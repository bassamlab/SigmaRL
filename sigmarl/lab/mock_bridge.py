# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""In-process loopback bridge: the lab interface backed by the kinematic
bicycle model. No ROS, no threads, no wall-clock time.

Used to validate the whole lab pipeline (frame transforms, command round
trip, WorldLab/Scenario wiring) against pure-VMAS ``WorldCustom`` runs: the
bicycle model is equivariant under the lab<->SigmaRL isometry, so a
``WorldLab + MockBridge`` rollout must reproduce a ``WorldCustom`` rollout up
to float32 round-off (and yaw wrapping, see below).
"""

import torch

from sigmarl.constants import AGENTS
from sigmarl.dynamics import KinematicBicycleModel
from sigmarl.helper_common import compute_applied_action_control
from sigmarl.lab.bridge_base import BridgeBase


class MockBridge(BridgeBase):
    def __init__(
        self,
        vehicle_ids,
        period_ms: int,
        world_x_dim: float,
        world_y_dim: float,
        device: str = "cpu",
        max_speed: float = None,
        min_speed: float = None,
    ):
        super().__init__(vehicle_ids, period_ms, world_x_dim, world_y_dim, device)

        # rl_action_type is fixed to "speed_steer": the position_speed
        # conversion (if any) already happened in WorldLab.step before the
        # targets reach the bridge — same as with the real lab.
        self._dynamics = KinematicBicycleModel(
            l_f=AGENTS["l_f"],
            l_r=AGENTS["l_r"],
            max_speed=AGENTS["max_speed"] if max_speed is None else max_speed,
            min_speed=AGENTS["min_speed"] if min_speed is None else min_speed,
            max_steering=AGENTS["max_steering"],
            min_steering=AGENTS["min_steering"],
            max_acc=AGENTS["max_acc"],
            min_acc=AGENTS["min_acc"],
            max_steering_rate=AGENTS["max_steering_rate"],
            min_steering_rate=AGENTS["min_steering_rate"],
            device=device,
            rl_action_type="speed_steer",
        )

        # Lab-frame state, one row per vehicle: (x, y, yaw, speed, steering)
        self._x = torch.zeros(self.n_vehicles, 5, dtype=torch.float32, device=device)
        # Lab-frame targets: (target_speed, target_steering)
        self._pending = torch.zeros(
            self.n_vehicles, 2, dtype=torch.float32, device=device
        )
        # After a respawn the first state snapshot reflects the freshly
        # placed fleet without time passing (mirrors VMAS, where reset
        # teleports instantaneously).
        self._skip_next_integration = False

    def apply_actions(self, target_actions: torch.Tensor) -> None:
        # Round-trip through the wire format (speed, curvature) so the command
        # pathway is exercised exactly as with the real lab.
        for a_i, (speed, curvature) in enumerate(
            self.commands_from_targets(target_actions)
        ):
            self._pending[a_i, 0] = speed
            self._pending[a_i, 1] = curvature * self.l_wb

    def apply_trajectories(
        self,
        trajectories: torch.Tensor,
        dt: float,
        initial_target_actions: torch.Tensor = None,
    ) -> None:
        """Process trajectory commands and prepare next-step integration.

        Round-trips through trajectories_to_lab to exercise the wire formatting.
        """
        _ = self.trajectories_to_lab(trajectories)
        for a_i in range(self.n_vehicles):
            if initial_target_actions is not None:
                self._pending[a_i, 0] = float(initial_target_actions[a_i, 0].item())
                self._pending[a_i, 1] = float(initial_target_actions[a_i, 1].item())
            elif trajectories.shape[1] > 1:
                self._pending[a_i, 0] = float(trajectories[a_i, 1, 3].item())
                self._pending[a_i, 1] = float(trajectories[a_i, 1, 4].item())

    def wait_for_state_list(self, timeout: float = None) -> dict:
        if self._skip_next_integration:
            self._skip_next_integration = False
            x = self._x
            return self.states_from_lab(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4])
        _, u = compute_applied_action_control(
            self._pending,
            self._x[:, 3],
            self._x[:, 4],
            self.period_s,
            dynamics=self._dynamics,
            rl_action_type="speed_steer",
        )
        x1, _, _ = self._dynamics.step(
            x0=self._x, u=u, dt=self.period_s, tick_per_step=1
        )
        # A real lab measurement (quaternion -> atan2) is always wrapped to
        # (-pi, pi]; keep the mock's yaw in that range so the frame transform
        # (which assumes wrapped input) stays exact.
        x1[:, 2] = torch.atan2(torch.sin(x1[:, 2]), torch.cos(x1[:, 2]))
        self._x = x1
        return self.states_from_lab(x1[:, 0], x1[:, 1], x1[:, 2], x1[:, 3], x1[:, 4])

    def respawn(self, vehicle_ids, spawn_states_sigmarl: torch.Tensor) -> None:
        spawn_states_sigmarl = spawn_states_sigmarl.to(self.device)
        pos_lab, yaw_lab = self.sigmarl_to_lab_pose(
            spawn_states_sigmarl[:, :2], spawn_states_sigmarl[:, 2]
        )
        for k, vid in enumerate(vehicle_ids):
            a_i = self.vehicle_ids.index(int(vid))
            self._x[a_i, 0:2] = pos_lab[k]
            self._x[a_i, 2] = yaw_lab[k]
            # Speed and steering are invariant under the frame isometry.
            self._x[a_i, 3] = spawn_states_sigmarl[k, 3]
            self._x[a_i, 4] = spawn_states_sigmarl[k, 4]
            self._pending[a_i] = 0.0
        self._skip_next_integration = True

    def manual_reset(self, vehicle_ids, reason: str = "") -> dict:
        """Offline rehearsal of the manual flow.

        The mock has no hands, so the "operator" moves nothing: this prompts,
        then reports the current state unchanged. Useful for exercising the
        prompt/validation logic without hardware. Never reached by
        ``test_lab_mock_equivalence``, which leaves ``lab_manual_reset`` at its
        default of False and so keeps using :py:meth:`respawn`.
        """
        self._prompt_manual_reset(vehicle_ids, reason)
        self._pending[:] = 0.0
        self._skip_next_integration = True
        x = self._x
        return self.states_from_lab(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4])

    def stop_vehicles(self) -> None:
        self._pending[:, 0] = 0.0

    def close(self) -> None:
        pass
