# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared, ROS-free logic for the lab physics backend.

A bridge connects ``WorldLab`` (the VMAS world whose ``step()`` delegates
physics to the CPM lab) with a physics provider: either the real lab via ROS 2
(``LabBridge``) or an in-process kinematic-bicycle loopback (``MockBridge``).

This module must stay importable without ROS: it holds the lab<->SigmaRL frame
transform, the VehicleState parsing math, and the SPEED_CURVATURE command
construction, all ported 1:1 from the previous ROS node
(``cpm_lab_sigmarl_control``) so both backends share one source of truth.
"""

import math
import os

import torch

from sigmarl.constants import AGENTS


class LabStopped(RuntimeError):
    """The lab requested a stop (SystemTrigger.STOP / emergency stop)."""


class LabStateTimeout(RuntimeError):
    """No fresh VehicleStateList arrived within the configured timeout."""


class BridgeBase:
    """Frame transforms + message math shared by LabBridge and MockBridge.

    The CPM lab and SigmaRL use different planar frames. The mapping is a
    self-inverse isometry (180-degree rotation followed by a shift by the
    world dimensions):

        p_sigmarl = -p_lab + [world_x_dim, world_y_dim]
        yaw_sigmarl = wrap(yaw_lab + pi)

    Speed and steering are invariant under this isometry.
    """

    def __init__(
        self,
        vehicle_ids,
        period_ms: int,
        world_x_dim: float,
        world_y_dim: float,
        device: str = "cpu",
    ):
        self.vehicle_ids = [int(v) for v in vehicle_ids]
        self.n_vehicles = len(self.vehicle_ids)
        self.period_s = period_ms / 1000.0
        self.device = device

        self.l_wb = AGENTS["l_wb"]
        self.l_r = AGENTS["l_r"]
        self.max_steering = AGENTS["max_steering"]

        self.shift = torch.tensor(
            [world_x_dim, world_y_dim], dtype=torch.float32, device=device
        )

    @staticmethod
    def _wrap_pi(rot: torch.Tensor) -> torch.Tensor:
        rot = rot + torch.pi
        rot = torch.where(rot > torch.pi, rot - 2 * torch.pi, rot)
        return rot

    def lab_to_sigmarl_pose(self, pos: torch.Tensor, yaw: torch.Tensor):
        """pos [..., 2], yaw [...] in lab frame -> SigmaRL frame."""
        return -pos + self.shift, self._wrap_pi(yaw)

    def sigmarl_to_lab_pose(self, pos: torch.Tensor, yaw: torch.Tensor):
        """The transform is self-inverse (rotation by -I is its own inverse)."""
        return -pos + self.shift, self._wrap_pi(yaw)

    def states_from_lab(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        yaw: torch.Tensor,
        speed: torch.Tensor,
        steering: torch.Tensor,
    ) -> dict:
        """Convert lab-frame measurements [n] into SigmaRL-frame state tensors.

        Returns a dict with pos [n, 2], rot/speed/steering/sideslip [n, 1],
        vel [n, 2]. Computing vel from the SigmaRL-frame rot is equivalent to
        computing it in the lab frame and rotating by pi (vel' = -vel), since
        speed and sideslip are frame-invariant.
        """
        pos_lab = torch.stack([x, y], dim=-1).to(self.device)
        pos, rot = self.lab_to_sigmarl_pose(pos_lab, yaw.to(self.device))
        speed = speed.to(self.device)
        steering = steering.to(self.device)

        sideslip = torch.atan(self.l_r / self.l_wb * torch.tan(steering))
        course = rot + sideslip
        vel = speed.unsqueeze(-1) * torch.stack(
            [torch.cos(course), torch.sin(course)], dim=-1
        )

        return {
            "pos": pos,
            "rot": rot.unsqueeze(-1),
            "speed": speed.unsqueeze(-1),
            "steering": steering.unsqueeze(-1),
            "sideslip": sideslip.unsqueeze(-1),
            "vel": vel,
        }

    def commands_from_targets(self, target_actions: torch.Tensor):
        """[n, 2] SigmaRL (target_speed, target_steering) -> [(speed, curvature)].

        curvature = steering / l_wb, matching the previous ROS node's
        _commands_from_act_speed_curvature (small-angle approximation of
        tan(steering) / l_wb; the lab's onboard controller was tuned against
        this convention).
        """
        commands = []
        for a_i in range(self.n_vehicles):
            speed = float(target_actions[a_i, 0].item())
            curvature = float(target_actions[a_i, 1].item()) / self.l_wb
            commands.append((speed, curvature))
        return commands

    @staticmethod
    def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
        """Convert a 2D planar yaw angle (radians) to a unit quaternion (qx, qy, qz, qw)."""
        half_yaw = float(yaw) * 0.5
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def trajectories_to_lab(self, trajectories: torch.Tensor):
        """Convert SigmaRL trajectories [n_vehicles, T, 5] to lab-frame waypoint data.

        Each trajectory contains (x, y, yaw, speed, steering) per waypoint.
        Returns a list (per vehicle) of lists of dicts containing:
        {
            "x": float,
            "y": float,
            "yaw": float,
            "speed": float,
            "distance": float,
            "steering": float,
        }
        """
        lab_trajs = []
        for a_i in range(self.n_vehicles):
            traj_i = trajectories[a_i]  # [T, 5]
            pos_sigmarl = traj_i[:, :2]
            yaw_sigmarl = traj_i[:, 2]
            pos_lab, yaw_lab = self.sigmarl_to_lab_pose(pos_sigmarl, yaw_sigmarl)

            points = []
            cum_dist = 0.0
            prev_x, prev_y = None, None
            for t_step in range(traj_i.shape[0]):
                x = float(pos_lab[t_step, 0].item())
                y = float(pos_lab[t_step, 1].item())
                yaw = float(yaw_lab[t_step].item())
                speed = float(traj_i[t_step, 3].item())
                steering = float(traj_i[t_step, 4].item())

                if prev_x is not None:
                    cum_dist += math.hypot(x - prev_x, y - prev_y)
                prev_x, prev_y = x, y

                points.append(
                    {
                        "x": x,
                        "y": y,
                        "yaw": yaw,
                        "speed": speed,
                        "distance": cum_dist,
                        "steering": steering,
                    }
                )
            lab_trajs.append(points)
        return lab_trajs

    # --- manual (human-in-the-loop) reset -----------------------------------

    def _prompt_manual_reset(self, vehicle_ids, reason: str) -> None:
        """Block until the operator confirms the vehicles are in place.

        Shared by both backends so the mock can rehearse the flow offline.
        Set ``SIGMARL_LAB_AUTO_CONFIRM=1`` to skip the blocking read (tests and
        unattended rehearsals only — never on hardware).
        """
        banner = (
            "\n"
            "==================== MANUAL RESET ====================\n"
            f"  reason   : {reason}\n"
            f"  vehicles : {list(vehicle_ids)}\n"
            "  The fleet is STOPPED and held stopped by the watchdog.\n"
            "  Place the vehicle(s) by hand. SigmaRL assigns the reference\n"
            "  paths from the MEASURED poses -- nothing is teleported and\n"
            "  no pose is sampled. Do NOT press STOP in the ControlCenter.\n"
            "======================================================\n"
            "  Press ENTER when the vehicles are in place: "
        )
        if os.environ.get("SIGMARL_LAB_AUTO_CONFIRM") == "1":
            print(banner + "[auto-confirmed]")
            return
        try:
            input(banner)
        except EOFError as exc:
            raise LabStopped(
                "stdin closed during a manual reset; manual mode needs an "
                "interactive terminal."
            ) from exc

    # --- interface implemented by subclasses -------------------------------

    def apply_actions(self, target_actions: torch.Tensor) -> None:
        """Send one (speed, steering) target per vehicle to the physics provider."""
        raise NotImplementedError

    def apply_trajectories(self, trajectories: torch.Tensor, dt: float) -> None:
        """Send multi-step trajectories [n_vehicles, T, 5] to the physics provider."""
        raise NotImplementedError

    def wait_for_state_list(self, timeout: float = None) -> dict:
        """Block until the next post-command state snapshot; return states_from_lab dict."""
        raise NotImplementedError

    def respawn(self, vehicle_ids, spawn_states_sigmarl: torch.Tensor) -> None:
        """Respawn the given vehicles at SigmaRL-frame spawn states.

        spawn_states_sigmarl is [k, 5] = (x, y, yaw, speed, steering). The
        simulation samples a random initial speed at reset; backends apply as
        much of the state as they physically can (the real lab can only place
        poses — respawned vehicles start at rest — while the mock applies the
        full state). The caller must pull the measured states afterwards.

        Not used when ``Parameters.lab_manual_reset`` is True — real vehicles
        cannot be teleported, so that mode uses :py:meth:`manual_reset`
        instead. The two are mutually exclusive within a run.
        """
        raise NotImplementedError

    def manual_reset(self, vehicle_ids, reason: str = "") -> dict:
        """Stop the fleet, wait for the operator, return a fresh snapshot.

        The human-in-the-loop counterpart of :py:meth:`respawn`, for real
        hardware. No pose is supplied and none is commanded: the operator
        places the vehicles by hand, and the returned measurements are ground
        truth from which the caller derives the reference paths.

        ``vehicle_ids`` only names which vehicles must be moved (for the
        prompt); the returned dict has one row per entry of
        ``self.vehicle_ids``, exactly like :py:meth:`wait_for_state_list`,
        because the operator may disturb any vehicle while working.
        """
        raise NotImplementedError

    def stop_vehicles(self) -> None:
        """Command zero speed to every vehicle."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources; must be idempotent."""
        raise NotImplementedError
