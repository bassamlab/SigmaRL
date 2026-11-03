# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""WorldStateLab: WorldStateRTSimulation with the initial-state *sampler* removed.

Real vehicles cannot be teleported. The operator places them by hand, so SigmaRL
must adopt whatever pose the localization reports instead of inventing one. This
class therefore overrides exactly one method -- ``_reset_init_state`` -- to
*assign* the best-matching reference path to the measured pose, and never calls
``set_pos`` / ``set_rot`` / ``set_vel`` / ``set_speed`` / ``set_steering`` /
``set_sideslip_angle``.

Everything else is inherited from ``WorldStateRTSimulation`` unchanged, because
``reward()`` and ``done()`` depend on it: ``update_state_before_rewarding``,
``update_collisions`` (including the swept-boundary check), the deadlock
tracker, the domain randomizer and the state buffer.

The cost function is ported from ``WorldStateRTReal._reset_init_state``
(``sigmarl/scenarios/world_state/world_state_rt/world_state_rt_real.py``), which
already solved this problem for the ROS high-level controller but is not
reachable from the scenario hierarchy. One deviation: a heading gate, see
``nearest_ref_path``.
"""

import torch
from termcolor import cprint

from sigmarl.helper_scenario import get_perpendicular_distances
from sigmarl.map_manager import MapManager
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt_sim import (
    WorldStateRTSimulation,
    WorldStateRTSimParameters,
)

# Beyond this heading error a reference path is considered to run *against* the
# vehicle and is discarded before the argmin.
YAW_GATE = torch.pi / 2

# Thresholds for colouring the per-agent diagnostic line red. Purely advisory.
SUSPICIOUS_LATERAL_M = 0.15
SUSPICIOUS_YAW_RAD = 0.6


def nearest_ref_path(pos, rot, ref_paths_scenario, yaw_gate: float = YAW_GATE):
    """Pick the reference path that best explains a measured pose.

    Args:
        pos: [2] measured position (SigmaRL frame).
        rot: scalar measured yaw (SigmaRL frame, wrapped to (-pi, pi]).
        ref_paths_scenario: list of reference-path dicts from the map manager.
        yaw_gate: paths whose heading error exceeds this are discarded before
            the argmin (unless *every* path is gated, in which case the gate is
            dropped so a choice is always made).

    Returns:
        (path_id, point_id, lateral_distance_m, abs_heading_error_rad).
        ``point_id`` follows the ``get_perpendicular_distances`` convention
        (nearest index + 1, i.e. the first point "in the future").

    The base cost ``(d * 100) ** 2 + |dyaw|`` is the one used by
    ``WorldStateRTReal``. It is dominated by lateral distance -- about 3 cm of
    offset outweighs a full 180-degree heading error -- so without the gate a
    car parked slightly off-centre can be matched to the oncoming lane.
    """
    distances = []
    point_ids = []
    yaw_errors = []
    for ref_path in ref_paths_scenario:
        distance, index = get_perpendicular_distances(pos, ref_path["center_line"])
        # index is argmin + 1, so index - 1 is always a valid yaw sample.
        index = int(index[0])
        d_yaw = ref_path["center_line_yaw"][index - 1] - rot
        d_yaw = (d_yaw + torch.pi) % (2 * torch.pi) - torch.pi

        distances.append(float(distance[0]))
        point_ids.append(index)
        yaw_errors.append(float(torch.as_tensor(d_yaw).abs()))

    lateral = torch.tensor(distances)
    heading = torch.tensor(yaw_errors)
    costs = (lateral * 100.0) ** 2 + heading

    against_traffic = heading > yaw_gate
    if not bool(against_traffic.all()):
        costs = torch.where(
            against_traffic, torch.full_like(costs, float("inf")), costs
        )

    path_id = int(torch.argmin(costs))
    return path_id, point_ids[path_id], distances[path_id], yaw_errors[path_id]


class WorldStateLab(WorldStateRTSimulation):
    """World state whose reset adopts measured poses instead of sampling them."""

    def __init__(self, params: WorldStateRTSimParameters, map: MapManager):
        super().__init__(params, map)
        # Set by ScenarioRoadTrafficLab so the diagnostic can name the physical
        # vehicle rather than only the agent index.
        self.vehicle_ids = None

    def _reset_init_state(
        self,
        agent_states,
        ref_paths_scenario,
        agent_index,
        env_index: int = None,
        is_reset_single_agent: bool = False,
        predefined_ref_path_idx_i=None,
        initial_state=None,
        initial_state_buffer=None,
    ):
        """Assign a reference path to the agent's *measured* pose.

        Deliberately writes only reference-path bookkeeping. ``entity.state`` is
        never touched, which is what preserves the hand-placed pose (and its
        measured, ~zero speed -- making ``is_zero_initial_speed`` a no-op here).
        """
        # These two inputs exist to *impose* a pose, which is exactly what
        # manual mode must never do. ScenarioRoadTrafficLab also guards them at
        # construction, because WorldStateRTSimulation.reset can apply a
        # predefined pose without ever calling this method.
        assert initial_state is None, (
            "Manual lab reset: reference paths must come from the measured "
            "pose. Set parameters.is_challenging_initial_state_buffer = False."
        )
        assert predefined_ref_path_idx_i is None, (
            "Manual lab reset: reference paths must come from the measured "
            "pose. Set parameters.predefined_ref_path_idx = None."
        )
        assert self.batch_dim == 1, "The lab backend drives one physical fleet."

        env_i = 0 if env_index is None else int(env_index)
        i_agent = int(agent_index)

        pos = agent_states[i_agent].pos[env_i]  # [2], measured
        rot = agent_states[i_agent].rot[env_i, 0]  # scalar, measured

        path_id, point_id, lateral, yaw_error = nearest_ref_path(
            pos, rot, ref_paths_scenario
        )

        self.ref_paths_agent_related.path_id[env_i, i_agent] = path_id
        self.ref_paths_agent_related.point_id[env_i, i_agent] = point_id

        self._report_assignment(
            i_agent, pos, rot, path_id, point_id, lateral, yaw_error
        )

        return ref_paths_scenario[path_id], path_id

    def _report_assignment(
        self, i_agent, pos, rot, path_id, point_id, lateral, yaw_error
    ):
        """Print what the measured pose was matched to, so a bad hand placement
        is visible before the vehicles start moving."""
        vehicle = (
            f" (vehicle {self.vehicle_ids[i_agent]})"
            if self.vehicle_ids is not None
            else ""
        )
        is_suspicious = lateral > SUSPICIOUS_LATERAL_M or yaw_error > SUSPICIOUS_YAW_RAD
        cprint(
            f"[WorldStateLab] agent {i_agent}{vehicle} @ "
            f"({float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(rot):+.2f} rad) "
            f"-> path {path_id}, point {point_id} "
            f"(lateral {lateral * 100:5.1f} cm, "
            f"heading err {yaw_error * 180 / torch.pi:4.0f} deg)",
            "red" if is_suspicious else "cyan",
        )
