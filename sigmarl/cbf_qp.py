# Copyright (c) 2025, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import cvxpy as cp
import time
import math
from math import ceil, atan2, pi
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from sigmarl.constants import (
    AGENTS,
    POSITION_SPEED_LATERAL_OFFSET_BOUND,
    POSITION_SPEED_LOOKAHEAD_DIST,
)
from sigmarl.dynamics import KinematicBicycleModel
from sigmarl.rectangle_approximation import RectangleCircleApproximation
from sigmarl.helper_common import (
    Parameters,
    compute_applied_action_control,
    compute_path_tracking_target_steering,
)
from sigmarl.pseudo_distance import PseudoDistance
from sigmarl.softmin_distance import SoftminDistance
from sigmarl.mtv_based_sm_predictor import SafetyMarginEstimatorModule  # TODO?
from sigmarl.helper_common import Vehicle

# -------- Grouping state structure --------
@dataclass
class AgentStateList:
    x: float
    y: float
    v: float


def _euclid(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return float(np.sqrt(d[0] * d[0] + d[1] * d[1]))


def group_agents_k_nearest(
    agent_states: List[AgentStateList],
    max_group_size: int,
) -> List[List[int]]:
    """
    K-seeded region growing with capacity constraints.

    Steps:
      1) Compute K = ceil(N / max_group_size).
      2) Select K spatially distributed seeds using farthest-point sampling.
      3) Assign remaining agents to the nearest seed group with available capacity.
      4) If nearest group is full, fall back to the next nearest group.

    Guarantees:
      - Exactly K groups
      - Group size <= max_group_size
      - Every agent is assigned exactly once
    """

    def squared_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive")

    n_agents = len(agent_states)
    if n_agents == 0:
        return []

    # --------------------------------------------------------
    # Determine target number of groups
    # --------------------------------------------------------
    K = int(math.ceil(n_agents / max_group_size))

    positions = [(s.x, s.y) for s in agent_states]

    # --------------------------------------------------------
    # Step 1: Farthest-point sampling for seed selection
    # --------------------------------------------------------
    seeds: List[int] = []

    # First seed: choose arbitrarily (index 0)
    seeds.append(0)

    while len(seeds) < K:
        best_candidate = None
        best_dist = -1.0

        for i in range(n_agents):
            if i in seeds:
                continue

            # Distance to closest existing seed
            d_min = min(squared_distance(positions[i], positions[s]) for s in seeds)

            if d_min > best_dist:
                best_dist = d_min
                best_candidate = i

        seeds.append(best_candidate)

    # --------------------------------------------------------
    # Step 2: Initialize groups with seeds
    # --------------------------------------------------------
    groups: List[List[int]] = [[s] for s in seeds]
    group_centroids: List[Tuple[float, float]] = [positions[s] for s in seeds]

    assigned = set(seeds)

    # --------------------------------------------------------
    # Step 3: Assign remaining agents
    # --------------------------------------------------------
    for i in range(n_agents):
        if i in assigned:
            continue

        # Order groups by distance to centroid
        dists = [
            (g_idx, squared_distance(positions[i], group_centroids[g_idx]))
            for g_idx in range(K)
        ]
        dists.sort(key=lambda x: x[1])

        # Try nearest group with remaining capacity
        placed = False
        for g_idx, _ in dists:
            if len(groups[g_idx]) < max_group_size:
                groups[g_idx].append(i)

                # Update centroid incrementally
                k = len(groups[g_idx])
                cx, cy = group_centroids[g_idx]
                px, py = positions[i]
                group_centroids[g_idx] = (
                    cx + (px - cx) / k,
                    cy + (py - cy) / k,
                )

                assigned.add(i)
                placed = True
                break

        if not placed:
            raise RuntimeError(
                "No available group with remaining capacity. "
                "This should not happen if K = ceil(N / max_group_size)."
            )

    # Final consistency check
    assert len(groups) == K
    assert all(len(g) <= max_group_size for g in groups)
    assert sorted(i for g in groups for i in g) == list(range(n_agents))

    return groups


@dataclass
class TrajectoryHistory:
    t: List[float] = field(default_factory=list)
    xy: List[np.ndarray] = field(default_factory=list)
    state: List[np.ndarray] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    u1: List[float] = field(default_factory=list)
    u2: List[float] = field(default_factory=list)
    evt: List[str] = field(default_factory=list)  # "", "COLL", "QP-INF"
    qp_solving_t: List[float] = field(default_factory=list)
    qp_solving_iter: List[int] = field(default_factory=list)
    lambda_history: List[np.ndarray] = field(default_factory=list)
    max_lane_slack: List[float] = field(default_factory=list)
    max_pair_slack: List[float] = field(default_factory=list)


class CBFQP:
    def __init__(self, env=None, env_idx: int = None, agent_idx: int = None, **kwargs):
        self.hist = TrajectoryHistory()

        self.env = env
        self.env_idx = env_idx
        self.agent_idx = agent_idx
        self.scenario = self.env.base_env.scenario_name
        self.map_pseudo_distance: PseudoDistance | SoftminDistance = (
            self.env.base_env.scenario_name.map_pseudo_distance
        )
        self.initialize_params()

        self.rec_cir_approx = RectangleCircleApproximation(
            self.length, self.width, self.parameters.n_circles_approximate_vehicle
        )
        self.circle_radius = self.rec_cir_approx.radius

        self.kbm = KinematicBicycleModel(
            l_f=self.l_f,
            l_r=self.l_r,
            max_speed=self.v_max,
            min_speed=self.v_min,
            max_steering=self.steering_max,
            min_steering=self.steering_min,
            max_acc=self.a_max,
            min_acc=self.a_min,
            max_steering_rate=self.steering_rate_max,
            min_steering_rate=self.steering_rate_min,
            device=self.device,
        )

        # Build QPs based on mode
        if self.parameters.is_solve_qp:
            if self.parameters.is_using_centralized_cbf:
                if self.parameters.is_grouping_agents:
                    self.build_grouped_cbf_qps()
                else:
                    self.build_centralized_cbf_qp()
            else:
                self.build_fully_decentralized_cbf_qps()

        self.cbf_solving_t = []

    def initialize_params(self):
        self.parameters: Parameters = self.env.base_env.scenario_name.parameters
        self.device = self.parameters.device
        self.dt = self.parameters.dt
        self.r = 2
        self.dt_taylor = float(self.r * self.dt)
        self.dx = 0.02
        self.dy = 0.02

        self.length = AGENTS["length"]
        self.width = AGENTS["width"]
        self.l_wb = AGENTS["l_wb"]
        self.l_f = AGENTS["l_f"]
        self.l_r = AGENTS["l_r"]

        self.v_max = torch.tensor(
            self.parameters.max_speed, device=self.device, dtype=torch.float32
        )
        self.v_min = torch.tensor(
            AGENTS["min_speed"], device=self.device, dtype=torch.float32
        )
        self.steering_max = torch.tensor(
            AGENTS["max_steering"], device=self.device, dtype=torch.float32
        )
        self.steering_min = torch.tensor(
            AGENTS["min_steering"], device=self.device, dtype=torch.float32
        )
        self.a_max = AGENTS["max_acc"]
        self.a_min = AGENTS["min_acc"]
        self.steering_rate_max = AGENTS["max_steering_rate"]
        self.steering_rate_min = AGENTS["min_steering_rate"]

        self.is_obs_noise = self.parameters.is_obs_noise
        self.obs_noise_level = self.parameters.obs_noise_level

        self.safety_buffer = 0.02 * self.width
        self.lambda_ttcbf_predefined = (
            0.2  # Should between 0 and 1. Higher -> more aggressive
        )
        self._warned_invalid_cbf_lambda = False
        self._warned_positive_centralized_slack = False

        # Nominal controller selection
        self.nom_controller_type = self.parameters.nom_controller_type
        self.rl_action_type = getattr(self.parameters, "rl_action_type", "speed_steer")
        self.lookahead_dist = float(
            getattr(self.parameters, "lookahead_dist", POSITION_SPEED_LOOKAHEAD_DIST)
        )
        self.position_speed_lateral_max = POSITION_SPEED_LATERAL_OFFSET_BOUND
        # CLF parameters
        self.lam_clf = float(getattr(self.parameters, "lam_clf", 2.0))
        self.ref_speed = float(self.parameters.ref_speed)  # m/s
        self.w_clf_relax = float(getattr(self.parameters, "w_clf_relax", 1.0))
        self.k_clf_heading = float(
            getattr(self.parameters, "k_clf_heading", 1.0)
        )  # nominal P gain → u2
        self.k_clf_speed = float(
            getattr(self.parameters, "k_clf_speed", 1.0)
        )  # nominal P gain → u1

        self.u_placeholder = torch.tensor([0.0, 0.0], dtype=torch.float32)

        self.nom_weight = 1 * np.diag(
            [10, 1]
        )  # Weight for nominal control tracking (acceleration, steering rate) in the cost function
        self.lane_slack_weight = (
            1e9  # Weight for lane boundary slack variable in the cost function
        )
        self.pair_slack_weight = (
            1e9  # Weight for inter-group agent slack variable in the cost function
        )
        self.cross_slack_weight = (
            1e9  # Weight for cross-group agent slack variable in the cost function
        )
        self.lambda_weight = 1e3  # Weight for lambda penalty in the cost function

    # ---------- helpers ----------
    @staticmethod
    def _wrap_angle(a: float) -> float:
        # map to (-pi, pi]
        a = (a + pi) % (2 * pi) - pi
        return a

    def _sanitize_raw_action(self, raw_actions):
        if isinstance(raw_actions, np.ndarray):
            raw_actions = torch.from_numpy(raw_actions)

        raw_actions = raw_actions.to(self.device, dtype=torch.float32).clone()
        if raw_actions.ndim == 1:
            raw_actions = raw_actions.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        if self.rl_action_type == "speed":
            raw_actions[:, 0] = torch.clamp(
                raw_actions[:, 0],
                min=-float(self.v_max.item()),
                max=float(self.v_max.item()),
            )
        elif self.rl_action_type == "position_speed":
            raw_actions[:, 0] = torch.clamp(
                raw_actions[:, 0],
                min=-self.position_speed_lateral_max,
                max=self.position_speed_lateral_max,
            )
            raw_actions[:, 1] = torch.clamp(
                raw_actions[:, 1],
                min=-float(self.v_max.item()),
                max=float(self.v_max.item()),
            )
        else:
            raw_actions[:, 0] = torch.clamp(
                raw_actions[:, 0],
                min=-float(self.v_max.item()),
                max=float(self.v_max.item()),
            )
            raw_actions[:, 1] = torch.clamp(
                raw_actions[:, 1],
                min=float(self.steering_min.item()),
                max=float(self.steering_max.item()),
            )

        if not is_batch:
            raw_actions = raw_actions.squeeze(0)
        return raw_actions

    def _ref_xy_from_tensordict(self, tensordict, agent_idx: int):
        ref = tensordict["agents", "info", "ref"][self.env_idx, agent_idx]
        n_points = ref.shape[-1] // 2
        ref_idx = min(2, n_points - 1)
        start = 2 * ref_idx
        return ref[start : start + 2]

    def _set_speed_target_action_override(self, agent_idx: int, target_action):
        if self.rl_action_type != "speed":
            return
        agent = self.env.base_env.scenario_name.world.agents[agent_idx]
        target_action = torch.as_tensor(
            target_action,
            device=self.device,
            dtype=torch.float32,
        ).reshape(2)
        if not hasattr(agent.dynamics, "speed_target_action_override"):
            batch_dim = agent.state.pos.shape[0]
            agent.dynamics.speed_target_action_override = torch.zeros(
                (batch_dim, 2), device=self.device, dtype=torch.float32
            )
            agent.dynamics.speed_target_action_override_mask = torch.zeros(
                batch_dim, device=self.device, dtype=torch.bool
            )
        agent.dynamics.speed_target_action_override[self.env_idx] = target_action
        agent.dynamics.speed_target_action_override_mask[self.env_idx] = True

    def _target_actions_from_current_actions(self, tensordict, states):
        raw_actions = tensordict[("agents", "action")][self.env_idx].clone()
        if self.rl_action_type != "speed":
            return self.raw_action_to_target_action(raw_actions)

        target_actions = torch.zeros(
            (raw_actions.shape[0], 2), device=self.device, dtype=torch.float32
        )
        agents = self.env.base_env.scenario_name.world.agents
        for i in range(raw_actions.shape[0]):
            override_mask = getattr(
                agents[i].dynamics, "speed_target_action_override_mask", None
            )
            if override_mask is not None and override_mask[self.env_idx]:
                target_actions[i] = agents[i].dynamics.speed_target_action_override[
                    self.env_idx
                ]
            else:
                target_actions[i] = self.raw_action_to_target_action(
                    raw_actions[i],
                    state=states[i],
                    ref_xy=self._ref_xy_from_tensordict(tensordict, i),
                )
        return target_actions

    def raw_action_to_target_action(self, raw_actions, state=None, ref_xy=None):
        raw_actions = self._sanitize_raw_action(raw_actions)
        if raw_actions.ndim == 1:
            raw_actions = raw_actions.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        if self.rl_action_type == "speed":
            if state is None or ref_xy is None:
                raise ValueError(
                    "speed rl_action_type requires state and ref_xy to recover target steering."
                )
            state = state.to(self.device, dtype=torch.float32)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            ref_xy = ref_xy.to(self.device, dtype=torch.float32)
            if ref_xy.ndim == 1:
                ref_xy = ref_xy.unsqueeze(0)
            target_speed = raw_actions[:, 0]
            steering_target = compute_path_tracking_target_steering(
                pos=state[:, 0:2],
                rot=state[:, 2],
                ref_point=ref_xy,
                l_wb=self.l_wb,
            )
            target_actions = torch.stack([target_speed, steering_target], dim=1)
        elif self.rl_action_type == "position_speed":
            lateral_offset = raw_actions[:, 0]
            target_speed = raw_actions[:, 1]
            ld = raw_actions.new_tensor(self.lookahead_dist)
            L = raw_actions.new_tensor(self.l_wb)
            steering_target = torch.atan(
                2.0 * L * lateral_offset / (ld**2 + lateral_offset**2)
            )
            target_actions = torch.stack([target_speed, steering_target], dim=1)
        else:
            target_actions = raw_actions

        target_actions[:, 0] = torch.clamp(
            target_actions[:, 0],
            min=-float(self.v_max.item()),
            max=float(self.v_max.item()),
        )
        target_actions[:, 1] = torch.clamp(
            target_actions[:, 1],
            min=float(self.steering_min.item()),
            max=float(self.steering_max.item()),
        )

        if not is_batch:
            target_actions = target_actions.squeeze(0)
        return target_actions

    def target_action_to_raw_action(self, target_actions):
        if isinstance(target_actions, np.ndarray):
            target_actions = torch.from_numpy(target_actions)

        target_actions = target_actions.to(self.device, dtype=torch.float32).clone()
        if target_actions.ndim == 1:
            target_actions = target_actions.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        target_actions[:, 0] = torch.clamp(
            target_actions[:, 0],
            min=-float(self.v_max.item()),
            max=float(self.v_max.item()),
        )
        target_actions[:, 1] = torch.clamp(
            target_actions[:, 1],
            min=float(self.steering_min.item()),
            max=float(self.steering_max.item()),
        )

        if self.rl_action_type == "speed":
            raw_actions = target_actions[:, 0:1]
            projected_target_actions = target_actions
        elif self.rl_action_type == "position_speed":
            # Invert the pure-pursuit mapping used in raw_action_to_target_action:
            #   delta = atan(L * kappa),  kappa = 2*y / (ld^2 + y^2)
            # where y is the lateral offset of the lookahead point.
            kappa = torch.tan(target_actions[:, 1]) / float(self.l_wb)
            ld = float(self.lookahead_dist)
            y_max = float(self.position_speed_lateral_max)
            # With the current parameters y_max <= ld, so curvature is monotonic
            # on the allowed lateral-offset interval and reaches its largest
            # representable magnitude at +/-y_max.
            max_kappa = 2.0 * y_max / (ld**2 + y_max**2) if y_max > 0 else 0.0
            kappa = torch.clamp(kappa, min=-max_kappa, max=max_kappa)
            # Solve kappa*y^2 - 2*y + kappa*ld^2 = 0 using the numerically stable
            # root y = kappa*ld^2 / (1 + sqrt(1 - (kappa*ld)^2)). This is the
            # branch that satisfies y -> 0 as kappa -> 0.
            discr = torch.clamp(1.0 - (kappa * ld) ** 2, min=0.0)
            lateral_offset = torch.where(
                kappa.abs() < 1e-8,
                torch.zeros_like(kappa),
                (kappa * (ld**2)) / (1.0 + torch.sqrt(discr)),
            )
            lateral_offset = torch.clamp(lateral_offset, min=-y_max, max=y_max)
            raw_actions = torch.stack([lateral_offset, target_actions[:, 0]], dim=1)
            # Recompute the target action after projection so callers can compare
            # the requested target with the closest representable position_speed
            # action under the lateral-offset bound.
            projected_target_actions = self.raw_action_to_target_action(raw_actions)
        else:
            raw_actions = target_actions
            projected_target_actions = target_actions

        if not is_batch:
            raw_actions = raw_actions.squeeze(0)
            projected_target_actions = projected_target_actions.squeeze(0)
        return raw_actions, projected_target_actions

    def _clf_errors_for_agent(
        self,
        state_i: torch.Tensor,
        ref_xy_i: torch.Tensor,
        ref_speed: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        heading and speed errors.
        heading error e_h = wrap(atan2(ref - pos) - psi)
        speed error e_v = ref_speed - v
        """
        x = float(state_i[0].item())
        y = float(state_i[1].item())
        psi = float(state_i[2].item())
        v = float(state_i[3].item())
        rx = float(ref_xy_i[0].item())
        ry = float(ref_xy_i[1].item())
        desired = atan2(ry - y, rx - x)
        e_h = self._wrap_angle(desired - psi)
        if ref_speed is None:
            ref_speed = self.ref_speed
        e_v = float(ref_speed) - v
        return e_h, e_v

    def _resolve_clf_reference(self, tensordict, state_i: torch.Tensor, agent_idx: int):
        if (
            self.parameters.uses_motion_policy
            and self.rl_action_type == "position_speed"
        ):
            raw_action = tensordict[("agents", "action")][
                self.env_idx, agent_idx
            ].clone()
            if self.is_obs_noise:
                raw_action = (
                    raw_action + torch.rand_like(raw_action) * self.obs_noise_level
                )
            raw_action = self._sanitize_raw_action(raw_action)
            lateral_offset = raw_action[0]
            target_speed = float(raw_action[1].item())

            x = state_i[0]
            y = state_i[1]
            psi = state_i[2]
            ld = state_i.new_tensor(self.lookahead_dist)
            ref_xy = torch.stack(
                [
                    x + torch.cos(psi) * ld - torch.sin(psi) * lateral_offset,
                    y + torch.sin(psi) * ld + torch.cos(psi) * lateral_offset,
                ]
            )
            return ref_xy, target_speed

        if self.parameters.uses_motion_policy and self.rl_action_type == "speed":
            raw_action = tensordict[("agents", "action")][
                self.env_idx, agent_idx
            ].clone()
            if self.is_obs_noise:
                raw_action = (
                    raw_action + torch.rand_like(raw_action) * self.obs_noise_level
                )
            raw_action = self._sanitize_raw_action(raw_action)
            return self._ref_xy_from_tensordict(tensordict, agent_idx), float(
                raw_action[0].item()
            )

        ref_xy = self._ref_xy_from_tensordict(tensordict, agent_idx)
        return ref_xy, self.ref_speed

    def _compute_nominal_control_for_agent(self, tensordict, state_i, agent_idx: int):
        if self.nom_controller_type == "rl":
            raw_action = tensordict[("agents", "action")][
                self.env_idx, agent_idx
            ].clone()
            if self.is_obs_noise:
                raw_action = (
                    raw_action + torch.rand_like(raw_action) * self.obs_noise_level
                )
            target_action, u_nom = self.rl_action_to_u(
                rl_actions=raw_action,
                v=state_i[3],
                steering=state_i[4],
                state=state_i,
                ref_xy=(
                    self._ref_xy_from_tensordict(tensordict, agent_idx)
                    if self.rl_action_type == "speed"
                    else None
                ),
            )
            return (
                u_nom.detach().cpu().numpy().reshape(2),
                target_action.detach(),
                0.0,
                0.0,
            )

        ref_xy, ref_speed = self._resolve_clf_reference(tensordict, state_i, agent_idx)
        e_h, e_v = self._clf_errors_for_agent(state_i, ref_xy, ref_speed)
        u_nom_np = np.array(
            [
                np.clip(self.k_clf_speed * e_v, self.a_min, self.a_max),
                np.clip(
                    self.k_clf_heading * e_h,
                    self.steering_rate_min,
                    self.steering_rate_max,
                ),
            ],
            dtype=np.float64,
        )
        target_action = self.u_to_target_action(
            u_nom_np, state_i[3], state_i[4]
        ).detach()
        return u_nom_np, target_action, e_h, e_v

    def _compute_applied_control_for_agent(self, tensordict, state_i, agent_idx: int):
        raw_action = tensordict[("agents", "action")][self.env_idx, agent_idx].clone()

        target_action, u_applied = self.rl_action_to_u(
            rl_actions=raw_action,
            v=state_i[3],
            steering=state_i[4],
            state=state_i,
            ref_xy=(
                self._ref_xy_from_tensordict(tensordict, agent_idx)
                if self.rl_action_type == "speed"
                else None
            ),
        )
        return (
            u_applied.detach().cpu().numpy().reshape(2),
            target_action.detach(),
        )

    def rl_action_to_u(
        self,
        rl_actions: torch.tensor,
        v: torch.tensor,
        steering: torch.tensor,
        state: torch.Tensor = None,
        ref_xy: torch.Tensor = None,
    ):
        """
        Convert the environment action representation into a target speed/steering
        pair and then into low-level acceleration / steering-rate inputs.

        Args:
            rl_actions: [target_speed, target_steering_angle] for "speed_steer",
                [lateral_offset, target_speed] for "position_speed", or
                [target_speed] for "speed"

        Returns:
            target_action: [target_speed, target_steering_angle]
            u: [acceleration, steering_rate]
        """
        return compute_applied_action_control(
            rl_actions,
            v,
            steering,
            self.dt,
            rl_action_type=self.rl_action_type,
            lookahead_dist=self.lookahead_dist,
            l_wb=self.l_wb,
            position_speed_lateral_max=self.position_speed_lateral_max,
            pos=state[..., 0:2] if state is not None else None,
            rot=state[..., 2] if state is not None else None,
            path_tracking_ref_point=ref_xy,
            target_speed_min=-float(self.v_max.item()),
            target_speed_max=float(self.v_max.item()),
            target_steering_min=float(self.steering_min.item()),
            target_steering_max=float(self.steering_max.item()),
            min_acc=self.a_min,
            max_acc=self.a_max,
            min_steering_rate=self.steering_rate_min,
            max_steering_rate=self.steering_rate_max,
        )

    def u_to_target_action(self, u, state_v, state_steering):
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)
        if (
            u.ndimension() == 1
            or state_v.ndimension() == 0
            or state_steering.ndimension() == 0
        ):
            u = u.unsqueeze(0)
            state_v = state_v.unsqueeze(0)
            state_steering = state_steering.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        u = u.to(self.device, dtype=torch.float32)
        state_v = state_v.to(self.device, dtype=torch.float32)
        state_steering = state_steering.to(self.device, dtype=torch.float32)

        v = state_v + u[:, 0] * self.dt
        steering = state_steering + u[:, 1] * self.dt
        steering = (steering + torch.pi) % (2 * torch.pi) - torch.pi
        v = torch.clamp(v, min=self.v_min, max=self.v_max)
        steering = torch.clamp(steering, min=self.steering_min, max=self.steering_max)
        target_action = torch.stack([v, steering], dim=1)
        if not is_batch:
            target_action = target_action.squeeze(0)
        return target_action

    def u_to_rl_action(self, u, state_v, state_steering, agent_idx: int = None):
        """
        Copy the original function
        """
        target_action = self.u_to_target_action(u, state_v, state_steering)
        if agent_idx is not None:
            self._set_speed_target_action_override(agent_idx, target_action)
        rl_action, _ = self.target_action_to_raw_action(target_action)
        return rl_action.to(self.device)

    def get_circle_centers(self, state: torch.Tensor) -> torch.Tensor:
        """
        Copy the original function
        """
        # --- Input Validation ---
        if not isinstance(state, torch.Tensor):
            raise TypeError("Input 'state' must be a torch.Tensor.")

        n_circles = self.parameters.n_circles_approximate_vehicle
        if n_circles < 1:
            raise ValueError("Number of circles (n_circles) must be at least 1.")
        if state.ndim == 0 or state.shape[0] < 3:
            raise ValueError("State tensor must contain at least [x, y, yaw].")

        vehicle_center_global = state[0:2]
        vehicle_yaw = state[2]
        other_state = state[3:]

        relative_centers_local = torch.tensor(
            self.rec_cir_approx.centers, dtype=state.dtype, device=state.device
        )

        cos_yaw = torch.cos(vehicle_yaw)
        sin_yaw = torch.sin(vehicle_yaw)
        rotation_matrix_global = torch.tensor(
            [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
            dtype=state.dtype,
            device=state.device,
        )

        rotated_vectors_global = torch.matmul(
            rotation_matrix_global, relative_centers_local.T
        ).T

        global_circle_centers = (
            rotated_vectors_global + vehicle_center_global.unsqueeze(0)
        )

        vehicle_yaw_expanded = (
            vehicle_yaw.unsqueeze(0).unsqueeze(0).expand(n_circles, 1)
        )
        other_state_expanded = other_state.unsqueeze(0).expand(n_circles, -1)

        circle_states = torch.cat(
            (global_circle_centers, vehicle_yaw_expanded, other_state_expanded), dim=1
        )
        return circle_states

    def estimate_agent_2_lane_safety_margin(self, agent_pos: torch.tensor, path_id):
        """
        agent_pos: torch.Size([2])
        path_id: Integer
        """
        offsets = torch.tensor(
            [
                [0, 0],
                [self.dx, 0],
                [0, self.dy],
                [-self.dx, 0],
                [0, -self.dy],
                [self.dx, self.dy],
                [self.dx, -self.dy],
                [-self.dx, self.dy],
                [-self.dx, -self.dy],
            ],
            device=self.device,
        )
        query_points = agent_pos.unsqueeze(0) + offsets

        time_pseudo_dis_start = time.time()
        if self.parameters.lane_distance_type == "softmin":
            (
                dleft_results,
                dright_results,
            ) = self.map_pseudo_distance.get_distance_with_derivatives(
                path_id, query_points, dx=self.dx, dy=self.dy
            )
        else:
            dleft_results, dright_results = self.map_pseudo_distance.get_distance(
                path_id, query_points
            )
        self.time_pseudo_dis += time.time() - time_pseudo_dis_start

        if self.parameters.lane_distance_type == "softmin":
            dleft, grad_left, hessian_left = dleft_results
            dright, grad_right, hessian_right = dright_results

            sm_left = dleft - self.circle_radius**2
            sm_right = dright - self.circle_radius**2

            return sm_left, grad_left, hessian_left, sm_right, grad_right, hessian_right
        if self.parameters.lane_distance_type != "pseudo":
            raise ValueError(
                f"Unsupported lane_distance_type: {self.parameters.lane_distance_type}"
            )

        (
            dleft,
            dleft_xplus,
            dleft_yplus,
            dleft_xminus,
            dleft_yminus,
            dleft_xplus_yplus,
            dleft_xplus_yminus,
            dleft_xminus_yplus,
            dleft_xminus_yminus,
        ) = dleft_results
        (
            dright,
            dright_xplus,
            dright_yplus,
            dright_xminus,
            dright_yminus,
            dright_xplus_yplus,
            dright_xplus_yminus,
            dright_xminus_yplus,
            dright_xminus_yminus,
        ) = dright_results

        dleft_dx = (dleft_xplus - dleft) / self.dx
        dleft_dy = (dleft_yplus - dleft) / self.dy
        dright_dx = (dright_xplus - dright) / self.dx
        dright_dy = (dright_yplus - dright) / self.dy

        dleft2_dx2 = (dleft_xplus - 2 * dleft + dleft_xminus) / (self.dx**2)
        dleft2_dy2 = (dleft_yplus - 2 * dleft + dleft_yminus) / (self.dy**2)
        dright2_dx2 = (dright_xplus - 2 * dright + dright_xminus) / (self.dx**2)
        dright2_dy2 = (dright_yplus - 2 * dright + dright_yminus) / (self.dy**2)
        dleft2_dxdy = (
            dleft_xplus_yplus
            - dleft_xplus_yminus
            - dleft_xminus_yplus
            + dleft_xminus_yminus
        ) / (4 * self.dx * self.dy)
        dright2_dxdy = (
            dright_xplus_yplus
            - dright_xplus_yminus
            - dright_xminus_yplus
            + dright_xminus_yminus
        ) / (4 * self.dx * self.dy)

        sm_left = dleft - self.circle_radius
        sm_right = dright - self.circle_radius

        grad_left = np.array([dleft_dx, dleft_dy])
        grad_right = np.array([dright_dx, dright_dy])

        hessian_left = np.array(
            [
                [dleft2_dx2, dleft2_dxdy],
                [dleft2_dxdy, dleft2_dy2],
            ]
        )
        hessian_right = np.array(
            [
                [dright2_dx2, dright2_dxdy],
                [dright2_dxdy, dright2_dy2],
            ]
        )

        return sm_left, grad_left, hessian_left, sm_right, grad_right, hessian_right

    def compute_dstate_2nd_time(self, u, state, dstate_time):
        """
        Copy the original function
        """
        u_1, u_2 = u
        k = self.l_r / self.l_wb
        dpsi = dstate_time[2].item()
        tan_delta = np.tan(state[4].item())
        beta = np.arctan(k * tan_delta)
        sec_delta_sq = 1 / np.cos(state[4].item()) ** 2
        tan_beta = k * tan_delta
        cos_beta = 1 / np.sqrt(1 + tan_beta**2)
        sin_beta = tan_beta * cos_beta

        dbeta = (k * sec_delta_sq * u_2) / (1 + (k * tan_delta) ** 2)

        ddx = u_1 * np.cos(state[2].item() + beta) - state[3].item() * np.sin(
            state[2].item() + beta
        ) * (dpsi + dbeta)
        ddy = u_1 * np.sin(state[2].item() + beta) + state[3].item() * np.cos(
            state[2].item() + beta
        ) * (dpsi + dbeta)

        ddpsi = (
            (u_1 / self.l_wb) * cos_beta * tan_delta
            + (state[3].item() / self.l_wb) * cos_beta * sec_delta_sq * u_2
            - (state[3].item() / self.l_wb) * sin_beta * tan_delta * dbeta
        )
        return ddx, ddy, ddpsi

    def compute_center_state_time_derivatives(
        self, psi, dstate_time_agent, ddx, ddy, ddpsi, idx
    ):
        """
        Copy the original function
        """
        psi = psi.item()
        dx = dstate_time_agent[0].item()
        dy = dstate_time_agent[1].item()
        dpsi = dstate_time_agent[2].item()

        delta_x, delta_y = self.rec_cir_approx.centers[idx]

        dx_center = dx - delta_x * np.sin(psi) * dpsi - delta_y * np.cos(psi) * dpsi
        dy_center = dy + delta_x * np.cos(psi) * dpsi - delta_y * np.sin(psi) * dpsi

        ddx_center = (
            ddx
            - delta_x * (np.sin(psi) * ddpsi + np.cos(psi) * dpsi * dpsi)
            - delta_y * (np.cos(psi) * ddpsi - np.sin(psi) * dpsi * dpsi)
        )
        ddy_center = (
            ddy
            + delta_x * (np.cos(psi) * ddpsi - np.sin(psi) * dpsi * dpsi)
            - delta_y * (np.sin(psi) * ddpsi + np.cos(psi) * dpsi * dpsi)
        )

        dpsi_center = dpsi
        ddpsi_center = ddpsi

        return dx_center, dy_center, dpsi_center, ddx_center, ddy_center, ddpsi_center

    # =========================
    # Centralized single-QP path (ungrouped)
    # =========================

    def build_centralized_cbf_qp(self):
        print(f"[INFO] Building centralized CBF-QP for env {self.env_idx}...")
        # Check is self._qp_cache exists
        if hasattr(self, "_qp_cache"):
            return
        n = int(self.parameters.n_agents)
        n_circles = int(self.parameters.n_circles_approximate_vehicle)
        self.n_agents = n

        # Decision variables
        u_var = cp.Variable((n, 2))
        s_bound = cp.Variable(2 * n * n_circles, nonneg=True)
        # CLF relaxations
        s_clf_head = cp.Variable(n, nonneg=True)
        s_clf_speed = cp.Variable(n, nonneg=True)

        # pair slacks
        pair_base = {}
        base = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                pair_base[(i, j)] = base
                base += n_circles * n_circles
        pair_count = base
        s_pair = cp.Variable(pair_count, nonneg=True) if pair_count > 0 else None

        def idx_bound(i_a, i_c, side):
            return 2 * (i_a * n_circles + i_c) + side

        def idx_pair(i_a, j_a, i_c, j_c):
            return pair_base[(i_a, j_a)] + i_c * n_circles + j_c

        cons = [
            self.a_min <= u_var[:, 0],
            u_var[:, 0] <= self.a_max,
            self.steering_rate_min <= u_var[:, 1],
            u_var[:, 1] <= self.steering_rate_max,
        ]

        # Parameters to update each step
        U_nom = cp.Parameter((n, 2))  # nominal accel, steer_rate

        # CLF parameters per agent
        e_head = cp.Parameter(n)  # heading error (desired_psi - psi)
        e_speed = cp.Parameter(n)  # speed error (v_ref - v)
        v_head_const = cp.Parameter(n)  # lam*0.5*e^2
        v_speed_const = cp.Parameter(n)  # lam*0.5*e^2

        # CLF constraints:
        # -e_head[i] * u_var[i,1] + v_head_const[i] - s_clf_head[i] <= 0
        # -e_speed[i]* u_var[i,0] + v_speed_const[i] - s_clf_speed[i] <= 0
        cons += [
            -e_head[i] * u_var[i, 1] + v_head_const[i] - s_clf_head[i] <= 0
            for i in range(n)
        ]
        cons += [
            -e_speed[i] * u_var[i, 0] + v_speed_const[i] - s_clf_speed[i] <= 0
            for i in range(n)
        ]

        # keep explicit handles to CBF safety constraints (lane + pairwise)
        cbf_lane_cons_L = [[None for _ in range(n_circles)] for _ in range(n)]
        cbf_lane_cons_R = [[None for _ in range(n_circles)] for _ in range(n)]
        cbf_pair_cons = {}  # (i, j, ci, cj) -> cvxpy.Constraint

        # Adaptive lambda variables for CBF constraints
        if self.parameters.adaptive_lambda:
            A_L = [[cp.Parameter((1, 2)) for _ in range(n_circles)] for _ in range(n)]
            b_L = [[cp.Parameter((1,)) for _ in range(n_circles)] for _ in range(n)]
            A_R = [[cp.Parameter((1, 2)) for _ in range(n_circles)] for _ in range(n)]
            b_R = [[cp.Parameter((1,)) for _ in range(n_circles)] for _ in range(n)]
            b0_L = [[cp.Parameter((1,)) for _ in range(n_circles)] for _ in range(n)]
            h_L = [[cp.Parameter((1,)) for _ in range(n_circles)] for _ in range(n)]
            b0_R = [[cp.Parameter((1,)) for _ in range(n_circles)] for _ in range(n)]
            h_R = [[cp.Parameter((1,)) for _ in range(n_circles)] for _ in range(n)]
            lambda_bound = cp.Variable(2 * n * n_circles)
            lambda_pair = cp.Variable(pair_count) if pair_count > 0 else None
            cons += [lambda_bound >= 0, lambda_bound <= 1]
            if lambda_pair is not None:
                cons += [lambda_pair >= 0, lambda_pair <= 1]
            for i in range(n):
                for c in range(n_circles):
                    cL = (
                        A_L[i][c] @ u_var[i, :]
                        + b0_L[i][c]
                        + h_L[i][c] * lambda_bound[idx_bound(i, c, 0)]
                        >= -s_bound[idx_bound(i, c, 0)]
                    )
                    cR = (
                        A_R[i][c] @ u_var[i, :]
                        + b0_R[i][c]
                        + h_R[i][c] * lambda_bound[idx_bound(i, c, 1)]
                        >= -s_bound[idx_bound(i, c, 1)]
                    )

                    cons += [cL, cR]
                    cbf_lane_cons_L[i][c] = cL
                    cbf_lane_cons_R[i][c] = cR

        else:
            A_L = [[cp.Parameter((1, 2)) for _ in range(n_circles)] for _ in range(n)]
            b_L = [[cp.Parameter((1,)) for _ in range(n_circles)] for _ in range(n)]
            A_R = [[cp.Parameter((1, 2)) for _ in range(n_circles)] for _ in range(n)]
            b_R = [[cp.Parameter((1,)) for _ in range(n_circles)] for _ in range(n)]
            for i in range(n):
                for c in range(n_circles):
                    cL = (
                        A_L[i][c] @ u_var[i, :] + b_L[i][c]
                        >= -s_bound[idx_bound(i, c, 0)]
                    )
                    cR = (
                        A_R[i][c] @ u_var[i, :] + b_R[i][c]
                        >= -s_bound[idx_bound(i, c, 1)]
                    )
                    cons += [cL, cR]
                    cbf_lane_cons_L[i][c] = cL
                    cbf_lane_cons_R[i][c] = cR

        # Vehicle pairwise CBF constraints
        if self.parameters.adaptive_lambda:
            Aij_i = {}
            Aij_j = {}
            b0ij = {}
            hij = {}
            for i in range(n - 1):
                for j in range(i + 1, n):
                    for ci in range(n_circles):
                        for cj in range(n_circles):
                            Aij_i[(i, j, ci, cj)] = cp.Parameter((1, 2))
                            Aij_j[(i, j, ci, cj)] = cp.Parameter((1, 2))
                            b0ij[(i, j, ci, cj)] = cp.Parameter((1,))
                            hij[(i, j, ci, cj)] = cp.Parameter((1,))
                            if s_pair is None:
                                cP = (
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + b0ij[(i, j, ci, cj)]
                                    + hij[(i, j, ci, cj)] * 1.0
                                    >= 0
                                )
                            else:
                                cP = (
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + b0ij[(i, j, ci, cj)]
                                    + hij[(i, j, ci, cj)]
                                    * lambda_pair[idx_pair(i, j, ci, cj)]
                                    >= -s_pair[idx_pair(i, j, ci, cj)]
                                )
                            cons.append(cP)
                            cbf_pair_cons[(i, j, ci, cj)] = cP

        else:
            Aij_i = {}
            Aij_j = {}
            bij = {}
            for i in range(n - 1):
                for j in range(i + 1, n):
                    for ci in range(n_circles):
                        for cj in range(n_circles):
                            Aij_i[(i, j, ci, cj)] = cp.Parameter((1, 2))
                            Aij_j[(i, j, ci, cj)] = cp.Parameter((1, 2))
                            bij[(i, j, ci, cj)] = cp.Parameter((1,))
                            if s_pair is None:
                                cP = (
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + bij[(i, j, ci, cj)]
                                    >= 0
                                )
                            else:
                                cP = (
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + bij[(i, j, ci, cj)]
                                    >= -s_pair[idx_pair(i, j, ci, cj)]
                                )

                            cons.append(cP)
                            cbf_pair_cons[(i, j, ci, cj)] = cP

        tracking = cp.sum_squares((u_var - U_nom) @ self.nom_weight)

        cost = tracking + self.lane_slack_weight * cp.sum_squares(s_bound)
        # CLF relaxation penalties
        cost += self.w_clf_relax * (
            cp.sum_squares(s_clf_head) + cp.sum_squares(s_clf_speed)
        )
        if s_pair is not None:
            cost += self.pair_slack_weight * cp.sum_squares(s_pair)

        if self.parameters.adaptive_lambda:
            cost += self.lambda_weight * cp.sum_squares(
                lambda_bound - self.lambda_ttcbf_predefined
            )
            if lambda_pair is not None:
                cost += self.lambda_weight * cp.sum_squares(lambda_pair)

        prob = cp.Problem(cp.Minimize(cost), cons)

        # Cache
        if self.parameters.adaptive_lambda:
            self._qp_cache = dict(
                n=n,
                n_circles=n_circles,
                u_var=u_var,
                s_bound=s_bound,
                s_pair=s_pair,
                s_clf_head=s_clf_head,
                s_clf_speed=s_clf_speed,
                U_nom=U_nom,
                e_head=e_head,
                e_speed=e_speed,
                v_head_const=v_head_const,
                v_speed_const=v_speed_const,
                A_L=A_L,
                b_L=b_L,
                A_R=A_R,
                b_R=b_R,
                b0_L=b0_L,
                h_L=h_L,
                b0_R=b0_R,
                h_R=h_R,
                Aij_i=Aij_i,
                Aij_j=Aij_j,
                b0ij=b0ij,
                hij=hij,
                lambda_bound=lambda_bound,
                lambda_pair=lambda_pair,
                idx_pair=idx_pair,
                pair_count=pair_count,
                cbf_lane_cons_L=cbf_lane_cons_L,
                cbf_lane_cons_R=cbf_lane_cons_R,
                cbf_pair_cons=cbf_pair_cons,
                prob=prob,
            )
        else:
            self._qp_cache = dict(
                n=n,
                n_circles=n_circles,
                u_var=u_var,
                s_bound=s_bound,
                s_pair=s_pair,
                s_clf_head=s_clf_head,
                s_clf_speed=s_clf_speed,
                U_nom=U_nom,
                e_head=e_head,
                e_speed=e_speed,
                v_head_const=v_head_const,
                v_speed_const=v_speed_const,
                A_L=A_L,
                b_L=b_L,
                A_R=A_R,
                b_R=b_R,
                Aij_i=Aij_i,
                Aij_j=Aij_j,
                bij=bij,
                idx_pair=idx_pair,
                pair_count=pair_count,
                cbf_lane_cons_L=cbf_lane_cons_L,
                cbf_lane_cons_R=cbf_lane_cons_R,
                cbf_pair_cons=cbf_pair_cons,
                prob=prob,
            )

        # Warm-start
        U_nom.value = np.zeros((n, 2))
        s_bound.value = np.zeros((2 * n * n_circles,))
        if s_pair is not None:
            s_pair.value = np.zeros((pair_count,))
        u_var.value = np.zeros((n, 2))
        self._qp_cache["s_clf_head"].value = np.zeros((n,))
        self._qp_cache["s_clf_speed"].value = np.zeros((n,))
        self._qp_cache["e_head"].value = np.zeros((n,))
        self._qp_cache["e_speed"].value = np.zeros((n,))
        self._qp_cache["v_head_const"].value = np.zeros((n,))
        self._qp_cache["v_speed_const"].value = np.zeros((n,))

    def update_centralized_cbf_qp(self, tensordict):
        # print(f"[INFO] Updating centralized CBF-QP for env {self.env_idx}...")
        # Requires self._qp_cache from build_centralized_cbf_qp
        C = self._qp_cache
        n, n_circles = C["n"], C["n_circles"]
        self.time_pseudo_dis = 0

        agents: list[Vehicle] = self.env.base_env.scenario_name.world.agents
        path_ids_t = tensordict["agents", "info", "path_id"][self.env_idx]
        path_ids = [int(path_ids_t[i].item()) for i in range(n)]

        # Buffers
        nominal_target_actions = torch.zeros(
            (n, 2), device=self.device, dtype=torch.float32
        )
        nom_acc_omega = np.zeros((n, 2), dtype=np.float64)

        # Precompute states and circle centers
        states = []
        circles_all = []
        for i in range(n):
            s = torch.cat(
                [
                    agents[i].state.pos[self.env_idx],
                    agents[i].state.rot[self.env_idx],
                    agents[i].state.speed[self.env_idx],
                    agents[i].state.steering[self.env_idx],
                ],
                dim=-1,
            )
            states.append(s)
            circles_all.append(self.get_circle_centers(s))

        learned_lambdas = self._get_env_cbf_lambdas(tensordict)

        # CLF errors and U_nom
        e_h_np = np.zeros((n,), dtype=np.float64)
        e_v_np = np.zeros((n,), dtype=np.float64)
        for i in range(n):
            (
                u_nom_i,
                target_action_i,
                e_h,
                e_v,
            ) = self._compute_nominal_control_for_agent(tensordict, states[i], i)
            nominal_target_actions[i, :] = target_action_i
            nom_acc_omega[i, :] = u_nom_i
            e_h_np[i] = e_h
            e_v_np[i] = e_v

        # Fill CLF parameters when in CLF mode; leave zeros for RL
        if self.nom_controller_type == "clf":
            C["e_head"].value = e_h_np
            C["e_speed"].value = e_v_np
            C["v_head_const"].value = self.lam_clf * 0.5 * (e_h_np**2)
            C["v_speed_const"].value = self.lam_clf * 0.5 * (e_v_np**2)
        else:
            C["e_head"].value = np.zeros(C["e_head"].shape)
            C["e_speed"].value = np.zeros(C["e_speed"].shape)
            C["v_head_const"].value = np.zeros(C["v_head_const"].shape)
            C["v_speed_const"].value = np.zeros(C["v_speed_const"].shape)

        # Fill lane and pairwise CBFs
        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe

        # Lane constraints
        for i in range(n):
            kins_i = self.linearized_center_kinematics_coeffs(states[i])
            for ci in range(n_circles):
                circle_pos_i = circles_all[i][ci][0:2]
                smL, gL, HL, smR, gR, HR = self.estimate_agent_2_lane_safety_margin(
                    circle_pos_i, path_ids[i]
                )
                if self.parameters.adaptive_lambda:
                    A_L_ic, b0_L_ic, h_L_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smL, gL, HL, self.dt_taylor, None
                    )
                    A_R_ic, b0_R_ic, h_R_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smR, gR, HR, self.dt_taylor, None
                    )
                    C["A_L"][i][ci].value = A_L_ic
                    C["b0_L"][i][ci].value = b0_L_ic
                    C["h_L"][i][ci].value = h_L_ic
                    C["A_R"][i][ci].value = A_R_ic
                    C["b0_R"][i][ci].value = b0_R_ic
                    C["h_R"][i][ci].value = h_R_ic
                else:
                    A_L_ic, b_L_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i,
                        ci,
                        smL,
                        gL,
                        HL,
                        self.dt_taylor,
                        self.lambda_ttcbf_predefined,
                    )
                    A_R_ic, b_R_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i,
                        ci,
                        smR,
                        gR,
                        HR,
                        self.dt_taylor,
                        self.lambda_ttcbf_predefined,
                    )
                    C["A_L"][i][ci].value = np.asarray(
                        A_L_ic, dtype=np.float64
                    ).reshape(1, 2)
                    C["b_L"][i][ci].value = np.asarray(
                        b_L_ic, dtype=np.float64
                    ).reshape(
                        1,
                    )
                    C["A_R"][i][ci].value = np.asarray(
                        A_R_ic, dtype=np.float64
                    ).reshape(1, 2)
                    C["b_R"][i][ci].value = np.asarray(
                        b_R_ic, dtype=np.float64
                    ).reshape(
                        1,
                    )

        # Pairwise constraints between agents
        for i in range(n - 1):
            for j in range(i + 1, n):
                if self.parameters.is_learn_cbf_lambda:
                    lam_i = self._get_agent_cbf_lambda(learned_lambdas, i)
                    lam_j = self._get_agent_cbf_lambda(learned_lambdas, j)
                    lambda_ttcbf = float((lam_i + lam_j) / 2.0)
                    # print(f"Learned lambda for pair ({i}, {j}): {lambda_ttcbf}")
                else:
                    lambda_ttcbf = self.lambda_ttcbf_predefined
                kins_i = self.linearized_center_kinematics_coeffs(states[i])
                kins_j = self.linearized_center_kinematics_coeffs(states[j])
                for ci in range(n_circles):
                    pi = circles_all[i][ci][0:2]
                    for cj in range(n_circles):
                        pj = circles_all[j][cj][0:2]
                        delta = pi - pj
                        dx = float(delta[0].item())
                        dy = float(delta[1].item())
                        if self.parameters.adaptive_lambda:
                            A_i, A_j, b0, h = self.ttcbf_pair_affine_coeffs(
                                kins_i,
                                kins_j,
                                ci,
                                cj,
                                dx,
                                dy,
                                d_safe_sq,
                                self.dt_taylor,
                                None,
                            )
                            C["Aij_i"][(i, j, ci, cj)].value = A_i
                            C["Aij_j"][(i, j, ci, cj)].value = A_j
                            C["b0ij"][(i, j, ci, cj)].value = b0
                            C["hij"][(i, j, ci, cj)].value = h
                        else:
                            A_i, A_j, b, h = self.ttcbf_pair_affine_coeffs(
                                kins_i,
                                kins_j,
                                ci,
                                cj,
                                dx,
                                dy,
                                d_safe_sq,
                                self.dt_taylor,
                                lambda_ttcbf,
                            )
                            C["Aij_i"][(i, j, ci, cj)].value = np.asarray(
                                A_i, dtype=np.float64
                            ).reshape(1, 2)
                            C["Aij_j"][(i, j, ci, cj)].value = np.asarray(
                                A_j, dtype=np.float64
                            ).reshape(1, 2)
                            C["bij"][(i, j, ci, cj)].value = np.asarray(
                                b, dtype=np.float64
                            ).reshape(
                                1,
                            )

        # Set U_nom and zeros
        C["U_nom"].value = nom_acc_omega

        C["s_bound"].value = np.zeros(C["s_bound"].shape)
        if C["s_pair"] is not None:
            C["s_pair"].value = np.zeros(C["s_pair"].shape)

        C["s_clf_head"].value = np.zeros(C["s_clf_head"].shape)
        C["s_clf_speed"].value = np.zeros(C["s_clf_speed"].shape)

        # Warm start
        C["u_var"].value = C["U_nom"].value.copy()
        if C["u_var"].value.shape != C["u_var"].shape:
            C["u_var"].value = np.zeros(C["u_var"].shape)

        # TODO Remove: instead of solving CBF, we check if the RL actions violate the QP constraints
        if self.parameters.is_solve_qp:
            # Solve
            prob: cp.Problem = C["prob"]
            invalid_param = self._find_invalid_qp_parameter(C, path="_qp_cache")

            solved = False
            if invalid_param is not None:
                print(
                    "[WARN] Skipping centralized OSQP solve due to invalid QP "
                    f"parameter at {invalid_param[0]}: {invalid_param[1]}"
                )
            else:
                try:
                    prob.solve(
                        solver=cp.OSQP,
                        warm_start=True,
                        max_iter=20000,
                        eps_abs=1e-5,
                        eps_rel=1e-5,
                        polish=True,
                        adaptive_rho=True,
                    )

                except cp.SolverError:
                    print("[WARN] OSQP solver failed, trying CLARABEL...")
                    try:
                        prob.solve(
                            solver=cp.CLARABEL,
                            warm_start=True,
                            tol_feas=1e-8,
                            tol_gap_abs=1e-8,
                            tol_gap_rel=1e-8,
                        )
                    except cp.SolverError:
                        print("[WARN] CLARABEL solver failed, trying SCS...")
                        try:
                            prob.solve(
                                solver=cp.SCS,
                                warm_start=True,
                                max_iters=100000,
                                eps=1e-4,
                            )
                        except cp.SolverError:
                            print(
                                "[ERROR] SCS solver also failed. Using nominal actions as fallback."
                            )

            solved = (
                prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
                and C["u_var"].value is not None
                and np.isfinite(C["u_var"].value).all()
            )
            if solved:
                self._record_centralized_slack_usage(C)

            if self.env.scenario_name.parameters.is_apply_cbf_action:
                # Only replace RL actions with CBF actions if is_apply_cbf_action is True
                if solved:
                    for i in range(n):
                        s_i = states[i]
                        actions_v_steer_safe = self.u_to_rl_action(
                            C["u_var"].value[i, :], s_i[3], s_i[4], agent_idx=i
                        ).detach()
                        tensordict[("agents", "action")][
                            self.env_idx, i
                        ] = actions_v_steer_safe
                else:
                    # Fallback: use nominal actions
                    for i in range(n):
                        s_i = states[i]
                        actions_v_steer_nom = self.u_to_rl_action(
                            nom_acc_omega[i, :], s_i[3], s_i[4], agent_idx=i
                        ).detach()
                        tensordict[("agents", "action")][
                            self.env_idx, i
                        ] = actions_v_steer_nom
                    self.hist.evt.append("QP-INF")

            applied_target_actions = self._target_actions_from_current_actions(
                tensordict, states
            ).detach()
            self.env.base_env.scenario_name.world_state.applied_action_vel[
                self.env_idx, :
            ] = applied_target_actions[:, 0]
            self.env.base_env.scenario_name.world_state.applied_action_steer[
                self.env_idx, :
            ] = applied_target_actions[:, 1]

            st = prob.solver_stats
            if hasattr(st, "solve_time"):
                self.hist.qp_solving_t.append(st.solve_time)
                # print(
                #     f"[INFO] Centralized QP solving time: {st.solve_time * 1000:.2f} ms. Over all time steps: {np.mean(self.hist.qp_solving_t)*1000:.2f} ms."
                # )
            if hasattr(st, "num_iters"):
                self.hist.qp_solving_iter.append(st.num_iters)

            # if self.parameters.adaptive_lambda and C["lambda_pair"] is not None:
            #     print(f"lambda_bound mean {np.mean(C['lambda_bound'].value):.4f}; max {np.max(C['lambda_bound'].value):.4f}; min {np.min(C['lambda_bound'].value):.4f}")
            #     print(f"lambda_pair mean {np.mean(C['lambda_pair'].value):.4f}; max {np.max(C['lambda_pair'].value):.4f}; min {np.min(C['lambda_pair'].value):.4f}")

            # For visualization of nominal actions
            if self.env.scenario_name.parameters.is_apply_cbf_action:
                self.env.base_env.scenario_name.world_state.nominal_action_vel[
                    self.env_idx, :
                ] = nominal_target_actions[:, 0]
                self.env.base_env.scenario_name.world_state.nominal_action_steer[
                    self.env_idx, :
                ] = nominal_target_actions[:, 1]
            else:
                # Store the CBF actions
                if solved:
                    compare_target_actions = torch.zeros_like(nominal_target_actions)
                    for i in range(n):
                        s_i = states[i]
                        compare_target_actions[i] = self.u_to_target_action(
                            C["u_var"].value[i, :], s_i[3], s_i[4]
                        ).detach()
                else:
                    compare_target_actions = nominal_target_actions
                    self.hist.evt.append("QP-INF")

                self.env.base_env.scenario_name.world_state.nominal_action_vel[
                    self.env_idx, :
                ] = compare_target_actions[:, 0]
                self.env.base_env.scenario_name.world_state.nominal_action_steer[
                    self.env_idx, :
                ] = compare_target_actions[:, 1]
        else:
            print(f"Env {self.env_idx}")
            if self.parameters.adaptive_lambda:
                # Ensure lambda variables have values before reading constraint.expr.value
                if C["lambda_bound"].value is None:
                    C["lambda_bound"].value = np.full(
                        C["lambda_bound"].shape,
                        float(self.lambda_ttcbf_predefined),
                        dtype=np.float64,
                    )
                if (
                    C.get("lambda_pair", None) is not None
                    and C["lambda_pair"].value is None
                ):
                    C["lambda_pair"].value = np.full(
                        C["lambda_pair"].shape,
                        float(self.lambda_ttcbf_predefined),
                        dtype=np.float64,
                    )

            # Read constraint.expr.value for CBF constraints only (lane + pairwise).
            self.nominal_cbf_constraint_values = (
                self._get_nominal_cbf_constraint_values(tol=1e-9)
            )

            r_left, r_right, r_pair = self.compute_nominal_cbf_violation_rewards()
            self.env.base_env.scenario_name.reward_info.rew_near_left_lane[
                self.env_idx, :
            ] = torch.tensor(r_left, device=self.device, dtype=torch.float32)
            self.env.base_env.scenario_name.reward_info.rew_near_right_lane[
                self.env_idx, :
            ] = torch.tensor(r_right, device=self.device, dtype=torch.float32)
            self.env.base_env.scenario_name.reward_info.rew_near_other_agents[
                self.env_idx, :
            ] = torch.tensor(r_pair, device=self.device, dtype=torch.float32)

        # print(f"obj: {prob.objective.value}")

        # print(f"abs vel: {(self.env.base_env.scenario_name.world_state.applied_action_vel[self.env_idx, :] - self.env.base_env.scenario_name.world_state.nominal_action_vel[self.env_idx, :]).abs()}")
        # print(f"abs steer: {(self.env.base_env.scenario_name.world_state.applied_action_steer[self.env_idx, :] - self.env.base_env.scenario_name.world_state.nominal_action_steer[self.env_idx, :]).abs()}")

    def build_fully_decentralized_cbf_qps(self):
        print(
            f"[INFO] Building fully decentralized CBF-QP for env {self.env_idx}, "
            f"agent {self.agent_idx}..."
        )
        if self.agent_idx is None:
            raise ValueError("Fully decentralized CBF-QP requires agent_idx to be set.")
        if hasattr(self, "_fd_qp_cache"):
            return

        n_agents = int(self.parameters.n_agents)
        n_circles = int(self.parameters.n_circles_approximate_vehicle)

        e_cap_user = getattr(self.parameters, "max_cross_neighbors", None)
        if e_cap_user is None:
            e_cap = max(0, n_agents - 1)
        else:
            e_cap = min(max(0, n_agents - 1), max(0, int(e_cap_user)))

        def idx_lane(i_c, side):
            return 2 * i_c + side

        cross_base = {}
        cross_count = 0
        for e_idx in range(e_cap):
            cross_base[e_idx] = cross_count
            cross_count += n_circles * n_circles

        u_var = cp.Variable(2)
        s_lane = cp.Variable(2 * n_circles, nonneg=True)
        s_cross = cp.Variable(cross_count, nonneg=True) if cross_count > 0 else None
        s_clf_head = cp.Variable(nonneg=True)
        s_clf_speed = cp.Variable(nonneg=True)

        cons = [
            self.a_min <= u_var[0],
            u_var[0] <= self.a_max,
            self.steering_rate_min <= u_var[1],
            u_var[1] <= self.steering_rate_max,
        ]

        u_nom = cp.Parameter(2)
        e_head = cp.Parameter()
        e_speed = cp.Parameter()
        v_head_const = cp.Parameter()
        v_speed_const = cp.Parameter()

        cons += [-e_head * u_var[1] + v_head_const - s_clf_head <= 0]
        cons += [-e_speed * u_var[0] + v_speed_const - s_clf_speed <= 0]

        a_l = [cp.Parameter((1, 2)) for _ in range(n_circles)]
        a_r = [cp.Parameter((1, 2)) for _ in range(n_circles)]
        if self.parameters.adaptive_lambda:
            b0_l = [cp.Parameter((1,)) for _ in range(n_circles)]
            h_l = [cp.Parameter((1,)) for _ in range(n_circles)]
            b0_r = [cp.Parameter((1,)) for _ in range(n_circles)]
            h_r = [cp.Parameter((1,)) for _ in range(n_circles)]
            b_l = None
            b_r = None
            lambda_lane = cp.Variable(2 * n_circles)
            lambda_lane_ref = cp.Parameter(2 * n_circles)
            cons += [lambda_lane >= 0, lambda_lane <= 1]
            if self.parameters.is_learn_cbf_lambda:
                cons += [lambda_lane == lambda_lane_ref]

            for ci in range(n_circles):
                cons += [
                    a_l[ci] @ u_var + b0_l[ci] + h_l[ci] * lambda_lane[idx_lane(ci, 0)]
                    >= -s_lane[idx_lane(ci, 0)]
                ]
                cons += [
                    a_r[ci] @ u_var + b0_r[ci] + h_r[ci] * lambda_lane[idx_lane(ci, 1)]
                    >= -s_lane[idx_lane(ci, 1)]
                ]
        else:
            b_l = [cp.Parameter((1,)) for _ in range(n_circles)]
            b_r = [cp.Parameter((1,)) for _ in range(n_circles)]
            b0_l = None
            h_l = None
            b0_r = None
            h_r = None
            lambda_lane = None
            lambda_lane_ref = None

            for ci in range(n_circles):
                cons += [a_l[ci] @ u_var + b_l[ci] >= -s_lane[idx_lane(ci, 0)]]
                cons += [a_r[ci] @ u_var + b_r[ci] >= -s_lane[idx_lane(ci, 1)]]

        cross_uses_symbolic_lambda = (
            self.parameters.adaptive_lambda and not self.parameters.is_learn_cbf_lambda
        )
        axc_i = {}
        if cross_uses_symbolic_lambda:
            b0xc = {}
            hxc = {}
            bxc = None
            lambdax_i = cp.Variable(cross_count) if cross_count > 0 else None
            if lambdax_i is not None:
                cons += [lambdax_i >= 0, lambdax_i <= 1]
        else:
            bxc = {}
            b0xc = None
            hxc = None
            lambdax_i = None

        if cross_count > 0:
            for e_idx in range(e_cap):
                for ci in range(n_circles):
                    for cj in range(n_circles):
                        axc_i[(e_idx, ci, cj)] = cp.Parameter((1, 2))
                        if cross_uses_symbolic_lambda:
                            b0xc[(e_idx, ci, cj)] = cp.Parameter((1,))
                            hxc[(e_idx, ci, cj)] = cp.Parameter((1,))
                            cons += [
                                axc_i[(e_idx, ci, cj)] @ u_var
                                + self._cross_cbf_drift_scale()
                                * (
                                    b0xc[(e_idx, ci, cj)]
                                    + hxc[(e_idx, ci, cj)]
                                    * lambdax_i[cross_base[e_idx] + ci * n_circles + cj]
                                )
                                >= -s_cross[cross_base[e_idx] + ci * n_circles + cj]
                            ]
                        else:
                            bxc[(e_idx, ci, cj)] = cp.Parameter((1,))
                            cons += [
                                axc_i[(e_idx, ci, cj)] @ u_var + bxc[(e_idx, ci, cj)]
                                >= -s_cross[cross_base[e_idx] + ci * n_circles + cj]
                            ]

        cost = cp.quad_form(u_var - u_nom, self.nom_weight)
        cost += self.lane_slack_weight * cp.sum_squares(s_lane)
        if s_cross is not None:
            cost += self.cross_slack_weight * cp.sum_squares(s_cross)
        cost += self.w_clf_relax * (cp.square(s_clf_head) + cp.square(s_clf_speed))

        if self.parameters.adaptive_lambda:
            cost += self.lambda_weight * cp.sum_squares(
                lambda_lane - self.lambda_ttcbf_predefined
            )
        if lambdax_i is not None:
            cost += self.lambda_weight * cp.sum_squares(
                lambdax_i - float(self.lambda_ttcbf_predefined)
            )

        prob = cp.Problem(cp.Minimize(cost), cons)

        self._fd_qp_cache = dict(
            n_agents=n_agents,
            n_circles=n_circles,
            e_cap=e_cap,
            cross_base=cross_base,
            cross_count=cross_count,
            u_var=u_var,
            U_nom=u_nom,
            s_lane=s_lane,
            s_cross=s_cross,
            s_clf_head=s_clf_head,
            s_clf_speed=s_clf_speed,
            e_head=e_head,
            e_speed=e_speed,
            v_head_const=v_head_const,
            v_speed_const=v_speed_const,
            A_L=a_l,
            b_L=b_l,
            b0_L=b0_l,
            h_L=h_l,
            A_R=a_r,
            b_R=b_r,
            b0_R=b0_r,
            h_R=h_r,
            lambda_lane=lambda_lane,
            lambda_lane_ref=lambda_lane_ref,
            Axc_i=axc_i,
            bxc=bxc,
            b0xc=b0xc,
            hxc=hxc,
            lambdax_i=lambdax_i,
            idx_lane=idx_lane,
            prob=prob,
        )

        u_nom.value = np.zeros((2,), dtype=np.float64)
        s_lane.value = np.zeros((2 * n_circles,), dtype=np.float64)
        if s_cross is not None:
            s_cross.value = np.zeros((cross_count,), dtype=np.float64)
        s_clf_head.value = 0.0
        s_clf_speed.value = 0.0
        e_head.value = 0.0
        e_speed.value = 0.0
        v_head_const.value = 0.0
        v_speed_const.value = 0.0
        u_var.value = np.zeros((2,), dtype=np.float64)
        if lambda_lane is not None:
            lambda_lane.value = np.full(
                lambda_lane.shape, float(self.lambda_ttcbf_predefined), dtype=np.float64
            )
            lambda_lane_ref.value = np.full(
                lambda_lane_ref.shape,
                float(self.lambda_ttcbf_predefined),
                dtype=np.float64,
            )
        if lambdax_i is not None:
            lambdax_i.value = np.full(
                lambdax_i.shape, float(self.lambda_ttcbf_predefined), dtype=np.float64
            )

    def update_fully_decentralized_cbf_qps(self, tensordict):
        assert hasattr(
            self, "_fd_qp_cache"
        ), "Call build_fully_decentralized_cbf_qps first."
        if self.agent_idx is None:
            raise ValueError("Fully decentralized CBF-QP requires agent_idx to be set.")

        cache = self._fd_qp_cache
        i = int(self.agent_idx)
        n_agents = int(cache["n_agents"])
        n_circles = int(cache["n_circles"])
        e_cap = int(cache["e_cap"])
        cbf_coupling_range = float(self.parameters.cbf_coupling_range)

        agents: List[Vehicle] = self.env.base_env.scenario_name.world.agents
        path_ids_t = tensordict["agents", "info", "path_id"][self.env_idx]
        path_ids = [int(path_ids_t[k].item()) for k in range(n_agents)]

        states = []
        centers = []
        centers_flat = []
        for k in range(n_agents):
            s = torch.cat(
                [
                    agents[k].state.pos[self.env_idx],
                    agents[k].state.rot[self.env_idx],
                    agents[k].state.speed[self.env_idx],
                    agents[k].state.steering[self.env_idx],
                ],
                dim=-1,
            )
            states.append(s)
            centers.append(self.get_circle_centers(s))
            centers_flat.append(
                np.array([float(s[0].item()), float(s[1].item())], dtype=np.float64)
            )

        learned_lambdas = self._get_env_cbf_lambdas(tensordict)

        (
            u_nom_np,
            nominal_target_i,
            e_head_i,
            e_speed_i,
        ) = self._compute_nominal_control_for_agent(tensordict, states[i], i)
        v_head_const_i = self.lam_clf * 0.5 * (e_head_i**2)
        v_speed_const_i = self.lam_clf * 0.5 * (e_speed_i**2)

        cache["U_nom"].value = u_nom_np
        cache["e_head"].value = float(e_head_i)
        cache["e_speed"].value = float(e_speed_i)
        cache["v_head_const"].value = float(v_head_const_i)
        cache["v_speed_const"].value = float(v_speed_const_i)

        kins_i = self.linearized_center_kinematics_coeffs(states[i])
        lane_lambda_ref = None
        if cache["lambda_lane"] is not None:
            lane_lambda_ref = np.full(
                cache["lambda_lane"].shape,
                float(self.lambda_ttcbf_predefined),
                dtype=np.float64,
            )
        # Lane constraints
        for ci in range(n_circles):
            p_ic = centers[i][ci][0:2]
            sm_l, g_l, h_l, sm_r, g_r, h_r = self.estimate_agent_2_lane_safety_margin(
                p_ic, path_ids[i]
            )
            if self.parameters.adaptive_lambda:
                a_l_ic, b0_l_ic, h_l_ic = self.ttcbf_lane_affine_coeffs(
                    kins_i, ci, sm_l, g_l, h_l, self.dt_taylor, None
                )
                a_r_ic, b0_r_ic, h_r_ic = self.ttcbf_lane_affine_coeffs(
                    kins_i, ci, sm_r, g_r, h_r, self.dt_taylor, None
                )
                cache["A_L"][ci].value = a_l_ic
                cache["b0_L"][ci].value = b0_l_ic
                cache["h_L"][ci].value = h_l_ic
                cache["A_R"][ci].value = a_r_ic
                cache["b0_R"][ci].value = b0_r_ic
                cache["h_R"][ci].value = h_r_ic
            else:
                a_l_ic, b_l_ic = self.ttcbf_lane_affine_coeffs(
                    kins_i,
                    ci,
                    sm_l,
                    g_l,
                    h_l,
                    self.dt_taylor,
                    self.lambda_ttcbf_predefined,
                )
                a_r_ic, b_r_ic = self.ttcbf_lane_affine_coeffs(
                    kins_i,
                    ci,
                    sm_r,
                    g_r,
                    h_r,
                    self.dt_taylor,
                    self.lambda_ttcbf_predefined,
                )
                cache["A_L"][ci].value = np.asarray(a_l_ic, dtype=np.float64).reshape(
                    1, 2
                )
                cache["b_L"][ci].value = np.asarray(b_l_ic, dtype=np.float64).reshape(
                    1,
                )
                cache["A_R"][ci].value = np.asarray(a_r_ic, dtype=np.float64).reshape(
                    1, 2
                )
                cache["b_R"][ci].value = np.asarray(b_r_ic, dtype=np.float64).reshape(
                    1,
                )

        nbrs: List[int] = []
        for j in range(n_agents):
            if j == i:
                continue
            if _euclid(centers_flat[i], centers_flat[j]) <= cbf_coupling_range + 1e-9:
                nbrs.append(j)
        nbrs = nbrs[:e_cap]

        cross_lambda_ref = None
        if cache["lambdax_i"] is not None:
            cross_lambda_ref = np.full(
                cache["lambdax_i"].shape,
                float(self.lambda_ttcbf_predefined),
                dtype=np.float64,
            )

        # Cross constraints with neighbors
        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe
        for e_idx in range(e_cap):
            if e_idx < len(nbrs):
                j = nbrs[e_idx]
                kins_j = self.linearized_center_kinematics_coeffs(states[j])
                for ci in range(n_circles):
                    for cj in range(n_circles):
                        lam_cross = (
                            self._get_agent_cbf_lambda(learned_lambdas, i)
                            if self.parameters.is_learn_cbf_lambda
                            else float(self.lambda_ttcbf_predefined)
                        )
                        pi = centers[i][ci][0:2]
                        pj = centers[j][cj][0:2]
                        dx = float((pi - pj)[0].item())
                        dy = float((pi - pj)[1].item())
                        if cache["lambdax_i"] is not None:
                            cross_lambda_ref[
                                cache["cross_base"][e_idx] + ci * n_circles + cj
                            ] = lam_cross
                            a_i, b0, h = self.ttcbf_pair_affine_coeffs_cross(
                                kins_i,
                                kins_j,
                                ci,
                                cj,
                                dx,
                                dy,
                                d_safe_sq,
                                self.dt_taylor,
                                None,
                            )
                            cache["Axc_i"][(e_idx, ci, cj)].value = a_i
                            cache["b0xc"][(e_idx, ci, cj)].value = b0
                            cache["hxc"][(e_idx, ci, cj)].value = h
                        else:
                            a_i, b = self.ttcbf_pair_affine_coeffs_cross(
                                kins_i,
                                kins_j,
                                ci,
                                cj,
                                dx,
                                dy,
                                d_safe_sq,
                                self.dt_taylor,
                                lam_cross,
                            )
                            cache["Axc_i"][(e_idx, ci, cj)].value = a_i
                            cache["bxc"][(e_idx, ci, cj)].value = b
            else:
                for ci in range(n_circles):
                    for cj in range(n_circles):
                        cache["Axc_i"][(e_idx, ci, cj)].value = np.zeros((1, 2))
                        if cache["lambdax_i"] is not None:
                            cache["b0xc"][(e_idx, ci, cj)].value = np.array([1.0])
                            cache["hxc"][(e_idx, ci, cj)].value = np.array([0.0])
                        else:
                            cache["bxc"][(e_idx, ci, cj)].value = np.array([1.0])

        cache["s_lane"].value = np.zeros(cache["s_lane"].shape)
        if cache["s_cross"] is not None:
            cache["s_cross"].value = np.zeros(cache["s_cross"].shape)
        cache["s_clf_head"].value = 0.0
        cache["s_clf_speed"].value = 0.0
        cache["u_var"].value = cache["U_nom"].value.copy()

        if cache["lambda_lane"] is not None:
            cache["lambda_lane_ref"].value = lane_lambda_ref
            cache["lambda_lane"].value = (
                lane_lambda_ref.copy()
                if self.parameters.is_learn_cbf_lambda
                else np.full(
                    cache["lambda_lane"].shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
            )
        if cache["lambdax_i"] is not None:
            cache["lambdax_i"].value = (
                cross_lambda_ref.copy()
                if self.parameters.is_learn_cbf_lambda
                else np.full(
                    cache["lambdax_i"].shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
            )

        prob: cp.Problem = cache["prob"]
        invalid_param = self._find_invalid_qp_parameter(
            cache, path=f"_fd_qp_cache[agent={i}]"
        )
        if invalid_param is not None:
            print(
                "[WARN] Skipping fully decentralized OSQP solve due to invalid QP "
                f"parameter at {invalid_param[0]}: {invalid_param[1]}"
            )
        else:
            try:
                prob.solve(
                    solver=cp.OSQP,
                    warm_start=True,
                    max_iter=20000,
                    eps_abs=1e-5,
                    eps_rel=1e-5,
                    polish=True,
                    adaptive_rho=True,
                )
            except (ValueError, cp.SolverError):
                print(
                    "[WARN] Fully decentralized OSQP solve failed, trying CLARABEL..."
                )
                try:
                    prob.solve(
                        solver=cp.CLARABEL,
                        warm_start=True,
                        tol_feas=1e-8,
                        tol_gap_abs=1e-8,
                        tol_gap_rel=1e-8,
                    )
                except cp.SolverError:
                    print("[WARN] CLARABEL solver failed, trying SCS...")
                    try:
                        prob.solve(
                            solver=cp.SCS,
                            warm_start=True,
                            max_iters=100000,
                            eps=1e-4,
                        )
                    except cp.SolverError:
                        print(
                            "[ERROR] SCS solver also failed. Using nominal actions as fallback."
                        )

        solved = (
            prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
            and cache["u_var"].value is not None
            and np.isfinite(cache["u_var"].value).all()
        )

        if solved:
            u_safe = cache["u_var"].value
            action_safe = self.u_to_rl_action(
                u_safe, states[i][3], states[i][4], agent_idx=i
            ).detach()
            tensordict[("agents", "action")][self.env_idx, i] = action_safe
        else:
            action_nom = self.u_to_rl_action(
                u_nom_np, states[i][3], states[i][4], agent_idx=i
            ).detach()
            tensordict[("agents", "action")][self.env_idx, i] = action_nom
            self.hist.evt.append("QP-INF")

        applied_target_i = self._target_actions_from_current_actions(
            tensordict, states
        )[i].detach()
        self.env.base_env.scenario_name.world_state.applied_action_vel[
            self.env_idx, i
        ] = applied_target_i[0]
        self.env.base_env.scenario_name.world_state.applied_action_steer[
            self.env_idx, i
        ] = applied_target_i[1]

        self.env.base_env.scenario_name.world_state.nominal_action_vel[
            self.env_idx, i
        ] = nominal_target_i[0]
        self.env.base_env.scenario_name.world_state.nominal_action_steer[
            self.env_idx, i
        ] = nominal_target_i[1]

        if i == 0:
            self.scenario.inter_groups = [[k] for k in range(n_agents)]
            self.scenario.cross_groups = {}
        if hasattr(self.scenario, "cross_groups") and isinstance(
            self.scenario.cross_groups, dict
        ):
            self.scenario.cross_groups[i] = nbrs

        st = prob.solver_stats
        if hasattr(st, "solve_time"):
            self.hist.qp_solving_t.append(st.solve_time)
        if hasattr(st, "num_iters"):
            self.hist.qp_solving_iter.append(st.num_iters)

    def _get_nominal_cbf_constraint_values(self, tol: float = 1e-9):
        """
        Evaluate CBF constraint residuals at the current variable values

        Returns:
            A dict with:
            - lane_L_expr[i, c], lane_R_expr[i, c]: constraint.expr.value (float)
            - pair_expr[(i, j, ci, cj)]: constraint.expr.value (float)
            - satisfied_* booleans using expr <= tol convention (CVXPY inequality canonical form)
        """

        if self.parameters.is_grouping_agents:
            raise ValueError("Not implemented for grouped QPs")

        C = self._qp_cache
        n, n_circles = C["n"], C["n_circles"]

        def _to_float(v):
            if v is None:
                return np.nan
            return float(np.asarray(v).squeeze())

        lane_L_expr = np.zeros((n, n_circles), dtype=np.float64)
        lane_R_expr = np.zeros((n, n_circles), dtype=np.float64)
        lane_L_ok = np.zeros((n, n_circles), dtype=bool)
        lane_R_ok = np.zeros((n, n_circles), dtype=bool)

        # Lane CBFs
        for i in range(n):
            for c in range(n_circles):
                cL = C["cbf_lane_cons_L"][i][c]
                cR = C["cbf_lane_cons_R"][i][c]

                vL = _to_float(cL.expr.value)
                vR = _to_float(cR.expr.value)

                lane_L_expr[i, c] = vL
                lane_R_expr[i, c] = vR

                # CVXPY stores inequalities in canonical form expr <= 0.
                # So "satisfied" means expr.value <= tol.
                lane_L_ok[i, c] = vL <= tol
                lane_R_ok[i, c] = vR <= tol

        # Pairwise CBFs
        pair_expr = {}
        pair_ok = {}
        for key, cP in C["cbf_pair_cons"].items():
            vP = _to_float(cP.expr.value)
            pair_expr[key] = vP
            pair_ok[key] = vP <= tol

        return dict(
            lane_L_expr=lane_L_expr,
            lane_R_expr=lane_R_expr,
            lane_L_ok=lane_L_ok,
            lane_R_ok=lane_R_ok,
            pair_expr=pair_expr,
            pair_ok=pair_ok,
        )

    def compute_nominal_cbf_violation_rewards(
        self,
        cbf_values: dict | None = None,
        agg: str = "max",
        eps: float = 1e-9,
    ):
        """
        Compute per-agent reward signals from CBF constraint violations at the nominal action.

        Reward definition (per signal):
        - If a constraint is violated: reward is negative.
        - More violation => more negative.
        - If not violated: reward is exactly 0.
        - Normalized to roughly [-1, 0] using a data-driven scale (worst violation in this step).

        Inputs:
        cbf_values:
            Dict returned by your nominal CBF evaluation, e.g. self.nominal_cbf_constraint_values.
            If None, this function uses self.nominal_cbf_constraint_values.
        agg:
            How to aggregate multiple violations per agent:
            - "max": uses the worst (largest) violation (recommended; insensitive to #constraints)
            - "sum": uses the sum of violations
        eps:
            Numerical epsilon.

        Returns:
        r_left, r_right, r_pair:
            Three Python lists, each of length n_agents, with values in [-1, 0].
        """
        if cbf_values is None:
            cbf_values = self.nominal_cbf_constraint_values

        lane_L_expr = np.asarray(
            cbf_values["lane_L_expr"], dtype=np.float64
        )  # (n, n_circles)
        lane_R_expr = np.asarray(
            cbf_values["lane_R_expr"], dtype=np.float64
        )  # (n, n_circles)
        pair_expr = cbf_values[
            "pair_expr"
        ]  # dict keyed by (i, j, ci, cj) -> scalar residual

        n = lane_L_expr.shape[0]

        # Helper: positive part, because expr > 0 indicates violation in CVXPY canonical form.
        def pos(x):
            return np.maximum(x, 0.0)

        # ---------- Lane violation magnitudes per agent ----------
        vL = pos(lane_L_expr)  # (n, n_circles)
        vR = pos(lane_R_expr)  # (n, n_circles)

        if agg == "sum":
            vL_agent = np.sum(vL, axis=1)  # (n,)
            vR_agent = np.sum(vR, axis=1)  # (n,)
        elif agg == "max":
            vL_agent = np.max(vL, axis=1)  # (n,)
            vR_agent = np.max(vR, axis=1)  # (n,)
        else:
            raise ValueError(f"Unknown agg='{agg}'. Use 'max' or 'sum'.")

        # ---------- Pairwise violation magnitudes per agent ----------
        vP_agent = np.zeros((n,), dtype=np.float64)

        # Accumulate violation to both agents involved in each pairwise constraint
        # Key format: (i, j, ci, cj)
        for (i, j, ci, cj), expr_val in pair_expr.items():
            v = max(
                float(expr_val), 0.0
            )  # violation magnitude for this specific circle-circle constraint
            if agg == "sum":
                vP_agent[i] += v
                vP_agent[j] += v
            else:  # agg == "max"
                vP_agent[i] = max(vP_agent[i], v)
                vP_agent[j] = max(vP_agent[j], v)

        # ---------- Normalization (step-wise, per signal) ----------
        # We normalize each reward channel by its maximum violation in this step, so outputs are in [-1, 0].
        # If there are no violations in a channel, scale becomes ~0 and we return exactly zeros.
        scale_L = float(np.max(vL_agent)) + eps
        scale_R = float(np.max(vR_agent)) + eps
        scale_P = float(np.max(vP_agent)) + eps

        # Convert violation magnitudes into rewards:
        #   - No violation => 0
        #   - Violation v => negative, with worst violation near -1
        def to_reward(v_agent, scale):
            v_norm = v_agent / scale
            v_norm = np.clip(v_norm, 0.0, 1.0)  # bounded normalization
            return (-v_norm).tolist()

        r_left = to_reward(vL_agent, scale_L)
        r_right = to_reward(vR_agent, scale_R)
        r_pair = to_reward(vP_agent, scale_P)

        return r_left, r_right, r_pair

    # =========================
    # Grouped multi-QP path
    # =========================

    def _get_env_cbf_lambdas(self, tensordict):
        if not self.parameters.is_learn_cbf_lambda:
            return None

        lambda_env = tensordict.get(("agents", "cbf", "lambda"))[self.env_idx]
        lambda_env = lambda_env.squeeze(-1).detach().cpu().numpy()
        lambda_env = np.asarray(lambda_env, dtype=np.float64)

        invalid_mask = ~np.isfinite(lambda_env)
        if invalid_mask.any():
            if not self._warned_invalid_cbf_lambda:
                invalid_agents = np.nonzero(invalid_mask)[0].tolist()
                print(
                    "[WARN] Non-finite learned CBF lambda detected in "
                    f"env {self.env_idx} for agents {invalid_agents}. "
                    f"Falling back to lambda_ttcbf={self.lambda_ttcbf_predefined:.3f}."
                )
                self._warned_invalid_cbf_lambda = True
            lambda_env = lambda_env.copy()
            lambda_env[invalid_mask] = float(self.lambda_ttcbf_predefined)

        return np.clip(lambda_env, 0.0, 1.0)

    def _get_agent_cbf_lambda(self, learned_lambdas, agent_idx: int) -> float:
        if learned_lambdas is None:
            return float(self.lambda_ttcbf_predefined)
        return float(learned_lambdas[agent_idx])

    def _cross_cbf_drift_scale(self) -> float:
        """Return the half-split factor for decentralized pair CBF terms.

        For one circle pair, the centralized Taylor CBF inequality has the
        affine form

            A_i u_i + A_j u_j + b0 + lambda h >= 0,

        where ``b0`` contains the state-only Taylor drift terms
        ``dot_h * Dt + 0.5 * Dt**2 * ddot_h_0`` and the control inputs only
        enter through ``A_i u_i`` and ``A_j u_j``.  The fully decentralized
        cross-agent constraints assign each agent only its own control block:

            A_i u_i + 0.5 * (b0 + lambda h) >= 0,
            A_j u_j + 0.5 * (b0 + lambda h) >= 0.

        Summing the two directed constraints recovers the centralized
        inequality, so satisfying both decentralized constraints is a
        sufficient condition for satisfying the centralized pair constraint.
        This split is independent of whether ``lambda`` is learned or fixed;
        learning changes the value of ``lambda``, not the algebraic split.
        """
        return 0.5

    def _record_centralized_slack_usage(self, C: dict) -> None:
        lane_slack = C["s_bound"].value
        pair_slack = None if C["s_pair"] is None else C["s_pair"].value

        lane_max = (
            float(np.max(lane_slack))
            if lane_slack is not None and np.size(lane_slack) > 0
            else 0.0
        )
        pair_max = (
            float(np.max(pair_slack))
            if pair_slack is not None and np.size(pair_slack) > 0
            else 0.0
        )

        self.hist.max_lane_slack.append(lane_max)
        self.hist.max_pair_slack.append(pair_max)

        if (
            not self._warned_positive_centralized_slack
            and max(lane_max, pair_max) > 1e-3
        ):
            print(
                "[WARN] Centralized CBF-QP required positive slack in "
                f"env {self.env_idx}: lane={lane_max:.3e}, pair={pair_max:.3e}. "
                "OSQP success does not imply hard safety when slack is active."
            )
            self._warned_positive_centralized_slack = True

    def _find_invalid_qp_parameter(self, obj, path="cache"):
        if isinstance(obj, cp.Parameter):
            if obj.value is None:
                return path, "value is None"
            arr = np.asarray(obj.value, dtype=np.float64)
            if not np.isfinite(arr).all():
                bad_idx = np.argwhere(~np.isfinite(arr))
                bad_suffix = ""
                if bad_idx.size > 0:
                    bad_suffix = (
                        f", first_bad_index={tuple(int(i) for i in bad_idx[0])}"
                    )
                return (
                    path,
                    "non-finite value"
                    f"{bad_suffix}, shape={arr.shape}, sample={arr.reshape(-1)[:4]}",
                )
            return None

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"prob", "pair_base", "cross_base", "idx_lane"}:
                    continue
                key_str = str(key)
                result = self._find_invalid_qp_parameter(
                    value, path=f"{path}[{key_str}]"
                )
                if result is not None:
                    return result
            return None

        if isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                result = self._find_invalid_qp_parameter(value, path=f"{path}[{idx}]")
                if result is not None:
                    return result
            return None

        return None

    def _debug_osqp_bound_update_failure(self, prob: cp.Problem, context: dict):
        print(
            "[DEBUG] OSQP bound update failure context: "
            f"env={context.get('env_idx')}, group={context.get('group_idx')}, "
            f"members={context.get('members')}, slot2id={context.get('slot2id')}, "
            f"cross_neighbors={context.get('cross_neighbors')}"
        )

        try:
            problem_data = prob.get_problem_data(cp.OSQP)
            data = problem_data[0] if isinstance(problem_data, tuple) else problem_data
        except Exception as exc:
            print(f"[DEBUG] Failed to extract OSQP problem data: {exc!r}")
            return

        if not isinstance(data, dict):
            print(
                "[DEBUG] Unexpected OSQP problem data type: " f"{type(data).__name__}"
            )
            return

        print(f"[DEBUG] OSQP data keys: {sorted(data.keys())}")

        lower = data.get("lower_bounds")
        upper = data.get("upper_bounds")
        if lower is None or upper is None:
            lower = data.get("l")
            upper = data.get("u")
        if lower is None or upper is None:
            beq = data.get("b")
            gineq = data.get("G")
            if beq is not None and gineq is not None:
                beq_arr = np.asarray(beq, dtype=np.float64).reshape(-1)
                gineq_arr = np.asarray(gineq, dtype=np.float64).reshape(-1)
                lower = np.concatenate(
                    [beq_arr, np.full(gineq_arr.shape, -np.inf, dtype=np.float64)]
                )
                upper = np.concatenate([beq_arr, gineq_arr])
            elif beq is not None:
                beq_arr = np.asarray(beq, dtype=np.float64).reshape(-1)
                lower = beq_arr
                upper = beq_arr
            elif gineq is not None:
                gineq_arr = np.asarray(gineq, dtype=np.float64).reshape(-1)
                lower = np.full(gineq_arr.shape, -np.inf, dtype=np.float64)
                upper = gineq_arr

        if lower is None or upper is None:
            print("[DEBUG] Could not reconstruct OSQP lower/upper bounds.")
            return

        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)

        if lower.shape != upper.shape:
            print(
                "[DEBUG] Bound shape mismatch: "
                f"lower={lower.shape}, upper={upper.shape}"
            )
            return

        nan_lower = np.flatnonzero(np.isnan(lower))
        nan_upper = np.flatnonzero(np.isnan(upper))
        if nan_lower.size > 0:
            print(
                "[DEBUG] NaN detected in lower bounds at rows "
                f"{nan_lower[:10].tolist()}"
            )
        if nan_upper.size > 0:
            print(
                "[DEBUG] NaN detected in upper bounds at rows "
                f"{nan_upper[:10].tolist()}"
            )

        finite_lower = lower[np.isfinite(lower)]
        finite_upper = upper[np.isfinite(upper)]
        if finite_lower.size > 0 and finite_upper.size > 0:
            print(
                "[DEBUG] Bound stats: "
                f"lower[min,max]=({finite_lower.min():.12e}, {finite_lower.max():.12e}), "
                f"upper[min,max]=({finite_upper.min():.12e}, {finite_upper.max():.12e})"
            )

        bad_rows = np.flatnonzero(lower > upper)
        if bad_rows.size == 0:
            print(
                "[DEBUG] No rows with lower > upper found in reconstructed OSQP data."
            )
            return

        print(
            "[DEBUG] Rows with lower > upper detected: "
            f"count={bad_rows.size}, first_rows={bad_rows[:10].tolist()}"
        )
        for row in bad_rows[:10]:
            print(
                "[DEBUG]   row="
                f"{int(row)} lower={lower[row]:.12e} upper={upper[row]:.12e}"
            )

    def build_grouped_cbf_qps(self):
        """
        Prebuild G group QPs. Each group has capacity m = max_group_size.
        Cross-group constraints are preallocated against up to E external agents per local agent.
        Includes CLF relaxations per local slot.
        """
        print("[INFO] Building grouped CBF-QPs...")
        if hasattr(self, "_group_qp_caches"):
            return

        N = int(self.parameters.n_agents)
        C = int(self.parameters.n_circles_approximate_vehicle)
        m = int(self.parameters.max_group_size)
        if m < 1:
            raise ValueError("max_group_size must be >= 1")
        G = int(ceil(N / m))

        self._group_qp_caches: List[dict] = []
        self._group_capacity = m
        self._group_count = G
        self._group_circle_count = C
        self._group_disable_osqp = m == 1
        if self._group_disable_osqp:
            print(
                "[INFO] Grouped QPs with max_group_size=1 will use CLARABEL "
                "directly to avoid OSQP bound-update failures."
            )

        # -----------------------------
        # Cross-group preallocation size E
        # -----------------------------
        # If there is only one group, there are no cross-group constraints.
        # Optional cap for conditioning/speed: set parameters.max_cross_neighbors (int).
        if G == 1:
            self._group_cross_cap = 0
        else:
            E_cap_user = getattr(self.parameters, "max_cross_neighbors", None)
            if E_cap_user is None:
                self._group_cross_cap = max(0, N - 1)
            else:
                E_cap_user = int(E_cap_user)
                if E_cap_user < 0:
                    E_cap_user = 0
                self._group_cross_cap = min(max(0, N - 1), E_cap_user)

        def idx_lane(i_slot, i_c, side):
            return 2 * (i_slot * C + i_c) + side

        def pair_slot_base(mcap, Cc):
            base = {}
            k = 0
            for i in range(mcap - 1):
                for j in range(i + 1, mcap):
                    base[(i, j)] = k
                    k += Cc * Cc
            return base, k

        pair_base, pair_count = pair_slot_base(m, C)

        # Cross indexing only if E > 0
        E = int(self._group_cross_cap)
        if E > 0:
            cross_base = {}
            kx = 0
            for i in range(m):
                for e in range(E):
                    cross_base[(i, e)] = kx
                    kx += C * C
            cross_count = kx
        else:
            cross_base = {}
            cross_count = 0

        for g in range(G):
            u_var = cp.Variable((m, 2))
            s_lane = cp.Variable(2 * m * C, nonneg=True)
            s_pair = cp.Variable(pair_count, nonneg=True) if pair_count > 0 else None

            # Cross slacks only if cross_count > 0
            s_cross = cp.Variable(cross_count, nonneg=True) if cross_count > 0 else None

            # CLF relaxations per slot
            s_clf_head = cp.Variable(m, nonneg=True)
            s_clf_speed = cp.Variable(m, nonneg=True)

            cons = [
                self.a_min <= u_var[:, 0],
                u_var[:, 0] <= self.a_max,
                self.steering_rate_min <= u_var[:, 1],
                u_var[:, 1] <= self.steering_rate_max,
            ]

            U_nom = cp.Parameter((m, 2))

            # CLF params per slot
            e_head = cp.Parameter(m)
            e_speed = cp.Parameter(m)
            v_head_const = cp.Parameter(m)
            v_speed_const = cp.Parameter(m)

            # CLF inequalities
            cons += [
                -e_head[i] * u_var[i, 1] + v_head_const[i] - s_clf_head[i] <= 0
                for i in range(m)
            ]
            cons += [
                -e_speed[i] * u_var[i, 0] + v_speed_const[i] - s_clf_speed[i] <= 0
                for i in range(m)
            ]

            # Lane params
            A_L = [[cp.Parameter((1, 2)) for _ in range(C)] for _ in range(m)]
            A_R = [[cp.Parameter((1, 2)) for _ in range(C)] for _ in range(m)]
            if self.parameters.adaptive_lambda:
                b0_L = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
                h_L = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
                b0_R = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
                h_R = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
                b_L = None
                b_R = None

                lambda_lane = cp.Variable(2 * m * C)
                lambda_lane_ref = cp.Parameter(2 * m * C)
                cons += [lambda_lane >= 0, lambda_lane <= 1]
                if self.parameters.is_learn_cbf_lambda:
                    cons += [lambda_lane == lambda_lane_ref]

                for i in range(m):
                    for c in range(C):
                        cons += [
                            A_L[i][c] @ u_var[i, :]
                            + b0_L[i][c]
                            + h_L[i][c] * lambda_lane[idx_lane(i, c, 0)]
                            >= -s_lane[idx_lane(i, c, 0)]
                        ]
                        cons += [
                            A_R[i][c] @ u_var[i, :]
                            + b0_R[i][c]
                            + h_R[i][c] * lambda_lane[idx_lane(i, c, 1)]
                            >= -s_lane[idx_lane(i, c, 1)]
                        ]
            else:
                b_L = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
                b_R = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
                b0_L = None
                h_L = None
                b0_R = None
                h_R = None
                lambda_lane = None
                lambda_lane_ref = None

                for i in range(m):
                    for c in range(C):
                        cons += [
                            A_L[i][c] @ u_var[i, :] + b_L[i][c]
                            >= -s_lane[idx_lane(i, c, 0)]
                        ]
                        cons += [
                            A_R[i][c] @ u_var[i, :] + b_R[i][c]
                            >= -s_lane[idx_lane(i, c, 1)]
                        ]

            pair_uses_symbolic_lambda = (
                self.parameters.adaptive_lambda
                and not self.parameters.is_learn_cbf_lambda
            )

            # Intra-group pairs
            Aij_i = {}
            Aij_j = {}
            if pair_uses_symbolic_lambda:
                b0ij = {}
                hij = {}
                bij = None
                lambda_pair = cp.Variable(pair_count) if pair_count > 0 else None
                lambda_pair_ref = None
                if lambda_pair is not None:
                    cons += [lambda_pair >= 0, lambda_pair <= 1]
            else:
                bij = {}
                b0ij = None
                hij = None
                lambda_pair = None
                lambda_pair_ref = None

            for i in range(m - 1):
                for j in range(i + 1, m):
                    for ci in range(C):
                        for cj in range(C):
                            Aij_i[(i, j, ci, cj)] = cp.Parameter((1, 2))
                            Aij_j[(i, j, ci, cj)] = cp.Parameter((1, 2))
                            if pair_uses_symbolic_lambda:
                                b0ij[(i, j, ci, cj)] = cp.Parameter((1,))
                                hij[(i, j, ci, cj)] = cp.Parameter((1,))
                            else:
                                bij[(i, j, ci, cj)] = cp.Parameter((1,))

                            if s_pair is None and pair_uses_symbolic_lambda:
                                cons += [
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + b0ij[(i, j, ci, cj)]
                                    + hij[(i, j, ci, cj)] * 1.0
                                    >= 0
                                ]
                            elif s_pair is None:
                                cons += [
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + bij[(i, j, ci, cj)]
                                    >= 0
                                ]
                            elif pair_uses_symbolic_lambda:
                                cons += [
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + b0ij[(i, j, ci, cj)]
                                    + hij[(i, j, ci, cj)]
                                    * lambda_pair[pair_base[(i, j)] + ci * C + cj]
                                    >= -s_pair[pair_base[(i, j)] + ci * C + cj]
                                ]
                            else:
                                cons += [
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + bij[(i, j, ci, cj)]
                                    >= -s_pair[pair_base[(i, j)] + ci * C + cj]
                                ]

            # -----------------------------
            # Cross-group (only i's control), only if E > 0
            # -----------------------------
            cross_uses_symbolic_lambda = (
                self.parameters.adaptive_lambda
                and not self.parameters.is_learn_cbf_lambda
            )
            Axc_i = {}
            if cross_uses_symbolic_lambda:
                b0xc = {}
                hxc = {}
                bxc = None
                lambdax_i = cp.Variable(cross_count) if cross_count > 0 else None
                lambdax_i_ref = None
                if lambdax_i is not None:
                    cons += [lambdax_i >= 0, lambdax_i <= 1]
            else:
                bxc = {}
                b0xc = None
                hxc = None
                lambdax_i = None
                lambdax_i_ref = None

            if cross_count > 0:
                for i in range(m):
                    for e in range(E):
                        for ci in range(C):
                            for cj in range(C):
                                Axc_i[(i, e, ci, cj)] = cp.Parameter((1, 2))
                                if cross_uses_symbolic_lambda:
                                    b0xc[(i, e, ci, cj)] = cp.Parameter((1,))
                                    hxc[(i, e, ci, cj)] = cp.Parameter((1,))
                                else:
                                    bxc[(i, e, ci, cj)] = cp.Parameter((1,))
                                # Active form
                                if s_cross is not None and cross_uses_symbolic_lambda:
                                    cons += [
                                        Axc_i[(i, e, ci, cj)] @ u_var[i, :]
                                        + self._cross_cbf_drift_scale()
                                        * (
                                            b0xc[(i, e, ci, cj)]
                                            + hxc[(i, e, ci, cj)]
                                            * lambdax_i[
                                                cross_base[(i, e)] + ci * C + cj
                                            ]
                                        )
                                        >= -s_cross[cross_base[(i, e)] + ci * C + cj]
                                    ]
                                elif s_cross is not None:
                                    cons += [
                                        Axc_i[(i, e, ci, cj)] @ u_var[i, :]
                                        + bxc[(i, e, ci, cj)]
                                        >= -s_cross[cross_base[(i, e)] + ci * C + cj]
                                    ]
                                elif cross_uses_symbolic_lambda:
                                    cons += [
                                        Axc_i[(i, e, ci, cj)] @ u_var[i, :]
                                        + b0xc[(i, e, ci, cj)]
                                        + hxc[(i, e, ci, cj)] * 1.0
                                        >= 0
                                    ]
                                else:
                                    cons += [
                                        Axc_i[(i, e, ci, cj)] @ u_var[i, :]
                                        + bxc[(i, e, ci, cj)]
                                        >= 0
                                    ]

            # Objective
            cost = cp.sum_squares((u_var - U_nom) @ self.nom_weight)
            cost += self.lane_slack_weight * cp.sum_squares(s_lane)
            if s_pair is not None:
                cost += self.pair_slack_weight * cp.sum_squares(s_pair)
            if s_cross is not None:
                cost += self.cross_slack_weight * cp.sum_squares(s_cross)

            # CLF penalties
            cost += self.w_clf_relax * (
                cp.sum_squares(s_clf_head) + cp.sum_squares(s_clf_speed)
            )

            if self.parameters.adaptive_lambda:
                cost += self.lambda_weight * cp.sum_squares(
                    lambda_lane - self.lambda_ttcbf_predefined
                )
                if lambda_pair is not None:
                    cost += self.lambda_weight * cp.sum_squares(
                        lambda_pair - self.lambda_ttcbf_predefined
                    )

            # Always regularize cross lambdas if they exist
            if lambdax_i is not None:
                cost += self.lambda_weight * cp.sum_squares(
                    lambdax_i - float(self.lambda_ttcbf_predefined)
                )

            prob = cp.Problem(cp.Minimize(cost), cons)

            cache = dict(
                m=m,
                C=C,
                E=E,
                pair_base=pair_base,
                pair_count=pair_count,
                cross_base=cross_base,
                cross_count=cross_count,
                u_var=u_var,
                U_nom=U_nom,
                s_lane=s_lane,
                s_pair=s_pair,
                s_cross=s_cross,
                s_clf_head=s_clf_head,
                s_clf_speed=s_clf_speed,
                e_head=e_head,
                e_speed=e_speed,
                v_head_const=v_head_const,
                v_speed_const=v_speed_const,
                A_L=A_L,
                b_L=b_L,
                b0_L=b0_L,
                h_L=h_L,
                A_R=A_R,
                b_R=b_R,
                b0_R=b0_R,
                h_R=h_R,
                lambda_lane=lambda_lane,
                lambda_lane_ref=lambda_lane_ref,
                Aij_i=Aij_i,
                Aij_j=Aij_j,
                bij=bij,
                b0ij=b0ij,
                hij=hij,
                lambda_pair=lambda_pair,
                lambda_pair_ref=lambda_pair_ref,
                Axc_i=Axc_i,
                bxc=bxc,
                b0xc=b0xc,
                hxc=hxc,
                lambdax_i=lambdax_i,
                lambdax_i_ref=lambdax_i_ref,
                idx_lane=idx_lane,
                prob=prob,
            )

            # Warm start (nominals, slacks, decision vars)
            U_nom.value = np.zeros((m, 2))
            s_lane.value = np.zeros((2 * m * C,))
            s_clf_head.value = np.zeros((m,))
            s_clf_speed.value = np.zeros((m,))
            e_head.value = np.zeros((m,))
            e_speed.value = np.zeros((m,))
            v_head_const.value = np.zeros((m,))
            v_speed_const.value = np.zeros((m,))
            if s_pair is not None:
                s_pair.value = np.zeros((pair_count,))
            if s_cross is not None:
                s_cross.value = np.zeros((cross_count,))
            u_var.value = np.zeros((m, 2))

            # Warm start lambdas (bias to lambda_ttcbf)
            if lambda_lane is not None:
                lambda_lane.value = np.full(
                    lambda_lane.shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
                lambda_lane_ref.value = np.full(
                    lambda_lane_ref.shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
            if lambda_pair is not None:
                lambda_pair.value = np.full(
                    lambda_pair.shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
            if lambdax_i is not None:
                lambdax_i.value = np.full(
                    lambdax_i.shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )

            self._group_qp_caches.append(cache)

        self._group_assignment: List[List[int]] = [[] for _ in range(G)]
        # Baseline: fixed groups throughout the simulation
        self.use_fixed_groups = True
        self.fixed_groups = None

    def update_grouped_cbf_qps(self, tensordict):
        """
        Recompute groups. Assign to fixed group slots. Update params including CLF.
        Solve each group QP. Write back actions.
        """
        assert hasattr(self, "_group_qp_caches"), "Call build_grouped_cbf_qps first."
        agents: List[Vehicle] = self.env.base_env.scenario_name.world.agents
        N = int(self.parameters.n_agents)
        C = self._group_circle_count
        m = self._group_capacity
        G = self._group_count
        cbf_coupling_range = float(self.parameters.cbf_coupling_range)
        learned_lambdas = self._get_env_cbf_lambdas(tensordict)

        # print(f"Env {self.env_idx}: Updating grouped CBF-QPs with learned lambdas: {learned_lambdas}")

        # Collect states and centers
        path_ids_t = tensordict["agents", "info", "path_id"][self.env_idx]
        path_ids = [int(path_ids_t[i].item()) for i in range(N)]
        states = []
        centers = []
        centers_flat = []
        for i in range(N):
            s = torch.cat(
                [
                    agents[i].state.pos[self.env_idx],
                    agents[i].state.rot[self.env_idx],
                    agents[i].state.speed[self.env_idx],
                    agents[i].state.steering[self.env_idx],
                ],
                dim=-1,
            )
            states.append(s)
            centers.append(self.get_circle_centers(s))
            centers_flat.append(
                np.array([float(s[0].item()), float(s[1].item())], dtype=np.float64)
            )

        # Grouping (baseline: compute once, then keep fixed)
        asl = [
            AgentStateList(x=cf[0], y=cf[1], v=float(states[i][3].item()))
            for i, cf in enumerate(centers_flat)
        ]

        if self.use_fixed_groups:
            if self.fixed_groups is None:
                groups0 = group_agents_k_nearest(asl, m)
                groups0 = [sorted(list(g)) for g in groups0]
                groups0.sort(key=lambda g: g[0] if len(g) > 0 else 10**9)
                if len(groups0) != G:
                    raise RuntimeError(
                        f"Fixed grouping produced {len(groups0)} groups, expected {G}."
                    )
                self.fixed_groups = groups0
            groups = self.fixed_groups
        else:
            groups = group_agents_k_nearest(asl, m)
            groups = [sorted(list(g)) for g in groups]
            groups.sort(key=lambda g: g[0] if len(g) > 0 else 10**9)

        # Precompute nominal actions or CLF errors for all agents
        U_nom_all = np.zeros((N, 2), dtype=np.float64)
        nominal_target_actions_all = torch.zeros(
            (N, 2), device=self.device, dtype=torch.float32
        )
        e_h_all = np.zeros((N,), dtype=np.float64)
        e_v_all = np.zeros((N,), dtype=np.float64)

        for i in range(N):
            (
                u_nom_i,
                target_action_i,
                e_h,
                e_v,
            ) = self._compute_nominal_control_for_agent(tensordict, states[i], i)
            U_nom_all[i, :] = u_nom_i
            nominal_target_actions_all[i, :] = target_action_i
            e_h_all[i] = e_h
            e_v_all[i] = e_v

        # Safety radius
        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe

        qp_solving_t_groups = []
        qp_solving_iter_groups = []
        cross_groups: Dict[int, List[int]] = {}
        compare_target_actions_all = nominal_target_actions_all.clone()

        for g in range(G):
            cache = self._group_qp_caches[g]
            mcap = cache["m"]
            Cc = cache["C"]
            Ecap = int(cache["E"])  # may be 0
            idx_lane = cache["idx_lane"]
            pair_base = cache["pair_base"]

            members = groups[g][:mcap]
            n_local = len(members)
            slot2id = [-1] * mcap
            for k in range(n_local):
                slot2id[k] = members[k]

            lane_lambda_ref = None
            if cache["lambda_lane"] is not None:
                lane_lambda_ref = np.full(
                    cache["lambda_lane"].shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
            pair_lambda_ref = None
            if cache["lambda_pair"] is not None:
                pair_lambda_ref = np.full(
                    cache["lambda_pair"].shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
            cross_lambda_ref = None
            if cache["lambdax_i"] is not None:
                cross_lambda_ref = np.full(
                    cache["lambdax_i"].shape,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )

            # U_nom and CLF params per slot
            U_loc = np.zeros((mcap, 2), dtype=np.float64)
            e_head_loc = np.zeros((mcap,), dtype=np.float64)
            e_speed_loc = np.zeros((mcap,), dtype=np.float64)
            v_head_const = np.zeros((mcap,), dtype=np.float64)
            v_speed_const = np.zeros((mcap,), dtype=np.float64)

            for k in range(mcap):
                if k < n_local:
                    gid = slot2id[k]
                    U_loc[k, :] = U_nom_all[gid, :]
                    if self.nom_controller_type == "clf":
                        e_head_loc[k] = e_h_all[gid]
                        e_speed_loc[k] = e_v_all[gid]
                        v_head_const[k] = self.lam_clf * 0.5 * (e_h_all[gid] ** 2)
                        v_speed_const[k] = self.lam_clf * 0.5 * (e_v_all[gid] ** 2)

            cache["U_nom"].value = U_loc
            cache["e_head"].value = e_head_loc
            cache["e_speed"].value = e_speed_loc
            cache["v_head_const"].value = v_head_const
            cache["v_speed_const"].value = v_speed_const

            # Lane constraints update
            for i_slot in range(mcap):
                if i_slot < n_local:
                    i = slot2id[i_slot]
                    kins_i = self.linearized_center_kinematics_coeffs(states[i])
                    for ci in range(Cc):
                        p_ic = centers[i][ci][0:2]
                        (
                            smL,
                            gL,
                            HL,
                            smR,
                            gR,
                            HR,
                        ) = self.estimate_agent_2_lane_safety_margin(p_ic, path_ids[i])
                        if self.parameters.adaptive_lambda:
                            A_L_ic, b0_L_ic, h_L_ic = self.ttcbf_lane_affine_coeffs(
                                kins_i, ci, smL, gL, HL, self.dt_taylor, None
                            )
                            A_R_ic, b0_R_ic, h_R_ic = self.ttcbf_lane_affine_coeffs(
                                kins_i, ci, smR, gR, HR, self.dt_taylor, None
                            )
                            cache["A_L"][i_slot][ci].value = A_L_ic
                            cache["b0_L"][i_slot][ci].value = b0_L_ic
                            cache["h_L"][i_slot][ci].value = h_L_ic
                            cache["A_R"][i_slot][ci].value = A_R_ic
                            cache["b0_R"][i_slot][ci].value = b0_R_ic
                            cache["h_R"][i_slot][ci].value = h_R_ic
                        else:
                            A_L_ic, b_L_ic = self.ttcbf_lane_affine_coeffs(
                                kins_i,
                                ci,
                                smL,
                                gL,
                                HL,
                                self.dt_taylor,
                                self.lambda_ttcbf_predefined,
                            )
                            A_R_ic, b_R_ic = self.ttcbf_lane_affine_coeffs(
                                kins_i,
                                ci,
                                smR,
                                gR,
                                HR,
                                self.dt_taylor,
                                self.lambda_ttcbf_predefined,
                            )
                            cache["A_L"][i_slot][ci].value = np.asarray(
                                A_L_ic, dtype=np.float64
                            ).reshape(1, 2)
                            cache["b_L"][i_slot][ci].value = np.asarray(
                                b_L_ic, dtype=np.float64
                            ).reshape(
                                1,
                            )
                            cache["A_R"][i_slot][ci].value = np.asarray(
                                A_R_ic, dtype=np.float64
                            ).reshape(1, 2)
                            cache["b_R"][i_slot][ci].value = np.asarray(
                                b_R_ic, dtype=np.float64
                            ).reshape(
                                1,
                            )
                else:
                    for ci in range(Cc):
                        cache["A_L"][i_slot][ci].value = np.zeros((1, 2))
                        cache["A_R"][i_slot][ci].value = np.zeros((1, 2))
                        if self.parameters.adaptive_lambda:
                            cache["b0_L"][i_slot][ci].value = np.array([1.0])
                            cache["h_L"][i_slot][ci].value = np.array([0.0])
                            cache["b0_R"][i_slot][ci].value = np.array([1.0])
                            cache["h_R"][i_slot][ci].value = np.array([0.0])
                        else:
                            cache["b_L"][i_slot][ci].value = np.array([1.0])
                            cache["b_R"][i_slot][ci].value = np.array([1.0])

            # Reset slacks and warm start
            cache["s_lane"].value = np.zeros(cache["s_lane"].shape)
            if cache["s_pair"] is not None:
                cache["s_pair"].value = np.zeros(cache["s_pair"].shape)
            if cache["s_cross"] is not None:
                cache["s_cross"].value = np.zeros(cache["s_cross"].shape)
            cache["s_clf_head"].value = np.zeros(cache["s_clf_head"].shape)
            cache["s_clf_speed"].value = np.zeros(cache["s_clf_speed"].shape)
            cache["u_var"].value = cache["U_nom"].value.copy()

            # Warm start lambdas every step (keeps behavior stable when constraints are deactivated)
            if cache["lambda_lane"] is not None:
                cache["lambda_lane_ref"].value = lane_lambda_ref
                cache["lambda_lane"].value = (
                    lane_lambda_ref.copy()
                    if self.parameters.is_learn_cbf_lambda
                    else np.full(
                        cache["lambda_lane"].shape,
                        float(self.lambda_ttcbf_predefined),
                        dtype=np.float64,
                    )
                )
            if cache["lambda_pair"] is not None:
                cache["lambda_pair"].value = (
                    pair_lambda_ref.copy()
                    if self.parameters.is_learn_cbf_lambda
                    else np.full(
                        cache["lambda_pair"].shape,
                        float(self.lambda_ttcbf_predefined),
                        dtype=np.float64,
                    )
                )
            if cache["lambdax_i"] is not None:
                cache["lambdax_i"].value = (
                    cross_lambda_ref.copy()
                    if self.parameters.is_learn_cbf_lambda
                    else np.full(
                        cache["lambdax_i"].shape,
                        float(self.lambda_ttcbf_predefined),
                        dtype=np.float64,
                    )
                )

            # Intra-group pairs
            for i_loc in range(mcap - 1):
                for j_loc in range(i_loc + 1, mcap):
                    for ci in range(Cc):
                        for cj in range(Cc):
                            key = (i_loc, j_loc, ci, cj)
                            if i_loc < n_local and j_loc < n_local:
                                i = slot2id[i_loc]
                                j = slot2id[j_loc]
                                lam_pair = (
                                    self._get_agent_cbf_lambda(learned_lambdas, i)
                                    if self.parameters.is_learn_cbf_lambda
                                    else float(self.lambda_ttcbf_predefined)
                                )
                                kins_i = self.linearized_center_kinematics_coeffs(
                                    states[i]
                                )
                                kins_j = self.linearized_center_kinematics_coeffs(
                                    states[j]
                                )
                                pi = centers[i][ci][0:2]
                                pj = centers[j][cj][0:2]
                                dx = float((pi - pj)[0].item())
                                dy = float((pi - pj)[1].item())
                                if cache["lambda_pair"] is not None:
                                    pair_lambda_ref[
                                        pair_base[(i_loc, j_loc)] + ci * Cc + cj
                                    ] = lam_pair
                                    A_i, A_j, b0, h = self.ttcbf_pair_affine_coeffs(
                                        kins_i,
                                        kins_j,
                                        ci,
                                        cj,
                                        dx,
                                        dy,
                                        d_safe_sq,
                                        self.dt_taylor,
                                        None,
                                    )
                                else:
                                    A_i, A_j, b, _ = self.ttcbf_pair_affine_coeffs(
                                        kins_i,
                                        kins_j,
                                        ci,
                                        cj,
                                        dx,
                                        dy,
                                        d_safe_sq,
                                        self.dt_taylor,
                                        lam_pair,
                                    )
                                cache["Aij_i"][key].value = A_i
                                cache["Aij_j"][key].value = A_j
                                if cache["lambda_pair"] is not None:
                                    cache["b0ij"][key].value = b0
                                    cache["hij"][key].value = h
                                else:
                                    cache["bij"][key].value = b
                            else:
                                cache["Aij_i"][key].value = np.zeros((1, 2))
                                cache["Aij_j"][key].value = np.zeros((1, 2))
                                if cache["lambda_pair"] is not None:
                                    cache["b0ij"][key].value = np.array([1.0])
                                    cache["hij"][key].value = np.array([0.0])
                                else:
                                    cache["bij"][key].value = np.array([1.0])

            # -----------------------------
            # Cross-group reciprocal (only if Ecap > 0)
            # -----------------------------
            if Ecap > 0 and cache["s_cross"] is not None:
                for i_loc in range(mcap):
                    if i_loc < n_local:
                        i = slot2id[i_loc]
                        nbrs: List[int] = []
                        for j in range(N):
                            if j in members:
                                continue
                            if (
                                _euclid(centers_flat[i], centers_flat[j])
                                <= cbf_coupling_range + 1e-9
                            ):
                                nbrs.append(j)
                        nbrs = nbrs[:Ecap]
                        cross_groups[i] = nbrs

                        for e_idx in range(Ecap):
                            if e_idx < len(nbrs):
                                j = nbrs[e_idx]
                                kins_i = self.linearized_center_kinematics_coeffs(
                                    states[i]
                                )
                                kins_j = self.linearized_center_kinematics_coeffs(
                                    states[j]
                                )
                                for ci in range(Cc):
                                    for cj in range(Cc):
                                        lam_cross = (
                                            self._get_agent_cbf_lambda(
                                                learned_lambdas, i
                                            )
                                            if self.parameters.is_learn_cbf_lambda
                                            else float(self.lambda_ttcbf_predefined)
                                        )
                                        pi = centers[i][ci][0:2]
                                        pj = centers[j][cj][0:2]
                                        dx = float((pi - pj)[0].item())
                                        dy = float((pi - pj)[1].item())
                                        if cache["lambdax_i"] is not None:
                                            cross_lambda_ref[
                                                cache["cross_base"][(i_loc, e_idx)]
                                                + ci * Cc
                                                + cj
                                            ] = lam_cross
                                            (
                                                A_i,
                                                b0,
                                                h,
                                            ) = self.ttcbf_pair_affine_coeffs_cross(
                                                kins_i,
                                                kins_j,
                                                ci,
                                                cj,
                                                dx,
                                                dy,
                                                d_safe_sq,
                                                self.dt_taylor,
                                                None,
                                            )
                                        else:
                                            (
                                                A_i,
                                                b,
                                            ) = self.ttcbf_pair_affine_coeffs_cross(
                                                kins_i,
                                                kins_j,
                                                ci,
                                                cj,
                                                dx,
                                                dy,
                                                d_safe_sq,
                                                self.dt_taylor,
                                                lam_cross,
                                            )
                                        cache["Axc_i"][
                                            (i_loc, e_idx, ci, cj)
                                        ].value = A_i
                                        if cache["lambdax_i"] is not None:
                                            cache["b0xc"][
                                                (i_loc, e_idx, ci, cj)
                                            ].value = b0
                                            cache["hxc"][
                                                (i_loc, e_idx, ci, cj)
                                            ].value = h
                                        else:
                                            cache["bxc"][
                                                (i_loc, e_idx, ci, cj)
                                            ].value = b
                            else:
                                for ci in range(Cc):
                                    for cj in range(Cc):
                                        cache["Axc_i"][
                                            (i_loc, e_idx, ci, cj)
                                        ].value = np.zeros((1, 2))
                                        if cache["lambdax_i"] is not None:
                                            cache["b0xc"][
                                                (i_loc, e_idx, ci, cj)
                                            ].value = np.array([1.0])
                                            cache["hxc"][
                                                (i_loc, e_idx, ci, cj)
                                            ].value = np.array([0.0])
                                        else:
                                            cache["bxc"][
                                                (i_loc, e_idx, ci, cj)
                                            ].value = np.array([1.0])
                    else:
                        for e_idx in range(Ecap):
                            for ci in range(Cc):
                                for cj in range(Cc):
                                    cache["Axc_i"][
                                        (i_loc, e_idx, ci, cj)
                                    ].value = np.zeros((1, 2))
                                    if cache["lambdax_i"] is not None:
                                        cache["b0xc"][
                                            (i_loc, e_idx, ci, cj)
                                        ].value = np.array([1.0])
                                        cache["hxc"][
                                            (i_loc, e_idx, ci, cj)
                                        ].value = np.array([0.0])
                                    else:
                                        cache["bxc"][
                                            (i_loc, e_idx, ci, cj)
                                        ].value = np.array([1.0])
            else:
                # No cross constraints prebuilt: record empty neighbor lists for visualization
                for k in range(n_local):
                    cross_groups[slot2id[k]] = []

            if cache["lambda_pair"] is not None:
                cache["lambda_pair"].value = pair_lambda_ref.copy()
            if cache["lambdax_i"] is not None:
                cache["lambdax_i"].value = cross_lambda_ref.copy()

            # Solve group QP
            prob: cp.Problem = cache["prob"]
            invalid_param = self._find_invalid_qp_parameter(
                cache, path=f"_group_qp_caches[{g}]"
            )
            if invalid_param is not None:
                print(
                    "[WARN] Skipping grouped OSQP solve due to invalid QP "
                    f"parameter at {invalid_param[0]}: {invalid_param[1]}"
                )
            else:
                osqp_failed = False
                if not self._group_disable_osqp:
                    try:
                        prob.solve(
                            solver=cp.OSQP,
                            warm_start=True,
                            max_iter=20000,
                            eps_abs=1e-5,
                            eps_rel=1e-5,
                            polish=True,
                            adaptive_rho=True,
                        )
                    except ValueError as exc:
                        osqp_failed = True
                        print(f"[WARN] OSQP update failed with ValueError: {exc}")
                        self._debug_osqp_bound_update_failure(
                            prob,
                            context=dict(
                                env_idx=self.env_idx,
                                group_idx=g,
                                members=members,
                                slot2id=slot2id,
                                cross_neighbors={
                                    gid: cross_groups.get(gid, [])
                                    for gid in members
                                    if gid != -1
                                },
                            ),
                        )
                        print(
                            "[WARN] Disabling OSQP for grouped QPs for the rest of "
                            "this run and switching to CLARABEL/SCS."
                        )
                        self._group_disable_osqp = True
                    except cp.SolverError:
                        osqp_failed = True
                else:
                    osqp_failed = True

                if osqp_failed:
                    # print("[WARN] Grouped solve using CLARABEL...")
                    try:
                        prob.solve(
                            solver=cp.CLARABEL,
                            warm_start=True,
                            tol_feas=1e-8,
                            tol_gap_abs=1e-8,
                            tol_gap_rel=1e-8,
                        )
                    except cp.SolverError:
                        print("[WARN] CLARABEL solver failed, trying SCS...")
                        try:
                            prob.solve(
                                solver=cp.SCS,
                                warm_start=True,
                                max_iters=100000,
                                eps=1e-4,
                            )
                        except cp.SolverError:
                            print(
                                "[ERROR] SCS solver also failed. Using nominal actions as fallback."
                            )

            solved = (
                prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
                and cache["u_var"].value is not None
                and np.isfinite(cache["u_var"].value).all()
            )

            if solved:
                u_opt = cache["u_var"].value
                for k in range(n_local):
                    gid = slot2id[k]
                    s_i = states[gid]
                    actions_v_steer_safe = self.u_to_rl_action(
                        u_opt[k, :], s_i[3], s_i[4], agent_idx=gid
                    ).detach()
                    tensordict[("agents", "action")][
                        self.env_idx, gid
                    ] = actions_v_steer_safe
                    compare_target_actions_all[gid] = self.u_to_target_action(
                        u_opt[k, :], s_i[3], s_i[4]
                    ).detach()
            else:
                for k in range(n_local):
                    gid = slot2id[k]
                    s_i = states[gid]
                    actions_v_steer_nom = self.u_to_rl_action(
                        U_nom_all[gid, :], s_i[3], s_i[4], agent_idx=gid
                    ).detach()
                    tensordict[("agents", "action")][
                        self.env_idx, gid
                    ] = actions_v_steer_nom
                    compare_target_actions_all[gid] = nominal_target_actions_all[gid]
                self.hist.evt.append("QP-INF")

            st = prob.solver_stats
            if hasattr(st, "solve_time"):
                qp_solving_t_groups.append(st.solve_time)
            if hasattr(st, "num_iters"):
                qp_solving_iter_groups.append(st.num_iters)

        # For later visualization
        self.scenario.inter_groups = groups
        self.scenario.cross_groups = cross_groups

        applied_target_actions_all = self._target_actions_from_current_actions(
            tensordict, states
        ).detach()
        self.env.base_env.scenario_name.world_state.applied_action_vel[
            self.env_idx, :
        ] = applied_target_actions_all[:, 0]
        self.env.base_env.scenario_name.world_state.applied_action_steer[
            self.env_idx, :
        ] = applied_target_actions_all[:, 1]

        self.hist.qp_solving_t.append(qp_solving_t_groups)
        t_avg = sum(qp_solving_t_groups) / max(1, len(qp_solving_t_groups))
        # print(
        #     f"[INFO] Semi-centralized QP solving time per group: {t_avg*1000:.2f} ms. "
        #     f"Over all time steps: {np.mean(np.array(self.hist.qp_solving_t))*1000:.2f} ms"
        # )
        self.hist.qp_solving_iter.append(qp_solving_iter_groups)

        # For visualization of nominal actions
        if self.env.scenario_name.parameters.is_apply_cbf_action:
            self.env.base_env.scenario_name.world_state.nominal_action_vel[
                self.env_idx, :
            ] = nominal_target_actions_all[:, 0]
            self.env.base_env.scenario_name.world_state.nominal_action_steer[
                self.env_idx, :
            ] = nominal_target_actions_all[:, 1]
        else:
            self.env.base_env.scenario_name.world_state.nominal_action_vel[
                self.env_idx, :
            ] = compare_target_actions_all[:, 0]
            self.env.base_env.scenario_name.world_state.nominal_action_steer[
                self.env_idx, :
            ] = compare_target_actions_all[:, 1]

    # =========================
    # Shared kinematics and TTCBF coefficient builders
    # =========================
    def linearized_center_kinematics_coeffs(self, state_agent):
        """
        Copy the original function
        """
        n_circles = int(self.parameters.n_circles_approximate_vehicle)

        dstate_time_agent = self.kbm.ode(None, state_agent, self.u_placeholder)
        psi = state_agent[2]
        u0 = np.array([0.0, 0.0], dtype=np.float64)
        e1 = np.array([1.0, 0.0], dtype=np.float64)
        e2 = np.array([0.0, 1.0], dtype=np.float64)

        ddx0, ddy0, ddpsi0 = self.compute_dstate_2nd_time(
            u0, state_agent, dstate_time_agent
        )
        ddx_e1, ddy_e1, ddpsi_e1 = self.compute_dstate_2nd_time(
            e1, state_agent, dstate_time_agent
        )
        ddx_e2, ddy_e2, ddpsi_e2 = self.compute_dstate_2nd_time(
            e2, state_agent, dstate_time_agent
        )

        kins = dict(dx=[], dy=[], a_ddx=[], c_ddx=[], a_ddy=[], c_ddy=[])

        for ci in range(n_circles):
            (
                dx0_c,
                dy0_c,
                dpsi0_c,
                ddx0_c,
                ddy0_c,
                ddpsi0_c,
            ) = self.compute_center_state_time_derivatives(
                psi, dstate_time_agent, ddx0, ddy0, ddpsi0, ci
            )
            _, _, _, ddx_e1_c, ddy_e1_c, _ = self.compute_center_state_time_derivatives(
                psi, dstate_time_agent, ddx_e1, ddy_e1, ddpsi_e1, ci
            )
            _, _, _, ddx_e2_c, ddy_e2_c, _ = self.compute_center_state_time_derivatives(
                psi, dstate_time_agent, ddx_e2, ddy_e2, ddpsi_e2, ci
            )

            a_ddx_c = np.array([ddx_e1_c - ddx0_c, ddx_e2_c - ddx0_c], dtype=np.float64)
            a_ddy_c = np.array([ddy_e1_c - ddy0_c, ddy_e2_c - ddy0_c], dtype=np.float64)

            kins["dx"].append(float(dx0_c))
            kins["dy"].append(float(dy0_c))
            kins["a_ddx"].append(a_ddx_c)
            kins["c_ddx"].append(float(ddx0_c))
            kins["a_ddy"].append(a_ddy_c)
            kins["c_ddy"].append(float(ddy0_c))

        return kins

    def ttcbf_lane_affine_coeffs(self, kins_i, ci, sm, grad, hess, dt, lam):
        """
        Copy the original function
        """
        if lam is None:
            h = float(sm - self.safety_buffer)
            g = np.asarray(grad, dtype=np.float64).reshape(
                2,
            )
            H = np.asarray(hess, dtype=np.float64).reshape(2, 2)
            dx = float(kins_i["dx"][ci])
            dy = float(kins_i["dy"][ci])

            a_x = np.asarray(kins_i["a_ddx"][ci], dtype=np.float64).reshape(
                2,
            )
            c_x = float(kins_i["c_ddx"][ci])
            a_y = np.asarray(kins_i["a_ddy"][ci], dtype=np.float64).reshape(
                2,
            )
            c_y = float(kins_i["c_ddy"][ci])

            A = 0.5 * (dt * dt) * (g[0] * a_x + g[1] * a_y)
            dot_h = g[0] * dx + g[1] * dy
            const_dd = (
                g[0] * c_x + g[1] * c_y + np.array([dx, dy]) @ H @ np.array([dx, dy])
            )
            b0 = dot_h * dt + 0.5 * (dt * dt) * const_dd
            return (
                A.reshape(1, 2),
                np.array([b0], dtype=np.float64),
                np.array([h], dtype=np.float64),
            )
        else:
            h = float(sm - self.safety_buffer)
            g = np.asarray(grad, dtype=np.float64).reshape(
                2,
            )
            H = np.asarray(hess, dtype=np.float64).reshape(2, 2)
            dx = float(kins_i["dx"][ci])
            dy = float(kins_i["dy"][ci])

            a_x = np.asarray(kins_i["a_ddx"][ci], dtype=np.float64).reshape(
                2,
            )
            c_x = float(kins_i["c_ddx"][ci])
            a_y = np.asarray(kins_i["a_ddy"][ci], dtype=np.float64).reshape(
                2,
            )
            c_y = float(kins_i["c_ddy"][ci])

            dot_h = g[0] * dx + g[1] * dy
            A = 0.5 * (dt * dt) * (g[0] * a_x + g[1] * a_y)
            const_dd = (
                g[0] * c_x
                + g[1] * c_y
                + np.array([dx, dy], dtype=np.float64)
                @ H
                @ np.array([dx, dy], dtype=np.float64)
            )
            b = lam * h + dot_h * dt + 0.5 * (dt * dt) * const_dd
            return A.reshape(1, 2), np.array([b], dtype=np.float64)

    def ttcbf_pair_affine_coeffs(
        self, kins_i, kins_j, ci, cj, dx, dy, d_safe_sq, dt, lam
    ):
        """
        Copy the original function
        """
        if lam is None:
            dx = float(dx)
            dy = float(dy)
            vxi = float(kins_i["dx"][ci])
            vyi = float(kins_i["dy"][ci])
            vxj = float(kins_j["dx"][cj])
            vyj = float(kins_j["dy"][cj])
            vrel_x = vxi - vxj
            vrel_y = vyi - vyj

            aix = np.asarray(kins_i["a_ddx"][ci], dtype=np.float64).reshape(
                2,
            )
            aiy = np.asarray(kins_i["a_ddy"][ci], dtype=np.float64).reshape(
                2,
            )
            ajx = np.asarray(kins_j["a_ddx"][cj], dtype=np.float64).reshape(
                2,
            )
            ajy = np.asarray(kins_j["a_ddy"][cj], dtype=np.float64).reshape(
                2,
            )
            cix = float(kins_i["c_ddx"][ci])
            ciy = float(kins_i["c_ddy"][ci])
            cjx = float(kins_j["c_ddx"][cj])
            cjy = float(kins_j["c_ddy"][cj])

            h = dx * dx + dy * dy - float(d_safe_sq)
            dh = 2.0 * (dx * vrel_x + dy * vrel_y)

            A_i = 0.5 * (dt * dt) * 2.0 * (dx * aix + dy * aiy)
            A_j = 0.5 * (dt * dt) * -2.0 * (dx * ajx + dy * ajy)
            const_ddh = 2.0 * (vrel_x * vrel_x + vrel_y * vrel_y) + 2.0 * (
                dx * (cix - cjx) + dy * (ciy - cjy)
            )
            b0 = dh * dt + 0.5 * (dt * dt) * const_ddh
            return (
                A_i.reshape(1, 2),
                A_j.reshape(1, 2),
                np.array([b0], dtype=np.float64),
                np.array([h], dtype=np.float64),
            )
        else:
            dx = float(dx)
            dy = float(dy)
            vxi = float(kins_i["dx"][ci])
            vyi = float(kins_i["dy"][ci])
            vxj = float(kins_j["dx"][cj])
            vyj = float(kins_j["dy"][cj])
            vrel_x = vxi - vxj
            vrel_y = vyi - vyj

            aix = np.asarray(kins_i["a_ddx"][ci], dtype=np.float64).reshape(
                2,
            )
            cix = float(kins_i["c_ddx"][ci])
            aiy = np.asarray(kins_i["a_ddy"][ci], dtype=np.float64).reshape(
                2,
            )
            ciy = float(kins_i["c_ddy"][ci])

            ajx = np.asarray(kins_j["a_ddx"][cj], dtype=np.float64).reshape(
                2,
            )
            cjx = float(kins_j["c_ddx"][cj])
            ajy = np.asarray(kins_j["a_ddy"][cj], dtype=np.float64).reshape(
                2,
            )
            cjy = float(kins_j["c_ddy"][cj])

            h = dx * dx + dy * dy - float(d_safe_sq)
            dh = 2.0 * (dx * vrel_x + dy * vrel_y)
            A_i = 0.5 * (dt * dt) * 2.0 * (dx * aix + dy * aiy)
            A_j = 0.5 * (dt * dt) * -2.0 * (dx * ajx + dy * ajy)
            const_ddh = 2.0 * (vrel_x * vrel_x + vrel_y * vrel_y) + 2.0 * (
                dx * (cix - cjx) + dy * (ciy - cjy)
            )
            b = lam * h + dh * dt + 0.5 * (dt * dt) * const_ddh
            return (
                A_i.reshape(1, 2),
                A_j.reshape(1, 2),
                np.array([b], dtype=np.float64),
                np.array([h], dtype=np.float64),
            )

    def ttcbf_pair_affine_coeffs_cross(
        self, kins_i, kins_j, ci, cj, dx, dy, d_safe_sq, dt, lam
    ):
        """
        Copy the original function
        """
        dx = float(dx)
        dy = float(dy)
        vxi = float(kins_i["dx"][ci])
        vyi = float(kins_i["dy"][ci])
        vxj = float(kins_j["dx"][cj])
        vyj = float(kins_j["dy"][cj])
        vrel_x = vxi - vxj
        vrel_y = vyi - vyj

        aix = np.asarray(kins_i["a_ddx"][ci], dtype=np.float64).reshape(
            2,
        )
        aiy = np.asarray(kins_i["a_ddy"][ci], dtype=np.float64).reshape(
            2,
        )
        cix = float(kins_i["c_ddx"][ci])
        ciy = float(kins_i["c_ddy"][ci])
        cjx = float(kins_j["c_ddx"][cj])
        cjy = float(kins_j["c_ddy"][cj])

        h = dx * dx + dy * dy - float(d_safe_sq)
        dh = 2.0 * (dx * vrel_x + dy * vrel_y)

        A_i = 0.5 * (dt * dt) * 2.0 * (dx * aix + dy * aiy)
        const_ddh = 2.0 * (vrel_x * vrel_x + vrel_y * vrel_y) + 2.0 * (
            dx * (cix - cjx) + dy * (ciy - cjy)
        )
        drift = dh * dt + 0.5 * (dt * dt) * const_ddh
        if lam is None:
            b0 = drift
            return (
                A_i.reshape(1, 2),
                np.array([b0], dtype=np.float64),
                np.array([h], dtype=np.float64),
            )

        b = self._cross_cbf_drift_scale() * (drift + h * lam)
        return A_i.reshape(1, 2), np.array([b], dtype=np.float64)

    # =========================
    # Public entry point per step
    # =========================
    def update_qp(self, tensordict):
        """
        Dispatch grouped or centralized update per time step.
        """
        self.time_pseudo_dis = 0

        if not self.parameters.is_solve_qp:
            if self.parameters.is_learn_cbf_lambda and (
                self.parameters.is_using_centralized_cbf or self.agent_idx == 0
            ):
                cbf_lambda_abrupt_change_penalty = (
                    self.compute_cbf_lambda_smoothness_penalty(tensordict)
                )
                self.env.base_env.scenario_name.reward_info.rew_cbf_lambda[
                    self.env_idx, :
                ] = torch.tensor(
                    cbf_lambda_abrupt_change_penalty,
                    device=self.device,
                    dtype=torch.float32,
                )

            if self.parameters.is_using_centralized_cbf or self.agent_idx == 0:
                self._update_nominal_cbf_margin_rewards(tensordict)
            return

        if not self.parameters.is_using_centralized_cbf:
            if self.parameters.is_learn_cbf_lambda and self.agent_idx == 0:
                cbf_lambda_abrupt_change_penalty = (
                    self.compute_cbf_lambda_smoothness_penalty(tensordict)
                )
                self.env.base_env.scenario_name.reward_info.rew_cbf_lambda[
                    self.env_idx, :
                ] = torch.tensor(
                    cbf_lambda_abrupt_change_penalty,
                    device=self.device,
                    dtype=torch.float32,
                )

            self.update_fully_decentralized_cbf_qps(tensordict)
        elif self.parameters.is_grouping_agents:
            self.update_grouped_cbf_qps(tensordict)
        else:
            if self.parameters.is_learn_cbf_lambda:
                cbf_lambda_abrupt_change_penalty = (
                    self.compute_cbf_lambda_smoothness_penalty(tensordict)
                )
                self.env.base_env.scenario_name.reward_info.rew_cbf_lambda[
                    self.env_idx, :
                ] = torch.tensor(
                    cbf_lambda_abrupt_change_penalty,
                    device=self.device,
                    dtype=torch.float32,
                )

            self.update_centralized_cbf_qp(tensordict)

    def _update_nominal_cbf_margin_rewards(self, tensordict):
        """Compute nominal CBF margins and write their reward channels."""
        cbf_margins = self.compute_nominal_cbf_constraint_margins(tensordict)
        (
            r_left,
            r_right,
            r_pair,
        ) = self.compute_cbf_violation_rewards_from_margins(cbf_margins)

        reward_info = self.env.base_env.scenario_name.reward_info
        reward_info.rew_near_left_lane[self.env_idx, :] = torch.tensor(
            r_left, device=self.device, dtype=torch.float32
        )
        reward_info.rew_near_right_lane[self.env_idx, :] = torch.tensor(
            r_right, device=self.device, dtype=torch.float32
        )
        reward_info.rew_near_other_agents[self.env_idx, :] = torch.tensor(
            r_pair, device=self.device, dtype=torch.float32
        )

    def compute_nominal_cbf_constraint_margins(
        self, tensordict, agg_pair_keys: bool = False
    ):
        """Compute nominal-action CBF margins without solving a QP.

        The lane-boundary margins use the same per-agent formulation in both
        centralized and fully decentralized CBF modes because each lane
        constraint depends only on the ego control ``u_i``.

        Vehicle-vehicle margins are selected by
        ``self.parameters.is_using_centralized_cbf``. In centralized mode, each
        pair margin evaluates the joint nominal action as
        ``A_i u_i + A_j u_j + b`` so the neighbor vehicle action is included. In
        fully decentralized mode, each directed cross margin evaluates only the
        ego action and uses the half-split drift formulation implemented by
        ``ttcbf_pair_affine_coeffs_cross``.

        The returned ``pair_mode`` tells
        ``compute_cbf_violation_rewards_from_margins`` whether a violated pair
        constraint should penalize both vehicles (centralized) or only the
        directed ego vehicle (fully decentralized).

        Margin convention:
        - Each CBF inequality is written as g(u) >= 0 (with slack assumed 0 here).
        - If g >= 0: satisfied
        - If g < 0: violated, with violation magnitude = -g

        Returns:
        dict with:
            lane_L_margin: (n, n_circles) margins for left boundary CBFs
            lane_R_margin: (n, n_circles) margins for right boundary CBFs
            pair_margin: dict (i, j, ci, cj) -> margin (float)
        """
        n = int(self.parameters.n_agents)
        n_circles = int(self.parameters.n_circles_approximate_vehicle)
        cbf_coupling_range = float(self.parameters.cbf_coupling_range)

        agents: list[Vehicle] = self.env.base_env.scenario_name.world.agents
        path_ids_t = tensordict["agents", "info", "path_id"][self.env_idx]
        path_ids = [int(path_ids_t[i].item()) for i in range(n)]

        # --------- Build states and circle centers (same as your update function) ---------
        states = []
        circles_all = []
        centers_flat = []
        for i in range(n):
            s = torch.cat(
                [
                    agents[i].state.pos[self.env_idx],
                    agents[i].state.rot[self.env_idx],
                    agents[i].state.speed[self.env_idx],
                    agents[i].state.steering[self.env_idx],
                ],
                dim=-1,
            )
            states.append(s)
            circles_all.append(self.get_circle_centers(s))
            centers_flat.append(
                np.array([float(s[0].item()), float(s[1].item())], dtype=np.float64)
            )

        # --------- Applied action u_applied (accel, steer_rate) per agent ---------
        u_applied = np.zeros((n, 2), dtype=np.float64)
        learned_lambdas = self._get_env_cbf_lambdas(tensordict)

        for i in range(n):
            u_applied_i, _ = self._compute_applied_control_for_agent(
                tensordict, states[i], i
            )
            u_applied[i, :] = u_applied_i

        # --------- Lane CBF margins g_L >= 0 and g_R >= 0 ---------
        lane_L_margin = np.zeros((n, n_circles), dtype=np.float64)
        lane_R_margin = np.zeros((n, n_circles), dtype=np.float64)

        for i in range(n):
            kins_i = self.linearized_center_kinematics_coeffs(states[i])
            ui = u_applied[i, :]  # shape (2,)
            for ci in range(n_circles):
                circle_pos_i = circles_all[i][ci][0:2]
                smL, gL, HL, smR, gR, HR = self.estimate_agent_2_lane_safety_margin(
                    circle_pos_i, path_ids[i]
                )

                if self.parameters.adaptive_lambda:
                    A_L_ic, b0_L_ic, h_L_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smL, gL, HL, self.dt_taylor, None
                    )
                    A_R_ic, b0_R_ic, h_R_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smR, gR, HR, self.dt_taylor, None
                    )

                    lam_i = (
                        self._get_agent_cbf_lambda(learned_lambdas, i)
                        if self.parameters.is_learn_cbf_lambda
                        else float(self.lambda_ttcbf_predefined)
                    )

                    g_left = (
                        float(np.asarray(A_L_ic).reshape(1, 2) @ ui.reshape(2, 1))
                        + float(np.asarray(b0_L_ic).squeeze())
                        + float(np.asarray(h_L_ic).squeeze()) * lam_i
                    )

                    g_right = (
                        float(np.asarray(A_R_ic).reshape(1, 2) @ ui.reshape(2, 1))
                        + float(np.asarray(b0_R_ic).squeeze())
                        + float(np.asarray(h_R_ic).squeeze()) * lam_i
                    )
                else:
                    lam_i = (
                        self._get_agent_cbf_lambda(learned_lambdas, i)
                        if self.parameters.is_learn_cbf_lambda
                        else float(self.lambda_ttcbf_predefined)
                    )
                    A_L_ic, b_L_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smL, gL, HL, self.dt_taylor, lam_i
                    )
                    A_R_ic, b_R_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smR, gR, HR, self.dt_taylor, lam_i
                    )

                    g_left = float(
                        np.asarray(A_L_ic).reshape(1, 2) @ ui.reshape(2, 1)
                    ) + float(np.asarray(b_L_ic).squeeze())

                    g_right = float(
                        np.asarray(A_R_ic).reshape(1, 2) @ ui.reshape(2, 1)
                    ) + float(np.asarray(b_R_ic).squeeze())

                lane_L_margin[i, ci] = g_left
                lane_R_margin[i, ci] = g_right

        # --------- Pairwise CBF margins g_ij >= 0 ---------
        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe

        pair_margin = {}  # (i, j, ci, cj) -> margin float
        pair_mode = (
            "centralized"
            if self.parameters.is_using_centralized_cbf
            else "fully_decentralized"
        )

        if self.parameters.is_using_centralized_cbf:
            for i in range(n - 1):
                kins_i = self.linearized_center_kinematics_coeffs(states[i])
                ui = u_applied[i, :]

                for j in range(i + 1, n):
                    lam_i = self._get_agent_cbf_lambda(learned_lambdas, i)
                    lam_j = self._get_agent_cbf_lambda(learned_lambdas, j)
                    lambda_ttcbf = (
                        float((lam_i + lam_j) / 2.0)
                        if self.parameters.is_learn_cbf_lambda
                        else float(self.lambda_ttcbf_predefined)
                    )

                    kins_j = self.linearized_center_kinematics_coeffs(states[j])
                    uj = u_applied[j, :]

                    for ci in range(n_circles):
                        pi = circles_all[i][ci][0:2]
                        for cj in range(n_circles):
                            pj = circles_all[j][cj][0:2]
                            delta = pi - pj
                            dx = float(delta[0].item())
                            dy = float(delta[1].item())

                            if self.parameters.adaptive_lambda:
                                A_i, A_j, b0, h = self.ttcbf_pair_affine_coeffs(
                                    kins_i,
                                    kins_j,
                                    ci,
                                    cj,
                                    dx,
                                    dy,
                                    d_safe_sq,
                                    self.dt_taylor,
                                    None,
                                )
                                g_pair = (
                                    float(
                                        np.asarray(A_i).reshape(1, 2) @ ui.reshape(2, 1)
                                    )
                                    + float(
                                        np.asarray(A_j).reshape(1, 2) @ uj.reshape(2, 1)
                                    )
                                    + float(np.asarray(b0).squeeze())
                                    + float(np.asarray(h).squeeze()) * lambda_ttcbf
                                )
                            else:
                                A_i, A_j, b, _ = self.ttcbf_pair_affine_coeffs(
                                    kins_i,
                                    kins_j,
                                    ci,
                                    cj,
                                    dx,
                                    dy,
                                    d_safe_sq,
                                    self.dt_taylor,
                                    lambda_ttcbf,
                                )
                                g_pair = (
                                    float(
                                        np.asarray(A_i).reshape(1, 2) @ ui.reshape(2, 1)
                                    )
                                    + float(
                                        np.asarray(A_j).reshape(1, 2) @ uj.reshape(2, 1)
                                    )
                                    + float(np.asarray(b).squeeze())
                                )

                            pair_margin[(i, j, ci, cj)] = float(g_pair)
        else:
            e_cap_user = getattr(self.parameters, "max_cross_neighbors", None)
            if e_cap_user is None:
                e_cap = max(0, n - 1)
            else:
                e_cap = min(max(0, n - 1), max(0, int(e_cap_user)))

            for i in range(n):
                kins_i = self.linearized_center_kinematics_coeffs(states[i])
                ui = u_applied[i, :]
                nbrs = []
                for j in range(n):
                    if j == i:
                        continue
                    if (
                        _euclid(centers_flat[i], centers_flat[j])
                        <= cbf_coupling_range + 1e-9
                    ):
                        nbrs.append(j)
                nbrs = nbrs[:e_cap]

                lam_i = (
                    self._get_agent_cbf_lambda(learned_lambdas, i)
                    if self.parameters.is_learn_cbf_lambda
                    else float(self.lambda_ttcbf_predefined)
                )

                for j in nbrs:
                    kins_j = self.linearized_center_kinematics_coeffs(states[j])
                    for ci in range(n_circles):
                        pi = circles_all[i][ci][0:2]
                        for cj in range(n_circles):
                            pj = circles_all[j][cj][0:2]
                            delta = pi - pj
                            dx = float(delta[0].item())
                            dy = float(delta[1].item())

                            if self.parameters.adaptive_lambda:
                                A_i, b0, h = self.ttcbf_pair_affine_coeffs_cross(
                                    kins_i,
                                    kins_j,
                                    ci,
                                    cj,
                                    dx,
                                    dy,
                                    d_safe_sq,
                                    self.dt_taylor,
                                    None,
                                )
                                g_pair = float(
                                    np.asarray(A_i).reshape(1, 2) @ ui.reshape(2, 1)
                                ) + self._cross_cbf_drift_scale() * (
                                    float(np.asarray(b0).squeeze())
                                    + float(np.asarray(h).squeeze()) * lam_i
                                )
                            else:
                                A_i, b = self.ttcbf_pair_affine_coeffs_cross(
                                    kins_i,
                                    kins_j,
                                    ci,
                                    cj,
                                    dx,
                                    dy,
                                    d_safe_sq,
                                    self.dt_taylor,
                                    lam_i,
                                )
                                g_pair = float(
                                    np.asarray(A_i).reshape(1, 2) @ ui.reshape(2, 1)
                                ) + float(np.asarray(b).squeeze())

                            pair_margin[(i, j, ci, cj)] = float(g_pair)
        return {
            "lane_L_margin": lane_L_margin,
            "lane_R_margin": lane_R_margin,
            "pair_margin": pair_margin,
            "pair_mode": pair_mode,
        }

    def compute_cbf_violation_rewards_from_margins(self, cbf_margins: dict):
        """Convert CBF constraint margins into bounded per-agent rewards.

        ``compute_nominal_cbf_constraint_margins`` returns raw margins for
        inequalities written as ``g(u) >= 0``. A nonnegative margin means the
        nominal action satisfies that CBF constraint and receives zero penalty.
        A negative margin is a violation and is kept as a negative reward
        candidate.

        Lane rewards are per-agent worst-case reductions over the footprint
        circles:

            v_left[i] = min_c lane_L_margin[i, c]
            v_right[i] = min_c lane_R_margin[i, c]

        Pair rewards are also worst-case reductions, but the owner of each pair
        violation depends on ``pair_mode``. In centralized mode, a stored pair
        margin ``(i, j, ci, cj)`` was computed from both actions
        ``A_i u_i + A_j u_j + b``; therefore a violation is assigned to both
        agents ``i`` and ``j``. In fully decentralized mode, a stored pair
        margin ``(i, j, ci, cj)`` is a directed ego constraint for agent ``i``;
        therefore only agent ``i`` receives that violation.

        Finally, each channel is scaled by ``h_nom`` and clipped to ``[-1, 0]``.
        Thus safe constraints give ``0``, a margin of ``-0.5 * h_nom`` gives
        ``-0.5``, and any margin less than or equal to ``-h_nom`` saturates at
        ``-1``. Multiple violations do not accumulate; the most negative margin
        for each channel determines the reward.

        Returns:
            ``r_left, r_right, r_pair`` as lists of length ``n_agents`` with
            values in ``[-1, 0]``.
        """

        lane_L = np.asarray(
            cbf_margins["lane_L_margin"], dtype=np.float64
        )  # (n, n_circles)
        lane_R = np.asarray(
            cbf_margins["lane_R_margin"], dtype=np.float64
        )  # (n, n_circles)
        pair_margin = cbf_margins["pair_margin"]  # dict (i, j, ci, cj) -> float
        pair_mode = cbf_margins.get("pair_mode", "centralized")

        n = lane_L.shape[0]

        vL_agent = np.min(lane_L, axis=1)
        vR_agent = np.min(lane_R, axis=1)

        # pairwise aggregation per agent
        vP_agent = np.zeros((n,), dtype=np.float64)
        for (i, j, ci, cj), g in pair_margin.items():
            v = min(float(g), 0.0)
            vP_agent[i] = min(vP_agent[i], v)
            if pair_mode == "centralized":
                vP_agent[j] = min(vP_agent[j], v)

        def to_reward(v_agent, scale):
            v_norm = np.clip(v_agent / scale, -1.0, 0.0)
            return v_norm.tolist()

        h_nom = (
            float(self.parameters.h_nom) if self.parameters.h_nom is not None else 0.1
        )
        r_left = to_reward(vL_agent, h_nom)
        r_right = to_reward(vR_agent, h_nom)
        r_pair = to_reward(vP_agent, h_nom)

        return r_left, r_right, r_pair

    def compute_cbf_lambda_smoothness_penalty(self, tensordict):
        """
        Compute an additional penalty reward if the CBF lambda changes abruptly from the last step.

        This is to encourage smoother lambda values if you are learning lambda.

        Returns:
        None, but writes to tensordict[("agents", "reward", "cbf_lambda_smoothness_penalty")]
        """
        cur_lam = (
            tensordict.get(("agents", "cbf", "lambda"))[self.env_idx]
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )  # shape (n_agents, 1)

        # A full env reset starts a new lambda trajectory for this controller.
        if int(self.scenario.timer.step[self.env_idx].item()) == 0:
            self.hist.lambda_history.clear()

        if len(self.hist.lambda_history) == 0:
            cbf_lambda_abrupt_change_penalty = np.zeros(
                cur_lam.shape[0], dtype=np.float64
            )
        else:
            his_lam = self.hist.lambda_history[-1]  # list of (n_agents, 1) arrays
            cbf_lambda_abrupt_change_penalty = (
                -self.parameters.cbf_lambda_rew_scale * abs(his_lam - cur_lam) / 1.0
            )

        # Penalize the magnitude of lambda itself to prevent it from growing too large
        cbf_lambda_abrupt_change_penalty -= (
            self.parameters.cbf_lambda_rew_scale_abs * cur_lam**2
        )

        # Append the current lambda to this env-specific history
        self.hist.lambda_history.append(cur_lam.copy())

        return cbf_lambda_abrupt_change_penalty
