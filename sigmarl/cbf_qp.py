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
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from sigmarl.constants import AGENTS
from sigmarl.dynamics import KinematicBicycleModel
from sigmarl.rectangle_approximation import RectangleCircleApproximation
from sigmarl.helper_common import Parameters
from sigmarl.pseudo_distance import PseudoDistance
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


def group_agents(
    agent_states: List[AgentStateList],
    max_group_size: int,
    max_group_radius: float,
) -> List[List[int]]:
    """
    Group agents into spatially coherent clusters with a hard capacity constraint.

    Key properties:
      - Scalability: avoids all-pairs distance checks via spatial hashing.
      - Spatial coherence: agents in the same group are mutually close.
      - Hard constraints:
          * group size <= max_group_size
          * group spatial radius <= max_group_radius

    The algorithm follows a region-growing strategy:
      1) Partition the plane using a uniform spatial grid.
      2) Iteratively grow groups from unassigned seed agents.
      3) Add neighboring agents only if they remain close to the current group centroid.
    """

    # -----------------------------
    # Input validation
    # -----------------------------
    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive")

    n_agents = len(agent_states)
    if n_agents == 0:
        return []

    # -----------------------------
    # Estimate spatial scale
    # -----------------------------
    # We compute an approximate spatial scale from the agent distribution.
    # This is used to define a grid cell size such that each cell contains
    # only a small number of agents on average, ensuring scalability.
    xs = [s.x for s in agent_states]
    ys = [s.y for s in agent_states]

    span_x = max(xs) - min(xs) if n_agents > 1 else 1.0
    span_y = max(ys) - min(ys) if n_agents > 1 else 1.0
    area = max(span_x * span_y, 1e-6)

    # Average inter-agent spacing heuristic
    cell_size = math.sqrt(area / n_agents)
    cell_size = max(cell_size, 1e-3)

    # -----------------------------
    # Spatial hashing
    # -----------------------------
    # Each agent is assigned to a grid cell. During grouping, we only
    # consider agents in the same or neighboring cells, which avoids
    # quadratic neighbor searches.
    def cell_index(x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor(x / cell_size)),
            int(math.floor(y / cell_size)),
        )

    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, s in enumerate(agent_states):
        grid[cell_index(s.x, s.y)].append(i)

    # -----------------------------
    # Group construction
    # -----------------------------
    # We maintain a set of agents that have not yet been assigned to any group.
    unassigned = set(range(n_agents))
    groups: List[List[int]] = []

    # Moore neighborhood offsets: check the current cell and its 8 neighbors.
    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # Continue until all agents are assigned to some group.
    while unassigned:
        # -----------------------------
        # Initialize a new group
        # -----------------------------
        # Pick an arbitrary unassigned agent as the seed.
        seed = next(iter(unassigned))
        unassigned.remove(seed)

        group: List[int] = [seed]
        queue = deque([seed])

        # Initialize the group centroid.
        # We update this incrementally as agents are added.
        cx = agent_states[seed].x
        cy = agent_states[seed].y

        # -----------------------------
        # Region growing
        # -----------------------------
        # We expand the group by exploring spatial neighbors of
        # agents already in the group, subject to size and radius constraints.
        while queue and len(group) < max_group_size:
            i = queue.popleft()
            s_i = agent_states[i]
            cell_x, cell_y = cell_index(s_i.x, s_i.y)

            # Examine agents in neighboring grid cells only.
            for dx, dy in neighbor_offsets:
                cell = (cell_x + dx, cell_y + dy)
                for j in grid.get(cell, []):
                    # Skip agents that are already assigned to a group.
                    if j not in unassigned:
                        continue

                    s_j = agent_states[j]

                    # -----------------------------
                    # Spatial coherence check
                    # -----------------------------
                    # Enforce that the candidate agent lies within
                    # max_group_radius of the current group centroid.
                    dx_c = s_j.x - cx
                    dy_c = s_j.y - cy
                    if dx_c * dx_c + dy_c * dy_c > max_group_radius * max_group_radius:
                        continue

                    # -----------------------------
                    # Accept agent into group
                    # -----------------------------
                    unassigned.remove(j)
                    queue.append(j)
                    group.append(j)

                    # Incremental centroid update:
                    # new_centroid = old_centroid + (x_j - old_centroid) / k
                    k = len(group)
                    cx += (s_j.x - cx) / k
                    cy += (s_j.y - cy) / k

                    # Stop growing if capacity is reached.
                    if len(group) >= max_group_size:
                        break

                if len(group) >= max_group_size:
                    break

        groups.append(group)

    return groups


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
    lambda_ttcbf: List[float] = field(default_factory=list)
    qp_solving_t: List[float] = field(default_factory=list)
    qp_solving_iter: List[int] = field(default_factory=list)


class CBFQP:
    def __init__(self, env=None, env_idx: int = None, agent_idx: int = None, **kwargs):
        self.hist = TrajectoryHistory()

        self.env = env
        self.env_idx = env_idx
        self.agent_idx = agent_idx
        self.scenario = self.env.base_env.scenario_name
        self.map_pseudo_distance: PseudoDistance = (
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
        if self.parameters.is_grouping_agents:
            self.build_grouped_cbf_qps()
        else:
            if self.parameters.is_solve_qp:
                self.build_centralized_cbf_qp()

        self.cbf_solving_t = []

    def initialize_params(self):
        self.parameters: Parameters = self.env.base_env.scenario_name.parameters
        self.device = self.parameters.device
        self.dt = self.parameters.dt
        self.r = 2
        self.dt_taylor = float(self.r * self.dt)
        self.dx = 0.02
        self.dy = 0.02

        self.adaptive_lambda = True if self.parameters.is_solve_qp else False

        self.length = AGENTS["length"]
        self.width = AGENTS["width"]
        self.l_wb = AGENTS["l_wb"]
        self.l_f = AGENTS["l_f"]
        self.l_r = AGENTS["l_r"]

        self.v_max = torch.tensor(
            AGENTS["max_speed"], device=self.device, dtype=torch.float32
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

        self.safety_buffer = 0
        self.lambda_ttcbf = 0.1

        # Nominal controller selection
        self.nom_controller_type = self.parameters.nom_controller_type
        # CLF parameters
        self.lam_clf = float(getattr(self.parameters, "lam_clf", 2.0))
        self.ref_speed = float(getattr(self.parameters, "ref_speed", 1.0))  # m/s
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

    def _clf_errors_for_agent(
        self, state_i: torch.Tensor, ref_xy_i: torch.Tensor
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
        e_v = self.ref_speed - v
        return e_h, e_v

    def rl_action_to_u(
        self, rl_actions: torch.tensor, v: torch.tensor, steering: torch.tensor
    ):
        """
        Copy the original function

        Args:
            rl_actions: [target_speed, target_steering_angle]

        Returns:
            u: [acceleration, steering_rate]
        """
        # Transform for btach calcultion
        if rl_actions.ndim == 1 or v.ndim == 0 or steering.ndim == 0:
            rl_actions = rl_actions.unsqueeze(0)
            v = v.unsqueeze(0)
            steering = steering.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        rl_actions[:, 0] = torch.clamp(rl_actions[:, 0], min=self.v_min, max=self.v_max)
        rl_actions[:, 1] = torch.clamp(
            rl_actions[:, 1], min=self.steering_min, max=self.steering_max
        )

        u_acc = (rl_actions[:, 0] - v) / self.dt
        u_steering_rate = (rl_actions[:, 1] - steering) / self.dt

        u_acc = torch.clamp(u_acc, min=self.a_min, max=self.a_max)
        u_steering_rate = torch.clamp(
            u_steering_rate, min=self.steering_rate_min, max=self.steering_rate_max
        )

        u = torch.stack((u_acc, u_steering_rate), dim=1)

        if not is_batch:
            u.squeeze(0)

        return rl_actions, u

    def u_to_rl_action(self, u, state_v, state_steering):
        """
        Copy the original function
        """
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

        v = state_v + u[:, 0] * self.dt
        steering = state_steering + u[:, 1] * self.dt
        steering = (steering + torch.pi) % (2 * torch.pi) - torch.pi
        v = torch.clamp(v, min=self.v_min, max=self.v_max)
        steering = torch.clamp(steering, min=self.steering_min, max=self.steering_max)
        rl_action = torch.stack([v, steering], dim=1)
        if not is_batch:
            rl_action = rl_action.squeeze(0)
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
        Copy the original function
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
        dleft_results, dright_results = self.map_pseudo_distance.get_distance(
            path_id, query_points
        )
        self.time_pseudo_dis += time.time() - time_pseudo_dis_start

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
        if self.adaptive_lambda:
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
        if self.adaptive_lambda:
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
                lambda_bound - self.lambda_ttcbf
            )
            if lambda_pair is not None:
                cost += self.lambda_weight * cp.sum_squares(lambda_pair)

        prob = cp.Problem(cp.Minimize(cost), cons)

        # Cache
        if self.adaptive_lambda:
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
        rl_actions = torch.zeros((n, 2), device=self.device, dtype=torch.float32)
        nominal_actions = np.zeros((n, 2), dtype=np.float64)

        nom_acc_omega = np.zeros((n, 2), dtype=np.float64)
        nom_v_steer = np.zeros((n, 2), dtype=np.float64)

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

        # CLF errors and U_nom
        e_h_np = np.zeros((n,), dtype=np.float64)
        e_v_np = np.zeros((n,), dtype=np.float64)
        for i in range(n):
            if self.nom_controller_type == "rl":
                rl_i = tensordict[("agents", "action")][self.env_idx, i].clone()
                if self.is_obs_noise:
                    rl_i = rl_i + torch.rand_like(rl_i) * self.obs_noise_level
                rl_i, u_nom_i = self.rl_action_to_u(
                    rl_actions=rl_i, v=states[i][3], steering=states[i][4]
                )
                rl_actions[i, :] = rl_i
                # nominal_actions[i, :] = u_nom_i.detach().cpu().numpy()

                nom_acc_omega[i, :] = u_nom_i.detach().cpu().numpy()
                nom_v_steer[i, :] = rl_i.detach().cpu().numpy()

            else:  # "clf"
                ref_xy = tensordict["agents", "info", "ref"][self.env_idx, i, 4:6]
                e_h, e_v = self._clf_errors_for_agent(states[i], ref_xy)
                e_h_np[i] = e_h
                e_v_np[i] = e_v
                # nominal P controller: u1 = k_v * e_v ; u2 = k_h * e_h
                u1_nom = np.clip(self.k_clf_speed * e_v, self.a_min, self.a_max)
                u2_nom = np.clip(
                    self.k_clf_heading * e_h,
                    self.steering_rate_min,
                    self.steering_rate_max,
                )
                # nominal_actions[i, 0] = u1_nom
                # nominal_actions[i, 1] = u2_nom
                nom_acc_omega[i, 0] = u1_nom
                nom_acc_omega[i, 1] = u2_nom
                nom_v_steer[i, 0] = states[i][3] + e_v
                nom_v_steer[i, 1] = e_h

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
                if self.adaptive_lambda:
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
                        kins_i, ci, smL, gL, HL, self.dt_taylor, self.lambda_ttcbf
                    )
                    A_R_ic, b_R_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smR, gR, HR, self.dt_taylor, self.lambda_ttcbf
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
                kins_i = self.linearized_center_kinematics_coeffs(states[i])
                kins_j = self.linearized_center_kinematics_coeffs(states[j])
                for ci in range(n_circles):
                    pi = circles_all[i][ci][0:2]
                    for cj in range(n_circles):
                        pj = circles_all[j][cj][0:2]
                        delta = pi - pj
                        dx = float(delta[0].item())
                        dy = float(delta[1].item())
                        if self.adaptive_lambda:
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
                                self.lambda_ttcbf,
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

            solved = False
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

            if self.env.scenario_name.parameters.is_apply_cbf_action:
                # Only replace RL actions with CBF actions if is_apply_cbf_action is True
                if solved:
                    for i in range(n):
                        s_i = states[i]
                        actions_v_steer_safe = self.u_to_rl_action(
                            C["u_var"].value[i, :], s_i[3], s_i[4]
                        ).detach()
                        tensordict[("agents", "action")][
                            self.env_idx, i
                        ] = actions_v_steer_safe
                else:
                    # Fallback: use nominal actions
                    for i in range(n):
                        s_i = states[i]
                        actions_v_steer_nom = self.u_to_rl_action(
                            nom_acc_omega[i, :], s_i[3], s_i[4]
                        ).detach()
                        tensordict[("agents", "action")][
                            self.env_idx, i
                        ] = actions_v_steer_nom
                    self.hist.evt.append("QP-INF")

            st = prob.solver_stats
            if hasattr(st, "solve_time"):
                self.hist.qp_solving_t.append(st.solve_time)
                # print(f"[INFO] Centralized QP solving time: {st.solve_time * 1000:.2f} ms. Over all time steps: {np.mean(self.hist.qp_solving_t)*1000:.2f} ms.")
            if hasattr(st, "num_iters"):
                self.hist.qp_solving_iter.append(st.num_iters)

            # if self.adaptive_lambda and C["lambda_pair"] is not None:
            #     print(f"lambda_bound mean {np.mean(C['lambda_bound'].value):.4f}; max {np.max(C['lambda_bound'].value):.4f}; min {np.min(C['lambda_bound'].value):.4f}")
            #     print(f"lambda_pair mean {np.mean(C['lambda_pair'].value):.4f}; max {np.max(C['lambda_pair'].value):.4f}; min {np.min(C['lambda_pair'].value):.4f}")

            # For visualization of nominal actions
            if self.env.scenario_name.parameters.is_apply_cbf_action:
                if self.nom_controller_type == "rl":
                    self.env.base_env.scenario_name.world_state.nominal_action_vel[
                        self.env_idx, :
                    ] = rl_actions[:, 0]
                    self.env.base_env.scenario_name.world_state.nominal_action_steer[
                        self.env_idx, :
                    ] = rl_actions[:, 1]
                else:
                    # na = torch.from_numpy(C["U_nom"].value).to(self.device)
                    self.env.base_env.scenario_name.world_state.nominal_action_vel[
                        self.env_idx, :
                    ] = torch.tensor(
                        nom_v_steer[:, 0], device=self.device, dtype=torch.float32
                    )
                    self.env.base_env.scenario_name.world_state.nominal_action_steer[
                        self.env_idx, :
                    ] = torch.tensor(
                        nom_v_steer[:, 1], device=self.device, dtype=torch.float32
                    )
            else:
                # Store the CBF actions
                if solved:
                    for i in range(n):
                        s_i = states[i]
                        actions_v_steer_safe = self.u_to_rl_action(
                            C["u_var"].value[i, :], s_i[3], s_i[4]
                        ).detach()

                        self.env.base_env.scenario_name.world_state.nominal_action_vel[
                            self.env_idx, i
                        ] = actions_v_steer_safe[0]
                        self.env.base_env.scenario_name.world_state.nominal_action_steer[
                            self.env_idx, i
                        ] = actions_v_steer_safe[
                            1
                        ]
                else:
                    # Fallback: use nominal actions
                    for i in range(n):
                        s_i = states[i]
                        actions_v_steer_nom = self.u_to_rl_action(
                            nom_acc_omega[i, :], s_i[3], s_i[4]
                        ).detach()

                        self.env.base_env.scenario_name.world_state.nominal_action_vel[
                            self.env_idx, i
                        ] = actions_v_steer_nom[0]
                        self.env.base_env.scenario_name.world_state.nominal_action_steer[
                            self.env_idx, i
                        ] = actions_v_steer_nom[
                            1
                        ]

                    self.hist.evt.append("QP-INF")
        else:
            if self.adaptive_lambda:
                # Ensure lambda variables have values before reading constraint.expr.value
                if C["lambda_bound"].value is None:
                    C["lambda_bound"].value = np.full(
                        C["lambda_bound"].shape,
                        float(self.lambda_ttcbf),
                        dtype=np.float64,
                    )
                if (
                    C.get("lambda_pair", None) is not None
                    and C["lambda_pair"].value is None
                ):
                    C["lambda_pair"].value = np.full(
                        C["lambda_pair"].shape,
                        float(self.lambda_ttcbf),
                        dtype=np.float64,
                    )

            # Read constraint.expr.value for CBF constraints only (lane + pairwise).
            self.nominal_cbf_constraint_values = (
                self._get_nominal_cbf_constraint_values(tol=1e-9)
            )

            r_left, r_right, r_pair = self.compute_nominal_cbf_violation_rewards()
            self.env.base_env.scenario_name.rew_left_bound[
                self.env_idx, :
            ] = torch.tensor(r_left, device=self.device, dtype=torch.float32)
            self.env.base_env.scenario_name.rew_right_bound[
                self.env_idx, :
            ] = torch.tensor(r_right, device=self.device, dtype=torch.float32)
            self.env.base_env.scenario_name.rew_agent_pair[
                self.env_idx, :
            ] = torch.tensor(r_pair, device=self.device, dtype=torch.float32)

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
    def build_grouped_cbf_qps(self):
        """
        Prebuild G group QPs. Each group has capacity m = max_group_size.
        Cross-group constraints are preallocated against up to (N-1) external agents per local agent.
        Includes CLF relaxations per local slot.
        """
        print("[INFO] Building grouped CBF-QPs...")
        if hasattr(self, "_group_qp_caches"):
            return

        N = int(self.parameters.n_agents)
        C = int(self.parameters.n_circles_approximate_vehicle)
        m = int(self.parameters.max_group_size)
        assert m >= 1
        G = int(ceil(N / m))

        self._group_qp_caches: List[dict] = []
        self._group_capacity = m
        self._group_count = G
        self._group_circle_count = C
        self._group_cross_cap = max(0, N - 1)

        def idx_lane(i_slot, i_c, side):
            return 2 * (i_slot * C + i_c) + side

        def pair_slot_base(mcap, C):
            base = {}
            k = 0
            for i in range(mcap - 1):
                for j in range(i + 1, mcap):
                    base[(i, j)] = k
                    k += C * C
            return base, k

        pair_base, pair_count = pair_slot_base(m, C)
        E = self._group_cross_cap
        cross_base = {}
        kx = 0
        for i in range(m):
            for e in range(E):
                cross_base[(i, e)] = kx
                kx += C * C
        cross_count = kx

        for g in range(G):
            u_var = cp.Variable((m, 2))
            s_lane = cp.Variable(2 * m * C, nonneg=True)
            s_pair = cp.Variable(pair_count, nonneg=True) if pair_count > 0 else None
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
            b0_L = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
            h_L = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
            A_R = [[cp.Parameter((1, 2)) for _ in range(C)] for _ in range(m)]
            b0_R = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
            h_R = [[cp.Parameter((1,)) for _ in range(C)] for _ in range(m)]
            lambda_lane = cp.Variable(2 * m * C)
            cons += [lambda_lane >= 0, lambda_lane <= 1]
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

            # Intra-group pairs
            Aij_i = {}
            Aij_j = {}
            b0ij = {}
            hij = {}
            lambda_pair = cp.Variable(pair_count) if pair_count > 0 else None
            if lambda_pair is not None:
                cons += [lambda_pair >= 0, lambda_pair <= 1]
            for i in range(m - 1):
                for j in range(i + 1, m):
                    for ci in range(C):
                        for cj in range(C):
                            Aij_i[(i, j, ci, cj)] = cp.Parameter((1, 2))
                            Aij_j[(i, j, ci, cj)] = cp.Parameter((1, 2))
                            b0ij[(i, j, ci, cj)] = cp.Parameter((1,))
                            hij[(i, j, ci, cj)] = cp.Parameter((1,))
                            if s_pair is None:
                                cons += [
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + b0ij[(i, j, ci, cj)]
                                    + hij[(i, j, ci, cj)] * 1.0
                                    >= 0
                                ]
                            else:
                                cons += [
                                    Aij_i[(i, j, ci, cj)] @ u_var[i, :]
                                    + Aij_j[(i, j, ci, cj)] @ u_var[j, :]
                                    + b0ij[(i, j, ci, cj)]
                                    + hij[(i, j, ci, cj)]
                                    * lambda_pair[pair_base[(i, j)] + ci * C + cj]
                                    >= -s_pair[pair_base[(i, j)] + ci * C + cj]
                                ]

            # Cross-group reciprocal (only i's control)
            Axc_i = {}
            b0xc = {}
            hxc = {}
            lambdax_i = cp.Variable(cross_count) if cross_count > 0 else None
            if lambdax_i is not None:
                cons += [lambdax_i >= 0, lambdax_i <= 1]
            for i in range(m):
                for e in range(E):
                    for ci in range(C):
                        for cj in range(C):
                            Axc_i[(i, e, ci, cj)] = cp.Parameter((1, 2))
                            b0xc[(i, e, ci, cj)] = cp.Parameter((1,))
                            hxc[(i, e, ci, cj)] = cp.Parameter((1,))
                            if s_cross is not None and lambdax_i is not None:
                                cons += [
                                    Axc_i[(i, e, ci, cj)] @ u_var[i, :]
                                    + b0xc[(i, e, ci, cj)]
                                    + self.parameters.rs
                                    * hxc[(i, e, ci, cj)]
                                    * lambdax_i[
                                        cross_base[(i, e)] + ci * C + cj
                                    ]  # TODO: Now receiprocal_sigma is fixed as 0.5 (defined by self.parameters.rs). Code adaptation is required if using other values because it is no longer symmetric.
                                    >= -s_cross[cross_base[(i, e)] + ci * C + cj]
                                ]
                            else:
                                cons += [
                                    Axc_i[(i, e, ci, cj)] @ u_var[i, :]
                                    + b0xc[(i, e, ci, cj)]
                                    + hxc[(i, e, ci, cj)] * 1.0
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
                    lambda_lane - self.lambda_ttcbf
                )
                if lambda_pair is not None:
                    cost += self.lambda_weight * cp.sum_squares(lambda_pair)

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
                b0_L=b0_L,
                h_L=h_L,
                A_R=A_R,
                b0_R=b0_R,
                h_R=h_R,
                lambda_lane=lambda_lane,
                Aij_i=Aij_i,
                Aij_j=Aij_j,
                b0ij=b0ij,
                hij=hij,
                lambda_pair=lambda_pair,
                Axc_i=Axc_i,
                b0xc=b0xc,
                hxc=hxc,
                lambdax_i=lambdax_i,
                idx_lane=idx_lane,
                prob=prob,
            )

            # Warm start
            U_nom.value = np.zeros((m, 2))
            s_lane.value = np.zeros((2 * m * C,))
            s_clf_head.value = np.zeros((m,))
            s_clf_speed.value = np.zeros((m,))
            e_head.value = np.zeros((m,))
            e_speed.value = np.zeros((m,))
            v_head_const.value = np.zeros((m,))
            v_speed_const.value = np.zeros((m,))
            lambda_lane.value = np.zeros((2 * m * C,))
            if pair_count is not None and lambda_pair is not None:
                lambda_pair.value = np.zeros((pair_count,))
            if cross_count is not None and lambdax_i is not None:
                lambdax_i.value = np.zeros((cross_count,))
            if s_pair is not None:
                s_pair.value = np.zeros((pair_count,))
            if s_cross is not None:
                s_cross.value = np.zeros((cross_count,))
            u_var.value = np.zeros((m, 2))
            self._group_qp_caches.append(cache)

        self._group_assignment: List[List[int]] = [[] for _ in range(G)]
        # Baseline: fixed groups throughout the simulation
        self.use_fixed_groups = True  # baseline switch
        self.fixed_groups = None  # will be computed once on first update

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
        E = self._group_cross_cap
        obs_range = float(self.parameters.observation_range)
        sigma = self.parameters.rs

        # Collect states and centers
        path_ids_t = tensordict["agents", "info", "path_id"][self.env_idx]
        path_ids = [int(path_ids_t[i].item()) for i in range(N)]
        states = []
        centers = []
        centers_flat = []
        for i in range(N):
            s = torch.cat(
                [
                    agents[i].state.pos,
                    agents[i].state.rot,
                    agents[i].state.speed,
                    agents[i].state.steering,
                ],
                dim=1,
            ).squeeze(0)
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

                # Sort agents inside each group so that:
                #   slot index k -> global agent id mapping is deterministic.
                # This avoids random permutations of decision-variable rows inside a QP.
                groups0 = [sorted(list(g)) for g in groups0]

                # Optional: also sort group list itself for reproducibility across runs.
                # Not strictly required because groups are frozen after this point.
                groups0.sort(key=lambda g: g[0] if len(g) > 0 else 10**9)

                if len(groups0) != G:
                    raise RuntimeError(
                        f"Fixed grouping produced {len(groups0)} groups, expected {G}."
                    )

                self.fixed_groups = groups0

            groups = self.fixed_groups
        else:
            # Recompute groups every step (dynamic grouping)
            groups = group_agents_k_nearest(asl, m)

            # ---- Deterministic ordering for warm-start and reproducibility ----

            # (1) Sort agents inside each group:
            #     ensures that slot k in the group QP always refers to the same
            #     global agent id, given the same group membership.
            groups = [sorted(list(g)) for g in groups]

            # (2) Sort the list of groups:
            #     ensures that group index g maps to the same QP cache across time,
            #     as long as group memberships do not change.
            #     We use the smallest agent id as a stable group label.
            groups.sort(key=lambda g: g[0] if len(g) > 0 else 10**9)

        # Precompute nominal actions or CLF errors for all agents
        U_nom_all = np.zeros((N, 2), dtype=np.float64)
        rl_actions_all = torch.zeros((N, 2), device=self.device, dtype=torch.float32)
        e_h_all = np.zeros((N,), dtype=np.float64)
        e_v_all = np.zeros((N,), dtype=np.float64)
        for i in range(N):
            if self.nom_controller_type == "rl":
                rl_i = tensordict[("agents", "action")][self.env_idx, i].clone()
                if self.is_obs_noise:
                    rl_i = rl_i + torch.rand_like(rl_i) * self.obs_noise_level
                rl_i, u_nom_i = self.rl_action_to_u(
                    rl_actions=rl_i, v=states[i][3], steering=states[i][4]
                )
                rl_actions_all[i, :] = rl_i
                U_nom_all[i, :] = u_nom_i.detach().cpu().numpy()
            else:
                ref_xy = tensordict["agents", "info", "ref"][self.env_idx, i, 4:6]
                e_h, e_v = self._clf_errors_for_agent(states[i], ref_xy)
                e_h_all[i] = e_h
                e_v_all[i] = e_v
                U_nom_all[i, 0] = np.clip(
                    self.k_clf_speed * e_v, self.a_min, self.a_max
                )  # u1
                U_nom_all[i, 1] = np.clip(
                    self.k_clf_heading * e_h,
                    self.steering_rate_min,
                    self.steering_rate_max,
                )  # u2

        # Safety radius
        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe

        qp_solving_t_groups = []
        qp_solving_iter_groups = []
        cross_groups = {}

        for g in range(G):
            cache = self._group_qp_caches[g]
            mcap = cache["m"]
            Cc = cache["C"]
            Ecap = cache["E"]
            idx_lane = cache["idx_lane"]

            members = groups[g][:mcap]
            n_local = len(members)
            slot2id = [-1] * mcap
            for k in range(n_local):
                slot2id[k] = members[k]

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
                else:
                    U_loc[k, :] = 0.0
                    e_head_loc[k] = 0.0
                    e_speed_loc[k] = 0.0
                    v_head_const[k] = 0.0
                    v_speed_const[k] = 0.0

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
                    for ci in range(Cc):
                        cache["A_L"][i_slot][ci].value = np.zeros((1, 2))
                        cache["b0_L"][i_slot][ci].value = np.array([1.0])
                        cache["h_L"][i_slot][ci].value = np.array([0.0])
                        cache["A_R"][i_slot][ci].value = np.zeros((1, 2))
                        cache["b0_R"][i_slot][ci].value = np.array([1.0])
                        cache["h_R"][i_slot][ci].value = np.array([0.0])

            # Reset slacks and warm start
            cache["s_lane"].value = np.zeros(cache["s_lane"].shape)
            if cache["s_pair"] is not None:
                cache["s_pair"].value = np.zeros(cache["s_pair"].shape)
            if cache["s_cross"] is not None:
                cache["s_cross"].value = np.zeros(cache["s_cross"].shape)
            cache["s_clf_head"].value = np.zeros(cache["s_clf_head"].shape)
            cache["s_clf_speed"].value = np.zeros(cache["s_clf_speed"].shape)
            cache["u_var"].value = cache["U_nom"].value.copy()
            cache["lambda_lane"].value = np.zeros(cache["lambda_lane"].shape)
            if cache["lambda_pair"] is not None:
                cache["lambda_pair"].value = np.zeros(cache["lambda_pair"].shape)
            if cache["lambdax_i"] is not None:
                cache["lambdax_i"].value = np.zeros(cache["lambdax_i"].shape)

            # Intra-group pairs
            for i_loc in range(mcap - 1):
                for j_loc in range(i_loc + 1, mcap):
                    for ci in range(Cc):
                        for cj in range(Cc):
                            key = (i_loc, j_loc, ci, cj)
                            if i_loc < n_local and j_loc < n_local:
                                i = slot2id[i_loc]
                                j = slot2id[j_loc]
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
                                cache["Aij_i"][key].value = A_i
                                cache["Aij_j"][key].value = A_j
                                cache["b0ij"][key].value = b0
                                cache["hij"][key].value = h
                            else:
                                cache["Aij_i"][key].value = np.zeros((1, 2))
                                cache["Aij_j"][key].value = np.zeros((1, 2))
                                cache["b0ij"][key].value = np.array([1.0])
                                cache["hij"][key].value = np.array([0.0])

            # Cross-group reciprocal
            for i_loc in range(mcap):
                if i_loc < n_local:
                    i = slot2id[i_loc]
                    nbrs: List[int] = []
                    for j in range(N):
                        if j in members:
                            continue
                        if (
                            _euclid(centers_flat[i], centers_flat[j])
                            <= obs_range + 1e-9
                        ):
                            nbrs.append(j)
                    nbrs = nbrs[:Ecap]

                    cross_groups[i] = nbrs
                    for e_idx in range(Ecap):
                        if e_idx < len(nbrs):
                            j = nbrs[e_idx]
                            kins_i = self.linearized_center_kinematics_coeffs(states[i])
                            kins_j = self.linearized_center_kinematics_coeffs(states[j])
                            for ci in range(Cc):
                                for cj in range(Cc):
                                    pi = centers[i][ci][0:2]
                                    pj = centers[j][cj][0:2]
                                    dx = float((pi - pj)[0].item())
                                    dy = float((pi - pj)[1].item())
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
                                        sigma_i=sigma,
                                    )
                                    cache["Axc_i"][(i_loc, e_idx, ci, cj)].value = A_i
                                    cache["b0xc"][(i_loc, e_idx, ci, cj)].value = b0
                                    cache["hxc"][(i_loc, e_idx, ci, cj)].value = h
                        else:
                            for ci in range(Cc):
                                for cj in range(Cc):
                                    cache["Axc_i"][
                                        (i_loc, e_idx, ci, cj)
                                    ].value = np.zeros((1, 2))
                                    cache["b0xc"][
                                        (i_loc, e_idx, ci, cj)
                                    ].value = np.array([1.0])
                                    cache["hxc"][
                                        (i_loc, e_idx, ci, cj)
                                    ].value = np.array([0.0])
                else:
                    for e_idx in range(Ecap):
                        for ci in range(Cc):
                            for cj in range(Cc):
                                cache["Axc_i"][(i_loc, e_idx, ci, cj)].value = np.zeros(
                                    (1, 2)
                                )
                                cache["b0xc"][(i_loc, e_idx, ci, cj)].value = np.array(
                                    [1.0]
                                )
                                cache["hxc"][(i_loc, e_idx, ci, cj)].value = np.array(
                                    [0.0]
                                )

            # Solve group QP
            # prob: cp.Problem = cache["prob"]
            # try:
            #     prob.solve(
            #         solver=cp.OSQP,
            #         warm_start=True,
            #         verbose=False,
            #         max_iter=40000,
            #         eps_abs=1e-5,
            #         eps_rel=1e-5,
            #         polish=True,
            #         adaptive_rho=True,
            #     )
            # except cp.SolverError:
            #     try:
            #         prob.solve(
            #             solver=cp.ECOS,
            #             warm_start=True,
            #             verbose=False,
            #             max_iters=20000,
            #             abstol=1e-8,
            #             reltol=1e-8,
            #         )
            #     except cp.SolverError:
            #         prob.solve(
            #             solver=cp.SCS,
            #             warm_start=True,
            #             verbose=False,
            #             max_iters=100000,
            #             eps=1e-4,
            #         )

            # # Write back
            # u_opt = cache["u_var"].value
            # for k in range(n_local):
            #     gid = slot2id[k]
            #     s_i = states[gid]
            #     actions_v_steer_safe = self.u_to_rl_action(
            #         u_opt[k, :], s_i[3], s_i[4]
            #     ).detach()
            #     tensordict[("agents", "action")][
            #         self.env_idx, gid
            #     ] = actions_v_steer_safe

            # Solve group QP
            prob: cp.Problem = cache["prob"]
            solved = False
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
                and cache["u_var"].value is not None
                and np.isfinite(cache["u_var"].value).all()
            )

            if solved:
                u_opt = cache["u_var"].value
                for k in range(n_local):
                    gid = slot2id[k]
                    s_i = states[gid]
                    actions_v_steer_safe = self.u_to_rl_action(
                        u_opt[k, :], s_i[3], s_i[4]
                    ).detach()
                    tensordict[("agents", "action")][
                        self.env_idx, gid
                    ] = actions_v_steer_safe
            else:
                # Fallback: nominal actions for this group
                for k in range(n_local):
                    gid = slot2id[k]
                    s_i = states[gid]
                    actions_v_steer_nom = self.u_to_rl_action(
                        U_nom_all[gid, :], s_i[3], s_i[4]
                    ).detach()
                    tensordict[("agents", "action")][
                        self.env_idx, gid
                    ] = actions_v_steer_nom
                self.hist.evt.append("QP-INF")

            st = prob.solver_stats
            if hasattr(st, "solve_time"):
                qp_solving_t_groups.append(st.solve_time)
            if hasattr(st, "num_iters"):
                qp_solving_iter_groups.append(st.num_iters)

        # Example: inter-groups = [[8, 10], [0, 12], [1, 2], [3, 4], [5, 6], [7, 9], [11, 13], [14]], cross-groups = {8: [0], 10: [0, 12], 0: [8, 10], 12: [10], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 9: [], 11: [], 13: [], 14: []}
        # print(f"inter-groups = {groups}, cross-groups = {cross_groups}")
        # For later visualization
        self.scenario.inter_groups = groups
        self.scenario.cross_groups = cross_groups

        self.hist.qp_solving_t.append(qp_solving_t_groups)
        t_avg = sum(qp_solving_t_groups) / max(1, len(qp_solving_t_groups))
        # print(f"[INFO] Semi-centralized QP solving time per group: {t_avg*1000:.2f} ms. Over all time steps: {np.mean(np.array(self.hist.qp_solving_t))*1000:.2f} ms")
        self.hist.qp_solving_iter.append(qp_solving_iter_groups)

        # For visualization of nominal actions
        if self.nom_controller_type == "rl":
            self.env.base_env.scenario_name.world_state.nominal_action_vel[
                self.env_idx, :
            ] = rl_actions_all[:, 0]
            self.env.base_env.scenario_name.world_state.nominal_action_steer[
                self.env_idx, :
            ] = rl_actions_all[:, 1]
        else:
            na = torch.from_numpy(U_nom_all).to(self.device)
            self.env.base_env.scenario_name.world_state.nominal_action_vel[
                self.env_idx, :
            ] = states[0][3].new_tensor(na[:, 0])
            self.env.base_env.scenario_name.world_state.nominal_action_steer[
                self.env_idx, :
            ] = states[0][4].new_tensor(na[:, 1])

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
        if self.adaptive_lambda:
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
        if self.adaptive_lambda:
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
        self, kins_i, kins_j, ci, cj, dx, dy, d_safe_sq, dt, lam, sigma_i: float = 0.5
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
        b0 = dh * dt + 0.5 * (dt * dt) * const_ddh
        return (
            A_i.reshape(1, 2),
            np.array([b0], dtype=np.float64),
            np.array([h], dtype=np.float64),
        )

    # =========================
    # Public entry point per step
    # =========================
    def update_qp(self, tensordict):
        """
        Dispatch grouped or centralized update per time step.
        """
        self.time_pseudo_dis = 0
        if self.parameters.is_grouping_agents:
            self.update_grouped_cbf_qps(tensordict)
        else:
            if self.parameters.is_solve_qp:
                self.update_centralized_cbf_qp(tensordict)
            else:
                cbf_margins = self.compute_nominal_cbf_constraint_margins(tensordict)
                (
                    r_left,
                    r_right,
                    r_pair,
                ) = self.compute_cbf_violation_rewards_from_margins(cbf_margins)
                self.env.base_env.scenario_name.rew_left_bound[
                    self.env_idx, :
                ] = torch.tensor(r_left, device=self.device, dtype=torch.float32)
                self.env.base_env.scenario_name.rew_right_bound[
                    self.env_idx, :
                ] = torch.tensor(r_right, device=self.device, dtype=torch.float32)
                self.env.base_env.scenario_name.rew_agent_pair[
                    self.env_idx, :
                ] = torch.tensor(r_pair, device=self.device, dtype=torch.float32)

    def compute_nominal_cbf_constraint_margins(
        self, tensordict, agg_pair_keys: bool = False
    ):
        """
        Compute nominal CBF constraint *margins* without CVXPY.

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

        agents: list[Vehicle] = self.env.base_env.scenario_name.world.agents
        path_ids_t = tensordict["agents", "info", "path_id"][self.env_idx]
        path_ids = [int(path_ids_t[i].item()) for i in range(n)]

        # --------- Build states and circle centers (same as your update function) ---------
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

        # --------- Nominal action u_nom (accel, steer_rate) per agent ---------
        u_nom = np.zeros((n, 2), dtype=np.float64)

        if self.nom_controller_type == "rl":
            for i in range(n):
                rl_i = tensordict[("agents", "action")][self.env_idx, i].clone()
                if self.is_obs_noise:
                    rl_i = rl_i + torch.rand_like(rl_i) * self.obs_noise_level

                # Your function returns (rl_actions, u_nom)
                _, u_nom_i = self.rl_action_to_u(
                    rl_actions=rl_i, v=states[i][3], steering=states[i][4]
                )
                u_nom[i, :] = u_nom_i.detach().cpu().numpy()
        else:
            # "clf" nominal controller
            for i in range(n):
                ref_xy = tensordict["agents", "info", "ref"][self.env_idx, i, 4:6]
                e_h, e_v = self._clf_errors_for_agent(states[i], ref_xy)
                u1_nom = np.clip(self.k_clf_speed * e_v, self.a_min, self.a_max)
                u2_nom = np.clip(
                    self.k_clf_heading * e_h,
                    self.steering_rate_min,
                    self.steering_rate_max,
                )
                u_nom[i, 0] = float(u1_nom)
                u_nom[i, 1] = float(u2_nom)

        # --------- Lane CBF margins g_L >= 0 and g_R >= 0 ---------
        lane_L_margin = np.zeros((n, n_circles), dtype=np.float64)
        lane_R_margin = np.zeros((n, n_circles), dtype=np.float64)

        for i in range(n):
            kins_i = self.linearized_center_kinematics_coeffs(states[i])
            ui = u_nom[i, :]  # shape (2,)
            for ci in range(n_circles):
                circle_pos_i = circles_all[i][ci][0:2]
                smL, gL, HL, smR, gR, HR = self.estimate_agent_2_lane_safety_margin(
                    circle_pos_i, path_ids[i]
                )

                if self.adaptive_lambda:
                    # returns (A, b0, h); inequality: A u + b0 + h*lambda >= 0
                    A_L_ic, b0_L_ic, h_L_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smL, gL, HL, self.dt_taylor, None
                    )
                    A_R_ic, b0_R_ic, h_R_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smR, gR, HR, self.dt_taylor, None
                    )

                    # If you do not optimize lambda, you must pick a value.
                    # Here we use your default lambda_ttcbf.
                    lam_L = float(self.lambda_ttcbf)
                    lam_R = float(self.lambda_ttcbf)

                    g_left = (
                        float(np.asarray(A_L_ic).reshape(1, 2) @ ui.reshape(2, 1))
                        + float(np.asarray(b0_L_ic).squeeze())
                        + float(np.asarray(h_L_ic).squeeze()) * lam_L
                    )

                    g_right = (
                        float(np.asarray(A_R_ic).reshape(1, 2) @ ui.reshape(2, 1))
                        + float(np.asarray(b0_R_ic).squeeze())
                        + float(np.asarray(h_R_ic).squeeze()) * lam_R
                    )
                else:
                    # returns (A, b); inequality: A u + b >= 0
                    A_L_ic, b_L_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smL, gL, HL, self.dt_taylor, self.lambda_ttcbf
                    )
                    A_R_ic, b_R_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smR, gR, HR, self.dt_taylor, self.lambda_ttcbf
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

        for i in range(n - 1):
            kins_i = self.linearized_center_kinematics_coeffs(states[i])
            ui = u_nom[i, :]

            for j in range(i + 1, n):
                kins_j = self.linearized_center_kinematics_coeffs(states[j])
                uj = u_nom[j, :]

                for ci in range(n_circles):
                    pi = circles_all[i][ci][0:2]
                    for cj in range(n_circles):
                        pj = circles_all[j][cj][0:2]
                        delta = pi - pj
                        dx = float(delta[0].item())
                        dy = float(delta[1].item())

                        if self.adaptive_lambda:
                            # returns (A_i, A_j, b0, h); inequality: A_i u_i + A_j u_j + b0 + h*lambda >= 0
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
                            lam = float(self.lambda_ttcbf)

                            g_pair = (
                                float(np.asarray(A_i).reshape(1, 2) @ ui.reshape(2, 1))
                                + float(
                                    np.asarray(A_j).reshape(1, 2) @ uj.reshape(2, 1)
                                )
                                + float(np.asarray(b0).squeeze())
                                + float(np.asarray(h).squeeze()) * lam
                            )
                        else:
                            # returns (A_i, A_j, b, h) but your non-adaptive branch uses b in the constraint
                            A_i, A_j, b, _ = self.ttcbf_pair_affine_coeffs(
                                kins_i,
                                kins_j,
                                ci,
                                cj,
                                dx,
                                dy,
                                d_safe_sq,
                                self.dt_taylor,
                                self.lambda_ttcbf,
                            )
                            g_pair = (
                                float(np.asarray(A_i).reshape(1, 2) @ ui.reshape(2, 1))
                                + float(
                                    np.asarray(A_j).reshape(1, 2) @ uj.reshape(2, 1)
                                )
                                + float(np.asarray(b).squeeze())
                            )

                        pair_margin[(i, j, ci, cj)] = float(g_pair)

        return {
            "lane_L_margin": lane_L_margin,
            "lane_R_margin": lane_R_margin,
            "pair_margin": pair_margin,
        }

    def compute_cbf_violation_rewards_from_margins(
        self, cbf_margins: dict, agg: str = "max", eps: float = 1e-9
    ):
        """
        Compute three per-agent reward signals from CBF margins (no CVXPY).

        For each constraint margin g:
        - violation magnitude = max(0, -g)
        - reward is 0 if no violation, else negative
        - normalized per channel so the worst violation in the step maps near -1

        Returns:
        r_left, r_right, r_pair: Python lists of length n_agents, values in [-1, 0]
        """
        lane_L = np.asarray(
            cbf_margins["lane_L_margin"], dtype=np.float64
        )  # (n, n_circles)
        lane_R = np.asarray(
            cbf_margins["lane_R_margin"], dtype=np.float64
        )  # (n, n_circles)
        pair_margin = cbf_margins["pair_margin"]  # dict (i, j, ci, cj) -> float

        n = lane_L.shape[0]

        # violation magnitudes
        vL = np.maximum(-lane_L, 0.0)  # (n, n_circles)
        vR = np.maximum(-lane_R, 0.0)  # (n, n_circles)

        if agg == "sum":
            vL_agent = np.sum(vL, axis=1)
            vR_agent = np.sum(vR, axis=1)
        elif agg == "max":
            vL_agent = np.max(vL, axis=1)
            vR_agent = np.max(vR, axis=1)
        else:
            raise ValueError(f"Unknown agg='{agg}'. Use 'max' or 'sum'.")

        # pairwise aggregation per agent
        vP_agent = np.zeros((n,), dtype=np.float64)
        for (i, j, ci, cj), g in pair_margin.items():
            v = max(-float(g), 0.0)
            if agg == "sum":
                vP_agent[i] += v
                vP_agent[j] += v
            else:  # "max"
                vP_agent[i] = max(vP_agent[i], v)
                vP_agent[j] = max(vP_agent[j], v)

        # normalize each channel to [-1, 0]
        scale_L = float(np.max(vL_agent)) + eps
        scale_R = float(np.max(vR_agent)) + eps
        scale_P = float(np.max(vP_agent)) + eps

        def to_reward(v_agent, scale):
            v_norm = np.clip(v_agent / scale, 0.0, 1.0)
            return (-v_norm).tolist()

        r_left = to_reward(vL_agent, scale_L)
        r_right = to_reward(vR_agent, scale_R)
        r_pair = to_reward(vP_agent, scale_P)

        return r_left, r_right, r_pair
