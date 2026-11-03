# Copyright (c) 2025, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import numpy as np
import proxsuite
import time
import math
from math import ceil, atan2, pi
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from sigmarl.constants import (
    ACCELERATION_SCALE,
    AGENTS,
    H_NOM,
    LENGTH_SCALE,
    POSITION_SPEED_LATERAL_OFFSET_BOUND,
    POSITION_SPEED_LOOKAHEAD_DIST,
    SPEED_SCALE,
)
from sigmarl.dynamics import KinematicBicycleModel
from sigmarl.rectangle_approximation import RectangleCircleApproximation
from sigmarl.helper_common import (
    Parameters,
    compute_applied_action_control,
    compute_path_tracking_target_steering,
)
from sigmarl.helper_group import (
    group_agents_by_interaction_risk,
    sort_neighbors_by_interaction_risk,
)
from sigmarl.helper_scenario import compute_pseudo_tangent_vector
from sigmarl.pseudo_distance import PseudoDistance
from sigmarl.softmin_distance import SoftminDistance
from sigmarl.lane_sat_barrier import LaneSatBarrier
from sigmarl.pair_sat_barrier import PairSatBarrier
from sigmarl.backup_horizon_cbf import WRITE_BARRIER, run_steps
from sigmarl.cbf_baselines import (
    make_backup_horizon,
    make_lane_baseline,
    make_pair_baseline,
    tuned_cbf_lambda,
)
from sigmarl.mtv_based_sm_predictor import SafetyMarginEstimatorModule  # TODO?
from sigmarl.helper_common import Vehicle

PROXQP_INF = 1e20
# Parts of the CBF time of a step (CBFQP.time_parts): the QP solves, the exact
# check of the solutions (with its repair rows), and the fallback controls
TIME_PARTS = ("qp", "check", "fallback")
PROXQP_SUCCESS_STATUSES = {
    proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED,
    proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE,
}


def qp_statistics(cbf_controllers):
    """Number of QP solves and failures (by status) summed over (nested lists of) CBFQP controllers."""
    if isinstance(cbf_controllers, CBFQP):
        return cbf_controllers.n_qp_solves, Counter(cbf_controllers.qp_failure_counts)
    n_solves, failures = 0, Counter()
    for ctrl in cbf_controllers:
        n, f = qp_statistics(ctrl)
        n_solves += n
        failures += f
    return n_solves, failures


def escape_statistics(cbf_controllers) -> Counter:
    """Standstill-escape counts summed over (nested lists of) CBFQP controllers."""
    if isinstance(cbf_controllers, CBFQP):
        return Counter(cbf_controllers.escape_stats)
    total = Counter()
    for ctrl in cbf_controllers:
        total += escape_statistics(ctrl)
    return total


# -------- Grouping state structure --------
@dataclass
class AgentStateList:
    x: float
    y: float
    v: float
    theta: float = 0.0


def _euclid(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return float(np.sqrt(d[0] * d[0] + d[1] * d[1]))


class _LazyAgentValues(Sequence):
    """Per-agent values fn(item), each computed on its first access and cached.

    The exact-footprint (rectangle_sat) barriers read neither the circle
    centers nor the kinematic coefficients that the circle barriers need for
    every agent, so these are computed only for the agents whose values are
    read.
    """

    def __init__(self, fn, items):
        self._fn = fn
        self._items = items
        self._values = [None] * len(items)

    def __getitem__(self, idx):
        if self._values[idx] is None:
            self._values[idx] = self._fn(self._items[idx])
        return self._values[idx]

    def __len__(self):
        return len(self._items)


class _LazyKinematics(Mapping):
    """Kinematic coefficients of one agent (CBFQP.linearized_center_kinematics_coeffs),
    computed on the first key access (see _LazyAgentValues)."""

    def __init__(self, ctrl, state):
        self._ctrl = ctrl
        self._state = state
        self._kins = None

    def _values(self) -> dict:
        if self._kins is None:
            self._kins = self._ctrl.linearized_center_kinematics_coeffs(self._state)
        return self._kins

    def __getitem__(self, key):
        return self._values()[key]

    def __iter__(self):
        return iter(self._values())

    def __len__(self):
        return len(self._values())


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
        self.lane_safety_buffer_sq = float(
            (self.circle_radius + self.safety_buffer) ** 2 - self.circle_radius**2
        )
        self.lane_safety_buffer = (
            self.lane_safety_buffer_sq
            if self.parameters.lane_distance_type == "softmin"
            else self.safety_buffer
        )
        self.lane_cbf_footprint = self.parameters.lane_cbf_footprint
        if self.lane_cbf_footprint not in ("circle", "rectangle", "rectangle_sat"):
            raise ValueError(
                f"Unsupported lane_cbf_footprint: {self.lane_cbf_footprint}"
            )
        if (
            self.lane_cbf_footprint == "rectangle"
            and self.parameters.lane_distance_type != "softmin"
        ):
            raise ValueError(
                "The rectangle lane CBF footprint requires softmin distances."
            )
        if (
            self.lane_cbf_footprint == "rectangle_sat"
            and self.parameters.is_aggregate_lane_veh_safety
        ):
            raise NotImplementedError(
                "The rectangle_sat lane barrier does not support aggregated lane "
                "safety."
            )
        # Rectangle segment covered by each circle: half-width and half-length.
        self.lane_rect_half_width = 0.5 * float(self.width)
        self.lane_rect_half_length = 0.5 * float(self.rec_cir_approx.segment)
        # Smoothing of |x| in the rectangle support (adds about b * eps at 0 deg).
        self.lane_rect_abs_eps = 0.05
        # Exact rectangle-polyline barrier (rectangle_sat), shared with the scenario
        self.lane_sat: LaneSatBarrier | None = self.scenario.lane_sat_barrier
        if (self.lane_cbf_footprint == "rectangle_sat") != (self.lane_sat is not None):
            raise ValueError(
                "The scenario must build a rectangle_sat barrier exactly when "
                "lane_cbf_footprint is rectangle_sat."
            )
        if self.lane_sat is not None and not math.isclose(
            self.lane_sat.safety_buffer, self.safety_buffer
        ):
            raise ValueError(
                "The scenario's rectangle_sat barrier uses a different safety "
                "buffer than the CBF-QP."
            )
        # Vehicle-vehicle CBF footprint; the exact rectangle pair barrier
        # (rectangle_sat) shares the lane barrier's prediction and repairs
        self.veh_veh_cbf_footprint = self.parameters.veh_veh_cbf_footprint
        if self.veh_veh_cbf_footprint not in ("circle", "mtv", "rectangle_sat"):
            raise ValueError(
                f"Unsupported veh_veh_cbf_footprint: {self.veh_veh_cbf_footprint}"
            )
        if self.veh_veh_cbf_footprint == "rectangle_sat" and self.lane_sat is None:
            raise ValueError(
                "veh_veh_cbf_footprint='rectangle_sat' needs "
                "lane_cbf_footprint='rectangle_sat'."
            )
        self.pair_sat: PairSatBarrier | None = (
            PairSatBarrier(self.length, self.width, self.pair_safety_buffer)
            if self.veh_veh_cbf_footprint == "rectangle_sat"
            else None
        )
        # Atoms whose nominal margin is below this band get rows before the solve
        self.lane_sat_initial_band = 0.01 * LENGTH_SCALE
        # Tolerance of the exact check; well below the safety buffer
        self.lane_sat_tol = 2e-5 * LENGTH_SCALE
        # Row tightening against linearization errors (the exact check decides)
        self.lane_sat_linearization_margin = 2e-4 * LENGTH_SCALE
        self.lane_sat_max_repairs = 3
        self.lane_sat_stats = defaultdict(int)
        # Slack activity: QPs whose CBF condition was infeasible (solver failure
        # or a lane/pair slack above slack_active_tol, in the barrier's unit)
        self.slack_active_tol = 5e-5 * LENGTH_SCALE
        self.slack_stats = defaultdict(int)
        self.last_slack_active = False
        # Wall-clock time [s] of the last update_qp per part (TIME_PARTS)
        self.time_parts = dict.fromkeys(TIME_PARTS, 0.0)
        # SOTA CBF baselines (comparison only, see sigmarl/cbf_baselines) that
        # replace the rectangle_sat lane or pair rows ("barrier swap")
        if (self.parameters.lane_cbf_baseline != "none" and self.lane_sat is None) or (
            self.parameters.veh_veh_cbf_baseline != "none" and self.pair_sat is None
        ):
            raise ValueError(
                "CBF baselines replace rectangle_sat rows and need the "
                "rectangle_sat lane and vehicle-vehicle footprints."
            )
        self.lane_baseline = make_lane_baseline(self.parameters.lane_cbf_baseline, self)
        self.pair_baseline = make_pair_baseline(
            self.parameters.veh_veh_cbf_baseline, self
        )
        # Backup-horizon TTCBF rows (sigmarl/backup_horizon_cbf.py)
        self.backup_horizon = None
        # (path ids, route bundle) of the backup-horizon margin reward
        self._bh_reward_bundle = None
        # Backup-horizon rows of the backup-value diagnostic (backup_values_all_envs)
        self._diag_backup_horizon = None
        if (
            self.parameters.condition_cbf_baseline != "none"
            and not self.parameters.is_backup_horizon_cbf
        ):
            raise ValueError("condition_cbf_baseline needs is_backup_horizon_cbf.")
        # Control of an agent whose QP is not solved; None resolves to the
        # backup control with the backup-horizon rows, else the nominal control
        self.failure_fallback = self.parameters.cbf_qp_failure_fallback
        if self.failure_fallback is None:
            self.failure_fallback = (
                "brake" if self.parameters.is_backup_horizon_cbf else "nominal"
            )
        if self.parameters.is_backup_horizon_cbf:
            if (
                self.pair_sat is None
                or self.lane_baseline is not None
                or self.pair_baseline is not None
                or self.parameters.is_learn_cbf_lambda
            ):
                raise ValueError(
                    "is_backup_horizon_cbf (the default) needs the rectangle_sat lane "
                    "and vehicle-vehicle footprints, no CBF baseline, and a fixed "
                    "lambda; set is_backup_horizon_cbf=False for the two-step "
                    "condition alone."
                )
            if self.failure_fallback != "brake":
                raise ValueError(
                    "is_backup_horizon_cbf needs cbf_qp_failure_fallback='brake' "
                    "(the backup control keeps the state recoverable)."
                )
            self.backup_horizon = make_backup_horizon(
                self.parameters.condition_cbf_baseline, self
            )
        # Whether this controller supports heterogeneous vehicles (see
        # _vehicle_params)
        self._is_heterogeneous_supported = (
            self.parameters.is_backup_horizon_cbf
            and self.parameters.condition_cbf_baseline == "none"
            and not self.parameters.is_two_step_rows
            and not self.parameters.is_using_centralized_cbf
            and not self.parameters.is_exact_backup_shield
            and not self.parameters.is_standstill_escape
        )
        self.shield_stats = defaultdict(int)
        # Route centerlines of the shield's backup steering, keyed by path id
        self._shield_route_cache = {}
        # Margin floors (<= 0) of the shield's lane [n] and pair [n, n]
        # constraints, and the predicted next states [n, 5] they belong to
        self._shield_lane_floor = None
        self._shield_pair_floor = None
        self._shield_prediction = None
        # Limits of (speed, steering) and of (acceleration, steering rate)
        self._shield_state_min = np.array([float(self.v_min), float(self.steering_min)])
        self._shield_state_max = np.array([float(self.v_max), float(self.steering_max)])
        self._shield_input_min = np.array(
            [float(self.a_min), float(self.steering_rate_min)]
        )
        self._shield_input_max = np.array(
            [float(self.a_max), float(self.steering_rate_max)]
        )
        # The same for one agent of the fully decentralized shield: its lane
        # floor, pair floors [n] and predicted next state [5]
        self._fd_shield_lane_floor = 0.0
        self._fd_shield_pair_floor = None
        self._fd_shield_prediction = None
        # Shield priority (see _shield_priority_key): simulation step at which
        # each agent became active, and its last checked position
        self._shield_active_since = {}
        self._shield_last_xy = {}

        # Standstill escape (see _standstill_escape_nominal): consecutive
        # stalled steps and the active escape plan of each agent
        if self.parameters.is_standstill_escape and self.lane_sat is None:
            raise ValueError(
                "The standstill escape needs lane_cbf_footprint='rectangle_sat'."
            )
        self._escape_stall_steps = defaultdict(int)
        self._escape_plans = {}
        # (pose, nominal angle) of each agent's last search without a plan;
        # it is not repeated while they change by less than escape_search_tol
        self._escape_failed_search = {}
        self.escape_stats = defaultdict(int)
        # [m/s] An agent slower than this is stopped
        self.escape_speed_tol = 0.02 * SPEED_SCALE
        # [m/s] Next speed below which the QP keeps an agent stopped
        self.escape_stop_tol = 1e-3 * SPEED_SCALE
        # [rad] Steering error below which an escape plan starts to move
        self.escape_angle_tol = 1e-3
        # Changes of (x, y, heading, nominal angle) that trigger a new search
        self.escape_search_tol = np.array(
            [1e-3 * float(self.length), 1e-3 * float(self.length), 2e-3, 2e-3]
        )
        # Stalled steps before an agent searches an escape plan
        self.escape_stall_steps = max(1, ceil(0.3 / self.dt))
        self.escape_n_angles = 25
        # Escape plans creep this distance [m] at this acceleration and speed
        self.escape_distance = 0.5 * float(self.length)
        self.escape_acc = min(0.5 * ACCELERATION_SCALE, self.a_max, -self.a_min)
        self.escape_speed = {1.0: min(0.1 * SPEED_SCALE, float(self.v_max))}
        # Reversing is an escape only if the agent profile allows it
        self.escape_reverse = float(self.v_min) < 0.0
        if self.escape_reverse:
            self.escape_speed[-1.0] = min(0.1 * SPEED_SCALE, -float(self.v_min))
        plan_time = (
            float(self.steering_max - self.steering_min) / self.steering_rate_max
            + self.escape_speed[1.0] / self.escape_acc
            + self.escape_distance / self.escape_speed[1.0]
        )
        # An escape plan that takes three times as long is given up
        self.escape_max_steps = ceil(3.0 * plan_time / self.dt)

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
        self.dx = 0.02 * LENGTH_SCALE
        self.dy = 0.02 * LENGTH_SCALE

        self.parameters.adaptive_lambda = False

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

        self.safety_buffer = 0.01 * self.width
        # Pair rows of the rectangle_sat vehicle-vehicle CBF; lane rows and the
        # backup shield keep safety_buffer (0.01 W interlocks vehicles in traffic)
        self.pair_safety_buffer = 0.05 * self.width
        # Fixed CBF decay rate in (0, 1]; higher is more aggressive. Unset: the
        # tuned value of the method (BH-TTCBF or a baseline), else 0.5
        cbf_lambda = self.parameters.cbf_lambda
        if cbf_lambda is None:
            cbf_lambda = tuned_cbf_lambda(self.parameters) or 0.5
        self.lambda_ttcbf_predefined = float(cbf_lambda)
        self._warned_invalid_cbf_lambda = False

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

        self.clf_nominal_mode = self.parameters.clf_nominal_mode
        # Tuning of the "pure_pursuit" nominal controller
        self.clf_steering_gain = 5.0  # [1/s] steering-angle error -> steering rate
        # Lookahead as a (fractional) index into the short-term reference points
        self.clf_lookahead_index = 1.0
        # [m/s^2] Lateral acceleration used to slow down for curves
        self.clf_max_lat_acc = 0.7 * ACCELERATION_SCALE
        self.is_cbf_limit_acc_to_nominal = bool(
            self.parameters.is_cbf_limit_acc_to_nominal
        )
        self.cross_cbf_responsibility = self.parameters.cross_cbf_responsibility

        self.u_placeholder = torch.tensor([0.0, 0.0], dtype=torch.float32)
        # QP statistics: number of solves and of failures (by status); failed QPs
        # use `_failure_fallback_control`
        self.n_qp_solves = 0
        self.qp_failure_counts = Counter()

        # Weight for nominal control tracking (acceleration, steering rate) in the
        # cost function, which is ||nom_weight (u - u_nom)||^2. The acceleration
        # weight was 10 before; that made braking 100 times more expensive than
        # steering, so the QP steered even where steering barely helps
        self.nom_weight = np.diag([1.0, 1.0])
        # Express the acceleration in the tracking cost and the speed CLF row in the
        # units of the lab vehicles, so that the weights above keep their meaning
        # for other agent profiles (identity for the lab profiles)
        self.nom_unit_scale = np.diag([1.0 / ACCELERATION_SCALE, 1.0])
        self.clf_speed_row_scale = 1.0 / SPEED_SCALE**2
        # Weights of the slack variables of the lane, inter-group pair, and
        # cross-group pair CBF rows. 1e10 is tuned for the lab vehicles; the slacks
        # have the units of the barriers ([m^2]), so dividing by LENGTH_SCALE**4 keeps
        # the trade-off between slack and control for other agent profiles (and
        # avoids an ill-conditioned QP at full scale)
        slack_weight = 1e10 / LENGTH_SCALE**4
        # The rectangle_sat lane barrier is in [m], so its slack scales with
        # LENGTH_SCALE**-2 instead; otherwise, at full scale, violating a lane by
        # 1 m is far cheaper than violating a pair barrier by the same distance
        if self.parameters.lane_cbf_footprint == "rectangle_sat":
            self.lane_slack_weight = 1e10 / LENGTH_SCALE**2
        else:
            self.lane_slack_weight = slack_weight
        # The same holds for the rectangle_sat pair barrier (inter- and
        # cross-group rows)
        if self.parameters.veh_veh_cbf_footprint == "rectangle_sat":
            self.pair_slack_weight = 1e10 / LENGTH_SCALE**2
        else:
            self.pair_slack_weight = slack_weight
        self.cross_slack_weight = self.pair_slack_weight
        self.lambda_weight = 1e3  # Weight for lambda penalty in the cost function
        self.is_debug_slack_var = bool(
            getattr(self.parameters, "is_debug_slack_var", False)
        )
        self.debug_slack_threshold = float(
            getattr(self.parameters, "debug_slack_threshold", 1e-3)
        )

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
            kappa = torch.tan(target_actions[:, 1]) / float(self.l_wb)
            ld = float(self.lookahead_dist)
            y_max = float(self.position_speed_lateral_max)
            max_kappa = 2.0 * y_max / (ld**2 + y_max**2) if y_max > 0 else 0.0
            kappa = torch.clamp(kappa, min=-max_kappa, max=max_kappa)
            discr = torch.clamp(1.0 - (kappa * ld) ** 2, min=0.0)
            lateral_offset = torch.where(
                kappa.abs() < 1e-8,
                torch.zeros_like(kappa),
                (kappa * (ld**2)) / (1.0 + torch.sqrt(discr)),
            )
            lateral_offset = torch.clamp(lateral_offset, min=-y_max, max=y_max)
            raw_actions = torch.stack([lateral_offset, target_actions[:, 0]], dim=1)
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
        ref_speed: float | None = None,
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

    def _clf_nominal_for_agent(
        self, tensordict, state_i: torch.Tensor, agent_idx: int
    ) -> Tuple[np.ndarray, float, float]:
        """Nominal CLF control [acc, steering_rate] and the CLF row errors (e_h, e_v).

        "heading_rate" (legacy): the steering rate is proportional to the heading
        error to the reference point, so the steering angle integrates the heading
        error without damping and overshoots in curves.
        "pure_pursuit": the steering angle tracks a pure-pursuit target angle with a
        first-order law, and the target speed is reduced in curves. e_h is then the
        steering-angle error, which keeps the CLF heading row consistent with the
        nominal control.
        """
        ref_xy, ref_speed = self._resolve_clf_reference(tensordict, agent_idx)
        if self.clf_nominal_mode == "heading_rate":
            e_h, e_v = self._clf_errors_for_agent(state_i, ref_xy, ref_speed)
            u_nom = np.array(
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
            return u_nom, e_h, e_v
        if self.clf_nominal_mode != "pure_pursuit":
            raise ValueError(f"Unknown clf_nominal_mode {self.clf_nominal_mode!r}.")

        x, y, psi, v, delta = (float(s) for s in state_i[0:5])
        ref = tensordict["agents", "info", "ref"][self.env_idx, agent_idx]
        pts = ref.reshape(-1, 2).detach().cpu().numpy().astype(np.float64)
        f = float(np.clip(self.clf_lookahead_index, 0.0, len(pts) - 1))
        k0 = int(np.floor(f))
        k1 = min(k0 + 1, len(pts) - 1)
        target = pts[k0] + (f - k0) * (pts[k1] - pts[k0])

        # Pure pursuit from the rear axle
        rear_x = x - self.l_r * math.cos(psi)
        rear_y = y - self.l_r * math.sin(psi)
        l_d = max(math.hypot(target[0] - rear_x, target[1] - rear_y), 1e-6)
        alpha = self._wrap_angle(atan2(target[1] - rear_y, target[0] - rear_x) - psi)
        delta_des = float(
            np.clip(
                math.atan(2.0 * self.l_wb * math.sin(alpha) / l_d),
                float(self.steering_min),
                float(self.steering_max),
            )
        )
        e_h = delta_des - delta

        # Slow down for the sharpest curvature ahead (pure pursuit or path points)
        kappa = abs(math.tan(delta_des)) / self.l_wb
        for k in range(len(pts) - 2):
            ab = pts[k + 1] - pts[k]
            bc = pts[k + 2] - pts[k + 1]
            ca = pts[k] - pts[k + 2]
            denom = np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ca)
            if denom > 1e-12:
                kappa = max(kappa, abs(2.0 * (ab[0] * bc[1] - ab[1] * bc[0])) / denom)
        v_des = float(ref_speed)
        if kappa > 1e-9:
            v_des = min(v_des, math.sqrt(self.clf_max_lat_acc / kappa))
        e_v = v_des - v

        u_nom = np.array(
            [
                np.clip(self.k_clf_speed * e_v, self.a_min, self.a_max),
                np.clip(
                    self.clf_steering_gain * e_h,
                    self.steering_rate_min,
                    self.steering_rate_max,
                ),
            ],
            dtype=np.float64,
        )
        return u_nom, e_h, e_v

    def _resolve_clf_reference(self, tensordict, agent_idx: int):
        ref_xy = self._ref_xy_from_tensordict(tensordict, agent_idx)
        if self.parameters.uses_motion_policy and self.rl_action_type == "speed":
            raw_action = tensordict[("agents", "action")][
                self.env_idx, agent_idx
            ].clone()
            if self.is_obs_noise:
                raw_action = (
                    raw_action + torch.rand_like(raw_action) * self.obs_noise_level
                )
            raw_action = self._sanitize_raw_action(raw_action)
            return ref_xy, float(raw_action[0].item())

        return ref_xy, self.ref_speed

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
        acc_limits=None,
    ):
        """
        Convert the environment action representation into a target speed/steering
        pair and then into low-level acceleration / steering-rate inputs.

        Args:
            rl_actions: [target_speed, target_steering_angle] for "speed_steer",
                [lateral_offset, target_speed] for "position_speed", or
                [target_speed] for "speed"
            acc_limits: the agent's (min, max) acceleration; None: the nominal
                ones.

        Returns:
            target_action: [target_speed, target_steering_angle]
            u: [acceleration, steering_rate]
        """
        min_acc, max_acc = (
            (self.a_min, self.a_max) if acc_limits is None else acc_limits
        )
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
            min_acc=min_acc,
            max_acc=max_acc,
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

    @staticmethod
    def _remove_overlapping_local_boundary_points(
        boundary: torch.Tensor, threshold: float = 1e-6
    ) -> torch.Tensor:
        """Remove consecutive duplicate points from a local boundary polyline."""
        if boundary.shape[0] < 2:
            return boundary
        keep = torch.ones(boundary.shape[0], device=boundary.device, dtype=torch.bool)
        keep[1:] = torch.diff(boundary, dim=0).norm(dim=-1) > threshold
        cleaned = boundary[keep]
        if cleaned.shape[0] >= 2:
            return cleaned
        return boundary[:2]

    def _get_local_lane_boundaries_for_agent(
        self, agent_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cleaned local left and right boundary polylines for one agent."""
        return self._get_local_lane_boundaries_for_env_agent(self.env_idx, agent_idx)

    def _get_local_lane_boundaries_for_env_agent(
        self, env_idx: int, agent_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cleaned local left and right boundary polylines for one env-agent."""
        ref_paths = self.scenario.world_state.ref_paths_agent_related
        left_boundary = ref_paths.nearing_points_left_boundary[env_idx, agent_idx]
        right_boundary = ref_paths.nearing_points_right_boundary[env_idx, agent_idx]
        return (
            self._remove_overlapping_local_boundary_points(left_boundary),
            self._remove_overlapping_local_boundary_points(right_boundary),
        )

    def _get_local_pseudo_distance(
        self,
        agent_idx: int,
        query_points: torch.Tensor,
        local_boundary_cache: dict | None = None,
    ):
        """Compute pseudo distances to the local left and right boundary polylines."""
        if local_boundary_cache is None:
            left_boundary, right_boundary = self._get_local_lane_boundaries_for_agent(
                agent_idx
            )
            left_tangent = compute_pseudo_tangent_vector(left_boundary)
            right_tangent = compute_pseudo_tangent_vector(right_boundary)
        else:
            cache_key = ("pseudo", agent_idx)
            cached = local_boundary_cache.get(cache_key)
            if cached is None:
                (
                    left_boundary,
                    right_boundary,
                ) = self._get_local_lane_boundaries_for_agent(agent_idx)
                cached = (
                    left_boundary,
                    right_boundary,
                    compute_pseudo_tangent_vector(left_boundary),
                    compute_pseudo_tangent_vector(right_boundary),
                )
                local_boundary_cache[cache_key] = cached
            left_boundary, right_boundary, left_tangent, right_tangent = cached
        pos = query_points.unsqueeze(1)

        left_distance, _ = self.map_pseudo_distance.get_pseudo_distance(
            left_tangent, left_boundary, pos
        )
        right_distance, _ = self.map_pseudo_distance.get_pseudo_distance(
            right_tangent, right_boundary, pos
        )
        return (
            left_distance.detach().cpu().numpy(),
            right_distance.detach().cpu().numpy(),
        )

    def _get_local_softmin_distance_with_derivatives(
        self,
        agent_idx: int,
        query_points: torch.Tensor,
        dx: float,
        dy: float,
        local_boundary_cache: dict | None = None,
    ):
        """Compute softmin boundary distances and derivatives from local polylines."""
        points_np = np.asarray(
            self.map_pseudo_distance._to_numpy(query_points), dtype=np.float64
        )
        if points_np.shape != (9, 2):
            raise ValueError(
                "query_points must have shape [9, 2] ordered as "
                "[center, +x, +y, -x, -y, +x+y, +x-y, -x+y, -x-y]."
            )

        left_cache, right_cache = self._get_local_softmin_boundary_cache(
            agent_idx, local_boundary_cache
        )

        left = self.map_pseudo_distance.derivatives_at_points(
            left_cache, points_np[None], dx, dy
        )
        right = self.map_pseudo_distance.derivatives_at_points(
            right_cache, points_np[None], dx, dy
        )
        return tuple(x[0] for x in left), tuple(x[0] for x in right)

    def _get_local_softmin_boundary_cache(
        self, agent_idx: int, local_boundary_cache: dict | None = None
    ):
        """Return prepared local softmin boundary caches for one agent."""
        return self._get_local_softmin_boundary_cache_for_env_agent(
            self.env_idx, agent_idx, local_boundary_cache
        )

    def _get_local_softmin_boundary_cache_for_env_agent(
        self,
        env_idx: int,
        agent_idx: int,
        local_boundary_cache: dict | None = None,
    ):
        """Return prepared local softmin boundary caches for one env-agent."""
        if local_boundary_cache is None:
            (
                left_boundary,
                right_boundary,
            ) = self._get_local_lane_boundaries_for_env_agent(env_idx, agent_idx)
            return (
                self.map_pseudo_distance._prepare_boundary(left_boundary),
                self.map_pseudo_distance._prepare_boundary(right_boundary),
            )

        cache_key = ("softmin", env_idx, agent_idx)
        cached = local_boundary_cache.get(cache_key)
        if cached is None:
            (
                left_boundary,
                right_boundary,
            ) = self._get_local_lane_boundaries_for_env_agent(env_idx, agent_idx)
            cached = (
                self.map_pseudo_distance._prepare_boundary(left_boundary),
                self.map_pseudo_distance._prepare_boundary(right_boundary),
            )
            local_boundary_cache[cache_key] = cached
        return cached

    def _get_local_softmin_distances_with_derivatives_batch(
        self,
        agent_idx: int,
        query_points: torch.Tensor,
        dx: float,
        dy: float,
        local_boundary_cache: dict | None = None,
    ):
        """Compute local softmin distances for batched circle stencils."""
        points_np = np.asarray(
            self.map_pseudo_distance._to_numpy(query_points), dtype=np.float64
        )
        if points_np.ndim != 3 or points_np.shape[1:] != (9, 2):
            raise ValueError(
                "query_points must have shape [N, 9, 2] ordered by stencil as "
                "[center, +x, +y, -x, -y, +x+y, +x-y, -x+y, -x-y]."
            )

        left_cache, right_cache = self._get_local_softmin_boundary_cache(
            agent_idx, local_boundary_cache
        )
        left = self.map_pseudo_distance.derivatives_at_points(
            left_cache, points_np, dx, dy
        )
        right = self.map_pseudo_distance.derivatives_at_points(
            right_cache, points_np, dx, dy
        )
        return left, right

    def _get_local_softmin_distances_with_derivatives_batch_for_env_agent(
        self,
        env_idx: int,
        agent_idx: int,
        query_points,
        dx: float,
        dy: float,
        local_boundary_cache: dict | None = None,
    ):
        """Compute local softmin distances for batched stencils of one env-agent."""
        points_np = np.asarray(
            self.map_pseudo_distance._to_numpy(query_points), dtype=np.float64
        )
        if points_np.ndim != 3 or points_np.shape[1:] != (9, 2):
            raise ValueError(
                "query_points must have shape [N, 9, 2] ordered by stencil as "
                "[center, +x, +y, -x, -y, +x+y, +x-y, -x+y, -x-y]."
            )

        left_cache, right_cache = self._get_local_softmin_boundary_cache_for_env_agent(
            env_idx, agent_idx, local_boundary_cache
        )
        left = self.map_pseudo_distance.derivatives_at_points(
            left_cache, points_np, dx, dy
        )
        right = self.map_pseudo_distance.derivatives_at_points(
            right_cache, points_np, dx, dy
        )
        return left, right

    def estimate_agent_2_lane_safety_margin(
        self,
        agent_pos: torch.tensor,
        path_id,
        agent_idx: int = None,
        local_boundary_cache: dict | None = None,
    ):
        """Estimate lane-boundary safety margins and derivatives for one circle."""
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
            if self.parameters.is_use_sampled_lane_boundary_for_cbf:
                if agent_idx is None:
                    raise ValueError(
                        "agent_idx is required when using local lane boundaries for CBF."
                    )
                (
                    dleft_results,
                    dright_results,
                ) = self._get_local_softmin_distance_with_derivatives(
                    agent_idx,
                    query_points,
                    dx=self.dx,
                    dy=self.dy,
                    local_boundary_cache=local_boundary_cache,
                )
            else:
                (
                    dleft_results,
                    dright_results,
                ) = self.map_pseudo_distance.get_distance_with_derivatives(
                    path_id, query_points, dx=self.dx, dy=self.dy
                )
        else:
            if self.parameters.is_use_sampled_lane_boundary_for_cbf:
                if agent_idx is None:
                    raise ValueError(
                        "agent_idx is required when using local lane boundaries for CBF."
                    )
                dleft_results, dright_results = self._get_local_pseudo_distance(
                    agent_idx,
                    query_points,
                    local_boundary_cache=local_boundary_cache,
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

    @staticmethod
    def _polyline_station_data(points: np.ndarray) -> tuple[np.ndarray, float]:
        """Return cumulative arc length data for a route polyline."""
        diffs = np.diff(points, axis=0)
        ds = np.linalg.norm(diffs, axis=1)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        return s, float(s[-1])

    def _write_low_level_controls_to_tensordict(
        self,
        tensordict,
        states: list[torch.Tensor],
        controls: np.ndarray,
        agent_ids=None,
    ) -> None:
        """Convert low-level controls (acceleration, steering rate) back to env actions."""
        if agent_ids is None:
            agent_ids = range(len(states))
        for local_idx, agent_id in enumerate(agent_ids):
            action_safe = self.u_to_rl_action(
                controls[local_idx],
                states[agent_id][3],
                states[agent_id][4],
                agent_idx=agent_id,
            ).detach()
            tensordict[("agents", "action")][self.env_idx, agent_id] = action_safe

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

    def _cbf_acc_upper_bound(self, u_nom_acc: float, acc_limits=None) -> float:
        """Upper acceleration bound of the CBF-QP.

        With is_cbf_limit_acc_to_nominal, the safety filter may brake but never
        accelerate beyond the nominal control (or beyond zero when the nominal
        brakes). Otherwise a vehicle-vehicle CBF can be satisfied by speeding away
        from a follower, which drives the vehicle too fast into curves.
        acc_limits: the agent's (min, max) acceleration; None: the nominal ones.
        """
        a_min, a_max = (self.a_min, self.a_max) if acc_limits is None else acc_limits
        if not self.is_cbf_limit_acc_to_nominal:
            return a_max
        return float(np.clip(max(float(u_nom_acc), 0.0), a_min, a_max))

    @staticmethod
    def _apply_box_bounds(lb: np.ndarray, ub: np.ndarray, slc: slice, lower, upper):
        if lower is not None:
            lb[slc] = np.asarray(lower, dtype=np.float64)
        if upper is not None:
            ub[slc] = np.asarray(upper, dtype=np.float64)

    @staticmethod
    def _append_qp_row(rows, lowers, uppers, n_vars: int, terms, lower, upper):
        row = np.zeros((n_vars,), dtype=np.float64)
        for idx, coeff in terms:
            coeff_arr = np.asarray(coeff, dtype=np.float64).reshape(-1)
            if isinstance(idx, slice):
                if idx.stop - idx.start != coeff_arr.size:
                    raise ValueError("QP row coefficient width mismatch.")
                row[idx] += coeff_arr
            else:
                row[int(idx)] += float(coeff_arr.squeeze())
        rows.append(row)
        lowers.append(float(lower))
        uppers.append(float(upper))

    @staticmethod
    def _add_quadratic_tracking_cost(
        H: np.ndarray, g: np.ndarray, slc: slice, weight: np.ndarray, ref: np.ndarray
    ):
        Q = np.asarray(weight, dtype=np.float64)
        if Q.ndim == 1:
            Q = np.diag(Q)
        ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        H[slc, slc] += 2.0 * Q
        g[slc] += -2.0 * (Q @ ref)

    @staticmethod
    def _add_quadratic_penalty(
        H: np.ndarray, g: np.ndarray, slc: slice, weight: float, ref=None
    ):
        size = slc.stop - slc.start
        H[slc, slc] += 2.0 * float(weight) * np.eye(size, dtype=np.float64)
        if ref is not None:
            ref_arr = np.asarray(ref, dtype=np.float64).reshape(-1)
            if ref_arr.size != size:
                raise ValueError("QP penalty reference width mismatch.")
            g[slc] += -2.0 * float(weight) * ref_arr

    @staticmethod
    def _is_within_box(x: np.ndarray, l_box: np.ndarray, u_box: np.ndarray) -> bool:
        """Whether a QP solution satisfies its box constraints up to a small tolerance."""
        tol_l = 1e-3 * (1.0 + np.abs(l_box))
        tol_u = 1e-3 * (1.0 + np.abs(u_box))
        return bool(np.all(x >= l_box - tol_l) and np.all(x <= u_box + tol_u))

    def _failure_fallback_control(self, state, agent_idx: int, u_nom) -> np.ndarray:
        """Control of an agent whose QP was not solved (`Parameters.cbf_qp_failure_fallback`):
        its nominal control, or braking with route-tracking steering (the shield's backup policy)."""
        if self.failure_fallback == "brake":
            t0 = time.perf_counter()
            X = np.array([[float(v) for v in state[:5]]], dtype=np.float64)
            bundle = self._shield_route_bundle([self._shield_route(agent_idx)])
            u = self._shield_backup_controls(
                X,
                bundle,
                np.ones(1, dtype=bool),
                params=self._vehicle_params([agent_idx]),
            )[0]
            self.time_parts["fallback"] += time.perf_counter() - t0
            return u
        return np.asarray(u_nom, dtype=np.float64).reshape(2)

    @staticmethod
    def _finalize_qp_rows(rows, lowers, uppers, n_vars: int):
        if len(rows) == 0:
            return (
                np.zeros((0, n_vars), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        return (
            np.vstack(rows).astype(np.float64, copy=False),
            np.asarray(lowers, dtype=np.float64),
            np.asarray(uppers, dtype=np.float64),
        )

    def _solve_proxqp_problem(
        self,
        H: np.ndarray,
        g: np.ndarray,
        C: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        l_box: np.ndarray,
        u_box: np.ndarray,
    ) -> dict:
        H = np.asarray(0.5 * (H + H.T), dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        l = np.asarray(l, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        l_box = np.asarray(l_box, dtype=np.float64)
        u_box = np.asarray(u_box, dtype=np.float64)

        qp = proxsuite.proxqp.dense.QP(H.shape[0], 0, C.shape[0], True)
        # Well-conditioned QPs converge in a few tens of iterations; the caps bound
        # the time spent on QPs that do not converge (they count as failures)
        qp.settings.max_iter = 1000
        qp.settings.max_iter_in = 100
        qp.settings.eps_abs = 1e-5
        qp.settings.eps_rel = 1e-5
        qp.settings.compute_timings = True
        qp.settings.verbose = False

        t0 = time.perf_counter()
        try:
            qp.init(H, g, None, None, C, l, u, l_box, u_box)
            qp.solve()
            x = np.asarray(qp.results.x, dtype=np.float64).copy()
            status = qp.results.info.status
            solved = status in PROXQP_SUCCESS_STATUSES and np.isfinite(x).all()
            if solved and not self._is_within_box(x, l_box, u_box):
                # Ill-conditioned QPs can be reported as solved with controls far
                # outside the actuator limits
                solved = False
                status = "solution outside the box constraints"
            iter_count = int(qp.results.info.iter)
            error = None
        except Exception as exc:
            x = None
            status = repr(exc)
            solved = False
            iter_count = 0
            error = exc
        solve_time = time.perf_counter() - t0
        self.time_parts["qp"] += solve_time
        self.n_qp_solves += 1
        if not solved:
            self.qp_failure_counts[str(status)] += 1
        return dict(
            solved=solved,
            x=x,
            status=status,
            iter=iter_count,
            solve_time=solve_time,
            error=error,
        )

    def _debug_sim_step(self):
        """Return the current simulator step for debug messages when available."""
        timer = getattr(self.scenario, "timer", None)
        step = getattr(timer, "step", None)
        if step is None:
            return "unknown"
        try:
            if isinstance(step, torch.Tensor):
                if step.ndim == 0:
                    return int(step.item())
                return int(step[self.env_idx].item())
            return int(step)
        except Exception:
            return "unknown"

    def _debug_path_id(self, path_ids, agent_idx: int):
        """Return a path id for an agent while tolerating missing debug context."""
        if path_ids is None:
            return "unknown"
        try:
            return int(path_ids[int(agent_idx)])
        except Exception:
            return "unknown"

    def _print_debug_slack_warning(self, **kwargs):
        """Print one detailed CBF slack warning when slack debugging is enabled."""
        if not self.is_debug_slack_var:
            return
        threshold = float(self.debug_slack_threshold)
        slack_value = float(kwargs.pop("slack_value"))
        if slack_value <= threshold:
            return

        details = [
            "[WARN][CBF-QP slack]",
            f"env={self.env_idx}",
            f"step={self._debug_sim_step()}",
            f"threshold={threshold:.6g}",
            f"slack={slack_value:.9g}",
        ]
        for key, value in kwargs.items():
            details.append(f"{key}={value}")
        print(", ".join(details), flush=True)

    def _debug_print_lane_slacks(
        self,
        x_opt: np.ndarray,
        s_lane_slice: slice,
        agent_ids,
        n_circles: int,
        path_ids,
        mode: str,
        group_idx=None,
    ):
        """Print lane-boundary CBF slack warnings for all agents in a QP solution.

        n_circles is the number of lane slacks per side and agent (1 for
        rectangle_sat, which has one barrier per side).
        """
        side_names = ("left_boundary", "right_boundary")
        for local_idx, agent_id in enumerate(agent_ids):
            for circle_idx in range(n_circles):
                for side_idx, side_name in enumerate(side_names):
                    var_idx = (
                        s_lane_slice.start
                        + 2 * (local_idx * n_circles + circle_idx)
                        + side_idx
                    )
                    self._print_debug_slack_warning(
                        mode=mode,
                        group=group_idx if group_idx is not None else "none",
                        collision_type="agent_lane",
                        vehicle=int(agent_id),
                        local_vehicle=int(local_idx),
                        path_id=self._debug_path_id(path_ids, agent_id),
                        circle=int(circle_idx),
                        boundary=side_name,
                        slack_var_idx=int(var_idx),
                        slack_value=float(x_opt[var_idx]),
                    )

    def _debug_print_pair_slacks(
        self,
        x_opt: np.ndarray,
        s_pair_slice: slice | None,
        agent_ids,
        pair_base: dict,
        n_circles: int,
        mode: str,
        group_idx=None,
    ):
        """Print in-QP agent-to-agent CBF slack warnings for a QP solution."""
        if s_pair_slice is None:
            return
        for i_loc in range(len(agent_ids) - 1):
            for j_loc in range(i_loc + 1, len(agent_ids)):
                base = pair_base[(i_loc, j_loc)]
                for ci in range(n_circles):
                    for cj in range(n_circles):
                        var_idx = s_pair_slice.start + base + ci * n_circles + cj
                        self._print_debug_slack_warning(
                            mode=mode,
                            group=group_idx if group_idx is not None else "none",
                            collision_type="agent_agent_pair",
                            vehicle_i=int(agent_ids[i_loc]),
                            vehicle_j=int(agent_ids[j_loc]),
                            local_vehicle_i=int(i_loc),
                            local_vehicle_j=int(j_loc),
                            circle_i=int(ci),
                            circle_j=int(cj),
                            slack_var_idx=int(var_idx),
                            slack_value=float(x_opt[var_idx]),
                        )

    def _debug_print_cross_slacks(
        self,
        x_opt: np.ndarray,
        s_cross_slice: slice | None,
        ego_agent_ids,
        neighbor_lists,
        cross_base: dict,
        n_circles: int,
        mode: str,
        group_idx=None,
    ):
        """Print cross-neighbor agent-to-agent CBF slack warnings for a QP solution."""
        if s_cross_slice is None:
            return
        for i_loc, ego_agent_id in enumerate(ego_agent_ids):
            for neighbor_rank, neighbor_id in enumerate(neighbor_lists[i_loc]):
                base = cross_base[(i_loc, neighbor_rank)]
                for ci in range(n_circles):
                    for cj in range(n_circles):
                        var_idx = s_cross_slice.start + base + ci * n_circles + cj
                        self._print_debug_slack_warning(
                            mode=mode,
                            group=group_idx if group_idx is not None else "none",
                            collision_type="agent_agent_cross",
                            vehicle_i=int(ego_agent_id),
                            vehicle_j=int(neighbor_id),
                            local_vehicle_i=int(i_loc),
                            neighbor_rank=int(neighbor_rank),
                            circle_i=int(ci),
                            circle_j=int(cj),
                            slack_var_idx=int(var_idx),
                            slack_value=float(x_opt[var_idx]),
                        )

    def _collect_agent_states(self, tensordict, n_agents: int):
        """Route ids, states [x, y, psi, v, delta] (tensors), lazily computed
        circle centers, positions [n, 2], and speeds [n] (float64) of the
        agents of this env."""
        agents: List[Vehicle] = self.env.base_env.scenario_name.world.agents
        path_ids_t = tensordict["agents", "info", "path_id"][self.env_idx]
        path_ids = [int(p) for p in path_ids_t[:n_agents].reshape(n_agents).tolist()]

        joint = torch.cat(
            [
                torch.stack(
                    [
                        getattr(agents[i].state, key)[self.env_idx]
                        for i in range(n_agents)
                    ]
                )
                for key in ("pos", "rot", "speed", "steering")
            ],
            dim=-1,
        )
        states = list(joint.unbind(0))
        # float32 to float64 is exact (as float(state_i[k].item()))
        joint_np = joint.detach().cpu().numpy().astype(np.float64)
        centers = _LazyAgentValues(self.get_circle_centers, states)
        return path_ids, states, centers, joint_np[:, 0:2], joint_np[:, 3]

    def _get_on_hold_agent_set(self) -> set[int]:
        """Return the set of agent indices that are on hold (finished in the joint exit task, or parked by a
        log replay). They get zero control and are ignored by the other agents' constraints."""
        on_hold = self.scenario.get_on_hold_agents_mask()
        if on_hold is None or not on_hold[self.env_idx].any():
            return set()
        return set(torch.nonzero(on_hold[self.env_idx], as_tuple=True)[0].tolist())

    def _get_uncontrolled_agent_set(self) -> set[int]:
        """Return the set of agent indices that follow recorded tracks (log replay).

        They do not react to the controlled agents: their own QP is skipped, and controlled agents take full
        responsibility for pair constraints with them, predicting them with zero control (constant speed and
        steering).
        """
        uncontrolled = self.scenario.get_uncontrolled_agents_mask()
        if uncontrolled is None or not uncontrolled[self.env_idx].any():
            return set()
        return set(torch.nonzero(uncontrolled[self.env_idx], as_tuple=True)[0].tolist())

    def _get_log_replay_passive_agent_set(self) -> set[int]:
        """Agents that a log replay parks or replays (empty without a log replay).

        The centralized QP fixes their controls to zero and skips their own lane and CLF rows, which would
        only add (possibly ill-conditioned) rows that their controls cannot affect.
        """
        if self.scenario.log_replay is None:
            return set()
        return self._get_on_hold_agent_set() | self._get_uncontrolled_agent_set()

    def prepare_fully_decentralized_step_cache(self, tensordict, controllers):
        """Share per-step state and kinematics across decentralized agent QPs."""
        if not controllers:
            return
        n_agents = int(self.parameters.n_agents)
        path_ids, states, centers, centers_flat, speeds = self._collect_agent_states(
            tensordict, n_agents
        )
        on_hold_set = self._get_on_hold_agent_set()
        cache = {
            "env_idx": self.env_idx,
            "path_ids": path_ids,
            "states": states,
            "centers": centers,
            "centers_flat": centers_flat,
            "speeds": speeds,
            "kins_all": [_LazyKinematics(self, state_i) for state_i in states],
            "local_boundary_cache": {},
            "on_hold_set": on_hold_set,
            # Round-1 broadcasts of the decentralized exact backup shield
            "shield_proposals": {},
        }
        for controller in controllers:
            controller._fd_step_cache = cache

    @staticmethod
    def clear_fully_decentralized_step_cache(controllers):
        """Clear the shared per-step decentralized cache from agent controllers."""
        for controller in controllers:
            if hasattr(controller, "_fd_step_cache"):
                controller._fd_step_cache = None

    @staticmethod
    def _build_pair_base(n_agents: int, n_circles: int) -> Tuple[dict, int]:
        pair_base = {}
        base = 0
        for i in range(n_agents - 1):
            for j in range(i + 1, n_agents):
                pair_base[(i, j)] = base
                base += n_circles * n_circles
        return pair_base, base

    def build_centralized_cbf_qp(self):
        print(f"[INFO] Building centralized CBF-QP for env {self.env_idx}...")
        if hasattr(self, "_qp_cache"):
            return
        n = int(self.parameters.n_agents)
        n_circles = int(self.parameters.n_circles_approximate_vehicle)
        pair_base, pair_count = self._build_pair_base(n, n_circles)
        self.n_agents = n
        self._qp_cache = dict(
            n=n,
            n_circles=n_circles,
            pair_base=pair_base,
            pair_count=pair_count,
        )

    def update_centralized_cbf_qp(self, tensordict):
        cache = self._qp_cache
        n = int(cache["n"])
        n_circles = int(cache["n_circles"])
        is_aggregate_veh_veh_safety = bool(
            getattr(self.parameters, "is_aggregate_veh_veh_safety", True)
        )
        is_aggregate_lane_veh_safety = bool(
            getattr(self.parameters, "is_aggregate_lane_veh_safety", False)
        )
        aggregate_safety_softmin_beta = float(
            getattr(
                self.parameters,
                "aggregate_safety_softmin_beta",
                400.0 / LENGTH_SCALE**2,
            )
        )
        is_use_mtv_for_veh_veh = self.veh_veh_cbf_footprint == "mtv"
        is_pair_sat = self.veh_veh_cbf_footprint == "rectangle_sat"
        # rectangle_sat needs one lane slack per side, like the aggregated barrier
        is_lane_sat = self.lane_cbf_footprint == "rectangle_sat"
        is_two_lane_slacks = is_aggregate_lane_veh_safety or is_lane_sat
        lane_count_per_agent = 2 if is_two_lane_slacks else 2 * n_circles
        pair_base, pair_count = self._build_pair_base(
            n,
            1
            if is_use_mtv_for_veh_veh or is_pair_sat or is_aggregate_veh_veh_safety
            else n_circles,
        )
        self.time_pseudo_dis = 0

        path_ids, states, circles_all, _, _ = self._collect_agent_states(tensordict, n)
        on_hold_set = self._get_on_hold_agent_set()
        passive_set = self._get_log_replay_passive_agent_set()
        if self.parameters.cbf_lambda_head == "pair":
            learned_lambdas = self._get_env_pair_cbf_lambdas(tensordict)
        else:
            learned_lambdas = self._get_env_cbf_lambdas(tensordict)

        nom_acc_omega = np.zeros((n, 2), dtype=np.float64)
        nominal_target_actions = torch.zeros(
            (n, 2), device=self.device, dtype=torch.float32
        )
        e_h_np = np.zeros((n,), dtype=np.float64)
        e_v_np = np.zeros((n,), dtype=np.float64)
        v_head_const = np.zeros((n,), dtype=np.float64)
        v_speed_const = np.zeros((n,), dtype=np.float64)

        for i in range(n):
            if i in passive_set:
                continue  # Zero nominal control
            if self.nom_controller_type == "rl":
                rl_i = tensordict[("agents", "action")][self.env_idx, i].clone()
                if self.is_obs_noise:
                    rl_i = rl_i + torch.rand_like(rl_i) * self.obs_noise_level
                target_action_i, u_nom_i = self.rl_action_to_u(
                    rl_actions=rl_i,
                    v=states[i][3],
                    steering=states[i][4],
                    state=states[i],
                    ref_xy=(
                        self._ref_xy_from_tensordict(tensordict, i)
                        if self.rl_action_type == "speed"
                        else None
                    ),
                )
                nominal_target_actions[i, :] = target_action_i.detach()
                nom_acc_omega[i, :] = u_nom_i.detach().cpu().numpy()
            else:
                u_nom_i, e_h, e_v = self._clf_nominal_for_agent(
                    tensordict, states[i], i
                )
                e_h_np[i] = e_h
                e_v_np[i] = e_v
                nom_acc_omega[i, :] = u_nom_i
                nominal_target_actions[i, :] = self.u_to_target_action(
                    nom_acc_omega[i, :], states[i][3], states[i][4]
                ).detach()
            if self.parameters.is_standstill_escape and i not in on_hold_set:
                (
                    nom_acc_omega[i, :],
                    e_h_np[i],
                    e_v_np[i],
                    nominal_target_actions[i, :],
                ) = self._standstill_escape_nominal(
                    i,
                    states[i],
                    nom_acc_omega[i, :],
                    e_h_np[i],
                    e_v_np[i],
                    nominal_target_actions[i, :],
                )
            # Zero for an RL nominal controller (no CLF errors)
            v_head_const[i] = self.lam_clf * 0.5 * (e_h_np[i] ** 2)
            v_speed_const[i] = (
                self.lam_clf * 0.5 * (e_v_np[i] ** 2) * self.clf_speed_row_scale
            )

        offset = 0
        u_slice = slice(offset, offset + 2 * n)
        offset = u_slice.stop
        s_lane_slice = slice(offset, offset + n * lane_count_per_agent)
        offset = s_lane_slice.stop
        s_pair_slice = slice(offset, offset + pair_count) if pair_count > 0 else None
        if s_pair_slice is not None:
            offset = s_pair_slice.stop
        s_clf_head_slice = slice(offset, offset + n)
        offset = s_clf_head_slice.stop
        s_clf_speed_slice = slice(offset, offset + n)
        offset = s_clf_speed_slice.stop
        lambda_lane_slice = None
        lambda_pair_slice = None
        if self.parameters.adaptive_lambda:
            lambda_lane_slice = slice(offset, offset + n * lane_count_per_agent)
            offset = lambda_lane_slice.stop
            if pair_count > 0:
                lambda_pair_slice = slice(offset, offset + pair_count)
                offset = lambda_pair_slice.stop
        n_vars = offset

        H = np.zeros((n_vars, n_vars), dtype=np.float64)
        g = np.zeros((n_vars,), dtype=np.float64)
        l_box = np.full((n_vars,), -PROXQP_INF, dtype=np.float64)
        u_box = np.full((n_vars,), PROXQP_INF, dtype=np.float64)
        rows = []
        lowers = []
        uppers = []

        scaled_nom_weight = self.nom_weight @ self.nom_unit_scale
        track_weight = scaled_nom_weight.T @ scaled_nom_weight
        for i in range(n):
            ui_slice = slice(u_slice.start + 2 * i, u_slice.start + 2 * i + 2)
            self._add_quadratic_tracking_cost(
                H, g, ui_slice, track_weight, nom_acc_omega[i, :]
            )
            if i in passive_set:
                # Parked agents stand still; recorded vehicles are predicted with zero control
                self._apply_box_bounds(l_box, u_box, ui_slice, 0.0, 0.0)
            else:
                self._apply_box_bounds(
                    l_box,
                    u_box,
                    ui_slice,
                    [self.a_min, self.steering_rate_min],
                    [
                        self._cbf_acc_upper_bound(nom_acc_omega[i, 0]),
                        self.steering_rate_max,
                    ],
                )

        self._apply_box_bounds(l_box, u_box, s_lane_slice, 0.0, PROXQP_INF)
        self._apply_box_bounds(l_box, u_box, s_clf_head_slice, 0.0, PROXQP_INF)
        self._apply_box_bounds(l_box, u_box, s_clf_speed_slice, 0.0, PROXQP_INF)
        if s_pair_slice is not None:
            self._apply_box_bounds(l_box, u_box, s_pair_slice, 0.0, PROXQP_INF)

        self._add_quadratic_penalty(H, g, s_lane_slice, self.lane_slack_weight)
        self._add_quadratic_penalty(H, g, s_clf_head_slice, self.w_clf_relax)
        self._add_quadratic_penalty(H, g, s_clf_speed_slice, self.w_clf_relax)
        if s_pair_slice is not None:
            self._add_quadratic_penalty(H, g, s_pair_slice, self.pair_slack_weight)

        if lambda_lane_slice is not None:
            self._apply_box_bounds(l_box, u_box, lambda_lane_slice, 0.0, 1.0)
            self._add_quadratic_penalty(
                H,
                g,
                lambda_lane_slice,
                self.lambda_weight,
                ref=np.full(
                    lambda_lane_slice.stop - lambda_lane_slice.start,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                ),
            )
        if lambda_pair_slice is not None:
            self._apply_box_bounds(l_box, u_box, lambda_pair_slice, 0.0, 1.0)
            self._add_quadratic_penalty(H, g, lambda_pair_slice, self.lambda_weight)

        for i in range(n):
            if i in passive_set:
                continue
            u_acc_idx = u_slice.start + 2 * i
            u_steer_idx = u_acc_idx + 1
            self._append_qp_row(
                rows,
                lowers,
                uppers,
                n_vars,
                [
                    (u_steer_idx, -e_h_np[i]),
                    (s_clf_head_slice.start + i, -1.0),
                ],
                lower=-PROXQP_INF,
                upper=-v_head_const[i],
            )
            self._append_qp_row(
                rows,
                lowers,
                uppers,
                n_vars,
                [
                    (u_acc_idx, -e_v_np[i] * self.clf_speed_row_scale),
                    (s_clf_speed_slice.start + i, -1.0),
                ],
                lower=-PROXQP_INF,
                upper=-v_speed_const[i],
            )

        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe
        body_circle_radius = float(math.sqrt(self.length**2 + self.width**2) / 2.0)
        body_circle_d_safe = float(2.0 * body_circle_radius + self.safety_buffer)
        body_circle_d_safe_sq = body_circle_d_safe * body_circle_d_safe
        kins_all = [_LazyKinematics(self, state_i) for state_i in states]
        local_boundary_cache = {}
        lane_stencil_offsets = None
        if (
            not is_lane_sat
            and self.parameters.lane_distance_type == "softmin"
            and self.parameters.is_use_sampled_lane_boundary_for_cbf
        ):
            lane_stencil_offsets = torch.tensor(
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

        def lane_slack_idx(agent_idx, circle_idx, side):
            if is_two_lane_slacks:
                return s_lane_slice.start + agent_idx * lane_count_per_agent + side
            return s_lane_slice.start + 2 * (agent_idx * n_circles + circle_idx) + side

        def pair_idx(i_a, j_a, i_c, j_c):
            if is_use_mtv_for_veh_veh or is_pair_sat or is_aggregate_veh_veh_safety:
                return pair_base[(i_a, j_a)]
            return pair_base[(i_a, j_a)] + i_c * n_circles + j_c

        sat_blocks = []
        for i in range(n):
            if i in passive_set:
                continue
            ui_slice = slice(u_slice.start + 2 * i, u_slice.start + 2 * i + 2)
            kins_i = kins_all[i]
            if self.lane_cbf_footprint == "rectangle_sat":
                time_pseudo_dis_start = time.time()
                sat_blocks.append(
                    self._append_lane_sat_rows(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        l_box,
                        u_box,
                        ui_slice,
                        states[i],
                        self.env_idx,
                        i,
                        nom_acc_omega[i, :],
                        slack_idx=(lane_slack_idx(i, 0, 0), lane_slack_idx(i, 0, 1)),
                        lambda_idx=(
                            None
                            if lambda_lane_slice is None
                            else (
                                lambda_lane_slice.start + i * lane_count_per_agent,
                                lambda_lane_slice.start + i * lane_count_per_agent + 1,
                            )
                        ),
                    )
                )
                self.time_pseudo_dis += time.time() - time_pseudo_dis_start
                continue
            lane_distance_results_all = None
            if lane_stencil_offsets is not None:
                query_points = circles_all[i][:, 0:2].unsqueeze(
                    1
                ) + lane_stencil_offsets.unsqueeze(0)
                time_pseudo_dis_start = time.time()
                lane_distance_results_all = (
                    self._get_local_softmin_distances_with_derivatives_batch(
                        i,
                        query_points,
                        dx=self.dx,
                        dy=self.dy,
                        local_boundary_cache=local_boundary_cache,
                    )
                )
                self.time_pseudo_dis += time.time() - time_pseudo_dis_start

            def lane_results_for_circle(circle_idx: int):
                if lane_distance_results_all is not None:
                    dleft_results_all, dright_results_all = lane_distance_results_all
                    return (
                        dleft_results_all[0][circle_idx] - self.circle_radius**2,
                        dleft_results_all[1][circle_idx],
                        dleft_results_all[2][circle_idx],
                        dright_results_all[0][circle_idx] - self.circle_radius**2,
                        dright_results_all[1][circle_idx],
                        dright_results_all[2][circle_idx],
                    )
                circle_pos_i = circles_all[i][circle_idx][0:2]
                return self.estimate_agent_2_lane_safety_margin(
                    circle_pos_i,
                    path_ids[i],
                    agent_idx=i,
                    local_boundary_cache=local_boundary_cache,
                )

            if is_aggregate_lane_veh_safety:
                lane_terms = {0: [], 1: []}
                for ci in range(n_circles):
                    smL, gL, HL, smR, gR, HR = lane_results_for_circle(ci)
                    lane_terms[0].append(
                        self.ttcbf_lane_affine_components(kins_i, ci, smL, gL, HL)
                    )
                    lane_terms[1].append(
                        self.ttcbf_lane_affine_components(kins_i, ci, smR, gR, HR)
                    )
                for side in (0, 1):
                    A_lane, b0_lane, h_lane = self.aggregate_softmin_taylor_cbf_terms(
                        lane_terms[side],
                        self.dt_taylor,
                        aggregate_safety_softmin_beta,
                    )
                    terms = [
                        (ui_slice, A_lane.reshape(-1)),
                        (lane_slack_idx(i, 0, side), 1.0),
                    ]
                    if self.parameters.adaptive_lambda:
                        terms.append(
                            (
                                lambda_lane_slice.start
                                + i * lane_count_per_agent
                                + side,
                                float(np.asarray(h_lane).squeeze()),
                            )
                        )
                        lower = -float(np.asarray(b0_lane).squeeze())
                    else:
                        lower = -float(
                            np.asarray(
                                b0_lane + self.lambda_ttcbf_predefined * h_lane
                            ).squeeze()
                        )
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        terms,
                        lower=lower,
                        upper=PROXQP_INF,
                    )
                continue

            for ci in range(n_circles):
                smL, gL, HL, smR, gR, HR = lane_results_for_circle(ci)
                if self.parameters.adaptive_lambda:
                    A_L_ic, b0_L_ic, h_L_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smL, gL, HL, self.dt_taylor, None
                    )
                    A_R_ic, b0_R_ic, h_R_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, smR, gR, HR, self.dt_taylor, None
                    )
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (ui_slice, A_L_ic.reshape(-1)),
                            (lane_slack_idx(i, ci, 0), 1.0),
                            (
                                lambda_lane_slice.start + 2 * (i * n_circles + ci),
                                float(np.asarray(h_L_ic).squeeze()),
                            ),
                        ],
                        lower=-float(np.asarray(b0_L_ic).squeeze()),
                        upper=PROXQP_INF,
                    )
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (ui_slice, A_R_ic.reshape(-1)),
                            (lane_slack_idx(i, ci, 1), 1.0),
                            (
                                lambda_lane_slice.start + 2 * (i * n_circles + ci) + 1,
                                float(np.asarray(h_R_ic).squeeze()),
                            ),
                        ],
                        lower=-float(np.asarray(b0_R_ic).squeeze()),
                        upper=PROXQP_INF,
                    )
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
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (ui_slice, A_L_ic.reshape(-1)),
                            (lane_slack_idx(i, ci, 0), 1.0),
                        ],
                        lower=-float(np.asarray(b_L_ic).squeeze()),
                        upper=PROXQP_INF,
                    )
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (ui_slice, A_R_ic.reshape(-1)),
                            (lane_slack_idx(i, ci, 1), 1.0),
                        ],
                        lower=-float(np.asarray(b_R_ic).squeeze()),
                        upper=PROXQP_INF,
                    )

        pair_sat_pairs = []
        for i in range(n - 1):
            ui_slice = slice(u_slice.start + 2 * i, u_slice.start + 2 * i + 2)
            for j in range(i + 1, n):
                if on_hold_set and (i in on_hold_set or j in on_hold_set):
                    continue
                if i in passive_set and j in passive_set:
                    continue
                uj_slice = slice(u_slice.start + 2 * j, u_slice.start + 2 * j + 2)
                if self.parameters.cbf_lambda_head == "pair":
                    lambda_ttcbf = float(learned_lambdas[i, j])
                elif self.parameters.is_learn_cbf_lambda:
                    lam_i = self._get_agent_cbf_lambda(learned_lambdas, i)
                    lam_j = self._get_agent_cbf_lambda(learned_lambdas, j)
                    lambda_ttcbf = float((lam_i + lam_j) / 2.0)
                else:
                    lambda_ttcbf = float(self.lambda_ttcbf_predefined)
                kins_i = kins_all[i]
                kins_j = kins_all[j]
                pair_offset = pair_idx(i, j, 0, 0)
                if is_pair_sat:
                    pair_sat_pairs.append(
                        dict(
                            a=i,
                            b=j,
                            slices=(ui_slice, uj_slice),
                            share=None,
                            lam=lambda_ttcbf,
                            slack_idx=(
                                None
                                if s_pair_slice is None
                                else s_pair_slice.start + pair_offset
                            ),
                            lambda_idx=(
                                None
                                if lambda_pair_slice is None
                                else lambda_pair_slice.start + pair_offset
                            ),
                        )
                    )
                    continue
                if is_use_mtv_for_veh_veh:
                    relative_pose = self._relative_pose_in_ego_frame(
                        states[i], states[j]
                    )
                    if self._is_mtv_pose_in_domain(relative_pose):
                        A_i, A_j, b0, h = self.ttcbf_mtv_affine_coeffs_pair(
                            states[i], states[j], self.dt_taylor
                        )
                    else:
                        kins_i_center = (
                            self.linearized_vehicle_center_kinematics_coeffs(states[i])
                        )
                        kins_j_center = (
                            self.linearized_vehicle_center_kinematics_coeffs(states[j])
                        )
                        dx_center = float(states[i][0].item() - states[j][0].item())
                        dy_center = float(states[i][1].item() - states[j][1].item())
                        A_i, A_j, b0, h = self.ttcbf_pair_affine_coeffs(
                            kins_i_center,
                            kins_j_center,
                            0,
                            0,
                            dx_center,
                            dy_center,
                            body_circle_d_safe_sq,
                            self.dt_taylor,
                            None,
                        )
                    pair_slack_term = (
                        []
                        if s_pair_slice is None
                        else [(s_pair_slice.start + pair_offset, 1.0)]
                    )
                    terms = [
                        (ui_slice, A_i.reshape(-1)),
                        (uj_slice, A_j.reshape(-1)),
                        *pair_slack_term,
                    ]
                    if lambda_pair_slice is not None:
                        terms.append(
                            (
                                lambda_pair_slice.start + pair_offset,
                                float(np.asarray(h).squeeze()),
                            )
                        )
                        lower = -float(np.asarray(b0).squeeze())
                    else:
                        lower = -float(np.asarray(b0 + lambda_ttcbf * h).squeeze())
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        terms,
                        lower=lower,
                        upper=PROXQP_INF,
                    )
                    continue

                if is_aggregate_veh_veh_safety:
                    pair_terms = []
                    for ci in range(n_circles):
                        pi = circles_all[i][ci][0:2]
                        for cj in range(n_circles):
                            pj = circles_all[j][cj][0:2]
                            dx = float((pi - pj)[0].item())
                            dy = float((pi - pj)[1].item())
                            pair_terms.append(
                                self.ttcbf_pair_affine_components(
                                    kins_i,
                                    kins_j,
                                    ci,
                                    cj,
                                    dx,
                                    dy,
                                    d_safe_sq,
                                )
                            )
                    A_i, A_j, b0, h = self.aggregate_softmin_taylor_pair_cbf_terms(
                        pair_terms,
                        self.dt_taylor,
                        aggregate_safety_softmin_beta,
                    )
                    pair_slack_term = (
                        []
                        if s_pair_slice is None
                        else [(s_pair_slice.start + pair_offset, 1.0)]
                    )
                    terms = [
                        (ui_slice, A_i.reshape(-1)),
                        (uj_slice, A_j.reshape(-1)),
                        *pair_slack_term,
                    ]
                    if lambda_pair_slice is not None:
                        terms.append(
                            (
                                lambda_pair_slice.start + pair_offset,
                                float(np.asarray(h).squeeze()),
                            )
                        )
                        lower = -float(np.asarray(b0).squeeze())
                    else:
                        lower = -float(np.asarray(b0 + lambda_ttcbf * h).squeeze())
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        terms,
                        lower=lower,
                        upper=PROXQP_INF,
                    )
                    continue

                for ci in range(n_circles):
                    pi = circles_all[i][ci][0:2]
                    for cj in range(n_circles):
                        pj = circles_all[j][cj][0:2]
                        dx = float((pi - pj)[0].item())
                        dy = float((pi - pj)[1].item())
                        pair_offset = pair_idx(i, j, ci, cj)
                        pair_slack_term = (
                            []
                            if s_pair_slice is None
                            else [(s_pair_slice.start + pair_offset, 1.0)]
                        )
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
                            terms = [
                                (ui_slice, A_i.reshape(-1)),
                                (uj_slice, A_j.reshape(-1)),
                                *pair_slack_term,
                            ]
                            if lambda_pair_slice is not None:
                                terms.append(
                                    (
                                        lambda_pair_slice.start + pair_offset,
                                        float(np.asarray(h).squeeze()),
                                    )
                                )
                                lower = -float(np.asarray(b0).squeeze())
                            else:
                                lower = -float(
                                    np.asarray(b0).squeeze() + np.asarray(h).squeeze()
                                )
                            self._append_qp_row(
                                rows,
                                lowers,
                                uppers,
                                n_vars,
                                terms,
                                lower=lower,
                                upper=PROXQP_INF,
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
                            self._append_qp_row(
                                rows,
                                lowers,
                                uppers,
                                n_vars,
                                [
                                    (ui_slice, A_i.reshape(-1)),
                                    (uj_slice, A_j.reshape(-1)),
                                    *pair_slack_term,
                                ],
                                lower=-float(np.asarray(b).squeeze()),
                                upper=PROXQP_INF,
                            )

        if pair_sat_pairs:
            states_np = {
                i: np.asarray([float(value) for value in states[i][:5]])
                for i in range(n)
            }
            u_lin = {
                i: np.clip(
                    nom_acc_omega[i, :],
                    l_box[u_slice.start + 2 * i : u_slice.start + 2 * i + 2],
                    u_box[u_slice.start + 2 * i : u_slice.start + 2 * i + 2],
                )
                for i in range(n)
            }
            sat_blocks.append(
                self._append_pair_sat_rows(
                    pair_sat_pairs,
                    states_np,
                    u_lin,
                    set(),
                    rows,
                    lowers,
                    uppers,
                    n_vars,
                )
            )

        if self.lane_cbf_footprint == "rectangle_sat":
            result = self._solve_qp_with_sat_repairs(
                H, g, rows, lowers, uppers, n_vars, l_box, u_box, sat_blocks
            )
        else:
            C_mat, l_vec, u_vec = self._finalize_qp_rows(rows, lowers, uppers, n_vars)
            result = self._solve_proxqp_problem(H, g, C_mat, l_vec, u_vec, l_box, u_box)
        self._qp_cache["last_result"] = result
        self._record_slack_activity(result, [s_lane_slice, s_pair_slice])

        solved = bool(result["solved"])
        if not solved and result["error"] is not None:
            print(
                "[WARN] Centralized proxsuite solve failed. Using the fallback controls "
                f"({self.failure_fallback}). status={result['status']}"
            )

        if solved and self.is_debug_slack_var:
            x_opt = result["x"]
            agent_ids = list(range(n))
            if not is_aggregate_lane_veh_safety:
                self._debug_print_lane_slacks(
                    x_opt,
                    s_lane_slice,
                    agent_ids,
                    1 if is_lane_sat else n_circles,
                    path_ids,
                    mode="centralized",
                )
            if not (
                is_use_mtv_for_veh_veh or is_pair_sat or is_aggregate_veh_veh_safety
            ):
                self._debug_print_pair_slacks(
                    x_opt,
                    s_pair_slice,
                    agent_ids,
                    pair_base,
                    n_circles,
                    mode="centralized",
                )

        if self.env.scenario_name.parameters.is_apply_cbf_action:
            u_apply = nom_acc_omega.copy()
            if solved:
                x_opt = result["x"]
                keep_if_safe = getattr(
                    self.parameters, "is_keep_nominal_when_safe", False
                )
                for i in range(n):
                    ui = x_opt[u_slice.start + 2 * i : u_slice.start + 2 * i + 2]
                    # The min-deviation QP returns u == u_nom for any agent whose
                    # nominal already satisfied the CBF (constraints inactive). So an
                    # unchanged u means "nominal is safe" -> leave the RL action in the
                    # tensordict untouched (exact, no solver-noise perturbation).
                    # Comparison is in control space (accel, steer_rate).
                    if keep_if_safe and np.allclose(ui, nom_acc_omega[i, :], atol=1e-4):
                        continue
                    # u_apply is written to the tensordict after the shield
                    u_apply[i, :] = ui
            else:
                self.hist.evt.append("QP-INF")
                for i in range(n):
                    u_apply[i, :] = self._failure_fallback_control(
                        states[i], i, nom_acc_omega[i, :]
                    )
            if self.parameters.is_standstill_escape:
                for i in range(n):
                    u_apply[i, :] = self._standstill_escape_control(
                        i, states[i], u_apply[i, :]
                    )
            if self.parameters.is_exact_backup_shield:
                u_apply, _ = self._apply_exact_backup_shield(
                    states, u_apply, on_hold_set
                )
            if on_hold_set:
                for held_idx in on_hold_set:
                    u_apply[held_idx, :] = 0.0
            self._write_low_level_controls_to_tensordict(tensordict, states, u_apply)

        applied_target_actions = self._target_actions_from_current_actions(
            tensordict, states
        ).detach()
        self.env.base_env.scenario_name.world_state.applied_action_vel[
            self.env_idx, :
        ] = applied_target_actions[:, 0]
        self.env.base_env.scenario_name.world_state.applied_action_steer[
            self.env_idx, :
        ] = applied_target_actions[:, 1]

        self.hist.qp_solving_t.append(result["solve_time"])
        self.hist.qp_solving_iter.append(result["iter"])

        if self.env.scenario_name.parameters.is_apply_cbf_action:
            self.env.base_env.scenario_name.world_state.nominal_action_vel[
                self.env_idx, :
            ] = nominal_target_actions[:, 0]
            self.env.base_env.scenario_name.world_state.nominal_action_steer[
                self.env_idx, :
            ] = nominal_target_actions[:, 1]
        else:
            if solved:
                x_opt = result["x"]
                compare_target_actions = torch.zeros_like(nominal_target_actions)
                for i in range(n):
                    ui = x_opt[u_slice.start + 2 * i : u_slice.start + 2 * i + 2]
                    compare_target_actions[i] = self.u_to_target_action(
                        ui, states[i][3], states[i][4]
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
        self._fd_qp_cache = dict(
            n_agents=n_agents,
            n_circles=n_circles,
            e_cap=e_cap,
        )

    def update_fully_decentralized_cbf_qps(self, tensordict):
        run_steps(self.update_fully_decentralized_cbf_qps_steps(tensordict))

    def update_fully_decentralized_cbf_qps_steps(self, tensordict):
        """Generator version of update_fully_decentralized_cbf_qps (see
        update_qp_steps)."""
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
        is_aggregate_veh_veh_safety = bool(
            getattr(self.parameters, "is_aggregate_veh_veh_safety", True)
        )
        is_aggregate_lane_veh_safety = bool(
            getattr(self.parameters, "is_aggregate_lane_veh_safety", False)
        )
        # This beta acts on safety margins in m^2. A value like lane_softmin_beta is
        # too small here because it adds a log(N) / beta clearance offset.
        aggregate_safety_softmin_beta = float(
            getattr(
                self.parameters,
                "aggregate_safety_softmin_beta",
                400.0 / LENGTH_SCALE**2,
            )
        )
        is_use_mtv_for_veh_veh = self.veh_veh_cbf_footprint == "mtv"
        is_pair_sat = self.veh_veh_cbf_footprint == "rectangle_sat"

        step_cache = getattr(self, "_fd_step_cache", None)
        if step_cache is not None and step_cache.get("env_idx") == self.env_idx:
            path_ids = step_cache["path_ids"]
            states = step_cache["states"]
            centers = step_cache["centers"]
            centers_flat = step_cache["centers_flat"]
            speeds = step_cache["speeds"]
            kins_all = step_cache["kins_all"]
            local_boundary_cache = step_cache["local_boundary_cache"]
            on_hold_set = step_cache.get("on_hold_set", None)
            if on_hold_set is None:
                on_hold_set = self._get_on_hold_agent_set()
        else:
            (
                path_ids,
                states,
                centers,
                centers_flat,
                speeds,
            ) = self._collect_agent_states(tensordict, n_agents)
            kins_all = None
            local_boundary_cache = {}
            on_hold_set = self._get_on_hold_agent_set()
        uncontrolled_set = self._get_uncontrolled_agent_set()

        if (on_hold_set and i in on_hold_set) or i in uncontrolled_set:
            yield WRITE_BARRIER
            # Not checked by the shield; its floors and priority reset once it
            # moves again
            self._fd_shield_prediction = None
            self._shield_last_xy.pop(i, None)
            zero_u = np.zeros(2, dtype=np.float64)
            action_zero = self.u_to_rl_action(
                zero_u, states[i][3], states[i][4], agent_idx=i
            ).detach()
            tensordict[("agents", "action")][self.env_idx, i] = action_zero
            self.env.base_env.scenario_name.world_state.applied_action_vel[
                self.env_idx, i
            ] = 0.0
            self.env.base_env.scenario_name.world_state.applied_action_steer[
                self.env_idx, i
            ] = 0.0
            self.env.base_env.scenario_name.world_state.nominal_action_vel[
                self.env_idx, i
            ] = 0.0
            self.env.base_env.scenario_name.world_state.nominal_action_steer[
                self.env_idx, i
            ] = 0.0
            if i == 0:
                self.scenario.inter_groups = [[k] for k in range(n_agents)]
                self.scenario.cross_groups = {}
            if hasattr(self.scenario, "cross_groups") and isinstance(
                self.scenario.cross_groups, dict
            ):
                self.scenario.cross_groups[i] = []
            self.hist.qp_solving_t.append(0.0)
            self.hist.qp_solving_iter.append(0)
            return

        learned_lambdas = self._get_env_cbf_lambdas(tensordict)

        a_min_i, a_max_i = self._agent_acc_limits(i)
        if self.nom_controller_type == "rl":
            rl_i = tensordict[("agents", "action")][self.env_idx, i].clone()
            if self.is_obs_noise:
                rl_i = rl_i + torch.rand_like(rl_i) * self.obs_noise_level
            nominal_target_i, u_nom_i = self.rl_action_to_u(
                rl_actions=rl_i,
                v=states[i][3],
                steering=states[i][4],
                state=states[i],
                ref_xy=(
                    self._ref_xy_from_tensordict(tensordict, i)
                    if self.rl_action_type == "speed"
                    else None
                ),
                acc_limits=(a_min_i, a_max_i),
            )
            u_nom_np = u_nom_i.detach().cpu().numpy().reshape(2)
            nominal_target_i = nominal_target_i.detach()
            e_head_i = 0.0
            e_speed_i = 0.0
        else:
            if self.scenario.is_heterogeneous_vehicles:
                raise NotImplementedError(
                    "Heterogeneous vehicle parameters need a policy as the "
                    "nominal controller."
                )
            u_nom_np, e_head_i, e_speed_i = self._clf_nominal_for_agent(
                tensordict, states[i], i
            )
            nominal_target_i = self.u_to_target_action(
                u_nom_np, states[i][3], states[i][4]
            ).detach()
        if self.parameters.is_standstill_escape:
            (
                u_nom_np,
                e_head_i,
                e_speed_i,
                nominal_target_i,
            ) = self._standstill_escape_nominal(
                i, states[i], u_nom_np, e_head_i, e_speed_i, nominal_target_i
            )
        v_head_const_i = self.lam_clf * 0.5 * (e_head_i**2)
        v_speed_const_i = (
            self.lam_clf * 0.5 * (e_speed_i**2) * self.clf_speed_row_scale
        )

        is_other = np.ones(n_agents, dtype=bool)
        is_other[i] = False
        if on_hold_set:
            is_other[list(on_hold_set)] = False
        others = np.nonzero(is_other)[0]
        coupling_ranges = np.full(len(others), float(cbf_coupling_range))
        if self.backup_horizon is not None:
            # Theorem 4 needs every vehicle that the backup tails can reach
            coupling_ranges = np.maximum(
                coupling_ranges,
                self.backup_horizon.coupling_ranges(speeds, i, others),
            )
        d = centers_flat[i] - centers_flat[others]
        distances = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
        nbrs = others[distances <= coupling_ranges + 1e-9].tolist()
        nbrs = nbrs[:e_cap]
        # rectangle_sat needs one lane slack per side, like the aggregated barrier
        is_lane_sat = self.lane_cbf_footprint == "rectangle_sat"
        lane_count = 2 if is_aggregate_lane_veh_safety or is_lane_sat else 2 * n_circles
        cross_count = (
            len(nbrs)
            if is_use_mtv_for_veh_veh or is_pair_sat or is_aggregate_veh_veh_safety
            else len(nbrs) * n_circles * n_circles
        )

        offset = 0
        u_slice = slice(offset, offset + 2)
        offset = u_slice.stop
        s_lane_slice = slice(offset, offset + lane_count)
        offset = s_lane_slice.stop
        s_cross_slice = slice(offset, offset + cross_count) if cross_count > 0 else None
        if s_cross_slice is not None:
            offset = s_cross_slice.stop
        s_clf_head_slice = slice(offset, offset + 1)
        offset = s_clf_head_slice.stop
        s_clf_speed_slice = slice(offset, offset + 1)
        offset = s_clf_speed_slice.stop
        lambda_lane_slice = None
        lambda_cross_slice = None
        if self.parameters.adaptive_lambda:
            lambda_lane_slice = slice(offset, offset + lane_count)
            offset = lambda_lane_slice.stop
            if not self.parameters.is_learn_cbf_lambda and cross_count > 0:
                lambda_cross_slice = slice(offset, offset + cross_count)
                offset = lambda_cross_slice.stop
        n_vars = offset

        H = np.zeros((n_vars, n_vars), dtype=np.float64)
        g = np.zeros((n_vars,), dtype=np.float64)
        l_box = np.full((n_vars,), -PROXQP_INF, dtype=np.float64)
        u_box = np.full((n_vars,), PROXQP_INF, dtype=np.float64)
        rows = []
        lowers = []
        uppers = []

        self._add_quadratic_tracking_cost(
            H,
            g,
            u_slice,
            (self.nom_weight @ self.nom_unit_scale).T
            @ (self.nom_weight @ self.nom_unit_scale),
            u_nom_np,
        )
        self._apply_box_bounds(
            l_box,
            u_box,
            u_slice,
            [a_min_i, self.steering_rate_min],
            [
                self._cbf_acc_upper_bound(u_nom_np[0], (a_min_i, a_max_i)),
                self.steering_rate_max,
            ],
        )
        self._apply_box_bounds(l_box, u_box, s_lane_slice, 0.0, PROXQP_INF)
        self._apply_box_bounds(l_box, u_box, s_clf_head_slice, 0.0, PROXQP_INF)
        self._apply_box_bounds(l_box, u_box, s_clf_speed_slice, 0.0, PROXQP_INF)
        self._add_quadratic_penalty(H, g, s_lane_slice, self.lane_slack_weight)
        self._add_quadratic_penalty(H, g, s_clf_head_slice, self.w_clf_relax)
        self._add_quadratic_penalty(H, g, s_clf_speed_slice, self.w_clf_relax)

        if s_cross_slice is not None:
            self._apply_box_bounds(l_box, u_box, s_cross_slice, 0.0, PROXQP_INF)
            self._add_quadratic_penalty(H, g, s_cross_slice, self.cross_slack_weight)

        if lambda_lane_slice is not None:
            self._apply_box_bounds(l_box, u_box, lambda_lane_slice, 0.0, 1.0)
            self._add_quadratic_penalty(
                H,
                g,
                lambda_lane_slice,
                self.lambda_weight,
                ref=np.full(
                    lambda_lane_slice.stop - lambda_lane_slice.start,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                ),
            )
            if self.parameters.is_learn_cbf_lambda:
                lane_lambda_ref = np.full(
                    lambda_lane_slice.stop - lambda_lane_slice.start,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
                for k, lam_ref in enumerate(lane_lambda_ref):
                    lam_idx = lambda_lane_slice.start + k
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [(lam_idx, 1.0)],
                        lower=lam_ref,
                        upper=lam_ref,
                    )

        if lambda_cross_slice is not None:
            self._apply_box_bounds(l_box, u_box, lambda_cross_slice, 0.0, 1.0)
            self._add_quadratic_penalty(
                H,
                g,
                lambda_cross_slice,
                self.lambda_weight,
                ref=np.full(
                    lambda_cross_slice.stop - lambda_cross_slice.start,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                ),
            )

        self._append_qp_row(
            rows,
            lowers,
            uppers,
            n_vars,
            [(u_slice.start + 1, -e_head_i), (s_clf_head_slice.start, -1.0)],
            lower=-PROXQP_INF,
            upper=-v_head_const_i,
        )
        self._append_qp_row(
            rows,
            lowers,
            uppers,
            n_vars,
            [
                (u_slice.start, -e_speed_i * self.clf_speed_row_scale),
                (s_clf_speed_slice.start, -1.0),
            ],
            lower=-PROXQP_INF,
            upper=-v_speed_const_i,
        )

        def lane_slack_idx(circle_idx, side):
            """Return the lane slack variable index for a circle-side constraint."""
            if is_aggregate_lane_veh_safety or is_lane_sat:
                return s_lane_slice.start + side
            return s_lane_slice.start + 2 * circle_idx + side

        kins_i = (
            kins_all[i]
            if kins_all is not None
            else self.linearized_center_kinematics_coeffs(states[i])
        )
        sat_blocks = []
        if is_lane_sat:
            time_pseudo_dis_start = time.time()
            sat_blocks.append(
                self._append_lane_sat_rows(
                    rows,
                    lowers,
                    uppers,
                    n_vars,
                    l_box,
                    u_box,
                    u_slice,
                    states[i],
                    self.env_idx,
                    i,
                    u_nom_np,
                    slack_idx=(lane_slack_idx(0, 0), lane_slack_idx(0, 1)),
                    lambda_idx=(
                        None
                        if lambda_lane_slice is None
                        else (lambda_lane_slice.start, lambda_lane_slice.start + 1)
                    ),
                )
            )
            self.time_pseudo_dis += time.time() - time_pseudo_dis_start
        lane_distance_results_all = None
        if (
            not is_lane_sat
            and self.parameters.lane_distance_type == "softmin"
            and self.parameters.is_use_sampled_lane_boundary_for_cbf
        ):
            lane_stencil_offsets = torch.tensor(
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
            query_points = centers[i][:, 0:2].unsqueeze(
                1
            ) + lane_stencil_offsets.unsqueeze(0)
            time_pseudo_dis_start = time.time()
            lane_distance_results_all = (
                self._get_local_softmin_distances_with_derivatives_batch(
                    i,
                    query_points,
                    dx=self.dx,
                    dy=self.dy,
                    local_boundary_cache=local_boundary_cache,
                )
            )
            self.time_pseudo_dis += time.time() - time_pseudo_dis_start

        def lane_results_for_circle(circle_idx: int):
            if lane_distance_results_all is not None:
                dleft_results_all, dright_results_all = lane_distance_results_all
                return (
                    dleft_results_all[0][circle_idx] - self.circle_radius**2,
                    dleft_results_all[1][circle_idx],
                    dleft_results_all[2][circle_idx],
                    dright_results_all[0][circle_idx] - self.circle_radius**2,
                    dright_results_all[1][circle_idx],
                    dright_results_all[2][circle_idx],
                )
            p_ic = centers[i][circle_idx][0:2]
            return self.estimate_agent_2_lane_safety_margin(
                p_ic,
                path_ids[i],
                agent_idx=i,
                local_boundary_cache=local_boundary_cache,
            )

        if is_lane_sat:
            pass
        elif is_aggregate_lane_veh_safety:
            lane_terms = {0: [], 1: []}
            for ci in range(n_circles):
                (
                    sm_l,
                    g_l,
                    h_l,
                    sm_r,
                    g_r,
                    h_r,
                ) = lane_results_for_circle(ci)
                lane_terms[0].append(
                    self.ttcbf_lane_affine_components(kins_i, ci, sm_l, g_l, h_l)
                )
                lane_terms[1].append(
                    self.ttcbf_lane_affine_components(kins_i, ci, sm_r, g_r, h_r)
                )

            for side in (0, 1):
                a_lane, b0_lane, h_lane = self.aggregate_softmin_taylor_cbf_terms(
                    lane_terms[side],
                    self.dt_taylor,
                    aggregate_safety_softmin_beta,
                )
                if self.parameters.adaptive_lambda:
                    lambda_idx = lambda_lane_slice.start + side
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (u_slice, a_lane.reshape(-1)),
                            (lane_slack_idx(0, side), 1.0),
                            (lambda_idx, float(np.asarray(h_lane).squeeze())),
                        ],
                        lower=-float(np.asarray(b0_lane).squeeze()),
                        upper=PROXQP_INF,
                    )
                else:
                    b_lane = b0_lane + self.lambda_ttcbf_predefined * h_lane
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (u_slice, a_lane.reshape(-1)),
                            (lane_slack_idx(0, side), 1.0),
                        ],
                        lower=-float(np.asarray(b_lane).squeeze()),
                        upper=PROXQP_INF,
                    )
        else:
            for ci in range(n_circles):
                (
                    sm_l,
                    g_l,
                    h_l,
                    sm_r,
                    g_r,
                    h_r,
                ) = lane_results_for_circle(ci)
                if self.parameters.adaptive_lambda:
                    a_l_ic, b0_l_ic, h_l_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, sm_l, g_l, h_l, self.dt_taylor, None
                    )
                    a_r_ic, b0_r_ic, h_r_ic = self.ttcbf_lane_affine_coeffs(
                        kins_i, ci, sm_r, g_r, h_r, self.dt_taylor, None
                    )
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (u_slice, a_l_ic.reshape(-1)),
                            (lane_slack_idx(ci, 0), 1.0),
                            (
                                lambda_lane_slice.start + 2 * ci,
                                float(np.asarray(h_l_ic).squeeze()),
                            ),
                        ],
                        lower=-float(np.asarray(b0_l_ic).squeeze()),
                        upper=PROXQP_INF,
                    )
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (u_slice, a_r_ic.reshape(-1)),
                            (lane_slack_idx(ci, 1), 1.0),
                            (
                                lambda_lane_slice.start + 2 * ci + 1,
                                float(np.asarray(h_r_ic).squeeze()),
                            ),
                        ],
                        lower=-float(np.asarray(b0_r_ic).squeeze()),
                        upper=PROXQP_INF,
                    )
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
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (u_slice, a_l_ic.reshape(-1)),
                            (lane_slack_idx(ci, 0), 1.0),
                        ],
                        lower=-float(np.asarray(b_l_ic).squeeze()),
                        upper=PROXQP_INF,
                    )
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (u_slice, a_r_ic.reshape(-1)),
                            (lane_slack_idx(ci, 1), 1.0),
                        ],
                        lower=-float(np.asarray(b_r_ic).squeeze()),
                        upper=PROXQP_INF,
                    )

        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe
        body_circle_radius = float(math.sqrt(self.length**2 + self.width**2) / 2.0)
        body_circle_d_safe = float(2.0 * body_circle_radius + self.safety_buffer)
        body_circle_d_safe_sq = body_circle_d_safe * body_circle_d_safe
        cross_scale = self._cross_cbf_drift_scale()
        lam_cross = (
            self._get_agent_cbf_lambda(learned_lambdas, i)
            if self.parameters.is_learn_cbf_lambda
            else float(self.lambda_ttcbf_predefined)
        )
        mtv_mode_counts = {"mtv": 0, "single_circle_fallback": 0}
        pair_sat_pairs = []

        for e_idx, j in enumerate(nbrs):
            cross_scale = (
                1.0
                if j in uncontrolled_set
                else self._cross_cbf_responsibility_share(states[i], states[j])
            )
            if is_pair_sat:
                pair_sat_pairs.append(
                    self._pair_sat_cross_pair(
                        i,
                        j,
                        u_slice,
                        cross_scale,
                        lam_cross,
                        None if s_cross_slice is None else s_cross_slice.start + e_idx,
                        (
                            None
                            if lambda_cross_slice is None
                            else lambda_cross_slice.start + e_idx
                        ),
                    )
                )
                continue
            if is_use_mtv_for_veh_veh:
                relative_pose = self._relative_pose_in_ego_frame(states[i], states[j])
                if self._is_mtv_pose_in_domain(relative_pose):
                    a_i, b0, h = self.ttcbf_mtv_affine_coeffs_cross(
                        states[i], states[j], self.dt_taylor
                    )
                    mtv_mode_counts["mtv"] += 1
                else:
                    kins_i_center = self.linearized_vehicle_center_kinematics_coeffs(
                        states[i]
                    )
                    kins_j_center = self.linearized_vehicle_center_kinematics_coeffs(
                        states[j]
                    )
                    dx_center = float(states[i][0].item() - states[j][0].item())
                    dy_center = float(states[i][1].item() - states[j][1].item())
                    a_i, b0, h = self.ttcbf_pair_affine_coeffs_cross(
                        kins_i_center,
                        kins_j_center,
                        0,
                        0,
                        dx_center,
                        dy_center,
                        body_circle_d_safe_sq,
                        self.dt_taylor,
                        None,
                    )
                    mtv_mode_counts["single_circle_fallback"] += 1

                slack_terms = (
                    []
                    if s_cross_slice is None
                    else [(s_cross_slice.start + e_idx, 1.0)]
                )
                if lambda_cross_slice is not None:
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (u_slice, a_i.reshape(-1)),
                            *slack_terms,
                            (
                                lambda_cross_slice.start + e_idx,
                                cross_scale * float(np.asarray(h).squeeze()),
                            ),
                        ],
                        lower=-cross_scale * float(np.asarray(b0).squeeze()),
                        upper=PROXQP_INF,
                    )
                else:
                    b = cross_scale * (b0 + lam_cross * h)
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [(u_slice, a_i.reshape(-1)), *slack_terms],
                        lower=-float(np.asarray(b).squeeze()),
                        upper=PROXQP_INF,
                    )
                continue

            kins_j = (
                kins_all[j]
                if kins_all is not None
                else self.linearized_center_kinematics_coeffs(states[j])
            )
            if is_aggregate_veh_veh_safety:
                pair_terms = []
                for ci in range(n_circles):
                    pi = centers[i][ci][0:2]
                    for cj in range(n_circles):
                        pj = centers[j][cj][0:2]
                        dx = float((pi - pj)[0].item())
                        dy = float((pi - pj)[1].item())
                        pair_terms.append(
                            self.ttcbf_pair_cross_affine_components(
                                kins_i,
                                kins_j,
                                ci,
                                cj,
                                dx,
                                dy,
                                d_safe_sq,
                            )
                        )
                a_i, b0, h = self.aggregate_softmin_taylor_cbf_terms(
                    pair_terms,
                    self.dt_taylor,
                    aggregate_safety_softmin_beta,
                )
                slack_terms = (
                    []
                    if s_cross_slice is None
                    else [(s_cross_slice.start + e_idx, 1.0)]
                )
                if lambda_cross_slice is not None:
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [
                            (u_slice, a_i.reshape(-1)),
                            *slack_terms,
                            (
                                lambda_cross_slice.start + e_idx,
                                cross_scale * float(np.asarray(h).squeeze()),
                            ),
                        ],
                        lower=-cross_scale * float(np.asarray(b0).squeeze()),
                        upper=PROXQP_INF,
                    )
                else:
                    b = cross_scale * (b0 + lam_cross * h)
                    self._append_qp_row(
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                        [(u_slice, a_i.reshape(-1)), *slack_terms],
                        lower=-float(np.asarray(b).squeeze()),
                        upper=PROXQP_INF,
                    )
                continue

            cross_base = e_idx * n_circles * n_circles
            for ci in range(n_circles):
                pi = centers[i][ci][0:2]
                for cj in range(n_circles):
                    pj = centers[j][cj][0:2]
                    dx = float((pi - pj)[0].item())
                    dy = float((pi - pj)[1].item())
                    cross_offset = cross_base + ci * n_circles + cj
                    slack_terms = (
                        []
                        if s_cross_slice is None
                        else [(s_cross_slice.start + cross_offset, 1.0)]
                    )
                    if lambda_cross_slice is not None:
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
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            [
                                (u_slice, a_i.reshape(-1)),
                                *slack_terms,
                                (
                                    lambda_cross_slice.start + cross_offset,
                                    cross_scale * float(np.asarray(h).squeeze()),
                                ),
                            ],
                            lower=-cross_scale * float(np.asarray(b0).squeeze()),
                            upper=PROXQP_INF,
                        )
                    else:
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
                        b = cross_scale * (b0 + lam_cross * h)
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            [(u_slice, a_i.reshape(-1)), *slack_terms],
                            lower=-float(np.asarray(b).squeeze()),
                            upper=PROXQP_INF,
                        )

        if pair_sat_pairs:
            sat_blocks.append(
                self._append_pair_sat_rows(
                    pair_sat_pairs,
                    {
                        k: np.asarray([float(value) for value in states[k][:5]])
                        for k in [i, *nbrs]
                    },
                    {i: np.clip(u_nom_np, l_box[u_slice], u_box[u_slice])},
                    uncontrolled_set,
                    rows,
                    lowers,
                    uppers,
                    n_vars,
                )
            )

        if is_lane_sat:
            result = yield from self._solve_qp_with_sat_repairs_steps(
                H, g, rows, lowers, uppers, n_vars, l_box, u_box, sat_blocks
            )
        else:
            C_mat, l_vec, u_vec = self._finalize_qp_rows(rows, lowers, uppers, n_vars)
            result = self._solve_proxqp_problem(H, g, C_mat, l_vec, u_vec, l_box, u_box)
        yield WRITE_BARRIER
        self._fd_qp_cache["last_result"] = result
        self._fd_qp_cache["last_mtv_mode_counts"] = mtv_mode_counts
        self._record_slack_activity(result, [s_lane_slice, s_cross_slice])

        if result["solved"]:
            if self.is_debug_slack_var:
                x_opt = result["x"]
                cross_base_debug = {
                    (0, e_idx): e_idx * n_circles * n_circles
                    for e_idx in range(len(nbrs))
                }
                if not is_aggregate_lane_veh_safety:
                    self._debug_print_lane_slacks(
                        x_opt,
                        s_lane_slice,
                        [i],
                        1 if is_lane_sat else n_circles,
                        path_ids,
                        mode="fully_decentralized",
                    )
                if not (
                    is_use_mtv_for_veh_veh or is_pair_sat or is_aggregate_veh_veh_safety
                ):
                    self._debug_print_cross_slacks(
                        x_opt,
                        s_cross_slice,
                        [i],
                        [nbrs],
                        cross_base_debug,
                        n_circles,
                        mode="fully_decentralized",
                    )
            u_apply_i = result["x"][u_slice]
            if self.parameters.is_standstill_escape:
                u_apply_i = self._standstill_escape_control(i, states[i], u_apply_i)
            if self.parameters.is_exact_backup_shield:
                step_cache["shield_proposals"][i] = self._fd_shield_propose(
                    states, u_apply_i
                )
            action_safe = self.u_to_rl_action(
                u_apply_i, states[i][3], states[i][4], agent_idx=i
            ).detach()
            tensordict[("agents", "action")][self.env_idx, i] = action_safe
        else:
            if result["error"] is not None:
                print(
                    "[WARN] Fully decentralized proxsuite solve failed. Using the fallback "
                    f"control ({self.failure_fallback}). status={result['status']}"
                )
            u_apply_i = self._failure_fallback_control(states[i], i, u_nom_np)
            if self.parameters.is_standstill_escape:
                u_apply_i = self._standstill_escape_control(i, states[i], u_apply_i)
            if self.parameters.is_exact_backup_shield:
                step_cache["shield_proposals"][i] = self._fd_shield_propose(
                    states, u_apply_i
                )
            action_nom = self.u_to_rl_action(
                u_apply_i, states[i][3], states[i][4], agent_idx=i
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

        self.hist.qp_solving_t.append(result["solve_time"])
        self.hist.qp_solving_iter.append(result["iter"])

    def _get_nominal_cbf_constraint_values(self, tol: float = 1e-9):
        margins = getattr(self, "_last_nominal_cbf_margins", None)
        if margins is None:
            raise ValueError(
                "Nominal CBF margins have not been computed yet in the proxsuite path."
            )

        lane_L_expr = -np.asarray(margins["lane_L_margin"], dtype=np.float64)
        lane_R_expr = -np.asarray(margins["lane_R_margin"], dtype=np.float64)
        pair_expr = {
            key: -float(value) for key, value in margins["pair_margin"].items()
        }
        return dict(
            lane_L_expr=lane_L_expr,
            lane_R_expr=lane_R_expr,
            lane_L_ok=lane_L_expr <= tol,
            lane_R_ok=lane_R_expr <= tol,
            pair_expr=pair_expr,
            pair_ok={key: value <= tol for key, value in pair_expr.items()},
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
        if self.parameters.cbf_lambda_head == "pair":
            raise NotImplementedError(
                "cbf_lambda_head='pair' is only supported by the centralized CBF-QP."
            )

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

        lambda_env = np.clip(lambda_env, 0.0, 1.0)
        self.scenario.set_cbf_lambda_render_rows(
            self.env_idx, [[(i, float(lam))] for i, lam in enumerate(lambda_env)]
        )
        return lambda_env

    def _get_env_pair_cbf_lambdas(self, tensordict) -> np.ndarray:
        """Return the symmetric pair-lambda matrix of this env from the pair head.

        Agent i proposes lambda_{i->j} for each valid observed neighbor j. A
        pair's decay rate is the mean of its two directed values if both
        agents observe each other, the single value if only one does, and
        ``lambda_ttcbf_predefined`` if neither does.

        Returns:
            ``[n_agents, n_agents]`` array of lambdas in [0, 1].
        """
        pair_lambda = (
            tensordict.get(("agents", "cbf", "pair_lambda"))[self.env_idx]
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )  # [n_agents, K]
        observation_provider = self.scenario.observation_provider
        neighbor_indices = (
            observation_provider.observations.nearing_agents_indices[self.env_idx]
            .cpu()
            .numpy()
        )  # [n_agents, K]
        neighbor_valid = (
            observation_provider.get_neighbor_valid()[self.env_idx].cpu().numpy()
        )  # [n_agents, K]

        n = pair_lambda.shape[0]
        lambda_sum = np.zeros((n, n), dtype=np.float64)
        lambda_count = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for k in range(pair_lambda.shape[1]):
                j = int(neighbor_indices[i, k])
                if j == i or not neighbor_valid[i, k]:
                    continue
                if not np.isfinite(pair_lambda[i, k]):
                    continue
                lambda_sum[i, j] += pair_lambda[i, k]
                lambda_sum[j, i] += pair_lambda[i, k]
                lambda_count[i, j] += 1.0
                lambda_count[j, i] += 1.0

        lambdas = np.full((n, n), float(self.lambda_ttcbf_predefined))
        has_value = lambda_count > 0
        lambdas[has_value] = lambda_sum[has_value] / lambda_count[has_value]
        lambdas = np.clip(lambdas, 0.0, 1.0)
        # Render each ego's pair lambdas with its valid observed neighbors
        self.scenario.set_cbf_lambda_render_rows(
            self.env_idx,
            [
                [
                    (int(j), float(lambdas[i, j]))
                    for j, valid in zip(neighbor_indices[i], neighbor_valid[i])
                    if valid and j != i
                ]
                for i in range(n)
            ],
        )
        return lambdas

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

    def _cross_cbf_responsibility_share(self, state_i, state_j) -> float:
        """Share of a decentralized pair CBF that ego i takes on (j takes the rest).

        "half" (legacy) splits every pair constraint equally (see
        _cross_cbf_drift_scale). "follower" gives the larger share to the agent
        that drives toward the other one, so a follower brakes instead of pushing
        its leader to speed away. With f_i = max(0, (p_j - p_i) . heading_i), the
        share is (f_i + c) / (f_i + f_j + 2 c), where c = vehicle length. Both
        agents compute complementary shares from the same states, so the two
        directed constraints still sum to the centralized one.
        """
        if self.cross_cbf_responsibility == "half":
            return self._cross_cbf_drift_scale()
        if self.cross_cbf_responsibility != "follower":
            raise ValueError(
                f"Unknown cross_cbf_responsibility {self.cross_cbf_responsibility!r}."
            )
        dx = float(state_j[0]) - float(state_i[0])
        dy = float(state_j[1]) - float(state_i[1])
        psi_i = float(state_i[2])
        psi_j = float(state_j[2])
        f_i = max(0.0, dx * math.cos(psi_i) + dy * math.sin(psi_i))
        f_j = max(0.0, -dx * math.cos(psi_j) - dy * math.sin(psi_j))
        c = float(self.length)
        return (f_i + c) / (f_i + f_j + 2.0 * c)

    def _find_invalid_qp_parameter(self, obj, path="cache"):
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj, dtype=np.float64)
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
        if np.isscalar(obj):
            if not np.isfinite(obj):
                return path, f"non-finite scalar {obj!r}"
            return None

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"pair_base", "cross_base", "idx_lane", "last_result"}:
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

    def _debug_osqp_bound_update_failure(self, prob, context: dict):
        print(
            "[DEBUG] Legacy OSQP debug hook is disabled in the proxsuite version. "
            f"context={context}"
        )

    def build_grouped_cbf_qps(self):
        return self._build_grouped_cbf_qps_proxsuite()

    def update_grouped_cbf_qps(self, tensordict):
        return self._update_grouped_cbf_qps_proxsuite(tensordict)

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

        # Heading terms used by the rectangle lane footprints.
        kins["psi"] = float(psi)
        kins["dpsi"] = float(dstate_time_agent[2])
        kins["a_ddpsi"] = np.array(
            [ddpsi_e1 - ddpsi0, ddpsi_e2 - ddpsi0], dtype=np.float64
        )
        kins["c_ddpsi"] = float(ddpsi0)

        return kins

    def _build_grouped_cbf_qps_proxsuite(self):
        print("[INFO] Building grouped CBF-QPs...")
        if hasattr(self, "_group_qp_caches"):
            return

        N = int(self.parameters.n_agents)
        C = int(self.parameters.n_circles_approximate_vehicle)
        m = int(self.parameters.max_group_size)
        if m < 1:
            raise ValueError("max_group_size must be >= 1")
        G = int(ceil(N / m))

        if G == 1:
            self._group_cross_cap = 0
        else:
            e_cap_user = getattr(self.parameters, "max_cross_neighbors", None)
            if e_cap_user is None:
                self._group_cross_cap = max(0, N - 1)
            else:
                self._group_cross_cap = min(max(0, N - 1), max(0, int(e_cap_user)))

        self._group_qp_caches = [{} for _ in range(G)]
        self._group_capacity = m
        self._group_count = G
        self._group_circle_count = C
        self._group_assignment = [[] for _ in range(G)]
        self.use_fixed_groups = False
        self.fixed_groups = None
        self._previous_interaction_groups = None

    def _grouping_interaction_range(self, cbf_coupling_range: float) -> float:
        configured_range = getattr(self.parameters, "grouping_interaction_range", None)
        if configured_range is not None:
            return float(configured_range)
        return max(float(cbf_coupling_range), 2.5 * float(self.circle_radius))

    def _grouping_closing_speed_scale(self) -> float:
        return float(self.parameters.grouping_closing_speed_scale)

    def _select_grouped_qp_agent_groups(
        self,
        agent_states: List[AgentStateList],
        max_group_size: int,
        expected_group_count: int,
        cbf_coupling_range: float,
    ) -> List[List[int]]:
        grouping_method = str(
            getattr(self.parameters, "grouping_method", "interaction_risk")
        ).lower()

        if grouping_method in {"k_nearest", "k-nearest", "nearest", "centroid"}:
            groups = group_agents_k_nearest(agent_states, max_group_size)
        elif grouping_method in {"interaction_risk", "risk", "risk_greedy"}:
            use_hysteresis = bool(
                getattr(self.parameters, "grouping_use_hysteresis", True)
            )
            groups = group_agents_by_interaction_risk(
                agent_states,
                max_group_size=max_group_size,
                interaction_range=self._grouping_interaction_range(cbf_coupling_range),
                previous_groups=(
                    self._previous_interaction_groups if use_hysteresis else None
                ),
                hysteresis_bonus=float(
                    getattr(self.parameters, "grouping_hysteresis_bonus", 0.25)
                ),
                hysteresis_keep_range_factor=float(
                    getattr(
                        self.parameters, "grouping_hysteresis_keep_range_factor", 1.25
                    )
                ),
                risk_threshold=float(
                    getattr(self.parameters, "grouping_risk_threshold", 0.0)
                ),
                closing_speed_scale=self._grouping_closing_speed_scale(),
            )
        else:
            raise ValueError(
                f"Unknown grouping_method={grouping_method!r}. "
                "Use 'interaction_risk' or 'k_nearest'."
            )

        groups = [sorted(list(g)) for g in groups]
        groups.sort(key=lambda g: g[0] if len(g) > 0 else 10**9)
        if len(groups) != expected_group_count:
            raise RuntimeError(
                f"Grouping produced {len(groups)} groups, expected {expected_group_count}."
            )
        self._previous_interaction_groups = [list(g) for g in groups]
        return groups

    def _update_grouped_cbf_qps_proxsuite(self, tensordict):
        assert hasattr(self, "_group_qp_caches"), "Call build_grouped_cbf_qps first."
        N = int(self.parameters.n_agents)
        C = self._group_circle_count
        m = self._group_capacity
        G = self._group_count
        cbf_coupling_range = float(self.parameters.cbf_coupling_range)
        learned_lambdas = self._get_env_cbf_lambdas(tensordict)
        is_aggregate_veh_veh_safety = bool(
            getattr(self.parameters, "is_aggregate_veh_veh_safety", True)
        )
        is_aggregate_lane_veh_safety = bool(
            getattr(self.parameters, "is_aggregate_lane_veh_safety", False)
        )
        aggregate_safety_softmin_beta = float(
            getattr(
                self.parameters,
                "aggregate_safety_softmin_beta",
                400.0 / LENGTH_SCALE**2,
            )
        )
        is_use_mtv_for_veh_veh = self.veh_veh_cbf_footprint == "mtv"
        is_pair_sat = self.veh_veh_cbf_footprint == "rectangle_sat"
        # rectangle_sat needs one lane slack per side, like the aggregated barrier
        is_lane_sat = self.lane_cbf_footprint == "rectangle_sat"
        is_two_lane_slacks = is_aggregate_lane_veh_safety or is_lane_sat
        lane_count_per_agent = 2 if is_two_lane_slacks else 2 * C
        pair_circle_factor = (
            1
            if is_use_mtv_for_veh_veh or is_pair_sat or is_aggregate_veh_veh_safety
            else C
        )
        cross_count_per_neighbor = (
            1
            if is_use_mtv_for_veh_veh or is_pair_sat or is_aggregate_veh_veh_safety
            else C * C
        )

        path_ids, states, centers, centers_flat, _ = self._collect_agent_states(
            tensordict, N
        )
        on_hold_set = self._get_on_hold_agent_set()
        asl = [
            AgentStateList(
                x=float(cf[0]),
                y=float(cf[1]),
                v=float(states[i][3].item()),
                theta=float(states[i][2].item()),
            )
            for i, cf in enumerate(centers_flat)
        ]

        if self.use_fixed_groups:
            if self.fixed_groups is None:
                self.fixed_groups = self._select_grouped_qp_agent_groups(
                    asl, m, G, cbf_coupling_range
                )
            groups = self.fixed_groups
        else:
            groups = self._select_grouped_qp_agent_groups(asl, m, G, cbf_coupling_range)

        U_nom_all = np.zeros((N, 2), dtype=np.float64)
        nominal_target_actions_all = torch.zeros(
            (N, 2), device=self.device, dtype=torch.float32
        )
        e_h_all = np.zeros((N,), dtype=np.float64)
        e_v_all = np.zeros((N,), dtype=np.float64)
        for i in range(N):
            if self.nom_controller_type == "rl":
                rl_i = tensordict[("agents", "action")][self.env_idx, i].clone()
                if self.is_obs_noise:
                    rl_i = rl_i + torch.rand_like(rl_i) * self.obs_noise_level
                target_action_i, u_nom_i = self.rl_action_to_u(
                    rl_actions=rl_i,
                    v=states[i][3],
                    steering=states[i][4],
                    state=states[i],
                    ref_xy=(
                        self._ref_xy_from_tensordict(tensordict, i)
                        if self.rl_action_type == "speed"
                        else None
                    ),
                )
                nominal_target_actions_all[i, :] = target_action_i.detach()
                U_nom_all[i, :] = u_nom_i.detach().cpu().numpy()
            else:
                U_nom_all[i, :], e_h, e_v = self._clf_nominal_for_agent(
                    tensordict, states[i], i
                )
                e_h_all[i] = e_h
                e_v_all[i] = e_v
                nominal_target_actions_all[i, :] = self.u_to_target_action(
                    U_nom_all[i, :], states[i][3], states[i][4]
                ).detach()
            if self.parameters.is_standstill_escape and i not in on_hold_set:
                (
                    U_nom_all[i, :],
                    e_h_all[i],
                    e_v_all[i],
                    nominal_target_actions_all[i, :],
                ) = self._standstill_escape_nominal(
                    i,
                    states[i],
                    U_nom_all[i, :],
                    e_h_all[i],
                    e_v_all[i],
                    nominal_target_actions_all[i, :],
                )

        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe
        body_circle_radius = float(math.sqrt(self.length**2 + self.width**2) / 2.0)
        body_circle_d_safe = float(2.0 * body_circle_radius + self.safety_buffer)
        body_circle_d_safe_sq = body_circle_d_safe * body_circle_d_safe
        scaled_nom_weight = self.nom_weight @ self.nom_unit_scale
        track_weight = scaled_nom_weight.T @ scaled_nom_weight
        cross_scale = self._cross_cbf_drift_scale()

        qp_solving_t_groups = []
        qp_solving_iter_groups = []
        cross_groups: Dict[int, List[int]] = {}

        for g_idx, members in enumerate(groups):
            members = members[:m]
            n_local = len(members)
            if n_local == 0:
                qp_solving_t_groups.append(0.0)
                qp_solving_iter_groups.append(0)
                continue

            pair_base, pair_count = self._build_pair_base(n_local, pair_circle_factor)
            neighbor_lists = []
            cross_base = {}
            cross_count = 0
            for i_loc, i in enumerate(members):
                nbrs = []
                if not (on_hold_set and i in on_hold_set):
                    for j in range(N):
                        if j in members or (on_hold_set and j in on_hold_set):
                            continue
                        if (
                            _euclid(centers_flat[i], centers_flat[j])
                            <= cbf_coupling_range + 1e-9
                        ):
                            nbrs.append(j)
                nbrs = sort_neighbors_by_interaction_risk(
                    asl,
                    i,
                    nbrs,
                    interaction_range=self._grouping_interaction_range(
                        cbf_coupling_range
                    ),
                    closing_speed_scale=self._grouping_closing_speed_scale(),
                )
                nbrs = nbrs[: int(self._group_cross_cap)]
                neighbor_lists.append(nbrs)
                cross_groups[i] = nbrs
                for e_idx in range(len(nbrs)):
                    cross_base[(i_loc, e_idx)] = cross_count
                    cross_count += cross_count_per_neighbor

            offset = 0
            u_slice = slice(offset, offset + 2 * n_local)
            offset = u_slice.stop
            s_lane_slice = slice(offset, offset + n_local * lane_count_per_agent)
            offset = s_lane_slice.stop
            s_pair_slice = (
                slice(offset, offset + pair_count) if pair_count > 0 else None
            )
            if s_pair_slice is not None:
                offset = s_pair_slice.stop
            s_cross_slice = (
                slice(offset, offset + cross_count) if cross_count > 0 else None
            )
            if s_cross_slice is not None:
                offset = s_cross_slice.stop
            s_clf_head_slice = slice(offset, offset + n_local)
            offset = s_clf_head_slice.stop
            s_clf_speed_slice = slice(offset, offset + n_local)
            offset = s_clf_speed_slice.stop
            lambda_lane_slice = None
            lambda_pair_slice = None
            lambda_cross_slice = None
            if self.parameters.adaptive_lambda:
                lambda_lane_slice = slice(
                    offset, offset + n_local * lane_count_per_agent
                )
                offset = lambda_lane_slice.stop
                if not self.parameters.is_learn_cbf_lambda and pair_count > 0:
                    lambda_pair_slice = slice(offset, offset + pair_count)
                    offset = lambda_pair_slice.stop
                if not self.parameters.is_learn_cbf_lambda and cross_count > 0:
                    lambda_cross_slice = slice(offset, offset + cross_count)
                    offset = lambda_cross_slice.stop
            n_vars = offset

            H = np.zeros((n_vars, n_vars), dtype=np.float64)
            g = np.zeros((n_vars,), dtype=np.float64)
            l_box = np.full((n_vars,), -PROXQP_INF, dtype=np.float64)
            u_box = np.full((n_vars,), PROXQP_INF, dtype=np.float64)
            rows = []
            lowers = []
            uppers = []

            for i_loc, gid in enumerate(members):
                ui_slice = slice(
                    u_slice.start + 2 * i_loc, u_slice.start + 2 * i_loc + 2
                )
                self._add_quadratic_tracking_cost(
                    H, g, ui_slice, track_weight, U_nom_all[gid, :]
                )
                self._apply_box_bounds(
                    l_box,
                    u_box,
                    ui_slice,
                    [self.a_min, self.steering_rate_min],
                    [
                        self._cbf_acc_upper_bound(U_nom_all[gid, 0]),
                        self.steering_rate_max,
                    ],
                )
                v_head_const = self.lam_clf * 0.5 * (e_h_all[gid] ** 2)
                v_speed_const = (
                    self.lam_clf * 0.5 * (e_v_all[gid] ** 2) * self.clf_speed_row_scale
                )
                self._append_qp_row(
                    rows,
                    lowers,
                    uppers,
                    n_vars,
                    [
                        (ui_slice.start + 1, -e_h_all[gid]),
                        (s_clf_head_slice.start + i_loc, -1.0),
                    ],
                    lower=-PROXQP_INF,
                    upper=-v_head_const,
                )
                self._append_qp_row(
                    rows,
                    lowers,
                    uppers,
                    n_vars,
                    [
                        (ui_slice.start, -e_v_all[gid] * self.clf_speed_row_scale),
                        (s_clf_speed_slice.start + i_loc, -1.0),
                    ],
                    lower=-PROXQP_INF,
                    upper=-v_speed_const,
                )

            self._apply_box_bounds(l_box, u_box, s_lane_slice, 0.0, PROXQP_INF)
            self._apply_box_bounds(l_box, u_box, s_clf_head_slice, 0.0, PROXQP_INF)
            self._apply_box_bounds(l_box, u_box, s_clf_speed_slice, 0.0, PROXQP_INF)
            self._add_quadratic_penalty(H, g, s_lane_slice, self.lane_slack_weight)
            self._add_quadratic_penalty(H, g, s_clf_head_slice, self.w_clf_relax)
            self._add_quadratic_penalty(H, g, s_clf_speed_slice, self.w_clf_relax)
            if s_pair_slice is not None:
                self._apply_box_bounds(l_box, u_box, s_pair_slice, 0.0, PROXQP_INF)
                self._add_quadratic_penalty(H, g, s_pair_slice, self.pair_slack_weight)
            if s_cross_slice is not None:
                self._apply_box_bounds(l_box, u_box, s_cross_slice, 0.0, PROXQP_INF)
                self._add_quadratic_penalty(
                    H, g, s_cross_slice, self.cross_slack_weight
                )

            if lambda_lane_slice is not None:
                self._apply_box_bounds(l_box, u_box, lambda_lane_slice, 0.0, 1.0)
                lane_ref = np.full(
                    lambda_lane_slice.stop - lambda_lane_slice.start,
                    float(self.lambda_ttcbf_predefined),
                    dtype=np.float64,
                )
                self._add_quadratic_penalty(
                    H, g, lambda_lane_slice, self.lambda_weight, ref=lane_ref
                )
                if self.parameters.is_learn_cbf_lambda:
                    for k, ref_val in enumerate(lane_ref):
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            [(lambda_lane_slice.start + k, 1.0)],
                            lower=ref_val,
                            upper=ref_val,
                        )
            if lambda_pair_slice is not None:
                self._apply_box_bounds(l_box, u_box, lambda_pair_slice, 0.0, 1.0)
                self._add_quadratic_penalty(
                    H,
                    g,
                    lambda_pair_slice,
                    self.lambda_weight,
                    ref=np.full(
                        lambda_pair_slice.stop - lambda_pair_slice.start,
                        float(self.lambda_ttcbf_predefined),
                        dtype=np.float64,
                    ),
                )
            if lambda_cross_slice is not None:
                self._apply_box_bounds(l_box, u_box, lambda_cross_slice, 0.0, 1.0)
                self._add_quadratic_penalty(
                    H,
                    g,
                    lambda_cross_slice,
                    self.lambda_weight,
                    ref=np.full(
                        lambda_cross_slice.stop - lambda_cross_slice.start,
                        float(self.lambda_ttcbf_predefined),
                        dtype=np.float64,
                    ),
                )

            def lane_slack_idx(i_loc, ci, side):
                if is_two_lane_slacks:
                    return s_lane_slice.start + i_loc * lane_count_per_agent + side
                return s_lane_slice.start + 2 * (i_loc * C + ci) + side

            kins_local = [
                self.linearized_center_kinematics_coeffs(states[gid]) for gid in members
            ]
            sat_blocks = []
            for i_loc, gid in enumerate(members):
                ui_slice = slice(
                    u_slice.start + 2 * i_loc, u_slice.start + 2 * i_loc + 2
                )
                kins_i = kins_local[i_loc]
                if self.lane_cbf_footprint == "rectangle_sat":
                    time_pseudo_dis_start = time.time()
                    sat_blocks.append(
                        self._append_lane_sat_rows(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            l_box,
                            u_box,
                            ui_slice,
                            states[gid],
                            self.env_idx,
                            gid,
                            U_nom_all[gid, :],
                            slack_idx=(
                                lane_slack_idx(i_loc, 0, 0),
                                lane_slack_idx(i_loc, 0, 1),
                            ),
                            lambda_idx=(
                                None
                                if lambda_lane_slice is None
                                else (
                                    lambda_lane_slice.start
                                    + i_loc * lane_count_per_agent,
                                    lambda_lane_slice.start
                                    + i_loc * lane_count_per_agent
                                    + 1,
                                )
                            ),
                        )
                    )
                    self.time_pseudo_dis += time.time() - time_pseudo_dis_start
                    continue
                if is_aggregate_lane_veh_safety:
                    lane_terms = {0: [], 1: []}
                    for ci in range(C):
                        circle_pos_i = centers[gid][ci][0:2]
                        (
                            smL,
                            gL,
                            HL,
                            smR,
                            gR,
                            HR,
                        ) = self.estimate_agent_2_lane_safety_margin(
                            circle_pos_i, path_ids[gid], agent_idx=gid
                        )
                        lane_terms[0].append(
                            self.ttcbf_lane_affine_components(kins_i, ci, smL, gL, HL)
                        )
                        lane_terms[1].append(
                            self.ttcbf_lane_affine_components(kins_i, ci, smR, gR, HR)
                        )
                    for side in (0, 1):
                        (
                            A_lane,
                            b0_lane,
                            h_lane,
                        ) = self.aggregate_softmin_taylor_cbf_terms(
                            lane_terms[side],
                            self.dt_taylor,
                            aggregate_safety_softmin_beta,
                        )
                        terms = [
                            (ui_slice, A_lane.reshape(-1)),
                            (lane_slack_idx(i_loc, 0, side), 1.0),
                        ]
                        if self.parameters.adaptive_lambda:
                            terms.append(
                                (
                                    lambda_lane_slice.start
                                    + i_loc * lane_count_per_agent
                                    + side,
                                    float(np.asarray(h_lane).squeeze()),
                                )
                            )
                            lower = -float(np.asarray(b0_lane).squeeze())
                        else:
                            lower = -float(
                                np.asarray(
                                    b0_lane + self.lambda_ttcbf_predefined * h_lane
                                ).squeeze()
                            )
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            terms,
                            lower=lower,
                            upper=PROXQP_INF,
                        )
                    continue

                for ci in range(C):
                    circle_pos_i = centers[gid][ci][0:2]
                    smL, gL, HL, smR, gR, HR = self.estimate_agent_2_lane_safety_margin(
                        circle_pos_i, path_ids[gid], agent_idx=gid
                    )
                    if self.parameters.adaptive_lambda:
                        A_L_ic, b0_L_ic, h_L_ic = self.ttcbf_lane_affine_coeffs(
                            kins_i, ci, smL, gL, HL, self.dt_taylor, None
                        )
                        A_R_ic, b0_R_ic, h_R_ic = self.ttcbf_lane_affine_coeffs(
                            kins_i, ci, smR, gR, HR, self.dt_taylor, None
                        )
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            [
                                (ui_slice, A_L_ic.reshape(-1)),
                                (lane_slack_idx(i_loc, ci, 0), 1.0),
                                (
                                    lambda_lane_slice.start + 2 * (i_loc * C + ci),
                                    float(np.asarray(h_L_ic).squeeze()),
                                ),
                            ],
                            lower=-float(np.asarray(b0_L_ic).squeeze()),
                            upper=PROXQP_INF,
                        )
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            [
                                (ui_slice, A_R_ic.reshape(-1)),
                                (lane_slack_idx(i_loc, ci, 1), 1.0),
                                (
                                    lambda_lane_slice.start + 2 * (i_loc * C + ci) + 1,
                                    float(np.asarray(h_R_ic).squeeze()),
                                ),
                            ],
                            lower=-float(np.asarray(b0_R_ic).squeeze()),
                            upper=PROXQP_INF,
                        )
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
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            [
                                (ui_slice, A_L_ic.reshape(-1)),
                                (lane_slack_idx(i_loc, ci, 0), 1.0),
                            ],
                            lower=-float(np.asarray(b_L_ic).squeeze()),
                            upper=PROXQP_INF,
                        )
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            [
                                (ui_slice, A_R_ic.reshape(-1)),
                                (lane_slack_idx(i_loc, ci, 1), 1.0),
                            ],
                            lower=-float(np.asarray(b_R_ic).squeeze()),
                            upper=PROXQP_INF,
                        )

            pair_sat_pairs = []
            for i_loc in range(n_local - 1):
                gid_i = members[i_loc]
                ui_slice = slice(
                    u_slice.start + 2 * i_loc, u_slice.start + 2 * i_loc + 2
                )
                kins_i = kins_local[i_loc]
                for j_loc in range(i_loc + 1, n_local):
                    gid_j = members[j_loc]
                    if on_hold_set and (gid_i in on_hold_set or gid_j in on_hold_set):
                        continue
                    uj_slice = slice(
                        u_slice.start + 2 * j_loc, u_slice.start + 2 * j_loc + 2
                    )
                    kins_j = kins_local[j_loc]
                    lam_pair = (
                        self._get_agent_cbf_lambda(learned_lambdas, gid_i)
                        if self.parameters.is_learn_cbf_lambda
                        else float(self.lambda_ttcbf_predefined)
                    )
                    pair_offset = pair_base[(i_loc, j_loc)]
                    if is_pair_sat:
                        pair_sat_pairs.append(
                            dict(
                                a=gid_i,
                                b=gid_j,
                                slices=(ui_slice, uj_slice),
                                share=None,
                                lam=lam_pair,
                                slack_idx=(
                                    None
                                    if s_pair_slice is None
                                    else s_pair_slice.start + pair_offset
                                ),
                                lambda_idx=(
                                    None
                                    if lambda_pair_slice is None
                                    else lambda_pair_slice.start + pair_offset
                                ),
                            )
                        )
                        continue
                    if is_use_mtv_for_veh_veh:
                        relative_pose = self._relative_pose_in_ego_frame(
                            states[gid_i], states[gid_j]
                        )
                        if self._is_mtv_pose_in_domain(relative_pose):
                            A_i, A_j, b0, h = self.ttcbf_mtv_affine_coeffs_pair(
                                states[gid_i], states[gid_j], self.dt_taylor
                            )
                        else:
                            kins_i_center = (
                                self.linearized_vehicle_center_kinematics_coeffs(
                                    states[gid_i]
                                )
                            )
                            kins_j_center = (
                                self.linearized_vehicle_center_kinematics_coeffs(
                                    states[gid_j]
                                )
                            )
                            dx_center = float(
                                states[gid_i][0].item() - states[gid_j][0].item()
                            )
                            dy_center = float(
                                states[gid_i][1].item() - states[gid_j][1].item()
                            )
                            A_i, A_j, b0, h = self.ttcbf_pair_affine_coeffs(
                                kins_i_center,
                                kins_j_center,
                                0,
                                0,
                                dx_center,
                                dy_center,
                                body_circle_d_safe_sq,
                                self.dt_taylor,
                                None,
                            )
                        slack_terms = (
                            []
                            if s_pair_slice is None
                            else [(s_pair_slice.start + pair_offset, 1.0)]
                        )
                        terms = [
                            (ui_slice, A_i.reshape(-1)),
                            (uj_slice, A_j.reshape(-1)),
                            *slack_terms,
                        ]
                        if lambda_pair_slice is not None:
                            terms.append(
                                (
                                    lambda_pair_slice.start + pair_offset,
                                    float(np.asarray(h).squeeze()),
                                )
                            )
                            lower = -float(np.asarray(b0).squeeze())
                        else:
                            lower = -float(np.asarray(b0 + lam_pair * h).squeeze())
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            terms,
                            lower=lower,
                            upper=PROXQP_INF,
                        )
                        continue

                    if is_aggregate_veh_veh_safety:
                        pair_terms = []
                        for ci in range(C):
                            pi = centers[gid_i][ci][0:2]
                            for cj in range(C):
                                pj = centers[gid_j][cj][0:2]
                                dx = float((pi - pj)[0].item())
                                dy = float((pi - pj)[1].item())
                                pair_terms.append(
                                    self.ttcbf_pair_affine_components(
                                        kins_i,
                                        kins_j,
                                        ci,
                                        cj,
                                        dx,
                                        dy,
                                        d_safe_sq,
                                    )
                                )
                        A_i, A_j, b0, h = self.aggregate_softmin_taylor_pair_cbf_terms(
                            pair_terms,
                            self.dt_taylor,
                            aggregate_safety_softmin_beta,
                        )
                        slack_terms = (
                            []
                            if s_pair_slice is None
                            else [(s_pair_slice.start + pair_offset, 1.0)]
                        )
                        terms = [
                            (ui_slice, A_i.reshape(-1)),
                            (uj_slice, A_j.reshape(-1)),
                            *slack_terms,
                        ]
                        if lambda_pair_slice is not None:
                            terms.append(
                                (
                                    lambda_pair_slice.start + pair_offset,
                                    float(np.asarray(h).squeeze()),
                                )
                            )
                            lower = -float(np.asarray(b0).squeeze())
                        else:
                            lower = -float(np.asarray(b0 + lam_pair * h).squeeze())
                        self._append_qp_row(
                            rows,
                            lowers,
                            uppers,
                            n_vars,
                            terms,
                            lower=lower,
                            upper=PROXQP_INF,
                        )
                        continue

                    for ci in range(C):
                        pi = centers[gid_i][ci][0:2]
                        for cj in range(C):
                            pj = centers[gid_j][cj][0:2]
                            dx = float((pi - pj)[0].item())
                            dy = float((pi - pj)[1].item())
                            pair_offset = pair_base[(i_loc, j_loc)] + ci * C + cj
                            slack_terms = (
                                []
                                if s_pair_slice is None
                                else [(s_pair_slice.start + pair_offset, 1.0)]
                            )
                            if lambda_pair_slice is not None:
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
                                self._append_qp_row(
                                    rows,
                                    lowers,
                                    uppers,
                                    n_vars,
                                    [
                                        (ui_slice, A_i.reshape(-1)),
                                        (uj_slice, A_j.reshape(-1)),
                                        *slack_terms,
                                        (
                                            lambda_pair_slice.start + pair_offset,
                                            float(np.asarray(h).squeeze()),
                                        ),
                                    ],
                                    lower=-float(np.asarray(b0).squeeze()),
                                    upper=PROXQP_INF,
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
                                self._append_qp_row(
                                    rows,
                                    lowers,
                                    uppers,
                                    n_vars,
                                    [
                                        (ui_slice, A_i.reshape(-1)),
                                        (uj_slice, A_j.reshape(-1)),
                                        *slack_terms,
                                    ],
                                    lower=-float(np.asarray(b).squeeze()),
                                    upper=PROXQP_INF,
                                )

            for i_loc, gid_i in enumerate(members):
                ui_slice = slice(
                    u_slice.start + 2 * i_loc, u_slice.start + 2 * i_loc + 2
                )
                kins_i = kins_local[i_loc]
                lam_cross = (
                    self._get_agent_cbf_lambda(learned_lambdas, gid_i)
                    if self.parameters.is_learn_cbf_lambda
                    else float(self.lambda_ttcbf_predefined)
                )
                for e_idx, gid_j in enumerate(neighbor_lists[i_loc]):
                    cross_scale = self._cross_cbf_responsibility_share(
                        states[gid_i], states[gid_j]
                    )
                    kins_j = self.linearized_center_kinematics_coeffs(states[gid_j])
                    cross_offset_base = cross_base[(i_loc, e_idx)]
                    if is_pair_sat:
                        pair_sat_pairs.append(
                            self._pair_sat_cross_pair(
                                gid_i,
                                gid_j,
                                ui_slice,
                                cross_scale,
                                lam_cross,
                                (
                                    None
                                    if s_cross_slice is None
                                    else s_cross_slice.start + cross_offset_base
                                ),
                                (
                                    None
                                    if lambda_cross_slice is None
                                    else lambda_cross_slice.start + cross_offset_base
                                ),
                            )
                        )
                        continue
                    if is_use_mtv_for_veh_veh:
                        relative_pose = self._relative_pose_in_ego_frame(
                            states[gid_i], states[gid_j]
                        )
                        if self._is_mtv_pose_in_domain(relative_pose):
                            A_i, b0, h = self.ttcbf_mtv_affine_coeffs_cross(
                                states[gid_i], states[gid_j], self.dt_taylor
                            )
                        else:
                            kins_i_center = (
                                self.linearized_vehicle_center_kinematics_coeffs(
                                    states[gid_i]
                                )
                            )
                            kins_j_center = (
                                self.linearized_vehicle_center_kinematics_coeffs(
                                    states[gid_j]
                                )
                            )
                            dx_center = float(
                                states[gid_i][0].item() - states[gid_j][0].item()
                            )
                            dy_center = float(
                                states[gid_i][1].item() - states[gid_j][1].item()
                            )
                            A_i, b0, h = self.ttcbf_pair_affine_coeffs_cross(
                                kins_i_center,
                                kins_j_center,
                                0,
                                0,
                                dx_center,
                                dy_center,
                                body_circle_d_safe_sq,
                                self.dt_taylor,
                                None,
                            )
                        slack_terms = (
                            []
                            if s_cross_slice is None
                            else [(s_cross_slice.start + cross_offset_base, 1.0)]
                        )
                        if lambda_cross_slice is not None:
                            self._append_qp_row(
                                rows,
                                lowers,
                                uppers,
                                n_vars,
                                [
                                    (ui_slice, A_i.reshape(-1)),
                                    *slack_terms,
                                    (
                                        lambda_cross_slice.start + cross_offset_base,
                                        cross_scale * float(np.asarray(h).squeeze()),
                                    ),
                                ],
                                lower=-cross_scale * float(np.asarray(b0).squeeze()),
                                upper=PROXQP_INF,
                            )
                        else:
                            b = cross_scale * (b0 + lam_cross * h)
                            self._append_qp_row(
                                rows,
                                lowers,
                                uppers,
                                n_vars,
                                [(ui_slice, A_i.reshape(-1)), *slack_terms],
                                lower=-float(np.asarray(b).squeeze()),
                                upper=PROXQP_INF,
                            )
                        continue

                    if is_aggregate_veh_veh_safety:
                        pair_terms = []
                        for ci in range(C):
                            pi = centers[gid_i][ci][0:2]
                            for cj in range(C):
                                pj = centers[gid_j][cj][0:2]
                                dx = float((pi - pj)[0].item())
                                dy = float((pi - pj)[1].item())
                                pair_terms.append(
                                    self.ttcbf_pair_cross_affine_components(
                                        kins_i,
                                        kins_j,
                                        ci,
                                        cj,
                                        dx,
                                        dy,
                                        d_safe_sq,
                                    )
                                )
                        A_i, b0, h = self.aggregate_softmin_taylor_cbf_terms(
                            pair_terms,
                            self.dt_taylor,
                            aggregate_safety_softmin_beta,
                        )
                        slack_terms = (
                            []
                            if s_cross_slice is None
                            else [(s_cross_slice.start + cross_offset_base, 1.0)]
                        )
                        if lambda_cross_slice is not None:
                            self._append_qp_row(
                                rows,
                                lowers,
                                uppers,
                                n_vars,
                                [
                                    (ui_slice, A_i.reshape(-1)),
                                    *slack_terms,
                                    (
                                        lambda_cross_slice.start + cross_offset_base,
                                        cross_scale * float(np.asarray(h).squeeze()),
                                    ),
                                ],
                                lower=-cross_scale * float(np.asarray(b0).squeeze()),
                                upper=PROXQP_INF,
                            )
                        else:
                            b = cross_scale * (b0 + lam_cross * h)
                            self._append_qp_row(
                                rows,
                                lowers,
                                uppers,
                                n_vars,
                                [(ui_slice, A_i.reshape(-1)), *slack_terms],
                                lower=-float(np.asarray(b).squeeze()),
                                upper=PROXQP_INF,
                            )
                        continue

                    for ci in range(C):
                        pi = centers[gid_i][ci][0:2]
                        for cj in range(C):
                            pj = centers[gid_j][cj][0:2]
                            dx = float((pi - pj)[0].item())
                            dy = float((pi - pj)[1].item())
                            cross_offset = cross_offset_base + ci * C + cj
                            slack_terms = (
                                []
                                if s_cross_slice is None
                                else [(s_cross_slice.start + cross_offset, 1.0)]
                            )
                            if lambda_cross_slice is not None:
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
                                self._append_qp_row(
                                    rows,
                                    lowers,
                                    uppers,
                                    n_vars,
                                    [
                                        (ui_slice, A_i.reshape(-1)),
                                        *slack_terms,
                                        (
                                            lambda_cross_slice.start + cross_offset,
                                            cross_scale
                                            * float(np.asarray(h).squeeze()),
                                        ),
                                    ],
                                    lower=-cross_scale
                                    * float(np.asarray(b0).squeeze()),
                                    upper=PROXQP_INF,
                                )
                            else:
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
                                b = cross_scale * (b0 + lam_cross * h)
                                self._append_qp_row(
                                    rows,
                                    lowers,
                                    uppers,
                                    n_vars,
                                    [(ui_slice, A_i.reshape(-1)), *slack_terms],
                                    lower=-float(np.asarray(b).squeeze()),
                                    upper=PROXQP_INF,
                                )

            if pair_sat_pairs:
                group_agents = set(members)
                for nbrs in neighbor_lists:
                    group_agents.update(nbrs)
                sat_blocks.append(
                    self._append_pair_sat_rows(
                        pair_sat_pairs,
                        {
                            k: np.asarray([float(value) for value in states[k][:5]])
                            for k in group_agents
                        },
                        {
                            gid: np.clip(
                                U_nom_all[gid, :],
                                l_box[
                                    u_slice.start
                                    + 2 * i_loc : u_slice.start
                                    + 2 * i_loc
                                    + 2
                                ],
                                u_box[
                                    u_slice.start
                                    + 2 * i_loc : u_slice.start
                                    + 2 * i_loc
                                    + 2
                                ],
                            )
                            for i_loc, gid in enumerate(members)
                        },
                        set(),
                        rows,
                        lowers,
                        uppers,
                        n_vars,
                    )
                )

            if self.lane_cbf_footprint == "rectangle_sat":
                result = self._solve_qp_with_sat_repairs(
                    H, g, rows, lowers, uppers, n_vars, l_box, u_box, sat_blocks
                )
            else:
                C_mat, l_vec, u_vec = self._finalize_qp_rows(
                    rows, lowers, uppers, n_vars
                )
                result = self._solve_proxqp_problem(
                    H, g, C_mat, l_vec, u_vec, l_box, u_box
                )
            self._group_qp_caches[g_idx] = dict(
                members=members,
                cross_neighbors={gid: cross_groups.get(gid, []) for gid in members},
                last_result=result,
            )
            self._record_slack_activity(
                result, [s_lane_slice, s_pair_slice, s_cross_slice]
            )

            if result["solved"]:
                x_opt = result["x"]
                if self.is_debug_slack_var:
                    if not is_aggregate_lane_veh_safety:
                        self._debug_print_lane_slacks(
                            x_opt,
                            s_lane_slice,
                            members,
                            1 if is_lane_sat else C,
                            path_ids,
                            mode="grouped",
                            group_idx=g_idx,
                        )
                    if not (
                        is_use_mtv_for_veh_veh
                        or is_pair_sat
                        or is_aggregate_veh_veh_safety
                    ):
                        self._debug_print_pair_slacks(
                            x_opt,
                            s_pair_slice,
                            members,
                            pair_base,
                            C,
                            mode="grouped",
                            group_idx=g_idx,
                        )
                        self._debug_print_cross_slacks(
                            x_opt,
                            s_cross_slice,
                            members,
                            neighbor_lists,
                            cross_base,
                            C,
                            mode="grouped",
                            group_idx=g_idx,
                        )
                for i_loc, gid in enumerate(members):
                    if on_hold_set and gid in on_hold_set:
                        zero_u = np.zeros(2, dtype=np.float64)
                        action_zero = self.u_to_rl_action(
                            zero_u, states[gid][3], states[gid][4], agent_idx=gid
                        ).detach()
                        tensordict[("agents", "action")][
                            self.env_idx, gid
                        ] = action_zero
                        continue
                    ui = x_opt[
                        u_slice.start + 2 * i_loc : u_slice.start + 2 * i_loc + 2
                    ]
                    action_safe = self.u_to_rl_action(
                        ui, states[gid][3], states[gid][4], agent_idx=gid
                    ).detach()
                    tensordict[("agents", "action")][self.env_idx, gid] = action_safe
            else:
                if result["error"] is not None:
                    print(
                        "[WARN] Grouped proxsuite solve failed. "
                        f"group={g_idx}, members={members}, status={result['status']}"
                    )
                for gid in members:
                    if on_hold_set and gid in on_hold_set:
                        zero_u = np.zeros(2, dtype=np.float64)
                        action_zero = self.u_to_rl_action(
                            zero_u, states[gid][3], states[gid][4], agent_idx=gid
                        ).detach()
                        tensordict[("agents", "action")][
                            self.env_idx, gid
                        ] = action_zero
                        continue
                    u_fallback = self._failure_fallback_control(
                        states[gid], gid, U_nom_all[gid, :]
                    )
                    action_nom = self.u_to_rl_action(
                        u_fallback, states[gid][3], states[gid][4], agent_idx=gid
                    ).detach()
                    tensordict[("agents", "action")][self.env_idx, gid] = action_nom
                self.hist.evt.append("QP-INF")

            qp_solving_t_groups.append(result["solve_time"])
            qp_solving_iter_groups.append(result["iter"])

        self.scenario.inter_groups = groups
        self.scenario.cross_groups = cross_groups

        if self.env.scenario_name.parameters.is_apply_cbf_action:
            u_apply_all = np.zeros((N, 2), dtype=np.float64)
            for i in range(N):
                u_apply_all[i, :], _ = self._compute_applied_control_for_agent(
                    tensordict, states[i], i
                )
                if self.parameters.is_standstill_escape:
                    u_apply_all[i, :] = self._standstill_escape_control(
                        i, states[i], u_apply_all[i, :]
                    )
            # Grouped QPs write per-group actions independently above. The
            # shield checks the complete joint action once afterward, so
            # cross-group interactions are covered as well.
            if self.parameters.is_exact_backup_shield:
                u_apply_all, _ = self._apply_exact_backup_shield(
                    states, u_apply_all, on_hold_set
                )
            if on_hold_set:
                for held_idx in on_hold_set:
                    u_apply_all[held_idx, :] = 0.0
            self._write_low_level_controls_to_tensordict(
                tensordict, states, u_apply_all
            )

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
        self.hist.qp_solving_iter.append(qp_solving_iter_groups)

        if self.env.scenario_name.parameters.is_apply_cbf_action:
            self.env.base_env.scenario_name.world_state.nominal_action_vel[
                self.env_idx, :
            ] = nominal_target_actions_all[:, 0]
            self.env.base_env.scenario_name.world_state.nominal_action_steer[
                self.env_idx, :
            ] = nominal_target_actions_all[:, 1]
        else:
            compare_target_actions_all = applied_target_actions_all
            self.env.base_env.scenario_name.world_state.nominal_action_vel[
                self.env_idx, :
            ] = compare_target_actions_all[:, 0]
            self.env.base_env.scenario_name.world_state.nominal_action_steer[
                self.env_idx, :
            ] = compare_target_actions_all[:, 1]

    def _get_mtv_safety_margin_estimator(self) -> SafetyMarginEstimatorModule:
        """Load and cache the learned MTV safety-margin estimator."""
        if not hasattr(self, "_mtv_safety_margin_estimator"):
            path_nn = getattr(
                self.parameters,
                "mtv_sm_predictor_path",
                "checkpoints/mtv_distance_approximate/sm_predictor.pth",
            )
            estimator = SafetyMarginEstimatorModule(
                length=self.length,
                width=self.width,
                path_nn=path_nn,
            )
            estimator.load_model()
            self._mtv_safety_margin_estimator = estimator
        return self._mtv_safety_margin_estimator

    def _relative_pose_in_ego_frame(
        self, state_i: torch.Tensor, state_j: torch.Tensor
    ) -> np.ndarray:
        """Return vehicle j pose in vehicle i's ego frame as [x, y, heading]."""
        xi, yi, psi_i = (
            float(state_i[0].item()),
            float(state_i[1].item()),
            float(state_i[2].item()),
        )
        xj, yj, psi_j = (
            float(state_j[0].item()),
            float(state_j[1].item()),
            float(state_j[2].item()),
        )
        dx = xj - xi
        dy = yj - yi
        cos_i = math.cos(psi_i)
        sin_i = math.sin(psi_i)
        return np.array(
            [
                cos_i * dx + sin_i * dy,
                -sin_i * dx + cos_i * dy,
                self._wrap_angle(psi_j - psi_i),
            ],
            dtype=np.float64,
        )

    def _is_mtv_pose_in_domain(self, relative_pose: np.ndarray) -> bool:
        """Return whether a relative pose lies inside the learned MTV domain."""
        estimator = self._get_mtv_safety_margin_estimator()
        x_rel = float(relative_pose[0])
        y_rel = float(relative_pose[1])
        return (
            estimator.x_min < x_rel < estimator.x_max
            and estimator.y_min < y_rel < estimator.y_max
        )

    def _estimate_mtv_safety_margin_with_derivatives(
        self, relative_pose: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate learned MTV margin and its first/second derivatives."""
        estimator = self._get_mtv_safety_margin_estimator()
        feature_norm = estimator.feature_normalizer.squeeze(0)
        label_norm = float(estimator.label_normalizer)

        pose = np.asarray(relative_pose, dtype=np.float64).reshape(3).copy()
        pose[2] = self._wrap_angle(float(pose[2]))
        inputs = torch.tensor(pose, dtype=torch.float32)
        normalized_inputs = inputs / feature_norm
        normalized_inputs.requires_grad_(True)

        sm_normalized = estimator.net(normalized_inputs)
        sm_predicted = sm_normalized * label_norm
        grad_normalized = torch.autograd.grad(
            sm_normalized,
            normalized_inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_predicted = (grad_normalized * label_norm) / feature_norm
        hessian_rows = []
        for row_idx in range(normalized_inputs.size(0)):
            grad2 = torch.autograd.grad(
                grad_normalized[row_idx],
                normalized_inputs,
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
            )[0]
            hessian_rows.append(
                grad2 * label_norm / (feature_norm[row_idx] * feature_norm)
            )
        hessian_predicted = torch.stack(hessian_rows)

        return (
            float(sm_predicted.detach().item()),
            grad_predicted.detach().cpu().numpy().astype(np.float64, copy=False),
            hessian_predicted.detach().cpu().numpy().astype(np.float64, copy=False),
        )

    def linearized_vehicle_center_kinematics_coeffs(self, state_agent):
        """Return center-point kinematics coefficients in the circle helper format."""
        dstate_time_agent = self.kbm.ode(None, state_agent, self.u_placeholder)
        u0 = np.array([0.0, 0.0], dtype=np.float64)
        e1 = np.array([1.0, 0.0], dtype=np.float64)
        e2 = np.array([0.0, 1.0], dtype=np.float64)

        ddx0, ddy0, _ = self.compute_dstate_2nd_time(u0, state_agent, dstate_time_agent)
        ddx_e1, ddy_e1, _ = self.compute_dstate_2nd_time(
            e1, state_agent, dstate_time_agent
        )
        ddx_e2, ddy_e2, _ = self.compute_dstate_2nd_time(
            e2, state_agent, dstate_time_agent
        )

        return dict(
            dx=[float(dstate_time_agent[0].item())],
            dy=[float(dstate_time_agent[1].item())],
            a_ddx=[np.array([ddx_e1 - ddx0, ddx_e2 - ddx0], dtype=np.float64)],
            c_ddx=[float(ddx0)],
            a_ddy=[np.array([ddy_e1 - ddy0, ddy_e2 - ddy0], dtype=np.float64)],
            c_ddy=[float(ddy0)],
        )

    def _relative_ego_dynamics_affine(
        self, state_i: torch.Tensor, state_j: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return relative ego-frame velocity and acceleration affine in ego control.

        The returned tuple is ``(dot_z, A_ddot_z, c_ddot_z)`` for
        ``z = [x_ji, y_ji, psi_ji]`` and ``ddot_z = A_ddot_z @ u_i + c_ddot_z``.
        """
        dstate_i = self.kbm.ode(None, state_i, self.u_placeholder)
        dstate_j = self.kbm.ode(None, state_j, self.u_placeholder)

        xi = float(state_i[0].item())
        yi = float(state_i[1].item())
        psi_i = float(state_i[2].item())
        xj = float(state_j[0].item())
        yj = float(state_j[1].item())
        x_ji_global = xj - xi
        y_ji_global = yj - yi

        dx_ji_global = float(dstate_j[0].item() - dstate_i[0].item())
        dy_ji_global = float(dstate_j[1].item() - dstate_i[1].item())
        dpsi_ji_global = float(dstate_j[2].item() - dstate_i[2].item())
        dpsi_i = float(dstate_i[2].item())
        cos_i = math.cos(psi_i)
        sin_i = math.sin(psi_i)

        dot_z = np.array(
            [
                cos_i * dx_ji_global
                - x_ji_global * sin_i * dpsi_i
                + sin_i * dy_ji_global
                + y_ji_global * cos_i * dpsi_i,
                cos_i * dy_ji_global
                - y_ji_global * sin_i * dpsi_i
                - sin_i * dx_ji_global
                - x_ji_global * cos_i * dpsi_i,
                dpsi_ji_global,
            ],
            dtype=np.float64,
        )

        ddx_j0, ddy_j0, ddpsi_j0 = self.compute_dstate_2nd_time(
            np.array([0.0, 0.0], dtype=np.float64), state_j, dstate_j
        )

        def ddot_for_ego_control(u_i: np.ndarray) -> np.ndarray:
            """Evaluate relative ego-frame second derivative for one ego input."""
            ddx_i, ddy_i, ddpsi_i = self.compute_dstate_2nd_time(u_i, state_i, dstate_i)
            ddx_ji_global = ddx_j0 - ddx_i
            ddy_ji_global = ddy_j0 - ddy_i
            ddpsi_ji_global = ddpsi_j0 - ddpsi_i
            return np.array(
                [
                    cos_i * ddx_ji_global
                    - 2.0 * dx_ji_global * sin_i * dpsi_i
                    - x_ji_global * cos_i * dpsi_i**2
                    - x_ji_global * sin_i * ddpsi_i
                    + sin_i * ddy_ji_global
                    + 2.0 * dy_ji_global * cos_i * dpsi_i
                    - y_ji_global * sin_i * dpsi_i**2
                    + y_ji_global * cos_i * ddpsi_i,
                    cos_i * ddy_ji_global
                    - 2.0 * dy_ji_global * sin_i * dpsi_i
                    - y_ji_global * cos_i * dpsi_i**2
                    - y_ji_global * sin_i * ddpsi_i
                    - sin_i * ddx_ji_global
                    - 2.0 * dx_ji_global * cos_i * dpsi_i
                    + x_ji_global * sin_i * dpsi_i**2
                    - x_ji_global * cos_i * ddpsi_i,
                    ddpsi_ji_global,
                ],
                dtype=np.float64,
            )

        ddot_0 = ddot_for_ego_control(np.array([0.0, 0.0], dtype=np.float64))
        ddot_e1 = ddot_for_ego_control(np.array([1.0, 0.0], dtype=np.float64))
        ddot_e2 = ddot_for_ego_control(np.array([0.0, 1.0], dtype=np.float64))
        A_ddot = np.column_stack((ddot_e1 - ddot_0, ddot_e2 - ddot_0))
        return dot_z, A_ddot, ddot_0

    def _relative_ego_dynamics_affine_pair(
        self, state_i: torch.Tensor, state_j: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return relative ego-frame dynamics affine in both vehicle controls.

        The returned tuple is ``(dot_z, A_i, A_j, c)`` for
        ``z = [x_ji, y_ji, psi_ji]`` and
        ``ddot_z = A_i @ u_i + A_j @ u_j + c``.
        """
        dstate_i = self.kbm.ode(None, state_i, self.u_placeholder)
        dstate_j = self.kbm.ode(None, state_j, self.u_placeholder)

        xi = float(state_i[0].item())
        yi = float(state_i[1].item())
        psi_i = float(state_i[2].item())
        xj = float(state_j[0].item())
        yj = float(state_j[1].item())
        x_ji_global = xj - xi
        y_ji_global = yj - yi

        dx_ji_global = float(dstate_j[0].item() - dstate_i[0].item())
        dy_ji_global = float(dstate_j[1].item() - dstate_i[1].item())
        dpsi_ji_global = float(dstate_j[2].item() - dstate_i[2].item())
        dpsi_i = float(dstate_i[2].item())
        cos_i = math.cos(psi_i)
        sin_i = math.sin(psi_i)

        dot_z = np.array(
            [
                cos_i * dx_ji_global
                - x_ji_global * sin_i * dpsi_i
                + sin_i * dy_ji_global
                + y_ji_global * cos_i * dpsi_i,
                cos_i * dy_ji_global
                - y_ji_global * sin_i * dpsi_i
                - sin_i * dx_ji_global
                - x_ji_global * cos_i * dpsi_i,
                dpsi_ji_global,
            ],
            dtype=np.float64,
        )

        def state_second_derivative(state, dstate, u):
            return self.compute_dstate_2nd_time(
                np.asarray(u, dtype=np.float64), state, dstate
            )

        def ddot_for_controls(u_i: np.ndarray, u_j: np.ndarray) -> np.ndarray:
            ddx_i, ddy_i, ddpsi_i = state_second_derivative(state_i, dstate_i, u_i)
            ddx_j, ddy_j, ddpsi_j = state_second_derivative(state_j, dstate_j, u_j)
            ddx_ji_global = ddx_j - ddx_i
            ddy_ji_global = ddy_j - ddy_i
            ddpsi_ji_global = ddpsi_j - ddpsi_i
            return np.array(
                [
                    cos_i * ddx_ji_global
                    - 2.0 * dx_ji_global * sin_i * dpsi_i
                    - x_ji_global * cos_i * dpsi_i**2
                    - x_ji_global * sin_i * ddpsi_i
                    + sin_i * ddy_ji_global
                    + 2.0 * dy_ji_global * cos_i * dpsi_i
                    - y_ji_global * sin_i * dpsi_i**2
                    + y_ji_global * cos_i * ddpsi_i,
                    cos_i * ddy_ji_global
                    - 2.0 * dy_ji_global * sin_i * dpsi_i
                    - y_ji_global * cos_i * dpsi_i**2
                    - y_ji_global * sin_i * ddpsi_i
                    - sin_i * ddx_ji_global
                    - 2.0 * dx_ji_global * cos_i * dpsi_i
                    + x_ji_global * sin_i * dpsi_i**2
                    - x_ji_global * cos_i * ddpsi_i,
                    ddpsi_ji_global,
                ],
                dtype=np.float64,
            )

        u0 = np.array([0.0, 0.0], dtype=np.float64)
        e1 = np.array([1.0, 0.0], dtype=np.float64)
        e2 = np.array([0.0, 1.0], dtype=np.float64)
        ddot_0 = ddot_for_controls(u0, u0)
        A_i = np.column_stack(
            (ddot_for_controls(e1, u0) - ddot_0, ddot_for_controls(e2, u0) - ddot_0)
        )
        A_j = np.column_stack(
            (ddot_for_controls(u0, e1) - ddot_0, ddot_for_controls(u0, e2) - ddot_0)
        )
        return dot_z, A_i, A_j, ddot_0

    def ttcbf_mtv_affine_coeffs_cross(
        self, state_i: torch.Tensor, state_j: torch.Tensor, dt: float
    ):
        """Return ego-control affine TTCBF coefficients from the MTV model."""
        relative_pose = self._relative_pose_in_ego_frame(state_i, state_j)
        sm, grad, hess = self._estimate_mtv_safety_margin_with_derivatives(
            relative_pose
        )
        estimator = self._get_mtv_safety_margin_estimator()
        h = float(sm - float(estimator.error_upper_bound) - self.safety_buffer)
        dot_z, A_ddot_z, c_ddot_z = self._relative_ego_dynamics_affine(state_i, state_j)

        grad = np.asarray(grad, dtype=np.float64).reshape(3)
        hess = np.asarray(hess, dtype=np.float64).reshape(3, 3)
        A = 0.5 * (dt * dt) * (grad @ A_ddot_z)
        dot_h = float(grad @ dot_z)
        const_dd = float(grad @ c_ddot_z + dot_z @ hess @ dot_z)
        b0 = dot_h * dt + 0.5 * (dt * dt) * const_dd
        return (
            A.reshape(1, 2),
            np.array([b0], dtype=np.float64),
            np.array([h], dtype=np.float64),
        )

    def ttcbf_mtv_affine_coeffs_pair(
        self, state_i: torch.Tensor, state_j: torch.Tensor, dt: float
    ):
        """Return two-control affine TTCBF coefficients from the MTV model."""
        relative_pose = self._relative_pose_in_ego_frame(state_i, state_j)
        sm, grad, hess = self._estimate_mtv_safety_margin_with_derivatives(
            relative_pose
        )
        estimator = self._get_mtv_safety_margin_estimator()
        h = float(sm - float(estimator.error_upper_bound) - self.safety_buffer)
        dot_z, A_ddot_i, A_ddot_j, c_ddot_z = self._relative_ego_dynamics_affine_pair(
            state_i, state_j
        )

        grad = np.asarray(grad, dtype=np.float64).reshape(3)
        hess = np.asarray(hess, dtype=np.float64).reshape(3, 3)
        A_i = 0.5 * (dt * dt) * (grad @ A_ddot_i)
        A_j = 0.5 * (dt * dt) * (grad @ A_ddot_j)
        dot_h = float(grad @ dot_z)
        const_dd = float(grad @ c_ddot_z + dot_z @ hess @ dot_z)
        b0 = dot_h * dt + 0.5 * (dt * dt) * const_dd
        return (
            A_i.reshape(1, 2),
            A_j.reshape(1, 2),
            np.array([b0], dtype=np.float64),
            np.array([h], dtype=np.float64),
        )

    @staticmethod
    def _stable_softmin_weights(values, beta: float):
        """Return the unnormalized softmin value and normalized softmin weights."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        beta = float(beta)
        if beta <= 0.0:
            raise ValueError("Softmin beta must be positive.")
        value_min = float(np.min(values))
        exp_values = np.exp(-beta * (values - value_min))
        weight_sum = float(np.sum(exp_values))
        weights = exp_values / weight_sum
        softmin_value = value_min - np.log(weight_sum) / beta
        return softmin_value, weights

    def aggregate_softmin_taylor_cbf_terms(self, terms, dt: float, beta: float):
        """
        Aggregate per-footprint CBF Taylor terms with a conservative softmin.

        Each term is a dictionary containing the barrier value ``h``, its current
        time derivative ``dot_h``, its control-free second-derivative part
        ``const_dd``, and its control-affine second-derivative coefficient ``A``.
        """
        if len(terms) == 0:
            raise ValueError("At least one CBF term is required for aggregation.")

        h_values = np.asarray([term["h"] for term in terms], dtype=np.float64)
        dot_values = np.asarray([term["dot_h"] for term in terms], dtype=np.float64)
        const_values = np.asarray(
            [term["const_dd"] for term in terms], dtype=np.float64
        )
        A_values = np.vstack(
            [np.asarray(term["A"], dtype=np.float64).reshape(1, 2) for term in terms]
        )

        h_softmin, weights = self._stable_softmin_weights(h_values, beta)
        dot_softmin = float(weights @ dot_values)
        const_softmin = float(weights @ const_values)
        dot_variance = float(weights @ (dot_values * dot_values) - dot_softmin**2)
        A_softmin = 0.5 * (dt * dt) * (weights @ A_values)
        drift = dot_softmin * dt + 0.5 * (dt * dt) * (
            const_softmin - float(beta) * dot_variance
        )

        return (
            A_softmin.reshape(1, 2),
            np.array([drift], dtype=np.float64),
            np.array([h_softmin], dtype=np.float64),
        )

    def _lane_barrier_taylor_terms(self, kins_i, ci, sm, grad, hess):
        """Return (h, dot_h, A, const_dd) of one lane barrier, ddot_h = A u + const_dd.

        Circle footprint: h = D - (r + buffer)^2, where D is the softmin squared
        distance of the circle center to the boundary.

        Rectangle footprint: certified barrier of the rectangle segment covered
        by the circle (see ``_lane_rect_geometry`` and ``_lane_rect_taylor_terms``).
        """
        g = np.asarray(grad, dtype=np.float64).reshape(2)
        H = np.asarray(hess, dtype=np.float64).reshape(2, 2)
        v = np.array([kins_i["dx"][ci], kins_i["dy"][ci]], dtype=np.float64)
        a_x = np.asarray(kins_i["a_ddx"][ci], dtype=np.float64).reshape(2)
        a_y = np.asarray(kins_i["a_ddy"][ci], dtype=np.float64).reshape(2)
        A_p = np.vstack([a_x, a_y])  # d^2 p / dt^2 = A_p u + c_p
        c_p = np.array([kins_i["c_ddx"][ci], kins_i["c_ddy"][ci]], dtype=np.float64)

        if self.lane_cbf_footprint == "rectangle":
            return self._lane_rect_taylor_terms(kins_i, sm, g, H, v, A_p, c_p)
        h = float(sm - self.lane_safety_buffer)
        dot_h = float(g @ v)
        const_dd = float(g @ c_p + v @ H @ v)
        return h, dot_h, g @ A_p, const_dd

    def _lane_rect_geometry(self, sm, g, H, psi: float):
        """Local boundary geometry and barrier value of the rectangle footprint.

        Uses the signed distance d = sqrt(D) of the slice center, the unit
        normal n, and the tangential Hessian w = t^T (grad^2 d) t, which equals
        kappa / (1 + kappa d) for a boundary arc of signed curvature kappa
        (positive when the boundary bulges toward the vehicle).

        The barrier is h = d - S(alpha) - K(q) - buffer:
        - S = b sqrt(s^2 + eps^2) + a sqrt(c^2 + eps^2), with s = n.e_lon and
          c = n.e_lat, is the exact support function of the Minkowski sum of
          two ellipses that contains the slice, so S >= a|c| + b|s|.
        - K = rho^2 q / (2 (1 - q rho)), q = max(0, -w), rho = slice
          half-diagonal, bounds the drop of d over the slice where the
          boundary curves around the vehicle (d concave). Where d is convex,
          the linearization already under-estimates d and K = 0.
        Then h >= 0 implies that the whole slice keeps the buffer.
        """
        D = max(float(sm) + self.circle_radius**2, 1e-12)
        d = math.sqrt(D)
        n = g / max(float(np.linalg.norm(g)), 1e-12)
        t = np.array([-n[1], n[0]], dtype=np.float64)
        w = float(t @ H @ t) / (2.0 * d)
        a = self.lane_rect_half_width
        b = self.lane_rect_half_length
        eps = self.lane_rect_abs_eps
        s = math.cos(psi) * n[0] + math.sin(psi) * n[1]
        c = -math.sin(psi) * n[0] + math.cos(psi) * n[1]
        sig_s = math.sqrt(s * s + eps * eps)
        sig_c = math.sqrt(c * c + eps * eps)
        rho = math.hypot(a, b)
        # Keep 1 - q rho > 0 (only matters for boundary radii below 2 rho)
        q = min(max(-w, 0.0), 0.5 / rho)
        den = 1.0 - q * rho
        K = 0.5 * rho * rho * q / den
        h = d - (b * sig_s + a * sig_c) - K - self.safety_buffer
        return dict(
            h=h, n=n, t=t, w=w, s=s, c=c, sig_s=sig_s, sig_c=sig_c, q=q, den=den
        )

    def _lane_rect_taylor_terms(self, kins_i, sm, g, H, v, A_p, c_p):
        """Exact (h, dot_h, A, const_dd) of the rectangle barrier.

        The support term depends on the pose only through the relative angle
        alpha between the boundary normal and the heading, whose rates are
        dot_alpha = w v_t - dot_psi and
        ddot_alpha = w t.ddot_p - 2 w^2 v_n v_t - ddot_psi,
        where v_n = n.dot_p and v_t = t.dot_p. With ddot_d = n.ddot_p + w v_t^2
        and dot_w = -w^2 v_n, the second derivative of h is exact on boundary
        arcs and lines; only the curvature variation along the boundary is
        neglected. All heading terms, including drift and squared rates, are kept.
        """
        geo = self._lane_rect_geometry(sm, g, H, float(kins_i["psi"]))
        n, t, w = geo["n"], geo["t"], geo["w"]
        s, c, sig_s, sig_c = geo["s"], geo["c"], geo["sig_s"], geo["sig_c"]
        a = self.lane_rect_half_width
        b = self.lane_rect_half_length
        rho = math.hypot(a, b)
        a_psi = np.asarray(kins_i["a_ddpsi"], dtype=np.float64).reshape(2)
        c_psi = float(kins_i["c_ddpsi"])
        dpsi = float(kins_i["dpsi"])

        # Support S(alpha) and its first two derivatives
        S1 = s * c * (a / sig_c - b / sig_s)
        S2 = b * (-(s * s - c * c) / sig_s - (s * c) ** 2 / sig_s**3) + a * (
            (s * s - c * c) / sig_c - (s * c) ** 2 / sig_c**3
        )

        v_n = float(n @ v)
        v_t = float(t @ v)
        dot_alpha = w * v_t - dpsi
        A_alpha = w * (t @ A_p) - a_psi
        c_alpha = w * float(t @ c_p) - 2.0 * w * w * v_n * v_t - c_psi
        A_d = n @ A_p
        c_d = float(n @ c_p) + w * v_t * v_t

        # Curvature margin K(q) with q = -w on the concave side
        if geo["q"] > 0.0:
            K1 = 0.5 * rho * rho / geo["den"] ** 2
            K2 = rho**3 / geo["den"] ** 3
            dot_q = w * w * v_n
            A_q = w * w * A_d
            c_q = -2.0 * w**3 * v_n * v_n + w * w * c_d
        else:
            K1 = K2 = dot_q = 0.0
            A_q = np.zeros(2, dtype=np.float64)
            c_q = 0.0

        dot_h = v_n - S1 * dot_alpha - K1 * dot_q
        A = A_d - S1 * A_alpha - K1 * A_q
        const_dd = c_d - S2 * dot_alpha**2 - S1 * c_alpha - K2 * dot_q**2 - K1 * c_q
        return geo["h"], float(dot_h), A, float(const_dd)

    def ttcbf_lane_affine_components(self, kins_i, ci, sm, grad, hess):
        """Return control-affine Taylor components for one lane-footprint barrier."""
        h, dot_h, A, const_dd = self._lane_barrier_taylor_terms(
            kins_i, ci, sm, grad, hess
        )
        return dict(A=A, h=h, dot_h=dot_h, const_dd=const_dd)

    def ttcbf_pair_cross_affine_components(
        self, kins_i, kins_j, ci, cj, dx, dy, d_safe_sq
    ):
        """Return ego-control Taylor components for one cross-agent circle pair."""
        dx = float(dx)
        dy = float(dy)
        vxi = float(kins_i["dx"][ci])
        vyi = float(kins_i["dy"][ci])
        vxj = float(kins_j["dx"][cj])
        vyj = float(kins_j["dy"][cj])
        vrel_x = vxi - vxj
        vrel_y = vyi - vyj

        aix = np.asarray(kins_i["a_ddx"][ci], dtype=np.float64).reshape(2)
        aiy = np.asarray(kins_i["a_ddy"][ci], dtype=np.float64).reshape(2)
        cix = float(kins_i["c_ddx"][ci])
        ciy = float(kins_i["c_ddy"][ci])
        cjx = float(kins_j["c_ddx"][cj])
        cjy = float(kins_j["c_ddy"][cj])

        h = dx * dx + dy * dy - float(d_safe_sq)
        dot_h = 2.0 * (dx * vrel_x + dy * vrel_y)
        A = 2.0 * (dx * aix + dy * aiy)
        const_dd = 2.0 * (vrel_x * vrel_x + vrel_y * vrel_y) + 2.0 * (
            dx * (cix - cjx) + dy * (ciy - cjy)
        )
        return dict(A=A, h=h, dot_h=dot_h, const_dd=const_dd)

    def ttcbf_pair_affine_components(self, kins_i, kins_j, ci, cj, dx, dy, d_safe_sq):
        """Return two-control Taylor components for one in-QP circle pair."""
        dx = float(dx)
        dy = float(dy)
        vxi = float(kins_i["dx"][ci])
        vyi = float(kins_i["dy"][ci])
        vxj = float(kins_j["dx"][cj])
        vyj = float(kins_j["dy"][cj])
        vrel_x = vxi - vxj
        vrel_y = vyi - vyj

        aix = np.asarray(kins_i["a_ddx"][ci], dtype=np.float64).reshape(2)
        aiy = np.asarray(kins_i["a_ddy"][ci], dtype=np.float64).reshape(2)
        ajx = np.asarray(kins_j["a_ddx"][cj], dtype=np.float64).reshape(2)
        ajy = np.asarray(kins_j["a_ddy"][cj], dtype=np.float64).reshape(2)
        cix = float(kins_i["c_ddx"][ci])
        ciy = float(kins_i["c_ddy"][ci])
        cjx = float(kins_j["c_ddx"][cj])
        cjy = float(kins_j["c_ddy"][cj])

        h = dx * dx + dy * dy - float(d_safe_sq)
        dot_h = 2.0 * (dx * vrel_x + dy * vrel_y)
        A_i = 2.0 * (dx * aix + dy * aiy)
        A_j = -2.0 * (dx * ajx + dy * ajy)
        const_dd = 2.0 * (vrel_x * vrel_x + vrel_y * vrel_y) + 2.0 * (
            dx * (cix - cjx) + dy * (ciy - cjy)
        )
        return dict(A_i=A_i, A_j=A_j, h=h, dot_h=dot_h, const_dd=const_dd)

    def aggregate_softmin_taylor_pair_cbf_terms(self, terms, dt: float, beta: float):
        """Aggregate in-QP pair TTCBF terms and keep both control blocks."""
        if len(terms) == 0:
            raise ValueError("At least one CBF term is required for aggregation.")

        h_values = np.asarray([term["h"] for term in terms], dtype=np.float64)
        dot_values = np.asarray([term["dot_h"] for term in terms], dtype=np.float64)
        const_values = np.asarray(
            [term["const_dd"] for term in terms], dtype=np.float64
        )
        A_i_values = np.vstack(
            [np.asarray(term["A_i"], dtype=np.float64).reshape(1, 2) for term in terms]
        )
        A_j_values = np.vstack(
            [np.asarray(term["A_j"], dtype=np.float64).reshape(1, 2) for term in terms]
        )

        h_softmin, weights = self._stable_softmin_weights(h_values, beta)
        dot_softmin = float(weights @ dot_values)
        const_softmin = float(weights @ const_values)
        dot_variance = float(weights @ (dot_values * dot_values) - dot_softmin**2)
        A_i_softmin = 0.5 * (dt * dt) * (weights @ A_i_values)
        A_j_softmin = 0.5 * (dt * dt) * (weights @ A_j_values)
        drift = dot_softmin * dt + 0.5 * (dt * dt) * (
            const_softmin - float(beta) * dot_variance
        )

        return (
            A_i_softmin.reshape(1, 2),
            A_j_softmin.reshape(1, 2),
            np.array([drift], dtype=np.float64),
            np.array([h_softmin], dtype=np.float64),
        )

    def ttcbf_lane_affine_coeffs(self, kins_i, ci, sm, grad, hess, dt, lam):
        """
        Return the Taylor-based lane CBF row A u >= -b.

        With lam=None, return (A, b0, h) so lam * h can be added by the caller.
        """
        h, dot_h, A_dd, const_dd = self._lane_barrier_taylor_terms(
            kins_i, ci, sm, grad, hess
        )
        A = 0.5 * (dt * dt) * A_dd
        b0 = dot_h * dt + 0.5 * (dt * dt) * const_dd
        if lam is None:
            return (
                A.reshape(1, 2),
                np.array([b0], dtype=np.float64),
                np.array([h], dtype=np.float64),
            )
        b = lam * h + b0
        return A.reshape(1, 2), np.array([b], dtype=np.float64)

    # =========================
    # Exact rectangle-polyline lane barrier (lane_cbf_footprint="rectangle_sat")
    # =========================
    #
    # The barrier geometry (facets, SAT atoms, and their gradients) lives in
    # sigmarl/lane_sat_barrier.py. Its kinks are handled exactly as a nonsmooth
    # (min/max) barrier: every atom gets its own constraint, so no smooth
    # surrogate has to predict the slope change at a polyline joint.
    #
    # The constraints use the exact two-step explicit-Euler map of WorldCustom:
    # the pose y at k+1 does not depend on u_k, the pose z at k+2 does. Each
    # atom alpha must satisfy alpha(z) >= (1 - lambda) alpha(y). The QP uses
    # linearizations in u; the solution is then checked with the exact map and
    # violated atoms are re-linearized at the solution (lazy constraints), so an
    # accepted control satisfies every exact constraint.

    def _lane_sat_predict(self, state_np: np.ndarray, u):
        """Two-step explicit-Euler pose prediction and its input Jacobian.

        WorldCustom integrates x_{k+1} = x_k + dt f(x_k, u_k), so the pose y at
        k+1 does not depend on u_k, while speed and steering do. The pose z at
        k+2 then depends on u_k through v_{k+1} = v + dt a and
        delta_{k+1} = delta + dt omega.

        Returns:
            pose_y (3,), pose_z (3,), and d pose_z / d u (3, 2).
        """
        dt = self.dt
        k = self.l_r / self.l_wb
        x, y, psi, v, delta = (float(value) for value in state_np)

        beta = math.atan(k * math.tan(delta))
        course = psi + beta
        pose_y = np.array(
            [
                x + dt * v * math.cos(course),
                y + dt * v * math.sin(course),
                psi + dt * v * math.sin(beta) / self.l_r,
            ]
        )

        v1 = v + dt * float(u[0])
        delta1 = delta + dt * float(u[1])
        tan1 = math.tan(delta1)
        beta1 = math.atan(k * tan1)
        dbeta_domega = dt * k / (math.cos(delta1) ** 2 * (1.0 + (k * tan1) ** 2))
        course1 = pose_y[2] + beta1
        cos_c, sin_c = math.cos(course1), math.sin(course1)
        cos_b, sin_b = math.cos(beta1), math.sin(beta1)
        pose_z = np.array(
            [
                pose_y[0] + dt * v1 * cos_c,
                pose_y[1] + dt * v1 * sin_c,
                pose_y[2] + dt * v1 * sin_b / self.l_r,
            ]
        )
        jacobian = np.array(
            [
                [dt * dt * cos_c, -dt * v1 * sin_c * dbeta_domega],
                [dt * dt * sin_c, dt * v1 * cos_c * dbeta_domega],
                [dt * dt * sin_b / self.l_r, dt * v1 * cos_b / self.l_r * dbeta_domega],
            ]
        )
        return pose_y, pose_z, jacobian

    def _lane_sat_candidate_values(self, cand: dict, u) -> np.ndarray:
        """Exact atom values of all candidate atoms at the k+2 pose under u."""
        _, pose_z, _ = self._lane_sat_predict(cand["state"], u)
        values = np.empty(cand["h"].shape[0], dtype=np.float64)
        for side, facets in enumerate(cand["facets"]):
            tables = self.lane_sat.atom_tables(facets, side, pose_z)
            for axis in range(4):
                mask = (cand["side"] == side) & (cand["axis"] == axis)
                values[mask] = tables[axis][cand["facet"][mask], cand["atom"][mask]]
        return values

    def _lane_sat_selection_control(
        self, facets_all, side_values, state_np, u_lin, u_lo, u_hi
    ) -> np.ndarray:
        """Control at which the SAT axes of the lane rows are selected.

        Keeping one axis per facet makes the rows convex but drops the other
        axes, so axes selected at a control that steers toward the boundary can
        exclude every safe control and leave the QP only its slack. The clipped
        nominal u_lin is kept if it satisfies the exact constraint
        B(z) >= (1 - lambda) B(y) on both sides. Otherwise, the closest probe
        control of the input box that satisfies it is used, or the probe with
        the largest exact margin if none does. If the best axis of every facet
        at that control is admissible, the control satisfies the selected rows.

        Args:
            facets_all: buffered facets of the left and right boundary.
            side_values: B(y) of each side, None for a side without facets.
            u_lo, u_hi: input box of the agent.
        """
        decay = 1.0 - self.lambda_ttcbf_predefined

        def exact_margin(u):
            _, pose_z, _ = self._lane_sat_predict(state_np, u)
            return min(
                self.lane_sat.value(facets, side, pose_z) - decay * side_values[side]
                for side, facets in enumerate(facets_all)
                if side_values[side] is not None
            )

        threshold = self.lane_sat_linearization_margin
        if exact_margin(u_lin) >= threshold:
            return u_lin
        acc_probes = np.unique(
            np.clip([u_lo[0], 0.0, u_lin[0], u_hi[0]], u_lo[0], u_hi[0])
        )
        steer_probes = np.unique(np.append(np.linspace(u_lo[1], u_hi[1], 5), u_lin[1]))
        probes = np.array([[a, w] for a in acc_probes for w in steer_probes])
        margins = np.array([exact_margin(u) for u in probes])
        self.lane_sat_stats["axis_reselections"] += 1
        feasible = margins >= threshold
        if not feasible.any():
            return probes[np.argmax(margins)]
        scale = np.maximum(u_hi - u_lo, 1e-9)
        distances = np.sum(((probes - u_lin) / scale) ** 2, axis=1)
        return probes[feasible][np.argmin(distances[feasible])]

    def _lane_sat_candidates(
        self, env_idx: int, agent_idx: int, state_np, u_lin, u_lo, u_hi
    ):
        """Select one SAT axis per facet and return its atoms as candidates.

        An axis is admissible if all its atoms are nonnegative at the pose y
        (fixed by the previous control). Among admissible axes, the one whose
        constraints alpha(z) >= (1 - lambda) B(y) have the largest minimum
        margin under the selection control (see _lane_sat_selection_control)
        is chosen, where B(y) is the side's barrier value.
        Since the barrier is a min over facets of a max over axes, these rows
        are sufficient for B(z) >= (1 - lambda) B(y). Using B(y) rather than
        each atom's own value avoids spurious stopping requirements from atoms
        far from the boundary (e.g., the ahead axis of a facet in front).

        The returned "h" holds B(y) of each candidate's side.
        """
        boundaries = self.lane_sat.local_boundaries(
            self.scenario.world_state, env_idx, agent_idx
        )
        facets_all = [
            self.lane_sat.facets(points, side) for side, points in enumerate(boundaries)
        ]
        pose_y, _, _ = self._lane_sat_predict(state_np, u_lin)
        side_values = [
            (
                self.lane_sat.value(facets, side, pose_y)
                if facets["start"].shape[0] > 0
                else None
            )
            for side, facets in enumerate(facets_all)
        ]
        u_sel = self._lane_sat_selection_control(
            facets_all, side_values, state_np, u_lin, u_lo, u_hi
        )
        _, pose_z, _ = self._lane_sat_predict(state_np, u_sel)

        sides, axes, facet_ids, atom_ids, h_values = [], [], [], [], []
        for side, facets in enumerate(facets_all):
            tables_y = self.lane_sat.atom_tables(facets, side, pose_y)
            tables_z = self.lane_sat.atom_tables(facets, side, pose_z)
            decay = 1.0 - self.lambda_ttcbf_predefined
            axis_y = np.stack([table.min(axis=1) for table in tables_y], axis=1)
            if axis_y.shape[0] == 0:
                continue
            side_value = side_values[side]
            axis_margin = np.stack(
                [(table_z - decay * side_value).min(axis=1) for table_z in tables_z],
                axis=1,
            )
            admissible = axis_y >= -self.lane_sat_tol
            score = np.where(admissible, axis_margin, -np.inf)
            chosen = np.where(
                admissible.any(axis=1), score.argmax(axis=1), axis_y.argmax(axis=1)
            )
            for facet, axis in enumerate(chosen):
                for atom in range(tables_y[axis].shape[1]):
                    sides.append(side)
                    axes.append(axis)
                    facet_ids.append(facet)
                    atom_ids.append(atom)
                    h_values.append(side_value)

        return dict(
            state=np.asarray(state_np, dtype=np.float64),
            facets=facets_all,
            side=np.asarray(sides, dtype=np.int64),
            axis=np.asarray(axes, dtype=np.int64),
            facet=np.asarray(facet_ids, dtype=np.int64),
            atom=np.asarray(atom_ids, dtype=np.int64),
            h=np.asarray(h_values, dtype=np.float64),
            linearized_at=[[] for _ in h_values],
        )

    def _lane_sat_append_row(
        self, block: dict, k: int, u_lin, rows, lowers, uppers, n_vars: int
    ):
        """Append the constraint of candidate atom k linearized at u_lin.

        alpha(z(u)) ~ alpha(z(u_lin)) + G (u - u_lin) >= (1 - lambda) B(y),
        with B(y) the barrier value of the atom's side at the pose y.
        """
        cand = block["cand"]
        side = int(cand["side"][k])
        axis = int(cand["axis"][k])
        facet = int(cand["facet"][k])
        atom = int(cand["atom"][k])
        h = float(cand["h"][k])
        u_lin = np.asarray(u_lin, dtype=np.float64)

        _, pose_z, jacobian = self._lane_sat_predict(cand["state"], u_lin)
        facets = cand["facets"][side]
        alpha = self.lane_sat.atom_tables(facets, side, pose_z)[axis][facet, atom]
        grad_u = (
            self.lane_sat.atom_gradient(facets, side, axis, facet, atom, pose_z)
            @ jacobian
        )
        constant = alpha - grad_u @ u_lin

        terms = [(block["ui_slice"], grad_u), (block["slack_idx"][side], 1.0)]
        if block["lambda_idx"] is not None:
            terms.append((block["lambda_idx"][side], h))
            lower = h - constant
        else:
            lower = (1.0 - self.lambda_ttcbf_predefined) * h - constant
        # Absorbs the usual linearization error so the exact check rarely fails
        lower += self.lane_sat_linearization_margin
        cand["linearized_at"][k].append(u_lin.copy())
        # Collinear facets give identical rows; keep only the tightest one
        for prev_grad, prev_lower in block["rows"].get(side, []):
            if np.allclose(prev_grad, grad_u, rtol=0.0, atol=1e-12) and (
                prev_lower >= lower - 1e-12
            ):
                return
        block["rows"].setdefault(side, []).append((grad_u, lower))
        self._append_qp_row(
            rows, lowers, uppers, n_vars, terms, lower=lower, upper=PROXQP_INF
        )
        self.lane_sat_stats["lane_rows"] += 1

    def _append_lane_sat_rows(
        self,
        rows,
        lowers,
        uppers,
        n_vars: int,
        l_box: np.ndarray,
        u_box: np.ndarray,
        ui_slice: slice,
        state,
        env_idx: int,
        agent_idx: int,
        u_nom,
        slack_idx,
        lambda_idx,
    ) -> dict:
        """Add the initial rectangle_sat lane rows of one agent to a QP.

        The input box is tightened so that the next speed and steering stay
        inside their limits (no clipping, so the Euler map is exact). Atoms that
        are nearly active at the clipped nominal control get rows now; all others
        are checked after the solve (see _solve_qp_with_sat_repairs).

        Returns:
            The lane block (candidates and variable indices) of this agent.
        """
        state_np = np.asarray([float(value) for value in state[:5]], dtype=np.float64)
        v, delta = state_np[3], state_np[4]
        lower = np.array(
            [
                (float(self.v_min) - v) / self.dt,
                (float(self.steering_min) - delta) / self.dt,
            ]
        )
        upper = np.array(
            [
                (float(self.v_max) - v) / self.dt,
                (float(self.steering_max) - delta) / self.dt,
            ]
        )
        hi = np.minimum(u_box[ui_slice], upper)
        lo = np.minimum(np.maximum(l_box[ui_slice], lower), hi)
        l_box[ui_slice] = lo
        u_box[ui_slice] = hi
        u_lin = np.clip(np.asarray(u_nom, dtype=np.float64), lo, hi)
        if self.lane_baseline is not None:
            self.lane_baseline.append_rows(
                rows,
                lowers,
                uppers,
                n_vars,
                ui_slice,
                state_np,
                u_lin,
                env_idx,
                agent_idx,
                slack_idx,
                lambda_idx,
            )
            return dict(kind="baseline")
        if self.backup_horizon is not None:
            self.backup_horizon.register_lane(
                env_idx, agent_idx, state_np, u_lin, ui_slice, slack_idx, lambda_idx
            )
            if not self.parameters.is_two_step_rows:
                return dict(kind="tail_only")

        cand = self._lane_sat_candidates(env_idx, agent_idx, state_np, u_lin, lo, hi)
        block = dict(
            kind="lane",
            cand=cand,
            agent_idx=agent_idx,
            ui_slice=ui_slice,
            slack_idx=slack_idx,
            lambda_idx=lambda_idx,
            rows={},
        )
        values = self._lane_sat_candidate_values(cand, u_lin)
        margins = values - (1.0 - self.lambda_ttcbf_predefined) * cand["h"]
        for k in np.nonzero(margins < self.lane_sat_initial_band)[0]:
            self._lane_sat_append_row(
                block, int(k), u_lin, rows, lowers, uppers, n_vars
            )
        return block

    def _lane_sat_violated_atoms(self, block: dict, x: np.ndarray, u) -> np.ndarray:
        """Indices of the block's atoms that violate the exact constraint under u.

        The decay rates lambda are read from the QP solution x when adaptive.
        """
        cand = block["cand"]
        values = self._lane_sat_candidate_values(cand, u)
        if block["lambda_idx"] is not None:
            lambdas = np.array(
                [x[block["lambda_idx"][side]] for side in cand["side"]],
                dtype=np.float64,
            )
        else:
            lambdas = np.full(values.shape, self.lambda_ttcbf_predefined)
        return np.nonzero(values < (1.0 - lambdas) * cand["h"] - self.lane_sat_tol)[0]

    def _lane_sat_add_violated_rows(
        self, block: dict, x: np.ndarray, rows, lowers, uppers, n_vars: int
    ) -> tuple[int, int]:
        """Check all atoms of a block exactly at a QP solution x.

        Violated atoms get a new row linearized at the solution.

        Returns:
            (number of added rows, number of violated atoms that already have a
            row linearized at this solution, i.e. cannot be repaired).
        """
        cand = block["cand"]
        u = np.asarray(x[block["ui_slice"]], dtype=np.float64)
        violated = self._lane_sat_violated_atoms(block, x, u)

        added = 0
        stuck = 0
        for k in violated:
            if any(
                np.allclose(u, u_prev, rtol=0.0, atol=1e-9)
                for u_prev in cand["linearized_at"][k]
            ):
                stuck += 1
                continue
            self._lane_sat_append_row(block, int(k), u, rows, lowers, uppers, n_vars)
            self.lane_sat_stats["repair_rows"] += 1
            added += 1
        return added, stuck

    def _record_slack_activity(self, result: dict, slack_slices) -> None:
        """Record whether a CBF-QP needed its lane or pair slacks.

        A QP counts as infeasible if the solver failed or a CBF slack exceeds
        slack_active_tol; ``last_slack_active`` accumulates over the QPs of the
        current ``update_qp`` call.
        """
        self.slack_stats["qps"] += 1
        if not result["solved"]:
            self.slack_stats["failed"] += 1
            self.last_slack_active = True
            return
        x = result["x"]
        slack = max(
            [float(np.max(x[slc])) for slc in slack_slices if slc is not None] or [0.0]
        )
        if slack > self.slack_active_tol:
            self.slack_stats["slack_active"] += 1
            self.last_slack_active = True

    def _solve_qp_with_sat_repairs(
        self, H, g, rows, lowers, uppers, n_vars, l_box, u_box, sat_blocks
    ) -> dict:
        """Solve a CBF-QP with repairs (see _solve_qp_with_sat_repairs_steps)."""
        return run_steps(
            self._solve_qp_with_sat_repairs_steps(
                H, g, rows, lowers, uppers, n_vars, l_box, u_box, sat_blocks
            )
        )

    def _solve_qp_with_sat_repairs_steps(
        self, H, g, rows, lowers, uppers, n_vars, l_box, u_box, sat_blocks
    ):
        """Solve a CBF-QP and repair rectangle_sat lane and pair rows until certified.

        After each solve, the exact two-step Euler map checks every candidate
        atom of every block (lane blocks, see _append_lane_sat_rows, and pair
        blocks, see _append_pair_sat_rows). Violated atoms (linearization error
        or lazily omitted rows) get rows linearized at the solution, and the QP
        is solved again, at most lane_sat_max_repairs times.
        result["sat_certified"] tells whether the returned solution satisfies
        all exact lane and pair constraints.

        With the backup-horizon rows (is_backup_horizon_cbf), their tail block
        is built here from the constraints registered by the row hooks and
        repaired in the same loop. If its exact conditions still fail at the
        end, the involved agents switch to their backup controls (see
        BackupHorizonCBF.fallback).

        Generator: yields the rollout requests of the backup-horizon rows (see
        sigmarl/backup_horizon_cbf.py) and returns the result dict.
        """
        tail = None
        if self.backup_horizon is not None:
            tail = yield from self.backup_horizon.build_steps(
                rows, lowers, uppers, n_vars
            )
            if self.backup_horizon.IS_ACCEPT_OR_REPLACE:
                return (
                    yield from self._accept_or_replace_steps(
                        H, g, n_vars, l_box, u_box, tail
                    )
                )
        tail_ok = True
        solve_time = 0.0
        for repair in range(self.lane_sat_max_repairs + 1):
            C_mat, l_vec, u_vec = self._finalize_qp_rows(rows, lowers, uppers, n_vars)
            result = self._solve_proxqp_problem(H, g, C_mat, l_vec, u_vec, l_box, u_box)
            # Constraint rows of the first and of every solve (size of the QPs)
            if repair == 0:
                self.lane_sat_stats["qp_rows_first"] += len(l_vec)
            self.lane_sat_stats["qp_rows_all_solves"] += len(l_vec)
            self.lane_sat_stats["qp_solves_all"] += 1
            solve_time += result["solve_time"]
            result["solve_time"] = solve_time
            result["sat_certified"] = False
            if not result["solved"]:
                break
            t_check = time.perf_counter()
            added = 0
            stuck = 0
            for block in sat_blocks:
                # Baseline rows are not repaired (see cbf_baselines); tail_only
                # blocks have no two-step rows (backup-horizon rows only)
                if block["kind"] in ("baseline", "tail_only"):
                    continue
                add_violated_rows = (
                    self._pair_sat_add_violated_rows
                    if block["kind"] == "pair"
                    else self._lane_sat_add_violated_rows
                )
                n_added, n_stuck = add_violated_rows(
                    block, result["x"], rows, lowers, uppers, n_vars
                )
                added += n_added
                stuck += n_stuck
            if tail is not None:
                n_added, n_stuck = yield from self.backup_horizon.check_steps(
                    tail, result["x"], rows, lowers, uppers, n_vars
                )
                tail_ok = n_added == 0 and n_stuck == 0
                added += n_added
                stuck += n_stuck
            self.time_parts["check"] += time.perf_counter() - t_check
            if added == 0:
                result["sat_certified"] = stuck == 0
                break
            self.lane_sat_stats["repair_solves"] += int(
                repair < self.lane_sat_max_repairs
            )

        self.lane_sat_stats["qp_calls"] += 1
        self.lane_sat_stats["qp_rows_last"] += len(l_vec)
        if not result["sat_certified"]:
            self.lane_sat_stats["uncertified"] += 1
        if tail is not None and result["solved"] and not tail_ok:
            t_fallback = time.perf_counter()
            yield from self.backup_horizon.fallback_steps(tail, result["x"])
            self.time_parts["fallback"] += time.perf_counter() - t_fallback
        return result

    def _accept_or_replace_steps(self, H, g, n_vars, l_box, u_box, tail):
        """Model predictive shielding (condition baseline "mps"): the QP without
        CBF rows returns the saturated nominal controls, which are kept unless
        the exact check of the tail block fails; then the vehicles of the
        violated constraints switch to their backup controls
        (BackupHorizonCBF.fallback). Generator (see
        _solve_qp_with_sat_repairs_steps)."""
        C_mat, l_vec, u_vec = self._finalize_qp_rows([], [], [], n_vars)
        result = self._solve_proxqp_problem(H, g, C_mat, l_vec, u_vec, l_box, u_box)
        result["sat_certified"] = bool(result["solved"])
        self.lane_sat_stats["qp_calls"] += 1
        if tail is not None and result["solved"]:
            is_replaced = yield from self.backup_horizon.fallback_steps(
                tail, result["x"]
            )
            self.lane_sat_stats["mps_replaced_qps"] += int(is_replaced)
        return result

    # =========================
    # Exact rectangle-rectangle vehicle barrier
    # (veh_veh_cbf_footprint="rectangle_sat")
    # =========================
    #
    # The pair barrier (sigmarl/pair_sat_barrier.py) is a max over 8
    # separating directions of a min over 4 corner atoms. As for the lane
    # barrier, one direction is selected per pair, and each of its atoms must
    # satisfy alpha(z_a, z_b) >= (1 - lambda) h(y_a, y_b) under the exact
    # two-step Euler map (y: poses at k+1, z: poses at k+2), which implies
    # h(z) >= (1 - lambda) h(y). Rows are linearized and repaired lazily until
    # the exact check passes (_solve_qp_with_sat_repairs).
    #
    # A cross pair (fully decentralized, or across groups) knows only the ego
    # control. With z0 the poses under zero input, the atom of a direction
    # owned by vehicle o splits exactly as
    #   alpha(z_a, z_b) = alpha(z0) + D_a + D_b + M,
    # where D_a = alpha(z_a, z0_b) - alpha(z0) changes only with u_a, and the
    # mixed term M = sigma (e_o - e0_o) . (q - q0) is bilinear in the change of
    # the owner's axis e_o and of the other vehicle's corner q. Hence
    # M >= -|e_o - e0_o| Q, where Q bounds the corner displacement of the
    # other vehicle over its input box, and |e_o - e0_o| = 2 |sin(dpsi_o / 2)|
    # depends only on the owner's control. Each agent takes the share s of
    # alpha(z0) - (1 - lambda) h(y), and the owner also takes the mixed term:
    #   D_ego - [ego owns] 2 |sin(dpsi_ego / 2)| Q_other
    #       + s_ego (alpha(z0) - (1 - lambda) h(y)) >= 0.
    # Both agents compute the same direction, alpha(z0), and h(y) from the
    # shared states, and their shares add up to one, so their two certified
    # rows imply the pair condition for any controls they apply. A vehicle that
    # keeps its zero-input heading (e.g., at standstill) pays no mixed term.

    def _pair_sat_input_box(self, state_np) -> tuple:
        """Input box (lower [..., 2], upper [..., 2]) of the controls the
        simulator can apply at a state [..., 5]: within the input limits and
        keeping the next speed and steering within theirs (see _shield_saturate)."""
        speed_steer = np.asarray(state_np[..., 3:5], dtype=np.float64)
        lower = np.maximum(
            (self._shield_state_min - speed_steer) / self.dt, self._shield_input_min
        )
        upper = np.minimum(
            (self._shield_state_max - speed_steer) / self.dt, self._shield_input_max
        )
        return lower, np.maximum(lower, upper)

    def _pair_sat_corner_displacement(self, state_np) -> float:
        """Bound [m] of how far any corner of the k+2 pose moves from its
        zero-input position over the input box.

        The heading psi_z = psi_y + dt v1 sin(beta1) / l_r is a product of an
        affine function of the acceleration and a monotone function of the
        steering rate, so its extremes lie at the corners of the box. The center
        moves by at most dt (|v1 - v| + |v| |beta1 - beta0|), and a corner
        additionally by its distance from the center times the heading change.
        """
        lower, upper = self._pair_sat_input_box(state_np)
        _, pose_zero, _ = self._lane_sat_predict(state_np, (0.0, 0.0))
        heading = max(
            abs(self._lane_sat_predict(state_np, (acc, rate))[1][2] - pose_zero[2])
            for acc in (lower[0], upper[0])
            for rate in (lower[1], upper[1])
        )
        k = self.l_r / self.l_wb
        delta = float(state_np[4])
        beta0 = math.atan(k * math.tan(delta))
        beta_change = max(
            abs(math.atan(k * math.tan(delta + self.dt * rate)) - beta0)
            for rate in (lower[1], upper[1])
        )
        acc_abs = max(abs(lower[0]), abs(upper[0]))
        center = self.dt * (self.dt * acc_abs + abs(float(state_np[3])) * beta_change)
        corner_radius = float(np.linalg.norm(self.pair_sat.half))
        return center + corner_radius * heading

    @staticmethod
    def _pair_sat_cross_pair(
        ego, other, ego_slice, share, lam, slack_idx, lambda_idx
    ) -> dict:
        """Pair entry (see _append_pair_sat_rows) of a cross pair, in the
        canonical agent order, so that both agents build the same pair."""
        a, b = sorted((ego, other))
        return dict(
            a=a,
            b=b,
            slices=(ego_slice, None) if ego == a else (None, ego_slice),
            share=share,
            lam=lam,
            slack_idx=slack_idx,
            lambda_idx=lambda_idx,
        )

    def _pair_sat_values(self, block: dict, poses_a, poses_b) -> np.ndarray:
        """Constrained values [P, 4] of the selected atoms at poses [P, 3]: the
        atoms minus the mixed-term allowance of an ego that owns the direction."""
        atoms = self.pair_sat.atoms(poses_a, poses_b)[
            np.arange(poses_a.shape[0]), block["direction"]
        ]
        owner_heading = np.where(block["owner"] == 0, poses_a[:, 2], poses_b[:, 2])
        allowance = (
            block["mixed"]
            * 2.0
            * np.abs(np.sin(0.5 * (owner_heading - block["owner_heading_zero"])))
        )
        return atoms - allowance[:, None]

    def _append_pair_sat_rows(
        self, pairs, states_np, u_lin, fixed_agents, rows, lowers, uppers, n_vars
    ) -> dict:
        """Add the initial rectangle_sat pair rows of a QP and return their block.

        A joint pair has both controls in the QP; a cross pair has only the ego
        control, and the other agent's control is unknown (or, for an agent in
        fixed_agents, predicted as zero). The direction is selected at the
        joint linearization controls for a joint pair, and at zero input for a
        cross pair, so that both agents of a cross pair select the same one.
        Atoms whose margin at u_lin is below lane_sat_initial_band get rows now;
        all others are checked after the solve.

        Args:
            pairs: one dict per pair with the agents "a" < "b", the QP control
                slices "slices" (a, b), None for a control not in the QP, the
                ego "share" (None for a joint pair), the decay rate "lam", and
                the variable indices "slack_idx" and "lambda_idx" (or None).
            states_np: states (x, y, psi, v, delta) [5] of the agents by index.
            u_lin: linearization control [2] of each agent with a QP control.
            fixed_agents: agents whose control is known to be zero.

        Returns:
            The pair block (see _pair_sat_add_violated_rows).
        """
        if self.pair_baseline is not None:
            self.pair_baseline.append_rows(
                pairs, states_np, u_lin, fixed_agents, rows, lowers, uppers, n_vars
            )
            return dict(kind="baseline")
        if self.backup_horizon is not None:
            self.backup_horizon.register_pairs(pairs, states_np, u_lin, fixed_agents)
            if not self.parameters.is_two_step_rows:
                return dict(kind="tail_only")
        block, pose_lin = self._pair_sat_block(pairs, states_np, u_lin, fixed_agents)
        values = self._pair_sat_values(block, *self._pair_sat_poses(pairs, pose_lin))
        margins = values - (block["rhs0"] - (block["rhs_lam"] * block["lam"])[:, None])
        for k, c in zip(*np.nonzero(margins < self.lane_sat_initial_band)):
            u_pair = [
                u_lin.get(agent, np.zeros(2))
                for agent in (pairs[k]["a"], pairs[k]["b"])
            ]
            self._pair_sat_append_row(
                block, int(k), int(c), u_pair, rows, lowers, uppers, n_vars
            )
        return block

    @staticmethod
    def _pair_sat_poses(pairs, poses: dict) -> tuple:
        """Poses [P, 3] of the agents a and b of each pair from poses by agent."""
        return (
            np.array([poses[p["a"]] for p in pairs]),
            np.array([poses[p["b"]] for p in pairs]),
        )

    def _pair_sat_block(self, pairs, states_np, u_lin, fixed_agents) -> tuple:
        """Select the direction of each pair and build its pair block without
        adding rows (see _append_pair_sat_rows for the arguments).

        Returns:
            The pair block and the k+2 poses by agent under u_lin (zero input
            for an agent without a QP control).
        """
        n_pairs = len(pairs)
        agents = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
        pose_y, pose_zero, pose_lin = {}, {}, {}
        for agent in agents:
            pose_y[agent], pose_zero[agent], _ = self._lane_sat_predict(
                states_np[agent], (0.0, 0.0)
            )
            pose_lin[agent] = (
                self._lane_sat_predict(states_np[agent], u_lin[agent])[1]
                if agent in u_lin
                else pose_zero[agent]
            )

        def pair_poses(poses):
            return self._pair_sat_poses(pairs, poses)

        atoms_y = self.pair_sat.atoms(*pair_poses(pose_y))  # [P, 8, 4]
        atoms_zero = self.pair_sat.atoms(*pair_poses(pose_zero))
        direction_y = atoms_y.min(axis=2)
        h = direction_y.max(axis=1)
        is_cross = np.array([p["share"] is not None for p in pairs])
        score = np.where(
            is_cross[:, None],
            atoms_zero.min(axis=2),
            self.pair_sat.atoms(*pair_poses(pose_lin)).min(axis=2),
        )
        admissible = direction_y >= -self.lane_sat_tol
        direction = np.where(
            admissible.any(axis=1),
            np.where(admissible, score, -np.inf).argmax(axis=1),
            direction_y.argmax(axis=1),
        )
        owner = (direction >= 4).astype(np.int64)

        # Requirement value >= rhs0 - rhs_lam lambda for each selected atom
        rhs0 = np.repeat(h[:, None], 4, axis=1)
        rhs_lam = h.copy()
        mixed = np.zeros(n_pairs)
        corner_bounds = {}
        for k, p in enumerate(pairs):
            share = p["share"]
            if share is None:
                continue
            alpha_zero = atoms_zero[k, direction[k]]
            rhs0[k] = (1.0 - share) * alpha_zero + share * h[k]
            rhs_lam[k] = share * h[k]
            if p["slices"][owner[k]] is not None:
                other = (p["a"], p["b"])[1 - owner[k]]
                if other not in fixed_agents:
                    if other not in corner_bounds:
                        corner_bounds[other] = self._pair_sat_corner_displacement(
                            states_np[other]
                        )
                    mixed[k] = corner_bounds[other]

        block = dict(
            kind="pair",
            pairs=pairs,
            states=states_np,
            direction=direction,
            owner=owner,
            owner_heading_zero=np.array(
                [pose_zero[(p["a"], p["b"])[o]][2] for p, o in zip(pairs, owner)]
            ),
            mixed=mixed,
            rhs0=rhs0,
            rhs_lam=rhs_lam,
            lam=np.array([p["lam"] for p in pairs], dtype=np.float64),
            linearized_at=[[[] for _ in range(4)] for _ in pairs],
            rows=[[] for _ in pairs],
        )
        return block, pose_lin

    def _pair_sat_append_row(
        self, block: dict, k: int, c: int, u_pair, rows, lowers, uppers, n_vars: int
    ):
        """Append the row of atom c of pair k linearized at the controls u_pair
        ([u_a, u_b]; a control not in the QP is zero)."""
        pair = block["pairs"][k]
        direction = int(block["direction"][k])
        owner = int(block["owner"][k])
        poses, jacobians = [], []
        for agent, u in zip((pair["a"], pair["b"]), u_pair):
            _, pose_z, jacobian = self._lane_sat_predict(block["states"][agent], u)
            poses.append(pose_z)
            jacobians.append(jacobian)
        alpha = self.pair_sat.atoms(poses[0][None], poses[1][None])[0, direction, c]
        gradients = list(self.pair_sat.atom_gradients(poses[0], poses[1], direction))
        if block["mixed"][k] > 0.0:
            # Mixed-term allowance 2 Q |sin(dpsi / 2)| of the owning ego
            dpsi = poses[owner][2] - block["owner_heading_zero"][k]
            alpha -= block["mixed"][k] * 2.0 * abs(math.sin(0.5 * dpsi))
            gradients[owner] = gradients[owner].copy()
            gradients[owner][:, 2] -= (
                block["mixed"][k] * np.sign(dpsi) * math.cos(0.5 * dpsi)
            )

        terms = []
        constant = alpha
        for ui_slice, gradient, jacobian, u in zip(
            pair["slices"], gradients, jacobians, u_pair
        ):
            if ui_slice is None:
                continue
            grad_u = gradient[c] @ jacobian
            terms.append((ui_slice, grad_u))
            constant -= grad_u @ np.asarray(u, dtype=np.float64)
        if pair["slack_idx"] is not None:
            terms.append((pair["slack_idx"], 1.0))
        rhs0 = float(block["rhs0"][k, c])
        rhs_lam = float(block["rhs_lam"][k])
        if pair["lambda_idx"] is not None:
            terms.append((pair["lambda_idx"], rhs_lam))
            lower = rhs0 - constant
        else:
            lower = rhs0 - rhs_lam * float(block["lam"][k]) - constant
        lower += self.lane_sat_linearization_margin
        block["linearized_at"][k][c].append(
            [np.array(u, dtype=np.float64) for u in u_pair]
        )
        row_key = np.concatenate([np.atleast_1d(term[1]) for term in terms])
        for prev_key, prev_lower in block["rows"][k]:
            if (
                prev_key.shape == row_key.shape
                and np.allclose(prev_key, row_key, rtol=0.0, atol=1e-12)
                and prev_lower >= lower - 1e-12
            ):
                return
        block["rows"][k].append((row_key, lower))
        self._append_qp_row(
            rows, lowers, uppers, n_vars, terms, lower=lower, upper=PROXQP_INF
        )
        self.lane_sat_stats["pair_rows"] += 1

    def _pair_sat_add_violated_rows(
        self, block: dict, x: np.ndarray, rows, lowers, uppers, n_vars: int
    ) -> tuple[int, int]:
        """Check all selected pair atoms exactly at a QP solution x.

        Violated atoms get a new row linearized at the solution.

        Returns:
            (number of added rows, number of violated atoms that already have a
            row linearized at this solution, i.e. cannot be repaired).
        """
        pairs = block["pairs"]
        controls = []
        for pair in pairs:
            controls.append(
                [
                    (
                        np.zeros(2)
                        if ui_slice is None
                        else np.asarray(x[ui_slice], dtype=np.float64)
                    )
                    for ui_slice in pair["slices"]
                ]
            )
        pose_cache = {}

        def pose_z(agent, u):
            key = (agent, float(u[0]), float(u[1]))
            if key not in pose_cache:
                pose_cache[key] = self._lane_sat_predict(block["states"][agent], u)[1]
            return pose_cache[key]

        values = self._pair_sat_values(
            block,
            np.array([pose_z(p["a"], u[0]) for p, u in zip(pairs, controls)]),
            np.array([pose_z(p["b"], u[1]) for p, u in zip(pairs, controls)]),
        )
        lambdas = np.array(
            [
                x[p["lambda_idx"]] if p["lambda_idx"] is not None else lam
                for p, lam in zip(pairs, block["lam"])
            ],
            dtype=np.float64,
        )
        required = block["rhs0"] - (block["rhs_lam"] * lambdas)[:, None]

        added = 0
        stuck = 0
        for k, c in zip(*np.nonzero(values < required - self.lane_sat_tol)):
            u_pair = controls[k]
            if any(
                all(
                    np.allclose(u, u_prev, rtol=0.0, atol=1e-9)
                    for u, u_prev in zip(u_pair, prev)
                )
                for prev in block["linearized_at"][k][c]
            ):
                stuck += 1
                continue
            self._pair_sat_append_row(
                block, int(k), int(c), u_pair, rows, lowers, uppers, n_vars
            )
            self.lane_sat_stats["pair_repair_rows"] += 1
            added += 1
        return added, stuck

    # =========================
    # Nominal rectangle_sat margins (CBF-margin rewards with is_solve_qp=False)
    # =========================
    #
    # No QP is solved: the exact constraints of the QP are evaluated at the
    # applied nominal control. Lane and joint pair margins are the exact
    # discrete-time conditions B(z) - (1 - lambda) B(y); a cross pair margin is
    # the ego's share of the pair condition (see the cross-pair note above).

    def _sat_nominal_control(self, state_np, u) -> np.ndarray:
        """Control the simulator applies for u: u clipped to the input box that
        keeps the next speed and steering within their limits."""
        lower, upper = self._pair_sat_input_box(state_np)
        return np.clip(np.asarray(u, dtype=np.float64), lower, upper)

    def _lane_sat_nominal_margins(
        self, agent_idx: int, state_np, u, lam: float
    ) -> np.ndarray:
        """Exact lane margins B(z(u)) - (1 - lam) B(y) [2] of the left and right
        boundary; inf for a side without facets."""
        boundaries = self.lane_sat.local_boundaries(
            self.scenario.world_state, self.env_idx, agent_idx
        )
        pose_y, pose_z, _ = self._lane_sat_predict(state_np, u)
        margins = np.full(2, np.inf)
        for side, points in enumerate(boundaries):
            facets = self.lane_sat.facets(points, side)
            if facets["start"].shape[0] == 0:
                continue
            margins[side] = self.lane_sat.value(facets, side, pose_z) - (
                1.0 - lam
            ) * self.lane_sat.value(facets, side, pose_y)
        return margins

    def _pair_sat_joint_nominal_margins(
        self, pairs, states_np, controls, lams
    ) -> np.ndarray:
        """Exact margins h(z_a, z_b) - (1 - lam) h(y_a, y_b) [P] of joint pairs.

        Args:
            pairs: one dict per pair with the agents "a" and "b".
            states_np: states (x, y, psi, v, delta) [5] of the agents by index.
            controls: applied control [2] of each agent by index.
            lams: decay rate of each pair [P].
        """
        pose_y, pose_z = {}, {}
        for agent, state_np in states_np.items():
            pose_y[agent], pose_z[agent], _ = self._lane_sat_predict(
                state_np, controls[agent]
            )
        h_y = self.pair_sat.values(*self._pair_sat_poses(pairs, pose_y))
        h_z = self.pair_sat.values(*self._pair_sat_poses(pairs, pose_z))
        return h_z - (1.0 - np.asarray(lams, dtype=np.float64)) * h_y

    def _pair_sat_cross_nominal_margins(
        self, ego: int, nbrs, shares, states_np, u_ego, lam: float, fixed_agents
    ) -> np.ndarray:
        """Directed margins [len(nbrs)] of the ego's cross-pair rows under u_ego,
        with the neighbors' controls unknown (see _append_pair_sat_rows).

        Each margin is the minimum over the selected direction's atoms of the
        constrained value minus its requirement, i.e., the ego's row residual.
        """
        pairs = [
            self._pair_sat_cross_pair(ego, j, slice(0, 2), share, lam, None, None)
            for j, share in zip(nbrs, shares)
        ]
        block, pose_lin = self._pair_sat_block(
            pairs, states_np, {ego: u_ego}, fixed_agents
        )
        values = self._pair_sat_values(block, *self._pair_sat_poses(pairs, pose_lin))
        required = block["rhs0"] - (block["rhs_lam"] * block["lam"])[:, None]
        return (values - required).min(axis=1)

    # The batched versions below repeat the floating-point operations of their
    # per-agent counterparts, so that both give bit-identical margins.

    def _lane_sat_predict_poses_batch(self, states: np.ndarray, u: np.ndarray):
        """Poses y and z [..., 3] of _lane_sat_predict for states [..., 5] and
        controls [..., 2] (without the Jacobian)."""
        dt = self.dt
        k = self.l_r / self.l_wb
        x, y, psi, v, delta = np.moveaxis(states, -1, 0)

        beta = np.arctan(k * np.tan(delta))
        course = psi + beta
        pose_y = np.stack(
            [
                x + dt * v * np.cos(course),
                y + dt * v * np.sin(course),
                psi + dt * v * np.sin(beta) / self.l_r,
            ],
            axis=-1,
        )

        v1 = v + dt * u[..., 0]
        delta1 = delta + dt * u[..., 1]
        beta1 = np.arctan(k * np.tan(delta1))
        course1 = pose_y[..., 2] + beta1
        pose_z = np.stack(
            [
                pose_y[..., 0] + dt * v1 * np.cos(course1),
                pose_y[..., 1] + dt * v1 * np.sin(course1),
                pose_y[..., 2] + dt * v1 * np.sin(beta1) / self.l_r,
            ],
            axis=-1,
        )
        return pose_y, pose_z

    def _pair_sat_corner_displacement_batch(self, states: np.ndarray) -> np.ndarray:
        """_pair_sat_corner_displacement [...] of states [..., 5]."""
        lower, upper = self._pair_sat_input_box(states)
        _, pose_zero = self._lane_sat_predict_poses_batch(
            states, np.zeros(states.shape[:-1] + (2,))
        )
        heading = np.max(
            [
                np.abs(
                    self._lane_sat_predict_poses_batch(
                        states, np.stack([acc, rate], axis=-1)
                    )[1][..., 2]
                    - pose_zero[..., 2]
                )
                for acc in (lower[..., 0], upper[..., 0])
                for rate in (lower[..., 1], upper[..., 1])
            ],
            axis=0,
        )
        k = self.l_r / self.l_wb
        delta = states[..., 4]
        beta0 = np.arctan(k * np.tan(delta))
        beta_change = np.max(
            [
                np.abs(np.arctan(k * np.tan(delta + self.dt * rate)) - beta0)
                for rate in (lower[..., 1], upper[..., 1])
            ],
            axis=0,
        )
        acc_abs = np.maximum(np.abs(lower[..., 0]), np.abs(upper[..., 0]))
        center = self.dt * (self.dt * acc_abs + np.abs(states[..., 3]) * beta_change)
        corner_radius = float(np.linalg.norm(self.pair_sat.half))
        return center + corner_radius * heading

    def _cross_cbf_responsibility_share_batch(self, states_i, states_j) -> np.ndarray:
        """_cross_cbf_responsibility_share [...] of state pairs [..., 5]."""
        if self.cross_cbf_responsibility == "half":
            return np.full(states_i.shape[:-1], self._cross_cbf_drift_scale())
        if self.cross_cbf_responsibility != "follower":
            raise ValueError(
                f"Unknown cross_cbf_responsibility {self.cross_cbf_responsibility!r}."
            )
        dx = states_j[..., 0] - states_i[..., 0]
        dy = states_j[..., 1] - states_i[..., 1]
        psi_i = states_i[..., 2]
        psi_j = states_j[..., 2]
        # np.where(x > 0, x, 0) is max(0.0, x) also for -0.0 and nan
        f_i = dx * np.cos(psi_i) + dy * np.sin(psi_i)
        f_i = np.where(f_i > 0.0, f_i, 0.0)
        f_j = -dx * np.cos(psi_j) - dy * np.sin(psi_j)
        f_j = np.where(f_j > 0.0, f_j, 0.0)
        c = float(self.length)
        return (f_i + c) / (f_i + f_j + 2.0 * c)

    def _update_nominal_sat_margin_rewards_all_envs(
        self, tensordict, controllers
    ) -> float:
        """Batched rectangle_sat version of _update_nominal_cbf_margin_rewards
        over all envs, agents, and pairs, with bit-identical rewards.

        Args:
            controllers: the controller of each env (centralized) or the list of
                agent controllers of each env (fully decentralized).

        Returns:
            The lane-barrier computation time [s] (pseudo-distance time).
        """
        n_envs = int(tensordict.batch_size[0])
        n = int(self.parameters.n_agents)
        is_centralized = self.parameters.is_using_centralized_cbf
        # The controller that writes an env's rewards in update_qp
        env_controllers = [
            controller if is_centralized else controller[0]
            for controller in controllers
        ]
        for controller in controllers:
            for agent_controller in (
                controller if isinstance(controller, list) else [controller]
            ):
                agent_controller.time_pseudo_dis = 0.0

        reward_info = self.env.base_env.scenario_name.reward_info
        if (
            self.parameters.is_learn_cbf_lambda
            and "cbflambda" in self.parameters.rew_method
        ):
            for env_idx, controller in enumerate(env_controllers):
                reward_info.rew_cbf_lambda[env_idx, :] = torch.tensor(
                    controller.compute_cbf_lambda_smoothness_penalty(tensordict),
                    device=self.device,
                    dtype=torch.float32,
                )

        states, u_sat = self._nominal_states_and_controls_batch(tensordict, n_envs, n)

        if self.parameters.is_learn_cbf_lambda:
            lams = np.stack(
                [
                    controller._get_env_cbf_lambdas(tensordict)
                    for controller in env_controllers
                ]
            )  # [E, n]
        else:
            lams = np.full((n_envs, n), float(self.lambda_ttcbf_predefined))

        # Lane margins B(z) - (1 - lambda) B(y) [E, n, 2], inf without facets
        time_lane_start = time.time()
        pose_y, pose_u = self._lane_sat_predict_poses_batch(states, u_sat)
        facets, valid = self.lane_sat.padded_local_facets(
            self.scenario.world_state, n_envs, n
        )
        n_facets = valid.shape[-1]
        facets = {
            name: value.reshape(n_envs * n * 2, n_facets, *value.shape[4:])
            for name, value in facets.items()
        }
        sides = np.tile([0, 1], n_envs * n)

        def lane_values(poses):
            return self.lane_sat.padded_values(
                facets,
                valid.reshape(-1, n_facets),
                sides,
                np.repeat(poses.reshape(-1, 3), 2, axis=0),
            ).reshape(n_envs, n, 2)

        with np.errstate(invalid="ignore"):
            lane_margins = np.where(
                valid.any(axis=-1),
                lane_values(pose_u) - (1.0 - lams[..., None]) * lane_values(pose_y),
                np.inf,
            )
        time_lane = time.time() - time_lane_start

        # Pair margins [E, n, n]: (i, j) holds the margin that agent i is
        # penalized for (inf for no pair)
        pair_margins = np.full((n_envs, n, n), np.inf)
        if is_centralized:
            idx_a, idx_b = np.triu_indices(n, k=1)

            def joint_values(poses):
                return self.pair_sat.values(
                    poses[:, idx_a].reshape(-1, 3), poses[:, idx_b].reshape(-1, 3)
                ).reshape(n_envs, -1)

            pair_lams = (
                (lams[:, idx_a] + lams[:, idx_b]) / 2.0
                if self.parameters.is_learn_cbf_lambda
                else lams[:, idx_a]
            )
            margins = joint_values(pose_u) - (1.0 - pair_lams) * joint_values(pose_y)
            pair_margins[:, idx_a, idx_b] = margins
            pair_margins[:, idx_b, idx_a] = margins
        else:
            uncontrolled_mask = self.scenario.get_uncontrolled_agents_mask()
            uncontrolled = (
                np.zeros((n_envs, n), dtype=bool)
                if uncontrolled_mask is None
                else uncontrolled_mask[:n_envs].cpu().numpy()
            )
            # Neighbors within the coupling range, the first e_cap by index
            delta = states[:, :, None, 0:2] - states[:, None, :, 0:2]
            distance = np.sqrt(
                delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]
            )
            is_nbr = (
                distance <= float(self.parameters.cbf_coupling_range) + 1e-9
            ) & ~np.eye(n, dtype=bool)
            e_cap_user = getattr(self.parameters, "max_cross_neighbors", None)
            if e_cap_user is not None:
                e_cap = min(max(0, n - 1), max(0, int(e_cap_user)))
                is_nbr &= np.cumsum(is_nbr, axis=2) <= e_cap
            env, ego, other = np.nonzero(is_nbr)
            if env.size > 0:
                pair_margins[env, ego, other] = self._pair_sat_cross_margins_batch(
                    states, pose_y, pose_u, uncontrolled, lams, env, ego, other
                )

        # Rewards as in compute_cbf_violation_rewards_from_margins; a nan pair
        # margin is ignored there
        self._write_nominal_margin_rewards(lane_margins, pair_margins)
        self.time_pseudo_dis = time_lane
        return time_lane

    def _nominal_states_and_controls_batch(self, tensordict, n_envs: int, n: int):
        """States [E, n, 5] (x, y, psi, v, delta) of all agents and their
        nominal controls [E, n, 2] (acceleration, steering rate) as the
        simulator applies them (within _pair_sat_input_box)."""
        agents: list[Vehicle] = self.env.base_env.scenario_name.world.agents
        states_t = torch.stack(
            [
                torch.cat(
                    [
                        agent.state.pos[:n_envs],
                        agent.state.rot[:n_envs],
                        agent.state.speed[:n_envs],
                        agent.state.steering[:n_envs],
                    ],
                    dim=-1,
                )
                for agent in agents[:n]
            ],
            dim=1,
        )  # [E, n, 5]
        states = states_t.detach().cpu().numpy().astype(np.float64)
        _, u_applied_t = self.rl_action_to_u(
            rl_actions=tensordict[("agents", "action")][:n_envs, :n].reshape(-1, 2),
            v=states_t[..., 3].reshape(-1),
            steering=states_t[..., 4].reshape(-1),
        )
        u_applied = (
            u_applied_t.detach().cpu().numpy().astype(np.float64).reshape(n_envs, n, 2)
        )
        lower, upper = self._pair_sat_input_box(states)
        return states, np.clip(u_applied, lower, upper)

    def _write_nominal_margin_rewards(self, lane_margins, pair_margins):
        """Write the CBF-margin reward channels of the first E envs from the
        lane margins [E, n, 2] and the pair margins [E, n, n] ((i, j): the
        margin agent i is penalized for; inf for no pair), as in
        compute_cbf_violation_rewards_from_margins (a nan pair margin is
        ignored)."""
        reward_info = self.env.base_env.scenario_name.reward_info
        n_envs = lane_margins.shape[0]
        pair_margins = np.where(np.isnan(pair_margins), np.inf, pair_margins)
        v_pair = pair_margins.min(axis=2)
        v_pair = np.where(v_pair < 0.0, v_pair, 0.0)
        h_nom_lane, h_nom_pair = self.cbf_margin_reward_scales()
        for channel, values, scale in (
            (reward_info.rew_near_left_lane, lane_margins[..., 0], h_nom_lane),
            (reward_info.rew_near_right_lane, lane_margins[..., 1], h_nom_lane),
            (reward_info.rew_near_other_agents, v_pair, h_nom_pair),
        ):
            channel[:n_envs, :] = torch.as_tensor(
                np.clip(values / scale, -1.0, 0.0),
                device=self.device,
                dtype=torch.float32,
            )

    def _update_nominal_bh_margin_rewards_all_envs(
        self, tensordict, controllers
    ) -> float:
        """Backup-horizon (BH-CBF) version of
        _update_nominal_sat_margin_rewards_all_envs (centralized CBF).

        Every lane side and every pair within the QP's coupling range gets the
        margin of its BH rows (see sigmarl/backup_horizon_cbf.py) under the
        nominal controls u of all agents:

            min_{t=1..N} m_e(Psi_t(u)) - (1 - lambda) max(B_e(x_k), 0),

        where Psi_t(u) applies u and then t - 1 backup steps, and the backup
        value B_e(x_k) = min_{t=0..N} m_e(phi^t) is the minimum along the
        backup rollout from x_k. The positive part keeps the margin negative
        while the nominal controls leave the state unrecoverable (B_e < 0 at
        k + 1), instead of rewarding a mere decay of a negative B_e. The pair
        margin is the signed distance minus the pair buffer, as in the
        joint BH rows. All envs, agents, and pairs share one
        vectorized rollout over the longest stopping horizon N. The poses of a
        stopped vehicle no longer change, so the barriers are evaluated only at
        the tail steps where a vehicle of the constraint still moves; the
        minima are the same as over the whole tail. The rewards are normalized
        as the one-step margins.

        Args:
            controllers: the controller of each env.

        Returns:
            The lane-barrier computation time [s] (pseudo-distance time).
        """
        n_envs = int(tensordict.batch_size[0])
        n = int(self.parameters.n_agents)
        for controller in controllers:
            controller.time_pseudo_dis = 0.0
        states, u_sat = self._nominal_states_and_controls_batch(tensordict, n_envs, n)
        lam = float(self.lambda_ttcbf_predefined)
        n_rows = n_envs * n
        X = states.reshape(n_rows, 5)

        # One rollout of the backup rows [0, n_rows) (backup from x_k) and the
        # nominal rows [n_rows, 2 n_rows) (u, then backup), row e n + i for
        # agent i of env e; the route bundle changes only with the paths
        path_ids = tuple(
            self.scenario.world_state.ref_paths_agent_related.path_id[:n_envs, :n]
            .flatten()
            .tolist()
        )
        if self._bh_reward_bundle is None or self._bh_reward_bundle[0] != path_ids:
            routes = [
                controllers[env_idx]._shield_route(agent_idx)
                for env_idx in range(n_envs)
                for agent_idx in range(n)
            ]
            self._bh_reward_bundle = (
                path_ids,
                self._shield_route_bundle(routes + routes),
            )
        bundle = self._bh_reward_bundle[1]
        X_rows = np.concatenate([X, X])
        U_rows = self._shield_backup_controls(X_rows, bundle, None)
        U_rows[n_rows:] = u_sat.reshape(n_rows, 2)
        n_steps = self._shield_horizon(X[:, 3])
        n_t = n_steps + 1
        X_t = self._shield_euler_step(X_rows, self._shield_saturate(X_rows, U_rows))
        poses = np.empty((n_t, 2 * n_rows, 3))
        # Tail steps to evaluate [T, 2 n_rows]: a pose differs from the
        # previous one only if the speed was nonzero; the nominal tail starts
        # at t = 1 (y does not depend on the controls)
        is_needed = np.ones((n_t, 2 * n_rows), dtype=bool)
        is_needed[0, n_rows:] = False
        for t in range(n_t):
            poses[t] = X_t[:, 0:3]
            if t == n_steps:
                break
            is_needed[t + 1] = X_t[:, 3] != 0.0
            bounds = self._shield_input_bounds(X_t)
            X_t = self._shield_euler_step(
                X_t, self._shield_backup_controls(X_t, bundle, None, bounds)
            )
        # The first nominal pose z (t = 1) is needed whatever the speed at y
        is_needed[1, n_rows:] = True

        def tail_minima(needed, values_of):
            """Minima over the tail [K] of K constraints with evaluated steps
            needed [T, K]; values_of(t, k) gives the values at the steps."""
            t_idx, k_idx = np.nonzero(needed)
            values = np.full(needed.shape, np.inf)
            values[t_idx, k_idx] = values_of(t_idx, k_idx)
            return values.min(axis=0)

        # Lane margins [E, n, 2] on windows that cover the backup's reach (as
        # BackupHorizonCBF._build_lane_entries), inf without facets
        time_lane_start = time.time()
        v1 = np.maximum(0.0, states[..., 3] + self.dt * float(self.a_max))
        reach = (
            2.0 * v1 * self.dt
            + v1 * v1 / (-2.0 * float(self.a_min))
            + self.backup_horizon.LANE_WINDOW_MARGIN
        )
        facets, valid = self.lane_sat.padded_local_facets(
            self.scenario.world_state, n_envs, n, ahead_distance=reach
        )
        n_facets = valid.shape[-1]
        # Boundary 2 r + side of agent row r
        facets = {
            name: value.reshape(n_rows * 2, n_facets, *value.shape[4:])
            for name, value in facets.items()
        }
        valid_rows = valid.reshape(-1, n_facets)

        def lane_values(t_idx, k_idx):
            """Values at tail step t_idx of boundary k_idx (row k_idx // 2 of
            the backup and k_idx // 2 + n_rows of the nominal tail)."""
            boundary = k_idx % (2 * n_rows)
            row = boundary // 2 + np.where(k_idx >= 2 * n_rows, n_rows, 0)
            return self.lane_sat.padded_values(
                {name: value[boundary] for name, value in facets.items()},
                valid_rows[boundary],
                boundary % 2,
                poses[t_idx, row],
            )

        lane_minima = tail_minima(np.repeat(is_needed, 2, axis=1), lane_values)
        lane_backup = lane_minima[: 2 * n_rows].reshape(n_envs, n, 2)
        lane_nominal = lane_minima[2 * n_rows :].reshape(n_envs, n, 2)
        with np.errstate(invalid="ignore"):
            lane_margins = np.where(
                valid.any(axis=-1),
                lane_nominal - (1.0 - lam) * np.maximum(lane_backup, 0.0),
                np.inf,
            )
        time_lane = time.time() - time_lane_start

        # Pair margins [E, n, n] of the pairs within the QP's coupling range
        # (see the neighbor sets of the centralized QP)
        idx_a, idx_b = np.triu_indices(n, k=1)
        speed_reach = self.backup_horizon.reach(states[..., 3])  # [E, n]
        coupling_range = np.maximum(
            float(self.parameters.cbf_coupling_range),
            speed_reach[:, idx_a]
            + speed_reach[:, idx_b]
            + self.backup_horizon.diagonal
            + float(self.pair_safety_buffer),
        )
        distance = np.linalg.norm(
            states[:, idx_a, 0:2] - states[:, idx_b, 0:2], axis=-1
        )
        env, pair = np.nonzero(distance <= coupling_range + 1e-9)
        pair_margins = np.full((n_envs, n, n), np.inf)
        if env.size > 0:
            a, b = idx_a[pair], idx_b[pair]
            # Skip the pairs whose margin is surely nonnegative (zero reward):
            # a pair value is at least |d| - 2 r - buffer at center distance d
            # (r the half diagonal), and B_e is at most the value at y
            value_y = self.backup_horizon.pair_values(
                poses[0, env * n + a], poses[0, env * n + b]
            )
            gap = np.linalg.norm(
                poses[1:, env * n + a + n_rows, 0:2]
                - poses[1:, env * n + b + n_rows, 0:2],
                axis=-1,
            ).min(axis=0)
            lower_bound = (
                gap - self.backup_horizon.diagonal - float(self.pair_safety_buffer)
            )
            is_close = lower_bound < (1.0 - lam) * np.maximum(value_y, 0.0)
            env, a, b = env[is_close], a[is_close], b[is_close]
        if env.size > 0:
            # Rows of the pairs' vehicles in the backup, then the nominal tail
            row_a = np.concatenate([env * n + a, env * n + a + n_rows])
            row_b = np.concatenate([env * n + b, env * n + b + n_rows])
            pair_minima = tail_minima(
                is_needed[:, row_a] | is_needed[:, row_b],
                lambda t_idx, k_idx: self.backup_horizon.pair_values(
                    poses[t_idx, row_a[k_idx]], poses[t_idx, row_b[k_idx]]
                ),
            )
            margins = pair_minima[env.size :] - (1.0 - lam) * np.maximum(
                pair_minima[: env.size], 0.0
            )
            pair_margins[env, a, b] = margins
            pair_margins[env, b, a] = margins

        self._write_nominal_margin_rewards(lane_margins, pair_margins)
        self.time_pseudo_dis = time_lane
        return time_lane

    def backup_values_all_envs(self, controllers) -> np.ndarray:
        """Joint backup value of every agent at the current states [E, n]
        (diagnostic of the evaluation, in both CBF modes and for every filter).

        The value of agent i is the smallest backup value B_e(x_k) = min_{t=0..N}
        m_e(Psi^b_t) of its lane sides and of its pairs within the coupling
        range, with our margins (rectangle_sat lane barrier, signed distance
        minus the pair buffer) along the joint route-tracking braking backup of
        all agents. All values are nonnegative if and only if the joint state is
        recoverable; it is certified (in C) if it is also safe.

        Args:
            controllers: the controller (centralized) or the agent controllers
                (fully decentralized) of each env.
        """
        if self._diag_backup_horizon is None:
            from sigmarl.backup_horizon_cbf import BackupHorizonCBF

            self._diag_backup_horizon = BackupHorizonCBF(self)
            # Keep the diagnostic's counters out of the filter's statistics
            self._diag_backup_horizon.stats = defaultdict(int)
        horizon = self._diag_backup_horizon
        n_envs = len(controllers)
        n = int(self.parameters.n_agents)
        agents: list[Vehicle] = self.env.base_env.scenario_name.world.agents
        states = (
            torch.stack(
                [
                    torch.cat(
                        [
                            agent.state.pos[:n_envs],
                            agent.state.rot[:n_envs],
                            agent.state.speed[:n_envs],
                            agent.state.steering[:n_envs],
                        ],
                        dim=-1,
                    )
                    for agent in agents[:n]
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )  # [E, n, 5]
        n_rows = n_envs * n
        X = states.reshape(n_rows, 5)
        env_controllers = [
            item[0] if isinstance(item, list) else item for item in controllers
        ]
        routes = [
            env_controllers[env_idx]._shield_route(agent_idx)
            for env_idx in range(n_envs)
            for agent_idx in range(n)
        ]
        bundle = self._shield_route_bundle(routes)
        # Vehicle parameters of the rows (None: nominal)
        params = self._vehicle_params(np.tile(np.arange(n), n_envs))
        n_steps = self._shield_horizon(X[:, 3], params)
        X_t = self._shield_euler_step(
            X,
            self._shield_saturate(
                X,
                self._shield_backup_controls(X, bundle, None, params=params),
                params=params,
            ),
            params,
        )
        poses = np.empty((n_steps + 1, n_rows, 3))
        for t in range(n_steps + 1):
            poses[t] = X_t[:, 0:3]
            if t == n_steps:
                break
            bounds = self._shield_input_bounds(X_t, params)
            X_t = self._shield_euler_step(
                X_t,
                self._shield_backup_controls(X_t, bundle, None, bounds, params),
                params,
            )

        # Lane sides on windows that cover the backup's reach
        if params is None:
            a_min, a_max = float(self.a_min), float(self.a_max)
            halves = None
        else:
            a_min = params["min_acc"].reshape(n_envs, n)
            a_max = params["max_acc"].reshape(n_envs, n)
            halves = 0.5 * np.column_stack([params["length"], params["width"]])
        v1 = np.maximum(0.0, states[..., 3] + self.dt * a_max)
        reach = (
            2.0 * v1 * self.dt + v1 * v1 / (-2.0 * a_min) + horizon.LANE_WINDOW_MARGIN
        )
        facets, valid = self.lane_sat.padded_local_facets(
            self.scenario.world_state, n_envs, n, ahead_distance=reach
        )
        n_facets = valid.shape[-1]
        facets = {
            name: np.repeat(
                value.reshape(1, n_rows * 2, n_facets, *value.shape[4:]),
                n_steps + 1,
                axis=0,
            ).reshape(-1, n_facets, *value.shape[4:])
            for name, value in facets.items()
        }
        valid_rows = np.tile(valid.reshape(-1, n_facets), (n_steps + 1, 1))
        has_facets = valid_rows.any(axis=1)
        lane = np.full(valid_rows.shape[0], np.inf)
        if has_facets.any():
            lane[has_facets] = self.lane_sat.padded_values(
                {name: value[has_facets] for name, value in facets.items()},
                valid_rows[has_facets],
                np.tile(np.arange(2 * n_rows) % 2, n_steps + 1)[has_facets],
                np.repeat(poses, 2, axis=1).reshape(-1, 3)[has_facets],
                (
                    None
                    if halves is None
                    else np.tile(np.repeat(halves, 2, axis=0), (n_steps + 1, 1))[
                        has_facets
                    ]
                ),
            )
        values = lane.reshape(n_steps + 1, n_envs, n, 2).min(axis=(0, 3))  # [E, n]

        # Pairs within the coupling range of the backup tails
        idx_a, idx_b = np.triu_indices(n, k=1)
        if params is None:
            speed_reach = horizon.reach(states[..., 3])  # [E, n]
            diagonal = horizon.diagonal
        else:
            speed_reach = horizon.reach(states[..., 3].reshape(-1), params).reshape(
                n_envs, n
            )
            circumradius = np.linalg.norm(halves, axis=1).reshape(n_envs, n)
            diagonal = circumradius[:, idx_a] + circumradius[:, idx_b]
        coupling_range = (
            speed_reach[:, idx_a]
            + speed_reach[:, idx_b]
            + diagonal
            + float(self.pair_safety_buffer)
        )
        distance = np.linalg.norm(
            states[:, idx_a, 0:2] - states[:, idx_b, 0:2], axis=-1
        )
        env, pair = np.nonzero(distance <= coupling_range)
        if env.size > 0:
            a, b = idx_a[pair], idx_b[pair]
            pair_halves = (
                ()
                if halves is None
                else (
                    np.tile(halves[env * n + a], (n_steps + 1, 1)),
                    np.tile(halves[env * n + b], (n_steps + 1, 1)),
                )
            )
            pair_values = horizon.pair_values(
                poses[:, env * n + a].reshape(-1, 3),
                poses[:, env * n + b].reshape(-1, 3),
                *pair_halves,
            )
            pair_values = pair_values.reshape(n_steps + 1, -1).min(axis=0)
            np.minimum.at(values, (env, a), pair_values)
            np.minimum.at(values, (env, b), pair_values)
        return values

    def _pair_sat_cross_margins_batch(
        self, states, pose_y, pose_u, uncontrolled, lams, env, ego, other
    ) -> np.ndarray:
        """_pair_sat_cross_nominal_margins of directed pairs (env, ego, other)
        [P] given per-agent arrays [E, n, ...] of the states, the poses y and z
        under the applied controls, the uncontrolled mask, and the lambdas."""
        _, pose_zero = self._lane_sat_predict_poses_batch(
            states, np.zeros(states.shape[:-1] + (2,))
        )
        a, b = np.minimum(ego, other), np.maximum(ego, other)
        ego_is_a = (ego == a)[:, None]
        rows = np.arange(env.size)

        atoms_y = self.pair_sat.atoms(pose_y[env, a], pose_y[env, b])  # [P, 8, 4]
        atoms_zero = self.pair_sat.atoms(pose_zero[env, a], pose_zero[env, b])
        direction_y = atoms_y.min(axis=2)
        h = direction_y.max(axis=1)
        admissible = direction_y >= -self.lane_sat_tol
        direction = np.where(
            admissible.any(axis=1),
            np.where(admissible, atoms_zero.min(axis=2), -np.inf).argmax(axis=1),
            direction_y.argmax(axis=1),
        )
        owner = direction >= 4

        share = np.where(
            uncontrolled[env, other],
            1.0,
            self._cross_cbf_responsibility_share_batch(
                states[env, ego], states[env, other]
            ),
        )
        rhs0 = (1.0 - share)[:, None] * atoms_zero[rows, direction] + (share * h)[
            :, None
        ]
        rhs_lam = share * h

        # Mixed-term allowance if the ego owns the direction (see _pair_sat_block)
        owner_agent = np.where(owner, b, a)
        has_mixed = (owner_agent == ego) & ~uncontrolled[env, other]
        mixed = np.where(
            has_mixed,
            self._pair_sat_corner_displacement_batch(states)[env, other],
            0.0,
        )
        pose_lin_a = np.where(ego_is_a, pose_u[env, a], pose_zero[env, a])
        pose_lin_b = np.where(ego_is_a, pose_zero[env, b], pose_u[env, b])
        owner_heading = np.where(owner, pose_lin_b[:, 2], pose_lin_a[:, 2])
        allowance = (
            mixed
            * 2.0
            * np.abs(np.sin(0.5 * (owner_heading - pose_zero[env, owner_agent, 2])))
        )
        values = (
            self.pair_sat.atoms(pose_lin_a, pose_lin_b)[rows, direction]
            - allowance[:, None]
        )
        required = rhs0 - (rhs_lam * lams[env, ego])[:, None]
        return (values - required).min(axis=1)

    # =========================
    # Standstill escape (Parameters.is_standstill_escape)
    # =========================
    #
    # The lane rows constrain the pose two steps ahead, which a steering rate
    # moves only in proportion to the next speed. A vehicle that stops next to
    # a lane boundary with its wheels turned toward it can therefore neither
    # drive on (the boundary) nor steer away (no effect at zero speed), and the
    # QP keeps it stopped for good. A stalled agent (stopped while its nominal
    # control accelerates) searches an escape plan: steer to a fixed angle at
    # standstill, then creep a short distance forward or, if no forward plan
    # exists and the agent profile allows it, backward. A plan is admissible
    # if the exact lane barrier B satisfies B' >= (1 - lambda) B at every step
    # (a violated barrier must not get worse).
    # The plan replaces the nominal control until the agent has covered the
    # distance. Only the input of the CBF-QP and the shield changes, so their
    # safety guarantees hold as before; each agent uses only its own state and
    # lane, so the fully decentralized mode needs no communication.

    def _escape_lane_facets(self, agent_idx: int) -> list:
        """Buffered lane facets [left, right] that cover an agent's escape plans."""
        boundaries = self.lane_sat.local_boundaries(
            self.scenario.world_state,
            self.env_idx,
            agent_idx,
            ahead_distance=self.escape_distance,
        )
        return [
            self.lane_sat.facets(points, side) for side, points in enumerate(boundaries)
        ]

    def _escape_acceleration(self, v, direction):
        """Acceleration of an escape plan: toward its creep speed in its
        direction (+1 forward, -1 backward)."""
        speed = np.where(
            np.asarray(direction) > 0,
            self.escape_speed[1.0],
            self.escape_speed.get(-1.0, 0.0),
        )
        return np.clip(
            (direction * speed - v) / self.dt, -self.escape_acc, self.escape_acc
        )

    def _escape_plan_margins(self, X0, angles, directions, facets):
        """Worst lane CBF margins of escape plans of one agent.

        A plan sets its steering angle at standstill (the pose does not change)
        and then creeps in its direction with fixed steering until it has
        covered escape_distance.

        Args:
            X0: start states [K, 5] (their speeds are taken as zero).
            angles: steering angles [K] of the plans.
            directions: +1 (forward) or -1 (backward) [K].
            facets: lane facets of the agent (see _escape_lane_facets).

        Returns:
            min over the plan of B_{t+1} - min((1 - lambda) B_t, B_t) [K]
            (admissible if >= 0), and the end states [K, 5].
        """
        decay = 1.0 - self.lambda_ttcbf_predefined
        X = X0.copy()
        X[:, 3] = 0.0
        X[:, 4] = angles
        B = self._shield_lane_margins(X[:, :3], facets)
        worst = np.full(X.shape[0], np.inf)
        distance = np.zeros(X.shape[0])
        active = np.ones(X.shape[0], dtype=bool)
        U = np.zeros((X.shape[0], 2), dtype=np.float64)
        for _ in range(self.escape_max_steps):
            U[:, 0] = self._escape_acceleration(X[:, 3], directions)
            X_next = self._shield_euler_step(X, self._shield_saturate(X, U))
            X_next[~active] = X[~active]
            distance += np.hypot(*(X_next[:, :2] - X[:, :2]).T)
            B_next = self._shield_lane_margins(X_next[:, :3], facets)
            # A violated barrier (B < 0, e.g., after a respawn) must not get worse
            required = np.minimum(decay * B, B)
            worst = np.where(active, np.minimum(worst, B_next - required), worst)
            X, B = X_next, B_next
            active &= distance < self.escape_distance
            if not active.any():
                break
        return worst, X

    def _escape_search(self, agent_idx: int, X0: np.ndarray, angle_nominal: float):
        """Escape plan (direction, steering angle) of a stalled agent, or None.

        Forward plans come first, with the angle closest to the nominal one.
        Otherwise, a backward plan is chosen after which a forward plan exists
        (forward angle closest to the nominal one, then the largest margin),
        or the backward plan with the largest margin. None also means that
        creeping forward at the nominal angle is admissible: the lane does not
        block the agent then.
        """
        facets = self._escape_lane_facets(agent_idx)
        angles = np.linspace(
            float(self.steering_min), float(self.steering_max), self.escape_n_angles
        )
        n = angles.shape[0]
        angle_nominal = float(
            np.clip(angle_nominal, float(self.steering_min), float(self.steering_max))
        )
        worst_nominal, _ = self._escape_plan_margins(
            X0[None], np.array([angle_nominal]), np.ones(1), facets
        )
        if worst_nominal[0] >= 0.0:
            self.escape_stats["not_blocked"] += 1
            return None
        worst, _ = self._escape_plan_margins(
            np.repeat(X0[None], n, axis=0), angles, np.ones(n), facets
        )
        admissible = worst >= 0.0
        if admissible.any():
            options = angles[admissible]
            return 1.0, float(options[np.argmin(np.abs(options - angle_nominal))])
        if not self.escape_reverse:
            self.escape_stats["not_found"] += 1
            return None
        worst_back, X_end = self._escape_plan_margins(
            np.repeat(X0[None], n, axis=0), angles, -np.ones(n), facets
        )
        back = np.nonzero(worst_back >= 0.0)[0]
        if back.shape[0] == 0:
            self.escape_stats["not_found"] += 1
            return None
        # Forward plans after each admissible backward plan
        worst_after, _ = self._escape_plan_margins(
            np.repeat(X_end[back], n, axis=0),
            np.tile(angles, back.shape[0]),
            np.ones(back.shape[0] * n),
            facets,
        )
        cost = np.where(
            worst_after.reshape(back.shape[0], n) >= 0.0,
            np.abs(angles - angle_nominal)[None, :],
            np.inf,
        ).min(axis=1)
        best = np.lexsort((-worst_back[back], cost))[0]
        return -1.0, float(angles[back[best]])

    def _standstill_escape_nominal(
        self, agent_idx: int, state, u_nom, e_h: float, e_v: float, target_action
    ):
        """Nominal control of an agent with the standstill escape applied.

        Updates the agent's stall counter, searches an escape plan once it has
        stalled for escape_stall_steps, and follows an active plan: brake to a
        stop and steer to the plan's angle, then creep in its direction.

        Args:
            u_nom: nominal [acceleration, steering rate].
            e_h, e_v: CLF heading and speed errors (0 for an RL nominal).
            target_action: nominal [target speed, target steering angle].

        Returns:
            (u_nom, e_h, e_v, target_action), replaced while a plan is active.
        """
        X = np.array([float(value) for value in state[:5]], dtype=np.float64)
        plan = self._escape_plans.get(agent_idx)
        if plan is not None:
            jump = np.hypot(*(X[:2] - plan["last_xy"]))
            plan["last_xy"] = X[:2]
            plan["steps"] += 1
            if jump > 0.5 * float(self.length):
                outcome = "respawned"
            elif np.hypot(*(X[:2] - plan["origin"])) >= self.escape_distance:
                outcome = "completed"
            elif plan["steps"] > self.escape_max_steps:
                outcome = "timed_out"
            else:
                outcome = None
            if outcome is not None:
                self.escape_stats[outcome] += 1
                del self._escape_plans[agent_idx]
                plan = None
        if plan is None:
            if abs(X[3]) >= self.escape_speed_tol or u_nom[0] <= 0.0:
                self._escape_stall_steps[agent_idx] = 0
                return u_nom, e_h, e_v, target_action
            self._escape_stall_steps[agent_idx] += 1
            if self._escape_stall_steps[agent_idx] < self.escape_stall_steps:
                return u_nom, e_h, e_v, target_action
            self._escape_stall_steps[agent_idx] = 0
            if self.nom_controller_type != "rl" and self.clf_nominal_mode == (
                "pure_pursuit"
            ):
                angle_nominal = X[4] + e_h  # e_h is the steering-angle error
            else:
                angle_nominal = float(target_action[1])
            # The plans depend only on the pose and the nominal angle, so a
            # search without a plan is not repeated for the same ones
            key = np.append(X[:3], angle_nominal)
            last_key = self._escape_failed_search.get(agent_idx)
            if last_key is not None and np.all(
                np.abs(key - last_key) <= self.escape_search_tol
            ):
                self.escape_stats["repeated_searches_skipped"] += 1
                return u_nom, e_h, e_v, target_action
            self.escape_stats["searches"] += 1
            found = self._escape_search(agent_idx, X, angle_nominal)
            if found is None:
                self._escape_failed_search[agent_idx] = key
                return u_nom, e_h, e_v, target_action
            direction, angle = found
            self.escape_stats["forward" if direction > 0 else "backward"] += 1
            plan = dict(
                direction=direction,
                angle=angle,
                origin=X[:2],
                last_xy=X[:2],
                steps=0,
                moving=False,
            )
            self._escape_plans[agent_idx] = plan

        if not plan["moving"] and abs(plan["angle"] - X[4]) <= self.escape_angle_tol:
            plan["moving"] = True
        if plan["moving"]:
            acc = float(self._escape_acceleration(X[3], plan["direction"]))
            speed = plan["direction"] * self.escape_speed[plan["direction"]]
        else:
            acc = float(np.clip(-X[3] / self.dt, self.a_min, self.a_max))
            speed = 0.0
        omega = float(
            np.clip(
                (plan["angle"] - X[4]) / self.dt,
                self.steering_rate_min,
                self.steering_rate_max,
            )
        )
        u_escape = np.array([acc, omega], dtype=np.float64)
        target = self.u_to_target_action(u_escape, state[3], state[4]).detach()
        if self.nom_controller_type == "rl":
            return u_escape, e_h, e_v, target
        return u_escape, plan["angle"] - X[4], speed - X[3], target

    def _standstill_escape_control(self, agent_idx: int, state, u) -> np.ndarray:
        """Filtered control of an agent, steered at standstill while its escape
        plan is still turning the wheels and the QP keeps it stopped.

        The next speed is then exactly 0, so the pose two steps ahead equals
        the pose one step ahead for every steering rate, and no barrier changes.
        """
        plan = self._escape_plans.get(agent_idx)
        if plan is None or plan["moving"]:
            return u
        v, delta = float(state[3]), float(state[4])
        acc = -v / self.dt
        if (
            abs(v + self.dt * float(u[0])) > self.escape_stop_tol
            or not self.a_min <= acc <= self.a_max
        ):
            return u
        omega = np.clip(
            (plan["angle"] - delta) / self.dt,
            self.steering_rate_min,
            self.steering_rate_max,
        )
        self.escape_stats["standstill_steering"] += 1
        return np.array([acc, omega], dtype=np.float64)

    # =========================
    # Exact backup shield (Parameters.is_exact_backup_shield)
    # =========================
    #
    # The CBF-QP rows only ask the barriers to decay at a bounded rate over the
    # next step, which does not ensure that the QP stays feasible under the
    # input limits (e.g., two vehicles that merge too fast). The shield adds
    # recursive feasibility with a backup policy (route-tracking emergency
    # braking): a state is recoverable if the joint backup rollout from it
    # keeps every vehicle rectangle clear of its lane boundaries and of all
    # other rectangles until every vehicle has stopped. A control is accepted
    # only if the state after it is recoverable. The rollouts use the exact
    # explicit-Euler map of WorldCustom and exact rectangle geometry, so the
    # backup control itself is always accepted from a recoverable state.
    #
    # A state that is not recoverable (e.g., a vehicle respawned across the
    # lane buffer) only arises when the simulator departs from the prediction.
    # The constraints that the joint backup cannot satisfy then get a negative
    # floor, their margin under the joint backup rollout: they must not get
    # worse than braking would make them, while all other constraints keep the
    # floor 0. The floors carry over while the prediction holds and rise to
    # the margins of the accepted rollout, so the same induction applies to
    # the relaxed constraints, and they return to 0 once the vehicles recover.

    def _vehicle_params(self, agents):
        """Vehicle parameters of agents as arrays by key (see
        ScenarioRoadTraffic.vehicle_param_rows), or None while every vehicle has the
        nominal parameters of the agent profile.

        Heterogeneous vehicles (e.g., recorded footprints in a log replay) are
        supported by the fully decentralized BH-CBF filter with the
        backup-horizon rows only (no two-step rows, CBF baselines, condition
        baselines, exact backup shield, or standstill escape).
        """
        params = self.scenario.vehicle_param_rows(agents)
        if params is not None and not self._is_heterogeneous_supported:
            raise NotImplementedError(
                "Heterogeneous vehicle parameters need the fully decentralized "
                "BH-CBF filter (is_backup_horizon_cbf, condition_cbf_baseline "
                "'none', no two-step rows, CBF baselines, exact backup shield, "
                "or standstill escape)."
            )
        return params

    def _agent_acc_limits(self, agent_idx: int) -> tuple:
        """Acceleration limits (min, max) [m/s^2] of an agent."""
        params = self._vehicle_params([agent_idx])
        if params is None:
            return self.a_min, self.a_max
        return float(params["min_acc"][0]), float(params["max_acc"][0])

    def _shield_route(self, agent_idx: int) -> dict:
        """Cached centerline of an agent's route for the backup steering law."""
        ref_paths = self.scenario.world_state.ref_paths_agent_related
        path_id = int(ref_paths.path_id[self.env_idx, agent_idx])
        route = self._shield_route_cache.get(path_id)
        if route is None:
            n_points = int(ref_paths.n_points_long_term[self.env_idx, agent_idx])
            points = (
                ref_paths.long_term[self.env_idx, agent_idx, :n_points]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            stations, total_length = self._polyline_station_data(points)
            segments = np.diff(points, axis=0)
            route = dict(
                points=points,
                stations=stations,
                total_length=total_length,
                start=points[:-1],
                segments=segments,
                seg_len_sq=np.maximum(np.sum(segments**2, axis=1), 1e-12),
                is_loop=bool(ref_paths.is_loop[self.env_idx, agent_idx]),
            )
            self._shield_route_cache[path_id] = route
        return route

    @staticmethod
    def _shield_route_bundle(routes) -> dict:
        """Concatenate the route segments of several agents for vectorized
        projection (see _shield_backup_controls). Agents without a route get
        no segments."""
        owners, offsets = [], [0]
        # Station shift of each route, so that the routes' stations do not overlap
        shift, shift_end = np.full(len(routes), np.nan), 0.0
        for i, route in enumerate(routes):
            n_seg = 0 if route is None else route["start"].shape[0]
            owners.append(np.full(n_seg, i))
            offsets.append(offsets[-1] + n_seg)
            if route is not None:
                shift[i] = shift_end
                shift_end += route["total_length"] + 1.0
        valid = [i for i, route in enumerate(routes) if route is not None]

        def cat(values, empty_shape):
            if not valid:
                return np.zeros(empty_shape)
            return np.concatenate([values(routes[i], i) for i in valid])

        owner = np.concatenate(owners).astype(int)
        has_route = np.array([route is not None for route in routes], dtype=bool)
        # Index of each agent among the agents with a route
        group = np.cumsum(has_route) - 1
        start = cat(lambda r, i: r["start"], (0, 2))
        segments = cat(lambda r, i: r["segments"], (0, 2))
        points = cat(lambda r, i: r["points"], (0, 2))
        total_length = np.array(
            [np.nan if r is None else r["total_length"] for r in routes]
        )
        is_loop = np.array([r is not None and r["is_loop"] for r in routes])
        stations = cat(lambda r, i: r["stations"] + shift[i], (0,))
        return dict(
            owner=owner,
            # Segments per agent: repeating a per-agent value by it gathers
            # the value at each segment (as indexing by owner, but faster)
            counts=np.diff(offsets),
            # Contiguous coordinate columns of start, segments, and points
            start_x=np.ascontiguousarray(start[:, 0]),
            start_y=np.ascontiguousarray(start[:, 1]),
            segments_x=np.ascontiguousarray(segments[:, 0]),
            segments_y=np.ascontiguousarray(segments[:, 1]),
            # Agents whose target station wraps around a closed route
            wraps=is_loop & (total_length > 1e-9),
            # Station interpolation pieces (rows, stations, points_x, points_y);
            # a merged bundle has one per merged bundle
            pieces=[
                (
                    slice(None),
                    stations,
                    np.ascontiguousarray(points[:, 0]),
                    np.ascontiguousarray(points[:, 1]),
                )
            ],
            # The segments of the agents with a route tile the segments in order
            group_starts=np.asarray(offsets)[:-1][has_route],
            group=group,
            segment_group=group[owner],
            segment_index=np.arange(owner.shape[0]),
            start=start,
            segments=segments,
            seg_len_sq=cat(lambda r, i: r["seg_len_sq"], (0,)),
            seg_station=cat(lambda r, i: r["stations"][:-1], (0,)),
            seg_station_diff=cat(lambda r, i: np.diff(r["stations"]), (0,)),
            points=points,
            stations=stations,
            shift=shift,
            total_length=total_length,
            is_loop=is_loop,
        )

    @staticmethod
    def _shield_merge_route_bundles(bundles) -> dict:
        """Merge the route bundles of several rollouts (see _shield_route_bundle)
        for one joint call of _shield_backup_controls with all agents moving.

        The station shift of a route depends on its bundle and enters the
        target interpolation, so each bundle keeps its own stations (one
        interpolation piece each): every agent gets the controls of its own
        bundle bit for bit.
        """
        n_rows = np.cumsum([0] + [b["counts"].shape[0] for b in bundles])
        n_segments = np.cumsum([0] + [b["seg_len_sq"].shape[0] for b in bundles])
        n_groups = np.cumsum([0] + [b["group_starts"].shape[0] for b in bundles])

        def cat(key):
            return np.concatenate([b[key] for b in bundles])

        return dict(
            counts=cat("counts"),
            start_x=cat("start_x"),
            start_y=cat("start_y"),
            segments_x=cat("segments_x"),
            segments_y=cat("segments_y"),
            seg_len_sq=cat("seg_len_sq"),
            seg_station=cat("seg_station"),
            seg_station_diff=cat("seg_station_diff"),
            group_starts=np.concatenate(
                [b["group_starts"] + n for b, n in zip(bundles, n_segments)]
            ),
            group=np.concatenate([b["group"] + n for b, n in zip(bundles, n_groups)]),
            segment_group=np.concatenate(
                [b["segment_group"] + n for b, n in zip(bundles, n_groups)]
            ),
            segment_index=np.arange(n_segments[-1]),
            shift=cat("shift"),
            total_length=cat("total_length"),
            wraps=cat("wraps"),
            pieces=[
                (slice(start, stop), stations, points_x, points_y)
                for b, start, stop in zip(bundles, n_rows[:-1], n_rows[1:])
                for _, stations, points_x, points_y in b["pieces"]
            ],
        )

    def _shield_backup_controls(
        self, X: np.ndarray, bundle, moving, bounds=None, params=None
    ) -> np.ndarray:
        """Backup controls of all agents (route-tracking emergency braking).

        Maximum braking without crossing zero speed, and pure-pursuit steering
        toward the route point one lookahead ahead of the state's projection,
        converted to a steering rate with gain 4.

        Args:
            X: states [n, 5].
            bundle: routes of the agents (see _shield_route_bundle).
            moving: agents that are controlled; the others get zero control.
                None: all agents (without the index gathers).
            bounds: input bounds of the states X (see _shield_input_bounds),
                computed here if None.
            params: vehicle parameters of the rows (see _vehicle_params); None:
                the nominal ones.
        """
        U = np.zeros((X.shape[0], 2), dtype=np.float64)
        if moving is None:
            agents = slice(None)
        else:
            agents = np.nonzero(moving)[0]
            if agents.shape[0] == 0:
                return U
        k_delta = 4.0
        base_lookahead = 2.0 * float(self.length)
        X_a = X[agents]
        speed = X_a[:, 3]
        v = np.maximum(0.0, speed)
        # Stops a reversing vehicle as well
        if params is None:
            a_min, a_max, l_wb = float(self.a_min), float(self.a_max), self.l_wb
        else:
            a_min = params["min_acc"][agents]
            a_max = params["max_acc"][agents]
            l_wb = params["l_wb"][agents]
        U[agents, 0] = np.clip(-speed / self.dt, a_min, a_max)
        lookahead = np.maximum(float(self.length), base_lookahead + 0.2 * v)

        # Closest route segment of each agent (first one on ties, as argmin).
        # Per coordinate: the same products and sums as the dot products over
        # the coordinate axis.
        counts = bundle["counts"]
        rel_x = X[:, 0].repeat(counts) - bundle["start_x"]
        rel_y = X[:, 1].repeat(counts) - bundle["start_y"]
        segments_x, segments_y = bundle["segments_x"], bundle["segments_y"]
        tau = np.clip(
            (rel_x * segments_x + rel_y * segments_y) / bundle["seg_len_sq"], 0.0, 1.0
        )
        gap_x = rel_x - tau * segments_x
        gap_y = rel_y - tau * segments_y
        dist_sq = gap_x * gap_x + gap_y * gap_y
        starts = bundle["group_starts"]
        group_min = np.minimum.reduceat(dist_sq, starts)
        first_min = np.minimum.reduceat(
            np.where(
                dist_sq == group_min[bundle["segment_group"]],
                bundle["segment_index"],
                dist_sq.shape[0],
            ),
            starts,
        )
        seg = first_min[bundle["group"][agents]]
        station = (
            bundle["seg_station"][seg] + tau[seg] * bundle["seg_station_diff"][seg]
        )
        target_station = station + lookahead
        total_length = bundle["total_length"][agents]
        loop = bundle["wraps"][agents]
        if loop.any():
            target_station[loop] %= total_length[loop]
        target_station = (
            np.clip(target_station, 0.0, total_length) + bundle["shift"][agents]
        )
        pieces = bundle["pieces"]
        if len(pieces) == 1:
            _, stations, points_x, points_y = pieces[0]
            target_x = np.interp(target_station, stations, points_x)
            target_y = np.interp(target_station, stations, points_y)
        else:  # Merged bundles (all agents moving)
            target_x = np.empty_like(target_station)
            target_y = np.empty_like(target_station)
            for rows, stations, points_x, points_y in pieces:
                target_x[rows] = np.interp(target_station[rows], stations, points_x)
                target_y[rows] = np.interp(target_station[rows], stations, points_y)
        alpha = self._wrap_angle(
            np.arctan2(target_y - X_a[:, 1], target_x - X_a[:, 0]) - X_a[:, 2]
        )
        # float(steering limits): read from the state limits, which hold them
        # as floats (converting the torch tensors costs more than this step)
        delta_des = np.clip(
            np.arctan2(2.0 * l_wb * np.sin(alpha), lookahead),
            self._shield_state_min[1],
            self._shield_state_max[1],
        )
        U[agents, 1] = np.clip(
            k_delta * self._wrap_angle(delta_des - X_a[:, 4]),
            float(self.steering_rate_min),
            float(self.steering_rate_max),
        )
        if bounds is None:
            bounds = self._shield_input_bounds(X, params)
        return self._shield_saturate(X, U, bounds)

    def _shield_horizon(self, speeds, params=None) -> int:
        """Backup steps until vehicles at these speeds [m/s] have stopped, plus
        two (the first control may still speed them up). params: vehicle
        parameters of the speeds (see _vehicle_params); None: the nominal ones."""
        speeds = np.asarray(speeds, dtype=np.float64)
        if params is None:
            steps = max(
                np.maximum(speeds, 0.0).max(initial=0.0)
                / (-float(self.a_min) * self.dt),
                np.maximum(-speeds, 0.0).max(initial=0.0)
                / (float(self.a_max) * self.dt),
            )
        else:
            steps = max(
                (np.maximum(speeds, 0.0) / (-params["min_acc"] * self.dt)).max(
                    initial=0.0
                ),
                (np.maximum(-speeds, 0.0) / (params["max_acc"] * self.dt)).max(
                    initial=0.0
                ),
            )
        return int(math.ceil(steps)) + 2

    def _shield_input_bounds(self, X: np.ndarray, params=None) -> tuple:
        """Bounds (lower [n, 2], upper [n, 2]) of the controls the simulator
        applies at states [n, 5] of vehicles with parameters params (see
        _shield_saturate and _vehicle_params; None: the nominal ones)."""
        if params is None:
            input_min, input_max = self._shield_input_min, self._shield_input_max
        else:
            n = X.shape[0]
            input_min = np.column_stack(
                [params["min_acc"], np.full(n, self._shield_input_min[1])]
            )
            input_max = np.column_stack(
                [params["max_acc"], np.full(n, self._shield_input_max[1])]
            )
        lower = np.maximum((self._shield_state_min - X[:, 3:5]) / self.dt, input_min)
        upper = np.minimum((self._shield_state_max - X[:, 3:5]) / self.dt, input_max)
        return lower, upper

    def _shield_saturate(
        self, X: np.ndarray, U: np.ndarray, bounds=None, params=None
    ) -> np.ndarray:
        """Controls [n, 2] as the simulator applies them at states [n, 5]: within
        the input limits and keeping the next speed and steering within theirs
        (see compute_applied_action_control). bounds: the bounds of X (see
        _shield_input_bounds), computed here with the vehicle parameters params
        if None."""
        lower, upper = (
            self._shield_input_bounds(X, params) if bounds is None else bounds
        )
        # np.minimum/np.maximum: the same as np.clip, with less call overhead
        return np.minimum(np.maximum(U, lower), upper)

    def _shield_euler_step(
        self, X: np.ndarray, U: np.ndarray, params=None
    ) -> np.ndarray:
        """One explicit-Euler step of the kinematic bicycle (as in WorldCustom)
        of vehicles with parameters params (see _vehicle_params; None: the
        nominal ones)."""
        l_r = self.l_r if params is None else params["l_r"]
        l_wb = self.l_wb if params is None else params["l_wb"]
        tan_steering = np.tan(X[:, 4])
        beta = np.arctan(l_r / l_wb * tan_steering)
        heading = X[:, 2] + beta
        dt_speed = self.dt * X[:, 3]
        X_next = X.copy()
        X_next[:, 0] += dt_speed * np.cos(heading)
        X_next[:, 1] += dt_speed * np.sin(heading)
        X_next[:, 2] += dt_speed / l_wb * tan_steering * np.cos(beta)
        X_next[:, 3:5] += self.dt * U
        return X_next

    def _shield_pair_separation(self, poses: np.ndarray) -> np.ndarray:
        """Separating-axis gap [m] of all rectangle pairs, a lower bound of their
        distance (negative if they overlap).

        Args:
            poses: poses (x, y, psi) [..., n, 3].

        Returns:
            Symmetric gaps [..., n, n].
        """
        cos_psi, sin_psi = np.cos(poses[..., 2]), np.sin(poses[..., 2])
        axes = np.stack(
            [np.stack([cos_psi, sin_psi], -1), np.stack([-sin_psi, cos_psi], -1)], -2
        )  # [..., n, 2 axes, 2]
        half = np.array([0.5 * self.length, 0.5 * self.width])
        local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * half  # [4, 2]
        center = poses[..., 0:2]
        corners = center[..., :, None, :] + local @ axes  # [..., n, 4, 2]
        # proj[..., i, j, a, c]: corner c of j on axis a of i, relative to i's center
        proj = np.einsum(
            "...iad,...ijcd->...ijac",
            axes,
            corners[..., None, :, :, :] - center[..., :, None, None, :],
        )
        gap = np.maximum(proj.min(axis=-1) - half, -proj.max(axis=-1) - half,).max(
            axis=-1
        )  # [..., i, j] gap along the axes of i
        return np.maximum(gap, np.swapaxes(gap, -1, -2))

    def _shield_trajectories(self, X0, U0, routes, agents, horizon) -> np.ndarray:
        """Poses of the backup rollout of some agents after their first control.

        The backup law of an agent depends only on its own state, so agents are
        rolled out independently. After an agent stops, its pose stays constant.

        Args:
            X0: states [n, 5].
            U0: first controls [n, 2].
            routes: route of each agent (see _shield_route).
            agents: indices of the agents to roll out.
            horizon: number of backup steps after the first predicted state.

        Returns:
            Poses [horizon + 1, len(agents), 3]; index t is the state after the
            first control and t backup steps.
        """
        agents = np.asarray(agents, dtype=int)
        bundle = self._shield_route_bundle([routes[i] for i in agents])
        moving = np.ones(agents.shape[0], dtype=bool)
        poses = np.empty((horizon + 1, agents.shape[0], 3))
        X = self._shield_euler_step(X0[agents], U0[agents])
        for t in range(horizon + 1):
            poses[t] = X[:, 0:3]
            moving &= np.abs(X[:, 3]) > 1e-9
            if t == horizon or not moving.any():
                poses[t + 1 :] = X[:, 0:3]
                break
            X = self._shield_euler_step(
                X, self._shield_backup_controls(X, bundle, moving)
            )
        return poses

    def _shield_lane_facets(self, agent_idx: int, speed: float) -> list:
        """Buffered lane facets [left, right] that cover an agent's backup
        rollout from the given speed [m/s]."""
        v = max(0.0, float(speed))
        reach = 2.0 * v * self.dt + v * v / (-2.0 * float(self.a_min))
        boundaries = self.lane_sat.local_boundaries(
            self.scenario.world_state, self.env_idx, agent_idx, ahead_distance=reach
        )
        return [
            self.lane_sat.facets(points, side) for side, points in enumerate(boundaries)
        ]

    def _shield_lane_margins(self, poses: np.ndarray, facets) -> np.ndarray:
        """Lane barrier values [T] of one agent along its poses [T, 3] [m]."""
        margins = np.full(poses.shape[0], np.inf)
        for side, side_facets in enumerate(facets):
            if side_facets["start"].shape[0] > 0:
                margins = np.minimum(
                    margins, self.lane_sat.values(side_facets, side, poses)
                )
        return margins

    def _apply_exact_backup_shield(
        self, states, u_apply: np.ndarray, on_hold_set=None
    ) -> tuple[np.ndarray, dict]:
        """Keep the joint control only if the next state stays recoverable.

        Agents whose backup rollout violates a margin floor switch to the
        backup control, repeated until the rollout is clear. Only the rollouts
        and lane margins of switched agents are recomputed in each repetition.
        If even the joint backup control is not clear (the state is not
        recoverable, e.g., after a respawn), the floors of the violated
        constraints are lowered to their margins under the joint backup control
        and the repair is repeated, so only the agents that would make a
        violated constraint worse than braking are switched.

        Args:
            states: agent states (x, y, psi, v, delta).
            u_apply: joint controls [n, 2] after the other filters.
            on_hold_set: agents that stand still and are ignored.

        Returns:
            The shielded controls and info (certified: the rollout keeps every
            margin >= 0; relaxed: floors were lowered; repaired agents; margins).
        """
        if self.lane_sat is None:
            raise ValueError(
                "The exact backup shield needs lane_cbf_footprint='rectangle_sat'."
            )
        n = len(states)
        X0 = np.array([[float(v) for v in s[:5]] for s in states], dtype=np.float64)
        moving = np.ones(n, dtype=bool)
        for i in on_hold_set or ():
            moving[i] = False
        checked = np.nonzero(moving)[0]
        routes = [self._shield_route(i) if moving[i] else None for i in range(n)]
        # Steps until every vehicle stops under maximum braking
        horizon = self._shield_horizon(X0[moving, 3])
        facets = [
            self._shield_lane_facets(i, X0[i, 3]) if moving[i] else [] for i in range(n)
        ]

        U = np.asarray(u_apply, dtype=np.float64).copy()
        U[~moving] = 0.0
        U = self._shield_saturate(X0, U)
        U_backup = self._shield_backup_controls(
            X0, self._shield_route_bundle(routes), moving
        )
        buffer = float(self.safety_buffer)
        lane_floor_all, pair_floor_all = self._shield_floors(X0, moving)
        lane_floor = lane_floor_all[checked]
        pair_floor = pair_floor_all[np.ix_(checked, checked)]
        # Poses [T, len(checked), 3] and lane margins of the checked agents
        poses = np.empty((horizon + 1, checked.shape[0], 3))
        lane = np.empty((horizon + 1, checked.shape[0]))
        diagonal = np.arange(checked.shape[0])
        U_input = U.copy()
        is_priority = bool(self.parameters.is_shield_priority)
        if is_priority:
            for i in np.nonzero(~moving)[0]:
                self._shield_last_xy.pop(int(i), None)
            keys = [self._shield_priority_key(int(i), X0[i]) for i in checked]
        relaxed = False
        while True:
            update = np.arange(checked.shape[0])
            repaired = set()
            while True:
                if update.shape[0] > 0:
                    poses[:, update] = self._shield_trajectories(
                        X0, U, routes, checked[update], horizon
                    )
                    for local in update:
                        lane[:, local] = self._shield_lane_margins(
                            poses[:, local], facets[checked[local]]
                        )
                sep = self._shield_pair_separation(poses) - buffer
                sep[:, diagonal, diagonal] = np.inf
                # Margins below the floors in the steps that the control affects
                lane_bad = (lane[1:] < lane_floor).any(axis=0)
                pair_bad = (sep[1:] < pair_floor).any(axis=0)
                violated = lane_bad | pair_bad.any(axis=1)
                if not violated.any():
                    break
                if is_priority:
                    new = self._shield_priority_switches(
                        checked, lane_bad, pair_bad, keys, repaired
                    )
                else:
                    new = {int(i) for i in checked[violated]} - repaired
                new = new or (set(checked.tolist()) - repaired)
                if not new:
                    break
                repaired |= new
                for i in new:
                    U[i] = U_backup[i]
                update = np.nonzero(np.isin(checked, sorted(new)))[0]
            if not violated.any() or relaxed:
                break
            # Not recoverable: every agent uses the backup control. Only the
            # violated constraints are relaxed: they may not get worse than under
            # it (tolerance against rounding). The repair restarts from the input.
            relaxed = True
            tol = float(self.lane_sat_tol)
            lane_min = lane[1:].min(axis=0)
            pair_min = sep[1:].min(axis=0)
            lane_floor = np.where(lane_min < lane_floor, lane_min - tol, lane_floor)
            pair_floor = np.where(pair_min < pair_floor, pair_min - tol, pair_floor)
            U = U_input.copy()
        if is_priority and repaired and not violated.any():
            released = self._shield_release_pass(
                X0,
                U,
                U_input,
                routes,
                facets,
                horizon,
                checked,
                keys,
                repaired,
                poses,
                lane,
                lane_floor,
                pair_floor,
            )
            if released:
                repaired -= released
                sep = self._shield_pair_separation(poses) - buffer
                sep[:, diagonal, diagonal] = np.inf
        margins = np.minimum(lane, sep.min(axis=2, initial=np.inf))
        committed = float(margins[0].min(initial=np.inf))
        worst = float(margins[1:].min(initial=np.inf))
        certified = worst >= 0.0
        # Floors of the next step: the margins of the accepted rollout, whose
        # tail is the backup rollout from the predicted next state
        self._shield_lane_floor = np.zeros(n)
        self._shield_lane_floor[checked] = np.minimum(0.0, lane[1:].min(axis=0))
        self._shield_pair_floor = np.zeros((n, n))
        self._shield_pair_floor[np.ix_(checked, checked)] = np.minimum(
            0.0, sep[1:].min(axis=0)
        )
        self._shield_prediction = self._shield_euler_step(X0, U)
        self._shield_prediction[~moving] = np.nan
        if self.parameters.is_visu_shield_rollout:
            self.scenario.set_shield_rollout_render(
                self.env_idx,
                {int(i): poses[:, k, 0:2] for k, i in enumerate(checked)},
            )
        self.shield_stats["calls"] += 1
        self.shield_stats["repaired_agent_steps"] += len(repaired)
        self.shield_stats["repaired_steps"] += int(bool(repaired))
        self.shield_stats["uncertified"] += int(not certified)
        self.shield_stats["relaxed"] += int(relaxed)
        if relaxed or violated.any():
            self.hist.evt.append("EXACT-SHIELD-INF")
        return U, dict(
            certified=certified,
            relaxed=relaxed,
            recoverable=not violated.any(),
            repaired=sorted(int(i) for i in repaired),
            worst_margin=worst,
            committed_margin=committed,
            min_floor=float(
                min(lane_floor.min(initial=0.0), pair_floor.min(initial=0.0))
            ),
        )

    def _shield_floors(self, X0: np.ndarray, moving) -> tuple[np.ndarray, np.ndarray]:
        """Margin floors of the shield's constraints at the current states.

        The floors of the previous step carry over for agents whose state
        matches its prediction. The constraints of all other agents (first
        step, respawn, reset, or released from hold) get the floor 0.

        Args:
            X0: states [n, 5].
            moving: agents that are checked.

        Returns:
            Lane floors [n] and pair floors [n, n] (<= 0).
        """
        n = X0.shape[0]
        if self._shield_prediction is None or self._shield_prediction.shape[0] != n:
            return np.zeros(n), np.zeros((n, n))
        lane_floor = self._shield_lane_floor.copy()
        pair_floor = self._shield_pair_floor.copy()
        error = np.abs(X0 - self._shield_prediction).max(axis=1)
        reset = ~(error <= float(self.lane_sat_tol)) | ~moving
        self.shield_stats["floor_resets"] += int(
            (lane_floor[reset] < 0.0).sum() + (pair_floor[reset] < 0.0).sum()
        )
        lane_floor[reset] = 0.0
        pair_floor[reset, :] = 0.0
        pair_floor[:, reset] = 0.0
        return lane_floor, pair_floor

    def _shield_priority_key(self, agent_idx: int, X: np.ndarray) -> tuple:
        """Priority of an agent in shield conflicts (Parameters.is_shield_priority):
        (simulation step at which it became active, index); smaller keys win.

        An agent becomes active when the shield checks it for the first time
        after a reset, a release from hold, or a respawn (a jump of more than
        half a vehicle length beyond its travel in one step), so conflicts are
        resolved in arrival order. The
        key depends only on the agent's own states, so the decentralized shield
        broadcasts it with its proposal.

        Args:
            X: the agent's state (x, y, psi, v, delta).
        """
        step = int(self.scenario.timer.step[self.env_idx])
        last = self._shield_last_xy.get(agent_idx)
        if (
            last is None
            or np.hypot(*(X[:2] - last))
            > abs(X[3]) * self.dt + 0.5 * float(self.length)
            or step < self._shield_active_since[agent_idx]
        ):
            self._shield_active_since[agent_idx] = step
        self._shield_last_xy[agent_idx] = X[:2].copy()
        return (self._shield_active_since[agent_idx], int(agent_idx))

    @staticmethod
    def _shield_priority_switches(checked, lane_bad, pair_bad, keys, repaired):
        """Agents that the centralized shield switches to the backup control
        next when it resolves conflicts by priority.

        An agent with a violated lane margin switches. Of a violated pair, the
        agent with lower priority yields first, and the other one only if the
        pair is still violated while the first one brakes. Without a priority,
        both agents of a stopped pair would brake and block each other for good.

        Args:
            checked: agent indices of the local indices [m].
            lane_bad: violated lane constraints [m].
            pair_bad: violated pair constraints [m, m].
            keys: priority keys of the checked agents (smaller wins).
            repaired: agents that already use the backup control.
        """
        new = {int(i) for i in checked[lane_bad]}
        for a, b in zip(*np.nonzero(np.triu(pair_bad, 1))):
            keeper, yielder = (a, b) if keys[a] < keys[b] else (b, a)
            yielder, keeper = int(checked[yielder]), int(checked[keeper])
            new.add(keeper if yielder in repaired else yielder)
        return new - repaired

    def _shield_release_pass(
        self,
        X0,
        U,
        U_input,
        routes,
        facets,
        horizon,
        checked,
        keys,
        repaired,
        poses,
        lane,
        lane_floor,
        pair_floor,
    ) -> set:
        """Give switched agents back their input controls where the accepted
        joint rollout allows it, in priority order (centralized shield).

        A vehicle may have yielded to one with higher priority that switched
        later as well; without this pass, both would stay stopped. An agent's
        input control is restored only if its new rollout keeps its lane margin
        and its pair gaps to all other current rollouts above their floors, so
        the joint rollout stays clear. Updates U, poses, and lane in place.

        Returns:
            The released agents.
        """
        buffer = float(self.safety_buffer)
        local_of = {int(i): k for k, i in enumerate(checked)}
        released = set()
        for i in sorted(repaired, key=lambda i: keys[local_of[i]]):
            k = local_of[i]
            U_try = U.copy()
            U_try[i] = U_input[i]
            poses_i = self._shield_trajectories(X0, U_try, routes, [i], horizon)[:, 0]
            lane_i = self._shield_lane_margins(poses_i, facets[i])
            if (lane_i[1:] < lane_floor[k]).any():
                continue
            sep_i = self._shield_separation(poses_i[:, None, :], poses) - buffer
            sep_i[:, k] = np.inf
            if (sep_i[1:] < pair_floor[k]).any():
                continue
            U[i] = U_input[i]
            poses[:, k] = poses_i
            lane[:, k] = lane_i
            released.add(i)
        self.shield_stats["released_agent_steps"] += len(released)
        return released

    # Exact backup shield of the fully decentralized CBF-QP
    #
    # Each agent decides only its own control, in two rounds. Round 1 (after
    # its QP): the agent rolls out two options, its QP control and its backup
    # control, each followed by the backup policy, and broadcasts both pose
    # sequences. Round 2 (after all broadcasts): the agent keeps its QP
    # control only if its own lane constraint and, for every neighbor j, the
    # pair constraint hold for every option of j that j may still take: j's
    # backup option always, and j's QP option only if j has priority (smaller
    # key of _shield_priority_key, arrival order; with
    # Parameters.is_shield_priority=False, the lower index). A neighbor without priority checks the pair against both options
    # of the agent, so it yields if the two QP options conflict. Otherwise the
    # agent takes its backup control. Every combination that can result is
    # thus checked by an agent that takes it, except (backup, backup), the
    # joint backup rollout, whose tail the previous step certified. This gives
    # the guarantee of the centralized shield (Theorem 2) with the margin
    # floors of _apply_exact_backup_shield. The floors are computed by both
    # agents of a pair from the same broadcast data, so they agree.

    def _shield_separation(self, poses_a: np.ndarray, poses_b: np.ndarray):
        """Separating-axis gap [m] of rectangle pairs at poses (x, y, psi)
        [..., 3] (broadcast). Same value as _shield_pair_separation and exactly
        symmetric in a and b."""
        half_length, half_width = 0.5 * self.length, 0.5 * self.width

        def axis_gap(a, b):
            # Gap along the two axes of a
            cos_a, sin_a = np.cos(a[..., 2]), np.sin(a[..., 2])
            cos_b, sin_b = np.cos(b[..., 2]), np.sin(b[..., 2])
            dx, dy = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
            c = np.abs(cos_a * cos_b + sin_a * sin_b)
            s = np.abs(sin_a * cos_b - cos_a * sin_b)
            lon = np.abs(dx * cos_a + dy * sin_a) - half_length
            lat = np.abs(dy * cos_a - dx * sin_a) - half_width
            return np.maximum(
                lon - (half_length * c + half_width * s),
                lat - (half_length * s + half_width * c),
            )

        return np.maximum(axis_gap(poses_a, poses_b), axis_gap(poses_b, poses_a))

    def _fd_shield_propose(self, states, u_apply_i) -> dict:
        """Round 1 of the decentralized shield: the ego agent's two options.

        Args:
            states: agent states (x, y, psi, v, delta).
            u_apply_i: ego control [2] after the QP and the other filters.

        Returns:
            The broadcast: controls [2 options, 2] (QP, backup), poses
            [T, 2, 3] and lane margins [T, 2] of their rollouts (index t: state
            after the first control and t backup steps), the bounding box of
            the centers, whether the state matched its prediction, and the
            agent's priority key.
        """
        if self.lane_sat is None:
            raise ValueError(
                "The exact backup shield needs lane_cbf_footprint='rectangle_sat'."
            )
        i = int(self.agent_idx)
        X = np.array([float(v) for v in states[i][:5]], dtype=np.float64)
        route = self._shield_route(i)
        U = np.empty((2, 2), dtype=np.float64)
        U[0] = np.asarray(u_apply_i, dtype=np.float64).reshape(2)
        U[1] = self._shield_backup_controls(
            X[None], self._shield_route_bundle([route]), np.ones(1, dtype=bool)
        )[0]
        U[0] = self._shield_saturate(X[None], U[0:1])[0]
        # Steps until both options have stopped under maximum braking
        speed = max(X[3], X[3] + self.dt * U[0, 0], 0.0)
        horizon = self._shield_horizon([X[3], X[3] + self.dt * U[0, 0]])
        poses = self._shield_trajectories(
            np.stack([X, X]), U, [route, route], [0, 1], horizon
        )
        lane = self._shield_lane_margins(
            poses.reshape(-1, 3), self._shield_lane_facets(i, speed)
        ).reshape(-1, 2)
        centers = poses[..., 0:2].reshape(-1, 2)
        matched = self._fd_shield_prediction is not None and bool(
            np.abs(X - self._fd_shield_prediction).max() <= float(self.lane_sat_tol)
        )
        priority = (
            self._shield_priority_key(i, X)
            if self.parameters.is_shield_priority
            else (0, i)
        )
        return dict(
            agent=i,
            priority=priority,
            state=X,
            controls=U,
            poses=poses,
            lane=lane,
            box=(centers.min(axis=0), centers.max(axis=0)),
            matched=matched,
        )

    def _fd_shield_decide(self, proposals: dict) -> tuple[np.ndarray, dict]:
        """Round 2 of the decentralized shield: keep the ego QP control or brake.

        Args:
            proposals: round-1 broadcasts of all checked agents, keyed by agent.

        Returns:
            The ego control and info (accepted: the QP control is kept;
            relaxed: floors were lowered; worst margin of the checked
            combinations of the chosen option; committed margin).
        """
        i = int(self.agent_idx)
        n = int(self.parameters.n_agents)
        own = proposals[i]
        buffer = float(self.safety_buffer)
        tol = float(self.lane_sat_tol)
        # Rollouts whose center boxes are this far apart keep a gap > buffer
        # (the gap is at least the center distance / sqrt(2) - L / 2 - radius)
        reach = math.sqrt(2.0) * (
            0.5 * self.length + 0.5 * math.hypot(self.length, self.width) + buffer
        )
        lo, hi = own["box"]
        nbrs = [
            j
            for j in sorted(proposals)
            if j != i
            and np.linalg.norm(
                np.maximum(
                    0.0,
                    np.maximum(
                        proposals[j]["box"][0] - hi, lo - proposals[j]["box"][1]
                    ),
                )
            )
            <= reach
        ]
        horizon = max(proposals[k]["poses"].shape[0] for k in [i] + nbrs)

        def pad(x):
            # Every rollout has stopped at its end, so its last value repeats
            return np.concatenate([x, np.repeat(x[-1:], horizon - x.shape[0], 0)])

        lane = pad(own["lane"])  # [T, 2 own options]
        if nbrs:
            # sep[t, j, own option, option of j]
            sep = (
                self._shield_separation(
                    pad(own["poses"])[:, None, :, None, :],
                    np.stack([pad(proposals[j]["poses"]) for j in nbrs], axis=1)[
                        :, :, None, :, :
                    ],
                )
                - buffer
            )
        else:
            sep = np.full((horizon, 0, 2, 2), np.inf)

        # Floors: the margins of the joint backup rollout from the current
        # state, which is the tail of the previous rollout, while it was
        # predicted; at least the previous floors and at most 0
        lane_floor = 0.0
        if own["matched"]:
            lane_floor = min(0.0, max(self._fd_shield_lane_floor, lane[:, 1].min()))
        pair_floor = np.zeros(len(nbrs))
        for k, j in enumerate(nbrs):
            if own["matched"] and proposals[j]["matched"]:
                pair_floor[k] = min(
                    0.0, max(self._fd_shield_pair_floor[j], sep[:, k, 1, 1].min())
                )
        # Not recoverable (e.g., after a respawn): the violated constraints may
        # not get worse than under the joint backup rollout
        lane_backup = lane[1:, 1].min()
        pair_backup = sep[1:, :, 1, 1].min(axis=0, initial=np.inf)
        relaxed_lane = bool(lane_backup < lane_floor)
        relaxed_pairs = [
            j for k, j in enumerate(nbrs) if pair_backup[k] < pair_floor[k]
        ]
        relaxed = relaxed_lane or bool(relaxed_pairs)
        if relaxed_lane:
            lane_floor = lane_backup - tol
        pair_floor = np.where(pair_backup < pair_floor, pair_backup - tol, pair_floor)

        # Options of each neighbor that it may take: backup, and QP with priority
        options = np.array(
            [[proposals[j]["priority"] < own["priority"], True] for j in nbrs],
            dtype=bool,
        ).reshape(-1, 2)
        pair_qp = np.where(options[None], sep[1:, :, 0, :], np.inf).min(
            axis=(0, 2), initial=np.inf
        )
        accepted = bool(
            lane[1:, 0].min() >= lane_floor and (pair_qp >= pair_floor).all()
        )
        option = 0 if accepted else 1
        u = own["controls"][option].copy()

        self.shield_stats["floor_resets"] += int(
            not own["matched"]
            and (
                self._fd_shield_lane_floor < 0.0
                or (
                    self._fd_shield_pair_floor is not None
                    and (self._fd_shield_pair_floor < 0.0).any()
                )
            )
        )
        self._fd_shield_lane_floor = lane_floor
        self._fd_shield_pair_floor = np.zeros(n)
        self._fd_shield_pair_floor[nbrs] = pair_floor
        self._fd_shield_prediction = self._shield_euler_step(
            own["state"][None], u[None]
        )[0]
        worst = min(
            lane[1:, option].min(),
            pair_qp.min(initial=np.inf)
            if accepted
            else pair_backup.min(initial=np.inf),
        )
        committed = min(lane[0, 0], sep[0, :, 0, 0].min(initial=np.inf))
        self.shield_stats["calls"] += 1
        self.shield_stats["repaired_agent_steps"] += int(not accepted)
        self.shield_stats["relaxed"] += int(relaxed)
        if relaxed:
            self.hist.evt.append("EXACT-SHIELD-INF")
        return u, dict(
            agent=i,
            accepted=accepted,
            relaxed=relaxed,
            relaxed_lane=relaxed_lane,
            relaxed_pairs=relaxed_pairs,
            neighbors=nbrs,
            worst_margin=float(worst),
            committed_margin=float(committed),
            min_floor=float(min(lane_floor, pair_floor.min(initial=0.0))),
        )

    def apply_fully_decentralized_exact_shield(self, tensordict, controllers):
        """Run round 2 of the decentralized shield for every agent of the env.

        Called after all decentralized QPs (round 1) of the env. An agent that
        does not keep its QP control gets its backup control.

        Returns:
            The round-2 info of each checked agent, keyed by agent; "time" is
            the agent's round-2 computation time [s].
        """
        cache = self._fd_step_cache
        proposals = cache["shield_proposals"]
        states = cache["states"]
        world_state = self.env.base_env.scenario_name.world_state
        infos = {}
        rollouts = {}
        for i in sorted(proposals):
            t_start = time.perf_counter()
            u_i, infos[i] = controllers[i]._fd_shield_decide(proposals)
            infos[i]["time"] = time.perf_counter() - t_start
            option = 0 if infos[i]["accepted"] else 1
            rollouts[i] = proposals[i]["poses"][:, option, 0:2]
            if infos[i]["accepted"]:
                continue
            self._write_low_level_controls_to_tensordict(
                tensordict, states, u_i[None], agent_ids=[i]
            )
            applied_target_i = self._target_actions_from_current_actions(
                tensordict, states
            )[i].detach()
            world_state.applied_action_vel[self.env_idx, i] = applied_target_i[0]
            world_state.applied_action_steer[self.env_idx, i] = applied_target_i[1]
        if self.parameters.is_visu_shield_rollout:
            self.scenario.set_shield_rollout_render(self.env_idx, rollouts)
        return infos

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
        run_steps(self.update_qp_steps(tensordict))

    def update_qp_steps(self, tensordict):
        """Generator version of update_qp: a fully decentralized QP yields its
        backup-horizon rollout requests and the WRITE_BARRIER before writing
        its results (see sigmarl/backup_horizon_cbf.py); the other modes run
        through without yielding."""
        self.time_pseudo_dis = 0
        self.last_slack_active = False
        self.time_parts = dict.fromkeys(TIME_PARTS, 0.0)
        uses_cbf_lambda_reward = (
            self.parameters.is_learn_cbf_lambda
            and "cbflambda" in self.parameters.rew_method
        )

        if not self.parameters.is_solve_qp:
            if uses_cbf_lambda_reward and (
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

            if "cbfmargin" in self.parameters.rew_method and (
                self.parameters.is_using_centralized_cbf or self.agent_idx == 0
            ):
                self._update_nominal_cbf_margin_rewards(tensordict)
            return

        if not self.parameters.is_using_centralized_cbf:
            if uses_cbf_lambda_reward and self.agent_idx == 0:
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

            yield from self.update_fully_decentralized_cbf_qps_steps(tensordict)
        elif self.parameters.is_grouping_agents:
            self.update_grouped_cbf_qps(tensordict)
        else:
            if uses_cbf_lambda_reward:
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
        if self.parameters.cbf_margin_reward_condition == "backup_horizon":
            raise NotImplementedError(
                "The backup-horizon margin reward is computed for all envs at "
                "once (update_nominal_cbf_margin_rewards_all_envs)."
            )
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

    def _circle_centers_xy_batch_np(self, states_np: np.ndarray) -> np.ndarray:
        centers = np.asarray(self.rec_cir_approx.centers, dtype=np.float64)
        x = states_np[..., 0]
        y = states_np[..., 1]
        psi = states_np[..., 2]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        local_x = centers[:, 0]
        local_y = centers[:, 1]
        circle_x = (
            x[..., None] + cos_psi[..., None] * local_x - sin_psi[..., None] * local_y
        )
        circle_y = (
            y[..., None] + sin_psi[..., None] * local_x + cos_psi[..., None] * local_y
        )
        return np.stack([circle_x, circle_y], axis=-1)

    def _linearized_center_kinematics_coeffs_batch_np(
        self, states_np: np.ndarray
    ) -> dict:
        centers = np.asarray(self.rec_cir_approx.centers, dtype=np.float64)
        psi = states_np[..., 2]
        v = states_np[..., 3]
        delta = states_np[..., 4]

        k = float(self.l_r / self.l_wb)
        tan_delta = np.tan(delta)
        beta = np.arctan(k * tan_delta)
        theta = psi + beta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        sec_delta_sq = 1.0 / (np.cos(delta) ** 2)
        tan_beta = k * tan_delta
        cos_beta = 1.0 / np.sqrt(1.0 + tan_beta**2)
        sin_beta = tan_beta * cos_beta

        dx = v * cos_theta
        dy = v * sin_theta
        dpsi = (v / self.l_wb) * tan_delta * cos_beta

        def second_derivative(u_1: float, u_2: float):
            dbeta = (k * sec_delta_sq * u_2) / (1.0 + (k * tan_delta) ** 2)
            ddx = u_1 * cos_theta - v * sin_theta * (dpsi + dbeta)
            ddy = u_1 * sin_theta + v * cos_theta * (dpsi + dbeta)
            ddpsi = (
                (u_1 / self.l_wb) * cos_beta * tan_delta
                + (v / self.l_wb) * cos_beta * sec_delta_sq * u_2
                - (v / self.l_wb) * sin_beta * tan_delta * dbeta
            )
            return ddx, ddy, ddpsi

        ddx0, ddy0, ddpsi0 = second_derivative(0.0, 0.0)
        ddx_e1, ddy_e1, ddpsi_e1 = second_derivative(1.0, 0.0)
        ddx_e2, ddy_e2, ddpsi_e2 = second_derivative(0.0, 1.0)

        delta_x = centers[:, 0]
        delta_y = centers[:, 1]
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        dpsi_sq = dpsi * dpsi

        def center_derivatives(ddx_body, ddy_body, ddpsi_body):
            dx_center = (
                dx[..., None]
                - delta_x * sin_psi[..., None] * dpsi[..., None]
                - delta_y * cos_psi[..., None] * dpsi[..., None]
            )
            dy_center = (
                dy[..., None]
                + delta_x * cos_psi[..., None] * dpsi[..., None]
                - delta_y * sin_psi[..., None] * dpsi[..., None]
            )
            ddx_center = (
                ddx_body[..., None]
                - delta_x
                * (
                    sin_psi[..., None] * ddpsi_body[..., None]
                    + cos_psi[..., None] * dpsi_sq[..., None]
                )
                - delta_y
                * (
                    cos_psi[..., None] * ddpsi_body[..., None]
                    - sin_psi[..., None] * dpsi_sq[..., None]
                )
            )
            ddy_center = (
                ddy_body[..., None]
                + delta_x
                * (
                    cos_psi[..., None] * ddpsi_body[..., None]
                    - sin_psi[..., None] * dpsi_sq[..., None]
                )
                - delta_y
                * (
                    sin_psi[..., None] * ddpsi_body[..., None]
                    + cos_psi[..., None] * dpsi_sq[..., None]
                )
            )
            return dx_center, dy_center, ddx_center, ddy_center

        dx_c, dy_c, ddx0_c, ddy0_c = center_derivatives(ddx0, ddy0, ddpsi0)
        _, _, ddx_e1_c, ddy_e1_c = center_derivatives(ddx_e1, ddy_e1, ddpsi_e1)
        _, _, ddx_e2_c, ddy_e2_c = center_derivatives(ddx_e2, ddy_e2, ddpsi_e2)

        return {
            "dx": dx_c,
            "dy": dy_c,
            "a_ddx": np.stack([ddx_e1_c - ddx0_c, ddx_e2_c - ddx0_c], axis=-1),
            "c_ddx": ddx0_c,
            "a_ddy": np.stack([ddy_e1_c - ddy0_c, ddy_e2_c - ddy0_c], axis=-1),
            "c_ddy": ddy0_c,
        }

    def _fixed_lambda_lane_margins_from_distance_results(
        self,
        kins: dict,
        env_idx: int,
        agent_idx: int,
        ui: np.ndarray,
        distance_results: tuple,
        lambda_ttcbf: float,
    ) -> np.ndarray:
        distance, grad, hess = distance_results
        h = distance - self.circle_radius**2 - self.lane_safety_buffer
        dx = kins["dx"][env_idx, agent_idx]
        dy = kins["dy"][env_idx, agent_idx]
        a_ddx = kins["a_ddx"][env_idx, agent_idx]
        a_ddy = kins["a_ddy"][env_idx, agent_idx]
        c_ddx = kins["c_ddx"][env_idx, agent_idx]
        c_ddy = kins["c_ddy"][env_idx, agent_idx]

        velocity = np.stack([dx, dy], axis=-1)
        dot_h = grad[:, 0] * dx + grad[:, 1] * dy
        control_affine = (
            0.5
            * (self.dt_taylor * self.dt_taylor)
            * (grad[:, 0, None] * a_ddx + grad[:, 1, None] * a_ddy)
        )
        const_dd = (
            grad[:, 0] * c_ddx
            + grad[:, 1] * c_ddy
            + np.einsum("ci,cij,cj->c", velocity, hess, velocity)
        )
        b = (
            lambda_ttcbf * h
            + dot_h * self.dt_taylor
            + 0.5 * (self.dt_taylor * self.dt_taylor) * const_dd
        )
        return np.einsum("ci,i->c", control_affine, ui) + b

    def update_nominal_cbf_margin_rewards_all_envs(self, tensordict, controllers=None):
        """Fast all-env nominal CBF margin reward update for the default path.

        With both footprints rectangle_sat, it runs the batched exact barriers
        (see ``_update_nominal_sat_margin_rewards_all_envs``) in both CBF modes;
        otherwise it covers the circle footprints in the centralized mode.
        Mixed footprints fall back to ``compute_nominal_cbf_constraint_margins``
        per env.

        Returns total pseudo-distance time when the fast path is applicable;
        otherwise returns ``None`` so callers can fall back to per-env updates.
        Without a QP, the margins only feed the "cbfmargin" reward and the
        "cbflambda" penalty, so nothing is computed (zero time) for other
        rewards.
        """
        is_margin_reward = (
            "cbfmargin" in self.parameters.rew_method
            or "cbflambda" in self.parameters.rew_method
        )
        if not self.parameters.is_solve_qp and not is_margin_reward:
            return 0.0
        if (
            self.lane_cbf_footprint == "rectangle_sat"
            and self.veh_veh_cbf_footprint == "rectangle_sat"
            and not self.parameters.is_solve_qp
            and not self.parameters.is_exact_backup_shield
            and self.rl_action_type == "speed_steer"
            and controllers is not None
        ):
            if self.parameters.cbf_margin_reward_condition == "backup_horizon":
                return self._update_nominal_bh_margin_rewards_all_envs(
                    tensordict, controllers
                )
            return self._update_nominal_sat_margin_rewards_all_envs(
                tensordict, controllers
            )
        if (
            self.parameters.cbf_margin_reward_condition == "backup_horizon"
            and "cbfmargin" in self.parameters.rew_method
        ):
            # Only the "cbfmargin" reward reads the margins; otherwise the
            # per-env QP path below runs as for the one-step condition
            raise NotImplementedError(
                "The backup-horizon margin reward needs the batched rectangle_sat "
                "path (rectangle_sat footprints, is_solve_qp=False, no exact "
                "backup shield, rl_action_type='speed_steer')."
            )
        if (
            self.lane_cbf_footprint == "rectangle_sat"
            or self.veh_veh_cbf_footprint == "rectangle_sat"
            or self.parameters.is_solve_qp
            or not self.parameters.is_using_centralized_cbf
            or self.parameters.is_learn_cbf_lambda
            or self.parameters.adaptive_lambda
            or self.rl_action_type != "speed_steer"
            or self.parameters.lane_distance_type != "softmin"
            or not self.parameters.is_use_sampled_lane_boundary_for_cbf
        ):
            return None

        n_envs = int(tensordict.batch_size[0])
        n_agents = int(self.parameters.n_agents)
        n_circles = int(self.parameters.n_circles_approximate_vehicle)
        if controllers is not None:
            if len(controllers) != n_envs or any(
                isinstance(controller, list) for controller in controllers
            ):
                return None

        agents: list[Vehicle] = self.env.base_env.scenario_name.world.agents
        states_t = torch.stack(
            [
                torch.cat(
                    [
                        agents[i].state.pos,
                        agents[i].state.rot,
                        agents[i].state.speed,
                        agents[i].state.steering,
                    ],
                    dim=-1,
                )
                for i in range(n_agents)
            ],
            dim=1,
        )
        states_np = states_t.detach().cpu().numpy().astype(np.float64, copy=False)
        circles_xy = self._circle_centers_xy_batch_np(states_np)
        kins = self._linearized_center_kinematics_coeffs_batch_np(states_np)

        actions = tensordict[("agents", "action")][:n_envs, :n_agents]
        if actions.shape[-1] != 2:
            return None
        _, u_applied_t = compute_applied_action_control(
            actions.reshape(n_envs * n_agents, 2),
            states_t[..., 3].reshape(n_envs * n_agents),
            states_t[..., 4].reshape(n_envs * n_agents),
            self.dt,
            rl_action_type=self.rl_action_type,
            lookahead_dist=self.lookahead_dist,
            l_wb=self.l_wb,
            position_speed_lateral_max=self.position_speed_lateral_max,
            target_speed_min=-float(self.v_max.item()),
            target_speed_max=float(self.v_max.item()),
            target_steering_min=float(self.steering_min.item()),
            target_steering_max=float(self.steering_max.item()),
            min_acc=self.a_min,
            max_acc=self.a_max,
            min_steering_rate=self.steering_rate_min,
            max_steering_rate=self.steering_rate_max,
        )
        u_applied = (
            u_applied_t.detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
            .reshape(n_envs, n_agents, 2)
        )

        v_left = np.zeros((n_envs, n_agents), dtype=np.float64)
        v_right = np.zeros((n_envs, n_agents), dtype=np.float64)
        v_pair = np.zeros((n_envs, n_agents), dtype=np.float64)
        lane_stencil_offsets = np.array(
            [
                [0.0, 0.0],
                [self.dx, 0.0],
                [0.0, self.dy],
                [-self.dx, 0.0],
                [0.0, -self.dy],
                [self.dx, self.dy],
                [self.dx, -self.dy],
                [-self.dx, self.dy],
                [-self.dx, -self.dy],
            ],
            dtype=np.float64,
        )

        pseudo_time = 0.0
        lambda_ttcbf = float(self.lambda_ttcbf_predefined)
        local_boundary_cache = {}
        for env_idx in range(n_envs):
            for agent_idx in range(n_agents):
                query_points = (
                    circles_xy[env_idx, agent_idx, :, None, :]
                    + lane_stencil_offsets[None, :, :]
                )
                time_pseudo_dis_start = time.time()
                (
                    dleft_results,
                    dright_results,
                ) = self._get_local_softmin_distances_with_derivatives_batch_for_env_agent(
                    env_idx,
                    agent_idx,
                    query_points,
                    dx=self.dx,
                    dy=self.dy,
                    local_boundary_cache=local_boundary_cache,
                )
                pseudo_time += time.time() - time_pseudo_dis_start

                left_margins = self._fixed_lambda_lane_margins_from_distance_results(
                    kins,
                    env_idx,
                    agent_idx,
                    u_applied[env_idx, agent_idx],
                    dleft_results,
                    lambda_ttcbf,
                )
                right_margins = self._fixed_lambda_lane_margins_from_distance_results(
                    kins,
                    env_idx,
                    agent_idx,
                    u_applied[env_idx, agent_idx],
                    dright_results,
                    lambda_ttcbf,
                )
                v_left[env_idx, agent_idx] = np.min(left_margins)
                v_right[env_idx, agent_idx] = np.min(right_margins)

        d_safe = float(2.0 * self.circle_radius + self.safety_buffer)
        d_safe_sq = d_safe * d_safe
        dt_sq = self.dt_taylor * self.dt_taylor
        for i in range(n_agents - 1):
            ui = u_applied[:, i, :]
            for j in range(i + 1, n_agents):
                uj = u_applied[:, j, :]
                for ci in range(n_circles):
                    pi = circles_xy[:, i, ci, :]
                    vxi = kins["dx"][:, i, ci]
                    vyi = kins["dy"][:, i, ci]
                    aix = kins["a_ddx"][:, i, ci, :]
                    aiy = kins["a_ddy"][:, i, ci, :]
                    cix = kins["c_ddx"][:, i, ci]
                    ciy = kins["c_ddy"][:, i, ci]
                    for cj in range(n_circles):
                        pj = circles_xy[:, j, cj, :]
                        dx = pi[:, 0] - pj[:, 0]
                        dy = pi[:, 1] - pj[:, 1]
                        vxj = kins["dx"][:, j, cj]
                        vyj = kins["dy"][:, j, cj]
                        ajx = kins["a_ddx"][:, j, cj, :]
                        ajy = kins["a_ddy"][:, j, cj, :]
                        cjx = kins["c_ddx"][:, j, cj]
                        cjy = kins["c_ddy"][:, j, cj]

                        vrel_x = vxi - vxj
                        vrel_y = vyi - vyj
                        h = dx * dx + dy * dy - d_safe_sq
                        dh = 2.0 * (dx * vrel_x + dy * vrel_y)
                        A_i = dt_sq * (dx[:, None] * aix + dy[:, None] * aiy)
                        A_j = -dt_sq * (dx[:, None] * ajx + dy[:, None] * ajy)
                        const_ddh = 2.0 * (vrel_x * vrel_x + vrel_y * vrel_y) + 2.0 * (
                            dx * (cix - cjx) + dy * (ciy - cjy)
                        )
                        g_pair = (
                            np.einsum("ei,ei->e", A_i, ui)
                            + np.einsum("ei,ei->e", A_j, uj)
                            + lambda_ttcbf * h
                            + dh * self.dt_taylor
                            + 0.5 * dt_sq * const_ddh
                        )
                        violation = np.minimum(g_pair, 0.0)
                        v_pair[:, i] = np.minimum(v_pair[:, i], violation)
                        v_pair[:, j] = np.minimum(v_pair[:, j], violation)

        h_nom_lane, h_nom_pair = self.cbf_margin_reward_scales()
        reward_info = self.env.base_env.scenario_name.reward_info
        reward_info.rew_near_left_lane[:n_envs, :] = torch.as_tensor(
            np.clip(v_left / h_nom_lane, -1.0, 0.0),
            device=self.device,
            dtype=torch.float32,
        )
        reward_info.rew_near_right_lane[:n_envs, :] = torch.as_tensor(
            np.clip(v_right / h_nom_lane, -1.0, 0.0),
            device=self.device,
            dtype=torch.float32,
        )
        reward_info.rew_near_other_agents[:n_envs, :] = torch.as_tensor(
            np.clip(v_pair / h_nom_pair, -1.0, 0.0),
            device=self.device,
            dtype=torch.float32,
        )

        if controllers is not None:
            for controller in controllers:
                controller.time_pseudo_dis = 0.0
        self.time_pseudo_dis = pseudo_time
        return pseudo_time

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

        The barriers follow the CBF-QP footprints. With
        ``lane_cbf_footprint="rectangle_sat"``, each side has one exact
        rectangle-polyline margin ``B(z) - (1 - lambda) B(y)`` under the
        two-step Euler map (see ``_lane_sat_nominal_margins``); otherwise, each
        covering circle has a Taylor CBF margin. With
        ``veh_veh_cbf_footprint="rectangle_sat"``, each pair has one exact
        rectangle-pair margin, keyed ``(i, j, 0, 0)``: the joint condition in
        centralized mode, and the ego's share of the cross-pair rows in fully
        decentralized mode (see ``_pair_sat_cross_nominal_margins``);
        otherwise, each circle pair has a Taylor CBF margin. The rectangle_sat
        margins are in m, the circle margins in m^2.

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
                ((n, 1) with the rectangle_sat lane footprint)
            lane_R_margin: (n, n_circles) margins for right boundary CBFs
                ((n, 1) with the rectangle_sat lane footprint)
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

        kins_all = [
            self.linearized_center_kinematics_coeffs(state_i) for state_i in states
        ]
        local_boundary_cache = {}

        is_lane_sat = self.lane_cbf_footprint == "rectangle_sat"
        is_pair_sat = self.veh_veh_cbf_footprint == "rectangle_sat"
        if is_lane_sat or is_pair_sat:
            states_np = {
                i: np.asarray([float(value) for value in states[i][:5]])
                for i in range(n)
            }
            u_sat = {
                i: self._sat_nominal_control(states_np[i], u_applied[i])
                for i in range(n)
            }

        # --------- Lane CBF margins g_L >= 0 and g_R >= 0 ---------
        if is_lane_sat:
            # One exact barrier per side instead of one per covering circle
            lane_L_margin = np.zeros((n, 1), dtype=np.float64)
            lane_R_margin = np.zeros((n, 1), dtype=np.float64)
            time_pseudo_dis_start = time.time()
            for i in range(n):
                lam_i = (
                    self._get_agent_cbf_lambda(learned_lambdas, i)
                    if self.parameters.is_learn_cbf_lambda
                    else float(self.lambda_ttcbf_predefined)
                )
                (
                    lane_L_margin[i, 0],
                    lane_R_margin[i, 0],
                ) = self._lane_sat_nominal_margins(i, states_np[i], u_sat[i], lam_i)
            self.time_pseudo_dis += time.time() - time_pseudo_dis_start
        else:
            lane_L_margin = np.zeros((n, n_circles), dtype=np.float64)
            lane_R_margin = np.zeros((n, n_circles), dtype=np.float64)
            use_batched_local_softmin = (
                self.parameters.lane_distance_type == "softmin"
                and self.parameters.is_use_sampled_lane_boundary_for_cbf
            )
            lane_stencil_offsets = None
            if use_batched_local_softmin:
                lane_stencil_offsets = torch.tensor(
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

            for i in range(n):
                kins_i = kins_all[i]
                ui = u_applied[i, :]  # shape (2,)
                if use_batched_local_softmin:
                    query_points = circles_all[i][:, 0:2].unsqueeze(
                        1
                    ) + lane_stencil_offsets.unsqueeze(0)
                    time_pseudo_dis_start = time.time()
                    (
                        dleft_results_all,
                        dright_results_all,
                    ) = self._get_local_softmin_distances_with_derivatives_batch(
                        i,
                        query_points,
                        dx=self.dx,
                        dy=self.dy,
                        local_boundary_cache=local_boundary_cache,
                    )
                    self.time_pseudo_dis += time.time() - time_pseudo_dis_start
                    smL_all = dleft_results_all[0] - self.circle_radius**2
                    gL_all = dleft_results_all[1]
                    HL_all = dleft_results_all[2]
                    smR_all = dright_results_all[0] - self.circle_radius**2
                    gR_all = dright_results_all[1]
                    HR_all = dright_results_all[2]

                for ci in range(n_circles):
                    if use_batched_local_softmin:
                        smL = smL_all[ci]
                        gL = gL_all[ci]
                        HL = HL_all[ci]
                        smR = smR_all[ci]
                        gR = gR_all[ci]
                        HR = HR_all[ci]
                    else:
                        circle_pos_i = circles_all[i][ci][0:2]
                        (
                            smL,
                            gL,
                            HL,
                            smR,
                            gR,
                            HR,
                        ) = self.estimate_agent_2_lane_safety_margin(
                            circle_pos_i,
                            path_ids[i],
                            agent_idx=i,
                            local_boundary_cache=local_boundary_cache,
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
        # Neighbor cap of the directed (fully decentralized) pair margins
        e_cap_user = getattr(self.parameters, "max_cross_neighbors", None)
        if e_cap_user is None:
            e_cap = max(0, n - 1)
        else:
            e_cap = min(max(0, n - 1), max(0, int(e_cap_user)))

        if is_pair_sat:
            # One exact rectangle-pair barrier per pair, keyed (i, j, 0, 0)
            if self.parameters.is_using_centralized_cbf:
                pairs = [dict(a=i, b=j) for i in range(n - 1) for j in range(i + 1, n)]
                lams = [
                    (
                        float(
                            (
                                self._get_agent_cbf_lambda(learned_lambdas, p["a"])
                                + self._get_agent_cbf_lambda(learned_lambdas, p["b"])
                            )
                            / 2.0
                        )
                        if self.parameters.is_learn_cbf_lambda
                        else float(self.lambda_ttcbf_predefined)
                    )
                    for p in pairs
                ]
                if pairs:
                    margins = self._pair_sat_joint_nominal_margins(
                        pairs, states_np, u_sat, lams
                    )
                    for p, g_pair in zip(pairs, margins):
                        pair_margin[(p["a"], p["b"], 0, 0)] = float(g_pair)
            else:
                uncontrolled_set = self._get_uncontrolled_agent_set()
                for i in range(n):
                    nbrs = [
                        j
                        for j in range(n)
                        if j != i
                        and _euclid(centers_flat[i], centers_flat[j])
                        <= cbf_coupling_range + 1e-9
                    ][:e_cap]
                    if not nbrs:
                        continue
                    shares = [
                        (
                            1.0
                            if j in uncontrolled_set
                            else self._cross_cbf_responsibility_share(
                                states[i], states[j]
                            )
                        )
                        for j in nbrs
                    ]
                    lam_i = (
                        self._get_agent_cbf_lambda(learned_lambdas, i)
                        if self.parameters.is_learn_cbf_lambda
                        else float(self.lambda_ttcbf_predefined)
                    )
                    margins = self._pair_sat_cross_nominal_margins(
                        i, nbrs, shares, states_np, u_sat[i], lam_i, uncontrolled_set
                    )
                    for j, g_pair in zip(nbrs, margins):
                        pair_margin[(i, j, 0, 0)] = float(g_pair)
        elif self.parameters.is_using_centralized_cbf:
            for i in range(n - 1):
                kins_i = kins_all[i]
                ui = u_applied[i, :]

                for j in range(i + 1, n):
                    lam_i = self._get_agent_cbf_lambda(learned_lambdas, i)
                    lam_j = self._get_agent_cbf_lambda(learned_lambdas, j)
                    lambda_ttcbf = (
                        float((lam_i + lam_j) / 2.0)
                        if self.parameters.is_learn_cbf_lambda
                        else float(self.lambda_ttcbf_predefined)
                    )

                    kins_j = kins_all[j]
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
            for i in range(n):
                kins_i = kins_all[i]
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
                    kins_j = kins_all[j]
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
        result = {
            "lane_L_margin": lane_L_margin,
            "lane_R_margin": lane_R_margin,
            "pair_margin": pair_margin,
            "pair_mode": pair_mode,
        }
        self._last_nominal_cbf_margins = result
        return result

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

        Finally, each channel is divided by its normalizer ``h_nom`` (see
        ``cbf_margin_reward_scales``) and clipped to ``[-1, 0]``.
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

        h_nom_lane, h_nom_pair = self.cbf_margin_reward_scales()

        r_left = to_reward(vL_agent, h_nom_lane)
        r_right = to_reward(vR_agent, h_nom_lane)
        r_pair = to_reward(vP_agent, h_nom_pair)

        return r_left, r_right, r_pair

    def cbf_margin_reward_scales(self) -> tuple[float, float]:
        """Normalizers (lane, pair) of the CBF-margin rewards in the margins' units.

        Each normalizer is h_nom (a unitless factor; None: H_NOM) times the
        largest margin drop one step can cause while the barrier is still
        nonnegative, i.e., closing on the barrier at the speed limit v_max:
        - rectangle_sat (B in m over one step dt): v_max dt for a lane and
          2 v_max dt for a pair (both vehicles closing).
        - circle (h = d^2 - d_safe^2 in m^2 over the Taylor horizon dt_taylor,
          with dh/dt = 2 d dd/dt and d = d_safe on the barrier):
          2 r v_max dt_taylor for a lane (circle radius r) and
          2 d_safe 2 v_max dt_taylor for a pair.
        A margin of minus the normalizer (reward -1) thus means closing at full speed
        when the barrier requires holding, for any agent profile and dt.
        """
        h_nom = H_NOM if self.parameters.h_nom is None else float(self.parameters.h_nom)
        v_max = max(abs(float(self.v_max)), abs(float(self.v_min)))
        if self.lane_cbf_footprint == "rectangle_sat":
            h_nom_lane = h_nom * v_max * self.dt
        else:
            h_nom_lane = h_nom * 2.0 * self.circle_radius * v_max * self.dt_taylor
        if self.veh_veh_cbf_footprint == "rectangle_sat":
            h_nom_pair = h_nom * 2.0 * v_max * self.dt
        else:
            d_safe = 2.0 * self.circle_radius + self.safety_buffer
            h_nom_pair = h_nom * 2.0 * d_safe * 2.0 * v_max * self.dt_taylor
        return h_nom_lane, h_nom_pair

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
