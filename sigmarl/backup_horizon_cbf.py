# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Backup-horizon CBF (BH-CBF) rows (Parameters.is_backup_horizon_cbf).

The exact two-step rows of the rectangle_sat pipeline only look at the pose
z at k+2. The backup-horizon rows extend them along a tail in which every
vehicle applies the backup policy of the exact shield (route-tracking
emergency braking) after its QP control u_k:
  Psi_t(u) = pi(phi^t(F(x, u))),  t = 0..N  (Psi_0 = y at k+1, Psi_1 = z),
with the stopping horizon N = ceil(v / (|a_min| dt)) + 2. With the backup
value B_e(X) = min_{t=0..N} m_e(Psi^b_t) of a constraint e (the minimum of
its margin along the braking tail Psi^b_t = pi(phi^{t+1}(X)) from X), the
rows are

  joint pair or lane:  m_e(Psi_t(u)) >= (1 - lambda) B_e(X_k),  t = 1..N  (BH)
  cross pair (budget): <w_t, q_c(Psi_t(u_i))> - min_c' <w_t, q_c'(Psi^b_{i,t})>
                       >= -s_ij (sd^b_t - eps_0 - (1 - lambda) B^E_ij)    (BD)

for every corner c of the ego in BD. The pair margin m_e of the joint rows is
the signed distance of the two rectangles minus the pair buffer, the same
margin as the budget rows use. The joint backup control satisfies (BH),
and the ego's own backup control satisfies (BD) whatever the other vehicle
does, so the conditions stay feasible from every recoverable state, and any
control that satisfies them keeps the state recoverable (recursive
feasibility; the exact backup shield is the case lambda = 1). BD bounds how
far the ego's corners may move toward the other vehicle, beyond the closest
corner of its own braking tail, along the witness direction w_t of the signed
distance sd of the two braking tails; the two shares s_ij + s_ji = 1 add up
to the pair condition in the signed distance without a mixed term, so the
budget rows of both vehicles imply the joint row of the pair.

Rows are linearized at the QP's linearization controls with forward
differences through the rollout and repaired at the QP solution like the
two-step rows (see CBFQP._solve_qp_with_sat_repairs): every tail step of
every constraint is checked exactly. If the tail rows remain violated after
the repairs, the controls of the vehicles involved are replaced by their
backup controls until the exact check passes (at worst the joint backup).
All rollouts of a QP are computed in one vectorized call.

The methods with the suffix ``_steps`` are generators that yield their
rollouts as RolloutRequest and receive the poses back, so that a driver can
serve the requests of several QPs in one vectorized call
(rollout_requests); run_steps serves them one at a time. Both give the same
poses bit for bit.
"""

import os
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from sigmarl.constants import LENGTH_SCALE

# Upper bound of one-sided QP rows (as in sigmarl/cbf_qp_proxsuite.py)
PROXQP_INF = 1e20
# Yielded by CBFQP.update_qp_steps after the QP of an agent is solved and
# before the agent writes its results, so that a driver of several agents
# keeps the order of their writes (answered with None)
WRITE_BARRIER = "write_barrier"


class RolloutRequest(NamedTuple):
    """Arguments of one BackupHorizonCBF.rollout call."""

    horizon: "BackupHorizonCBF"
    X0: np.ndarray
    U0: np.ndarray
    routes: list
    hold: np.ndarray
    n_steps: int
    # Vehicle parameters of the rows (ScenarioRoadTraffic.vehicle_param_rows); None:
    # the nominal ones
    params: dict = None


def rollout_requests(requests) -> list:
    """Poses of several rollout requests (of controllers of the same
    scenario) in one vectorized rollout; each equals the poses of its own
    BackupHorizonCBF.rollout call bit for bit."""
    return requests[0].horizon.rollout_batch(requests)


def run_steps(steps):
    """Run a ``_steps`` generator, serving each rollout request at once.

    Returns:
        The generator's return value.
    """
    reply = None
    while True:
        try:
            request = steps.send(reply)
        except StopIteration as stop:
            return stop.value
        reply = None if request is WRITE_BARRIER else rollout_requests([request])[0]


def take_params(params, rows):
    """Vehicle parameters (ScenarioRoadTraffic.vehicle_param_rows) of some rows; None
    stays None (nominal parameters)."""
    if params is None:
        return None
    return {key: value[rows] for key, value in params.items()}


def param_halves(params) -> np.ndarray:
    """Half lengths and half widths [m, 2] of vehicles with parameters params."""
    return 0.5 * np.column_stack([params["length"], params["width"]])


def rectangle_corners(poses: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Corners [..., 4, 2] of rectangles with half sizes (l, w) at poses [..., 3].

    half: [2] for all rectangles, or [..., 2] per rectangle (broadcast against
    the leading dimensions of poses).
    """
    cos_psi, sin_psi = np.cos(poses[..., 2]), np.sin(poses[..., 2])
    e_lon = np.stack([cos_psi, sin_psi], axis=-1)[..., None, :]
    e_lat = np.stack([-sin_psi, cos_psi], axis=-1)[..., None, :]
    signs = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    if np.ndim(half) == 1:
        local = signs * half
        return poses[..., None, 0:2] + local[:, 0:1] * e_lon + local[:, 1:2] * e_lat
    local = signs * np.asarray(half)[..., None, :]  # [..., 4, 2]
    return poses[..., None, 0:2] + local[..., 0:1] * e_lon + local[..., 1:2] * e_lat


def signed_distance(
    poses_a: np.ndarray, poses_b: np.ndarray, half: np.ndarray, half_b=None
):
    """Signed distance of rectangle pairs and its witness direction.

    The signed distance is the Euclidean distance of disjoint rectangles and
    minus the penetration depth of overlapping ones. It equals the largest
    projection gap over all unit directions and is attained along the
    witness direction (pointing from b to a).

    Args:
        poses_a, poses_b: poses [K, 3].
        half: half length and half width [2] of all rectangles, or [K, 2] of
            the rectangles a.
        half_b: half sizes [K, 2] of the rectangles b; None: half.

    Returns:
        Signed distances [K] and unit witness directions [K, 2].
    """
    corners_a = rectangle_corners(poses_a, half)
    corners_b = rectangle_corners(poses_b, half if half_b is None else half_b)
    k = np.arange(poses_a.shape[0])

    def body_axes(poses):
        cos_psi, sin_psi = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        return [
            np.stack([cos_psi, sin_psi], axis=1),
            np.stack([-sin_psi, cos_psi], axis=1),
        ]

    # Separating axis test over the four body axes: exact for the overlap
    # depth and for whether the rectangles are disjoint
    axes = np.stack(body_axes(poses_a) + body_axes(poses_b), axis=1)  # [K, 4, 2]
    proj_a = np.einsum("kxd,kcd->kxc", axes, corners_a)
    proj_b = np.einsum("kxd,kcd->kxc", axes, corners_b)
    a_ahead = proj_a.min(axis=2) - proj_b.max(axis=2)
    b_ahead = proj_b.min(axis=2) - proj_a.max(axis=2)
    gaps = np.maximum(a_ahead, b_ahead)
    best = gaps.argmax(axis=1)
    sat_gap = gaps[k, best]
    sign = np.where(a_ahead[k, best] >= b_ahead[k, best], 1.0, -1.0)
    witness = axes[k, best] * sign[:, None]
    distance = sat_gap.copy()

    disjoint = sat_gap > 0.0
    if disjoint.any():

        def corner_to_edge(points, polygon):
            """Vectors [K, 16, 2] from the closest edge points of polygon to
            each corner of points."""
            start = polygon
            segment = np.roll(polygon, -1, axis=1) - polygon
            rel = points[:, :, None, :] - start[:, None, :, :]  # [K, 4, 4, 2]
            tau = np.clip(
                np.einsum("kced,ked->kce", rel, segment)
                / np.einsum("ked,ked->ke", segment, segment)[:, None, :],
                0.0,
                1.0,
            )
            return (rel - tau[..., None] * segment[:, None]).reshape(-1, 16, 2)

        vectors = np.concatenate(
            [
                corner_to_edge(corners_a, corners_b),
                -corner_to_edge(corners_b, corners_a),
            ],
            axis=1,
        )  # from b to a
        lengths = np.linalg.norm(vectors, axis=2)
        closest = lengths.argmin(axis=1)
        euclid = lengths[k, closest]
        direction = vectors[k, closest] / np.maximum(euclid, 1e-12)[:, None]
        distance = np.where(disjoint, euclid, sat_gap)
        witness = np.where(disjoint[:, None], direction, witness)
    return distance, witness


def signed_distance_canonical(poses_a, poses_b, half, is_a_first, half_b=None):
    """signed_distance with the pair evaluated in a fixed order: (a, b) where
    is_a_first [K] and (b, a) elsewhere, the witness negated back to point
    from b to a. Both vehicles of a pair thus get the same floating-point
    values (with opposite witnesses). Half sizes as in signed_distance."""
    first = np.where(is_a_first[:, None], poses_a, poses_b)
    second = np.where(is_a_first[:, None], poses_b, poses_a)
    if half_b is None:
        distance, witness = signed_distance(first, second, half)
    else:
        distance, witness = signed_distance(
            first,
            second,
            np.where(is_a_first[:, None], half, half_b),
            np.where(is_a_first[:, None], half_b, half),
        )
    return distance, np.where(is_a_first[:, None], witness, -witness)


class BackupHorizonCBF:
    """Backup-horizon tail rows of one CBFQP controller.

    The pair and lane hooks of the controller register their constraints
    (``register_pairs``, ``register_lane``) while a QP is built; ``build``
    adds the initial rows and returns the tail block of the QP, which
    ``check`` repairs at each solution and ``fallback`` enforces at the end
    (see CBFQP._solve_qp_with_sat_repairs).
    """

    # Slack coefficient of the tail rows: they share the slack of their pair
    # or lane side, scaled so that a violated tail row barely relaxes the
    # two-step rows of the same constraint
    SLACK_GAIN = 10.0
    # Tail steps whose margin at the linearization controls is below this
    # band [m] get initial rows (at most MAX_INITIAL_ROWS per constraint); the
    # others are checked after the solve. BH_ROW_BAND=inf and a large
    # BH_MAX_INITIAL_ROWS put every tail step into the first QP (no lazy rows)
    ROW_BAND = float(os.environ.get("BH_ROW_BAND", 0.05 * LENGTH_SCALE))
    MAX_INITIAL_ROWS = int(os.environ.get("BH_MAX_INITIAL_ROWS", 4))
    # Forward-difference step of the controls (acceleration, steering rate)
    FD_STEP = 1e-4
    # Extra route distance [m] of the lane window beyond the backup's reach
    LANE_WINDOW_MARGIN = 0.05 * LENGTH_SCALE
    # Entry kinds, evaluated by _kind_values (subclasses may add kinds)
    KINDS = ("pair", "budget", "lane")
    # Accept-or-replace (model predictive shielding): skip the QP and keep the
    # nominal control unless the exact check fails (see
    # CBFQP._solve_qp_with_sat_repairs_steps)
    IS_ACCEPT_OR_REPLACE = False

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.half = np.array([0.5 * float(ctrl.length), 0.5 * float(ctrl.width)])
        self.diagonal = 2.0 * float(np.linalg.norm(self.half))
        # Diagnostics, written to the cbf_statistics file with the lane_sat stats
        self.stats = ctrl.lane_sat_stats
        self._pending_pairs = []
        self._pending_lanes = []

    # ---------------------------------------------------------------
    # Registration (while a QP is built)
    # ---------------------------------------------------------------

    def register_pairs(self, pairs, states_np, u_lin, fixed_agents):
        """Register the pairs of a QP (arguments of CBFQP._append_pair_sat_rows)."""
        for pair in pairs:
            if pair["lambda_idx"] is not None:
                raise NotImplementedError(
                    "The backup-horizon rows support only a fixed lambda."
                )
        self._pending_pairs.append((pairs, states_np, dict(u_lin), set(fixed_agents)))

    def register_lane(
        self,
        env_idx: int,
        agent_idx: int,
        state_np,
        u_lin,
        ui_slice,
        slack_idx,
        lambda_idx,
    ):
        """Register the lane constraints of one agent of a QP."""
        if lambda_idx is not None:
            raise NotImplementedError(
                "The backup-horizon rows support only a fixed lambda."
            )
        self._pending_lanes.append(
            dict(
                env_idx=env_idx,
                agent=agent_idx,
                state=np.asarray(state_np, dtype=np.float64),
                u_lin=np.asarray(u_lin, dtype=np.float64),
                slice=ui_slice,
                slack=slack_idx,
            )
        )

    def params_of(self, agents):
        """Vehicle parameters of agents (see CBFQP._vehicle_params); None while
        every vehicle has the nominal parameters."""
        return self.ctrl._vehicle_params(agents)

    def pair_values(
        self, poses_a: np.ndarray, poses_b: np.ndarray, half_a=None, half_b=None
    ) -> np.ndarray:
        """Pair margins [K] of rectangle pairs at poses [K, 3]: their signed
        distance minus the pair buffer (the margin of the joint rows). half_a,
        half_b: half sizes [K, 2] of the rectangles; None: the nominal ones."""
        if half_a is None:
            distance, _ = signed_distance(poses_a, poses_b, self.half)
        else:
            distance, _ = signed_distance(poses_a, poses_b, half_a, half_b)
        return distance - float(self.ctrl.pair_safety_buffer)

    def pair_distance_witness(self, poses_a, poses_b, half, is_a_first, half_b=None):
        """Distances [K] of the footprints of pairs (without the pair buffer)
        and their witness directions [K, 2] from b to a, evaluated in a fixed
        order (see signed_distance_canonical): the geometry of the budget rows."""
        return signed_distance_canonical(poses_a, poses_b, half, is_a_first, half_b)

    def support_points(self, poses: np.ndarray, half) -> np.ndarray:
        """Points [..., P, 2] of the footprints at poses [..., 3] whose smallest
        projection on a direction is the footprint's support up to a constant
        (the budget rows compare it with that of the braking tail): the
        corners of the rectangles."""
        return rectangle_corners(poses, half)

    def reach(self, speed, params=None):
        """Upper bound [m] of the center displacement of a vehicle over the
        poses of its tail (k+1, k+2, then backup braking until standstill);
        elementwise for an array of speeds (of vehicles with parameters
        params; None: the nominal ones)."""
        ctrl = self.ctrl
        a_max = float(ctrl.a_max) if params is None else params["max_acc"]
        a_min = float(ctrl.a_min) if params is None else params["min_acc"]
        v = np.maximum(0.0, speed)
        v1 = v + ctrl.dt * a_max
        return v * ctrl.dt + 2.0 * v1 * ctrl.dt + v1 * v1 / (-2.0 * a_min)

    def coupling_ranges(
        self, speeds: np.ndarray, agent: int, others: np.ndarray
    ) -> np.ndarray:
        """Center distances [m] [K] beyond which the backup tails of a vehicle
        and of each of others [K] cannot come within the pair safety buffer
        (decentralized neighbor set of Theorem 4). speeds: [n] speeds of all
        vehicles. Elementwise, so each value equals that of the pair alone."""
        rows = np.concatenate([[agent], others]).astype(int)
        params = self.params_of(rows)
        reach = self.reach(speeds[rows], params)
        buffer = float(self.ctrl.pair_safety_buffer)
        if params is None:
            return reach[0] + reach[1:] + self.diagonal + buffer
        circumradii = np.linalg.norm(param_halves(params), axis=1)
        return reach[0] + reach[1:] + (circumradii[0] + circumradii[1:]) + buffer

    def entry_values(self, env_idx: int, agent: int, others, held) -> tuple:
        """Smallest margins of the constraints of one vehicle at the current
        joint state, at the current poses and along the braking tails (a
        certified entry, e.g., of a log replay, needs both >= 0: x in C).

        The constraints and their evaluation are those of the filter's rows:
        the vehicle's lane sides on the window that covers its backup's reach
        (rectangle_sat lane barrier, buffers included) and its pairs with the
        other vehicles (signed distance minus the pair buffer). Every vehicle
        follows the route-tracking braking backup, except the held ones
        (non-reactive, e.g., replayed), which keep zero input (constant speed
        and steering) as in the filter. The other vehicles' own constraints
        are not evaluated.

        Args:
            env_idx: env of the vehicles.
            agent: agent index of the vehicle.
            others: agent indices of the other vehicles on the map.
            held: the subset of others predicted with zero input.

        Returns:
            (lane value, pair value) [m]; inf without facets or other vehicles.
        """
        ctrl = self.ctrl
        indices = [agent] + list(others)
        world_agents = ctrl.scenario.world.agents
        X = np.array(
            [
                [
                    float(world_agents[i].state.pos[env_idx, 0]),
                    float(world_agents[i].state.pos[env_idx, 1]),
                    float(world_agents[i].state.rot[env_idx, 0]),
                    float(world_agents[i].state.speed[env_idx, 0]),
                    float(world_agents[i].state.steering[env_idx, 0]),
                ]
                for i in indices
            ],
            dtype=np.float64,
        )
        held = set(held)
        hold = np.array([i in held for i in indices], dtype=bool)
        routes = [ctrl._shield_route(i) for i in indices]
        params = self.params_of(indices)
        U0 = np.where(hold[:, None], 0.0, self.backup_controls(X, routes, params))
        tails = self.rollout(
            X,
            U0,
            routes,
            hold,
            self.horizon(X[~hold, 3], take_params(params, ~hold)),
            params,
        )
        poses = np.concatenate([X[None, :, 0:3], tails])  # [T + 1, m, 3]

        a_max = float(ctrl.a_max) if params is None else params["max_acc"][0]
        a_min = float(ctrl.a_min) if params is None else params["min_acc"][0]
        v1 = max(0.0, X[0, 3] + ctrl.dt * a_max)
        reach = 2.0 * v1 * ctrl.dt + v1 * v1 / (-2.0 * a_min) + self.LANE_WINDOW_MARGIN
        boundaries = ctrl.lane_sat.local_boundaries(
            ctrl.scenario.world_state, env_idx, agent, ahead_distance=reach
        )
        halves = None if params is None else param_halves(params)
        lane_value = np.inf
        for side, points in enumerate(boundaries):
            facets = ctrl.lane_sat.facets(points, side)
            if facets["start"].shape[0] > 0:
                values = ctrl.lane_sat.values(
                    facets, side, poses[:, 0], None if halves is None else halves[0]
                )
                lane_value = min(lane_value, float(values.min()))

        pair_value = np.inf
        n_others = len(indices) - 1
        if n_others > 0:
            ego = np.repeat(poses[:, :1], n_others, axis=1)
            pair_halves = (
                ()
                if halves is None
                else (
                    np.tile(halves[:1], (poses.shape[0] * n_others, 1)),
                    np.tile(halves[1:], (poses.shape[0], 1)),
                )
            )
            values = self.pair_values(
                ego.reshape(-1, 3), poses[:, 1:].reshape(-1, 3), *pair_halves
            ).reshape(poses.shape[0], n_others)
            # Each pair up to its own stopping horizon (that of a QP with the
            # two vehicles); it matters only for a held vehicle, which keeps
            # moving after the others stopped
            for k in range(n_others):
                rows = np.array([0, k + 1])[~hold[[0, k + 1]]]
                n_steps = self.horizon(X[rows, 3], take_params(params, rows))
                pair_value = min(pair_value, float(values[: n_steps + 2, k].min()))
        return lane_value, pair_value

    # ---------------------------------------------------------------
    # Rollouts
    # ---------------------------------------------------------------

    def horizon(self, speeds, params=None) -> int:
        """Stopping horizon of the backup from these speeds [m/s] of vehicles
        with parameters params (see CBFQP._shield_horizon): the tail steps
        after which every vehicle that starts the tail at k+1 has stopped."""
        return self.ctrl._shield_horizon(speeds, params)

    def backup_controls(self, X: np.ndarray, routes, params=None) -> np.ndarray:
        """Backup controls [m, 2] at states X [m, 5] with one route and one row
        of vehicle parameters (params) per row."""
        ctrl = self.ctrl
        return ctrl._shield_backup_controls(
            X,
            ctrl._shield_route_bundle(routes),
            np.ones(X.shape[0], dtype=bool),
            params=params,
        )

    def rollout(self, X0, U0, routes, hold, n_steps: int, params=None) -> np.ndarray:
        """Poses [n_steps + 1, m, 3] of m rollouts that apply U0 [m, 2] at
        X0 [m, 5] and then the backup policy, or zero input for rows with
        hold (vehicles predicted with constant speed and steering). Index t
        is the pose after the first control and t tail steps. params: vehicle
        parameters of the rows."""
        return self.rollout_batch(
            [RolloutRequest(self, X0, U0, routes, hold, n_steps, params)]
        )[0]

    def rollout_batch(self, requests) -> list:
        """Poses of several rollout requests (see rollout) in one vectorized
        rollout over all their rows, up to the longest horizon.

        Rows do not interact, and each request keeps its own route bundle for
        the station interpolation, so each request gets the poses of its own
        rollout bit for bit (as a fresh C-contiguous array, the layout of a
        single rollout).
        """
        ctrl = self.ctrl
        if len(requests) == 1:
            X0, U0, hold = requests[0].X0, requests[0].U0, requests[0].hold
            params = requests[0].params
            bundle = ctrl._shield_route_bundle(requests[0].routes)
        else:
            X0 = np.concatenate([request.X0 for request in requests])
            U0 = np.concatenate([request.U0 for request in requests])
            hold = np.concatenate([request.hold for request in requests])
            is_nominal = [request.params is None for request in requests]
            assert all(is_nominal) or not any(is_nominal), "Mixed vehicle parameters."
            params = (
                None
                if is_nominal[0]
                else {
                    key: np.concatenate([request.params[key] for request in requests])
                    for key in requests[0].params
                }
            )
            bundle = ctrl._shield_merge_route_bundles(
                [ctrl._shield_route_bundle(request.routes) for request in requests]
            )
        n_steps = max(request.n_steps for request in requests)
        is_any_hold = bool(hold.any())
        X = ctrl._shield_euler_step(
            X0, ctrl._shield_saturate(X0, U0, params=params), params
        )
        poses = np.empty((n_steps + 1, X0.shape[0], 3))
        for t in range(n_steps + 1):
            poses[t] = X[:, 0:3]
            if t == n_steps:
                break
            # The backup controls are saturated already; only rows with hold
            # change before the second saturation
            bounds = ctrl._shield_input_bounds(X, params)
            U = ctrl._shield_backup_controls(X, bundle, None, bounds, params)
            if is_any_hold:
                U[hold] = 0.0
                U = ctrl._shield_saturate(X, U, bounds)
            X = ctrl._shield_euler_step(X, U, params)
        if len(requests) == 1:
            return [poses]
        rows = np.cumsum([0] + [request.X0.shape[0] for request in requests])
        return [
            np.ascontiguousarray(poses[: request.n_steps + 1, start:stop])
            for request, start, stop in zip(requests, rows[:-1], rows[1:])
        ]

    def fd_controls(self, X: np.ndarray, U: np.ndarray, params=None) -> tuple:
        """Forward-difference controls of agents at states X [m, 5] around U
        [m, 2] (vehicle parameters params): the perturbed controls [m, 2
        (column), 2] and the signed steps [m, 2]. A step points away from the
        input limit it would cross."""
        ctrl = self.ctrl
        steps = np.empty_like(U)
        perturbed = np.repeat(U[:, None, :], 2, axis=1)
        for col in range(2):
            plus = U.copy()
            plus[:, col] += self.FD_STEP
            applied = ctrl._shield_saturate(X, plus, params=params)
            is_free = np.abs(applied[:, col] - plus[:, col]) < 1e-12
            steps[:, col] = np.where(is_free, self.FD_STEP, -self.FD_STEP)
            perturbed[:, col, col] += steps[:, col]
        return perturbed, steps

    # ---------------------------------------------------------------
    # Tail block of a QP
    # ---------------------------------------------------------------

    def build(self, rows, lowers, uppers, n_vars: int):
        """Build the tail block of the registered constraints and add its
        initial rows; returns None if nothing is registered."""
        return run_steps(self.build_steps(rows, lowers, uppers, n_vars))

    def build_steps(self, rows, lowers, uppers, n_vars: int):
        """Generator version of build (see the module docstring)."""
        pending_pairs, pending_lanes = self._pending_pairs, self._pending_lanes
        self._pending_pairs, self._pending_lanes = [], []
        if not pending_pairs and not pending_lanes:
            return None
        ctrl = self.ctrl

        # Agent table: state, QP slice and linearization control (controlled
        # agents), and whether the agent is predicted with zero input
        agents = {}

        def add_agent(agent, state, ui_slice=None, u_lin=None, hold=False):
            entry = agents.setdefault(
                agent,
                dict(
                    state=np.asarray(state, dtype=np.float64),
                    slice=None,
                    u=None,
                    hold=hold,
                ),
            )
            if ui_slice is not None:
                entry["slice"] = ui_slice
                entry["u"] = np.asarray(u_lin, dtype=np.float64)

        for lane in pending_lanes:
            add_agent(lane["agent"], lane["state"], lane["slice"], lane["u_lin"])
        for pairs, states_np, u_lin, fixed_agents in pending_pairs:
            for pair in pairs:
                for agent, ui_slice in zip((pair["a"], pair["b"]), pair["slices"]):
                    if ui_slice is not None:
                        add_agent(agent, states_np[agent], ui_slice, u_lin[agent])
                    else:
                        add_agent(agent, states_np[agent], hold=agent in fixed_agents)
        order = sorted(agents)
        controlled = [k for k, a in enumerate(order) if agents[a]["slice"] is not None]
        if not controlled:
            return None
        X = np.array([agents[a]["state"][:5] for a in order])
        hold = np.array([agents[a]["hold"] for a in order], dtype=bool)
        routes = [ctrl._shield_route(a) for a in order]
        params = self.params_of(order)
        block = dict(
            kind="tail",
            index={agent: k for k, agent in enumerate(order)},
            X=X,
            hold=hold,
            routes=routes,
            # Vehicle parameters and half sizes [A, 2] of the agents (None:
            # nominal)
            params=params,
            half=None if params is None else param_halves(params),
            backup=self.backup_controls(X, routes, params),
            slices=[agents[a]["slice"] for a in order],
            controlled=np.array(controlled, dtype=int),
            slot={k: s for s, k in enumerate(controlled)},
            n_steps=(
                self.horizon(X[~hold, 3], take_params(params, ~hold))
                if (~hold).any()
                else 2
            ),
            entries=[],
            rhs=[],
            linearized_at=defaultdict(list),
        )
        U_lin = np.array([agents[order[k]]["u"] for k in controlled])
        # One rollout call: the reference rollouts of all agents (backup, or
        # zero input with hold) and the linearization controls with their
        # forward differences
        poses_c, steps = yield from self._rollouts_steps(
            block, U_lin, n_var=3, is_with_ref=True
        )

        self._build_pair_entries(block, pending_pairs)
        self._build_lane_entries(block, pending_lanes)
        if not block["entries"]:
            return None
        block["rhs"] = np.concatenate(block["rhs"], axis=0)  # [E, T]
        poses = self._assemble(block, poses_c)
        values = self._values(block, poses)
        margins = values - block["rhs"]
        selected = []
        for e in np.nonzero((margins < self.ROW_BAND).any(axis=1))[0]:
            candidates = np.nonzero(margins[e] < self.ROW_BAND)[0]
            candidates = candidates[np.argsort(margins[e, candidates])]
            selected += [(int(e), int(t)) for t in candidates[: self.MAX_INITIAL_ROWS]]
        if selected:
            grads = self._gradients(block, poses, steps, {e for e, _ in selected})
            for e, t in selected:
                self._append_row(
                    block, e, t, values, grads, U_lin, rows, lowers, uppers, n_vars
                )
            self.stats["bh_rows"] += len(selected)
        self.stats["bh_blocks"] += 1
        return block

    def _add_entries(self, block, kind: str, agents, slacks, rhs, **arrays):
        """Append the entries of one kind (agents: table indices [K, n])."""
        start = len(block["entries"])
        block["entries"] += [
            dict(agents=tuple(int(a) for a in agent_row), slack=slack)
            for agent_row, slack in zip(agents, slacks)
        ]
        rhs[:, 0] = -np.inf  # y does not depend on the controls
        block["rhs"].append(rhs)
        block[kind] = dict(idx=np.arange(start, len(block["entries"])), **arrays)

    def decay_rate(self, lam):
        """Decay rate lambda of the rows (subclasses may override it)."""
        return lam

    def _decay_rhs(self, ref_values: np.ndarray, lam) -> np.ndarray:
        """Right-hand sides [E, T] of the rows of E constraints from their
        margins [T, E] along the reference (backup) rollout: one value per
        constraint, (1 - lambda) times its backup value (BH)."""
        backup_value = ref_values.min(axis=0)
        return np.repeat(((1.0 - lam) * backup_value)[:, None], ref_values.shape[0], 1)

    def _build_cross_entries(self, block, ie, io, cross):
        """Budget rows (BD) of the cross pairs: ego ie and other agent io
        (table indices), with their pair dicts."""
        ref = block["ref"]
        n_t = ref.shape[0]
        lam = self.decay_rate(np.array([float(c[2]["lam"]) for c in cross]))
        share = np.array([float(c[2]["share"]) for c in cross])
        # Agent table indices follow the agent indices, so both vehicles of a
        # pair evaluate it in the same order
        if block["half"] is None:
            half_ego, halves = self.half, (self.half,)
        else:
            half_ego = block["half"][ie]
            halves = (
                np.tile(half_ego, (n_t, 1)),
                np.tile(block["half"][io], (n_t, 1)),
            )
        distance, witness = self.pair_distance_witness(
            ref[:, ie].reshape(-1, 3),
            ref[:, io].reshape(-1, 3),
            halves[0],
            np.tile(ie < io, n_t),
            *halves[1:],
        )
        margin = (distance - float(self.ctrl.pair_safety_buffer)).reshape(n_t, -1)
        backup_value = margin.min(axis=0)
        witness = witness.reshape(n_t, -1, 2)
        # Support of the ego's braking tail toward the other vehicle [T, K]:
        # the projection of its closest corner on the witness
        ref_support = np.einsum(
            "tkd,tkcd->tkc", witness, self.support_points(ref[:, ie], half_ego)
        ).min(axis=2)
        self._add_entries(
            block,
            "budget",
            ie[:, None],
            [c[2]["slack_idx"] for c in cross],
            np.zeros((len(cross), n_t)),
            ego=ie,
            witness=witness,
            ref_support=ref_support,
            budget=share[None, :] * (margin - ((1.0 - lam) * backup_value)[None, :]),
        )
        self.stats["bh_pairs_unrecoverable"] += int((backup_value < 0.0).sum())

    def _build_pair_entries(self, block, pending_pairs):
        """Joint pairs (both controls in the QP, rows on the signed-distance margins)
        and cross pairs (budget rows of the QP's agent)."""
        ref, index = block["ref"], block["index"]
        joint, cross = [], []
        for pairs, _, _, _ in pending_pairs:
            for pair in pairs:
                a, b = index[pair["a"]], index[pair["b"]]
                if pair["share"] is None:
                    joint.append((a, b, pair))
                else:
                    ego_first = pair["slices"][0] is not None
                    cross.append(((a, b) if ego_first else (b, a)) + (pair,))
        n_t = ref.shape[0]

        if joint:
            ia = np.array([j[0] for j in joint])
            ib = np.array([j[1] for j in joint])
            lam = self.decay_rate(np.array([float(j[2]["lam"]) for j in joint]))
            ref_values = self.pair_values(
                ref[:, ia].reshape(-1, 3),
                ref[:, ib].reshape(-1, 3),
                *self._pair_halves(block, ia, ib, n_t),
            ).reshape(n_t, -1)
            backup_value = ref_values.min(axis=0)
            rhs = self._decay_rhs(ref_values, lam)
            self._add_entries(
                block,
                "pair",
                np.stack([ia, ib], axis=1),
                [j[2]["slack_idx"] for j in joint],
                rhs,
                a=ia,
                b=ib,
            )
            self.stats["bh_pairs_unrecoverable"] += int((backup_value < 0.0).sum())

        if cross:
            ie = np.array([c[0] for c in cross])
            io = np.array([c[1] for c in cross])
            self._build_cross_entries(block, ie, io, cross)

    def _pair_halves(self, block, ia, ib, n_t: int) -> tuple:
        """Half sizes (of a, of b) [n_t * K, 2] of pairs of agent table indices
        ia, ib [K] at n_t time steps (rows t-major), or () for the nominal
        ones (the arguments half_a, half_b of pair_values)."""
        if block["half"] is None:
            return ()
        return (
            np.tile(block["half"][ia], (n_t, 1)),
            np.tile(block["half"][ib], (n_t, 1)),
        )

    def _build_lane_entries(self, block, pending_lanes):
        """One entry per agent and boundary side, with zero-padded facets."""
        ctrl = self.ctrl
        index = block["index"]
        params = block["params"]
        a_min = -float(ctrl.a_min)
        a_max = float(ctrl.a_max)
        sides = []
        for lane in pending_lanes:
            if params is not None:
                k = index[lane["agent"]]
                a_min = -float(params["min_acc"][k])
                a_max = float(params["max_acc"][k])
            v1 = max(0.0, float(lane["state"][3]) + ctrl.dt * a_max)
            reach = (
                2.0 * v1 * ctrl.dt + v1 * v1 / (2.0 * a_min) + self.LANE_WINDOW_MARGIN
            )
            boundaries = ctrl.lane_sat.local_boundaries(
                ctrl.scenario.world_state,
                lane["env_idx"],
                lane["agent"],
                ahead_distance=reach,
            )
            for side, points in enumerate(boundaries):
                facets = ctrl.lane_sat.facets(points, side)
                if facets["start"].shape[0] > 0:
                    sides.append(
                        (index[lane["agent"]], side, facets, lane["slack"][side])
                    )
        if not sides:
            return
        n_facets = max(facets["start"].shape[0] for _, _, facets, _ in sides)
        padded = {
            key: np.zeros((len(sides), n_facets) + sides[0][2][key].shape[1:])
            for key in ("start", "end", "normal", "offset", "buffer")
        }
        valid = np.zeros((len(sides), n_facets), dtype=bool)
        for e, (_, _, facets, _) in enumerate(sides):
            count = facets["start"].shape[0]
            for key in padded:
                padded[key][e, :count] = facets[key]
            valid[e, :count] = True
        agent = np.array([k for k, _, _, _ in sides])
        lane = dict(
            agent=agent,
            side=np.array([side for _, side, _, _ in sides]),
            facets=padded,
            valid=valid,
            # Half sizes [L, 2] of the agents (None: nominal)
            half=None if block["half"] is None else block["half"][agent],
        )
        ref_values = self._lane_values(lane, block["ref"][:, agent])
        backup_value = ref_values.min(axis=0)
        rhs = self._decay_rhs(
            ref_values, self.decay_rate(float(ctrl.lambda_ttcbf_predefined))
        )
        self._add_entries(
            block,
            "lane",
            agent[:, None],
            [slack for _, _, _, slack in sides],
            rhs,
            **lane,
        )
        self.stats["bh_lanes_unrecoverable"] += int((backup_value < 0.0).sum())

    # ---------------------------------------------------------------
    # Exact evaluation
    # ---------------------------------------------------------------

    def _rollouts_steps(
        self, block, U: np.ndarray, n_var: int, is_with_ref: bool = False
    ):
        """Roll out the controlled agents under U [C, 2] (n_var = 3 adds the
        forward differences of both controls) in one request, together with
        the reference rollouts of all agents if is_with_ref (stored in
        block["ref"], [T, A, 3]). Generator (see the module docstring).

        Returns:
            Poses [T, C, n_var, 3] of the controlled agents and the signed
            forward-difference steps [C, 2] (None for n_var = 1).
        """
        controlled = block["controlled"]
        X_c = block["X"][controlled]
        routes_c = [block["routes"][k] for k in controlled]
        steps = None
        if n_var == 3:
            perturbed, steps = self.fd_controls(
                X_c, U, take_params(block["params"], controlled)
            )
            U_rows = np.concatenate([U[:, None], perturbed], axis=1).reshape(-1, 2)
        else:
            U_rows = U
        X_rows = np.repeat(X_c, n_var, axis=0)
        row_agents = np.repeat(controlled, n_var)
        routes = [route for route in routes_c for _ in range(n_var)]
        hold = np.zeros(X_rows.shape[0], dtype=bool)
        n_ref = 0
        if is_with_ref:
            X, ref_hold = block["X"], block["hold"]
            U_ref = np.where(ref_hold[:, None], 0.0, block["backup"])
            n_ref = X.shape[0]
            X_rows = np.concatenate([X, X_rows])
            U_rows = np.concatenate([U_ref, U_rows])
            row_agents = np.concatenate([np.arange(n_ref), row_agents])
            routes = block["routes"] + routes
            hold = np.concatenate([ref_hold, hold])
        poses = yield RolloutRequest(
            self,
            X_rows,
            U_rows,
            routes,
            hold,
            block["n_steps"],
            take_params(block["params"], row_agents),
        )
        if is_with_ref:
            block["ref"] = poses[:, :n_ref]
        n_t = poses.shape[0]
        return poses[:, n_ref:].reshape(n_t, controlled.shape[0], n_var, 3), steps

    def _assemble(self, block, poses_c: np.ndarray) -> np.ndarray:
        """Poses [T, A, V, 3] of all agents: the controlled agents' rollouts and
        the other agents' reference rollouts."""
        poses = np.repeat(block["ref"][:, :, None, :], poses_c.shape[2], axis=2)
        poses[:, block["controlled"]] = poses_c
        return poses

    def _lane_values(self, lane: dict, poses: np.ndarray, sub=None) -> np.ndarray:
        """Lane values [T, L] of lane entries (positions sub, all if None) at
        poses [T, L, 3]."""
        n_t, n_lanes = poses.shape[0], poses.shape[1]
        boundary = np.arange(lane["agent"].shape[0]) if sub is None else sub
        boundary = np.tile(boundary, n_t)
        values = self.ctrl.lane_sat.padded_values(
            {key: value[boundary] for key, value in lane["facets"].items()},
            lane["valid"][boundary],
            lane["side"][boundary],
            poses.reshape(-1, 3),
            None if lane["half"] is None else lane["half"][boundary],
        )
        return values.reshape(n_t, n_lanes)

    def _kind_values(self, block, poses, kind: str, sub, variant_a=0, variant_b=0):
        """Values [len(sub), T] of the entries sub (positions within a kind),
        with the given control variants of their first and second agent."""
        data = block[kind]
        if kind == "pair":
            pa = poses[:, data["a"][sub], variant_a]
            pb = poses[:, data["b"][sub], variant_b]
            values = self.pair_values(
                pa.reshape(-1, 3),
                pb.reshape(-1, 3),
                *self._pair_halves(block, data["a"][sub], data["b"][sub], pa.shape[0]),
            ).reshape(pa.shape[0], -1)
        elif kind == "budget":
            corners = self.support_points(
                poses[:, data["ego"][sub], variant_a],
                self.half if block["half"] is None else block["half"][data["ego"][sub]],
            )
            # Closest corner of the candidate tail beyond the closest corner of
            # the braking tail (the rows of all four corners as one value)
            shift = (
                np.einsum("tbd,tbcd->tbc", data["witness"][:, sub], corners).min(axis=2)
                - data["ref_support"][:, sub]
            )
            values = shift + data["budget"][:, sub]
        else:
            values = self._lane_values(
                data, poses[:, data["agent"][sub], variant_a], sub
            )
        return values.T

    def _values(self, block, poses) -> np.ndarray:
        """Exact values [E, T] of all entries under the base controls."""
        values = np.empty(block["rhs"].shape)
        for kind in self.KINDS:
            if kind in block:
                sub = np.arange(block[kind]["idx"].shape[0])
                values[block[kind]["idx"]] = self._kind_values(block, poses, kind, sub)
        return values

    def _gradients(self, block, poses, steps, entries) -> np.ndarray:
        """Forward differences d value / d u [E, T, 2 agents, 2] of the given
        entries (agent slots follow entry["agents"]; zero elsewhere)."""
        grads = np.zeros(block["rhs"].shape + (2, 2))
        slot = block["slot"]
        for kind in self.KINDS:
            if kind not in block:
                continue
            idx = block[kind]["idx"]
            sub = np.nonzero(np.isin(idx, list(entries)))[0]
            if sub.shape[0] == 0:
                continue
            base = self._kind_values(block, poses, kind, sub)
            for s in range(2 if kind == "pair" else 1):
                members = np.array([block["entries"][e]["agents"][s] for e in idx[sub]])
                step = steps[[slot[int(m)] for m in members]]  # [K, 2]
                for col in range(2):
                    variants = (1 + col, 0) if s == 0 else (0, 1 + col)
                    perturbed = self._kind_values(block, poses, kind, sub, *variants)
                    grads[idx[sub], :, s, col] = (perturbed - base) / step[:, col, None]
        return grads

    # ---------------------------------------------------------------
    # Rows, exact check, and fallback
    # ---------------------------------------------------------------

    def _append_row(
        self, block, e: int, t: int, values, grads, U, rows, lowers, uppers, n_vars
    ):
        """Row of tail step t of entry e, linearized at the controls U [C, 2]."""
        entry = block["entries"][e]
        slot = block["slot"]
        terms, constant = [], float(values[e, t])
        for s, agent in enumerate(entry["agents"]):
            grad = grads[e, t, s].copy()
            terms.append((block["slices"][agent], grad))
            constant -= float(grad @ U[slot[agent]])
        if entry["slack"] is not None:
            terms.append((entry["slack"], self.SLACK_GAIN))
        lower = (
            float(block["rhs"][e, t])
            - constant
            + self.ctrl.lane_sat_linearization_margin
        )
        self.ctrl._append_qp_row(rows, lowers, uppers, n_vars, terms, lower, PROXQP_INF)
        block["linearized_at"][(e, t)].append(
            np.array([U[slot[a]] for a in entry["agents"]])
        )

    def controls_of(self, block, x: np.ndarray) -> np.ndarray:
        """Controls [C, 2] of the controlled agents in a QP solution x."""
        return np.array(
            [
                np.asarray(x[block["slices"][k]], dtype=np.float64)
                for k in block["controlled"]
            ]
        )

    def violated(self, block, values) -> np.ndarray:
        """Violated (entry, step) index pairs [V, 2] of exact values [E, T]."""
        return np.argwhere(values < block["rhs"] - self.ctrl.lane_sat_tol)

    def check(self, block, x, rows, lowers, uppers, n_vars) -> tuple:
        """Exact check of all tail steps at a QP solution x; violated steps get
        rows linearized at x (see CBFQP._solve_qp_with_sat_repairs).

        Returns:
            (number of added rows, number of violated steps that already have
            a row linearized at x).
        """
        return run_steps(self.check_steps(block, x, rows, lowers, uppers, n_vars))

    def check_steps(self, block, x, rows, lowers, uppers, n_vars):
        """Generator version of check (see the module docstring)."""
        U = self.controls_of(block, x)
        poses_c, steps = yield from self._rollouts_steps(block, U, n_var=3)
        poses = self._assemble(block, poses_c)
        values = self._values(block, poses)
        repair, stuck = [], 0
        for e, t in self.violated(block, values):
            e, t = int(e), int(t)
            u_now = np.array(
                [U[block["slot"][a]] for a in block["entries"][e]["agents"]]
            )
            if any(
                np.allclose(u_now, prev, rtol=0.0, atol=1e-9)
                for prev in block["linearized_at"][(e, t)]
            ):
                stuck += 1
            else:
                repair.append((e, t))
        if repair:
            grads = self._gradients(block, poses, steps, {e for e, _ in repair})
            for e, t in repair:
                self._append_row(
                    block, e, t, values, grads, U, rows, lowers, uppers, n_vars
                )
            self.stats["bh_repair_rows"] += len(repair)
        return len(repair), stuck

    def fallback(self, block, x: np.ndarray) -> bool:
        """Enforce the exact tail conditions on a solution x in place.

        Agents of violated constraints switch to their backup controls until
        the exact check passes; the joint backup control satisfies every
        condition from a recoverable state. Returns whether x was changed.
        """
        return run_steps(self.fallback_steps(block, x))

    def fallback_steps(self, block, x: np.ndarray):
        """Generator version of fallback (see the module docstring)."""
        U = self.controls_of(block, x)
        controlled = block["controlled"]
        slot = block["slot"]
        switched = np.zeros(controlled.shape[0], dtype=bool)
        while True:
            poses_c, _ = yield from self._rollouts_steps(block, U, n_var=1)
            bad = self.violated(
                block, self._values(block, self._assemble(block, poses_c))
            )
            new = np.zeros_like(switched)
            for e in np.unique(bad[:, 0]):
                for agent in block["entries"][e]["agents"]:
                    new[slot[agent]] = True
            new &= ~switched
            if not new.any():
                break  # Clear, or not recoverable even under the backup
            switched |= new
            U[new] = block["backup"][controlled[new]]
        if not switched.any():
            return False
        for s in np.nonzero(switched)[0]:
            x[block["slices"][controlled[s]]] = U[s]
        self.stats["bh_fallback_qps"] += 1
        self.stats["bh_fallback_agents"] += int(switched.sum())
        return True
