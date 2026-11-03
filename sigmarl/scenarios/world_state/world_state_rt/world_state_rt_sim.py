import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from termcolor import cprint

from sigmarl.backup_horizon_cbf import BackupHorizonCBF
from sigmarl.constants import AGENTS, THRESHOLD
from sigmarl.helper_scenario import (
    get_distances_between_agents,
    get_rectangle_vertices,
    InitialStateBuffer,
    interX,
    Collisions,
)
from sigmarl.lane_sat_barrier import LaneSatBarrier
from sigmarl.map_manager import MapManager
from sigmarl.scenarios.observations.observation_provider import AgentState
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt import (
    WorldStateRT,
    WorldStateRTParameters,
)
from sigmarl.scenarios.world_state.world_state_sim import WorldStateSim


@dataclass
class WorldStateRTSimParameters(WorldStateRTParameters):
    scenario_type: str
    cpm_scenario_probabilities: List[float]
    random_seed: int = 0
    is_testing_mode: bool = False  # Whether it is in testing mode
    is_enable_debug_diagnostics: bool = False  # Whether to emit per-step diagnostics
    is_zero_initial_speed: bool = True  # Whether to set the initial speed of the agents to zero. If False, the initial speed is sampled in [0, 20% max speed].
    is_fast_inference_mode: bool = (
        False  # Whether fast inference mode is active (e.g. skip collision checks)
    )
    is_observe_vertices: bool = (
        False  # Whether vehicle vertices are observed by the policy
    )


# Sideslip angle at full steering and the smallest turning radius [m] of the
# center (kinematic bicycle): the heading turns by at most 1 / _MIN_TURN_RADIUS
# per meter traveled, and the velocity is at most _MAX_SIDESLIP off the heading.
_MAX_SIDESLIP = math.atan(
    AGENTS["l_r"] / AGENTS["l_wb"] * math.tan(AGENTS["max_steering"])
)
_MIN_TURN_RADIUS = AGENTS["l_r"] / math.sin(_MAX_SIDESLIP)


def _braking_distance(speed: torch.Tensor, dt: float, min_acc: float) -> torch.Tensor:
    """Distance [m] a vehicle travels under maximum braking until it stops.

    Includes one step at the current speed: a vehicle reacts to a spawn in the
    step after it, and under explicit Euler a step travels the current speed.
    """
    v = speed.abs()
    return v * dt + v**2 / (-2.0 * min_acc)


def _min_progress(path_length: torch.Tensor) -> torch.Tensor:
    """Lower bound [m] on how far a vehicle's center advances along its initial
    heading over a path of the given length.

    The direction of motion deviates from the initial heading by at most
    s / R + beta after s meters (R = _MIN_TURN_RADIUS, beta = _MAX_SIDESLIP),
    so the advance is at least the integral of cos(min(s / R + beta, pi)). The
    integrand is first positive and then negative, so where the result is
    negative, it also bounds how far behind its start the center can get over
    any shorter path.
    """
    radius, beta = _MIN_TURN_RADIUS, _MAX_SIDESLIP
    angle = (path_length / radius + beta).clamp(max=math.pi)
    return radius * (torch.sin(angle) - math.sin(beta)) - (
        path_length - (math.pi - beta) * radius
    ).clamp_min(0.0)


def spawn_clearances(
    candidates: torch.Tensor,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    speeds: torch.Tensor,
    dt: float,
    min_acc: float,
    spawn_speed: float,
) -> torch.Tensor:
    """Clearance [m] of each spawn candidate from the other vehicles (>= 0: safe).

    A spawn is safe if all vehicles can brake to a stop without a collision:
    under maximum braking a vehicle's center moves at most its braking distance
    D, and its footprint stays within half a diagonal of its center, so a
    spawn at least D + THRESHOLD["initial_distance"] (1.2 diagonals) away from
    every vehicle is safe in any direction. A spawned vehicle at rest may also
    be closer behind a vehicle: it is safe if it lies at least
    max(0, -_min_progress(D)) + THRESHOLD["initial_distance"] behind the
    vehicle along its direction of travel. A spawn with a nonzero initial
    speed adds its own braking distance and uses the first rule only.

    Args:
        candidates: spawn positions [P, 2].
        positions: positions [M, 2] of the other vehicles.
        rotations: headings [M] of the other vehicles.
        speeds: signed speeds [M] of the other vehicles.
        dt: step size [s].
        min_acc: maximum deceleration (negative) [m/s^2].
        spawn_speed: initial speed of the spawned vehicle [m/s].

    Returns:
        The smallest clearance [P] over the other vehicles (inf if there are none).
    """
    if positions.shape[0] == 0:
        return candidates.new_full((candidates.shape[0],), float("inf"))
    buffer = THRESHOLD["initial_distance"]
    braking = _braking_distance(speeds, dt, min_acc)
    offsets = candidates.unsqueeze(1) - positions.unsqueeze(0)  # [P, M, 2]
    own_braking = float(_braking_distance(torch.tensor(spawn_speed), dt, min_acc))
    clearance = offsets.norm(dim=-1) - (braking + own_braking + buffer)
    if spawn_speed == 0.0:
        # Moving backward reverses the direction of travel
        sign = torch.where(speeds < 0.0, -1.0, 1.0)
        direction = torch.stack(
            (sign * torch.cos(rotations), sign * torch.sin(rotations)), dim=-1
        )
        behind = -(offsets * direction).sum(-1) - (
            (-_min_progress(braking)).clamp_min(0.0) + buffer
        )
        clearance = torch.maximum(clearance, behind)
    return clearance.min(dim=1).values


def _is_spawn_crossing_lane_boundaries(ref_path: dict, point_id: int) -> bool:
    """Whether a vehicle placed on a centerline point, aligned with the
    centerline yaw, crosses the left or right (shared) lane boundary.

    Only the boundary points within one vehicle length of the spawn point are
    checked, which keeps the check cheap and ignores far-away route parts.
    """
    length = float(AGENTS["length"])
    n_window = int(length / float(ref_path["center_line_vec_mean_length"])) + 2
    lo = max(point_id - n_window, 0)
    hi = point_id + n_window + 1
    vertices = get_rectangle_vertices(
        ref_path["center_line"][point_id],
        ref_path["center_line_yaw"][point_id],
        float(AGENTS["width"]),
        length,
    )
    return bool(
        interX(vertices, ref_path["left_boundary_shared"][lo:hi].unsqueeze(0))
        | interX(vertices, ref_path["right_boundary_shared"][lo:hi].unsqueeze(0))
    )


def _spawn_lane_margins(
    barrier: LaneSatBarrier, ref_path: dict, point_ids: np.ndarray, dt: float
) -> np.ndarray:
    """Lane margins [K] of the rectangle_sat lane barrier (with its buffers) of
    vehicles at rest on centerline points of a route, aligned with the
    centerline yaw, over the lane window the lane CBF uses for them."""
    center = ref_path["center_line"].detach().cpu().numpy().astype(np.float64)
    yaw = ref_path["center_line_yaw"].detach().cpu().numpy().astype(np.float64)
    geometry = dict(
        cum=np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1)))
        ),
        left=ref_path["left_boundary_shared"].detach().cpu().numpy().astype(np.float64),
        right=ref_path["right_boundary_shared"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        is_loop=bool(ref_path["is_loop"]),
    )
    # Window extension of a vehicle at rest (BackupHorizonCBF._build_lane_entries)
    v1 = dt * float(AGENTS["max_acc"])
    ahead = (
        2.0 * v1 * dt
        + v1 * v1 / (-2.0 * float(AGENTS["min_acc"]))
        + BackupHorizonCBF.LANE_WINDOW_MARGIN
    )
    poses = np.c_[center[point_ids], yaw.reshape(-1)[point_ids]]
    return barrier.route_point_margins(geometry, point_ids, poses, ahead)


def _get_pairwise_polyline_intersections(polylines: torch.Tensor) -> torch.Tensor:
    """Return a symmetric intersection matrix for batched closed polylines."""
    if polylines.ndim != 4 or polylines.shape[-1] != 2:
        raise ValueError("polylines must have shape [batch, polylines, points, 2].")

    batch_dim, n_polylines, n_points, _ = polylines.shape
    intersections = torch.zeros(
        batch_dim,
        n_polylines,
        n_polylines,
        dtype=torch.bool,
        device=polylines.device,
    )
    if n_polylines < 2:
        return intersections

    pair_indices = torch.triu_indices(
        n_polylines,
        n_polylines,
        offset=1,
        device=polylines.device,
    )
    first = polylines[:, pair_indices[0]].reshape(-1, n_points, 2)
    second = polylines[:, pair_indices[1]].reshape(-1, n_points, 2)
    pair_intersections = interX(first, second, is_return_points=False).reshape(
        batch_dim, -1
    )

    intersections[:, pair_indices[0], pair_indices[1]] = pair_intersections
    intersections[:, pair_indices[1], pair_indices[0]] = pair_intersections
    return intersections


def _get_agentwise_polyline_intersections(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Return intersections for aligned batched agent polylines."""
    if first.ndim != 4 or second.ndim != 4:
        raise ValueError("first and second must have shape [batch, agents, points, 2].")
    if first.shape[:2] != second.shape[:2] or first.shape[-1] != 2:
        raise ValueError("first and second must align on batch and agent dimensions.")
    if second.shape[-1] != 2:
        raise ValueError("Polyline coordinates must have size 2.")

    batch_dim, n_agents = first.shape[:2]
    first_flat = first.reshape(
        batch_dim * n_agents,
        first.shape[-2],
        2,
    )
    second_flat = second.reshape(
        batch_dim * n_agents,
        second.shape[-2],
        2,
    )
    return interX(first_flat, second_flat, is_return_points=False).reshape(
        batch_dim, n_agents
    )


def derive_reset_seed(
    base_seed: Optional[int], reset_index: int, env_index: Optional[int] = 0
) -> int:
    """Derive a deterministic 63-bit seed from base seed, reset index, and env index."""
    base = 0 if base_seed is None else int(base_seed)
    k = int(reset_index)
    e = 0 if env_index is None else int(env_index)
    return (base * 1_000_003 + k * 10_007 + e) % (2**63 - 1)


class WorldStateRTSimulation(WorldStateRT, WorldStateSim):
    def __init__(self, params: WorldStateRTSimParameters, map: MapManager):
        # Spawn candidates per set of routes (see _spawn_candidates)
        self._spawn_candidate_cache = {}
        # Spawns without a safe point (see _generate_feasible_initial_positions)
        self.n_spawn_fallbacks = 0
        super().__init__(params, map)

        self.params = params

    def _init_stateful_parameters(self):
        super()._init_stateful_parameters()

        # Initialize collision matrix (for the simulated RT env we keep track of collisions for rewarding)
        self.collisions = Collisions(
            with_agents=torch.zeros(
                (self.batch_dim, self.n_agents, self.n_agents),
                device=self.device,
                dtype=torch.bool,
            ),
            with_lanelets=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.bool
            ),
            with_entry_segments=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.bool
            ),
            with_exit_segments=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.bool
            ),
        )

    def init_ref_paths_agent_related_from_buffer(
        self, env_index: int, initial_state_buffer: InitialStateBuffer
    ):
        initial_state = initial_state_buffer.get_random()
        self.ref_paths_agent_related.scenario_id[env_index] = initial_state[
            :, initial_state_buffer.idx_scenario
        ]  # Update
        self.ref_paths_agent_related.path_id[env_index] = initial_state[
            :, initial_state_buffer.idx_path
        ]  # Update
        self.ref_paths_agent_related.point_id[env_index] = initial_state[
            :, initial_state_buffer.idx_point
        ]  # Update

        return initial_state

    def reset(
        self,
        agent_states: List[AgentState],
        env_index: int = None,
        agent_index: int = None,
        predefined_ref_path_idx=None,
        initial_state=None,
        initial_state_buffer=None,
        reset_index: int = None,
    ):

        if reset_index is not None and agent_index is None:
            generator = torch.Generator(device=self.world.device)
            seed = derive_reset_seed(self.params.random_seed, reset_index, env_index)
            generator.manual_seed(seed)
        else:
            generator = None

        reset_indices = (
            range(self.n_agents) if agent_index is None else agent_index.unsqueeze(0)
        )

        is_reset_single_agent = agent_index is not None

        (ref_paths_scenario, extended_points) = self._reset_scenario_related_ref_paths(
            env_index=env_index, agent_index=agent_index, generator=generator
        )

        for i_agent in reset_indices:
            if predefined_ref_path_idx is not None:
                predefined_ref_path_idx_i = predefined_ref_path_idx[i_agent]
            else:
                predefined_ref_path_idx_i = None

            if (agent_index is None) and (predefined_ref_path_idx is not None):
                # agent_index is None only when initializing the world, and we only use the predefined reference path in this case (not intermediate reset)
                # Use predefined reference path indices if provided
                path_id = predefined_ref_path_idx[i_agent]
                ref_path = ref_paths_scenario[path_id]
                initial_state_i = initial_state[i_agent]
                self.world.agents[i_agent].set_pos(
                    initial_state_i[0:2], batch_index=env_index
                )
                self.world.agents[i_agent].set_rot(
                    initial_state_i[2], batch_index=env_index
                )
                self.world.agents[i_agent].set_vel(
                    torch.zeros(2, device=self.params.device, dtype=torch.float32),
                    batch_index=env_index,
                )
                self.world.agents[i_agent].set_speed(
                    torch.zeros(1, device=self.params.device, dtype=torch.float32),
                    batch_index=env_index,
                )
                self.world.agents[i_agent].set_steering(
                    torch.zeros(1, device=self.params.device, dtype=torch.float32),
                    batch_index=env_index,
                )
                self.world.agents[i_agent].set_sideslip_angle(
                    torch.zeros(1, device=self.params.device, dtype=torch.float32),
                    batch_index=env_index,
                )
            else:
                ref_path, path_id = self._reset_init_state(
                    agent_states,
                    ref_paths_scenario,
                    env_index=env_index,
                    agent_index=i_agent,
                    is_reset_single_agent=is_reset_single_agent,
                    predefined_ref_path_idx_i=predefined_ref_path_idx_i,
                    initial_state=initial_state,
                    initial_state_buffer=initial_state_buffer,
                    generator=generator,
                )

            self._reset_agent_related_ref_path(
                env_index, i_agent, ref_path, path_id, extended_points
            )

    def _reset_init_state(
        self,
        agent_states: List[AgentState],
        ref_paths_scenario,
        agent_index: int,
        env_index: int = None,
        is_reset_single_agent: bool = False,
        predefined_ref_path_idx_i=None,
        initial_state=None,
        initial_state_buffer: InitialStateBuffer = None,
        generator: torch.Generator = None,
    ):
        """
        This function resets the initial position, rotation, and velocity for an agent based on the provided
        initial state buffer if it is used. Otherwise, it randomly generates initial states ensuring they
        are feasible and do not collide with other agents.
        """

        agents = self.world.agents

        if initial_state:
            # fetch reference path from buffer
            path_id = initial_state[agent_index, initial_state_buffer.idx_path].int()
            ref_path = ref_paths_scenario[path_id]

            # apply state to world
            agents[agent_index].set_pos(
                initial_state[agent_index, 0:2], batch_index=env_index
            )
            agents[agent_index].set_rot(
                initial_state[agent_index, 2], batch_index=env_index
            )
            agents[agent_index].set_vel(
                initial_state[agent_index, 3:5], batch_index=env_index
            )

            return ref_path, path_id

        # generate feasible initial positions for vehicles
        ref_path, path_id, random_point_id = self._generate_feasible_initial_positions(
            agents,
            ref_paths_scenario,
            env_index,
            agent_index,
            is_reset_single_agent,
            predefined_ref_path_idx_i,
            generator=generator,
        )
        rot_start = ref_path["center_line_yaw"][random_point_id]
        steering_start = torch.zeros_like(rot_start, device=self.world.device)
        sideslip_start = torch.zeros_like(
            steering_start, device=self.world.device
        )  # Sideslip angle is zero since the steering is zero

        if self.params.is_zero_initial_speed:
            speed_start = 0.0 * torch.ones(
                1, dtype=torch.float32, device=self.world.device
            )  # Zero initial velocity
        else:
            speed_start = (
                torch.rand(
                    1,
                    dtype=torch.float32,
                    device=self.world.device,
                    generator=generator,
                )
                * agents[agent_index].max_speed
                * 0.2
            )  # Random initial velocity in [0, 20% max speed]
        vel_start = torch.hstack(
            [
                speed_start * torch.cos(sideslip_start + rot_start),
                speed_start * torch.sin(sideslip_start + rot_start),
            ]
        )

        # apply calculated state to current world
        agents[agent_index].set_rot(rot_start, batch_index=env_index)
        agents[agent_index].set_steering(steering_start, batch_index=env_index)
        agents[agent_index].set_sideslip_angle(sideslip_start, batch_index=env_index)
        agents[agent_index].set_speed(speed_start, batch_index=env_index)
        agents[agent_index].set_vel(vel_start, batch_index=env_index)

        return ref_path, path_id

    def _generate_feasible_initial_positions(
        self,
        agents,
        ref_paths_scenario,
        env_index,
        agent_index,
        is_reset_single_agent,
        predefined_ref_path_idx_i,
        generator: torch.Generator = None,
    ):
        """Sample a spawn point that every vehicle can safely brake for.

        All candidate points (see _spawn_candidates) are checked at once
        against the other vehicles (all of them for a respawn, the ones placed
        before for a full reset) with spawn_clearances. A feasible point is
        drawn with the probability of the former rejection sampler (a uniform
        route, then a uniform point on it). If no point is feasible, the one
        with the largest clearance is used and counted in n_spawn_fallbacks.
        """
        candidates = self._spawn_candidates(ref_paths_scenario)
        weights = candidates["weights"]
        if predefined_ref_path_idx_i is not None:
            weights = weights * (candidates["path_ids"] == predefined_ref_path_idx_i)

        others = (
            [i for i in range(self.n_agents) if i != agent_index]
            if is_reset_single_agent
            else list(range(agent_index))
        )
        spawn_speed = (
            0.0
            if self.params.is_zero_initial_speed
            else 0.2 * float(agents[agent_index].max_speed)
        )  # Highest possible initial speed
        others = torch.tensor(others, dtype=torch.long, device=self.world.device)
        clearance = spawn_clearances(
            candidates["points"],
            torch.stack([agent.state.pos[env_index] for agent in agents])[others],
            torch.stack([agent.state.rot[env_index] for agent in agents]).reshape(-1)[
                others
            ],
            torch.stack([agent.state.speed[env_index] for agent in agents]).reshape(-1)[
                others
            ],
            float(self.world.dt),
            float(agents[agent_index].dynamics.min_acc),
            spawn_speed,
        )
        feasible_weights = weights * (clearance >= 0.0)
        if feasible_weights.sum() > 0:
            k = int(torch.multinomial(feasible_weights, 1, generator=generator))
        else:
            k = int(torch.where(weights > 0, clearance, -float("inf")).argmax())
            self.n_spawn_fallbacks += 1
            cprint(
                f"[WARN] No safe spawn point for agent {agent_index} in env "
                f"{int(env_index)}; using the one with the largest clearance "
                f"({float(clearance[k]):.2f} m).",
                "yellow",
            )

        path_id = int(candidates["path_ids"][k])
        point_id = int(candidates["point_ids"][k])
        self.ref_paths_agent_related.path_id[env_index, agent_index] = path_id
        self.ref_paths_agent_related.point_id[env_index, agent_index] = point_id
        ref_path = ref_paths_scenario[path_id]
        agents[agent_index].set_pos(
            ref_path["center_line"][point_id], batch_index=env_index
        )
        return ref_path, path_id, point_id

    def _spawn_candidates(self, ref_paths_scenario) -> dict:
        """Spawn candidates of a set of routes, built once and cached.

        The candidates are the centerline points 2 to 3/4 of each route at
        which a vehicle aligned with the centerline does not cross a lane
        boundary and has a nonnegative lane margin of the rectangle_sat lane
        barrier, buffers included, so that a vehicle spawned at rest starts
        inside the lane CBF's certified set. Each has the weight
        1 / (n_routes * n_points_in_range of its route), the probability of
        drawing it with a uniform route and point.

        Returns:
            dict with points [P, 2], path_ids [P], point_ids [P], weights [P].
        """
        key = id(ref_paths_scenario)
        if key not in self._spawn_candidate_cache:
            points, path_ids, point_ids, weights = [], [], [], []
            # The lane barrier of the lane CBF (see road_traffic.py)
            barrier = LaneSatBarrier(
                length=AGENTS["length"],
                width=AGENTS["width"],
                safety_buffer=0.01 * AGENTS["width"],
            )
            for path_id, ref_path in enumerate(ref_paths_scenario):
                point_range = range(2, int(3 * ref_path["center_line"].shape[0] / 4))
                route_points = np.array(
                    [
                        point_id
                        for point_id in point_range
                        if not _is_spawn_crossing_lane_boundaries(ref_path, point_id)
                    ],
                    dtype=int,
                )
                if route_points.shape[0] > 0:
                    margins = _spawn_lane_margins(
                        barrier, ref_path, route_points, float(self.world.dt)
                    )
                    route_points = route_points[margins >= 0.0]
                for point_id in route_points.tolist():
                    points.append(ref_path["center_line"][point_id])
                    path_ids.append(path_id)
                    point_ids.append(point_id)
                    weights.append(
                        1.0 / (len(ref_paths_scenario) * max(len(point_range), 1))
                    )
            if not points:
                raise ValueError(
                    "No spawn point on any route keeps the vehicle inside its lane "
                    "boundaries; is the map scaled for the agent profile "
                    "(scale_map)?"
                )
            device = self.world.device
            self._spawn_candidate_cache[key] = {
                "points": torch.stack(points).to(device),
                "path_ids": torch.tensor(path_ids, device=device),
                "point_ids": torch.tensor(point_ids, device=device),
                "weights": torch.tensor(weights, device=device),
                "routes": ref_paths_scenario,  # Keeps the key's object alive
            }
        return self._spawn_candidate_cache[key]

    def _reset_scenario_related_ref_paths(
        self,
        env_index: int = None,
        agent_index: int = None,
        generator: torch.Generator = None,
    ) -> tuple:

        if self.params.scenario_type == "cpm_intersection_extended":
            ref_paths_scenario = (
                self.ref_paths_map_related.long_term_intersection_extended
            )
            extended_points = (
                self.ref_paths_map_related.point_extended_intersection_extended
            )
            self.ref_paths_agent_related.scenario_id[env_index, :] = 0

            return ref_paths_scenario, extended_points

        if self.params.scenario_type != "cpm_mixed":
            ref_paths_scenario = self.ref_paths_map_related.long_term_all
            extended_points = self.ref_paths_map_related.point_extended_all
            # 0 for others, 1 for intersection, 2 for merge-in, 3 for merge-out scenario
            self.ref_paths_agent_related.scenario_id[env_index, :] = 0

            return ref_paths_scenario, extended_points

        if agent_index is not None:
            scenario_id = self.ref_paths_agent_related.scenario_id[
                env_index, agent_index
            ]  # Keep the same scenario
        else:
            scenario_id = (
                torch.multinomial(
                    torch.tensor(
                        self.params.cpm_scenario_probabilities,
                        device=self.world.device,
                        dtype=torch.float32,
                    ),
                    1,
                    replacement=True,
                    generator=generator,
                ).item()
                + 1
            )  # A random integer {1, 2, 3}

            self.ref_paths_agent_related.scenario_id[env_index, :] = scenario_id

        if scenario_id == 1:
            # Intersection scenario
            ref_paths_scenario = self.ref_paths_map_related.long_term_intersection
            extended_points = self.ref_paths_map_related.point_extended_intersection
        elif scenario_id == 2:
            # Merge-in scenario
            ref_paths_scenario = self.ref_paths_map_related.long_term_merge_in
            extended_points = self.ref_paths_map_related.point_extended_merge_in
        else:  # scenario_id == 3
            # Merge-out scenario
            ref_paths_scenario = self.ref_paths_map_related.long_term_merge_out
            extended_points = self.ref_paths_map_related.point_extended_merge_out

        return ref_paths_scenario, extended_points

    def update_mutual_distances(self, agent_states, env_index=slice(None)):
        mutual_distances = get_distances_between_agents(
            data=torch.stack(
                [self.world.agents[i].state.pos for i in range(self.n_agents)]
            )
            if self.distances.type == "c2c"
            else self.vertices,
            distance_type=self.distances.type,
            is_set_diagonal=True,
            x_semidim=self.world.x_semidim,
            y_semidim=self.world.y_semidim,
        )

        self.distances.agents[env_index, :, :] = mutual_distances[env_index, :, :]

    """

    """

    def update_collisions(self):
        """Update collision matrices and print collision diagnostics in testing mode."""
        is_testing_mode = self.params.is_testing_mode
        is_enable_debug_diagnostics = getattr(
            self.params, "is_enable_debug_diagnostics", False
        )

        if self.distances.type == "c2c":
            self.collisions.with_agents |= _get_pairwise_polyline_intersections(
                self.vertices
            )
        elif self.distances.type == "mtv":
            # Overlapping agents have a negative mtv-based distance
            self.collisions.with_agents[:] = self.distances.agents <= 0

        collision_with_lanelets = _get_agentwise_polyline_intersections(
            self.vertices,
            self.ref_paths_agent_related.nearing_points_left_boundary,
        ) | _get_agentwise_polyline_intersections(
            self.vertices,
            self.ref_paths_agent_related.nearing_points_right_boundary,
        )
        if self.state_buffer is not None and self.state_buffer.valid_size > 0:
            current_positions = torch.stack(
                [agent.state.pos for agent in self.world.agents], dim=1
            )
            center_sweeps = torch.stack(
                (
                    self.state_buffer.get_latest(n=1)[:, :, 0:2],
                    current_positions,
                ),
                dim=2,
            )
            # Keep the anti-tunneling check on the agent's local lane window.
            # A full route boundary may contain future branches that fold back
            # across the agent's current lane in roundabout geometries.
            collision_with_lanelets |= _get_agentwise_polyline_intersections(
                center_sweeps,
                self.ref_paths_agent_related.nearing_points_left_boundary,
            ) | _get_agentwise_polyline_intersections(
                center_sweeps,
                self.ref_paths_agent_related.nearing_points_right_boundary,
            )
        self.collisions.with_lanelets |= collision_with_lanelets

        for a_i in range(self.n_agents):
            # Check for collisions with entry or exit segments (only need if agents' reference paths are not a loop)
            if not self.ref_paths_agent_related.is_loop[:, a_i].any():
                self.collisions.with_entry_segments[:, a_i] = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.entry[:, a_i],
                    is_return_points=False,
                )
                self.collisions.with_exit_segments[:, a_i] = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.exit[:, a_i],
                    is_return_points=False,
                )

            if is_testing_mode and is_enable_debug_diagnostics:
                collision_with_agents = self.collisions.with_agents[:, a_i].any(dim=1)
                collision_with_lanelets = self.collisions.with_lanelets[:, a_i]
                collision_with_entry = self.collisions.with_entry_segments[:, a_i]
                collision_with_exit = self.collisions.with_exit_segments[:, a_i]
                collision_envs = (
                    collision_with_agents
                    | collision_with_lanelets
                    | collision_with_entry
                    | collision_with_exit
                )
                if collision_envs.any():
                    env_indices = collision_envs.nonzero(as_tuple=True)[0]
                    for env_index in env_indices.tolist():
                        other_agents = self.collisions.with_agents[
                            env_index, a_i
                        ].nonzero(as_tuple=True)[0]
                        message_prefix = (
                            "[DEBUG] Exit reached: "
                            if collision_with_exit[env_index]
                            else "[DEBUG] Collision detected: "
                        )
                        message_color = (
                            "yellow" if collision_with_exit[env_index] else "red"
                        )
                        cprint(
                            message_prefix + f"env={env_index}, agent={a_i}, "
                            f"other_agents={other_agents.tolist()}, "
                            f"lanelet={self.collisions.with_lanelets[env_index, a_i].item()}, "
                            f"entry={self.collisions.with_entry_segments[env_index, a_i].item()}, "
                            f"exit={self.collisions.with_exit_segments[env_index, a_i].item()}",
                            message_color,
                        )

    def reset_collisions(self, env_index=slice(None)):
        self.collisions.with_agents[env_index, :, :] = False
        self.collisions.with_lanelets[env_index, :] = False
        self.collisions.with_entry_segments[env_index, :] = False
        self.collisions.with_exit_segments[env_index, :] = False

    def update_state_before_rewarding(self, agent_index):
        """Update some states (such as mutual distances between agents, vertices of each agent, and
        collision matrices) that will be used before rewarding agents.
        """

        agent_states = self.get_agent_state_list()

        if agent_index == 0:  # Avoid repeated computations

            self.reset_collisions()

            self.update_vertices(agent_states)

        self.update_distances(agent_states, agent_index)

        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        if agent_index == 0:  # Avoid repeated computations

            self.update_collisions()

    def update_state_after_step(self):
        """Prepare collision and distance state once after a world step.

        VMAS calls :meth:`ScenarioRoadTraffic.post_step` before it requests
        rewards, observations, and infos.  Keeping the original agent-order
        operations here preserves the numerical behavior of
        ``update_state_before_rewarding`` while avoiding repeated construction
        of the agent-state list from each reward call. Smooth sampling with
        simple road distances uses the all-agent distance kernel.
        """
        agent_states = self.get_agent_state_list()

        self.reset_collisions()
        # Skip corner vertex computation if fast inference is on and vertices are unobserved.
        if not self.params.is_fast_inference_mode or self.params.is_observe_vertices:
            self.update_vertices(agent_states)

        if self.params.is_smooth_sampling and self.params.is_use_simple_road_distance:
            self.update_mutual_distances(agent_states)
            self._update_smooth_sampled_ref_paths_all_agents(agent_states)
            self._update_simple_road_distances_all_agents(agent_states)
            if not self.params.is_fast_inference_mode:
                self.update_collisions()
        else:
            for agent_index in range(self.n_agents):
                self.update_distances(agent_states, agent_index)
            if not self.params.is_fast_inference_mode:
                self.update_collisions()
