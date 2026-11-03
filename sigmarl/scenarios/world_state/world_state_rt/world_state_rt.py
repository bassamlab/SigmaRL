from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import torch

from sigmarl.constants import AGENTS, LENGTH_SCALE
from sigmarl.helper_scenario import (
    Distances,
    ReferencePathsAgentRelated,
    get_progress_on_polyline,
    get_perpendicular_distances,
    get_rectangle_vertices,
    get_reference_progress_aligned_sampled_points,
    get_route_progress_aligned_sampled_points,
    get_short_term_reference_path,
    ReferencePathsMapRelated,
)
from sigmarl.map_manager import MapManager
from sigmarl.scenarios.observations.observation_provider import AgentState
from sigmarl.scenarios.world_state.world_state import WorldState, WorldStateParameters


@dataclass
class WorldStateRTParameters(WorldStateParameters):
    # distances parameters
    distance_type: str
    # ref path parameters
    max_ref_path_points: int
    n_points_nearing_boundary: int
    n_points_short_term: int
    short_term_ref_path_spacing_m: float
    is_smooth_sampling: bool
    is_use_simple_road_distance: bool
    observe_distance_to_boundaries: bool


"""
Implementation of the world state class for the road_traffic scenario.
"""


class WorldStateRT(WorldState):
    def __init__(self, params: WorldStateRTParameters, map: MapManager):
        self.map = (
            map  # Do not move this line, it is needed for the super().__init__() call
        )
        super().__init__(params)

        self.params = params

        self.ref_paths_map_related = ReferencePathsMapRelated(
            long_term_all=self.map.parser.reference_paths,
            long_term_intersection=self.map.parser.reference_paths_intersection,
            long_term_intersection_extended=self.map.parser.reference_paths_intersection_extended,
            long_term_merge_in=self.map.parser.reference_paths_merge_in,
            long_term_merge_out=self.map.parser.reference_paths_merge_out,
            point_extended_all=torch.zeros(
                (
                    len(self.map.parser.reference_paths),
                    self.params.n_points_short_term,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),  # Not interesting, may be useful in the future
            point_extended_intersection=torch.zeros(
                (
                    len(self.map.parser.reference_paths_intersection),
                    self.params.n_points_short_term,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),
            point_extended_intersection_extended=torch.zeros(
                (
                    len(self.map.parser.reference_paths_intersection_extended),
                    self.params.n_points_short_term,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),
            point_extended_merge_in=torch.zeros(
                (
                    len(self.map.parser.reference_paths_merge_in),
                    self.params.n_points_short_term,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),
            point_extended_merge_out=torch.zeros(
                (
                    len(self.map.parser.reference_paths_merge_out),
                    self.params.n_points_short_term,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),
            sample_interval=torch.tensor(
                self.params.short_term_ref_path_spacing_m,
                device=self.device,
                dtype=torch.float32,
            ),
        )

        self._extend_map_related_ref_path()

    @abstractmethod
    def reset(self, agent_states):
        pass

    @abstractmethod
    def _reset_scenario_related_ref_paths(self) -> tuple:
        pass

    @abstractmethod
    def _reset_init_state(
        self, agent_states: List[AgentState], ref_paths_scenario, agent_index: int
    ):
        pass

    @abstractmethod
    def update_mutual_distances(self, agent_states, env_index=slice(None)):
        pass

    def _init_stateful_parameters(self):
        self.distances = Distances(
            type=self.params.distance_type,  # Type of distances between agents
            agents=torch.zeros(
                self.batch_dim, self.n_agents, self.n_agents, dtype=torch.float32
            ),
            left_boundaries=torch.zeros(
                (self.batch_dim, self.n_agents, 4),
                device=self.device,
                dtype=torch.float32,
            ),  # Distances from the four vertices to the left boundary
            right_boundaries=torch.zeros(
                (self.batch_dim, self.n_agents, 4),
                device=self.device,
                dtype=torch.float32,
            ),  # Distances from the four vertices to the right boundary
            boundaries=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.float32
            ),
            ref_paths=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.float32
            ),
            closest_point_on_ref_path=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),
            closest_point_on_left_b=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),
            closest_point_on_right_b=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),
        )

        max_num_lanelets = len(self.map.parser.lanelets_all)

        # Initialize agent-specific reference paths, which will be determined in `reset_world_at` function
        self.ref_paths_agent_related = ReferencePathsAgentRelated(
            long_term=torch.zeros(
                (self.batch_dim, self.n_agents, self.params.max_ref_path_points, 2),
                device=self.device,
                dtype=torch.float32,
            ),  # Long-term reference paths of agents
            long_term_vec_normalized=torch.zeros(
                (self.batch_dim, self.n_agents, self.params.max_ref_path_points, 2),
                device=self.device,
                dtype=torch.float32,
            ),
            left_boundary=torch.zeros(
                (self.batch_dim, self.n_agents, self.params.max_ref_path_points, 2),
                device=self.device,
                dtype=torch.float32,
            ),
            right_boundary=torch.zeros(
                (self.batch_dim, self.n_agents, self.params.max_ref_path_points, 2),
                device=self.device,
                dtype=torch.float32,
            ),
            entry=torch.zeros(
                (self.batch_dim, self.n_agents, 2, 2),
                device=self.device,
                dtype=torch.float32,
            ),
            exit=torch.zeros(
                (self.batch_dim, self.n_agents, 2, 2),
                device=self.device,
                dtype=torch.float32,
            ),
            is_loop=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.bool
            ),
            n_points_long_term=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),
            n_points_left_b=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),
            n_points_right_b=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),
            short_term=torch.zeros(
                (self.batch_dim, self.n_agents, self.params.n_points_short_term, 2),
                device=self.device,
                dtype=torch.float32,
            ),  # Short-term reference path
            short_term_indices=torch.zeros(
                (self.batch_dim, self.n_agents, self.params.n_points_short_term),
                device=self.device,
                dtype=torch.int32,
            ),
            n_points_nearing_boundary=torch.tensor(
                self.params.n_points_nearing_boundary,
                device=self.device,
                dtype=torch.int32,
            ),
            nearing_points_left_boundary=torch.zeros(
                (
                    self.batch_dim,
                    self.n_agents,
                    self.params.n_points_nearing_boundary,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),  # Nearing left boundary
            nearing_points_right_boundary=torch.zeros(
                (
                    self.batch_dim,
                    self.n_agents,
                    self.params.n_points_nearing_boundary,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),  # Nearing right boundary
            scenario_id=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),  # Which scenarios agents are (1 for intersection, 2 for merge-in, 3 for merge-out)
            path_id=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),  # Which paths agents are
            point_id=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),  # Which points agents are
            ref_lanelet_ids=torch.zeros(
                (self.batch_dim, self.n_agents, max_num_lanelets),
                device=self.device,
                dtype=torch.int32,
            ),  # Lanelet IDs of the reference path
            n_ref_lanelet_ids=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.int32
            ),  # Number of lanelet IDs in the reference path (used for slicing later)
            ref_lanelet_segment_points=torch.zeros(
                (self.batch_dim, self.n_agents, max_num_lanelets + 1, 2),
                device=self.device,
                dtype=torch.float32,
            ),  # Connection points (segment startpoints and endpoints) of the lanelets for the reference path
        )

        n_segments = self.params.max_ref_path_points - 1
        segment_cache_shape = (self.batch_dim, self.n_agents, n_segments)
        point_cache_shape = (
            self.batch_dim,
            self.n_agents,
            self.params.max_ref_path_points,
        )
        self._ref_path_segment_vectors = torch.zeros(
            (*segment_cache_shape, 2), device=self.device, dtype=torch.float32
        )
        self._ref_path_segment_lengths = torch.zeros(
            segment_cache_shape, device=self.device, dtype=torch.float32
        )
        self._ref_path_segment_lengths_squared = torch.zeros(
            segment_cache_shape, device=self.device, dtype=torch.float32
        )
        self._ref_path_valid_segments = torch.zeros(
            segment_cache_shape, device=self.device, dtype=torch.bool
        )
        self._ref_path_cumulative_lengths = torch.zeros(
            point_cache_shape, device=self.device, dtype=torch.float32
        )
        self._ref_path_sampling_cumulative_lengths = torch.zeros(
            point_cache_shape, device=self.device, dtype=torch.float32
        )
        self._ref_path_total_lengths = torch.zeros(
            (self.batch_dim, self.n_agents),
            device=self.device,
            dtype=torch.float32,
        )
        self._route_progress = torch.zeros(
            (self.batch_dim, self.n_agents),
            device=self.device,
            dtype=torch.float32,
        )
        self._route_progress_previous = torch.zeros_like(self._route_progress)
        self._route_progress_total_lengths = torch.zeros_like(self._route_progress)
        self._route_progress_cache_valid = torch.zeros(
            (self.batch_dim, self.n_agents),
            device=self.device,
            dtype=torch.bool,
        )
        self._route_progress_association_window = None
        self._left_boundary_segment_vectors = torch.zeros(
            (*segment_cache_shape, 2), device=self.device, dtype=torch.float32
        )
        self._right_boundary_segment_vectors = torch.zeros(
            (*segment_cache_shape, 2), device=self.device, dtype=torch.float32
        )
        self._sampling_batch_indices = torch.arange(
            self.batch_dim * self.n_agents, device=self.device, dtype=torch.long
        )
        self._short_term_progress_offsets = (
            torch.arange(
                self.params.n_points_short_term,
                device=self.device,
                dtype=torch.float32,
            )
            + 1
        )
        self._boundary_progress_offsets = torch.arange(
            self.params.n_points_nearing_boundary,
            device=self.device,
            dtype=torch.float32,
        ) - round(self.params.n_points_nearing_boundary / 2)

        # The shape of each agent is considered a rectangle with 4 vertices.
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros(
            (self.batch_dim, self.n_agents, 5, 2),
            device=self.device,
            dtype=torch.float32,
        )
        # Footprint of each agent [m] (see ScenarioRoadTraffic.set_vehicle_params)
        self.agent_lengths = [AGENTS["length"]] * self.n_agents
        self.agent_widths = [AGENTS["width"]] * self.n_agents

        # Initialize variables to store nominal actions (used only when CBF is applied)
        self.nominal_action_vel = torch.zeros(
            (self.batch_dim, self.n_agents), device=self.device, dtype=torch.float32
        )
        self.nominal_action_steer = torch.zeros(
            (self.batch_dim, self.n_agents), device=self.device, dtype=torch.float32
        )
        self.applied_action_vel = torch.zeros(
            (self.batch_dim, self.n_agents), device=self.device, dtype=torch.float32
        )  # In case CBF is not used, applied action = nominal action
        self.applied_action_steer = torch.zeros(
            (self.batch_dim, self.n_agents), device=self.device, dtype=torch.float32
        )

    def _extend_map_related_ref_path(self):
        # Extended the reference path by several points along the last vector of the center line
        idx_broadcasting_entend = torch.arange(
            1,
            self.params.n_points_short_term + 1,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        for idx, i_path in enumerate(self.map.parser.reference_paths):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            direction = direction / torch.clamp(
                direction.norm(), min=torch.finfo(center_line_i.dtype).eps
            )
            self.ref_paths_map_related.point_extended_all[idx, :] = (
                center_line_i[-1]
                + idx_broadcasting_entend
                * self.params.short_term_ref_path_spacing_m
                * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_intersection):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            direction = direction / torch.clamp(
                direction.norm(), min=torch.finfo(center_line_i.dtype).eps
            )
            self.ref_paths_map_related.point_extended_intersection[idx, :] = (
                center_line_i[-1]
                + idx_broadcasting_entend
                * self.params.short_term_ref_path_spacing_m
                * direction
            )
        for idx, i_path in enumerate(
            self.map.parser.reference_paths_intersection_extended
        ):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            direction = direction / torch.clamp(
                direction.norm(), min=torch.finfo(center_line_i.dtype).eps
            )
            self.ref_paths_map_related.point_extended_intersection_extended[idx, :] = (
                center_line_i[-1]
                + idx_broadcasting_entend
                * self.params.short_term_ref_path_spacing_m
                * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_merge_in):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            direction = direction / torch.clamp(
                direction.norm(), min=torch.finfo(center_line_i.dtype).eps
            )
            self.ref_paths_map_related.point_extended_merge_in[idx, :] = (
                center_line_i[-1]
                + idx_broadcasting_entend
                * self.params.short_term_ref_path_spacing_m
                * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_merge_out):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            direction = direction / torch.clamp(
                direction.norm(), min=torch.finfo(center_line_i.dtype).eps
            )
            self.ref_paths_map_related.point_extended_merge_out[idx, :] = (
                center_line_i[-1]
                + idx_broadcasting_entend
                * self.params.short_term_ref_path_spacing_m
                * direction
            )

    def _reset_agent_related_ref_path(
        self, env_i, agent_index, ref_path, path_id, extended_points
    ):
        """
        This function resets the agent-related reference paths and updates various related attributes
        for a specified agent in an environment.
        """
        center_line = ref_path["center_line"]
        center_line_vec_normalized = ref_path["center_line_vec_normalized"]
        left_boundary_shared = ref_path["left_boundary_shared"]
        right_boundary_shared = ref_path["right_boundary_shared"]

        # Long-term reference paths for agents
        n_points_long_term = center_line.shape[0]

        self.ref_paths_agent_related.long_term[
            env_i, agent_index, 0:n_points_long_term, :
        ] = center_line

        self.ref_paths_agent_related.long_term[
            env_i,
            agent_index,
            n_points_long_term : (n_points_long_term + self.params.n_points_short_term),
            :,
        ] = extended_points[path_id, :, :]

        self.ref_paths_agent_related.long_term[
            env_i,
            agent_index,
            (n_points_long_term + self.params.n_points_short_term) :,
            :,
        ] = extended_points[path_id, -1, :]

        self.ref_paths_agent_related.n_points_long_term[
            env_i, agent_index
        ] = n_points_long_term

        self.ref_paths_agent_related.long_term_vec_normalized[
            env_i, agent_index, 0 : n_points_long_term - 1, :
        ] = center_line_vec_normalized

        self.ref_paths_agent_related.long_term_vec_normalized[
            env_i,
            agent_index,
            (n_points_long_term - 1) : (
                n_points_long_term - 1 + self.params.n_points_short_term
            ),
            :,
        ] = center_line_vec_normalized[-1, :]

        n_points_left_b = left_boundary_shared.shape[0]

        self.ref_paths_agent_related.left_boundary[
            env_i, agent_index, 0:n_points_left_b, :
        ] = left_boundary_shared

        self.ref_paths_agent_related.left_boundary[
            env_i, agent_index, n_points_left_b:, :
        ] = left_boundary_shared[-1, :]

        self.ref_paths_agent_related.n_points_left_b[
            env_i, agent_index
        ] = n_points_left_b

        n_points_right_b = right_boundary_shared.shape[0]

        self.ref_paths_agent_related.right_boundary[
            env_i, agent_index, 0:n_points_right_b, :
        ] = right_boundary_shared

        self.ref_paths_agent_related.right_boundary[
            env_i, agent_index, n_points_right_b:, :
        ] = right_boundary_shared[-1, :]

        self.ref_paths_agent_related.n_points_right_b[
            env_i, agent_index
        ] = n_points_right_b

        self.ref_paths_agent_related.entry[
            env_i, agent_index, 0, :
        ] = left_boundary_shared[0, :]
        self.ref_paths_agent_related.entry[
            env_i, agent_index, 1, :
        ] = right_boundary_shared[0, :]

        self.ref_paths_agent_related.exit[
            env_i, agent_index, 0, :
        ] = left_boundary_shared[-1, :]
        self.ref_paths_agent_related.exit[
            env_i, agent_index, 1, :
        ] = right_boundary_shared[-1, :]

        self.ref_paths_agent_related.is_loop[env_i, agent_index] = ref_path["is_loop"]

        # Store information for determining the current lanelet ID of each agent
        self.ref_paths_agent_related.n_ref_lanelet_ids[env_i, agent_index] = len(
            ref_path["lanelet_IDs"]
        )
        self.ref_paths_agent_related.ref_lanelet_ids[env_i, agent_index, :] = 0
        self.ref_paths_agent_related.ref_lanelet_ids[
            env_i, agent_index, : len(ref_path["lanelet_IDs"])
        ] = torch.tensor(ref_path["lanelet_IDs"], dtype=torch.int32)
        self.ref_paths_agent_related.ref_lanelet_segment_points[
            env_i, agent_index, : len(ref_path["lanelet_IDs"]) + 1
        ] = self.map.get_ref_lanelet_segment_points(ref_path["lanelet_IDs"])
        self._refresh_ref_path_sampling_cache(env_i, agent_index)
        # A path assignment/reset invalidates the previous route station. The
        # next sampling update initializes from a global projection once.
        self._route_progress_cache_valid[env_i, agent_index] = False

    def _refresh_ref_path_sampling_cache(self, env_index, agent_index: int):
        """Refresh static sampling metadata after assigning an agent path."""
        polyline = self.ref_paths_agent_related.long_term[env_index, agent_index]
        n_points = self.ref_paths_agent_related.n_points_long_term[
            env_index, agent_index
        ]
        is_single_env = polyline.ndim == 2
        if is_single_env:
            polyline = polyline.unsqueeze(0)
            n_points = n_points.unsqueeze(0)

        segment_vectors = polyline[:, 1:, :] - polyline[:, :-1, :]
        segment_lengths = torch.norm(segment_vectors, dim=-1)
        segment_lengths_squared = torch.sum(segment_vectors**2, dim=-1)
        segment_indices = torch.arange(
            segment_lengths.shape[-1], device=self.device
        ).unsqueeze(0)
        valid_segments = segment_indices < (n_points - 1).unsqueeze(-1)
        valid_segments = valid_segments & (
            segment_lengths > torch.finfo(polyline.dtype).eps
        )
        cumulative_lengths = torch.cat(
            (
                torch.zeros(
                    (polyline.shape[0], 1),
                    device=self.device,
                    dtype=polyline.dtype,
                ),
                torch.cumsum(segment_lengths, dim=-1),
            ),
            dim=-1,
        )
        sampling_segment_lengths = torch.where(
            valid_segments,
            segment_lengths,
            torch.zeros((), device=self.device, dtype=polyline.dtype),
        )
        sampling_cumulative_lengths = torch.cat(
            (
                torch.zeros(
                    (polyline.shape[0], 1),
                    device=self.device,
                    dtype=polyline.dtype,
                ),
                torch.cumsum(sampling_segment_lengths, dim=-1),
            ),
            dim=-1,
        )
        batch_indices = torch.arange(polyline.shape[0], device=self.device)
        last_point_indices = torch.clamp(n_points.to(torch.long) - 1, min=0)
        total_lengths = cumulative_lengths[batch_indices, last_point_indices]

        def restore_env_shape(value):
            return value.squeeze(0) if is_single_env else value

        self._ref_path_segment_vectors[env_index, agent_index] = restore_env_shape(
            segment_vectors
        )
        self._ref_path_segment_lengths[env_index, agent_index] = restore_env_shape(
            segment_lengths
        )
        self._ref_path_segment_lengths_squared[
            env_index, agent_index
        ] = restore_env_shape(segment_lengths_squared)
        self._ref_path_valid_segments[env_index, agent_index] = restore_env_shape(
            valid_segments
        )
        self._ref_path_cumulative_lengths[env_index, agent_index] = restore_env_shape(
            cumulative_lengths
        )
        self._ref_path_sampling_cumulative_lengths[
            env_index, agent_index
        ] = restore_env_shape(sampling_cumulative_lengths)
        self._ref_path_total_lengths[env_index, agent_index] = restore_env_shape(
            total_lengths
        )

        left_boundary = self.ref_paths_agent_related.left_boundary[
            env_index, agent_index
        ]
        right_boundary = self.ref_paths_agent_related.right_boundary[
            env_index, agent_index
        ]
        self._left_boundary_segment_vectors[env_index, agent_index] = (
            left_boundary[..., 1:, :] - left_boundary[..., :-1, :]
        )
        self._right_boundary_segment_vectors[env_index, agent_index] = (
            right_boundary[..., 1:, :] - right_boundary[..., :-1, :]
        )

    @staticmethod
    def _ensure_batched_points(points: torch.Tensor) -> torch.Tensor:
        """Return points with an explicit batch dimension."""
        if points.ndim == 2:
            return points.unsqueeze(0)
        return points

    @staticmethod
    def _min_distances_to_sampled_points(
        query_points: torch.Tensor, sampled_points: torch.Tensor
    ) -> torch.Tensor:
        """Compute minimum Euclidean distances from query points to sampled points."""
        query_points = WorldStateRT._ensure_batched_points(query_points)
        sampled_points = WorldStateRT._ensure_batched_points(sampled_points)
        distances = torch.linalg.vector_norm(
            query_points.unsqueeze(-2) - sampled_points.unsqueeze(-3),
            dim=-1,
        )
        return distances.min(dim=-1).values

    def _assign_agent_distance_values(
        self,
        env_index,
        agent_index: int,
        ref_path_distances: torch.Tensor,
        left_boundary_distances: torch.Tensor,
        right_boundary_distances: torch.Tensor,
    ):
        """Assign centerline and boundary distance tensors for one agent."""
        if ref_path_distances.ndim > 0 and ref_path_distances.shape[0] == 1:
            ref_path_distances = ref_path_distances.squeeze(0)
        if left_boundary_distances.ndim > 1 and left_boundary_distances.shape[0] == 1:
            left_boundary_distances = left_boundary_distances.squeeze(0)
        if right_boundary_distances.ndim > 1 and right_boundary_distances.shape[0] == 1:
            right_boundary_distances = right_boundary_distances.squeeze(0)

        self.distances.ref_paths[env_index, agent_index] = ref_path_distances
        self.distances.left_boundaries[env_index, agent_index] = left_boundary_distances
        self.distances.right_boundaries[
            env_index, agent_index
        ] = right_boundary_distances
        self.distances.boundaries[env_index, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[env_index, agent_index],
                    self.distances.right_boundaries[env_index, agent_index],
                )
            ),
            dim=-1,
        )

    def _update_simple_road_distances(self, agent_pos, env_index, agent_index: int):
        """Update road distances using local sampled centerline and boundary points."""
        center_points = (
            agent_pos.unsqueeze(0).unsqueeze(0)
            if agent_pos.ndim == 1
            else agent_pos.unsqueeze(1)
        )
        vertices = self.vertices[env_index, agent_index, :4, :]
        vertices = vertices.unsqueeze(0) if vertices.ndim == 2 else vertices

        ref_path_distances = self._min_distances_to_sampled_points(
            center_points,
            self.ref_paths_agent_related.short_term[env_index, agent_index],
        ).squeeze(-1)
        left_boundary_distances = self._min_distances_to_sampled_points(
            vertices,
            self.ref_paths_agent_related.nearing_points_left_boundary[
                env_index, agent_index
            ],
        )
        right_boundary_distances = self._min_distances_to_sampled_points(
            vertices,
            self.ref_paths_agent_related.nearing_points_right_boundary[
                env_index, agent_index
            ],
        )

        self._assign_agent_distance_values(
            env_index=env_index,
            agent_index=agent_index,
            ref_path_distances=ref_path_distances,
            left_boundary_distances=left_boundary_distances,
            right_boundary_distances=right_boundary_distances,
        )

    def _update_simple_road_distances_all_agents(self, agent_states: List[AgentState]):
        """Update simple road distances for every environment and agent together."""
        agent_positions = torch.stack(
            [agent_state.pos for agent_state in agent_states], dim=1
        )
        vertices = self.vertices[:, :, :4, :]

        ref_path_distances = self._min_distances_to_sampled_points(
            agent_positions.unsqueeze(-2),
            self.ref_paths_agent_related.short_term,
        ).squeeze(-1)
        left_boundary_distances = self._min_distances_to_sampled_points(
            vertices,
            self.ref_paths_agent_related.nearing_points_left_boundary,
        )
        right_boundary_distances = self._min_distances_to_sampled_points(
            vertices,
            self.ref_paths_agent_related.nearing_points_right_boundary,
        )

        self.distances.ref_paths[:] = ref_path_distances
        self.distances.left_boundaries[:] = left_boundary_distances
        self.distances.right_boundaries[:] = right_boundary_distances
        self.distances.boundaries[:], _ = torch.min(
            torch.cat((left_boundary_distances, right_boundary_distances), dim=-1),
            dim=-1,
        )

    @staticmethod
    def _average_boundary_indices(
        vertex_indices: torch.Tensor,
        n_points_boundary: torch.Tensor,
        is_loop: torch.Tensor,
    ) -> torch.Tensor:
        """Average vertex closest indices while respecting loop seam wraparound.

        Args:
            vertex_indices: Closest boundary indices for vehicle vertices with shape
                ``[B, 4]`` or ``[4]``.
            n_points_boundary: Number of valid boundary points with shape ``[B]`` or
                scalar ``[]``.
            is_loop: Whether each boundary belongs to a looped reference path with
                shape ``[B]`` or scalar ``[]``.

        Returns:
            Rounded average closest boundary index with shape ``[B]``. For looped
            paths, indices near the first/last point are averaged across the seam.
        """
        if vertex_indices.ndim == 1:
            vertex_indices = vertex_indices.unsqueeze(0)
        if n_points_boundary.ndim == 0:
            n_points_boundary = n_points_boundary.unsqueeze(0)
        if is_loop.ndim == 0:
            is_loop = is_loop.unsqueeze(0)

        n_points_boundary = n_points_boundary.to(
            device=vertex_indices.device, dtype=torch.long
        )
        is_loop = is_loop.to(device=vertex_indices.device, dtype=torch.bool)
        indices = vertex_indices.to(torch.float32)
        period = torch.clamp(n_points_boundary - 1, min=1).to(indices.dtype)

        anchor = indices[:, :1]
        delta = indices - anchor
        delta = torch.where(
            is_loop.unsqueeze(-1) & (delta > period.unsqueeze(-1) / 2),
            delta - period.unsqueeze(-1),
            delta,
        )
        delta = torch.where(
            is_loop.unsqueeze(-1) & (delta < -period.unsqueeze(-1) / 2),
            delta + period.unsqueeze(-1),
            delta,
        )
        averaged = torch.round(anchor.squeeze(-1) + delta.mean(dim=-1)).to(torch.long)
        averaged = torch.where(
            is_loop,
            averaged % period.to(torch.long),
            torch.minimum(torch.clamp(averaged, min=0), n_points_boundary - 1),
        )
        return averaged

    def _update_exact_road_distances(self, agent_pos, env_index, agent_index: int):
        """Update road distances using exact perpendicular distances to full polylines."""
        left_boundary = self.ref_paths_agent_related.left_boundary[
            env_index, agent_index
        ]
        right_boundary = self.ref_paths_agent_related.right_boundary[
            env_index, agent_index
        ]
        n_points_left_b = self.ref_paths_agent_related.n_points_left_b[
            env_index, agent_index
        ]
        n_points_right_b = self.ref_paths_agent_related.n_points_right_b[
            env_index, agent_index
        ]

        (
            self.distances.ref_paths[env_index, agent_index],
            self.distances.closest_point_on_ref_path[env_index, agent_index],
        ) = get_perpendicular_distances(
            point=agent_pos,
            polyline=self.ref_paths_agent_related.long_term[env_index, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                env_index, agent_index
            ],
        )

        vertices = self.vertices[env_index, agent_index, :4, :]
        is_single_env = vertices.ndim == 2
        vertices_batched = vertices.unsqueeze(0) if is_single_env else vertices
        batch_size, n_vertices, _ = vertices_batched.shape
        vertices_flat = vertices_batched.reshape(batch_size * n_vertices, -1)

        left_boundary_batched = (
            left_boundary.unsqueeze(0) if left_boundary.ndim == 2 else left_boundary
        )
        right_boundary_batched = (
            right_boundary.unsqueeze(0) if right_boundary.ndim == 2 else right_boundary
        )
        left_boundary_repeated = left_boundary_batched.repeat_interleave(
            n_vertices, dim=0
        )
        right_boundary_repeated = right_boundary_batched.repeat_interleave(
            n_vertices, dim=0
        )
        n_points_left_repeated = (
            n_points_left_b.unsqueeze(0)
            if n_points_left_b.ndim == 0
            else n_points_left_b
        ).repeat_interleave(n_vertices)
        n_points_right_repeated = (
            n_points_right_b.unsqueeze(0)
            if n_points_right_b.ndim == 0
            else n_points_right_b
        ).repeat_interleave(n_vertices)

        left_vertex_distances, left_vertex_indices = get_perpendicular_distances(
            point=vertices_flat,
            polyline=left_boundary_repeated,
            n_points_long_term=n_points_left_repeated,
        )
        right_vertex_distances, right_vertex_indices = get_perpendicular_distances(
            point=vertices_flat,
            polyline=right_boundary_repeated,
            n_points_long_term=n_points_right_repeated,
        )
        left_vertex_distances = left_vertex_distances.reshape(batch_size, n_vertices)
        right_vertex_distances = right_vertex_distances.reshape(batch_size, n_vertices)
        left_vertex_indices = left_vertex_indices.reshape(batch_size, n_vertices)
        right_vertex_indices = right_vertex_indices.reshape(batch_size, n_vertices)

        is_loop = self.ref_paths_agent_related.is_loop[env_index, agent_index]
        left_boundary_index = self._average_boundary_indices(
            vertex_indices=left_vertex_indices,
            n_points_boundary=n_points_left_b,
            is_loop=is_loop,
        )
        right_boundary_index = self._average_boundary_indices(
            vertex_indices=right_vertex_indices,
            n_points_boundary=n_points_right_b,
            is_loop=is_loop,
        )

        if is_single_env:
            left_vertex_distances = left_vertex_distances.squeeze(0)
            right_vertex_distances = right_vertex_distances.squeeze(0)
            left_boundary_index = left_boundary_index.squeeze(0)
            right_boundary_index = right_boundary_index.squeeze(0)
        self.distances.closest_point_on_left_b[
            env_index, agent_index
        ] = left_boundary_index
        self.distances.closest_point_on_right_b[
            env_index, agent_index
        ] = right_boundary_index
        self.distances.left_boundaries[env_index, agent_index] = left_vertex_distances
        self.distances.right_boundaries[env_index, agent_index] = right_vertex_distances

        self.distances.boundaries[env_index, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[env_index, agent_index],
                    self.distances.right_boundaries[env_index, agent_index],
                )
            ),
            dim=-1,
        )

    def _get_route_progress_association_window(self, dtype):
        """Return a conservative one-step route-progress association window."""
        cached_window = self._route_progress_association_window
        if cached_window is not None and cached_window.dtype == dtype:
            return cached_window

        max_speed = max(float(agent.max_speed) for agent in self.world.agents)

        # Cover one maximum-speed timestep plus projection/interpolation noise,
        # while remaining much smaller than separated stations at a crossing.
        station_margin = max(
            0.05 * LENGTH_SCALE,
            0.5 * float(self.params.short_term_ref_path_spacing_m),
        )
        self._route_progress_association_window = torch.as_tensor(
            max_speed * float(self.world.dt) + station_margin,
            device=self.device,
            dtype=dtype,
        )
        return self._route_progress_association_window

    @staticmethod
    def _get_progress_from_cached_paths(
        points: torch.Tensor,
        polylines: torch.Tensor,
        segment_vectors: torch.Tensor,
        segment_lengths: torch.Tensor,
        segment_lengths_squared: torch.Tensor,
        valid_segments: torch.Tensor,
        cumulative_lengths: torch.Tensor,
        total_lengths: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        previous_progress: torch.Tensor | None = None,
        previous_progress_valid: torch.Tensor | None = None,
        is_loop: torch.Tensor | None = None,
        progress_window: torch.Tensor | None = None,
    ):
        """Compute route progress for a flattened batch using cached geometry.

        When continuity arguments are supplied, candidate projections are
        restricted to a local arc-length window around the previous station.
        The geometric projection remains fully vectorized; the global nearest
        result is used only if no valid local candidate exists.
        """
        line_starts = polylines[:, :-1, :]
        line_lens_safe = torch.clamp(
            segment_lengths_squared, min=torch.finfo(polylines.dtype).eps
        )
        point_vectors = points.unsqueeze(1) - line_starts
        projected_lengths = (
            torch.sum(point_vectors * segment_vectors, dim=-1) / line_lens_safe
        )
        clamped_lengths = torch.clamp(projected_lengths, 0, 1)
        closest_points = line_starts + segment_vectors * clamped_lengths.unsqueeze(-1)
        distances = torch.norm(closest_points - points.unsqueeze(1), dim=-1)
        distances = torch.where(
            valid_segments,
            distances,
            torch.full_like(distances, torch.finfo(polylines.dtype).max),
        )

        if progress_window is not None:
            if previous_progress is None or previous_progress_valid is None:
                raise ValueError(
                    "previous_progress and previous_progress_valid are required "
                    "when progress_window is provided"
                )

            eps = torch.finfo(polylines.dtype).eps
            route_total_safe = torch.clamp(total_lengths, min=eps)
            previous_progress = previous_progress.to(
                device=polylines.device, dtype=polylines.dtype
            )
            previous_progress_valid = previous_progress_valid.to(
                device=polylines.device, dtype=torch.bool
            )
            progress_window = torch.as_tensor(
                progress_window, device=polylines.device, dtype=polylines.dtype
            )
            if progress_window.ndim == 0:
                progress_window = progress_window.expand(points.shape[0])
            if is_loop is None:
                is_loop = torch.zeros(
                    points.shape[0], device=polylines.device, dtype=torch.bool
                )
            else:
                is_loop = is_loop.to(device=polylines.device, dtype=torch.bool)

            # Reuse the projection fractions already computed for the distance
            # test to obtain each segment's candidate route progress.
            candidate_progress = cumulative_lengths[:, :-1] + (
                clamped_lengths * segment_lengths
            )
            progress_delta = candidate_progress - previous_progress.unsqueeze(-1)
            wrapped_delta = torch.remainder(
                progress_delta + 0.5 * route_total_safe.unsqueeze(-1),
                route_total_safe.unsqueeze(-1),
            ) - 0.5 * route_total_safe.unsqueeze(-1)
            progress_delta = torch.where(
                is_loop.unsqueeze(-1), wrapped_delta, progress_delta
            )
            locally_reachable = (
                (~previous_progress_valid).unsqueeze(-1)
                | (progress_delta.abs() <= progress_window.unsqueeze(-1))
            ) & valid_segments
            has_local_candidate = locally_reachable.any(dim=-1)
            local_distances = torch.where(
                locally_reachable,
                distances,
                torch.full_like(distances, torch.finfo(polylines.dtype).max),
            )
            distances = torch.where(
                has_local_candidate.unsqueeze(-1), local_distances, distances
            )

        perpendicular_distances, closest_segment_indices = torch.min(distances, dim=-1)
        if batch_indices is None:
            batch_indices = torch.arange(points.shape[0], device=points.device)
        progress = (
            cumulative_lengths[batch_indices, closest_segment_indices]
            + clamped_lengths[batch_indices, closest_segment_indices]
            * segment_lengths[batch_indices, closest_segment_indices]
        )
        return perpendicular_distances, progress, total_lengths

    def _get_progress_for_selected_paths(
        self, points: torch.Tensor, env_index, agent_index: int
    ):
        """Project points onto selected paths with continuity constraints."""
        polylines = self.ref_paths_agent_related.long_term[env_index, agent_index]
        segment_vectors = self._ref_path_segment_vectors[env_index, agent_index]
        segment_lengths = self._ref_path_segment_lengths[env_index, agent_index]
        segment_lengths_squared = self._ref_path_segment_lengths_squared[
            env_index, agent_index
        ]
        valid_segments = self._ref_path_valid_segments[env_index, agent_index]
        cumulative_lengths = self._ref_path_cumulative_lengths[env_index, agent_index]
        total_lengths = self._ref_path_total_lengths[env_index, agent_index]
        previous_progress = self._route_progress[env_index, agent_index]
        previous_progress_valid = self._route_progress_cache_valid[
            env_index, agent_index
        ]
        is_loop = self.ref_paths_agent_related.is_loop[env_index, agent_index]

        if polylines.ndim == 2:
            polylines = polylines.unsqueeze(0)
            segment_vectors = segment_vectors.unsqueeze(0)
            segment_lengths = segment_lengths.unsqueeze(0)
            segment_lengths_squared = segment_lengths_squared.unsqueeze(0)
            valid_segments = valid_segments.unsqueeze(0)
            cumulative_lengths = cumulative_lengths.unsqueeze(0)
            total_lengths = total_lengths.unsqueeze(0)
            previous_progress = previous_progress.unsqueeze(0)
            previous_progress_valid = previous_progress_valid.unsqueeze(0)
            is_loop = is_loop.unsqueeze(0)

        if points.ndim == 1:
            points = points.unsqueeze(0)
        elif points.ndim == 3 and points.shape[-2] == 1:
            points = points.squeeze(-2)
        progress_window = self._get_route_progress_association_window(polylines.dtype)
        return self._get_progress_from_cached_paths(
            points=points,
            polylines=polylines,
            segment_vectors=segment_vectors,
            segment_lengths=segment_lengths,
            segment_lengths_squared=segment_lengths_squared,
            valid_segments=valid_segments,
            cumulative_lengths=cumulative_lengths,
            total_lengths=total_lengths,
            batch_indices=torch.arange(points.shape[0], device=points.device),
            previous_progress=previous_progress,
            previous_progress_valid=previous_progress_valid,
            is_loop=is_loop,
            progress_window=progress_window,
        )

    @staticmethod
    def _sample_cached_paths(
        polylines: torch.Tensor,
        interpolation_vectors: torch.Tensor,
        progress_values: torch.Tensor,
        n_points: torch.Tensor,
        is_loop: torch.Tensor,
        segment_lengths: torch.Tensor,
        cumulative_lengths: torch.Tensor,
        total_lengths: torch.Tensor,
        *,
        wrap_progress: bool,
        is_extend_non_loop: bool,
        batch_indices: torch.Tensor | None = None,
    ):
        """Sample flattened paths while reusing their static station grids."""
        batch_size, n_samples = progress_values.shape
        eps = torch.finfo(polylines.dtype).eps
        if batch_indices is None:
            batch_indices = torch.arange(batch_size, device=polylines.device)
        n_points = n_points.to(device=polylines.device, dtype=torch.long)
        last_point_indices = torch.clamp(n_points - 1, min=0)
        total_lengths_safe = torch.clamp(total_lengths, min=eps)

        if wrap_progress:
            progress_for_sampling = torch.where(
                is_loop.unsqueeze(-1),
                progress_values % total_lengths_safe.unsqueeze(-1),
                progress_values,
            )
        else:
            progress_for_sampling = progress_values
        before_start = progress_for_sampling < 0
        after_end = progress_for_sampling > total_lengths.unsqueeze(-1)
        progress_clamped = torch.minimum(
            torch.clamp(progress_for_sampling, min=0),
            total_lengths.unsqueeze(-1),
        )

        segment_indices = torch.searchsorted(cumulative_lengths, progress_clamped)
        upper_segment_indices = torch.clamp(n_points - 1, min=1).unsqueeze(-1)
        segment_indices = torch.minimum(
            torch.clamp(segment_indices, min=1), upper_segment_indices
        )
        segment_starts = torch.gather(cumulative_lengths, 1, segment_indices - 1)
        segment_ends = torch.gather(cumulative_lengths, 1, segment_indices)
        segment_denominators = torch.clamp(segment_ends - segment_starts, min=eps)
        segment_ratios = (progress_clamped - segment_starts) / segment_denominators

        gather_start_indices = (
            (segment_indices - 1).unsqueeze(-1).expand(-1, -1, polylines.shape[-1])
        )
        gather_end_indices = segment_indices.unsqueeze(-1).expand(
            -1, -1, polylines.shape[-1]
        )
        start_points = torch.gather(polylines, 1, gather_start_indices)
        end_points = torch.gather(polylines, 1, gather_end_indices)
        sampled_points = start_points + segment_ratios.unsqueeze(-1) * (
            end_points - start_points
        )

        degenerate_polylines = total_lengths <= eps
        if torch.any(degenerate_polylines):
            sampled_points = torch.where(
                degenerate_polylines.view(batch_size, 1, 1),
                polylines[:, :1, :].expand(-1, n_samples, -1),
                sampled_points,
            )

        if is_extend_non_loop:
            non_loop = ~is_loop
            needs_before = before_start & non_loop.unsqueeze(-1)
            needs_after = after_end & non_loop.unsqueeze(-1)
            if torch.any(needs_before):
                first_lengths = torch.clamp(segment_lengths[:, 0], min=eps)
                before_points = (
                    polylines[:, :1, :]
                    + (progress_for_sampling / first_lengths.unsqueeze(-1)).unsqueeze(
                        -1
                    )
                    * interpolation_vectors[:, :1, :]
                )
                sampled_points = torch.where(
                    needs_before.unsqueeze(-1), before_points, sampled_points
                )
            if torch.any(needs_after):
                last_segment_indices = torch.clamp(n_points - 2, min=0)
                last_lengths = torch.clamp(
                    segment_lengths[batch_indices, last_segment_indices],
                    min=eps,
                )
                last_points = polylines[batch_indices, last_point_indices, :]
                last_vectors = interpolation_vectors[
                    batch_indices, last_segment_indices, :
                ]
                after_points = last_points.unsqueeze(1) + (
                    (progress_for_sampling - total_lengths.unsqueeze(-1))
                    / last_lengths.unsqueeze(-1)
                ).unsqueeze(-1) * last_vectors.unsqueeze(1)
                sampled_points = torch.where(
                    needs_after.unsqueeze(-1), after_points, sampled_points
                )

        return sampled_points

    def snapshot_route_progress(self):
        """Save the last completed state's route progress before a new update."""
        valid = self._route_progress_cache_valid
        self._route_progress_previous[valid] = self._route_progress[valid]

    def _update_smooth_sampled_ref_paths_all_agents(
        self, agent_states: List[AgentState]
    ):
        """Update every smooth local path in one environment-agent batch."""
        batch_agent_count = self.batch_dim * self.n_agents
        positions = torch.stack(
            [agent_state.pos for agent_state in agent_states], dim=1
        ).reshape(batch_agent_count, 2)
        polylines = self.ref_paths_agent_related.long_term.reshape(
            batch_agent_count, self.params.max_ref_path_points, 2
        )
        n_points = self.ref_paths_agent_related.n_points_long_term.reshape(-1)
        is_loop = self.ref_paths_agent_related.is_loop.reshape(-1)
        segment_vectors = self._ref_path_segment_vectors.reshape(
            batch_agent_count, self.params.max_ref_path_points - 1, 2
        )
        segment_lengths = self._ref_path_segment_lengths.reshape(
            batch_agent_count, self.params.max_ref_path_points - 1
        )
        segment_lengths_squared = self._ref_path_segment_lengths_squared.reshape(
            batch_agent_count, self.params.max_ref_path_points - 1
        )
        valid_segments = self._ref_path_valid_segments.reshape(
            batch_agent_count, self.params.max_ref_path_points - 1
        )
        cumulative_lengths = self._ref_path_cumulative_lengths.reshape(
            batch_agent_count, self.params.max_ref_path_points
        )
        sampling_cumulative_lengths = (
            self._ref_path_sampling_cumulative_lengths.reshape(
                batch_agent_count, self.params.max_ref_path_points
            )
        )
        total_lengths = self._ref_path_total_lengths.reshape(-1)
        previous_progress = self._route_progress.reshape(-1)
        previous_progress_valid = self._route_progress_cache_valid.reshape(-1)
        progress_window = self._get_route_progress_association_window(polylines.dtype)

        _, route_progress, route_total_lengths = self._get_progress_from_cached_paths(
            points=positions,
            polylines=polylines,
            segment_vectors=segment_vectors,
            segment_lengths=segment_lengths,
            segment_lengths_squared=segment_lengths_squared,
            valid_segments=valid_segments,
            cumulative_lengths=cumulative_lengths,
            total_lengths=total_lengths,
            batch_indices=self._sampling_batch_indices,
            previous_progress=previous_progress,
            previous_progress_valid=previous_progress_valid,
            is_loop=is_loop,
            progress_window=progress_window,
        )
        self._route_progress[:] = route_progress.reshape(self.batch_dim, self.n_agents)
        self._route_progress_total_lengths[:] = route_total_lengths.reshape(
            self.batch_dim, self.n_agents
        )
        self._route_progress_cache_valid[:] = True
        route_total_safe = torch.clamp(
            route_total_lengths, min=torch.finfo(polylines.dtype).eps
        )
        progress_ratios = torch.clamp(route_progress / route_total_safe, 0, 1)
        base_progress = progress_ratios * total_lengths
        center_progress = (
            base_progress.unsqueeze(-1)
            + self.params.short_term_ref_path_spacing_m
            * self._short_term_progress_offsets
        )
        short_term = self._sample_cached_paths(
            polylines=polylines,
            interpolation_vectors=segment_vectors,
            progress_values=center_progress,
            n_points=n_points,
            is_loop=is_loop,
            segment_lengths=segment_lengths,
            cumulative_lengths=sampling_cumulative_lengths,
            total_lengths=total_lengths,
            wrap_progress=True,
            is_extend_non_loop=True,
            batch_indices=self._sampling_batch_indices,
        )

        wrapped_route_progress = torch.where(
            is_loop.unsqueeze(-1),
            route_progress.unsqueeze(-1) % route_total_safe.unsqueeze(-1),
            route_progress.unsqueeze(-1),
        )
        boundary_spacing = total_lengths / torch.clamp(
            n_points.to(polylines.dtype) - 1,
            min=torch.ones((), device=self.device, dtype=polylines.dtype),
        )
        boundary_progress = (
            wrapped_route_progress
            + boundary_spacing.unsqueeze(-1) * self._boundary_progress_offsets
        )
        left_boundaries = self.ref_paths_agent_related.left_boundary.reshape(
            batch_agent_count, self.params.max_ref_path_points, 2
        )
        right_boundaries = self.ref_paths_agent_related.right_boundary.reshape(
            batch_agent_count, self.params.max_ref_path_points, 2
        )
        left_boundary_vectors = self._left_boundary_segment_vectors.reshape(
            batch_agent_count, self.params.max_ref_path_points - 1, 2
        )
        right_boundary_vectors = self._right_boundary_segment_vectors.reshape(
            batch_agent_count, self.params.max_ref_path_points - 1, 2
        )
        nearing_left = self._sample_cached_paths(
            polylines=left_boundaries,
            interpolation_vectors=left_boundary_vectors,
            progress_values=boundary_progress,
            n_points=n_points,
            is_loop=is_loop,
            segment_lengths=segment_lengths,
            cumulative_lengths=sampling_cumulative_lengths,
            total_lengths=total_lengths,
            wrap_progress=False,
            is_extend_non_loop=True,
            batch_indices=self._sampling_batch_indices,
        )
        nearing_right = self._sample_cached_paths(
            polylines=right_boundaries,
            interpolation_vectors=right_boundary_vectors,
            progress_values=boundary_progress,
            n_points=n_points,
            is_loop=is_loop,
            segment_lengths=segment_lengths,
            cumulative_lengths=sampling_cumulative_lengths,
            total_lengths=total_lengths,
            wrap_progress=False,
            is_extend_non_loop=True,
            batch_indices=self._sampling_batch_indices,
        )

        self.ref_paths_agent_related.short_term[:] = short_term.reshape(
            self.batch_dim,
            self.n_agents,
            self.params.n_points_short_term,
            2,
        )
        self.ref_paths_agent_related.nearing_points_left_boundary[
            :
        ] = nearing_left.reshape(
            self.batch_dim,
            self.n_agents,
            self.params.n_points_nearing_boundary,
            2,
        )
        self.ref_paths_agent_related.nearing_points_right_boundary[
            :
        ] = nearing_right.reshape(
            self.batch_dim,
            self.n_agents,
            self.params.n_points_nearing_boundary,
            2,
        )

    def _update_sampled_ref_paths_agent_related(
        self, agent_pos, env_index, agent_index: int
    ):
        """Update short-term centerline and local boundary samples for one agent."""
        if self.params.is_smooth_sampling:
            (
                _,
                route_progress,
                route_total_length,
            ) = self._get_progress_for_selected_paths(
                points=agent_pos,
                env_index=env_index,
                agent_index=agent_index,
            )
            target_progress = self._route_progress[env_index, agent_index]
            self._route_progress[env_index, agent_index] = route_progress.reshape(
                target_progress.shape
            )
            self._route_progress_total_lengths[
                env_index, agent_index
            ] = route_total_length.reshape(target_progress.shape)
            self._route_progress_cache_valid[env_index, agent_index] = True
            self.ref_paths_agent_related.short_term[
                env_index, agent_index
            ] = get_route_progress_aligned_sampled_points(
                polyline=self.ref_paths_agent_related.long_term[env_index, agent_index],
                route_progress=route_progress,
                route_total_length=route_total_length,
                n_points_to_return=self.params.n_points_short_term,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[
                    env_index, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_index, agent_index
                ],
                n_points_shift=1,
                sample_spacing=self.params.short_term_ref_path_spacing_m,
                is_extend_non_loop=True,
            )
            self.ref_paths_agent_related.nearing_points_left_boundary[
                env_index, agent_index
            ] = get_reference_progress_aligned_sampled_points(
                reference_polyline=self.ref_paths_agent_related.long_term[
                    env_index, agent_index
                ],
                aligned_polyline=self.ref_paths_agent_related.left_boundary[
                    env_index, agent_index
                ],
                route_progress=route_progress,
                n_points_to_return=self.params.n_points_nearing_boundary,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[
                    env_index, agent_index
                ],
                n_points_reference=self.ref_paths_agent_related.n_points_long_term[
                    env_index, agent_index
                ],
                sample_interval=1,
                n_points_shift=-round(self.params.n_points_nearing_boundary / 2),
                is_extend_non_loop=True,
            )
            self.ref_paths_agent_related.nearing_points_right_boundary[
                env_index, agent_index
            ] = get_reference_progress_aligned_sampled_points(
                reference_polyline=self.ref_paths_agent_related.long_term[
                    env_index, agent_index
                ],
                aligned_polyline=self.ref_paths_agent_related.right_boundary[
                    env_index, agent_index
                ],
                route_progress=route_progress,
                n_points_to_return=self.params.n_points_nearing_boundary,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[
                    env_index, agent_index
                ],
                n_points_reference=self.ref_paths_agent_related.n_points_long_term[
                    env_index, agent_index
                ],
                sample_interval=1,
                n_points_shift=-round(self.params.n_points_nearing_boundary / 2),
                is_extend_non_loop=True,
            )
        else:
            _, route_progress, route_total_length = get_progress_on_polyline(
                point=agent_pos,
                polyline=self.ref_paths_agent_related.long_term[env_index, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_index, agent_index
                ],
            )
            target_progress = self._route_progress[env_index, agent_index]
            self._route_progress[env_index, agent_index] = route_progress.reshape(
                target_progress.shape
            )
            self._route_progress_total_lengths[
                env_index, agent_index
            ] = route_total_length.reshape(target_progress.shape)
            self._route_progress_cache_valid[env_index, agent_index] = True
            (
                _,
                self.distances.closest_point_on_ref_path[env_index, agent_index],
            ) = get_perpendicular_distances(
                point=agent_pos,
                polyline=self.ref_paths_agent_related.long_term[env_index, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_index, agent_index
                ],
            )
            (
                _,
                self.distances.closest_point_on_left_b[env_index, agent_index],
            ) = get_perpendicular_distances(
                point=agent_pos,
                polyline=self.ref_paths_agent_related.left_boundary[
                    env_index, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                    env_index, agent_index
                ],
            )
            (
                _,
                self.distances.closest_point_on_right_b[env_index, agent_index],
            ) = get_perpendicular_distances(
                point=agent_pos,
                polyline=self.ref_paths_agent_related.right_boundary[
                    env_index, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                    env_index, agent_index
                ],
            )

            (
                self.ref_paths_agent_related.short_term[env_index, agent_index],
                self.ref_paths_agent_related.short_term_indices[env_index, agent_index],
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.long_term[env_index, agent_index],
                index_closest_point=self.distances.closest_point_on_ref_path[
                    env_index, agent_index
                ],
                n_points_to_return=self.params.n_points_short_term,
                device=self.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[
                    env_index, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_index, agent_index
                ],
                sample_interval=1,
                n_points_shift=1,
                is_extend_non_loop=True,
            )
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_index, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[
                    env_index, agent_index
                ],
                index_closest_point=self.distances.closest_point_on_left_b[
                    env_index, agent_index
                ],
                n_points_to_return=self.params.n_points_nearing_boundary,
                device=self.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[
                    env_index, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                    env_index, agent_index
                ],
                sample_interval=1,
                n_points_shift=-round(self.params.n_points_nearing_boundary / 2),
                is_extend_non_loop=True,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_index, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[
                    env_index, agent_index
                ],
                index_closest_point=self.distances.closest_point_on_right_b[
                    env_index, agent_index
                ],
                n_points_to_return=self.params.n_points_nearing_boundary,
                device=self.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[
                    env_index, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                    env_index, agent_index
                ],
                sample_interval=1,
                n_points_shift=-round(self.params.n_points_nearing_boundary / 2),
                is_extend_non_loop=True,
            )

    def reset_init_distances_and_short_term_ref_path(
        self, agent_states: List[AgentState], env_j, agent_index
    ):
        """
        This function calculates the distances from the agent's center of gravity (CG) to its reference path and boundaries,
        and computes the positions of the four vertices of the agent. It also determines the short-term reference paths
        for the agent based on the long-term reference paths and the agent's current position.
        """
        agent_pos = agent_states[agent_index].pos[env_j, :]

        self.vertices[env_j, agent_index] = get_rectangle_vertices(
            center=agent_pos,
            yaw=agent_states[agent_index].rot[env_j, :],
            width=self.agent_widths[agent_index],
            length=self.agent_lengths[agent_index],
            is_close_shape=True,
        )

        self._update_sampled_ref_paths_agent_related(
            agent_pos=agent_pos, env_index=env_j, agent_index=agent_index
        )
        self._route_progress_previous[env_j, agent_index] = self._route_progress[
            env_j, agent_index
        ]
        self._route_progress_cache_valid[env_j, agent_index] = True
        if self.params.is_use_simple_road_distance:
            self._update_simple_road_distances(
                agent_pos=agent_pos, env_index=env_j, agent_index=agent_index
            )
        else:
            self._update_exact_road_distances(
                agent_pos=agent_pos, env_index=env_j, agent_index=agent_index
            )

    """
    ------------------------------- UPDATES -------------------------------
    """

    def update_distances(self, agent_states: List[AgentState], agent_index: int):
        if agent_index == 0:
            self.update_mutual_distances(agent_states)
            if self.params.is_smooth_sampling:
                self._update_smooth_sampled_ref_paths_all_agents(agent_states)

        agent_pos = agent_states[agent_index].pos
        if not self.params.is_smooth_sampling:
            self._update_sampled_ref_paths_agent_related(
                agent_pos=agent_pos,
                env_index=slice(None),
                agent_index=agent_index,
            )
        if self.params.is_use_simple_road_distance:
            self._update_simple_road_distances(
                agent_pos=agent_pos, env_index=slice(None), agent_index=agent_index
            )
        else:
            self._update_exact_road_distances(
                agent_pos=agent_pos, env_index=slice(None), agent_index=agent_index
            )

    def update_vertices(self, agent_states: List[AgentState]):
        for a_i in range(self.n_agents):
            self.vertices[:, a_i] = get_rectangle_vertices(
                center=agent_states[a_i].pos,
                yaw=agent_states[a_i].rot,
                width=self.agent_widths[a_i],
                length=self.agent_lengths[a_i],
                is_close_shape=True,
            )

    def update_ref_paths_agent_related(self, agent_index: int):
        self._update_sampled_ref_paths_agent_related(
            agent_pos=self.world.agents[agent_index].state.pos,
            env_index=slice(None),
            agent_index=agent_index,
        )
