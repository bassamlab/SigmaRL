from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import torch

from sigmarl.constants import AGENTS
from sigmarl.helper_scenario import (
    Distances,
    ReferencePathsAgentRelated,
    get_perpendicular_distances,
    get_rectangle_vertices,
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
    sample_interval_ref_path: int
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
            long_term_merge_in=self.map.parser.reference_paths_merge_in,
            long_term_merge_out=self.map.parser.reference_paths_merge_out,
            point_extended_all=torch.zeros(
                (
                    len(self.map.parser.reference_paths),
                    self.params.n_points_short_term
                    * self.params.sample_interval_ref_path,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),  # Not interesting, may be useful in the future
            point_extended_intersection=torch.zeros(
                (
                    len(self.map.parser.reference_paths_intersection),
                    self.params.n_points_short_term
                    * self.params.sample_interval_ref_path,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),
            point_extended_merge_in=torch.zeros(
                (
                    len(self.map.parser.reference_paths_merge_in),
                    self.params.n_points_short_term
                    * self.params.sample_interval_ref_path,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),
            point_extended_merge_out=torch.zeros(
                (
                    len(self.map.parser.reference_paths_merge_out),
                    self.params.n_points_short_term
                    * self.params.sample_interval_ref_path,
                    2,
                ),
                device=self.device,
                dtype=torch.float32,
            ),
            sample_interval=torch.tensor(
                self.params.sample_interval_ref_path,
                device=self.device,
                dtype=torch.int32,
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
                (self.batch_dim, self.n_agents, 1 + 4),
                device=self.device,
                dtype=torch.float32,
            ),  # The first entry for the center, the last 4 entries for the four vertices
            right_boundaries=torch.zeros(
                (self.batch_dim, self.n_agents, 1 + 4),
                device=self.device,
                dtype=torch.float32,
            ),
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

        # The shape of each agent is considered a rectangle with 4 vertices.
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros(
            (self.batch_dim, self.n_agents, 5, 2),
            device=self.device,
            dtype=torch.float32,
        )

    def _extend_map_related_ref_path(self):
        # Extended the reference path by several points along the last vector of the center line
        idx_broadcasting_entend = torch.arange(
            1,
            self.params.n_points_short_term * self.params.sample_interval_ref_path + 1,
            device=self.device,
            dtype=torch.int32,
        ).unsqueeze(1)

        for idx, i_path in enumerate(self.map.parser.reference_paths):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_all[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_intersection):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_intersection[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_merge_in):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_in[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_merge_out):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_out[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )

    def _reset_agent_related_ref_path(
        self, env_i, agent_index, ref_path, path_id, extended_points
    ):
        """
        This function resets the agent-related reference paths and updates various related attributes
        for a specified agent in an environment.
        """
        # Long-term reference paths for agents
        n_points_long_term = ref_path["center_line"].shape[0]

        self.ref_paths_agent_related.long_term[
            env_i, agent_index, 0:n_points_long_term, :
        ] = ref_path["center_line"]

        self.ref_paths_agent_related.long_term[
            env_i,
            agent_index,
            n_points_long_term : (
                n_points_long_term
                + self.params.n_points_short_term * self.params.sample_interval_ref_path
            ),
            :,
        ] = extended_points[path_id, :, :]

        self.ref_paths_agent_related.long_term[
            env_i,
            agent_index,
            (
                n_points_long_term
                + self.params.n_points_short_term * self.params.sample_interval_ref_path
            ) :,
            :,
        ] = extended_points[path_id, -1, :]

        self.ref_paths_agent_related.n_points_long_term[
            env_i, agent_index
        ] = n_points_long_term

        self.ref_paths_agent_related.long_term_vec_normalized[
            env_i, agent_index, 0 : n_points_long_term - 1, :
        ] = ref_path["center_line_vec_normalized"]

        self.ref_paths_agent_related.long_term_vec_normalized[
            env_i,
            agent_index,
            (n_points_long_term - 1) : (
                n_points_long_term
                - 1
                + self.params.n_points_short_term * self.params.sample_interval_ref_path
            ),
            :,
        ] = ref_path["center_line_vec_normalized"][-1, :]

        n_points_left_b = ref_path["left_boundary_shared"].shape[0]

        self.ref_paths_agent_related.left_boundary[
            env_i, agent_index, 0:n_points_left_b, :
        ] = ref_path["left_boundary_shared"]

        self.ref_paths_agent_related.left_boundary[
            env_i, agent_index, n_points_left_b:, :
        ] = ref_path["left_boundary_shared"][-1, :]

        self.ref_paths_agent_related.n_points_left_b[
            env_i, agent_index
        ] = n_points_left_b

        n_points_right_b = ref_path["right_boundary_shared"].shape[0]

        self.ref_paths_agent_related.right_boundary[
            env_i, agent_index, 0:n_points_right_b, :
        ] = ref_path["right_boundary_shared"]

        self.ref_paths_agent_related.right_boundary[
            env_i, agent_index, n_points_right_b:, :
        ] = ref_path["right_boundary_shared"][-1, :]

        self.ref_paths_agent_related.n_points_right_b[
            env_i, agent_index
        ] = n_points_right_b

        self.ref_paths_agent_related.entry[env_i, agent_index, 0, :] = ref_path[
            "left_boundary_shared"
        ][0, :]
        self.ref_paths_agent_related.entry[env_i, agent_index, 1, :] = ref_path[
            "right_boundary_shared"
        ][0, :]

        self.ref_paths_agent_related.exit[env_i, agent_index, 0, :] = ref_path[
            "left_boundary_shared"
        ][-1, :]
        self.ref_paths_agent_related.exit[env_i, agent_index, 1, :] = ref_path[
            "right_boundary_shared"
        ][-1, :]

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

    def reset_init_distances_and_short_term_ref_path(
        self, agent_states: List[AgentState], env_j, agent_index
    ):
        """
        This function calculates the distances from the agent's center of gravity (CG) to its reference path and boundaries,
        and computes the positions of the four vertices of the agent. It also determines the short-term reference paths
        for the agent based on the long-term reference paths and the agent's current position.
        """
        # Distance from the center of gravity (CG) of the agent to its reference path
        (
            self.distances.ref_paths[env_j, agent_index],
            self.distances.closest_point_on_ref_path[env_j, agent_index],
        ) = get_perpendicular_distances(
            point=agent_states[agent_index].pos[env_j, :],
            polyline=self.ref_paths_agent_related.long_term[env_j, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                env_j, agent_index
            ],
        )
        # Distances from CG to left boundary
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[env_j, agent_index],
        ) = get_perpendicular_distances(
            point=agent_states[agent_index].pos[env_j, :],
            polyline=self.ref_paths_agent_related.left_boundary[env_j, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                env_j, agent_index
            ],
        )
        self.distances.left_boundaries[env_j, agent_index, 0] = center_2_left_b - (
            AGENTS["width"] / 2
        )
        # Distances from CG to right boundary
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[env_j, agent_index],
        ) = get_perpendicular_distances(
            point=agent_states[agent_index].pos[env_j, :],
            polyline=self.ref_paths_agent_related.right_boundary[env_j, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                env_j, agent_index
            ],
        )
        self.distances.right_boundaries[env_j, agent_index, 0] = center_2_right_b - (
            AGENTS["width"] / 2
        )
        # Calculate the positions of the four vertices of the agents
        self.vertices[env_j, agent_index] = get_rectangle_vertices(
            center=agent_states[agent_index].pos[env_j, :],
            yaw=agent_states[agent_index].rot[env_j, :],
            width=AGENTS["width"],
            length=AGENTS["length"],
            is_close_shape=True,
        )
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            (
                self.distances.left_boundaries[env_j, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[env_j, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                    env_j, agent_index
                ],
            )
            (
                self.distances.right_boundaries[env_j, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[
                    env_j, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                    env_j, agent_index
                ],
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[env_j, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[env_j, agent_index],
                    self.distances.right_boundaries[env_j, agent_index],
                )
            ),
            dim=-1,
        )

        # Get the short-term reference paths
        (
            self.ref_paths_agent_related.short_term[env_j, agent_index],
            _,
        ) = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[env_j, agent_index],
            index_closest_point=self.distances.closest_point_on_ref_path[
                env_j, agent_index
            ],
            n_points_to_return=self.params.n_points_short_term,
            device=self.device,
            is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                env_j, agent_index
            ],
            sample_interval=self.params.sample_interval_ref_path,
            n_points_shift=1,
        )

        if not self.params.observe_distance_to_boundaries:
            # Get nearing points on boundaries
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_j, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[env_j, agent_index],
                index_closest_point=self.distances.closest_point_on_left_b[
                    env_j, agent_index
                ],
                n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                device=self.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[
                    env_j, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_j, agent_index
                ],
                sample_interval=1,
                n_points_shift=1,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_j, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[
                    env_j, agent_index
                ],
                index_closest_point=self.distances.closest_point_on_right_b[
                    env_j, agent_index
                ],
                n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                device=self.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[
                    env_j, agent_index
                ],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_j, agent_index
                ],
                sample_interval=1,
                n_points_shift=1,
            )

    """
    ------------------------------- UPDATES -------------------------------
    """

    def update_distances(self, agent_states: List[AgentState], agent_index: int):
        if agent_index == 0:
            self.update_mutual_distances(agent_states)

        # Distance from the center of gravity (CG) of the agent to its reference path
        (
            self.distances.ref_paths[:, agent_index],
            self.distances.closest_point_on_ref_path[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent_states[agent_index].pos,
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                :, agent_index
            ],
        )
        # Distances from CG to left boundary
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent_states[agent_index].pos[:, :],
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                :, agent_index
            ],
        )
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - (
            AGENTS["width"] / 2
        )
        # Distances from CG to right boundary
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent_states[agent_index].pos[:, :],
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                :, agent_index
            ],
        )
        self.distances.right_boundaries[:, agent_index, 0] = center_2_right_b - (
            AGENTS["width"] / 2
        )
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            (
                self.distances.left_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                    :, agent_index
                ],
            )
            (
                self.distances.right_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                    :, agent_index
                ],
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[:, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:, agent_index],
                    self.distances.right_boundaries[:, agent_index],
                )
            ),
            dim=-1,
        )

    def update_vertices(self, agent_states: List[AgentState]):
        for a_i in range(self.n_agents):
            self.vertices[:, a_i] = get_rectangle_vertices(
                center=agent_states[a_i].pos,
                yaw=agent_states[a_i].rot,
                width=AGENTS["width"],
                length=AGENTS["length"],
                is_close_shape=True,
            )

    def update_ref_paths_agent_related(self, agent_index: int):
        (
            self.ref_paths_agent_related.short_term[:, agent_index],
            _,
        ) = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            index_closest_point=self.distances.closest_point_on_ref_path[
                :, agent_index
            ],
            n_points_to_return=self.params.n_points_short_term,
            device=self.device,
            is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                :, agent_index
            ],
            sample_interval=self.params.sample_interval_ref_path,
        )

        if not self.params.observe_distance_to_boundaries:
            # Get nearing points on boundaries
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_left_b[
                    :, agent_index
                ],
                n_points_to_return=self.params.n_points_nearing_boundary,
                device=self.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
                sample_interval=1,
                n_points_shift=-2,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_right_b[
                    :, agent_index
                ],
                n_points_to_return=self.params.n_points_nearing_boundary,
                device=self.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
                sample_interval=1,
                n_points_shift=-2,
            )
