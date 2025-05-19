from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from sigmarl.constants import AGENTS
from sigmarl.helper_scenario import (
    Thresholds,
    Constants,
    Normalizers,
    Observations,
    CircularBuffer,
    angle_eliminate_two_pi,
    transform_from_global_to_local_coordinate,
)
from sigmarl.helper_training import WorldCustom
from sigmarl.map_manager import MapManager
from sigmarl.scenarios.observations.observation_provider import (
    ObservationProviderParameters,
    ObservationProvider,
    AgentState,
)
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt import WorldStateRT
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt_sim import (
    WorldStateRTSimulation,
)


@dataclass
class ObservationProviderParametersRT(ObservationProviderParameters):
    n_observed_steps: int
    n_points_nearing_boundary: int
    n_nearing_agents_observed: int
    n_points_short_term: int
    is_observe_distance_to_boundaries: bool
    is_ego_view: bool
    is_using_opponent_modeling: bool
    is_apply_mask: bool
    is_partial_observation: bool
    is_obs_steering: bool
    is_observe_vertices: bool
    is_observe_distance_to_agents: bool
    is_observe_ref_path_other_agents: bool
    is_observe_distance_to_center_line: bool


class ObservationProviderRT(ObservationProvider):
    def __init__(
        self,
        params: ObservationProviderParametersRT,
        constants: Constants,
        normalizers: Normalizers,
        thresholds: Thresholds,
        map: MapManager,
        world_state: WorldStateRT,
    ):

        super().__init__(params, constants, normalizers)

        self.params = params

        self.thresholds = thresholds
        self.map = map
        self.world_state = world_state

        self.observations = Observations(
            n_nearing_agents=torch.tensor(
                params.n_nearing_agents_observed,
                device=self.device,
                dtype=torch.int32,
            ),
            obs_noise_level=torch.tensor(
                params.obs_noise_level, device=self.device, dtype=torch.float32
            ),
            n_stored_steps=torch.tensor(
                params.n_stored_steps, device=self.device, dtype=torch.int32
            ),
            n_observed_steps=torch.tensor(
                params.n_observed_steps, device=self.device, dtype=torch.int32
            ),
            nearing_agents_indices=torch.zeros(
                (self.batch_dim, self.n_agents, params.n_nearing_agents_observed),
                device=self.device,
                dtype=torch.int32,
            ),
        )

        # initialize observations
        assert (
            self.observations.n_stored_steps >= 1
        ), "The number of stored steps should be at least 1."
        assert (
            self.observations.n_observed_steps >= 1
        ), "The number of observed steps should be at least 1."
        assert (
            self.observations.n_stored_steps >= self.observations.n_observed_steps
        ), "The number of stored steps should be greater or equal than the number of observed steps."

        if self.params.is_ego_view:
            self.observations.past_pos = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_rot = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vertices = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        4,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vel = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.params.n_points_short_term,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.params.n_points_nearing_boundary,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.params.n_points_nearing_boundary,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
        else:
            # Bird view
            self.observations.past_pos = CircularBuffer(
                torch.zeros(
                    (self.params.n_stored_steps, self.batch_dim, self.n_agents, 2),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_rot = CircularBuffer(
                torch.zeros(
                    (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vertices = CircularBuffer(
                torch.zeros(
                    (self.params.n_stored_steps, self.batch_dim, self.n_agents, 4, 2),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vel = CircularBuffer(
                torch.zeros(
                    (self.params.n_stored_steps, self.batch_dim, self.n_agents, 2),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.params.n_points_short_term,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.params.n_points_nearing_boundary,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        self.params.n_stored_steps,
                        self.batch_dim,
                        self.n_agents,
                        self.params.n_points_nearing_boundary,
                        2,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            )

        self.observations.past_action_vel = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_action_steering = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_ref_path = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_boundaries = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_left_boundary = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_right_boundary = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_agents = CircularBuffer(
            torch.zeros(
                (
                    self.params.n_stored_steps,
                    self.batch_dim,
                    self.n_agents,
                    self.n_agents,
                ),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_lengths = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_widths = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.observations.past_steering = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )
        )

    """
    ------------------------------- STATE MANAGEMENT -------------------------------
    """

    def update_state(self, agent_states: List[AgentState]):
        positions_global = torch.stack([s.pos for s in agent_states], dim=0).transpose(
            0, 1
        )
        rotations_global = (
            torch.stack([s.rot for s in agent_states], dim=0)
            .transpose(0, 1)
            .squeeze(-1)
        )
        steering_global = angle_eliminate_two_pi(
            torch.stack([s.steering for s in agent_states], dim=0)
            .transpose(0, 1)
            .squeeze(-1)
        )

        lengths_global = torch.tensor(
            [AGENTS["length"] for _ in range(len(agent_states))],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.batch_dim, 1)

        widths_global = torch.tensor(
            [AGENTS["width"] for _ in range(len(agent_states))],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.batch_dim, 1)

        # Add new observation & normalize
        self.observations.past_distance_to_agents.add(
            self.world_state.distances.agents / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_ref_path.add(
            self.world_state.distances.ref_paths / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_left_boundary.add(
            torch.min(self.world_state.distances.left_boundaries, dim=-1)[0]
            / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_right_boundary.add(
            torch.min(self.world_state.distances.right_boundaries, dim=-1)[0]
            / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_boundaries.add(
            self.world_state.distances.boundaries / self.normalizers.distance_lanelet
        )
        self.observations.past_lengths.add(
            lengths_global / self.normalizers.distance_agent
        )  # Use distance to agents as the normalizer
        self.observations.past_widths.add(
            widths_global / self.normalizers.distance_agent
        )
        self.observations.past_steering.add(steering_global / self.normalizers.rot)

        if self.params.is_ego_view:
            pos_i_others = torch.zeros(
                (self.batch_dim, self.n_agents, self.n_agents, 2),
                device=self.device,
                dtype=torch.float32,
            )  # Positions of other agents relative to agent i
            rot_i_others = torch.zeros(
                (self.batch_dim, self.n_agents, self.n_agents),
                device=self.device,
                dtype=torch.float32,
            )  # Rotations of other agents relative to agent i
            vel_i_others = torch.zeros(
                (self.batch_dim, self.n_agents, self.n_agents, 2),
                device=self.device,
                dtype=torch.float32,
            )  # Velocities of other agents relative to agent i
            ref_i_others = torch.zeros_like(
                (self.observations.past_short_term_ref_points.get_latest())
            )  # Reference paths of other agents relative to agent i
            l_b_i_others = torch.zeros_like(
                (self.observations.past_left_boundary.get_latest())
            )  # Left boundaries of other agents relative to agent i
            r_b_i_others = torch.zeros_like(
                (self.observations.past_right_boundary.get_latest())
            )  # Right boundaries of other agents relative to agent i
            ver_i_others = torch.zeros_like(
                (self.observations.past_vertices.get_latest())
            )  # Vertices of other agents relative to agent i

            for a_i in range(self.n_agents):
                pos_i = agent_states[a_i].pos
                rot_i = agent_states[a_i].rot

                # Store new observation - position
                pos_i_others[:, a_i] = transform_from_global_to_local_coordinate(
                    pos_i=pos_i,
                    pos_j=positions_global,
                    rot_i=rot_i,
                )

                # Store new observation - rotation
                rot_i_others[:, a_i] = angle_eliminate_two_pi(rotations_global - rot_i)

                for a_j in range(self.n_agents):
                    # Store new observation - velocities
                    rot_rel = rot_i_others[:, a_i, a_j].unsqueeze(1)
                    vel_abs = torch.norm(agent_states[a_j].vel, dim=1).unsqueeze(
                        1
                    )  # TODO Check if relative velocities here are better
                    vel_i_others[:, a_i, a_j] = torch.hstack(
                        (vel_abs * torch.cos(rot_rel), vel_abs * torch.sin(rot_rel))
                    )

                    # Store new observation - reference paths
                    ref_i_others[
                        :, a_i, a_j
                    ] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=self.world_state.ref_paths_agent_related.short_term[
                            :, a_j
                        ],
                        rot_i=rot_i,
                    )

                    # Store new observation - left boundary
                    if not self.params.is_observe_distance_to_boundaries:
                        l_b_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.world_state.ref_paths_agent_related.nearing_points_left_boundary[
                                :, a_j
                            ],
                            rot_i=rot_i,
                        )

                        # Store new observation - right boundary
                        r_b_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.world_state.ref_paths_agent_related.nearing_points_right_boundary[
                                :, a_j
                            ],
                            rot_i=rot_i,
                        )

                    # Store new observation - vertices
                    ver_i_others[
                        :, a_i, a_j
                    ] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=self.world_state.vertices[:, a_j, 0:4, :],
                        rot_i=rot_i,
                    )
            # Add new observations & normalize
            self.observations.past_pos.add(
                pos_i_others
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
            self.observations.past_vel.add(vel_i_others / self.normalizers.v)
            self.observations.past_short_term_ref_points.add(
                ref_i_others
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_left_boundary.add(
                l_b_i_others
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_right_boundary.add(
                r_b_i_others
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_vertices.add(
                ver_i_others
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )

        else:  # Global coordinate system
            # Store new observations
            self.observations.past_pos.add(
                positions_global
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_vel.add(
                torch.stack([s.vel for s in agent_states], dim=1) / self.normalizers.v
            )
            self.observations.past_rot.add(
                angle_eliminate_two_pi(rotations_global[:]) / self.normalizers.rot
            )
            self.observations.past_vertices.add(
                self.world_state.vertices[:, :, 0:4, :]
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_short_term_ref_points.add(
                self.world_state.ref_paths_agent_related.short_term[:]
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_left_boundary.add(
                self.world_state.ref_paths_agent_related.nearing_points_left_boundary
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_right_boundary.add(
                self.world_state.ref_paths_agent_related.nearing_points_right_boundary
                / (
                    self.normalizers.pos
                    if self.params.is_ego_view
                    else self.normalizers.pos_world
                )
            )

            if self.params.is_apply_mask:
                # Determine the current lanelet IDs of all agents of all envs for later use
                self.map.determine_current_lanelet(positions_global)

    """
    ------------------------------- OBSERVATIONS -------------------------------
    """

    def get_observation(self, agent_index):
        # Observation of other agents
        obs_other_agents = self._observe_other_agents(agent_index)

        obs_self = self._observe_self(agent_index)

        obs_self.append(obs_other_agents)  # Append the observations of other agents

        obs_all = [o for o in obs_self if o is not None]  # Filter out None values

        obs = torch.hstack(obs_all)  # Convert from list to tensor

        if self.params.is_using_opponent_modeling:
            # Zero-padding as a placeholder for actions of surrounding agents
            obs = F.pad(
                obs,
                (0, self.params.n_nearing_agents_observed * AGENTS["n_actions"]),
            )

        if self.params.is_obs_noise:
            # Add sensor noise if required
            obs = obs + (
                self.observations.obs_noise_level
                * torch.rand_like(obs, device=self.device, dtype=torch.float32)
            )

        return obs

    def _observe_other_agents(self, agent_index):
        """Observe surrounding agents."""
        ##################################################
        ## Observation of other agents
        ##################################################
        if self.params.is_partial_observation:
            # Each agent observes only a fixed number of nearest agents
            (
                nearing_agents_distances,
                self.observations.nearing_agents_indices[:, agent_index],
            ) = torch.topk(
                self.world_state.distances.agents[:, agent_index],
                k=self.observations.n_nearing_agents,
                largest=False,
            )

            if self.params.is_apply_mask:
                # Two kinds of agents will be masked by ego agents:
                # 1. By distance: agents that are distant to the ego agents
                # 2. By lanelet relation: agents whose lanelets are not the neighboring lanelets or the same lanelets of the ego agents
                masked_agents_by_distance = (
                    nearing_agents_distances >= self.thresholds.distance_mask_agents
                )
                # print(f"masked_agents_by_distance = {masked_agents_by_distance}")
                if len(self.map.parser.neighboring_lanelets_idx) != 0:
                    # Mask agents by lanelets
                    masked_agents_by_lanelets = (
                        self.map.determine_masked_agents_by_lanelets(
                            agent_index,
                            self.observations.nearing_agents_indices[:, agent_index],
                        )
                    )
                else:
                    masked_agents_by_lanelets = torch.zeros(
                        (
                            self.batch_dim,
                            self.params.n_nearing_agents_observed,
                        ),
                        device=self.device,
                        dtype=torch.bool,
                    )

                masked_agents = masked_agents_by_distance | masked_agents_by_lanelets

            else:
                # Otherwise no agents will be masked
                masked_agents = torch.zeros(
                    (self.batch_dim, self.params.n_nearing_agents_observed),
                    device=self.device,
                    dtype=torch.bool,
                )

            indexing_tuple_1 = (
                (self.constants.env_idx_broadcasting,)
                + ((agent_index,) if self.params.is_ego_view else ())
                + (self.observations.nearing_agents_indices[:, agent_index],)
            )

            # Positions of nearing agents
            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 2]
            obs_pos_other_agents[
                masked_agents
            ] = self.constants.mask_one  # Position mask

            # Rotations of nearing agents
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_rot_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Rotation mask

            # Lengths and widths of nearing agents
            obs_lengths_other_agents = self.observations.past_lengths.get_latest()[
                self.constants.env_idx_broadcasting,
                self.observations.nearing_agents_indices[:, agent_index],
            ]
            obs_widths_other_agents = self.observations.past_widths.get_latest()[
                self.constants.env_idx_broadcasting,
                self.observations.nearing_agents_indices[:, agent_index],
            ]
            obs_steering_other_agents = self.observations.past_steering.get_latest()[
                self.constants.env_idx_broadcasting,
                self.observations.nearing_agents_indices[:, agent_index],
            ]
            obs_steering_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Steering mask

            # Velocities of nearing agents
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_vel_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Velocity mask

            # Reference paths of nearing agents
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_1
                ]
            )  # [batch_size, n_nearing_agents, n_points_short_term, 2]
            obs_ref_path_other_agents[
                masked_agents
            ] = self.constants.mask_one  # Reference-path mask

            # vertices of nearing agents
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 4, 2]
            obs_vertices_other_agents[
                masked_agents
            ] = self.constants.mask_one  # Reference-path mask

            # Distances to nearing agents
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[
                    self.constants.env_idx_broadcasting,
                    agent_index,
                    self.observations.nearing_agents_indices[:, agent_index],
                ]
            )  # [batch_size, n_nearing_agents]
            obs_distance_other_agents[
                masked_agents
            ] = self.constants.mask_one  # Distance mask

        else:
            indexing_tuple_2 = (self.constants.env_idx_broadcasting.squeeze(-1),) + (
                (agent_index,) if self.params.is_ego_view else ()
            )

            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 2]
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, (n_agents)]
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 2]
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_2
                ]
            )  # [batch_size, n_agents, n_points_short_term, 2]
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 4, 2]
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[indexing_tuple_2]
            )  # [batch_size, n_agents]
            obs_distance_other_agents[
                indexing_tuple_2
            ] = 0  # Reset self-self distance to zero
            obs_lengths_other_agents = self.observations.past_lengths.get_latest()[
                indexing_tuple_2
            ]
            obs_widths_other_agents = self.observations.past_widths.get_latest()[
                indexing_tuple_2
            ]
            obs_steering_other_agents = self.observations.past_steering.get_latest()[
                indexing_tuple_2
            ]

        # Flatten the last dimensions to combine all features into a single dimension
        obs_pos_other_agents_flat = obs_pos_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_rot_other_agents_flat = obs_rot_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vel_other_agents_flat = obs_vel_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vertices_other_agents_flat = obs_vertices_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_distance_other_agents_flat = obs_distance_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_lengths_other_agents_flat = obs_lengths_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_widths_other_agents_flat = obs_widths_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_steering_other_agents_flat = obs_steering_other_agents.reshape(
            self.batch_dim, self.observations.n_nearing_agents, -1
        )

        # Observation of other agents
        obs_others_list = [
            (
                obs_vertices_other_agents_flat
                if self.params.is_observe_vertices
                else torch.cat(  # [other] vertices
                    [
                        obs_pos_other_agents_flat,  # [others] positions
                        obs_rot_other_agents_flat,  # [others] rotations
                        obs_lengths_other_agents_flat,  # [others] lengths
                        obs_widths_other_agents_flat,  # [others] widths
                    ],
                    dim=-1,
                )
            ),
            obs_vel_other_agents_flat,  # [others] velocities
            obs_steering_other_agents_flat
            if self.params.is_obs_steering
            else None,  # [others] steering angles
            (
                obs_distance_other_agents_flat
                if self.params.is_observe_distance_to_agents
                else None
            ),  # [others] mutual distances
            (
                obs_ref_path_other_agents_flat
                if self.params.is_observe_ref_path_other_agents
                else None
            ),  # [others] reference paths
        ]
        obs_others_list = [
            o for o in obs_others_list if o is not None
        ]  # Filter out None values
        obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(
            self.batch_dim, -1
        )  # [batch_size, -1]

        return obs_other_agents

    def _observe_self(self, agent_index):
        """Observe the given agent itself."""
        indexing_tuple_3 = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index,) if self.params.is_ego_view else ())
        )
        indexing_tuple_vel = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index, 0) if self.params.is_ego_view else ())
        )  # In local coordinate system, only the first component is interesting, as the second is always 0
        # All observations
        obs_self = [
            (
                None
                if self.params.is_ego_view
                else self.observations.past_pos.get_latest()[indexing_tuple_3].reshape(
                    self.batch_dim, -1
                )
            ),  # [own] position,
            (
                None
                if self.params.is_ego_view
                else self.observations.past_rot.get_latest()[indexing_tuple_3].reshape(
                    self.batch_dim, -1
                )
            ),  # [own] rotation,
            self.observations.past_vel.get_latest()[indexing_tuple_vel].reshape(
                self.batch_dim, -1
            ),  # [own] velocity
            self.observations.past_steering.get_latest()[:, agent_index].reshape(
                self.batch_dim, -1
            )
            if self.params.is_obs_steering
            else None,  # [own] steering angle
            self.observations.past_short_term_ref_points.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.batch_dim, -1
            ),  # [own] short-term reference path
            (
                self.observations.past_distance_to_ref_path.get_latest()[
                    :, agent_index
                ].reshape(self.batch_dim, -1)
                if self.params.is_observe_distance_to_center_line
                else None
            ),  # [own] distances to reference paths
            (
                self.observations.past_distance_to_left_boundary.get_latest()[
                    :, agent_index
                ].reshape(self.batch_dim, -1)
                if self.params.is_observe_distance_to_boundaries
                else self.observations.past_left_boundary.get_latest()[
                    indexing_tuple_3
                ].reshape(self.batch_dim, -1)
            ),  # [own] left boundaries
            (
                self.observations.past_distance_to_right_boundary.get_latest()[
                    :, agent_index
                ].reshape(self.batch_dim, -1)
                if self.params.is_observe_distance_to_boundaries
                else self.observations.past_right_boundary.get_latest()[
                    indexing_tuple_3
                ].reshape(self.batch_dim, -1)
            ),  # [own] right boundaries
        ]

        return obs_self


# todo maybe put into own file
class RoadTrafficObservationProviderSimulation(ObservationProviderRT):
    def __init__(
        self,
        params: ObservationProviderParametersRT,
        constants: Constants,
        normalizers: Normalizers,
        thresholds: Thresholds,
        map: MapManager,
        world_state: WorldStateRTSimulation,
    ):
        super().__init__(params, constants, normalizers, thresholds, map, world_state)

        self.world_state = world_state

    def update_state(self, world: WorldCustom):

        super().update_state(self.world_state.get_agent_state_list())

        # Add new observation - actions & normalize
        if world.agents[0].action.u is None:
            self.observations.past_action_vel.add(self.constants.empty_action_vel)
            self.observations.past_action_steering.add(
                self.constants.empty_action_steering
            )
        else:
            self.observations.past_action_vel.add(
                torch.stack([a.action.u[:, 0] for a in world.agents], dim=1)
                / self.normalizers.v
            )
            self.observations.past_action_steering.add(
                torch.stack([a.action.u[:, 1] for a in world.agents], dim=1)
                / self.normalizers.steering
            )


# todo maybe put into own file
class RoadTrafficObservationProviderReal(ObservationProviderRT):
    def update_state(self, agent_states: List[AgentState]):
        # update world state first
        for a_i in range(len(agent_states)):
            self.world_state.update_distances(agent_states, a_i)

            self.world_state.update_vertices(agent_states)

            self.world_state.update_ref_paths_agent_related(a_i)

        # then execute observation specific updates
        super().update_state(agent_states)
