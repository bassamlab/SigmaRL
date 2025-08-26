from dataclasses import dataclass
from typing import List

import torch
from termcolor import cprint

from sigmarl.helper_scenario import (
    get_distances_between_agents,
    InitialStateBuffer,
    interX,
    Collisions,
)
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
    reset_agent_min_distance: torch.Tensor


class WorldStateRTSimulation(WorldStateRT, WorldStateSim):
    def __init__(self, params: WorldStateRTSimParameters, map: MapManager):
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
    ):

        reset_indices = (
            range(self.n_agents) if agent_index is None else agent_index.unsqueeze(0)
        )

        is_reset_single_agent = agent_index is not None

        (ref_paths_scenario, extended_points) = self._reset_scenario_related_ref_paths(
            env_index=env_index, agent_index=agent_index
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
        )
        rot_start = ref_path["center_line_yaw"][random_point_id]
        steering_start = torch.zeros_like(rot_start, device=self.world.device)
        sideslip_start = torch.zeros_like(
            steering_start, device=self.world.device
        )  # Sideslip angle is zero since the steering is zero

        speed_start = (
            torch.rand(1, dtype=torch.float32, device=self.world.device)
            * agents[agent_index].max_speed
        )  # Random initial velocity
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
    ):
        is_feasible_initial_position_found = False
        random_count = 0

        ref_path, path_id, random_point_id = None, None, None

        # Randomly generate initial states for each agent
        while not is_feasible_initial_position_found:

            if random_count >= 20:
                cprint(
                    f"Reset agent(s): random_count = {random_count}.",
                    "grey",
                )
            random_count += 1

            if predefined_ref_path_idx_i is not None:
                path_id = predefined_ref_path_idx_i
            else:
                path_id = torch.randint(
                    0, len(ref_paths_scenario), (1,)
                ).item()  # Select randomly a path

            self.ref_paths_agent_related.path_id[env_index, agent_index] = path_id

            ref_path = ref_paths_scenario[path_id]

            num_points = ref_path["center_line"].shape[0]

            if self.params.scenario_type == "CPM_mixed":
                # In the mixed scenarios of the CPM case, we avoid using the beginning part of a path, making agents encounter each other more frequently. Additionally, We avoid initializing agents to be at a very end of a path.
                start_point_idx = 6
                end_point_idx = int(num_points / 2)
            else:
                start_point_idx = 3  # Do not set to an overly small value to make sure agents are fully inside its lane
                end_point_idx = num_points - 3

            random_point_id = torch.randint(
                start_point_idx, end_point_idx, (1,)
            ).item()  # choose random point from previously defined range

            self.ref_paths_agent_related.point_id[
                env_index, agent_index
            ] = random_point_id  # Update

            position_start = ref_path["center_line"][random_point_id]
            agents[agent_index].set_pos(position_start, batch_index=env_index)

            # Check if the initial position is feasible
            if not is_reset_single_agent:
                if agent_index == 0:
                    # The initial position of the first agent is always feasible
                    is_feasible_initial_position_found = True
                    continue
                else:
                    positions = torch.stack(
                        [
                            self.world.agents[i].state.pos[env_index]
                            for i in range(agent_index + 1)
                        ]
                    )
            else:
                # Check if the initial position of the agent to be reset is collision-free with other agents
                positions = torch.stack(
                    [
                        self.world.agents[i].state.pos[env_index]
                        for i in range(self.n_agents)
                    ]
                )

            diff_sq = (
                positions[agent_index, :] - positions
            ) ** 2  # Calculate pairwise squared differences in positions
            initial_mutual_distances_sq = torch.sum(diff_sq, dim=-1)

            initial_mutual_distances_sq[agent_index] = (
                torch.max(initial_mutual_distances_sq) + 1
            )  # Set self-to-self distance to a sufficiently high value

            min_distance_sq = torch.min(initial_mutual_distances_sq)

            is_feasible_initial_position_found = min_distance_sq >= (
                self.params.reset_agent_min_distance**2
            )

        return ref_path, path_id, random_point_id

    def _reset_scenario_related_ref_paths(
        self, env_index: int = None, agent_index: int = None
    ) -> tuple:

        if self.params.scenario_type != "CPM_mixed":
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
        for a_i in range(self.n_agents):
            # Update the collision matrices
            if self.distances.type == "c2c":
                for a_j in range(a_i + 1, self.n_agents):
                    # Check for collisions between agents using the interX function
                    collision_batch_index = interX(
                        self.vertices[:, a_i], self.vertices[:, a_j], False
                    )
                    self.collisions.with_agents[
                        torch.nonzero(collision_batch_index), a_i, a_j
                    ] = True
                    self.collisions.with_agents[
                        torch.nonzero(collision_batch_index), a_j, a_i
                    ] = True
            elif self.distances.type == "mtv":
                # Two agents collide if their mtv-based distance is zero
                self.collisions.with_agents[:] = self.distances.agents == 0

            # Check for collisions between agents and lanelet boundaries
            collision_with_left_boundary = interX(
                L1=self.vertices[:, a_i],
                L2=self.ref_paths_agent_related.left_boundary[:, a_i],
                is_return_points=False,
            )  # [batch_dim]
            collision_with_right_boundary = interX(
                L1=self.vertices[:, a_i],
                L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                is_return_points=False,
            )  # [batch_dim]
            self.collisions.with_lanelets[
                (collision_with_left_boundary | collision_with_right_boundary), a_i
            ] = True

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

        self.update_distances(agent_states, agent_index)

        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        if agent_index == 0:  # Avoid repeated computations

            self.reset_collisions()

            self.update_vertices(agent_states)

            self.update_collisions()

    def update_state_after_rewarding(self, agent_index: int):
        """
        Update some states (such as previous positions and short-term reference paths) after rewarding agents.
        """
        self.update_ref_paths_agent_related(agent_index)
