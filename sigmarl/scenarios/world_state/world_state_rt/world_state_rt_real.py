from typing import List

import torch

from sigmarl.constants import SCENARIOS
from sigmarl.helper_scenario import (
    get_distances_between_agents,
    get_perpendicular_distances,
)
from sigmarl.scenarios.observations.observation_provider import AgentState
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt import WorldStateRT


class WorldStateRTReal(WorldStateRT):
    def reset(self, agent_states: List[AgentState]):
        """
        in the real world case this method is only called for initialization purposes,
        therefore we always "reset" all agents simultaneously.
        """

        (ref_paths_scenario, extended_points) = self._reset_scenario_related_ref_paths()

        for i_agent in range(self.n_agents):
            ref_path, path_id = self._reset_init_state(
                agent_states, ref_paths_scenario, i_agent
            )

            self._reset_agent_related_ref_path(
                slice(None), i_agent, ref_path, path_id, extended_points
            )

            self.reset_init_distances_and_short_term_ref_path(
                agent_states, slice(None), i_agent
            )
        self.update_mutual_distances(agent_states)

    def _reset_scenario_related_ref_paths(self) -> tuple:
        # currently we only support the full map for real world applications

        ref_paths_scenario = self.ref_paths_map_related.long_term_all
        extended_points = self.ref_paths_map_related.point_extended_all

        self.ref_paths_agent_related.scenario_id[:, :] = 0

        return ref_paths_scenario, extended_points

    def _reset_init_state(
        self, agent_states: List[AgentState], ref_paths_scenario, agent_index: int
    ):

        agent_pos = agent_states[agent_index].pos

        # center lines have different number of points
        distances = [
            get_perpendicular_distances(agent_pos, ref_path["center_line"])
            for ref_path in ref_paths_scenario
        ]

        point_indices = [int(d[1]) for d in distances]

        distances = [d[0] for d in distances]

        # compute relative yaw between given yaws in [-pi, pi]
        def get_relative_yaw(yaw1: float, yaw2: float) -> float:
            relative_yaw = yaw2 - yaw1  # [0, 2*pi]
            # wrap to [-pi, pi] and use absolute value because 0 and 2*pi are the same
            relative_yaw = (relative_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            return abs(relative_yaw)

        agent_rot = agent_states[agent_index].rot

        rel_yaws = [
            get_relative_yaw(
                agent_rot,
                ref_paths_scenario[path_idx]["center_line_yaw"][point_idx - 1],
            )
            for path_idx, point_idx in enumerate(point_indices)
        ]

        costs = [
            (distance * 100) ** 2 + yaw for distance, yaw in zip(distances, rel_yaws)
        ]

        path_id = torch.argmin(torch.tensor(costs)).item()
        ref_path = ref_paths_scenario[path_id]

        self.ref_paths_agent_related.path_id[:, agent_index] = path_id
        self.ref_paths_agent_related.point_id[:, agent_index] = point_indices[path_id]

        return ref_path, path_id

    def update_mutual_distances(self, agent_states, env_index=slice(None)):
        mutual_distances = get_distances_between_agents(
            data=torch.stack([agent_states[i].pos for i in range(self.n_agents)])
            if self.distances.type == "c2c"
            else self.vertices,
            distance_type=self.distances.type,
            is_set_diagonal=True,
            x_semidim=torch.tensor(
                SCENARIOS["CPM_entire"]["world_x_dim"],
                device=self.device,
                dtype=torch.float32,
            ),  # currently we only support the full map for real world applications
            y_semidim=torch.tensor(
                SCENARIOS["CPM_entire"]["world_x_dim"],
                device=self.device,
                dtype=torch.float32,
            ),  # currently we only support the full map for real world applications
        )

        self.distances.agents[env_index, :, :] = mutual_distances[env_index, :, :]
