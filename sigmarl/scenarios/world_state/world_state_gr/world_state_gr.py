from dataclasses import dataclass
from typing import List

import torch

from sigmarl.constants import AGENTS
from sigmarl.helper_scenario import (
    Distances,
    get_perpendicular_distances,
    get_rectangle_vertices,
)
from sigmarl.scenarios.observations.observation_provider import AgentState
from sigmarl.scenarios.world_state.world_state import WorldState, WorldStateParameters


@dataclass
class WorldStateGRParameters(WorldStateParameters):
    goal: torch.Tensor
    original_pos: torch.Tensor


class WorldStateGR(WorldState):
    def __init__(self, params: WorldStateGRParameters):
        super().__init__(params)

        self.params = params

    def _init_stateful_parameters(self):

        self.distances = Distances(
            type="c2c",  # Type of distances between agents
            agents=torch.zeros(
                self.batch_dim, self.n_agents, self.n_agents, dtype=torch.float32
            ),
            ref_paths=torch.zeros(
                (self.batch_dim, self.n_agents), device=self.device, dtype=torch.float32
            ),
        )

        # The shape of each agent is considered a rectangle with 4 vertices.
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros(
            (self.batch_dim, self.n_agents, 5, 2),
            device=self.device,
            dtype=torch.float32,
        )

    def update_distances(
        self, agent_states: List[AgentState], agent_index: int, env_index=slice(None)
    ):
        # Distance from the center of gravity (CG) of the agent to its reference path
        (self.distances.ref_paths[env_index, 0], _,) = get_perpendicular_distances(
            point=agent_states[agent_index].pos[env_index],
            polyline=torch.stack(
                [self.params.original_pos[env_index], self.params.goal[env_index]],
                dim=1,
            ),
        )

    def update_vertices(self, agent_states):
        # in the goal reaching env there is only a single agent
        self.vertices[:, 0] = get_rectangle_vertices(
            center=agent_states[0].pos,
            yaw=agent_states[0].rot,
            width=AGENTS["width"],
            length=AGENTS["length"],
            is_close_shape=True,
        )

    def reset(self, agent_states: List[AgentState]):
        pass  # todo
