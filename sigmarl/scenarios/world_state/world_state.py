from abc import abstractmethod, ABC
from typing import List

from vmas.simulator.core import AgentState

import torch

from dataclasses import dataclass

@dataclass
class WorldStateParameters:
    batch_dim: int
    n_agents: int
    device: torch.device

"""
This class contains the stateful attributes of an environment (real or simulated).
"""

class WorldState(ABC):
    def __init__(self, params: WorldStateParameters):

        self.n_agents = params.n_agents
        self.batch_dim = params.batch_dim
        self.device = params.device

        self.params = params

        self._init_stateful_parameters()

    """
    Depending on the environment there are different stateful attributes.
    However, all environments keep track of the stateful attributes distances and vertices.
    """
    @abstractmethod
    def _init_stateful_parameters(self):
        pass

    """Methods in this class are concerned with updating the stateful values of the environment"""
    @abstractmethod
    def reset(self, agent_states: List[AgentState]):
        pass

    @abstractmethod
    def update_distances(self, agent_states: List[AgentState], agent_index: int):
        pass

    @abstractmethod
    def update_vertices(self, agent_states: List[AgentState]):
        pass