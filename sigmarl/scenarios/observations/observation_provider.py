from abc import abstractmethod
from typing import List

import torch

from dataclasses import dataclass

from sigmarl.helper_scenario import Constants, Normalizers


@dataclass
class ObservationProviderParameters:
    batch_dim: int
    n_agents: int
    device: torch.device
    # observation specific
    n_stored_steps: int
    obs_noise_level: float
    is_obs_noise: bool


@dataclass
class AgentState:
    pos: torch.tensor
    rot: torch.tensor
    steering: torch.tensor
    vel: torch.tensor


class ObservationProvider:
    def __init__(
        self,
        params: ObservationProviderParameters,
        constants: Constants,
        normalizers: Normalizers,
    ):
        self.batch_dim = params.batch_dim
        self.n_agents = params.n_agents
        self.device = params.device

        self.params = params

        self.constants = constants
        self.normalizers = normalizers

    @abstractmethod
    def init_state(self, agent_states: List[AgentState]):
        pass

    @abstractmethod
    def update_state(self, agent_states: List[AgentState]):
        pass

    @abstractmethod
    def get_observation(self, agent_index: int) -> torch.Tensor:
        pass
