from abc import abstractmethod
from typing import List

import torch

from sigmarl.constants import AGENTS
from sigmarl.helper_scenario import Observations, CircularBuffer, Constants, Normalizers
from sigmarl.helper_training import WorldCustom
from sigmarl.scenarios.observations.observation_provider import (
    ObservationProvider,
    ObservationProviderParameters,
    AgentState,
)
from sigmarl.scenarios.world_state.world_state_gr.world_state_gr import WorldStateGR
from sigmarl.scenarios.world_state.world_state_gr.world_state_gr_sim import (
    WorldStateGRSimulation,
)


class ObservationProviderGR(ObservationProvider):
    def __init__(
        self,
        params: ObservationProviderParameters,
        constants: Constants,
        normalizers: Normalizers,
        short_term_ref_path,
        world_state: WorldStateGR,
    ):

        super().__init__(params, constants, normalizers)

        self.short_term_ref_path = short_term_ref_path
        self.world_state = world_state

        self.observations = Observations(
            obs_noise_level=torch.tensor(
                self.params.obs_noise_level, device=self.device, dtype=torch.float32
            ),
        )

        self.observations.past_action_steering = CircularBuffer(
            torch.zeros(
                (self.params.n_stored_steps, self.batch_dim, self.n_agents),
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

        self.agent_pos = None
        self.agent_rot = None
        self.agent_speed = None
        self.agent_steering = None

    def update_state(self, agent_states: List[AgentState]):
        self.agent_pos = [s.pos for s in agent_states]
        self.agent_rot = [s.rot for s in agent_states]
        self.agent_steering = [s.steering for s in agent_states]
        self.agent_speed = [
            torch.norm(s.vel, p=2, dim=1).unsqueeze(1) for s in agent_states
        ]

    def get_observation(self, agent_index: int) -> torch.Tensor:

        # Compute the vectors from the agent's position to the short-term reference paths
        vectors_to_ref = self.short_term_ref_path - self.agent_pos[
            agent_index
        ].unsqueeze(1)

        # Calculate the angles of these vectors
        angles_to_ref = (
            torch.atan2(vectors_to_ref[..., 1], vectors_to_ref[..., 0])
        ).unsqueeze(-1)

        # Compute the angle difference between the vectors and the agent's rotation
        angle_difference = angles_to_ref - self.agent_rot[agent_index].unsqueeze(1)

        # Calculate the length of the vectors
        vector_lengths = torch.norm(vectors_to_ref, dim=-1).unsqueeze(-1)

        # Project the vectors into the agent's ego view using the angle difference
        short_term_ref_ego_view_x = vector_lengths * torch.cos(angle_difference)
        short_term_ref_ego_view_y = vector_lengths * torch.sin(angle_difference)

        short_term_ref_ego_view = torch.cat(
            (short_term_ref_ego_view_x, short_term_ref_ego_view_y), dim=-1
        )

        obs = torch.hstack(
            [
                self.agent_speed[agent_index] / self.normalizers.v,
                self.agent_steering[agent_index] / self.normalizers.steering,
                (short_term_ref_ego_view / self.normalizers.pos).reshape(
                    self.batch_dim, -1
                ),
                self.world_state.distances.ref_paths / self.normalizers.distance_ref,
            ]
        )

        if self.params.is_add_noise:
            # Add sensor noise if required
            obs = obs + (
                self.observations.obs_noise_level
                * torch.rand_like(obs, device=self.device, dtype=torch.float32)
            )

        return obs


class ObservationProviderGRSimulation(ObservationProviderGR):
    def __init__(
        self,
        params: ObservationProviderParameters,
        constants: Constants,
        normalizers: Normalizers,
        short_term_ref_path,
        world_state: WorldStateGRSimulation,
    ):
        super().__init__(
            params, constants, normalizers, short_term_ref_path, world_state
        )

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
