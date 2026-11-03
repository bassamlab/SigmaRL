#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Line, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from sigmarl.scenarios._time_step_overlay import TimeStepOverlayScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils
from sigmarl.helper_scenario import Distances, Observations


class Scenario(TimeStepOverlayScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.line_length = kwargs.pop("line_length", 2)
        line_mass = kwargs.pop("line_mass", 30)
        self.desired_velocity = kwargs.pop("desired_velocity", 0.05)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(
            batch_dim,
            device,
            dt=getattr(getattr(self, "parameters", None), "dt", 0.05),
        )
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(name=f"agent_{i}", u_multiplier=0.6, shape=Sphere(0.03))
            world.add_agent(agent)
        # Add landmarks
        self.line = Landmark(
            name="line",
            collide=True,
            rotatable=True,
            shape=Line(length=self.line_length),
            mass=line_mass,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)
        center = Landmark(
            name="center",
            shape=Sphere(radius=0.02),
            collide=False,
            color=Color.BLACK,
        )
        world.add_landmark(center)

        # containers for prioritized MARL features
        self.distances = Distances(type="c2c")
        self.observations = Observations(
            n_nearing_agents=0, nearing_agents_indices=None
        )

        return world

    def compute_agent_distances(self):
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        d = torch.norm(diff, dim=-1)
        d.diagonal(dim1=1, dim2=2).fill_(1e9)
        self.distances.agents = d

        n_envs, n_agents, _ = d.shape
        k = min(2, n_agents - 1)
        nearest_idx = torch.argsort(d, dim=-1)[..., :k]
        self.observations.nearing_agents_indices = nearest_idx
        self.observations.n_nearing_agents = k

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            # Random pos between -1 and 1
            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

        self.line.set_rot(
            torch.zeros(
                (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -torch.pi / 2,
                torch.pi / 2,
            ),
            batch_index=env_index,
        )

        # update initial pair-wise distances
        self.compute_agent_distances()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            # keep distance matrix up-to-date each step (once per env)
            self.compute_agent_distances()
            self.rew = (self.line.state.ang_vel.abs() - self.desired_velocity).abs()

        return -self.rew

    def observation(self, agent: Agent):
        line_end_1 = torch.cat(
            [
                (self.line_length / 2) * torch.cos(self.line.state.rot),
                (self.line_length / 2) * torch.sin(self.line.state.rot),
            ],
            dim=1,
        )
        line_end_2 = -line_end_1

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.line.state.pos - agent.state.pos,
                line_end_1 - agent.state.pos,
                line_end_2 - agent.state.pos,
                self.line.state.rot % torch.pi,
                self.line.state.ang_vel.abs(),
                (self.line.state.ang_vel.abs() - self.desired_velocity).abs(),
            ],
            dim=-1,
        )

    def info(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        obs_without_actions = self.observation(agent)
        n_pad = self.observations.n_nearing_agents * 2
        obs_with_actions = torch.nn.functional.pad(
            obs_without_actions.clone(), (0, n_pad)
        )
        if self.observations.nearing_agents_indices is None:
            nearing_agents_indices = torch.empty(
                self.world.batch_dim,
                0,
                device=self.world.device,
                dtype=torch.int64,
            )
        else:
            nearing_agents_indices = self.observations.nearing_agents_indices[
                :, agent_index
            ]
        return {
            "nearing_agents_indices": nearing_agents_indices,
            "observation_without_actions": obs_without_actions,
            "observation_with_actions": obs_with_actions,
        }


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions is True, "Heuristic for continuous actions only"

        index_line_extrema = 6

        pos_agent = observation[:, :2]
        pos_end2_agent = observation[:, index_line_extrema + 2 : index_line_extrema + 4]

        pos_end2 = pos_end2_agent + pos_agent

        pos_end2_shifted = TorchUtils.rotate_vector(
            pos_end2,
            torch.tensor(torch.pi / 4, device=observation.device).expand(
                pos_end2.shape[0]
            ),
        )

        pos_end2_shifted_agent = pos_end2_shifted - pos_agent

        action_agent = torch.clamp(
            pos_end2_shifted_agent,
            min=-u_range,
            max=u_range,
        )

        return action_agent


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        desired_velocity=0.05,
        n_agents=4,
        line_length=2,
        line_mass=30,
    )
