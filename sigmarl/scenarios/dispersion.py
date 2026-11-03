#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from sigmarl.scenarios._time_step_overlay import TimeStepOverlayScenario
from vmas.simulator.utils import Color, ScenarioUtils

from sigmarl.helper_scenario import Distances, Observations


class Scenario(TimeStepOverlayScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.share_reward = kwargs.pop("share_reward", False)
        self.penalise_by_time = kwargs.pop("penalise_by_time", False)
        self.food_radius = kwargs.pop("food_radius", 0.05)
        self.pos_range = kwargs.pop("pos_range", 1.0)
        n_food = kwargs.pop("n_food", n_agents)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(
            batch_dim,
            device,
            dt=getattr(getattr(self, "parameters", None), "dt", 0.05),
            x_semidim=self.pos_range,
            y_semidim=self.pos_range,
        )
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=0.035),
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(n_food):
            food = Landmark(
                name=f"food_{i}",
                collide=False,
                shape=Sphere(radius=self.food_radius),
                color=Color.GREEN,
            )
            world.add_landmark(food)

        # Store pairwise distances and nearest-agent indices for opponent
        # modeling and reward redistribution.
        self.distances = Distances(type="c2c")
        self.observations = Observations(
            n_nearing_agents=0,
            nearing_agents_indices=None,
        )

        return world

    def compute_agent_distances(self):
        """Compute pairwise distances and nearest neighbors for all agents."""
        positions = torch.stack(
            [agent.state.pos for agent in self.world.agents],
            dim=1,
        )
        distances = torch.cdist(positions, positions)
        distances.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        self.distances.agents = distances

        n_agents = len(self.world.agents)
        n_nearing_agents = min(
            self.parameters.n_nearing_agents_observed,
            n_agents - 1,
        )
        self.observations.nearing_agents_indices = torch.argsort(
            distances,
            dim=-1,
        )[..., :n_nearing_agents]
        self.observations.n_nearing_agents = n_nearing_agents

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    self.world.dim_p,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )
        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.pos_range,
                    self.pos_range,
                ),
                batch_index=env_index,
            )
            if env_index is None:
                landmark.eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.just_eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
            else:
                landmark.eaten[env_index] = False
                landmark.just_eaten[env_index] = False
                landmark.is_rendering[env_index] = True

        self.compute_agent_distances()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            self.compute_agent_distances()

        rews = torch.zeros(self.world.batch_dim, device=self.world.device)

        for landmark in self.world.landmarks:
            if is_first:
                landmark.how_many_on_food = torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - landmark.state.pos, dim=1
                        )
                        < a.shape.radius + landmark.shape.radius
                        for a in self.world.agents
                    ],
                    dim=1,
                ).sum(-1)
                landmark.anyone_on_food = landmark.how_many_on_food > 0
                landmark.just_eaten[landmark.anyone_on_food] = True

            assert (landmark.how_many_on_food <= len(self.world.agents)).all()

            if self.share_reward:
                rews[landmark.just_eaten * ~landmark.eaten] += 1
            else:
                on_food = (
                    torch.linalg.vector_norm(
                        agent.state.pos - landmark.state.pos, dim=1
                    )
                    < agent.shape.radius + landmark.shape.radius
                )
                eating_rew = landmark.how_many_on_food.reciprocal().nan_to_num(
                    posinf=0, neginf=0
                )
                rews[on_food * ~landmark.eaten] += eating_rew[on_food * ~landmark.eaten]

            if is_last:
                landmark.eaten += landmark.just_eaten
                landmark.just_eaten[:] = False
                landmark.is_rendering[landmark.eaten] = False

        if self.penalise_by_time:
            rews[rews == 0] = -0.01
        return rews

    def observation(self, agent: Agent):
        obs = []
        for landmark in self.world.landmarks:
            obs.append(
                torch.cat(
                    [
                        landmark.state.pos - agent.state.pos,
                        landmark.eaten.to(torch.int).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
        return torch.cat(
            [agent.state.pos, agent.state.vel, *obs],
            dim=-1,
        )

    def info(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        nearing_agents_indices = self.observations.nearing_agents_indices[
            :, agent_index
        ].to(torch.int32)

        obs_without_actions = self.observation(agent)
        n_action_features = self.world.dim_p
        obs_with_actions = torch.nn.functional.pad(
            obs_without_actions.clone(),
            (0, self.observations.n_nearing_agents * n_action_features),
        )

        return {
            "nearing_agents_indices": nearing_agents_indices,
            "observation_without_actions": obs_without_actions,
            "observation_with_actions": obs_with_actions,
        }

    def done(self):
        return torch.all(
            torch.stack(
                [landmark.eaten for landmark in self.world.landmarks],
                dim=1,
            ),
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
        share_reward=False,
        penalise_by_tim=False,
    )
