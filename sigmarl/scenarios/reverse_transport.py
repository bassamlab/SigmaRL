#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from sigmarl.scenarios._time_step_overlay import TimeStepOverlayScenario
from vmas.simulator.utils import Color, ScenarioUtils
from sigmarl.helper_scenario import Distances, Observations


class Scenario(TimeStepOverlayScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 4)
        self.package_width = kwargs.pop("package_width", 0.6)
        self.package_length = kwargs.pop("package_length", 0.6)
        self.package_mass = kwargs.pop("package_mass", 50)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.shaping_factor = 100

        # Make world
        world = World(
            batch_dim,
            device,
            dt=getattr(getattr(self, "parameters", None), "dt", 0.05),
            contact_margin=6e-3,
            substeps=5,
            collision_force=500,
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(name=f"agent_{i}", shape=Sphere(0.03), u_multiplier=0.5)
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.09),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)

        self.package = Landmark(
            name=f"package {i}",
            collide=True,
            movable=True,
            mass=self.package_mass,
            shape=Box(
                length=self.package_length,
                width=self.package_width,
                hollow=True,
            ),
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)

        self.stored_base_observations = [None] * self.n_agents
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
        package_pos = torch.zeros(
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
        )

        self.package.set_pos(
            package_pos,
            batch_index=env_index,
        )
        for agent in self.world.agents:
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (
                                (1, 1)
                                if env_index is not None
                                else (self.world.batch_dim, 1)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.package_length / 2 + agent.shape.radius,
                            self.package_length / 2 - agent.shape.radius,
                        ),
                        torch.zeros(
                            (
                                (1, 1)
                                if env_index is not None
                                else (self.world.batch_dim, 1)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.package_width / 2 + agent.shape.radius,
                            self.package_width / 2 - agent.shape.radius,
                        ),
                    ],
                    dim=1,
                )
                + package_pos,
                batch_index=env_index,
            )

        self.package.goal.set_pos(
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

        if env_index is None:
            self.package.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
            self.package.on_goal = torch.zeros(
                self.world.batch_dim,
                dtype=torch.bool,
                device=self.world.device,
            )
        else:
            self.package.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )
            self.package.on_goal[env_index] = False

        # update initial pair-wise distances
        self.compute_agent_distances()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )

            self.compute_agent_distances()

            self.package.dist_to_goal = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )
            self.package.on_goal = self.world.is_overlapping(
                self.package, self.package.goal
            )
            self.package.color = torch.tensor(
                Color.RED.value, device=self.world.device, dtype=torch.float32
            ).repeat(self.world.batch_dim, 1)
            self.package.color[self.package.on_goal] = torch.tensor(
                Color.GREEN.value,
                device=self.world.device,
                dtype=torch.float32,
            )

            package_shaping = self.package.dist_to_goal * self.shaping_factor
            self.rew[~self.package.on_goal] += (
                self.package.global_shaping[~self.package.on_goal]
                - package_shaping[~self.package.on_goal]
            )
            self.package.global_shaping = package_shaping

            self.rew[~self.package.on_goal] += (
                self.package.global_shaping[~self.package.on_goal]
                - package_shaping[~self.package.on_goal]
            )
            self.package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        obs = torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.package.state.vel,
                self.package.state.pos - agent.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
            ],
            dim=-1,
        )
        self.stored_base_observations[agent_index] = obs  # <-- ADD THIS LINE
        return obs

    def info(self, agent: Agent):
        """
        This function computes and returns a dictionary of extra information for the agent.
        """
        agent_index = self.world.agents.index(agent)

        # Nearing agents are the ones except the agent itself (full observability)
        nearing_agents_indices = torch.tensor(
            [i for i in range(self.n_agents) if i != agent_index],
            device=self.world.device,
            dtype=torch.int32,
        ).repeat(self.world.batch_dim, 1)

        # Use the observation stored from the main observation() call
        obs_without_actions = self.stored_base_observations[agent_index]

        # Create a padded version for use with opponent modeling or action propagation
        n_pad = self.observations.n_nearing_agents * 2  # 2 actions per agent
        obs_with_actions = torch.nn.functional.pad(
            obs_without_actions.clone(),
            (0, n_pad),
        )

        # --- Create the info dictionary ---
        info_dict = {
            "nearing_agents_indices": nearing_agents_indices,
            "observation_without_actions": obs_without_actions,
            "observation_with_actions": obs_with_actions,
        }

        return info_dict

    def done(self):
        return self.package.on_goal


if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(
        scenario=scenario,
        control_two_agents=False,
        shared_reward=False,
    )
