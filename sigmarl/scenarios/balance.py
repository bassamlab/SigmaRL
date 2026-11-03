#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from sigmarl.scenarios._time_step_overlay import TimeStepOverlayScenario
from vmas.simulator.utils import Color, ScenarioUtils, Y
from sigmarl.helper_scenario import Distances, Observations


class Scenario(TimeStepOverlayScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 3)
        self.package_mass = kwargs.pop("package_mass", 5)
        self.random_package_pos_on_line = kwargs.pop(
            "random_package_pos_on_line", False
        )
        self.shared_rew = kwargs.pop("shared_rew", True)
        self.global_reward_weight = kwargs.pop("global_reward_weight", 0.5)
        self.local_reward_weight = kwargs.pop("local_reward_weight", 0.5)
        self.support_shaping_factor = kwargs.pop("support_shaping_factor", 100)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert self.n_agents > 1
        assert self.global_reward_weight >= 0
        assert self.local_reward_weight >= 0
        assert self.support_shaping_factor >= 0

        self.line_length = 0.8
        self.agent_radius = 0.03

        self.shaping_factor = 100
        self.fall_reward = -10

        self.visualize_semidims = False

        # Make world
        world = World(
            batch_dim,
            device,
            dt=getattr(getattr(self, "parameters", None), "dt", 0.05),
            gravity=(0.0, -0.05),
            y_semidim=1,
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
            )
            agent.support_offset = torch.tensor(
                [
                    -(self.line_length - self.agent_radius) / 2
                    + i * (self.line_length - self.agent_radius) / (self.n_agents - 1),
                    -self.agent_radius * 2,
                ],
                device=device,
                dtype=torch.float32,
            )
            agent.support_rew = torch.zeros(
                batch_dim, device=device, dtype=torch.float32
            )
            world.add_agent(agent)

        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.package = Landmark(
            name="package",
            collide=True,
            movable=True,
            shape=Sphere(),
            mass=self.package_mass,
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)
        # Add landmarks

        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True,
            movable=True,
            rotatable=True,
            mass=5,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)

        self.floor = Landmark(
            name="floor",
            collide=True,
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(self.floor)

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()

        # create empty containers so helper_training and opponent-modeling can find them
        self.distances = Distances(type="c2c")
        # we will fill nearing_agents_indices in compute_agent_distances
        self.observations = Observations(
            n_nearing_agents=0, nearing_agents_indices=None
        )

        return world

    def compute_agent_distances(self):
        """Compute centre-to-centre distances between every pair of agents."""
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)  # [B,N,2]
        # ||p_i - p_j||
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        d = torch.norm(diff, dim=-1)  # [B,N,N]
        # put something large on the diagonal if you do not want self-distance
        d.diagonal(dim1=1, dim2=2).fill_(1e9)
        self.distances.agents = d

        # also compute the indices of the two closest neighbours for opponent modeling
        n_envs, n_agents, _ = d.shape
        k = min(2, n_agents - 1)  # default: 2 nearest agents
        nearest_idx = torch.argsort(d, dim=-1)[..., :k]  # [B,N,k]
        self.observations.nearing_agents_indices = nearest_idx
        # store how many agents are being tracked
        # use a plain integer so helper modules expecting an int work correctly
        self.observations.n_nearing_agents = k

    def reset_world_at(self, env_index: int = None):
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.agent_radius * 2,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    (
                        -self.line_length / 2 + self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                    (
                        self.line_length / 2 - self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )

        for agent in self.world.agents:
            agent.set_pos(
                line_pos + agent.support_offset,
                batch_index=env_index,
            )

        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )

        self.floor.set_pos(
            torch.tensor(
                [
                    0,
                    -self.world.y_semidim
                    - self.floor.shape.width / 2
                    - self.agent_radius,
                ],
                device=self.world.device,
            ),
            batch_index=env_index,
        )

        # update initial pair-wise distances
        self.compute_agent_distances()
        self.compute_on_the_ground()
        for agent in self.world.agents:
            support_shaping = self.compute_support_error(agent) * (
                self.support_shaping_factor
            )
            if env_index is None:
                agent.support_shaping = support_shaping
            else:
                agent.support_shaping[env_index] = support_shaping[env_index]
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

    def compute_on_the_ground(self):
        self.on_the_ground = self.world.is_overlapping(
            self.line, self.floor
        ) + self.world.is_overlapping(self.package, self.floor)

    def compute_support_error(self, agent: Agent):
        """Return the distance from an agent to its assigned point beneath the line."""
        line_rotation = self.line.state.rot
        cos_rotation = torch.cos(line_rotation)
        sin_rotation = torch.sin(line_rotation)
        offset_x, offset_y = agent.support_offset
        rotated_offset = torch.cat(
            [
                cos_rotation * offset_x - sin_rotation * offset_y,
                sin_rotation * offset_x + cos_rotation * offset_y,
            ],
            dim=-1,
        )
        support_position = self.line.state.pos + rotated_offset
        return torch.linalg.vector_norm(agent.state.pos - support_position, dim=-1)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.ground_rew[:] = 0

            self.compute_on_the_ground()
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )

            self.ground_rew[self.on_the_ground] = self.fall_reward

            global_shaping = self.package_dist * self.shaping_factor
            self.pos_rew = self.global_shaping - global_shaping
            self.global_shaping = global_shaping

            for world_agent in self.world.agents:
                support_shaping = self.compute_support_error(world_agent) * (
                    self.support_shaping_factor
                )
                world_agent.support_rew = world_agent.support_shaping - support_shaping
                world_agent.support_shaping = support_shaping

            # keep distance matrix up-to-date each step (once per env)
            self.compute_agent_distances()

        if self.shared_rew:
            return self.ground_rew + self.pos_rew
        return (
            self.ground_rew
            + self.global_reward_weight * self.pos_rew
            + self.local_reward_weight * agent.support_rew
        )

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.package.state.pos,
                agent.state.pos - self.line.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
                self.package.state.vel,
                self.line.state.vel,
                self.line.state.ang_vel,
                self.line.state.rot % torch.pi,
            ],
            dim=-1,
        )

    def done(self):
        return self.on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )

    def info(self, agent: Agent):
        agent_index = self.world.agents.index(agent)

        nearing_agents_indices = torch.tensor(
            [i for i in range(self.n_agents) if i != agent_index],
            device=self.world.device,
            dtype=torch.int32,
        ).repeat(self.world.batch_dim, 1)

        obs_without_actions = self.observation(agent)

        n_pad = self.observations.n_nearing_agents * 2
        obs_with_actions = torch.nn.functional.pad(
            obs_without_actions.clone(),
            (0, n_pad),
        )

        # Create a new local dictionary on every call
        info_dict = {
            "pos_rew": self.pos_rew,
            "ground_rew": self.ground_rew,
            "support_rew": agent.support_rew,
            "nearing_agents_indices": nearing_agents_indices,
            "observation_without_actions": obs_without_actions,
            "observation_with_actions": obs_with_actions,
        }

        return info_dict


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        batch_dim = observation.shape[0]

        index_package_goal_pos = 8
        dist_package_goal = observation[
            :, index_package_goal_pos : index_package_goal_pos + 2
        ]
        y_distance_ge_0 = dist_package_goal[:, Y] >= 0

        if self.continuous_actions:
            action_agent = torch.clamp(
                torch.stack(
                    [
                        torch.zeros(batch_dim, device=observation.device),
                        -dist_package_goal[:, Y],
                    ],
                    dim=1,
                ),
                min=-u_range,
                max=u_range,
            )
            action_agent[:, Y][y_distance_ge_0] = 0
        else:
            action_agent = torch.full((batch_dim,), 4, device=observation.device)
            action_agent[y_distance_ge_0] = 0
        return action_agent


if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(
        scenario=scenario,
        control_two_agents=False,
        shared_rew=False,
        n_agents=3,
        package_mass=5,
        random_package_pos_on_line=False,
    )
