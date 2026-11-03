# This file is an adapted version of a file from https://github.com/proroklab/VectorizedMultiAgentSimulator
# Copyright (c) 2022-2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.
#
# Adapted by Jianye Xu, 2025
#
# This file is licensed under the GNU General Public License v3.0
# You should have received a copy of the GNU General Public License
# along with this script. If not, see <https://www.gnu.org/licenses/>.

import math

import torch

from typing import Optional

from vmas import render_interactively
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from sigmarl.scenarios._time_step_overlay import TimeStepOverlayScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils
from sigmarl.helper_scenario import Distances, Observations

from sigmarl.helper_training import Parameters


class Scenario(TimeStepOverlayScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = 2
        self.rew = torch.zeros(
            batch_dim, self.n_agents, device=device, dtype=torch.float32
        )
        self.rew_with_propagation = torch.zeros(
            batch_dim, self.n_agents, device=device, dtype=torch.float32
        )
        self.pos_rew = torch.zeros(
            batch_dim, self.n_agents, device=device, dtype=torch.float32
        )
        self.deviation_pelanity = torch.zeros(
            batch_dim, self.n_agents, device=device, dtype=torch.float32
        )
        self.energy_rew = torch.zeros(
            batch_dim, self.n_agents, device=device, dtype=torch.float32
        )
        self.final_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        self.step = 0
        self.v_range = kwargs.pop("v_range", 0.5)
        self.a_range = kwargs.pop("a_range", 1)
        self.obs_noise = kwargs.pop("obs_noise", 0)
        self.box_agents = kwargs.pop("box_agents", False)
        self.linear_friction = kwargs.pop("linear_friction", 0.1)
        self.mirror_passage = kwargs.pop("mirror_passage", False)
        self.done_on_completion = kwargs.pop("done_on_completion", False)
        self.observe_rel_pos = kwargs.pop("observe_rel_pos", True)
        self.observe_vel_sur = kwargs.pop(
            "observe_vel_sur", False
        )  # observe velocity of surrounding agents
        self.observe_rel_pos_to_goal_sur = kwargs.pop(
            "observe_rel_pos_to_goal_sur", False
        )
        self.is_reward_propagation = kwargs.pop("is_reward_propagation", False)
        self.is_share_reward = kwargs.pop("is_share_reward", True)

        # Reward params
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1.0)
        self.final_reward = kwargs.pop("final_reward", 0.01)
        self.energy_reward_coeff = kwargs.pop("energy_rew_coeff", 0.005)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", 0)
        self.passage_collision_penalty = kwargs.pop("passage_collision_penalty", 0)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", 0)
        self.centerline_deviation_penalty = kwargs.pop(
            "centerline_deviation_penalty", 0.005
        )
        print(f"centerline_deviation_penalty: {self.centerline_deviation_penalty}")

        # Params for real world deployment on robomasters
        self.use_velocity_controller = kwargs.pop("use_velocity_controller", True)
        self.min_input_norm = kwargs.pop("min_input_norm", 0.08)
        self.dt_delay = kwargs.pop("dt_delay", 0)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.viewer_size = (1600, 700)

        controller_params = [2, 6, 0.002]

        self.f_range = self.a_range + self.linear_friction
        self.u_range = self.v_range if self.use_velocity_controller else self.f_range

        # Make world
        world = World(
            batch_dim,
            device,
            drag=0,
            dt=getattr(getattr(self, "parameters", None), "dt", 0.05),
            linear_friction=self.linear_friction,
            substeps=16 if self.box_agents else 5,
            collision_force=10000 if self.box_agents else 500,
        )

        self.agent_radius = 0.16
        self.agent_box_length = 0.32
        self.agent_box_width = 0.24

        self.spawn_pos_noise = 0.02
        self.min_collision_distance = 0.005

        # Add agents
        blue_agent = Agent(
            name="agent_0",
            rotatable=False,
            linear_friction=self.linear_friction,
            shape=(
                Sphere(radius=self.agent_radius)
                if not self.box_agents
                else Box(length=self.agent_box_length, width=self.agent_box_width)
            ),
            u_range=self.u_range,
            f_range=self.f_range,
            v_range=self.v_range,
            render_action=True,
        )
        if self.use_velocity_controller:
            blue_agent.controller = VelocityController(
                blue_agent, world, controller_params, "standard"
            )
        blue_goal = Landmark(
            name="goal_0",
            collide=False,
            shape=Sphere(radius=self.agent_radius / 2),
            color=Color.BLUE,
        )
        blue_agent.goal = blue_goal
        world.add_agent(blue_agent)
        world.add_landmark(blue_goal)

        green_agent = Agent(
            name="agent_1",
            color=Color.GREEN,
            linear_friction=self.linear_friction,
            shape=(
                Sphere(radius=self.agent_radius)
                if not self.box_agents
                else Box(length=self.agent_box_length, width=self.agent_box_width)
            ),
            rotatable=False,
            u_range=self.u_range,
            f_range=self.f_range,
            v_range=self.v_range,
            render_action=True,
        )
        if self.use_velocity_controller:
            green_agent.controller = VelocityController(
                green_agent, world, controller_params, "standard"
            )
        green_goal = Landmark(
            name="goal_1",
            collide=False,
            shape=Sphere(radius=self.agent_radius / 2),
            color=Color.GREEN,
        )
        green_agent.goal = green_goal
        world.add_agent(green_agent)
        world.add_landmark(green_goal)

        null_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device)
        blue_agent.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        green_agent.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        # control delayed by n dts

        self.spawn_map(world)

        for agent in world.agents:
            agent.agent_collision_rew = torch.zeros(batch_dim, device=device)
            agent.obstacle_collision_rew = agent.agent_collision_rew.clone()

        self.stored_base_observations = [None] * 2  # Two agents

        # containers for prioritized MARL features
        self.distances = Distances(type="c2c")
        self.observations = Observations(
            n_nearing_agents=self.parameters.n_nearing_agents_observed,
            nearing_agents_indices=None,
        )

        return world

    def compute_agent_distances(self):
        """Compute centre-to-centre distances between every pair of agents."""
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        d = torch.norm(diff, dim=-1)
        d.diagonal(dim1=1, dim2=2).fill_(1e9)
        self.distances.agents = d

        n_envs, n_agents, _ = d.shape
        # Dynamically determine the number of neighbors 'k' to consider.
        # It should be the smaller value between the configured number of observed agents
        # and the actual number of other agents available in the scenario (n_agents - 1).
        k = min(self.parameters.n_nearing_agents_observed, n_agents - 1)

        # print(f"Number of agents: {n_agents}")
        # print(f"Indices of neighboring agents: {self.observations.nearing_agents_indices}")

        nearest_idx = torch.argsort(d, dim=-1)[..., :k]
        self.observations.nearing_agents_indices = nearest_idx
        self.observations.n_nearing_agents = k

    def reset_world_at(self, env_index: int = None):
        self.step = 0
        self.rew = torch.zeros(
            self.world.batch_dim,
            self.n_agents,
            device=self.world.device,
            dtype=torch.float32,
        )
        self.rew_with_propagation = torch.zeros(
            self.world.batch_dim,
            self.n_agents,
            device=self.world.device,
            dtype=torch.float32,
        )
        self.pos_rew = torch.zeros(
            self.world.batch_dim,
            self.n_agents,
            device=self.world.device,
            dtype=torch.float32,
        )
        self.energy_rew = torch.zeros(
            self.world.batch_dim,
            self.n_agents,
            device=self.world.device,
            dtype=torch.float32,
        )
        self.deviation_pelanity = torch.zeros(
            self.world.batch_dim,
            self.n_agents,
            device=self.world.device,
            dtype=torch.float32,
        )
        self.final_rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        self.world.agents[0].set_pos(
            torch.tensor(
                [-(self.scenario_length / 2 - self.agent_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            )
            + torch.zeros(self.world.dim_p, device=self.world.device,).uniform_(
                -self.spawn_pos_noise,
                self.spawn_pos_noise,
            ),
            batch_index=env_index,
        )
        if self.use_velocity_controller:
            self.world.agents[0].controller.reset(env_index)
        self.world.landmarks[0].set_pos(
            torch.tensor(
                [(self.scenario_length / 2 - self.goal_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.world.agents[1].set_pos(
            torch.tensor(
                [self.scenario_length / 2 - self.agent_dist_from_wall, 0.0],
                dtype=torch.float32,
                device=self.world.device,
            )
            + torch.zeros(self.world.dim_p, device=self.world.device,).uniform_(
                -self.spawn_pos_noise,
                self.spawn_pos_noise,
            ),
            batch_index=env_index,
        )
        if self.use_velocity_controller:
            self.world.agents[1].controller.reset(env_index)
        self.world.landmarks[1].set_pos(
            torch.tensor(
                [-(self.scenario_length / 2 - self.goal_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

        self.reset_map(env_index)
        for agent in self.world.agents:
            if env_index is None:
                agent.shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

        if env_index is None:
            self.goal_reached = torch.full(
                (self.world.batch_dim,), False, device=self.world.device
            )
        else:
            self.goal_reached[env_index] = False

        # update initial pair-wise distances
        self.compute_agent_distances()

    def process_action(self, agent: Agent):
        if self.use_velocity_controller:
            # Use queue for delay
            agent.input_queue.append(agent.action.u.clone())
            agent.action.u = agent.input_queue.pop(0)

            # Clamp square to circle
            agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, self.u_range)

            # Zero small input
            action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
            agent.action.u[action_norm < self.min_input_norm] = 0

            # Copy action
            agent.vel_action = agent.action.u.clone()

            # Reset controller
            vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 1e-3
            agent.controller.reset(vel_is_zero)

            agent.controller.process_force()

    def reward(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        is_first = agent == self.world.agents[0]

        blue_agent = self.world.agents[0]
        green_agent = self.world.agents[-1]

        if is_first:
            # Initialize
            self.rew[:] = 0
            self.rew_with_propagation[:] = 0
            self.pos_rew[:] = 0
            self.energy_rew[:] = 0
            self.final_rew[:] = 0

            # keep distance matrix up-to-date each step
            self.compute_agent_distances()

            self.blue_distance = torch.linalg.vector_norm(
                blue_agent.state.pos - blue_agent.goal.state.pos,
                dim=1,
            )
            self.green_distance = torch.linalg.vector_norm(
                green_agent.state.pos - green_agent.goal.state.pos,
                dim=1,
            )
            self.blue_on_goal = self.blue_distance < blue_agent.goal.shape.radius
            self.green_on_goal = self.green_distance < green_agent.goal.shape.radius
            self.goal_reached = self.green_on_goal * self.blue_on_goal

            self.step += 1
            # Print time steps if it can be divided by 10
            if self.step % 200 == 0:
                print(f"Step: {self.step}")
                if self.green_on_goal.any():
                    print(f"green_on_goal: {self.green_on_goal}")
                if self.blue_on_goal.any():
                    print(f"blue_on_goal: {self.blue_on_goal}")

            green_shaping = self.green_distance * self.pos_shaping_factor
            self.green_rew = green_agent.shaping - green_shaping
            green_agent.shaping = green_shaping

            blue_shaping = self.blue_distance * self.pos_shaping_factor
            self.blue_rew = blue_agent.shaping - blue_shaping
            blue_agent.shaping = blue_shaping

            self.pos_rew[:, 0] = self.blue_rew
            self.pos_rew[:, 1] = self.green_rew

            self.final_rew[self.goal_reached] = self.final_reward

            # If agents devide from centerline, penalize; exceptions are when they are close to each other and make way for each other
            is_close_mask = (blue_agent.state.pos - green_agent.state.pos).norm(
                dim=-1
            ) <= 4 * (blue_agent.goal.shape.radius + green_agent.goal.shape.radius)
            is_acceptable_deviation_mask_blue = (
                blue_agent.state.pos[:, 1].abs() <= self.corridor_width / 10
            )
            is_acceptable_deviation_mask_green = (
                green_agent.state.pos[:, 1].abs() <= self.corridor_width / 10
            )

            blue_deviation_penalty = (
                -blue_agent.state.pos[:, 1].abs()
                * self.centerline_deviation_penalty
                * (1 - is_close_mask.float())
                * (1 - is_acceptable_deviation_mask_blue.float())
            )
            green_deviation_penalty = (
                -green_agent.state.pos[:, 1].abs()
                * self.centerline_deviation_penalty
                * (1 - is_close_mask.float())
                * (1 - is_acceptable_deviation_mask_green.float())
            )

            self.deviation_pelanity[:, 0] = blue_deviation_penalty
            self.deviation_pelanity[:, 1] = green_deviation_penalty

            # Energy reward
            self.energy_rew[:, 0] = (
                -torch.linalg.vector_norm(blue_agent.action.u, dim=-1)
                / math.sqrt(self.world.dim_p * (blue_agent.f_range**2))
                * self.energy_reward_coeff
            )
            self.energy_rew[:, 1] = (
                -torch.linalg.vector_norm(green_agent.action.u, dim=-1)
                / math.sqrt(self.world.dim_p * (green_agent.f_range**2))
                * self.energy_reward_coeff
            )

            self.rew = (
                self.pos_rew
                + self.deviation_pelanity
                + self.final_rew.unsqueeze(-1)
                + self.energy_rew
            )

        agent.agent_collision_rew[:] = 0
        agent.obstacle_collision_rew[:] = 0
        for a in self.world.agents:
            if a != agent:
                agent.agent_collision_rew[
                    self.world.get_distance(agent, a) <= self.min_collision_distance
                ] += self.agent_collision_penalty
        for landmark in self.world.landmarks:
            if self.world.collides(agent, landmark):
                if landmark in (
                    [*self.passage_1, *self.passage_2]
                    if self.mirror_passage is True
                    else [*self.passage_1]
                ):
                    penalty = self.passage_collision_penalty
                else:
                    penalty = self.obstacle_collision_penalty
                agent.obstacle_collision_rew[
                    self.world.get_distance(agent, landmark)
                    <= self.min_collision_distance
                ] += penalty

        if self.is_share_reward:
            if self.is_reward_propagation:
                if is_first:
                    for i in range(self.n_agents):
                        self.rew_with_propagation[:, i] = self.rew[:, 0 : i + 1].sum(
                            dim=-1
                        ) / (i + 1)
                    self.rew_with_propagation[:, :] = self.rew_with_propagation.sum(
                        dim=-1
                    ).unsqueeze(-1)
                return self.rew_with_propagation[:, agent_index]
            else:
                return self.rew.sum(dim=-1)
        else:
            # Agents have their own rewards
            # Reward propagation (assume agents with lower indices have higher priorities)
            if self.is_reward_propagation:
                # Agent-specific reward is scaled by its priority (agents with lower index have higher priority)
                # Divide by (agent_index + 1) to give more priority to lower indexed agents
                self.rew_with_propagation[:, agent_index] += self.rew[
                    :, agent_index
                ] / (agent_index + 1)

                # Propagate the reward to all lower priority agents (higher index agents)
                for i in range(agent_index + 1, self.n_agents):
                    # Distribute the reward to the lower priority agents
                    # Divide by (i + 1) to appropriately scale the reward according to the agent's priority
                    self.rew_with_propagation[:, i] += self.rew[:, agent_index] / (
                        i + 1
                    )

                # Return the propagated reward for the current agent
                return self.rew_with_propagation[:, agent_index]
            else:
                # If reward propagation is disabled, return the original reward for the agent
                return self.rew[:, agent_index]

    def observation(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        pos_sur = []
        vel_sur = []  # Velocities of surrounding agents
        rel_pos_to_goal_sur = []
        for a in self.world.agents:
            if a != agent:
                pos_sur.append(a.state.pos)
                vel_sur.append(a.state.vel)
                rel_pos_to_goal_sur.append(a.goal.state.pos - a.state.pos)

                # # Observe the actions of surrounding agents
                # if a.action.u is not None:
                #     u_sur_agents.append(a.action.u)
                #     # If agents are far away (more than two times their radius), zero-mask their actions
                #     mask = pos_sur[-1].norm(dim=-1) > 4 * agent.shape.radius
                #     u_sur_agents[-1] = u_sur_agents[-1] * (1 - mask.unsqueeze(-1).float())
                # else:
                #     u_sur_agents.append(torch.zeros_like(agent.state.pos, device=self.world.device, dtype=torch.float32))

        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.goal.state.pos - agent.state.pos,
        ]

        if self.observe_rel_pos:
            observations += pos_sur

        if self.observe_vel_sur:
            observations += vel_sur

        if self.observe_rel_pos_to_goal_sur:
            observations += rel_pos_to_goal_sur

        if self.obs_noise > 0:
            for i, obs in enumerate(observations):
                noise = torch.zeros(*obs.shape, device=self.world.device,).uniform_(
                    -self.obs_noise,
                    self.obs_noise,
                )
                observations[i] = obs + noise

        obs = torch.cat(
            observations,
            dim=-1,
        )

        # Store observation for reuse in `info()`
        self.stored_base_observations[agent_index] = obs

        return obs

    def info(self, agent: Agent):
        agent_index = self.world.agents.index(agent)

        # Nearing agents are the ones except the agent itself (assume full observability)
        nearing_agents_indices = torch.tensor(
            [i for i in range(self.n_agents) if i != agent_index],
            device=self.world.device,
            dtype=torch.int32,
        ).repeat(self.world.batch_dim, 1)

        n_pad = 2 * (self.n_agents - 1)  # Each surrounding agent has 2 actions
        obs_with_actions = torch.nn.functional.pad(
            self.stored_base_observations[agent_index].clone(),
            (0, n_pad),
        )

        return {
            "pos_rew": self.pos_rew[:, agent_index],
            "final_rew": self.final_rew,
            "energy_rew": self.energy_rew[:, agent_index],
            "agent_collision_rew": agent.agent_collision_rew,
            "obstacle_collision_rew": agent.obstacle_collision_rew,
            "nearing_agents_indices": nearing_agents_indices,
            "observation_without_actions": self.stored_base_observations[agent_index],
            "observation_with_actions": obs_with_actions,
        }

    def spawn_map(self, world: World):
        self.scenario_length = 5
        self.passage_length = 0.4
        self.passage_width = 0.4  # box obstacle length
        self.corridor_width = self.passage_length
        self.small_ceiling_length = (self.scenario_length / 2) - (
            self.passage_length / 2
        )
        self.goal_dist_from_wall = self.agent_radius + 0.05
        self.agent_dist_from_wall = 0.5

        self.walls = []
        for i in range(2):
            landmark = Landmark(
                name=f"wall {i}",
                collide=True,
                shape=Line(length=self.corridor_width),
                color=Color.BLACK,
            )
            self.walls.append(landmark)
            world.add_landmark(landmark)
        self.small_ceilings_1 = []
        for i in range(2):
            landmark = Landmark(
                name=f"ceil 1 {i}",
                collide=True,
                shape=Line(length=self.small_ceiling_length),
                color=Color.BLACK,
            )
            self.small_ceilings_1.append(landmark)
            world.add_landmark(landmark)
        self.passage_1 = []
        for i in range(3):
            landmark = Landmark(
                name=f"ceil 2 {i}",
                collide=True,
                shape=Line(
                    length=self.passage_length if i == 2 else self.passage_width
                ),
                color=Color.BLACK,
            )
            self.passage_1.append(landmark)
            world.add_landmark(landmark)

        if self.mirror_passage:
            self.small_ceilings_2 = []
            for i in range(2):
                landmark = Landmark(
                    name=f"ceil 12 {i}",
                    collide=True,
                    shape=Line(length=self.small_ceiling_length),
                    color=Color.BLACK,
                )
                self.small_ceilings_2.append(landmark)
                world.add_landmark(landmark)
            self.passage_2 = []
            for i in range(3):
                landmark = Landmark(
                    name=f"ceil 22 {i}",
                    collide=True,
                    shape=Line(
                        length=self.passage_length if i == 2 else self.passage_width
                    ),
                    color=Color.BLACK,
                )
                self.passage_2.append(landmark)
                world.add_landmark(landmark)
        else:
            # Add landmarks
            landmark = Landmark(
                name="floor",
                collide=True,
                shape=Line(length=self.scenario_length),
                color=Color.BLACK,
            )
            self.floor = landmark
            world.add_landmark(landmark)

    def reset_map(self, env_index):
        # Walls
        for i, landmark in enumerate(self.walls):
            landmark.set_pos(
                torch.tensor(
                    [
                        (
                            -self.scenario_length / 2
                            if i == 0
                            else self.scenario_length / 2
                        ),
                        0.0,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

        # Ceiling
        small_ceiling_pos = self.small_ceiling_length / 2 - self.scenario_length / 2
        for i, landmark in enumerate(self.small_ceilings_1):
            landmark.set_pos(
                torch.tensor(
                    [
                        -small_ceiling_pos if i == 0 else small_ceiling_pos,
                        self.passage_length / 2,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

        # Asymmetric hole
        for i, landmark in enumerate(self.passage_1[:-1]):
            landmark.set_pos(
                torch.tensor(
                    [
                        -self.passage_length / 2 if i == 0 else self.passage_length / 2,
                        self.passage_length / 2 + self.passage_width / 2,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        self.passage_1[-1].set_pos(
            torch.tensor(
                [0, self.passage_length / 2 + self.passage_width],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

        if self.mirror_passage:
            # Ceiling
            for i, landmark in enumerate(self.small_ceilings_2):
                landmark.set_pos(
                    torch.tensor(
                        [
                            -small_ceiling_pos if i == 0 else small_ceiling_pos,
                            -self.passage_length / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

            # Asymmetric hole
            for i, landmark in enumerate(self.passage_2[:-1]):
                landmark.set_pos(
                    torch.tensor(
                        [
                            (
                                -self.passage_length / 2
                                if i == 0
                                else self.passage_length / 2
                            ),
                            -self.passage_length / 2 - self.passage_width / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor(
                        [torch.pi / 2],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            self.passage_2[-1].set_pos(
                torch.tensor(
                    [0, -self.passage_length / 2 - self.passage_width],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        else:
            # Floor
            self.floor.set_pos(
                torch.tensor(
                    [0, -self.passage_length / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def done(self):
        if self.done_on_completion:
            return self.goal_reached
        else:
            return torch.zeros_like(self.goal_reached)


if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(
        scenario=scenario,
        control_two_agents=False,
        shared_reward=False,
        # done_on_completion=True
    )
