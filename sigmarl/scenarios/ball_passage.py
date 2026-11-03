#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from sigmarl.scenarios._time_step_overlay import TimeStepOverlayScenario
from vmas.simulator.utils import Color, ScenarioUtils, X, Y
from sigmarl.helper_scenario import Distances, Observations


class Scenario(TimeStepOverlayScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_passages = kwargs.pop("n_passages", 1)
        self.fixed_passage = kwargs.pop("fixed_passage", False)
        self.random_start_angle = kwargs.pop("random_start_angle", True)

        # ball-passage tuning independent of the shared training configuration.
        self.shared_rew = kwargs.pop("shared_rew", False)
        self.success_reward = float(kwargs.pop("success_reward", 2.0))
        success_distance = kwargs.pop("success_distance", None)
        self.agent_pos_shaping_factor = float(
            kwargs.pop("agent_pos_shaping_factor", 0.1)
        )
        pushing_distance = kwargs.pop("pushing_distance", None)
        self.pushing_angle = float(kwargs.pop("pushing_angle", torch.pi / 6))
        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert 1 <= self.n_passages <= 20

        self.pos_shaping_factor = 1
        self.collision_reward = -0.06

        self.n_agents = 2

        self.agent_spacing = 0.5
        self.agent_radius = 0.03333
        self.ball_radius = self.agent_radius
        self.passage_width = 0.2
        self.passage_length = 0.103
        # By default, success means that the ball and goal spheres overlap.
        self.success_distance = (
            2 * self.ball_radius
            if success_distance is None
            else float(success_distance)
        )
        # Agents are guided to two symmetric contact points behind the ball.
        self.pushing_distance = (
            self.agent_radius + self.ball_radius
            if pushing_distance is None
            else float(pushing_distance)
        )

        if self.success_reward < 0:
            raise ValueError("`success_reward` must be non-negative.")
        if not isinstance(self.shared_rew, bool):
            raise ValueError("`shared_rew` must be a boolean.")
        if self.success_distance <= 0:
            raise ValueError("`success_distance` must be positive.")
        if self.agent_pos_shaping_factor < 0:
            raise ValueError("`agent_pos_shaping_factor` must be non-negative.")
        if self.pushing_distance <= 0:
            raise ValueError("`pushing_distance` must be positive.")
        if not 0 <= self.pushing_angle < torch.pi / 2:
            raise ValueError("`pushing_angle` must be in [0, pi / 2).")

        self.visualize_semidims = False

        # Make world
        world = World(
            batch_dim,
            device,
            dt=getattr(getattr(self, "parameters", None), "dt", 0.05),
            x_semidim=1,
            y_semidim=1,
            drag=0,
            linear_friction=0.0,
        )
        # Add agents
        for i in range(2):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
                mass=2,
                drag=0.25,
            )
            world.add_agent(agent)

        self.goal = Landmark(
            name="goal",
            shape=Sphere(radius=self.ball_radius),
            collide=False,
            color=Color.GREEN,
        )
        world.add_landmark(self.goal)

        self.ball = Landmark(
            name="ball",
            shape=Sphere(radius=self.ball_radius),
            collide=True,
            movable=True,
            mass=1,
            color=Color.BLACK,
            linear_friction=0.02,
        )
        world.add_landmark(self.ball)

        self.create_passage_map(world)

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.collision_rew = self.pos_rew.clone()
        self.agent_pos_rew = self.pos_rew.clone()
        self.success_rew = self.pos_rew.clone()
        self.agent_pos_shaping = self.pos_rew.clone()
        self.agent_pos_rew_per_agent = torch.zeros(
            batch_dim,
            self.n_agents,
            device=device,
            dtype=torch.float32,
        )
        self.agent_pos_shaping_per_agent = self.agent_pos_rew_per_agent.clone()
        self.agent_collision_rew = self.agent_pos_rew_per_agent.clone()
        self.ball_collision_rew = self.pos_rew.clone()
        self.success_achieved = torch.zeros(
            batch_dim,
            device=device,
            dtype=torch.bool,
        )
        # Infrastructure needed for training system
        self.stored_base_observations = [None] * self.n_agents
        self.distances = Distances(type="c2c")
        self.observations = Observations(
            n_nearing_agents=self.n_agents - 1,  # Each agent observes all other agents
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
        n_nearing_agents_configured = n_agents - 1
        k = min(n_nearing_agents_configured, n_agents - 1)

        nearest_idx = torch.argsort(d, dim=-1)[..., :k]
        self.observations.nearing_agents_indices = nearest_idx
        self.observations.n_nearing_agents = k

    def _closest_passage_position(self):
        passage_positions = torch.stack(
            [p.state.pos for p in self.passages if not p.collide],
            dim=1,
        )
        closest_indices = torch.linalg.vector_norm(
            passage_positions - self.ball.state.pos.unsqueeze(1),
            dim=-1,
        ).argmin(dim=1)
        batch_indices = torch.arange(
            self.world.batch_dim,
            device=self.world.device,
        )
        return passage_positions[batch_indices, closest_indices]

    def _get_pushing_slots(self):
        """Return two symmetric useful contact positions behind the ball."""
        closest_passage = self._closest_passage_position()
        ball_passed = self.ball.state.pos[:, Y] > 0
        active_target = torch.where(
            ball_passed.unsqueeze(-1),
            self.goal.state.pos,
            closest_passage,
        )

        target_direction = active_target - self.ball.state.pos
        target_norm = torch.linalg.vector_norm(
            target_direction,
            dim=-1,
            keepdim=True,
        )
        fallback_direction = torch.zeros_like(target_direction)
        fallback_direction[:, Y] = 1
        target_direction = torch.where(
            target_norm > 1e-6,
            target_direction / target_norm.clamp_min(1e-6),
            fallback_direction,
        )
        perpendicular = torch.stack(
            [-target_direction[:, Y], target_direction[:, X]],
            dim=-1,
        )

        angle = torch.as_tensor(
            self.pushing_angle,
            device=self.world.device,
            dtype=torch.float32,
        )
        backward = -target_direction * torch.cos(angle)
        lateral = perpendicular * torch.sin(angle)
        slot_offsets = torch.stack(
            [backward + lateral, backward - lateral],
            dim=1,
        )
        return self.ball.state.pos.unsqueeze(1) + (self.pushing_distance * slot_offsets)

    def _compute_agent_pos_shaping_per_agent(self):
        """Return each agent's cost under the minimum-cost slot assignment."""
        agent_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents],
            dim=1,
        )
        pushing_slots = self._get_pushing_slots()
        distances = torch.linalg.vector_norm(
            agent_positions.unsqueeze(2) - pushing_slots.unsqueeze(1),
            dim=-1,
        )
        direct_assignment = distances[:, 0, 0] + distances[:, 1, 1]
        swapped_assignment = distances[:, 0, 1] + distances[:, 1, 0]
        use_direct_assignment = direct_assignment <= swapped_assignment
        agent_0_distance = torch.where(
            use_direct_assignment,
            distances[:, 0, 0],
            distances[:, 0, 1],
        )
        agent_1_distance = torch.where(
            use_direct_assignment,
            distances[:, 1, 1],
            distances[:, 1, 0],
        )
        return (
            torch.stack([agent_0_distance, agent_1_distance], dim=1)
            * self.agent_pos_shaping_factor
        )

    def _compute_agent_pos_shaping(self):
        """Permutation-invariant mean distance potential for the team."""
        return self._compute_agent_pos_shaping_per_agent().mean(dim=1)

    def _is_success(self):
        return (
            torch.linalg.vector_norm(
                self.ball.state.pos - self.goal.state.pos,
                dim=1,
            )
            <= self.success_distance
        )

    def reset_world_at(self, env_index: int = None):
        start_angle = torch.zeros(
            (1, 1) if env_index is not None else (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -torch.pi / 2 if self.random_start_angle else -torch.pi / 2,
            torch.pi / 2 if self.random_start_angle else -torch.pi / 2,
        )

        start_delta_x = (self.agent_spacing / 2) * torch.cos(start_angle)
        start_delta_x_abs = start_delta_x.abs()
        min_x_start = -self.world.x_semidim + (self.agent_radius + start_delta_x_abs)
        max_x_start = self.world.x_semidim - (self.agent_radius + start_delta_x_abs)
        start_delta_y = (self.agent_spacing / 2) * torch.sin(start_angle)
        start_delta_y_abs = start_delta_y.abs()
        min_y_start = -self.world.y_semidim + (self.agent_radius + start_delta_y_abs)
        max_y_start = (
            -2 * self.agent_radius - self.passage_width / 2 - start_delta_y_abs
        )

        min_x_goal = -self.world.x_semidim + self.agent_radius
        max_x_goal = self.world.x_semidim - self.agent_radius
        min_y_goal = 2 * self.agent_radius + self.passage_width / 2
        max_y_goal = self.world.y_semidim - self.agent_radius

        ball_pos = torch.cat(
            [
                (min_x_start - max_x_start)
                * torch.rand(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                )
                + max_x_start,
                (min_y_start - max_y_start)
                * torch.rand(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                )
                + max_y_start,
            ],
            dim=1,
        )

        self.ball.set_pos(
            ball_pos,
            batch_index=env_index,
        )

        for i, agent in enumerate(self.world.agents):
            if i == 0:
                agent.set_pos(
                    ball_pos - torch.cat([start_delta_x, start_delta_y], dim=1),
                    batch_index=env_index,
                )
            else:
                agent.set_pos(
                    ball_pos + torch.cat([start_delta_x, start_delta_y], dim=1),
                    batch_index=env_index,
                )

        self.goal.set_pos(
            torch.cat(
                [
                    (min_x_goal - max_x_goal)
                    * torch.rand(
                        (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                    + max_x_goal,
                    (min_y_goal - max_y_goal)
                    * torch.rand(
                        (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                    + max_y_goal,
                ],
                dim=1,
            ),
            batch_index=env_index,
        )

        self.spawn_passage_map(env_index)

        if env_index is None:
            self.ball.pos_shaping_pre = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.ball.state.pos - p.state.pos, dim=1
                        )
                        for p in self.passages
                        if not p.collide
                    ],
                    dim=1,
                ).min(dim=1)[0]
                * self.pos_shaping_factor
            )
            self.ball.pos_shaping_post = (
                torch.linalg.vector_norm(
                    self.ball.state.pos - self.goal.state.pos, dim=1
                )
                * self.pos_shaping_factor
            )

        else:
            self.ball.pos_shaping_pre[env_index] = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.ball.state.pos[env_index] - p.state.pos[env_index],
                        ).unsqueeze(-1)
                        for p in self.passages
                        if not p.collide
                    ],
                    dim=1,
                ).min(dim=1)[0]
                * self.pos_shaping_factor
            )
            self.ball.pos_shaping_post[env_index] = (
                torch.linalg.vector_norm(
                    self.ball.state.pos[env_index] - self.goal.state.pos[env_index],
                )
                * self.pos_shaping_factor
            )

        current_agent_pos_shaping_per_agent = (
            self._compute_agent_pos_shaping_per_agent()
        )
        current_agent_pos_shaping = current_agent_pos_shaping_per_agent.mean(dim=1)
        if env_index is None:
            self.agent_pos_shaping_per_agent = current_agent_pos_shaping_per_agent
            self.agent_pos_shaping = current_agent_pos_shaping
            self.success_achieved[:] = False
        else:
            self.agent_pos_shaping_per_agent[
                env_index
            ] = current_agent_pos_shaping_per_agent[env_index]
            self.agent_pos_shaping[env_index] = current_agent_pos_shaping[env_index]
            self.success_achieved[env_index] = False

        # Update initial pair-wise distances
        self.compute_agent_distances()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        agent_index = self.world.agents.index(agent)

        if is_first:
            # Keep distance matrix up-to-date each step
            self.compute_agent_distances()

            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
            self.pos_rew[:] = 0
            self.collision_rew[:] = 0
            self.agent_pos_rew[:] = 0
            self.agent_pos_rew_per_agent[:] = 0
            self.agent_collision_rew[:] = 0
            self.ball_collision_rew[:] = 0
            self.success_rew[:] = 0

            ball_passed = self.ball.state.pos[:, Y] > 0

            # Pos shaping
            ball_dist_to_closest_pass = torch.stack(
                [
                    torch.linalg.vector_norm(self.ball.state.pos - p.state.pos, dim=1)
                    for p in self.passages
                    if not p.collide
                ],
                dim=1,
            ).min(dim=1)[0]
            ball_shaping = ball_dist_to_closest_pass * self.pos_shaping_factor
            self.pos_rew[~ball_passed] += (self.ball.pos_shaping_pre - ball_shaping)[
                ~ball_passed
            ]
            self.ball.pos_shaping_pre = ball_shaping

            ball_dist_to_goal = torch.linalg.vector_norm(
                self.ball.state.pos - self.goal.state.pos,
                dim=1,
            )
            ball_shaping = ball_dist_to_goal * self.pos_shaping_factor
            self.pos_rew[ball_passed] += (self.ball.pos_shaping_post - ball_shaping)[
                ball_passed
            ]
            self.ball.pos_shaping_post = ball_shaping

            # Assign both pushing slots jointly, then retain each agent's local
            # potential delta for optional non-shared credit assignment.
            agent_pos_shaping_per_agent = self._compute_agent_pos_shaping_per_agent()
            self.agent_pos_rew_per_agent += (
                self.agent_pos_shaping_per_agent - agent_pos_shaping_per_agent
            )
            agent_pos_shaping = agent_pos_shaping_per_agent.mean(dim=1)
            self.agent_pos_rew += self.agent_pos_rew_per_agent.mean(dim=1)
            self.agent_pos_shaping_per_agent = agent_pos_shaping_per_agent
            self.agent_pos_shaping = agent_pos_shaping

            # Award the terminal bonus once when physical goal overlap begins.
            success = self._is_success()
            newly_successful = success & ~self.success_achieved
            self.success_rew[newly_successful] = self.success_reward
            self.success_achieved |= success

            # Agent collisions
            for collision_agent_index, collision_agent in enumerate(self.world.agents):
                for passage in self.passages:
                    if passage.collide:
                        self.agent_collision_rew[
                            self.world.is_overlapping(collision_agent, passage),
                            collision_agent_index,
                        ] += self.collision_reward

            # Ball collisions
            for p in self.passages:
                if p.collide:
                    self.ball_collision_rew[
                        self.world.is_overlapping(p, self.ball)
                    ] += self.collision_reward

            self.collision_rew = (
                self.agent_collision_rew.sum(dim=1) + self.ball_collision_rew
            )
            if self.shared_rew:
                self.rew = (
                    self.pos_rew
                    + self.agent_pos_rew
                    + self.success_rew
                    + self.collision_rew
                )
            else:
                shared_components = (
                    self.pos_rew + self.success_rew + self.ball_collision_rew
                )
                self.rew = (
                    shared_components.unsqueeze(-1)
                    + self.agent_pos_rew_per_agent
                    + self.agent_collision_rew
                )

        return self.rew if self.shared_rew else self.rew[:, agent_index]

    def observation(self, agent: Agent):
        agent_index = self.world.agents.index(agent)

        # get positions of all entities in this agent's reference frame
        passage_obs = []
        for passage in self.passages:
            if not passage.collide:
                passage_obs.append(agent.state.pos - passage.state.pos)
        teammate = next(
            other_agent for other_agent in self.world.agents if other_agent is not agent
        )

        obs = torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.goal.state.pos,
                agent.state.pos - self.ball.state.pos,
                self.ball.state.vel,
                teammate.state.pos - agent.state.pos,
                teammate.state.vel - agent.state.vel,
                *passage_obs,
            ],
            dim=-1,
        )

        # Store observation for reuse in `info()`
        self.stored_base_observations[agent_index] = obs

        return obs

    def done(self):
        return (
            self._is_success()
            | (-self.world.x_semidim + self.ball_radius >= self.ball.state.pos[:, X])
            | (self.ball.state.pos[:, X] >= self.world.x_semidim - self.ball_radius)
            | (-self.world.y_semidim + self.ball_radius >= self.ball.state.pos[:, Y])
            | (self.ball.state.pos[:, Y] >= self.world.y_semidim - self.ball_radius)
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        agent_index = self.world.agents.index(agent)
        local_collision_rew = (
            self.agent_collision_rew[:, agent_index] + self.ball_collision_rew
        )

        # Use the nearing agents indices computed during agent distance calculation
        if self.observations.nearing_agents_indices is not None:
            nearing_agents_indices = self.observations.nearing_agents_indices[
                :, agent_index
            ]
        else:
            # Fallback: create nearing agents indices manually (all other agents)
            nearing_agents_indices = torch.tensor(
                [i for i in range(self.n_agents) if i != agent_index],
                device=self.world.device,
                dtype=torch.int32,
            ).repeat(self.world.batch_dim, 1)

        # Base observation (same as returned by the observation() method)
        obs_without_actions = self.stored_base_observations[agent_index]

        # Create a padded version for use with opponent modeling or action propagation
        # The padding size depends on the number of other agents to observe
        n_pad = self.observations.n_nearing_agents * 2  # 2 actions per agent
        obs_with_actions = torch.nn.functional.pad(
            obs_without_actions.clone(),
            (0, n_pad),
        )

        return {
            "pos_rew": self.pos_rew,
            "agent_pos_rew": (
                self.agent_pos_rew
                if self.shared_rew
                else self.agent_pos_rew_per_agent[:, agent_index]
            ),
            "agent_pos_rew_local": self.agent_pos_rew_per_agent[:, agent_index],
            "success_rew": self.success_rew,
            "collision_rew": (
                self.collision_rew if self.shared_rew else local_collision_rew
            ),
            "agent_collision_rew": self.agent_collision_rew[:, agent_index],
            "ball_collision_rew": self.ball_collision_rew,
            "is_success": self._is_success(),
            "nearing_agents_indices": nearing_agents_indices,
            "observation_without_actions": obs_without_actions,
            "observation_with_actions": obs_with_actions,
        }

    def create_passage_map(self, world: World):
        # Add landmarks
        self.passages = []
        n_boxes = int(
            (2 * world.x_semidim + 2 * self.agent_radius) // self.passage_length
        )

        def removed(i):
            return (
                (n_boxes // 2) - self.n_passages / 2
                <= i
                < (n_boxes // 2) + self.n_passages / 2
            )

        for i in range(n_boxes):
            passage = Landmark(
                name=f"passage {i}",
                collide=not removed(i),
                movable=False,
                shape=Box(length=self.passage_length, width=self.passage_width),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            self.passages.append(passage)
            world.add_landmark(passage)

    def spawn_passage_map(self, env_index):
        if not self.fixed_passage:
            order = torch.randperm(len(self.passages)).tolist()
            self.passages_to_place = [self.passages[i] for i in order]
        else:
            self.passages_to_place = self.passages
        for i, passage in enumerate(self.passages_to_place):
            if not passage.collide:
                passage.is_rendering[:] = False
            passage.neighbour = False
            try:
                passage.neighbour += not self.passages_to_place[i - 1].collide
            except IndexError:
                pass
            try:
                passage.neighbour += not self.passages_to_place[i + 1].collide
            except IndexError:
                pass
            pos = torch.tensor(
                [
                    -1
                    - self.agent_radius
                    + self.passage_length / 2
                    + self.passage_length * i,
                    0.0,
                ],
                dtype=torch.float32,
                device=self.world.device,
            )
            passage.neighbour *= passage.collide
            passage.set_pos(
                pos,
                batch_index=env_index,
            )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []
        # Perimeter
        for i in range(4):
            geom = Line(length=2 + self.agent_radius * 2).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.world.x_semidim + self.agent_radius
                        if i == 0
                        else -self.world.x_semidim - self.agent_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.world.x_semidim + self.agent_radius
                        if i == 1
                        else -self.world.x_semidim - self.agent_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        geoms.extend(self.time_step_overlay(env_index))

        return geoms


if __name__ == "__main__":
    render_interactively(Scenario(), control_two_agents=True)
