#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from sigmarl.scenarios._time_step_overlay import TimeStepOverlayScenario
from vmas.simulator.utils import Color, ScenarioUtils
from sigmarl.helper_scenario import Distances, Observations

DEFAULT_ENERGY_COEFF = 0.02
DEFAULT_GOAL_REWARD = 5.0
DEFAULT_PROGRESS_COEFF = 1.0
DEFAULT_WORLD_SEMIDIM = 1.2
DEFAULT_VELOCITY_NORMALIZER = 1.0
DEFAULT_DT = 0.05
DEFAULT_BOUNDARY_PENALTY = 1.0


class Scenario(TimeStepOverlayScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.energy_coeff = kwargs.pop(
            "energy_coeff", DEFAULT_ENERGY_COEFF
        )  # Weight of team energy penalty
        self.goal_reward = kwargs.pop("goal_reward", DEFAULT_GOAL_REWARD)
        self.progress_coeff = kwargs.pop("progress_coeff", DEFAULT_PROGRESS_COEFF)
        self.world_semidim = kwargs.pop("world_semidim", DEFAULT_WORLD_SEMIDIM)
        self.velocity_normalizer = kwargs.pop(
            "velocity_normalizer", DEFAULT_VELOCITY_NORMALIZER
        )
        self.boundary_penalty = kwargs.pop("boundary_penalty", DEFAULT_BOUNDARY_PENALTY)
        self.shared_rew = kwargs.pop("shared_rew", False)
        parameters = getattr(self, "parameters", None)
        self.dt = kwargs.pop("dt", getattr(parameters, "dt", DEFAULT_DT))
        self.start_same_point = kwargs.pop("start_same_point", False)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        if n_agents < 1:
            raise ValueError("`n_agents` must be at least 1.")
        if self.energy_coeff < 0:
            raise ValueError("`energy_coeff` must be non-negative.")
        if self.goal_reward <= 0:
            raise ValueError("`goal_reward` must be positive.")
        if self.progress_coeff < 0:
            raise ValueError("`progress_coeff` must be non-negative.")
        if self.world_semidim <= 1:
            raise ValueError(
                "`world_semidim` must be greater than 1 to contain reset positions."
            )
        if self.velocity_normalizer <= 0:
            raise ValueError("`velocity_normalizer` must be positive.")
        if self.boundary_penalty < 0:
            raise ValueError("`boundary_penalty` must be non-negative.")
        if not isinstance(self.shared_rew, bool):
            raise ValueError("`shared_rew` must be a boolean.")
        if self.dt <= 0:
            raise ValueError("`dt` must be positive.")

        self.agent_radius = 0.05
        self.goal_radius = 0.03

        # Make world
        world = World(
            batch_dim,
            device,
            dt=getattr(getattr(self, "parameters", None), "dt", 0.05),
            x_semidim=self.world_semidim,
            y_semidim=self.world_semidim,
        )
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
            )
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=self.goal_radius),
            color=Color.GREEN,
        )
        world.add_landmark(goal)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.progress_rew = self.pos_rew.clone()
        self.energy_rew_local = torch.zeros(batch_dim, n_agents, device=device)
        self.boundary_rew_local = self.energy_rew_local.clone()
        self.energy_rew = self.energy_rew_local.clone()
        self.boundary_rew = self.energy_rew_local.clone()
        self.rew = self.energy_rew_local.clone()
        self.boundary_collisions = torch.zeros(
            batch_dim, n_agents, device=device, dtype=torch.bool
        )
        self.previous_min_goal_distance = self.pos_rew.clone()
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)

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

    def compute_goal_distances(self):
        """Compute each agent's centre-to-centre distance from the goal."""
        goal_pos = self.world.landmarks[0].state.pos
        return torch.stack(
            [
                torch.linalg.vector_norm(agent.state.pos - goal_pos, dim=-1)
                for agent in self.world.agents
            ],
            dim=1,
        )

    def compute_boundary_collisions(self):
        """Return which agent spheres are touching a world boundary."""
        boundary_limit = self.world_semidim - self.agent_radius
        positions = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        return torch.any(
            positions.abs() >= boundary_limit - 1e-6,
            dim=-1,
        )

    def reset_agents_at_bounds(self):
        """Respawn only boundary-colliding agents and zero their velocity."""
        collision_indices = torch.where(self.boundary_collisions)
        for env_index_tensor, agent_index_tensor in zip(*collision_indices):
            env_index = int(env_index_tensor.item())
            agent_index = int(agent_index_tensor.item())
            agent = self.world.agents[agent_index]

            occupied_positions = torch.stack(
                [
                    entity.state.pos[env_index]
                    for entity in self.world.entities
                    if entity is not agent
                ],
                dim=0,
            ).unsqueeze(0)
            ScenarioUtils.spawn_entities_randomly(
                [agent],
                self.world,
                env_index,
                self.goal_radius + self.agent_radius + 0.01,
                x_bounds=(-1, 1),
                y_bounds=(-1, 1),
                occupied_positions=occupied_positions,
            )
            agent.set_vel(
                torch.zeros(
                    1,
                    self.world.dim_p,
                    device=self.world.device,
                    dtype=torch.float,
                ),
                batch_index=env_index,
            )

    def reset_world_at(self, env_index: int = None):
        if self.start_same_point:
            for agent in self.world.agents:
                agent.set_pos(
                    torch.zeros(
                        (1, 2) if env_index is not None else (self.world.batch_dim, 2),
                        device=self.world.device,
                        dtype=torch.float,
                    ),
                    batch_index=env_index,
                )
                agent.set_vel(
                    torch.zeros(
                        (1, 2) if env_index is not None else (self.world.batch_dim, 2),
                        device=self.world.device,
                        dtype=torch.float,
                    ),
                    batch_index=env_index,
                )
            ScenarioUtils.spawn_entities_randomly(
                self.world.landmarks,
                self.world,
                env_index,
                self.goal_radius + self.agent_radius + 0.01,
                x_bounds=(-1, 1),
                y_bounds=(-1, 1),
                occupied_positions=torch.zeros(
                    1 if env_index is not None else self.world.batch_dim,
                    1,
                    2,
                    device=self.world.device,
                    dtype=torch.float,
                ),
            )
        else:
            ScenarioUtils.spawn_entities_randomly(
                self.world.policy_agents + self.world.landmarks,
                self.world,
                env_index,
                self.goal_radius + self.agent_radius + 0.01,
                x_bounds=(-1, 1),
                y_bounds=(-1, 1),
            )
            for agent in self.world.agents:
                agent.set_vel(
                    torch.zeros(
                        (1, 2) if env_index is not None else (self.world.batch_dim, 2),
                        device=self.world.device,
                        dtype=torch.float,
                    ),
                    batch_index=env_index,
                )

        for landmark in self.world.landmarks:
            if env_index is None:
                landmark.eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
                self._done[:] = False
            else:
                landmark.eaten[env_index] = False
                landmark.is_rendering[env_index] = True
                self._done[env_index] = False

        # update initial pair-wise distances
        self.compute_agent_distances()
        min_goal_distance = self.compute_goal_distances().min(dim=1).values
        if env_index is None:
            self.previous_min_goal_distance.copy_(min_goal_distance)
            self.pos_rew.zero_()
            self.progress_rew.zero_()
            self.energy_rew_local.zero_()
            self.boundary_rew_local.zero_()
            self.energy_rew.zero_()
            self.boundary_rew.zero_()
            self.rew.zero_()
            self.boundary_collisions.zero_()
        else:
            self.previous_min_goal_distance[env_index] = min_goal_distance[env_index]
            self.pos_rew[env_index] = 0
            self.progress_rew[env_index] = 0
            self.energy_rew_local[env_index] = 0
            self.boundary_rew_local[env_index] = 0
            self.energy_rew[env_index] = 0
            self.boundary_rew[env_index] = 0
            self.rew[env_index] = 0
            self.boundary_collisions[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        # Debug: check agent actions and positions for NaNs
        for a in self.world.agents:
            if hasattr(a, "action") and a.action is not None:
                if torch.isnan(a.action.u).any():
                    print(f"[ERROR] NaN in action for agent {a.name}: {a.action.u}")
                    assert not torch.isnan(
                        a.action.u
                    ).any(), "NaN detected in agent action!"
            if torch.isnan(a.state.pos).any():
                print(f"[ERROR] NaN in position for agent {a.name}: {a.state.pos}")
                assert not torch.isnan(
                    a.state.pos
                ).any(), "NaN detected in agent position!"
            if torch.isnan(a.state.vel).any():
                print(f"[ERROR] NaN in velocity for agent {a.name}: {a.state.vel}")
                assert not torch.isnan(
                    a.state.vel
                ).any(), "NaN detected in agent velocity!"

        # Debug: check landmark positions and eaten flags
        for landmark in self.world.landmarks:
            if torch.isnan(landmark.state.pos).any():
                print(
                    f"[ERROR] NaN in position for landmark {landmark.name}: {landmark.state.pos}"
                )
                assert not torch.isnan(
                    landmark.state.pos
                ).any(), "NaN detected in landmark position!"
            if hasattr(landmark, "eaten") and torch.isnan(landmark.eaten).any():
                print(
                    f"[ERROR] NaN in eaten flag for landmark {landmark.name}: {landmark.eaten}"
                )
                assert not torch.isnan(
                    landmark.eaten
                ).any(), "NaN detected in landmark eaten flag!"

        if is_first:
            goal_distances = self.compute_goal_distances()
            min_goal_distance = goal_distances.min(dim=1).values
            self.any_eaten = min_goal_distance < self.agent_radius + self.goal_radius
            self._done.copy_(self.any_eaten)

            self.progress_rew.copy_(
                self.progress_coeff
                * (self.previous_min_goal_distance - min_goal_distance)
            )

            self.pos_rew.zero_()
            newly_reached = self.any_eaten & ~self.world.landmarks[0].eaten
            self.pos_rew[newly_reached] = self.goal_reward

            self.energy_rew_local.copy_(
                self.energy_coeff
                * -torch.stack(
                    [
                        torch.linalg.vector_norm(a.action.u, dim=-1)
                        / math.sqrt(
                            self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2)
                        )
                        for a in self.world.agents
                    ],
                    dim=1,
                )
            )
            self.boundary_collisions.copy_(self.compute_boundary_collisions())
            self.boundary_rew_local.copy_(
                -self.boundary_penalty * self.boundary_collisions
            )

            if self.shared_rew:
                self.energy_rew.copy_(
                    self.energy_rew_local.mean(dim=1, keepdim=True).expand_as(
                        self.energy_rew
                    )
                )
                self.boundary_rew.copy_(
                    self.boundary_rew_local.mean(dim=1, keepdim=True).expand_as(
                        self.boundary_rew
                    )
                )
            else:
                self.energy_rew.copy_(self.energy_rew_local)
                self.boundary_rew.copy_(self.boundary_rew_local)

            shared_task_rew = self.pos_rew + self.progress_rew
            self.rew.copy_(
                shared_task_rew.unsqueeze(-1) + self.energy_rew + self.boundary_rew
            )

            self.reset_agents_at_bounds()
            self.previous_min_goal_distance.copy_(
                self.compute_goal_distances().min(dim=1).values
            )
            # Keep distance metadata consistent with post-respawn observations.
            self.compute_agent_distances()

        if is_last:
            self.world.landmarks[0].eaten[self.any_eaten] = True
            self.world.landmarks[0].is_rendering[self.any_eaten] = False

        agent_index = self.world.agents.index(agent)
        return self.rew[:, agent_index]

    def observation(self, agent: Agent):
        goal_distances = self.compute_goal_distances()
        agent_index = self.world.agents.index(agent)
        agent_goal_distance = goal_distances[:, agent_index]
        min_goal_distance, closest_agent_index = goal_distances.min(dim=1)
        is_closest_to_goal = (closest_agent_index == agent_index).unsqueeze(-1)
        max_world_distance = 2 * math.sqrt(2) * self.world_semidim
        goal_distance_advantage = (
            (agent_goal_distance - min_goal_distance) / max_world_distance
        ).unsqueeze(-1)

        obs = torch.cat(
            [
                (agent.state.pos / self.world_semidim).clamp(-1, 1),
                (agent.state.vel / self.velocity_normalizer).clamp(-1, 1),
                (
                    (self.world.landmarks[0].state.pos - agent.state.pos)
                    / (2 * self.world_semidim)
                ).clamp(-1, 1),
                self.world.landmarks[0].eaten.unsqueeze(-1),
                is_closest_to_goal.to(agent.state.pos.dtype),
                goal_distance_advantage.clamp(0, 1),
            ],
            dim=-1,
        )
        # NaN check for debugging
        if torch.isnan(obs).any():
            print(f"[ERROR] NaN in observation for agent {agent.name}", obs)
            assert not torch.isnan(obs).any(), "NaN detected in observation!"
        return obs

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        agent_index = self.world.agents.index(agent)

        nearing_agents_indices = torch.tensor(
            [i for i in range(len(self.world.agents)) if i != agent_index],
            device=self.world.device,
            dtype=torch.int32,
        ).repeat(self.world.batch_dim, 1)

        obs_without_actions = self.observation(agent)

        n_pad = self.observations.n_nearing_agents * 2
        obs_with_actions = torch.nn.functional.pad(
            obs_without_actions.clone(),
            (0, n_pad),
        )

        info_dict = {
            "pos_rew": self.pos_rew,
            "progress_rew": self.progress_rew,
            "energy_rew": self.energy_rew[:, agent_index],
            "energy_rew_local": self.energy_rew_local[:, agent_index],
            "boundary_rew": self.boundary_rew[:, agent_index],
            "boundary_rew_local": self.boundary_rew_local[:, agent_index],
            "boundary_collision": self.boundary_collisions[:, agent_index],
            "nearing_agents_indices": nearing_agents_indices,
            "observation_without_actions": obs_without_actions,
            "observation_with_actions": obs_with_actions,
        }
        return info_dict

    def done(self):
        return self._done


if __name__ == "__main__":
    render_interactively(
        Scenario(),
        control_two_agents=True,
        n_agents=4,
        energy_coeff=DEFAULT_ENERGY_COEFF,
    )
