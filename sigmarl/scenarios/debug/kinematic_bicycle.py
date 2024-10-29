# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Box
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation

from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

from sigmarl.constants import AGENTS

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from sigmarl.dynamics import KinematicBicycleModel
from sigmarl.helper_training import Vehicle, WorldCustom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Kinematic bicycle model example scenario
        """
        self.n_agents = kwargs.get("n_agents", 2)
        width = kwargs.get("width", 0.1)  # Agent width
        l_f = kwargs.get(
            "l_f", 0.1
        )  # Distance between the front axle and the center of gravity
        l_r = kwargs.get(
            "l_r", 0.1
        )  # Distance between the rear axle and the center of gravity
        max_steering = kwargs.get(
            "max_steering",
            torch.tensor(35.0, device=device, dtype=torch.float32).deg2rad(),
        )
        max_speed = kwargs.get(
            "max_speed", torch.tensor(1.0, device=device, dtype=torch.float32)
        )

        # Make world
        world = WorldCustom(batch_dim, device, substeps=10, collision_force=500)

        for i in range(self.n_agents):
            # Use the kinematic bicycle model for the first agent
            agent = Vehicle(
                name=f"bicycle_{i}",
                shape=Box(length=l_f + l_r, width=width),
                collide=True,
                render_action=True,
                max_speed=max_speed,
                u_range=[1, max_steering],
                u_multiplier=[1, 1],
                dynamics=KinematicBicycleModel(  # Use the kinematic bicycle model for each agent
                    l_f=AGENTS["l_f"],
                    l_r=AGENTS["l_r"],
                    max_speed=max_speed,
                    max_steering=max_steering,
                    device=world.device,
                ),
            )

            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.1,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim)

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            agent.state.vel,
        ]
        return torch.cat(
            observations,
            dim=-1,
        )

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Agent rotation
        for agent in self.world.agents:
            color = Color.BLACK.value
            line = rendering.Line(
                (0, 0),
                (0.1, 0),
                width=1,
            )
            xform = rendering.Transform()
            xform.set_rotation(agent.state.rot[env_index])
            xform.set_translation(*agent.state.pos[env_index])
            line.add_attr(xform)
            line.set_color(*color)
            geoms.append(line)

        return geoms


# ... and the code to run the simulation.
if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(
        scenario,
        control_two_agents=False,
        width=0.1,
        l_f=0.1,
        l_r=0.1,
        display_info=True,
    )
