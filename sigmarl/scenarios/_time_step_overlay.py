from typing import Optional

import torch

from vmas.simulator.core import World
from vmas.simulator.scenario import BaseScenario


class TimeStepOverlayScenario(BaseScenario):
    """Base scenario that renders elapsed simulation time and episode step."""

    def env_make_world(
        self,
        batch_dim: int,
        device: torch.device,
        **kwargs,
    ) -> World:
        world = super().env_make_world(batch_dim, device, **kwargs)
        self.time_step = torch.zeros(
            batch_dim,
            device=device,
            dtype=torch.int64,
        )
        return world

    def env_reset_world_at(self, env_index: Optional[int]):
        super().env_reset_world_at(env_index)
        if env_index is None:
            self.time_step[:] = 0
        else:
            self.time_step[env_index] = 0

    def post_step(self):
        """Advance the per-environment simulation step counter."""
        self.time_step += 1

    def time_step_overlay(self, env_index: int = 0):
        """Create top-left simulation time and step labels."""
        from vmas.simulator import rendering

        from sigmarl.render_backend import compute_text_scale

        scale = compute_text_scale(*self.viewer_size)
        time_step = int(self.time_step[env_index].item())
        text_x = 10 * scale
        text_y = self.viewer_size[1] - 25 * scale
        return [
            rendering.TextLine(
                text=text,
                x=text_x,
                y=text_y - line_index * 20 * scale,
                font_size=14 * scale,
            )
            for line_index, text in enumerate(
                (f"t: {time_step * self.world.dt:.2f} s", f"k: {time_step}")
            )
        ]

    def extra_render(self, env_index: int = 0):
        """Render the simulation time and step labels."""
        return self.time_step_overlay(env_index)
