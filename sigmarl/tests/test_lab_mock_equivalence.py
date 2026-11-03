# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Equivalence test: physics_backend="mock" must reproduce "vmas".

The lab backend replaces only the physics integration of WorldCustom with a
bridge round trip (SigmaRL frame -> lab frame -> SPEED_CURVATURE wire format
-> integration -> measured states -> SigmaRL frame). With the MockBridge the
integration is the same kinematic bicycle model, so a rollout must match a
pure-VMAS rollout up to float32 round-off introduced by the frame round trip.

Two accepted deviations:

1. The mock wraps yaw to (-pi, pi] each step (a real lab measurement is
   always wrapped), while VMAS integrates rot unbounded. Rotations are
   therefore compared modulo 2*pi. If a trajectory crosses the +-pi
   boundary, raw-rot-dependent observations would differ by a 2*pi offset;
   the fixed seed and mild action table below avoid that.
2. On reset, the lab scenario re-anchors the short-term reference-path
   window to the *measured* spawn pose (load-bearing on real hardware,
   where placement error is real). The float32 frame round trip can flip a
   knife-edge nearest-point argmin by one path point, transiently shifting
   the window for the post-reset observation before it re-converges.
   Observations are therefore not compared on done (reset) steps.
"""

import random
import unittest

import torch

from sigmarl.helper_common import Parameters
from sigmarl.mappo_cavs import MAPPOCAVs

SEED = 7
N_AGENTS = 4
N_STEPS = 50


def make_parameters(backend: str) -> Parameters:
    parameters = Parameters.from_json("sigmarl/config.json")
    parameters.scenario_type = "cpm_entire"
    parameters.n_agents = N_AGENTS
    parameters.dt = 0.1
    parameters.num_vmas_envs = 1
    parameters.max_steps = 64
    parameters.random_seed = SEED
    parameters.is_prb = False
    parameters.is_using_prioritized_marl = False
    parameters.is_using_cbf_training = False
    parameters.is_using_cbf_testing = False
    parameters.is_testing_mode = False
    parameters.is_save_intermediate_model = False
    # Disable per-step randomness in the observation pipeline so the two
    # sequential runs consume the global RNG identically.
    parameters.is_obs_noise = False
    parameters.visible_agent_dropout_probability = 0.0
    parameters.is_challenging_initial_state_buffer = False
    parameters.physics_backend = backend
    parameters.lab_period_ms = 100
    return parameters


def build_env(backend: str):
    parameters = make_parameters(backend)
    trainer = MAPPOCAVs(parameters)  # seeds torch + random from random_seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    return trainer._setup_environment()


def action_table(n_steps: int, n_agents: int) -> torch.Tensor:
    """Deterministic, mild open-loop actions: [T, n_agents, 2]."""
    t = torch.arange(n_steps, dtype=torch.float32).unsqueeze(1)
    phase = torch.arange(n_agents, dtype=torch.float32).unsqueeze(0)
    speeds = 0.25 + 0.15 * torch.sin((t + 3.0 * phase) / 7.0)
    steers = 0.12 * torch.sin((t + 5.0 * phase) / 5.0)
    return torch.stack([speeds, steers], dim=-1)


def run_lockstep(env, actions: torch.Tensor):
    """Step the env open-loop and record per-step states/rewards/dones."""
    torch.manual_seed(SEED + 1)
    random.seed(SEED + 1)
    td = env.reset()
    records = []
    for step in range(actions.shape[0]):
        td["agents", "action"] = actions[step].unsqueeze(0)
        td, td_ = env.step_and_maybe_reset(td)
        agents = env.scenario.world.agents
        records.append(
            {
                "pos": torch.stack([a.state.pos[0].clone() for a in agents]),
                "rot": torch.stack([a.state.rot[0].clone() for a in agents]),
                "speed": torch.stack([a.state.speed[0].clone() for a in agents]),
                "steering": torch.stack([a.state.steering[0].clone() for a in agents]),
                "vel": torch.stack([a.state.vel[0].clone() for a in agents]),
                "obs": td_["agents", "observation"][0].clone(),
                "reward": td["next", "agents", "reward"][0].clone(),
                "done": td["next", "done"][0].clone(),
            }
        )
        td = td_
    return records


def assert_angles_close(a: torch.Tensor, b: torch.Tensor, atol: float):
    diff = torch.atan2(torch.sin(a - b), torch.cos(a - b))
    torch.testing.assert_close(diff, torch.zeros_like(diff), atol=atol, rtol=0.0)


class TestLabMockEquivalence(unittest.TestCase):
    def test_rollout_equivalence(self):
        env_vmas = build_env("vmas")
        env_mock = build_env("mock")
        actions = action_table(N_STEPS, N_AGENTS)

        rec_vmas = run_lockstep(env_vmas, actions)
        rec_mock = run_lockstep(env_mock, actions)

        for step, (a, b) in enumerate(zip(rec_vmas, rec_mock)):
            msg = f"mismatch at step {step}"
            torch.testing.assert_close(a["pos"], b["pos"], atol=2e-3, rtol=0.0, msg=msg)
            assert_angles_close(a["rot"], b["rot"], atol=2e-3)
            torch.testing.assert_close(
                a["speed"], b["speed"], atol=2e-3, rtol=0.0, msg=msg
            )
            torch.testing.assert_close(
                a["steering"], b["steering"], atol=2e-3, rtol=0.0, msg=msg
            )
            torch.testing.assert_close(a["vel"], b["vel"], atol=2e-3, rtol=0.0, msg=msg)
            if not bool(a["done"].any()):
                # Post-reset observations may transiently differ by one
                # reference-path point (deviation 2 in the module docstring).
                torch.testing.assert_close(
                    a["obs"], b["obs"], atol=5e-3, rtol=0.0, msg=msg
                )
            torch.testing.assert_close(
                a["reward"], b["reward"], atol=5e-3, rtol=0.0, msg=msg
            )
            self.assertTrue(
                torch.equal(a["done"], b["done"]),
                f"done mismatch at step {step}",
            )

    def test_reset_equivalence(self):
        """env.reset() after stepping must produce identical fresh spawns."""
        env_vmas = build_env("vmas")
        env_mock = build_env("mock")
        actions = action_table(5, N_AGENTS)
        run_lockstep(env_vmas, actions)
        run_lockstep(env_mock, actions)

        torch.manual_seed(SEED + 2)
        random.seed(SEED + 2)
        td_vmas = env_vmas.reset()
        torch.manual_seed(SEED + 2)
        random.seed(SEED + 2)
        td_mock = env_mock.reset()

        for i in range(N_AGENTS):
            a = env_vmas.scenario.world.agents[i].state
            b = env_mock.scenario.world.agents[i].state
            torch.testing.assert_close(a.pos, b.pos, atol=1e-5, rtol=0.0)
            assert_angles_close(a.rot, b.rot, atol=1e-5)
            torch.testing.assert_close(a.speed, b.speed, atol=1e-6, rtol=0.0)
            torch.testing.assert_close(a.steering, b.steering, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(
            td_vmas["agents", "observation"],
            td_mock["agents", "observation"],
            atol=1e-4,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
