# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Manual (human-in-the-loop) lab reset: poses are measured, never sampled.

On real hardware vehicles cannot be teleported. With
``Parameters.lab_manual_reset = True`` the operator places them by hand and
SigmaRL must adopt those poses and derive the reference paths from them.

These tests run against the MockBridge with ``SIGMARL_LAB_AUTO_CONFIRM=1`` so
the operator prompt does not block. The mock's "hands" are the test itself: it
writes lab-frame poses into ``bridge._x`` before the environment is built, and
then asserts that those exact poses survive the reset. The default
(``lab_manual_reset = False``) respawn/teleport path is covered by
``test_lab_mock_equivalence`` and is deliberately untouched.
"""

import os
import random
import unittest
from unittest import mock

import torch

from sigmarl.helper_common import Parameters
from sigmarl.lab import scenario_road_traffic_lab as scenario_module
from sigmarl.lab.bridge_base import LabStopped
from sigmarl.mappo_cavs import MAPPOCAVs
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt_sim import (
    WorldStateRTSimulation,
)
from sigmarl.lab.world_state_lab import WorldStateLab

SEED = 11
N_AGENTS = 4
# Index along each reference path's center line at which a vehicle is "placed".
PLACEMENT_POINT = 20


def make_parameters(manual_reset: bool = True) -> Parameters:
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
    parameters.is_obs_noise = False
    parameters.visible_agent_dropout_probability = 0.0
    parameters.is_challenging_initial_state_buffer = False
    parameters.physics_backend = "mock"
    parameters.lab_period_ms = 100
    parameters.lab_manual_reset = manual_reset
    # Deliberately ON to prove the manual path ignores it: the measured speed
    # of a hand-placed (stationary) vehicle is what must be adopted.
    parameters.is_zero_initial_speed = False
    return parameters


def pick_spread_placements(world_state, n_agents: int):
    """Choose n_agents well-separated center-line points, SigmaRL frame.

    Returns (pos [n, 2], yaw [n]). Greedy farthest-point selection over the
    map's reference paths so the placement passes the scenario's minimum
    inter-vehicle distance check.
    """
    ref_paths = world_state.ref_paths_map_related.long_term_all
    candidates = []
    for path_id, ref_path in enumerate(ref_paths):
        center_line = ref_path["center_line"]
        idx = min(PLACEMENT_POINT, center_line.shape[0] - 2)
        candidates.append(
            (
                path_id,
                center_line[idx].clone(),
                ref_path["center_line_yaw"][idx].clone(),
            )
        )

    chosen = [candidates[0]]
    while len(chosen) < n_agents:
        best, best_distance = None, -1.0
        for candidate in candidates:
            distance = min(
                float(torch.norm(candidate[1] - taken[1])) for taken in chosen
            )
            if distance > best_distance:
                best, best_distance = candidate, distance
        chosen.append(best)
        candidates.remove(best)

    pos = torch.stack([c[1] for c in chosen])
    yaw = torch.stack([c[2].reshape(()) for c in chosen])
    return pos, yaw


def build_manual_env(parameters):
    """Build the env with the mock fleet pre-placed at spread-out poses.

    ``make_bridge`` is patched so the placement lands in the bridge *before*
    VMAS' construction-time reset reads it — the mock's default state is every
    vehicle at the origin, which the placement validator correctly rejects.
    """
    trainer = MAPPOCAVs(parameters)
    torch.manual_seed(SEED)
    random.seed(SEED)

    placed = {}
    real_make_bridge = scenario_module.make_bridge

    def make_bridge_with_placement(scenario, device):
        bridge = real_make_bridge(scenario, device)
        pos, yaw = pick_spread_placements(scenario.world_state, scenario.n_agents)
        pos_lab, yaw_lab = bridge.sigmarl_to_lab_pose(pos, yaw)
        bridge._x[:, 0:2] = pos_lab
        bridge._x[:, 2] = yaw_lab
        bridge._x[:, 3] = 0.0  # a hand-placed vehicle is at rest
        bridge._x[:, 4] = 0.0
        placed["pos"] = pos
        placed["yaw"] = yaw
        return bridge

    with mock.patch.object(scenario_module, "make_bridge", make_bridge_with_placement):
        env = trainer._setup_environment()
    return env, placed


class TestLabManualReset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_auto_confirm = os.environ.get("SIGMARL_LAB_AUTO_CONFIRM")
        os.environ["SIGMARL_LAB_AUTO_CONFIRM"] = "1"

    @classmethod
    def tearDownClass(cls):
        if cls._previous_auto_confirm is None:
            os.environ.pop("SIGMARL_LAB_AUTO_CONFIRM", None)
        else:
            os.environ["SIGMARL_LAB_AUTO_CONFIRM"] = cls._previous_auto_confirm

    def test_world_state_class_depends_on_mode(self):
        """Manual mode swaps the world state; the default must not."""
        env, _ = build_manual_env(make_parameters(manual_reset=True))
        self.assertIsInstance(env.scenario.world_state, WorldStateLab)

        trainer = MAPPOCAVs(make_parameters(manual_reset=False))
        default_env = trainer._setup_environment()
        self.assertIsInstance(default_env.scenario.world_state, WorldStateRTSimulation)
        self.assertNotIsInstance(default_env.scenario.world_state, WorldStateLab)

    def test_reset_adopts_measured_poses(self):
        """The placed poses must survive reset verbatim -- nothing is sampled."""
        parameters = make_parameters(manual_reset=True)
        env, placed = build_manual_env(parameters)
        env.reset()

        scenario = env.scenario
        for i_agent in range(N_AGENTS):
            agent = scenario.world.agents[i_agent]
            torch.testing.assert_close(
                agent.state.pos[0], placed["pos"][i_agent], atol=1e-5, rtol=0
            )
            # Yaw is wrapped by the measurement path; compare modulo 2*pi.
            delta = float(agent.state.rot[0, 0] - placed["yaw"][i_agent])
            delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
            self.assertLess(abs(delta), 1e-5)
            # is_zero_initial_speed is False, yet a hand-placed vehicle is at
            # rest: the measured speed wins because nothing is sampled.
            self.assertAlmostEqual(float(agent.state.speed[0, 0]), 0.0, places=5)

    def test_assigned_path_matches_placement(self):
        """Each agent's reference path must pass close to where it was placed."""
        env, placed = build_manual_env(make_parameters(manual_reset=True))
        env.reset()

        world_state = env.scenario.world_state
        ref_paths = world_state.ref_paths_map_related.long_term_all
        for i_agent in range(N_AGENTS):
            path_id = int(world_state.ref_paths_agent_related.path_id[0, i_agent])
            center_line = ref_paths[path_id]["center_line"]
            lateral = float(
                torch.min(torch.norm(center_line - placed["pos"][i_agent], dim=-1))
            )
            self.assertLess(
                lateral,
                0.05,
                f"agent {i_agent} was assigned path {path_id}, which passes "
                f"{lateral * 100:.1f} cm from its placed pose",
            )

    def test_no_spurious_collision_after_reset(self):
        """A valid placement must not report collisions on the first step.

        Guards the two failure modes that make manual mode loop forever: a
        stale reference path, and a stale state-buffer row feeding the
        swept-boundary check.
        """
        env, _ = build_manual_env(make_parameters(manual_reset=True))
        td = env.reset()

        collisions = env.scenario.world_state.collisions
        self.assertFalse(bool(collisions.with_agents.any()))
        self.assertFalse(bool(collisions.with_lanelets.any()))

        # One mild forward step: nothing should collide.
        td.set(
            ("agents", "action"),
            torch.stack(
                [
                    torch.full((N_AGENTS, 1), 0.3),  # speed
                    torch.zeros(N_AGENTS, 1),  # steering
                ],
                dim=-1,
            ).reshape(1, N_AGENTS, 2),
        )
        td = env.step(td)
        self.assertFalse(
            bool(td.get(("next", "agents", "info", "is_collision_with_lanelets")).any())
        )
        self.assertFalse(
            bool(td.get(("next", "agents", "info", "is_collision_with_agents")).any())
        )

    def test_single_agent_reset_reanchors_every_agent(self):
        """A single-agent reset must still re-derive paths for the others.

        Mirrors what done() does mid-episode, including the 0-dim tensor
        indices it passes. The operator may move any vehicle at that prompt, so
        every agent's path must be re-derived from its measured pose.
        """
        env, _ = build_manual_env(make_parameters(manual_reset=True))
        env.reset()
        scenario = env.scenario

        # The "operator" moves a vehicle that is NOT the one being reset onto a
        # different reference path. Use a 5th spread-out spot so it does not
        # land on top of one of the other three.
        moved = 1
        reset_agent = 2
        pos, yaw = pick_spread_placements(scenario.world_state, N_AGENTS + 1)
        new_pos, new_yaw = pos[N_AGENTS].clone(), yaw[N_AGENTS].clone()
        pos_lab, yaw_lab = scenario.bridge.sigmarl_to_lab_pose(
            new_pos.unsqueeze(0), new_yaw.reshape(1)
        )
        scenario.bridge._x[moved, 0:2] = pos_lab[0]
        scenario.bridge._x[moved, 2] = yaw_lab[0]

        scenario.reset_world_at(
            env_index=torch.tensor(0), agent_index=torch.tensor(reset_agent)
        )

        # The moved (non-reset) agent adopted its new pose ...
        torch.testing.assert_close(
            scenario.world.agents[moved].state.pos[0], new_pos, atol=1e-5, rtol=0
        )
        # ... and got a reference path that actually passes through it.
        ref_paths = scenario.world_state.ref_paths_map_related.long_term_all
        path_id = int(scenario.world_state.ref_paths_agent_related.path_id[0, moved])
        lateral = float(
            torch.min(torch.norm(ref_paths[path_id]["center_line"] - new_pos, dim=-1))
        )
        self.assertLess(lateral, 0.05)

        # The whole state-buffer row was refreshed, not just the reset agent's,
        # so the swept-boundary check and progress reward see no phantom jump.
        latest = scenario.state_buffer.get_latest(n=1)
        for i_agent in range(N_AGENTS):
            torch.testing.assert_close(
                latest[0, i_agent, 0:2],
                scenario.world.agents[i_agent].state.pos[0],
                atol=1e-5,
                rtol=0,
            )

    def test_invalid_placement_is_rejected(self):
        """Overlapping vehicles must be rejected rather than silently accepted."""
        env, _ = build_manual_env(make_parameters(manual_reset=True))
        scenario = env.scenario
        # Park every vehicle on top of vehicle 0.
        scenario.bridge._x[:, 0:2] = scenario.bridge._x[0, 0:2]

        with self.assertRaises(LabStopped):
            scenario.reset_world_at()

    def test_pose_imposing_settings_are_guarded(self):
        """Settings that would overwrite the hand placement must fail fast."""
        parameters = make_parameters(manual_reset=True)
        parameters.is_challenging_initial_state_buffer = True
        with self.assertRaises(AssertionError):
            build_manual_env(parameters)


if __name__ == "__main__":
    unittest.main()
