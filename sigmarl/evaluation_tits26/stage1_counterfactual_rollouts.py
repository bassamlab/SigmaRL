# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 1 counterfactual rollouts: the same traffic with and without the
BH-CBF filter, branched from identical joint states.

One deployment runs as in Stage 1 (stage1_zero_shot.py: TTC-reward policy of
iteration 100, fully decentralized BH-CBF filter, spawns at rest, exact backup
shield and standstill escape off) in env 0 of a batched VMAS environment
(the master, always filtered). The other envs are branches: every `--every`
steps, one branch env copies the master's joint state (poses, speeds,
steering angles, routes) right after the dynamics of a step, before the
observations are computed, and from then on all of its vehicles apply the
nominal (unfiltered) policy actions, for `--branches * --every` steps, until
it copies the master again. A collision in a branch that the master avoids
is a scene of the paper's counterfactual figure
(paper_fig_stage1_counterfactual.py).

`--validate` filters the branches as well: they must then reproduce the
master exactly (up to respawns), which checks the state copy.

Output: <out>/<checkpoint label>/<map>/seed<k>.npz with, per step and env,
the poses [T, E, n, 3], speeds [T, E, n], inter-vehicle and lane collision
flags [T, E, n], and the master's filter interventions (|applied - nominal|
[T, n, 2]), the nominal actions of all envs [T, E, n, 2]; `branch_start` [T, E] marks the steps at which an env copied the
master.

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.stage1_counterfactual_rollouts \\
        --checkpoint checkpoints/tits26/stage1_pretrain/round_location0/stage1_ttc_seed_2 \\
        --map round_location0 --seeds 301 302 [--out outputs/tits26/stage1_counterfactual]
"""

import argparse
import json
import os
import runpy
import sys

import numpy as np
import torch

from sigmarl.evaluation_tits26.settings import N_AGENTS
from sigmarl.evaluation_tits26.stage1_zero_shot import checkpoint_label

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TUNED = os.path.join(REPO_ROOT, "sigmarl/cbf_baselines/tuned_defaults.json")
POLICY_ITER = 100


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--every", type=int, default=10, help="Steps between branches.")
    parser.add_argument("--branches", type=int, default=6, help="Branch envs.")
    parser.add_argument("--out", default="outputs/tits26/stage1_counterfactual")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


class _Built(Exception):
    """Carries the objects that mappo_cavs built out of main_testing_cbf_filter.py."""


def build(checkpoint: str, scenario: str, seed: int, n_envs: int, out_dir: str):
    """(env, policy, cbf controllers) as main_testing_cbf_filter.py builds them
    for the Stage 1 BH-CBF deployment, with `n_envs` envs."""
    import sigmarl.mappo_cavs as mappo_module

    tuned = json.load(open(TUNED))["shield_off/ours/dec"]
    original = mappo_module.mappo_cavs

    def patched(parameters):
        parameters.num_vmas_envs = n_envs
        raise _Built(original(parameters=parameters))

    argv = sys.argv
    sys.argv = [
        "main_testing_cbf_filter.py", "--nominal", "rl", "--path", os.path.abspath(checkpoint),
        "--scenario_type", scenario, "--n_agents", str(N_AGENTS[scenario]),
        "--max_steps", "100000", "--max_group_size", "1", "--seed", str(seed),
        "--is_zero_initial_speed", "--no-is_visu_result", "--no-is_save_simulation_video",
        "--no-is_exact_backup_shield", "--no-is_standstill_escape", "--output_dir", out_dir,
        "--is_load_final_model", "--cbf_lambda", str(tuned["cbf_lambda"]),
        "--lane_cbf_baseline", "none", "--veh_veh_cbf_baseline", "none",
        "--is_backup_horizon_cbf", "--no-is_two_step_rows",
    ]  # fmt: skip
    mappo_module.mappo_cavs = patched
    try:
        runpy.run_path(
            os.path.join(REPO_ROOT, "main_testing_cbf_filter.py"), run_name="__main__"
        )
    except _Built as built:
        env, decision_making_module, _, _, cbf_controllers, _ = built.args[0]
    finally:
        mappo_module.mappo_cavs = original
        sys.argv = argv
    policy = decision_making_module.policy
    policy.load_state_dict(
        torch.load(
            os.path.join(checkpoint, "checkpoints_cross_iter", f"policy_iter{POLICY_ITER}.pth"),
            weights_only=True,
        )
    )  # fmt: skip
    return env, policy, cbf_controllers


class Brancher:
    """Copies the master's (env 0) joint state into branch envs inside the
    scenario's post_step, i.e., after the dynamics and before the shared
    transition state and the observations are computed."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.pending = []
        original = scenario.post_step

        def post_step():
            for env_index in self.pending:
                self.copy(env_index)
            self.pending = []
            original()

        scenario.post_step = post_step

    def copy(self, env_index: int):
        scenario = self.scenario
        path_ids = scenario.world_state.ref_paths_agent_related.path_id
        for i, agent in enumerate(scenario.world.agents):
            scenario.place_agent(
                env_index, i, agent.state.pos[0].clone(), float(agent.state.rot[0, 0]),
                float(agent.state.speed[0, 0]), float(agent.state.steering[0, 0]),
                path_id=int(path_ids[0, i]),
            )  # fmt: skip
            scenario.sync_agent_history(env_index, i)


def run(env, policy, cbf_controllers, args) -> dict:
    """The master and the branches for args.steps steps."""
    from tqdm import tqdm
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    from sigmarl.helper_training import cbf_constrained_policy_main

    scenario = env.base_env.scenario_name
    brancher = Brancher(scenario)
    n_envs = args.branches + 1
    # The branches are not filtered (unless validating): no controllers
    controllers = list(cbf_controllers)
    if not args.validate:
        controllers = [controllers[0]] + [[] for _ in range(args.branches)]
    record = {k: [] for k in ("pose", "speed", "veh_col", "lane_col", "intervention", "nominal")}
    branch_start = np.zeros((args.steps, n_envs), dtype=bool)
    with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        for step in tqdm(range(args.steps), desc="steps", leave=False):
            cbf_constrained_policy_main(td, policy, controllers)
            nominal = policy._rl_action_cache
            action = td.get(("agents", "action"))
            record["intervention"].append((action[0] - nominal[0]).abs().numpy())
            record["nominal"].append(nominal.numpy().copy())
            if not args.validate:
                action[1:] = nominal[1:]
            # The branch env whose turn it is copies the master after this step
            if step % args.every == 0:
                branch = 1 + (step // args.every) % args.branches
                brancher.pending = [branch]
                branch_start[step, branch] = True
            td = env.step(td)
            info = td["next", "agents", "info"]
            record["pose"].append(
                torch.cat([info["pos"], info["rot"]], dim=-1).numpy()
            )
            record["speed"].append(info["vel"].norm(dim=-1).numpy())
            record["veh_col"].append(info["is_collision_with_agents"][..., 0].numpy())
            record["lane_col"].append(info["is_collision_with_lanelets"][..., 0].numpy())
            td = env._step_mdp(td)
            td = env.maybe_reset(td)
    result = {k: np.stack(v) for k, v in record.items()}
    result["branch_start"] = branch_start
    return result


def main():
    args = parse_args()
    label = checkpoint_label(args.checkpoint) + f"__iter{POLICY_ITER}"
    out_dir = os.path.join(args.out, label, args.map)
    os.makedirs(out_dir, exist_ok=True)
    for seed in args.seeds:
        path = os.path.join(out_dir, f"seed{seed}" + ("_validate" if args.validate else "") + ".npz")
        if os.path.exists(path):
            continue
        env, policy, cbf_controllers = build(
            args.checkpoint, args.map, seed, args.branches + 1, os.path.join(out_dir, "_log")
        )
        result = run(env, policy, cbf_controllers, args)
        np.savez_compressed(path, every=args.every, branches=args.branches, **result)
        print(f"[INFO] Saved {path}", flush=True)


if __name__ == "__main__":
    main()
