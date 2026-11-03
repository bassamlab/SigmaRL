# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""BH-CBF braking tails for the video demo (video.py).

The evaluation stores the vehicles' poses, but not the backup that BH-CBF
predicts from them. This script reruns a deployment with the BH-CBF filter
exactly as the evaluation ran it and, at every step of a time span, rolls out
the backup (route-following emergency braking) of every vehicle from the joint
state that the filter reads, until standstill (as paper_fig_headline.py does
for one step). A state is captured only if its positions match those of the
stored rollout, so the tails belong to the replayed poses.

Three sources (subcommands), each writing <out>/<name>.npz with the steps,
the tails [S, H, n, 3] (padded with the standstill pose), and for Stage 4 the
vehicle keys:

- ``deployment --run <run folder> --steps a b``: a Stage 2 (zero-shot) deployment
  (main_testing_cbf_filter.py with its command_testing.txt); steps index the
  run's out_td (state after step t);
- ``counterfactual --npz <rollout> --steps a b``: the filtered master (env 0)
  of a counterfactual rollout (stage1_counterfactual_rollouts.py), rerun with
  its branches; steps index the npz;
- ``humans --episode <name> --seed <policy seed>``: a Stage 4 episode
  (evaluation_human_comparison/run.py as stage3_human_comparison.py runs it);
  tails of the active vehicles at every step, keyed by time.

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.video_demo.tails deployment \\
        --run outputs/tits26/stage1_filters_ttc_iter100/<label>/<map>/ours/dec/seed202 --steps 0 250
"""

import argparse
import glob
import os
import pickle
import runpy
import shlex
import shutil
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = "outputs/tits26/video_demo/tails"
COUNTERFACTUAL = "outputs/tits26/stage1_counterfactual"
STAGE2 = "outputs/tits26/stage2_finetune"
STAGE3 = "outputs/tits26/stage3_human_ft_recorded_maxlen6.5/natural"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s1 = sub.add_parser("deployment")
    s1.add_argument("--run", required=True)
    s1.add_argument("--steps", type=int, nargs=2, required=True)
    cf = sub.add_parser("counterfactual")
    cf.add_argument(
        "--npz", required=True, help="Relative to the counterfactual outputs."
    )
    cf.add_argument("--steps", type=int, nargs=2, required=True)
    s3 = sub.add_parser("humans")
    s3.add_argument("--episode", required=True)
    s3.add_argument("--seed", type=int, required=True, help="Source policy seed.")
    for command in (s1, cf, s3):
        command.add_argument(
            "--name", required=True, help="Output file name (without .npz)."
        )
        command.add_argument("--out", default=OUT)
    return parser.parse_args()


def tails_at(ctrl, X: np.ndarray) -> np.ndarray:
    """Poses [H, n, 3]: the joint state X [n, 5], then the backup of every
    vehicle from X until all stand still."""
    horizon = ctrl.backup_horizon
    n = X.shape[0]
    routes = [ctrl._shield_route(i) for i in range(n)]
    tails = horizon.rollout(
        X,
        horizon.backup_controls(X, routes),
        routes,
        np.zeros(n, dtype=bool),
        horizon.horizon(X[:, 3]),
    )
    return np.concatenate([X[None, :, 0:3], tails])


def joint_state(result) -> np.ndarray:
    """X [n, 5] from the output of CBFQP._collect_agent_states."""
    return np.array([state.tolist() for state in result[1]], dtype=np.float64)


def pad(tails: list) -> np.ndarray:
    """Stack tails of different lengths, repeating each one's standstill pose."""
    length = max(len(t) for t in tails)
    return np.stack(
        [np.concatenate([t, np.repeat(t[-1:], length - len(t), 0)]) for t in tails]
    )


class TargetCapture:
    """Captures the tails of env 0 whenever a filter reads a joint state whose
    positions match a wanted step of a stored rollout (targets [T, n, 2])."""

    def __init__(self, targets: np.ndarray, steps: range):
        self.targets = targets
        self.wanted = np.array([s for s in steps if s < len(targets)])
        self.tails = {}

    def install(self):
        from sigmarl.cbf_qp_proxsuite import CBFQP

        collect = CBFQP._collect_agent_states
        capture = self

        def collect_and_capture(ctrl, tensordict, n_agents):
            result = collect(ctrl, tensordict, n_agents)
            if ctrl.env_idx == 0 and len(capture.tails) < len(capture.wanted):
                diff = np.abs(capture.targets[capture.wanted] - result[3]).max(
                    axis=(1, 2)
                )
                for index in np.nonzero(diff < 1e-4)[0]:
                    step = int(capture.wanted[index])
                    if step not in capture.tails:
                        capture.tails[step] = tails_at(ctrl, joint_state(result))
            return result

        CBFQP._collect_agent_states = collect_and_capture

    def save(self, path: str):
        steps = sorted(self.tails)
        missing = sorted(set(self.wanted.tolist()) - set(steps))
        if missing:
            print(
                f"[WARNING] No filter read the stored state at {len(missing)} steps: {missing}"
            )
        np.savez_compressed(
            path, steps=np.array(steps), tails=pad([self.tails[s] for s in steps])
        )
        print(f"[INFO] Saved {path} ({len(steps)} steps)")


def deployment(args):
    """Rerun a Stage 2 deployment as its command_testing.txt and capture the tails."""
    (path,) = glob.glob(os.path.join(args.run, "out_td_*.td"))
    td = torch.load(path, weights_only=False)
    targets = td["next", "agents", "info", "pos"][0].numpy()
    start, end = args.steps
    capture = TargetCapture(targets, range(start, end + 1))
    capture.install()
    with open(os.path.join(args.run, "command_testing.txt")) as file:
        tokens = shlex.split(file.read())
    argv = tokens[
        tokens.index(
            next(t for t in tokens if t.endswith("main_testing_cbf_filter.py"))
        )
        + 1 :
    ]
    replace = {
        "--max_steps": str(end + 3),
        "--output_dir": os.path.join(args.out, "_sim", args.name),
    }
    for flag, value in replace.items():
        argv[argv.index(flag) + 1] = value
    sys.argv = ["main_testing_cbf_filter.py"] + argv
    runpy.run_path(
        os.path.join(REPO_ROOT, "main_testing_cbf_filter.py"), run_name="__main__"
    )
    capture.save(os.path.join(args.out, f"{args.name}.npz"))


def counterfactual(args):
    """Rerun a counterfactual rollout (master and branches) up to the span's
    end and capture the master's tails."""
    from sigmarl.evaluation_tits26 import stage1_counterfactual_rollouts as rollouts

    path = os.path.join(COUNTERFACTUAL, args.npz)
    data = np.load(path)
    label, scenario, name = os.path.normpath(args.npz).split(os.sep)
    source_map, folder = label.split("__")[:2]
    checkpoint = os.path.join("checkpoints/tits26/stage1_pretrain", source_map, folder)
    seed = int(name.removeprefix("seed").removesuffix(".npz"))
    start, end = args.steps
    capture = TargetCapture(data["pose"][:, 0, :, :2], range(start, end + 1))
    capture.install()
    n_envs = int(data["branches"]) + 1
    env, policy, controllers = rollouts.build(
        checkpoint, scenario, seed, n_envs, os.path.join(args.out, "_sim", args.name)
    )
    rollouts.run(env, policy, controllers, argparse.Namespace(
        steps=end + 2, every=int(data["every"]), branches=int(data["branches"]), validate=False))  # fmt: skip
    capture.save(os.path.join(args.out, f"{args.name}.npz"))


def humans(args):
    """Rerun one Stage 4 episode and capture the tails of the active vehicles at every step."""
    from sigmarl.evaluation_human_comparison import run as human
    from sigmarl.evaluation_human_comparison.episodes import (
        homogeneous_episodes,
        load_location0,
        max_concurrent_slots,
    )
    from sigmarl.evaluation_tits26.stage3_human_comparison import policy_source
    from sigmarl.parse_lanelet2 import ParseLanelet2
    from sigmarl.round_tracks import RouteMatcher

    policy = os.path.join(
        STAGE2,
        f"round_location0__stage1_ttc_seed_{args.seed}__iter100/round_location0/ours/seed1",
    )
    source, model_file, _ = policy_source(policy, 100)
    # The agents per environment of the paper's run: from all episodes, as run.py computes them
    parser = ParseLanelet2("round_location0", "cpu")
    routes = RouteMatcher(parser.reference_paths).routes
    data = load_location0(
        human.DEFAULT_DATA_DIR, parser, os.path.join(STAGE3, "matched_tracks.pkl")
    )
    specs = homogeneous_episodes(data, max_vehicle_length=6.5)
    n_slots = max(max_concurrent_slots(s, data, routes, 0.1) for s in specs) + 2

    captured = []  # (time, {key: tail [H, 3]})
    control = human.ControllerRunner._control

    def control_and_capture(runner, td, controlled_slots):
        out = control(runner, td, controlled_slots)
        replay = runner.scenario.log_replay
        ctrl = runner.first_cbf_controller
        result = ctrl._collect_agent_states(td, len(runner.scenario.world.agents))
        tails = tails_at(ctrl, joint_state(result))
        keys = {
            v.slot: v.track.track_id for v in replay.controlled if v.status == "active"
        }
        captured.append(
            (float(replay.t), {key: tails[:, slot] for slot, key in keys.items()})
        )
        return out

    human.ControllerRunner._control = control_and_capture
    out_dir = os.path.join(args.out, "_sim", args.name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(STAGE3, "matched_tracks.pkl"), out_dir)
    human.main([
        "--controller", "marl_filter", "--model_path", source, "--model_file", model_file,
        "--method", "ours", "--filter_mode", "dec", "--label", "rerun", "--out_dir", out_dir,
        "--workers", "1", "--entry_check", "certified", "--vehicles", "recorded",
        "--max_vehicle_length", "6.5", "--episodes", args.episode, "--n_slots", str(n_slots),
        "--data_dir", human.DEFAULT_DATA_DIR,
    ])  # fmt: skip
    # The rerun must reproduce the paper's episode
    with open(
        os.path.join(out_dir, "homogeneous", "rerun", f"{args.episode}.pkl"), "rb"
    ) as file:
        rerun = pickle.load(file)
    paper = os.path.join(
        STAGE3, "homogeneous",
        f"round_location0__stage1_ttc_seed_{args.seed}__iter100__ft-round_location0-ours-seed1__iter100__ours_dec",
        f"{args.episode}.pkl",
    )  # fmt: skip
    with open(paper, "rb") as file:
        reference = pickle.load(file)
    deviation = max(
        np.abs(np.asarray(a["log"]["x"]) - np.asarray(b["log"]["x"])).max(initial=0.0)
        for a, b in zip(rerun["vehicles"], reference["vehicles"])
    )
    print(
        f"[INFO] Largest deviation of the rerun from the paper's episode: {deviation:.2e} m"
    )
    with open(os.path.join(args.out, f"{args.name}.pkl"), "wb") as file:
        pickle.dump(dict(captured=captured, deviation=deviation), file)
    print(
        f"[INFO] Saved {os.path.join(args.out, args.name)}.pkl ({len(captured)} steps)"
    )


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"deployment": deployment, "counterfactual": counterfactual, "humans": humans}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
