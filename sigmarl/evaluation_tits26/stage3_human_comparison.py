# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 3 of the T-ITS evaluation: comparison with the human drivers of rounD
location 0.

For every policy and filter, runs sigmarl/evaluation_human_comparison/run.py
(controller marl_filter, homogeneous episodes: every car and van of a recorded
window without other road users is driven by the policy with the filter) into
<out_root>/homogeneous/<label>, then analyze.py over all labels, which writes
the comparison with the humans to <out_root>/homogeneous/ (summary.md,
vehicles.csv, figures). Finished episodes are skipped, so the script resumes.

Each controlled vehicle has the parameters of the recorded vehicle it replaces
(--vehicles recorded, default): its recorded footprint, axle distances scaled
with its length, and the acceleration limits of its class (cars [-6, 3],
vans [-5, 2.5] m/s^2; episodes.CLASS_ACCELERATION). --vehicles nominal gives
every vehicle the 4.6 m x 2.0 m car of Stages 1 and 2. Cars and vans longer than
--max_vehicle_length (default 6.5 m) count as heavy vehicles, so the windows end
where they appear (humans in such long vans often cut the lane boundaries).

Labels: <training map>__<policy folder>__<method>_<mode>; for a fine-tuning
checkpoint (--iter, Stage 2 runs):
<source label>__ft-<target map>-<method>-seed<k>__iter<N>__<method>_<mode>.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.stage3_human_comparison \\
        --policies checkpoints/tits26/stage1_pretrain/*/stage1_bh_seed_* \\
        [--methods ours] [--mode dec] [--workers 10] [--out_root outputs/tits26/stage3_human] \\
        [--entry_check certified] [--vehicles recorded] [--max_vehicle_length 6.5]
    ... --policies outputs/tits26/stage2_finetune/<run> ... --iter 100   # fine-tuned
"""

import argparse
import json
import os
import subprocess
import sys

from sigmarl.evaluation_tits26.stage1_zero_shot import checkpoint_label
from sigmarl.evaluation_tits26.before_stages_filter_benchmark import REPO_ROOT, base_env

CHECKPOINT_DIR = "checkpoints_cross_iter"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--policies",
        nargs="+",
        required=True,
        help="Pre-trained checkpoint folders, or Stage 2 run folders with --iter.",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Fine-tuning iteration of the Stage 2 runs (policy_iter<k>.pth).",
    )
    parser.add_argument("--methods", nargs="+", default=["ours"])
    parser.add_argument("--mode", choices=("dec", "cen"), default="dec")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out_root", default="outputs/tits26/stage3_human")
    parser.add_argument(
        "--entry_check",
        choices=("certified", "distance"),
        default="certified",
        help="Entry of the controlled vehicles (run.py --entry_check): certified "
        "joint state, or the earlier distance gate.",
    )
    parser.add_argument(
        "--vehicles",
        choices=("recorded", "nominal"),
        default="recorded",
        help="Parameters of the controlled vehicles (run.py --vehicles): those of "
        "the recorded vehicles, or the nominal car of Stages 1 and 2.",
    )
    parser.add_argument(
        "--max_vehicle_length",
        type=float,
        default=6.5,
        help="Cars and vans longer than this [m] count as heavy vehicles and end the "
        "homogeneous windows (run.py --max_vehicle_length; a negative value: no limit).",
    )
    parser.add_argument(
        "--analyze_only", action="store_true", help="Only run analyze.py."
    )
    return parser.parse_args()


def policy_source(policy: str, iteration) -> tuple:
    """(checkpoint folder with the parameters, policy weights or None for its
    final model, label prefix)."""
    if iteration is None:
        return policy, None, checkpoint_label(policy)
    with open(os.path.join(policy, "run_config.json")) as file:
        run_config = json.load(file)
    model_file = os.path.join(policy, CHECKPOINT_DIR, f"policy_iter{iteration}.pth")
    # Run folder <source label>/<target map>/<method>/seed<k>
    source_label, target, method, seed = os.path.normpath(policy).split(os.sep)[-4:]
    label = f"{source_label}__ft-{target}-{method}-{seed}__iter{iteration}"
    return run_config["source"], model_file, label


def main():
    args = parse_args()
    env = base_env()
    labels = []
    for policy in args.policies:
        source, model_file, prefix = policy_source(policy, args.iter)
        for method in args.methods:
            label = f"{prefix}__{method}_{args.mode}"
            labels.append(label)
            if args.analyze_only:
                continue
            command = [
                sys.executable,
                "-m",
                "sigmarl.evaluation_human_comparison.run",
                "--controller",
                "marl_filter",
                "--model_path",
                source,
                "--method",
                method,
                "--filter_mode",
                args.mode,
                "--label",
                label,
                "--out_dir",
                args.out_root,
                "--workers",
                str(args.workers),
                "--entry_check",
                args.entry_check,
                "--vehicles",
                args.vehicles,
            ]
            if model_file is not None:
                command += ["--model_file", model_file]
            if args.max_vehicle_length >= 0.0:
                command += ["--max_vehicle_length", str(args.max_vehicle_length)]
            print(f"[INFO] {label}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sigmarl.evaluation_human_comparison.analyze",
            "--controllers",
            ",".join(labels),
            "--out_dir",
            args.out_root,
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
