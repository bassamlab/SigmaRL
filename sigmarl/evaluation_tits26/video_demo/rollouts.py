# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Rollouts for the video demo (video.py) that the evaluation outputs do
not store: Stage 3 policies before and after filter-in-the-loop fine-tuning,
deployed without a filter.

The Stage 3 evaluation (stage2_finetune_run.py eval) keeps only summary
statistics. This script reruns the same evaluation for one checkpoint
(--iter) of one fine-tuning run (--run) and saves the rollout's out_td. It
works on a copy of the run's configuration and checkpoint in --out, so the
run folder of the paper stays untouched (eval appends to a JSONL file in
the run folder).

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.video_demo.rollouts \\
        --run outputs/tits26/stage2_finetune/round_location0__stage1_ttc_seed_1__iter100/ind_location4/ours/seed1 \\
        --iter 0 100 --seed 201
"""

import argparse
import os
import shutil
import subprocess
import sys

import torch

OUT = "outputs/tits26/video_demo/stage2"
CHECKPOINT_DIR = "checkpoints_cross_iter"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="Stage 3 fine-tuning run folder.")
    parser.add_argument("--iter", type=int, nargs="+", default=[0, 100])
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--max_steps", type=int, default=601)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_dir(args) -> str:
    """The copy of the run (configuration and checkpoints) in --out."""
    parts = os.path.normpath(args.run).split(os.sep)
    return os.path.join(args.out, "__".join(parts[-4:]))


def prepare(args) -> str:
    target = run_dir(args)
    os.makedirs(os.path.join(target, CHECKPOINT_DIR), exist_ok=True)
    shutil.copy(os.path.join(args.run, "run_config.json"), target)
    for it in args.iter:
        name = f"policy_iter{it}.pth"
        shutil.copy(os.path.join(args.run, CHECKPOINT_DIR, name),
                    os.path.join(target, CHECKPOINT_DIR, name))  # fmt: skip
    return target


def worker(args):
    """Evaluate one checkpoint in this process and save its out_td."""
    import sigmarl.evaluation_tits26.stage2_finetune_run as stage2

    target = run_dir(args)
    summarize = stage2.summarize_rollout

    def summarize_and_save(out_td, parameters):
        path = os.path.join(
            target, f"out_td_iter{args.worker}_nofilter_seed{args.seed}.td"
        )
        torch.save(out_td.cpu(), path)
        print(f"[INFO] Saved {path}")
        return summarize(out_td, parameters)

    stage2.summarize_rollout = summarize_and_save
    stage2.evaluate(
        argparse.Namespace(
            run=target,
            iter=str(args.worker),
            filter=False,
            eval_method=None,
            seed=args.seed,
            num_vmas_envs=1,
            max_steps=args.max_steps,
            log_backup_value=False,
        )
    )


def main():
    args = parse_args()
    if args.worker is not None:
        worker(args)
        return
    prepare(args)
    # One checkpoint per process (see stage2_finetune_run.py eval)
    for index, it in enumerate(args.iter):
        print(f"[{index + 1}/{len(args.iter)}] iteration {it}")
        subprocess.run(
            [sys.executable, "-m", "sigmarl.evaluation_tits26.video_demo.rollouts",
             "--run", args.run, "--iter", *map(str, args.iter), "--seed", str(args.seed),
             "--max_steps", str(args.max_steps), "--out", args.out, "--worker", str(it)],
            check=True,
        )  # fmt: skip


if __name__ == "__main__":
    main()
