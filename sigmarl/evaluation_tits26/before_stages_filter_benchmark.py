# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run the filter benchmark before Stages 1-3: method x mode x block x seed
rollouts of main_testing_cbf_filter.py (CLF nominal, full-scale
round_location0). METHODS defines the safety filters of all stages.

Every method uses the rectangle_sat pipeline and swaps one or both row types
(see sigmarl/cbf_baselines). Blocks: "shield_off" (exact backup shield and
standstill escape off for all methods) and "shield_on" (both on for all).
Runs whose evaluation summary exists are skipped, so the script resumes.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.before_stages_filter_benchmark
    ... --light                       # 1 seed, 201 steps, for debugging
    ... --methods ours b3_disk --modes cen --blocks shield_off --seeds 1 2
    ... --methods b1_lse_sat --env SIGMARL_B1_LSE_KAPPA=40 --tag kappa40
    ... --tuned <out_root>/tuned.json   # per (block, method, mode) lambda and
                                        # environment from before_stages_filter_tuning.py
                                        # (default: the tracked tuned values,
                                        # sigmarl/cbf_baselines/tuned_defaults.json)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# Method -> (lane-boundary rows, inter-vehicle rows); "none" keeps ours
METHODS = {
    "ours": ("none", "none"),  # BH-TTCBF, the proposed method (plus METHOD_ARGS)
    "ours_onestep": ("none", "none"),  # our barriers, one-step condition only
    "b6_backup_cbf": ("none", "none"),  # plus METHOD_ARGS
    "b1_lse_sat": ("b1_lse_sat", "b1_lse_sat"),
    "b2_frenet_point": ("b2_frenet_point", "none"),
    "b2_frenet_heading": ("b2_frenet_heading", "none"),
    "b3_wang": ("none", "b3_wang"),
    "b3_disk": ("none", "b3_disk"),
    "b4_collision_cone": ("none", "b4_collision_cone"),
    "b5_convex_containment": ("b5_convex_containment", "none"),
    "t2_dual_distance": ("t2_dual_distance", "t2_dual_distance"),
    # Variants of the backup-horizon rows (sigmarl/cbf_baselines/bh_variants.py)
    "bh_half_split": ("none", "none"),
    "b1_bh": ("none", "none"),
    "b3_bh": ("none", "none"),
    "bh_capsule": ("none", "none"),
    "mps": ("none", "none"),
    "unfiltered": None,  # --no-is_cbf_filter; centralized, shield_off block only
}
# Extra arguments of a method
METHOD_ARGS = {
    # Explicit flags, so that the method names do not depend on the defaults
    "ours": ["--is_backup_horizon_cbf", "--no-is_two_step_rows"],
    "ours_onestep": ["--no-is_backup_horizon_cbf"],
    # As published (Chen et al.): flow-point rows only, no two-step rows
    "b6_backup_cbf": [
        "--condition_cbf_baseline",
        "b6_backup_cbf",
        "--no-is_two_step_rows",
    ],
    **{
        name: ["--condition_cbf_baseline", name, "--no-is_two_step_rows"]
        for name in ("bh_half_split", "b1_bh", "b3_bh", "bh_capsule", "mps")
    },
}
MODES = {"cen": "none", "dec": "1"}
BLOCKS = {
    "shield_off": ["--no-is_exact_backup_shield", "--no-is_standstill_escape"],
    "shield_on": ["--is_exact_backup_shield", "--is_standstill_escape"],
}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max_steps", type=int, default=601)
    parser.add_argument("--n_agents", type=int, default=10)
    parser.add_argument("--scenario_type", default="round_location0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out_root",
        default=os.path.join(
            REPO_ROOT, "outputs", "tits26", "before_stages_filter_benchmark"
        ),
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Debug run: seed 1, 201 steps, outputs under <out_root>/light.",
    )
    parser.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Extra environment variables KEY=VALUE, e.g., for a kappa sweep.",
    )
    parser.add_argument("--tag", default="", help="Suffix of the method folders.")
    parser.add_argument(
        "--cbf_lambda",
        type=float,
        default=None,
        help="CBF decay rate of all runs (default: the Parameters default).",
    )
    parser.add_argument(
        "--tuned",
        default=os.path.join(
            REPO_ROOT, "sigmarl", "cbf_baselines", "tuned_defaults.json"
        ),
        help=(
            "JSON from before_stages_filter_tuning.py with the selected cbf_lambda and "
            "environment per block, method, and mode; overrides --cbf_lambda. "
            "Pass '' to use --cbf_lambda for all methods."
        ),
    )
    args = parser.parse_args()
    if args.light:
        args.seeds = [1]
        args.max_steps = 201
        args.out_root = os.path.join(args.out_root, "light")
    return args


def build_jobs(args, base_env: dict) -> list:
    """(output folder, command, environment) of every run without a summary."""
    tuned = json.load(open(args.tuned)) if args.tuned else {}
    jobs = []
    for block in args.blocks:
        for method in args.methods:
            for mode in args.modes:
                # The unfiltered reference has no shield; run it once
                if METHODS[method] is None and (mode != "cen" or block != "shield_off"):
                    continue
                for seed in args.seeds:
                    name = method + (f"_{args.tag}" if args.tag else "")
                    out_dir = os.path.join(
                        args.out_root, block, name, mode, f"seed{seed}"
                    )
                    if glob.glob(os.path.join(out_dir, "evaluation_summary_*.json")):
                        continue
                    command = [
                        sys.executable,
                        os.path.join(REPO_ROOT, "main_testing_cbf_filter.py"),
                        "--nominal",
                        "clf",
                        "--scenario_type",
                        args.scenario_type,
                        "--n_agents",
                        str(args.n_agents),
                        "--max_steps",
                        str(args.max_steps),
                        "--max_group_size",
                        MODES[mode],
                        "--seed",
                        str(seed),
                        "--no-is_visu_result",
                        "--no-is_save_simulation_video",
                        "--output_dir",
                        out_dir,
                        *BLOCKS[block],
                    ]
                    filter_args, filter_env = method_args(
                        method, mode, block, tuned, args.cbf_lambda
                    )
                    env = dict(base_env)
                    env.update(filter_env)
                    jobs.append((out_dir, command + filter_args, env))
    return jobs


def method_args(method: str, mode: str, block: str, tuned: dict, cbf_lambda=None):
    """Arguments of main_testing_cbf_filter.py that select a method (with its
    tuned lambda and own parameters, if tuned has them for the block and
    mode; else cbf_lambda, if given) and its environment variables."""
    config = tuned.get(f"{block}/{method}/{mode}", {})
    cbf_lambda = config.get("cbf_lambda", cbf_lambda)
    command = [] if cbf_lambda is None else ["--cbf_lambda", str(cbf_lambda)]
    if METHODS[method] is None:
        command.append("--no-is_cbf_filter")
    else:
        lane, veh = METHODS[method]
        command += [
            "--lane_cbf_baseline",
            lane,
            "--veh_veh_cbf_baseline",
            veh,
            *METHOD_ARGS.get(method, []),
        ]
    return command, dict(config.get("env", {}))


def run_job(out_dir: str, command: list, env: dict) -> tuple:
    """Run one rollout; its stdout and stderr go to <out_dir>/stdout.txt."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stdout.txt"), "w") as log:
        code = subprocess.run(
            command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    return out_dir, code


def base_env(items=()) -> dict:
    """Environment of the rollouts (one thread each) plus KEY=VALUE items."""
    env = dict(os.environ)
    env.update(
        SIGMARL_AGENT_PROFILE="full_scale",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    for item in items:
        key, value = item.split("=", 1)
        env[key] = value
    return env


def main():
    args = parse_args()
    run_jobs(build_jobs(args, base_env(args.env)), args.workers)


def run_jobs(jobs: list, workers: int) -> list:
    """Run (output folder, command, environment) jobs in parallel with a
    progress bar; returns the failed output folders."""
    print(f"[INFO] {len(jobs)} runs to do, {workers} in parallel.")
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, *job) for job in jobs]
        with tqdm(total=len(futures), unit="run") as bar:
            for future in as_completed(futures):
                out_dir, code = future.result()
                if code != 0:
                    failed.append(out_dir)
                    tqdm.write(f"[FAILED] {out_dir} (exit {code})")
                bar.update(1)
    print(f"[INFO] Done; {len(failed)} failed.")
    for out_dir in failed:
        print(f"  {out_dir}/stdout.txt")
    return failed


if __name__ == "__main__":
    main()
