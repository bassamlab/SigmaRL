# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 1 of the T-ITS evaluation: zero-shot deployment of pre-trained MARL
policies with each safety filter.

Every checkpoint (a MAPPO policy pre-trained on one map) is deployed on every
map of --maps (its own map and the unseen ones) with every method of
sigmarl/evaluation_tits26/before_stages_filter_benchmark.py (METHODS), centralized
and fully decentralized, through main_testing_cbf_filter.py with the policy as
the nominal controller. Spawns are at rest, the exact backup shield and the
standstill escape are off, and lambda and the baselines' own parameters are
the tuned values (sigmarl/cbf_baselines/tuned_defaults.json, block
shield_off). Runs whose evaluation summary exists are skipped, so the script
resumes. Outputs: <out_root>/<checkpoint>/<deploy map>/<method>/<mode>/seed<k>,
with <checkpoint> = <training map>__<checkpoint folder name>.

``--summarize`` writes <out_root>/runs.csv (one row per run, see
before_stages_filter_summary.run_metrics) and <out_root>/summary.md and summary.csv: per method
and mode, the mean and standard error over runs on the own maps and on the
unseen maps. Collisions after an infeasible step are pooled over the runs.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.stage1_zero_shot \\
        --checkpoints checkpoints/<date>/round_location0/<run> ... \\
        --maps round_location0 ind_location4 \\
        --out_root outputs/<date>-stage1 [--workers 8]
    ... --methods ours ours_onestep --modes dec --seeds 1   # subset
    ... --methods ours --cbf_lambda 0.1 --tag lam0.1        # lambda ablation
    ... --summarize
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from tqdm import tqdm

from sigmarl.evaluation_tits26.settings import MAPS, N_AGENTS
from sigmarl.evaluation_tits26.before_stages_filter_benchmark import (
    METHODS,
    MODES,
    REPO_ROOT,
    base_env,
    method_args,
    run_jobs,
)

DEFAULT_METHODS = [
    "ours",
    "ours_onestep",
    "b1_lse_sat",
    "b2_frenet_heading",
    "b3_wang",
    "t2_dual_distance",
    "bh_capsule",
    "b6_backup_cbf",
    "bh_half_split",
    "mps",
    "unfiltered",
]
BLOCK = "shield_off"  # The exact backup shield is not part of the paper
# Metrics of the summary: (key, title, format)
SUMMARY_METRICS = [
    ("veh_col", "Inter-veh. col.", "{:.2f}"),
    ("lane_col", "Lane col.", "{:.2f}"),
    ("throughput", "Throughput [veh/s]", "{:.3f}"),
    ("speed", "Speed [m/s]", "{:.2f}"),
    ("intervention", "Intervention [%]", "{:.1f}"),
    ("infeas", "Infeasible steps [%]", "{:.1f}"),
    ("col_after_infeas", "Col. after infeasible step [%]", "{:.0f}"),
    ("fallback", "Fallback [1/1000]", "{:.2f}"),
    ("outside_c", "Steps outside C [%]", "{:.2f}"),
    ("min_backup", "Min backup value [m]", "{:.3f}"),
    ("t_seq", "CBF seq. [ms]", "{:.1f}"),
    ("t_par", "CBF par. [ms]", "{:.1f}"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[],
        help="Checkpoint folders (each with its reward*_data.json).",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(MAPS),
        help="Deployment maps (scenario types).",
    )
    parser.add_argument(
        "--n_agents",
        nargs="*",
        default=[],
        help="Vehicles per map as MAP=N (default: settings.N_AGENTS).",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max_steps", type=int, default=601)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max_running",
        type=int,
        default=None,
        help="Start a run only while fewer than this many training and rollout "
        "processes run on the machine (see stage1_pretrain.n_running).",
    )
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--tag", default="", help="Suffix of the method folders.")
    parser.add_argument(
        "--cbf_lambda",
        type=float,
        default=None,
        help="Decay rate of all runs instead of the tuned values (ablation).",
    )
    parser.add_argument(
        "--tuned",
        default=os.path.join(
            REPO_ROOT, "sigmarl", "cbf_baselines", "tuned_defaults.json"
        ),
        help="Tuned lambda and environment per method and mode ('' for none).",
    )
    parser.add_argument(
        "--is_load_final_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deploy final_*.pth (the last iteration, default) instead of the "
        "best-reward model.",
    )
    parser.add_argument(
        "--policy_iter",
        type=int,
        default=None,
        help="Deploy the policy of this training iteration "
        "(checkpoints_cross_iter/policy_iter<N>.pth); labels get the suffix __iter<N>.",
    )
    parser.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Extra environment variables KEY=VALUE.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Only summarize the finished runs under --out_root.",
    )
    args = parser.parse_args()
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        parser.error(f"Unknown methods {unknown}; one of {sorted(METHODS)}.")
    if args.cbf_lambda is not None:
        args.tuned = ""
    return args


def checkpoint_scenario(checkpoint: str) -> str:
    """Training map of a checkpoint (from its reward*_data.json)."""
    data_files = glob.glob(os.path.join(checkpoint, "reward*_data.json"))
    if not data_files:
        raise FileNotFoundError(f"No reward*_data.json in '{checkpoint}'.")
    with open(data_files[0]) as file:
        return json.load(file)["parameters"]["scenario_type"]


def checkpoint_label(checkpoint: str) -> str:
    """<training map>__<checkpoint folder name>."""
    name = os.path.basename(os.path.normpath(checkpoint))
    return f"{checkpoint_scenario(checkpoint)}__{name}"


def n_agents_of(args) -> dict:
    """Vehicles per deployment map: settings.N_AGENTS, else the scenario's
    n_agents, overridden by --n_agents MAP=N."""
    from sigmarl.constants import SCENARIOS

    counts = {
        name: N_AGENTS.get(name, int(SCENARIOS[name]["n_agents"])) for name in args.maps
    }
    for item in args.n_agents:
        name, value = item.split("=", 1)
        counts[name] = int(value)
    return counts


def build_jobs(args, env_base: dict) -> list:
    """(output folder, command, environment) of every run without a summary."""
    tuned = json.load(open(args.tuned)) if args.tuned else {}
    counts = n_agents_of(args)
    jobs = []
    for checkpoint in args.checkpoints:
        label = checkpoint_label(checkpoint)
        if args.policy_iter is not None:
            label += f"__iter{args.policy_iter}"
        for deploy_map in args.maps:
            for method in args.methods:
                for mode in args.modes:
                    # The unfiltered reference is the same in both modes
                    if METHODS[method] is None and mode != "cen":
                        continue
                    for seed in args.seeds:
                        name = method + (f"_{args.tag}" if args.tag else "")
                        out_dir = os.path.join(
                            args.out_root, label, deploy_map, name, mode, f"seed{seed}"
                        )
                        if glob.glob(
                            os.path.join(out_dir, "evaluation_summary_*.json")
                        ):
                            continue
                        command = [
                            sys.executable,
                            os.path.join(REPO_ROOT, "main_testing_cbf_filter.py"),
                            "--nominal",
                            "rl",
                            "--path",
                            os.path.abspath(checkpoint),
                            "--scenario_type",
                            deploy_map,
                            "--n_agents",
                            str(counts[deploy_map]),
                            "--max_steps",
                            str(args.max_steps),
                            "--max_group_size",
                            MODES[mode],
                            "--seed",
                            str(seed),
                            "--is_zero_initial_speed",
                            "--is_log_backup_value",
                            "--no-is_visu_result",
                            "--no-is_save_simulation_video",
                            "--no-is_exact_backup_shield",
                            "--no-is_standstill_escape",
                            "--output_dir",
                            out_dir,
                        ]
                        if args.is_load_final_model:
                            command.append("--is_load_final_model")
                        if args.policy_iter is not None:
                            command += ["--policy_iter", str(args.policy_iter)]
                        filter_args, filter_env = method_args(
                            method, mode, BLOCK, tuned, args.cbf_lambda
                        )
                        env = dict(env_base)
                        env.update(filter_env)
                        jobs.append((out_dir, command + filter_args, env))
    return jobs


def summarize(out_root: str):
    """Write runs.csv, summary.csv, and summary.md (see the module docstring)."""
    from sigmarl.evaluation_tits26.before_stages_filter_summary import run_metrics

    run_dirs = sorted(
        os.path.dirname(path)
        for path in glob.glob(f"{out_root}/*/*/*/*/seed*/cbf_statistics_*.json")
    )
    runs = []
    for run_dir in tqdm(run_dirs, unit="run"):
        label, deploy_map, method, mode, seed = run_dir.split(os.sep)[-5:]
        train_map = label.split("__")[0]
        row = dict(
            checkpoint=label,
            train_map=train_map,
            deploy_map=deploy_map,
            relation="own" if deploy_map == train_map else "unseen",
            method=method,
            mode=mode,
            seed=int(seed.removeprefix("seed")),
        )
        row.update(run_metrics(run_dir))
        runs.append(row)
    if not runs:
        print(f"[INFO] No finished runs under {out_root}.")
        return
    with open(os.path.join(out_root, "runs.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)

    groups = {}
    for row in runs:
        key = (row["mode"], row["method"], row["relation"])
        groups.setdefault(key, []).append(row)
    order = {name: k for k, name in enumerate(METHODS)}
    rows = []
    for (mode, method, relation), members in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            order.get(item[0][1].split("_lam")[0], 99),
            item[0][1],
            item[0][2],
        ),
    ):
        row = dict(mode=mode, method=method, relation=relation, n_runs=len(members))
        for key, _, _ in SUMMARY_METRICS:
            if key == "col_after_infeas":
                n_onsets = sum(m["n_onsets"] for m in members)
                n_after = sum(m["n_onsets_after_infeasible"] for m in members)
                row[key] = 100.0 * n_after / n_onsets if n_onsets else float("nan")
                row[key + "_se"] = float("nan")
                continue
            values = np.array([m[key] for m in members], dtype=np.float64)
            if key == "min_backup":
                row[key], row[key + "_se"] = float(np.min(values)), float("nan")
                continue
            row[key] = float(np.mean(values))
            row[key + "_se"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else float("nan")
            )
        rows.append(row)
    with open(os.path.join(out_root, "summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Stage 1: zero-shot deployment",
        "",
        "Mean over runs (+- standard error); collisions per run (601 steps), "
        "spawn collisions excluded. Min backup value: smallest over runs. "
        "CBF time: sequential (actual loop) and parallelized (per-agent "
        "estimate, equal to sequential when centralized).",
        "",
        "| Mode | Method | Maps | Runs | "
        + " | ".join(title for _, title, _ in SUMMARY_METRICS)
        + " |",
        "|---" * (4 + len(SUMMARY_METRICS)) + "|",
    ]
    for row in rows:
        cells = []
        for key, _, fmt in SUMMARY_METRICS:
            if np.isnan(row[key + "_se"]):
                cells.append(fmt.format(row[key]))
            else:
                cells.append(f"{fmt.format(row[key])} ± {fmt.format(row[key + '_se'])}")
        lines.append(
            f"| {row['mode']} | {row['method']} | {row['relation']} | "
            f"{row['n_runs']} | " + " | ".join(cells) + " |"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_root, "summary.md"), "w") as file:
        file.write(text)
    print(text)


def main():
    args = parse_args()
    if args.summarize:
        summarize(args.out_root)
        return
    if not args.checkpoints:
        raise SystemExit("--checkpoints is required unless --summarize is given.")
    if args.max_running is not None:
        from sigmarl.evaluation_tits26 import stage1_pretrain
        from sigmarl.evaluation_tits26 import before_stages_filter_benchmark

        before_stages_filter_benchmark.run_job = stage1_pretrain.gated_run_job(
            args.max_running
        )
    run_jobs(build_jobs(args, base_env(args.env)), args.workers)


if __name__ == "__main__":
    main()
