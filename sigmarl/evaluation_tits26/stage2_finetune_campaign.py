# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 2 of the T-ITS evaluation: filter-guarded fine-tuning campaign.

Runs stage2_finetune_run.py for every pre-trained source checkpoint
(--source_model: final, or a training iteration iter<N>), target map (the
source's own map and the unseen ones), method (a filter of
before_stages_filter_benchmark.METHODS, or none / none_bh without a filter), and seed, then
evaluates the saved iteration checkpoints without a filter and with the run's
filter (BH-CBF for the runs without one). Finished trainings and evaluations
are skipped, so every subcommand resumes. Outputs:
<out_root>/<training map>__<source folder>[__iter<N>]/<target map>/<method>/seed<k>.

Subcommands:
    train      fine-tune (one process per run)
    eval       evaluate the checkpoints of all runs (one process per checkpoint)
    summarize  write runs.csv, summary.csv/.md (per method and own/unseen
               target: training collisions, and the metrics of the final
               checkpoint without and with the filter, mean and standard error
               over runs), and curves.csv (per method, own/unseen, and
               iteration: training statistics and evaluation metrics)

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.stage2_finetune_campaign train \\
        --sources checkpoints/<date>/round_location0/<run> ... \\
        --maps round_location0 ind_location4 \\
        --out_root outputs/tits26/stage2_finetune [--source_model iter100] [--workers 10]
    ... eval --out_root outputs/tits26/stage2_finetune [--workers 10]
    ... summarize --out_root outputs/tits26/stage2_finetune
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

from sigmarl.evaluation_tits26.stage1_zero_shot import checkpoint_label, n_agents_of
from sigmarl.evaluation_tits26.before_stages_filter_benchmark import (
    METHODS,
    REPO_ROOT,
    base_env,
    run_jobs,
)

SCRIPT = os.path.join(
    REPO_ROOT, "sigmarl", "evaluation_tits26", "stage2_finetune_run.py"
)
DEFAULT_METHODS = [
    "ours",
    "b6_backup_cbf",
    "mps",
    "t2_dual_distance",
    "none",
    "none_bh",
]
NO_FILTER_METHODS = ("none", "none_bh")
# Metrics of the summary: (key, title, format)
SUMMARY_METRICS = [
    ("train_col", "Training col. [colliding agent-steps, total]", "{:.0f}"),
    ("train_outside_c", "Training outside C [% env-steps]", "{:.2f}"),
    ("train_min_backup", "Training min. backup value [m]", "{:.3f}"),
    ("train_intervention_first", "Training int., first 10 it. [%]", "{:.1f}"),
    ("train_intervention", "Training int., last 10 it. [%]", "{:.1f}"),
    ("train_fallback", "Training fallback, last 10 it. [1/1000]", "{:.2f}"),
    ("nofilter_col", "No filter: col. [1/env-min]", "{:.2f}"),
    ("nofilter_throughput", "No filter: throughput [veh/s]", "{:.3f}"),
    ("filter_col", "Filter: col. [1/env-min]", "{:.3f}"),
    ("filter_intervention", "Filter: int. [%]", "{:.1f}"),
    ("filter_fallback", "Filter: fallback [1/1000]", "{:.2f}"),
    ("filter_outside_c", "Filter: outside C [% env-steps]", "{:.2f}"),
    ("filter_throughput", "Filter: throughput [veh/s]", "{:.3f}"),
    ("wall_h", "Training time [h]", "{:.1f}"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "eval", "summarize"):
        command = sub.add_parser(name)
        command.add_argument("--out_root", required=True)
        if name != "summarize":
            command.add_argument("--workers", type=int, default=4)
            command.add_argument(
                "--max_running",
                type=int,
                default=None,
                help="Start a run only while fewer than this many training and "
                "rollout processes run on the machine (stage1_pretrain.n_running).",
            )
            command.add_argument(
                "--env", nargs="*", default=[], help="Extra KEY=VALUE variables."
            )
    tr = sub.choices["train"]
    tr.add_argument("--sources", nargs="+", required=True)
    tr.add_argument(
        "--source_model",
        default="final",
        help="Source model: final or iter<N> (the label gets the suffix __iter<N>).",
    )
    tr.add_argument("--maps", nargs="+", required=True, help="Target maps.")
    tr.add_argument(
        "--n_agents",
        nargs="*",
        default=[],
        help="Vehicles per map as MAP=N (default: settings.N_AGENTS).",
    )
    tr.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=[*NO_FILTER_METHODS, *METHODS],
    )
    tr.add_argument("--mode", default="dec", choices=["dec", "cen"])
    tr.add_argument("--seeds", nargs="+", type=int, default=[1])
    tr.add_argument("--n_iters", type=int, default=100)
    tr.add_argument("--save_every", type=int, default=10)
    tr.add_argument(
        "--train_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Further arguments of stage2_finetune_run.py train (last).",
    )
    ev = sub.choices["eval"]
    ev.add_argument(
        "--iters",
        nargs="+",
        type=int,
        default=None,
        help="Checkpoint iterations (default: all saved ones).",
    )
    ev.add_argument("--eval_seeds", nargs="+", type=int, default=[100, 101, 102])
    ev.add_argument(
        "--filters",
        nargs="+",
        default=["nofilter", "filter"],
        choices=["nofilter", "filter"],
    )
    ev.add_argument("--num_vmas_envs", type=int, default=8)
    ev.add_argument("--max_steps", type=int, default=601)
    return parser.parse_args()


def train_jobs(args, env: dict) -> list:
    """(run folder, command, environment) of every unfinished training."""
    counts = n_agents_of(args)
    save_iters = [str(i) for i in range(0, args.n_iters + 1, args.save_every)]
    if save_iters[-1] != str(args.n_iters):
        save_iters.append(str(args.n_iters))
    jobs = []
    for source in args.sources:
        label = checkpoint_label(source)
        if args.source_model != "final":
            label += f"__{args.source_model}"
        for target in args.maps:
            for method in args.methods:
                for seed in args.seeds:
                    run = os.path.join(
                        args.out_root, label, target, method, f"seed{seed}"
                    )
                    final = os.path.join(
                        run, "checkpoints_cross_iter", f"policy_iter{args.n_iters}.pth"
                    )
                    if os.path.exists(final):
                        continue
                    command = [
                        sys.executable,
                        SCRIPT,
                        "train",
                        "--source",
                        os.path.abspath(source),
                        "--source_model",
                        args.source_model,
                        "--scenario_type",
                        target,
                        "--n_agents",
                        str(counts[target]),
                        "--method",
                        method,
                        "--mode",
                        args.mode,
                        "--seed",
                        str(seed),
                        "--n_iters",
                        str(args.n_iters),
                        "--save_iters",
                        *save_iters,
                        "--out",
                        run,
                        *args.train_args,
                    ]
                    jobs.append((run, command, env))
    return jobs


def run_folders(out_root: str) -> list:
    """Fine-tuning run folders under out_root."""
    return sorted(
        os.path.dirname(path)
        for path in glob.glob(f"{out_root}/*/*/*/seed*/run_config.json")
    )


def evaluated(path: str) -> set:
    """Iterations already in an evaluation file."""
    if not os.path.exists(path):
        return set()
    with open(path) as file:
        return {int(json.loads(line)["iter"]) for line in file if line.strip()}


def eval_jobs(args, env: dict) -> list:
    """(log folder, command, environment) of every missing evaluation."""
    jobs = []
    for run in run_folders(args.out_root):
        saved = sorted(
            int(os.path.basename(path).removeprefix("policy_iter").removesuffix(".pth"))
            for path in glob.glob(f"{run}/checkpoints_cross_iter/policy_iter*.pth")
        )
        iters = saved if args.iters is None else sorted(set(saved) & set(args.iters))
        for tag in args.filters:
            for seed in args.eval_seeds:
                done = evaluated(os.path.join(run, f"eval_{tag}_seed{seed}.jsonl"))
                for i_iter in iters:
                    if i_iter in done:
                        continue
                    log_dir = os.path.join(
                        run, "eval_logs", f"{tag}_seed{seed}_iter{i_iter}"
                    )
                    command = [
                        sys.executable,
                        SCRIPT,
                        "eval",
                        "--run",
                        run,
                        "--iter",
                        str(i_iter),
                        "--filter" if tag == "filter" else "--no-filter",
                        "--seed",
                        str(seed),
                        "--num_vmas_envs",
                        str(args.num_vmas_envs),
                        "--max_steps",
                        str(args.max_steps),
                    ]
                    jobs.append((log_dir, command, env))
    return jobs


def per_minute(result: dict, key: str) -> float:
    """Events of an evaluation per environment-minute."""
    return result[key] / (result["num_envs"] * result["duration_s"] / 60.0)


def mean_of(results: list, key: str) -> float:
    """Mean of a field over results (nan where it was not logged)."""
    values = [r[key] for r in results if r.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def min_of(results: list, key: str) -> float:
    """Minimum of a field over results (nan where it was not logged)."""
    values = [r[key] for r in results if r.get(key) is not None]
    return float(min(values)) if values else float("nan")


def train_window(stats: list, key: str, rows: slice, scale: float = 1.0) -> float:
    """Mean of a train_stats.csv column over the rows of an iteration window."""
    values = [float(s[key]) for s in stats[rows] if s.get(key) not in (None, "")]
    return scale * float(np.mean(values)) if values else float("nan")


def eval_rows(run: str, tag: str) -> dict:
    """Evaluation metrics per iteration of a run, averaged over the seeds."""
    by_iter = {}
    for path in glob.glob(os.path.join(run, f"eval_{tag}_seed*.jsonl")):
        with open(path) as file:
            for line in file:
                if line.strip():
                    result = json.loads(line)
                    by_iter.setdefault(int(result["iter"]), []).append(result)
    rows = {}
    for i_iter, results in by_iter.items():
        rows[i_iter] = dict(
            veh_col=np.mean([per_minute(r, "agent_agent_events") for r in results]),
            lane_col=np.mean([per_minute(r, "agent_lane_events") for r in results]),
            col=np.mean([per_minute(r, "collision_events") for r in results]),
            throughput=np.mean([r["throughput_veh_s"] for r in results]),
            speed=np.mean([r["mean_speed"] for r in results]),
            intervention=100.0 * mean_of(results, "intervention_rate"),
            fallback=mean_of(results, "fallback_per_1000"),
            outside_c=100.0 * mean_of(results, "outside_c_env_step_rate"),
            min_backup=min_of(results, "min_backup_value"),
            n_seeds=len(results),
        )
    return rows


def summarize(out_root: str):
    """Write runs.csv, summary.csv/.md, and curves.csv (see the module docstring)."""
    runs, curves = [], {}
    for run in run_folders(out_root):
        label, target, method, seed = run.split(os.sep)[-4:]
        train_map = label.split("__")[0]
        relation = "own" if target == train_map else "unseen"
        stats_path = os.path.join(run, "train_stats.csv")
        if not os.path.exists(stats_path):
            continue
        with open(stats_path) as file:
            stats = list(csv.DictReader(file))
        if not stats:
            continue
        nofilter, withfilter = eval_rows(run, "nofilter"), eval_rows(run, "filter")
        last = stats[-1]
        backups = [float(s["min_backup_value"]) for s in stats if s["min_backup_value"]]
        row = dict(
            source=label,
            train_map=train_map,
            target_map=target,
            relation=relation,
            method=method,
            seed=int(seed.removeprefix("seed")),
            iters=int(last["iter"]),
            train_veh_col=int(last["cum_agent_collision_steps"]),
            train_lane_col=int(last["cum_lane_collision_steps"]),
            train_col=int(last["cum_agent_collision_steps"])
            + int(last["cum_lane_collision_steps"]),
            train_outside_c=train_window(
                stats, "outside_c_env_step_rate", slice(None), 100.0
            ),
            train_min_backup=min(backups) if backups else float("nan"),
            train_intervention_first=train_window(
                stats, "intervention_rate", slice(0, 10), 100.0
            ),
            train_intervention=train_window(
                stats, "intervention_rate", slice(-10, None), 100.0
            ),
            train_fallback=train_window(stats, "fallback_per_1000", slice(-10, None)),
            train_goals=train_window(stats, "goal_steps", slice(-10, None)),
            train_speed=train_window(stats, "mean_speed", slice(-10, None)),
            wall_h=float(last["wall_s"]) / 3600.0,
        )
        final = max(set(nofilter) | set(withfilter), default=None)
        for prefix, rows, keys in (
            (
                "nofilter",
                nofilter,
                ("col", "veh_col", "lane_col", "throughput", "speed"),
            ),
            (
                "filter",
                withfilter,
                ("col", "intervention", "fallback", "outside_c", "throughput", "speed"),
            ),
        ):
            for key in keys:
                row[f"{prefix}_{key}"] = (
                    rows[final][key] if final in rows else float("nan")
                )
        row["final_eval_iter"] = final
        runs.append(row)
        for stat in stats:
            i_iter = int(stat["iter"])
            entry = curves.setdefault((method, relation, i_iter), {})
            entry.setdefault("train_col", []).append(
                int(stat["agent_collision_steps"]) + int(stat["lane_collision_steps"])
            )
            for key, scale in (
                ("intervention_rate", 100.0),
                ("fallback_per_1000", 1.0),
                ("slack_env_step_rate", 100.0),
                ("outside_c_env_step_rate", 100.0),
                ("goal_steps", 1.0),
                ("mean_speed", 1.0),
                ("episode_reward", 1.0),
            ):
                if stat.get(key) not in (None, ""):
                    entry.setdefault(f"train_{key}", []).append(
                        scale * float(stat[key])
                    )
        for prefix, rows in (("nofilter", nofilter), ("filter", withfilter)):
            for i_iter, metrics in rows.items():
                entry = curves.setdefault((method, relation, i_iter), {})
                for key in (
                    "col",
                    "throughput",
                    "speed",
                    "intervention",
                    "fallback",
                    "outside_c",
                ):
                    entry.setdefault(f"{prefix}_{key}", []).append(metrics[key])
    if not runs:
        print(f"[INFO] No fine-tuning runs under {out_root}.")
        return
    with open(os.path.join(out_root, "runs.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)

    curve_keys = sorted({key for entry in curves.values() for key in entry})
    with open(os.path.join(out_root, "curves.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "relation", "iter", *curve_keys])
        for (method, relation, i_iter), entry in sorted(curves.items()):
            writer.writerow(
                [method, relation, i_iter]
                + [
                    float(np.mean(entry[key])) if key in entry else ""
                    for key in curve_keys
                ]
            )

    order = {name: k for k, name in enumerate(DEFAULT_METHODS)}
    groups = {}
    for row in runs:
        groups.setdefault((row["method"], row["relation"]), []).append(row)
    rows = []
    for (method, relation), members in sorted(
        groups.items(), key=lambda item: (order.get(item[0][0], 99), item[0])
    ):
        row = dict(method=method, relation=relation, n_runs=len(members))
        for key, _, _ in SUMMARY_METRICS:
            values = np.array([m[key] for m in members], dtype=np.float64)
            values = values[~np.isnan(values)]
            row[key] = float(np.mean(values)) if values.size else float("nan")
            row[key + "_se"] = (
                float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else float("nan")
            )
        rows.append(row)
    with open(os.path.join(out_root, "summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage 2: filter-guarded fine-tuning",
        "",
        "Mean over runs (+- standard error). Training col.: colliding agent-steps "
        "over all fine-tuning iterations; training outside C: share of env-steps "
        "whose joint state is not certified (our margins, as in Stage 1), over all "
        "iterations; training int./fallback: first or last 10 iterations. "
        "Evaluation: last evaluated checkpoint, mean over evaluation seeds; "
        "'Filter' uses the run's filter (BH-CBF for the runs without one).",
        "",
        "| Method | Target | Runs | "
        + " | ".join(title for _, title, _ in SUMMARY_METRICS)
        + " |",
        "|---" * (3 + len(SUMMARY_METRICS)) + "|",
    ]
    for row in rows:
        cells = []
        for key, _, fmt in SUMMARY_METRICS:
            if np.isnan(row[key + "_se"]):
                cells.append(fmt.format(row[key]))
            else:
                cells.append(f"{fmt.format(row[key])} ± {fmt.format(row[key + '_se'])}")
        lines.append(
            f"| {row['method']} | {row['relation']} | {row['n_runs']} | "
            + " | ".join(cells)
            + " |"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_root, "summary.md"), "w") as file:
        file.write(text)
    print(text)


def main():
    args = parse_args()
    if args.command == "summarize":
        summarize(args.out_root)
        return
    if args.max_running is not None:
        from sigmarl.evaluation_tits26 import stage1_pretrain
        from sigmarl.evaluation_tits26 import before_stages_filter_benchmark

        before_stages_filter_benchmark.run_job = stage1_pretrain.gated_run_job(
            args.max_running
        )
    env = base_env(args.env)
    jobs = train_jobs(args, env) if args.command == "train" else eval_jobs(args, env)
    run_jobs(jobs, args.workers)


if __name__ == "__main__":
    main()
