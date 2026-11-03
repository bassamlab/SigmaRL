# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Computation time of BH-CBF (centralized and fully decentralized) over the
number of vehicles.

The nominal controllers are the Stage 2 source policies: the five TTC-reward
policies pre-trained on the roundabout, at iteration 100. Each policy runs
once per vehicle count and mode (one VMAS env, spawns at rest, environment
seed = policy seed, shield and standstill escape off, tuned lambda) through
main_testing_cbf_filter.py, which logs the per-step CBF time and its parts
(sigmarl/cbf_qp_proxsuite.py, TIME_PARTS): the QP solves, the exact check of
the solutions (with its repair rows), and the fallback to backup controls.
The rest is the construction of the QP (backup rollouts, linearization,
rows). For the fully decentralized filter, all times are the parallelized
estimate: the shared preparation plus the slowest agent's update, with the
parts of that agent.

The summary skips the first --n_skipped steps of each run (initialization
noise) and keeps the next --n_kept steps (runs of 105 steps by default; longer
runs keep the same steps). It writes <out_root>/times.csv (one row per step), summary.csv
(mean, median, p95 per vehicle count and mode), and the bar plot
fig_computation_time_{a,b,legend}.{pdf,png} (centralized,
decentralized, legend).

Runs whose CBF statistics exist are skipped, so the script resumes.
The rollouts run under ``taskpolicy -t 0 -l 0`` (highest throughput and
latency tiers; macOS cannot pin performance cores, but the scheduler prefers
them for such tasks).

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.computation_time_scaling \\
        [--n_agents 2 4 ... 30] [--workers 5] [--out_root outputs/tits26/computation_time]
    ... --summarize
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from sigmarl.evaluation_tits26.before_stages_filter_benchmark import (
    MODES,
    REPO_ROOT,
    base_env,
    method_args,
    run_jobs,
)
from sigmarl.evaluation_tits26.stage1_zero_shot import BLOCK

MAP = "round_location0"
SOURCES = "checkpoints/tits26/stage1_pretrain/round_location0/stage1_ttc_seed_{}"
SOURCE_ITER = 100
METHOD = "ours"
# Parts of the per-step time: (key, label, color); "other" is the rest
PARTS = [
    ("qp", "QP solves", "#00549F"),
    ("check", "Exact check", "#57AB27"),
    ("fallback", "Fallback", "#CC071E"),
    ("other", "QP construction", "#C7DDF2"),
]
# y-ticks of the centralized and decentralized panels (outer ticks: limits),
# used while the bars fit below the last one; else ticks from MaxNLocator
YTICKS_CEN = [0, 3000, 6000, 9000]
YTICKS_DEC = [0, 3, 6, 9]
# Inset of the centralized panel: vehicle counts up to INSET_MAX_N, y-ticks [ms]
INSET_MAX_N = 20
INSET_YTICKS = [0, 50, 100, 150]
STEP_MS = 100  # Sampling step Delta t [ms]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n_agents", nargs="+", type=int, default=range(2, 31, 2))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--max_steps", type=int, default=106)
    parser.add_argument("--n_skipped", type=int, default=5)
    parser.add_argument("--n_kept", type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--out_root", default=os.path.join("outputs", "tits26", "computation_time")
    )
    parser.add_argument(
        "--tuned",
        default=os.path.join(
            REPO_ROOT, "sigmarl", "cbf_baselines", "tuned_defaults.json"
        ),
    )
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def run_dir(out_root: str, n_agents: int, mode: str, seed: int) -> str:
    return os.path.join(out_root, f"n{n_agents:02d}", mode, f"seed{seed}")


def build_jobs(args, env: dict) -> list:
    """(output folder, command, environment) of every run without statistics."""
    tuned = json.load(open(args.tuned))
    prefix = [shutil.which("taskpolicy"), "-t", "0", "-l", "0"]
    jobs = []
    for n_agents in args.n_agents:
        for mode in args.modes:
            for seed in args.seeds:
                out_dir = run_dir(args.out_root, n_agents, mode, seed)
                if glob.glob(os.path.join(out_dir, "cbf_statistics_*.json")):
                    continue
                command = [
                    sys.executable,
                    os.path.join(REPO_ROOT, "main_testing_cbf_filter.py"),
                    "--nominal",
                    "rl",
                    "--path",
                    os.path.abspath(SOURCES.format(seed)),
                    "--policy_iter",
                    str(SOURCE_ITER),
                    "--scenario_type",
                    MAP,
                    "--n_agents",
                    str(n_agents),
                    "--max_steps",
                    str(args.max_steps),
                    "--max_group_size",
                    MODES[mode],
                    "--seed",
                    str(seed),
                    "--is_zero_initial_speed",
                    "--no-is_visu_result",
                    "--no-is_save_simulation_video",
                    "--no-is_exact_backup_shield",
                    "--no-is_standstill_escape",
                    "--output_dir",
                    out_dir,
                ]
                filter_args, filter_env = method_args(METHOD, mode, BLOCK, tuned)
                job_env = dict(env)
                job_env.update(filter_env)
                jobs.append((out_dir, prefix + command + filter_args, job_env))
    return jobs


def summarize(out_root: str, n_skipped: int, n_kept: int):
    """times.csv, summary.csv, and the bar plot of all finished runs."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out_root, "n*/*/seed*"))):
        files = glob.glob(os.path.join(path, "cbf_statistics_*.json"))
        if not files:
            continue
        n_agents = int(path.split(os.sep)[-3][1:])
        mode, seed = path.split(os.sep)[-2], int(path.split(os.sep)[-1][4:])
        times = json.load(open(files[0]))["time_cbf_per_step_ms"]
        for step in range(n_skipped, n_skipped + n_kept):
            row = dict(n_agents=n_agents, mode=mode, seed=seed, step=step)
            row.update({key: values[step] for key, values in times.items()})
            row["other"] = row["total"] - sum(row[p] for p, *_ in PARTS[:-1])
            rows.append(row)
    if not rows:
        print("[INFO] No finished runs.")
        return
    keys = ["n_agents", "mode", "seed", "step", "total"] + [p for p, *_ in PARTS]
    with open(os.path.join(out_root, "times.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in rows)
    summary = []
    for (n_agents, mode), group in groupby(rows).items():
        entry = dict(n_agents=n_agents, mode=mode, n=len(group))
        for key in ["total"] + [p for p, *_ in PARTS]:
            values = np.array([row[key] for row in group])
            entry[f"{key}_mean"] = values.mean()
            entry[f"{key}_median"] = np.median(values)
            entry[f"{key}_p95"] = np.percentile(values, 95)
        summary.append(entry)
    with open(os.path.join(out_root, "summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        for entry in summary:
            writer.writerow(
                {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in entry.items()}
            )
    print(f"[INFO] Saved {out_root}/summary.csv ({len(rows)} steps)")
    for entry in summary:
        print(
            f"n={entry['n_agents']:2d} {entry['mode']}: total "
            f"{entry['total_mean']:.1f} ms (p95 {entry['total_p95']:.1f}), "
            + ", ".join(f"{p} {entry[f'{p}_mean']:.2f}" for p, *_ in PARTS)
        )
    plot(summary, os.path.join(out_root, "fig_computation_time"))


def groupby(rows: list) -> dict:
    """Rows per (vehicle count, mode), sorted."""
    groups = {}
    for row in rows:
        groups.setdefault((row["n_agents"], row["mode"]), []).append(row)
    return dict(sorted(groups.items()))


def stacked_bars(ax, entries: list) -> tuple:
    """Stacked mean parts per vehicle count; returns the counts and totals."""
    x = np.array([e["n_agents"] for e in entries])
    bottom = np.zeros(len(entries))
    for part, label, color in PARTS:
        height = np.array([e[f"{part}_mean"] for e in entries])
        ax.bar(x, height, 1.5, bottom=bottom, color=color, lw=0, label=label)
        bottom += height
    return x, bottom


def inset_small_counts(ax, entries: list):
    """Inset of the bars up to INSET_MAX_N vehicles with the sampling step
    Delta t (dashed), in the empty upper left of the centralized panel."""
    inset = ax.inset_axes([0.21, 0.4, 0.45, 0.56])
    stacked_bars(inset, [e for e in entries if e["n_agents"] <= INSET_MAX_N])
    inset.axhline(STEP_MS, color="0.3", lw=0.6, ls=(0, (3, 1.5)))
    inset.text(1, STEP_MS + 4, r"$\Delta t$", fontsize=7, color="0.3", va="bottom")
    inset.set_xlim(0, INSET_MAX_N + 2)
    inset.set_xticks([2, 10, 20])
    inset.set_ylim(INSET_YTICKS[0], INSET_YTICKS[-1])
    inset.set_yticks(INSET_YTICKS)
    inset.tick_params(labelsize=6, length=1.5, pad=1)
    inset.grid(axis="y", color="0.9", lw=0.4)
    inset.set_axisbelow(True)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)


def plot(summary: list, path: str):
    """Stacked bars of the mean time per step and its parts over the number of
    vehicles: centralized (<path>_a), decentralized (<path>_b), and their
    common legend (<path>_legend)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 8,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )
    figures = []
    for suffix, mode, yticks in (("a", "cen", YTICKS_CEN), ("b", "dec", YTICKS_DEC)):
        entries = [e for e in summary if e["mode"] == mode]
        if not entries:
            continue
        fig = plt.figure(figsize=(1.75, 1.3))
        ax = fig.add_axes([0.25, 0.27, 0.72, 0.69])
        x, bottom = stacked_bars(ax, entries)
        if mode == "cen":
            inset_small_counts(ax, entries)
        n_max = int(x.max())
        ax.set_xlim(0, n_max + 2)
        ax.set_xticks(
            range(2, n_max + 1, 4) if n_max <= 30 else [2, *range(10, n_max + 1, 10)]
        )
        if bottom.max() > yticks[-1]:
            yticks = MaxNLocator(nbins=4).tick_values(0, bottom.max())
            yticks = [t for t in yticks if t < bottom.max()] + [
                t for t in yticks if t >= bottom.max()
            ][:1]
        ax.set_ylim(yticks[0], yticks[-1])
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:g}" for t in yticks])
        if yticks[-1] >= 10000:
            # Room for five-digit tick labels
            ax.set_position([0.29, 0.27, 0.68, 0.69])
        ax.set_xlabel("Number of vehicles", labelpad=1)
        ax.set_ylabel("Time per step [ms]", labelpad=2)
        ax.grid(axis="y", color="0.9", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        figures.append((fig, suffix))
    fig_legend = plt.figure(figsize=(3.5, 0.2))
    handles, labels = ax.get_legend_handles_labels()
    fig_legend.legend(handles, labels, loc="center", ncol=len(PARTS), frameon=False,
                      handlelength=1.2, columnspacing=1.2, borderaxespad=0)  # fmt: skip
    figures.append((fig_legend, "legend"))
    for fig, suffix in figures:
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}_{suffix}.{ext}", dpi=300)
        plt.close(fig)
    print(f"[INFO] Saved {path}_{{a,b,legend}}.pdf")


def main():
    args = parse_args()
    if not args.summarize:
        run_jobs(build_jobs(args, base_env()), args.workers)
    summarize(args.out_root, args.n_skipped, args.n_kept)


if __name__ == "__main__":
    main()
