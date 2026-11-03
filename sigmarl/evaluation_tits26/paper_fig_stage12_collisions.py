# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Paper figure of Stages 1 and 2: collisions of the TTC-reward policies over
pre-training and fine-tuning.

(a) Collisions of the training rollouts: the first 100
pre-training iterations of the 10 TTC-reward policies (both maps), then the
100 fine-tuning iterations from their iteration-100 checkpoints, per fine-tuning
condition (20 runs each: 10 sources x own and unseen map), trailing moving
average. (b) Colliding vehicles of the fine-tuning checkpoints deployed
without a filter (evaluation seed 201, every 20 iterations). Mean over runs,
with the standard deviation over runs in (a) and the standard error in (b).

It writes two versions that differ in the collision unit: per time
(fig_stage12_collisions_*: (a) per agent-minute, (b) per environment-minute)
and per km driven (fig_stage12_collisions_{a,b}_per_km; (a) on a symmetric
log scale, linear below 1/km, so that the collision-free curves stay visible).
Both share fig_stage12_collisions_legend. The
distance is the route progress in Stage 1 (stage1_plot_training.py) and the
mean speed times the agent time in Stage 2; both agree for vehicles that
follow their lanes.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.paper_fig_stage12_collisions \\
        [--out outputs/tits26/paper_figs]
"""

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sigmarl.evaluation_tits26.settings import STAGE2_EXCLUDED_LANE_STEPS
from sigmarl.evaluation_tits26.stage1_plot_training import load_runs, moving_average

PRETRAIN_ROOT = "checkpoints/tits26/stage1_pretrain"
FINETUNE = "outputs/tits26/stage2_finetune"
PRETRAIN_ITERS = 100
# Fine-tuning conditions: (key, label, color, line style, line width, marker),
# RWTH colors. Distinct markers keep the curves at zero apart in (a).
BLACK = "#000000"
CONDITIONS = [
    ("none", "Without filter, TTC reward", BLACK, (0, (3, 1.5)), 0.9, "x"),
    ("none_bh", "Without filter, BH-CBF reward", "#8EBAE5", (0, (3, 1.5)), 0.9, "v"),
    ("t2_dual_distance", "B-Dual", "#56AB27", "-", 0.9, "D"),
    ("mps", "B-MPS", "#E30066", "-", 0.9, "^"),
    ("b6_backup_cbf", "B-Backup", "#7B6FAC", "-", 0.9, "s"),
    ("ours", "BH-CBF (ours)", "#00549F", "-", 1.4, "o"),
]
# Geometry ablation, drawn only with --with_geometry_ablation
GEOMETRY_ABLATION = ("bh_capsule", "A-Geometry", "#006165", "-", 0.9, "P")
MARKER_SIZE = 2.6
DT = 0.1  # [s]
# Figure versions: (file suffix, per km driven)
VERSIONS = [("", False), ("_per_km", True)]
# Markers in (a) every MARK_EVERY iterations, staggered per condition
MARK_EVERY = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="outputs/tits26/paper_figs")
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument(
        "--with_geometry_ablation",
        action="store_true",
        help="Also draw A-Geometry (bh_capsule), before BH-CBF.",
    )
    return parser.parse_args()


def pretrain_curve(window: int, per_km: bool):
    """Mean and standard deviation over the runs of both maps of the TTC
    pre-training collisions per agent-minute or per km (smoothed per run as in
    stage1_plot_training.py)."""
    key = "collisions_per_km" if per_km else "collisions"
    series = np.array(
        [
            moving_average(run[key], window)[: PRETRAIN_ITERS + 1]
            for run in load_runs(PRETRAIN_ROOT)
            if run["reward"] == "ttc"
        ]
    )
    return np.arange(series.shape[1]), series.mean(0), series.std(0, ddof=1)


def finetune_runs(method: str) -> list:
    return sorted(glob.glob(os.path.join(FINETUNE, "*__iter100", "*", method, "seed1")))


def training_collisions(method: str, window: int, per_km: bool):
    """Collisions of the training rollouts per agent-minute or per km, per
    iteration (smoothed per run), mean and standard deviation over runs; without
    the excluded lane-boundary steps of settings.STAGE2_EXCLUDED_LANE_STEPS."""
    series = []
    for run in finetune_runs(method):
        with open(os.path.join(run, "train_stats.csv")) as file:
            rows = list(csv.DictReader(file))
        name = os.path.relpath(run, FINETUNE)
        values = np.array(
            [
                (
                    float(r["agent_collision_steps"])
                    + float(r["lane_collision_steps"])
                    - STAGE2_EXCLUDED_LANE_STEPS.get((name, int(r["iter"])), 0)
                )
                / (
                    float(r["agent_steps"]) * DT * float(r["mean_speed"]) / 1000.0
                    if per_km
                    else float(r["agent_steps"]) * DT / 60.0
                )
                for r in rows
            ]
        )
        series.append(moving_average(values, window))
    series = np.array(series)
    iters = np.arange(1, series.shape[1] + 1)
    return iters, series.mean(0), series.std(0, ddof=1)


def deployment_collisions(method: str, per_km: bool):
    """Colliding vehicles per environment-minute or per km without a filter
    at the evaluated iterations (seed 201), mean and standard error over
    runs."""
    per_iter = {}
    for run in finetune_runs(method):
        with open(os.path.join(run, "run_config.json")) as file:
            n_agents = json.load(file)["n_agents"]
        with open(os.path.join(run, "eval_nofilter_seed201.jsonl")) as file:
            for line in file:
                entry = json.loads(line)
                if entry["iter"] % 20:
                    continue
                env_seconds = entry["num_envs"] * entry["duration_s"]
                exposure = (
                    env_seconds * n_agents * entry["mean_speed"] / 1000.0
                    if per_km
                    else env_seconds / 60.0
                )
                per_iter.setdefault(entry["iter"], []).append(
                    entry["collision_events"] / exposure
                )
    iters = np.array(sorted(per_iter))
    values = [np.array(per_iter[i]) for i in iters]
    means = np.array([v.mean() for v in values])
    ses = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in values])
    return iters, means, ses


def plot(name: str, window: int, per_km: bool, conditions: list):
    """Write the three subfigure files (a), (b), and legend under name."""
    # Three files that the paper places as subfigures: (a) training rollouts
    # over pre-training and fine-tuning (full width), (b) deployment without a
    # filter (half width), and the legend (the other half)
    ticks = range(0, PRETRAIN_ITERS + 1, 20)
    unit_a, unit_b = ("km", "km") if per_km else ("agent-min", "env-min")

    fig_train = plt.figure(figsize=(3.5, 1.32))
    # Per km, the wider tick labels and the top tick label need more room
    ax_train = fig_train.add_axes(
        [0.145, 0.25, 0.83, 0.69] if per_km else [0.145, 0.25, 0.83, 0.72]
    )
    iters, mean, sd = pretrain_curve(window, per_km)
    ax_train.plot(iters, mean, color=BLACK, lw=0.9, label="Pre-training (TTC reward)")
    ax_train.fill_between(iters, mean - sd, mean + sd, color=BLACK, alpha=0.2, lw=0)
    for index, (method, label, color, style, width_, marker) in enumerate(conditions):
        it, m, s = training_collisions(method, window, per_km)
        x = PRETRAIN_ITERS + it
        ax_train.plot(x, m, color=color, ls=style, lw=width_, label=label,
                      marker=marker, ms=MARKER_SIZE, mew=0.7, mec=color,
                      markevery=(4 + 3 * index, MARK_EVERY))  # fmt: skip
        ax_train.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)
    if per_km:
        # Linear below 1/km, so that the curves at zero stay visible
        ax_train.set_yscale("symlog", linthresh=1.0, linscale=0.6)
        # Same top as the collisions per km of fig_stage1_pretrain
        y_top = 100.0
        ax_train.set_ylim(-0.3, y_top)
        ax_train.set_yticks([0, 1, 10, 100])
        ax_train.set_yticklabels(
            ["0"] + [f"$\\mathdefault{{10^{{{k}}}}}$" for k in range(3)]
        )
        ax_train.yaxis.set_minor_locator(plt.NullLocator())
    else:
        # The curves at zero sit slightly above the x-axis, so that their markers show
        y_top = 1.38
        ax_train.set_ylim(-0.07, y_top)
        ax_train.set_yticks([0, 0.5, 1.0])
    ax_train.axvline(PRETRAIN_ITERS, color="0.6", lw=0.5, zorder=0)
    ax_train.set_xlim(0, 2 * PRETRAIN_ITERS)
    all_ticks = list(ticks) + [PRETRAIN_ITERS + t for t in ticks[1:]]
    ax_train.set_xticks(all_ticks)
    ax_train.set_xticklabels(
        [str(t) if t < PRETRAIN_ITERS else "100|0" if t == PRETRAIN_ITERS
         else str(t - PRETRAIN_ITERS) for t in all_ticks]
    )  # fmt: skip
    ax_train.set_xlabel("Iteration", labelpad=1)
    # One line per km as in (b); "[1/agent-min]" is too long for one line
    ax_train.set_ylabel(f"Collisions{' ' if per_km else '\n'}[1/{unit_a}]")
    ax_train.text(PRETRAIN_ITERS * 0.97, y_top, "Stage 1: pre-training", ha="right",
                  va="top", fontsize=7, color="0.35")  # fmt: skip
    ax_train.text(PRETRAIN_ITERS * 1.03, y_top, "Stage 3: fine-tuning", ha="left",
                  va="top", fontsize=7, color="0.35")  # fmt: skip

    fig_eval = plt.figure(figsize=(1.75, 1.32))
    ax_eval = fig_eval.add_axes([0.25, 0.25, 0.68, 0.72])
    for method, label, color, style, width_, marker in conditions:
        it, m, s = deployment_collisions(method, per_km)
        ax_eval.plot(it, m, color=color, ls=style, lw=width_, marker=marker,
                     ms=MARKER_SIZE, mew=0.7, mec=color)  # fmt: skip
        ax_eval.fill_between(it, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax_eval.set_ylim(bottom=0)
    ax_eval.set_xlim(0, PRETRAIN_ITERS)
    ax_eval.set_xticks(ticks)
    ax_eval.set_xlabel("Fine-tuning iteration", labelpad=1)
    ax_eval.set_ylabel(f"Collisions [1/{unit_b}]")

    for ax in (ax_train, ax_eval):
        ax.grid(axis="y", color="0.9", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig_legend = plt.figure(figsize=(1.75, 1.32))
    handles, labels = ax_train.get_legend_handles_labels()
    # Centered vertically on the axes of (b) and, as far as its width allows,
    # horizontally under the pre-training part of (a)
    legend = fig_legend.legend(handles[::-1], labels[::-1], loc="center",
                               frameon=False, handlelength=2.2, labelspacing=0.3,
                               borderaxespad=0.0, fontsize=6.5)  # fmt: skip
    fig_legend.canvas.draw()
    half_width = legend.get_window_extent().width / fig_legend.dpi / 2 / 1.75
    center_x = min((0.145 + 0.83 / 4) * 3.5 / 1.75, 1.0 - half_width - 0.01)
    legend.set_bbox_to_anchor((center_x, 0.25 + 0.72 / 2))

    # The legend is the same in both versions
    figures = [(fig_train, "a"), (fig_eval, "b")]
    if per_km:
        plt.close(fig_legend)
    else:
        figures.append((fig_legend, "legend"))
    for fig, part in figures:
        path = f"{name}_{part}" + ("_per_km" if per_km else "")
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved {path}.pdf")


def main():
    args = parse_args()
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
        }
    )
    os.makedirs(args.out, exist_ok=True)
    name = os.path.join(args.out, "fig_stage12_collisions")
    conditions = CONDITIONS
    if args.with_geometry_ablation:
        conditions = [*CONDITIONS[:-1], GEOMETRY_ABLATION, CONDITIONS[-1]]
    for _, per_km in VERSIONS:
        plot(name, args.smooth, per_km, conditions)


if __name__ == "__main__":
    main()
