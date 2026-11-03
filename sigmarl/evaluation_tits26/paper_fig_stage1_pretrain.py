# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Paper figure of Stage 1: collisions and speed of the training rollouts
over pre-training, per map and safety reward.

Reads curves.csv of stage1_plot_training.py (per map, reward, and iteration:
mean and standard error over the 5 training seeds of the trailing moving
average) and plots the collisions (top) and the speed relative to the speed
limit (bottom), one column per map. It writes two versions that differ in the
collision unit: per agent-minute (fig_stage1_pretrain) and per km driven along
the route (fig_stage1_pretrain_per_km, log scale). The per-minute rate also
falls when vehicles slow down, e.g., the near-standstill policy at iteration 0
collides rarely per minute but often per km.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.paper_fig_stage1_pretrain \\
        [--out outputs/tits26/paper_figs]
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES = "outputs/tits26/stage1_pretrain/curves.csv"
MAPS = [("round_location0", "Roundabout"), ("ind_location4", "Intersection")]
SPEED = ("speed_ratio", "Speed /\nspeed limit")
SPEED_TICKS = [0.0, 0.2, 0.4, 0.6]
# Figure versions: (file suffix, collision metric, y-label, log scale, y-ticks;
# the outer ticks are the y-limits)
VERSIONS = [
    ("", "collisions", "Collisions\n[1/agent-min]", False, [0.0, 0.5, 1.0, 1.5]),
    ("_per_km", "collisions_per_km", "Collisions\n[1/km]", True, [0.1, 1, 10, 100]),
]
# Rewards: (key, label, color, line style, line width, marker), RWTH colors;
# TTC in black as in fig_stage12_collisions
REWARDS = [
    ("ttc", "TTC", "#000000", (0, (3, 1.5)), 0.9, "x"),
    ("onestep", "One-Step CBF", "#A11135", "-", 0.9, "s"),
    ("bh", "BH-CBF (ours)", "#00549F", "-", 1.4, "o"),
]
MARKER_SIZE = 2.6
# Markers every MARK_EVERY iterations, staggered per reward
MARK_EVERY = 100
# Iteration of the unsafe source policies of Stages 1 and 2 (TTC reward)
SOURCE_ITER = 100


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="outputs/tits26/paper_figs")
    return parser.parse_args()


def load_curves() -> dict:
    """{(map, reward, metric): (iterations, mean, se)} from curves.csv."""
    rows = {}
    with open(CURVES) as file:
        for row in csv.DictReader(file):
            key = (row["map"], row["reward"], row["metric"])
            rows.setdefault(key, []).append(
                (int(row["iteration"]), float(row["mean"]), float(row["se"]))
            )
    return {
        key: tuple(np.array(v).T)
        for key, v in ((k, sorted(r)) for k, r in rows.items())
    }


def plot(curves: dict, collision: tuple, path: str):
    """Collisions (top, given as (metric, y-label, log scale, y-ticks)) and
    speed (bottom) over the iterations, one column per map."""
    metric_c, ylabel_c, is_log, yticks_c = collision
    metrics = [(metric_c, ylabel_c), SPEED]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.3), sharex="col", sharey="row")
    for col, (map_name, title) in enumerate(MAPS):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for index, (reward, label, color, style, width, marker) in enumerate(
                REWARDS
            ):
                it, mean, se = curves[(map_name, reward, metric)]
                ax.plot(it, mean, color=color, ls=style, lw=width, label=label,
                        marker=marker, ms=MARKER_SIZE, mew=0.7, mec=color,
                        markevery=(20 + 30 * index, MARK_EVERY))  # fmt: skip
                ax.fill_between(it, mean - se, mean + se, color=color, alpha=0.15, lw=0)
            ax.set_xlim(0, 500)
            ax.set_xticks(range(0, 501, 100))
            if row == 0 and is_log:
                ax.set_yscale("log")
            ax.grid(axis="y", color="0.9", lw=0.4)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=8, pad=2)
                # Point to the source policies on the TTC curve, gray as the
                # labels in fig_stage12_collisions
                it, mean, _ = curves[(map_name, "ttc", metric)]
                source = mean[it == SOURCE_ITER][0]
                ax.vlines(SOURCE_ITER, yticks_c[0], source, color="0.6", lw=0.5, ls=":",
                          zorder=0)  # fmt: skip
                ax.annotate("Unsafe source policies",
                            xy=(SOURCE_ITER, source),
                            xytext=(SOURCE_ITER + 20, 0.86), textcoords=("data", "axes fraction"),
                            ha="left", va="top", fontsize=7, color="0.35",
                            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.5,
                                            mutation_scale=5, shrinkA=1, shrinkB=1.5,
                                            relpos=(0.15, 0)))  # fmt: skip
            if row == 1:
                ax.set_xlabel("Pre-training iteration", labelpad=1)
            if col == 0:
                ax.set_ylabel(ylabel)
    for ax, ticks in ((axes[0, 0], yticks_c), (axes[1, 0], SPEED_TICKS)):
        ax.set_ylim(ticks[0], ticks[-1])
        ax.set_yticks(ticks)
        if not is_log or ax is axes[1, 0]:
            ax.set_yticklabels([f"{t:.1f}" for t in ticks])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(REWARDS), frameon=False,
               handlelength=2.2, columnspacing=1.2, borderaxespad=0.1)  # fmt: skip
    fig.subplots_adjust(
        left=0.16, right=0.965, bottom=0.15, top=0.85, wspace=0.2, hspace=0.22
    )
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
    curves = load_curves()
    os.makedirs(args.out, exist_ok=True)
    for suffix, *collision in VERSIONS:
        plot(curves, collision, os.path.join(args.out, f"fig_stage1_pretrain{suffix}"))


if __name__ == "__main__":
    main()
