# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 1 of the T-ITS evaluation: learning curves and convergence check of
pre-training.

Reads the per-iteration data of every run under --root
(<map>/stage1_<reward>_seed_<k>/reward*_data.json) and writes to --out:

- training_curves.pdf: per map, the collision rate per agent-minute and per
  km driven along the route, and the speed ratio of the training rollouts over
  the iterations (mean and standard error over seeds, moving average), one
  line per reward. These metrics do not depend on the training reward
  (MAPPOCAVs._compute_task_metrics);
- training_return.pdf: the training return (not comparable between rewards);
- curves.csv: the plotted values;
- convergence.csv: per run, the mean over the last --block iterations against
  the highest mean over any --block consecutive iterations, for the return and
  the task reward; a run passes if the gap is at most --tol of the highest
  mean.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.stage1_plot_training \\
        [--root checkpoints/tits26/stage1_pretrain] [--out outputs/tits26/stage1_pretrain]
"""

import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Rewards in a fixed order with fixed colors and line styles
REWARDS = {
    "ttc": ("TTC", "#2a78d6", ":"),
    "onestep": ("One-Step CBF", "#eb6834", "--"),
    "bh": ("BH-CBF (ours)", "#1baf7a", "-"),
}
MAP_NAMES = {
    "round_location0": "rounD (roundabout)",
    "ind_location4": "inD (intersection)",
}
# Plotted metrics: (key, axis label)
METRICS = [
    ("collisions", "Collisions [1/agent-min]"),
    ("collisions_per_km", "Collisions [1/km]"),
    ("speed_ratio", "Speed / speed limit"),
]
# Also in curves.csv: the task reward (progress + sparse terms per agent-second,
# dominated by the progress term) and the goal rate (raised by collisions,
# since a colliding vehicle respawns at a random point of its route)
CSV_METRICS = ["return", "task_reward", "goals"] + [key for key, _ in METRICS]
# Metrics plotted on a log scale
LOG_METRICS = {"collisions_per_km"}
RUN_PATTERN = re.compile(r"stage1_(?P<reward>[a-z]+)_seed_(?P<seed>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default="checkpoints/tits26/stage1_pretrain")
    parser.add_argument("--out", default="outputs/tits26/stage1_pretrain")
    parser.add_argument(
        "--smooth", type=int, default=10, help="Moving-average window [iterations]."
    )
    parser.add_argument(
        "--block", type=int, default=50, help="Window of the convergence check."
    )
    parser.add_argument("--tol", type=float, default=0.05)
    return parser.parse_args()


def load_runs(root: str) -> list:
    """One dict per run with its per-iteration series."""
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(root, "*", "stage1_*"))):
        match = RUN_PATTERN.search(os.path.basename(run_dir))
        files = glob.glob(os.path.join(run_dir, "reward*_data.json"))
        if match is None or not files:
            continue
        # Every iteration rewrites the data file of the best model, so the
        # newest one holds the whole history
        with open(max(files, key=os.path.getmtime)) as file:
            data = json.load(file)
        metrics = data["task_metrics_list"]
        collisions = np.array(
            [m["veh_collisions"] + m["lane_collisions"] for m in metrics]
        )
        speed_ratio = np.array([m["speed_ratio"] for m in metrics])
        # Route distance per agent-minute [km]
        km_per_agent_min = speed_ratio * data["parameters"]["max_speed"] * 60 / 1000
        runs.append(
            {
                "map": os.path.basename(os.path.dirname(run_dir)),
                "reward": match["reward"],
                "seed": int(match["seed"]),
                "finished": os.path.exists(os.path.join(run_dir, "final_policy.pth")),
                "return": np.array(data["episode_reward_mean_list"], dtype=float),
                "task_reward": np.array([m["task_reward"] for m in metrics]),
                "collisions": collisions,
                "collisions_per_km": collisions / km_per_agent_min,
                "speed_ratio": speed_ratio,
                "goals": np.array([m["goals"] for m in metrics]),
            }
        )
    return runs


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average (shorter window at the start)."""
    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(1, len(values) + 1)
    start = np.maximum(idx - window, 0)
    return (cumsum[idx] - cumsum[start]) / (idx - start)


def aggregate(runs: list, key: str, window: int):
    """Mean and standard error over seeds of the smoothed series, over the
    iterations that all runs have reached."""
    series = [moving_average(run[key], window) for run in runs]
    length = min(len(s) for s in series)
    stacked = np.stack([s[:length] for s in series])
    se = (
        stacked.std(axis=0, ddof=1) / np.sqrt(len(series))
        if len(series) > 1
        else np.zeros(length)
    )
    return stacked.mean(axis=0), se


def convergence(values: np.ndarray, block: int) -> tuple:
    """(mean of the last block, highest block mean, relative gap)."""
    if len(values) < block:
        return float("nan"), float("nan"), float("nan")
    block_means = np.convolve(values, np.ones(block) / block, mode="valid")
    last, best = float(block_means[-1]), float(block_means.max())
    return last, best, (best - last) / abs(best)


def direct_labels(ax, labels: list):
    """Label each line at its end, pushed apart vertically so that the labels
    do not overlap."""
    is_log = ax.get_yscale() == "log"
    to_axis = np.log10 if is_log else (lambda y: y)
    y_low, y_high = (to_axis(y) for y in ax.get_ylim())
    gap = 0.07 * (y_high - y_low)
    labels = sorted([to_axis(y), x, name] for y, x, name in labels)
    for k in range(1, len(labels)):
        labels[k][0] = max(labels[k][0], labels[k - 1][0] + gap)
    if is_log:
        labels = [[10**y, x, name] for y, x, name in labels]
    for y, x, name in labels:
        ax.annotate(
            name,
            (x, y),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=6,
            color="#52514e",
            va="center",
        )


def plot(groups: dict, maps: list, metrics: list, window: int, path: str):
    """One column per map, one row per metric, one line per reward."""
    fig, axes = plt.subplots(
        len(metrics),
        len(maps),
        figsize=(3.4 * len(maps), 2.1 * len(metrics)),
        squeeze=False,
        sharex=True,
    )
    for col, map_name in enumerate(maps):
        for row, (key, label) in enumerate(metrics):
            ax = axes[row, col]
            labels = []
            for reward, (name, color, style) in REWARDS.items():
                runs = groups.get((map_name, reward))
                if not runs:
                    continue
                mean, se = aggregate(runs, key, window)
                it = np.arange(len(mean))
                ax.plot(it, mean, color=color, linestyle=style, linewidth=1.5)
                ax.fill_between(it, mean - se, mean + se, color=color, alpha=0.18, lw=0)
                labels.append([float(mean[-1]), it[-1], name])
            if key in LOG_METRICS:
                ax.set_yscale("log")
            direct_labels(ax, labels)
            ax.grid(color="#e4e3df", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            if row == 0:
                ax.set_title(MAP_NAMES.get(map_name, map_name), fontsize=8)
            if row == len(metrics) - 1:
                ax.set_xlabel("Iteration", fontsize=7)
    handles = [
        plt.Line2D([], [], color=color, linestyle=style, linewidth=1.5)
        for _, color, style in REWARDS.values()
    ]
    fig.legend(
        handles,
        [name for name, _, _ in REWARDS.values()],
        loc="upper center",
        ncol=len(REWARDS),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No runs under '{args.root}'.")
    os.makedirs(args.out, exist_ok=True)
    groups = {}
    for run in runs:
        groups.setdefault((run["map"], run["reward"]), []).append(run)
    maps = [name for name in MAP_NAMES if any(run["map"] == name for run in runs)]

    plot(
        groups,
        maps,
        METRICS,
        args.smooth,
        os.path.join(args.out, "training_curves.pdf"),
    )
    plot(
        groups,
        maps,
        [("return", "Training return")],
        args.smooth,
        os.path.join(args.out, "training_return.pdf"),
    )

    with open(os.path.join(args.out, "curves.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["map", "reward", "metric", "iteration", "mean", "se", "n_seeds"]
        )
        for (map_name, reward), group in sorted(groups.items()):
            for key in CSV_METRICS:
                mean, se = aggregate(group, key, args.smooth)
                for it, (m, s) in enumerate(zip(mean, se)):
                    writer.writerow(
                        [map_name, reward, key, it, f"{m:.5f}", f"{s:.5f}", len(group)]
                    )

    rows = []
    for run in sorted(runs, key=lambda r: (r["map"], r["reward"], r["seed"])):
        row = {
            "map": run["map"],
            "reward": run["reward"],
            "seed": run["seed"],
            "iterations": len(run["return"]),
            "finished": run["finished"],
        }
        for key in ("return", "task_reward"):
            last, best, gap = convergence(run[key], args.block)
            row.update(
                {
                    f"{key}_last": round(last, 4),
                    f"{key}_best": round(best, 4),
                    f"{key}_gap": round(gap, 4),
                    f"{key}_ok": bool(gap <= args.tol),
                }
            )
        rows.append(row)
    with open(os.path.join(args.out, "convergence.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Convergence (last {args.block} iterations vs. best, tolerance {args.tol:.0%}):"
    )
    for row in rows:
        print(
            f"  {row['map']:16s} {row['reward']:8s} seed {row['seed']} "
            f"({row['iterations']} it): return gap {row['return_gap']:+.1%}, "
            f"task reward gap {row['task_reward_gap']:+.1%}"
            f"{'' if row['return_ok'] and row['task_reward_ok'] else '  <-- not converged'}"
        )
    print(
        f"Wrote {args.out}/training_curves.pdf, training_return.pdf, curves.csv, convergence.csv"
    )


if __name__ == "__main__":
    main()
