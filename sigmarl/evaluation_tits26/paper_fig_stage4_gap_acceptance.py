# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Paper figure of Stage 4: gap acceptance of the recorded humans and the BH-CBF
fine-tuned policies at the roundabout entries.

Reads events.csv of stage3_gap_acceptance.py and writes three subfigures that
share one column, (a) on the left and (b) above (c) on the right: (a) a recorded
entry decision with the entering and the circulating vehicle, the estimated merge
point, and the arc distance, trimmed to the north entry and annotated in place of
a legend (_a_full: the same over the whole roundabout, for slides); (b) the speed
of the recorded humans over the route distance to the merge point, whose lowest
median sets the decision distance D_REF; (c) the probability that an entering
vehicle enters the ring before the circulating vehicle over the time to arrival
of the circulating vehicle, with the fit and style of fig_acceptance in
stage3_gap_acceptance_plots.py.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.paper_fig_stage4_gap_acceptance \\
        [--out outputs/tits26/paper_figs]
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from sigmarl.evaluation_human_comparison.metrics import RING_CENTER
from sigmarl.evaluation_tits26.paper_fig_headline import RWTH_PALETTE
from sigmarl.evaluation_tits26.stage3_gap_acceptance import (
    D_REF, POLICY_LABEL, RESULTS, human_trajectories, route_geometry,
)  # fmt: skip
from sigmarl.evaluation_tits26.stage3_gap_acceptance_plots import (
    HUMAN, OUT, POLICY, acceptance_panel, decisions, draw_definition_scene, load,
)  # fmt: skip
from sigmarl.parse_lanelet2 import ParseLanelet2  # Imported before main() sets the rcParams, which it overrides
from sigmarl.round_tracks import RouteMatcher

# Subfigure sizes [in]: (a) 0.40 \columnwidth (3.5 in) wide on the left; (b) and (c) 0.58 \columnwidth wide,
# stacked on the right, so that (a) is as tall as (b) and (c) with the caption of (b) between them
WIDTH_A = 0.40 * 3.5
WIDTH_R = 0.58 * 3.5
HEIGHT_B = 1.0
HEIGHT_C = 1.35
D_MAX = 20.0  # [m] Route distance before the merge point shown in (b)
MARGIN = dict(left=0.40, right=0.04, bottom=0.31, top=0.03)  # [in] Axes margins of (b) and (c), aligned
YLABEL = (-0.15, 0.5)  # Position of the y labels of (b) and (c) in axes coordinates, aligned
# Windows of (a) relative to the ring center [m]: trimmed for the paper, and the whole roundabout (slides), at
# the same scale so that both look the same
X_A = (-12.0, 26.0)
Y_A = (-33.0, 37.0)
X_FULL = (-38.0, 38.0)
Y_FULL = (-33.0, 43.0)
# White halo keeps the labels readable over the dashed circles
ANNOTATION = dict(fontsize=7, color="0.15", zorder=7, path_effects=[patheffects.withStroke(linewidth=1.5, foreground="w")])
LEADER = dict(arrowstyle="-", color="0.35", lw=0.4, shrinkA=0.5, shrinkB=1.5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="outputs/tits26/paper_figs")
    return parser.parse_args()


def plot_scene(events, path, window=(X_A, Y_A)):
    """(a): the recorded decision, annotated, in the window (x, y) relative to the ring center."""
    scale = WIDTH_A / (X_A[1] - X_A[0])  # [in/m]
    (x0, x1), (y0, y1) = window
    fig = plt.figure(figsize=(scale * (x1 - x0), scale * (y1 - y0)))
    ax = fig.add_axes([0, 0, 1, 1])
    scene = draw_definition_scene(ax, events, OUT, chevron_lw=0.6, star_size=7, arc_lw=2.0)
    ax.set_xlim(RING_CENTER[0] + x0, RING_CENTER[0] + x1)
    ax.set_ylim(RING_CENTER[1] + y0, RING_CENTER[1] + y1)
    ax.set_aspect("equal", adjustable="datalim")
    i, j, point = scene["poses"]["i"][:2], scene["poses"]["j"][:2], scene["point"]
    on_ring = lambda radius, deg: RING_CENTER + radius * np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    mid = on_ring(20.5, np.degrees(scene["arc"][40]))
    route = scene["route"][np.argmin(np.linalg.norm(scene["route"] - (point + i) / 2, axis=1))]
    labels = [
        # (text, point, text position, color): blue and green as the entering and the circulating vehicle
        ("Entering\nvehicle", i + [1.5, 0.0], i + [10.5, 1.5], POLICY[0]),
        ("Reference\nroute", route, route + [-9.5, 3.0], POLICY[0]),
        ("Merge\npoint", point + [-0.8, -0.8], point + [-4.0, -8.5], "0.15"),
        ("Arc\ndistance", mid, RING_CENTER + [15.5, 24.0], RWTH_PALETTE[3][0]),
        ("Circulating\nvehicle", j + [-1.0, 1.5], RING_CENTER + [7.8, 9.8], RWTH_PALETTE[3][0]),
        ("Centerline", on_ring(20.5, 0.0), RING_CENTER + [9.5, 0.5], "0.15"),
        ("Outer centerline", on_ring(23.0, -12.0), RING_CENTER + [8.5, -4.8], "0.15"),
    ]
    for text, xy, xytext, color in labels:
        ax.annotate(text, xy=xy, xytext=xytext, ha="center", va="center", arrowprops=dict(LEADER, shrinkB=0),
                    **dict(ANNOTATION, color=color))  # fmt: skip
    save(fig, path)


def panel_axes(height):
    """Figure of the right column with axes inside MARGIN."""
    fig = plt.figure(figsize=(WIDTH_R, height))
    m = MARGIN
    return fig, fig.add_axes([m["left"] / WIDTH_R, m["bottom"] / height, 1 - (m["left"] + m["right"]) / WIDTH_R,
                              1 - (m["bottom"] + m["top"]) / height])  # fmt: skip


def approach_speed():
    """Speeds [n, len(D)] of the recorded humans that drive through the ring when they first reach the route
    distance D before their merge point; only vehicles recorded over the whole last D_MAX, so that every
    distance shows the same vehicles. Cached in OUT."""
    path = os.path.join(OUT, "approach_speed.npz")
    if os.path.exists(path):
        cache = np.load(path)
        return cache["D"], cache["speed"]
    geometry = route_geometry(RouteMatcher(ParseLanelet2("round_location0", "cpu").reference_paths).routes)
    with open(os.path.join(RESULTS, "natural", "matched_tracks.pkl"), "rb") as file:
        data = pickle.load(file)
    # The windows (specs) are the same for all policies; take them from the first natural policy
    folder = os.path.join(RESULTS, "natural", "homogeneous", POLICY_LABEL.format(map="round_location0", seed=1))
    D = np.arange(0.0, D_MAX + 1e-6, 0.1)
    speed = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".pkl"):
            continue
        with open(os.path.join(folder, name), "rb") as file:
            spec = pickle.load(file)["spec"]
        for vehicle in human_trajectories(spec, data[spec.recording_id]):
            g = geometry.get(vehicle["route"])
            d = None if g is None else g["s_merge"] - vehicle["s"]
            if d is None or d[0] < D_MAX or d.min() > 0:
                continue
            speed.append([vehicle["v"][np.flatnonzero(d <= x)[0]] for x in D])
    np.savez(path, D=D, speed=np.array(speed))
    return D, np.array(speed)


def plot_speed(path):
    """(b): speed of the humans over the route distance to the merge point; its lowest median is D_REF."""
    D, speed = approach_speed()
    q1, med, q3 = np.percentile(speed, [25, 50, 75], axis=0)
    k = int(np.argmin(med))
    print(f"[INFO] Lowest median speed of {len(speed)} humans: {med[k]:.2f} m/s at {D[k]:.1f} m (D_REF = {D_REF} m)")
    fig, ax = panel_axes(HEIGHT_B)
    ax.fill_between(D, q1, q3, color=HUMAN[1], alpha=0.5, lw=0)
    ax.plot(D, med, color=HUMAN[0], lw=1.0)
    ax.axvline(D[k], color="0.35", lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.plot(D[k], med[k], marker="o", ms=2.5, color=HUMAN[0], zorder=5)
    ax.text(D[k] - 0.4, 0.5, f"{D[k]:.1f} m", ha="left", va="bottom", color=HUMAN[0], fontsize=6.5)
    ax.set_xlim(D_MAX, 0)
    ax.set_ylim(0, 8.5)
    ax.set_xticks(np.arange(D_MAX, -1, -5))
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Distance to the merge point [m]", labelpad=1)
    ax.set_ylabel("Speed [m/s]")
    ax.yaxis.set_label_coords(*YLABEL)
    save(fig, path)


def plot_acceptance(events, path):
    """(c): probability of entering first over the time to arrival."""
    fig, ax = panel_axes(HEIGHT_C)
    critical = acceptance_panel(ax, decisions(events), "tta", np.random.default_rng(1), legend=False)
    # Critical values next to their dotted lines, with a short horizontal mark at P = 0.5; the legend names only
    # the groups
    for group, color in (("policy", POLICY[0]), ("human", HUMAN[0])):
        ax.plot(critical[group], 0.5, marker="_", ms=6, mew=0.9, color=color, zorder=5)
    ax.text(critical["policy"] + 0.12, 0.4, f"{critical['policy']:.2f} s", ha="left", va="center",
            color=POLICY[0], fontsize=6.5)  # fmt: skip
    ax.text(critical["human"] + 0.12, 0.4, f"{critical['human']:.2f} s", ha="left", va="center",
            color=HUMAN[0], fontsize=6.5)  # fmt: skip
    handles = ax.get_lines()
    ax.legend([h for h in handles if h.get_label().startswith(("Humans", "BH-CBF"))], ["Humans", "BH-CBF (ours)"],
              loc="lower right", frameon=False, fontsize=6.5, handlelength=1.2, handletextpad=0.5,
              borderaxespad=0.1, labelspacing=0.25)  # fmt: skip
    ax.set_xlabel("Time to arrival [s]", labelpad=1)
    ax.set_ylabel("Prob. of entering first")
    ax.yaxis.set_label_coords(*YLABEL)
    save(fig, path)


def save(fig, path):
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
    events = load(os.path.join(OUT, "events.csv"))
    os.makedirs(args.out, exist_ok=True)
    plot_scene(events, os.path.join(args.out, "fig_stage4_gap_acceptance_a"))
    plot_scene(events, os.path.join(args.out, "fig_stage4_gap_acceptance_a_full"), window=(X_FULL, Y_FULL))
    plot_speed(os.path.join(args.out, "fig_stage4_gap_acceptance_b"))
    plot_acceptance(events, os.path.join(args.out, "fig_stage4_gap_acceptance_c"))


if __name__ == "__main__":
    main()
