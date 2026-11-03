# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Paper figure of Stage 3: footprints of the BH-CBF-filtered policy and of
the human drivers in the same recorded traffic, four scenes (a)-(d).

Each scene is one time span of one Stage 3 episode (stage3_human_comparison.py,
final run in outputs/tits26/stage3_human_ft_recorded_maxlen6.5): the key
vehicles as driven by the fine-tuned policy with the decentralized BH-CBF
filter (filled footprints) and as recorded (outlined footprints, recorded
size), in the same color per vehicle, every second; transparency encodes time
as in the Stage 1 footprint figures, and a chevron marks the heading of the
last footprint. Every vehicle starts at its recorded state, so both
footprints of a vehicle coincide at its start if it appears at a footprint
time. Other vehicles are not drawn.

Each footprint carries its time stamp: a thin leader line (solid for the
policy, dashed for the human) from the side of the footprint to the label,
on the side given per series (vehicle and driver) in the scene's "labels",
which also shift single labels where they would overlap. The footprints of
a standing vehicle share one label, and so do coinciding footprints of the
policy and the human (labeled by the policy).

The panels are drawn at full column width with large text and lines, since
the paper shows them at half size (2 x 2).

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.paper_fig_stage3_footprints [--scenes a b c d]
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon

from sigmarl.evaluation_tits26.paper_fig_headline import (
    RWTH_PALETTE,
    heading_chevron,
    rectangle,
)
from sigmarl.evaluation_human_comparison.analyze import controlled_size
from sigmarl.evaluation_tits26.paper_fig_stage1_counterfactual import time_key
from sigmarl.evaluation_tits26.paper_fig_stage1_footprints import (
    ALPHA_MIN,
    HALF,
    draw_road,
)

RESULTS = "outputs/tits26/stage3_human_ft_recorded_maxlen6.5/natural"
CONTROLLER = (
    "round_location0__stage1_ttc_seed_{}__iter100__ft-round_location0-ours-seed1"
    "__iter100__ours_dec"
)
# Blue, green, purple, magenta
KEY_COLORS = [RWTH_PALETTE[i][0] for i in (0, 3, 2, 6)]
# Text sizes [pt] and line widths [pt] at full size (the paper halves them)
LABEL_SIZE = 14.0
TIME_SIZE = 11.0
KEY_SIZE = 12.0
EDGE = 0.8  # Filled footprints
OUTLINE = 1.2  # Human footprints
LEADER_WIDTH = 0.7
ROAD_WIDTH = 0.8
LEADER = 2.6  # Length of the leader lines [m]
SHARED = 1.0  # Distance [m] below which both footprints of a vehicle share one label
LABEL_EVERY = 2  # Time stamps every LABEL_EVERY seconds, on the first footprint, and on standing groups
WINDOW = (64.0, 64.0)  # Width and height of every panel [m]
FIGSIZE = (3.4, 3.4 * WINDOW[1] / WINDOW[0])
# Scenes: source policy seed, start time t0 [s] and duration [s] of the time
# span, key vehicles (track ids, numbered 1, 2, ... in this order), lower left
# corner [m] of the window, offsets [m] of the number labels from the first
# footprints, and the time stamps per (track id, "c" policy or "h" human):
# side of the leader lines (+1 left, -1 right of the heading); skipped
# times; groups (first, last) [s] of footprints under one label, in place
# of the standing groups; per time (the first of a group), the side, a shift
# (dx, dy) [m] of the label end, and the anchor of the leader along the
# footprint (fraction of the half length ahead of the middle); and the leader
# length [m]. Standing footprints share one label (standing_groups), and
# coinciding footprints of the policy and the human share the policy's
SCENES = {
    # Enters at once, ahead of a circulating vehicle that is still far away
    "a": dict(
        episode="rec11_h0303.3", seed=1, t0=311.72, duration=7.0, keys=[166, 164],
        corner=(33.0, 32.5), offsets=[(-3.0, -4.0), (-5.0, -2.5)],
        labels={
            (166, "c"): dict(side=-1, sides={1: 1, 2: 1, 6: 1}),
            (166, "h"): dict(side=-1, shift={2: (0.5, -1.5), 4: (0.5, -2.0)}),
            (164, "c"): dict(side=1, length=3.6, shift={0: (-3.0, -2.0)}),
            (164, "h"): dict(side=-1),
        },
        keys_box=dict(time=(42.5, 90.6, 7.0, 1.6), style=(35.0, 86.4)),
    ),
    # A queue of two follows a circulating vehicle into the ring
    "b": dict(
        episode="rec19_h0037.9", seed=1, t0=37.88, duration=7.0, keys=[63, 66, 59],
        corner=(40.0, 36.0), offsets=[(6.5, -7.0), (5.0, 1.5), (-5.0, -1.0)],
        labels={
            (63, "c"): dict(side=-1, sides={0: 1, 2: 1}),
            (63, "h"): dict(side=-1, groups=[(2, 7)], shift={2: (-1.0, -0.5)}),
            (66, "c"): dict(side=1, sides={2: -1}),
            (66, "h"): dict(side=-1, groups=[(3, 7)], shift={3: (-1.0, 1.0)}),
            (59, "c"): dict(side=1, sides={0: -1}),
            (59, "h"): dict(side=-1),
        },
    ),
    # Keeps its speed into the ring, ahead of a circulating vehicle
    "c": dict(
        episode="rec05_h0700.5", seed=1, t0=714.02, duration=7.0, keys=[462, 463],
        corner=(68.0, 27.0), offsets=[(0.0, 4.5), (6.0, 1.0)],
        labels={
            (462, "c"): dict(side=1, shift={4: (-2.0, 1.5)}),
            (462, "h"): dict(side=-1, shift={4: (2.0, 1.0)}),
            (463, "c"): dict(side=1, shift={6: (-0.5, -2.0)}),
            (463, "h"): dict(side=-1),
        },
    ),
    # Enters ahead of a circulating vehicle, which brakes briefly to let it in
    "d": dict(
        episode="rec11_h0067.9", seed=1, t0=67.98, duration=7.0, keys=[43, 37],
        corner=(50.0, 13.0), offsets=[(-4.5, -1.0), (-2.0, 4.5)],
        labels={
            (43, "c"): dict(side=-1),
            (43, "h"): dict(side=1),
            (37, "c"): dict(side=1, shift={6: (-1.5, 0.5)}),
            (37, "h"): dict(side=-1),
        },
    ),
}  # fmt: skip


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenes", nargs="+", default=list(SCENES), choices=list(SCENES)
    )
    parser.add_argument("--out", default="outputs/tits26/paper_figs")
    return parser.parse_args()


def controlled_pose(vehicle: dict, t: float):
    """Pose of a controlled vehicle at time t from its log (None if absent)."""
    log = vehicle["log"]
    times = np.asarray(log["t"])
    if len(times) == 0 or not times[0] - 1e-6 <= t <= times[-1] + 1e-6:
        return None
    i = int(np.argmin(np.abs(times - t)))
    return np.array([log["x"][i], log["y"][i], log["yaw"][i]])


def human_pose(track, t: float):
    """Recorded pose at time t (None outside the track)."""
    if not track.t_start - 1e-6 <= t <= track.t_end + 1e-6:
        return None
    return np.array(
        [
            np.interp(t, track.t, track.xy[:, 0]),
            np.interp(t, track.t, track.xy[:, 1]),
            np.interp(t, track.t, np.unwrap(track.heading)),
        ]
    )


def style_key(ax, xy: tuple, fontsize: float = KEY_SIZE):
    """Explains the two footprint styles: filled (BH-CBF) and outlined (human)."""
    x, y = xy
    for row, (label, filled) in enumerate((("BH-CBF (ours)", True), ("Human", False))):
        pose = np.array([x + HALF[0], y - 4.0 * row, 0.0])
        ax.add_patch(
            Polygon(rectangle(pose, HALF), fc="0.35" if filled else "none", ec="0.35",
                    lw=OUTLINE, zorder=9)
        )  # fmt: skip
        ax.text(x + 2 * HALF[0] + 1.2, pose[1], label, ha="left", va="center",
                fontsize=fontsize, zorder=9)  # fmt: skip


def time_label(
    ax,
    pose,
    half,
    side: int,
    text: str,
    color,
    shift=(0.0, 0.0),
    length: float = LEADER,
    anchor: float = 0.0,
    dashed: bool = False,
):
    """A leader line from a footprint's long side (left or right of its
    heading; at `anchor` times the half length ahead of the middle) outward,
    with the time stamp at its end; dashed for human drivers."""
    heading = np.array([np.cos(pose[2]), np.sin(pose[2])])
    normal = side * np.array([-heading[1], heading[0]])
    start = pose[:2] + half[1] * normal + anchor * half[0] * heading
    end = start + length * normal + np.asarray(shift)
    direction = end - start
    ax.plot(*np.stack([start, end]).T, color=color, lw=LEADER_WIDTH, zorder=6,
            linestyle=(0, (2.5, 1.5)) if dashed else "-")  # fmt: skip
    ha = "left" if direction[0] > 0.3 * abs(direction[1]) else (
        "right" if direction[0] < -0.3 * abs(direction[1]) else "center")  # fmt: skip
    va = "bottom" if direction[1] > 0 else "top"
    if ha != "center":
        va = "center"
    pad = 0.4 * direction / np.linalg.norm(direction)
    ax.text(*(end + pad), text, ha=ha, va=va, fontsize=TIME_SIZE, color=color, zorder=6)


def standing_groups(poses: dict, radius: float = 1.0) -> list:
    """(first, last) [s] of runs of consecutive footprints within `radius`
    [m] of the run's first one (a standing or creeping vehicle)."""
    times = sorted(poses)
    groups, start = [], 0
    for i in range(1, len(times) + 1):
        if (
            i == len(times)
            or np.linalg.norm(poses[times[i]][:2] - poses[times[start]][:2]) > radius
        ):
            if i - start > 1:
                groups.append((times[start], times[i - 1]))
            start = i
    return groups


def draw_series(
    ax, poses: dict, half, color, world: str, spec: dict, duration: float, shared: set
):
    """Footprints, last-heading chevron, and time stamps of one vehicle as
    driven by the policy (world "c") or the human ("h"). Standing footprints
    share one label; the human's labels at the `shared` times, where both
    footprints coincide, are left to the policy's."""

    def alpha(step):
        return ALPHA_MIN + (1 - ALPHA_MIN) * step / duration

    for s, pose in poses.items():
        if world == "c":
            patch = dict(fc=to_rgba(color, alpha(s)), ec=color, lw=EDGE, zorder=3)
        else:
            patch = dict(fc="none", ec=to_rgba(color, max(alpha(s), 0.3)), lw=OUTLINE,
                         zorder=4)  # fmt: skip
        ax.add_patch(Polygon(rectangle(pose, half), **patch))
    chevron = heading_chevron(poses[max(poses)], half)
    ax.plot(*chevron.T, color="white" if world == "c" else color, lw=EDGE,
            solid_joinstyle="miter", solid_capstyle="butt", zorder=5)  # fmt: skip
    length = spec.get("length", LEADER)
    skip = set(spec.get("skip", [])) | (shared if world == "h" else set())
    grouped = set()
    for lo, hi in spec.get("groups", standing_groups(poses)):
        members = [s for s in poses if lo <= s <= hi]
        grouped.update(members)
        if lo in skip:
            continue
        time_label(ax, poses[members[len(members) // 2]], half,
                   spec.get("sides", {}).get(lo, spec["side"]), f"{lo:g}\u2013{hi:g} s",
                   color, spec.get("shift", {}).get(lo, (0.0, 0.0)), length,
                   spec.get("anchor", {}).get(lo, 0.0), dashed=world == "h")  # fmt: skip
    for s, pose in poses.items():
        if s in grouped or s in skip or (s % LABEL_EVERY and s != min(poses)):
            continue
        time_label(ax, pose, half, spec.get("sides", {}).get(s, spec["side"]), f"{s:g} s",
                   color, spec.get("shift", {}).get(s, (0.0, 0.0)), length,
                   spec.get("anchor", {}).get(s, 0.0), dashed=world == "h")  # fmt: skip


def plot_scene(name: str, scene: dict, recordings: dict, out: str):
    path = os.path.join(
        RESULTS,
        "homogeneous",
        CONTROLLER.format(scene["seed"]),
        f"{scene['episode']}.pkl",
    )
    with open(path, "rb") as file:
        result = pickle.load(file)
    tracks = {t.track_id: t for t in recordings[result["spec"].recording_id].tracks}
    vehicles = {v["key"][1]: v for v in result["vehicles"]}
    steps = np.arange(0, scene["duration"] + 1e-6, 1.0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_road(ax, "round_location0")
    for patch in ax.patches:
        patch.set_linewidth(ROAD_WIDTH)
    for index in reversed(range(len(scene["keys"]))):
        key, color = scene["keys"][index], KEY_COLORS[index]
        track = tracks[key]
        series = {
            "c": ({s: controlled_pose(vehicles[key], scene["t0"] + s) for s in steps},
                  np.array(controlled_size(vehicles[key])) / 2),
            "h": ({s: human_pose(track, scene["t0"] + s) for s in steps},
                  np.array([track.length, track.width]) / 2),
        }  # fmt: skip
        series = {w: ({s: p for s, p in poses.items() if p is not None}, half)
                  for w, (poses, half) in series.items()}  # fmt: skip
        (poses_c, _), (poses_h, _) = series["c"], series["h"]
        shared = {s for s in poses_c if s in poses_h
                  and np.linalg.norm(poses_c[s][:2] - poses_h[s][:2]) < SHARED}  # fmt: skip
        for world, (poses, half) in series.items():
            draw_series(ax, poses, half, color, world,
                        scene["labels"].get((key, world), dict(side=1)), steps[-1], shared)  # fmt: skip
        first = poses_c[min(poses_c)]
        ax.text(*(first[:2] + np.asarray(scene["offsets"][index])), str(index + 1),
                ha="center", va="center", fontsize=LABEL_SIZE, color=color, zorder=7,
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=color, lw=EDGE))  # fmt: skip
    x0, y0 = scene["corner"]
    ax.set_xlim(x0, x0 + WINDOW[0])
    ax.set_ylim(y0, y0 + WINDOW[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(ROAD_WIDTH)
        spine.set_color("0.3")
    if "keys_box" in scene:
        time_key(ax, list(scene["keys_box"]["time"]), fontsize=KEY_SIZE)
        style_key(ax, scene["keys_box"]["style"])
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    stem = os.path.join(out, f"fig_stage3_footprints_{name}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {stem}.pdf")


def main():
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 8,
            "pdf.fonttype": 42,
        }
    )
    with open(os.path.join(RESULTS, "matched_tracks.pkl"), "rb") as file:
        recordings = pickle.load(file)
    os.makedirs(args.out, exist_ok=True)
    for name in args.scenes:
        plot_scene(name, SCENES[name], recordings, args.out)


if __name__ == "__main__":
    main()
