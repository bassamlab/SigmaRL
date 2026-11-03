# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Paper figure of Stage 1: footprints of vehicles filtered by BH-CBF over
time (transparency encodes time: the earlier, the more transparent).

The scenes come from the Stage 1 deployments of the TTC-reward policies of
iteration 100 with the fully decentralized BH-CBF filter
(outputs/tits26/stage1_filters_ttc_iter100, out_td files). The road is drawn
as in the headline figure (paper_fig_headline.py), from one of its snapshot
files, which hold the lane boundaries of the map.

Two commands:

1. ``preview``: draws large previews of given scenes (--scene, repeated),
   for choosing the paper's scenes;
2. ``plot``: draws the paper figure, one PDF per --scene (<name>_a, _b, ...);
   the paper places them as subfigures.

A scene is "<run folder>:<first step>:<steps>:<key vehicles>:<x0,x1,y0,y1>
[:<label offsets>[:<time key>]]", with the run folder relative to the Stage 1
output folder, the key vehicles separated by "+" (drawn in color and numbered
1, 2, ...; all others in gray), the plot window in meters, the offsets
"dx,dy;dx,dy;..." [m] of the number labels from the key vehicles' last
footprints, and the time key's box "x,y,width,height" [m] in the panel.

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale \\
        python -m sigmarl.evaluation_tits26.paper_fig_stage1_footprints plot \\
        --scene <scene> --scene <scene> [--out outputs/tits26/paper_figs]
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon

from sigmarl.evaluation_tits26.paper_fig_headline import (
    CONTEXT_GRAY,
    RWTH_PALETTE,
    heading_chevron,
    rectangle,
)

STAGE1 = "outputs/tits26/stage1_filters_ttc_iter100"
ROAD = {
    "round_location0": "outputs/tits26/paper_fig_headline/round_location0/snapshot_step298.npz",
    "ind_location4": "outputs/tits26/paper_fig_headline/ind_location4/snapshot_step546.npz",
}
HALF = np.array([2.3, 1.0])  # Half length and width of the full-scale vehicle [m]
# Key vehicles: blue, green, purple (RWTH, full strength)
KEY_COLORS = [RWTH_PALETTE[0][0], RWTH_PALETTE[3][0], RWTH_PALETTE[2][0]]
ALPHA_MIN = 0.12


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["preview", "plot"])
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--every", type=int, default=10, help="Steps between footprints.")
    parser.add_argument("--out", default="outputs/tits26/paper_figs")
    parser.add_argument("--name", default="fig_stage1_footprints")
    parser.add_argument("--figsize", type=float, nargs=2, default=[1.72, 1.72])
    return parser.parse_args()


def parse_scene(text: str) -> dict:
    run, start, steps, keys, window, *extra = text.split(":")
    n_keys = len(keys.split("+"))
    offsets = (
        [[float(v) for v in o.split(",")] for o in extra[0].split(";")]
        if extra and extra[0]
        else [[0.0, 3.5]] * n_keys
    )
    return dict(
        label_offsets=np.array(offsets),
        time_key=[float(v) for v in extra[1].split(",")] if len(extra) > 1 else None,
        run=os.path.join(STAGE1, run),
        start=int(start),
        steps=int(steps),
        keys=[int(k) for k in keys.split("+")],
        window=[float(v) for v in window.split(",")],
        map=run.split("/")[1],
    )


def load_run(run: str) -> dict:
    """Poses [T, n, 3] and speeds [T, n] after each step of a deployment."""
    (path,) = glob.glob(os.path.join(run, "out_td_*.td"))
    td = torch.load(path, weights_only=False)
    info = td["next", "agents", "info"]
    pos = info["pos"][0].numpy()
    rot = info["rot"][0, ..., 0].numpy()
    return dict(
        pose=np.concatenate([pos, rot[..., None]], axis=-1),
        speed=info["vel"][0].norm(dim=-1).numpy(),
    )


def draw_road(ax, scenario: str):
    """The road as in the headline figure: the union of the lanelets."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    data = np.load(ROAD[scenario])
    keys = sorted(
        key.removeprefix("left_boundary_")
        for key in data.files
        if key.startswith("left_boundary_")
    )
    road = unary_union(
        [
            ShapelyPolygon(
                np.vstack(
                    [data[f"left_boundary_{key}"], data[f"right_boundary_{key}"][::-1]]
                )
            ).buffer(0.05)
            for key in keys
        ]
    )
    for part in road.geoms if road.geom_type == "MultiPolygon" else [road]:
        ax.add_patch(
            Polygon(np.asarray(part.exterior.coords), fc="0.93", ec="0.45", lw=0.5,
                    zorder=0)
        )  # fmt: skip
        for hole in part.interiors:
            ax.add_patch(
                Polygon(np.asarray(hole.coords), fc="white", ec="0.45", lw=0.5,
                        zorder=1)
            )  # fmt: skip


def in_window(xy: np.ndarray, window: list, margin: float = 3.0) -> bool:
    x0, x1, y0, y1 = window
    return x0 - margin <= xy[0] <= x1 + margin and y0 - margin <= xy[1] <= y1 + margin


def draw_scene(ax, scene: dict, every: int, annotate: bool):
    """Footprints of every vehicle in the window from the first step on,
    every `every` steps, with the path of its center (other vehicles: only
    the path and the last footprint). The last footprint in the window is
    outlined and carries a heading chevron (at the end of the time span, or
    where the vehicle leaves the window). Key vehicles in color and numbered
    1, 2, ... in their order in the scene, the others in gray.
    A vehicle's track ends at a respawn (jump)."""
    data = load_run(scene["run"])
    pose = data["pose"]
    start, end = scene["start"], scene["start"] + scene["steps"]

    def time_alpha(t):
        return ALPHA_MIN + (1 - ALPHA_MIN) * (t - start) / (end - start)

    draw_road(ax, scene["map"])
    n = pose.shape[1]
    order = [k for k in range(n) if k not in scene["keys"]] + scene["keys"][::-1]
    for k in order:
        is_key = k in scene["keys"]
        color = KEY_COLORS[scene["keys"].index(k)] if is_key else CONTEXT_GRAY[0]
        zorder = 3 if is_key else 2
        last = end
        for t in range(start + 1, end + 1):
            if np.linalg.norm(pose[t, k, :2] - pose[t - 1, k, :2]) > 3.0:
                last = t - 1  # respawned
                break
        visible = [t for t in range(start, last + 1) if in_window(pose[t, k, :2], scene["window"], 0.0)]
        if not visible:
            continue
        final = visible[-1]
        # Other vehicles: only the path and the last footprint, to reduce clutter
        shown = [t for t in range(start, final, every) if t in visible] if is_key else []
        for t in shown:
            ax.add_patch(
                Polygon(rectangle(pose[t, k], HALF), fc=color, alpha=time_alpha(t),
                        ec="none", zorder=zorder)
            )  # fmt: skip
        # Last footprint: outlined (in white if it is opaque enough, else in its color)
        alpha = time_alpha(final)
        mark = "white" if alpha > 0.5 else color
        ax.add_patch(
            Polygon(rectangle(pose[final, k], HALF), fc=to_rgba(color, alpha), ec=mark,
                    lw=0.5, zorder=zorder + 0.3)
        )  # fmt: skip
        ax.plot(pose[start : last + 1, k, 0], pose[start : last + 1, k, 1], color=color,
                lw=0.6, zorder=zorder + 0.2)  # fmt: skip
        chevron = heading_chevron(pose[final, k], HALF)
        ax.plot(chevron[:, 0], chevron[:, 1], color=mark, lw=0.6,
                solid_joinstyle="miter", solid_capstyle="butt",
                zorder=zorder + 0.4)  # fmt: skip
        if is_key:
            label = scene["keys"].index(k)
            offset = scene["label_offsets"][label]
            ax.text(*(pose[final, k, :2] + offset), str(label + 1), ha="center",
                    va="center", fontsize=6.5, color=color, zorder=7,
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=color,
                              lw=0.5))  # fmt: skip
        if annotate:
            ax.text(*pose[start, k, :2], str(k), fontsize=6, zorder=6)
    x0, x1, y0, y1 = scene["window"]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def time_key(ax, seconds: float, box: list):
    """A gray bar from transparent to opaque in a free area of a panel (box
    x, y, width, height in meters) that explains the time encoding."""
    key = ax.inset_axes(box, transform=ax.transData)
    gradient = np.linspace(0, 1, 256)[None, :]
    colors = np.zeros((1, 256, 4))
    colors[..., :3] = 0.35
    colors[..., 3] = ALPHA_MIN + (1 - ALPHA_MIN) * gradient
    key.imshow(colors, aspect="auto", extent=[0, seconds, 0, 1])
    key.set_yticks([])
    key.set_xticks([0, seconds])
    key.set_xticklabels(["0 s", f"{seconds:.0f} s"])
    key.tick_params(length=1.5, width=0.4, pad=1, labelsize=6.5)
    key.set_title("Time", fontsize=6.5, pad=1.5)
    for spine in key.spines.values():
        spine.set_visible(False)


def main():
    args = parse_args()
    scenes = [parse_scene(text) for text in args.scene]
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["Times"], "font.size": 8}
    )
    if args.command == "preview":
        for index, scene in enumerate(scenes):
            fig, ax = plt.subplots(figsize=(7, 7))
            draw_scene(ax, scene, args.every, annotate=True)
            path = os.path.join(args.out, f"preview_{index}.png")
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print(f"[INFO] Saved {path}")
        return
    seconds = scenes[0]["steps"] * 0.1
    for index, scene in enumerate(scenes):
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        draw_scene(ax, scene, args.every, annotate=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("0.3")
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
        if scene["time_key"] is not None:
            time_key(ax, seconds, scene["time_key"])
        name = os.path.join(args.out, f"{args.name}_{chr(ord('a') + index)}")
        for ext in ("pdf", "png"):
            fig.savefig(f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved {name}.pdf")


if __name__ == "__main__":
    main()
