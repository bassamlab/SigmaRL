# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Paper figure of Stage 1: scenes in which the nominal policy collides and
BH-CBF avoids the collision, from the same joint state.

The scenes come from stage1_counterfactual_rollouts.py: a branch env copies
the filtered master's joint state and then applies the nominal policy. Each
scene is drawn twice over the same time span, from the branch time to the
nominal collision: without filter (the branch, ending in the collision,
marked in red) and with BH-CBF (the master). Footprints as in
paper_fig_stage1_footprints.py (transparency encodes time; key vehicles in
color and numbered, the others in gray).

Two commands:

1. ``search``: lists the candidate scenes of the rollouts (a collision of the
   nominal branch 2-8 s after the branch time that the master avoids, while
   the master's filter intervenes on a colliding vehicle) in
   <out>/counterfactual_candidates.csv, with a preview PNG per candidate;
2. ``plot``: draws the paper figure, one PDF per --scene and row
   (<name>_<a, b, ...>_{nominal,filter}); the paper places them as
   subfigures.

A scene is "<npz path relative to --rollouts>:<branch env>:<branch step>:
<collision step>:<key vehicles a+b+...>:<x0,x1,y0,y1>[:<label offsets
dx,dy;...>[:<time key box x,y,w,h>]]".

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.paper_fig_stage1_counterfactual search
    ... plot --scene <scene> --scene <scene> ...
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon

from sigmarl.evaluation_tits26.paper_fig_headline import (
    CONTEXT_GRAY,
    RWTH_PALETTE,
    heading_chevron,
    rectangle,
)
from sigmarl.evaluation_tits26.paper_fig_stage1_footprints import (
    ALPHA_MIN,
    HALF,
    KEY_COLORS,
    draw_road,
    in_window,
)

ROLLOUTS = "outputs/tits26/stage1_counterfactual"
COLLISION_RED = RWTH_PALETTE[5][0]
JUMP = 3.0  # Displacement per step that marks a respawn [m]
# Text sizes [pt] of the panels, which the paper scales to 50 % (one row of 8 scenes)
LABEL_SIZE = 11.0
KEY_SIZE = 12.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["search", "plot"])
    parser.add_argument("--rollouts", default=ROLLOUTS)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--every", type=int, default=10, help="Steps between footprints.")
    parser.add_argument("--min_lead", type=int, default=20, help="Steps before the collision.")
    parser.add_argument("--out", default="outputs/tits26/paper_figs")
    parser.add_argument("--name", default="fig_stage1_counterfactual")
    parser.add_argument("--figsize", type=float, nargs=2, default=[1.72, 1.72])
    return parser.parse_args()


def respawned(pose: np.ndarray) -> np.ndarray:
    """[T, ...] flags of a jump of more than JUMP from the previous step."""
    jump = np.zeros(pose.shape[:-1], dtype=bool)
    jump[1:] = np.linalg.norm(pose[1:, ..., :2] - pose[:-1, ..., :2], axis=-1) > JUMP
    return jump


def map_of(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def search(args):
    """Candidate scenes of all rollouts (see the module docstring)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(args.rollouts, "*", "*", "seed*.npz"))):
        if path.endswith("_validate.npz"):
            continue
        data = np.load(path)
        pose, speed, veh_col = data["pose"], data["speed"], data["veh_col"]
        horizon = int(data["every"]) * int(data["branches"])
        jump = respawned(pose)
        steps, n_envs, n = veh_col.shape
        for env in range(1, n_envs):
            for start in np.where(data["branch_start"][:, env])[0]:
                end = min(steps, start + horizon)
                hits = np.where(veh_col[start + 1 : end, env].any(axis=-1))[0]
                if not len(hits) or hits[0] + 1 < args.min_lead:
                    continue
                t_col = start + 1 + hits[0]
                agents = np.where(veh_col[t_col, env])[0]
                # The colliding pair: the two closest of the flagged vehicles
                xy = pose[t_col, env, agents, :2]
                dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + 1e9 * np.eye(len(agents))
                a, b = np.unravel_index(np.argmin(dist), dist.shape)
                pair = [int(agents[a]), int(agents[b])]
                span = slice(start + 1, min(steps, t_col + 11))
                if jump[span, 0][:, pair].any() or jump[start + 1 : t_col + 1, env][:, pair].any():
                    continue
                if veh_col[span, 0][:, pair].any() or data["lane_col"][span, 0][:, pair].any():
                    continue
                intervention = data["intervention"][start : t_col + 1][:, pair].sum(axis=(0, 2))
                if intervention.max() < 0.5:
                    continue
                # The yielding vehicle: the one the filter slows down most
                slow = speed[t_col, env, pair] - speed[t_col, 0, pair]
                yielder = pair[int(np.argmax(slow))]
                other = pair[1 - int(np.argmax(slow))]
                dyaw = pose[t_col, 0, yielder, 2] - pose[t_col, 0, other, 2]
                dyaw = abs((dyaw + np.pi) % (2 * np.pi) - np.pi)
                rows.append(
                    dict(
                        npz=os.path.relpath(path, args.rollouts), env=env, start=int(start),
                        t_col=int(t_col), lead_s=(t_col - start) / 10, yielder=yielder,
                        other=other, rel_heading_deg=round(float(np.degrees(dyaw))),
                        speed_drop=round(float(slow.max()), 2),
                        min_speed_filter=round(float(speed[start : t_col + 1, 0, yielder].min()), 2),
                        x=round(float(pose[t_col, env, yielder, 0]), 1),
                        y=round(float(pose[t_col, env, yielder, 1]), 1),
                    )
                )  # fmt: skip
    os.makedirs(args.out, exist_ok=True)
    table = os.path.join(args.out, "counterfactual_candidates.csv")
    with open(table, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] {len(rows)} candidates in {table}")
    previews = os.path.join(args.out, "counterfactual_previews")
    os.makedirs(previews, exist_ok=True)
    for index, row in enumerate(rows):
        xy = np.array([row["x"], row["y"]])
        window = [xy[0] - 22, xy[0] + 22, xy[1] - 22, xy[1] + 22]
        scene = dict(
            npz=os.path.join(args.rollouts, row["npz"]), env=row["env"], start=row["start"],
            t_col=row["t_col"], keys=[row["yielder"], row["other"]], window=window,
            label_offsets=np.array([[0.0, 4.5]] * 2), time_key=None,
        )  # fmt: skip
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, is_filter in zip(axes, (False, True)):
            draw_scene(ax, scene, args.every, is_filter, annotate=True)
        axes[0].set_title(f"#{index} nominal: {row['npz']} env {row['env']}")
        axes[1].set_title(f"BH-CBF, yielder {row['yielder']}, rel. heading {row['rel_heading_deg']}")
        fig.savefig(os.path.join(previews, f"cand_{index:03d}.png"), dpi=80)
        plt.close(fig)
    print(f"[INFO] Previews in {previews}")


def parse_scene(text: str, rollouts: str) -> dict:
    npz, env, start, t_col, keys, window, *extra = text.split(":")
    n_keys = len(keys.split("+"))
    offsets = (
        [[float(v) for v in o.split(",")] for o in extra[0].split(";")]
        if extra and extra[0]
        else [[0.0, 4.5]] * n_keys
    )
    return dict(
        npz=os.path.join(rollouts, npz),
        env=int(env),
        start=int(start),
        t_col=int(t_col),
        keys=[int(k) for k in keys.split("+")],
        window=[float(v) for v in window.split(",")],
        label_offsets=np.array(offsets),
        time_key=[float(v) for v in extra[1].split(",")] if len(extra) > 1 else None,
    )


def draw_scene(ax, scene: dict, every: int, is_filter: bool, annotate: bool):
    """Footprints from the branch step to the nominal collision step, without
    filter (the branch env; the colliding footprints outlined in red, with a
    red mark at the contact) or with BH-CBF (the master, env 0). Key vehicles
    in color and numbered, every `every` steps; the others in gray, only
    with their path and last footprint. A track ends at a respawn."""
    data = np.load(scene["npz"])
    env = 0 if is_filter else scene["env"]
    pose = data["pose"][:, env]
    start, end = scene["start"], scene["t_col"]
    colliding = set() if is_filter else set(np.where(data["veh_col"][end, env])[0])
    jump = respawned(pose)

    def time_alpha(t):
        return ALPHA_MIN + (1 - ALPHA_MIN) * (t - start) / (end - start)

    draw_road(ax, map_of(scene["npz"]))
    n = pose.shape[1]
    order = [k for k in range(n) if k not in scene["keys"]] + scene["keys"][::-1]
    for k in order:
        is_key = k in scene["keys"]
        color = KEY_COLORS[scene["keys"].index(k)] if is_key else CONTEXT_GRAY[0]
        zorder = 3 if is_key else 2
        jumps = np.where(jump[start + 1 : end + 1, k])[0]
        last = start + jumps[0] if len(jumps) else end
        visible = [t for t in range(start, last + 1) if in_window(pose[t, k, :2], scene["window"], 0.0)]
        if not visible:
            continue
        final = visible[-1]
        shown = [t for t in range(start, final, every) if t in visible] if is_key else []
        for t in shown:
            ax.add_patch(
                Polygon(rectangle(pose[t, k], HALF), fc=color, alpha=time_alpha(t),
                        ec="none", zorder=zorder)
            )  # fmt: skip
        alpha = time_alpha(final)
        mark = "white" if alpha > 0.5 else color
        is_hit = k in colliding and final == end
        ax.add_patch(
            Polygon(rectangle(pose[final, k], HALF), fc=to_rgba(color, alpha),
                    ec=COLLISION_RED if is_hit else mark, lw=0.9 if is_hit else 0.5,
                    zorder=zorder + 0.3)
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
                    va="center", fontsize=LABEL_SIZE, color=color, zorder=7,
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=color,
                              lw=0.9))  # fmt: skip
        if annotate:
            ax.text(*pose[start, k, :2], str(k), fontsize=6, zorder=6)
    if not is_filter:
        hit = sorted(colliding)
        pair = [k for k in scene["keys"] if k in colliding][:2] or hit[:2]
        contact = pose[end, pair, :2].mean(axis=0)
        ax.plot(*contact, marker="*", ms=7, mfc=COLLISION_RED, mec="white", mew=0.5,
                zorder=8)  # fmt: skip
    x0, x1, y0, y1 = scene["window"]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def time_key(ax, box: list, fontsize: float = KEY_SIZE):
    """A gray bar from transparent to opaque in a free area of a panel (box
    x, y, width, height in meters) that explains the time encoding, from the
    common start state to the end of the time span; "Time" above, "start"
    and "end" beside the bar. Large text, so that it stays legible when the
    panel is scaled down."""
    x, y, width, height = box
    key = ax.inset_axes(box, transform=ax.transData)
    gradient = np.linspace(0, 1, 256)[None, :]
    colors = np.zeros((1, 256, 4))
    colors[..., :3] = 0.35
    colors[..., 3] = ALPHA_MIN + (1 - ALPHA_MIN) * gradient
    key.imshow(colors, aspect="auto", extent=[0, 1, 0, 1])
    key.set_xticks([])
    key.set_yticks([])
    for spine in key.spines.values():
        spine.set_visible(False)
    middle = y + height / 2
    ax.text(x - 0.6, middle, "start", ha="right", va="center", fontsize=fontsize, zorder=9)
    ax.text(x + width + 0.6, middle, "end", ha="left", va="center", fontsize=fontsize, zorder=9)
    ax.text(x + width / 2, y + height + 0.5, "Time", ha="center", va="bottom",
            fontsize=fontsize, zorder=9)  # fmt: skip


def plot(args):
    scenes = [parse_scene(text, args.rollouts) for text in args.scene]
    os.makedirs(args.out, exist_ok=True)
    for index, scene in enumerate(scenes):
        for is_filter in (False, True):
            fig, ax = plt.subplots(figsize=tuple(args.figsize))
            draw_scene(ax, scene, args.every, is_filter, annotate=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("0.3")
            fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
            if scene["time_key"] is not None and not is_filter:
                time_key(ax, scene["time_key"])
            row = "filter" if is_filter else "nominal"
            name = os.path.join(args.out, f"{args.name}_{chr(ord('a') + index)}_{row}")
            for ext in ("pdf", "png"):
                fig.savefig(f"{name}.{ext}", dpi=300)
            plt.close(fig)
            print(f"[INFO] Saved {name}.pdf")


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
    if args.command == "search":
        search(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
