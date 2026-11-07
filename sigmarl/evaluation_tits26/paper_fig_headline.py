# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Headline figure of the T-ITS paper (first page): snapshots of the
roundabout and the intersection with the BH-CBF braking tails of the vehicles.

The snapshot is a state of a Stage 1 deployment: a TTC-reward policy at
training iteration 100 (pre-trained on the map, seed 1), filtered by the fully
decentralized BH-CBF filter with its tuned lambda. At that state, every
vehicle's backup (route-following emergency braking) is rolled out until
standstill; the figure draws the vehicles' rectangles, their braking tails, and
the lane boundaries, and highlights the pair with the smallest backup value.

Three steps per map (--map), each writing to --out:

1. ``select``: scores the steps of a finished Stage 1 run (its out_td file)
   and writes candidates.png with the best candidate steps (vehicle positions
   only), for choosing --step.
2. ``simulate``: reruns that deployment (main_testing_cbf_filter.py, in this
   process, with the same arguments as Stage 1) up to --step, captures the
   joint state when the filter reads it at --step, and saves the vehicle
   states, the braking tails, the pair backup values, and the map to
   snapshot_step<k>.npz (several steps in one run). The deployment is
   deterministic; a state is captured only if its positions match those of
   the reference run.
3. ``plot``: draws fig_headline_step<k>_<style>.pdf (and .png) from
   snapshot_step<k>.npz.

Usage (from the repository root, single core):
    SIGMARL_AGENT_PROFILE=full_scale \\
        python -m sigmarl.evaluation_tits26.paper_fig_headline select
    ... paper_fig_headline simulate --step <k>
    ... paper_fig_headline plot [--xlim x0 x1 --ylim y0 y1]

The paper's figure (two subfigures; outputs in --out = OUT/<map>):
    ... simulate --map round_location0 --step 298
    ... plot --map round_location0 --step 298 --xlim 40 130 --ylim 31 79 --figsize 3.5 1.87
    ... simulate --map ind_location4 --step 546
    ... plot --map ind_location4 --step 546 --min_heading_diff 45 --xlim 68 140 \\
        --ylim 45.5 75 --figsize 3.5 1.43 --inset 131.5 53.5 12 --zoom 10 \\
        --label_offset 2.4 0.3
copied from fig_headline_step<k>_focus_blue.pdf to the paper's
fig_headline_{round,ind}.pdf.

Full views of the maps (whole road only, without vehicles; the height follows
the map's aspect ratio), copied from fig_headline_step<k>_focus_blue_full.pdf to
the paper's fig_headline_{round,ind}_full.pdf:
    ... plot --map round_location0 --step 298 --full
    ... plot --map ind_location4 --step 546 --min_heading_diff 45 --full
"""

import argparse
import glob
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Vehicles per map (sigmarl/evaluation_tits26/settings.py)
N_AGENTS = {"round_location0": 16, "ind_location4": 12}
SEED = 201
OUT = "outputs/tits26/paper_fig_headline"


def checkpoint_of(scenario: str) -> str:
    """The TTC-reward policy (seed 1) pre-trained on the map."""
    return f"checkpoints/tits26/stage1_pretrain/{scenario}/stage1_ttc_seed_1"


def reference_of(scenario: str) -> str:
    """The Stage 1 run of that policy (iteration 100) with BH-CBF dec on its own map."""
    return (
        f"outputs/tits26/stage1_filters_ttc_iter100/"
        f"{scenario}__stage1_ttc_seed_1__iter100/{scenario}/ours/dec/seed{SEED}"
    )


def filter_args(scenario: str) -> list:
    """Arguments of the reference run (see its command_testing.txt)."""
    return [
        "--nominal", "rl",
        "--scenario_type", scenario,
        "--n_agents", str(N_AGENTS[scenario]),
        "--max_group_size", "1",
        "--seed", str(SEED),
        "--is_zero_initial_speed",
        "--is_log_backup_value",
        "--no-is_visu_result",
        "--no-is_save_simulation_video",
        "--no-is_exact_backup_shield",
        "--no-is_standstill_escape",
        "--is_load_final_model",
        "--policy_iter", "100",
        "--cbf_lambda", "0.5",
        "--lane_cbf_baseline", "none",
        "--veh_veh_cbf_baseline", "none",
        "--is_backup_horizon_cbf",
        "--no-is_two_step_rows",
    ]  # fmt: skip


# RWTH colors
RWTH_BLUE = "#00549F"
RWTH_BLUE_25 = "#C7DDF2"
RWTH_RED = "#CC061F"
# (full strength, 25 % tint) of the RWTH colors, in order of preference
RWTH_PALETTE = [
    ("#00549F", "#C7DDF2"),  # blue
    ("#016165", "#BFD0D1"),  # petrol
    ("#7B6FAC", "#DEDAEB"),  # purple
    ("#56AB27", "#DDEBCE"),  # green
    ("#0298A1", "#CAE7E7"),  # turquoise
    ("#CC061F", "#F3CDBB"),  # red
    ("#E30066", "#F9D2DA"),  # magenta
    ("#A11135", "#E5C5C0"),  # bordeaux
]
# Vehicle colors (body, tail) per --style: "mono" (all blue), "palette" (one
# RWTH color per vehicle), "focus" (gray context, the highlighted pair in color),
# "focus_blue" (light blue context, the highlighted pair in color)
CONTEXT_GRAY = ("#7A7A7A", "#DADADA")
CONTEXT_BLUE = ("#8EBAE5", "#E9F1FA")
FOCUS_COLORS = [("#00549F", "#C7DDF2"), ("#56AB27", "#DDEBCE")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["select", "simulate", "plot"])
    parser.add_argument("--map", choices=list(N_AGENTS), default="round_location0")
    parser.add_argument("--out", help=f"Output folder (default: {OUT}/<map>).")
    parser.add_argument(
        "--step", type=int, nargs="+",
        help="Steps of the snapshots (simulate); the step to plot (plot).",
    )  # fmt: skip
    parser.add_argument(
        "--min_heading_diff", type=float, default=0.0,
        help="Default highlighted pair: smallest backup value among the pairs whose "
        "headings differ by at least this angle [deg] (plot).",
    )  # fmt: skip
    parser.add_argument("--n_candidates", type=int, default=12)
    parser.add_argument("--min_step", type=int, default=150)
    parser.add_argument("--xlim", type=float, nargs=2, help="Plot window [m].")
    parser.add_argument("--ylim", type=float, nargs=2, help="Plot window [m].")
    parser.add_argument(
        "--style", choices=["mono", "palette", "focus", "focus_blue"], default="focus_blue",
        help="Vehicle colors (plot); see RWTH_PALETTE.",
    )  # fmt: skip
    parser.add_argument(
        "--zoom", type=float, default=11.0,
        help="Side [m] of the window around the closest approach in the inset (plot).",
    )  # fmt: skip
    parser.add_argument(
        "--label_distance", type=float, default=2.4,
        help="Distance [m] of the clearance label from the gap in the inset (plot).",
    )  # fmt: skip
    parser.add_argument(
        "--label_offset", type=float, nargs=2,
        help="Offset [m] of the clearance label from the gap's middle (plot; "
        "default: --label_distance above the arrow).",
    )  # fmt: skip
    parser.add_argument(
        "--highlight",
        type=int,
        nargs=2,
        help="Highlighted pair (default: the pair with the smallest backup value).",
    )
    parser.add_argument(
        "--inset", type=float, nargs=3, metavar=("X", "Y", "SIDE"),
        help="Inset center and side [m] (default: on the largest island in view).",
    )  # fmt: skip
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=[3.5, 2.5], help="[in] (plot)."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Whole road without vehicles and inset; the height of --figsize follows "
        "the map (plot).",
    )  # fmt: skip
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(OUT, args.map)
    args.reference = reference_of(args.map)
    return args


def load_reference(run_dir: str) -> dict:
    """Positions [T, n, 2], headings [T, n], speeds [T, n], and collision
    flags [T, n] of the reference run, index t = state before step t."""
    (path,) = glob.glob(os.path.join(run_dir, "out_td_*.td"))
    td = torch.load(path, weights_only=False)
    info = td["agents", "info"]
    collided = (info["is_collision_with_agents"][0, ..., 0] > 0) | (
        info["is_collision_with_lanelets"][0, ..., 0] > 0
    )
    return dict(
        pos=info["pos"][0].numpy(),
        rot=info["rot"][0, ..., 0].numpy(),
        speed=info["vel"][0].norm(dim=-1).numpy(),
        collided=collided.numpy(),
    )


def select(args):
    """Score every step by the vehicles that move and by the close pairs of
    moving vehicles, and draw the best candidates."""
    ref = load_reference(args.reference)
    n_steps = ref["pos"].shape[0]
    scores = np.full(n_steps, -np.inf)
    for t in range(args.min_step, n_steps):
        moving = ref["speed"][t] > 2.0
        if (
            ref["collided"][max(0, t - 5) : t + 6].any()
            or moving.sum() < len(moving) / 2
        ):
            continue
        p = ref["pos"][t][moving]
        dist = np.linalg.norm(p[:, None] - p[None], axis=-1)
        close = np.triu(dist < 9.0, k=1).sum()
        scores[t] = moving.sum() + 2.0 * close
    # Best steps at least 20 steps apart
    candidates = []
    for t in np.argsort(-scores):
        if not np.isfinite(scores[t]) or len(candidates) == args.n_candidates:
            break
        if all(abs(t - c) >= 20 for c in candidates):
            candidates.append(int(t))
    n_cols = 4
    n_rows = int(np.ceil(len(candidates) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for ax, t in zip(np.ravel(axes), candidates):
        pos, rot, speed = ref["pos"][t], ref["rot"][t], ref["speed"][t]
        ax.scatter(ref["pos"][:, :, 0], ref["pos"][:, :, 1], s=0.2, c="0.85")
        ax.quiver(
            pos[:, 0], pos[:, 1], speed * np.cos(rot), speed * np.sin(rot),
            color=RWTH_BLUE, angles="xy", scale_units="xy", scale=1.0,
        )  # fmt: skip
        for i, (x, y) in enumerate(pos):
            ax.text(x, y, str(i), fontsize=6)
        ax.set_title(f"step {t}, score {scores[t]:.0f}")
        ax.set_aspect("equal")
    os.makedirs(args.out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "candidates.png"), dpi=110)
    print(f"[INFO] Candidates {candidates}; see {args.out}/candidates.png")


def braking_tails(ctrl, X: np.ndarray) -> tuple:
    """Braking tails [N + 2, n, 3] (the poses at X, then the backup from X
    until every vehicle stands still) and pair backup values [n, n] (the
    smallest pair margin along the two tails) at the joint state X [n, 5]."""
    horizon = ctrl.backup_horizon
    n = X.shape[0]
    routes = [ctrl._shield_route(i) for i in range(n)]
    tails = horizon.rollout(
        X,
        horizon.backup_controls(X, routes),
        routes,
        np.zeros(n, dtype=bool),
        horizon.horizon(X[:, 3]),
    )
    tails = np.concatenate([X[None, :, 0:3], tails])
    pair_values = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            value = horizon.pair_values(tails[1:, i], tails[1:, j]).min()
            pair_values[i, j] = pair_values[j, i] = value
    return tails, pair_values


def simulate(args):
    """Rerun the reference deployment up to the largest --step and save one
    snapshot per step (snapshot_step<k>.npz).

    The joint state is captured when the first filter reads it at a step of
    --step (the environment resets after the rollout, so the final
    state is not available afterwards)."""
    from sigmarl.cbf_qp_proxsuite import CBFQP

    if not args.step:
        raise ValueError("simulate needs --step.")
    ref = load_reference(args.reference)
    targets = {step: ref["pos"][step] for step in args.step}
    snapshots = {}
    collect = CBFQP._collect_agent_states

    def collect_and_capture(ctrl, tensordict, n_agents):
        result = collect(ctrl, tensordict, n_agents)
        if ctrl.env_idx == 0:
            X = np.array([state.tolist() for state in result[1]], dtype=np.float64)
            for step, target in targets.items():
                if step not in snapshots and np.abs(X[:, :2] - target).max() < 1e-4:
                    tails, pair_values = braking_tails(ctrl, X)
                    snapshots[step] = dict(
                        states=X,
                        tails=tails,
                        pair_values=pair_values,
                        half=ctrl.backup_horizon.half,
                    )
        return result

    CBFQP._collect_agent_states = collect_and_capture
    # max_steps - 1 steps are simulated; the filter reads the state of each step
    sys.argv = (
        [
            "main_testing_cbf_filter.py",
            "--path",
            os.path.abspath(checkpoint_of(args.map)),
        ]
        + filter_args(args.map)
        + ["--max_steps", str(max(args.step) + 2)]
        + ["--output_dir", os.path.join(args.out, "sim")]
    )
    run = runpy.run_path(
        os.path.join(REPO_ROOT, "main_testing_cbf_filter.py"), run_name="__main__"
    )
    missing = sorted(set(args.step) - set(snapshots))
    if missing:
        print(f"[WARNING] No filter read the reference state at the steps {missing}.")

    boundaries = {}
    for key, lanelet in run["scenario"]._get_rendered_lanelets().items():
        for side in ("left_boundary", "right_boundary"):
            boundaries[f"{side}_{key}"] = np.asarray(lanelet[side], dtype=np.float64)
    for step, snapshot in snapshots.items():
        path = os.path.join(args.out, f"snapshot_step{step}.npz")
        np.savez(path, step=step, **snapshot, **boundaries)
        print(f"[INFO] Saved {path}")


def rectangle(pose: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Corners [4, 2] of a vehicle rectangle with half sizes (l, w) at a pose."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * half
    return pose[:2] + local @ np.array([[c, s], [-s, c]])


def heading_chevron(pose: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Points [3, 2] of a chevron (">") in the front half of a vehicle
    rectangle, pointing in its heading direction."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    local = np.array([[0.25, 0.55], [0.75, 0.0], [0.25, -0.55]]) * half
    return pose[:2] + local @ np.array([[c, s], [-s, c]])


def closest_points(pose_a, pose_b, half) -> tuple:
    """Closest points of two vehicle rectangles."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import nearest_points

    a, b = nearest_points(
        ShapelyPolygon(rectangle(pose_a, half)), ShapelyPolygon(rectangle(pose_b, half))
    )
    return np.array(a.coords[0]), np.array(b.coords[0])


def plot(args):
    """Draw the headline figure from snapshot.npz: the road, every vehicle
    (filled) with its braking tail (light, ending in the outlined standstill
    pose), and the closest approach of the tails of the highlighted pair,
    magnified in an inset on the central island with its clearance."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    (step,) = args.step
    data = np.load(os.path.join(args.out, f"snapshot_step{step}.npz"))
    tails, half = data["tails"], data["half"]
    pair_values = data["pair_values"]
    n = tails.shape[1]
    if args.highlight is not None:
        pair = tuple(args.highlight)
    else:
        heading = data["states"][:, 2]
        diff = np.abs((heading[:, None] - heading[None] + np.pi) % (2 * np.pi) - np.pi)
        values = np.where(
            np.degrees(diff) >= args.min_heading_diff, pair_values, np.inf
        )
        pair = tuple(int(k) for k in np.unravel_index(np.argmin(values), values.shape))
    keys = sorted(
        key.removeprefix("left_boundary_")
        for key in data.files
        if key.startswith("left_boundary_")
    )
    # Road: the union of the lanelets
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
    parts = road.geoms if road.geom_type == "MultiPolygon" else [road]
    # Closest approach of the highlighted pair along their tails
    i, j = pair
    distances = [
        np.linalg.norm(np.subtract(*closest_points(tails[t, i], tails[t, j], half)))
        for t in range(1, tails.shape[0])
    ]
    t_min = 1 + int(np.argmin(distances))
    a, b = closest_points(tails[t_min, i], tails[t_min, j], half)

    def colors_of(k):
        if args.style == "palette":
            return RWTH_PALETTE[k % len(RWTH_PALETTE)]
        if args.style in ("focus", "focus_blue"):
            if k in pair:
                return FOCUS_COLORS[pair.index(k)]
            return CONTEXT_GRAY if args.style == "focus" else CONTEXT_BLUE
        return RWTH_BLUE, RWTH_BLUE_25

    def draw_scene(ax, vehicles=True):
        for part in parts:
            ax.add_patch(
                Polygon(np.asarray(part.exterior.coords), fc="0.93", ec="0.45",
                        lw=0.5, zorder=0)
            )  # fmt: skip
            for hole in part.interiors:
                ax.add_patch(
                    Polygon(np.asarray(hole.coords), fc="white", ec="0.45", lw=0.5,
                            zorder=1)
                )  # fmt: skip
        ax.set_aspect("equal")
        if not vehicles:
            return
        for k in range(n):
            body, light = colors_of(k)
            tail = tails[1:, k]
            if np.linalg.norm(tail[-1, :2] - tails[0, k, :2]) > 0.05:
                for pose in tail:
                    ax.add_patch(
                        Polygon(rectangle(pose, half), fc=light, ec="none", zorder=2)
                    )
                # Standstill pose (thin outline) and the path of the center to it
                ax.add_patch(
                    Polygon(rectangle(tail[-1], half), fill=False, ec=body, lw=0.35,
                            zorder=3)
                )  # fmt: skip
                ax.plot(tails[:, k, 0], tails[:, k, 1], color=body, lw=0.4, zorder=3)
            ax.add_patch(
                Polygon(rectangle(tails[0, k], half), fc=body, ec="none", zorder=4)
            )
            chevron = heading_chevron(tails[0, k], half)
            ax.plot(chevron[:, 0], chevron[:, 1], color="white", lw=0.6,
                    solid_joinstyle="miter", solid_capstyle="butt", zorder=4.5)  # fmt: skip
        for k in pair:
            ax.add_patch(
                Polygon(rectangle(tails[t_min, k], half), fill=False, ec=RWTH_RED,
                        lw=0.6, zorder=5)
            )  # fmt: skip

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 8,
            "pdf.fonttype": 42,
        }
    )
    if args.full:
        # Whole road with a small margin
        x0, y0, x1, y1 = road.bounds
        args.xlim, args.ylim = [x0 - 1, x1 + 1], [y0 - 1, y1 + 1]
        args.figsize[1] = args.figsize[0] * (y1 - y0 + 2) / (x1 - x0 + 2)
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    draw_scene(ax, vehicles=not args.full)
    if args.xlim is not None:
        ax.set_xlim(*args.xlim)
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    ax.axis("off")
    if args.full:
        fig.tight_layout(pad=0.05)
        name = os.path.join(args.out, f"fig_headline_step{step}_{args.style}_full")
        for ext in ("pdf", "png"):
            fig.savefig(f"{name}.{ext}", dpi=300)
        print(f"[INFO] Saved {name}.pdf")
        return

    # Inset on the largest island in view: the closest approach, magnified,
    # with a dimension arrow of the clearance
    if args.inset is not None:
        center, side = np.array(args.inset[:2]), args.inset[2]
    else:
        holes = [ShapelyPolygon(hole) for part in parts for hole in part.interiors]
        island = max(holes, key=lambda hole: hole.area)
        center = np.asarray(island.centroid.coords[0])
        side = 1.3 * np.sqrt(island.area / np.pi)
    inset = ax.inset_axes(
        [*(center - side / 2), side, side], transform=ax.transData, zorder=8
    )
    draw_scene(inset)
    middle = (a + b) / 2
    inset.set_xlim(middle[0] - args.zoom / 2, middle[0] + args.zoom / 2)
    inset.set_ylim(middle[1] - args.zoom / 2, middle[1] + args.zoom / 2)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.3")
    inset.annotate(
        "", xy=a, xytext=b, zorder=6,
        arrowprops=dict(arrowstyle="<|-|>,head_length=0.3,head_width=0.15",
                        color=RWTH_RED, lw=0.6, shrinkA=0, shrinkB=0),
    )  # fmt: skip
    if args.label_offset is not None:
        label_xy = middle + np.asarray(args.label_offset)
    else:
        normal = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        normal *= 1.0 if normal[1] > 0 else -1.0  # Label above the arrow
        label_xy = (
            middle
            + args.label_distance * normal
            + 0.6 * (b - a) / np.linalg.norm(b - a)
        )
    inset.text(
        *label_xy, f"{min(distances):.1f} m", ha="center", va="center",
        color=RWTH_RED, fontsize=7, zorder=7,
    )  # fmt: skip
    _, connectors = ax.indicate_inset_zoom(inset, edgecolor="0.3", lw=0.5, alpha=1.0)
    for connector in connectors:
        connector.set_linewidth(0.4)
    print(
        f"[INFO] Highlighted pair {pair}: backup value {pair_values[pair]:.2f} m, "
        f"closest approach {min(distances):.2f} m at tail step {t_min}"
    )
    fig.tight_layout(pad=0.05)
    name = os.path.join(args.out, f"fig_headline_step{step}_{args.style}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=300)
    print(f"[INFO] Saved {name}.pdf")


if __name__ == "__main__":
    arguments = parse_args()
    {"select": select, "simulate": simulate, "plot": plot}[arguments.command](arguments)
