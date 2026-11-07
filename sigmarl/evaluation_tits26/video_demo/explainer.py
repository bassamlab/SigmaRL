# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Explainer video of the T-ITS paper: how the BH-CBF filter works (about 3 minutes).

The supplementary video (video.py) shows what BH-CBF achieves over the four
evaluation stages. This video zooms into single filter problems and shows how
the filter reaches its decision, in the order of the paper's method section:

- one_step: why the one-step condition is not enough (a vehicle approaches a
  stopped vehicle on a straight road; both filters simulated here with the
  paper's model, parameters, and margin);
- backup: the backup and candidate predictions of one vehicle and the BH-CBF
  condition on its lane margin, at one step of a Stage 2 deployment;
- control_plane: the same filter problem in the plane of the controls
  (acceleration, steering rate): the exact set of controls that satisfy all
  inequalities, the policy's control, the backup control, the QP candidate,
  and the filtered control; then the following steps;
- recursive: why the filter stays feasible (the candidate prediction at k is
  the backup prediction at k + 1);
- pair: two vehicles and the decentralized filter (shares, corner budgets
  along the separating normal, one problem per vehicle).

All quantities of the deployment scenes come from the filter itself: the
filter problems are recorded by explainer_capture.py (OUT/explainer/*.pkl)
from a rerun of the Stage 2 comparison run with BH-CBF (the run of the
``repair`` clip of video.py). Clips go to OUT/explainer/clips/<nn>_<name>.mp4;
``assemble`` concatenates them to OUT/bh_cbf_explainer.mp4.

Usage (from the repository root, with the shell function R of video.py):
    R explainer_capture --run <EXPLAINER_RUN> --steps 94 120 --agents 9 --name explainer_lane
    R explainer_capture --run <EXPLAINER_RUN> --steps 112 155 --agents 6 10 --name explainer_pair
    R explainer render [--only <clip> ...] [--preview]
    R explainer assemble
"""

import argparse
import glob
import os
import pickle
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patheffects
from matplotlib.colors import to_rgba
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D

from sigmarl.backup_horizon_cbf import rectangle_corners, signed_distance
from sigmarl.evaluation_tits26.video_demo.video import (
    BLUE, CARLITO, CONTEXT, DT, FPS, GREEN, HALF, HEIGHT, JUMP, MUTED, OUT, PURPLE, RED,
    REPAIR_RUN, TEXT, WIDTH, Clip, Fleet, card, heading_chevron, interpolate, load_td,
    new_figure, rectangle, road, tint,
)  # fmt: skip

EXPLAINER = f"{OUT}/explainer"
EXPLAINER_RUN = REPAIR_RUN  # Stage 2 comparison run with BH-CBF (TTC policy, iteration 100)
# One vehicle (lane constraint): the filter problem of vehicle LANE_AGENT at LANE_STEP, the steps
# of the real-time approach before it, and the steps shown one by one after it
LANE_NAME, LANE_AGENT, LANE_STEP = "explainer_lane", 9, 114
LANE_LEAD, LANE_NEXT = 20, 2
# Two vehicles: i approaches j; the step of the paused filter problems, the real-time approach,
# and the real-time continuation
PAIR_NAME, PAIR_I, PAIR_J, PAIR_STEP = "explainer_pair", 10, 6, 130
PAIR_LEAD, PAIR_AFTER = 15, 20
VIEW = 30.0  # Side of the vehicle-centered map view [m]
PAIR_VIEW = 40.0  # Side of the pair's map window [m]
J_COLOR = GREEN
BACKUP_COLOR = "#8C8C8C"  # Backup predictions (gray)
NOMINAL = PURPLE  # The policy's control and its candidate prediction
INFEASIBLE = tint(RED, 0.13)
ROUTE_COLOR = "#0098A1"  # Route centerline (RWTH turquoise)

# Layout (figure fractions): map left, plots right, a caption across the bottom
MAP_BOX = [0.035, 0.165, 0.65 * HEIGHT / WIDTH, 0.65]
RIGHT = (0.455, 0.965)  # x range of the right column
PANEL_TOP, PANEL_BOTTOM = 0.815, 0.165


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["render", "assemble"])
    parser.add_argument("--only", nargs="+", help="Render only these clips (names).")
    parser.add_argument(
        "--preview", action="store_true",
        help="Write one PNG per clip (its middle frame) instead of the clips.",
    )  # fmt: skip
    parser.add_argument("--out", default=EXPLAINER)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data


def load_problems(name: str) -> tuple:
    """(meta, {(step, agent): filter problem}) recorded by explainer_capture.py."""
    with open(os.path.join(EXPLAINER, f"{name}.pkl"), "rb") as file:
        data = pickle.load(file)
    return data["meta"], {(p["step"], p["agent"]): p for p in data["problems"]}


def window_poses(first: int, last: int) -> np.ndarray:
    """Poses [T, n, 3] after the steps first, ..., last of the run; each
    vehicle only between its respawns around the window (NaN elsewhere)."""
    pose = load_td(EXPLAINER_RUN)["pose"][first : last + 1].copy()
    jump = np.linalg.norm(pose[1:, :, :2] - pose[:-1, :, :2], axis=-1) > JUMP
    for k in range(pose.shape[1]):
        hits = np.nonzero(jump[:, k])[0] + 1
        if len(hits):
            pose[hits[0] :, k] = np.nan  # From its first respawn in the window on
    return pose


def entry(problem: dict, label: tuple) -> int:
    return problem["labels"].index(label)


def speeds(poses: np.ndarray) -> np.ndarray:
    """Speeds [T] along a prediction [T, 3] (t = 1, ...): an Euler step moves
    the pose by dt times the speed."""
    step = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) / DT
    return np.concatenate([step, [0.0]])


def pair_margins(meta, poses_a, poses_b) -> np.ndarray:
    """Pair margins [T] of two predictions [T, 3]."""
    distance, _ = signed_distance(poses_a, poses_b, meta["half"])
    return distance - meta["pair_buffer"]


# ---------------------------------------------------------------------------
# Drawing helpers


def setup_fonts():
    # Carlito: metrically compatible with Calibri, the font of the institute's slides
    for path in glob.glob(os.path.join(CARLITO, "Carlito-*.ttf")):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Carlito"],
                         "mathtext.fontset": "cm"})  # fmt: skip


def white_stroke(width: float = 4):
    return [patheffects.withStroke(linewidth=width, foreground="white")]


class MapPanel:
    """A map panel, heading-up around a pose (view transform) or north-up."""

    def __init__(self, fig, box=MAP_BOX, view: float = VIEW, scenario="round_location0"):
        self.ax = ax = fig.add_axes(box)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.6")
        road(ax, scenario)
        self.transform = Affine2D()
        for artist in [*ax.patches, *ax.lines, *ax.collections]:
            artist.set_transform(self.transform + ax.transData)
        self.set_window(view)
        self.clock = ax.text(0.02, 0.975, "", transform=ax.transAxes, fontsize=20, color=MUTED,
                             va="top", zorder=20,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))  # fmt: skip

    def set_window(self, view: float, center=(0.0, 0.0)):
        self.ax.set_xlim(center[0] - view / 2, center[0] + view / 2)
        self.ax.set_ylim(center[1] - view / 2, center[1] + view / 2)
        self.ax.set_aspect("equal")

    def look_from(self, pose: np.ndarray, ahead: float = 0.22 * VIEW):
        """Heading up, the pose `ahead` meters below the center."""
        self.transform.clear().translate(-pose[0], -pose[1]).rotate(np.pi / 2 - pose[2])
        self.transform.translate(0, -ahead)

    def add(self, artist):
        if isinstance(artist, PolyCollection):
            self.ax.add_collection(artist)
        elif isinstance(artist, Line2D):
            self.ax.add_line(artist)
        else:
            self.ax.add_artist(artist)
        artist.set_transform(self.transform + self.ax.transData)
        return artist

    def fleet(self, colors: list) -> Fleet:
        fleet = Fleet(self.ax, HALF, colors)
        for artist in fleet.patches + fleet.chevrons:
            artist.set_transform(self.transform + self.ax.transData)
        return fleet

    def text(self, text: str, color: str, size: float = 26):
        label = self.ax.text(0, 0, text, fontsize=size, color=color, weight="bold", ha="center",
                             va="center", zorder=12, path_effects=white_stroke())  # fmt: skip
        label.set_transform(self.transform + self.ax.transData)
        label.set_clip_on(True)
        return label

    def note(self, text: str, point, color: str = TEXT, ha: str = "center", va: str = "center",
             size: float = 20, visible: bool = True):  # fmt: skip
        """An annotation at a world point (upright on the screen)."""
        label = self.ax.text(point[0], point[1], text, fontsize=size, color=color, ha=ha, va=va, zorder=12,
                             path_effects=white_stroke(), visible=visible)  # fmt: skip
        label.set_transform(self.transform + self.ax.transData)
        label.set_clip_on(True)
        return label


def beside(pose: np.ndarray, ahead: float, right: float) -> np.ndarray:
    """The world point `ahead` meters along the heading of pose and `right`
    meters to its right."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pose[:2] + ahead * np.array([c, s]) + right * np.array([s, -c])


def on_polyline(points: np.ndarray, pose: np.ndarray, ahead: float) -> np.ndarray:
    """The point of a polyline [P, 2] that lies `ahead` meters along the
    heading of pose (of several such points, the one nearest to pose)."""
    along = (points - pose[:2]) @ np.array([np.cos(pose[2]), np.sin(pose[2])])
    hits = [pa + (ahead - a) / (b - a) * (pb - pa)
            for a, b, pa, pb in zip(along[:-1], along[1:], points[:-1], points[1:])
            if (a - ahead) * (b - ahead) <= 0 and a != b]  # fmt: skip
    if not hits:
        return points[np.argmin(np.abs(along - ahead))]
    return min(hits, key=lambda q: np.linalg.norm(q - pose[:2]))


class Footprints:
    """Footprints of a prediction: light fills with edges, the standstill
    footprint outlined (as the trails of video.py)."""

    def __init__(self, panel: MapPanel, color: str, zorder: float = 2.0, strength: float = 0.28):
        self.color, self.strength = color, strength
        self.fill = panel.add(PolyCollection([], linewidths=0.8, zorder=zorder))
        self.stop = panel.add(PolyCollection([], facecolors="none", linewidths=1.6,
                                             zorder=zorder + 0.05))  # fmt: skip

    def update(self, poses, upto: int = None, color: str = None):
        """poses [T, 3] (None: hide); only the first `upto` of them."""
        color = color or self.color
        if poses is None or len(poses) == 0:
            self.fill.set_verts([])
            self.stop.set_verts([])
            return
        poses = poses if upto is None else poses[:upto]
        self.fill.set_verts([rectangle(p, HALF) for p in poses])
        self.fill.set_facecolors([tint(color, self.strength)] * len(poses))
        self.fill.set_edgecolors([tint(color, 0.6)] * len(poses))
        self.stop.set_verts([rectangle(poses[-1], HALF)])
        self.stop.set_edgecolors([tint(color, 0.85)])


class Outlines:
    """Rectangles outlined in one color (e.g., red at violated poses)."""

    def __init__(self, panel: MapPanel, color: str = RED, width: float = 3.0, zorder: float = 6,
                 linestyle="solid"):  # fmt: skip
        self.collection = panel.add(PolyCollection([], facecolors="none", edgecolors=color,
                                                   linewidths=width, linestyles=linestyle, zorder=zorder))  # fmt: skip

    def update(self, poses):
        """poses [T, 3] (None: hide)."""
        self.collection.set_verts([] if poses is None else [rectangle(p, HALF) for p in poses])


class Caption:
    """Up to two caption lines across the bottom of the frame."""

    def __init__(self, fig, y: float = 0.105, size: float = 25):
        self.fig, self.y, self.size = fig, y, size
        self.texts = []

    def set(self, *lines, color=TEXT):
        for text in self.texts:
            text.remove()
        self.texts = [self.fig.text(0.035, self.y - 0.052 * k, line, fontsize=self.size, color=color,
                                    va="center") for k, line in enumerate(lines)]  # fmt: skip


def style_plot(ax, xlabel: str = None, ylabel: str = None):
    ax.tick_params(labelsize=16, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20, color=TEXT)


def hold(clip: Clip, fig, seconds: float):
    clip.add(fig, int(round(seconds * FPS)))


# ---------------------------------------------------------------------------
# Scene: one-step condition vs. BH-CBF on a straight road


def simulate_line(meta: dict, mode: str, v0: float, gap: float, steps: int = 70) -> list:
    """A vehicle approaches a stopped vehicle on a straight road (Euler steps
    of the paper's model, heading fixed), with the policy keeping its speed
    (nominal acceleration zero), filtered by the one-step condition ("onestep")
    or by BH-CBF ("bh") on the pair margin (bumper gap minus the clearance).
    Returns one dict per step: position x of the center, speed v,
    acceleration a, the backup prediction (positions until standstill), the
    filter's state, and the bumper gap."""
    dt, a_min, a_max = meta["dt"], meta["a_min"], meta["a_max"]
    lam = meta["lam_lane"]
    length = 2 * meta["half"][0]
    obstacle = gap + length  # Center of the stopped vehicle
    margin = lambda x: obstacle - x - length - meta["pair_buffer"]  # noqa: E731

    def prediction(x, v, a):
        """Positions at t = 1, 2, ... under a now, then the backup."""
        x, v = x + dt * v, max(0.0, v + dt * a)
        out = [x]
        while v > 1e-9:
            b = max(a_min, -v / dt)
            x, v = x + dt * v, max(0.0, v + dt * b)
            out.append(x)
        return np.array(out)

    states, x, v, state = [], 0.0, v0, "cruise"
    for _ in range(steps):
        lower = max(a_min, -v / dt)
        backup = prediction(x, v, lower)
        if mode == "onestep":
            bound = (lam * margin(x + dt * v) - dt * v) / dt**2  # From the one-step condition
            a = min(0.0, bound, a_max)
            if a < 0 and state == "cruise":
                state = "brake"
            if bound < lower:
                a, state = lower, "infeasible"
        else:
            value = margin(backup[-1])  # Backup value: the margin at standstill
            ok = lambda a: margin(prediction(x, v, a)[-1]) >= (1 - lam) * value - 1e-9  # noqa: E731
            if ok(0.0):
                a = 0.0
            else:
                low, high = lower, 0.0
                for _ in range(60):
                    mid = 0.5 * (low + high)
                    low, high = (mid, high) if ok(mid) else (low, mid)
                a = low
                state = "brake"
        gap_now = obstacle - x - length
        states.append(dict(x=x, v=v, a=a, state=state, backup=np.concatenate([[x], backup]),
                           gap=gap_now))  # fmt: skip
        if gap_now <= 0:
            states[-1].update(state="collision", x=obstacle - length, gap=0.0)
            break
        if v < 0.05 and state != "cruise":  # Standstill (moves less than 5 mm per step)
            states[-1]["state"] = "stopped"
            break
        x, v = x + dt * v, max(0.0, v + dt * a)
    return states, obstacle


def scene_one_step(clip, speed: float = 0.5):
    meta, _ = load_problems(LANE_NAME)
    v0, gap = 11.7, 26.0
    runs = {mode: simulate_line(meta, mode, v0, gap) for mode in ("onestep", "bh")}
    obstacle = runs["bh"][1]
    fig = new_figure(
        "Why look ahead until standstill?",
        f"A vehicle at {v0} m/s approaches a stopped vehicle; the policy keeps its speed. "
        "Both filters use the same margin, model, and parameters.",
    )
    x0, x1 = -4.0, obstacle + 5.5
    strip_w = 0.6
    strip_h = strip_w * WIDTH * 9.0 / (x1 - x0) / HEIGHT
    strips = {}
    plot = fig.add_axes([0.715, 0.23, 0.25, 0.53])
    style_plot(plot, "Time [s]", "Speed [m/s]")
    plot.set_ylim(-0.5, 13.5)
    duration = max(len(r[0]) for r in runs.values()) * DT
    plot.set_xlim(0, duration + 0.2)
    lines = {}
    for row, (mode, title, color) in enumerate((("onestep", "One-step CBF filter", TEXT),
                                                ("bh", "BH-CBF filter (ours)", BLUE))):  # fmt: skip
        ax = fig.add_axes([0.035, 0.53 - row * 0.33, strip_w, strip_h])
        ax.set_xlim(x0, x1)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((x0, -3.5), x1 - x0, 7.0, fc="#ECECEC", ec="none", zorder=0))
        for y in (-3.5, 3.5):
            ax.plot([x0, x1], [y, y], color="0.45", lw=1.5, zorder=1)
        ax.set_title(title, fontsize=26, color=color, weight="bold", loc="left", pad=6)
        ax.add_patch(Polygon(rectangle(np.array([obstacle, 0.0, 0.0]), HALF), fc=CONTEXT, ec="white",
                             zorder=3))  # fmt: skip
        ax.text(obstacle, -1.35, "stopped vehicle", fontsize=17, color=MUTED, ha="center", va="top", zorder=5)
        tail = PolyCollection([], edgecolors="none", zorder=2)
        stop = PolyCollection([], facecolors="none", linewidths=1.6, zorder=2.1)
        ax.add_collection(tail)
        ax.add_collection(stop)
        vehicle = Polygon(np.zeros((4, 2)), fc=BLUE, ec="white", zorder=4)
        ax.add_patch(vehicle)
        chevron = Line2D([], [], color="white", lw=1.4, zorder=4.1)
        ax.add_line(chevron)
        (mark,) = ax.plot([], [], marker="*", ms=30, color=RED, mec="white", mew=1.2, zorder=7)
        status = ax.text(x0 + 0.5, -4.3, "", fontsize=22, color=TEXT, va="top", ha="left")
        tail_text = ax.text(0, 1.35, "", fontsize=17, color=BLUE, ha="center", va="bottom", zorder=5)
        strips[mode] = (ax, tail, stop, vehicle, chevron, mark, status, tail_text)
        (lines[mode],) = plot.plot([], [], color=color, lw=3)
    (red_line,) = plot.plot([], [], color=RED, lw=3)
    plot.legend([lines["onestep"], lines["bh"]], ["One-step CBF filter", "BH-CBF filter (ours)"],
                fontsize=17, frameon=False, loc="lower left")  # fmt: skip
    fig.text(
        0.965, 0.04,
        f"Simulated with the paper's model: Euler steps, Δt = {meta['dt']:g} s, acceleration ≥ −{-meta["a_min"]:g} m/s², λ = {meta['lam_lane']:g}.",
        fontsize=19, color=MUTED, ha="right", va="center",
    )  # fmt: skip
    texts = {
        "onestep": {"cruise": ("Keeps its speed: the one-step condition holds", TEXT),
                    "brake": ("Brakes only now: the margin has become small", TEXT),
                    "infeasible": ("Infeasible: even full braking violates the one-step condition", RED),
                    "collision": ("Collision at {v:.1f} m/s", RED)},
        "bh": {"cruise": ("Keeps its speed: its backup prediction stops in time", TEXT),
               "brake": ("Brakes once the gap at the end of its backup prediction would shrink too fast", BLUE),
               "stopped": ("Stopped with the required clearance of {gap:.1f} m, no collision", BLUE)},
    }  # fmt: skip
    # Fractional steps of the frames: at `speed`, with a pause where the one-step filter becomes
    # infeasible, and a hold at the end
    n_frames = int(duration / speed * FPS) + 1
    taus = [min(frame, n_frames - 1) * speed / (FPS * DT) for frame in range(n_frames + int(2.5 * FPS))]
    k_inf = next(k for k, q in enumerate(runs["onestep"][0]) if q["state"] == "infeasible")
    cut = next(i for i, tau in enumerate(taus) if tau >= k_inf)
    taus = taus[:cut] + [float(k_inf)] * int(2.5 * FPS) + taus[cut:]
    for tau in taus:
        for mode, (states, _) in runs.items():
            ax, tail, stop, vehicle, chevron, mark, status, tail_text = strips[mode]
            k = min(int(tau), len(states) - 1)
            s = states[k]
            f = 0.0 if k == len(states) - 1 else tau - k
            x = s["x"] + f * (states[min(k + 1, len(states) - 1)]["x"] - s["x"])
            pose = np.array([x, 0.0, 0.0])
            vehicle.set_xy(rectangle(pose, HALF))
            points = heading_chevron(pose, HALF)
            chevron.set_data(points[:, 0], points[:, 1])
            if mode == "bh" and s["state"] != "stopped":
                tail.set_verts([rectangle(np.array([p, 0.0, 0.0]), HALF) for p in s["backup"][1:]])
                tail.set_facecolors([tint(BLUE, 0.28)])
                stop.set_verts([rectangle(np.array([s["backup"][-1], 0.0, 0.0]), HALF)])
                stop.set_edgecolors([tint(BLUE, 0.85)])
                tail_text.set_position((0.5 * (s["backup"][1] + s["backup"][-1]), 1.35))
                tail_text.set_text("backup prediction (until standstill)")
            else:
                tail.set_verts([])
                stop.set_verts([])
                tail_text.set_text("")
            if s["state"] == "collision":
                mark.set_data([obstacle - HALF[0]], [0.0])
                vehicle.set_edgecolor(RED)
                vehicle.set_linewidth(3)
            text, color = texts[mode][s["state"]]
            status.set_text(text.format(v=states[k - 1]["v"] if k else s["v"], gap=s["gap"]))
            status.set_color(color)
            t = np.arange(k + 1) * DT
            v = np.array([q["v"] for q in states[: k + 1]])
            if s["state"] == "collision":
                v[-1] = states[k - 1]["v"]
            lines[mode].set_data(t, v)
            if mode == "onestep":
                bad = [i for i, q in enumerate(states[: k + 1]) if q["state"] in ("infeasible", "collision")]
                if bad:
                    red_line.set_data(t[bad[0] - 1 if bad[0] else 0 :], v[bad[0] - 1 if bad[0] else 0 :])
        clip.add(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scenes of one vehicle (lane constraint)


def lane_setup(title: str, subtitle: str, n: int = 16):
    """Figure, map panel, fleet (n vehicles), and labels of the lane scenes."""
    fig = new_figure(title, subtitle)
    panel = MapPanel(fig)
    colors = [CONTEXT] * n
    colors[LANE_AGENT] = BLUE
    fleet = panel.fleet(colors)
    label = panel.text("$i$", BLUE)
    return fig, panel, fleet, label


def place_label(label, pose, side: float = 3.0):
    visible = not np.isnan(pose).any()
    label.set_visible(visible)
    if visible:
        label.set_position(pose[:2] + side * np.array([np.cos(pose[2] + np.pi / 2), np.sin(pose[2] + np.pi / 2)]))


def ahead_note(panel: MapPanel, poses: np.ndarray, ego: int):
    """A note beside the nearest vehicle ahead of vehicle ego in its lane (hidden; empty if none)."""
    note = panel.note("", (0.0, 0.0), MUTED, ha="left", size=17, visible=False)
    update_ahead(note, poses, ego)
    return note


def update_ahead(note, poses: np.ndarray, ego: int):
    pose = poses[ego]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rel = poses[:, :2] - pose[:2]
    along, lateral = rel @ np.array([c, s]), rel @ np.array([s, -c])
    ahead = [k for k in range(len(poses))
             if k != ego and not np.isnan(poses[k]).any() and 2.0 < along[k] < 20.0 and abs(lateral[k]) < 2.5]  # fmt: skip
    if not ahead:
        note.set_text("")
        return
    k = min(ahead, key=lambda k: along[k])
    note.set_position(beside(poses[k], 0.0, 1.8))
    note.set_text("vehicle ahead: handled by\npair constraints (later)")


def violated_constraint(problem: dict) -> int:
    """The entry with the smallest slack of the policy's control (t >= 2)."""
    slack = (problem["values_nom"] - problem["rhs"])[:, 1:].min(axis=1)
    return int(np.argmin(slack))


def boundary_line(panel: MapPanel, problem: dict, e: int):
    """The facets of the lane constraint e, drawn thick."""
    lane = problem["lane"]
    s = list(lane["idx"]).index(e)
    valid = lane["valid"][s]
    points = np.concatenate([lane["start"][s][valid], lane["end"][s][valid][-1:]])
    line = Line2D(points[:, 0], points[:, 1], color=TEXT, lw=3.5, zorder=1.5, solid_capstyle="round")
    return panel.add(line)


def boundary_note(panel: MapPanel, line, pose: np.ndarray, side: str, ahead: float = 12.0):
    """The name of the boundary beside its thick line, on the outside of the lane."""
    point = on_polyline(np.column_stack(line.get_data()), pose, ahead)
    sign = -1.0 if side == "left" else 1.0
    return panel.note(f"{side} boundary", beside(np.array([*point, pose[2]]), 0.0, 0.8 * sign),
                      ha="right" if side == "left" else "left", visible=False)  # fmt: skip


def scene_backup(clip):
    """The backup prediction, the candidate prediction of the policy's
    control, and the BH-CBF condition on the lane margin; then the sweep from
    the policy's control to the filtered one."""
    meta, problems = load_problems(LANE_NAME)
    problem = problems[(LANE_STEP, LANE_AGENT)]
    e = violated_constraint(problem)
    side = {"L": "left", "R": "right"}[problem["labels"][e][1]]
    first = LANE_STEP - LANE_LEAD
    pose = window_poses(first, LANE_STEP + 1)
    n = pose.shape[1]
    fig, panel, fleet, label = lane_setup(
        "The backup prediction and the BH-CBF condition",
        "Vehicle $i$ (blue) in the roundabout, driven by a MARL policy without safety guarantees. One step $k$ of the filter, paused.",
    )
    caption = Caption(fig)
    route = panel.add(Line2D(*problem["routes"][problem["ego"]].T, color=ROUTE_COLOR, lw=2.2, zorder=1.6,
                             visible=False))  # fmt: skip
    boundary = boundary_line(panel, problem, e)
    boundary.set_visible(False)
    backup_fp = Footprints(panel, BACKUP_COLOR, zorder=2.0)
    candidate_fp = Footprints(panel, NOMINAL, zorder=2.2, strength=0.22)
    bad = Outlines(panel, width=1.5)

    # Right: speed and margin along the predictions over t = 1, ..., N
    T = len(problem["prediction_backup"])
    t = np.arange(1, T + 1)
    x0, x1 = RIGHT[0] + 0.07, RIGHT[1]
    ax_v = fig.add_axes([x0, 0.635, x1 - x0, 0.17])
    ax_h = fig.add_axes([x0, 0.255, x1 - x0, 0.33])
    style_plot(ax_v, None, "Speed [m/s]")
    style_plot(ax_h, "Prediction step $t$", f"Margin $h_e$ [m]\n({side} boundary)")
    for ax in (ax_v, ax_h):
        ax.set_xlim(0.5, T + 0.5)
        ax.set_xticks([1, 2] + list(range(5, T + 1, 5)))
    ax_v.set_xticklabels([])
    v_backup = speeds(problem["prediction_backup"])
    h_backup = problem["values_backup"][e]
    rhs = problem["rhs"][e, 1]
    path_h = problem["path_values"][:, e]
    lo = min(h_backup.min(), path_h.min(), problem["values_nom"][e].min())
    hi = max(h_backup.max(), path_h.max(), problem["values_nom"][e].max())
    ax_h.set_ylim(lo - 0.15 * (hi - lo) - 0.1, hi + 0.12 * (hi - lo))
    ax_v.set_ylim(-0.5, max(v_backup.max(), speeds(problem["prediction_nom"]).max()) + 1.5)
    (gray_v,) = ax_v.plot([], [], color=BACKUP_COLOR, lw=3, marker="o", ms=4)
    (gray_h,) = ax_h.plot([], [], color=BACKUP_COLOR, lw=3, marker="o", ms=4)
    (cand_v,) = ax_v.plot([], [], color=NOMINAL, lw=3, marker="o", ms=4)
    (cand_h,) = ax_h.plot([], [], color=NOMINAL, lw=3, marker="o", ms=4)
    (bad_h,) = ax_h.plot([], [], ls="none", marker="o", ms=11, mfc="none", mec=RED, mew=2.5)
    (bound,) = ax_h.plot([], [], color=RED, lw=2.5, ls="--")
    bound_text = ax_h.text(T + 0.4, rhs, "", fontsize=20, color=RED, ha="right", va="top")
    (b_dot,) = ax_h.plot([], [], ls="none", marker="o", ms=13, color=BACKUP_COLOR, mec="white", mew=1.5, zorder=5)
    b_text = ax_h.text(0, 0, "", fontsize=20, color=MUTED, ha="center", va="bottom")
    legend = fig.legend(
        [Line2D([], [], color=BACKUP_COLOR, lw=3), Line2D([], [], color=NOMINAL, lw=3),
         Line2D([], [], color=BLUE, lw=3)],
        ["Backup prediction", "Candidate (policy)", "Candidate (filtered)"],
        fontsize=18, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(x0 - 0.01, 0.845),
    )  # fmt: skip
    legend.set_visible(False)

    def show_candidate(poses, values, color):
        candidate_fp.update(poses, color=color)
        v = speeds(poses)
        cand_v.set_data(t, v)
        cand_v.set_color(color)
        cand_h.set_data(t, values)
        cand_h.set_color(color)
        violated = np.nonzero(values[1:] < rhs - meta["tol"])[0] + 1
        bad_h.set_data(t[violated], values[violated])
        bad.update(poses[violated])
        return len(violated)

    # Real-time approach
    for frame in range(int(LANE_LEAD * DT * FPS) + 1):
        tau = frame / (FPS * DT)
        poses = interpolate(pose, tau)
        panel.look_from(poses[LANE_AGENT])
        fleet.update(poses)
        place_label(label, poses[LANE_AGENT])
        panel.clock.set_text(f"time {(first + tau) * DT:4.1f} s   (real time)")
        clip.add(fig)
    poses = pose[LANE_STEP - first]
    panel.look_from(poses[LANE_AGENT])
    panel.clock.set_text(f"time {LANE_STEP * DT:4.1f} s   (paused at step $k$)")
    hold(clip, fig, 0.5)
    ego = poses[LANE_AGENT]
    route_point = on_polyline(problem["routes"][problem["ego"]], ego, -5.5)
    route_note = panel.note("route centerline", beside(np.array([*route_point, ego[2]]), 0.0, 1.0), ROUTE_COLOR,
                            ha="left")  # fmt: skip
    boundary_text = boundary_note(panel, boundary, ego, side)
    ahead_text = ahead_note(panel, poses, LANE_AGENT)

    # The backup prediction, step by step
    route.set_visible(True)
    caption.set("Backup prediction (gray): brake as hard as possible from now on and steer along the route centerline,",
                "until standstill. The prediction steps $t = 1, \\dots, N$ cover the braking until the vehicle has stopped.")  # fmt: skip
    for upto in range(1, T + 1):
        backup_fp.update(problem["prediction_backup"], upto)
        gray_v.set_data(t[:upto], v_backup[:upto])
        gray_h.set_data(t[:upto], h_backup[:upto])
        clip.add(fig, 3)
    ahead_text.set_visible(ahead_text.get_text() != "")
    hold(clip, fig, 3.5)
    boundary.set_visible(True)
    boundary_text.set_visible(True)
    b_t = int(np.argmin(h_backup))
    b_dot.set_data([t[b_t]], [h_backup[b_t]])
    b_text.set_position((t[b_t], h_backup[b_t] + 0.04 * (hi - lo)))
    b_text.set_text("$B_e(\\boldsymbol{x}_k)$")
    caption.set(f"Margin $h_e$ to the {side} boundary along the backup prediction. Its minimum is the backup value $B_e(\\boldsymbol{{x}}_k)$.")
    hold(clip, fig, 3.0)
    bound.set_data([1.5, T + 0.5], [rhs, rhs])
    bound_text.set_text("$(1-\\lambda)\\,B_e(\\boldsymbol{x}_k)$")
    caption.set(f"Margin $h_e$ to the {side} boundary along the backup prediction. Its minimum is the backup value $B_e(\\boldsymbol{{x}}_k)$.",
                "The BH-CBF condition: the margin of each candidate prediction must stay above $(1-\\lambda)\\,B_e(\\boldsymbol{x}_k)$ for $t = 2, \\dots, N$.")  # fmt: skip
    hold(clip, fig, 5.0)

    # The candidate prediction of the policy's control
    legend.set_visible(True)
    a, w = problem["u_nom"]
    count = show_candidate(problem["prediction_nom"], problem["values_nom"][e], NOMINAL)
    caption.set(f"Candidate prediction (purple): the policy's control now ($a = {a:.1f}$ m/s², $\\omega = {w:.2f}$ rad/s), then the same backup.",
                f"It steers toward the {side} boundary. The margin stays positive but shrinks too fast: below the red line at {count} steps.")  # fmt: skip
    hold(clip, fig, 7.0)

    # Sweep from the policy's control to the filtered one
    caption.set("The filter moves the control until the margin stays above the red line.")
    hold(clip, fig, 1.0)
    n_path = len(problem["path_s"])
    for frame in range(int(4.0 * FPS)):
        s = int(round(frame / (4.0 * FPS - 1) * (n_path - 1)))
        color = BLUE if s == n_path - 1 else NOMINAL
        show_candidate(problem["path_poses"][s], problem["path_values"][s, e], color)
        clip.add(fig)
    a, w = problem["u_applied"]
    show_candidate(problem["prediction_applied"], problem["values_applied"][e], BLUE)
    caption.set(f"Filtered control (blue): $a = {a:.1f}$ m/s², $\\omega = {w:.2f}$ rad/s. It still brakes fully, but steers less.",
                "Its candidate prediction keeps the margin above $(1-\\lambda)\\,B_e(\\boldsymbol{x}_k)$.")  # fmt: skip
    hold(clip, fig, 6.0)
    plt.close(fig)


class ControlPlane:
    """The plane of the controls (acceleration, steering rate) of one filter
    problem: the admissible box, the exact set of controls that satisfy all
    BH-CBF inequalities of the vehicle (blue; red tint: violated), and markers.
    The axes are scaled by the QP weights, so that distances look as the QP
    measures them."""

    def __init__(self, ax, meta: dict, title: str = None, color: str = BLUE, labels: bool = True,
                 notes: bool = False):  # fmt: skip
        """notes: also name the QP candidate, the linearized constraints, and the cost level."""
        self.ax, self.meta, self.color, self.notes = ax, meta, color, notes
        style_plot(ax, "Acceleration $a$ [m/s²]", "Steering rate $\\omega$ [rad/s]")
        W = meta["W"]
        ax.set_aspect(np.sqrt(W[0, 0] / W[1, 1]) ** -1 * 1.0)
        if title:
            ax.set_title(title, fontsize=24, color=color, weight="bold", pad=8)
        self.fills = []
        self.box = ax.add_patch(Rectangle((0, 0), 1, 1, fc="none", ec=TEXT, lw=2, zorder=3))
        self.rows = []
        self.circle = ax.add_patch(Ellipse((0, 0), 0, 0, fc="none", ec=MUTED, lw=1.5, ls="--", zorder=4,
                                           visible=False))  # fmt: skip
        self.arrow = None
        kw = dict(ls="none", zorder=6, mew=2.5)
        (self.nominal,) = ax.plot([], [], marker="o", ms=17, mfc="white", mec=NOMINAL, **kw)
        (self.backup,) = ax.plot([], [], marker="D", ms=13, mfc=BACKUP_COLOR, mec="white", **kw)
        (self.applied,) = ax.plot([], [], marker="o", ms=15, mfc=color, mec="white", **kw)
        (self.candidate,) = ax.plot([], [], marker="X", ms=16, mfc=TEXT, mec="white", mew=1.5, ls="none",
                                    zorder=7)  # fmt: skip
        self.labels = {}
        self.row_note = ax.text(0, 0, "linearized constraints", fontsize=18, color=TEXT, ha="right", va="bottom",
                                zorder=8, path_effects=white_stroke(3), visible=False)  # fmt: skip
        self.circle_note = ax.text(0, 0, "same QP cost", fontsize=18, color=MUTED, ha="left", va="bottom",
                                   zorder=8, path_effects=white_stroke(3), visible=False)  # fmt: skip
        if labels:
            keys = (("nominal", "policy", NOMINAL), ("backup", "backup", MUTED), ("applied", "filtered", color))
            for key, text, c in keys + ((("candidate", "QP candidate", TEXT),) if notes else ()):
                self.labels[key] = ax.annotate(text, (0, 0), xytext=(0, 0), textcoords="offset points",
                                               fontsize=19, color=c, weight="bold", zorder=8, va="center",
                                               path_effects=white_stroke(3), visible=False)  # fmt: skip

    def set_problem(self, problem: dict, limits=None):
        ax, meta = self.ax, self.meta
        for artist in self.fills:
            artist.remove()
        lower, upper = problem["bounds"]
        if limits is None:
            pad = 0.06 * (upper - lower)
            limits = (lower[0] - pad[0], upper[0] + pad[0], lower[1] - pad[1], upper[1] + pad[1])
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        # Ticks at the bounds of the acceleration and at multiples of 3 m/s² between them
        ticks = np.arange(np.ceil(lower[0] / 3) * 3, upper[0] + 1e-9, 3.0)
        ax.set_xticks(np.unique(np.round(np.concatenate([[lower[0]], ticks, [upper[0]]]), 6)))
        a, w = problem["grid_a"], problem["grid_w"]
        field = problem["grid_slack"].min(axis=-1).T
        filled = ax.contourf(a, w, field, levels=[-1e9, -meta["tol"], 1e9],
                             colors=[INFEASIBLE, tint(self.color, 0.3)], zorder=1)  # fmt: skip
        edge = ax.contour(a, w, field, levels=[-meta["tol"]], colors=[self.color], linewidths=2.5, linestyles="solid", zorder=2)
        self.fills = [filled, edge]
        self.box.set_xy(lower)
        self.box.set_width(upper[0] - lower[0])
        self.box.set_height(upper[1] - lower[1])

    def mark(self, key: str, u):
        """Show (u: control [2]) or hide (u: None) a marker and its label."""
        marker = getattr(self, key)
        if u is None:
            marker.set_data([], [])
        else:
            marker.set_data([u[0]], [u[1]])
        self._place_labels()

    def _place_labels(self):
        """Labels beside their markers, on the side away from the nearer edge
        of the plane; one label where the policy's control is kept."""
        x0, x1 = self.ax.get_xlim()
        points = {key: getattr(self, key).get_xydata() for key in self.labels}
        same = (len(points.get("nominal", [])) and len(points.get("applied", []))
                and np.allclose(points["nominal"][0], points["applied"][0], atol=1e-3))  # fmt: skip
        for key, label in self.labels.items():
            data = points[key]
            visible = len(data) > 0 and not (same and key == "applied")
            label.set_visible(bool(visible))
            if not visible:
                continue
            u = data[0]
            left = u[0] > 0.5 * (x0 + x1)
            label.xy = (u[0], u[1])
            label.set_text({"nominal": "policy = filtered" if same else "policy", "backup": "backup",
                            "applied": "filtered", "candidate": "QP candidate"}[key])  # fmt: skip
            label.set_ha("right" if left else "left")
            label.xyann = (-14 if left else 14, {"nominal": 12, "backup": -12, "applied": -14, "candidate": -16}[key])
        # Labels that overlap an earlier one move to the other side (vertically, then horizontally)
        renderer = self.ax.figure.canvas.get_renderer()
        placed = []
        for label in self.labels.values():
            if not label.get_visible():
                continue
            for flip in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                dx, dy = label.xyann
                trial = (abs(dx) * np.sign(dx) * flip[0], dy * flip[1])
                label.xyann = trial
                label.set_ha({1: "left", -1: "right"}[int(np.sign(trial[0]))])
                box = label.get_window_extent(renderer)
                if not any(box.overlaps(other) for other in placed):
                    break
                label.xyann = (dx, dy)
            label.set_ha({1: "left", -1: "right"}[int(np.sign(label.xyann[0]))])
            placed.append(label.get_window_extent(renderer))

    def show_rows(self, rows: dict, lin_margin: float, rhs: np.ndarray, u_lin):
        """QP constraints: value + grad (u - u_lin) >= rhs + margin, as dashed lines."""
        for line in self.rows:
            line.remove()
        self.rows = []
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        for (e, t), (value, grad) in rows.items():
            c = rhs[e, t] + lin_margin - value + grad @ u_lin
            if abs(grad[1]) * (x1 - x0) >= abs(grad[0]) * (y1 - y0) and abs(grad[1]) > 1e-12:
                xs = np.array([x0, x1])
                ys = (c - grad[0] * xs) / grad[1]
            elif abs(grad[0]) > 1e-12:
                ys = np.array([y0, y1])
                xs = (c - grad[1] * ys) / grad[0]
            else:
                continue
            if max(ys) < y0 or min(ys) > y1 or max(xs) < x0 or min(xs) > x1:
                continue  # The line misses the plane (a nearly constant row)
            (line,) = self.ax.plot(xs, ys, color=TEXT, lw=1.4, ls=(0, (4, 3)), zorder=4)
            self.rows.append(line)
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        x = x1 - 0.02 * (x1 - x0)
        heights = [y for y in (np.interp(x, *sorted_xy(line)) for line in self.rows) if y0 < y < y1]
        if self.notes and heights:
            # Above the highest line at the right end of the plane
            self.row_note.set_position((x, min(max(heights), y1 - 0.08 * (y1 - y0)) + 0.01 * (y1 - y0)))
            self.row_note.set_visible(True)

    def hide_rows(self):
        for line in self.rows:
            line.remove()
        self.rows = []
        self.row_note.set_visible(False)

    def show_projection(self, u_from, u_to):
        if self.arrow is not None:
            self.arrow.remove()
            self.arrow = None
        if u_from is None or np.allclose(u_from, u_to):
            self.circle.set_visible(False)
            self.circle_note.set_visible(False)
            return
        self.arrow = self.ax.add_patch(FancyArrowPatch(u_from, u_to, arrowstyle="-|>,head_length=8,head_width=5",
                                                       color=self.color, lw=2.5, shrinkA=9, shrinkB=9, zorder=5))  # fmt: skip
        # The QP's cost level through the filtered control: a circle in the weighted axes
        W = self.meta["W"]
        d = np.asarray(u_to) - np.asarray(u_from)
        r = np.sqrt(d @ W @ d)
        self.circle.set_center(u_from)
        self.circle.width, self.circle.height = 2 * r / np.sqrt(W[0, 0]), 2 * r / np.sqrt(W[1, 1])
        self.circle.set_visible(True)
        if self.notes:
            # Upper right on the circle, or lower right if that leaves the plane
            top = u_from[1] + np.sin(np.pi / 4) * r / np.sqrt(W[1, 1])
            angle = np.pi / 4 if top < self.ax.get_ylim()[1] - 0.06 * np.ptp(self.ax.get_ylim()) else -np.pi / 4
            self.circle_note.set_position((u_from[0] + np.cos(angle) * r / np.sqrt(W[0, 0]),
                                           u_from[1] + np.sin(angle) * r / np.sqrt(W[1, 1])))  # fmt: skip
            self.circle_note.set_va("bottom" if angle > 0 else "top")
            self.circle_note.set_visible(True)


def sorted_xy(line) -> tuple:
    """The data (x, y) of a line, sorted by x (for np.interp)."""
    x, y = (np.asarray(v, dtype=float) for v in line.get_data())
    order = np.argsort(x)
    return x[order], y[order]


def scene_control_plane(clip):
    """The filter problem of the lane scene in the plane of the controls;
    then the following steps, one by one."""
    meta, problems = load_problems(LANE_NAME)
    problem = problems[(LANE_STEP, LANE_AGENT)]
    pose = window_poses(LANE_STEP, LANE_STEP + LANE_NEXT + 1)
    n = pose.shape[1]
    fig, panel, fleet, label = lane_setup(
        "The filter in the plane of the controls",
        "The same filter problem. Each point on the right is a control $\\boldsymbol{u}^i = (a^i, \\omega^i)$ of vehicle $i$ at step $k$.",
    )
    caption = Caption(fig)
    nominal_fp = Footprints(panel, NOMINAL, zorder=2.1, strength=0.22)
    backup_fp = Footprints(panel, BACKUP_COLOR, zorder=2.0)
    applied_fp = Footprints(panel, BLUE, zorder=2.2)
    candidate_fp = Outlines(panel, TEXT, width=1.3, zorder=2.3, linestyle=(0, (3, 2)))  # QP candidate
    ax = fig.add_axes([RIGHT[0] + 0.07, PANEL_BOTTOM + 0.07, RIGHT[1] - RIGHT[0] - 0.08, PANEL_TOP - PANEL_BOTTOM - 0.07])
    plane = ControlPlane(ax, meta, notes=True)
    plane.set_problem(problem)
    legend = [Rectangle((0, 0), 1, 1, fc=tint(BLUE, 0.3), ec=BLUE, lw=2),
              Rectangle((0, 0), 1, 1, fc=INFEASIBLE, ec="none"),
              Rectangle((0, 0), 1, 1, fc="none", ec=TEXT, lw=2)]  # fmt: skip
    fig.legend(legend, ["All inequalities of $i$ hold", "At least one violated", "Admissible controls"],
               fontsize=18, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(RIGHT[0] + 0.03, 0.845),
               columnspacing=1.2, handlelength=1.6)  # fmt: skip

    def show_step(step):
        poses = pose[step - LANE_STEP]
        panel.look_from(poses[LANE_AGENT])
        fleet.update(poses)
        place_label(label, poses[LANE_AGENT])
        update_ahead(ahead_text, poses, LANE_AGENT)
        ahead_text.set_visible(ahead_text.get_text() != "")
        panel.clock.set_text(f"time {step * DT:4.1f} s   (step $k{'+' + str(step - LANE_STEP) if step > LANE_STEP else ''}$, paused)")

    ahead_text = ahead_note(panel, pose[0], LANE_AGENT)
    show_step(LANE_STEP)
    caption.set("Box: the admissible controls. Blue: the controls whose candidate prediction satisfies every inequality of vehicle $i$",
                "(lane boundaries, and neighbors as shown later), evaluated exactly on a grid for this illustration.")  # fmt: skip
    hold(clip, fig, 7.0)
    plane.mark("nominal", problem["u_nom"])
    nominal_fp.update(problem["prediction_nom"])
    caption.set("Purple: the policy's control $\\boldsymbol{u}^i_{\\mathrm{nom},k}$ lies outside the blue set, as in the previous scene.")
    hold(clip, fig, 4.0)
    plane.mark("backup", problem["backup_control"])
    backup_fp.update(problem["prediction_backup"])
    caption.set("Gray: the backup control $\\boldsymbol{\\kappa}^i(\\boldsymbol{x}^i_k)$ (full braking along the route). It lies in the blue set whenever",
                "the state is recoverable, so the filter problem has a solution. The next scene shows that this holds at every step.")  # fmt: skip
    hold(clip, fig, 7.0)
    plane.show_rows(problem["initial_rows"], meta["lin_margin"], problem["rhs"], problem["u_nom"])
    rounds = problem["rounds"]
    caption.set("The Quadratic Program (QP) uses the inequalities, linearized at the policy's control (dashed lines),",
                "and returns the candidate control closest to the policy's control that satisfies them (dashed outline on the map).")  # fmt: skip
    plane.mark("candidate", rounds[0]["u"])
    candidate_fp.update(rounds[0]["prediction"])
    hold(clip, fig, 7.0)
    for r, rnd in enumerate(rounds[1:]):
        # The added QP constraints, linearized at the previous candidate
        plane.show_rows(rounds[r].get("rows", {}), meta["lin_margin"], problem["rhs"], rounds[r]["u"])
        plane.mark("candidate", rnd["u"])
        candidate_fp.update(rnd["prediction"])
        caption.set(f"Repair {r + 1}: the violated inequalities are added, linearized at the candidate, and the QP is solved again.")
        hold(clip, fig, 3.0)
    passed = not rounds[-1]["violated"]
    caption.set("The exact check of every inequality at every $t$ " + ("accepts the candidate." if passed else
                "rejects the candidate; the vehicle falls back to its backup control."),
                "Otherwise, the violated inequalities would be added and the QP solved again; the backup control is the last resort."
                if passed else "")  # fmt: skip
    hold(clip, fig, 6.0)
    caption.set("Blue: the filtered control $\\boldsymbol{u}^i_k$. The axes are scaled by the QP weights, so the QP cost grows with the distance.")
    plane.hide_rows()
    plane.mark("candidate", None)
    candidate_fp.update(None)
    plane.mark("applied", problem["u_applied"])
    plane.show_projection(problem["u_nom"], problem["u_applied"])
    applied_fp.update(problem["prediction_applied"])
    hold(clip, fig, 6.0)

    # The following steps, one by one: a new problem at every step
    for step in range(LANE_STEP + 1, LANE_STEP + LANE_NEXT + 1):
        p = problems.get((step, LANE_AGENT))
        if p is None:
            continue
        show_step(step)
        plane.set_problem(p)
        plane.mark("nominal", p["u_nom"])
        plane.mark("backup", p["backup_control"])
        plane.mark("applied", p["u_applied"])
        plane.show_projection(p["u_nom"], p["u_applied"])
        nominal_fp.update(p["prediction_nom"])
        backup_fp.update(None)
        applied_fp.update(p["prediction_applied"])
        moved = not np.allclose(p["u_nom"], p["u_applied"], atol=1e-3)
        caption.set(f"Step $k+{step - LANE_STEP}$: a new filter problem from the new state. "
                    + ("The policy's control is outside the blue set, so the filter moves it." if moved else
                       "The policy's control is inside the blue set, so the filter keeps it."),
                    "The policy's control jumps from step to step; the filter changes it only where needed.")  # fmt: skip
        hold(clip, fig, 5.0)
    plt.close(fig)


def scene_recursive(clip):
    """The candidate prediction of the filtered control at k is the backup
    prediction at k + 1, so the backup value keeps its bound and the backup
    control stays feasible."""
    meta, problems = load_problems(LANE_NAME)
    now, nxt = problems[(LANE_STEP, LANE_AGENT)], problems[(LANE_STEP + 1, LANE_AGENT)]
    e = violated_constraint(now)
    label_e = now["labels"][e]
    e1 = entry(nxt, label_e)
    side = {"L": "left", "R": "right"}[label_e[1]]
    pose = window_poses(LANE_STEP, LANE_STEP + 1)
    fig, panel, fleet, label = lane_setup(
        "Why the filter always finds a safe control",
        "Vehicle $i$ at steps $k$ and $k+1$. Recursive feasibility in one picture.",
    )
    caption = Caption(fig)
    boundary = boundary_line(panel, now, e)
    applied_fp = Footprints(panel, BLUE, zorder=2.2)
    next_fp = panel.add(PolyCollection([], facecolors="none", edgecolors=BACKUP_COLOR, linewidths=2.2,
                                       linestyles="--", zorder=3))  # fmt: skip
    boundary_note(panel, boundary, pose[0, LANE_AGENT], side).set_visible(True)
    T = len(now["prediction_applied"])
    x0, x1 = RIGHT[0] + 0.07, RIGHT[1]
    ax = fig.add_axes([x0, 0.255, x1 - x0, 0.53])
    style_plot(ax, "Prediction step $t$ (counted from step $k$)", f"Margin $h_e$ [m] ({side} boundary)")
    steps = np.arange(1, T + 1)
    ax.set_xlim(0.5, T + 1.5)
    ax.set_xticks([1, 2] + list(range(5, T + 2, 5)))
    h_now = now["values_applied"][e]
    h_backup_now = now["values_backup"][e]
    h_next = nxt["values_backup"][e1]
    rhs = now["rhs"][e, 1]
    B_now, B_next = h_backup_now.min(), h_next.min()
    rhs_next = nxt["rhs"][e1, 1]  # (1 - lambda) B_e(x_{k+1})
    values = np.concatenate([h_now, h_next, h_backup_now])
    lo, hi = values.min(), values.max()
    ax.set_ylim(min(lo, rhs, rhs_next) - 0.12 * (hi - lo), hi + 0.15 * (hi - lo))
    (gray,) = ax.plot(steps, h_backup_now, color=BACKUP_COLOR, lw=3, marker="o", ms=4)
    (blue,) = ax.plot([], [], color=BLUE, lw=3, marker="o", ms=5)
    (nxt_line,) = ax.plot([], [], color=BACKUP_COLOR, lw=2.5, ls="--", marker="o", ms=11, mfc="none", mew=2)
    ax.plot([1.5, T + 1.5], [rhs, rhs], color=RED, lw=2.5, ls="--")
    ax.text(T + 1.4, rhs, "$(1-\\lambda)\\,B_e(\\boldsymbol{x}_k)$", fontsize=20, color=RED, ha="right", va="top")
    (b_now,) = ax.plot([steps[np.argmin(h_backup_now)]], [B_now], ls="none", marker="o", ms=13, color=BACKUP_COLOR,
                       mec="white", mew=1.5, zorder=5)  # fmt: skip
    b_now_text = ax.text(steps[np.argmin(h_backup_now)], B_now + 0.04 * (hi - lo), "$B_e(\\boldsymbol{x}_k)$",
                         fontsize=20, color=MUTED, ha="center", va="bottom")  # fmt: skip
    (b_next,) = ax.plot([], [], ls="none", marker="o", ms=15, color=BLUE, mec="white", mew=1.5, zorder=6)
    b_next_text = ax.text(0, 0, "", fontsize=20, color=BLUE, ha="center", va="top")
    fig.legend([Line2D([], [], color=BACKUP_COLOR, lw=3), Line2D([], [], color=BLUE, lw=3),
                Line2D([], [], color=BACKUP_COLOR, lw=2.5, ls="--", marker="o", mfc="none", ms=10)],
               ["Backup at $k$", "Filtered candidate at $k$", "Backup at $k+1$"],
               fontsize=18, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(x0 - 0.01, 0.845))  # fmt: skip

    poses = pose[0]
    panel.look_from(poses[LANE_AGENT])
    fleet.update(poses)
    place_label(label, poses[LANE_AGENT])
    panel.clock.set_text(f"time {LANE_STEP * DT:4.1f} s   (step $k$)")
    applied_fp.update(now["prediction_applied"])
    blue.set_data(steps, h_now)
    caption.set("At step $k$: the candidate prediction of the filtered control (applied now, then the backup) is blue.",
                "Its margin stays above $(1-\\lambda)\\,B_e(\\boldsymbol{x}_k)$ at every $t \\geq 2$.")  # fmt: skip
    hold(clip, fig, 6.0)
    # One step: the vehicle moves (map fixed), then the new backup prediction
    for frame in range(int(1.0 * FPS) + 1):
        tau = frame / FPS
        poses = interpolate(pose, tau)
        fleet.update(poses)
        place_label(label, poses[LANE_AGENT])
        clip.add(fig)
    panel.clock.set_text(f"time {(LANE_STEP + 1) * DT:4.1f} s   (step $k+1$)")
    next_fp.set_verts([rectangle(p, HALF) for p in nxt["prediction_backup"]])
    nxt_line.set_data(steps[:len(h_next)] + 1, h_next)
    caption.set("At step $k+1$, the backup prediction (dashed) is exactly the candidate prediction made at $k$: after the applied control,",
                "both brake along the route. So the new backup value $B_e(\\boldsymbol{x}_{k+1})$ is the smallest blue margin from $t = 2$ on.")  # fmt: skip
    hold(clip, fig, 8.0)
    t_next = steps[int(np.argmin(h_next))] + 1
    b_next.set_data([t_next], [B_next])
    b_next_text.set_position((t_next, B_next - 0.05 * (hi - lo)))
    b_next_text.set_text("$B_e(\\boldsymbol{x}_{k+1})$")
    caption.set("Hence $B_e(\\boldsymbol{x}_{k+1}) \\geq (1-\\lambda)\\,B_e(\\boldsymbol{x}_k) \\geq 0$: the state stays recoverable, and the backup control",
                "satisfies the BH-CBF condition again at $k+1$.")  # fmt: skip
    hold(clip, fig, 4.0)
    # The new bound of step k + 1 (its prediction steps start one step later)
    ax.plot([2.5, T + 1.5], [rhs_next, rhs_next], color=tint(RED, 0.5), lw=2.5, ls="--")
    ax.text(T + 1.4, rhs_next, "$(1-\\lambda)\\,B_e(\\boldsymbol{x}_{k+1})$", fontsize=20, color=tint(RED, 0.6), ha="right",
            va="top")  # fmt: skip
    caption.set("The bound of step $k+1$ (light red): the backup prediction at $k+1$ (dashed) stays above it, so the filter has a solution.",
                "By induction, the filter finds a safe control at every step, starting from a recoverable state (under the paper's model).")  # fmt: skip
    hold(clip, fig, 6.0)
    del b_now, b_now_text, gray
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scene of two vehicles (decentralized filter)


def budget_geometry(problem: dict, other: int, t: int) -> dict:
    """Separating normal (from the other vehicle to this one), the support of
    this vehicle's backup footprint along it, and its share of the budget at
    prediction index t."""
    e = entry(problem, ("pair", other))
    k = list(problem["budget"]["idx"]).index(e)
    return dict(normal=problem["budget"]["witness"][t, k], support=problem["budget"]["ref_support"][t, k],
                share_budget=problem["budget"]["budget"][t, k], share=problem["shares"][k][0], e=e)  # fmt: skip


def band(normal, support, width, center, length=9.0):
    """A strip perpendicular to normal: projections in [support - width,
    support] (the share of a vehicle toward the other), around center."""
    tangent = np.array([-normal[1], normal[0]])
    base = center - (center @ normal) * normal
    p0 = base + (support - width) * normal
    p1 = base + support * normal
    return np.array([p0 - length / 2 * tangent, p1 - length / 2 * tangent, p1 + length / 2 * tangent,
                     p0 + length / 2 * tangent])  # fmt: skip


def scene_pair(clip, speed: float = 0.5):
    """Two vehicles: the backup value of their pair, the shares of the
    budget, the corner budgets at one prediction step, and the two filter
    problems, one per vehicle."""
    meta, problems = load_problems(PAIR_NAME)
    pi, pj = problems[(PAIR_STEP, PAIR_I)], problems[(PAIR_STEP, PAIR_J)]
    first = PAIR_STEP - PAIR_LEAD
    pose = window_poses(first, PAIR_STEP + PAIR_AFTER + 1)
    n = pose.shape[1]
    colors = [CONTEXT] * n
    colors[PAIR_I], colors[PAIR_J] = BLUE, J_COLOR
    fig = new_figure(
        "Two vehicles: the fully decentralized filter",
        "Vehicle $i$ (blue) approaches vehicle $j$ (green). Each solves its own filter problem from the broadcast states and routes.",
    )
    caption = Caption(fig)
    panel = MapPanel(fig, view=PAIR_VIEW)
    fleet = panel.fleet(colors)
    labels = {PAIR_I: panel.text("$i$", BLUE), PAIR_J: panel.text("$j$", J_COLOR)}
    center = 0.5 * (pose[PAIR_LEAD, PAIR_I, :2] + pose[PAIR_LEAD, PAIR_J, :2])
    panel.set_window(PAIR_VIEW, center)
    backup_i = Footprints(panel, BLUE, zorder=2.0, strength=0.22)
    backup_j = Footprints(panel, J_COLOR, zorder=2.0, strength=0.22)
    # Zoomed elements at one prediction step
    shares_fill = panel.add(PolyCollection([], edgecolors="none", zorder=1.8))
    sep_line = panel.add(Line2D([], [], color=TEXT, lw=1.6, ls=(0, (6, 4)), zorder=2.6))
    rects = panel.add(PolyCollection([], facecolors="none", linewidths=2.4, zorder=4))
    dots = panel.add(Line2D([], [], ls="none", marker="o", ms=12, color=RED, mec="white", mew=1.5, zorder=7))
    arrow = None

    ii, jj = pi["agents"].index(PAIR_I), pi["agents"].index(PAIR_J)
    ref_i, ref_j = pi["braking"][:, ii], pi["braking"][:, jj]
    T = len(ref_i)
    t = np.arange(1, T + 1)
    h_backup = pair_margins(meta, ref_i, ref_j)
    B = h_backup.min()
    lam = pi["shares"][list(pi["budget"]["idx"]).index(entry(pi, ("pair", PAIR_J)))][1]
    floor = (1 - lam) * B
    rho = budget_geometry(pi, PAIR_J, 0)["share"]
    eta = h_backup - floor
    # Candidate predictions: both policies' controls, and both filtered controls
    nj = len(pj["prediction_nom"])
    h_nom = pair_margins(meta, pi["prediction_nom"][: min(T, nj)], pj["prediction_nom"][: min(T, nj)])
    h_app = pair_margins(meta, pi["prediction_applied"][: min(T, nj)], pj["prediction_applied"][: min(T, nj)])

    x0, x1 = RIGHT[0] + 0.07, RIGHT[1]
    ax = fig.add_axes([x0, 0.255, x1 - x0, 0.53])
    style_plot(ax, "Prediction step $t$", "Pair margin $h^{ij}$ [m]")
    ax.set_xlim(0.5, T + 0.5)
    ax.set_xticks([1, 2] + list(range(5, T + 1, 5)))
    top = max(h_backup.max(), h_nom.max(), h_app.max())
    ax.set_ylim(min(h_nom.min(), floor) - 0.6, top + 0.8)
    (gray,) = ax.plot([], [], color=BACKUP_COLOR, lw=3, marker="o", ms=4)
    (bound,) = ax.plot([], [], color=RED, lw=2.5, ls="--")
    bound_text = ax.text(2.2, floor - 0.08, "", fontsize=20, color=RED, ha="left", va="top")
    (b_dot,) = ax.plot([], [], ls="none", marker="o", ms=13, color=BACKUP_COLOR, mec="white", mew=1.5, zorder=5)
    b_text = ax.text(0, 0, "", fontsize=20, color=MUTED, ha="center", va="bottom")
    fills = []
    (nom_line,) = ax.plot([], [], color=NOMINAL, lw=3, marker="o", ms=4)
    (app_line,) = ax.plot([], [], color=BLUE, lw=3, marker="o", ms=4)
    t_mark = ax.axvline(0, color=TEXT, lw=1.2, ls=":", visible=False)
    t_text = ax.text(0, 1.0, "", transform=ax.get_xaxis_transform(), fontsize=20, color=TEXT, ha="center",
                     va="bottom")  # fmt: skip
    nom_text = ax.text(0, 0, "", fontsize=20, color=NOMINAL, ha="left", va="center", path_effects=white_stroke(3))
    app_text = ax.text(0, 0, "", fontsize=20, color=BLUE, ha="left", va="top", path_effects=white_stroke(3))

    def show_map(poses, tau_text):
        fleet.update(poses)
        for k, lab in labels.items():
            place_label(lab, poses[k], 3.2)
        panel.clock.set_text(tau_text)

    # Real-time approach
    for frame in range(int(PAIR_LEAD * DT * FPS) + 1):
        tau = frame / (FPS * DT)
        show_map(interpolate(pose, tau), f"time {(first + tau) * DT:4.1f} s   (real time)")
        clip.add(fig)
    show_map(pose[PAIR_LEAD], f"time {PAIR_STEP * DT:4.1f} s   (paused at step $k$)")
    hold(clip, fig, 0.5)

    # Backup predictions and the pair's backup value
    caption.set("Both vehicles compute both backup predictions (light) from the broadcast states and routes.",
                "Their pair margin $h^{ij}$ (gray) has its minimum, the backup value $B^{ij}(\\boldsymbol{x}_k)$, where the predictions come closest.")  # fmt: skip
    for upto in range(1, T + 1):
        backup_i.update(ref_i, upto)
        backup_j.update(ref_j, upto)
        gray.set_data(t[:upto], h_backup[:upto])
        clip.add(fig, 3)
    hold(clip, fig, 1.0)
    tb = int(np.argmin(h_backup))
    b_dot.set_data([t[tb]], [B])
    b_text.set_position((t[tb], B + 0.25))
    b_text.set_text("$B^{ij}(\\boldsymbol{x}_k)$")
    bound.set_data([1.5, T + 0.5], [floor, floor])
    bound_text.set_text("$(1-\\lambda)\\,B^{ij}(\\boldsymbol{x}_k)$")
    hold(clip, fig, 4.5)

    # Shares of the budget
    fills.append(ax.fill_between(t[1:], floor, floor + rho * eta[1:], color=tint(BLUE, 0.45), lw=0, zorder=0))
    fills.append(ax.fill_between(t[1:], floor + rho * eta[1:], h_backup[1:], color=tint(J_COLOR, 0.45), lw=0, zorder=0))
    # The budget (a double arrow from the red line to the gray curve) and the two shares
    tq = 11
    ax.annotate("", (tq, floor), (tq, h_backup[tq - 1]), zorder=3,
                             arrowprops=dict(arrowstyle="<|-|>", color=TEXT, lw=1.6, shrinkA=0, shrinkB=0))  # fmt: skip
    ax.text(tq + 0.3, floor + 0.55 * eta[tq - 1], "$\\eta^{ij}_{k+t|k}$", fontsize=22, color=TEXT,
                         ha="left", va="center", path_effects=white_stroke(3))  # fmt: skip
    ts_i = 5
    ax.text(ts_i, floor + 0.45 * rho * eta[ts_i - 1], "$\\rho^{ij}\\eta^{ij}_{k+t|k}$", fontsize=22,
                         color=TEXT, ha="center", va="center", path_effects=white_stroke(3))  # fmt: skip
    ts_j = 7
    ax.annotate("$\\rho^{ji}\\eta^{ij}_{k+t|k}$", (ts_j, h_backup[ts_j - 1] - 0.5 * (1 - rho) * eta[ts_j - 1]),
                             (ts_j + 2.2, h_backup[ts_j - 1] + 1.6), fontsize=22, color=J_COLOR, ha="left", va="center",
                             path_effects=white_stroke(3),
                             arrowprops=dict(arrowstyle="-", color=J_COLOR, lw=1.6, shrinkA=2, shrinkB=0))  # fmt: skip
    caption.set("The budget $\\eta^{ij}_{k+t|k}$ between the gray curve and the red line is split into shares: "
                f"$\\rho^{{ij}} = {rho:.2f}$ for $i$, which approaches $j$,",
                f"and $\\rho^{{ji}} = {1 - rho:.2f}$ for $j$. Each vehicle may deviate from its backup prediction only within its share.")  # fmt: skip
    hold(clip, fig, 7.0)

    # Corner budgets at one prediction step: zoom into the step where the policy's control of i violates most
    gi = budget_geometry(pi, PAIR_J, 0)
    slack_i = (pi["values_nom"][gi["e"]] - pi["rhs"][gi["e"]])[1:]
    ts = int(np.argmin(slack_i)) + 1  # Code index of the step
    gi = budget_geometry(pi, PAIR_J, ts)
    gj = budget_geometry(pj, PAIR_I, ts)
    mid = 0.5 * (ref_i[ts, :2] + ref_j[ts, :2])
    t_mark.set_xdata([ts + 1, ts + 1])
    t_mark.set_visible(True)
    t_text.set_x(ts + 1)
    t_text.set_text(f"$t = {ts + 1}$")
    zoom = 11.0
    for frame in range(int(1.5 * FPS) + 1):
        f = 0.5 - 0.5 * np.cos(np.pi * frame / (1.5 * FPS))
        panel.set_window(PAIR_VIEW + f * (zoom - PAIR_VIEW), (1 - f) * center + f * mid)
        if frame == 0:
            backup_i.update(ref_i[ts : ts + 1])
            backup_j.update(ref_j[ts : ts + 1])
        clip.add(fig)
    for lab in labels.values():
        lab.set_visible(False)
    fleet.update(np.full((n, 3), np.nan))  # Only the footprints at this step while zoomed
    normal = gi["normal"]
    shares_fill.set_verts([band(normal, gi["support"], gi["share_budget"], mid),
                           band(-normal, gj["support"], gj["share_budget"], mid)])  # fmt: skip
    shares_fill.set_facecolors([tint(BLUE, 0.45), tint(J_COLOR, 0.45)])
    tangent = np.array([-normal[1], normal[0]])
    # Separating line halfway between the backup footprints, perpendicular to the normal
    sep = 0.5 * (gi["support"] - gj["support"])
    base = mid - (mid @ normal) * normal + sep * normal
    sep_line.set_data([base[0] - 5 * tangent[0], base[0] + 5 * tangent[0]], [base[1] - 5 * tangent[1], base[1] + 5 * tangent[1]])
    arrow = panel.add(FancyArrowPatch(base - 1.6 * normal, base + 0.4 * normal, arrowstyle="-|>,head_length=10,head_width=6",
                                      color=TEXT, lw=2, zorder=5))  # fmt: skip
    n_text = panel.ax.text(*(base - 0.9 * normal + 2.2 * tangent), "$\\boldsymbol{n}^{ij}_{k+t|k}$", fontsize=24,
                           color=TEXT, ha="center", va="center", zorder=12, path_effects=white_stroke())  # fmt: skip
    n_text.set_transform(panel.transform + panel.ax.transData)
    # The shares beside the lower part of their strips (on the side of their vehicle), and the vehicles
    down = tangent if tangent[1] < 0 else -tangent
    foot = mid - (mid @ normal) * normal + 3.7 * down  # Projection zero along the normal

    def share_note(text, direction, support, color):
        return panel.note(text, foot + (support + 0.25) * direction,
                          color, ha="right" if direction[0] < 0 else "left", size=22)

    zoom_notes = [
        share_note("$\\rho^{ij}\\eta^{ij}_{k+t|k}$", normal, gi["support"], BLUE),
        share_note("$\\rho^{ji}\\eta^{ij}_{k+t|k}$", -normal, gj["support"], J_COLOR),
        panel.note("$i$", ref_i[ts, :2] + np.array([0.0, 2.3]), BLUE, size=26),
        panel.note("$j$", ref_j[ts, :2] + np.array([0.0, 2.3]), J_COLOR, size=26),
    ]
    caption.set(f"Zoom into prediction step $t = {ts + 1}$: a line (dashed) with normal $\\boldsymbol{{n}}^{{ij}}_{{k+t|k}}$ separates the backup footprints of $i$ and $j$.",
                "Each corner of $i$ may move toward $j$ beyond its backup footprint by at most its share $\\rho^{ij}\\eta^{ij}_{k+t|k}$, and likewise $j$.")  # fmt: skip
    hold(clip, fig, 8.0)
    # The policy's controls
    cand = [pi["prediction_nom"][ts], pj["prediction_nom"][ts]]
    rects.set_verts([rectangle(p, HALF) for p in cand])
    rects.set_facecolors([to_rgba(NOMINAL, 0.22)] * 2)
    rects.set_edgecolors([NOMINAL, NOMINAL])
    corners_i = rectangle_corners(cand[0], HALF)
    crossing = corners_i[corners_i @ normal < gi["support"] - gi["share_budget"] - meta["tol"]]
    dots.set_data(crossing[:, 0], crossing[:, 1])
    nom_line.set_data(t[: len(h_nom)], h_nom)
    nom_text.set_position((len(h_nom) + 0.3, h_nom[-1]))
    nom_text.set_text("policies")
    count = {1: "one corner", 2: "two corners", 3: "three corners", 4: "all corners"}.get(len(crossing), "no corner")
    caption.set(f"Purple: the footprints at this step under the policies' controls. Vehicle $i$ accelerates, and {count} of $i$ leave its strip (red),",
                "so $i$ violates its corner inequalities. Here, the pair margin of the two candidate predictions also falls below the red line.")  # fmt: skip
    hold(clip, fig, 7.0)
    cand = [pi["prediction_applied"][ts], pj["prediction_applied"][ts]]
    rects.set_verts([rectangle(p, HALF) for p in cand])
    rects.set_facecolors(["none", "none"])
    rects.set_edgecolors([BLUE, J_COLOR])
    dots.set_data([], [])
    app_line.set_data(t[: len(h_app)], h_app)
    app_text.set_position((len(h_app) + 0.3, h_app[-1] - 0.12))
    app_text.set_text("filtered")
    a = pi["u_applied"][0]
    caption.set(f"Filtered (outlines): the filter of $i$ brakes ($a = {a:.1f}$ m/s²) to stay in its strip. The filter of $j$ keeps its policy's control,",
                "which already stays in its own strip. Both shares together keep the pair margin above the red line.")  # fmt: skip
    hold(clip, fig, 8.0)

    # The two filter problems
    for frame in range(int(1.2 * FPS) + 1):
        f = 0.5 + 0.5 * np.cos(np.pi * frame / (1.2 * FPS))
        panel.set_window(PAIR_VIEW + f * (zoom - PAIR_VIEW), (1 - f) * center + f * mid)
        clip.add(fig)
    for artist in (shares_fill,):
        artist.set_verts([])
    rects.set_verts([])
    sep_line.set_data([], [])
    arrow.remove()
    n_text.remove()
    for note in zoom_notes:
        note.remove()
    show_map(pose[PAIR_LEAD], f"time {PAIR_STEP * DT:4.1f} s   (paused at step $k$)")
    backup_i.update(pi["prediction_applied"])
    backup_j.update(pj["prediction_applied"])
    ax.remove()
    planes = []
    gap = 0.06
    width = (RIGHT[1] - RIGHT[0] - 0.07 - gap) / 2
    for k, (p, title, color) in enumerate(((pi, "Filter of vehicle $i$", BLUE), (pj, "Filter of vehicle $j$", J_COLOR))):
        pax = fig.add_axes([RIGHT[0] + 0.06 + k * (width + gap), PANEL_BOTTOM + 0.1, width, PANEL_TOP - PANEL_BOTTOM - 0.14])
        plane = ControlPlane(pax, meta, title, color)
        plane.set_problem(p)
        plane.mark("nominal", p["u_nom"])
        plane.mark("backup", p["backup_control"])
        plane.mark("applied", p["u_applied"])
        plane.show_projection(p["u_nom"], p["u_applied"])
        planes.append(plane)
    fig.legend([Rectangle((0, 0), 1, 1, fc=tint(BLUE, 0.3), ec=BLUE, lw=2),
                Rectangle((0, 0), 1, 1, fc=tint(J_COLOR, 0.3), ec=J_COLOR, lw=2),
                Rectangle((0, 0), 1, 1, fc=INFEASIBLE, ec="none")],
               ["All inequalities of $i$ hold", "All inequalities of $j$ hold", "At least one violated"],
               fontsize=17, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(RIGHT[0] - 0.005, 0.845),
               columnspacing=1.2, handlelength=1.6)  # fmt: skip
    caption.set("Two filter problems, one per vehicle, each over its own control; they share only the states and routes broadcast once per step.",
                "As the shares add up to one, both solutions together satisfy the pair's BH-CBF condition. Map: filtered candidate predictions.")  # fmt: skip
    hold(clip, fig, 9.0)

    # Continuation at half speed with the predictions of both vehicles
    caption.set("At every step, each vehicle solves its own filter problem again from the newly broadcast states.",
                "Map: each vehicle's filtered candidate prediction until standstill (future poses, not current ones).")  # fmt: skip
    drawn = {}
    for frame in range(int(PAIR_AFTER * DT * FPS / speed) + 1):
        tau = PAIR_LEAD + frame * speed / (FPS * DT)
        poses = interpolate(pose, tau)
        step = first + int(np.floor(tau + 1e-9))
        for vehicle, footprints in ((PAIR_I, backup_i), (PAIR_J, backup_j)):
            p = problems.get((step, vehicle))
            footprints.update(None if p is None else p["prediction_applied"])
        for vehicle, plane in ((PAIR_I, planes[0]), (PAIR_J, planes[1])):
            p = problems.get((step, vehicle))
            if p is not None and drawn.get(vehicle) != step:
                drawn[vehicle] = step
                plane.set_problem(p)
                plane.mark("nominal", p["u_nom"])
                plane.mark("backup", p["backup_control"])
                plane.mark("applied", p["u_applied"])
                plane.show_projection(p["u_nom"], p["u_applied"])
        visible = ~np.isnan(poses[[PAIR_I, PAIR_J], 0])
        if visible.all():  # Follow the pair
            panel.set_window(PAIR_VIEW, 0.5 * (poses[PAIR_I, :2] + poses[PAIR_J, :2]))
        show_map(poses, f"time {(first + tau) * DT:4.1f} s   (0.5× speed)")
        clip.add(fig)
    hold(clip, fig, 1.0)
    plt.close(fig)


def scene_summary(clip):
    fig = new_figure("BH-CBF in short")
    items = [
        "At every step, predicts each vehicle braking to a standstill along its route",
        "Keeps the smallest margin along this prediction from shrinking too fast:\nrecursively feasible under control bounds",
        "One small QP per vehicle and step, checked exactly;\nthe backup control as the last resort",
        "Fully decentralized: each vehicle stays within its share of the budget;\none broadcast of states and routes per step",
    ]  # fmt: skip
    renderer = fig.canvas.get_renderer()
    gap = 0.045  # Between items (figure fraction)
    texts = [(fig.text(0.07, 0, f"{k + 1}.", fontsize=30, color=BLUE, weight="bold", va="top"),
              fig.text(0.1, 0, text, fontsize=30, color=TEXT, va="top", linespacing=1.3))
             for k, text in enumerate(items)]  # fmt: skip
    heights = [body.get_window_extent(renderer).height / fig.bbox.height for _, body in texts]
    # The block centered between the subtitle and the closing line
    y = 0.47 + 0.5 * (sum(heights) + gap * (len(items) - 1))
    for (number, body), height in zip(texts, heights):
        number.set_y(y)
        body.set_y(y)
        y -= height + gap
    fig.text(0.07, 0.08, "Details and proofs: see the paper.   Project page: bassamlab.github.io/BH-CBF",
             fontsize=24, color=MUTED, va="center")  # fmt: skip
    hold(clip, fig, 10.0)
    plt.close(fig)


CLIPS = [
    ("intro", lambda c: card(c, "Inside the BH-CBF filter",
                             "How one filter problem is set up and solved: one vehicle, then two vehicles", 3.5)),
    ("one_step", scene_one_step),
    ("backup", scene_backup),
    ("control_plane", scene_control_plane),
    ("recursive", scene_recursive),
    ("pair", scene_pair),
    ("summary", scene_summary),
]  # fmt: skip


def render(args):
    setup_fonts()
    folder = os.path.join(args.out, "previews" if args.preview else "clips")
    os.makedirs(folder, exist_ok=True)
    selected = [(os.path.join(folder, f"{i:02d}_{name}.mp4"), name, fn)
                for i, (name, fn) in enumerate(CLIPS) if not args.only or name in args.only]  # fmt: skip
    for count, (path, name, fn) in enumerate(selected):
        print(f"[{count + 1}/{len(selected)}] {name}")
        clip = Clip(path, preview=args.preview)
        fn(clip)
        clip.close()


def assemble(args):
    clips = sorted(glob.glob(os.path.join(args.out, "clips", "*.mp4")))
    listing = os.path.join(args.out, "clips", "list.txt")
    with open(listing, "w") as file:
        file.writelines(f"file '{os.path.abspath(c)}'\n" for c in clips)
    target = os.path.join(OUT, "bh_cbf_explainer.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
                    "-c", "copy", "-movflags", "+faststart", target], check=True)  # fmt: skip
    seconds = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", target],
        capture_output=True, text=True, check=True).stdout)  # fmt: skip
    print(f"[INFO] Saved {target} ({seconds:.1f} s, {len(clips)} clips)")


def main():
    args = parse_args()
    if args.command == "render":
        render(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
