# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Video demo of the T-ITS paper (supplementary material, about 3 minutes).

The video follows the paper's four evaluation stages: pre-training with
safety rewards (1), zero-shot deployment with run-time filters (2),
filter-in-the-loop fine-tuning (3), and the comparison with human drivers (4).
It replays rollouts of the evaluation; it simulates nothing itself.
Wherever the BH-CBF filter acts (except in Stage 4, where they would hide
the comparison), it draws the braking tails that the filter predicts at each step (light trails until standstill; tails.py
captures them by rerunning the deployment). Its scenes and their sources:

- backup horizon: two Stage 2 deployments with BH-CBF, one per map
  (outputs/tits26/stage1_filters_ttc_iter100, out_td files; the output
  folders keep the names of the former three-stage numbering);
- Stage 1 rewards: policies pre-trained with the TTC, the one-step CBF, and
  the BH-CBF-informed reward, deployed without a filter from the same start
  (outputs/tits26/stage1_zero_shot, unfiltered runs);
- Stage 2 counterfactuals: the scenes (a), (b), (e), (f) of the paper's
  counterfactual figure (stage1_counterfactual_rollouts.py,
  outputs/tits26/stage1_counterfactual);
- Stage 2 filter comparison: one policy and seed deployed without a filter,
  with the one-step CBF filter, and with BH-CBF, from the same start;
- Stage 3: one source policy before and after fine-tuning with BH-CBF in the
  loop, both deployed without a filter (rollouts.py, OUT/stage2);
- Stage 4: the four scenes of the paper's human-comparison figure, recorded humans
  (outlined) and the fine-tuned, filtered policy (filled) in one panel, a thin
  line joining the two versions of each vehicle, red where the human is ahead
  (outputs/tits26/stage3_human_ft_recorded_maxlen6.5); without braking tails.

The time windows avoid deadlocks (vehicles standing still for seconds in the
middle of the map). Each scene is rendered with matplotlib and piped to
ffmpeg as one clip (OUT/clips/<nn>_<name>.mp4); ``assemble`` concatenates the
clips to OUT/bh_cbf_video_demo.mp4.
The map panels of every scene are also saved without titles and fades in
OUT/raw/<scene>.mp4, for reuse in slides. The cover and the closing card take the logos and
photos in OUT/assets (logo_i11_rwth.png, logo_unibw.png, photo_<name>.jpg);
the text is set in Carlito, metrically compatible with Calibri (bundled with
LibreOffice, CARLITO).

Usage (from the repository root, with the shell function
R() { SIGMARL_AGENT_PROFILE=full_scale python -m "sigmarl.evaluation_tits26.video_demo.$1" "${@:2}"; }):
    # Inputs that the evaluation outputs do not hold (once)
    R rollouts --run outputs/tits26/stage2_finetune/round_location0__stage1_ttc_seed_1__iter100/ind_location4/ours/seed1 --iter 0 100 --seed 205
    R tails deployment --run <BACKUP_ROUND run> --steps 250 350 --name backup_round
    R tails deployment --run <BACKUP_IND run> --steps 40 140 --name backup_ind
    R tails deployment --run <COMPARISON BH-CBF run> --steps 0 250 --name cmp_round
    R tails counterfactual --npz <scene npz> --steps <branch> <collision + 30> --name cf_<a..d>
    R tails humans --episode <episode> --seed 1 --name s3_<a..d>
    R repair capture --run <REPAIR_RUN> --steps 0 300 --name repair_round
    # Clips and video
    R video render [--only <clip> ...] [--preview]
    R video assemble
    # Extra clips, not part of the video (OUT/extra/<name>.mp4)
    R video render --only repair stage4_side_by_side
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
import torch
from matplotlib import font_manager, patheffects
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from sigmarl.evaluation_tits26.paper_fig_headline import heading_chevron, rectangle
from sigmarl.evaluation_tits26.paper_fig_stage1_footprints import HALF, draw_road

OUT = "outputs/tits26/video_demo"
TAILS = f"{OUT}/tails"
ASSETS = f"{OUT}/assets"  # Logos and photos of the cover
CARLITO = "/Applications/LibreOffice.app/Contents/Resources/fonts/truetype"
# Output folders of the evaluation, named after the former three-stage numbering
STAGE1 = "outputs/tits26/stage1_filters_ttc_iter100"
REWARDS_ROOT = "outputs/tits26/stage1_zero_shot"
COUNTERFACTUAL = "outputs/tits26/stage1_counterfactual"
HUMAN = "outputs/tits26/stage3_human_ft_recorded_maxlen6.5/natural"
HUMAN_CONTROLLER = (
    "round_location0__stage1_ttc_seed_{}__iter100__ft-round_location0-ours-seed1"
    "__iter100__ours_dec"
)

WIDTH, HEIGHT, FPS = 1920, 1080, 30
DT = 0.1  # Sampling period of the rollouts [s]
FADE = 8  # Frames of the fade from and to white at the ends of a clip
JUMP = 3.0  # Displacement per step that marks a respawn [m]

# RWTH colors (2026_T-ITS_CBF-Safeguard-MARL/00_ps.sty)
BLUE, BLUE50 = "#00549F", "#8EBAE5"
GREEN = "#56AB27"
PURPLE = "#7B6FAC"
MAGENTA = "#E30066"
RED = "#CC061F"
ORANGE = "#F6A800"
CONTEXT = "#9C9EA0"  # Other vehicles
TEXT = "#1A1A1A"
MUTED = "#606062"
KEY_COLORS = [BLUE, GREEN, PURPLE, MAGENTA]
# Links of the overlaid Stage 4 scenes: neutral, or red where the human is ahead
# of its policy vehicle by more than LEAD along the recorded path [m]
LINK, HUMAN_AHEAD, LEAD = TEXT, RED, 0.5

# Backup-horizon scenes: Stage 2 deployments with BH-CBF (TTC policy of iteration 100)
BACKUP_ROUND = f"{STAGE1}/round_location0__stage1_ttc_seed_1__iter100/round_location0/ours/dec/seed201"
BACKUP_IND = (
    f"{STAGE1}/ind_location4__stage1_ttc_seed_4__iter100/ind_location4/ours/dec/seed202"
)
# Stage 2 counterfactual scenes of the paper (fig. footprints (a), (b), (e), (f)):
# npz (relative to COUNTERFACTUAL), branch env, branch step, collision step, key vehicles, window, tails
COUNTERFACTUAL_SCENES = [
    ("round_location0__stage1_ttc_seed_2__iter100/round_location0/seed301.npz", 5, 680, 714, [4, 15], (61, 101, 14, 54), "Roundabout", "cf_a"),
    ("round_location0__stage1_ttc_seed_1__iter100/round_location0/seed301.npz", 6, 770, 805, [14, 0], (60, 100, 15, 55), "Roundabout", "cf_b"),
    ("round_location0__stage1_ttc_seed_1__iter100/ind_location4/seed301.npz", 6, 1090, 1120, [2, 7], (82, 122, 42, 82), "Intersection (unseen map)", "cf_c"),
    ("round_location0__stage1_ttc_seed_3__iter100/ind_location4/seed301.npz", 2, 1130, 1175, [3, 1], (81, 121, 46, 86), "Intersection (unseen map)", "cf_d"),
]  # fmt: skip
# Steps that the filtered panel continues after the nominal collision (per scene, default 30)
CONTINUE = {"cf_a": 50, "cf_b": 50}
# Stage 1 rewards: the policies of training seed 1 on the roundabout (iteration 500), deployed
# without a filter from the start (TTC: 2 collisions, one-step CBF: 3, BH-CBF: 0)
REWARDS_RUN = f"{REWARDS_ROOT}/round_location0__stage1_{{}}_seed_1/round_location0/unfiltered/cen/seed203"
REWARDS = [
    ("TTC reward", REWARDS_RUN.format("ttc"), None, False),
    ("One-step CBF reward", REWARDS_RUN.format("onestep"), None, False),
    ("BH-CBF-informed reward (ours)", REWARDS_RUN.format("bh"), None, True),
]
REWARDS_STEPS = (0, 250)
# Stage 2 filter comparison: one TTC-reward policy (iteration 100) and seed, three filters,
# from the start of the deployment (no filter: 3 collisions, one-step: 2, BH-CBF: 0)
COMPARISON_RUN = f"{STAGE1}/round_location0__stage1_ttc_seed_4__iter100/round_location0"
COMPARISON = [
    ("No filter", f"{COMPARISON_RUN}/unfiltered/cen/seed202", None, False),
    ("One-step CBF filter", f"{COMPARISON_RUN}/ours_onestep/dec/seed202", None, False),
    ("BH-CBF filter (ours)", f"{COMPARISON_RUN}/ours/dec/seed202", "cmp_round", True),
]
COMPARISON_STEPS = (0, 250)
COMPARISON_WINDOW = (36, 124, 25, 85)
# Stage 3: the roundabout-trained TTC policy (seed 1, iteration 100) fine-tuned on the intersection
FINETUNE_RUN = f"{OUT}/stage2/round_location0__stage1_ttc_seed_1__iter100__ind_location4__ours__seed1"
FINETUNE_SEED = 205
FINETUNE_STEPS = (0, 300)
FINETUNE_WINDOW = (70, 140, 36, 90)
# Stage 4 scenes of the paper (fig. human-footprints; (a), (b) cut from the paper): episode, start time [s],
# duration [s], key vehicles (track ids), lower left corner of the 64 m window, tails, caption
HUMAN_SCENES = [
    ("rec11_h0303.3", 311.2, 10.5, [166, 164], (33.0, 32.5), "s3_a",
     "The policy enters at once, ahead of a circulating vehicle that is still far away; the human waits 3.5 s."),
    ("rec19_h0037.9", 37.4, 10.5, [63, 66, 59], (40.0, 36.0), "s3_b",
     "Two queued vehicles enter as a platoon right behind a circulating vehicle; the humans wait about 6 s."),
    ("rec05_h0700.5", 713.5, 10.5, [462, 463], (68.0, 27.0), "s3_c",
     "The policy keeps its speed into the ring; the human brakes to a standstill and waits 3 s."),
    ("rec11_h0067.9", 67.5, 10.5, [43, 37], (50.0, 13.0), "s3_d",
     "The policy enters ahead of a close circulating vehicle, which brakes briefly; the human waits 5.4 s."),
]  # fmt: skip
HUMAN_SEED = 1  # Source policy of the fine-tuned policy
HUMAN_WINDOW = 64.0
# Side-by-side Stage 4 scenes: labels with arrows to the blue vehicle in both panels, at a fixed
# position [m]. Times [s from the scene start]: the human starts to slow down (left: "Slow down";
# right, the policy: "Decisively enter"), the human's stop (speed below 1 m/s; left: "Stop"), and
# the end of the policy's label, 1 s after it reaches the ring (within 20.5 m of its center)
HUMAN_CUES = {  # slow down, (stop from, stop to), policy label until, label position
    "s3_a": (1.6, (4.4, 7.9), 5.1, (43.0, 68.0)),
    "s3_b": (1.6, (3.3, 8.8), 4.7, (69.5, 88.0)),
    "s3_c": (0.5, (4.4, 7.5), 5.0, (116.0, 68.0)),
    "s3_d": (0.7, (3.7, 9.0), 5.1, (88.0, 20.0)),
}
# Repair rounds of the certified filter (repair.py): the Stage 2 comparison run with BH-CBF,
# one filter problem (step, vehicle i) with a repair round, steps shown before and after it
REPAIR = f"{OUT}/repair"
REPAIR_NAME = "repair_round"
REPAIR_RUN = f"{COMPARISON_RUN}/ours/dec/seed202"
REPAIR_EVENT = (147, 6)
REPAIR_LEAD = (30, 30)
REPAIR_VIEW = 36.0  # Side of the vehicle-centered view [m]
CELL_GRAY = "#E6E6E6"
NEIGHBOR_COLORS = [GREEN, MAGENTA, "#0098A1"]  # Purple marks the candidate's prediction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["render", "assemble"])
    parser.add_argument("--only", nargs="+", help="Render only these clips (names).")
    parser.add_argument(
        "--preview", action="store_true",
        help="Write one PNG per clip (its middle frame) instead of the clips.",
    )  # fmt: skip
    parser.add_argument("--out", default=OUT)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Frames and clips


class Clip:
    """Frames of one scene, piped to ffmpeg as an H.264 clip; fades from and to
    white over FADE frames at its ends. With preview=True, it saves only the
    middle frame as PNG."""

    def __init__(self, path: str, preview: bool = False):
        self.path, self.preview = path, preview
        self.frames = []  # Last FADE frames (fade-out) or all frames (preview)
        self.count = 0
        if not preview:
            self.process = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                 "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", path],
                stdin=subprocess.PIPE,
            )  # fmt: skip

    raw = None  # ffmpeg process of the raw video being written alongside, if any

    def start_raw(self, name: str, fig, axes: list):
        """Also write the map panels of the following frames (cropped to the
        panels and their titles, without fades) to RAW/<name>.mp4, for reuse
        in slides."""
        if self.preview:
            return
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_window_extent(renderer) for ax in axes]
        boxes += [ax.title.get_window_extent(renderer) for ax in axes]
        pad = 12
        x0 = max(0, int(min(b.x0 for b in boxes)) - pad)
        x1 = min(WIDTH, int(max(b.x1 for b in boxes)) + pad)
        top = max(0, HEIGHT - int(max(b.y1 for b in boxes)) - pad)
        bottom = min(HEIGHT, HEIGHT - int(min(b.y0 for b in boxes)) + pad)
        width, height = (x1 - x0) // 2 * 2, (bottom - top) // 2 * 2  # Even for yuv420p
        self.crop = (slice(top, top + height), slice(x0, x0 + width))
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(self.path)), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        self.raw_path = os.path.join(raw_dir, f"{name}.mp4")
        self.raw = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", self.raw_path],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    def stop_raw(self):
        if self.raw is None:
            return
        self.raw.stdin.close()
        self.raw.wait()
        self.raw = None
        print(f"[INFO] Saved {self.raw_path}")

    def add(self, fig, repeat: int = 1):
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        if self.raw is not None:
            cropped = np.ascontiguousarray(frame[self.crop]).tobytes()
            for _ in range(repeat):
                self.raw.stdin.write(cropped)
        for _ in range(repeat):
            self.frames.append(frame)
            if self.preview:
                continue
            if len(self.frames) > FADE:
                self._write(self.frames.pop(0))

    def _write(self, frame):
        if self.count < FADE:
            frame = fade(frame, (self.count + 1) / (FADE + 1))
        self.process.stdin.write(frame.tobytes())
        self.count += 1

    def close(self):
        if self.preview:
            plt.imsave(
                self.path.replace(".mp4", ".png"), self.frames[len(self.frames) // 2]
            )
            print(f"[INFO] Saved {self.path.replace('.mp4', '.png')}")
            return
        n = len(self.frames)
        for i, frame in enumerate(self.frames):
            self._write(fade(frame, (n - i) / (n + 1)))
        self.process.stdin.close()
        self.process.wait()
        print(f"[INFO] Saved {self.path} ({self.count / FPS:.1f} s)")


def fade(frame: np.ndarray, weight: float) -> np.ndarray:
    """The frame blended with white (weight 1: the frame itself)."""
    return (255 - weight * (255 - frame.astype(np.float32))).astype(np.uint8)


def new_figure(title: str = None, subtitle: str = None):
    """A white 1920 x 1080 frame with a title and a subtitle."""
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor("white")
    if title:
        fig.text(
            0.035, 0.935, title, fontsize=38, weight="bold", color=BLUE, va="center"
        )
    if subtitle:
        fig.text(0.035, 0.875, subtitle, fontsize=22, color=MUTED, va="center")
    return fig


def panel_axes(
    fig,
    n: int,
    window: tuple,
    top: float = 0.79,
    bottom: float = 0.1,
    gap: float = 0.025,
    margin: float = 0.035,
) -> list:
    """n equal-aspect map panels side by side, as large as fits."""
    x0, x1, y0, y1 = window
    aspect = (y1 - y0) / (x1 - x0)
    width = (1 - 2 * margin - (n - 1) * gap) / n
    height = top - bottom
    # Fit the window's aspect ratio in the cell (figure coordinates are not square)
    if width * WIDTH * aspect > height * HEIGHT:
        width = height * HEIGHT / aspect / WIDTH
    else:
        height = width * WIDTH * aspect / HEIGHT
    total = n * width + (n - 1) * gap
    left = (1 - total) / 2
    axes = []
    for k in range(n):
        ax = fig.add_axes(
            [
                left + k * (width + gap),
                bottom + (top - bottom - height) / 2,
                width,
                height,
            ]
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.6")
            spine.set_linewidth(1.0)
        axes.append(ax)
    return axes


def panel_title(ax, text: str, color: str = TEXT):
    ax.set_title(text, fontsize=26, color=color, weight="bold", pad=10)


def footer(fig, text: str = "", right: str = None):
    """The clock (left, returned for updates) and an optional note (right)."""
    if right:
        fig.text(0.965, 0.04, right, fontsize=20, color=MUTED, ha="right", va="center")
    return fig.text(0.035, 0.04, text, fontsize=20, color=MUTED, va="center")


def time_text(seconds: float, speed: float) -> str:
    rate = "real time" if speed == 1 else f"{speed:g}× speed"
    return f"t = {seconds:4.1f} s   ({rate})"


def card(clip: Clip, title: str, subtitle: str, seconds: float = 3.0):
    """A section card: a title and a subtitle, centered."""
    fig = new_figure()
    fig.text(
        0.5,
        0.53,
        title,
        fontsize=40,
        color=BLUE,
        weight="bold",
        ha="center",
        va="center",
    )
    fig.text(0.5, 0.44, subtitle, fontsize=26, color=MUTED, ha="center", va="center")
    clip.add(fig, int(seconds * FPS))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Vehicles, tails, collisions


def interpolate(pose: np.ndarray, tau: float) -> np.ndarray:
    """Poses [n, 3] at fractional step tau of poses [T, n, 3] (NaN: absent);
    no interpolation across a respawn."""
    i = int(np.clip(np.floor(tau), 0, len(pose) - 1))
    j = min(i + 1, len(pose) - 1)
    f = tau - i
    a, b = pose[i], pose[j]
    out = a.copy()
    dh = (b[:, 2] - a[:, 2] + np.pi) % (2 * np.pi) - np.pi
    out[:, :2] = a[:, :2] + f * (b[:, :2] - a[:, :2])
    out[:, 2] = a[:, 2] + f * dh
    jump = np.linalg.norm(b[:, :2] - a[:, :2], axis=-1) > JUMP
    out[jump] = a[jump]
    return out


def hide_after_respawn(pose: np.ndarray) -> np.ndarray:
    """A copy of poses [T, n, 3] that is NaN from each vehicle's first respawn on
    (a vehicle that reaches its exit respawns at a random entry, which differs
    between rollouts)."""
    pose = pose.copy()
    jump = np.linalg.norm(pose[1:, :, :2] - pose[:-1, :, :2], axis=-1) > JUMP
    for k in range(pose.shape[1]):
        hits = np.nonzero(jump[:, k])[0]
        if len(hits):
            pose[hits[0] + 1 :, k] = np.nan
    return pose


def progress(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Arc length [m] along a densely sampled path [M, 2] of its vertex nearest
    to each of points [T, 2] (NaN stays NaN)."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=-1))])
    nearest = np.linalg.norm(points[:, None] - path[None], axis=-1)
    out = np.full(len(points), np.nan)
    valid = ~np.isnan(points).any(-1)
    out[valid] = arc[np.argmin(nearest[valid], axis=1)]
    return out


class Fleet:
    """Vehicle rectangles with heading chevrons on one axes: filled, or
    outlined (recorded human drivers)."""

    def __init__(
        self,
        ax,
        half: np.ndarray,
        colors: list,
        filled: bool = True,
        zorder: float = 3,
        face: str = "white",
    ):
        self.half = np.broadcast_to(half, (len(colors), 2))
        self.patches, self.chevrons = [], []
        for color in colors:
            if filled:
                patch = Polygon(
                    np.zeros((4, 2)), fc=color, ec="white", lw=0.8, zorder=zorder
                )
                mark = "white"
            else:
                # Twice the width, clipped to the footprint: the outline lies inside
                # it, so an outlined vehicle looks as large as a filled one
                patch = Polygon(
                    np.zeros((4, 2)), fc=face, ec=color, lw=4.4, zorder=zorder
                )
                mark = color
            ax.add_patch(patch)
            if not filled:
                patch.set_clip_path(patch)
            chevron = Line2D(
                [], [], color=mark, lw=1.4, zorder=zorder + 0.1, solid_joinstyle="miter"
            )
            ax.add_line(chevron)
            self.patches.append(patch)
            self.chevrons.append(chevron)

    def update(self, poses: np.ndarray):
        for k, (patch, chevron) in enumerate(zip(self.patches, self.chevrons)):
            visible = not np.isnan(poses[k]).any()
            patch.set_visible(visible)
            chevron.set_visible(visible)
            if visible:
                patch.set_xy(rectangle(poses[k], self.half[k]))
                points = heading_chevron(poses[k], self.half[k])
                chevron.set_data(points[:, 0], points[:, 1])


class Tails:
    """BH-CBF braking tails: the predicted footprints until standstill (light,
    in the vehicle's color) and the standstill footprint (outlined)."""

    def __init__(self, ax, half: np.ndarray, colors: list):
        self.half = np.broadcast_to(half, (len(colors), 2))
        self.colors = colors
        self.fill = PolyCollection([], edgecolors="none", zorder=2)
        self.stop = PolyCollection([], facecolors="none", linewidths=1.2, zorder=2.5)
        ax.add_collection(self.fill)
        ax.add_collection(self.stop)

    def update(self, tails: dict, visible: np.ndarray):
        """tails: vehicle index -> poses [H, 3] (from the current pose on)."""
        verts, faces, stops, edges = [], [], [], []
        for k, tail in tails.items():
            if not visible[k]:
                continue
            for pose in tail[1:]:
                verts.append(rectangle(pose, self.half[k]))
                faces.append(tint(self.colors[k], 0.25))
            stops.append(rectangle(tail[-1], self.half[k]))
            edges.append(tint(self.colors[k], 0.55))
        self.fill.set_verts(verts)
        self.fill.set_facecolors(faces)
        self.stop.set_verts(stops)
        self.stop.set_edgecolors(edges)


def tint(color: str, strength: float) -> tuple:
    """An opaque tint of a color (strength 1: the color, 0: white); opaque, so
    that overlapping footprints of a tail keep the same shade."""
    rgb = np.array(to_rgba(color)[:3])
    return (*(1 - strength * (1 - rgb)), 1.0)


def load_tails(name: str) -> tuple:
    """(steps [S], tails [S, H, n, 3]) captured by tails.py."""
    data = np.load(os.path.join(TAILS, f"{name}.npz"))
    return data["steps"], data["tails"]


def tails_at(steps: np.ndarray, tails: np.ndarray, step: float) -> dict:
    """Tails of the latest captured step at or before `step` (a few steps of a
    rerun have none: the previous one is held for 0.1 s)."""
    index = max(0, int(np.searchsorted(steps, np.floor(step + 1e-9), side="right")) - 1)
    return {k: tails[index][:, k] for k in range(tails.shape[2])}


class Collisions:
    """Collision markers: at the onset step of a collision, the vehicle's
    rectangle outlined in red with a burst, fading out over `hold` steps; and a
    counter of collisions (a collision of two vehicles counts once)."""

    def __init__(
        self, ax, events: list, half: np.ndarray, hold: int = 25, counter_xy=None
    ):
        self.events, self.hold, self.half = events, hold, half  # (step, pose, kind)
        self.patches, self.marks = [], []
        for _ in events:
            patch = Polygon(
                np.zeros((4, 2)), fc="none", ec=RED, lw=3.5, alpha=0.0, zorder=6
            )
            ax.add_patch(patch)
            (mark,) = ax.plot([], [], marker="*", ms=26, color=RED, mec="white", mew=1.2,
                              zorder=7, alpha=0.0)  # fmt: skip
            self.patches.append(patch)
            self.marks.append(mark)
        self.counter = None
        if counter_xy is not None:
            self.counter = ax.text(*counter_xy, "", transform=ax.transAxes, fontsize=24,
                                   weight="bold", ha="left", va="top", color=TEXT,
                                   bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                             ec="0.7", alpha=0.9), zorder=9)  # fmt: skip

    def update(self, tau: float):
        vehicle, lane = 0, 0
        for (step, pose, kind), patch, mark in zip(
            self.events, self.patches, self.marks
        ):
            if tau >= step:
                vehicle += kind == "vehicle"
                lane += kind == "lane"
            age = tau - step
            alpha = 0.0 if age < 0 or age > self.hold else 1.0 - 0.7 * age / self.hold
            patch.set_alpha(alpha)
            patch.set_xy(rectangle(pose, self.half))
            mark.set_alpha(alpha)
            mark.set_data([pose[0]], [pose[1]])
        if self.counter is not None:
            count = int(np.ceil(vehicle / 2)) + lane
            self.counter.set_text(f"Collisions: {count}")
            self.counter.set_color(RED if count else GREEN)


def collision_events(pose, veh_col, lane_col, start: int, end: int) -> list:
    """(step, pose, kind) of every collision onset in [start, end] (relative
    steps from start)."""
    events = []
    veh_col, lane_col = veh_col.astype(bool), lane_col.astype(bool)
    for t in range(start, end + 1):
        for flags, kind in ((veh_col, "vehicle"), (lane_col, "lane")):
            prev = flags[t - 1] if t > 0 else np.zeros_like(flags[t])
            for k in np.nonzero(flags[t] & ~prev)[0]:
                events.append((t - start, pose[t, k].copy(), kind))
    return events


def load_td(run: str) -> dict:
    """Poses [T, n, 3] and collision flags [T, n] after each step of a deployment
    (`run`: its out_td file, or the run folder that holds one)."""
    (path,) = (
        [run] if run.endswith(".td") else glob.glob(os.path.join(run, "out_td_*.td"))
    )
    td = torch.load(path, weights_only=False)
    info = td["next", "agents", "info"]
    return dict(
        pose=np.concatenate([info["pos"][0].numpy(), info["rot"][0].numpy()], axis=-1),
        veh_col=(info["is_collision_with_agents"][0, ..., 0] > 0).numpy(),
        lane_col=(info["is_collision_with_lanelets"][0, ..., 0] > 0).numpy(),
    )


def road(ax, scenario: str):
    draw_road(ax, scenario)
    for patch in ax.patches:
        patch.set_linewidth(1.0)


# ---------------------------------------------------------------------------
# Scenes


def scene_title(clip):
    """The cover: title, authors with their institutions' logos and photos."""
    fig = new_figure()
    for y, line in ((0.86, "Safeguarding Multi-Agent Reinforcement Learning for Connected and"),
                    (0.795, "Automated Vehicles with Recursively Feasible Control Barrier Functions")):  # fmt: skip
        fig.text(
            0.5,
            y,
            line,
            fontsize=42,
            color=BLUE,
            weight="bold",
            ha="center",
            va="center",
        )

    def image(name: str, center: tuple, width: float):
        """An image of ASSETS, `width` in figure widths, centered at `center`."""
        picture = plt.imread(os.path.join(ASSETS, name))
        height = width * WIDTH / HEIGHT * picture.shape[0] / picture.shape[1]
        ax = fig.add_axes(
            [center[0] - width / 2, center[1] - height / 2, width, height]
        )
        ax.imshow(picture)
        ax.axis("off")

    groups = [
        (0.265, "Jianye Xu", "logo_i11_rwth.png", 0.33, [("photo_xu.jpg", 0.265)],
         ["Department of Computer Science", "RWTH Aachen University", "Germany"]),
        (0.735, "Yousef Sayed, Bassam Alrifaee", "logo_unibw.png", 0.32,
         [("photo_sayed.jpg", 0.669), ("photo_alrifaee.jpg", 0.801)],
         ["Department of Aerospace Engineering", "University of the Bundeswehr Munich", "Germany"]),
    ]  # fmt: skip
    for x, names, logo, logo_width, photos, affiliation in groups:
        fig.text(x, 0.65, names, fontsize=38, color=BLUE, ha="center", va="center")
        image(logo, (x, 0.557), logo_width)
        for photo, photo_x in photos:
            image(photo, (photo_x, 0.397), 0.118)
        for k, line in enumerate(affiliation):
            fig.text(
                x,
                0.253 - 0.039 * k,
                line,
                fontsize=26,
                color=TEXT,
                ha="center",
                va="center",
            )
    clip.add(fig, 5 * FPS)
    plt.close(fig)


def scene_contributions(clip):
    """Our contributions: a lead line and numbered points."""
    fig = new_figure("Contributions")
    fig.text(0.05, 0.8, "We propose the Backup-Horizon Control Barrier Function (BH-CBF) with the following key properties:",
             fontsize=30, color=TEXT, va="center")  # fmt: skip
    for i, text in enumerate([
        "Recursively feasible",
        "Considers exact vehicle and road geometry",
        "Fully decentralized and real-time: one broadcast of vehicle states and routes per step",
        "Safeguards Multi-Agent Reinforcement Learning (MARL) policies over their life cycle:\n"
        "pre-training → run-time filter → safe fine-tuning",
        "14% shorter median travel time than recorded human drivers",
    ]):  # fmt: skip
        y = 0.68 - 0.1 * i - (0.025 if i == 4 else 0)
        fig.text(0.07, y, f"{i + 1}.", fontsize=30, color=ORANGE, weight="bold", va="top")
        fig.text(0.1, y, text, fontsize=30, color=TEXT, va="top", linespacing=1.3)
    clip.add(fig, 10 * FPS)
    plt.close(fig)


def scene_backup(
    clip,
    run: str,
    tails_name: str,
    scenario: str,
    steps: tuple,
    window: tuple,
    label: str,
    speed: float = 1.0,
):
    """BH-CBF braking tails at every step of a Stage 2 deployment."""
    start, end = steps
    pose = load_td(run)["pose"][start : end + 1]
    tail_steps, tails = load_tails(tails_name)
    n = pose.shape[1]
    fig = new_figure(
        "Key idea: the backup horizon",
        "At every step, BH-CBF predicts route-following emergency braking of every vehicle until standstill (light trails).",
    )
    (ax,) = panel_axes(fig, 1, window, top=0.8, bottom=0.12)
    road(ax, scenario)
    panel_title(ax, label)
    trails = Tails(ax, HALF, [BLUE] * n)
    fleet = Fleet(ax, HALF, [BLUE] * n)
    clock = footer(
        fig,
        right="The filter keeps the smallest margin along these trails from shrinking too fast.",
    )
    clip.start_raw(tails_name, fig, [ax])
    for frame in range(int((end - start) * DT / speed * FPS) + 1):
        tau = frame * speed / (FPS * DT)
        poses = interpolate(pose, tau)
        trails.update(tails_at(tail_steps, tails, start + tau), ~np.isnan(poses[:, 0]))
        fleet.update(poses)
        clock.set_text(time_text(tau * DT, speed))
        clip.add(fig)
    clip.stop_raw()
    plt.close(fig)


def scene_counterfactuals(clip):
    """The paper's counterfactual scenes: from the same joint state, the
    nominal policy collides (left, frozen at the collision) and BH-CBF avoids
    the collision (right, continues for CONTINUE steps)."""
    for index, (npz, env, start, hit, keys, window, label, name) in enumerate(
        COUNTERFACTUAL_SCENES
    ):
        data = np.load(os.path.join(COUNTERFACTUAL, npz))
        scenario = os.path.basename(os.path.dirname(npz))
        n = data["pose"].shape[2]
        colors = [CONTEXT] * n
        for rank, k in enumerate(keys):
            colors[k] = KEY_COLORS[rank]
        fig = new_figure(
            "Stage 2: zero-shot deployment of an unsafe policy",
            f"Scene {index + 1} of 4 – {label}. Both panels start from the same joint state; blue and green are the conflicting vehicles.",
        )
        axes = panel_axes(fig, 2, window)
        road(axes[0], scenario)
        road(axes[1], scenario)
        panel_title(axes[0], "Policy alone (no filter)", RED)
        panel_title(axes[1], "Policy + BH-CBF filter", BLUE)
        nominal = hide_after_respawn(data["pose"][start : hit + 1, env])
        filtered = hide_after_respawn(
            data["pose"][start : hit + CONTINUE.get(name, 30) + 1, 0]
        )
        events = collision_events(
            data["pose"][:, env],
            data["veh_col"][:, env],
            data["lane_col"][:, env],
            start,
            hit,
        )
        left = (
            Fleet(axes[0], HALF, colors),
            Collisions(axes[0], events, HALF, hold=10**6),
        )
        trails = Tails(axes[1], HALF, colors)
        right = Fleet(axes[1], HALF, colors)
        tail_steps, tails = load_tails(name)
        crash = axes[0].text(0.5, 0.06, "", transform=axes[0].transAxes, fontsize=22, weight="bold",
                             color=RED, ha="center", va="bottom", zorder=9,
                             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=RED))  # fmt: skip
        crash.set_visible(False)
        clock = footer(
            fig,
            right="Vehicles that reach their exit respawn at random entries; they are not drawn.",
        )
        lead, total = (
            hit - start,
            len(filtered) - 1,
        )  # The rollout may end before CONTINUE steps
        clip.start_raw(f"stage2_counterfactual_{index + 1}", fig, axes)
        timeline = (
            [0.0] * FPS
            + list(np.arange(0, total * DT * FPS + 1) / (DT * FPS))
            + [float(total)] * FPS
        )
        for tau in timeline:
            nominal_tau = min(tau, lead)
            left[0].update(interpolate(nominal, nominal_tau))
            left[1].update(nominal_tau)
            if tau >= lead:
                crash.set_text("Collision (paused)")
                crash.set_visible(True)
            poses = interpolate(filtered, tau)
            trails.update(
                tails_at(tail_steps, tails, start + tau), ~np.isnan(poses[:, 0])
            )
            right.update(poses)
            clock.set_text(time_text(tau * DT, 1))
            clip.add(fig)
        clip.stop_raw()
        plt.close(fig)


def scene_panels(
    clip, title: str, subtitle: str, runs: list, steps: tuple, raw: str, speed: float = 2.0
):
    """Roundabout deployments side by side from the same start, with
    collision counters; runs: (panel title, run, tails name or None, ours)."""
    start, end = steps
    fig = new_figure(title, subtitle)
    axes = panel_axes(fig, len(runs), COMPARISON_WINDOW)
    series = []
    for ax, (label, run, tails_name, ours) in zip(axes, runs):
        data = load_td(run)
        road(ax, "round_location0")
        panel_title(ax, label, BLUE if ours else TEXT)
        pose = data["pose"][start : end + 1]
        n = pose.shape[1]
        events = collision_events(
            data["pose"], data["veh_col"], data["lane_col"], start, end
        )
        trails = (
            (Tails(ax, HALF, [BLUE] * n), *load_tails(tails_name))
            if tails_name
            else None
        )
        series.append((pose, trails, Fleet(ax, HALF, [BLUE] * n),
                       Collisions(ax, events, HALF, counter_xy=(0.03, 0.97))))  # fmt: skip
    clock = footer(
        fig,
        right="Red: collision with a vehicle or the road boundary (the vehicle respawns).",
    )
    clip.start_raw(raw, fig, axes)
    for frame in range(int((end - start) * DT / speed * FPS) + 1):
        tau = frame * speed / (FPS * DT)
        for pose, trails, fleet, collisions in series:
            poses = interpolate(pose, tau)
            if trails is not None:
                trails[0].update(
                    tails_at(trails[1], trails[2], start + tau), ~np.isnan(poses[:, 0])
                )
            fleet.update(poses)
            collisions.update(tau)
        clock.set_text(time_text(tau * DT, speed))
        clip.add(fig)
    clip.add(fig, FPS)
    clip.stop_raw()
    plt.close(fig)


def scene_rewards(clip):
    """Policies pre-trained with each safety reward, deployed without a filter."""
    scene_panels(
        clip,
        "Stage 1: pre-training with safety rewards",
        "Policies pre-trained for 500 iterations with each safety reward, deployed without a filter from the same start.",
        REWARDS,
        REWARDS_STEPS,
        "stage1_rewards",
    )


def scene_comparison(clip):
    """One policy and seed without a filter, with the one-step CBF filter,
    and with BH-CBF, from the same start."""
    scene_panels(
        clip,
        "Stage 2: one-step CBF filter vs. BH-CBF",
        "Same policy, same start. The one-step filter uses our exact margins but looks only one step ahead: its Quadratic Program (QP) can become infeasible.",
        COMPARISON,
        COMPARISON_STEPS,
        "stage2_comparison",
    )


def scene_finetuning(clip, speed: float = 2.0):
    """A source policy before and after fine-tuning with BH-CBF in the loop,
    both deployed without a filter on the intersection (unseen in pre-training)."""
    start, end = FINETUNE_STEPS
    fig = new_figure(
        "Stage 3: fine-tuning with the BH-CBF filter in the loop",
        "Fine-tuning on the intersection is collision-free, and the policy itself becomes safe. Both panels: no filter at deployment.",
    )
    axes = panel_axes(fig, 2, FINETUNE_WINDOW)
    series = []
    for ax, (title, it) in zip(
        axes, (("Before fine-tuning", 0), ("After fine-tuning (100 iterations)", 100))
    ):
        data = load_td(
            os.path.join(FINETUNE_RUN, f"out_td_iter{it}_nofilter_seed{FINETUNE_SEED}.td")
        )
        road(ax, "ind_location4")
        panel_title(ax, title, BLUE if it else TEXT)
        pose = data["pose"][start : end + 1]
        events = collision_events(
            data["pose"], data["veh_col"], data["lane_col"], start, end
        )
        series.append((pose, Fleet(ax, HALF, [BLUE] * pose.shape[1]),
                       Collisions(ax, events, HALF, counter_xy=(0.02, 0.97))))  # fmt: skip
    clock = footer(fig, right="No filter acts here, so no braking trails are drawn.")
    clip.start_raw("stage3_finetuning", fig, axes)
    for frame in range(int((end - start) * DT / speed * FPS) + 1):
        tau = frame * speed / (FPS * DT)
        for pose, fleet, collisions in series:
            fleet.update(interpolate(pose, tau))
            collisions.update(tau)
        clock.set_text(time_text(tau * DT, speed))
        clip.add(fig)
    clip.add(fig, FPS)
    clip.stop_raw()
    plt.close(fig)


def scene_humans(clip, speed: float = 1.0, overlay: bool = True):
    """The paper's Stage 4 scenes: recorded humans (outlined) and the
    fine-tuned policy with the BH-CBF filter (filled), each vehicle starting
    at its recorded state. With overlay=True in one panel, a thin line joining
    the two versions of each vehicle (red where the human is ahead); otherwise
    side by side, with the policy's braking tails."""
    from sigmarl.evaluation_human_comparison.analyze import controlled_size

    with open(os.path.join(HUMAN, "matched_tracks.pkl"), "rb") as file:
        recordings = pickle.load(file)
    for index, (episode, t0, duration, keys, corner, name, caption) in enumerate(
        HUMAN_SCENES
    ):
        path = os.path.join(
            HUMAN,
            "homogeneous",
            HUMAN_CONTROLLER.format(HUMAN_SEED),
            f"{episode}.pkl",
        )
        with open(path, "rb") as file:
            result = pickle.load(file)
        tracks = {t.track_id: t for t in recordings[result["spec"].recording_id].tracks}
        vehicles = result["vehicles"]
        ids = [v["key"][1] for v in vehicles]
        colors = [KEY_COLORS[keys.index(i)] if i in keys else CONTEXT for i in ids]
        times = t0 + np.arange(0.0, duration + 1e-9, DT)

        def controlled(v):
            log = v["log"]
            t = np.asarray(log["t"])
            if len(t) == 0:
                return np.full((len(times), 3), np.nan)
            pose = np.stack([np.interp(times, t, log["x"]), np.interp(times, t, log["y"]),
                             np.interp(times, t, np.unwrap(log["yaw"]))], axis=-1)  # fmt: skip
            pose[(times < t[0] - 1e-6) | (times > t[-1] + 1e-6)] = np.nan
            return pose

        def human(track):
            pose = np.stack([np.interp(times, track.t, track.xy[:, 0]),
                             np.interp(times, track.t, track.xy[:, 1]),
                             np.interp(times, track.t, np.unwrap(track.heading))], axis=-1)  # fmt: skip
            pose[(times < track.t_start - 1e-6) | (times > track.t_end + 1e-6)] = np.nan
            return pose

        pose_c = np.stack([controlled(v) for v in vehicles], axis=1)
        pose_h = np.stack([human(tracks[i]) for i in ids], axis=1)
        # Progress of both versions along the recorded path [m]; positive gap: the policy is ahead
        gap = np.stack([progress(pose_c[:, k, :2], tracks[i].xy) - progress(pose_h[:, k, :2], tracks[i].xy)
                        for k, i in enumerate(ids)], axis=1)  # fmt: skip
        half_c = np.array([controlled_size(v) for v in vehicles]) / 2
        half_h = np.array([[tracks[i].length, tracks[i].width] for i in ids]) / 2
        with open(os.path.join(TAILS, f"{name}.pkl"), "rb") as file:
            captured = pickle.load(file)["captured"]
        tail_times = np.array([t for t, _ in captured])
        column = {i: k for k, i in enumerate(ids)}

        fig = new_figure(
            "Stage 4: comparison with human drivers in recorded traffic",
            f"Scene {index + 1} of 4 – {caption}",
        )
        x0, y0 = corner
        window = (x0, x0 + HUMAN_WINDOW, y0, y0 + HUMAN_WINDOW)
        if overlay:
            axes = panel_axes(fig, 1, window)
            road(axes[0], "round_location0")
            # Under the vehicles; the outlines on top keep both versions visible
            links = LineCollection([], zorder=2.8)
            axes[0].add_collection(links)
            policy = Fleet(axes[0], half_c, colors)
            humans = Fleet(axes[0], half_h, colors, filled=False, zorder=4, face="none")
            note = "Filled: our controlled vehicles. Outlined: human drivers. Each vehicle enters at its recorded time and state."
        else:
            axes = panel_axes(fig, 2, window)
            for ax, title in zip(
                axes, ("Recorded human drivers", "Fine-tuned policy + BH-CBF filter")
            ):
                road(ax, "round_location0")
                panel_title(ax, title, BLUE if "BH-CBF" in title else TEXT)
            humans = Fleet(axes[0], half_h, colors, filled=False)
            # Labels with arrows to the blue vehicle: the human slows down and stops, the policy enters
            slow, (stop_from, stop_to), enter_to, label_xy = HUMAN_CUES[name]
            cue_h, cue_c = (ax.annotate(
                "", xy=label_xy, xytext=label_xy, fontsize=26, weight="bold", color=BLUE,
                ha="center", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.35", color=BLUE,
                                lw=2.5, shrinkA=14, shrinkB=22),
                path_effects=[patheffects.withStroke(linewidth=5, foreground="white")],
            ) for ax in axes)  # fmt: skip
            trails = Tails(axes[1], half_c, colors)
            policy = Fleet(axes[1], half_c, colors)
            note = "Each vehicle enters at its recorded time and state; a color marks the same vehicle."
        clock = footer(fig, right=note)
        prefix = "stage4_scene" if overlay else "stage4_side_by_side_scene"
        clip.start_raw(f"{prefix}_{index + 1}", fig, axes)
        steps = len(times) - 1
        for frame in range(int(steps * DT / speed * FPS) + 1):
            tau = frame * speed / (FPS * DT)
            poses_h = interpolate(pose_h, tau)
            humans.update(poses_h)
            poses = interpolate(pose_c, tau)
            if overlay:
                both = ~np.isnan(poses[:, 0]) & ~np.isnan(poses_h[:, 0])
                links.set_segments([[poses_h[k, :2], poses[k, :2]] for k in np.nonzero(both)[0]])
                ahead = np.array([np.interp(tau, np.arange(len(gap)), gap[:, k]) for k in range(gap.shape[1])])
                links.set_colors([HUMAN_AHEAD if ahead[k] < -LEAD else LINK
                                  for k in np.nonzero(both)[0]])  # fmt: skip
                links.set_linewidths(1.0)
            i = max(
                0,
                int(
                    np.searchsorted(
                        tail_times, times[0] + np.floor(tau + 1e-9) * DT + 1e-6
                    )
                )
                - 1,
            )
            if not overlay:
                seconds, k = tau * DT, column[keys[0]]
                text = "Slow down" if slow <= seconds < stop_from else "Stop" if stop_from <= seconds <= stop_to else ""
                cue_h.set_text(text)
                cue_h.set_visible(bool(text) and not np.isnan(poses_h[k, 0]))
                cue_h.xy = poses_h[k, :2]
                cue_c.set_text("Decisively enter")
                cue_c.set_visible(slow <= seconds <= enter_to and not np.isnan(poses[k, 0]))
                cue_c.xy = poses[k, :2]
                trails.update({column[key]: tail for key, tail in captured[i][1].items() if key in column},
                              ~np.isnan(poses[:, 0]))  # fmt: skip
            policy.update(poses)
            clock.set_text(time_text(tau * DT, speed))
            clip.add(fig)
        clip.add(fig, 2 * FPS if overlay else FPS // 2)  # Time to look at the links
        clip.stop_raw()
        plt.close(fig)


def scene_logos(clip):
    """The closing card: the logos of both universities."""
    fig = new_figure()
    for name, y, width in (("logo_i11_rwth.png", 0.67, 0.7), ("logo_unibw.png", 0.33, 0.7)):
        picture = plt.imread(os.path.join(ASSETS, name))
        height = width * WIDTH / HEIGHT * picture.shape[0] / picture.shape[1]
        ax = fig.add_axes([0.5 - width / 2, y - height / 2, width, height])
        ax.imshow(picture)
        ax.axis("off")
    clip.add(fig, 3 * FPS)
    plt.close(fig)


def scene_repair(clip):
    """One filter problem of one vehicle with a repair round (repair.py),
    seen from the vehicle (heading up): the deployment at real time up to the
    step, then the rounds of the certified filter, paused, then real time
    again. Right: one cell per QP constraint (safety constraint e, prediction
    step t)."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D

    step, ego = REPAIR_EVENT
    math_font = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "cm"  # Math in the look of the paper
    before, after = REPAIR_LEAD
    with open(os.path.join(REPAIR, f"{REPAIR_NAME}.pkl"), "rb") as file:
        problems = {(p["step"], p["agent"]): p for p in pickle.load(file)}
    problem = problems[(step, ego)]
    pose = load_td(REPAIR_RUN)["pose"][step - before : step + after + 1].copy()
    n = pose.shape[1]
    # Each vehicle only between the respawns around the step (a respawn jumps to an entry)
    jump = np.linalg.norm(pose[1:, :, :2] - pose[:-1, :, :2], axis=-1) > JUMP
    for k in range(n):
        hits = np.nonzero(jump[:, k])[0] + 1
        pose[: max([h for h in hits if h <= before], default=0), k] = np.nan
        pose[min([h for h in hits if h > before], default=len(pose)) :, k] = np.nan

    # Safety constraints e: lane sides first, then the neighbors j in N^i; the
    # neighbors with QP constraints or violations get key colors
    labels = problem["labels"]
    rows = sorted(range(len(labels)), key=lambda e: (labels[e][0] != "lane", labels[e][1]))
    involved = {e for e, _ in problem["initial"]}
    for r in problem["rounds"]:
        involved |= {e for e, _ in r["violated"]}
    colors = [CONTEXT] * n
    colors[ego] = BLUE
    key = iter(NEIGHBOR_COLORS)
    names = {}
    for e in rows:
        kind, other = labels[e]
        if kind == "lane":
            names[e] = (f"$\\mathrm{{{other}}}$", TEXT)
            continue
        colors[other] = next(key) if e in involved else MUTED
        names[e] = (f"$j_{{{len(names) - 1}}}$", colors[other])

    fig = new_figure(
        "Inside the Filter: Repair by Lazy Constraint Generation",
        "Vehicle $i$ (blue) in the roundabout. Trails: predictions until standstill.",
    )
    # Map panel, heading up: the view transform maps the world to vehicle coordinates
    half_view = REPAIR_VIEW / 2
    (ax,) = panel_axes(fig, 1, (-half_view, half_view, -half_view, half_view),
                       top=0.82, bottom=0.08, margin=0.035)  # fmt: skip
    ax.set_position([0.035, 0.08, 0.74 * HEIGHT / WIDTH, 0.74])
    road(ax, "round_location0")
    view = Affine2D()
    braking = PolyCollection([], edgecolors="none", zorder=2)
    own = PolyCollection([], linewidths=1.0, zorder=2.2)
    marks = PolyCollection([], facecolors="none", edgecolors=RED, linewidths=3.0, zorder=5)
    for collection in (braking, own, marks):
        ax.add_collection(collection)
    fleet = Fleet(ax, HALF, colors)
    tags = {ego: ax.text(0, 0, "$i$", fontsize=26, color=BLUE, weight="bold", ha="center", va="center")}
    for e, (name, color) in names.items():
        if labels[e][0] == "pair":
            tags[labels[e][1]] = ax.text(0, 0, name, fontsize=26, color=color, weight="bold",
                                         ha="center", va="center")  # fmt: skip
    for tag in tags.values():
        tag.set_path_effects([patheffects.withStroke(linewidth=4, foreground="white")])
        tag.set_zorder(8)
        tag.set_clip_on(True)
    for artist in [*ax.patches, *ax.lines, *ax.collections, *tags.values()]:
        artist.set_transform(view + ax.transData)

    def update_vehicles(poses):
        fleet.update(poses)
        for k, tag in tags.items():
            visible = not np.isnan(poses[k]).any()
            tag.set_visible(visible)
            if visible:
                tag.set_position(poses[k, :2] + 3.2 * np.array([np.cos(poses[k, 2] + np.pi / 2),
                                                                 np.sin(poses[k, 2] + np.pi / 2)]))  # fmt: skip

    def look_from(p):
        view.clear().translate(-p[0], -p[1]).rotate(np.pi / 2 - p[2]).translate(0, -REPAIR_VIEW * 0.2)

    def draw_predictions(record, prediction, color, violated=()):
        """Braking predictions of the colored neighbors, the vehicle's own
        prediction, and red outlines at the violated (e, t)."""
        verts, faces = [], []
        for k, agent in enumerate(record["agents"]):
            if agent != ego and colors[agent] not in (CONTEXT, MUTED):
                for p in record["braking"][:, k]:
                    verts.append(rectangle(p, HALF))
                    faces.append(tint(colors[agent], 0.22))
        braking.set_verts(verts)
        braking.set_facecolors(faces)
        own.set_verts([rectangle(p, HALF) for p in prediction])
        own.set_facecolors([tint(color, 0.2)] * len(prediction))
        own.set_edgecolors([tint(color, 0.6)] * len(prediction))
        red = []
        for e, c in violated:
            red.append(rectangle(prediction[c], HALF))
            if record["labels"][e][0] == "pair":
                k = record["agents"].index(record["labels"][e][1])
                red.append(rectangle(record["braking"][c, k], HALF))
        marks.set_verts(red)

    def applied(record):
        if record["fallback"]:
            return record["prediction_fallback"]
        return record["rounds"][-1]["prediction"] if record["rounds"] else record["prediction_nom"]

    # Grid: one cell per QP constraint (e, t), t = 2, ..., N
    n_t = problem["rhs"].shape[1]
    columns = n_t - 1
    gax = fig.add_axes([0.5, 0.80 - 0.032 * (len(rows) + 1.2), 0.465, 0.032 * (len(rows) + 1.2)])
    gax.set_xlim(-1.6, columns)
    gax.set_ylim(len(rows), -1.2)
    gax.axis("off")
    gax.text(-0.8, -0.6, "$t$", fontsize=20, color=MUTED, ha="center", va="center")
    for col in range(columns):
        if col % 3 == 0 or col == columns - 1:
            gax.text(col + 0.5, -0.6, str(col + 2), fontsize=16, color=MUTED, ha="center", va="center")
    cells, inner, crosses = {}, {}, {}
    for r, e in enumerate(rows):
        name, color = names[e]
        gax.text(-0.8, r + 0.5, name, fontsize=20, color=color, ha="center", va="center", weight="bold")
        for col in range(columns):
            c = col + 1  # code index of t = col + 2
            cells[e, c] = gax.add_patch(Rectangle((col, r), 1, 1, fc=CELL_GRAY, ec="white", lw=1.5))
            inner[e, c] = gax.add_patch(Rectangle((col + 0.22, r + 0.22), 0.56, 0.56, fc=PURPLE,
                                                  ec="white", lw=1.0, visible=False))  # fmt: skip
            crosses[e, c] = gax.text(col + 0.5, r + 0.52, "×", fontsize=26, color=RED, weight="bold",
                                     ha="center", va="center", visible=False,
                                     path_effects=[patheffects.withStroke(linewidth=4, foreground="white")])  # fmt: skip
    initial = set(problem["initial"])
    # Legend and the status lines of the rounds
    y0 = gax.get_position().y0 - 0.06
    for k, (swatch, text) in enumerate(((BLUE, "In QP, linearized at $\\boldsymbol{u}^i_{\\mathrm{nom},k}$"),
                                         (PURPLE, "Added, linearized at candidate $\\hat{\\boldsymbol{u}}^i$"),
                                         (CELL_GRAY, "Not in QP, only checked"),
                                         (None, "Violated in the exact check"))):  # fmt: skip
        x = 0.5 + 0.235 * (k % 2)
        y = y0 - 0.045 * (k // 2)
        if swatch is None:
            fig.text(x + 0.008, y, "×", fontsize=26, color=RED, weight="bold", ha="center", va="center")
        else:
            fig.patches.append(Rectangle((x, y - 0.013), 0.016, 0.026, fc=swatch, transform=fig.transFigure))
        fig.text(x + 0.024, y, text, fontsize=19, color=TEXT, va="center")
    status = []  # Text pieces of the status lines
    clock = footer(fig, right="Each cell is one QP constraint: safety constraint $e$ (row) at prediction step $t$ (column).")

    def show_status(lines):
        """Status lines, each a list of (text, color) pieces set one after another."""
        for text in status:
            text.remove()
        status.clear()
        renderer = fig.canvas.get_renderer()
        for k, pieces in enumerate(lines):
            x = 0.5
            for piece, color in pieces:
                text = fig.text(x, y0 - 0.14 - 0.052 * k, piece, fontsize=22, color=color, va="center")
                x += text.get_window_extent(renderer).width / WIDTH
                status.append(text)

    def real_time(first: int, last: int):
        for frame in range(int((last - first) * DT * FPS) + 1):
            tau = first + frame / (FPS * DT)
            poses = interpolate(pose, tau)
            look_from(poses[ego])
            record = problems.get((step - before + int(tau), ego))
            if record is not None:
                draw_predictions(record, applied(record), BLUE)
            update_vehicles(poses)
            clock.set_text(f"Time {(step - before + tau) * DT:4.1f} s   (real time)")
            clip.add(fig)

    clip.start_raw("repair_round", fig, [ax, gax])
    show_status([])
    real_time(0, before)
    # Paused at the step: the rounds of the filter problem
    poses = pose[before]
    look_from(poses[ego])
    update_vehicles(poses)
    for cell in initial:
        cells[cell].set_facecolor(BLUE)
    clock.set_text(f"Time {step * DT:4.1f} s   (paused: one filter problem of vehicle $i$)")
    rounds = problem["rounds"]
    first = rounds[0]
    omitted = sum(v not in initial for v in first["violated"])
    lines = [[(f"Round 1: QP with {len(initial)} QP constraints near their bound, linearized at "
              "$\\boldsymbol{u}^i_{\\mathrm{nom},k}$", TEXT)]]  # fmt: skip
    draw_predictions(problem, problem["prediction_nom"], BLUE50)
    show_status(lines)
    clip.add(fig, int(3.0 * FPS))
    for r, rnd in enumerate(rounds):
        if r > 0:
            for cell in crosses.values():
                cell.set_visible(False)
            for cell in rnd_prev["added"]:
                if cell in initial:
                    inner[cell].set_visible(True)
                else:
                    cells[cell].set_facecolor(PURPLE)
            draw_predictions(problem, rnd_prev["prediction"], PURPLE)
            lines = lines[-1:] + [[(f"Round {r + 1}: + {len(rnd_prev['added'])} QP constraints linearized at "
                                   "$\\hat{\\boldsymbol{u}}^i$, earlier ones kept → re-solve", TEXT)]]  # fmt: skip
            show_status(lines)
            clip.add(fig, int(3.0 * FPS))
        a, w = rnd["u"]
        lines = lines + [[("Candidate $\\hat{\\boldsymbol{u}}^i$: " + f"$a = {a:.1f}\\,\\mathrm{{m/s^2}}$, ".replace("-", "{-}")
                          + f"$\\omega = {w:.2f}\\,\\mathrm{{rad/s}}$".replace("-", "{-}"), TEXT)]]
        draw_predictions(problem, rnd["prediction"], PURPLE)
        show_status(lines)
        clip.add(fig, int(2.5 * FPS))
        if rnd["violated"]:
            for cell in rnd["violated"]:
                crosses[cell].set_visible(True)
            # Leading spaces: matplotlib drops the trailing ones from a text's extent
            pieces = [("Exact check (every $e$ and $t$, no slack):", TEXT),
                      (f" {len(rnd['violated'])} violated", RED)]
            if r == 0:
                pieces.append((f", {omitted} of them not in the QP", TEXT))
            lines = lines + [pieces]
            draw_predictions(problem, rnd["prediction"], PURPLE, rnd["violated"])
            show_status(lines)
            clip.add(fig, int(3.5 * FPS))
        else:
            lines = lines + [[("Exact check", TEXT), (" passes", BLUE),
                              (" → apply $\\hat{\\boldsymbol{u}}^i$", TEXT)]]
            show_status(lines)
            clip.add(fig, int(3.5 * FPS))
        rnd_prev = rnd
    show_status([])
    marks.set_verts([])
    real_time(before, before + after)
    clip.stop_raw()
    plt.close(fig)
    plt.rcParams["mathtext.fontset"] = math_font


CLIPS = [
    ("title", scene_title),
    ("contributions", scene_contributions),
    ("backup_round", lambda c: scene_backup(c, BACKUP_ROUND, "backup_round", "round_location0", (250, 350),
                                            (40, 130, 31, 79), "Roundabout (rounD)")),
    ("backup_ind", lambda c: scene_backup(c, BACKUP_IND, "backup_ind", "ind_location4", (40, 140),
                                          (64, 146, 38, 82), "Intersection (inD)")),
    ("stage1_card", lambda c: card(c, "Stage 1: pre-training with safety rewards",
                                   "Ten policies per reward: Time-To-Collision (TTC), one-step CBF, and our BH-CBF-informed reward")),
    ("stage1_rewards", scene_rewards),
    ("stage2_card", lambda c: card(c, "Stage 2: zero-shot deployment with run-time filters",
                                   "Unsafe MARL policies: TTC reward, pre-trained for only 100 iterations")),
    ("stage2_counterfactual", scene_counterfactuals),
    ("stage2_comparison", scene_comparison),
    ("stage3_card", lambda c: card(c, "Stage 3: filter-in-the-loop fine-tuning",
                                   "The roundabout policy is fine-tuned on the intersection with BH-CBF in the loop")),
    ("stage3_finetuning", scene_finetuning),
    ("stage4_card", lambda c: card(c, "Stage 4: comparison with human drivers",
                                   "112 recorded roundabout windows, 1264 vehicles with their recorded footprints, over 115 km")),
    ("stage4", scene_humans),
    ("logos", scene_logos),
]  # fmt: skip
# Clips outside the video (written to OUT/extra, which assemble skips; render them with --only)
EXTRAS = [
    ("repair", scene_repair),
    ("stage4_side_by_side", lambda c: scene_humans(c, overlay=False)),
]


def render(args):
    # Carlito: metrically compatible with Calibri, the font of the institute's slides
    for path in glob.glob(os.path.join(CARLITO, "Carlito-*.ttf")):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Carlito"]})
    folder = os.path.join(args.out, "previews" if args.preview else "clips")
    os.makedirs(folder, exist_ok=True)
    selected = [
        (os.path.join(folder, f"{i:02d}_{name}.mp4"), name, fn)
        for i, (name, fn) in enumerate(CLIPS)
        if not args.only or name in args.only
    ]
    extra = os.path.join(args.out, "previews" if args.preview else "extra")
    os.makedirs(extra, exist_ok=True)
    selected += [
        (os.path.join(extra, f"{name}.mp4"), name, fn)
        for name, fn in EXTRAS
        if args.only and name in args.only
    ]
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
    target = os.path.join(args.out, "bh_cbf_video_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-c", "copy", "-movflags", "+faststart", target],
                   check=True)  # fmt: skip
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
