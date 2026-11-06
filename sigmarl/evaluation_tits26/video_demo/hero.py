# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Background clip of the project page's header (bassamlab.github.io/BH-CBF).

The backup-horizon scene of the video (BH-CBF braking tails on the roundabout,
video.BACKUP_ROUND) as a wide, full-bleed loop: no title, frame, or text. The
last LOOP seconds crossfade into the first ones, so the clip loops without a
jump. It replays the deployment and its cached tails; it simulates nothing.

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_tits26.video_demo.hero
Output: OUT/hero/hero_round.mp4 and hero_round.jpg (its first frame, the poster).
"""

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sigmarl.evaluation_tits26.paper_fig_stage1_footprints import HALF
from sigmarl.evaluation_tits26.video_demo.video import (
    BACKUP_ROUND, BLUE, DT, FPS, OUT, Fleet, Tails, interpolate, load_tails, load_td, road,
    tails_at,
)  # fmt: skip

WIDTH, HEIGHT = 1920, 800
WINDOW = (18.5, 138.5, 28.0, 78.0)  # Roundabout centered, aspect WIDTH / HEIGHT [m]
STEPS = (250, 350)  # Steps with cached tails (backup_round)
LOOP = 1.0  # Crossfade at the loop point [s]
CRF = 26


def main():
    start, end = STEPS
    pose = load_td(BACKUP_ROUND)["pose"][start : end + 1]
    tail_steps, tails = load_tails("backup_round")
    n = pose.shape[1]
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*WINDOW[:2])
    ax.set_ylim(*WINDOW[2:])
    ax.set_aspect("equal")
    ax.axis("off")
    road(ax, "round_location0")
    trails = Tails(ax, HALF, [BLUE] * n)
    fleet = Fleet(ax, HALF, [BLUE] * n)

    def frame(f: int) -> np.ndarray:
        tau = f / (FPS * DT)
        poses = interpolate(pose, tau)
        trails.update(tails_at(tail_steps, tails, start + tau), ~np.isnan(poses[:, 0]))
        fleet.update(poses)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)

    folder = os.path.join(OUT, "hero")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "hero_round.mp4")
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "slow", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", path],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    total, fade = int((end - start) * DT * FPS), int(LOOP * FPS)
    # Frames fade..total-1; the last `fade` ones blend into frames 0..fade-1, which
    # lead on to frame `fade`, where the loop restarts
    for f in range(fade, total):
        image = frame(f)
        if f >= total - fade:
            weight = (f - (total - fade) + 1) / (fade + 1)
            image = (1 - weight) * image + weight * frame(f - (total - fade))
        if f == fade:
            plt.imsave(path.replace(".mp4", ".jpg"), image.astype(np.uint8))
        process.stdin.write(image.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()
    plt.close(fig)
    print(f"[INFO] Saved {path} ({(total - fade) / FPS:.1f} s)")


if __name__ == "__main__":
    main()
