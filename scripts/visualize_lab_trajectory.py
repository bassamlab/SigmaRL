#!/usr/bin/env python3
# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize MARL virtual actions and appended constant actions in TRAJECTORY mode.

This script runs a trained SigmaRL policy in TRAJECTORY mode (on mock or lab backend)
and generates visual frames depicting:
  1. Road network boundaries and lane centerlines.
  2. Vehicle footprints (oriented bounding boxes) and heading directions.
  3. MARL policy-generated trajectory waypoints (Color 1).
  4. Appended constant action / deceleration padding waypoints (Color 2).

Usage:
python scripts/visualize_lab_trajectory.py --model-dir checkpoints/icra27/lab --lab-traj-horizon 3 --n-steps 30 --save-frames 0,5,10,15,20,25 --output-dir outputs/visualize_lab_trajectory
"""

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from sigmarl.helper_common import Parameters
from sigmarl.helper_scenario import get_rectangle_vertices
from sigmarl.helper_training import SaveData
from sigmarl.mappo_cavs import mappo_cavs


def find_reward_json(model_dir: str) -> str:
    for name in os.listdir(model_dir):
        if name.endswith(".json") and name.startswith("reward"):
            return os.path.join(model_dir, name)
    raise FileNotFoundError(f"No reward*.json found in {model_dir}")


# Distinct visual styling
COLOR_MARL = "#0066FF"  # Vibrant Royal Blue
COLOR_CONSTANT = "#FF5500"  # Vibrant Orange-Red
VEHICLE_COLORS = [
    "#2CA02C",  # Green
    "#9467BD",  # Purple
    "#17BECF",  # Cyan
    "#E377C2",  # Pink
    "#BCBD22",  # Yellow-Green
    "#8C564B",  # Brown
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize virtual MARL actions and appended constant actions in trajectory mode."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints/icra27/lab",
        help="Path to trained model directory containing config.json.",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "lab"],
        default="mock",
        help="Physics backend: 'mock' (default) or 'lab'.",
    )
    parser.add_argument(
        "--scenario-type",
        type=str,
        default="cpm_intersection_extended",
        help="Scenario type.",
    )
    parser.add_argument(
        "--lab-traj-horizon",
        type=int,
        default=3,
        help="Number of steps for policy-generated virtual trajectory (default: 3).",
    )
    parser.add_argument(
        "--period-ms",
        type=int,
        default=50,
        help="Control period in milliseconds (default: 50 ms).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help="Number of rollout steps to simulate (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for evaluation (default: 1).",
    )
    parser.add_argument(
        "--vehicle-ids",
        type=str,
        default="1,2,3,4",
        help="Comma-separated vehicle IDs (default: '1,2,3,4').",
    )
    parser.add_argument(
        "--save-frames",
        type=str,
        default="0,5,10,15,20,25",
        help="Comma-separated list of step indices to save as images, or 'all'.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save frame every N steps when --save-frames is empty.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualize_lab_trajectory",
        help="Directory where frame images will be saved.",
    )
    parser.add_argument(
        "--view-mode",
        choices=["full", "zoom", "both"],
        default="full",
        help="Camera view: 'full' (whole map), 'zoom' (close-up on vehicles), or 'both'.",
    )
    parser.add_argument(
        "--create-summary-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to create a combined multi-frame comparison overview.",
    )
    return parser.parse_args()


def render_map_lanelets(ax, map_parser, scenario_type: str):
    """Render road lanelet boundaries and centerlines."""
    for i, lanelet in enumerate(map_parser.lanelets_all):
        if scenario_type == "cpm_mixed":
            if not any((i + 1) in p for p in map_parser.path_intersection):
                continue

        left_b = lanelet["left_boundary"].cpu().numpy()
        right_b = lanelet["right_boundary"].cpu().numpy()
        center_l = lanelet["center_line"].cpu().numpy()

        # Fill lanelet surface subtly
        poly_pts = np.vstack([left_b, right_b[::-1]])
        ax.add_patch(
            patches.Polygon(
                poly_pts,
                closed=True,
                facecolor="#E8E8E8",
                edgecolor="none",
                zorder=1,
            )
        )

        # Draw road boundaries
        ax.plot(
            left_b[:, 0],
            left_b[:, 1],
            color="#555555",
            linewidth=1.2,
            zorder=2,
        )
        ax.plot(
            right_b[:, 0],
            right_b[:, 1],
            color="#555555",
            linewidth=1.2,
            zorder=2,
        )
        # Centerline dashed
        ax.plot(
            center_l[:, 0],
            center_l[:, 1],
            color="#BBBBBB",
            linestyle="--",
            linewidth=0.8,
            zorder=2,
        )


def plot_frame(
    env,
    step: int,
    dt: float,
    horizon: int,
    trajectories: torch.Tensor,
    output_path: Path,
    title_suffix: str = "",
    view_mode: str = "full",
):
    """Plot a single simulation frame with vehicle footprints and trajectories."""
    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)

    # 1. Render Map
    map_parser = env.scenario.map.parser
    render_map_lanelets(ax, map_parser, env.scenario.parameters.scenario_type)

    agents = env.scenario.world.agents
    n_agents = len(agents)
    T_total = trajectories.shape[1] if trajectories is not None else 0
    n_pad = max(0, T_total - (horizon + 1))

    # 2. Render each vehicle footprint and planned trajectory
    for i, agent in enumerate(agents):
        color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
        pos = agent.state.pos[0].detach().cpu()
        rot = agent.state.rot[0].detach().cpu()
        speed = agent.state.speed[0].item()

        # Compute oriented bounding box (footprint)
        vertices = (
            get_rectangle_vertices(
                pos,
                rot,
                width=agent.shape.width,
                length=agent.shape.length,
                is_close_shape=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        # Plot vehicle footprint
        ax.add_patch(
            patches.Polygon(
                vertices,
                closed=True,
                facecolor=color,
                edgecolor="#111111",
                linewidth=1.6,
                alpha=0.45,
                zorder=5,
            )
        )

        # Heading indicator arrow from center
        yaw = rot.item()
        arrow_len = agent.shape.length * 0.65
        dx = arrow_len * math.cos(yaw)
        dy = arrow_len * math.sin(yaw)
        ax.annotate(
            "",
            xy=(pos[0].item() + dx, pos[1].item() + dy),
            xytext=(pos[0].item(), pos[1].item()),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#111111",
                lw=1.5,
                mutation_scale=12,
            ),
            zorder=6,
        )

        # Label vehicle ID and current speed
        ax.text(
            pos[0].item() - 0.12,
            pos[1].item() + 0.14,
            f"V{i+1} ({speed:.2f} m/s)",
            fontsize=8,
            fontweight="bold",
            color="#111111",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=color,
                alpha=0.85,
                lw=1.0,
            ),
            zorder=7,
        )

        # 3. Plot trajectory points
        if trajectories is not None:
            traj_i = trajectories[i].detach().cpu().numpy()  # [T_total, 5]

            # MARL policy actions (points 0 .. horizon)
            marl_pts = traj_i[: horizon + 1, :2]
            ax.plot(
                marl_pts[:, 0],
                marl_pts[:, 1],
                color=COLOR_MARL,
                linestyle="-",
                linewidth=2.2,
                label="MARL Actions" if i == 0 else "",
                zorder=8,
            )
            ax.scatter(
                marl_pts[1:, 0],
                marl_pts[1:, 1],
                color=COLOR_MARL,
                edgecolors="white",
                linewidths=0.8,
                s=40,
                marker="o",
                zorder=9,
            )

            # Appended constant actions (deceleration/centering padding)
            if T_total > horizon + 1:
                const_pts = traj_i[horizon:, :2]
                ax.plot(
                    const_pts[:, 0],
                    const_pts[:, 1],
                    color=COLOR_CONSTANT,
                    linestyle="--",
                    linewidth=2.0,
                    label="Appended Constant Actions" if i == 0 else "",
                    zorder=8,
                )
                ax.scatter(
                    const_pts[1:, 0],
                    const_pts[1:, 1],
                    color=COLOR_CONSTANT,
                    edgecolors="white",
                    linewidths=0.8,
                    s=42,
                    marker="D",
                    zorder=9,
                )

    # Styling and Info Overlay
    time_s = step * dt
    marl_time_s = horizon * dt
    total_lead_s = (T_total - 1) * dt if T_total > 0 else 0.0

    ax.set_title(
        f"Lab Trajectory Mode - Step {step} (t = {time_s:.2f}s){title_suffix}\n"
        f"MARL Horizon: {horizon} steps ({marl_time_s:.2f}s) | "
        f"Padding: {n_pad} steps (Total Lead: {total_lead_s:.2f}s)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    # Info box
    info_text = (
        f"Step: {step}\n"
        f"Simulation Time: {time_s:.2f} s\n"
        f"MARL Horizon (H): {horizon} steps ({marl_time_s:.2f} s)\n"
        f"Constant Decel: {n_pad} steps ({n_pad * dt:.2f} s)\n"
        f"MPC Lead Time: {total_lead_s:.2f} s (>= 0.80s target, >= 0.36s req.)"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.92,
        ),
        zorder=10,
    )

    # Custom Legend
    legend_elements = [
        patches.Patch(
            facecolor="#888888",
            edgecolor="#111111",
            alpha=0.5,
            label="Vehicle Footprint",
        ),
        plt.Line2D(
            [0],
            [0],
            color=COLOR_MARL,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            label=f"Virtual MARL Actions (H={horizon})",
        ),
        plt.Line2D(
            [0],
            [0],
            color=COLOR_CONSTANT,
            marker="D",
            linestyle="--",
            linewidth=2.0,
            markersize=6,
            label=f"Appended Constant Actions ({n_pad} steps)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.92,
        fontsize=8.5,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]", fontsize=9)
    ax.set_ylabel("Y [m]", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Framing around vehicles or whole map
    if view_mode == "zoom":
        all_x = [a.state.pos[0, 0].item() for a in agents]
        all_y = [a.state.pos[0, 1].item() for a in agents]
        if trajectories is not None:
            all_x.extend(trajectories[:, :, 0].reshape(-1).tolist())
            all_y.extend(trajectories[:, :, 1].reshape(-1).tolist())
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        span = max(max_x - min_x, max_y - min_y, 0.8) + 0.6
        ax.set_xlim(cx - span / 2.0, cx + span / 2.0)
        ax.set_ylim(cy - span / 2.0, cy + span / 2.0)
    else:
        ax.set_xlim(-0.2, 5.0)
        ax.set_ylim(-0.2, 4.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved frame to {output_path}")


def create_summary_grid_image(frame_paths: list[Path], output_path: Path):
    """Combine multiple frames into a clean summary comparison grid."""
    import matplotlib.image as mpimg

    n_frames = len(frame_paths)
    if n_frames == 0:
        return

    cols = min(3, n_frames)
    rows = math.ceil(n_frames / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), dpi=150)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, f_path in enumerate(frame_paths):
        img = mpimg.imread(f_path)
        axes[idx].imshow(img)
        axes[idx].axis("off")
        step_num = f_path.stem.split("_")[-1]
        axes[idx].set_title(f"Step {step_num}", fontsize=11, fontweight="bold")

    for idx in range(n_frames, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "MARL Virtual Actions vs Appended Constant Actions Across Steps",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved summary comparison grid to {output_path}")


def main():
    args = parse_args()
    vehicle_ids = [int(x.strip()) for x in args.vehicle_ids.split(",")]
    n_agents = len(vehicle_ids)

    json_path = find_reward_json(args.model_dir)
    with open(json_path, "r") as f:
        parameters: Parameters = SaveData.from_dict(json.load(f)).parameters

    parameters.n_agents = n_agents
    parameters.scenario_type = args.scenario_type
    parameters.num_vmas_envs = 1
    parameters.max_steps = args.n_steps + 5
    parameters.random_seed = args.seed
    parameters.physics_backend = args.backend
    parameters.lab_vehicle_ids = vehicle_ids
    parameters.lab_period_ms = args.period_ms
    parameters.lab_mode = "trajectory"
    parameters.lab_traj_horizon = args.lab_traj_horizon
    parameters.is_testing_mode = True
    parameters.is_load_model = True
    parameters.is_continue_train = False
    parameters.is_load_out_td = False
    parameters.where_to_save = args.model_dir
    parameters.without_lanelet_reset = True
    parameters.is_fast_inference_mode = True
    parameters.is_save_simulation_video = False
    parameters.is_real_time_rendering = False
    parameters.is_using_cbf_testing = False
    parameters.is_using_cbf_training = False
    parameters.is_apply_cbf_action = False
    parameters.is_solve_qp = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse which frames to save
    if args.save_frames.strip().lower() == "all":
        save_step_set = set(range(args.n_steps))
    elif args.save_frames.strip():
        save_step_set = {
            int(x.strip()) for x in args.save_frames.split(",") if x.strip()
        }
    else:
        save_step_set = {s for s in range(args.n_steps) if s % args.save_interval == 0}

    print(f"[INFO] Initializing environment with {n_agents} agents...")
    print(f"[INFO] Horizon: {args.lab_traj_horizon} MARL steps (dt={parameters.dt}s)")
    print(f"[INFO] Saving frames at steps: {sorted(list(save_step_set))}")

    env, decision_making_module, _, _, _, parameters = mappo_cavs(parameters=parameters)
    env.scenario.set_policy(decision_making_module.policy)

    saved_frame_paths = []

    with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()

        for step in range(args.n_steps):
            td = decision_making_module.policy(td)
            td, td_ = env.step_and_maybe_reset(td)

            # Retrieve planned trajectory from WorldLab
            world = env.scenario.world
            trajectories = getattr(world, "last_trajectories", None)
            horizon = getattr(world, "last_marl_horizon", args.lab_traj_horizon)

            if step in save_step_set:
                if args.view_mode in ("full", "both"):
                    suffix = "_full" if args.view_mode == "both" else ""
                    frame_path = output_dir / f"frame_step_{step:04d}{suffix}.png"
                    plot_frame(
                        env=env,
                        step=step,
                        dt=parameters.dt,
                        horizon=horizon,
                        trajectories=trajectories,
                        output_path=frame_path,
                        view_mode="full",
                    )
                    saved_frame_paths.append(frame_path)
                if args.view_mode in ("zoom", "both"):
                    frame_path_zoom = output_dir / f"frame_step_{step:04d}_zoom.png"
                    plot_frame(
                        env=env,
                        step=step,
                        dt=parameters.dt,
                        horizon=horizon,
                        trajectories=trajectories,
                        output_path=frame_path_zoom,
                        title_suffix=" [Zoom]",
                        view_mode="zoom",
                    )
                    if args.view_mode == "zoom":
                        saved_frame_paths.append(frame_path_zoom)

            td = td_

    if args.create_summary_grid and len(saved_frame_paths) > 1:
        grid_path = output_dir / "summary_grid.png"
        create_summary_grid_image(saved_frame_paths, grid_path)

    print(f"[SUCCESS] All frames saved to {output_dir}")


if __name__ == "__main__":
    main()
