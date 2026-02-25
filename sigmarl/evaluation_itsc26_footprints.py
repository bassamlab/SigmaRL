from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from sigmarl.helper_common import is_latex_available
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import matplotlib.animation as animation
from typing import List

plt.rcParams.update(
    {
        "font.size": 10,  # slightly smaller for IEEE single-column
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "serif",
        "text.usetex": is_latex_available(),
        # PDF export quality
        "pdf.fonttype": 42,  # embed TrueType fonts (better in paper PDFs)
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
)

from sigmarl.map_manager import MapManager


@dataclass(frozen=True)
class VehicleFootprintStyle:
    goal_edge_color: str = "black"
    collision_edge_color: str = "tab:red"
    default_edge_color: str = "black"
    default_linewidth: float = 1.0
    terminal_linewidth: float = 2.5


def animate_vehicle_footprints_from_td(
    td_path: str | Path,
    k0: int,
    veh_length: float,
    veh_width: float,
    *,
    style: VehicleFootprintStyle = VehicleFootprintStyle(),
    fps: int = 10,
    step_interval: int = 1,
    is_show_after_reset: bool = False,  # default: do NOT show after reset
    out_mp4: Optional[str | Path] = None,
) -> Path:
    """
    Animate vehicle footprints and save as mp4.

    Two modes:
    - is_show_after_reset=False (default): for each vehicle n, stop visualizing it after its first terminal
      event (reach goal or collide with lanelets/agents) occurring at or after k0.
    - is_show_after_reset=True: keep visualizing the vehicle across the whole rollout (including resets).

    Style matches the static plot:
    - same vehicle colors (skip tab:red)
    - same polygon edge rules for terminal event (goal_edge_color / collision_edge_color)
    - same vehicle ID circle (solid) and white ID text
    - no alpha encoding of time (use alpha=1.0 for faces)
    """
    td_path = Path(td_path)
    td = torch.load(td_path, weights_only=False)

    # Required
    key_pos = ("agents", "info", "pos")
    key_rot = ("agents", "info", "rot")
    key_goal = ("agents", "info", "is_reach_goal")
    key_coll_lane = ("agents", "info", "is_collision_with_lanelets")
    key_coll_agents = ("agents", "info", "is_collision_with_agents")

    pos = _to_numpy_bool(td.get(key_pos))  # [T,N,2]
    rot = _to_numpy_bool(td.get(key_rot))  # [T,N,1]
    goal_raw = _to_numpy_bool(td.get(key_goal))  # [T,N,1]
    lane_raw = _to_numpy_bool(td.get(key_coll_lane))  # [T,N,1]
    ag_raw = _to_numpy_bool(td.get(key_coll_agents))  # [T,N,1]

    T, N, _ = pos.shape

    if not (0 <= k0 < T):
        raise ValueError(f"k0 must be in [0, {T-1}], got {k0}")
    if step_interval <= 0:
        raise ValueError(f"step_interval must be positive, got {step_interval}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    def _event_bool(arr_tnc: np.ndarray) -> np.ndarray:
        return arr_tnc[..., 0].astype(np.float64) != 0.0

    goal_hit = _event_bool(goal_raw)  # [T,N]
    coll_lane = _event_bool(lane_raw)  # [T,N]
    coll_agents = _event_bool(ag_raw)  # [T,N]

    # Determine terminal step per vehicle (first event >= k0)
    kend = np.full((N,), T - 1, dtype=int)
    terminal_cause = np.array(["none"] * N, dtype=object)
    for n in range(N):
        candidates: List[Tuple[int, str]] = []
        for arr, name in [
            (goal_hit, "goal"),
            (coll_lane, "coll_lane"),
            (coll_agents, "coll_agents"),
        ]:
            idx = np.where(arr[k0:, n])[0]
            if idx.size > 0:
                candidates.append((k0 + int(idx[0]), name))
        if candidates:
            tmin, cause = min(candidates, key=lambda x: x[0])
            kend[n] = tmin
            terminal_cause[n] = cause

    # Visualize intersection (background)
    map = MapManager(scenario_type="cpm_mixed")
    fig, ax = map.parser.visualize_map(is_visu_intersection_only=True)
    fig.set_constrained_layout(False)
    fig.set_size_inches(5.0, 5.0, forward=True)
    fig.tight_layout(pad=0.0)

    # Vehicle colors (skip tab:red)
    cmap = plt.get_cmap("tab10")
    allowed_idx = [0, 1, 2, 4, 5, 6, 7, 8, 9]
    colors = [cmap(allowed_idx[i % len(allowed_idx)]) for i in range(N)]

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    # Pre-create patches and texts (fast animation)
    polys: List[Polygon] = []
    id_circles: List[Circle] = []
    id_texts: List[plt.Text] = []

    for n in range(N):
        # Initial dummy footprint
        center0 = pos[k0, n, :]
        heading0 = float(rot[k0, n, 0])
        corners0 = _rect_corners_world(center0, heading0, veh_length, veh_width)

        poly = Polygon(
            corners0,
            closed=True,
            facecolor=colors[n],
            edgecolor=style.default_edge_color,
            linewidth=style.default_linewidth,
            alpha=1.0,
            zorder=20,
        )
        ax.add_patch(poly)
        polys.append(poly)

        # Vehicle ID circle at current center (radius = veh_width/2 as in your script)
        circle = Circle(
            (float(center0[0]), float(center0[1])),
            radius=float(veh_width) / 2,
            facecolor=colors[n],
            edgecolor="none",
            alpha=1.0,
            zorder=40,
        )
        ax.add_patch(circle)
        id_circles.append(circle)

        txt = ax.text(
            float(center0[0]),
            float(center0[1]),
            f"{n+1}",
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            zorder=41,
        )
        id_texts.append(txt)

    # Frames to animate
    # k_frames = list(range(k0, T, step_interval))

    if not is_show_after_reset:
        k_stop = int(np.max(kend))  # last terminal time among vehicles
        k_frames = list(range(k0, k_stop + 1, step_interval))
        if len(k_frames) == 0 or k_frames[-1] != k_stop:
            k_frames.append(k_stop)  # ensure final frame is included
    else:
        k_frames = list(range(k0, T, step_interval))

    def _is_reset_step(n: int, k: int) -> bool:
        """
        Heuristic reset detection: large jump in position between k-1 and k.
        This avoids needing extra flags from td.
        """
        if k <= k0:
            return False
        dp = pos[k, n, :] - pos[k - 1, n, :]
        return float(np.linalg.norm(dp)) > 1.5 * veh_length  # threshold in meters

    def _update(frame_idx: int):
        k = k_frames[frame_idx]

        for n in range(N):
            # Visibility logic
            if not is_show_after_reset:
                # Stop drawing after first terminal event
                if k > int(kend[n]):
                    polys[n].set_visible(False)
                    id_circles[n].set_visible(False)
                    id_texts[n].set_visible(False)
                    continue
            else:
                # Show full episode including resets; optionally hide right at reset frame to reduce artifacts
                # (comment out if you want continuous display)
                if _is_reset_step(n, k):
                    # show after reset is allowed; keep visible
                    pass

                polys[n].set_visible(True)
                id_circles[n].set_visible(True)
                id_texts[n].set_visible(True)

            center = pos[k, n, :]
            heading = float(rot[k, n, 0])
            corners = _rect_corners_world(center, heading, veh_length, veh_width)

            # Edge style: highlight terminal frame only (if in no-reset mode)
            if (not is_show_after_reset) and (k == int(kend[n])):
                if terminal_cause[n] == "goal":
                    edge_c = style.goal_edge_color
                    lw = style.terminal_linewidth
                elif terminal_cause[n] in ("coll_lane", "coll_agents"):
                    edge_c = style.collision_edge_color
                    lw = style.terminal_linewidth
                else:
                    edge_c = style.default_edge_color
                    lw = style.default_linewidth
            else:
                edge_c = style.default_edge_color
                lw = style.default_linewidth

            polys[n].set_xy(corners)
            polys[n].set_edgecolor(edge_c)
            polys[n].set_linewidth(lw)

            # Update ID marker position
            id_circles[n].center = (float(center[0]), float(center[1]))
            id_texts[n].set_position((float(center[0]), float(center[1])))

        # Optionally show a time annotation in title
        # ax.set_title(f"$t={0.1*k:.1f}\\,\\mathrm{{s}}$")
        return polys + id_circles + id_texts

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(k_frames),
        interval=1000.0 / fps,
        blit=True,
    )

    # Output path
    if out_mp4 is None:
        rew_method = next(
            (p for p in td_path.parts if p.startswith("rew_method_")),
            "rew_method_unknown",
        )
        suffix = "with_reset" if is_show_after_reset else "no_reset"
        out_mp4 = td_path.parent / f"anim_footprints_{rew_method}_k{k0}_{suffix}.mp4"
    else:
        out_mp4 = Path(out_mp4)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            # "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vf",
            "scale=1920:-2",  # width=1920px, height auto, even
            "-preset",
            "slow",
            "-crf",
            "16",
        ],
    )
    ani.save(str(out_mp4), writer=writer, dpi=450)

    plt.close(fig)
    print(f"[INFO] Animation saved to {out_mp4}")
    return out_mp4


def _to_numpy_bool(td_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor with shape [B,T,N,C] to numpy; expects B=1.
    """
    if td_tensor.ndim != 4:
        raise ValueError(f"Expected [B,T,N,C], got {tuple(td_tensor.shape)}")
    if td_tensor.shape[0] != 1:
        raise ValueError(f"Expected B=1 in saved td, got B={td_tensor.shape[0]}")
    return td_tensor.detach().cpu().numpy()[0]  # [T,N,C]


def _rect_corners_world(
    center_xy: np.ndarray, heading_rad: float, length: float, width: float
) -> np.ndarray:
    """
    Return rectangle corners (4,2) in world frame, counter-clockwise.
    """
    hl = 0.5 * length
    hw = 0.5 * width

    # Local corners (vehicle frame): x forward, y left
    local = np.array(
        [
            [hl, hw],
            [hl, -hw],
            [-hl, -hw],
            [-hl, hw],
        ],
        dtype=np.float64,
    )  # (4,2)

    c = float(np.cos(heading_rad))
    s = float(np.sin(heading_rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)  # (2,2)

    world = local @ R.T
    world += center_xy.reshape(1, 2)
    return world


def plot_vehicle_footprints_from_td(
    td_path: str | Path,
    k0: int,
    step_interval: int,
    veh_length: float,
    veh_width: float,
    *,
    style: VehicleFootprintStyle = VehicleFootprintStyle(),
    alpha_min: float = 0.15,
    alpha_max: float = 1.0,
) -> plt.Axes:
    """
    Plot vehicle rectangle footprints from a saved TensorDict.

    Inputs
    - td_path: path to the .td file saved by torch.load
    - k0: initial time step index to start plotting
    - step_interval: plot every `step_interval` steps
    - veh_length, veh_width: rectangle size

    Behavior
    - For each vehicle n, determine terminal step kend[n] as the earliest time >= k0
      when the vehicle reaches the goal OR collides with lane boundaries OR collides with other agents.
      If none occurs, kend[n] is the last available time step.
    - Plot footprints for steps k0, k0+step_interval, ..., kend[n].
    - Use distinct face color for each vehicle.
    - Use increasing transparency with time.
    - Mark the final footprint edge:
        - goal: style.goal_edge_color
        - collision with agents or lane: style.collision_edge_color
        - otherwise: style.default_edge_color
      Face color stays the vehicle color.
    """
    td_path = Path(td_path)
    td = torch.load(td_path)

    # Required
    key_pos = ("agents", "info", "pos")
    key_rot = ("agents", "info", "rot")
    if key_pos is None or key_rot is None:
        raise KeyError(
            "Missing required keys ('agents','info','pos') and/or ('agents','info','rot')."
        )

    # Terminal / event keys (robust to naming)
    key_goal = ("agents", "info", "is_reach_goal")

    key_coll_lane = ("agents", "info", "is_collision_with_lanelets")

    key_coll_agents = ("agents", "info", "is_collision_with_agents")

    pos = _to_numpy_bool(td.get(key_pos))  # [T,N,2]
    rot = _to_numpy_bool(td.get(key_rot))  # [T,N,1]
    T, N, _ = pos.shape

    dt = 0.1  # [s]

    time_offset_min = 1.3 * veh_width  # horizontal-ish vehicles
    time_offset_max = 2.0 * veh_width  # vertical-ish vehicles

    time_text_fontsize = 7
    time_text_bbox = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.8)

    if not (0 <= k0 < T):
        raise ValueError(f"k0 must be in [0, {T-1}], got {k0}")
    if step_interval <= 0:
        raise ValueError(f"step_interval must be positive, got {step_interval}")

    def _event_bool(arr_tnc: np.ndarray) -> np.ndarray:
        """
        Convert [T,N,1] to boolean [T,N]
        """
        v = arr_tnc[..., 0]
        return v.astype(np.float64) != 0.0

    goal_hit = np.zeros((T, N), dtype=bool)
    coll_lane = np.zeros((T, N), dtype=bool)
    coll_agents = np.zeros((T, N), dtype=bool)

    g_raw = _to_numpy_bool(td.get(key_goal))

    goal_hit = _event_bool(g_raw)

    cl_raw = _to_numpy_bool(td.get(key_coll_lane))
    coll_lane = _event_bool(cl_raw)

    ca_raw = _to_numpy_bool(td.get(key_coll_agents))
    coll_agents = _event_bool(ca_raw)

    # Determine terminal step per vehicle
    kend = np.full((N,), T - 1, dtype=int)
    terminal_cause = np.array(["none"] * N, dtype=object)

    for n in range(N):
        # earliest event time >= k0 among {goal, coll_lane, coll_agents}
        candidates: list[Tuple[int, str]] = []
        for arr, name in [
            (goal_hit, "goal"),
            (coll_lane, "coll_lane"),
            (coll_agents, "coll_agents"),
        ]:
            idx = np.where(arr[k0:, n])[0]
            if idx.size > 0:
                candidates.append((k0 + int(idx[0]), name))
        if candidates:
            tmin, cause = min(candidates, key=lambda x: x[0])
            kend[n] = tmin
            terminal_cause[n] = cause

    # Visualize intersection
    map = MapManager(
        scenario_type="cpm_mixed",
    )
    fig, ax = map.parser.visualize_map(is_visu_intersection_only=True)
    fig.set_constrained_layout(False)

    fig.tight_layout(pad=0.0)

    # Distinct colors per vehicle (skip tab:red because it is reserved for collisions)
    cmap = plt.get_cmap("tab10")
    allowed_idx = [0, 1, 2, 4, 5, 6, 7, 8, 9]  # skip 3 (tab:red)
    colors = [cmap(allowed_idx[i % len(allowed_idx)]) for i in range(N)]

    for n in range(N):
        # k_list = list(range(k0, int(kend[n]) + 1, step_interval))
        # if len(k_list) == 0:
        #     continue

        kend_n = int(kend[n])

        # Base list: k0, k0+step_interval, ..., <= kend_n
        k_list = list(range(k0, kend_n + 1, step_interval))

        # Ensure final step is always plotted (unlucky case)
        if len(k_list) == 0 or k_list[-1] != kend_n:
            k_list.append(kend_n)

        if len(k_list) == 1:
            alphas = [alpha_max]
        else:
            alphas = np.linspace(alpha_min, alpha_max, num=len(k_list)).tolist()

        for ii, k in enumerate(k_list):
            center = pos[k, n, :]
            heading = float(rot[k, n, 0])
            corners = _rect_corners_world(center, heading, veh_length, veh_width)

            is_first = ii == 0
            is_last = k == int(kend[n])

            # Unit vector along vehicle width direction (left normal of heading)
            nx = float(-np.sin(heading))
            ny = float(np.cos(heading))

            # Dynamic offset magnitude: smaller if horizontal, larger if vertical
            s = abs(float(np.sin(heading)))  # 0 (horizontal) -> 1 (vertical)
            time_text_offset = float(
                time_offset_min + (time_offset_max - time_offset_min) * s
            )

            if is_first or is_last:
                t_sec = k * dt

                tx = float(center[0] + time_text_offset * nx)
                ty = float(center[1] + time_text_offset * ny)

                ax.text(
                    tx,
                    ty,
                    rf"${t_sec:.1f}\,\mathrm{{s}}$",
                    ha="center",
                    va="center",
                    fontsize=time_text_fontsize,
                    color=colors[n],
                    bbox=time_text_bbox,
                    zorder=30,
                )

            is_last = k == int(kend[n])
            if is_last:
                if terminal_cause[n] == "goal":
                    edge_c = style.goal_edge_color
                    lw = style.terminal_linewidth
                elif terminal_cause[n] in ("coll_lane", "coll_agents"):
                    edge_c = style.collision_edge_color
                    lw = style.terminal_linewidth
                else:
                    edge_c = style.default_edge_color
                    lw = style.default_linewidth
                alpha_face = alphas[ii]
                alpha_edge = 1.0
            else:
                edge_c = style.default_edge_color
                lw = style.default_linewidth
                alpha_face = alphas[ii]
                alpha_edge = 0.8

            poly = Polygon(
                corners,
                closed=True,
                facecolor=colors[n],
                edgecolor=edge_c,
                linewidth=lw,
                alpha=alpha_face,
            )
            ax.add_patch(poly)

        # Label each vehicle near its first footprint
        # Vehicle ID: solid circle at initial center, radius = veh_width
        c0 = pos[k0, n, :]
        circle = Circle(
            (float(c0[0]), float(c0[1])),
            radius=float(veh_width) / 2,
            facecolor=colors[n],
            edgecolor="none",
            alpha=1.0,
            zorder=40,
        )
        ax.add_patch(circle)

        ax.text(
            float(c0[0]),
            float(c0[1]),
            f"{n+1}",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            zorder=41,
        )

    ax.set_xlim(0.55, 3.95)
    ax.set_ylim(0.465, 3.56)

    # Enforce equal scaling first
    ax.set_aspect("equal", adjustable="box")

    # Adapt figure height to match data aspect ratio (reduces blank margins)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    data_aspect = abs((y1 - y0) / (x1 - x0))

    fig_w = fig.get_size_inches()[0]
    fig_h = fig_w * data_aspect
    fig.set_size_inches(fig_w, fig_h, forward=True)

    ax.grid(True, alpha=0.25)

    # Build output path in the same folder
    td_path = Path(td_path)
    rew_method = next(
        (p for p in td_path.parts if p.startswith("rew_method_")), "rew_method_unknown"
    )

    if rew_method == "rew_method_cbf":
        out_pdf = td_path.parent / f"fig_footprints_{rew_method}_k{k0}.pdf"
    else:
        out_pdf = td_path.parent / f"fig_footprints_{rew_method}.pdf"

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.0)

    print(f"[INFO] Fig saved to {out_pdf}")

    return ax


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    from constants import AGENTS

    case_list = [1, 2, 3, 4, 5]
    # case_list = [
    #     5
    # ]
    for case in case_list:
        if case == 1:
            td_path = "checkpoints/itsc26_new/cpm_mixed_do_not_apply_cbf_action/rew_method_cbf/reward_progress10.0/seed1/out_td_cpm_mixed_nagents4_seed1.td"
            # k0_list = [69, 130, 206, 332, 400, 539]
            k0_list = [69, 206, 332, 539]

        if case == 2:
            td_path = "checkpoints/itsc26_new/cpm_mixed_do_not_apply_cbf_action/rew_method_distance/reward_progress10.0/seed1/out_td_cpm_mixed_nagents4_seed1.td"
            k0_list = [32]

        if case == 3:
            td_path = "checkpoints/itsc26_new/cpm_mixed_do_not_apply_cbf_action/rew_method_ttc/reward_progress10.0/seed1/out_td_cpm_mixed_nagents4_seed1.td"
            k0_list = [38]

        if case == 4:
            td_path = "checkpoints/itsc26_new/cpm_mixed_do_not_apply_cbf_action/rew_method_distance_sparse/reward_progress10.0/seed1/out_td_cpm_mixed_nagents4_seed1.td"
            k0_list = [340]

        if case == 5:
            td_path = "checkpoints/itsc26_new/cpm_mixed_do_not_apply_cbf_action/rew_method_ttc_sparse/reward_progress10.0/seed1/out_td_cpm_mixed_nagents4_seed1.td"
            k0_list = [256]

        for k0 in k0_list:
            ax = plot_vehicle_footprints_from_td(
                td_path=td_path,
                k0=k0,
                step_interval=5,
                veh_length=float(AGENTS["length"]),
                veh_width=float(AGENTS["width"]),
            )

            _ = animate_vehicle_footprints_from_td(
                td_path=td_path,
                k0=k0,
                step_interval=1,
                veh_length=float(AGENTS["length"]),
                veh_width=float(AGENTS["width"]),
                fps=30,
                is_show_after_reset=False,  # include resets
            )

        # plt.show()
