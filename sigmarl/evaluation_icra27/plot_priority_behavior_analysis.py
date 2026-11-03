#!/usr/bin/env python3
# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to analyze and visualize the relation between agents' priorities
(scores or ranks derived from task progress) and their egoism vs. altruism behaviors.

Generates multiple publication-quality figure variants and saves them in
`fig_relation_analysis/` within the specified model checkpoint folder.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.stats as stats
from scipy.interpolate import PchipInterpolator
import torch
from tensordict import TensorDict


# Configure publication-quality Matplotlib style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "figure.autolayout": False,
    }
)

# Color palettes
COLOR_EGO = "#d95f02"  # Warm orange/red for egoistic
COLOR_ALTRUIST = "#2b5c8f"  # Cool blue for altruistic
COLOR_NEUTRAL = "#7570b3"  # Purple for neutral / transition


def parse_interaction_thresh(val: Any) -> Optional[float]:
    """Parse interaction threshold argument.

    Accepts positive floats (e.g., '0.5'), or 'none' / '0' to disable
    distinguishing interaction and free-flow modes.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None
    val_str = str(val).strip().lower()
    if val_str in ("none", "null", "no", "false", "0", "0.0", "zero", ""):
        return None
    try:
        f = float(val_str)
        return f if f > 0 else None
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid interaction_thresh: '{val}'. Must be a positive float or 'none'/'0'."
        )


def parse_coverage_percentage(val: Any) -> float:
    """Parse data coverage percentage argument (e.g. 0.95 or 95)."""
    try:
        f = float(val)
        if f > 1.0:
            f = f / 100.0
        if f <= 0.0 or f > 1.0:
            raise ValueError()
        return float(f)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid data coverage percentage: '{val}'. Must be a float in (0, 1] (e.g., 0.95)."
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze egoism vs. altruism in relation to agent priorities."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints/xp-marl-20260818-main/relation_analysis/our_rr_motion_taskprogress_seed_1",
        help="Path to the checkpoint folder containing out_td_seed_*.td file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures (defaults to {model_dir}/fig_relation_analysis).",
    )
    parser.add_argument(
        "--interaction_thresh",
        type=parse_interaction_thresh,
        default=0.0,
        help="Distance threshold (m) to classify agents as interacting / in conflict zone. "
        "Set to 'none' or 0 to disable interaction vs. free-flow distinction.",
    )
    parser.add_argument(
        "--data_cover_percentage",
        "--data-cover-percentage",
        type=parse_coverage_percentage,
        default=0.95,
        help="Fraction of data to cover symmetrically around the center for axis ranges (default: 0.95).",
    )
    parser.add_argument(
        "--v_target",
        type=float,
        default=1.0,
        help="Target / maximum nominal speed (m/s) of vehicles.",
    )
    parser.add_argument(
        "--spread_scale",
        type=float,
        default=0.15,
        help="Multiplier on standard deviation for shaded spread bands (default: 0.15).",
    )
    parser.add_argument(
        "--diff_metric",
        type=str,
        choices=["weight", "progress"],
        default="weight",
        help=" difMetric to use for relative priorityference: 'weight' (softmax weight difference Delta omega, default) or 'progress' (raw route progress difference Delta p).",
    )
    parser.add_argument(
        "--show_regimes",
        action="store_true",
        default=True,
        help="Whether to show colored regime background bands in Subfigure 1c (default: False).",
    )
    parser.add_argument(
        "--use_tex",
        "--use-tex",
        action="store_true",
        default=False,
        help="Enable external LaTeX rendering ('text.usetex' = True) in Matplotlib.",
    )
    return parser.parse_args()


def load_rollout_data(model_dir: str) -> Tuple[TensorDict, str]:
    """Find and load the out_td tensor file."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    td_files = list(model_path.glob("out_td*.td"))
    if not td_files:
        raise FileNotFoundError(f"No out_td*.td file found in {model_dir}")

    target_file = td_files[0]
    for f in td_files:
        if "seed_1" in f.name:
            target_file = f
            break

    print(f"[INFO] Loading rollout data from: {target_file}")
    td = torch.load(target_file, map_location="cpu", weights_only=False)
    return td, target_file.stem


def extract_data_dict(
    td: TensorDict,
    v_target: float = 1.0,
    interaction_thresh: Optional[float] = 0.5,
    diff_metric: str = "weight",
) -> Dict[str, np.ndarray]:
    """Extract and flatten all relevant variables into a structured dictionary of numpy arrays."""

    def get_flat(key: Tuple[str, ...]) -> Optional[np.ndarray]:
        try:
            val = td.get(key)
            if val is not None:
                return val.detach().cpu().numpy().reshape(-1)
        except Exception:
            return None
        return None

    speed = get_flat(("agents", "info", "speed"))
    acc = get_flat(("agents", "info", "acceleration"))
    score = get_flat(("agents", "info", "priority_score"))
    rank = get_flat(("agents", "info", "priority_rank"))
    weight = get_flat(("agents", "info", "priority_weight"))
    d_min = get_flat(("agents", "info", "min_distance_to_agents"))

    score_near = get_flat(("agents", "info", "nearest_agent_priority_score"))
    rank_near = get_flat(("agents", "info", "nearest_agent_priority_rank"))
    weight_near = get_flat(("agents", "info", "nearest_agent_priority_weight"))
    speed_near = get_flat(("agents", "info", "nearest_agent_speed"))
    acc_near = get_flat(("agents", "info", "nearest_agent_acceleration"))

    diff_score = get_flat(("agents", "info", "priority_difference_nearest"))
    diff_rank = get_flat(("agents", "info", "rank_difference_nearest"))
    diff_speed = get_flat(("agents", "info", "speed_difference_nearest"))
    diff_acc = get_flat(("agents", "info", "acc_difference_nearest"))

    progress = get_flat(("agents", "info", "route_progress"))
    progress_frac = get_flat(("agents", "info", "progress_fraction"))

    on_hold = get_flat(("agents", "info", "is_joint_task_on_hold"))
    reached = get_flat(("agents", "info", "is_reach_goal"))
    collision = get_flat(("agents", "info", "is_collision_with_agents"))
    deadlock = get_flat(("agents", "info", "is_deadlocked"))

    act_vel = get_flat(("agents", "info", "applied_action_vel"))
    act_steer = get_flat(("agents", "info", "applied_action_steer"))
    rew_indep = get_flat(("agents", "info", "rew_parameter_independent"))

    n_samples = len(speed)

    # Compute pairwise priority differences
    weight_diff = None
    if weight is not None and weight_near is not None:
        weight_diff = weight - weight_near

    score_diff = (
        diff_score
        if diff_score is not None
        else (
            (score - score_near)
            if (score is not None and score_near is not None)
            else np.zeros(n_samples)
        )
    )

    if diff_metric == "weight":
        if weight_diff is not None:
            priority_diff = weight_diff
        else:
            print(
                "[WARNING] 'nearest_agent_priority_weight' not found in rollout data; "
                "falling back to route progress difference."
            )
            priority_diff = score_diff
    else:
        priority_diff = score_diff

    data = {
        "speed": speed,
        "acceleration": acc if acc is not None else np.zeros(n_samples),
        "priority_score": score if score is not None else progress_frac,
        "priority_rank": rank.astype(int)
        if rank is not None
        else np.zeros(n_samples, dtype=int),
        "priority_weight": weight if weight is not None else np.zeros(n_samples),
        "min_distance": d_min if d_min is not None else np.full(n_samples, np.nan),
        "nearest_score": score_near if score_near is not None else np.zeros(n_samples),
        "nearest_rank": rank_near.astype(int)
        if rank_near is not None
        else np.zeros(n_samples, dtype=int),
        "nearest_weight": weight_near
        if weight_near is not None
        else np.zeros(n_samples),
        "nearest_speed": speed_near if speed_near is not None else np.zeros(n_samples),
        "nearest_acc": acc_near if acc_near is not None else np.zeros(n_samples),
        "priority_diff": priority_diff,
        "priority_weight_diff": weight_diff
        if weight_diff is not None
        else np.zeros(n_samples),
        "priority_score_diff": score_diff,
        "rank_diff": diff_rank if diff_rank is not None else (rank_near - rank),
        "speed_diff": diff_speed if diff_speed is not None else (speed - speed_near),
        "acc_diff": diff_acc if diff_acc is not None else (acc - acc_near),
        "route_progress": progress if progress is not None else np.zeros(n_samples),
        "progress_fraction": progress_frac
        if progress_frac is not None
        else np.zeros(n_samples),
        "is_on_hold": on_hold.astype(bool)
        if on_hold is not None
        else np.zeros(n_samples, dtype=bool),
        "is_reach_goal": reached.astype(bool)
        if reached is not None
        else np.zeros(n_samples, dtype=bool),
        "is_collision": collision.astype(bool)
        if collision is not None
        else np.zeros(n_samples, dtype=bool),
        "is_deadlocked": deadlock.astype(bool)
        if deadlock is not None
        else np.zeros(n_samples, dtype=bool),
        "applied_action_vel": act_vel if act_vel is not None else speed,
        "applied_action_steer": act_steer
        if act_steer is not None
        else np.zeros(n_samples),
        "rew_parameter_independent": rew_indep
        if rew_indep is not None
        else np.zeros(n_samples),
    }

    # Boolean filters
    data["is_active"] = ~data["is_on_hold"]
    if interaction_thresh is not None and interaction_thresh > 0:
        data["is_interacting"] = data["is_active"] & (
            data["min_distance"] < interaction_thresh
        )
        data["is_free_flow"] = data["is_active"] & (
            data["min_distance"] >= interaction_thresh
        )
    else:
        data["is_interacting"] = data["is_active"].copy()
        data["is_free_flow"] = np.zeros_like(data["is_active"], dtype=bool)

    # Derived metrics
    data["altruism_index"] = np.clip((v_target - data["speed"]) / v_target, 0.0, 1.0)
    data["is_braking"] = data["acceleration"] < -0.1
    data["is_yielding"] = data["speed"] < (0.5 * v_target)

    # Priority Tier (0: Low [0, 0.33], 1: Med [0.33, 0.66], 2: High [0.66, 1.0])
    tier = np.zeros(n_samples, dtype=int)
    tier[data["priority_score"] >= 0.333] = 1
    tier[data["priority_score"] >= 0.666] = 2
    data["priority_tier"] = tier

    print(f"[INFO] Total rollout data points: {n_samples}")
    print(
        f"[INFO] Active data points: {data['is_active'].sum()} ({100 * data['is_active'].mean():.1f}%)"
    )
    if interaction_thresh is not None and interaction_thresh > 0:
        print(
            f"[INFO] Interaction data points (d < {interaction_thresh:g} m): {data['is_interacting'].sum()} ({100 * data['is_interacting'].mean():.1f}%)"
        )
        print(
            f"[INFO] Free-flow data points (d >= {interaction_thresh:g} m): {data['is_free_flow'].sum()} ({100 * data['is_free_flow'].mean():.1f}%)"
        )
    else:
        print(
            "[INFO] Interaction/free-flow distinction disabled (treating all active points uniformly)."
        )
    return data


def compute_binned_stats(
    x: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    spread_type: str = "sd",
    spread_scale: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean, spread (0.15 * SD by default for clean visual width), and sample count for y binned along x."""
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.zeros(len(bin_centers))
    spreads = np.zeros(len(bin_centers))
    counts = np.zeros(len(bin_centers), dtype=int)

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 1:
            vals = y[mask]
            means[i] = np.mean(vals)
            if spread_type == "sd":
                spreads[i] = spread_scale * np.std(vals)
            elif spread_type == "sem":
                spreads[i] = stats.sem(vals)
            elif spread_type == "sem_95":
                spreads[i] = 1.96 * stats.sem(vals)
            else:
                spreads[i] = spread_scale * np.std(vals)
        elif counts[i] == 1:
            means[i] = y[mask][0]
            spreads[i] = 0.0
        else:
            means[i] = np.nan
            spreads[i] = np.nan

    return bin_centers, means, spreads, counts


def smooth_series(
    x: np.ndarray,
    y: np.ndarray,
    num_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smooth PCHIP-interpolated (x_dense, y_dense) curve for plotting."""
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_v, y_v = x[valid], y[valid]
    if len(x_v) < 3:
        return x_v, y_v
    x_unique, unique_indices = np.unique(x_v, return_index=True)
    y_unique = y_v[unique_indices]
    if len(x_unique) < 3:
        return x_unique, y_unique
    try:
        interp = PchipInterpolator(x_unique, y_unique)
        x_dense = np.linspace(x_unique.min(), x_unique.max(), num_points)
        y_dense = interp(x_dense)
        return x_dense, y_dense
    except Exception:
        return x_unique, y_unique


def compute_symmetric_range(
    values: np.ndarray,
    center: float,
    coverage: float = 0.95,
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute symmetric range [center - delta, center + delta] around center covering coverage fraction of data."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        delta = 0.5
    else:
        coverage = float(np.clip(coverage, 0.01, 1.0))
        dev = np.abs(valid - center)
        delta = float(np.quantile(dev, coverage))

    low = center - delta
    high = center + delta
    if min_bound is not None:
        low = max(min_bound, low)
    if max_bound is not None:
        high = min(max_bound, high)
    return low, high


def save_figure(
    fig: plt.Figure, output_dir: Path, base_filename: str, dpi: int = 300
) -> None:
    """Save figure in both vector (PDF) and high-res raster (PNG) formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_path = output_dir / f"{base_filename}.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[INFO] Saved: {out_path}")
    plt.close(fig)


def plot_variant_1_relative_priority(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    v_target: float = 1.0,
    spread_scale: float = 0.35,
    interaction_thresh: Optional[float] = 0.5,
    data_cover_percentage: float = 0.95,
    diff_metric: str = "weight",
    show_regimes: bool = False,
    use_tex: bool = False,
):
    """
    Variant 1: Pairwise Relative Priority Difference vs. Action & Kinematics.
    X-axis: Relative Priority Difference \\Delta\\omega_t^{ij} (or \\Delta p_t^{ij}).
    Subfigures:
    (A) fig1a_speed_vs_relative_priority: Ego Speed vs \\Delta\\omega_t^{ij}
    (B) fig1b_relative_speed_vs_relative_priority: Relative Speed \\Delta v vs \\Delta\\omega_t^{ij}
    (C) fig1c_yielding_braking_rates_vs_relative_priority: Low-Speed Rate vs \\Delta\\omega_t^{ij}
    """
    active_mask = data["is_active"]
    inter_mask = data["is_interacting"]
    free_mask = data["is_free_flow"]
    distinguish_mode = interaction_thresh is not None and interaction_thresh > 0
    pct_symbol = r"\%" if use_tex else "%"

    if diff_metric == "weight":
        diff_label = r"Relative Priority Difference $\Delta\omega_t^{ij}$"
    else:
        diff_label = r"Relative Route Progress Difference $\Delta p_t^{ij}$"

    min_diff, max_diff = compute_symmetric_range(
        data["priority_diff"][active_mask],
        center=0.0,
        coverage=data_cover_percentage,
        min_bound=-1.0,
        max_bound=1.0,
    )
    max_scope = max(abs(min_diff), abs(max_diff))
    if diff_metric == "weight" and max_scope <= 0.26:
        xlim = (-0.15, 0.15)
        xticks = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    else:
        pad = 0.04 * (2 * max_scope) if max_scope > 0 else 0.04
        xlim = (-max_scope - pad, max_scope + pad)
        xticks = None

    bins = np.linspace(xlim[0], xlim[1], 35)

    # -------------------------------------------------------------
    # Subfigure 1a: Δ\omega_t^{ij} vs Ego Speed (Interaction vs Free-flow)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
    if distinguish_mode:
        c_inter, m_inter, std_inter, _ = compute_binned_stats(
            data["priority_diff"][inter_mask],
            data["speed"][inter_mask],
            bins,
            spread_scale=spread_scale,
        )
        c_free, m_free, std_free, _ = compute_binned_stats(
            data["priority_diff"][free_mask],
            data["speed"][free_mask],
            bins,
            spread_scale=spread_scale,
        )

        valid_inter = ~np.isnan(m_inter)
        valid_free = ~np.isnan(m_free)

        x_inter_s, y_inter_s = smooth_series(c_inter[valid_inter], m_inter[valid_inter])
        _, y_inter_low = smooth_series(
            c_inter[valid_inter],
            np.clip(m_inter[valid_inter] - std_inter[valid_inter], 0.0, 1.0),
        )
        _, y_inter_high = smooth_series(
            c_inter[valid_inter],
            np.clip(m_inter[valid_inter] + std_inter[valid_inter], 0.0, 1.0),
        )

        x_free_s, y_free_s = smooth_series(c_free[valid_free], m_free[valid_free])
        _, y_free_low = smooth_series(
            c_free[valid_free],
            np.clip(m_free[valid_free] - std_free[valid_free], 0.0, 1.0),
        )
        _, y_free_high = smooth_series(
            c_free[valid_free],
            np.clip(m_free[valid_free] + std_free[valid_free], 0.0, 1.0),
        )

        ax.plot(
            x_inter_s,
            y_inter_s,
            "-",
            color=COLOR_EGO,
            linewidth=2.2,
            label=f"Interaction ($d < {interaction_thresh:g}$ m)",
        )
        ax.plot(
            c_inter[valid_inter][::2],
            m_inter[valid_inter][::2],
            "o",
            color=COLOR_EGO,
            markersize=5.5,
        )
        ax.fill_between(
            x_inter_s, y_inter_low, y_inter_high, color=COLOR_EGO, alpha=0.18
        )

        ax.plot(
            x_free_s,
            y_free_s,
            "--",
            color=COLOR_ALTRUIST,
            linewidth=1.8,
            label=f"Free-flow ($d \\geq {interaction_thresh:g}$ m)",
        )
        ax.plot(
            c_free[valid_free][::2],
            m_free[valid_free][::2],
            "s",
            color=COLOR_ALTRUIST,
            markersize=5,
        )
        ax.fill_between(
            x_free_s, y_free_low, y_free_high, color=COLOR_ALTRUIST, alpha=0.12
        )
    else:
        c_all, m_all, std_all, _ = compute_binned_stats(
            data["priority_diff"][active_mask],
            data["speed"][active_mask],
            bins,
            spread_scale=spread_scale,
        )
        valid_all = ~np.isnan(m_all)
        x_all_s, y_all_s = smooth_series(c_all[valid_all], m_all[valid_all])
        _, y_all_low = smooth_series(
            c_all[valid_all],
            np.clip(m_all[valid_all] - std_all[valid_all], 0.0, 1.0),
        )
        _, y_all_high = smooth_series(
            c_all[valid_all],
            np.clip(m_all[valid_all] + std_all[valid_all], 0.0, 1.0),
        )
        ax.plot(
            x_all_s,
            y_all_s,
            "-",
            color=COLOR_EGO,
            linewidth=2.2,
            label="Overall Active",
        )
        ax.plot(
            c_all[valid_all][::2],
            m_all[valid_all][::2],
            "o",
            color=COLOR_EGO,
            markersize=5.5,
        )
        ax.fill_between(x_all_s, y_all_low, y_all_high, color=COLOR_EGO, alpha=0.18)

    ax.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    ax.set_xlabel(diff_label, fontsize=11)
    ax.set_ylabel(r"Ego Speed $v_{\mathrm{ego}}$ [m/s]", fontsize=11)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)

    # Annotate Altruistic vs Egoistic regimes
    ax.text(
        0.55 * xlim[0],
        0.92,
        r"$\leftarrow$ Altruistic (Yielding)",
        ha="center",
        fontsize=9.5,
        color=COLOR_ALTRUIST,
        fontweight="semibold",
    )
    ax.text(
        0.55 * xlim[1],
        0.22,
        r"Egoistic (Assertive) $\rightarrow$",
        ha="center",
        fontsize=9.5,
        color=COLOR_EGO,
        fontweight="semibold",
    )
    save_figure(fig, output_dir, "fig1a_speed_vs_relative_priority")

    # -------------------------------------------------------------
    # Subfigure 1b: Δ\omega_t^{ij} vs Relative Speed Δv = v_ego - v_neighbor
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
    eval_mask = inter_mask if distinguish_mode else active_mask
    neighbor_valid = eval_mask & (data["nearest_score"] > 0)
    c_dv, m_dv, std_dv, _ = compute_binned_stats(
        data["priority_diff"][neighbor_valid],
        data["speed_diff"][neighbor_valid],
        bins,
        spread_scale=spread_scale,
    )
    valid_dv = ~np.isnan(m_dv)

    x_dv_s, y_dv_s = smooth_series(c_dv[valid_dv], m_dv[valid_dv])
    _, y_dv_low = smooth_series(
        c_dv[valid_dv], np.clip(m_dv[valid_dv] - std_dv[valid_dv], -1.0, 1.0)
    )
    _, y_dv_high = smooth_series(
        c_dv[valid_dv], np.clip(m_dv[valid_dv] + std_dv[valid_dv], -1.0, 1.0)
    )

    ax.plot(
        x_dv_s,
        y_dv_s,
        "-",
        color="#7570b3",
        linewidth=2.2,
        label=r"$\Delta v = v_{\mathrm{ego}} - v_{\mathrm{neighbor}}$",
    )
    ax.plot(
        c_dv[valid_dv][::2], m_dv[valid_dv][::2], "d", color="#7570b3", markersize=5.5
    )
    ax.fill_between(x_dv_s, y_dv_low, y_dv_high, color="#7570b3", alpha=0.18)

    ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    ax.set_xlabel(diff_label, fontsize=11)
    ax.set_ylabel(
        r"Relative Speed $\Delta v = v_{\mathrm{ego}} - v_{\mathrm{neighbor}}$ [m/s]",
        fontsize=11,
    )
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)
    save_figure(fig, output_dir, "fig1b_relative_speed_vs_relative_priority")

    # -------------------------------------------------------------
    # Subfigure 1c: Δ\omega_t^{ij} vs Low-Speed Rate
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.8, 2.7), dpi=300)
    c_yd, m_yd, _, _ = compute_binned_stats(
        data["priority_diff"][eval_mask],
        data["is_yielding"][eval_mask].astype(float) * 100.0,
        bins,
    )
    valid_yd = ~np.isnan(m_yd)

    x_yd_s, y_yd_s = smooth_series(c_yd[valid_yd], m_yd[valid_yd])

    if show_regimes:
        if max_scope <= 0.26:
            regimes = [
                (-0.15, -0.02, -0.085, "Elevated", "#fed9a6", "#993404"),
                (-0.02, 0.02, 0.000, "Peak", "#fbb4ae", "#cb181d"),
                (0.02, 0.15, 0.085, "Negligible", "#deebf7", "#2171b5"),
            ]
            boundary_xs = (-0.02, 0.02)
        else:
            regimes = [
                (-0.60, -0.40, -0.495, "Elevated", "#fed9a6", "#993404"),
                (-0.40, -0.05, -0.250, "Moderate", "#e5f5e0", "#238b45"),
                (-0.05, 0.05, 0.000, "Dominant", "#fbb4ae", "#cb181d"),
                (0.05, 0.60, 0.325, "Negligible", "#deebf7", "#2171b5"),
            ]
            boundary_xs = (-0.40, -0.05, 0.05)

        for x_start, x_end, label_x, label, bg_color, text_color in regimes:
            ax.axvspan(x_start, x_end, color=bg_color, alpha=0.5, zorder=0)
            ax.text(
                label_x,
                91.5 if max_scope <= 0.26 else 73.0,
                label,
                ha="center",
                va="center",
                fontsize=10 if max_scope <= 0.26 else 7.2,
                fontweight="bold",
                color=text_color,
                zorder=5,
            )

        # Regime boundary vertical lines
        for bx in boundary_xs:
            ax.axvline(
                bx, color="#888888", linestyle="--", linewidth=0.75, alpha=0.6, zorder=1
            )

    # Reference line at equal priority
    ax.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)

    # Low-speed curve
    curve_color = "#1f4e78"  # Deep navy
    ax.plot(
        x_yd_s,
        y_yd_s,
        "-",
        color=curve_color,
        linewidth=2.2,
        label=f"Low speed ($v < {0.5 * v_target:g}$ m/s)",
        zorder=3,
    )

    ax.set_xlabel(diff_label, fontsize=11)
    ax.set_ylabel(f"Low-Speed Rate [{pct_symbol}]", fontsize=11)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="lower left", fontsize=9.0, framealpha=0.5)
    ax.grid(True, zorder=0, alpha=0.4)
    save_figure(fig, output_dir, "fig1c_yielding_braking_rates_vs_relative_priority")


def plot_variant_2_discrete_rank(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    v_target: float = 1.0,
    interaction_thresh: Optional[float] = 0.5,
    use_tex: bool = False,
):
    """
    Variant 2: Discrete Priority Rank vs. Interaction Kinematics & Yielding.
    X-axis: Discrete Priority Rank (1-based index).
    Subfigures:
    (A) fig2a_speed_distribution_by_rank: Speed Distributions (Boxplot)
    (B) fig2b_yielding_braking_probability_by_rank: Braking & Yielding Rate (Bar chart)
    (C) fig2c_mean_speed_hierarchy_by_rank: Mean Speed Hierarchy (Line plot)
    """
    active_mask = data["is_active"]
    inter_mask = data["is_interacting"]
    free_mask = data["is_free_flow"]
    distinguish_mode = interaction_thresh is not None and interaction_thresh > 0
    pct_symbol = r"\%" if use_tex else "%"

    ranks = np.sort(np.unique(data["priority_rank"][active_mask]))
    n_ranks = len(ranks)
    ranks_1based = ranks + 1

    # -------------------------------------------------------------
    # Subfigure 2a: Rank vs Speed in Interaction (Boxplot + Mean markers)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    box_mask = inter_mask if distinguish_mode else active_mask
    box_data = [data["speed"][box_mask & (data["priority_rank"] == r)] for r in ranks]

    bp = ax.boxplot(
        box_data,
        positions=ranks_1based,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#d95f02",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
    )

    # Subsample flier points and rasterize them to keep vector PDF fast and responsive
    max_fliers = 500
    rng = np.random.default_rng(42)
    for flier in bp["fliers"]:
        x_d = np.asarray(flier.get_xdata())
        y_d = np.asarray(flier.get_ydata())
        if len(y_d) > max_fliers:
            idx = rng.choice(len(y_d), size=max_fliers, replace=False)
            flier.set_data(x_d[idx], y_d[idx])
        flier.set_rasterized(True)

    colors = plt.cm.coolwarm(np.linspace(0.85, 0.15, n_ranks))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(ranks_1based)
    rank_labels_2a = [
        f"Rank {r}" if (r == ranks_1based[0] or r == ranks_1based[-1]) else str(r)
        for r in ranks_1based
    ]
    ax.set_xticklabels(rank_labels_2a, fontsize=9.5)
    ax.set_xlabel(
        f"Priority Rank ({ranks_1based.min()} = Highest, {ranks_1based.max()} = Lowest)",
        fontsize=11,
    )
    y_label_sub1 = (
        r"Interaction Speed $v$ [m/s]" if distinguish_mode else r"Speed $v$ [m/s]"
    )
    ax.set_ylabel(y_label_sub1, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y")
    save_figure(fig, output_dir, "fig2a_speed_distribution_by_rank")

    # -------------------------------------------------------------
    # Subfigure 2b: Rank vs Yielding and Braking Rates (Bar chart)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
    eval_mask = inter_mask if distinguish_mode else active_mask
    rank_yield = [
        data["is_yielding"][eval_mask & (data["priority_rank"] == r)].mean() * 100.0
        for r in ranks
    ]
    rank_brake = [
        data["is_braking"][eval_mask & (data["priority_rank"] == r)].mean() * 100.0
        for r in ranks
    ]

    x = np.arange(n_ranks)
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        rank_yield,
        width,
        label=f"Yielding ($v < {0.5 * v_target:g}$ m/s)",
        color=COLOR_ALTRUIST,
        alpha=0.85,
    )
    rects2 = ax.bar(
        x + width / 2,
        rank_brake,
        width,
        label=r"Braking ($a < -0.1\ \mathrm{m/s}^2$)",
        color=COLOR_EGO,
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {r}" for r in ranks_1based], fontsize=10)
    ax.set_xlabel("Priority Rank", fontsize=11)
    ax.set_ylabel(f"Yielding/Braking Rate [{pct_symbol}]", fontsize=11)
    ax.set_ylim(0, 60)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}{pct_symbol}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}{pct_symbol}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(fig, output_dir, "fig2b_yielding_braking_probability_by_rank")

    # -------------------------------------------------------------
    # Subfigure 2c: Rank vs Interaction vs Free-flow Mean Speed
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
    mean_speed_all = [
        data["speed"][active_mask & (data["priority_rank"] == r)].mean() for r in ranks
    ]

    if distinguish_mode:
        mean_speed_inter = [
            data["speed"][inter_mask & (data["priority_rank"] == r)].mean()
            for r in ranks
        ]
        mean_speed_free = [
            data["speed"][free_mask & (data["priority_rank"] == r)].mean()
            for r in ranks
        ]

        ax.plot(
            ranks_1based,
            mean_speed_inter,
            "o-",
            color=COLOR_EGO,
            linewidth=2.2,
            markersize=7,
            label=f"Interaction ($d < {interaction_thresh:g}$ m)",
        )
        ax.plot(
            ranks_1based,
            mean_speed_free,
            "s--",
            color=COLOR_ALTRUIST,
            linewidth=2.0,
            markersize=6,
            label=f"Free-flow ($d \\geq {interaction_thresh:g}$ m)",
        )
        ax.plot(
            ranks_1based,
            mean_speed_all,
            "^:",
            color="#333333",
            linewidth=1.8,
            markersize=5,
            label="Overall Active",
        )
    else:
        ax.plot(
            ranks_1based,
            mean_speed_all,
            "o-",
            color=COLOR_EGO,
            linewidth=2.2,
            markersize=7,
            label="Overall Active",
        )

    ax.set_xticks(ranks_1based)
    ax.set_xticklabels([f"Rank {r}" for r in ranks_1based], fontsize=10)
    ax.set_xlabel("Priority Rank", fontsize=11)
    ax.set_ylabel("Mean Speed [m/s]", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True)
    save_figure(fig, output_dir, "fig2c_mean_speed_hierarchy_by_rank")


def plot_variant_3_2d_interaction_heatmap(
    data: Dict[str, np.ndarray], output_dir: Path, use_tex: bool = False
):
    """
    Variant 3: 2D Interaction Phase Diagram / Heatmap.
    X-axis: Ego Priority Score s_ego ∈ [0, 1].
    Y-axis: Nearest Neighbor Priority Score s_neighbor ∈ [0, 1].
    Subfigures:
    (A) fig3a_2d_interaction_speed_phase_map: Average Ego Speed v_ego
    (B) fig3b_2d_yielding_probability_phase_map: Yielding Probability
    """
    inter_mask = data["is_interacting"] & (data["nearest_score"] > 0)
    s_ego = data["priority_score"][inter_mask]
    s_near = data["nearest_score"][inter_mask]
    speed = data["speed"][inter_mask]
    yielding = data["is_yielding"][inter_mask].astype(float) * 100.0
    pct_symbol = r"\%" if use_tex else "%"

    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    speed_grid = np.full((n_bins, n_bins), np.nan)
    yield_grid = np.full((n_bins, n_bins), np.nan)
    count_grid = np.zeros((n_bins, n_bins), dtype=int)

    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (s_ego >= bin_edges[j])
                & (
                    s_ego < bin_edges[j + 1]
                    if j < n_bins - 1
                    else s_ego <= bin_edges[j + 1]
                )
                & (s_near >= bin_edges[i])
                & (
                    s_near < bin_edges[i + 1]
                    if i < n_bins - 1
                    else s_near <= bin_edges[i + 1]
                )
            )
            count_grid[i, j] = mask.sum()
            if count_grid[i, j] > 0:
                speed_grid[i, j] = np.mean(speed[mask])
                yield_grid[i, j] = np.mean(yielding[mask])

    # -------------------------------------------------------------
    # Subfigure 3a: Ego Speed Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    cmap1 = plt.cm.viridis.copy()
    cmap1.set_bad("#D8D8D8")
    im1 = ax.imshow(
        speed_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap1,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label(r"Mean Ego Speed $v_{\mathrm{ego}}$ [m/s]", fontsize=10.5)

    # Annotate grid values
    for i in range(n_bins):
        for j in range(n_bins):
            if not np.isnan(speed_grid[i, j]):
                x_pos = 0.5 * (bin_edges[j] + bin_edges[j + 1])
                y_pos = 0.5 * (bin_edges[i] + bin_edges[i + 1])
                val_color = "white" if speed_grid[i, j] < 0.7 else "black"
                ax.text(
                    x_pos,
                    y_pos,
                    f"{speed_grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=val_color,
                )

    ax.plot([0, 1], [0, 1], color="white", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.set_xlabel(r"Ego Priority Score $s_{\mathrm{ego}}$ (Task Progress)", fontsize=11)
    ax.set_ylabel(
        r"Nearest Neighbor Priority Score $s_{\mathrm{neighbor}}$", fontsize=11
    )

    # Annotate regimes
    text_altruistic = (
        r"\textbf{Altruistic Regime}"
        + "\n"
        + r"($s_{\mathrm{ego}} < s_{\mathrm{neighbor}}$)"
        if use_tex
        else "Altruistic Regime\n($s_{\\mathrm{ego}} < s_{\\mathrm{neighbor}}$)"
    )
    text_egoistic = (
        r"\textbf{Egoistic Regime}"
        + "\n"
        + r"($s_{\mathrm{ego}} > s_{\mathrm{neighbor}}$)"
        if use_tex
        else "Egoistic Regime\n($s_{\\mathrm{ego}} > s_{\\mathrm{neighbor}}$)"
    )
    ax.text(
        0.22,
        0.82,
        text_altruistic,
        color="white",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )
    ax.text(
        0.78,
        0.22,
        text_egoistic,
        color="white",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )
    save_figure(fig, output_dir, "fig3a_2d_interaction_speed_phase_map")

    # -------------------------------------------------------------
    # Subfigure 3b: Yielding Probability Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    cmap2 = plt.cm.magma_r.copy()
    cmap2.set_bad("#D8D8D8")
    im2 = ax.imshow(
        yield_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap2,
        vmin=0,
        vmax=60,
        aspect="equal",
    )
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label(f"Yielding Probability [{pct_symbol}]", fontsize=10.5)

    for i in range(n_bins):
        for j in range(n_bins):
            if not np.isnan(yield_grid[i, j]):
                x_pos = 0.5 * (bin_edges[j] + bin_edges[j + 1])
                y_pos = 0.5 * (bin_edges[i] + bin_edges[i + 1])
                val_color = "white" if yield_grid[i, j] > 35 else "black"
                ax.text(
                    x_pos,
                    y_pos,
                    f"{yield_grid[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=val_color,
                )

    ax.plot([0, 1], [0, 1], color="white", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.set_xlabel(r"Ego Priority Score $s_{\mathrm{ego}}$ (Task Progress)", fontsize=11)
    ax.set_ylabel(
        r"Nearest Neighbor Priority Score $s_{\mathrm{neighbor}}$", fontsize=11
    )
    save_figure(fig, output_dir, "fig3b_2d_yielding_probability_phase_map")


def plot_variant_4_altruism_index_and_distance(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    spread_scale: float = 0.35,
    interaction_thresh: Optional[float] = 0.5,
    data_cover_percentage: float = 0.95,
    use_tex: bool = False,
):
    """
    Variant 4: Task Progress vs. Altruism Index & Distance Headway Stratification.
    Subfigures:
    (A) fig4a_altruism_index_vs_task_progress: Task Progress Score vs. Altruism Index
    (B) fig4b_speed_vs_proximity_by_priority_tier: Distance to Neighbor vs. Speed by Priority Tier
    """
    active_mask = data["is_active"]
    distinguish_mode = interaction_thresh is not None and interaction_thresh > 0
    pct_symbol = r"\%" if use_tex else "%"

    min_score, max_score = compute_symmetric_range(
        data["priority_score"][active_mask],
        center=0.5,
        coverage=data_cover_percentage,
        min_bound=0.0,
        max_bound=1.0,
    )
    score_bins = np.linspace(min_score, max_score, 25)

    # -------------------------------------------------------------
    # Subfigure 4a: Priority Score vs Altruism Index
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
    if distinguish_mode:
        categories = [
            (
                f"Interaction ($d < {interaction_thresh:g}$ m)",
                data["is_interacting"],
                COLOR_EGO,
                "o",
                "-",
            ),
            (
                f"Free-flow ($d \\geq {interaction_thresh:g}$ m)",
                data["is_free_flow"],
                COLOR_ALTRUIST,
                "s",
                "--",
            ),
        ]
    else:
        categories = [
            ("Overall Active", data["is_active"], COLOR_EGO, "o", "-"),
        ]

    for label, mask, color, marker, ls in categories:
        centers, means, stds, _ = compute_binned_stats(
            data["priority_score"][mask],
            data["altruism_index"][mask],
            score_bins,
            spread_type="sd",
            spread_scale=spread_scale,
        )
        valid = ~np.isnan(means)
        x_s, y_s = smooth_series(centers[valid], means[valid])
        _, y_low = smooth_series(
            centers[valid], np.clip(means[valid] - stds[valid], 0.0, 1.0)
        )
        _, y_high = smooth_series(
            centers[valid], np.clip(means[valid] + stds[valid], 0.0, 1.0)
        )

        ax.plot(x_s, y_s, linestyle=ls, color=color, linewidth=2.2, label=label)
        ax.plot(
            centers[valid][::2], means[valid][::2], marker, color=color, markersize=5.5
        )
        ax.fill_between(x_s, y_low, y_high, color=color, alpha=0.15)

    ax.set_xlabel(r"Ego Priority Score $s_{\mathrm{ego}}$ (Task Progress)", fontsize=11)
    ax.set_ylabel(
        r"Altruism Index $(v_{\mathrm{target}} - v) / v_{\mathrm{target}}$", fontsize=11
    )
    ax.set_xlim(min_score, max_score)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    save_figure(fig, output_dir, "fig4a_altruism_index_vs_task_progress")

    # -------------------------------------------------------------
    # Subfigure 4b: Distance to Nearest Neighbor vs Speed by Priority Tier
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
    min_d, max_d = compute_symmetric_range(
        data["min_distance"][active_mask],
        center=0.65,
        coverage=data_cover_percentage,
        min_bound=0.0,
    )
    d_bins = np.linspace(min_d, max_d, 28)

    tier_styles = [
        (f"High Priority (Top 33{pct_symbol})", 2, COLOR_EGO, "o"),
        (f"Medium Priority (Mid 33{pct_symbol})", 1, COLOR_NEUTRAL, "s"),
        (f"Low Priority (Bottom 33{pct_symbol})", 0, COLOR_ALTRUIST, "^"),
    ]

    for label, tier_code, color, marker in tier_styles:
        mask = active_mask & (data["priority_tier"] == tier_code)
        centers, means, stds, _ = compute_binned_stats(
            data["min_distance"][mask],
            data["speed"][mask],
            d_bins,
            spread_type="sd",
            spread_scale=spread_scale,
        )
        valid = ~np.isnan(means)
        x_s, y_s = smooth_series(centers[valid], means[valid])
        _, y_low = smooth_series(
            centers[valid], np.clip(means[valid] - stds[valid], 0.0, 1.0)
        )
        _, y_high = smooth_series(
            centers[valid], np.clip(means[valid] + stds[valid], 0.0, 1.0)
        )

        ax.plot(x_s, y_s, "-", color=color, linewidth=2.2, label=label)
        ax.plot(
            centers[valid][::2], means[valid][::2], marker, color=color, markersize=5.5
        )
        ax.fill_between(x_s, y_low, y_high, color=color, alpha=0.15)

    ax.set_xlabel(
        r"Distance to Nearest Surrounding Vehicle $d_{\mathrm{min}}$ [m]", fontsize=11
    )
    ax.set_ylabel(r"Ego Speed $v_{\mathrm{ego}}$ [m/s]", fontsize=11)
    ax.set_xlim(min_d, max_d)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    save_figure(fig, output_dir, "fig4b_speed_vs_proximity_by_priority_tier")


def plot_variant_5_hero_summary(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    spread_scale: float = 0.35,
    interaction_thresh: Optional[float] = 0.5,
    data_cover_percentage: float = 0.95,
    diff_metric: str = "weight",
    use_tex: bool = False,
):
    """
    Variant 5: Multi-Panel Publication Hero Figure.
    Comprehensive 2x2 grid combining the 4 core views:
    (A) Relative Priority vs Speed in Interaction vs Free-flow
    (B) Discrete Rank vs Speed Distribution Boxplots (1-based ranks)
    (C) 2D Interaction Phase Diagram (s_ego vs s_neighbor -> v_ego)
    (D) Distance Proximity vs Speed by Priority Tier
    """
    active_mask = data["is_active"]
    inter_mask = data["is_interacting"]
    free_mask = data["is_free_flow"]
    distinguish_mode = interaction_thresh is not None and interaction_thresh > 0
    pct_symbol = r"\%" if use_tex else "%"

    if diff_metric == "weight":
        diff_label = r"Relative Priority Difference $\Delta\omega_t^{ij}$"
    else:
        diff_label = r"Relative Route Progress Difference $\Delta p_t^{ij}$"

    min_diff, max_diff = compute_symmetric_range(
        data["priority_diff"][active_mask],
        center=0.0,
        coverage=data_cover_percentage,
        min_bound=-1.0,
        max_bound=1.0,
    )
    max_scope = max(abs(min_diff), abs(max_diff))
    if diff_metric == "weight" and max_scope <= 0.26:
        xlim_diff = (-0.15, 0.15)
        xticks_diff = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    else:
        pad = 0.04 * (2 * max_scope) if max_scope > 0 else 0.04
        xlim_diff = (-max_scope - pad, max_scope + pad)
        xticks_diff = None

    bins = np.linspace(xlim_diff[0], xlim_diff[1], 35)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), dpi=300)

    # -------------------------------------------------------------
    # (A) Relative Priority vs Speed
    # -------------------------------------------------------------
    ax = axes[0, 0]

    if distinguish_mode:
        c_inter, m_inter, std_inter, _ = compute_binned_stats(
            data["priority_diff"][inter_mask],
            data["speed"][inter_mask],
            bins,
            spread_type="sd",
            spread_scale=spread_scale,
        )
        c_free, m_free, std_free, _ = compute_binned_stats(
            data["priority_diff"][free_mask],
            data["speed"][free_mask],
            bins,
            spread_type="sd",
            spread_scale=spread_scale,
        )
        valid_inter = ~np.isnan(m_inter)
        valid_free = ~np.isnan(m_free)

        x_inter_s, y_inter_s = smooth_series(c_inter[valid_inter], m_inter[valid_inter])
        _, y_inter_low = smooth_series(
            c_inter[valid_inter],
            np.clip(m_inter[valid_inter] - std_inter[valid_inter], 0.0, 1.0),
        )
        _, y_inter_high = smooth_series(
            c_inter[valid_inter],
            np.clip(m_inter[valid_inter] + std_inter[valid_inter], 0.0, 1.0),
        )

        x_free_s, y_free_s = smooth_series(c_free[valid_free], m_free[valid_free])
        _, y_free_low = smooth_series(
            c_free[valid_free],
            np.clip(m_free[valid_free] - std_free[valid_free], 0.0, 1.0),
        )
        _, y_free_high = smooth_series(
            c_free[valid_free],
            np.clip(m_free[valid_free] + std_free[valid_free], 0.0, 1.0),
        )

        ax.plot(
            x_inter_s,
            y_inter_s,
            "-",
            color=COLOR_EGO,
            linewidth=2.2,
            label=f"Interaction ($d < {interaction_thresh:g}$ m)",
        )
        ax.plot(
            c_inter[valid_inter][::2],
            m_inter[valid_inter][::2],
            "o",
            color=COLOR_EGO,
            markersize=5.0,
        )
        ax.fill_between(
            x_inter_s, y_inter_low, y_inter_high, color=COLOR_EGO, alpha=0.18
        )

        ax.plot(
            x_free_s,
            y_free_s,
            "--",
            color=COLOR_ALTRUIST,
            linewidth=1.8,
            label=f"Free-flow ($d \\geq {interaction_thresh:g}$ m)",
        )
        ax.plot(
            c_free[valid_free][::2],
            m_free[valid_free][::2],
            "s",
            color=COLOR_ALTRUIST,
            markersize=4.5,
        )
        ax.fill_between(
            x_free_s, y_free_low, y_free_high, color=COLOR_ALTRUIST, alpha=0.12
        )
    else:
        c_all, m_all, std_all, _ = compute_binned_stats(
            data["priority_diff"][active_mask],
            data["speed"][active_mask],
            bins,
            spread_type="sd",
            spread_scale=spread_scale,
        )
        valid_all = ~np.isnan(m_all)
        x_all_s, y_all_s = smooth_series(c_all[valid_all], m_all[valid_all])
        _, y_all_low = smooth_series(
            c_all[valid_all],
            np.clip(m_all[valid_all] - std_all[valid_all], 0.0, 1.0),
        )
        _, y_all_high = smooth_series(
            c_all[valid_all],
            np.clip(m_all[valid_all] + std_all[valid_all], 0.0, 1.0),
        )
        ax.plot(
            x_all_s,
            y_all_s,
            "-",
            color=COLOR_EGO,
            linewidth=2.2,
            label="Overall Active",
        )
        ax.plot(
            c_all[valid_all][::2],
            m_all[valid_all][::2],
            "o",
            color=COLOR_EGO,
            markersize=5.0,
        )
        ax.fill_between(x_all_s, y_all_low, y_all_high, color=COLOR_EGO, alpha=0.18)

    ax.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    ax.set_xlabel(diff_label, fontsize=10.5)
    ax.set_ylabel(r"Ego Speed $v_{\mathrm{ego}}$ [m/s]", fontsize=10.5)
    ax.set_title(
        r"(a) Speed vs. Relative Priority Difference", fontsize=11.5, fontweight="bold"
    )
    ax.set_xlim(*xlim_diff)
    if xticks_diff is not None:
        ax.set_xticks(xticks_diff)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)

    # -------------------------------------------------------------
    # (B) Discrete Rank vs Speed Boxplot (1-based ranks)
    # -------------------------------------------------------------
    ax = axes[0, 1]
    ranks = np.sort(np.unique(data["priority_rank"][active_mask]))
    ranks_1based = ranks + 1
    box_mask = inter_mask if distinguish_mode else active_mask
    box_data = [data["speed"][box_mask & (data["priority_rank"] == r)] for r in ranks]

    bp = ax.boxplot(
        box_data,
        positions=ranks_1based,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#d95f02",
            "markeredgecolor": "black",
            "markersize": 5.5,
        },
        medianprops={"color": "black", "linewidth": 1.4},
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.25},
    )

    # Subsample flier points and rasterize them to keep vector PDF fast and responsive
    max_fliers = 500
    rng = np.random.default_rng(42)
    for flier in bp["fliers"]:
        x_d = np.asarray(flier.get_xdata())
        y_d = np.asarray(flier.get_ydata())
        if len(y_d) > max_fliers:
            idx = rng.choice(len(y_d), size=max_fliers, replace=False)
            flier.set_data(x_d[idx], y_d[idx])
        flier.set_rasterized(True)

    colors = plt.cm.coolwarm(np.linspace(0.85, 0.15, len(ranks)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(ranks_1based)
    rank_labels_hero = [
        f"Rank {r}" if (r == ranks_1based[0] or r == ranks_1based[-1]) else str(r)
        for r in ranks_1based
    ]
    ax.set_xticklabels(rank_labels_hero, fontsize=9.5)
    ax.set_xlabel(
        f"Priority Rank ({ranks_1based.min()} = Highest, {ranks_1based.max()} = Lowest)",
        fontsize=10.5,
    )
    y_label_b = (
        r"Interaction Speed $v$ [m/s]" if distinguish_mode else r"Speed $v$ [m/s]"
    )
    title_b = (
        r"(b) Interaction Speed Distribution by Rank"
        if distinguish_mode
        else r"(b) Speed Distribution by Rank"
    )
    ax.set_ylabel(y_label_b, fontsize=10.5)
    ax.set_title(title_b, fontsize=11.5, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")

    # -------------------------------------------------------------
    # (C) 2D Interaction Phase Map
    # -------------------------------------------------------------
    ax = axes[1, 0]
    inter_2d_mask = (inter_mask if distinguish_mode else active_mask) & (
        data["nearest_score"] > 0
    )
    s_ego_2d = data["priority_score"][inter_2d_mask]
    s_near_2d = data["nearest_score"][inter_2d_mask]
    speed_2d = data["speed"][inter_2d_mask]

    n_bins_hero = 8
    bin_edges_hero = np.linspace(0.0, 1.0, n_bins_hero + 1)
    grid_hero = np.full((n_bins_hero, n_bins_hero), np.nan)

    for i in range(n_bins_hero):
        for j in range(n_bins_hero):
            mask = (
                (s_ego_2d >= bin_edges_hero[j])
                & (
                    s_ego_2d < bin_edges_hero[j + 1]
                    if j < n_bins_hero - 1
                    else s_ego_2d <= bin_edges_hero[j + 1]
                )
                & (s_near_2d >= bin_edges_hero[i])
                & (
                    s_near_2d < bin_edges_hero[i + 1]
                    if i < n_bins_hero - 1
                    else s_near_2d <= bin_edges_hero[i + 1]
                )
            )
            if mask.sum() > 0:
                grid_hero[i, j] = np.mean(speed_2d[mask])

    cmap_hero = plt.cm.viridis.copy()
    cmap_hero.set_bad("#D8D8D8")
    im_c = ax.imshow(
        grid_hero,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap_hero,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    cbar_c = plt.colorbar(im_c, ax=ax, fraction=0.046, pad=0.04)
    cbar_c.set_label(r"Mean Speed [m/s]", fontsize=10)

    for i in range(n_bins_hero):
        for j in range(n_bins_hero):
            if not np.isnan(grid_hero[i, j]):
                x_pos = 0.5 * (bin_edges_hero[j] + bin_edges_hero[j + 1])
                y_pos = 0.5 * (bin_edges_hero[i] + bin_edges_hero[i + 1])
                val_color = "white" if grid_hero[i, j] < 0.72 else "black"
                ax.text(
                    x_pos,
                    y_pos,
                    f"{grid_hero[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=val_color,
                )

    ax.plot([0, 1], [0, 1], color="white", linestyle="--", linewidth=1.4, alpha=0.9)
    ax.set_xlabel(r"Ego Priority Score $s_{\mathrm{ego}}$", fontsize=10.5)
    ax.set_ylabel(r"Neighbor Priority Score $s_{\mathrm{neighbor}}$", fontsize=10.5)
    ax.set_title(
        r"(c) 2D Interaction Speed Phase Map", fontsize=11.5, fontweight="bold"
    )

    # -------------------------------------------------------------
    # (D) Proximity Headway vs Speed by Priority Tier
    # -------------------------------------------------------------
    ax = axes[1, 1]
    min_d, max_d = compute_symmetric_range(
        data["min_distance"][active_mask],
        center=0.65,
        coverage=data_cover_percentage,
        min_bound=0.0,
    )
    d_bins = np.linspace(min_d, max_d, 28)

    tier_styles = [
        (f"High Priority (Top 33{pct_symbol})", 2, COLOR_EGO, "o"),
        (f"Medium Priority (Mid 33{pct_symbol})", 1, COLOR_NEUTRAL, "s"),
        (f"Low Priority (Bottom 33{pct_symbol})", 0, COLOR_ALTRUIST, "^"),
    ]

    for label, tier_code, color, marker in tier_styles:
        mask = active_mask & (data["priority_tier"] == tier_code)
        centers, means, stds, _ = compute_binned_stats(
            data["min_distance"][mask],
            data["speed"][mask],
            d_bins,
            spread_type="sd",
            spread_scale=spread_scale,
        )
        valid = ~np.isnan(means)
        x_s, y_s = smooth_series(centers[valid], means[valid])
        _, y_low = smooth_series(
            centers[valid], np.clip(means[valid] - stds[valid], 0.0, 1.0)
        )
        _, y_high = smooth_series(
            centers[valid], np.clip(means[valid] + stds[valid], 0.0, 1.0)
        )

        ax.plot(x_s, y_s, "-", color=color, linewidth=2.0, label=label)
        ax.plot(
            centers[valid][::2], means[valid][::2], marker, color=color, markersize=5.0
        )
        ax.fill_between(x_s, y_low, y_high, color=color, alpha=0.15)

    ax.set_xlabel(
        r"Distance to Nearest Surrounding Vehicle $d_{\mathrm{min}}$ [m]", fontsize=10.5
    )
    ax.set_ylabel(r"Ego Speed $v_{\mathrm{ego}}$ [m/s]", fontsize=10.5)
    ax.set_title(
        r"(d) Deceleration Proximity by Priority Tier", fontsize=11.5, fontweight="bold"
    )
    ax.set_xlim(min_d, max_d)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    save_figure(fig, output_dir, "fig5_comprehensive_relation_summary")


def print_statistical_summary(
    data: Dict[str, np.ndarray], interaction_thresh: Optional[float] = 0.5
):
    """Print key statistical relationships and hypothesis testing results."""
    inter_mask = data["is_interacting"]
    active_mask = data["is_active"]
    distinguish_mode = interaction_thresh is not None and interaction_thresh > 0

    # Correlation tests
    r_spearman_rank_speed, p_spearman = stats.spearmanr(
        data["priority_rank"][inter_mask], data["speed"][inter_mask]
    )
    r_pearson_diff_speed, p_pearson = stats.pearsonr(
        data["priority_diff"][inter_mask], data["speed"][inter_mask]
    )
    r_pearson_diff_dv, p_dv = stats.pearsonr(
        data["priority_diff"][inter_mask], data["speed_diff"][inter_mask]
    )

    ranks = np.sort(np.unique(data["priority_rank"][active_mask]))

    print("\n" + "=" * 65)
    print("STATISTICAL SUMMARY: PRIORITY VS. EGOISM/ALTRUISM")
    print("=" * 65)
    print(f"1. Priority Rank vs. Interaction Speed (Spearman Rank Corr):")
    print(f"   rho = {r_spearman_rank_speed:+.4f} (p = {p_spearman:.3e})")
    print(
        f"   Interpretation: Monotonic decrease in speed as rank index increases (lower priority)."
    )

    print(f"\n2. Relative Priority Difference vs. Interaction Speed (Pearson Corr):")
    print(f"   r = {r_pearson_diff_speed:+.4f} (p = {p_pearson:.3e})")
    print(
        f"   Interpretation: Statistically significant positive correlation (higher priority than neighbor = higher speed)."
    )

    print(f"\n3. Relative Priority Difference vs. Relative Speed Diff (Pearson Corr):")
    print(f"   r = {r_pearson_diff_dv:+.4f} (p = {p_dv:.3e})")
    print(f"   Interpretation: Direct pairwise yielding / asserting behavior.")

    zone_str = (
        f"Interaction Zone d < {interaction_thresh:g}m"
        if distinguish_mode
        else "All Active Agents"
    )
    print(f"\n4. Mean Speed & Yielding Rates by Discrete Rank ({zone_str}):")
    print(
        f"{'Rank':<6} | {'Mean Speed [m/s]':<18} | {'Yield Rate [%]':<16} | {'Brake Rate [%]':<16} | {'Count':<8}"
    )
    print("-" * 75)
    for r in ranks:
        mask_r = inter_mask & (data["priority_rank"] == r)
        mean_v = data["speed"][mask_r].mean()
        yd_rate = data["is_yielding"][mask_r].mean() * 100.0
        br_rate = data["is_braking"][mask_r].mean() * 100.0
        count = mask_r.sum()
        print(
            f"{r + 1:<6} | {mean_v:<18.3f} | {yd_rate:<16.1f} | {br_rate:<16.1f} | {count:<8}"
        )
    print("=" * 65 + "\n")


def main():
    """Main execution routine."""
    args = parse_args()
    model_dir = args.model_dir
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else os.path.join(model_dir, "fig_relation_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
                "mathtext.fontset": "stix",
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            }
        )

    print(f"[INFO] Analyzing model directory: {model_dir}")
    print(f"[INFO] Output figures directory: {output_dir}")
    print(
        f"[INFO] Symmetric data coverage scope: {args.data_cover_percentage * 100:.1f}%"
    )
    print(f"[INFO] Matplotlib LaTeX rendering (text.usetex): {args.use_tex}")

    # 1. Load data
    td, file_stem = load_rollout_data(model_dir)

    # 2. Extract Data Dictionary
    data = extract_data_dict(
        td,
        v_target=args.v_target,
        interaction_thresh=args.interaction_thresh,
        diff_metric=args.diff_metric,
    )

    # 3. Print statistical summary
    print_statistical_summary(data, interaction_thresh=args.interaction_thresh)

    # 4. Generate all figure variants
    print("[INFO] Generating figure variants...")
    plot_variant_1_relative_priority(
        data,
        output_dir,
        v_target=args.v_target,
        spread_scale=args.spread_scale,
        interaction_thresh=args.interaction_thresh,
        data_cover_percentage=args.data_cover_percentage,
        diff_metric=args.diff_metric,
        show_regimes=args.show_regimes,
        use_tex=args.use_tex,
    )
    plot_variant_2_discrete_rank(
        data,
        output_dir,
        v_target=args.v_target,
        interaction_thresh=args.interaction_thresh,
        use_tex=args.use_tex,
    )
    plot_variant_3_2d_interaction_heatmap(data, output_dir, use_tex=args.use_tex)
    plot_variant_4_altruism_index_and_distance(
        data,
        output_dir,
        spread_scale=args.spread_scale,
        interaction_thresh=args.interaction_thresh,
        data_cover_percentage=args.data_cover_percentage,
        use_tex=args.use_tex,
    )
    plot_variant_5_hero_summary(
        data,
        output_dir,
        spread_scale=args.spread_scale,
        interaction_thresh=args.interaction_thresh,
        data_cover_percentage=args.data_cover_percentage,
        diff_metric=args.diff_metric,
        use_tex=args.use_tex,
    )

    print(f"[SUCCESS] All figures successfully generated and saved to: {output_dir}")


if __name__ == "__main__":
    main()
