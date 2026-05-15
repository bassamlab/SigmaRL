import os
import json
import time

import torch

from sigmarl.helper_training import SaveData, Parameters
from tensordict import TensorDict

from sigmarl.helper_common import is_latex_available, save_video
from sigmarl.mappo_cavs import mappo_cavs

from typing import Callable, Optional, Dict, List
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize, LinearSegmentedColormap, FuncNorm
import numpy as np

MetricFn = Callable[[Path, str, int, int], Optional[float]]


METHOD_STYLE = {
    "CBF (our)": {
        "color": "tab:blue",
        "ls": "-",
        "lw": 1.2,
        "marker": None,
        "ms": 0,
        "mew": 1.0,
        "markevery": None,
    },
    "Distance": {
        "color": "tab:green",
        "ls": "-",
        "lw": 1.2,
        "marker": None,
        "ms": 4.5,
        "mew": 1.0,
        "markevery": 200,
    },
    "TTC": {
        "color": "tab:orange",
        "ls": "-",
        "lw": 1.2,
        "marker": None,
        "ms": 5.0,
        "mew": 1.0,
        "markevery": 200,
    },
}


DEFAULT_STYLE = {"color": "0.25", "ls": "-", "lw": 1.2}  # fallback: dark gray

FRAME_PER_ITERATION = 128 * 32
COLORMAP_VALUE_FONTSIZE = 11
IS_LATEX = is_latex_available()

plt.rcParams.update(
    {
        "font.size": 12,  # slightly smaller for IEEE single-column
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "font.family": "serif",
        "text.usetex": IS_LATEX,
        # PDF export quality
        "pdf.fonttype": 42,  # embed TrueType fonts (better in paper PDFs)
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
)


MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "*",
    "<",
    ">",
]  # enough for <=10 ta values


def _marker_map_for_values(vals: List[float]) -> Dict[float, str]:
    """Assign a distinct marker to each threshold value."""
    vals = sorted(set([float(v) for v in vals]))
    if len(vals) > len(MARKERS):
        raise ValueError(f"Too many unique ta values ({len(vals)}). Add more markers.")
    return {v: MARKERS[i] for i, v in enumerate(vals)}


def _truncate_colormap(cmap, minval: float = 0.2, maxval: float = 1.0, n: int = 256):
    """Return a colormap restricted to a subrange of an existing colormap."""
    if not (0.0 <= minval < maxval <= 1.0):
        raise ValueError("Require 0 <= minval < maxval <= 1.")
    return LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}_{minval:.2f}_{maxval:.2f}",
        cmap(np.linspace(minval, maxval, n)),
    )


def save_reward_curves_per_method_sweep_coded(
    threshold_sweep_paths: List[str],
    cbf_h_sweep_paths: List[str],
    out_dir: Path,
    frames_per_iteration: int = 1,
) -> None:
    """
    One figure per method.
      - CBF: color encodes h (Blues), shown as psi_cbf^th.
      - Distance / TTC: color encodes tb (Blues), marker encodes ta.

    Curves are truncated to frames <= 2e6 and xlim is set to [0, 2e6].
    Colormap is truncated to remove the near-white region (lowest 20%).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    max_frames = 2.0e6
    max_T = int(max_frames / float(frames_per_iteration)) + 1  # inclusive endpoint

    # Colormap: remove near-white (lowest 20%)
    base_cmap = plt.get_cmap("Blues")
    cmap = _truncate_colormap(base_cmap, minval=0.2, maxval=1.0)

    # Collect all model dirs and attach parsed params
    entries = []  # {method, curve, tb, ta, h}
    all_dirs = [
        Path(p) for p in (threshold_sweep_paths + cbf_h_sweep_paths) if Path(p).is_dir()
    ]

    for d in all_dirs:
        method_label = _exp_label_from_path(str(d))
        curve = _load_episode_reward_mean_list(d)
        if curve is None:
            continue
        entries.append(
            {
                "method": method_label,
                "curve": curve,
                "tb": _tb_from_path(str(d)),
                "ta": _ta_from_path(str(d)),
                "h": _h_from_path(str(d)),
            }
        )

    methods = sorted({e["method"] for e in entries}, key=_method_sort_key)

    for method in methods:
        method_entries = [e for e in entries if e["method"] == method]
        if len(method_entries) == 0:
            continue

        fig, ax = plt.subplots(figsize=(4.2, 2.2), constrained_layout=True)
        style_base = METHOD_STYLE.get(method, DEFAULT_STYLE)

        def _x_frames(T: int) -> List[float]:
            """Return frame indices for the truncated curve length."""
            # truncate by max_T
            T = min(T, max_T)
            return [frames_per_iteration * t for t in range(T)]

        # Axis formatting
        ax.set_xlim((0.0, max_frames))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e6:g}"))
        ax.set_xlabel(r"Frames ($\times10^6$)")
        ax.set_ylabel("Episode Reward")
        # ax.set_title(_wrap_method_label(method, is_line_break=False))
        _style_axis(ax)

        is_cbf_family = "CBF" in method

        if is_cbf_family:
            # Color by h, labeled as psi_cbf^th
            h_vals = [e["h"] for e in method_entries if e["h"] is not None]
            if len(h_vals) == 0:
                for e in method_entries:
                    y = _smooth_curve_ema(e["curve"][:max_T], alpha=0.03)
                    x = _x_frames(len(y))
                    ax.plot(
                        x,
                        y,
                        color=style_base["color"],
                        linestyle=style_base["ls"],
                        linewidth=style_base["lw"],
                    )
            else:
                h_min, h_max = float(min(h_vals)), float(max(h_vals))
                norm = Normalize(vmin=h_min, vmax=h_max)

                for e in method_entries:
                    y = _smooth_curve_ema(e["curve"][:max_T], alpha=0.03)
                    x = _x_frames(len(y))
                    h = float(e["h"]) if e["h"] is not None else h_min
                    ax.plot(
                        x,
                        y,
                        color=cmap(norm(h)),
                        linestyle=style_base["ls"],
                        linewidth=style_base["lw"],
                    )

                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.01)
                cbar.set_label(r"CBF Const. Thr. $\psi_\mathrm{cbf}^\mathrm{th}$")
                cbar.set_ticks([round(h_min, 2), round(h_max, 2)])

        else:
            # Color by tb, marker by ta
            tb_vals = [e["tb"] for e in method_entries if e["tb"] is not None]
            ta_vals = [e["ta"] for e in method_entries if e["ta"] is not None]

            if len(tb_vals) == 0 or len(ta_vals) == 0:
                for e in method_entries:
                    y = _smooth_curve_ema(e["curve"][:max_T], alpha=0.03)
                    x = _x_frames(len(y))
                    ax.plot(
                        x, y, color=style_base["color"], linestyle="-", linewidth=1.2
                    )
            else:
                tb_min, tb_max = float(min(tb_vals)), float(max(tb_vals))
                norm = Normalize(vmin=tb_min, vmax=tb_max)

                marker_map = _marker_map_for_values([float(v) for v in ta_vals])

                for e in method_entries:
                    y = _smooth_curve_ema(e["curve"][:max_T], alpha=0.03)
                    x = _x_frames(len(y))

                    tb = float(e["tb"]) if e["tb"] is not None else tb_min
                    ta = (
                        float(e["ta"])
                        if e["ta"] is not None
                        else sorted(marker_map.keys())[0]
                    )

                    ax.plot(
                        x,
                        y,
                        color=cmap(norm(tb)),
                        linestyle="-",
                        linewidth=1.2,
                        marker=marker_map[ta],
                        markersize=4.5,
                        markeredgewidth=0.8,
                        markevery=200,  # readable density
                    )

                # Colorbar label: tb as road distance threshold
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.01)
                cbar.set_label(r"Road Dist. Thr. $d_\mathrm{road}^\mathrm{th}$")
                cbar.set_ticks([round(tb_min, 4), round(tb_max, 4)])

                # Marker legend: ta label depends on method type
                handles, labels = [], []
                for ta in sorted(marker_map.keys()):
                    mk = marker_map[ta]
                    hdl = ax.plot(
                        [], [], color="0.2", marker=mk, linestyle="None", markersize=6.0
                    )[
                        0
                    ]  # TODO remove
                    # hdl = ax.plot(
                    #     [], [], color="0.2", marker=mk, linestyle="None",
                    #     markersize=6.0, markerfacecolor="none", markeredgewidth=0.9
                    # )[0]
                    handles.append(hdl)
                    labels.append(f"{ta:g}")

                if "distance" in method.lower():
                    legend_title = r"$d_\mathrm{veh}^\mathrm{th}$"
                else:
                    legend_title = r"$t_\mathrm{ttc}^\mathrm{th}$"

                leg = ax.legend(
                    handles,
                    labels,
                    title=legend_title,
                    loc="lower left",  # avoids covering early curves
                    frameon=True,
                    framealpha=0.6,
                    borderpad=0.6,
                    labelspacing=0.5,
                    handletextpad=0.6,
                )
                leg.get_title().set_fontstyle("italic")
                for t in leg.get_texts():
                    t.set_fontstyle("italic")

        # Save
        out_pdf = out_dir / (
            "fig_training_reward_curve_"
            + method.replace(" ", "_")
            .replace("+", "plus")
            .replace("(", "")
            .replace(")", "")
            + ".pdf"
        )
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        print(f"[INFO] Saved: {out_pdf}")


def _robust_stats_from_values(vals: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for finite values."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "q10": float("nan"),
            "q50": float("nan"),
            "q90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": float(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _sobol_from_grid(z: np.ndarray) -> Dict[str, float]:
    """
    Variance-based sensitivity indices computed on a discrete grid z[ta_idx, tb_idx],
    assuming uniform distribution over the finite grid points.

    Returns first-order S_tb, S_ta, interaction S_int, and total-effect T_tb, T_ta.
    """
    z = np.asarray(z, dtype=float)

    # Require at least some finite cells
    finite = np.isfinite(z)
    if not np.any(finite):
        return {
            "V": float("nan"),
            "S_tb": float("nan"),
            "S_ta": float("nan"),
            "S_int": float("nan"),
            "T_tb": float("nan"),
            "T_ta": float("nan"),
        }

    # Overall variance over all finite cells
    z_vals = z[finite]
    V = float(np.var(z_vals, ddof=0))
    if V <= 1e-12:
        return {
            "V": V,
            "S_tb": 0.0,
            "S_ta": 0.0,
            "S_int": 0.0,
            "T_tb": 0.0,
            "T_ta": 0.0,
        }

    # Conditional means: E[Y|tb] and E[Y|ta]
    m_tb = np.nanmean(z, axis=0)  # over ta, shape [n_tb]
    m_ta = np.nanmean(z, axis=1)  # over tb, shape [n_ta]

    V_tb = (
        float(np.var(m_tb[np.isfinite(m_tb)], ddof=0))
        if np.any(np.isfinite(m_tb))
        else 0.0
    )
    V_ta = (
        float(np.var(m_ta[np.isfinite(m_ta)], ddof=0))
        if np.any(np.isfinite(m_ta))
        else 0.0
    )

    # Interaction variance (clip to avoid small negative due to missing cells / numerics)
    V_int = max(0.0, V - V_tb - V_ta)

    S_tb = V_tb / V
    S_ta = V_ta / V
    S_int = V_int / V

    # Two-input identity: T_tb = 1 - S_ta, T_ta = 1 - S_tb
    T_tb = 1.0 - S_ta
    T_ta = 1.0 - S_tb

    return {
        "V": V,
        "S_tb": S_tb,
        "S_ta": S_ta,
        "S_int": S_int,
        "T_tb": T_tb,
        "T_ta": T_ta,
    }


def _print_robustness_report_2d(method_label: str, p_val: float, z: np.ndarray) -> None:
    """Print robustness statistics for a two-dimensional threshold sweep."""
    stats = _robust_stats_from_values(z[np.isfinite(z)])
    sob = _sobol_from_grid(z)

    print("-" * 80)
    print(f"[ROBUSTNESS] Method={method_label}, p={p_val:g} (grid over t_a x t_b)")
    print(
        f"  Count={int(stats['count'])}, Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, "
        f"Q10={stats['q10']:.3f}, Median={stats['q50']:.3f}, Q90={stats['q90']:.3f}, "
        f"Min={stats['min']:.3f}, Max={stats['max']:.3f}"
    )
    print(
        f"  Var(V)={sob['V']:.6f}, "
        f"S_tb={sob['S_tb']:.3f}, S_ta={sob['S_ta']:.3f}, S_int={sob['S_int']:.3f}, "
        f"T_tb={sob['T_tb']:.3f}, T_ta={sob['T_ta']:.3f}"
    )


def _print_robustness_report_1d(
    tag: str, x_vals: List[float], y_vals: List[float]
) -> None:
    """Print robustness statistics for a one-dimensional sweep."""
    y = np.asarray(y_vals, dtype=float)
    stats = _robust_stats_from_values(y[np.isfinite(y)])
    print("-" * 80)
    print(f"[ROBUSTNESS] {tag} (1D sweep)")
    print(
        f"  Count={int(stats['count'])}, Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, "
        f"Q10={stats['q10']:.3f}, Median={stats['q50']:.3f}, Q90={stats['q90']:.3f}, "
        f"Min={stats['min']:.3f}, Max={stats['max']:.3f}"
    )


def make_reward_diverging_norm(vmin: float, vmax: float):
    """Create the nonlinear diverging normalization used for reward heatmaps."""
    # vmin should be < 0, vmax > 0 for best effect
    def _forward(x):
        """Map reward values to normalized color positions."""
        result = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        result[pos] = 0.5 + 0.5 * np.clip(x[pos], 0, 10) / 10.0
        log_min = np.log1p(abs(vmin))
        result[neg] = 0.5 - 0.5 * np.log1p(np.abs(np.clip(x[neg], vmin, 0))) / log_min
        return result

    def _inverse(x):
        """Map normalized color positions back to reward values."""
        result = np.empty_like(x, dtype=float)
        pos = x >= 0.5
        neg = ~pos
        result[pos] = (x[pos] - 0.5) * 2.0 * 10.0
        log_min = np.log1p(abs(vmin))
        result[neg] = -(np.expm1((0.5 - x[neg]) * 2.0 * log_min))
        return result

    return FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)


def save_sensitivity_colormap_1d_pdf_metric(
    z: np.ndarray,
    x_vals: List[float],
    out_pdf: Path,
    title: str,
    xlabel: str,
    cbar_label: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "Blues",
    norm=None,
    fmt: str = "{:.1f}",
    is_highlight_best_value: bool = True,
    is_higher_better: bool = True,
) -> None:
    """
    Save a 1D sensitivity colormap as a 1xK heatmap.

    Args:
        z: shape [1, len(x_vals)].
        x_vals: x-axis tick values.
        out_pdf: output pdf path.
        title: figure title.
        xlabel: x-axis label.
        cbar_label: colorbar label.
        vmin/vmax: used when norm is None.
        cmap_name: matplotlib colormap name.
        norm: optional matplotlib norm (overrides vmin/vmax).
        fmt: annotation format string.
    """
    if z.ndim != 2 or z.shape[0] != 1 or z.shape[1] != len(x_vals):
        raise ValueError(f"Expected z shape [1, {len(x_vals)}], got {z.shape}")

    fig, ax = plt.subplots(figsize=(4.2, 2.0), constrained_layout=True)

    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_over("white")
    cmap.set_under("white")

    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    _annotate_heatmap_values(
        ax=ax,
        z=z,
        im=im,
        fmt=fmt,
        fontsize=COLORMAP_VALUE_FONTSIZE,
        is_highlight_best_value=is_highlight_best_value,
        is_higher_better=is_higher_better,
    )

    ax.set_xticks(list(range(len(x_vals))))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])

    # Hide the dummy y axis (single row heatmap)
    ax.set_yticks([0])
    ax.set_yticklabels([""])

    ax.set_xlabel(xlabel)
    # ax.set_title(title)

    _style_axis(ax)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
    cbar.set_label(cbar_label)
    cbar.set_ticks([round(vmin, 1), 0.0, round(vmax, 1)])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_sensitivity_colormap_pdf(
    z: np.ndarray,
    tb_vals: List[float],
    ta_vals: List[float],
    out_pdf: Path,
    title: str,
    cbar_label: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "Blues",
    method_label: str = "",
    norm=None,  # NEW
    fmt: str = "{:.1f}",  # NEW
    is_highlight_best_value: bool = True,
    is_higher_better: bool = True,
) -> None:
    """
    Save a tb (x) vs ta (y) colormap.

    z shape: [len(ta_vals), len(tb_vals)]  (rows=ta, cols=tb)
    """
    fig, ax = plt.subplots(figsize=(4.2, 2.0), constrained_layout=True)

    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_over("white")
    cmap.set_under("white")

    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    _annotate_heatmap_values(
        ax=ax,
        z=z,
        im=im,
        fmt=fmt,
        fontsize=COLORMAP_VALUE_FONTSIZE,
        is_highlight_best_value=is_highlight_best_value,
        is_higher_better=is_higher_better,
    )

    ax.set_xticks(list(range(len(tb_vals))))
    ax.set_yticks(list(range(len(ta_vals))))
    ax.set_xticklabels([f"{v:g}" for v in tb_vals])
    ax.set_yticklabels([f"{v:g}" for v in ta_vals])

    ax.set_xlabel(r"Road Distance Threshold $d_\mathrm{road}^\mathrm{th}$")
    if "distance" in method_label.lower():
        ax.set_ylabel(r"Veh. Dist. Thr. $d_\mathrm{veh}^\mathrm{th}$")
    elif "ttc" in method_label.lower():
        ax.set_ylabel(r"TTC Threshold $t_\mathrm{ttc}^\mathrm{th}$")

    # ax.set_title(title)

    _style_axis(ax)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
    cbar.set_label(cbar_label)
    cbar.set_ticks([round(vmin, 1), 0.0, round(vmax, 1)])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def get_name_suffix(scenario_type: str, n_agents: int, random_seed: int) -> str:
    """Build the filename suffix for one evaluation rollout."""
    return f"{scenario_type}_nagents{n_agents}_seed{random_seed}"


def _load_episode_reward_mean_list(model_dir: Path) -> Optional[List[float]]:
    """
    Load episode_reward_mean_list from reward*.json in a given training folder.
    Returns None if not found or malformed.
    """
    json_files = sorted(model_dir.glob("reward*.json"))
    if len(json_files) == 0:
        return None

    # Use the first match; adjust if you want a specific naming rule
    path = json_files[0]
    try:
        with open(path, "r") as f:
            data = json.load(f)
        lst = data.get("episode_reward_mean_list", None)
        if not isinstance(lst, list) or len(lst) == 0:
            return None
        return [float(x) for x in lst]
    except Exception:
        return None


def _wrap_method_label(s: str, is_line_break=True) -> str:
    """
    Insert line breaks to shorten long method names for x-axis tick labels.
    """
    if s == "CBF (our)":
        return "CBF\n(our)" if is_line_break else "CBF (our)"
    if s == "TTC":
        return "TTC"
    if s == "Distance":
        return "Distance"
    return s


def _reward_method_from_path(p: str) -> str:
    """
    Extract reward method from a path containing '/rew_method_cbf', etc.
    Returns 'unknown' if not found.
    """
    m = re.search(r"/rew_method_([^/]+)", p)
    return m.group(1) if m else "unknown"


def _reward_method_label(m: str) -> str:
    """Return the plot label for a reward method key."""
    label_map = {
        "cbf": "CBF (our)",
        "distance": "Distance",
        "ttc": "TTC",
    }
    return label_map.get(m, m)


def _method_sort_key(label: str) -> int:
    """Return a stable ordering key for the supported reward methods."""
    priority = {
        "CBF (our)": 0,
        "Distance": 10,
        "TTC": 20,
    }
    return priority.get(label, 10**6)


def _exp_label_from_path(p: str) -> str:
    """
    New experiment label: reward method only, extracted from '/rew_method_*'.
    """
    m = _reward_method_from_path(p)
    return _reward_method_label(m)


def _smooth_curve_ema(y: List[float], alpha: float) -> List[float]:
    """
    Exponential moving average smoothing.
    alpha in (0, 1]. Smaller alpha => more smoothing.
    """
    if len(y) == 0:
        return y
    out = [float(y[0])]
    for t in range(1, len(y)):
        out.append(alpha * float(y[t]) + (1.0 - alpha) * out[-1])
    return out


def trim_td(out_td: TensorDict, keys_to_keep=None) -> TensorDict:
    """
    Trim a TensorDict to keep only selected fields.

    Args:
        out_td: Original TensorDict.

    Returns:
        A new TensorDict containing only the specified keys.
    """
    if keys_to_keep is None:
        keys_to_keep = [
            ("agents", "info", "pos"),
            ("agents", "info", "rot"),
            ("agents", "info", "vel"),
            ("agents", "info", "ref"),
            ("agents", "info", "ref_lanelet_ids"),
            ("agents", "info", "is_collision_with_agents"),
            ("agents", "info", "is_collision_with_lanelets"),
            ("agents", "info", "is_reach_goal"),
            ("agents", "info", "rew_reach_goal"),
            ("agents", "info", "rew_collide_other_agents"),
            ("agents", "info", "rew_collide_lane"),
            ("agents", "info", "applied_action_vel"),  # speed, torch.Size([B, T, N, 1])
            (
                "agents",
                "info",
                "applied_action_steer",
            ),  # steering, torch.Size([B, T, N, 1])
            ("agents", "info", "nominal_action_vel"),
            ("agents", "info", "nominal_action_steer"),
        ]

    trimmed = TensorDict(
        {},
        batch_size=out_td.batch_size,
        device=out_td.device,
    )

    for key in keys_to_keep:
        trimmed.set(key, out_td.get(key))

    return trimmed


def run_evaluations():
    """Evaluate all configured model folders and save rollout artifacts."""
    # Each element of path_list is a folder that contains one or more .pth files
    model_folders = [p for p in path_list if os.path.isdir(p)]

    # Optional: progress accounting (approx)
    total_models = len(model_folders)

    total_evaluations = total_models * len(scenario_specs) * len(random_seed_list)
    current_evaluation = 0

    current_evaluation = 0
    for train_path in model_folders:
        # Find config json (you used reward*.json). Keep your logic.
        path_to_json_file: Optional[str] = None
        for file in os.listdir(train_path):
            if file.endswith(".json") and file.startswith("reward"):
                path_to_json_file = os.path.join(train_path, file)
                break
        if path_to_json_file is None:
            print(f"[WARNING] No config JSON found in {train_path}, skipping.")
            continue

        with open(path_to_json_file, "r") as file:
            data = json.load(file)

        # Enumerate all .pth models under this folder
        pth_files = sorted([f for f in os.listdir(train_path) if f.endswith(".pth")])
        if len(pth_files) == 0:
            print(f"[WARNING] No .pth files found in {train_path}, skipping.")
            continue

        for scenario_type, n_agents in scenario_specs:
            for random_seed in random_seed_list:
                current_evaluation += 1
                print("=" * 80)
                print(
                    f"[INFO] Evaluation {current_evaluation}/{total_evaluations}: {train_path}"
                )
                print("=" * 80)

                # Make a fresh copy of parameters per run (avoid state carry-over)
                parameters: Parameters = SaveData.from_dict(data).parameters

                parameters.is_testing_mode = True
                parameters.is_real_time_rendering = False
                parameters.is_save_eval_results = True
                parameters.is_load_model = True

                parameters.is_using_cbf_training = False
                parameters.is_using_cbf_testing = True
                parameters.is_solve_qp = True
                parameters.is_apply_cbf_action = "do_not_apply" not in train_path

                parameters.is_load_final_model = "final" in train_path
                parameters.is_load_out_td = False
                parameters.max_steps = 600
                parameters.num_vmas_envs = 1

                parameters.scenario_type = scenario_type
                parameters.n_agents = n_agents

                parameters.is_save_simulation_video = (
                    True if random_seed == random_seed_list[0] else False
                )  # Only save video for the first seed to save disk space
                parameters.is_visualize_short_term_path = True
                parameters.is_visualize_lane_boundary = False
                parameters.is_visualize_extra_info = True
                parameters.random_seed = random_seed
                parameters.is_continue_train = False
                parameters.lane_width = 0.25

                # Save into the same folder that holds this .pth
                parameters.where_to_save = train_path

                name_suffix = get_name_suffix(scenario_type, n_agents, random_seed)
                out_td_filename = f"out_td_{name_suffix}.td"
                video_basename = f"video_{name_suffix}"
                task_performance_filename = f"task_performance_{name_suffix}.json"
                task_performance_path = os.path.join(
                    train_path, task_performance_filename
                )
                out_td_path = os.path.join(train_path, out_td_filename)

                if os.path.exists(out_td_path) and os.path.exists(
                    task_performance_path
                ):
                    print(f"[INFO] Skipping existing output: {train_path}")
                    continue

                (
                    env,
                    decision_making_module,
                    optimization_module,
                    priority_module,
                    cbf_controllers,
                    parameters,
                ) = mappo_cavs(parameters=parameters)

                t_0 = time.time()
                out_td, frame_list = env.rollout(
                    max_steps=parameters.max_steps - 1,
                    policy=decision_making_module.policy,
                    priority_module=priority_module,
                    cbf_controllers=cbf_controllers,
                    callback=lambda env, _: env.render(
                        mode="rgb_array", visualize_when_rgb=False
                    )
                    if parameters.is_save_simulation_video
                    else None,  # Mode can be "human" or "rgb_array". Use "rgb_array" for headless evaluation.
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    is_save_simulation_video=parameters.is_save_simulation_video,
                )
                print(f"[INFO] Simulation time: {time.time() - t_0:.2f} seconds")

                task_performance_filename = f"task_performance_{name_suffix}.json"
                env.scenario_name.num_task_tries  # example value: tensor([11], dtype=torch.int32)
                env.scenario_name.task_success_times  # example value: tensor([3], dtype=torch.int32)

                # Pack into a dictionary
                task_performance_data = {
                    "num_task_tries": int(env.scenario_name.num_task_tries.item()),
                    "task_success_times": int(
                        env.scenario_name.task_success_times.item()
                    ),
                }

                # Save to JSON
                with open(task_performance_path, "w") as f:
                    json.dump(task_performance_data, f, indent=4)

                print(f"[INFO] Saved task performance to {task_performance_path}")

                if parameters.is_save_eval_results:
                    is_trim_td = True
                    if is_trim_td:
                        out_td = trim_td(out_td)

                    torch.save(out_td, out_td_path)
                    print(f"[INFO] Simulation outputs saved to {out_td_path}")

                    if parameters.is_save_simulation_video:
                        save_video(
                            os.path.join(train_path, video_basename),
                            frame_list,
                            fps=1 / parameters.dt,
                            fmt="mp4",
                            quality="low",
                        )
                        print(
                            f"[INFO] Video saved to {os.path.join(train_path, video_basename)}.mp4"
                        )


def _load_out_td(out_td_path: Path):
    """Load a saved rollout TensorDict from disk."""
    return torch.load(out_td_path, weights_only=False)


def compute_total_reward(out_td, is_consider_driving_comfort: bool = True) -> float:
    """
    Return total event reward plus an optional driving-comfort penalty.
    """
    # -----------------------------
    # Event-based reward (existing)
    # -----------------------------
    key_reach = ("agents", "info", "rew_reach_goal")
    key_coll_agents = ("agents", "info", "rew_collide_other_agents")
    key_coll_lane = ("agents", "info", "rew_collide_lane")

    r_reach = out_td.get(key_reach).float()
    r_coll_agents = out_td.get(key_coll_agents).float()
    r_coll_lane = out_td.get(key_coll_lane).float()

    r_event_sum = (r_reach + r_coll_agents + r_coll_lane).sum()

    if not is_consider_driving_comfort:
        return float(r_event_sum.item())

    # -----------------------------
    # Comfort penalty (new)
    # -----------------------------
    dt = 0.1  # seconds

    key_speed = ("agents", "info", "applied_action_vel")  # speed [m/s], shape [B,T,N,1]
    key_steer = (
        "agents",
        "info",
        "applied_action_steer",
    )  # steering angle [rad], shape [B,T,N,1]

    v = out_td.get(key_speed).float()
    delta = out_td.get(key_steer).float()

    # First differences (pad t=0 with zeros)
    dv = v[:, 1:] - v[:, :-1]  # [B,T-1,N,1]
    ddelta = delta[:, 1:] - delta[:, :-1]  # [B,T-1,N,1]

    a = dv / dt  # [B,T-1,N,1]  [m/s^2]
    steer_rate = ddelta / dt  # [B,T-1,N,1]  [rad/s]

    zero = torch.zeros_like(v[:, :1])  # [B,1,N,1]
    a = torch.cat([zero, a], dim=1)  # [B,T,N,1]
    steer_rate = torch.cat([zero, steer_rate], dim=1)

    # Second differences for jerk / steering-rate jerk
    da = a[:, 1:] - a[:, :-1]
    dsteer_rate = steer_rate[:, 1:] - steer_rate[:, :-1]

    jerk = da / dt  # [B,T-1,N,1]  [m/s^3]
    steer_rate_jerk = dsteer_rate / dt  # [B,T-1,N,1]  [rad/s^2]

    jerk = torch.cat([zero, jerk], dim=1)  # [B,T,N,1]
    steer_rate_jerk = torch.cat([zero, steer_rate_jerk], dim=1)

    # Normalizing constants (reasonable defaults)
    a_norm = 3.0  # [m/s^2]
    jerk_norm = 20.0  # [m/s^3]

    w_comf = 0.2
    r_a = -w_comf * (a.abs() / a_norm).pow(2).mean()
    r_jerk = -w_comf * (jerk.abs() / jerk_norm).pow(2).mean()
    r_comf = r_a + r_jerk
    # print(f"r_a: {r_a:.2f}, r_jerk: {r_jerk:.2f}, r_comf: {r_comf:.2f}")

    return float((r_event_sum + r_comf).item())


def compute_cbf_activation_percentage(out_td) -> float:
    """
    Compute the mean absolute difference between nominal and applied actions,
    averaged over all batches, time steps, agents, and action dimensions (vel, steer).

    Expected shapes (for each key):
        [B, T, N, 1]
    """
    key_applied_vel = ("agents", "info", "applied_action_vel")
    key_applied_steer = ("agents", "info", "applied_action_steer")
    key_nominal_vel = ("agents", "info", "nominal_action_vel")
    key_nominal_steer = ("agents", "info", "nominal_action_steer")

    applied_vel = out_td.get(key_applied_vel).float()
    applied_steer = out_td.get(key_applied_steer).float()
    nominal_vel = out_td.get(key_nominal_vel).float()
    nominal_steer = out_td.get(key_nominal_steer).float()

    # Differences: [B, T, N, 1]
    d_vel = applied_vel - nominal_vel
    d_steer = applied_steer - nominal_steer

    # Mean absolute difference over all dims (B, T, N, 1) and over both channels (vel+steer)
    mean_abs_diff = 0.5 * (d_vel.abs().mean() + d_steer.abs().mean())

    applied_vel_abs_mean = applied_vel.abs().mean()
    applied_steer_abs_mean = applied_steer.abs().mean()
    mean_abs_diff_normalized = 0.5 * (
        d_vel.abs().mean() / (applied_vel_abs_mean + 1e-9)
        + d_steer.abs().mean() / (applied_steer_abs_mean + 1e-9)
    )

    # Activation count: any channel differs (with tolerance)
    eps = 1e-9
    activated = (d_vel.abs() > eps) | (d_steer.abs() > eps)  # [B, T, N, 1]
    n_activations = int(activated.sum().item())
    total_elements = activated.numel()

    return float(100 * mean_abs_diff_normalized.item()), float(
        100 * n_activations / total_elements
    )


def _load_task_performance(task_perf_path: Path) -> Optional[Dict[str, int]]:
    """
    Load task performance JSON:
      {"num_task_tries": int, "task_success_times": int}
    Returns None if not found or malformed.
    """
    if not task_perf_path.exists():
        return None
    try:
        with open(task_perf_path, "r") as f:
            d = json.load(f)
        a = d.get("num_task_tries", None)
        b = d.get("task_success_times", None)
        if a is None or b is None:
            return None
        return {"num_task_tries": int(a), "task_success_times": int(b)}
    except Exception:
        return None


# -----------------------------
# Plotting
# -----------------------------
def _style_axis(ax: plt.Axes) -> None:
    """Apply the shared publication plot axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", linewidth=0.6, alpha=0.40)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.15)

    # Start from a clean tick state
    ax.minorticks_off()

    # Tick direction "in" for both major and minor
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6)


# -----------------------------
# Distance/TTC threshold sensitivity helpers
# -----------------------------
def _p_from_path(p: str) -> Optional[float]:
    """Extract the p sweep value from a model path."""
    m = re.search(r"/p(-?[0-9]+(?:\.[0-9]+)?)/", p)
    return float(m.group(1)) if m else None


def _tb_from_path(p: str) -> Optional[float]:
    """Extract the road-distance threshold tb from a model path."""
    m = re.search(r"/tb(-?[0-9]+(?:\.[0-9]+)?)/", p)
    return float(m.group(1)) if m else None


def _ta_from_path(p: str) -> Optional[float]:
    """Extract the auxiliary threshold ta from a model path."""
    m = re.search(r"/ta(-?[0-9]+(?:\.[0-9]+)?)/", p)
    return float(m.group(1)) if m else None


def _annotate_heatmap_values(
    ax: plt.Axes,
    z: np.ndarray,
    im,
    fmt: str = "{:.1f}",
    fontsize: int = 8,
    is_highlight_best_value: bool = True,
    is_higher_better: bool = True,
) -> None:
    """
    Annotate each cell with its numeric value.
    z: array shown by imshow (same shape).
    Skips NaNs.
    Text color adapts using the colormap normalization.
    """
    if z.size == 0:
        return

    # Use the mid-point of the current color scale to decide text color
    vmin, vmax = im.get_clim()
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = np.nanmin(z)
        vmax = np.nanmax(z)
    thresh = 0.5 * (vmin + vmax)

    best_mask = np.zeros_like(z, dtype=bool)
    finite = np.isfinite(z)
    if is_highlight_best_value and np.any(finite):
        best_value = np.nanmax(z) if is_higher_better else np.nanmin(z)
        best_mask = finite & np.isclose(z, best_value, rtol=1e-9, atol=1e-12)

    nrows, ncols = z.shape
    for i in range(nrows):
        for j in range(ncols):
            val = z[i, j]
            if not np.isfinite(val):
                continue

            color = "white" if (val > 0.5 * vmax or val < 0.5 * vmin) else "black"
            is_best = bool(best_mask[i, j])
            if is_best:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.8,
                        zorder=3,
                    )
                )
            ax.text(
                j,
                i,
                fmt.format(val),
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
                fontweight="bold" if is_best else "normal",
                clip_on=True,
                bbox=dict(facecolor="none", edgecolor="none", alpha=0.35, pad=0.15),
            )


def _is_higher_better_metric(metric_tag: str) -> bool:
    """
    Decide the optimization direction for metric colormaps.

    Reward, success, and throughput-style metrics are maximized. Cost/risk/
    intervention metrics are minimized.
    """
    tag = metric_tag.lower()
    lower_is_better_tokens = (
        "collision",
        "crash",
        "failure",
        "violation",
        "error",
        "loss",
        "cost",
        "cbf_activation",
    )
    return not any(token in tag for token in lower_is_better_tokens)


def plot_threshold_sweep_colormaps_metric(
    path_list: List[str],
    scenario_specs: List[tuple],
    random_seed_list: List[int],
    fig_dir: Path,
    metric_fn: MetricFn,
    metric_tag: str,
    cbar_label: str,
    fmt: str = "{:.1f}",
    only_p_values: Optional[List[float]] = None,
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
    cmap_name: str = "Blues",
    norm=None,
    is_highlight_best_value: bool = True,
) -> None:
    """
    Plot metric heatmaps for Distance/TTC threshold sweeps over tb and ta.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[tuple, List[Path]] = {}
    tb_set: Dict[tuple, set] = {}
    ta_set: Dict[tuple, set] = {}

    for s in path_list:
        d = Path(s)
        if not d.is_dir():
            continue

        method_label = _exp_label_from_path(str(d))
        p_val = _p_from_path(str(d))
        tb_val = _tb_from_path(str(d))
        ta_val = _ta_from_path(str(d))
        if p_val is None or tb_val is None or ta_val is None:
            continue
        if only_p_values is not None and (p_val not in set(only_p_values)):
            continue

        key = (method_label, p_val)
        groups.setdefault(key, []).append(d)
        tb_set.setdefault(key, set()).add(tb_val)
        ta_set.setdefault(key, set()).add(ta_val)

    for (method_label, p_val), dirs in sorted(
        groups.items(), key=lambda x: (_method_sort_key(x[0][0]), x[0][1])
    ):
        tb_vals = sorted(tb_set[(method_label, p_val)])
        ta_vals = sorted(ta_set[(method_label, p_val)])
        tb_to_j = {v: j for j, v in enumerate(tb_vals)}
        ta_to_i = {v: i for i, v in enumerate(ta_vals)}

        cell_values: List[List[List[float]]] = [
            [[] for _ in range(len(tb_vals))] for _ in range(len(ta_vals))
        ]

        for d in dirs:
            tb_val = _tb_from_path(str(d))
            ta_val = _ta_from_path(str(d))
            if tb_val is None or ta_val is None:
                continue
            i = ta_to_i[ta_val]
            j = tb_to_j[tb_val]

            for scenario_type, n_agents in scenario_specs:
                for eval_seed in random_seed_list:
                    try:
                        v = metric_fn(d, scenario_type, n_agents, eval_seed)
                        if v is not None and np.isfinite(v):
                            cell_values[i][j].append(float(v))
                    except Exception:
                        continue

        z = np.full((len(ta_vals), len(tb_vals)), np.nan, dtype=float)
        for i in range(len(ta_vals)):
            for j in range(len(tb_vals)):
                vals = cell_values[i][j]
                if len(vals) > 0:
                    z[i, j] = float(np.mean(vals))

        if metric_tag == "total_reward":
            _print_robustness_report_2d(method_label=method_label, p_val=p_val, z=z)

        vmin = global_vmin
        vmax = global_vmax
        if vmin is None or vmax is None:
            finite = np.isfinite(z)
            if np.any(finite):
                vmin, vmax = float(np.nanmin(z)), float(np.nanmax(z))

        out_pdf = (
            fig_dir / f"fig_colormap_{metric_tag.lower()}_{method_label.lower()}.pdf"
        )
        title = f"{_wrap_method_label(method_label, is_line_break=False)} (p={p_val:g})"
        is_higher_better = _is_higher_better_metric(metric_tag)

        save_sensitivity_colormap_pdf(
            z=z,
            tb_vals=tb_vals,
            ta_vals=ta_vals,
            out_pdf=out_pdf,
            title=title,
            cbar_label=cbar_label,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            norm=norm,
            fmt=fmt,
            method_label=method_label,
            is_highlight_best_value=is_highlight_best_value,
            is_higher_better=is_higher_better,
        )

        print(f"[INFO] Saved: {out_pdf}")


def plot_cbf_h_sweep_colormap_metric(
    path_list: List[str],
    scenario_specs: List[tuple],
    random_seed_list: List[int],
    fig_dir: Path,
    metric_fn: MetricFn,
    metric_tag: str,
    cbar_label: str,
    fmt: str = "{:.1f}",
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
    cmap_name: str = "Blues",
    norm=None,
    is_highlight_best_value: bool = True,
) -> None:
    """
    Plot a metric heatmap for the CBF threshold h sweep.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    dirs_by_h: Dict[float, List[Path]] = {}
    for s in path_list:
        d = Path(s)
        if not d.is_dir():
            continue
        if _reward_method_from_path(str(d)) != "cbf":
            continue
        h_val = _h_from_path(str(d))
        if h_val is None:
            continue
        dirs_by_h.setdefault(h_val, []).append(d)

    if len(dirs_by_h) == 0:
        print("[WARNING] No CBF h-folders found.")
        return

    h_vals = sorted(dirs_by_h.keys())
    means = []

    for h in h_vals:
        vals = []
        for d in dirs_by_h[h]:
            for scenario_type, n_agents in scenario_specs:
                for eval_seed in random_seed_list:
                    try:
                        v = metric_fn(d, scenario_type, n_agents, eval_seed)
                        if v is not None and np.isfinite(v):
                            vals.append(float(v))
                    except Exception:
                        continue
        means.append(float(np.mean(vals)) if len(vals) > 0 else float("nan"))

    z = np.asarray(means, dtype=float)[None, :]

    if metric_tag == "total_reward":
        _print_robustness_report_1d(
            tag="CBF total_reward vs h", x_vals=h_vals, y_vals=means
        )

    vmin = global_vmin
    vmax = global_vmax
    if vmin is None or vmax is None:
        finite = np.isfinite(z)
        if np.any(finite):
            vmin, vmax = float(np.nanmin(z)), float(np.nanmax(z))

    out_pdf = fig_dir / f"fig_colormap_{metric_tag}_cbf.pdf"
    title = "CBF (our)"
    is_higher_better = _is_higher_better_metric(metric_tag)

    save_sensitivity_colormap_1d_pdf_metric(
        z=z,
        x_vals=h_vals,
        out_pdf=out_pdf,
        title=title,
        xlabel=r"CBF Constraint Threshold $\psi_\mathrm{cbf}^\mathrm{th}$",
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        cbar_label=cbar_label,
        fmt=fmt,
        norm=norm,
        is_highlight_best_value=is_highlight_best_value,
        is_higher_better=is_higher_better,
    )

    print(f"[INFO] Saved: {out_pdf}")


# -----------------------------
# CBF threshold sensitivity helpers
# -----------------------------


def _h_from_path(p: str) -> Optional[float]:
    """Extract the CBF h threshold from a model path."""
    m = re.search(r"/h(-?[0-9]+(?:\.[0-9]+)?)/", p)
    return float(m.group(1)) if m else None


def metric_total_reward(
    d: Path, scenario_type: str, n_agents: int, eval_seed: int
) -> Optional[float]:
    """Load one rollout and return its total reward metric."""
    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
    out_td_path = d / f"out_td_{name_suffix}.td"
    if not out_td_path.exists():
        return None
    out_td = _load_out_td(out_td_path)
    return float(compute_total_reward(out_td))


def metric_cbf_activation_degree(
    d: Path, scenario_type: str, n_agents: int, eval_seed: int
) -> Optional[float]:
    """Load one rollout and return its normalized CBF activation degree."""
    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
    out_td_path = d / f"out_td_{name_suffix}.td"
    if not out_td_path.exists():
        return None
    out_td = _load_out_td(out_td_path)
    deg, _rate = compute_cbf_activation_percentage(out_td)
    return float(deg)


def metric_cbf_activation_rate(
    d: Path, scenario_type: str, n_agents: int, eval_seed: int
) -> Optional[float]:
    """Load one rollout and return its CBF activation rate."""
    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
    out_td_path = d / f"out_td_{name_suffix}.td"
    if not out_td_path.exists():
        return None
    out_td = _load_out_td(out_td_path)
    _deg, rate = compute_cbf_activation_percentage(out_td)
    return float(rate)


def metric_task_attempts_total(
    d: Path, scenario_type: str, n_agents: int, eval_seed: int
) -> Optional[float]:
    """Load one task-performance file and return the total task attempts."""
    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
    tp = _load_task_performance(d / f"task_performance_{name_suffix}.json")
    if tp is None:
        return None
    return float(tp["num_task_tries"])


def metric_task_success_rate(
    d: Path, scenario_type: str, n_agents: int, eval_seed: int
) -> Optional[float]:
    """Load one task-performance file and return task success percentage."""
    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
    tp = _load_task_performance(d / f"task_performance_{name_suffix}.json")
    if tp is None or tp["num_task_tries"] <= 0:
        return None
    return float(100.0 * tp["task_success_times"] / tp["num_task_tries"])


if __name__ == "__main__":
    is_highlight_best_value = True

    scenario_specs = [("cpm_mixed", 4)]
    print(f"[INFO] Evaluating scenarios: {scenario_specs}")

    policy_parent_folder = "checkpoints/itsc26"

    threshold_sweep_paths = []
    for root, dirs, files in os.walk(policy_parent_folder):
        if any(f.endswith(".pth") for f in files):
            method = _reward_method_from_path(root)
            is_supported_threshold_method = method in {"distance", "ttc"}
            if is_supported_threshold_method:
                threshold_sweep_paths.append(root)

    cbf_h_sweep_paths = []
    for root, dirs, files in os.walk(policy_parent_folder):
        if any(f.endswith(".pth") for f in files):
            is_cbf_h_sweep_folder = (
                (_reward_method_from_path(root) == "cbf")
                and ("reward_progress0.1" in root)
                and ("seed1" in root)
                and ("h0.02" not in root)
            )
            if is_cbf_h_sweep_folder:
                cbf_h_sweep_paths.append(root)

    random_seed_list = [1, 2, 3, 4]

    path_list = list(dict.fromkeys(threshold_sweep_paths + cbf_h_sweep_paths))
    run_evaluations()

    fig_dir = Path(policy_parent_folder) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    reward_vmin = -10.0
    reward_vmax = 10.0
    reward_norm = make_reward_diverging_norm(reward_vmin, reward_vmax)

    plot_threshold_sweep_colormaps_metric(
        path_list=threshold_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_total_reward,
        metric_tag="total_reward",
        cbar_label="Total Reward",
        fmt="{:.1f}",
        only_p_values=[-1.0],
        global_vmin=reward_vmin,
        global_vmax=reward_vmax,
        cmap_name="RdBu",
        norm=reward_norm,
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_cbf_h_sweep_colormap_metric(
        path_list=cbf_h_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_total_reward,
        metric_tag="total_reward",
        cbar_label="Total Reward",
        fmt="{:.1f}",
        global_vmin=reward_vmin,
        global_vmax=reward_vmax,
        cmap_name="RdBu",
        norm=reward_norm,
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_threshold_sweep_colormaps_metric(
        path_list=threshold_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_cbf_activation_degree,
        metric_tag="cbf_activation_degree",
        cbar_label=r"CBF Act. Deg. [\%]",
        fmt="{:.1f}",
        only_p_values=[-1.0],
        global_vmin=0.0,
        global_vmax=100.0,
        cmap_name="Blues",
        norm=None,
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_cbf_h_sweep_colormap_metric(
        path_list=cbf_h_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_cbf_activation_degree,
        metric_tag="cbf_activation_degree",
        cbar_label=r"CBF Act. Deg. [\%]",
        fmt="{:.1f}",
        global_vmin=0.0,
        global_vmax=100.0,
        cmap_name="Blues",
        norm=None,
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_threshold_sweep_colormaps_metric(
        path_list=threshold_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_cbf_activation_rate,
        metric_tag="cbf_activation_rate",
        cbar_label=r"CBF Act. Rate [\%]",
        fmt="{:.1f}",
        only_p_values=[-1.0],
        global_vmin=0.0,
        global_vmax=100.0,
        cmap_name="Blues",
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_cbf_h_sweep_colormap_metric(
        path_list=cbf_h_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_cbf_activation_rate,
        metric_tag="cbf_activation_rate",
        cbar_label=r"CBF Act. Rate [\%]",
        fmt="{:.1f}",
        global_vmin=0.0,
        global_vmax=100.0,
        cmap_name="Blues",
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_threshold_sweep_colormaps_metric(
        path_list=threshold_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_task_success_rate,
        metric_tag="task_success_rate",
        cbar_label=r"Task Success Rate [\%]",
        fmt="{:.1f}",
        only_p_values=[-1.0],
        global_vmin=0.0,
        global_vmax=100.0,
        cmap_name="Blues",
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_cbf_h_sweep_colormap_metric(
        path_list=cbf_h_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_task_success_rate,
        metric_tag="task_success_rate",
        cbar_label=r"Task Success Rate [\%]",
        fmt="{:.1f}",
        global_vmin=0.0,
        global_vmax=100.0,
        cmap_name="Blues",
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_threshold_sweep_colormaps_metric(
        path_list=threshold_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_task_attempts_total,
        metric_tag="task_attempts_total",
        cbar_label=r"\#Task Attempts",
        fmt="{:.0f}",
        only_p_values=[-1.0],
        global_vmin=None,
        global_vmax=None,
        cmap_name="Blues",
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_cbf_h_sweep_colormap_metric(
        path_list=cbf_h_sweep_paths,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_task_attempts_total,
        metric_tag="task_attempts_total",
        cbar_label=r"\#Task Attempts",
        fmt="{:.0f}",
        global_vmin=None,
        global_vmax=None,
        cmap_name="Blues",
        is_highlight_best_value=is_highlight_best_value,
    )

    save_reward_curves_per_method_sweep_coded(
        threshold_sweep_paths=threshold_sweep_paths,
        cbf_h_sweep_paths=cbf_h_sweep_paths,
        out_dir=fig_dir,
        frames_per_iteration=FRAME_PER_ITERATION,
    )
