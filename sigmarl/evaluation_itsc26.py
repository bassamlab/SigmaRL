import os
import json
import time

import torch

from sigmarl.helper_training import SaveData, Parameters, SaveData
from tensordict import TensorDict

from sigmarl.helper_common import is_latex_available, save_video
from sigmarl.mappo_cavs import mappo_cavs

from typing import Optional, Dict, List, Any, Tuple
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import math
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import (
    TwoSlopeNorm,
    SymLogNorm,
    Normalize,
    LinearSegmentedColormap,
    FuncNorm,
)
import numpy as np
from typing import Callable

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
    "CBF+Sparse (our)": {
        "color": "tab:blue",
        "ls": "--",
        "lw": 1.2,
        "marker": None,
        "ms": 0,
        "mew": 1.0,
        "markevery": None,
    },
    "Distance": {  # Distance V1
        "color": "tab:green",
        "ls": "-",
        "lw": 1.2,
        "marker": None,
        "ms": 4.5,
        "mew": 1.0,
        "markevery": 200,
    },
    "Distance+Sparse": {  # Distance V2
        "color": "tab:green",
        "ls": "--",
        "lw": 1.2,
        "marker": None,
        "ms": 4.5,
        "mew": 1.0,
        "markevery": 200,
    },
    "TTC": {  # TTC V1
        "color": "tab:orange",
        "ls": "-",
        "lw": 1.2,
        "marker": None,
        "ms": 5.0,
        "mew": 1.0,
        "markevery": 200,
    },
    "TTC+Sparse": {  # TTC V2
        "color": "tab:orange",
        "ls": "--",
        "lw": 1.2,
        "marker": None,
        "ms": 5.0,
        "mew": 1.0,
        "markevery": 200,
    },
    "Sparse": {
        "color": "#F0E442",
        "ls": "--",
        "lw": 1.4,
        "marker": None,
        "ms": 0,
        "mew": 1.0,
        "markevery": None,
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
        "legend.fontsize": 12,
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
    vals = sorted(set([float(v) for v in vals]))
    if len(vals) > len(MARKERS):
        raise ValueError(f"Too many unique ta values ({len(vals)}). Add more markers.")
    return {v: MARKERS[i] for i, v in enumerate(vals)}


def _truncate_colormap(cmap, minval: float = 0.2, maxval: float = 1.0, n: int = 256):
    if not (0.0 <= minval < maxval <= 1.0):
        raise ValueError("Require 0 <= minval < maxval <= 1.")
    return LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}_{minval:.2f}_{maxval:.2f}",
        cmap(np.linspace(minval, maxval, n)),
    )


def save_reward_curves_per_method_hparam_coded_case23(
    path_list_case2: List[str],
    path_list_case3: List[str],
    out_dir: Path,
    frames_per_iteration: int = 1,
) -> None:
    """
    One figure per method.
      - CBF / CBF+Sparse: color encodes h (Blues), shown as psi_cbf^th.
      - Distance / Distance+Sparse / TTC / TTC+Sparse: color encodes tb (Blues), marker encodes ta.

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
        Path(p) for p in (path_list_case2 + path_list_case3) if Path(p).is_dir()
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

        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
        style_base = METHOD_STYLE.get(method, DEFAULT_STYLE)

        def _x_frames(T: int) -> List[float]:
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
                cbar.set_label(
                    r"CBF Constraint Threshold $\psi_\mathrm{cbf}^\mathrm{th}$"
                )
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
                cbar.set_label(r"Road Distance Threshold $d_\mathrm{road}^\mathrm{th}$")
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
    # vmin should be < 0, vmax > 0 for best effect
    def _forward(x):
        result = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        result[pos] = 0.5 + 0.5 * np.clip(x[pos], 0, 10) / 10.0
        log_min = np.log1p(abs(vmin))
        result[neg] = 0.5 - 0.5 * np.log1p(np.abs(np.clip(x[neg], vmin, 0))) / log_min
        return result

    def _inverse(x):
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

    fig, ax = plt.subplots(figsize=(4.2, 2.2), constrained_layout=True)

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
    fig, ax = plt.subplots(figsize=(4.2, 2.2), constrained_layout=True)

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
        ax.set_ylabel(r"Vehicle Distance Threshold $d_\mathrm{veh}^\mathrm{th}$")
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


def _aggregate_curves(
    curves: List[List[float]],
    smoothing: str = "ema",
    ema_alpha: float = 0.05,
    rolling_window: int = 25,
):
    """
    Align curves by truncating to the minimum length.
    Optionally smooth each curve first, then compute mean and std.

    smoothing: "none" | "ema" | "rolling"
    """
    if len(curves) == 0:
        return None, None, None

    min_len = min(len(c) for c in curves)
    if min_len <= 1:
        return None, None, None

    proc = []
    for c in curves:
        c = [float(v) for v in c[:min_len]]
        if smoothing == "ema":
            c = _smooth_curve_ema(c, alpha=ema_alpha)
        elif smoothing == "rolling":
            c = _smooth_curve_rolling_mean(c, window=rolling_window)
        elif smoothing == "none":
            pass
        else:
            raise ValueError(f"Unknown smoothing={smoothing}")
        proc.append(c)

    arr = torch.tensor(proc, dtype=torch.float32)  # [n_runs, T]
    mean = arr.mean(dim=0)
    std = arr.std(dim=0, unbiased=False) if arr.shape[0] > 1 else torch.zeros_like(mean)
    x = list(range(min_len))

    # If you want standard error of the mean instead of std, uncomment below:
    # sem = std / (arr.shape[0] ** 0.5)
    # return x, mean.tolist(), sem.tolist()

    return x, mean.tolist(), std.tolist()


def _wrap_method_label(s: str, is_line_break=True) -> str:
    """
    Insert line breaks to shorten long method names for x-axis tick labels.
    """
    if s == "CBF (our)":
        return "CBF\n(our)" if is_line_break else "CBF (our)"
    if s == "CBF+Sparse (our)":
        return "CBF\nV2 (our)" if is_line_break else "CBF V2 (our)"
    if s == "Distance+Sparse":
        return "Distance\nV2" if is_line_break else "Distance V2"
    if s == "TTC+Sparse":
        return "TTC\nV2" if is_line_break else "TTC V2"
    if s == "TTC":
        return "TTC\nV1" if is_line_break else "TTC V1"
    if s == "Distance":
        return "Distance\nV1" if is_line_break else "Distance V1"
    return s


def _reward_method_from_path(p: str) -> str:
    """
    Extract reward method from a path containing '/rew_method_cbf', etc.
    Returns 'unknown' if not found.
    """
    m = re.search(r"/rew_method_([^/]+)", p)
    return m.group(1) if m else "unknown"


def _reward_method_label(m: str) -> str:
    label_map = {
        "cbf": "CBF (our)",
        "cbf_sparse": "CBF+Sparse (our)",
        "distance_sparse": "Distance+Sparse",
        "ttc_sparse": "TTC+Sparse",
        "distance": "Distance",
        "sparse": "Sparse",
        "ttc": "TTC",
        "distance_old": "Distance (Old)",
    }
    return label_map.get(m, m)


def _method_sort_key(label: str) -> int:
    priority = {
        "CBF (our)": 0,
        "CBF+Sparse (our)": 10,
        "Distance": 20,
        "Distance+Sparse": 30,
        "TTC": 40,
        "TTC+Sparse": 50,
        "Sparse": 60,
    }
    return priority.get(label, 10**6)


def _seed_from_path(p: str) -> Optional[int]:
    """
    Extract seed index from a path containing '/seed3', etc.
    """
    m = re.search(r"/seed(\d+)", p)
    return int(m.group(1)) if m else None


def _path_sort_key(p: str):
    """
    Primary: reward method (custom order)
    Secondary: reward_progress (ascending, if present)
    Tertiary: seed (ascending)
    """
    method = _reward_method_from_path(p)
    rp = _reward_progress_from_path(p)
    seed = _seed_from_path(p)

    method_priority = {
        "cbf": 0,
        "distance": 1,
        "distance_old": 2,
        "sparse": 3,
        "ttc": 4,
    }
    method_key = method_priority.get(method, 10**6)

    rp_key = rp if rp is not None else float("inf")
    seed_key = seed if seed is not None else 10**9
    return (method_key, rp_key, seed_key)


def _reward_progress_from_path(p: str) -> Optional[float]:
    """
    Extract reward_progress value from a path containing 'reward_progress2.5', etc.
    Returns None if not found.
    """
    m = re.search(r"reward_progress([0-9]+(?:\.[0-9]+)?)", p)
    return float(m.group(1)) if m else None


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


def _smooth_curve_rolling_mean(y: List[float], window: int) -> List[float]:
    """
    Rolling mean with a causal window (uses past values only).
    window >= 1.
    """
    if window <= 1 or len(y) == 0:
        return [float(v) for v in y]
    out = []
    s = 0.0
    q = []
    for v in y:
        v = float(v)
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
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
                parameters.rew_method = "sparse"  # Reward method: {"distance", "cbf_constraint", "cbf_qp", "ttc", "sparse"}
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
    return torch.load(out_td_path, weights_only=False)


def compute_collision_rate_percent(out_td) -> float:
    """
    Return collision rate in percent.

    out_td[("agents","info","is_collision_with_agents")] has shape:
      [B, T, N, 1]   (B=batch_size, T=n_steps, N=n_agents)
    Same for is_collision_with_lanelets.

    Default definition:
      percentage of time steps where at least one agent collides
      (with another agent or with boundaries), averaged over B.
    """
    key_agents = ("agents", "info", "is_collision_with_agents")
    key_lane = ("agents", "info", "is_collision_with_lanelets")

    coll_agents = out_td.get(key_agents).bool()  # [B,T,N,1]
    coll_lane = out_td.get(key_lane).bool()  # [B,T,N,1]
    coll = coll_agents | coll_lane  # [B,T,N,1]

    if COLLISION_MODE == "per_timestep_any_agent":
        # any collision among agents at each time step
        coll_bt = coll.any(dim=(2, 3))  # [B,T]
        return coll_bt.float().mean().item() * 100.0

    if COLLISION_MODE == "per_agent_timestep":
        # mean over all agent-time pairs (and B)
        return coll.float().mean().item() * 100.0  # Usually less than the above method

    raise ValueError(f"Unknown COLLISION_MODE={COLLISION_MODE}")


def compute_average_speed(out_td) -> float:
    """
    Return average speed [m/s] across all agents and all time steps (and envs).
    Uses key:
      ("agents","info","vel") -> [..., 2]
    """
    key_vel = ("agents", "info", "vel")
    vel = out_td.get(key_vel).float()

    # speed = ||v||_2 on the last dim
    speed = torch.linalg.norm(vel, dim=-1)

    # Average over all entries (time * env * agent)
    return speed.mean().item()


def compute_total_reward(out_td) -> float:
    """
    Return the total reward as a Python float.

    Reward keys:
      ("agents","info","rew_reach_goal")
      ("agents","info","rew_collide_other_agents")
      ("agents","info","rew_collide_lane")

    Assumes each tensor is broadcast-compatible (typically [B, T, N, 1]).
    Computes: mean over all entries (B * T * N * ...).
    """
    key_reach = ("agents", "info", "rew_reach_goal")
    key_coll_agents = ("agents", "info", "rew_collide_other_agents")
    key_coll_lane = ("agents", "info", "rew_collide_lane")

    r_reach = out_td.get(key_reach).float()
    r_coll_agents = out_td.get(key_coll_agents).float()
    r_coll_lane = out_td.get(key_coll_lane).float()

    r_total = r_reach + r_coll_agents + r_coll_lane
    return r_total.sum().item()


def compute_total_reward(out_td, is_consider_driving_comfort: bool = True) -> float:
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", linewidth=0.6, alpha=0.40)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.15)

    # Start from a clean tick state
    ax.minorticks_off()

    # Tick direction "in" for both major and minor
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6)


def save_boxplot_pdf(
    data_by_method: Dict[str, List[float]],
    method_order: List[str],
    ylabel: str,
    out_pdf: Path,
    unit: Optional[str] = "",
    ylim: Optional[tuple] = None,
    show_fliers: bool = False,  # default cleaner for paper
    loc_text: str = "center",  # "center" | "right"
    type_text: str = "average",  # "average" | "median"
    is_show_x_ticks: bool = True,
    is_show_x_label: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.2), constrained_layout=True)

    data = [data_by_method.get(m, []) for m in method_order]

    bp = ax.boxplot(
        data,
        showfliers=show_fliers,
        widths=0.55,
        patch_artist=True,
        showmeans=False,
        medianprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )
    iqr_eps = 1e-9  # or something like 1e-6 if your values are floats with noise

    for i, m in enumerate(method_order):
        style = METHOD_STYLE.get(m, DEFAULT_STYLE)

        # boxes
        bp["boxes"][i].set_facecolor(style["color"])
        bp["boxes"][i].set_edgecolor("black")
        bp["boxes"][i].set_linewidth(1.0)

        # whiskers and caps (2 each per box)
        bp["whiskers"][2 * i].set_color("black")
        bp["whiskers"][2 * i + 1].set_color("black")
        bp["caps"][2 * i].set_color("black")
        bp["caps"][2 * i + 1].set_color("black")

        # median color depends on IQR collapse
        vals = data[i]  # same order as method_order
        if vals is None or len(vals) == 0:
            is_collapsed = False
        else:
            arr = np.asarray(vals, dtype=float)
            q1, q3 = np.percentile(arr, [25, 75])
            is_collapsed = abs(q3 - q1) < iqr_eps

        bp["medians"][i].set_color(style["color"] if is_collapsed else "white")
        bp["medians"][i].set_linewidth(0.5)

    if ylim is not None:
        ax.set_ylim(ylim)
    y0, y1 = ax.get_ylim()
    yr = max(y1 - y0, 1e-12)

    pad_y = 0.02 * yr  # keep text inside axes when clamped

    # box width must match bp widths=0.55
    box_width = 0.55
    half_box = 0.5 * box_width
    gap_x = 0.04  # small gap from box edge when loc_text="right"

    if loc_text not in {"center", "right"}:
        raise ValueError(f"Unknown loc_text={loc_text}, must be 'center' or 'right'")
    if type_text not in {"average", "median"}:
        raise ValueError(
            f"Unknown type_text={type_text}, must be 'average' or 'median'"
        )

    for i, vals in enumerate(data, start=1):  # box centers are 1..K
        if vals is None or len(vals) == 0:
            continue

        v = torch.tensor([float(x) for x in vals], dtype=torch.float32)
        stat_val = (
            float(v.mean().item())
            if type_text == "average"
            else float(v.median().item())
        )

        # y position: at the statistic; clamp inside visible range
        y_text = stat_val
        va = "center"
        if y_text > y1:
            y_text = y1 - pad_y
            va = "top"
        elif y_text < y0:
            y_text = y0 + pad_y
            va = "bottom"

        # x position + alignment
        if loc_text == "right":
            x_text = i + half_box + gap_x
            ha = "left"
        else:  # "center"
            x_text = i
            ha = "center"
            # tightly above the statistic (unless clamped)
            if y0 < stat_val < y1:
                y_text = min(stat_val + pad_y, y1 - pad_y)
                va = "bottom"

        # format with optional unit
        if unit is None or unit == "":
            text_str = f"{stat_val:.2f}"
        else:
            if IS_LATEX and unit == "%":
                text_str = f"{stat_val:.2f}\\%"
            else:
                text_str = f"{stat_val:.2f}{unit}"

        ax.text(
            x_text,
            y_text,
            text_str,
            color="tab:red",
            ha=ha,
            va=va,
            clip_on=True,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0.15),
        )

    ax.set_ylabel(ylabel)

    # X label on/off
    if is_show_x_label:
        ax.set_xlabel("Reward Methods")
    else:
        ax.set_xlabel("")

    if is_show_x_ticks:
        ax.set_xticks(list(range(1, len(method_order) + 1)))

        if IS_LATEX:
            # LaTeX path: make labels italic via LaTeX, and replace '\n' with '\\' using \shortstack
            def _tex_italic_multiline(s: str) -> str:
                parts = _wrap_method_label(s).split("\n")
                # italicize each line explicitly
                parts = [rf"\textit{{{p}}}" for p in parts]
                return r"\shortstack{" + r"\\".join(parts) + "}"

            ax.set_xticklabels([_tex_italic_multiline(m) for m in method_order])
        else:
            # Matplotlib text path: normal newline + fontstyle works
            ax.set_xticklabels([_wrap_method_label(m) for m in method_order])
            for t in ax.get_xticklabels():
                t.set_fontstyle("italic")

        ax.tick_params(axis="x", rotation=0)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    _style_axis(ax)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_reward_curve_pdf(
    curves_by_method: Dict[str, List[List[float]]],
    method_order: List[str],
    out_pdf: Path,
    ylabel: str = "Episode Reward",
    frames_per_iteration: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.2), constrained_layout=True)

    for method in method_order:
        curves = curves_by_method.get(method, [])
        x, mean, std = _aggregate_curves(
            curves,
            smoothing="ema",
            ema_alpha=0.03,
        )
        if x is None:
            continue

        x_frames = [frames_per_iteration * xi for xi in x]

        x = x_frames

        style = METHOD_STYLE.get(method, DEFAULT_STYLE)
        method_legend = _wrap_method_label(method, is_line_break=False)
        line = ax.plot(
            x,
            mean,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            marker=style.get("marker", None),
            markersize=style.get("ms", 0),
            markeredgewidth=style.get("mew", 1.0),
            markevery=style.get("markevery", None),
            label=rf"\textit{{{method_legend}}}",
        )[0]

        # Use STD (run-to-run spread) for the shaded area
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.2, linewidth=0.0)

    def _nice_step(x: float) -> float:
        """Return a 'nice' step size close to x using {1,2,5}x10^k."""
        if x <= 0:
            return 1.0
        k = math.floor(math.log10(x))
        base = 10**k
        m = x / base
        if m <= 1:
            return 1 * base
        if m <= 2:
            return 2 * base
        if m <= 5:
            return 5 * base
        return 10 * base

    # Nice x ticks in frames
    if ax.lines:
        # Find xmax in frames from plotted data
        # Use axis limit (not data max) to define ticks
        xmax_frames = ax.get_xlim()[1]  # this is 4e6

        raw_step = xmax_frames / 5.0
        step_frames = _nice_step(raw_step)

        n_steps = int(math.floor(xmax_frames / step_frames))
        xticks = [k * step_frames for k in range(0, n_steps + 1)]  # no +2
        ax.set_xticks(xticks)

        # Display ticks in millions with clean numbers
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e6:g}"))
        ax.set_xlabel(r"Frames ($\times10^6$)")

    _style_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_xlim((0.0, 4.0e6))
    ax.set_ylim((-0.5, 1.0))
    ax.set_yticks(np.arange(-0.4, 1.1, 0.2))

    handles, labels = ax.get_legend_handles_labels()

    order = [0, 3, 1, 4, 2]

    handles = [handles[i] for i in order if i < len(handles)]
    labels = [labels[i] for i in order if i < len(labels)]

    leg = ax.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        frameon=True,
        framealpha=0.4,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    for t in leg.get_texts():
        t.set_fontstyle("italic")

    # ax.legend(frameon=True, framealpha=0.4, loc="upper left")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_figures_case_1(fig_dir: Path) -> None:
    # Derive unique reward methods
    # exp_labels = sorted({_exp_label_from_path(p) for p in path_list})
    exp_labels = sorted(
        {_exp_label_from_path(p) for p in path_list}, key=_method_sort_key
    )
    method_order = exp_labels

    method_order = exp_labels

    coll_rates: Dict[str, Dict[str, List[float]]] = {}
    avg_speeds: Dict[str, Dict[str, List[float]]] = {}
    avg_rews: Dict[str, Dict[str, List[float]]] = {}
    train_reward_curves: Dict[str, List[List[float]]] = {m: [] for m in method_order}
    cbf_activation_degree: Dict[str, Dict[str, List[float]]] = {}
    cbf_activation_rates: Dict[str, Dict[str, List[float]]] = {}
    task_num_tries: Dict[str, Dict[str, List[float]]] = {}
    task_success_times: Dict[str, Dict[str, List[float]]] = {}
    task_success_rate: Dict[str, Dict[str, List[float]]] = {}

    # Training reward curves (independent of scenario)
    for model_dir in path_list:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            continue

        method_label = _exp_label_from_path(str(model_dir))

        curve = _load_episode_reward_mean_list(model_dir)
        if curve is None:
            continue

        train_reward_curves[method_label].append(curve)

    for scenario_type, n_agents in scenario_specs:
        coll_rates[scenario_type] = {m: [] for m in method_order}
        avg_speeds[scenario_type] = {m: [] for m in method_order}
        avg_rews[scenario_type] = {m: [] for m in method_order}
        cbf_activation_degree[scenario_type] = {m: [] for m in method_order}
        cbf_activation_rates[scenario_type] = {m: [] for m in method_order}
        task_num_tries[scenario_type] = {m: [] for m in method_order}
        task_success_times[scenario_type] = {m: [] for m in method_order}
        task_success_rate[scenario_type] = {m: [] for m in method_order}

        # Each entry in path_list is a model folder (often .../h*/seed*)
        for model_dir in path_list:
            method_label = _exp_label_from_path(str(model_dir))

            model_dir = Path(model_dir)
            if not model_dir.is_dir():
                continue

            # Collect all out_td files for all models in this folder:
            # you used: out_td_{model_tag}_{scenario...}.td
            for eval_seed in random_seed_list:
                name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)

                # Match any model_tag prefix
                pattern = f"out_td_{name_suffix}.td"
                matches = list(model_dir.glob(pattern))
                if len(matches) == 0:
                    continue

                for out_td_path in matches:
                    try:
                        out_td = _load_out_td(out_td_path)
                        cr = compute_collision_rate_percent(out_td)
                        vbar = compute_average_speed(out_td)
                        rew = compute_total_reward(out_td)
                        cbf_act_deg, cbf_act_times = compute_cbf_activation_percentage(
                            out_td
                        )
                        task_perf_path = (
                            model_dir / f"task_performance_{name_suffix}.json"
                        )
                        tp = _load_task_performance(task_perf_path)

                        coll_rates[scenario_type][method_label].append(cr)
                        avg_speeds[scenario_type][method_label].append(vbar)
                        avg_rews[scenario_type][method_label].append(rew)
                        cbf_activation_degree[scenario_type][method_label].append(
                            cbf_act_deg
                        )
                        cbf_activation_rates[scenario_type][method_label].append(
                            cbf_act_times
                        )

                        task_num_tries[scenario_type][method_label].append(
                            float(tp["num_task_tries"])
                        )
                        task_success_times[scenario_type][method_label].append(
                            float(tp["task_success_times"])
                        )
                        task_success_rate[scenario_type][method_label].append(
                            float(100 * tp["task_success_times"] / tp["num_task_tries"])
                        )

                    except Exception as e:
                        print(f"[WARNING] Failed on {out_td_path}: {e}")

        # Per-scenario plots
        out_coll_pdf = fig_dir / "fig_collision_rate.pdf"
        out_speed_pdf = fig_dir / "fig_speed.pdf"
        out_rew_pdf = fig_dir / "fig_total_reward.pdf"
        out_train_rew_pdf = fig_dir / "fig_training_reward_curve.pdf"
        out_cbf_act_degree_pdf = fig_dir / "fig_cbf_activation_degree.pdf"
        out_cbf_act_rate_pdf = fig_dir / "fig_cbf_activation_rate.pdf"
        out_task_tries_pdf = fig_dir / "fig_task_attempts_total.pdf"
        out_success_task_tries_pdf = fig_dir / "fig_task_attempts_success.pdf"
        out_task_success_pdf = fig_dir / "fig_task_success_rate.pdf"

        log_path = fig_dir / f"{scenario_type}_summary.txt"

        # Use CBF (our) as reference method
        ref_method = "CBF (our)" if ("CBF (our)" in method_order) else None

        ref_stats = None
        if ref_method is not None:
            cr_ref = coll_rates[scenario_type][ref_method]
            v_ref = avg_speeds[scenario_type][ref_method]
            rew_ref = avg_rews[scenario_type][ref_method]
            deg_ref = cbf_activation_degree[scenario_type][ref_method]
            t_ref = cbf_activation_rates[scenario_type][ref_method]
            task_num_tries_ref = task_num_tries[scenario_type][ref_method]
            task_success_times_ref = task_success_times[scenario_type][ref_method]
            task_success_rate_ref = task_success_rate[scenario_type][ref_method]

            if len(cr_ref) > 0:
                ref_stats = {
                    "cr": sum(cr_ref) / len(cr_ref),
                    "v": sum(v_ref) / len(v_ref),
                    "rew": sum(rew_ref) / len(rew_ref),
                    "deg": sum(deg_ref) / len(deg_ref),
                    "t": sum(t_ref) / len(t_ref),
                    "task_num_tries": sum(task_num_tries_ref) / len(task_num_tries_ref),
                    "task_success_times": sum(task_success_times_ref)
                    / len(task_success_times_ref),
                    "task_success_rate": sum(task_success_rate_ref)
                    / len(task_success_rate_ref),
                }

        with open(log_path, "w", encoding="utf-8") as f:
            header = f"--- Scenario: {scenario_type} ---"
            print(header)
            f.write(header + "\n")

            if ref_stats is None:
                warn = '[WARNING] Reference method "distance" not found or has no data; improvements will not be logged.'
                print(warn)
                f.write(warn + "\n")

            for method in method_order:
                cr_list = coll_rates[scenario_type][method]
                vbar_list = avg_speeds[scenario_type][method]
                rew_list = avg_rews[scenario_type][method]
                cbf_act_deg_list = cbf_activation_degree[scenario_type][method]
                cbf_act_time_list = cbf_activation_rates[scenario_type][method]
                task_num_tries_ref_list = task_num_tries[scenario_type][method]
                task_success_times_ref_list = task_success_times[scenario_type][method]
                task_success_rate_ref_list = task_success_rate[scenario_type][method]

                if len(cr_list) == 0:
                    line = f"[INFO] Method: {method}: No data"
                    print(line)
                    f.write(line + "\n")
                    continue

                cr_avg = sum(cr_list) / len(cr_list)
                vbar_avg = sum(vbar_list) / len(vbar_list)
                rew_avg = sum(rew_list) / len(rew_list)
                cbf_act_deg_avg = sum(cbf_act_deg_list) / len(cbf_act_deg_list)
                cbf_act_time_avg = sum(cbf_act_time_list) / len(cbf_act_time_list)
                task_num_tries_ref_avg = sum(task_num_tries_ref_list) / len(
                    task_num_tries_ref_list
                )
                task_success_times_ref_avg = sum(task_success_times_ref_list) / len(
                    task_success_times_ref_list
                )
                task_success_rate_ref_avg = sum(task_success_rate_ref_list) / len(
                    task_success_rate_ref_list
                )

                line = (
                    f"[INFO] Method: {method}: "
                    # f"Collision Rate: {cr_avg:.3f} %, "
                    # f"Avg Speed: {vbar_avg:.3f} m/s, "
                    f"Avg Total Reward: {rew_avg:.3f}, "
                    f"CBF Activation Degree: {cbf_act_deg_avg:.3f} %, "
                    # f"CBF Activation Rate: {cbf_act_time_avg:.3f} %, "
                    # f"Task Num Tries: {task_num_tries_ref_avg:.3f}, "
                    f"Task Success Times: {task_success_times_ref_avg:.3f}, "
                    f"Task Success Rate: {task_success_rate_ref_avg:.3f}."
                )
                print(line)
                f.write(line + "\n")

                # Log percentage improvement/degradation of CBF (our) vs each other method
                if ref_stats is not None and method != ref_method:

                    def _pct_cbf_better_higher_is_better(
                        cbf: float, other: float
                    ) -> float:
                        # positive if CBF higher than other
                        if abs(other) < 1e-12:
                            return float("nan")
                        return 100.0 * (cbf - other) / abs(other)

                    def _pct_cbf_better_lower_is_better(
                        cbf: float, other: float
                    ) -> float:
                        # positive if CBF lower than other
                        if abs(other) < 1e-12:
                            return float("nan")
                        return 100.0 * (other - cbf) / abs(other)

                    # CBF values (reference)
                    cbf_cr = ref_stats["cr"]
                    cbf_v = ref_stats["v"]
                    cbf_rew = ref_stats["rew"]
                    cbf_deg = ref_stats["deg"]
                    cbf_t = ref_stats["t"]
                    cbf_task_num_tries = ref_stats["task_num_tries"]
                    cbf_task_success_times = ref_stats["task_success_times"]
                    cbf_task_success_rate = ref_stats["task_success_rate"]

                    # Percentage "CBF vs method" (positive = CBF better)
                    p_cr = _pct_cbf_better_lower_is_better(cbf_cr, cr_avg)
                    p_v = _pct_cbf_better_higher_is_better(cbf_v, vbar_avg)
                    p_rew = _pct_cbf_better_higher_is_better(cbf_rew, rew_avg)
                    p_deg = _pct_cbf_better_lower_is_better(cbf_deg, cbf_act_deg_avg)
                    p_t = _pct_cbf_better_lower_is_better(cbf_t, cbf_act_time_avg)
                    p_task_num_tries = _pct_cbf_better_lower_is_better(
                        cbf_task_num_tries, task_num_tries_ref_avg
                    )
                    p_task_success_times = _pct_cbf_better_lower_is_better(
                        cbf_task_success_times, task_success_times_ref_avg
                    )
                    p_task_success_rate = _pct_cbf_better_lower_is_better(
                        cbf_task_success_rate, task_success_rate_ref_avg
                    )

                    ref_name = _wrap_method_label(ref_method, is_line_break=False)
                    method_name = _wrap_method_label(method, is_line_break=False)

                    line_imp = (
                        f"    {ref_name} vs {method_name}: "
                        # f"Collision Rate {p_cr:+.1f} %, "
                        # f"Avg Speed {p_v:+.1f} %, "
                        f"Avg Total Reward {p_rew:+.1f} %, "
                        f"CBF Activation Degree {p_deg:+.1f} %, "
                        # f"CBF Activation Rate {p_t:+.1f} %, "
                        # f"Task Attempts {p_task_num_tries:+.1f} %, "
                        f"Task Success Times {p_task_success_times:+.1f} %, "
                        f"Task Success Rate {p_task_success_rate:+.1f} %"
                    )

                    print(line_imp)
                    f.write(line_imp + "\n")

        save_boxplot_pdf(
            data_by_method=coll_rates[scenario_type],
            method_order=method_order,
            ylabel=r"Collision Rate [\%]",
            out_pdf=out_coll_pdf,
            unit="%",
            ylim=(0, 1),
        )

        save_boxplot_pdf(
            data_by_method=avg_speeds[scenario_type],
            method_order=method_order,
            ylabel=r"Speed [$\mathrm{m/s}$]",
            out_pdf=out_speed_pdf,
            unit=" m/s",
            ylim=(0, 0.8),
        )

        save_boxplot_pdf(
            data_by_method=avg_rews[scenario_type],
            method_order=method_order,
            ylabel=r"Total Reward",
            out_pdf=out_rew_pdf,
            unit="",
            ylim=(0, 10),
        )

        save_reward_curve_pdf(
            curves_by_method=train_reward_curves,
            method_order=method_order,
            out_pdf=out_train_rew_pdf,
            ylabel="Episode Reward",
            frames_per_iteration=FRAME_PER_ITERATION,
        )

        save_boxplot_pdf(
            data_by_method=cbf_activation_degree[scenario_type],
            method_order=method_order,
            ylabel=r"CBF Act. Degree [\%]",
            out_pdf=out_cbf_act_degree_pdf,
            unit="%",
            ylim=(0, 20),
        )

        save_boxplot_pdf(
            data_by_method=cbf_activation_rates[scenario_type],
            method_order=method_order,
            ylabel=r"CBF Act. Ratio [\%]",
            out_pdf=out_cbf_act_rate_pdf,
            unit="%",
            ylim=(0, 20),
        )

        save_boxplot_pdf(
            data_by_method=task_num_tries[scenario_type],
            method_order=method_order,
            ylabel=r"\#Task Attempts",
            out_pdf=out_task_tries_pdf,
            unit="",
            ylim=(0, 50),
        )

        save_boxplot_pdf(
            data_by_method=task_success_times[scenario_type],
            method_order=method_order,
            ylabel=r"\#Successful Task Attempts",
            out_pdf=out_success_task_tries_pdf,
            unit="",
            ylim=(0, 50),
        )

        save_boxplot_pdf(
            data_by_method=task_success_rate[scenario_type],
            method_order=method_order,
            ylabel=r"Task Success Rate [\%]",
            out_pdf=out_task_success_pdf,
            unit="%",
            ylim=(0, 110),
        )

        print(f"[INFO] Saved: {out_coll_pdf}")
        print(f"[INFO] Saved: {out_speed_pdf}")
        print(f"[INFO] Saved: {out_rew_pdf}")
        print(f"[INFO] Saved: {out_train_rew_pdf}")
        print(f"[INFO] Saved: {out_cbf_act_degree_pdf}")
        print(f"[INFO] Saved: {out_cbf_act_rate_pdf}")
        print(f"[INFO] Saved: {out_task_tries_pdf}")
        print(f"[INFO] Saved: {out_task_success_pdf}")


# -----------------------------
# Case 2: Sensitivity colormap (Total Reward over tb x ta)
# -----------------------------
def _p_from_path(p: str) -> Optional[float]:
    m = re.search(r"/p(-?[0-9]+(?:\.[0-9]+)?)/", p)
    return float(m.group(1)) if m else None


def _tb_from_path(p: str) -> Optional[float]:
    m = re.search(r"/tb(-?[0-9]+(?:\.[0-9]+)?)/", p)
    return float(m.group(1)) if m else None


def _ta_from_path(p: str) -> Optional[float]:
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


def _collect_total_rewards_from_model_dirs(
    model_dirs: List[Path],
    scenario_specs: List[tuple],
    random_seed_list: List[int],
) -> List[float]:
    """
    Collect all Total Reward values across provided training folders,
    averaged per rollout file (one value per out_td), for global colormap scaling.
    """
    vals: List[float] = []
    for d in model_dirs:
        if not d.is_dir():
            continue
        for scenario_type, n_agents in scenario_specs:
            for eval_seed in random_seed_list:
                name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
                out_td_path = d / f"out_td_{name_suffix}.td"
                if not out_td_path.exists():
                    continue
                try:
                    out_td = _load_out_td(out_td_path)
                    vals.append(float(compute_total_reward(out_td)))
                except Exception:
                    continue
    return vals


def _global_vmin_vmax_for_sensitivity(
    path_list: List[str],
    scenario_specs: List[tuple],
    random_seed_list: List[int],
) -> tuple[Optional[float], Optional[float]]:
    """
    One global (vmin, vmax) for ALL sensitivity plots to enable cross-method comparison.
    """
    model_dirs = [Path(p) for p in path_list if Path(p).is_dir()]
    vals = _collect_total_rewards_from_model_dirs(
        model_dirs, scenario_specs, random_seed_list
    )
    if len(vals) == 0:
        return None, None
    return float(np.nanmin(vals)), float(np.nanmax(vals))


def plot_case2_sensitivity_colormaps(
    path_list: List[str],
    scenario_specs: List[tuple],
    random_seed_list: List[int],
    fig_dir: Path,
    only_p_values: Optional[List[float]] = None,
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
    is_highlight_best_value: bool = True,
) -> None:

    """
    For each (reward method, p) pair, create a tb (x) vs ta (y) colormap
    where each cell is the mean Total Reward averaged over:
      - scenario_specs (usually a single scenario for Case 2)
      - random_seed_list (evaluation seeds)

    only_p_values: if not None, restrict to these p values (example: [-1.0]).
    shared_color_scale_within_method: if True, uses a shared (vmin,vmax)
      across all p for the same method to keep comparisons consistent.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect folders grouped by (method_label, p)
    groups: Dict[tuple, List[Path]] = {}
    tb_set: Dict[tuple, set] = {}
    ta_set: Dict[tuple, set] = {}

    for s in path_list:
        model_dir = Path(s)
        if not model_dir.is_dir():
            continue

        method_label = _exp_label_from_path(str(model_dir))
        p_val = _p_from_path(str(model_dir))
        tb_val = _tb_from_path(str(model_dir))
        ta_val = _ta_from_path(str(model_dir))

        if p_val is None or tb_val is None or ta_val is None:
            continue
        if only_p_values is not None and (p_val not in set(only_p_values)):
            continue

        key = (method_label, p_val)
        groups.setdefault(key, []).append(model_dir)
        tb_set.setdefault(key, set()).add(tb_val)
        ta_set.setdefault(key, set()).add(ta_val)

    # Generate one heatmap per (method, p)
    for (method_label, p_val), dirs in sorted(
        groups.items(), key=lambda x: (_method_sort_key(x[0][0]), x[0][1])
    ):
        tb_vals = sorted(tb_set[(method_label, p_val)])
        ta_vals = sorted(ta_set[(method_label, p_val)])

        tb_to_j = {v: j for j, v in enumerate(tb_vals)}
        ta_to_i = {v: i for i, v in enumerate(ta_vals)}

        # Accumulate rewards per cell
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

            # Average over scenario_specs and evaluation seeds
            for scenario_type, n_agents in scenario_specs:
                for eval_seed in random_seed_list:
                    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
                    out_td_path = d / f"out_td_{name_suffix}.td"
                    if not out_td_path.exists():
                        continue
                    try:
                        out_td = _load_out_td(out_td_path)
                        rew = compute_total_reward(out_td)
                        cell_values[i][j].append(float(rew))
                    except Exception:
                        continue

        # Build Z as mean reward per cell (NaN if missing)
        z = np.full((len(ta_vals), len(tb_vals)), np.nan, dtype=float)
        for i in range(len(ta_vals)):
            for j in range(len(tb_vals)):
                vals = cell_values[i][j]
                if len(vals) > 0:
                    z[i, j] = float(np.mean(vals))

        vmin = global_vmin
        vmax = global_vmax
        if vmin is None or vmax is None:
            finite = np.isfinite(z)
            if np.any(finite):
                vmin, vmax = float(np.nanmin(z)), float(np.nanmax(z))

        out_pdf = fig_dir / f"fig_colormap_total_reward_{method_label.lower()}.pdf"

        title = f"{_wrap_method_label(method_label, is_line_break=False)}"
        save_sensitivity_colormap_pdf(
            z=z,
            tb_vals=tb_vals,
            ta_vals=ta_vals,
            out_pdf=out_pdf,
            title=title,
            cbar_label="Total Reward",
            vmin=vmin,
            vmax=vmax,
            cmap_name="Blues",
            method_label=method_label,
            is_highlight_best_value=is_highlight_best_value,
            is_higher_better=True,
        )

        print(f"[INFO] Saved: {out_pdf}")


def plot_case2_sensitivity_colormaps_metric(
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


def plot_case3_cbf_h_colormap_metric(
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
        print("[WARNING] No CBF h-folders found for Case 3.")
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

    # reuse your 1D saver, but update it analogously to accept cbar_label/fmt/norm
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
# Case 3: CBF sensitivity colormap (Total Reward over h)
# -----------------------------


def _h_from_path(p: str) -> Optional[float]:
    m = re.search(r"/h(-?[0-9]+(?:\.[0-9]+)?)/", p)
    return float(m.group(1)) if m else None


def save_sensitivity_colormap_1d_pdf(
    z: np.ndarray,
    x_vals: List[float],
    out_pdf: Path,
    title: str,
    xlabel: str = r"$h$",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "Blues",
    is_highlight_best_value: bool = True,
) -> None:
    """
    Save a 1D sensitivity colormap as a 1xK heatmap.
    z shape: [1, len(x_vals)]
    """
    fig, ax = plt.subplots(figsize=(4.2, 2.2), constrained_layout=True)

    z_clipped = np.clip(z, -10, 10)

    norm = make_reward_diverging_norm(vmin, vmax)

    cmap = plt.get_cmap("RdBu").copy()
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
        fmt="{:.1f}",
        fontsize=COLORMAP_VALUE_FONTSIZE,
        is_highlight_best_value=is_highlight_best_value,
        is_higher_better=True,
    )

    ax.set_xticks(list(range(len(x_vals))))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.set_yticks([0])
    ax.set_yticklabels([""])  # hide the dummy y label

    ax.set_xlabel(xlabel)
    # ax.set_title(title)

    _style_axis(ax)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
    cbar.set_label("Total Reward")

    finite_vals = z[np.isfinite(z)]
    if len(finite_vals) > 0:
        data_min = float(np.min(finite_vals))
        data_max = float(np.max(finite_vals))
        cbar.ax.set_ylim(data_min, data_max)
        # Generate ~5 clean ticks within the actual data range
        tick_step = (data_max - data_min) / 4.0
        magnitude = 10 ** np.floor(np.log10(abs(tick_step))) if tick_step > 0 else 1
        tick_step_nice = np.round(tick_step / magnitude) * magnitude
        ticks = np.arange(
            np.ceil(data_min / tick_step_nice) * tick_step_nice,
            data_max + tick_step_nice * 0.5,
            tick_step_nice,
        )
        cbar.set_ticks(ticks)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_case3_cbf_h_colormap(
    path_list: List[str],
    scenario_specs: List[tuple],
    random_seed_list: List[int],
    fig_dir: Path,
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
    is_highlight_best_value: bool = True,
) -> None:
    """
    Create a 1xK colormap for CBF over h, where each cell is the mean Total Reward
    averaged over (scenario_specs x random_seed_list).

    Assumes path_list contains only seed1 training folders, but evaluation still uses random_seed_list.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect model dirs keyed by h
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
        print("[WARNING] No CBF h-folders found for Case 3.")
        return

    h_vals = sorted(dirs_by_h.keys())

    # Compute mean reward for each h
    rewards_mean = []
    all_rewards = []

    for h in h_vals:
        vals = []
        for d in dirs_by_h[h]:
            for scenario_type, n_agents in scenario_specs:
                for eval_seed in random_seed_list:
                    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
                    out_td_path = d / f"out_td_{name_suffix}.td"
                    if not out_td_path.exists():
                        continue
                    try:
                        out_td = _load_out_td(out_td_path)
                        rew = compute_total_reward(out_td)
                        vals.append(float(rew))
                    except Exception:
                        continue

        if len(vals) == 0:
            rewards_mean.append(float("nan"))
        else:
            rewards_mean.append(float(np.mean(vals)))
            all_rewards.extend(vals)

    # Build Z as 1xK
    z = np.asarray(rewards_mean, dtype=float)[None, :]  # shape [1, K]

    vmin = global_vmin
    vmax = global_vmax
    if vmin is None or vmax is None:
        finite = np.isfinite(z)
        if np.any(finite):
            vmin, vmax = float(np.nanmin(z)), float(np.nanmax(z))

    out_pdf = fig_dir / "fig_colormap_total_reward_cbf_h.pdf"
    title = "CBF (our)"

    save_sensitivity_colormap_1d_pdf(
        z=z,
        x_vals=h_vals,
        out_pdf=out_pdf,
        title=title,
        xlabel=r"$h$",
        vmin=vmin,
        vmax=vmax,
        cmap_name="Blues",
        is_highlight_best_value=is_highlight_best_value,
    )

    print(f"[INFO] Saved: {out_pdf}")


def metric_total_reward(
    d: Path, scenario_type: str, n_agents: int, eval_seed: int
) -> Optional[float]:
    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
    out_td_path = d / f"out_td_{name_suffix}.td"
    if not out_td_path.exists():
        return None
    out_td = _load_out_td(out_td_path)
    return float(compute_total_reward(out_td))


def metric_cbf_activation_degree(
    d: Path, scenario_type: str, n_agents: int, eval_seed: int
) -> Optional[float]:
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
    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
    tp = _load_task_performance(d / f"task_performance_{name_suffix}.json")
    if tp is None:
        return None
    return float(tp["num_task_tries"])


def metric_task_success_rate(
    d: Path, scenario_type: str, n_agents: int, eval_seed: int
) -> Optional[float]:
    name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)
    tp = _load_task_performance(d / f"task_performance_{name_suffix}.json")
    if tp is None or tp["num_task_tries"] <= 0:
        return None
    return float(100.0 * tp["task_success_times"] / tp["num_task_tries"])


def plot_sensitivity_case2_and_case3(
    path_list_case2: List[str],
    path_list_case3: List[str],
    scenario_specs: List[tuple],
    random_seed_list: List[int],
    fig_dir: Path,
    only_p_values: Optional[List[float]] = None,
    is_highlight_best_value: bool = True,
) -> None:
    """
    Plot Case 2 (tb x ta) and Case 3 (h) using ONE shared global color scale.
    """
    merged = list(
        dict.fromkeys(path_list_case2 + path_list_case3)
    )  # unique, stable order
    global_vmin, global_vmax = _global_vmin_vmax_for_sensitivity(
        merged, scenario_specs, random_seed_list
    )

    # TODO
    # abs_max = max(abs(global_vmin), abs(global_vmax))
    # global_vmin, global_vmax = -abs_max, abs_max

    print(f"[INFO] Global colormap scale: vmin={global_vmin}, vmax={global_vmax}")

    plot_case2_sensitivity_colormaps(
        path_list=path_list_case2,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        only_p_values=only_p_values,
        global_vmin=global_vmin,
        global_vmax=global_vmax,
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_case3_cbf_h_colormap(
        path_list=path_list_case3,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        global_vmin=global_vmin,
        global_vmax=global_vmax,
        is_highlight_best_value=is_highlight_best_value,
    )


if __name__ == "__main__":
    is_highlight_best_value = True

    scenario_specs = [("cpm_mixed", 4)]
    print(f"[INFO] Evaluating scenarios: {scenario_specs}")

    policy_parent_folder = "checkpoints/itsc26_sensitivity_new/cpm_mixed"

    filtered_case2 = []
    for root, dirs, files in os.walk(policy_parent_folder):
        if any(f.endswith(".pth") for f in files):
            is_case2_folder = (
                ("reward_progress0.1" in root)
                and ("p-1.0" in root)
                and ("seed1" in root)
                and ("tb0.001" not in root)
                and ("ta1.0" not in root)
            )
            is_cbf_folder = ("/rew_method_cbf" in root) or (
                "/rew_method_cbf_sparse" in root
            )
            if is_case2_folder and not is_cbf_folder:
                filtered_case2.append(root)

    filtered_case3 = []
    for root, dirs, files in os.walk(policy_parent_folder):
        if any(f.endswith(".pth") for f in files):
            is_case3_folder = (
                ("rew_method_cbf" in root)
                and ("reward_progress0.1" in root)
                and ("seed1" in root)
                and ("h0.02" not in root)
            )
            if is_case3_folder:
                filtered_case3.append(root)

    path_list_case2 = filtered_case2
    path_list_case3 = filtered_case3

    random_seed_list = [1, 2, 3, 4]
    COLLISION_MODE = "per_agent_timestep"

    path_list = list(dict.fromkeys(path_list_case2 + path_list_case3))
    run_evaluations()

    fig_dir = Path(policy_parent_folder) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    reward_vmin = -10.0
    reward_vmax = 10.0
    reward_norm = make_reward_diverging_norm(reward_vmin, reward_vmax)

    plot_case2_sensitivity_colormaps_metric(
        path_list=path_list_case2,
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

    plot_case3_cbf_h_colormap_metric(
        path_list=path_list_case3,
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

    plot_case2_sensitivity_colormaps_metric(
        path_list=path_list_case2,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_cbf_activation_degree,
        metric_tag="cbf_activation_degree",
        cbar_label=r"CBF Act. Degree [\%]",
        fmt="{:.1f}",
        only_p_values=[-1.0],
        global_vmin=0.0,
        global_vmax=100.0,
        cmap_name="Blues",
        norm=None,
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_case3_cbf_h_colormap_metric(
        path_list=path_list_case3,
        scenario_specs=scenario_specs,
        random_seed_list=random_seed_list,
        fig_dir=fig_dir,
        metric_fn=metric_cbf_activation_degree,
        metric_tag="cbf_activation_degree",
        cbar_label=r"CBF Act. Degree [\%]",
        fmt="{:.1f}",
        global_vmin=0.0,
        global_vmax=100.0,
        cmap_name="Blues",
        norm=None,
        is_highlight_best_value=is_highlight_best_value,
    )

    plot_case2_sensitivity_colormaps_metric(
        path_list=path_list_case2,
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

    plot_case3_cbf_h_colormap_metric(
        path_list=path_list_case3,
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

    plot_case2_sensitivity_colormaps_metric(
        path_list=path_list_case2,
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

    plot_case3_cbf_h_colormap_metric(
        path_list=path_list_case3,
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

    plot_case2_sensitivity_colormaps_metric(
        path_list=path_list_case2,
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

    plot_case3_cbf_h_colormap_metric(
        path_list=path_list_case3,
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

    save_reward_curves_per_method_hparam_coded_case23(
        path_list_case2=path_list_case2,
        path_list_case3=path_list_case3,
        out_dir=fig_dir,
        frames_per_iteration=FRAME_PER_ITERATION,
    )
