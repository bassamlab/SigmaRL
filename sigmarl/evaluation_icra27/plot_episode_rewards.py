#!/usr/bin/env python3
# Copyright (c) 2024-2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usage:
    python sigmarl/evaluation_icra27/plot_episode_rewards.py --path checkpoints/icra27/main/cpm_mixed
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
    VPacker,
)
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import IdentityTransform
import numpy as np

SEED_RE = re.compile(r"^(?P<tag>.+)_seed_?(?P<seed>\d+)$")

DEFAULT_SMOOTHING_WINDOW = 15
ENV_STEPS_PER_MILLION = 1_000_000


def get_academic_rc_params(use_tex: bool = False) -> Dict[str, Any]:
    """Configure publication-quality Matplotlib styling for academic robotics papers.

    Uses Times New Roman serif font by default with Matplotlib's built-in mathtext engine
    for cross-platform compatibility. Full external LaTeX compilation can be enabled via use_tex.
    """
    params: Dict[str, Any] = {
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.50,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,  # Embed TrueType fonts in PDF for IEEE/ACM compliance
        "ps.fonttype": 42,
    }
    # if use_tex:
    #     params["text.latex.preamble"] = (
    #         r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathptmx}"
    #     )
    return params


# Method registry defining display labels, academic color palettes, line styles,
# drawing hierarchy (zorder), logical grouping, and sorting order.
METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    # ---------------- Proposed Full Method ----------------
    "our_rr_motion_taskprogress": {
        "label": "Our",
        "color": "#1f77b4",  # Primary Royal Blue
        "linestyle": "-",
        "linewidth": 2.2,
        "zorder": 10,
        "group": "proposed",
        "order": 1,
    },
    # ---------------- Ablation Study Variants ----------------
    "our_rr_only_taskprogress": {
        "label": "w/o pri. cond.",
        "color": "#9467bd",  # Amethyst Purple
        "linestyle": "-.",
        "linewidth": 1.8,
        "zorder": 8,
        "group": "ablation",
        "order": 2,
    },
    "our_motion_only_taskprogress": {
        "label": "w/o rew. red.",
        "color": "#2ca02c",  # Green
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 7,
        "group": "ablation",
        "order": 3,
    },
    "our_rr_motion_taskprogress_permute": {
        "label": "Shuffled pri.",
        "color": "#8c564b",  # Saddle Brown
        "linestyle": ":",
        "linewidth": 1.8,
        "zorder": 6,
        "group": "ablation",
        "order": 4,
    },
    # ---------------- Baseline Methods ----------------
    "vanilla": {
        "label": "Vanilla",
        "color": "#7f7f7f",  # Slate Gray
        "linestyle": "-",
        "linewidth": 1.8,
        "zorder": 4,
        "group": "baseline",
        "order": 5,
    },
    "om_dedicated": {
        "label": "OM-PPO",
        "color": "#ff7f0e",  # Vivid Amber / Orange
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 5,
        "group": "baseline",
        "order": 6,
    },
    "mtppo_motion": {
        "label": "MT-PPO",
        "color": "#aa177c",  # Pink
        "linestyle": "-.",
        "linewidth": 1.8,
        "zorder": 3,
        "group": "baseline",
        "order": 7,
    },
}

FALLBACK_CMAP = plt.get_cmap("tab10")


def positive_window(value: str) -> int:
    """Parse a positive rolling-window length for centered smoothing."""
    try:
        window = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("window must be an integer") from error
    if window <= 0:
        raise argparse.ArgumentTypeError("window must be positive")
    return window


def optional_float(value: str) -> float | None:
    """Parse a float or 'none' / 'null' to None."""
    if value.lower() in ("none", "null", "no", "off"):
        return None
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}") from error


def parse_keywords(value: str) -> tuple[str, ...]:
    """Parse hyphen-separated alternative substrings for run directories."""
    return tuple(keyword for keyword in value.split("-") if keyword)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(
            "checkpoints/xp-marl-20260825-main-real_dyn_2-new-hyperparams/cpm_mixed"
        ),
        help=(
            "Experiment path containing scenario folders, or one scenario "
            "folder containing <method>_seed[_]<seed> directories."
        ),
    )
    parser.add_argument(
        "--keywords",
        type=parse_keywords,
        default=(),
        help=(
            "Hyphen-separated keywords. Include only method/seed subfolders "
            "whose names contain at least one keyword."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where per-scenario figures are saved (default: "
            "<path>/00_fig, with a keyword subfolder added when filtering)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    parser.add_argument(
        "--smoothing-window",
        type=positive_window,
        default=DEFAULT_SMOOTHING_WINDOW,
        metavar="N",
        help=(
            "Centered rolling-mean window in training iterations "
            f"(default: {DEFAULT_SMOOTHING_WINDOW}; use 1 for raw curves)."
        ),
    )
    parser.add_argument(
        "--show-raw-mean",
        action="store_true",
        help="Show the unsmoothed cross-seed mean as a faint dotted reference.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[6.0, 2.4],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure width and height in inches.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=[0.0, 20.0],
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits (default: 0.0 20.0).",
    )
    parser.add_argument(
        "--ytick-interval",
        type=float,
        default=5.0,
        help="Y-axis tick interval (default: 5.0).",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Main X-axis limits in millions of environment steps (default: auto-detected).",
    )
    parser.add_argument(
        "--inset",
        type=str,
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Inset axis mode for methods with extended training horizons "
            "(default: auto; enabled if a method runs >= 1.5x longer than standard)."
        ),
    )
    parser.add_argument(
        "--inset-bounds",
        type=float,
        nargs=4,
        default=[0.54, 0.10, 0.37, 0.32],
        metavar=("X0", "Y0", "WIDTH", "HEIGHT"),
        help="Inset axes bounds in normalized axes coordinates (default: 0.54 0.10 0.37 0.32).",
    )
    parser.add_argument(
        "--inset-title",
        type=str,
        default="Full Horizon",
        help="Title for the inset axes (default: 'Full Horizon').",
    )
    parser.add_argument(
        "--inset-hline",
        type=optional_float,
        default=12.6,
        help=(
            "Optional y-value for a horizontal reference line in the inset axes "
            "(default: 12.6; pass 'none' to disable)."
        ),
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Enable external LaTeX engine for text and math rendering (requires system LaTeX).",
    )
    return parser.parse_args()


def steps_per_iteration(parameters: object) -> int | None:
    """Return collected environment steps per iteration from run metadata."""
    if not isinstance(parameters, dict):
        return None

    configured_frames = parameters.get("frames_per_batch")
    if isinstance(configured_frames, (int, float)) and configured_frames > 0:
        return int(configured_frames)

    num_envs = parameters.get("num_vmas_envs")
    max_steps = parameters.get("max_steps")
    if (
        isinstance(num_envs, (int, float))
        and isinstance(max_steps, (int, float))
        and num_envs > 0
        and max_steps > 0
    ):
        return int(num_envs * max_steps)
    return None


def load_seed_series(
    seed_dir: Path,
) -> tuple[np.ndarray, int | None, int | None] | None:
    """Pick the longest reward list, configured length, and rollout size."""
    best = None
    json_files = sorted(seed_dir.glob("reward*_data.json"))
    if not json_files:
        json_files = sorted(seed_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        series = data.get("episode_reward_mean_list")
        if not isinstance(series, list) or not series:
            continue
        try:
            arr = np.asarray(series, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 1:
            continue
        parameters = data.get("parameters")
        configured_length = (
            parameters.get("n_iters") if isinstance(parameters, dict) else None
        )
        if not isinstance(configured_length, int) or configured_length <= 0:
            configured_length = None
        configured_steps = steps_per_iteration(parameters)
        if best is None or len(arr) > len(best[0]):
            best = arr, configured_length, configured_steps
    return best


def parse_seed_directory(name: str) -> tuple[str, int] | None:
    """Return a method tag and seed from a supported directory name."""
    match = SEED_RE.fullmatch(name)
    if match is None:
        return None
    return match.group("tag"), int(match.group("seed"))


def collect_method_series(
    scenario_dir: Path, tag: str, keywords: tuple[str, ...] = ()
) -> list[tuple[int, np.ndarray, int | None, int | None]]:
    runs = []
    for child in sorted(scenario_dir.iterdir()):
        if not child.is_dir():
            continue
        if keywords and not any(keyword in child.name for keyword in keywords):
            continue
        parsed = parse_seed_directory(child.name)
        if parsed is None or parsed[0] != tag:
            continue
        loaded = load_seed_series(child)
        if loaded is None:
            print(f"  [warn] no valid episode_reward_mean_list in {child}")
            continue
        series, configured_length, configured_steps = loaded
        runs.append((parsed[1], series, configured_length, configured_steps))
    return runs


def align_series(series_list: list[np.ndarray]) -> np.ndarray:
    """Right-pad series with NaN so all available observations are retained."""
    max_len = max(len(series) for series in series_list)
    aligned = np.full((len(series_list), max_len), np.nan)
    for row, series in enumerate(series_list):
        aligned[row, : len(series)] = series
    return aligned


def centered_rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    """Return a NaN-aware centered rolling mean without changing the input."""
    values = np.asarray(series, dtype=float)
    if window == 1 or len(values) <= 1:
        return values.copy()

    effective_window = min(window, len(values))
    finite = np.isfinite(values)
    kernel = np.ones(effective_window, dtype=float)
    sums = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(float), kernel, mode="same")
    smoothed = np.full(values.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=smoothed, where=counts > 0)
    return smoothed


def method_sort_key(tag: str) -> tuple[int, str]:
    """Sort methods by their designated academic priority order."""
    spec = METHOD_SPECS.get(tag)
    if spec is not None:
        return (spec.get("order", 100), tag)
    return (200, tag)


def discover_method_tags(
    scenario_dir: Path, keywords: tuple[str, ...] = ()
) -> list[str]:
    """Discover method tags from child directories ending in a seed suffix."""
    tags = set()
    for child in scenario_dir.iterdir():
        if not child.is_dir():
            continue
        if keywords and not any(keyword in child.name for keyword in keywords):
            continue
        parsed = parse_seed_directory(child.name)
        if parsed is not None:
            tags.add(parsed[0])
    return sorted(tags, key=method_sort_key)


def discover_scenarios(root: Path, keywords: tuple[str, ...] = ()) -> list[str]:
    """Return scenario names found below ``root``."""
    if discover_method_tags(root, keywords):
        return [root.name]

    scenarios = []
    for scenario_dir in sorted(root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        if discover_method_tags(scenario_dir, keywords):
            scenarios.append(scenario_dir.name)
    return scenarios


def get_method_style(tag: str, fallback_idx: int) -> Dict[str, Any]:
    """Return full styling metadata for a given method tag."""
    if tag in METHOD_SPECS:
        return METHOD_SPECS[tag].copy()

    # Dynamic fallback styling
    label = tag.replace("_", " ").title()
    color = to_hex(FALLBACK_CMAP(fallback_idx % FALLBACK_CMAP.N))
    return {
        "label": label,
        "color": color,
        "linestyle": "-",
        "linewidth": 1.8,
        "zorder": 2,
        "group": "other",
        "order": 100 + fallback_idx,
    }


class _FramedLegendBox(VPacker):
    """VPacker that draws a styled background patch with border around itself."""

    def __init__(
        self,
        children: list[Any],
        pad: float = 3.5,
        sep: float = 2.5,
        align: str = "left",
        edgecolor: str = "#CCCCCC",
        facecolor: str = "#FFFFFF",
        alpha: float = 0.50,
        linewidth: float = 0.75,
        boxstyle: str = "round,pad=0,rounding_size=2.5",
    ) -> None:
        super().__init__(children=children, pad=pad, sep=sep, align=align)
        self.edgecolor = edgecolor
        self.facecolor = facecolor
        self.alpha = alpha
        self.linewidth = linewidth
        self.boxstyle = boxstyle

    def draw(self, renderer: Any) -> None:
        bbox, offsets = self._get_bbox_and_child_offsets(renderer)
        px, py = self.get_offset(bbox, renderer)
        patch = FancyBboxPatch(
            (px + bbox.x0, py + bbox.y0),
            bbox.width,
            bbox.height,
            boxstyle=self.boxstyle,
            edgecolor=self.edgecolor,
            facecolor=self.facecolor,
            alpha=self.alpha,
            linewidth=self.linewidth,
            mutation_scale=1.0,
            transform=IdentityTransform(),
            zorder=1,
        )
        patch.draw(renderer)
        for c, (ox, oy) in zip(self.get_visible_children(), offsets):
            c.set_offset((px + ox, py + oy))
            c.draw(renderer)
        self.stale = False


def _create_legend_item_box(
    label: str,
    color: str,
    linestyle: str,
    linewidth: float,
    font_size: float = 9.0,
    handle_len: float = 18.0,
    handle_height: float = 8.0,
) -> HPacker:
    """Create a horizontal box with a styled line handle and text label."""
    da = DrawingArea(handle_len, handle_height, 0, -2)
    line = Line2D(
        [0, handle_len],
        [handle_height / 2, handle_height / 2],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    da.add_artist(line)
    text = TextArea(label, textprops=dict(fontsize=font_size, color="#222222"))
    return HPacker(children=[da, text], align="center", pad=0, sep=5)


def _create_category_sub_box(
    title: str,
    items: list[HPacker],
    use_tex: bool = False,
    font_size: float = 9.0,
    pad: float = 3.0,
    sep: float = 2.5,
) -> _FramedLegendBox:
    """Create a nested framed sub-box for a category with title and items."""
    title_text = f"\\textbf{{{title}}}" if use_tex else title
    title_props = dict(fontsize=font_size, color="#222222")
    if not use_tex:
        title_props["fontweight"] = "bold"
    title_box = TextArea(title_text, textprops=title_props)
    return _FramedLegendBox(
        children=[title_box] + items,
        pad=pad,
        sep=sep,
        align="left",
        edgecolor="#CCCCCC",
        facecolor="#FFFFFF",
        alpha=0.50,
        linewidth=0.75,
        boxstyle="round,pad=0,rounding_size=2.5",
    )


def plot_single_panel(
    ax: plt.Axes,
    scenario_dir: Path,
    tags: list[str],
    smoothing_window: int,
    show_raw_mean: bool,
    keywords: tuple[str, ...] = (),
    use_tex: bool = False,
    ylim: tuple[float, float] = (0.0, 20.0),
    ytick_interval: float = 5.0,
    xlim: tuple[float, float] | None = None,
    inset_mode: str = "auto",
    inset_bounds: tuple[float, float, float, float] = (0.54, 0.10, 0.37, 0.32),
    inset_title: str = "Full Horizon",
    inset_hline: float | None = 12.0,
) -> int:
    """Plot training curves on a specific Axes object with optional inset axis for extended runs."""
    curve_data: list[dict[str, Any]] = []

    for idx, tag in enumerate(tags):
        runs = collect_method_series(scenario_dir, tag, keywords)
        if not runs:
            continue

        style = get_method_style(tag, idx)

        series_list = [series for _, series, _, _ in runs]
        configured_steps = {steps for _, _, _, steps in runs if steps is not None}
        if len(configured_steps) > 1:
            raise ValueError(
                f"{scenario_dir.name}/{tag} has inconsistent environment steps per "
                f"iteration: {sorted(configured_steps)}"
            )
        steps_per_iter = next(iter(configured_steps), None)
        if steps_per_iter is None:
            raise ValueError(
                f"{scenario_dir.name}/{tag} is missing num_vmas_envs/max_steps "
                "metadata needed for the environment-step x-axis"
            )

        smoothed_series_list = [
            centered_rolling_mean(series, smoothing_window) for series in series_list
        ]
        aligned = align_series(smoothed_series_list)
        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)
        x = np.arange(1, len(mean) + 1) * steps_per_iter / ENV_STEPS_PER_MILLION

        raw_mean = None
        if show_raw_mean:
            raw_aligned = align_series(series_list)
            raw_mean = np.nanmean(raw_aligned, axis=0)

        curve_data.append(
            {
                "tag": tag,
                "style": style,
                "x": x,
                "mean": mean,
                "std": std,
                "raw_mean": raw_mean,
                "max_x": float(x[-1]) if len(x) > 0 else 0.0,
            }
        )

    if not curve_data:
        return 0

    max_xs = [d["max_x"] for d in curve_data if d["max_x"] > 0]
    global_max_x = max(max_xs) if max_xs else 0.0

    # Determine common budget max_x (the standard horizon of typical methods)
    if len(max_xs) > 1:
        shorter_xs = [v for v in max_xs if v <= 0.67 * global_max_x]
        if shorter_xs:
            common_max_x = max(shorter_xs)
        else:
            common_max_x = global_max_x
    else:
        common_max_x = global_max_x

    show_inset = False
    if inset_mode == "always":
        show_inset = True
    elif inset_mode == "auto":
        show_inset = global_max_x >= 1.5 * common_max_x and common_max_x > 0

    if xlim is not None:
        main_xlim = tuple(xlim)
    elif show_inset:
        main_xlim = (0.0, common_max_x)
    else:
        main_xlim = (0.0, global_max_x) if global_max_x > 0 else (0.0, None)

    n_plotted = 0
    plotted_handles: dict[str, Any] = {}
    plotted_styles: dict[str, dict[str, Any]] = {}

    for d in curve_data:
        tag = d["tag"]
        style = d["style"]
        label = style["label"]
        color = style["color"]
        linestyle = style["linestyle"]
        linewidth = style["linewidth"]
        zorder = style["zorder"]
        x = d["x"]
        mean = d["mean"]
        std = d["std"]
        raw_mean = d["raw_mean"]

        if raw_mean is not None:
            ax.plot(
                x,
                raw_mean,
                color=color,
                linestyle=":",
                linewidth=0.7,
                alpha=0.35,
                zorder=1,
            )

        # Shaded standard deviation ribbon
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=zorder - 1,
        )

        # Mean curve
        (line,) = ax.plot(
            x,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
        )
        plotted_handles[tag] = line
        plotted_styles[tag] = style
        n_plotted += 1

    if n_plotted > 0:
        ax.set_xlabel(r"Environment Steps ($\times 10^6$)")
        ax.set_ylabel(r"Mean Episode Reward")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
            if ytick_interval is not None and ytick_interval > 0:
                ax.set_yticks(
                    np.arange(ylim[0], ylim[1] + ytick_interval * 0.5, ytick_interval)
                )

        if main_xlim[1] is not None:
            ax.set_xlim(main_xlim[0], main_xlim[1])
        else:
            ax.set_xlim(left=main_xlim[0])

        proposed_tags = [
            t
            for t in tags
            if t in plotted_handles and plotted_styles[t].get("group") == "proposed"
        ]
        baseline_tags = [
            t
            for t in tags
            if t in plotted_handles and plotted_styles[t].get("group") == "baseline"
        ]
        ablation_tags = [
            t
            for t in tags
            if t in plotted_handles and plotted_styles[t].get("group") == "ablation"
        ]
        other_tags = [
            t
            for t in tags
            if t in plotted_handles
            and plotted_styles[t].get("group")
            not in ("proposed", "baseline", "ablation")
        ]

        if baseline_tags and ablation_tags:
            # Side-by-side categorized layout:
            # From left to right: Our, Baselines block, Ablation Variants block (top horizontally aligned)
            sub_boxes: list[Any] = []

            if proposed_tags:
                proposed_items = [
                    _create_legend_item_box(
                        plotted_styles[t]["label"],
                        plotted_styles[t]["color"],
                        plotted_styles[t]["linestyle"],
                        plotted_styles[t]["linewidth"],
                        font_size=9.0,
                        handle_len=18.0,
                    )
                    for t in proposed_tags
                ]
                box_proposed = _FramedLegendBox(
                    children=proposed_items,
                    pad=3.0,
                    sep=2.5,
                    align="left",
                    edgecolor="#CCCCCC",
                    facecolor="#FFFFFF",
                    alpha=0.50,
                    linewidth=0.75,
                    boxstyle="round,pad=0,rounding_size=2.5",
                )
                sub_boxes.append(box_proposed)

            baseline_items = [
                _create_legend_item_box(
                    plotted_styles[t]["label"],
                    plotted_styles[t]["color"],
                    plotted_styles[t]["linestyle"],
                    plotted_styles[t]["linewidth"],
                    font_size=9.0,
                )
                for t in baseline_tags
            ]
            box_baselines = _create_category_sub_box(
                "Baselines", baseline_items, use_tex=use_tex
            )
            sub_boxes.append(box_baselines)

            ablation_items = [
                _create_legend_item_box(
                    plotted_styles[t]["label"],
                    plotted_styles[t]["color"],
                    plotted_styles[t]["linestyle"],
                    plotted_styles[t]["linewidth"],
                    font_size=9.0,
                )
                for t in ablation_tags
            ]
            box_ablations = _create_category_sub_box(
                "Ablation Variants", ablation_items, use_tex=use_tex
            )
            sub_boxes.append(box_ablations)

            legend_packer = HPacker(children=sub_boxes, align="top", pad=0.0, sep=4.0)

            anchored_box = AnchoredOffsetbox(
                loc="upper left",
                child=legend_packer,
                pad=0.0,
                borderpad=0.4,
                frameon=False,
                bbox_to_anchor=None,
                bbox_transform=ax.transAxes,
            )

            ax.add_artist(anchored_box)
        else:
            ncol = 2 if n_plotted > 4 else 1
            ax.legend(loc="upper left", ncol=ncol, framealpha=0.50)

        # Plot Inset Axis for Extended Training Horizon
        if show_inset:
            ax_inset = ax.inset_axes(list(inset_bounds))
            ax_inset.set_facecolor("#FFFFFF")

            # Re-plot all curves on the inset
            for d in curve_data:
                style = d["style"]
                color = style["color"]
                linestyle = style["linestyle"]
                linewidth = style["linewidth"] * 0.85
                zorder = style["zorder"]
                x = d["x"]
                mean = d["mean"]
                std = d["std"]

                ax_inset.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    zorder=zorder - 1,
                )
                ax_inset.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    zorder=zorder,
                )

            # Highlight the common budget region on the inset
            if 0 < common_max_x < global_max_x:
                ax_inset.axvspan(
                    0,
                    common_max_x,
                    color="#444444",
                    alpha=0.07,
                    linestyle=":",
                    linewidth=0.5,
                    zorder=0,
                )

            if inset_hline is not None:
                ax_inset.axhline(
                    y=inset_hline,
                    color="#555555",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.8,
                    zorder=2,
                )

            ax_inset.set_xlim(0, global_max_x)
            if ylim is not None:
                ax_inset.set_ylim(ylim[0], ylim[1])

            # Inset title placed cleanly inside top-center
            if inset_title:
                inset_title_text = (
                    f"\\textbf{{{inset_title}}}" if use_tex else inset_title
                )
                inset_title_props = dict(
                    fontsize=8.0,
                    color="#333333",
                    ha="center",
                    va="top",
                    transform=ax_inset.transAxes,
                    zorder=10,
                )
                if not use_tex:
                    inset_title_props["fontweight"] = "bold"
                ax_inset.text(0.50, 0.90, inset_title_text, **inset_title_props)

            # Clean and legible tick formatting
            if global_max_x >= 12.0:
                ax_inset.set_xticks(np.arange(0, int(np.ceil(global_max_x)) + 1, 4))
            else:
                ax_inset.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))

            if ylim == (0.0, 20.0):
                ax_inset.set_yticks([0.0, 10.0, 20.0])
            else:
                ax_inset.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))

            ax_inset.tick_params(
                axis="both", which="major", labelsize=7.0, pad=1.5, length=2.0
            )
            ax_inset.grid(
                True, linestyle="--", linewidth=0.5, alpha=0.65, color="#E0E0E0"
            )
            for spine in ax_inset.spines.values():
                spine.set_edgecolor("#999999")
                spine.set_linewidth(0.75)

    return n_plotted


def save_figure(
    fig: plt.Figure, output_dir: Path, base_filename: str, dpi: int
) -> None:
    """Save figure in both vector (PDF) and high-res raster (PNG) formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = output_dir / f"{base_filename}.{ext}"
        fig.savefig(out_path, dpi=dpi)
        print(f"Saved: {out_path}")
    plt.close(fig)


def plot_scenario(
    scenario: str,
    root: Path,
    output_dir: Path,
    dpi: int,
    smoothing_window: int,
    show_raw_mean: bool,
    figsize: tuple[float, float],
    ylim: tuple[float, float] = (0.0, 20.0),
    ytick_interval: float = 5.0,
    xlim: tuple[float, float] | None = None,
    inset_mode: str = "auto",
    inset_bounds: tuple[float, float, float, float] = (0.54, 0.10, 0.37, 0.32),
    inset_title: str = "Full Horizon",
    inset_hline: float | None = 12.0,
    keywords: tuple[str, ...] = (),
    use_tex: bool = False,
) -> None:
    scenario_dir = (
        root
        if root.name == scenario and discover_method_tags(root, keywords)
        else root / scenario
    )
    if not scenario_dir.exists():
        print(f"[skip] {scenario}: {scenario_dir} not found")
        return

    method_tags = discover_method_tags(scenario_dir, keywords)
    if not method_tags:
        print(f"[skip] {scenario}: no methods discovered")
        return

    # Main combined reward curve figure
    fig, ax = plt.subplots(figsize=figsize)
    n_plotted = plot_single_panel(
        ax,
        scenario_dir,
        method_tags,
        smoothing_window,
        show_raw_mean,
        keywords=keywords,
        use_tex=use_tex,
        ylim=ylim,
        ytick_interval=ytick_interval,
        xlim=xlim,
        inset_mode=inset_mode,
        inset_bounds=inset_bounds,
        inset_title=inset_title,
        inset_hline=inset_hline,
    )
    if n_plotted > 0:
        save_figure(fig, output_dir, f"figure_reward_{scenario}", dpi)
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"Path not found: {args.path}")
    if not args.path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {args.path}")
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = args.path / "00_fig"
    if args.keywords:
        output_dir /= "-".join(args.keywords)

    # Apply academic Matplotlib style parameters
    plt.rcParams.update(get_academic_rc_params(use_tex=args.use_tex))

    scenarios = discover_scenarios(args.path, args.keywords)
    if not scenarios:
        print(f"No scenario directories found under {args.path}")
        return

    for scenario in scenarios:
        print(f"=== Processing scenario: {scenario} ===")
        plot_scenario(
            scenario=scenario,
            root=args.path,
            output_dir=output_dir,
            dpi=args.dpi,
            smoothing_window=args.smoothing_window,
            show_raw_mean=args.show_raw_mean,
            figsize=tuple(args.figsize),
            ylim=tuple(args.ylim),
            ytick_interval=args.ytick_interval,
            xlim=tuple(args.xlim) if args.xlim is not None else None,
            inset_mode=args.inset,
            inset_bounds=tuple(args.inset_bounds),
            inset_title=args.inset_title,
            inset_hline=args.inset_hline,
            keywords=args.keywords,
            use_tex=args.use_tex,
        )


if __name__ == "__main__":
    main()
