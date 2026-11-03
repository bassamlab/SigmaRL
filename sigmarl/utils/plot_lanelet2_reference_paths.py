# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Plot the reference paths of a Lanelet2 scenario on the full OSM map to check that they are valid.

Writes to the output directory:
- path_<k>.png: one figure per reference path,
- all_paths_overlay.png: all reference paths in one map,
- all_paths_grid.png: all reference paths as small multiples in one figure,
- summary.csv: the checks below per path.

Checks per path (failed checks are printed and shown in red in the figure titles):
- route: consecutive lanelets are successors, the first has no predecessor, the last no successor,
- inside: every center-line point lies between the left and the right shared boundary,
- min width: narrowest corridor between the shared boundaries,
- min radius: tightest turn radius of the center line (heading change between consecutive segments over their
  length); below about 5 m a car cannot follow it,
- max jump: largest step between consecutive boundary points, relative to the center-line step there.

Example (from the repository root):
    python sigmarl/utils/plot_lanelet2_reference_paths.py --scenario_type round_location0_1to18
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from sigmarl.constants import SCENARIOS
from sigmarl.parse_lanelet2 import ParseLanelet2

WAY_STYLES = {
    # (type, subtype) -> line style; first match on type + subtype, then on type only
    ("line_thin", "solid"): dict(color="0.45", linestyle="-", linewidth=0.6),
    ("line_thin", "dashed"): dict(color="0.45", linestyle=(0, (4, 3)), linewidth=0.6),
    ("road_border", None): dict(color="black", linestyle="-", linewidth=0.9),
    ("curbstone", None): dict(color="0.3", linestyle="-", linewidth=0.9),
    ("virtual", None): dict(color="0.7", linestyle=(0, (1, 2)), linewidth=0.6),
}
DEFAULT_WAY_STYLE = dict(color="0.6", linestyle="-", linewidth=0.5)

MIN_RADIUS_M = 5.0  # Tighter center-line turns fail the check (about the minimum turning radius of a car)
MAX_JUMP_RATIO = (
    4.0  # Larger boundary steps (relative to the center-line step) fail the check
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scenario_type",
        type=str,
        default="round_location0_1to18",
        help="A scenario with 'map_format': 'lanelet2' in SCENARIOS.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/reference_paths_<scenario_type>).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def check_reference_path(map_parser, path):
    """Return the checks of one reference path as a dict (lengths and widths in real meters)."""
    lanelets = map_parser.lanelets_all
    ids = path["lanelet_IDs"]
    is_route_ok = (
        not lanelets[ids[0] - 1]["predecessor"]
        and not lanelets[ids[-1] - 1]["successor"]
        and all(b in lanelets[a - 1]["successor"] for a, b in zip(ids[:-1], ids[1:]))
    )

    center = path["center_line"]
    left = path["left_boundary_shared"]
    right = path["right_boundary_shared"]
    tangent = torch.gradient(center, dim=0)[0]

    def cross(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    n_outside = int(
        ((cross(tangent, left - center) <= 0) | (cross(tangent, right - center) >= 0))
        .sum()
        .item()
    )

    real_per_map_unit = 1.0 / map_parser._map_factor
    yaw = path["center_line_yaw"]
    kinks = torch.atan2(torch.sin(yaw.diff()), torch.cos(yaw.diff())).abs()
    center_step = center.diff(dim=0).norm(dim=1)
    radii = (center_step[:-1] + center_step[1:]) / 2 / kinks.clamp(min=1e-9)
    jump_ratio = max(
        (boundary.diff(dim=0).norm(dim=1) / center_step.clamp(min=1e-9)).max().item()
        for boundary in (left, right)
    )
    checks = {
        "n_lanelets": len(ids),
        "length_m": center_step.sum().item() * real_per_map_unit,
        "route_ok": is_route_ok,
        "n_points_outside": n_outside,
        "min_width_m": (left - right).norm(dim=1).min().item() * real_per_map_unit,
        "min_radius_m": radii.min().item() * real_per_map_unit,
        "min_radius_point": int(radii.argmin().item()) + 1,
        "max_jump_ratio": jump_ratio,
        "osm_ids": " ".join(str(map_parser.lanelet_osm_ids[i - 1]) for i in ids),
    }
    checks["is_valid"] = (
        is_route_ok
        and n_outside == 0
        and checks["min_radius_m"] >= MIN_RADIUS_M
        and jump_ratio <= MAX_JUMP_RATIO
    )
    return checks


def plot_background(ax, map_parser):
    """Draw all ways of the OSM file and fill the road lanelets."""
    for lanelet in map_parser.lanelets_all:
        outline = torch.cat(
            [lanelet["left_boundary"], lanelet["right_boundary"].flip(0)]
        )
        ax.add_patch(
            Polygon(
                outline.numpy(),
                closed=True,
                facecolor="0.93",
                edgecolor="none",
                zorder=0,
            )
        )
    for way in map_parser._ways.values():
        points = np.array([map_parser._nodes[n] for n in way["nodes"]])
        tags = way["tags"]
        style = WAY_STYLES.get(
            (tags.get("type"), tags.get("subtype")),
            WAY_STYLES.get((tags.get("type"), None), DEFAULT_WAY_STYLE),
        )
        ax.plot(points[:, 0], points[:, 1], zorder=1, **style)


def plot_direction_arrows(ax, center, color, n_arrows=6, zorder=4):
    """Draw arrows along a center line in its direction."""
    for k in np.linspace(2, len(center) - 3, n_arrows).astype(int):
        ax.annotate(
            "",
            xy=center[k + 1].tolist(),
            xytext=center[k - 1].tolist(),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, mutation_scale=8),
            zorder=zorder,
        )


def plot_reference_path(ax, map_parser, path, is_label_lanelets=True):
    """Draw the corridor, boundaries, center line, and lanelet IDs of one reference path."""
    left = path["left_boundary_shared"]
    right = path["right_boundary_shared"]
    center = path["center_line"]
    corridor = torch.cat([left, right.flip(0)])
    ax.add_patch(
        Polygon(
            corridor.numpy(),
            closed=True,
            facecolor="tab:blue",
            alpha=0.2,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.plot(
        left[:, 0],
        left[:, 1],
        color="tab:blue",
        linewidth=1.0,
        zorder=3,
        label="left shared boundary",
    )
    ax.plot(
        right[:, 0],
        right[:, 1],
        color="tab:red",
        linewidth=1.0,
        zorder=3,
        label="right shared boundary",
    )
    ax.plot(
        center[:, 0],
        center[:, 1],
        color="black",
        linestyle="--",
        linewidth=0.8,
        zorder=3,
        label="center line",
    )
    plot_direction_arrows(ax, center, "black")
    ax.plot(
        *center[0], marker="^", color="tab:green", markersize=7, zorder=5, label="start"
    )
    ax.plot(
        *center[-1], marker="s", color="tab:red", markersize=6, zorder=5, label="end"
    )
    if is_label_lanelets:
        for i in path["lanelet_IDs"]:
            lanelet_center = map_parser.lanelets_all[i - 1]["center_line"]
            ax.text(
                *lanelet_center[len(lanelet_center) // 2].tolist(),
                str(map_parser.lanelet_osm_ids[i - 1]),
                fontsize=4,
                ha="center",
                va="center",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )


def format_title(k, checks):
    return (
        f"Path {k}: {checks['n_lanelets']} lanelets, {checks['length_m']:.0f} m, "
        f"min width {checks['min_width_m']:.1f} m, min radius {checks['min_radius_m']:.1f} m\n"
        f"route {'OK' if checks['route_ok'] else 'BROKEN'}, "
        f"{checks['n_points_outside']} center points outside, max jump ratio {checks['max_jump_ratio']:.1f}"
    )


def setup_axes(ax, map_parser):
    ax.set_aspect("equal")
    ax.set_xlim(0, map_parser.bounds["world_x_dim"])
    ax.set_ylim(0, map_parser.bounds["world_y_dim"])
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    args = parse_args()
    output_dir = Path(
        args.output_dir or f"outputs/reference_paths_{args.scenario_type}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    assert (
        SCENARIOS[args.scenario_type].get("map_format") == "lanelet2"
    ), "Not a Lanelet2 scenario."
    map_parser = ParseLanelet2(scenario_type=args.scenario_type, device="cpu")
    paths = map_parser.reference_paths
    checks_all = [check_reference_path(map_parser, path) for path in paths]
    figsize = (
        8,
        8 * map_parser.bounds["world_y_dim"] / map_parser.bounds["world_x_dim"],
    )

    # One figure per reference path
    for k, (path, checks) in enumerate(zip(paths, checks_all)):
        fig, ax = plt.subplots(figsize=figsize)
        plot_background(ax, map_parser)
        plot_reference_path(ax, map_parser, path)
        ax.plot(
            *path["center_line"][checks["min_radius_point"]],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="magenta",
            markersize=10,
            zorder=6,
            label="tightest turn",
        )
        setup_axes(ax, map_parser)
        ax.set_title(
            format_title(k, checks),
            fontsize=8,
            color="black" if checks["is_valid"] else "red",
        )
        ax.legend(loc="lower left", fontsize=6)
        fig.savefig(output_dir / f"path_{k:02d}.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[{k + 1}/{len(paths)}] saved path_{k:02d}.png", end="\r")
    print()

    # All reference paths in one map
    fig, ax = plt.subplots(figsize=(figsize[0] * 1.5, figsize[1] * 1.5))
    plot_background(ax, map_parser)
    colors = plt.cm.turbo(np.linspace(0, 1, len(paths)))
    for k, (path, color) in enumerate(zip(paths, colors)):
        center = path["center_line"]
        ax.plot(
            center[:, 0], center[:, 1], color=color, linewidth=0.8, alpha=0.8, zorder=3
        )
        plot_direction_arrows(ax, center, color, n_arrows=3)
        ax.plot(*center[0], marker="^", color="tab:green", markersize=6, zorder=5)
        ax.plot(*center[-1], marker="s", color="tab:red", markersize=5, zorder=5)
        ax.text(
            *center[len(center) // 2].tolist(),
            str(k),
            fontsize=5,
            color=color,
            zorder=6,
        )
    setup_axes(ax, map_parser)
    n_valid = sum(c["is_valid"] for c in checks_all)
    ax.set_title(
        f"{args.scenario_type}: {len(paths)} reference paths ({n_valid} pass all checks); "
        "center lines, green ^ start, red square end",
        fontsize=9,
    )
    fig.savefig(output_dir / "all_paths_overlay.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # All reference paths as small multiples
    n_cols = 6
    n_rows = math.ceil(len(paths) / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3 * figsize[1] / figsize[0] + 0.5)
    )
    for ax in axs.flat:
        ax.axis("off")
    for k, (ax, path, checks) in enumerate(zip(axs.flat, paths, checks_all)):
        plot_background(ax, map_parser)
        plot_reference_path(ax, map_parser, path, is_label_lanelets=False)
        setup_axes(ax, map_parser)
        ax.set_title(
            f"{k}: {checks['n_lanelets']} lanelets, {checks['length_m']:.0f} m",
            fontsize=7,
            color="black" if checks["is_valid"] else "red",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "all_paths_grid.png", dpi=args.dpi)
    plt.close(fig)

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path"] + list(checks_all[0]))
        writer.writeheader()
        for k, checks in enumerate(checks_all):
            writer.writerow({"path": k, **checks})

    for k, checks in enumerate(checks_all):
        if not checks["is_valid"]:
            print(f"[WARNING] {format_title(k, checks)}".replace("\n", "; "))
    print(
        f"{n_valid}/{len(paths)} reference paths pass all checks. Figures and summary.csv in {output_dir}"
    )


if __name__ == "__main__":
    main()
