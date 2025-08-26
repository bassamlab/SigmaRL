"""Run the following in the terminal
python sigmarl/eva_at25/marl_evaluation.py \
  --td_path "checkpoints/at25/sigmarl/seed3/init2-1.td" \
  --ref_path_ids "6|2|3" \
  --scenario CPM_entire \
  --per_unit_m 100 \
  --max_steps 1800 \
  --make_plots \
  --rotate180
"""

import argparse
import os
import json
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.path import Path

from sigmarl.helper_scenario import (
    get_distances_between_agents,
    interX,
    get_rectangle_vertices,
    get_perpendicular_distances,
)
from sigmarl.parse_xml import ParseXML


def total_agent_distance_m(pos: torch.Tensor) -> float:
    diffs = pos[1:] - pos[:-1]
    step_dist = diffs.norm(dim=-1)
    return float(step_dist.sum().item())


def per_agent_step_distance(pos: torch.Tensor) -> torch.Tensor:
    diffs = torch.zeros_like(pos[..., 0])
    diffs[1:] = (pos[1:] - pos[:-1]).norm(dim=-1)
    return diffs


def get_agent_agent_collision_timesteps(
    a2a_distances: torch.Tensor, n_1: int = 3, n_2: int = 5
) -> dict:
    num_steps, num_agents, _ = a2a_distances.shape
    collisions = {i: [] for i in range(num_agents)}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            distance_ij = a2a_distances[:, i, j]
            neg_count = 0
            pos_count = 0
            in_collision = False
            for t, d in enumerate(distance_ij):
                d = float(d)
                if d < 0:
                    neg_count += 1
                    pos_count = 0
                    if not in_collision and neg_count >= n_1:
                        collisions[i].append(t)
                        collisions[j].append(t)
                        in_collision = True
                else:
                    pos_count += 1
                    neg_count = 0
                    if in_collision and pos_count >= n_2:
                        in_collision = False
    return collisions


def create_lane_polygon(left, right):
    lane_polygon = torch.cat([left, torch.flip(right, dims=[0])], dim=0)
    return Path(lane_polygon.cpu().numpy())


def is_outside_polygon(polygon: Path, footprint: torch.Tensor) -> bool:
    return not polygon.contains_points(footprint.cpu().numpy()).all()


def get_boundary_violation_timesteps(
    vertices: torch.Tensor, boundaries: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict:
    num_steps, num_agents, _, _ = vertices.shape
    violation_timesteps = {i: [] for i in range(num_agents)}
    for i in range(num_agents):
        left, right = boundaries[i]
        lane_poly = create_lane_polygon(left, right)
        for t in range(num_steps):
            footprint = vertices[t, i]
            intersects = interX(footprint.unsqueeze(0), left.unsqueeze(0)) or interX(
                footprint.unsqueeze(0), right.unsqueeze(0)
            )
            outside = is_outside_polygon(lane_poly, footprint)
            if intersects or outside:
                violation_timesteps[i].append(t)
    return violation_timesteps


def parse_ref_ids(s: str) -> List[int]:
    s = s.replace(" ", "").replace("|", ",")
    return [int(x) for x in s.split(",") if x != ""]


# --- Rotation helpers ---
def rot180_xy_t(xy_t: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
    return torch.stack([2 * cx - xy_t[..., 0], 2 * cy - xy_t[..., 1]], dim=-1)


def rot180_angle_t(theta_t: torch.Tensor) -> torch.Tensor:
    return (theta_t + torch.pi) % (2 * torch.pi)


def map_center_from_lanelets(px: "ParseXML") -> tuple[float, float]:
    xs, ys = [], []
    for ll in px.lanelets_all:
        for k in ("left_boundary", "right_boundary"):
            arr = ll[k]
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])
    x_min = float(torch.cat(xs).min().item())
    x_max = float(torch.cat(xs).max().item())
    y_min = float(torch.cat(ys).min().item())
    y_max = float(torch.cat(ys).max().item())
    print(f"Map center {(0.5 * (x_min + x_max), 0.5 * (y_min + y_max))}")
    return (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))


def evaluate_single_run(
    td_path: str,
    ref_path_ids: List[int],
    scenario: str,
    veh_width: float,
    veh_length: float,
    per_unit_m: float,
    max_steps: int | None = None,
    n1: int = 3,
    n2: int = 5,
    make_plots: bool = False,
    rotate180: bool = False,
    center_x: float | None = None,
    center_y: float | None = None,
):
    # Load
    td = torch.load(td_path, weights_only=False)
    num_steps_full = td["pos"].shape[0]
    num_steps = (
        min(max_steps, num_steps_full) if max_steps is not None else num_steps_full
    )

    w = veh_width
    l = veh_length

    pos = td["pos"][:num_steps]
    rot = td["rot"][:num_steps]
    vel = td["vel"][:num_steps]
    speed = vel.norm(dim=-1)

    num_steps, num_agents, _ = pos.shape

    # Map and ref paths
    px = ParseXML(scenario_type=scenario)
    ref_paths = [px.reference_paths[rid] for rid in ref_path_ids]
    if len(ref_paths) < num_agents:
        raise ValueError(
            f"Need at least {num_agents} ref_path_ids; got {len(ref_paths)}"
        )
    center_lines = [rp["center_line"] for rp in ref_paths[:num_agents]]
    left_boundaries = [rp["left_boundary"] for rp in ref_paths[:num_agents]]
    right_boundaries = [rp["right_boundary"] for rp in ref_paths[:num_agents]]

    # Apply rotation if requested
    if rotate180:
        print("[INFO] Rotate 180")
        if (center_x is None) or (center_y is None):
            cx, cy = map_center_from_lanelets(px)
        else:
            cx, cy = float(center_x), float(center_y)

        pos = rot180_xy_t(pos, cx, cy)
        rot = rot180_angle_t(rot)

        center_lines = [rot180_xy_t(cl, cx, cy) for cl in center_lines]
        left_boundaries = [rot180_xy_t(lb, cx, cy) for lb in left_boundaries]
        right_boundaries = [rot180_xy_t(rb, cx, cy) for rb in right_boundaries]

    # Geometry
    is_closed_shape = True
    vertices = torch.zeros(num_steps, num_agents, 5 if is_closed_shape else 4, 2)
    a2ref_distances = torch.zeros(num_steps, num_agents)
    for i_a in range(num_agents):
        vertices[:, i_a] = get_rectangle_vertices(
            pos[:, i_a, :], rot[:, i_a].unsqueeze(-1), w, l, True
        )
        a2ref_distances[:, i_a], _ = get_perpendicular_distances(
            pos[:, i_a], center_lines[i_a]
        )

    # A2A distances and events
    a2a_distances = get_distances_between_agents(
        vertices, distance_type="mtv", is_set_diagonal=False
    )
    a2a_distances.diagonal(dim1=-2, dim2=-1).fill_(10)
    a2a_collision_steps = get_agent_agent_collision_timesteps(
        a2a_distances, n_1=n1, n_2=n2
    )

    # Lane violation timesteps (no hysteresis)
    boundary_collision_steps = get_boundary_violation_timesteps(
        vertices, list(zip(left_boundaries, right_boundaries))
    )

    # Distance-normalized metrics
    total_a2a = sum(len(steps) for steps in a2a_collision_steps.values())
    total_dist_m = total_agent_distance_m(pos)
    units = total_dist_m / per_unit_m if total_dist_m > 0 else float("nan")
    cra_a_per_unit = total_a2a / units if units else float("nan")

    step_dist_m = per_agent_step_distance(pos)
    lane_violation_m = 0.0
    for i in range(num_agents):
        for t in boundary_collision_steps[i]:
            lane_violation_m += float(step_dist_m[t, i])
    lane_violation_per_unit = lane_violation_m / units if units else float("nan")

    cd_vals = []
    for i in range(num_agents):
        d_i, _ = get_perpendicular_distances(pos[:, i], center_lines[i])
        cd_vals.append(float(d_i.mean().item()))
    cd_mean = float(np.mean(cd_vals)) if len(cd_vals) else np.nan
    as_mean = float(speed.mean().item())

    metrics = {
        "td_path": td_path,
        "num_steps": int(num_steps),
        "num_agents": int(num_agents),
        "per_unit_m": float(per_unit_m),
        "raw_a2a_event_count": int(total_a2a),
        "total_distance_m": float(total_dist_m),
        "lane_violation_m": float(lane_violation_m),
        "CRA_A_per_unit": float(cra_a_per_unit),
        "LaneViolation_per_unit": float(lane_violation_per_unit),
        "CD_mean_m": float(cd_mean),
        "AS_mean_mps": float(as_mean),
    }

    if make_plots:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 11,
                "axes.titlesize": 11,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "font.family": "serif",
                "text.usetex": False,
            }
        )
        plt.figure(figsize=(10, 6))
        for lanelet in px.lanelets_all:
            left_bound = lanelet["left_boundary"]
            right_bound = lanelet["right_boundary"]
            if rotate180:
                left_bound = rot180_xy_t(left_bound, cx, cy)
                right_bound = rot180_xy_t(right_bound, cx, cy)
            left_line_marking = lanelet["left_line_marking"]
            right_line_marking = lanelet["right_line_marking"]
            plt.plot(
                left_bound[:, 0],
                left_bound[:, 1],
                linestyle="--" if left_line_marking == "dashed" else "-",
                color="grey",
                linewidth=0.5,
            )
            plt.plot(
                right_bound[:, 0],
                right_bound[:, 1],
                linestyle="--" if right_line_marking == "dashed" else "-",
                color="grey",
                linewidth=0.5,
            )
        plt.scatter(pos[:, 0, 0], pos[:, 0, 1], label="veh 1")
        if num_agents > 1:
            plt.scatter(pos[:, 1, 0], pos[:, 1, 1], label="veh 2")
        if num_agents > 2:
            plt.scatter(pos[:, 2, 0], pos[:, 2, 1], label="veh 3")
        for i, cl in enumerate(center_lines[:num_agents]):
            plt.plot(cl[:, 0], cl[:, 1], label=f"ref {i+1}", linewidth=1.0)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.gca().set_aspect("equal")
        plt.legend()
        plt.tight_layout()
        out_pdf = os.path.splitext(os.path.basename(td_path))[0] + "_footprint.pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved {out_pdf}")
        plt.close()

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--td_path", type=str, required=True)
    ap.add_argument("--ref_path_ids", type=str, required=True)
    ap.add_argument("--scenario", type=str, default="CPM_entire")
    ap.add_argument("--veh_width", type=float, default=0.107)
    ap.add_argument("--veh_length", type=float, default=0.220)
    ap.add_argument("--per_unit_m", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=1800)
    ap.add_argument("--n1", type=int, default=3)
    ap.add_argument("--n2", type=int, default=5)
    ap.add_argument("--make_plots", action="store_true")
    ap.add_argument(
        "--rotate180", action="store_true", help="Rotate run by 180 degrees"
    )
    ap.add_argument(
        "--center_x", type=float, default=None, help="Optional rotation center X"
    )
    ap.add_argument(
        "--center_y", type=float, default=None, help="Optional rotation center Y"
    )
    ap.add_argument("--json_out", type=str, default=None)
    args = ap.parse_args()

    ref_ids = parse_ref_ids(args.ref_path_ids)
    res = evaluate_single_run(
        td_path=args.td_path,
        ref_path_ids=ref_ids,
        scenario=args.scenario,
        veh_width=args.veh_width,
        veh_length=args.veh_length,
        per_unit_m=args.per_unit_m,
        max_steps=args.max_steps,
        n1=args.n1,
        n2=args.n2,
        make_plots=args.make_plots,
        rotate180=args.rotate180,
        center_x=args.center_x,
        center_y=args.center_y,
    )
    print(json.dumps(res, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
