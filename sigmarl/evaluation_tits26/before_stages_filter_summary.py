# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize the filter benchmark (see before_stages_filter_benchmark.py).

Per (block, method, mode), metrics are averaged over seeds (mean +- standard
error) and written to <out_root>/summary_<block>.md and summary.csv.

Collision counts are vehicle-steps flagged by the scenario's collision checker
(a colliding vehicle respawns, so each count is one collision of one vehicle;
an inter-vehicle collision counts both vehicles). Flags within RESPAWN_STEPS
steps after a (re)spawn are excluded and listed separately ("spawn"), since
they come from the spawn and not from the filter. The exact rectangle-overlap
onsets of vehicle pairs (respawns excluded) cross-check the inter-vehicle
counts. CBF times: sequential is the actual wall-clock loop time per step;
parallelized is the per-agent parallel-execution estimate (equal to the
sequential time for centralized controllers).

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_tits26.before_stages_filter_summary [--out_root ...]
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from sigmarl.constants import AGENTS
from sigmarl.evaluation_tits26.before_stages_filter_benchmark import METHODS, REPO_ROOT
from sigmarl.pair_sat_barrier import PairSatBarrier

# A position jump larger than this in one step [m] is a (re)spawn
RESPAWN_JUMP = 3.0
RESPAWN_STEPS = 3
# A vehicle slower than this [m/s] is stopped
STOP_SPEED = 0.1
# A collision counts as "after an infeasible step" if a CBF-QP of the env
# needed slack or failed within this many steps before it
INFEASIBLE_WINDOW = 15
# Normalized target change above which the filter counts as intervening (as in
# sigmarl/evaluation_tits26/stage2_finetune_run.py)
INTERVENTION_TOL = 1e-3
# [m] Backup value below which a state counts as unrecoverable (outside the
# certified set); above the exact check's tolerance
UNRECOVERABLE_TOL = 1e-3

METRICS = [
    ("total_col", "Total col.", "{:.1f}"),
    ("veh_col", "Inter-veh. col.", "{:.1f}"),
    ("lane_col", "Lane col.", "{:.1f}"),
    ("overlap", "Exact overlaps", "{:.1f}"),
    ("spawn_col", "Spawn col.", "{:.1f}"),
    ("goals", "Goals", "{:.1f}"),
    ("throughput", "Throughput [veh/s]", "{:.3f}"),
    ("speed", "Speed [m/s]", "{:.2f}"),
    ("stopped", "Stopped [%]", "{:.1f}"),
    ("min_gap", "Min gap [m]", "{:.3f}"),
    ("qp_fail", "QP fail [1/1000]", "{:.2f}"),
    ("infeas", "Infeasible steps [%]", "{:.1f}"),
    ("col_after_infeas", "Col. after infeasible step [%]", "{:.0f}"),
    ("uncert", "Uncertified [%]", "{:.2f}"),
    ("shield", "Shield [1/1000]", "{:.2f}"),
    ("intervention", "Intervention [%]", "{:.1f}"),
    ("fallback", "Fallback [1/1000]", "{:.2f}"),
    ("min_backup", "Min backup value [m]", "{:.3f}"),
    ("outside_c", "Steps outside C [%]", "{:.2f}"),
    ("t_seq", "CBF seq. [ms]", "{:.1f}"),
    ("t_par", "CBF par. [ms]", "{:.1f}"),
    ("t_seq_p95", "CBF seq. p95 [ms]", "{:.1f}"),
]


def respawn_mask(positions: np.ndarray) -> np.ndarray:
    """[T, n] steps within RESPAWN_STEPS steps after a position jump (respawn)."""
    jump = np.linalg.norm(np.diff(positions, axis=0), axis=2) > RESPAWN_JUMP
    mask = np.zeros(positions.shape[:2], dtype=bool)
    mask[0] = True
    for k in range(1, RESPAWN_STEPS + 1):
        mask[k:] |= jump[: jump.shape[0] - k + 1]
    return mask


def run_metrics(run_dir: str) -> dict:
    """Metrics of one rollout folder."""
    summary = json.load(open(glob.glob(f"{run_dir}/evaluation_summary_*.json")[0]))
    stats = summary["statistics"]
    cbf = json.load(open(glob.glob(f"{run_dir}/cbf_statistics_*.json")[0]))
    td = torch.load(glob.glob(f"{run_dir}/out_td_*.td")[0], weights_only=False)
    info = td["next", "agents", "info"]
    pos = info["pos"][0].numpy()
    rot = info["rot"][0].numpy().reshape(pos.shape[:2])
    speed = np.linalg.norm(info["vel"][0].numpy(), axis=2)
    veh_flag = info["is_collision_with_agents"][0].numpy().reshape(pos.shape[:2])
    lane_flag = info["is_collision_with_lanelets"][0].numpy().reshape(pos.shape[:2])
    spawn = respawn_mask(pos)
    # 1 if any CBF-QP of the step needed slack or failed (see helper_training.py)
    infeasible = td["cbf_slack_active"][0].numpy() > 0.5

    poses = np.concatenate([pos, rot[..., None]], axis=2)
    gap_barrier = PairSatBarrier(AGENTS["length"], AGENTS["width"], 0.0)
    n = pos.shape[1]
    overlaps, min_gap = 0, np.inf
    onsets = []  # steps of collision onsets (exact overlaps and lane flags)
    for a in range(n):
        for b in range(a + 1, n):
            ok = ~(spawn[:, a] | spawn[:, b])
            if not ok.any():
                continue
            gap = gap_barrier.values(poses[:, a], poses[:, b])
            min_gap = min(min_gap, float(gap[ok].min()))
            hit = (gap < 0.0) & ok
            onset = np.nonzero(hit & ~np.r_[False, hit[:-1]])[0]
            overlaps += onset.size
            onsets.extend(onset.tolist())

    onsets.extend(np.nonzero(lane_flag.astype(bool) & ~spawn)[0].tolist())
    after_infeasible = [
        bool(infeasible[max(0, t - INFEASIBLE_WINDOW) : t + 1].any()) for t in onsets
    ]
    veh_col = int((veh_flag.astype(bool) & ~spawn).sum())
    lane_col = int((lane_flag.astype(bool) & ~spawn).sum())
    spawn_col = int(((veh_flag.astype(bool) | lane_flag.astype(bool)) & spawn).sum())
    # Filter interventions and backup values of agent-steps outside respawns
    if "cbf_dev_vel" in td.keys():
        dev = np.maximum(td["cbf_dev_vel"][0].numpy(), td["cbf_dev_steer"][0].numpy())
        intervention = 100.0 * float((dev[~spawn] > INTERVENTION_TOL).mean())
    else:
        intervention = float("nan")
    if "backup_value" in td.keys():
        backup = np.where(spawn, np.inf, td["backup_value"][0].numpy())
        min_backup = float(backup.min())
        outside_c = 100.0 * float((backup < -UNRECOVERABLE_TOL).any(axis=1).mean())
    else:
        min_backup = outside_c = float("nan")
    n_solves = max(1, cbf["qp_solves"])
    sat = cbf["lane_sat_stats"]
    agent_steps = pos.shape[0] * n
    return dict(
        total_col=veh_col + lane_col,
        veh_col=veh_col,
        lane_col=lane_col,
        overlap=overlaps,
        spawn_col=spawn_col,
        goals=stats["goal_reaching"]["agent_step_count"],
        throughput=stats["traffic_throughput"]["vehicles_per_second"],
        speed=stats["speed_mps"]["mean"],
        stopped=100.0 * float((speed[~spawn] < STOP_SPEED).mean()),
        min_gap=min_gap,
        qp_fail=1e3 * sum(cbf["qp_failures"].values()) / n_solves,
        infeas=100.0 * float(infeasible.mean()),
        col_after_infeas=float("nan"),  # pooled over seeds in main()
        n_onsets=len(onsets),
        n_onsets_after_infeasible=int(sum(after_infeasible)),
        uncert=(
            100.0 * sat.get("uncertified", 0) / sat["qp_calls"]
            if sat.get("qp_calls")
            else float("nan")
        ),
        shield=1e3 * cbf["shield_stats"].get("repaired_agent_steps", 0) / agent_steps,
        intervention=intervention,
        fallback=1e3 * sat.get("bh_fallback_agents", 0) / agent_steps,
        min_backup=min_backup,
        outside_c=outside_c,
        t_seq=cbf["time_cbf_sequential"]["mean_ms"],
        t_par=cbf["time_cbf_parallelized"]["mean_ms"],
        t_seq_p95=cbf["time_cbf_sequential"]["p95_ms"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_root",
        default=os.path.join(
            REPO_ROOT, "outputs", "tits26", "before_stages_filter_benchmark"
        ),
    )
    args = parser.parse_args()

    groups = {}
    for run_dir in sorted(glob.glob(f"{args.out_root}/*/*/*/seed*")):
        if not glob.glob(f"{run_dir}/cbf_statistics_*.json"):
            continue
        block, method, mode = run_dir.split(os.sep)[-4:-1]
        groups.setdefault((block, method, mode), []).append(run_dir)

    rows = []
    for (block, method, mode), run_dirs in tqdm(sorted(groups.items())):
        per_seed = [run_metrics(d) for d in run_dirs]
        row = dict(block=block, method=method, mode=mode, n_seeds=len(per_seed))
        for key, _, _ in METRICS:
            if key == "col_after_infeas":
                n_onsets = sum(m["n_onsets"] for m in per_seed)
                n_after = sum(m["n_onsets_after_infeasible"] for m in per_seed)
                value = 100.0 * n_after / n_onsets if n_onsets else float("nan")
                row[key], row[key + "_se"], row[key + "_min"] = value, np.nan, value
                continue
            values = np.array([m[key] for m in per_seed], dtype=np.float64)
            row[key] = float(np.mean(values))
            row[key + "_se"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else float("nan")
            )
            row[key + "_min"] = float(np.min(values))
        rows.append(row)

    order = {name: k for k, name in enumerate(METHODS)}
    rows.sort(
        key=lambda r: (
            r["block"],
            r["mode"],
            order.get(r["method"].split("_lam")[0].split("_kappa")[0], 99),
            r["method"],
        )
    )
    with open(f"{args.out_root}/summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for block in sorted({r["block"] for r in rows}):
        lines = [
            f"# SOTA CBF comparison, block {block}",
            "",
            "Mean over seeds (+- standard error). Min gap and min backup value: "
            "smallest over seeds.",
            "",
            "| Mode | Method | Seeds | " + " | ".join(t for _, t, _ in METRICS) + " |",
            "|---" * (3 + len(METRICS)) + "|",
        ]
        for r in rows:
            if r["block"] != block:
                continue
            cells = []
            for key, _, fmt in METRICS:
                if key in ("min_gap", "min_backup"):
                    cells.append(fmt.format(r[key + "_min"]))
                elif np.isnan(r[key + "_se"]):
                    cells.append(fmt.format(r[key]))
                else:
                    cells.append(f"{fmt.format(r[key])} ± {fmt.format(r[key + '_se'])}")
            lines.append(
                f"| {r['mode']} | {r['method']} | {r['n_seeds']} | "
                + " | ".join(cells)
                + " |"
            )
        text = "\n".join(lines) + "\n"
        open(f"{args.out_root}/summary_{block}.md", "w").write(text)
        print(text)


if __name__ == "__main__":
    main()
