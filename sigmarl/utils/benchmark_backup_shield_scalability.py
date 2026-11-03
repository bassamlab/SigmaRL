"""Benchmark the timing overhead of the exact backup shield against agent count.

All runs use the rectangle_sat lane barrier, which the shield requires. For
fully decentralized runs, the sequential time is the wall-clock CBF loop time,
and the parallelized time is a rough estimate: the loop time divided by the
number of agents.

Example:
    python sigmarl/utils/benchmark_backup_shield_scalability.py \
        --agent_counts 4 \
        --steps 301 \
        --seed 1 \
        --is_save_simulation_video \
        --is_visu_result \
        --output_dir outputs/20260527/backup_shield_scalability
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_counts", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--scenario_type",
        choices=["auto", "cpm_mixed", "cpm_entire"],
        default="auto",
        help=(
            "Scenario used for all runs. The default 'auto' follows the repo "
            "convention: cpm_mixed for fewer than 7 agents and cpm_entire for "
            "7 or more agents."
        ),
    )
    parser.add_argument(
        "--is_save_simulation_video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward simulation-video saving to main_testing_clf_cbf_qp.py.",
    )
    parser.add_argument(
        "--is_visu_result",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward result visualization to main_testing_clf_cbf_qp.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/backup_shield_scalability"),
    )
    return parser.parse_args()


def scenario_type_for_agents(args, n_agents: int) -> str:
    """Select the benchmark scenario using the repo-wide default convention."""
    if args.scenario_type != "auto":
        return args.scenario_type
    return "cpm_mixed" if n_agents < 7 else "cpm_entire"


def mode_specs():
    return [
        {
            "mode": "centralized_baseline",
            "architecture": "centralized",
            "backup": "none",
            "extra": ["--max_group_size", "none", "--no-is_exact_backup_shield"],
            "baseline_key": "centralized",
        },
        {
            "mode": "centralized_shield",
            "architecture": "centralized",
            "backup": "shield",
            "extra": ["--max_group_size", "none", "--is_exact_backup_shield"],
            "baseline_key": "centralized",
        },
        {
            "mode": "fd_baseline",
            "architecture": "fully_decentralized",
            "backup": "none",
            "extra": ["--max_group_size", "1", "--no-is_exact_backup_shield"],
            "baseline_key": "fully_decentralized",
        },
        {
            "mode": "fd_shield",
            "architecture": "fully_decentralized",
            "backup": "shield",
            "extra": ["--max_group_size", "1", "--is_exact_backup_shield"],
            "baseline_key": "fully_decentralized",
        },
    ]


def td_filename(scenario_type: str, n_agents: int, seed: int, is_fd: bool) -> str:
    group = "1_interaction_risk" if is_fd else "None"
    return (
        f"out_td_{scenario_type}"
        f"_agents{n_agents}"
        f"_max_group_size{group}"
        "_proxsuite"
        f"_seed{seed}.td"
    )


def summarize_td(path: Path, architecture: str, n_agents: int) -> dict:
    td = torch.load(path, map_location="cpu", weights_only=False)
    time_cbf = td.get("time_cbf").float().reshape(-1)
    finite_time = time_cbf[torch.isfinite(time_cbf)]
    is_fully_decentralized = architecture == "fully_decentralized"
    parallelized_scale = float(n_agents if is_fully_decentralized else 1)
    finite_time_parallelized_initial = finite_time / parallelized_scale
    metrics = {
        "transition_steps": int(time_cbf.numel()),
        "time_cbf_loop_mean_ms": float(finite_time.mean().item() * 1e3),
        "time_cbf_loop_p95_ms": float(torch.quantile(finite_time, 0.95).item() * 1e3),
        "time_cbf_loop_max_ms": float(finite_time.max().item() * 1e3),
        "time_cbf_parallelized_mean_ms": float(
            finite_time_parallelized_initial.mean().item() * 1e3
        ),
        "time_cbf_parallelized_p95_ms": float(
            torch.quantile(finite_time_parallelized_initial, 0.95).item() * 1e3
        ),
        "time_cbf_parallelized_max_ms": float(
            finite_time_parallelized_initial.max().item() * 1e3
        ),
        "time_cbf_parallelized_scale": parallelized_scale,
        "time_cbf_timing_interpretation": (
            "parallelized_per_agent_estimate"
            if is_fully_decentralized
            else "joint_wall_time"
        ),
    }
    metrics["time_cbf_mean_ms"] = metrics["time_cbf_parallelized_mean_ms"]
    metrics["time_cbf_p95_ms"] = metrics["time_cbf_parallelized_p95_ms"]
    metrics["time_cbf_max_ms"] = metrics["time_cbf_parallelized_max_ms"]
    for name, key in [
        ("collisions_agents", ("agents", "info", "is_collision_with_agents")),
        ("collisions_lane", ("agents", "info", "is_collision_with_lanelets")),
        ("deadlocks", ("agents", "info", "is_deadlocked")),
    ]:
        value = td.get(key)
        metrics[name] = int(value.sum().item()) if value is not None else -1

    pos = td.get(("agents", "info", "pos"))
    if pos is not None:
        if pos.ndim == 4:
            pos = pos[0]
        min_dist = float("inf")
        for t in range(pos.shape[0]):
            for i, j in itertools.combinations(range(pos.shape[1]), 2):
                dist = float(torch.linalg.norm(pos[t, i] - pos[t, j]).item())
                min_dist = min(min_dist, dist)
        metrics["min_center_distance_m"] = min_dist
    else:
        metrics["min_center_distance_m"] = float("nan")
    return metrics


def run_case(args, spec: dict, n_agents: int) -> dict:
    run_dir = args.output_dir / spec["mode"] / f"agents{n_agents}"
    run_dir.mkdir(parents=True, exist_ok=True)
    is_fd = spec["architecture"] == "fully_decentralized"
    scenario_type = scenario_type_for_agents(args, n_agents)
    command = [
        sys.executable,
        "main_testing_clf_cbf_qp.py",
        "--scenario_type",
        scenario_type,
        "--n_agents",
        str(n_agents),
        "--seed",
        str(args.seed),
        "--max_steps",
        str(args.steps),
        "--no-is_real_time_rendering",
        "--output_dir",
        str(run_dir),
        "--lane_cbf_footprint",
        "rectangle_sat",
        *spec["extra"],
    ]
    command.append(
        "--is_save_simulation_video"
        if args.is_save_simulation_video
        else "--no-is_save_simulation_video"
    )
    command.append("--is_visu_result" if args.is_visu_result else "--no-is_visu_result")
    t0 = time.time()
    row = {
        "mode": spec["mode"],
        "architecture": spec["architecture"],
        "backup": spec["backup"],
        "baseline_key": spec["baseline_key"],
        "n_agents": n_agents,
        "scenario_type": scenario_type,
        "status": "ok",
        "wall_time_s": 0.0,
    }
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=args.timeout)
        row["wall_time_s"] = time.time() - t0
        row["returncode"] = process.returncode
        log_path = run_dir / "run.log"
        log_path.write_text(stdout or "", encoding="utf-8")
        if process.returncode != 0:
            row["status"] = "failed"
            row["error"] = f"returncode={process.returncode}"
            return row
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
        row["wall_time_s"] = time.time() - t0
        row["returncode"] = -1
        row["status"] = "timeout"
        row["error"] = f"timeout_after={args.timeout}"
        timeout_output = stdout or exc.stdout or ""
        if isinstance(timeout_output, bytes):
            timeout_output = timeout_output.decode("utf-8", errors="replace")
        (run_dir / "run.log").write_text(timeout_output, encoding="utf-8")
        return row

    td_path = run_dir / td_filename(scenario_type, n_agents, args.seed, is_fd)
    if not td_path.exists():
        row["status"] = "missing_td"
        row["error"] = str(td_path)
        return row
    row.update(summarize_td(td_path, spec["architecture"], n_agents))
    return row


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row["backup"] == "none":
            baselines[(row["baseline_key"], row["n_agents"])] = row[
                "time_cbf_parallelized_mean_ms"
            ]

    for row in rows:
        baseline = baselines.get((row["baseline_key"], row["n_agents"]))
        if row.get("status") == "ok" and baseline and baseline > 0:
            row["mean_parallelized_overhead_ratio"] = (
                row["time_cbf_parallelized_mean_ms"] / baseline
            )
            row["mean_parallelized_overhead_ms"] = (
                row["time_cbf_parallelized_mean_ms"] - baseline
            )
            row["mean_overhead_ratio"] = row["mean_parallelized_overhead_ratio"]
            row["mean_overhead_ms"] = row["mean_parallelized_overhead_ms"]
        else:
            row["mean_parallelized_overhead_ratio"] = ""
            row["mean_parallelized_overhead_ms"] = ""
            row["mean_overhead_ratio"] = ""
            row["mean_overhead_ms"] = ""

    fieldnames = sorted({key for row in rows for key in row.keys()})
    csv_path = output_dir / "backup_shield_scalability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "backup_shield_scalability.md"
    columns = [
        "mode",
        "n_agents",
        "scenario_type",
        "status",
        "time_cbf_parallelized_mean_ms",
        "time_cbf_parallelized_p95_ms",
        "time_cbf_parallelized_max_ms",
        "mean_parallelized_overhead_ratio",
        "time_cbf_loop_mean_ms",
        "time_cbf_timing_interpretation",
        "collisions_agents",
        "collisions_lane",
        "deadlocks",
        "min_center_distance_m",
    ]
    lines = ["# Backup Shield Scalability", "", "|" + "|".join(columns) + "|"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def progress_line(
    completed: int, total: int, start_time: float, width: int = 24
) -> str:
    """Return a compact progress/time bar without adding a tqdm dependency."""
    completed = min(max(int(completed), 0), total)
    fraction = completed / total if total else 1.0
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    if completed > 0 and completed < total:
        eta = elapsed * (total - completed) / completed
        eta_text = format_duration(eta)
    elif completed >= total:
        eta_text = "0s"
    else:
        eta_text = "unknown"
    return (
        f"[{completed:02d}/{total:02d}] |{bar}| {100.0 * fraction:5.1f}% "
        f"elapsed={format_duration(elapsed)} eta={eta_text}"
    )


def main():
    args = parse_args()
    rows = []
    specs = mode_specs()
    total_runs = len(args.agent_counts) * len(specs)
    start_time = time.time()
    for n_agents in args.agent_counts:
        scenario_type = scenario_type_for_agents(args, n_agents)
        for spec in specs:
            print(
                f"{progress_line(len(rows), total_runs, start_time)} "
                f"RUN {spec['mode']} agents={n_agents} scenario={scenario_type}",
                flush=True,
            )
            row = run_case(args, spec, n_agents)
            rows.append(row)
            print(
                f"{progress_line(len(rows), total_runs, start_time)} "
                f"DONE {spec['mode']} agents={n_agents} scenario={scenario_type} "
                "status="
                f"{row.get('status')} parallelized_mean_ms="
                f"{row.get('time_cbf_parallelized_mean_ms')}",
                flush=True,
            )
            write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
