"""Benchmark learned target speed against fixed-speed CLF-CBF-QP.

Example:
    python sigmarl/utils/benchmark_learned_target_speed_vs_clf_cbf_qp.py \
        --agent_counts 3 --seeds 1 --steps 5 --no-parallel
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "marl_for_cavs_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
XDG_CACHE_HOME = Path(tempfile.gettempdir()) / "marl_for_cavs_cache"
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import torch

from sigmarl.helper_common import Parameters, save_video, visu_td_results
from sigmarl.helper_training import (
    SaveData,
    compute_parameter_independent_reward_per_second,
)
from sigmarl.mappo_cavs import mappo_cavs


DEFAULT_AGENT_COUNTS = [3, 4, 5, 6]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
# DEFAULT_MODEL_DIR = Path("checkpoints/20260529-backup-cbf/ttc/seed1")
DEFAULT_MODEL_DIR = Path(
    "checkpoints/20260604-backup-cbf-learned-ref-speed/ttc/reward_progress0.2/seed1"
)
# DEFAULT_MODEL_DIR = Path("checkpoints/20260604-backup-cbf-learned-ref-speed/ttc/reward_progress0.5/seed1")
# DEFAULT_MODEL_DIR = Path("checkpoints/20260604-backup-cbf-learned-ref-speed/ttc/reward_progress1.0/seed1")
DEFAULT_OUTPUT_DIR = (
    Path("outputs")
    / f"{date.today():%Y%m%d}_benchmark_learned_target_speed_vs_clf_cbf_qp_reward_progress0.2"
    # / f"{date.today():%Y%m%d}_benchmark_learned_target_speed_vs_clf_cbf_qp_reward_progress0.5"
    # / f"{date.today():%Y%m%d}_benchmark_learned_target_speed_vs_clf_cbf_qp_reward_progress1.0"
)
SCENARIO_TYPE = "cpm_mixed"
METRIC_FIELDS = [
    "parameter_independent_reward",
    "collision_rate",
    "deadlock_rate",
    "goal_reaching_rate",
    "avg_speed",
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    learned_target_speed: bool


@dataclass(frozen=True)
class BenchmarkJob:
    method: MethodSpec
    n_agents: int
    seed: int
    run_dir: Path
    td_path: Path
    video_path: Path
    figure_path: Path


METHODS = [
    MethodSpec("method1_learned_target_speed", learned_target_speed=True),
    MethodSpec("method2_fixed_target_speed", learned_target_speed=False),
]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR


def find_first_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {label} matching {pattern!r} found in {directory}")
    return matches[0]


def load_saved_parameters(model_dir: Path) -> Parameters:
    json_path = find_first_file(model_dir, "reward*.json", "saved-data JSON")
    with json_path.open("r", encoding="utf-8") as file:
        saved_data = SaveData.from_dict(json.load(file))
    return saved_data.parameters


def find_motion_policy_checkpoint(model_dir: Path, parameters: Parameters) -> Path:
    if parameters.is_load_final_model:
        final_path = model_dir / "final_motion_policy.pth"
        if not final_path.exists():
            raise FileNotFoundError(
                f"parameters.is_load_final_model=True, but {final_path} does not exist."
            )
        print(f"[INFO] Loading final motion policy: {final_path}")
        return final_path

    matches = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file()
        and "reward" in path.name
        and "motion_policy" in path.name
        and path.suffix == ".pth"
    )
    if not matches:
        raise FileNotFoundError(
            "No reward motion-policy checkpoint found in "
            f"{model_dir}. Expected a .pth file whose name contains both "
            "'reward' and 'motion_policy'."
        )

    print(f"[INFO] Loading reward motion policy: {matches[0]}")
    return matches[0]


def apply_common_parameters(
    parameters: Parameters,
    args: argparse.Namespace,
    job: BenchmarkJob,
) -> Parameters:
    load_final_model = bool(getattr(parameters, "is_load_final_model", False))

    parameters.dt = float(args.dt)
    parameters.where_to_save = str(job.run_dir)
    parameters.is_testing_mode = True
    parameters.is_real_time_rendering = False
    parameters.is_save_eval_results = False
    parameters.is_load_model = False
    parameters.is_load_final_model = load_final_model
    parameters.is_load_out_td = False
    parameters.max_steps = int(args.steps) + 1
    parameters.num_vmas_envs = 1
    parameters.scenario_type = SCENARIO_TYPE
    parameters.n_agents = int(job.n_agents)
    parameters.lane_width = 0.25
    parameters.is_save_simulation_video = True
    parameters.is_visualize_short_term_path = True
    parameters.is_visualize_lane_boundary = False
    parameters.is_visualize_extra_info = True
    parameters.random_seed = int(job.seed)
    parameters.is_continue_train = False
    parameters.rew_method = "sparse"

    parameters.is_using_centralized_cbf = True
    parameters.is_using_cbf_testing = True
    parameters.is_using_cbf_training = False
    parameters.is_solve_qp = True
    parameters.is_apply_cbf_action = True
    parameters.is_use_sampled_lane_boundary_for_cbf = True
    parameters.nom_controller_type = "clf"
    parameters.lane_distance_type = args.lane_distance_type
    parameters.is_proxsuite_qp = True
    parameters.is_grouping_agents = False
    parameters.max_group_size = 2
    parameters.grouping_method = "interaction_risk"
    parameters.grouping_use_hysteresis = True
    parameters.adaptive_lambda = True
    parameters.is_learn_cbf_lambda = False
    parameters.is_alternating_action_lambda_training = False
    parameters.n_circles_approximate_vehicle = 3
    parameters.is_log_cbf_boundary_distance = False
    parameters.is_aggregate_veh_veh_safety = True
    parameters.is_aggregate_lane_veh_safety = False
    parameters.veh_veh_cbf_footprint = args.veh_veh_cbf_footprint
    # The backup-horizon rows need the rectangle_sat footprints
    parameters.is_backup_horizon_cbf = args.veh_veh_cbf_footprint == "rectangle_sat"
    parameters.is_exact_backup_shield = False
    parameters.is_visu_shield_rollout = False

    return parameters


def parameters_for_job(args: argparse.Namespace, job: BenchmarkJob) -> Parameters:
    if job.method.learned_target_speed:
        parameters = load_saved_parameters(args.model_dir)
        parameters = apply_common_parameters(parameters, args, job)
        motion_policy_path = find_motion_policy_checkpoint(args.model_dir, parameters)
        parameters.rl_action_type = "speed"
        parameters.is_train_motion_policy = True
        parameters.is_train_lambda_policy = False
        parameters.is_load_pretrained_motion_policy = True
        parameters.is_load_pretrained_lambda_policy = False
        parameters.pretrained_motion_policy_path = str(motion_policy_path)
        parameters.pretrained_lambda_policy_path = ""
        return parameters

    parameters = apply_common_parameters(Parameters(), args, job)
    parameters.is_train_motion_policy = False
    parameters.is_train_lambda_policy = False
    parameters.is_load_pretrained_motion_policy = False
    parameters.is_load_pretrained_lambda_policy = False
    parameters.pretrained_motion_policy_path = ""
    parameters.pretrained_lambda_policy_path = ""
    return parameters


def get_tensor(out_td: Any, key: Any):
    try:
        return out_td.get(key)
    except Exception:
        return None


def squeeze_scalar_field(tensor):
    if tensor is None:
        return None
    tensor = torch.as_tensor(tensor).detach().cpu()
    while tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def finite_mean(tensor) -> float:
    tensor = torch.as_tensor(tensor).detach().cpu().float()
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return math.nan
    return float(finite.mean().item())


def compute_metrics(out_td: Any) -> dict[str, float]:
    parameter_independent_reward = compute_parameter_independent_reward_per_second(
        out_td
    )

    veh_collision = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "is_collision_with_agents"))
    )
    lane_collision = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "is_collision_with_lanelets"))
    )
    if veh_collision is None and lane_collision is None:
        collision_rate = math.nan
    elif veh_collision is None:
        collision_rate = float(lane_collision.bool().float().mean().item())
    elif lane_collision is None:
        collision_rate = float(veh_collision.bool().float().mean().item())
    else:
        collision_rate = float(
            torch.logical_or(veh_collision.bool(), lane_collision.bool())
            .float()
            .mean()
            .item()
        )

    deadlock = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "is_deadlocked"))
    )
    goal = squeeze_scalar_field(get_tensor(out_td, ("agents", "info", "is_reach_goal")))

    vel = get_tensor(out_td, ("agents", "info", "vel"))
    if vel is None:
        avg_speed = math.nan
    else:
        vel = torch.as_tensor(vel).detach().cpu().float()
        avg_speed = finite_mean(torch.linalg.norm(vel[..., :2], dim=-1))

    return {
        "parameter_independent_reward": parameter_independent_reward,
        "collision_rate": collision_rate,
        "deadlock_rate": (
            math.nan
            if deadlock is None
            else float(deadlock.bool().float().mean().item())
        ),
        "goal_reaching_rate": (
            math.nan if goal is None else float(goal.bool().float().mean().item())
        ),
        "avg_speed": avg_speed,
    }


def save_rollout_video(path: Path, frame_list: list[Any], fps: int) -> str:
    if shutil.which("ffmpeg") is not None:
        return save_video(
            str(path.with_suffix("")),
            frame_list,
            fps=fps,
            fmt="mp4",
            quality="high",
        )

    import cv2

    if not frame_list:
        raise ValueError("frame_list is empty. Cannot save video.")

    path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = frame_list[0]
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frame_list:
        writer.write(cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_RGB2BGR))
    writer.release()
    return str(path)


def close_render_window(env: Any) -> None:
    """Close VMAS/pyglet render windows opened during RGB-array capture."""

    candidates = [
        env,
        getattr(env, "base_env", None),
        getattr(getattr(env, "base_env", None), "_env", None),
        getattr(getattr(env, "base_env", None), "env", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        viewer = getattr(candidate, "viewer", None)
        if viewer is not None and hasattr(viewer, "close"):
            try:
                viewer.close()
            except Exception:
                pass
            try:
                candidate.viewer = None
            except Exception:
                pass
        if hasattr(candidate, "close"):
            try:
                candidate.close()
            except Exception:
                pass


def run_job(args: argparse.Namespace, job: BenchmarkJob) -> dict[str, Any]:
    start_time = time.time()
    job.run_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = base_row(job)
    env = None

    try:
        parameters = parameters_for_job(args, job)
        (
            env,
            decision_making_module,
            _optimization_module,
            priority_module,
            cbf_controllers,
            parameters,
        ) = mappo_cavs(parameters=parameters)

        render_callback = lambda env, _: env.render(  # noqa: E731
            mode="rgb_array", visualize_when_rgb=False
        )
        out_td, frame_list = env.rollout(
            max_steps=parameters.max_steps - 1,
            policy=decision_making_module.policy,
            priority_module=priority_module,
            cbf_controllers=cbf_controllers,
            callback=render_callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=True,
        )

        torch.save(out_td, job.td_path)
        row["video_path"] = save_rollout_video(
            job.video_path,
            frame_list,
            fps=int(round(1.0 / parameters.dt)),
        )
        row["figure_path"] = visu_td_results(
            out_td=out_td,
            parameters=parameters,
            vehicle_ids="all",
            env_index=0,
            save_dir=str(job.run_dir),
            fig_name=job.figure_path.name,
            show=False,
        )
        row.update(compute_metrics(out_td))
        row["status"] = "ok"
    except Exception as exc:  # pragma: no cover - exercised by failed rollouts.
        row["status"] = "failed"
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc()
    finally:
        close_render_window(env)

    row["wall_time_s"] = time.time() - start_time
    return row


def base_row(job: BenchmarkJob) -> dict[str, Any]:
    return {
        "method": job.method.name,
        "scenario_type": SCENARIO_TYPE,
        "n_agents": job.n_agents,
        "seed": job.seed,
        "parameter_independent_reward": math.nan,
        "collision_rate": math.nan,
        "deadlock_rate": math.nan,
        "goal_reaching_rate": math.nan,
        "avg_speed": math.nan,
        "td_path": str(job.td_path),
        "video_path": str(job.video_path),
        "figure_path": str(job.figure_path),
        "wall_time_s": "",
        "status": "pending",
        "error": "",
        "traceback": "",
    }


def load_existing_row(job: BenchmarkJob) -> dict[str, Any]:
    row = base_row(job)
    if not job.td_path.exists():
        row["status"] = "missing_td"
        row["error"] = str(job.td_path)
        return row
    out_td = torch.load(job.td_path, weights_only=False, map_location="cpu")
    row.update(compute_metrics(out_td))
    row["status"] = "ok"
    row["video_path"] = str(job.video_path) if job.video_path.exists() else ""
    row["figure_path"] = str(job.figure_path) if job.figure_path.exists() else ""
    return row


def build_jobs(args: argparse.Namespace) -> list[BenchmarkJob]:
    jobs: list[BenchmarkJob] = []
    for method in METHODS:
        for n_agents in args.agent_counts:
            for seed in args.seeds:
                run_dir = (
                    args.output_dir / method.name / f"agents{n_agents}" / f"seed{seed}"
                )
                jobs.append(
                    BenchmarkJob(
                        method=method,
                        n_agents=n_agents,
                        seed=seed,
                        run_dir=run_dir,
                        td_path=run_dir / "out_td.pt",
                        video_path=run_dir / "video.mp4",
                        figure_path=run_dir / "td_results_profiles.png",
                    )
                )
    return jobs


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def finite_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for record in records:
        if record.get("status") != "ok":
            continue
        value = record.get(field)
        if value is None or value == "":
            continue
        value_float = float(value)
        if math.isfinite(value_float):
            values.append(value_float)
    return values


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["method"]), int(record["n_agents"])), []).append(
            record
        )

    summaries = []
    for (method, n_agents), group_records in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "method": method,
            "scenario_type": SCENARIO_TYPE,
            "n_agents": n_agents,
            "n_successful_seeds": sum(
                1 for record in group_records if record.get("status") == "ok"
            ),
        }
        for field in METRIC_FIELDS:
            values = finite_values(group_records, field)
            summary[f"{field}_mean"] = mean(values) if values else math.nan
            summary[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def print_summary(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        print("[WARN] No successful runs to summarize.")
        return

    print("\nSummary across seeds")
    for record in summaries:
        print(
            "{method:<29s} agents={n_agents:<2d} "
            "parameter_independent_reward="
            "{parameter_independent_reward_mean:9.3f}+/-"
            "{parameter_independent_reward_std:7.3f} "
            "collision={collision_rate_mean:7.5f}+/-{collision_rate_std:7.5f} "
            "deadlock={deadlock_rate_mean:7.5f}+/-{deadlock_rate_std:7.5f} "
            "goal={goal_reaching_rate_mean:7.5f}+/-{goal_reaching_rate_std:7.5f} "
            "speed={avg_speed_mean:6.3f}+/-{avg_speed_std:6.3f}".format(**record)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark learned target speed vs fixed-speed CLF-CBF-QP."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Folder containing reward*.json and *motion_policy.pth for method 1.",
    )
    parser.add_argument(
        "--agent_counts",
        type=parse_int_list,
        default=DEFAULT_AGENT_COUNTS,
        help="Comma-separated agent counts. Default: 3,4,5,6.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds. Default: 1,2,3,4,5.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Number of saved rollout transitions.",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir(),
        help="Root directory for rollout outputs and metric CSVs.",
    )
    parser.add_argument(
        "--lane_distance_type",
        choices=["pseudo", "softmin"],
        default="softmin",
    )
    parser.add_argument(
        "--veh_veh_cbf_footprint",
        choices=["circle", "mtv", "rectangle_sat"],
        default="rectangle_sat",
        help="Shared vehicle-vehicle CBF footprint for both methods.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run rollout jobs with a process pool.",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--skip_if_exist",
        action="store_true",
        help="Skip rollouts whose out_td.pt already exists.",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only recompute metrics from existing TensorDict outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "info.txt").write_text(str(DEFAULT_MODEL_DIR), encoding="utf-8")

    jobs = build_jobs(args)
    pending_jobs = [
        job
        for job in jobs
        if not args.aggregate_only and not (args.skip_if_exist and job.td_path.exists())
    ]

    print(f"[INFO] Output root: {args.output_dir}")
    print(f"[INFO] Model dir: {args.model_dir}")
    print(f"[INFO] Prepared {len(jobs)} jobs; running {len(pending_jobs)} jobs.")

    run_rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    if pending_jobs:
        if args.parallel:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_job = {
                    executor.submit(run_job, args, job): job for job in pending_jobs
                }
                for future in as_completed(future_to_job):
                    row = future.result()
                    key = (str(row["method"]), int(row["n_agents"]), int(row["seed"]))
                    run_rows_by_key[key] = row
                    print(
                        f"[{row['status'].upper()}] {row['method']} "
                        f"agents={row['n_agents']} seed={row['seed']} "
                        f"wall={row['wall_time_s']:.1f}s"
                    )
        else:
            for job in pending_jobs:
                row = run_job(args, job)
                key = (str(row["method"]), int(row["n_agents"]), int(row["seed"]))
                run_rows_by_key[key] = row
                print(
                    f"[{row['status'].upper()}] {row['method']} "
                    f"agents={row['n_agents']} seed={row['seed']} "
                    f"wall={row['wall_time_s']:.1f}s"
                )

    records = []
    for job in jobs:
        key = (job.method.name, job.n_agents, job.seed)
        records.append(run_rows_by_key.get(key) or load_existing_row(job))

    raw_fields = [
        "method",
        "scenario_type",
        "n_agents",
        "seed",
        *METRIC_FIELDS,
        "td_path",
        "video_path",
        "figure_path",
        "wall_time_s",
        "status",
        "error",
        "traceback",
    ]
    write_csv(args.output_dir / "raw_metrics.csv", records, raw_fields)

    summaries = summarize_records(records)
    summary_fields = [
        "method",
        "scenario_type",
        "n_agents",
        "n_successful_seeds",
    ]
    for field in METRIC_FIELDS:
        summary_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(args.output_dir / "summary_metrics.csv", summaries, summary_fields)
    print_summary(summaries)

    failed = [record for record in records if record.get("status") not in {"ok"}]
    if failed:
        print(f"[ERROR] {len(failed)} runs are missing or failed. See raw_metrics.csv.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
