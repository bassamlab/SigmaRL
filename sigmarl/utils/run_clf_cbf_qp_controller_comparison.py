"""Run CLF-CBF-QP controller comparisons for aggregation and MTV safety modes.

Examples:
    # Fully centralized CBF-QP.
    python sigmarl/utils/run_clf_cbf_qp_controller_comparison.py \
        --max_group_size none \
        --output_dir outputs/compare_clf_cbf_qp_centralized
        --skip_if_exist

    # Grouped CBF-QP.
    python sigmarl/utils/run_clf_cbf_qp_controller_comparison.py \
        --max_group_size 2 \
        --output_dir outputs/compare_clf_cbf_qp_grouped_m2
        --skip_if_exist

The script stores one rollout per scenario/config/seed subfolder, then writes
per-seed metrics and cross-seed summaries under the output root.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, stdev

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for this utility.
    tqdm = None


DEFAULT_SCRIPT = Path("main_testing_clf_cbf_qp.py")
DEFAULT_GROUPING_METHOD = "interaction_risk"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_MAX_STEPS = 601
DEFAULT_SCENARIOS = {
    "cpm_mixed": 4,
    "cpm_entire": 10,
}

METRIC_FIELDS = [
    "total_reward",
    "collision_rate",
    "vehicle_collision_rate",
    "lane_collision_rate",
    "avg_speed",
]


@dataclass(frozen=True)
class ControllerConfig:
    """Configuration switch values for one controller variant."""

    name: str
    is_aggregate_veh_veh_safety: bool
    veh_veh_cbf_footprint: str


@dataclass(frozen=True)
class ScenarioSpec:
    """Scenario name and agent count for one experiment family."""

    scenario_type: str
    n_agents: int


@dataclass(frozen=True)
class ComparisonJob:
    """One concrete rollout command and its output locations."""

    scenario: ScenarioSpec
    config: ControllerConfig
    seed: int
    run_dir: Path
    td_path: Path
    log_path: Path
    command: list[str]


CONFIGS = [
    ControllerConfig(
        name="aggregate_circles",
        is_aggregate_veh_veh_safety=True,
        veh_veh_cbf_footprint="circle",
    ),
    ControllerConfig(
        name="nonaggregate_circles",
        is_aggregate_veh_veh_safety=False,
        veh_veh_cbf_footprint="circle",
    ),
    ControllerConfig(
        name="mtv_single_row",
        is_aggregate_veh_veh_safety=False,
        veh_veh_cbf_footprint="mtv",
    ),
]


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list for CLI options."""

    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def parse_optional_int(value: str) -> int | None:
    """Parse an integer CLI value while accepting 'none' for disabled grouping."""

    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def default_output_dir() -> Path:
    """Return the dated default output directory requested for this comparison."""

    return Path("outputs") / f"{date.today():%Y%m%d}_compare_clf_cbf_qp"


def td_filename(
    scenario_type: str,
    n_agents: int,
    max_group_size: int | None,
    grouping_method: str,
    seed: int,
) -> str:
    """Return the output TensorDict filename used by main_testing_clf_cbf_qp.py."""

    grouping_suffix = (
        f"_max_group_size{max_group_size}_{grouping_method}"
        if max_group_size is not None
        else "_max_group_sizeNone"
    )
    return (
        f"out_td_{scenario_type}_agents{n_agents}"
        f"{grouping_suffix}_proxsuite_seed{seed}.td"
    )


def scenario_specs(args: argparse.Namespace) -> list[ScenarioSpec]:
    """Build the scenario list selected by CLI arguments."""

    specs = [
        ScenarioSpec(scenario_type=scenario_type, n_agents=n_agents)
        for scenario_type, n_agents in DEFAULT_SCENARIOS.items()
    ]
    if args.scenario_type == "all":
        return specs
    return [spec for spec in specs if spec.scenario_type == args.scenario_type]


def build_command(
    args: argparse.Namespace,
    scenario: ScenarioSpec,
    config: ControllerConfig,
    seed: int,
    run_dir: Path,
) -> list[str]:
    """Build the rollout command for one scenario/config/seed combination."""

    command = [
        args.python_exec,
        str(args.script),
        "--seed",
        str(seed),
        "--scenario_type",
        scenario.scenario_type,
        "--n_agents",
        str(scenario.n_agents),
        "--max_steps",
        str(args.max_steps),
        "--lane_distance_type",
        args.lane_distance_type,
        "--is_proxsuite_qp",
        "--max_group_size",
        str(args.max_group_size),
        "--grouping_method",
        args.grouping_method,
        "--output_dir",
        str(run_dir),
    ]

    if config.is_aggregate_veh_veh_safety:
        command.append("--is_aggregate_veh_veh_safety")
    else:
        command.append("--no-is_aggregate_veh_veh_safety")

    command.extend(["--veh_veh_cbf_footprint", config.veh_veh_cbf_footprint])

    if not args.save_video:
        command.append("--no-is_save_simulation_video")
    if not args.save_td_profile_figures:
        command.append("--no-is_visu_result")
    if not args.real_time_rendering:
        command.append("--no-is_real_time_rendering")

    return command


def build_jobs(args: argparse.Namespace) -> tuple[list[ComparisonJob], int]:
    """Create rollout jobs and count skipped runs when outputs already exist."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[ComparisonJob] = []
    skipped = 0
    for scenario in scenario_specs(args):
        for config in CONFIGS:
            for seed in args.seeds:
                run_dir = (
                    output_dir
                    / scenario.scenario_type
                    / f"agents{scenario.n_agents}"
                    / config.name
                    / f"seed{seed}"
                )
                td_path = run_dir / td_filename(
                    scenario_type=scenario.scenario_type,
                    n_agents=scenario.n_agents,
                    max_group_size=args.max_group_size,
                    grouping_method=args.grouping_method,
                    seed=seed,
                )
                log_path = run_dir / "run.log"
                if args.skip_if_exist and not args.aggregate_only and td_path.exists():
                    skipped += 1
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)
                command = build_command(
                    args=args,
                    scenario=scenario,
                    config=config,
                    seed=seed,
                    run_dir=run_dir,
                )
                jobs.append(
                    ComparisonJob(
                        scenario=scenario,
                        config=config,
                        seed=seed,
                        run_dir=run_dir,
                        td_path=td_path,
                        log_path=log_path,
                        command=command,
                    )
                )

    return jobs, skipped


def run_job(job: ComparisonJob) -> tuple[ComparisonJob, float, int]:
    """Run one rollout command and write stdout/stderr to its log file."""

    start_time = time.time()
    env = os.environ.copy()
    with job.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("[COMMAND] " + " ".join(job.command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            job.command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    return job, time.time() - start_time, completed.returncode


def progress_iter(iterable, total: int, desc: str):
    """Wrap an iterable in tqdm when tqdm is installed."""

    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="run")
    return iterable


def run_sequential(jobs: list[ComparisonJob]) -> list[tuple[ComparisonJob, float, int]]:
    """Run rollout jobs one at a time."""

    results = []
    for job in progress_iter(jobs, total=len(jobs), desc="Rollouts"):
        results.append(run_job(job))
    return results


def run_parallel(
    jobs: list[ComparisonJob],
    workers: int,
) -> list[tuple[ComparisonJob, float, int]]:
    """Run rollout jobs in parallel worker processes."""

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in progress_iter(
            as_completed(futures),
            total=len(futures),
            desc="Rollouts",
        ):
            results.append(future.result())
    return results


def get_tensor(out_td, key):
    """Return a TensorDict field or None if the field is absent."""

    try:
        return out_td.get(key)
    except Exception:
        return None


def squeeze_scalar_field(tensor):
    """Convert a scalar-valued TensorDict field to a CPU tensor without trailing ones."""

    if tensor is None:
        return None
    import torch

    tensor = torch.as_tensor(tensor).detach().cpu()
    while tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def finite_mean(tensor) -> float:
    """Return the finite mean of a tensor or NaN when no finite values exist."""

    if tensor is None:
        return math.nan
    import torch

    tensor = torch.as_tensor(tensor).detach().cpu().float()
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return math.nan
    return float(finite.mean().item())


def reward_total(out_td) -> float:
    """Compute total rollout reward over all recorded steps and agents."""

    import torch

    reward = squeeze_scalar_field(get_tensor(out_td, ("agents", "info", "rew_total")))
    if reward is None:
        reward_parts = [
            squeeze_scalar_field(get_tensor(out_td, key))
            for key in [
                ("agents", "info", "rew_reach_goal"),
                ("agents", "info", "rew_collide_other_agents"),
                ("agents", "info", "rew_collide_lane"),
                ("agents", "info", "rew_deadlock"),
            ]
        ]
        reward_parts = [item.float() for item in reward_parts if item is not None]
        if not reward_parts:
            return math.nan
        reward = torch.stack(reward_parts, dim=0).sum(dim=0)
    return float(reward.float().sum().item())


def collision_rates(out_td) -> tuple[float, float, float]:
    """Compute any-collision, vehicle-collision, and lane-collision rates."""

    import torch

    veh_collision = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "is_collision_with_agents"))
    )
    lane_collision = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "is_collision_with_lanelets"))
    )
    if veh_collision is None and lane_collision is None:
        return math.nan, math.nan, math.nan
    if veh_collision is None:
        lane_bool = lane_collision.bool()
        return (
            float(lane_bool.float().mean().item()),
            math.nan,
            float(lane_bool.float().mean().item()),
        )
    if lane_collision is None:
        veh_bool = veh_collision.bool()
        return (
            float(veh_bool.float().mean().item()),
            float(veh_bool.float().mean().item()),
            math.nan,
        )

    veh_bool = veh_collision.bool()
    lane_bool = lane_collision.bool()
    any_collision = torch.logical_or(veh_bool, lane_bool)
    return (
        float(any_collision.float().mean().item()),
        float(veh_bool.float().mean().item()),
        float(lane_bool.float().mean().item()),
    )


def average_speed(out_td) -> float:
    """Compute average absolute vehicle speed over all recorded steps and agents."""

    import torch

    speed = squeeze_scalar_field(get_tensor(out_td, ("agents", "info", "speed")))
    if speed is not None:
        return finite_mean(speed.abs())

    vel = get_tensor(out_td, ("agents", "info", "vel"))
    if vel is None:
        return math.nan
    vel = torch.as_tensor(vel).detach().cpu().float()
    if vel.ndim < 4 or vel.shape[-1] < 2:
        return math.nan
    return finite_mean(torch.linalg.norm(vel[..., :2], dim=-1))


def load_run_metrics(job: ComparisonJob) -> dict[str, object]:
    """Load one rollout TensorDict and compute comparison metrics."""

    import torch

    out_td = torch.load(job.td_path, weights_only=False, map_location="cpu")
    collision_rate, vehicle_collision_rate, lane_collision_rate = collision_rates(
        out_td
    )
    return {
        "scenario_type": job.scenario.scenario_type,
        "n_agents": job.scenario.n_agents,
        "config": job.config.name,
        "is_aggregate_veh_veh_safety": job.config.is_aggregate_veh_veh_safety,
        "veh_veh_cbf_footprint": job.config.veh_veh_cbf_footprint,
        "seed": job.seed,
        "td_path": str(job.td_path),
        "total_reward": reward_total(out_td),
        "collision_rate": collision_rate,
        "vehicle_collision_rate": vehicle_collision_rate,
        "lane_collision_rate": lane_collision_rate,
        "avg_speed": average_speed(out_td),
    }


def write_csv(
    path: Path, records: list[dict[str, object]], fieldnames: list[str]
) -> None:
    """Write records to CSV, creating parent directories if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def finite_values(records: list[dict[str, object]], field: str) -> list[float]:
    """Extract finite numeric values from records for one metric field."""

    values: list[float] = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        value_float = float(value)
        if math.isfinite(value_float):
            values.append(value_float)
    return values


def summarize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-seed records by scenario and controller config."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["scenario_type"]), str(record["config"]))
        grouped.setdefault(key, []).append(record)

    summaries: list[dict[str, object]] = []
    for (scenario_type, config_name), group_records in sorted(grouped.items()):
        config = next(config for config in CONFIGS if config.name == config_name)
        summary: dict[str, object] = {
            "scenario_type": scenario_type,
            "n_agents": group_records[0]["n_agents"],
            "config": config_name,
            "is_aggregate_veh_veh_safety": config.is_aggregate_veh_veh_safety,
            "veh_veh_cbf_footprint": config.veh_veh_cbf_footprint,
            "n_seeds": len(group_records),
        }
        for field in METRIC_FIELDS:
            values = finite_values(group_records, field)
            summary[f"{field}_mean"] = mean(values) if values else math.nan
            summary[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def print_summary(summaries: list[dict[str, object]]) -> None:
    """Print a compact text summary after aggregation."""

    if not summaries:
        print("[WARN] No metrics were available to summarize.")
        return

    print("\nSummary across seeds")
    for record in summaries:
        print(
            "{scenario_type:>10s} | {config:<22s} | "
            "reward {total_reward_mean:10.3f} +/- {total_reward_std:8.3f} | "
            "collision {collision_rate_mean:8.5f} +/- {collision_rate_std:8.5f} | "
            "speed {avg_speed_mean:7.4f} +/- {avg_speed_std:7.4f}".format(**record)
        )


def collect_metrics(jobs: list[ComparisonJob]) -> list[dict[str, object]]:
    """Collect metrics for all jobs whose TensorDict output exists."""

    records: list[dict[str, object]] = []
    for job in jobs:
        if not job.td_path.exists():
            print(f"[WARN] Missing TensorDict output: {job.td_path}")
            continue
        records.append(load_run_metrics(job))
    return records


def write_command_manifest(output_dir: Path, jobs: list[ComparisonJob]) -> None:
    """Write all generated commands to a CSV manifest."""

    records = [
        {
            "scenario_type": job.scenario.scenario_type,
            "n_agents": job.scenario.n_agents,
            "config": job.config.name,
            "seed": job.seed,
            "run_dir": str(job.run_dir),
            "td_path": str(job.td_path),
            "log_path": str(job.log_path),
            "command": " ".join(job.command),
        }
        for job in jobs
    ]
    write_csv(
        output_dir / "command_manifest.csv",
        records,
        [
            "scenario_type",
            "n_agents",
            "config",
            "seed",
            "run_dir",
            "td_path",
            "log_path",
            "command",
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison utility."""

    parser = argparse.ArgumentParser(
        description=(
            "Run main_testing_clf_cbf_qp.py across aggregation, non-aggregation, "
            "and MTV vehicle-vehicle CBF configurations."
        )
    )
    parser.add_argument(
        "--scenario_type",
        choices=["all", "cpm_mixed", "cpm_entire"],
        default="all",
        help="Scenario subset to run.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds. Default: 1,2,3,4,5.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Number of rollout steps configured for each run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir(),
        help="Root directory for rollout outputs, logs, and metric CSVs.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT,
        help="Path to main_testing_clf_cbf_qp.py.",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used for each rollout subprocess.",
    )
    parser.add_argument(
        "--lane_distance_type",
        choices=["pseudo", "softmin"],
        default="softmin",
        help="Lane distance type passed to main_testing_clf_cbf_qp.py.",
    )
    parser.add_argument(
        "--max_group_size",
        type=parse_optional_int,
        default=1,
        help=(
            "Maximum group size. Use 'none' for fully centralized CBF, "
            "1 for fully decentralized CBF, and values >1 for grouped CBF."
        ),
    )
    parser.add_argument(
        "--grouping_method",
        default=DEFAULT_GROUPING_METHOD,
        choices=["interaction_risk", "k_nearest"],
        help="Grouping method passed to main_testing_clf_cbf_qp.py.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run rollout subprocesses in parallel.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers when --parallel is set.",
    )
    parser.add_argument(
        "--skip_if_exist",
        action="store_true",
        help="Skip rollout jobs whose TensorDict output already exists.",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Do not run rollouts; only recompute metrics from existing outputs.",
    )
    parser.add_argument(
        "--save_video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save simulation videos. Disabled by default for headless sweeps.",
    )
    parser.add_argument(
        "--save_td_profile_figures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save TD profile and CBF boundary figures. Disabled by default.",
    )
    parser.add_argument(
        "--real_time_rendering",
        action="store_true",
        help="Pause rendering to match real time. Disabled by default.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the comparison sweep and write metric summaries."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    jobs, skipped = build_jobs(args)
    write_command_manifest(output_dir, jobs)

    print(f"[INFO] Output root: {output_dir}")
    print(f"[INFO] Prepared {len(jobs)} jobs; skipped {skipped} existing jobs.")

    if not args.aggregate_only and jobs:
        if args.parallel:
            results = run_parallel(jobs, workers=args.workers)
        else:
            results = run_sequential(jobs)

        failed = [result for result in results if result[2] != 0]
        for job, duration, returncode in sorted(
            results,
            key=lambda item: (
                item[0].scenario.scenario_type,
                item[0].config.name,
                item[0].seed,
            ),
        ):
            status = "OK" if returncode == 0 else f"FAIL({returncode})"
            print(
                f"[{status}] {job.scenario.scenario_type} {job.config.name} "
                f"seed={job.seed} duration={duration:.1f}s log={job.log_path}"
            )
        if failed:
            print(f"[ERROR] {len(failed)} rollout jobs failed. Check run.log files.")
            return 1

    records = collect_metrics(jobs)
    raw_fields = [
        "scenario_type",
        "n_agents",
        "config",
        "is_aggregate_veh_veh_safety",
        "veh_veh_cbf_footprint",
        "seed",
        "td_path",
        *METRIC_FIELDS,
    ]
    write_csv(output_dir / "raw_metrics.csv", records, raw_fields)

    summaries = summarize_records(records)
    summary_fields = [
        "scenario_type",
        "n_agents",
        "config",
        "is_aggregate_veh_veh_safety",
        "veh_veh_cbf_footprint",
        "n_seeds",
    ]
    for field in METRIC_FIELDS:
        summary_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(output_dir / "summary_metrics.csv", summaries, summary_fields)
    print_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
