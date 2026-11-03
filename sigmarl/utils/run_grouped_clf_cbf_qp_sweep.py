# python run_grouped_clf_cbf_qp_sweep.py --parallel --workers 4 --skip_if_exist

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency.
    tqdm = None


DEFAULT_OUTPUT_DIR = Path("checkpoints/20260503-grouped_clf_cbf_qp_comparison_v2")
DEFAULT_SCRIPT = Path("main_testing_clf_cbf_qp.py")
DEFAULT_GROUPING_METHOD = "interaction_risk"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class SweepJob:
    scenario_type: str
    n_agents: int
    max_group_size: int
    seed: int
    output_path: Path
    log_path: Path
    command: list[str]


def parse_int_list(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def group_sizes_for(scenario_type: str, n_agents: int) -> list[int]:
    if scenario_type == "cpm_mixed":
        return list(range(1, n_agents + 1))
    if scenario_type == "cpm_entire":
        return [1] + list(range(2, n_agents + 1, 2))
    raise ValueError(f"Unsupported scenario type: {scenario_type}")


def output_name(
    scenario_type: str,
    n_agents: int,
    max_group_size: int,
    grouping_method: str,
    seed: int,
) -> str:
    return (
        f"out_td_{scenario_type}_agents{n_agents}"
        f"_max_group_size{max_group_size}_{grouping_method}"
        f"_proxsuite_seed{seed}.td"
    )


def build_jobs(args: argparse.Namespace) -> tuple[list[SweepJob], int]:
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scenario_agent_counts = {
        "cpm_mixed": [4, 5, 6],
        "cpm_entire": [8, 12, 16],
    }
    if args.scenario_type != "all":
        scenario_agent_counts = {
            args.scenario_type: scenario_agent_counts[args.scenario_type]
        }

    jobs: list[SweepJob] = []
    skipped = 0
    script_path = str(Path(args.script))

    for scenario_type, agent_counts in scenario_agent_counts.items():
        for n_agents in agent_counts:
            for max_group_size in group_sizes_for(scenario_type, n_agents):
                for seed in args.seeds:
                    td_path = output_dir / output_name(
                        scenario_type=scenario_type,
                        n_agents=n_agents,
                        max_group_size=max_group_size,
                        grouping_method=args.grouping_method,
                        seed=seed,
                    )
                    log_path = log_dir / (
                        f"{scenario_type}_agents{n_agents}"
                        f"_group{max_group_size}_seed{seed}.log"
                    )

                    if args.skip_if_exist and td_path.exists():
                        skipped += 1
                        continue

                    command = [
                        args.python_exec,
                        script_path,
                        str(seed),
                        "--scenario_type",
                        scenario_type,
                        "--n_agents",
                        str(n_agents),
                        "--max_group_size",
                        str(max_group_size),
                        "--grouping_method",
                        args.grouping_method,
                        "--max_steps",
                        str(args.max_steps),
                        "--output_dir",
                        str(output_dir),
                        "--is_proxsuite_qp",
                    ]

                    if not args.save_video:
                        command.append("--no-is_save_simulation_video")
                    if not args.save_td_profile_figures:
                        command.append("--no-is_visu_result")
                    if not args.real_time_rendering:
                        command.append("--no-is_real_time_rendering")

                    jobs.append(
                        SweepJob(
                            scenario_type=scenario_type,
                            n_agents=n_agents,
                            max_group_size=max_group_size,
                            seed=seed,
                            output_path=td_path,
                            log_path=log_path,
                            command=command,
                        )
                    )

    return jobs, skipped


def run_job(job: SweepJob) -> tuple[SweepJob, float, int]:
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
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="run")
    return iterable


def run_sequential(jobs: list[SweepJob]) -> list[tuple[SweepJob, float, int]]:
    results = []
    for job in progress_iter(jobs, total=len(jobs), desc="Total progress"):
        results.append(run_job(job))
    return results


def run_parallel(
    jobs: list[SweepJob],
    workers: int,
) -> list[tuple[SweepJob, float, int]]:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in progress_iter(
            as_completed(futures),
            total=len(futures),
            desc="Total progress",
        ):
            results.append(future.result())
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the grouped CLF-CBF-QP parameter sweep."
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
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for outputs and logs.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT,
        help="Testing entrypoint to run.",
    )
    parser.add_argument(
        "--python_exec",
        default=sys.executable,
        help="Python executable used for each run.",
    )
    parser.add_argument(
        "--grouping_method",
        choices=["interaction_risk", "k_nearest"],
        default=DEFAULT_GROUPING_METHOD,
        help="Grouping method passed to the testing script.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="Rollout steps passed to the testing script.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use parallel workers. Default is sequential execution.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers when --parallel is enabled.",
    )
    parser.add_argument(
        "--skip_if_exist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip a run when its expected TD output already exists.",
    )
    parser.add_argument(
        "--save_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save simulation videos from each run.",
    )
    parser.add_argument(
        "--save_td_profile_figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-run TD profile figures.",
    )
    parser.add_argument(
        "--real_time_rendering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use real-time rendering pauses during each run.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1.")

    jobs, skipped = build_jobs(args)

    print(f"[INFO] Jobs to run: {len(jobs)}")
    print(f"[INFO] Jobs skipped: {skipped}")
    print(f"[INFO] Output directory: {Path(args.output_dir)}")
    print(f"[INFO] Logs directory: {Path(args.output_dir) / 'logs'}")

    if args.dry_run:
        for job in jobs:
            print(" ".join(job.command))
        return

    if not jobs:
        print("[INFO] Nothing to do.")
        return

    start_time = time.time()
    if args.parallel:
        print(f"[INFO] Running with {args.workers} parallel workers.")
        results = run_parallel(jobs, workers=args.workers)
    else:
        print("[INFO] Running sequentially.")
        results = run_sequential(jobs)

    failures = [(job, elapsed, code) for job, elapsed, code in results if code != 0]
    elapsed_total = time.time() - start_time

    print(f"[INFO] Finished {len(results)} jobs in {elapsed_total:.2f} s.")
    if failures:
        print(f"[ERROR] {len(failures)} jobs failed:")
        for job, elapsed, code in failures:
            print(
                f"  code={code} elapsed={elapsed:.2f}s "
                f"{job.scenario_type} agents={job.n_agents} "
                f"group={job.max_group_size} seed={job.seed} "
                f"log={job.log_path}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
