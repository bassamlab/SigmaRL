#!/usr/bin/env python3
"""Evaluate a folder of saved models over a configurable evaluation sweep.

The sweep is defined by :data:`SCENARIO_TYPES`, :data:`N_AGENTS`, and
:data:`SEEDS`. Every model folder is evaluated for every compatible scenario,
agent count, and seed combination. Summary JSON artifacts include the sweep
settings in their names so that different jobs can share one model folder.
Trimmed rollout TensorDicts are saved only when explicitly requested.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import math
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Optional

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

from sigmarl.constants import SCENARIOS
from sigmarl.helper_common import VideoWriter, trim_td
from sigmarl.helper_training import (
    Parameters,
    SaveData,
    _task_performance_from_rollout,
    save_evaluation_summary,
)
from sigmarl.mappo_cavs import mappo_cavs


DEFAULT_NUM_WORKERS = 6
DEFAULT_MAX_STEPS = 2400
N_EVAL_ENVS = 16

POLICY_FILE_PATTERN = "reward*_policy.pth"

# Keep these collections easy to edit when adding a new evaluation setting.
# SCENARIO_TYPES = ("cpm_mixed", "cpm_entire")
# N_AGENTS = {
#     "cpm_mixed": (3, 4, 5, 6, 7),
#     "cpm_entire": (8, 10, 12, 14, 16),
# }
SCENARIO_TYPES = (
    "cpm_mixed",
    # "cpm_intersection_extended",
    # "cpm_entire",
    # "interchange_3",
    "intersection_4",
    "roundabout_2",
)

N_AGENTS = {
    "cpm_mixed": (6, 8, 10),
    "cpm_intersection_extended": (6, 10, 14),
    "cpm_entire": (12, 14, 16, 18, 20),
    "interchange_3": (6, 11, 16),
    "intersection_4": (6, 11, 16),
    "roundabout_2": (6, 11, 16),
}

SCENARIO_SCALE_MAP = {
    "interchange_3": 1.2,
    "intersection_4": 1.2,
    "roundabout_2": 1.2,
}
SCENARIO_LANE_WIDTH = {
    "interchange_3": 0.30,
    "intersection_4": 0.30,
    "roundabout_2": 0.30,
}

SEEDS = [101, 102, 103, 104]

# Scenario settings for cross-iteration evaluation
SCENARIO_TYPES_CROSS_ITER = ("cpm_mixed",)
N_AGENTS_CROSS_ITER = {
    "cpm_mixed": (10,),
}
SEEDS_CROSS_ITER = [101]

METRICS = (
    (
        "parameter_independent_reward_per_agent_per_minute",
        ("parameter_independent_reward_per_agent_per_minute",),
        "float",
    ),
    (
        "vehicles_per_second",
        ("statistics", "traffic_throughput", "vehicles_per_second"),
        "float",
    ),
    ("mean_speed_mps", ("statistics", "speed_mps", "mean"), "float"),
    ("collision_rate_steps", ("collision_rate_steps",), "float"),
    ("collision_rate_agents", ("collision_rate_agents",), "float"),
    ("num_task_tries", ("task_performance", "num_task_tries"), "integer"),
    (
        "task_success_rate",
        ("task_performance", "task_success_rate"),
        "float",
    ),
)
PLOT_METRIC_NAMES = (
    "parameter_independent_reward_per_agent_per_minute",
    "mean_speed_mps",
    "collision_rate_steps",
    "task_success_rate",
)
PLOT_METRICS = tuple(metric for metric in METRICS if metric[0] in PLOT_METRIC_NAMES)
PLOT_METRIC_LABELS = {
    "parameter_independent_reward_per_agent_per_minute": (
        "Parameter-independent reward per agent per minute"
    ),
    "mean_speed_mps": "Agent mean speed (m/s)",
    "collision_rate_steps": r"Collision rate (\%)",
}
PLOT_METRIC_SCALES = {
    "collision_rate_steps": 100.0,
}
PLOT_Y_LIMITS = {
    "cpm_entire": {
        "parameter_independent_reward_per_agent_per_minute": (-5.0, 5.0),
        "collision_rate_steps": (0.0, 1.5),
    },
    "cpm_intersection_extended": {
        "parameter_independent_reward_per_agent_per_minute": (-12.0, 2.0),
        "collision_rate_steps": (0.0, 5.0),
    },
    "cpm_mixed": {
        "parameter_independent_reward_per_agent_per_minute": (-6.0, 16.0),
        "collision_rate_steps": (0.0, 1.5),
    },
    "intersection_4": {
        "parameter_independent_reward_per_agent_per_minute": (-16.0, 16.0),
        "collision_rate_steps": (0.0, 1.5),
    },
    "interchange_3": {
        "parameter_independent_reward_per_agent_per_minute": (-16.0, 16.0),
        "collision_rate_steps": (0.0, 1.5),
    },
}
PLOT_GLOBAL_Y_LIMITS = {
    "mean_speed_mps": (0.0, 1.0),
    "task_success_rate": (0.0, 1.0),
}
PLOT_DEFAULT_Y_LIMITS = {
    "parameter_independent_reward_per_agent_per_minute": (-16.0, 16.0),
    "collision_rate_steps": (0.0, 1.5),
}
MISSING_VALUE = "—"


def display_path(path: Path) -> str:
    """Return a path formatted relative to the current working directory."""

    return os.path.relpath(path, Path.cwd())


def parse_keywords(value: str) -> tuple[str, ...]:
    """Parse hyphen-separated alternative substrings for model folders."""
    return tuple(keyword for keyword in value.split("-") if keyword)


def parse_selected_cross_iter_checkpoints(
    tokens: Optional[Sequence[Union[str, int]]]
) -> Optional[list[int]]:
    """Parse list of checkpoint iterations from CLI argument.

    Supports:
    - Multiple space-separated tokens: ['100', '150', '200']
    - Single bracketed string: ['[100, 150, 200]'] or ['[100,150,200]']
    - Unquoted shell tokens: ['[100,', '150,', '200]']
    - Comma-separated string: ['100,150,200']
    """
    if tokens is None:
        return None
    raw = " ".join(str(t) for t in tokens).strip()
    raw = raw.strip("[](){}")
    if not raw.strip():
        return []
    parts = raw.replace(",", " ").split()
    return [int(p) for p in parts]


def parse_cross_iter_ego_indices(
    tokens: Optional[Sequence[Union[str, int]]]
) -> tuple[int, ...]:
    """Parse tuple of ego agent indices from CLI argument.

    Supports:
    - Single integer or string: '1' or 1
    - Single bracketed index: '[1]'
    - Multiple space-separated tokens: ['0', '1']
    - Bracketed string: ['[0, 1]'] or ['[0,1]']
    - Unquoted shell tokens: ['[0,', '1]']
    - Comma-separated string: ['0,1']
    - None or empty: returns (0,)
    """
    if tokens is None:
        return (0,)
    raw = " ".join(str(t) for t in tokens).strip()
    raw = raw.strip("[](){}")
    if not raw.strip():
        return (0,)
    parts = raw.replace(",", " ").split()
    indices = [int(p) for p in parts]
    if not indices:
        return (0,)
    for idx in indices:
        if idx < 0:
            raise ValueError(f"Agent index must be non-negative, got {idx}.")
    return tuple(dict.fromkeys(indices))


METHOD_CATEGORIES = (
    (
        "vanilla",
        ("vanilla"),
    ),
    (
        "our",
        (
            "our_rr_motion_taskprogress_permute",
            "our_rr_only_taskprogress",
            "our_motion_only_taskprogress",
            "our_rr_motion_taskprogress",
        ),
    ),
    (
        "opponent_modeling",
        (
            "om_dedicated",
            "om_shared",
        ),
    ),
    (
        "mtppo",
        ("mtppo_motion",),
    ),
)
METHOD_ORDER = tuple(
    method_name
    for _, category_methods in METHOD_CATEGORIES
    for method_name in category_methods
)
METHOD_ORDER_INDEX = {
    method_name: index for index, method_name in enumerate(METHOD_ORDER)
}
METHOD_COLORS = {
    "vanilla": "#4c72b0",
    "our_rr_motion_taskprogress_permute": "#7A4A2F",
    "our_rr_only_taskprogress": "#f1c7a5",
    "our_motion_only_taskprogress": "#f99e53",
    "our_rr_motion_taskprogress": "#893200",
    "om_dedicated": "#55a868",
    "om_shared": "#a0c4a3",
    "mtppo_motion": "#2b6b3a",
    "our_rr_motion_random": "#f1c7a5",
    "our_rr_motion_fixed": "#f1c7a5",
    "our_rr_motion_taskprogress_group": "#890080",
}
UNKNOWN_METHOD_COLOR = "#9e9e9e"
PLOT_FONT_SIZE = 9
PLOT_WIDTH = 6.0
PLOT_HEIGHT = 12.0


@dataclass(frozen=True)
class EvaluationJob:
    """One model/scenario/agent-count/seed evaluation."""

    model_path: Path
    scenario_type: str
    n_agents: int
    seed: int
    is_cross_iter_test: bool = False
    cross_iter_ego_indices: tuple[int, ...] = (0,)
    ego_iter: Optional[int] = None
    peer_iter: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.cross_iter_ego_indices, tuple):
            object.__setattr__(
                self, "cross_iter_ego_indices", tuple(self.cross_iter_ego_indices)
            )

    @property
    def ego_tag(self) -> str:
        """Tag representing the designated set of ego agent indices."""
        if not self.cross_iter_ego_indices:
            return "0"
        return "_".join(str(i) for i in self.cross_iter_ego_indices)

    @property
    def settings_tag(self) -> str:
        """Return the stable tag shared by all artifacts for this job."""

        base = (
            f"scenario_{self.scenario_type}_n_agents_{self.n_agents}"
            f"_seed_{self.seed}"
        )
        if (
            self.is_cross_iter_test
            and self.ego_iter is not None
            and self.peer_iter is not None
        ):
            return (
                f"{base}_cross_iter_ego{self.ego_tag}"
                f"_iter{self.ego_iter}_peer{self.peer_iter}"
            )
        return base

    @property
    def out_td_path(self) -> Path:
        return self.model_path / f"out_td_{self.settings_tag}.td"

    @property
    def summary_path(self) -> Path:
        return self.model_path / f"evaluation_summary_{self.settings_tag}.json"

    @property
    def video_path(self) -> Path:
        return self.model_path / f"video_ego{self.ego_tag}_{self.settings_tag}.mp4"


@dataclass(frozen=True)
class ModelEvaluationAggregate:
    """Evaluation metrics aggregated across one model's testing seeds."""

    metric_values: Mapping[str, float]
    mean_task_success_times: float | None
    mean_num_task_tries: float | None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the batch sweep."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Parent folder containing trained-model subfolders.",
    )
    parser.add_argument(
        "--is-hpc",
        "--is_hpc",
        dest="is_hpc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Submit one HPC job per model folder instead of running local "
            "parallel workers (default: False)."
        ),
    )
    parser.add_argument(
        "--hpc-submit-script",
        "--hpc_submit_script",
        dest="hpc_submit_script",
        type=Path,
        default=None,
        help=(
            "Private per-model HPC submission script. If omitted, use "
            "$SIGMARL_HPC_SUBMIT_SCRIPT."
        ),
    )
    parser.add_argument(
        "--is-hpc-worker",
        "--is_hpc_worker",
        dest="is_hpc_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-workers",
        "--workers",
        dest="num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Number of local parallel rollout workers; ignored by sequential "
            f"HPC worker jobs (default: {DEFAULT_NUM_WORKERS})."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=(
            "Number of environment steps per rollout. Use a small value such as "
            "100 for a quick smoke test. "
            f"(default: {DEFAULT_MAX_STEPS})"
        ),
    )
    parser.add_argument(
        "--is-skip",
        "--is_skip",
        dest="is_skip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip jobs whose summary JSON already exists "
            f"(default: False; use --no-is-skip to rerun)."
        ),
    )
    parser.add_argument(
        "--is-plot-only",
        "--is_plot_only",
        dest="is_plot_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip rollout evaluation and summary-report generation, and only "
            "regenerate the figures (default: False)."
        ),
    )
    parser.add_argument(
        "--keywords",
        type=parse_keywords,
        default=None,
        help=(
            "Plot-only mode: hyphen-separated keywords. Include only model "
            "subfolders whose names contain at least one keyword."
        ),
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Render and save an MP4 for every evaluated job (disabled by default).",
    )
    parser.add_argument(
        "--save-out-td",
        action="store_true",
        help=(
            "Save the trimmed rollout TensorDict for every job "
            "(disabled by default)."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "Markdown path for the final summary report "
            "(default: <path>/evaluation_summary.md)."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal places for floating-point report values (default: 6).",
    )
    parser.add_argument(
        "--is-cross-iter-test",
        "--is_cross_iter_test",
        dest="is_cross_iter_test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run cross iteration testing loading different checkpoints for "
            "the ego agent vs peer agents (default: False)."
        ),
    )
    parser.add_argument(
        "--cross-iter-ego-indices",
        "--cross_iter_ego_indices",
        dest="cross_iter_ego_indices",
        nargs="+",
        default=None,
        help=(
            "Agent index or indices designated as ego agents for cross-iteration "
            "testing (e.g. 0, 1, 0 1, [1], or [0, 1]). Default is [0]."
        ),
    )
    parser.add_argument(
        "--peer-iter",
        "--peer_iter",
        dest="peer_iter",
        type=int,
        default=None,
        help="Specific checkpoint iteration for peer agents in cross-iter test (default: all available).",
    )
    parser.add_argument(
        "--ego-iter",
        "--ego_iter",
        dest="ego_iter",
        type=int,
        default=None,
        help="Specific checkpoint iteration for ego agent in cross-iter test (default: all available).",
    )
    parser.add_argument(
        "--selected-cross-iter-checkpoints",
        "--selected_cross_iter_checkpoints",
        dest="selected_cross_iter_checkpoints",
        nargs="+",
        default=None,
        help=(
            "Specific checkpoint iterations to evaluate in cross-iter test "
            "(e.g. 100 150 200 or [100, 150, 200]). Default is None (all "
            "available cross pairs)."
        ),
    )
    parser.add_argument(
        "--usetex",
        "--use-tex",
        "--use_tex",
        dest="usetex",
        action="store_true",
        default=False,
        help="Enable external LaTeX rendering ('text.usetex' = True) in Matplotlib.",
    )
    parser.add_argument(
        "--usetex-font",
        "--usetex_font",
        dest="usetex_font",
        choices=["times", "helvetica"],
        default="times",
        help=(
            "LaTeX font family when --usetex is enabled ('times' for Times Roman serif "
            "or 'helvetica' for Helvetica sans-serif; default: 'times')."
        ),
    )
    args = parser.parse_args(argv)
    if args.selected_cross_iter_checkpoints is not None:
        try:
            parsed_iters = parse_selected_cross_iter_checkpoints(
                args.selected_cross_iter_checkpoints
            )
            if not parsed_iters:
                parser.error(
                    "--selected_cross_iter_checkpoints must contain at least one iteration number."
                )
            args.selected_cross_iter_checkpoints = parsed_iters
            args.is_cross_iter_test = True
        except ValueError as exc:
            parser.error(f"Failed to parse --selected_cross_iter_checkpoints: {exc}")
    if args.cross_iter_ego_indices is not None:
        try:
            parsed_indices = parse_cross_iter_ego_indices(args.cross_iter_ego_indices)
            if not parsed_indices:
                parser.error(
                    "--cross-iter-ego-indices must contain at least one agent index."
                )
            args.cross_iter_ego_indices = parsed_indices
        except ValueError as exc:
            parser.error(f"Failed to parse --cross-iter-ego-indices: {exc}")
    else:
        args.cross_iter_ego_indices = (0,)
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.precision < 0:
        parser.error("--precision must be non-negative")
    if args.is_hpc and args.summary_output is not None:
        parser.error(
            "--summary-output cannot be used with --is-hpc; each submitted "
            "model writes its own report."
        )
    if args.is_hpc and args.is_plot_only:
        parser.error("--is-plot-only cannot be used with --is-hpc.")
    if args.keywords is not None and not args.is_plot_only:
        parser.error("--keywords can only be used with --is-plot-only.")
    return args


def resolve_hpc_submit_script(configured_path: Optional[Path]) -> Path:
    """Resolve and validate the private launcher used for HPC submissions."""

    path_value = configured_path or os.environ.get("SIGMARL_HPC_SUBMIT_SCRIPT")
    if path_value is None:
        raise ValueError(
            "HPC mode requires a submission script: pass --hpc-submit-script "
            "or set $SIGMARL_HPC_SUBMIT_SCRIPT."
        )
    script_path = Path(path_value).expanduser()
    if not script_path.is_absolute():
        script_path = Path.cwd() / script_path
    script_path = script_path.resolve()

    if not script_path.is_file():
        raise FileNotFoundError(
            f"HPC submission script not found: {script_path}. Pass a valid "
            "--hpc-submit-script or set $SIGMARL_HPC_SUBMIT_SCRIPT."
        )
    return script_path


def model_path_for_hpc(model_path: Path) -> str:
    """Return a model path that can be remapped by the HPC launcher."""

    repository_root = Path(__file__).resolve().parent
    try:
        return model_path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(model_path)


def hpc_child_arguments(args: argparse.Namespace) -> list[str]:
    """Build evaluator arguments forwarded to each submitted model job."""

    child_arguments = [
        "--no-is-hpc",
        "--is-hpc-worker",
        "--max-steps",
        str(args.max_steps),
        "--precision",
        str(args.precision),
        "--is-skip" if args.is_skip else "--no-is-skip",
    ]
    if args.save_video:
        child_arguments.append("--save-video")
    if args.save_out_td:
        child_arguments.append("--save-out-td")
    if getattr(args, "is_cross_iter_test", False):
        child_arguments.append("--is-cross-iter-test")
        child_arguments.append("--cross-iter-ego-indices")
        child_arguments.extend(str(idx) for idx in args.cross_iter_ego_indices)
        if args.peer_iter is not None:
            child_arguments.extend(["--peer-iter", str(args.peer_iter)])
        if args.ego_iter is not None:
            child_arguments.extend(["--ego-iter", str(args.ego_iter)])
        if getattr(args, "selected_cross_iter_checkpoints", None) is not None:
            child_arguments.append("--selected-cross-iter-checkpoints")
            child_arguments.extend(
                str(it) for it in args.selected_cross_iter_checkpoints
            )
    return child_arguments


def submit_hpc_jobs(model_paths: Sequence[Path], args: argparse.Namespace) -> None:
    """Submit one private-launcher invocation for each trained model folder."""

    script_path = resolve_hpc_submit_script(args.hpc_submit_script)
    child_arguments = hpc_child_arguments(args)
    failures: list[tuple[Path, str]] = []

    print(f"[INFO] Using private HPC submission script: {display_path(script_path)}")
    for model_path in tqdm(
        model_paths,
        desc="Submitting HPC model jobs",
        unit="model",
    ):
        command = [
            "bash",
            str(script_path),
            model_path_for_hpc(model_path),
            *child_arguments,
        ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            if isinstance(error, subprocess.CalledProcessError):
                error_text = f"launcher exited with status {error.returncode}"
            else:
                error_text = f"{type(error).__name__}: {error}"
            failures.append((model_path, error_text))
            tqdm.write(
                f"[ERROR] HPC submission failed for {display_path(model_path)}: "
                f"{error_text}"
            )

    if failures:
        failed_models = ", ".join(model_path.name for model_path, _ in failures)
        raise RuntimeError(f"{len(failures)} HPC submission(s) failed: {failed_models}")


def contains_trained_policy(path: Path, is_cross_iter_test: bool = False) -> bool:
    """Return whether a folder contains at least one saved policy file."""

    if is_cross_iter_test:
        cross_dir = path / "checkpoints_cross_iter"
        if cross_dir.is_dir():
            with os.scandir(cross_dir) as entries:
                if any(
                    entry.is_file() and fnmatch.fnmatch(entry.name, "policy_iter*.pth")
                    for entry in entries
                ):
                    return True

    with os.scandir(path) as entries:
        return any(
            entry.is_file() and fnmatch.fnmatch(entry.name, POLICY_FILE_PATTERN)
            for entry in entries
        )


def find_policy_folders(
    root_path: Path, is_cross_iter_test: bool = False
) -> list[Path]:
    """Find model folders at the root or within its first two levels."""

    root_path = root_path.expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Model root is not a directory: {root_path}")

    root_contains_policy = contains_trained_policy(
        root_path, is_cross_iter_test=is_cross_iter_test
    )
    policy_folders: list[Path] = []
    first_level_folders = sorted(
        (entry for entry in root_path.iterdir() if entry.is_dir()),
        key=lambda entry: entry.name,
    )

    for first_level_folder in first_level_folders:
        if contains_trained_policy(
            first_level_folder, is_cross_iter_test=is_cross_iter_test
        ):
            policy_folders.append(first_level_folder)

        second_level_folders = sorted(
            (entry for entry in first_level_folder.iterdir() if entry.is_dir()),
            key=lambda entry: entry.name,
        )
        policy_folders.extend(
            second_level_folder
            for second_level_folder in second_level_folders
            if contains_trained_policy(
                second_level_folder, is_cross_iter_test=is_cross_iter_test
            )
        )

    if not policy_folders and root_contains_policy:
        policy_folders.append(root_path)

    return policy_folders


def get_cross_iter_checkpoints(model_path: Path) -> list[int]:
    """Find all saved iteration numbers in checkpoints_cross_iter."""
    cross_dir = model_path / "checkpoints_cross_iter"
    if not cross_dir.is_dir():
        return []
    pattern = re.compile(r"^policy_iter(\d+)\.pth$")
    iters = []
    for entry in cross_dir.iterdir():
        if entry.is_file():
            match = pattern.match(entry.name)
            if match:
                iters.append(int(match.group(1)))
    return sorted(iters)


def build_jobs(
    model_paths: Sequence[Path],
    is_cross_iter_test: bool = False,
    cross_iter_ego_indices: Sequence[int] = (0,),
    peer_iter: Optional[int] = None,
    ego_iter: Optional[int] = None,
    selected_cross_iter_checkpoints: Optional[Sequence[int]] = None,
) -> list[EvaluationJob]:
    """Create one job for every model and configured sweep combination."""

    ego_indices_tuple = tuple(cross_iter_ego_indices)

    scenarios = SCENARIO_TYPES_CROSS_ITER if is_cross_iter_test else SCENARIO_TYPES
    agent_counts = N_AGENTS_CROSS_ITER if is_cross_iter_test else N_AGENTS
    seeds = SEEDS_CROSS_ITER if is_cross_iter_test else SEEDS

    if not scenarios:
        raise ValueError("SCENARIO_TYPES must contain at least one scenario.")
    if not seeds:
        raise ValueError("SEEDS must contain at least one seed.")

    jobs: list[EvaluationJob] = []
    for model_path in model_paths:
        if is_cross_iter_test:
            available_iters = get_cross_iter_checkpoints(model_path)
            if not available_iters:
                raise FileNotFoundError(
                    f"No cross-iteration checkpoints found in {model_path / 'checkpoints_cross_iter'}."
                )
            if selected_cross_iter_checkpoints is not None:
                missing = [
                    it
                    for it in selected_cross_iter_checkpoints
                    if it not in available_iters
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Selected cross-iteration checkpoint(s) {missing} not found in "
                        f"{model_path / 'checkpoints_cross_iter'}. "
                        f"Available iterations are: {available_iters}."
                    )
                base_iters = [it for it in sorted(set(selected_cross_iter_checkpoints))]
            else:
                base_iters = available_iters

            ego_iters = [ego_iter] if ego_iter is not None else base_iters
            peer_iters = [peer_iter] if peer_iter is not None else base_iters

            for scenario_type in scenarios:
                if scenario_type not in SCENARIOS:
                    raise ValueError(
                        f"Unknown scenario '{scenario_type}' in SCENARIO_TYPES_CROSS_ITER."
                    )
                if scenario_type not in agent_counts:
                    raise ValueError(
                        f"No N_AGENTS entry configured for scenario '{scenario_type}'."
                    )
                for n_agents in agent_counts[scenario_type]:
                    if n_agents <= 0:
                        raise ValueError(
                            f"Agent counts must be positive, got {n_agents} "
                            f"for '{scenario_type}'."
                        )
                    for idx in ego_indices_tuple:
                        if idx < 0 or idx >= n_agents:
                            raise ValueError(
                                f"Ego agent index {idx} is out of bounds for scenario "
                                f"'{scenario_type}' with n_agents={n_agents}."
                            )
                    for seed in seeds:
                        for e_iter in ego_iters:
                            for p_iter in peer_iters:
                                jobs.append(
                                    EvaluationJob(
                                        model_path=model_path,
                                        scenario_type=scenario_type,
                                        n_agents=n_agents,
                                        seed=seed,
                                        is_cross_iter_test=True,
                                        cross_iter_ego_indices=ego_indices_tuple,
                                        ego_iter=e_iter,
                                        peer_iter=p_iter,
                                    )
                                )
        else:
            for scenario_type in scenarios:
                if scenario_type not in SCENARIOS:
                    raise ValueError(
                        f"Unknown scenario '{scenario_type}' in SCENARIO_TYPES."
                    )
                if scenario_type not in agent_counts:
                    raise ValueError(
                        f"No N_AGENTS entry configured for scenario '{scenario_type}'."
                    )
                for n_agents in agent_counts[scenario_type]:
                    if n_agents <= 0:
                        raise ValueError(
                            f"Agent counts must be positive, got {n_agents} "
                            f"for '{scenario_type}'."
                        )
                    for seed in seeds:
                        jobs.append(
                            EvaluationJob(
                                model_path=model_path,
                                scenario_type=scenario_type,
                                n_agents=n_agents,
                                seed=seed,
                            )
                        )
    return jobs


def find_training_json(model_path: Path) -> Path:
    """Return the deterministic reward-named training metadata file."""

    json_files = sorted(
        path
        for path in model_path.iterdir()
        if path.is_file() and path.name.startswith("reward") and path.suffix == ".json"
    )
    if not json_files:
        raise FileNotFoundError(f"No reward*.json file found in {model_path}.")
    return json_files[0]


def job_is_complete(job: EvaluationJob) -> bool:
    """Return whether the corresponding summary JSON file exists."""

    return job.summary_path.is_file()


def _apply_testing_parameters(
    parameters: Parameters,
    job: EvaluationJob,
    max_steps: int,
    save_video_enabled: bool,
) -> None:
    """Apply evaluation settings without changing the saved model metadata."""

    parameters.where_to_save = str(job.model_path)
    parameters.is_testing_mode = True
    parameters.is_real_time_rendering = False
    parameters.is_save_eval_results = False
    parameters.is_load_model = True
    parameters.is_load_final_model = False
    parameters.is_load_out_td = False
    # The environment uses this value as its configured horizon, while the
    # rollout below executes exactly ``max_steps`` transitions.
    parameters.max_steps = max_steps + 1
    parameters.observed_agents_render_ego_index = 0
    parameters.num_vmas_envs = N_EVAL_ENVS
    parameters.is_joint_task = True

    parameters.scenario_type = job.scenario_type
    parameters.n_agents = job.n_agents
    parameters.scale_map = SCENARIO_SCALE_MAP.get(
        job.scenario_type, getattr(parameters, "scale_map", 1.0)
    )
    parameters.lane_width = SCENARIO_LANE_WIDTH.get(
        job.scenario_type, getattr(parameters, "lane_width", 0.30)
    )
    parameters.is_save_simulation_video = save_video_enabled
    parameters.is_visualize_short_term_path = True
    parameters.is_visualize_lane_boundary = False
    parameters.is_visualize_extra_info = True
    parameters.is_visualize_observed_agents = True

    parameters.random_seed = job.seed
    parameters.is_continue_train = False
    parameters.rew_method = "sparse"

    if job.is_cross_iter_test:
        parameters.is_cross_iter_test = True
        primary_ego = job.cross_iter_ego_indices[0] if job.cross_iter_ego_indices else 0
        parameters.observed_agents_render_ego_index = primary_ego
        parameters.cross_iter_ego_indices = job.cross_iter_ego_indices
        parameters.ego_iter = job.ego_iter
        parameters.peer_iter = job.peer_iter
        parameters.ego_policy_path = str(
            job.model_path / "checkpoints_cross_iter" / f"policy_iter{job.ego_iter}.pth"
        )
        parameters.peer_policy_path = str(
            job.model_path
            / "checkpoints_cross_iter"
            / f"policy_iter{job.peer_iter}.pth"
        )


def run_rollout(
    job: EvaluationJob,
    max_steps: int,
    save_video_enabled: bool,
    save_out_td: bool,
    job_index: int,
    progress_queue: Any,
) -> str:
    """Run one deterministic rollout while keeping worker logs off the bars."""

    with open(os.devnull, "w") as worker_output:
        with contextlib.redirect_stdout(worker_output), contextlib.redirect_stderr(
            worker_output
        ):
            return _run_rollout(
                job,
                max_steps,
                save_video_enabled,
                save_out_td,
                job_index,
                progress_queue,
            )


def _run_rollout(
    job: EvaluationJob,
    max_steps: int,
    save_video_enabled: bool,
    save_out_td: bool,
    job_index: int,
    progress_queue: Any,
) -> str:
    """Load one model, execute one sweep job, and save its artifacts."""

    collision_penalty = -2.0

    training_json = find_training_json(job.model_path)
    with training_json.open("r", encoding="utf-8") as json_file:
        saved_data = SaveData.from_dict(json.load(json_file))
    parameters: Parameters = saved_data.parameters
    _apply_testing_parameters(parameters, job, max_steps, save_video_enabled)

    (
        env,
        decision_making_module,
        _optimization_module,
        priority_module,
        cbf_controllers,
        parameters,
    ) = mappo_cavs(parameters=parameters)

    def render_callback(current_env: Any, _: Any) -> Any:
        if progress_queue is not None:
            progress_queue.put((job_index, 1))
        if save_video_enabled:
            return current_env.render(mode="rgb_array", visualize_when_rgb=False)
        return None

    # Stream frames to the video file instead of keeping them in memory
    video_writer = (
        VideoWriter(
            str(job.video_path.with_suffix("")),
            fps=1 / parameters.dt,
            quality="very_high",
        )
        if save_video_enabled
        else None
    )
    t_0 = time.time()
    try:
        with torch.inference_mode(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ):
            out_td, _ = env.rollout(
                max_steps=max_steps,
                policy=decision_making_module.policy,
                priority_module=priority_module,
                cbf_controllers=cbf_controllers,
                callback=render_callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
                is_save_simulation_video=save_video_enabled,
                frame_sink=video_writer,
            )
        simulation_time = time.time() - t_0
        num_frames = out_td.numel()
        simulation_speed = num_frames / simulation_time if simulation_time > 0 else 0.0
        real_time_factor = (
            num_frames * parameters.dt / simulation_time if simulation_time > 0 else 0.0
        )
        print(f"Simulation time: {simulation_time:.2f} s")
        print(f"Simulation speed: {simulation_speed:.2f} FPS")
        print(f"Real-time factor: {real_time_factor:.2f}x")

        out_td = trim_td(out_td, parameters=parameters)

        scenario = env.scenario
        task_perf = _task_performance_from_rollout(out_td)
        if task_perf is not None:
            num_task_tries = task_perf["num_task_tries"]
            task_success_times = task_perf["task_success_times"]
        else:
            num_task_tries = int(scenario.num_task_tries.sum().item())
            task_success_times = int(scenario.task_success_times.sum().item())
        save_evaluation_summary(
            out_td,
            str(job.model_path),
            f"_{job.settings_tag}",
            random_seed=job.seed,
            task_performance={
                "is_joint_task": bool(getattr(parameters, "is_joint_task", False)),
                "num_task_tries": num_task_tries,
                "task_success_times": task_success_times,
            },
            collision_penalty=collision_penalty,
            computation_efficiency={
                "simulation_time": simulation_time,
                "simulation_speed": simulation_speed,
                "real_time_factor": real_time_factor,
            },
        )
        if save_out_td:
            torch.save(out_td, job.out_td_path)

        if video_writer is not None:
            video_writer.close()
    except BaseException:
        if video_writer is not None:
            video_writer.abort()
        raise
    finally:
        env.close()

    return str(job.model_path)


def _run_batch_sequential(
    pending_jobs: Sequence[EvaluationJob],
    total_jobs: int,
    skipped_count: int,
    max_steps: int,
    save_video_enabled: bool,
    save_out_td: bool,
) -> list[tuple[EvaluationJob, str]]:
    """Run evaluation settings serially without creating worker processes."""

    progress_steps = max(max_steps - 1, 0)
    failures: list[tuple[EvaluationJob, str]] = []
    with tqdm(
        total=total_jobs,
        initial=skipped_count,
        desc="Global evaluation (sequential)",
        position=0,
        unit="rollout",
    ) as global_progress:
        for job_index, job in enumerate(pending_jobs):
            progress = tqdm(
                total=progress_steps,
                desc=(
                    f"Rollout {job_index + 1}/{len(pending_jobs)}: "
                    f"{job.model_path.name} | {job.scenario_type} "
                    f"n={job.n_agents} seed={job.seed}"
                ),
                position=1,
                leave=False,
                unit="step",
            )
            try:
                run_rollout(
                    job,
                    max_steps,
                    save_video_enabled,
                    save_out_td,
                    job_index,
                    progress_queue=None,
                )
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
                failures.append((job, error_text))
                progress.set_description(
                    f"Rollout {job_index + 1}/{len(pending_jobs)} failed"
                )
                tqdm.write(
                    f"[ERROR] Rollout failed for {display_path(job.model_path)} "
                    f"({job.settings_tag}): {error_text}"
                )
            else:
                tqdm.write(
                    f"[INFO] Rollout completed for {display_path(job.model_path)} "
                    f"({job.settings_tag})"
                )
                progress.n = progress.total
            progress.refresh()
            progress.close()
            global_progress.update(1)

    return failures


def run_batch(
    jobs: Sequence[EvaluationJob],
    max_steps: int,
    num_workers: int,
    save_video_enabled: bool,
    save_out_td: bool,
    is_skip: bool,
    is_sequential: bool = False,
) -> list[tuple[EvaluationJob, str]]:
    """Schedule pending jobs and update global and per-job progress bars."""

    pending_jobs = [job for job in jobs if not is_skip or not job_is_complete(job)]
    skipped_count = len(jobs) - len(pending_jobs)
    print(f"[INFO] Skipped {skipped_count} completed job(s) " f"(is_skip={is_skip}).")
    if not pending_jobs:
        return []

    if is_sequential:
        print(
            "[INFO] Running evaluation settings sequentially; "
            "no local worker pool will be created."
        )
        return _run_batch_sequential(
            pending_jobs,
            total_jobs=len(jobs),
            skipped_count=skipped_count,
            max_steps=max_steps,
            save_video_enabled=save_video_enabled,
            save_out_td=save_out_td,
        )

    worker_count = min(num_workers, len(pending_jobs))
    progress_steps = max(max_steps - 1, 0)
    multiprocessing_context = multiprocessing.get_context("spawn")
    failures: list[tuple[EvaluationJob, str]] = []

    with multiprocessing_context.Manager() as manager:
        progress_queue = manager.Queue()
        with tqdm(
            total=len(jobs),
            initial=skipped_count,
            desc="Global evaluation",
            position=0,
            unit="rollout",
        ) as global_progress:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing_context,
            ) as executor:
                next_job_index = 0
                future_info: dict[Any, tuple[int, int, EvaluationJob]] = {}
                rollout_progress: dict[int, Any] = {}

                def submit_job(slot: int) -> Any:
                    nonlocal next_job_index
                    job_index = next_job_index
                    next_job_index += 1
                    job = pending_jobs[job_index]
                    rollout_progress[job_index] = tqdm(
                        total=progress_steps,
                        desc=(
                            f"Rollout {job_index + 1}/{len(pending_jobs)}: "
                            f"{job.model_path.name} | {job.scenario_type} "
                            f"n={job.n_agents} seed={job.seed}"
                        ),
                        position=slot + 1,
                        leave=False,
                        unit="step",
                    )
                    future = executor.submit(
                        run_rollout,
                        job,
                        max_steps,
                        save_video_enabled,
                        save_out_td,
                        job_index,
                        progress_queue,
                    )
                    future_info[future] = (job_index, slot, job)
                    return future

                for slot in range(worker_count):
                    submit_job(slot)

                pending = set(future_info)
                while pending:
                    _drain_progress_queue(progress_queue, rollout_progress)
                    completed, pending = wait(
                        pending,
                        timeout=0.1,
                        return_when=FIRST_COMPLETED,
                    )
                    if not completed:
                        continue

                    for future in completed:
                        job_index, slot, job = future_info.pop(future)
                        progress = rollout_progress.pop(job_index)
                        try:
                            future.result()
                        except Exception as error:
                            error_text = f"{type(error).__name__}: {error}"
                            failures.append((job, error_text))
                            progress.set_description(
                                f"Rollout {job_index + 1}/{len(pending_jobs)} failed"
                            )
                            tqdm.write(
                                f"[ERROR] Rollout failed for "
                                f"{display_path(job.model_path)} "
                                f"({job.settings_tag}): {error_text}"
                            )
                        else:
                            tqdm.write(
                                f"[INFO] Rollout completed for "
                                f"{display_path(job.model_path)} "
                                f"({job.settings_tag})"
                            )
                            progress.n = progress.total
                        progress.refresh()
                        progress.close()
                        global_progress.update(1)

                        if next_job_index < len(pending_jobs):
                            pending.add(submit_job(slot))

                _drain_progress_queue(progress_queue, rollout_progress)

    return failures


def _drain_progress_queue(
    progress_queue: Any, rollout_progress: Mapping[int, Any]
) -> None:
    """Apply queued per-step progress events to their corresponding bars."""

    while True:
        try:
            job_index, step_count = progress_queue.get_nowait()
        except queue.Empty:
            return
        progress = rollout_progress.get(job_index)
        if progress is not None:
            progress.update(step_count)


def nested_value(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Return a nested value, or ``None`` when the path is unavailable."""

    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def format_metric_value(value: Any, value_type: str, precision: int) -> str:
    """Format one summary metric for a Markdown table."""

    if value is None:
        return MISSING_VALUE
    if value_type == "integer":
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)) and float(value).is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{precision}f}"


def read_summary(path: Path) -> Mapping[str, Any] | None:
    """Read one summary JSON and report malformed files as unavailable."""

    try:
        with path.open("r", encoding="utf-8") as summary_file:
            data = json.load(summary_file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        print(
            f"[warning] Could not read {display_path(path)}: {error}",
            file=sys.stderr,
        )
        return None
    if not isinstance(data, Mapping):
        print(
            f"[warning] Expected an object in {display_path(path)}",
            file=sys.stderr,
        )
        return None
    return data


def _as_finite_float(value: Any) -> float | None:
    """Convert a JSON number to a finite float for plotting."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _collect_setting_summaries(
    setting_jobs: Sequence[EvaluationJob],
) -> tuple[tuple[int, ...], list[Path], dict[Path, dict[int, Mapping[str, Any]]],]:
    """Load summaries grouped by model path for one scenario/agent setting."""

    expected_seeds = tuple(sorted({job.seed for job in setting_jobs}))
    model_paths = sorted(
        {job.model_path for job in setting_jobs},
        key=_method_sort_key,
    )
    summaries_by_model: dict[Path, dict[int, Mapping[str, Any]]] = {
        model_path: {} for model_path in model_paths
    }
    for job in setting_jobs:
        summary = read_summary(job.summary_path)
        if summary is not None:
            summaries_by_model[job.model_path][job.seed] = summary
    return expected_seeds, model_paths, summaries_by_model


def _aggregate_model_summaries(
    summaries_by_seed: Mapping[int, Mapping[str, Any]],
    expected_seeds: Sequence[int],
) -> ModelEvaluationAggregate:
    """Aggregate testing seeds into one result for a trained model.

    Ordinary metrics are averaged across complete testing-seed summaries. The
    task-success and throughput ratios are recomputed from their stored
    numerators and denominators so that unequal per-seed event counts do not
    receive an unintended equal-ratio weighting.
    """

    metric_values: dict[str, float] = {}
    scalar_values: dict[str, list[float]] = {
        metric_name: [] for metric_name, _, _ in METRICS
    }
    task_success_times: list[float] = []
    task_tries: list[float] = []
    successful_vehicle_counts: list[float] = []
    simulated_durations: list[float] = []

    for seed in expected_seeds:
        summary = summaries_by_seed.get(seed)
        if summary is None:
            continue

        for metric_name, metric_path, _ in METRICS:
            if metric_name in {"task_success_rate", "vehicles_per_second"}:
                continue
            value = _as_finite_float(nested_value(summary, metric_path))
            if value is not None:
                scalar_values[metric_name].append(value)

        task_success_times_value = _as_finite_float(
            nested_value(summary, ("task_performance", "task_success_times"))
        )
        task_tries_value = _as_finite_float(
            nested_value(summary, ("task_performance", "num_task_tries"))
        )
        if task_success_times_value is not None and task_tries_value is not None:
            task_success_times.append(task_success_times_value)
            task_tries.append(task_tries_value)

        successful_vehicle_count = _as_finite_float(
            nested_value(
                summary,
                (
                    "statistics",
                    "traffic_throughput",
                    "successful_vehicle_count",
                ),
            )
        )
        simulated_duration = _as_finite_float(
            nested_value(
                summary,
                ("statistics", "traffic_throughput", "total_simulated_duration_s"),
            )
        )
        if successful_vehicle_count is not None and simulated_duration is not None:
            successful_vehicle_counts.append(successful_vehicle_count)
            simulated_durations.append(simulated_duration)

    expected_count = len(expected_seeds)
    if expected_count:
        for metric_name, values in scalar_values.items():
            if len(values) == expected_count:
                metric_values[metric_name] = fmean(values)

    if (
        expected_count
        and len(task_success_times) == expected_count
        and len(task_tries) == expected_count
    ):
        total_task_tries = sum(task_tries)
        metric_values["num_task_tries"] = fmean(task_tries)
        metric_values["task_success_rate"] = (
            sum(task_success_times) / total_task_tries if total_task_tries else 0.0
        )
        mean_task_success_times = fmean(task_success_times)
        mean_num_task_tries = fmean(task_tries)
    else:
        mean_task_success_times = None
        mean_num_task_tries = None

    if (
        expected_count
        and len(successful_vehicle_counts) == expected_count
        and len(simulated_durations) == expected_count
    ):
        total_simulated_duration = sum(simulated_durations)
        if total_simulated_duration > 0:
            metric_values["vehicles_per_second"] = (
                sum(successful_vehicle_counts) / total_simulated_duration
            )

    return ModelEvaluationAggregate(
        metric_values=metric_values,
        mean_task_success_times=mean_task_success_times,
        mean_num_task_tries=mean_num_task_tries,
    )


def _method_label(model_path: Path) -> str:
    """Return a user-facing method name without a training-seed suffix."""

    return re.sub(r"_seed_?\d+$", "", model_path.name)


METHOD_DISPLAY_NAMES = {
    "our_rr_motion_taskprogress": "Our",
    "our_rr_motion_taskprogress_permute": "Shuffled pri.",
    "our_rr_only_taskprogress": "w/o pri. cond.",
    "our_motion_only_taskprogress": "w/o rew. red.",
    "vanilla": "Vanilla",
    "om_dedicated": "OM-PPO",
    "om_shared": "OM-PPO (Shared)",
    "mtppo_motion": "MT-PPO",
    "our_rr_motion_random": "Random Priority",
    "our_rr_motion_fixed": "Fixed Priority",
    "our_rr_motion_taskprogress_group": "Group Priority",
}


def _method_display_label(method_name: str) -> str:
    """Return academic display label for a method name."""

    if method_name in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[method_name]
    return method_name.replace("opponent_modeling", "OM-PPO").replace("_", " ").title()


def _method_sort_key(model_path: Path) -> tuple[int, str, str, str]:
    """Sort known methods by category order, then keep unknown names stable."""

    method_name = _method_label(model_path)
    return (
        METHOD_ORDER_INDEX.get(method_name, len(METHOD_ORDER)),
        method_name,
        model_path.name,
        str(model_path),
    )


def get_figure_directory(
    root_path: Path,
    keywords: Sequence[str] = (),
) -> Path:
    """Return the output directory for summary figures."""
    figure_directory = root_path / "00_fig"
    if keywords:
        figure_directory /= "-".join(keywords)
    return figure_directory


def configure_matplotlib_fonts(
    usetex: bool = False, usetex_font: str = "times"
) -> None:
    """Configure Matplotlib backend, LaTeX typesetting, and paper typography."""
    import matplotlib

    matplotlib.use("Agg")
    if usetex:
        font_rc: dict[str, Any] = {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
        if usetex_font.lower() == "helvetica":
            font_rc.update(
                {
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                }
            )
        else:
            font_rc.update(
                {
                    "font.family": "serif",
                    "font.serif": [
                        "Times New Roman",
                        "Times",
                        "DejaVu Serif",
                        "serif",
                    ],
                    "mathtext.fontset": "stix",
                }
            )
        matplotlib.rcParams.update(font_rc)
    else:
        matplotlib.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            }
        )


def save_summary_barplots(
    root_path: Path,
    jobs: Sequence[EvaluationJob],
    keywords: Sequence[str] = (),
    usetex: bool = False,
    usetex_font: str = "times",
) -> list[Path]:
    """Save method-level barplots from hierarchically aggregated summaries.

    Each trained model contributes one value after its complete evaluation-seed
    summaries are aggregated. Bars show the mean across trained models, with
    standard deviations and individual model values.
    """

    configure_matplotlib_fonts(usetex=usetex, usetex_font=usetex_font)
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, int], list[EvaluationJob]] = defaultdict(list)
    for job in jobs:
        grouped[(job.scenario_type, job.n_agents)].append(job)

    figure_directory = get_figure_directory(root_path, keywords)
    figure_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    def save_metric_figure(
        metric_plot_data: Mapping[str, tuple[list[str], list[list[float]]]],
        success_try_annotations: Sequence[str] | None,
        figure_width: float,
        figure_height: float,
        figure_title: str,
        output_path: Path,
    ) -> list[Path]:
        """Render one trained-model aggregate summary figure."""

        figure, axes = plt.subplots(
            nrows=len(PLOT_METRICS),
            ncols=1,
            figsize=(figure_width, figure_height),
        )
        try:
            for panel_index, (axis, (metric_name, _, _)) in enumerate(
                zip(axes, PLOT_METRICS)
            ):
                is_last_panel = panel_index == len(PLOT_METRICS) - 1
                metric_label = PLOT_METRIC_LABELS.get(metric_name, metric_name)
                axis.set_title(metric_label, fontsize=PLOT_FONT_SIZE)
                axis_limits = PLOT_GLOBAL_Y_LIMITS.get(
                    metric_name,
                    PLOT_Y_LIMITS.get(scenario_type, {}).get(
                        metric_name,
                        PLOT_DEFAULT_Y_LIMITS.get(metric_name),
                    ),
                )
                if axis_limits is not None:
                    axis.set_ylim(*axis_limits)
                plot_data = metric_plot_data.get(metric_name)
                if plot_data is None:
                    axis.text(
                        0.5,
                        0.5,
                        "No complete seed data",
                        ha="center",
                        va="center",
                        fontsize=PLOT_FONT_SIZE,
                        transform=axis.transAxes,
                    )
                    axis.set_xticks([])
                    axis.set_yticks([])
                    continue

                method_names, raw_values_by_method = plot_data
                metric_scale = PLOT_METRIC_SCALES.get(metric_name, 1.0)
                values_by_method = [
                    [value * metric_scale for value in values]
                    for values in raw_values_by_method
                ]
                means = [fmean(values) for values in values_by_method]
                standard_deviations = [pstdev(values) for values in values_by_method]
                method_colors = [
                    METHOD_COLORS.get(method_name, UNKNOWN_METHOD_COLOR)
                    for method_name in method_names
                ]
                x_positions = list(range(len(method_names)))

                axis.bar(
                    x_positions,
                    means,
                    yerr=standard_deviations,
                    capsize=4,
                    color=method_colors,
                    edgecolor="#333333",
                    linewidth=0.8,
                    zorder=2,
                )

                # Keep the trained-model observations visible while retaining
                # a conventional mean bar for each method.
                model_spread = 0.24
                for method_index, values in enumerate(values_by_method):
                    sample_count = len(values)
                    denominator = max(sample_count - 1, 1)
                    for sample_index, value in enumerate(values):
                        offset = (
                            model_spread
                            * (sample_index - (sample_count - 1) / 2)
                            / denominator
                        )
                        axis.scatter(
                            [x_positions[method_index] + offset],
                            [value],
                            color="black",
                            s=22,
                            zorder=3,
                        )

                baseline_index = next(
                    (
                        index
                        for index, method_name in enumerate(method_names)
                        if method_name == "vanilla"
                    ),
                    None,
                )
                if baseline_index is None:
                    print(
                        f"[WARNING] Vanilla baseline not found for "
                        f"{metric_name} ({scenario_type}, n_agents={n_agents}); "
                        "skipping baseline reference line.",
                        file=sys.stderr,
                    )
                else:
                    baseline_mean = means[baseline_index]
                    axis.axhline(
                        baseline_mean,
                        color=METHOD_COLORS["vanilla"],
                        linestyle="--",
                        linewidth=1.6,
                        label=f"Vanilla mean ({baseline_mean:.6g})",
                        zorder=1.5,
                    )

                lower_is_better = metric_name.startswith("collision_rate_")
                best_index = (
                    min(range(len(means)), key=means.__getitem__)
                    if lower_is_better
                    else max(range(len(means)), key=means.__getitem__)
                )
                best_method_name = method_names[best_index]
                best_mean = means[best_index]
                best_prefix = "Best (lowest)" if lower_is_better else "Best"
                best_label = f"{best_prefix}: {best_method_name} mean ({best_mean:.6g})"
                if best_method_name == "vanilla":
                    best_label = (
                        f"Vanilla mean / {best_prefix.lower()} ({best_mean:.6g})"
                    )
                axis.axhline(
                    best_mean,
                    color=method_colors[best_index],
                    linestyle="--",
                    linewidth=1.6,
                    label=best_label,
                    zorder=1.5,
                )

                if (
                    metric_name == "task_success_rate"
                    and success_try_annotations is not None
                ):
                    for x_position, mean, standard_deviation, annotation in zip(
                        x_positions,
                        means,
                        standard_deviations,
                        success_try_annotations,
                    ):
                        axis.annotate(
                            annotation,
                            (x_position, mean + standard_deviation),
                            xytext=(0, 6),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=PLOT_FONT_SIZE,
                            clip_on=False,
                        )

                display_method_names = [
                    _method_display_label(method_name) for method_name in method_names
                ]
                if is_last_panel:
                    axis.set_xticks(
                        x_positions,
                        display_method_names,
                        rotation=45,
                        ha="right",
                    )
                    axis.set_xlabel("Method", fontsize=PLOT_FONT_SIZE)
                else:
                    axis.set_xticks(x_positions)
                    axis.tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        labelbottom=False,
                    )
                axis.set_ylabel(metric_label, fontsize=PLOT_FONT_SIZE)
                axis.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
                axis.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
                axis.set_axisbelow(True)
                axis.legend(loc="best", fontsize=PLOT_FONT_SIZE)

            png_output_path = output_path.with_suffix(".png")
            pdf_output_path = output_path.with_suffix(".pdf")
            figure.tight_layout(rect=(0, 0, 1, 0.98))
            figure.savefig(png_output_path, dpi=300)
            figure.savefig(pdf_output_path, dpi=300)
        finally:
            plt.close(figure)
        return [png_output_path, pdf_output_path]

    figure_progress = tqdm(
        sorted(grouped.items()),
        desc="Saving summary figures",
        unit="setting",
    )
    for (scenario_type, n_agents), setting_jobs in figure_progress:
        figure_progress.set_postfix(
            scenario=scenario_type,
            n_agents=n_agents,
            refresh=False,
        )
        expected_seeds, model_paths, summaries_by_model = _collect_setting_summaries(
            setting_jobs
        )
        aggregates_by_model = {
            model_path: _aggregate_model_summaries(
                summaries_by_model[model_path], expected_seeds
            )
            for model_path in model_paths
        }
        complete_model_paths_by_metric: dict[str, list[Path]] = {}
        for metric_name, _, _ in METRICS:
            complete_methods = [
                model_path
                for model_path in model_paths
                if metric_name in aggregates_by_model[model_path].metric_values
            ]
            incomplete_methods = [
                model_path
                for model_path in model_paths
                if model_path not in complete_methods
            ]
            if incomplete_methods:
                names = ", ".join(path.name for path in incomplete_methods)
                print(
                    f"[WARNING] Skipping incomplete seed data for "
                    f"{metric_name} ({scenario_type}, n_agents={n_agents}): {names}",
                    file=sys.stderr,
                )
            if not complete_methods or not expected_seeds:
                print(
                    f"[WARNING] No complete seed data for {metric_name} "
                    f"({scenario_type}, n_agents={n_agents}); skipping subplot.",
                    file=sys.stderr,
                )
                continue
            complete_model_paths_by_metric[metric_name] = complete_methods

        if not complete_model_paths_by_metric:
            continue

        complete_try_paths = set(
            complete_model_paths_by_metric.get("num_task_tries", [])
        )

        def format_success_try_annotation(model_paths: Sequence[Path]) -> str:
            """Format mean successful tries over mean total tries."""

            complete_paths = [
                model_path
                for model_path in model_paths
                if model_path in complete_try_paths
            ]
            if not complete_paths:
                return "— / —"

            tries_values: list[float] = []
            success_try_values: list[float] = []
            for model_path in complete_paths:
                aggregate = aggregates_by_model[model_path]
                if (
                    aggregate.mean_num_task_tries is None
                    or aggregate.mean_task_success_times is None
                ):
                    continue
                tries_values.append(aggregate.mean_num_task_tries)
                success_try_values.append(aggregate.mean_task_success_times)
            if not tries_values:
                return "— / —"
            return f"{fmean(success_try_values):.1f} / {fmean(tries_values):.1f}"

        seed_count = len(expected_seeds)

        merged_metric_plot_data: dict[str, tuple[list[str], list[list[float]]]] = {}
        for metric_name, _, _ in METRICS:
            complete_methods = complete_model_paths_by_metric.get(metric_name)
            if complete_methods is None:
                continue

            values_by_method: dict[str, list[float]] = defaultdict(list)
            for model_path in complete_methods:
                values_by_method[_method_label(model_path)].append(
                    aggregates_by_model[model_path].metric_values[metric_name]
                )
            merged_method_names = sorted(
                values_by_method,
                key=lambda method_name: (
                    METHOD_ORDER_INDEX.get(method_name, len(METHOD_ORDER)),
                    method_name,
                ),
            )
            merged_metric_plot_data[metric_name] = (
                merged_method_names,
                [values_by_method[method_name] for method_name in merged_method_names],
            )

        merged_success_annotations = None
        merged_success_model_paths_by_method: dict[str, list[Path]] = defaultdict(list)
        for model_path in complete_model_paths_by_metric.get("task_success_rate", []):
            merged_success_model_paths_by_method[_method_label(model_path)].append(
                model_path
            )
        merged_success_data = merged_metric_plot_data.get("task_success_rate")
        if merged_success_data is not None:
            merged_success_annotations = [
                format_success_try_annotation(
                    merged_success_model_paths_by_method[method_name]
                )
                for method_name in merged_success_data[0]
            ]

        merged_method_count = max(
            (
                len(merged_metric_plot_data[metric_name][0])
                for metric_name, _, _ in PLOT_METRICS
                if metric_name in merged_metric_plot_data
            ),
            default=0,
        )
        if merged_method_count:
            merged_output_path = (
                figure_directory
                / f"{scenario_type}_n_agents_{n_agents}_metrics_merged_seeds.png"
            )
            saved_paths.extend(
                save_metric_figure(
                    merged_metric_plot_data,
                    success_try_annotations=merged_success_annotations,
                    figure_width=PLOT_WIDTH,
                    figure_height=PLOT_HEIGHT,
                    figure_title=(
                        f"{scenario_type}, n_agents={n_agents} "
                        f"(trained-model aggregates; {seed_count} evaluation seeds/model)"
                    ),
                    output_path=merged_output_path,
                )
            )

    figure_progress.close()
    return saved_paths


def save_cross_iter_heatmaps(
    root_path: Path,
    jobs: Sequence[EvaluationJob],
    keywords: Sequence[str] = (),
    usetex: bool = False,
    usetex_font: str = "times",
) -> list[Path]:
    """Save cross-iteration CR and TSR colormaps with colorbars and cell annotations.

    Aggregates metrics for each method and scenario setting across parallel environments
    and evaluation seeds for each training seed, then averages across training seeds.
    """
    configure_matplotlib_fonts(usetex=usetex, usetex_font=usetex_font)
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    if not jobs:
        return []

    figure_directory = get_figure_directory(root_path, keywords)
    figure_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    grouped: dict[tuple[str, int, str], list[EvaluationJob]] = defaultdict(list)
    for job in jobs:
        grouped[(job.scenario_type, job.n_agents, job.ego_tag)].append(job)

    for (scenario_type, n_agents, ego_tag), setting_jobs in sorted(grouped.items()):
        models_by_method: dict[str, list[Path]] = defaultdict(list)
        for job in setting_jobs:
            method = _method_label(job.model_path)
            if job.model_path not in models_by_method[method]:
                models_by_method[method].append(job.model_path)

        sorted_methods = sorted(
            models_by_method.keys(),
            key=lambda m: (METHOD_ORDER_INDEX.get(m, len(METHOD_ORDER)), m),
        )

        for method_name in sorted_methods:
            method_models = models_by_method[method_name]
            method_jobs = [
                j for j in setting_jobs if _method_label(j.model_path) == method_name
            ]

            sorted_ego = sorted(
                {j.ego_iter for j in method_jobs if j.ego_iter is not None}
            )
            sorted_peer = sorted(
                {j.peer_iter for j in method_jobs if j.peer_iter is not None}
            )

            if not sorted_ego or not sorted_peer:
                continue

            cr_grid = np.full((len(sorted_peer), len(sorted_ego)), np.nan)
            tsr_grid = np.full((len(sorted_peer), len(sorted_ego)), np.nan)

            for p_idx, peer_iter in enumerate(sorted_peer):
                for e_idx, ego_iter in enumerate(sorted_ego):
                    model_cr_list: list[float] = []
                    model_tsr_list: list[float] = []
                    for model_path in method_models:
                        eval_seed_crs: list[float] = []
                        eval_seed_tsrs: list[float] = []
                        for j in method_jobs:
                            if (
                                j.model_path == model_path
                                and j.ego_iter == ego_iter
                                and j.peer_iter == peer_iter
                            ):
                                summ = read_summary(j.summary_path)
                                if summ is not None:
                                    cr_val = nested_value(
                                        summ, ("collision_rate_steps",)
                                    )
                                    if cr_val is not None:
                                        eval_seed_crs.append(float(cr_val) * 100.0)
                                    tsr_val = nested_value(
                                        summ,
                                        ("task_performance", "task_success_rate"),
                                    )
                                    if tsr_val is not None:
                                        eval_seed_tsrs.append(float(tsr_val) * 100.0)
                        if eval_seed_crs:
                            model_cr_list.append(fmean(eval_seed_crs))
                        if eval_seed_tsrs:
                            model_tsr_list.append(fmean(eval_seed_tsrs))

                    if model_cr_list:
                        cr_grid[p_idx, e_idx] = fmean(model_cr_list)
                    if model_tsr_list:
                        tsr_grid[p_idx, e_idx] = fmean(model_tsr_list)

            pct_sym = r"\%" if usetex else "%"
            specs = [
                ("CR", f"CR [{pct_sym}]", cr_grid, "YlOrRd"),
                ("TSR", f"TSR [{pct_sym}]", tsr_grid, "Blues"),
            ]

            for metric_tag, cbar_label, grid, cmap_name in specs:
                fig, ax = plt.subplots(figsize=(4.0, 1.8), constrained_layout=True)
                cmap = plt.get_cmap(cmap_name).copy()
                cmap.set_bad("#E0E0E0")

                if metric_tag == "TSR":
                    vmin, vmax = 40.0, 95.0
                else:
                    valid_vals = grid[~np.isnan(grid)]
                    if len(valid_vals) > 0:
                        vmin = float(np.min(valid_vals))
                        vmax = float(np.max(valid_vals))
                        if vmin == vmax:
                            vmin = max(0.0, vmin - 1.0)
                            vmax = vmax + 1.0
                    else:
                        vmin, vmax = 0.0, 1.0

                im = ax.imshow(
                    grid,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                ax.set_xticks(range(len(sorted_ego)))
                ax.set_yticks(range(len(sorted_peer)))
                ax.set_xticklabels([str(e) for e in sorted_ego], fontsize=11)
                ax.set_yticklabels([str(p) for p in sorted_peer], fontsize=11)
                ax.set_xlabel("Ego Checkp. Iter.", fontsize=13)
                ax.set_ylabel("Peer Checkp. Iter.", fontsize=13)

                is_highlight_fixed_ego_and_vary_peers: bool = True
                if is_highlight_fixed_ego_and_vary_peers:
                    target_mins = [
                        float(np.min(valid)) if len(valid) > 0 else None
                        for x_i in range(grid.shape[1])
                        for valid in [grid[:, x_i][~np.isnan(grid[:, x_i])]]
                    ]
                else:
                    target_mins = [
                        float(np.min(valid)) if len(valid) > 0 else None
                        for y_i in range(grid.shape[0])
                        for valid in [grid[y_i, :][~np.isnan(grid[y_i, :])]]
                    ]

                for y_i in range(grid.shape[0]):
                    for x_i in range(grid.shape[1]):
                        val = grid[y_i, x_i]
                        min_val = (
                            target_mins[x_i]
                            if is_highlight_fixed_ego_and_vary_peers
                            else target_mins[y_i]
                        )
                        if (
                            min_val is not None
                            and not np.isnan(val)
                            and np.isclose(val, min_val)
                        ):
                            ellipse = mpatches.Ellipse(
                                (x_i, y_i + 0.05),
                                width=0.75,
                                height=0.68,
                                fill=False,
                                edgecolor="tab:red",
                                alpha=0.65,
                                linewidth=1.2,
                                zorder=2,
                            )
                            ax.add_patch(ellipse)

                        text = "N/A" if np.isnan(val) else f"{val:.1f}"
                        is_diagonal = sorted_ego[x_i] == sorted_peer[y_i]
                        display_text = (
                            rf"\textbf{{{text}}}" if (usetex and is_diagonal) else text
                        )
                        ax.text(
                            x_i,
                            y_i,
                            display_text,
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=12,
                            fontweight="bold" if is_diagonal else "normal",
                            zorder=3,
                        )

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if metric_tag == "TSR":
                    cbar.ax.set_ylim(50, 85)
                    cbar.set_ticks([50, 60, 70, 80, 85])
                cbar.set_label(cbar_label, fontsize=13)
                cbar.ax.tick_params(labelsize=11)

                ego_suffix = f"_ego{ego_tag}" if ego_tag != "0" else ""
                base_filename = (
                    f"{scenario_type}_n_agents_{n_agents}_{method_name}"
                    f"_cross_iter{ego_suffix}_{metric_tag}"
                )
                png_path = figure_directory / f"{base_filename}.png"
                pdf_path = figure_directory / f"{base_filename}.pdf"
                fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
                fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
                plt.close(fig)

                saved_paths.append(png_path)
                saved_paths.append(pdf_path)

    return saved_paths


def markdown_escape(value: str) -> str:
    """Escape a value before placing it in a Markdown table cell."""

    return value.replace("|", r"\|").replace("\n", " ")


@dataclass
class MethodSettingAggregate:
    """Aggregated metrics across trained models (seeds) for one method and setting."""

    method_key: str
    display_label: str
    model_count: int
    metric_means: dict[str, float]
    metric_stds: dict[str, float]
    metric_mins: dict[str, float]
    metric_maxs: dict[str, float]
    mean_success_tries: float | None = None
    mean_total_tries: float | None = None


def collect_method_setting_aggregates(
    setting_jobs: Sequence[EvaluationJob],
) -> tuple[list[int], list[MethodSettingAggregate]]:
    """Compute cross-seed aggregates per method for one scenario/agent setting."""

    expected_seeds, model_paths, summaries_by_model = _collect_setting_summaries(
        setting_jobs
    )
    aggregates_by_model = {
        model_path: _aggregate_model_summaries(
            summaries_by_model[model_path], expected_seeds
        )
        for model_path in model_paths
    }

    models_by_method: dict[str, list[Path]] = defaultdict(list)
    for model_path in model_paths:
        models_by_method[_method_label(model_path)].append(model_path)

    sorted_methods = sorted(
        models_by_method.keys(),
        key=lambda m: (METHOD_ORDER_INDEX.get(m, len(METHOD_ORDER)), m),
    )

    method_aggregates: list[MethodSettingAggregate] = []
    for method_key in sorted_methods:
        method_model_paths = models_by_method[method_key]
        display_label = _method_display_label(method_key)

        metric_means: dict[str, float] = {}
        metric_stds: dict[str, float] = {}
        metric_mins: dict[str, float] = {}
        metric_maxs: dict[str, float] = {}

        for metric_name, _, _ in METRICS:
            values = [
                aggregates_by_model[path].metric_values[metric_name]
                for path in method_model_paths
                if metric_name in aggregates_by_model[path].metric_values
            ]
            if values:
                metric_means[metric_name] = fmean(values)
                metric_stds[metric_name] = pstdev(values) if len(values) > 1 else 0.0
                metric_mins[metric_name] = min(values)
                metric_maxs[metric_name] = max(values)

        success_times = [
            aggregates_by_model[path].mean_task_success_times
            for path in method_model_paths
            if aggregates_by_model[path].mean_task_success_times is not None
        ]
        total_tries = [
            aggregates_by_model[path].mean_num_task_tries
            for path in method_model_paths
            if aggregates_by_model[path].mean_num_task_tries is not None
        ]

        mean_success_tries = fmean(success_times) if success_times else None
        mean_total_tries = fmean(total_tries) if total_tries else None

        method_aggregates.append(
            MethodSettingAggregate(
                method_key=method_key,
                display_label=display_label,
                model_count=len(method_model_paths),
                metric_means=metric_means,
                metric_stds=metric_stds,
                metric_mins=metric_mins,
                metric_maxs=metric_maxs,
                mean_success_tries=mean_success_tries,
                mean_total_tries=mean_total_tries,
            )
        )

    return expected_seeds, method_aggregates


def format_stat(
    mean: float | None,
    std: float | None,
    scale: float = 1.0,
    unit: str = "",
    precision: int = 2,
    is_bold: bool = False,
    is_latex: bool = False,
) -> str:
    """Format a mean ± std value cleanly for Markdown or LaTeX."""

    if mean is None:
        return r"\textemdash" if is_latex else MISSING_VALUE
    scaled_mean = mean * scale
    if is_latex:
        if std is None or math.isnan(std):
            formatted = f"{scaled_mean:.{precision}f}{unit}"
            return f"\\textbf{{{formatted}}}" if is_bold else formatted
        scaled_std = std * scale
        val_str = f"{scaled_mean:.{precision}f}"
        std_str = f"{scaled_std:.{precision}f}{unit}"
        if is_bold:
            return f"\\textbf{{{val_str}}} $\\pm$ \\textbf{{{std_str}}}"
        return f"{val_str} $\\pm$ {std_str}"

    if std is None or math.isnan(std):
        formatted = f"{scaled_mean:.{precision}f}{unit}"
    else:
        scaled_std = std * scale
        formatted = f"{scaled_mean:.{precision}f} ± {scaled_std:.{precision}f}{unit}"
    if is_bold:
        return f"**{formatted}**"
    return formatted


def format_academic_number(val: float | None) -> str:
    """Format a number according to universal academic precision rules:

    1. If |val| < 1.0:
       - Default: 2 digits after the comma (e.g. 0.34, 0.55, 0.99, 0.07).
       - Exception: If rounding to 2 digits yields 0.00 (or if |val| < 0.01), retain as
         many decimal digits as needed until the first non-zero digit (e.g. 0.004, 0.008, 0.0000006).
    2. If |val| >= 1.0:
       - 1.0 <= |val| < 10.0: 3 valid digits (1 integer + 2 decimals, e.g. 1.23, 9.01).
       - 10.0 <= |val| < 100.0: 3 valid digits (2 integers + 1 decimal, e.g. 56.6, 99.2, 33.0).
       - |val| >= 100.0: retain at least 1 digit after comma (e.g. 134.2, 678.1, 100.0).
    """

    if val is None:
        return r"\textemdash"
    if abs(val) < 1e-12:
        return "0.00"

    sign = "-" if val < 0.0 else ""
    abs_v = abs(val)

    if abs_v < 1.0:
        if round(abs_v, 2) == 0.0 or abs_v < 0.01:
            first_nonzero_pos = int(math.floor(-math.log10(abs_v))) + 1
            return f"{sign}{abs_v:.{first_nonzero_pos}f}"
        return f"{sign}{abs_v:.2f}"
    elif abs_v < 10.0:
        return f"{sign}{abs_v:.2f}"
    elif abs_v < 100.0:
        return f"{sign}{abs_v:.1f}"
    else:
        return f"{sign}{abs_v:.1f}"


def build_method_summary_markdown(
    grouped_jobs: dict[tuple[str, int], list[EvaluationJob]],
) -> str:
    """Generate clean, publication-ready method-level comparison Markdown tables."""

    lines = [
        "# Method-Level Evaluation Summary",
        "",
        "Aggregated performance metrics across independent training seeds "
        "(reported as mean ± standard deviation across models).",
        "",
    ]

    for (scenario_type, n_agents), setting_jobs in sorted(grouped_jobs.items()):
        expected_seeds, aggregates = collect_method_setting_aggregates(setting_jobs)
        if not aggregates:
            continue

        seed_label = ", ".join(str(s) for s in expected_seeds)
        lines.extend(
            [
                f"## `{scenario_type}`, `n_agents={n_agents}`",
                f"*Evaluation seeds per model: {seed_label}*",
                "",
                "| Method | Models | Success Rate (%) | Success / Tries | Collision Rate (%) | Mean Speed (m/s) | Throughput (veh/s) | Reward / min |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        # Identify best values for bolding
        best_success = max(
            (
                agg.metric_means.get("task_success_rate")
                for agg in aggregates
                if "task_success_rate" in agg.metric_means
            ),
            default=None,
        )
        best_collision = min(
            (
                agg.metric_means.get("collision_rate_steps")
                for agg in aggregates
                if "collision_rate_steps" in agg.metric_means
            ),
            default=None,
        )
        best_speed = max(
            (
                agg.metric_means.get("mean_speed_mps")
                for agg in aggregates
                if "mean_speed_mps" in agg.metric_means
            ),
            default=None,
        )
        best_throughput = max(
            (
                agg.metric_means.get("vehicles_per_second")
                for agg in aggregates
                if "vehicles_per_second" in agg.metric_means
            ),
            default=None,
        )
        best_reward = max(
            (
                agg.metric_means.get(
                    "parameter_independent_reward_per_agent_per_minute"
                )
                for agg in aggregates
                if "parameter_independent_reward_per_agent_per_minute"
                in agg.metric_means
            ),
            default=None,
        )

        for agg in aggregates:
            succ_m = agg.metric_means.get("task_success_rate")
            succ_s = agg.metric_stds.get("task_success_rate")
            succ_str = format_stat(
                succ_m,
                succ_s,
                scale=100.0,
                unit="%",
                precision=2,
                is_bold=(
                    succ_m is not None
                    and succ_m == best_success
                    and len(aggregates) > 1
                ),
            )

            if agg.mean_success_tries is not None and agg.mean_total_tries is not None:
                tries_str = f"{agg.mean_success_tries:.1f} / {agg.mean_total_tries:.1f}"
            else:
                tries_str = MISSING_VALUE

            coll_m = agg.metric_means.get("collision_rate_steps")
            coll_s = agg.metric_stds.get("collision_rate_steps")
            coll_str = format_stat(
                coll_m,
                coll_s,
                scale=100.0,
                unit="%",
                precision=3,
                is_bold=(
                    coll_m is not None
                    and coll_m == best_collision
                    and len(aggregates) > 1
                ),
            )

            speed_m = agg.metric_means.get("mean_speed_mps")
            speed_s = agg.metric_stds.get("mean_speed_mps")
            speed_str = format_stat(
                speed_m,
                speed_s,
                scale=1.0,
                precision=3,
                is_bold=(
                    speed_m is not None
                    and speed_m == best_speed
                    and len(aggregates) > 1
                ),
            )

            thru_m = agg.metric_means.get("vehicles_per_second")
            thru_s = agg.metric_stds.get("vehicles_per_second")
            thru_str = format_stat(
                thru_m,
                thru_s,
                scale=1.0,
                precision=3,
                is_bold=(
                    thru_m is not None
                    and thru_m == best_throughput
                    and len(aggregates) > 1
                ),
            )

            rew_m = agg.metric_means.get(
                "parameter_independent_reward_per_agent_per_minute"
            )
            rew_s = agg.metric_stds.get(
                "parameter_independent_reward_per_agent_per_minute"
            )
            rew_str = format_stat(
                rew_m,
                rew_s,
                scale=1.0,
                precision=2,
                is_bold=(
                    rew_m is not None and rew_m == best_reward and len(aggregates) > 1
                ),
            )

            lines.append(
                f"| **{agg.display_label}** | {agg.model_count} | {succ_str} | {tries_str} | {coll_str} | {speed_str} | {thru_str} | {rew_str} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


SCENARIO_DISPLAY_NAMES = {
    "cpm_mixed": "CPM Mixed",
    "intersection_4": "Intersection",
    "roundabout_2": "Roundabout",
    "cpm_entire": "CPM Entire",
    "interchange_3": "Interchange",
    "cpm_intersection_extended": "Intersection (Ext.)",
}


def build_hierarchical_latex_table(
    grouped_jobs: dict[tuple[str, int], list[EvaluationJob]],
    method_columns: list[tuple[str, str]],
    caption: str,
    label: str,
    cr_precision: int = 2,
    tsr_precision: int = 2,
) -> str:
    """Build a publication-style hierarchical multi-scenario/agent-count LaTeX table."""

    scenarios: dict[str, list[int]] = defaultdict(list)
    for scenario_type, n_agents in sorted(grouped_jobs.keys()):
        if n_agents not in scenarios[scenario_type]:
            scenarios[scenario_type].append(n_agents)

    col_spec = "lll" + "c" * len(method_columns)
    headers = [r"\textbf{Scenario}", r"\textbf{\#Agents}", r"\textbf{Metrics}"] + [
        f"\\textbf{{{col_label}}}" for _, col_label in method_columns
    ]

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    scenario_keys = sorted(scenarios.keys())
    for s_idx, scenario_type in enumerate(scenario_keys):
        agent_counts = sorted(scenarios[scenario_type])
        scenario_display = SCENARIO_DISPLAY_NAMES.get(
            scenario_type, scenario_type.replace("_", " ").title()
        )
        total_rows = len(agent_counts) * 2

        for a_idx, n_agents in enumerate(agent_counts):
            setting_jobs = grouped_jobs.get((scenario_type, n_agents), [])
            _, aggregates = collect_method_setting_aggregates(setting_jobs)
            agg_by_key = {agg.method_key: agg for agg in aggregates}

            cr_means: dict[str, float | None] = {}
            cr_stds: dict[str, float | None] = {}
            tsr_means: dict[str, float | None] = {}
            tsr_stds: dict[str, float | None] = {}

            for m_key, _ in method_columns:
                agg = agg_by_key.get(m_key)
                if agg is not None:
                    cr_means[m_key] = agg.metric_means.get("collision_rate_steps")
                    cr_stds[m_key] = agg.metric_stds.get("collision_rate_steps")
                    tsr_means[m_key] = agg.metric_means.get("task_success_rate")
                    tsr_stds[m_key] = agg.metric_stds.get("task_success_rate")
                else:
                    cr_means[m_key] = None
                    cr_stds[m_key] = None
                    tsr_means[m_key] = None
                    tsr_stds[m_key] = None

            valid_cr = [v for v in cr_means.values() if v is not None]
            best_cr = min(valid_cr) if valid_cr else None

            valid_tsr = [v for v in tsr_means.values() if v is not None]
            best_tsr = max(valid_tsr) if valid_tsr else None

            cr_cells = []
            for m_key, _ in method_columns:
                m_val = cr_means[m_key]
                s_val = cr_stds[m_key]
                prec = (
                    3 if (m_val is not None and m_val * 100.0 < 0.05) else cr_precision
                )
                cr_str = format_stat(
                    m_val,
                    s_val,
                    scale=100.0,
                    precision=prec,
                    is_latex=True,
                    is_bold=(
                        m_val is not None and m_val == best_cr and len(valid_cr) > 1
                    ),
                )
                cr_cells.append(cr_str)

            tsr_cells = []
            for m_key, _ in method_columns:
                m_val = tsr_means[m_key]
                s_val = tsr_stds[m_key]
                tsr_str = format_stat(
                    m_val,
                    s_val,
                    scale=100.0,
                    precision=tsr_precision,
                    is_latex=True,
                    is_bold=(
                        m_val is not None and m_val == best_tsr and len(valid_tsr) > 1
                    ),
                )
                tsr_cells.append(tsr_str)

            scenario_prefix = (
                f"\\multirow{{{total_rows}}}{{*}}{{{scenario_display}}}"
                if a_idx == 0
                else ""
            )
            agent_prefix = f"\\multirow{{2}}{{*}}{{{n_agents}}}"

            lines.append(
                f"{scenario_prefix} & {agent_prefix} & CR [\\%] & "
                + " & ".join(cr_cells)
                + r" \\"
            )
            lines.append(f" & & TSR [\\%] & " + " & ".join(tsr_cells) + r" \\")

            if a_idx < len(agent_counts) - 1:
                lines.append(f"\\cmidrule(lr){{2-{3 + len(method_columns)}}}")

        if s_idx < len(scenario_keys) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


COMPARISON_METHOD_COLUMNS = [
    ("our_rr_motion_taskprogress", "Our"),
    ("vanilla", "Vanilla"),
    ("om_dedicated", "OM-PPO"),
    ("mtppo_motion", "MT-PPO"),
]

ABLATION_METHOD_COLUMNS = [
    ("our_rr_motion_taskprogress", "Our"),
    ("our_rr_motion_taskprogress_permute", "Shuffled pri."),
    ("our_rr_only_taskprogress", "w/o pri. cond."),
    ("our_motion_only_taskprogress", "w/o rew. red."),
]

COMBINED_COMPACT_METHOD_COLUMNS = [
    ("vanilla", "Vanilla"),
    ("om_dedicated", "OM-PPO"),
    ("mtppo_motion", "MT-PPO"),
    ("our_rr_motion_taskprogress", "Our"),
    ("our_rr_motion_taskprogress_permute", "Shuffled pri."),
    ("our_rr_only_taskprogress", "w/o pri. cond."),
    ("our_motion_only_taskprogress", "w/o rew. red."),
]


def build_comparison_latex_table(
    grouped_jobs: dict[tuple[str, int], list[EvaluationJob]],
) -> str:
    """Generate the baseline comparison LaTeX table for state-of-the-art methods."""

    caption = (
        "Comparison with state-of-the-art methods across scenarios and vehicle densities. "
        "Metrics are reported as mean $\\pm$ standard deviation across 5 independent training seeds "
        "(evaluated over 4 seeds per model). The best result in each row is highlighted in \\textbf{bold}."
    )
    return build_hierarchical_latex_table(
        grouped_jobs,
        COMPARISON_METHOD_COLUMNS,
        caption=caption,
        label="tab:sota_comparison",
    )


def build_ablation_latex_table(
    grouped_jobs: dict[tuple[str, int], list[EvaluationJob]],
) -> str:
    """Generate the ablation study LaTeX table."""

    caption = (
        "Ablation study across scenarios and vehicle densities. "
        "Metrics are reported as mean $\\pm$ standard deviation across 5 independent training seeds "
        "(evaluated over 4 seeds per model). The best result in each row is highlighted in \\textbf{bold}."
    )
    return build_hierarchical_latex_table(
        grouped_jobs,
        ABLATION_METHOD_COLUMNS,
        caption=caption,
        label="tab:ablation_study",
    )


def build_combined_compact_latex_table(
    grouped_jobs: dict[tuple[str, int], list[EvaluationJob]],
    caption: str = (
        "Quantitative comparison with baseline MARL methods and ablation variants "
        "across scenarios and vehicle densities (reported as CR [\\%] / TSR [\\%])."
    ),
    label: str = "tab:combined_evaluation_compact",
) -> str:
    """Generate a compact single-line-per-setting LaTeX table combining baselines and ablation variants."""

    scenarios: dict[str, list[int]] = defaultdict(list)
    for scenario_type, n_agents in sorted(grouped_jobs.keys()):
        if n_agents not in scenarios[scenario_type]:
            scenarios[scenario_type].append(n_agents)

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\footnotesize",
        r"\begin{tabular}{ll lll c rrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{Baselines}} & \textbf{Proposed} & \multicolumn{3}{c}{\textbf{Ablation Variants}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-6} \cmidrule(lr){7-9}",
        r"\textbf{Scenario} & \textbf{$N$} & \textbf{Vanilla} & \textbf{OM-PPO} & \textbf{MT-PPO} & \textbf{Our} & \textbf{Shuffled pri.} & \textbf{w/o pri. cond.} & \textbf{w/o rew. red.} \\",
        r"\midrule",
    ]

    scenario_keys = sorted(scenarios.keys())
    for s_idx, scenario_type in enumerate(scenario_keys):
        agent_counts = sorted(scenarios[scenario_type])
        scenario_display = SCENARIO_DISPLAY_NAMES.get(
            scenario_type, scenario_type.replace("_", " ").title()
        )
        total_rows = len(agent_counts)

        for a_idx, n_agents in enumerate(agent_counts):
            setting_jobs = grouped_jobs.get((scenario_type, n_agents), [])
            _, aggregates = collect_method_setting_aggregates(setting_jobs)
            agg_by_key = {agg.method_key: agg for agg in aggregates}

            cr_means: dict[str, float | None] = {}
            tsr_means: dict[str, float | None] = {}

            for m_key, _ in COMBINED_COMPACT_METHOD_COLUMNS:
                agg = agg_by_key.get(m_key)
                if agg is not None:
                    cr_means[m_key] = agg.metric_means.get("collision_rate_steps")
                    tsr_means[m_key] = agg.metric_means.get("task_success_rate")
                else:
                    cr_means[m_key] = None
                    tsr_means[m_key] = None

            valid_cr = [v for v in cr_means.values() if v is not None]
            best_cr = min(valid_cr) if valid_cr else None

            valid_tsr = [v for v in tsr_means.values() if v is not None]
            best_tsr = max(valid_tsr) if valid_tsr else None

            cells = []
            for m_key, _ in COMBINED_COMPACT_METHOD_COLUMNS:
                cr_val = cr_means[m_key]
                tsr_val = tsr_means[m_key]

                if cr_val is None or tsr_val is None:
                    cells.append(r"\textemdash")
                else:
                    cr_str_val = format_academic_number(cr_val * 100.0)
                    tsr_str_val = format_academic_number(tsr_val * 100.0)

                    best_cr_str = (
                        format_academic_number(best_cr * 100.0)
                        if best_cr is not None
                        else None
                    )
                    best_tsr_str = (
                        format_academic_number(best_tsr * 100.0)
                        if best_tsr is not None
                        else None
                    )

                    if cr_str_val == best_cr_str and len(valid_cr) > 1:
                        cr_formatted = f"\\textbf{{{cr_str_val}}}"
                    else:
                        cr_formatted = cr_str_val

                    if tsr_str_val == best_tsr_str and len(valid_tsr) > 1:
                        tsr_formatted = f"\\textbf{{{tsr_str_val}}}"
                    else:
                        tsr_formatted = tsr_str_val

                    cells.append(f"{cr_formatted} / {tsr_formatted}")

            if a_idx == 0:
                lines.append(f"\\multirow{{{total_rows}}}{{*}}{{{scenario_display}}}")
            lines.append(f" & {n_agents:<2} & " + " & ".join(cells) + r" \\")

        if s_idx < len(scenario_keys) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_method_summary_latex(
    grouped_jobs: dict[tuple[str, int], list[EvaluationJob]],
) -> str:
    """Generate LaTeX tables for manuscript inclusion (SOTA comparison, ablation study, and combined compact)."""

    comp_table = build_comparison_latex_table(grouped_jobs)
    abla_table = build_ablation_latex_table(grouped_jobs)
    compact_table = build_combined_compact_latex_table(grouped_jobs)

    lines = [
        "% ===========================================================================",
        "% LaTeX Tables: Evaluation Summary for Academic Manuscript",
        "% Requires \\usepackage{booktabs} and \\usepackage{multirow}",
        "% ===========================================================================",
        "",
        "% ---------------------------------------------------------------------------",
        "% Table 1: State-of-the-Art / Baseline Comparison (Detailed: Mean ± Std)",
        "% ---------------------------------------------------------------------------",
        comp_table,
        "",
        "% ---------------------------------------------------------------------------",
        "% Table 2: Ablation Study (Detailed: Mean ± Std)",
        "% ---------------------------------------------------------------------------",
        abla_table,
        "",
        "% ---------------------------------------------------------------------------",
        "% Table 3: Combined Evaluation & Ablation (Compact: CR [%] / TSR [%])",
        "% ---------------------------------------------------------------------------",
        compact_table,
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_method_summary_csv(
    grouped_jobs: dict[tuple[str, int], list[EvaluationJob]],
) -> str:
    """Generate a clean wide CSV table of method-level metrics."""

    headers = [
        "scenario",
        "n_agents",
        "method_key",
        "method_label",
        "model_count",
        "task_success_rate_mean",
        "task_success_rate_std",
        "mean_success_tries",
        "mean_total_tries",
        "collision_rate_steps_mean",
        "collision_rate_steps_std",
        "collision_rate_agents_mean",
        "collision_rate_agents_std",
        "mean_speed_mps_mean",
        "mean_speed_mps_std",
        "vehicles_per_second_mean",
        "vehicles_per_second_std",
        "parameter_independent_reward_mean",
        "parameter_independent_reward_std",
    ]

    rows = [",".join(headers)]
    for (scenario_type, n_agents), setting_jobs in sorted(grouped_jobs.items()):
        _, aggregates = collect_method_setting_aggregates(setting_jobs)
        for agg in aggregates:
            row = [
                scenario_type,
                str(n_agents),
                agg.method_key,
                f'"{agg.display_label}"',
                str(agg.model_count),
                f"{agg.metric_means.get('task_success_rate', ''):.6f}"
                if "task_success_rate" in agg.metric_means
                else "",
                f"{agg.metric_stds.get('task_success_rate', ''):.6f}"
                if "task_success_rate" in agg.metric_stds
                else "",
                f"{agg.mean_success_tries:.2f}"
                if agg.mean_success_tries is not None
                else "",
                f"{agg.mean_total_tries:.2f}"
                if agg.mean_total_tries is not None
                else "",
                f"{agg.metric_means.get('collision_rate_steps', ''):.6f}"
                if "collision_rate_steps" in agg.metric_means
                else "",
                f"{agg.metric_stds.get('collision_rate_steps', ''):.6f}"
                if "collision_rate_steps" in agg.metric_stds
                else "",
                f"{agg.metric_means.get('collision_rate_agents', ''):.6f}"
                if "collision_rate_agents" in agg.metric_means
                else "",
                f"{agg.metric_stds.get('collision_rate_agents', ''):.6f}"
                if "collision_rate_agents" in agg.metric_stds
                else "",
                f"{agg.metric_means.get('mean_speed_mps', ''):.6f}"
                if "mean_speed_mps" in agg.metric_means
                else "",
                f"{agg.metric_stds.get('mean_speed_mps', ''):.6f}"
                if "mean_speed_mps" in agg.metric_stds
                else "",
                f"{agg.metric_means.get('vehicles_per_second', ''):.6f}"
                if "vehicles_per_second" in agg.metric_means
                else "",
                f"{agg.metric_stds.get('vehicles_per_second', ''):.6f}"
                if "vehicles_per_second" in agg.metric_stds
                else "",
                f"{agg.metric_means.get('parameter_independent_reward_per_agent_per_minute', ''):.6f}"
                if "parameter_independent_reward_per_agent_per_minute"
                in agg.metric_means
                else "",
                f"{agg.metric_stds.get('parameter_independent_reward_per_agent_per_minute', ''):.6f}"
                if "parameter_independent_reward_per_agent_per_minute"
                in agg.metric_stds
                else "",
            ]
            rows.append(",".join(row))

    return "\n".join(rows).rstrip() + "\n"


def build_summary_report(
    root_path: Path,
    jobs: Sequence[EvaluationJob],
    precision: int,
) -> str:
    """Build raw per-job, per-model, and method-level evaluation summary tables."""

    grouped: dict[tuple[str, int], list[EvaluationJob]] = defaultdict(list)
    for job in jobs:
        grouped[(job.scenario_type, job.n_agents)].append(job)

    headers = [name for name, _, _ in METRICS]
    lines = [f"# Evaluation summary: `{display_path(root_path)}`", ""]
    for (scenario_type, n_agents), setting_jobs in sorted(grouped.items()):
        expected_seeds, model_paths, summaries_by_model = _collect_setting_summaries(
            setting_jobs
        )
        aggregates_by_model = {
            model_path: _aggregate_model_summaries(
                summaries_by_model[model_path], expected_seeds
            )
            for model_path in model_paths
        }

        # 1. Method-level aggregates (across training seeds)
        _, method_aggregates = collect_method_setting_aggregates(setting_jobs)
        lines.extend(
            [
                f"## `{scenario_type}`, `n_agents={n_agents}`",
                "",
                "### Method-level aggregates (mean ± std across trained models)",
                "",
                "| Method | Models | Success Rate (%) | Success / Tries | Collision Rate (%) | Mean Speed (m/s) | Throughput (veh/s) | Reward / min |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for agg in method_aggregates:
            succ_str = format_stat(
                agg.metric_means.get("task_success_rate"),
                agg.metric_stds.get("task_success_rate"),
                scale=100.0,
                unit="%",
                precision=2,
            )
            tries_str = (
                f"{agg.mean_success_tries:.1f} / {agg.mean_total_tries:.1f}"
                if agg.mean_success_tries is not None
                and agg.mean_total_tries is not None
                else MISSING_VALUE
            )
            coll_str = format_stat(
                agg.metric_means.get("collision_rate_steps"),
                agg.metric_stds.get("collision_rate_steps"),
                scale=100.0,
                unit="%",
                precision=3,
            )
            speed_str = format_stat(
                agg.metric_means.get("mean_speed_mps"),
                agg.metric_stds.get("mean_speed_mps"),
                scale=1.0,
                precision=3,
            )
            thru_str = format_stat(
                agg.metric_means.get("vehicles_per_second"),
                agg.metric_stds.get("vehicles_per_second"),
                scale=1.0,
                precision=3,
            )
            rew_str = format_stat(
                agg.metric_means.get(
                    "parameter_independent_reward_per_agent_per_minute"
                ),
                agg.metric_stds.get(
                    "parameter_independent_reward_per_agent_per_minute"
                ),
                scale=1.0,
                precision=2,
            )
            lines.append(
                f"| **{agg.display_label}** | {agg.model_count} | {succ_str} | {tries_str} | {coll_str} | {speed_str} | {thru_str} | {rew_str} |"
            )
        lines.append("")

        # 2. Per-model aggregate table
        lines.extend(
            [
                "### Testing seeds aggregated within each trained model",
                "",
                "| Model | Evaluation seeds | " + " | ".join(headers) + " |",
                "|---|---|" + "---:|" * len(headers),
            ]
        )
        expected_seed_label = ", ".join(str(seed) for seed in expected_seeds)
        for model_path in model_paths:
            aggregate = aggregates_by_model[model_path]
            values = [
                format_metric_value(
                    aggregate.metric_values.get(metric_name), value_type, precision
                )
                for metric_name, _, value_type in METRICS
            ]
            lines.append(
                "| `"
                + markdown_escape(model_path.name)
                + "` | "
                + expected_seed_label
                + " | "
                + " | ".join(values)
                + " |"
            )
        lines.append("")

        # 3. Raw per-job table
        lines.extend(
            [
                "### Raw individual evaluation seed runs",
                "",
                "| Model | Seed | " + " | ".join(headers) + " |",
                "|---|---:|" + "---:|" * len(headers),
            ]
        )
        for job in sorted(
            setting_jobs, key=lambda item: (item.model_path.name, item.seed)
        ):
            summary = summaries_by_model[job.model_path].get(job.seed)
            values = (
                [MISSING_VALUE] * len(METRICS)
                if summary is None
                else [
                    format_metric_value(
                        nested_value(summary, metric_path), value_type, precision
                    )
                    for _, metric_path, value_type in METRICS
                ]
            )
            lines.append(
                "| `"
                + markdown_escape(job.model_path.name)
                + "` | "
                + str(job.seed)
                + " | "
                + " | ".join(values)
                + " |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_cross_iter_summary_report(
    root_path: Path,
    jobs: Sequence[EvaluationJob],
    precision: int,
) -> tuple[str, str]:
    """Build cross-iteration matrix and detailed run tables in Markdown and CSV."""

    lines = [
        f"# Cross-Iteration Evaluation Summary: `{display_path(root_path)}`",
        "",
        "This evaluation measures policy robustness against changing peers by assigning "
        "the ego agent and peer agents policies from different training iterations.",
        "",
    ]

    csv_rows = [
        "model,ego_iter,peer_iter,seed," + ",".join(name for name, _, _ in METRICS)
    ]

    jobs_by_model: dict[Path, list[EvaluationJob]] = defaultdict(list)
    for job in jobs:
        jobs_by_model[job.model_path].append(job)

    for model_path, m_jobs in sorted(
        jobs_by_model.items(), key=lambda item: item[0].name
    ):
        lines.append(f"## Model: `{model_path.name}`")
        lines.append("")

        results: dict[tuple[int, int], Mapping[str, Any]] = {}
        ego_iters_set: set[int] = set()
        peer_iters_set: set[int] = set()
        for job in m_jobs:
            if job.ego_iter is not None and job.peer_iter is not None:
                ego_iters_set.add(job.ego_iter)
                peer_iters_set.add(job.peer_iter)
                summary = read_summary(job.summary_path)
                if summary is not None:
                    results[(job.ego_iter, job.peer_iter)] = summary

        sorted_ego = sorted(ego_iters_set)
        sorted_peer = sorted(peer_iters_set)

        lines.append(
            "### Parameter-Independent Reward / min (Ego Iter [Row] vs Peer Iter [Column])"
        )
        lines.append("")
        header = (
            "| Ego \\ Peer | " + " | ".join(f"Iter {p}" for p in sorted_peer) + " |"
        )
        sep = "|---|" + "---:|" * len(sorted_peer)
        lines.extend([header, sep])
        for e in sorted_ego:
            row_vals = []
            for p in sorted_peer:
                summ = results.get((e, p))
                if summ is not None:
                    rew = nested_value(
                        summ,
                        ("parameter_independent_reward_per_agent_per_minute",),
                    )
                    row_vals.append(
                        f"{float(rew):.{precision}f}"
                        if rew is not None
                        else MISSING_VALUE
                    )
                else:
                    row_vals.append(MISSING_VALUE)
            lines.append(f"| **Iter {e}** | " + " | ".join(row_vals) + " |")
        lines.append("")

        lines.append("### Detailed Runs")
        lines.append("")
        headers = [name for name, _, _ in METRICS]
        lines.append("| Ego Iter | Peer Iter | Seed | " + " | ".join(headers) + " |")
        lines.append("|---:|---:|---:|" + "---:|" * len(headers))
        for job in sorted(
            m_jobs,
            key=lambda j: (
                j.ego_iter if j.ego_iter is not None else 0,
                j.peer_iter if j.peer_iter is not None else 0,
            ),
        ):
            summ = results.get(
                (
                    job.ego_iter if job.ego_iter is not None else 0,
                    job.peer_iter if job.peer_iter is not None else 0,
                )
            )
            if summ is not None:
                vals = [
                    format_metric_value(nested_value(summ, m_path), v_type, precision)
                    for _, m_path, v_type in METRICS
                ]
                csv_vals = [
                    str(
                        nested_value(summ, m_path)
                        if nested_value(summ, m_path) is not None
                        else ""
                    )
                    for _, m_path, _ in METRICS
                ]
            else:
                vals = [MISSING_VALUE] * len(METRICS)
                csv_vals = [""] * len(METRICS)
            lines.append(
                f"| {job.ego_iter} | {job.peer_iter} | {job.seed} | "
                + " | ".join(vals)
                + " |"
            )
            csv_rows.append(
                f"{model_path.name},{job.ego_iter},{job.peer_iter},{job.seed},"
                + ",".join(csv_vals)
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", "\n".join(csv_rows).rstrip() + "\n"


def save_summary_tables(
    root_path: Path,
    jobs: Sequence[EvaluationJob],
    keywords: Sequence[str] = (),
    precision: int = 4,
    summary_output_override: Path | None = None,
) -> list[Path]:
    """Save comprehensive evaluation summary tables in Markdown, LaTeX, and CSV formats."""

    figure_directory = get_figure_directory(root_path, keywords)
    figure_directory.mkdir(parents=True, exist_ok=True)
    root_path.mkdir(parents=True, exist_ok=True)

    if jobs and jobs[0].is_cross_iter_test:
        cross_report, cross_csv = build_cross_iter_summary_report(
            root_path, jobs, precision=precision
        )
        saved_paths: list[Path] = []
        main_summary_md = summary_output_override or root_path / "evaluation_summary.md"
        main_summary_md.write_text(cross_report, encoding="utf-8")
        saved_paths.append(main_summary_md)
        fig_summary_md = figure_directory / "evaluation_summary.md"
        fig_summary_md.write_text(cross_report, encoding="utf-8")
        saved_paths.append(fig_summary_md)

        csv_file = root_path / "evaluation_cross_iter_summary.csv"
        csv_file.write_text(cross_csv, encoding="utf-8")
        saved_paths.append(csv_file)
        fig_csv_file = figure_directory / "evaluation_cross_iter_summary.csv"
        fig_csv_file.write_text(cross_csv, encoding="utf-8")
        saved_paths.append(fig_csv_file)
        return saved_paths

    grouped: dict[tuple[str, int], list[EvaluationJob]] = defaultdict(list)
    for job in jobs:
        grouped[(job.scenario_type, job.n_agents)].append(job)

    full_report = build_summary_report(root_path, jobs, precision=precision)
    method_markdown = build_method_summary_markdown(grouped)
    latex_tables = build_method_summary_latex(grouped)
    csv_table = build_method_summary_csv(grouped)

    saved_paths: list[Path] = []

    # 1. Full Evaluation Summary Markdown
    main_summary_md = summary_output_override or root_path / "evaluation_summary.md"
    main_summary_md.write_text(full_report, encoding="utf-8")
    saved_paths.append(main_summary_md)

    # 2. Method-Level Summary Markdown
    method_summary_md = root_path / "evaluation_method_summary.md"
    method_summary_md.write_text(method_markdown, encoding="utf-8")
    saved_paths.append(method_summary_md)

    # Also mirror method summary to figure directory
    fig_method_summary_md = figure_directory / "evaluation_method_summary.md"
    fig_method_summary_md.write_text(method_markdown, encoding="utf-8")
    saved_paths.append(fig_method_summary_md)

    # 3. LaTeX Table Exports
    comparison_latex = build_comparison_latex_table(grouped)
    ablation_latex = build_ablation_latex_table(grouped)

    latex_file = root_path / "evaluation_tables.tex"
    latex_file.write_text(latex_tables, encoding="utf-8")
    saved_paths.append(latex_file)

    fig_latex_file = figure_directory / "evaluation_tables.tex"
    fig_latex_file.write_text(latex_tables, encoding="utf-8")
    saved_paths.append(fig_latex_file)

    comp_file = root_path / "evaluation_table_comparison.tex"
    comp_file.write_text(comparison_latex, encoding="utf-8")
    saved_paths.append(comp_file)

    fig_comp_file = figure_directory / "evaluation_table_comparison.tex"
    fig_comp_file.write_text(comparison_latex, encoding="utf-8")
    saved_paths.append(fig_comp_file)

    abla_file = root_path / "evaluation_table_ablation.tex"
    abla_file.write_text(ablation_latex, encoding="utf-8")
    saved_paths.append(abla_file)

    fig_abla_file = figure_directory / "evaluation_table_ablation.tex"
    fig_abla_file.write_text(ablation_latex, encoding="utf-8")
    saved_paths.append(fig_abla_file)

    compact_latex = build_combined_compact_latex_table(grouped)

    compact_file = root_path / "evaluation_table_combined_compact.tex"
    compact_file.write_text(compact_latex, encoding="utf-8")
    saved_paths.append(compact_file)

    fig_compact_file = figure_directory / "evaluation_table_combined_compact.tex"
    fig_compact_file.write_text(compact_latex, encoding="utf-8")
    saved_paths.append(fig_compact_file)

    # 4. CSV Table Export
    csv_file = root_path / "evaluation_method_summary.csv"
    csv_file.write_text(csv_table, encoding="utf-8")
    saved_paths.append(csv_file)

    fig_csv_file = figure_directory / "evaluation_method_summary.csv"
    fig_csv_file.write_text(csv_table, encoding="utf-8")
    saved_paths.append(fig_csv_file)

    return saved_paths


def main() -> None:
    """Discover models, run the configured sweep, and print its summary."""

    args = parse_args()
    root_path = args.path.expanduser().resolve()
    model_paths = find_policy_folders(
        root_path, is_cross_iter_test=args.is_cross_iter_test
    )
    if not model_paths:
        expected_desc = (
            "checkpoints_cross_iter/policy_iter*.pth"
            if args.is_cross_iter_test
            else POLICY_FILE_PATTERN
        )
        raise FileNotFoundError(
            f"No folders containing {expected_desc} found in "
            f"{root_path} or its first two directory levels."
        )
    keywords = args.keywords or ()
    if args.is_plot_only and keywords:
        model_paths = [
            model_path
            for model_path in model_paths
            if any(keyword in model_path.name for keyword in keywords)
        ]

    jobs = build_jobs(
        model_paths,
        is_cross_iter_test=args.is_cross_iter_test,
        cross_iter_ego_indices=args.cross_iter_ego_indices,
        peer_iter=args.peer_iter,
        ego_iter=args.ego_iter,
        selected_cross_iter_checkpoints=args.selected_cross_iter_checkpoints,
    )

    active_scenarios = (
        SCENARIO_TYPES_CROSS_ITER if args.is_cross_iter_test else SCENARIO_TYPES
    )
    active_agent_counts = N_AGENTS_CROSS_ITER if args.is_cross_iter_test else N_AGENTS
    active_seeds = SEEDS_CROSS_ITER if args.is_cross_iter_test else SEEDS

    print(
        f"[INFO] Found {len(model_paths)} trained model folder(s) under "
        f"{display_path(root_path)}."
    )
    print(
        f"[INFO] Configured {len(active_scenarios)} scenario(s), "
        f"{sum(len(active_agent_counts[scenario]) for scenario in active_scenarios)} "
        f"scenario/agent setting(s), and {len(active_seeds)} seed(s)."
    )
    if args.is_cross_iter_test:
        if args.selected_cross_iter_checkpoints is not None:
            n_sel = len(args.selected_cross_iter_checkpoints)
            print(
                f"[INFO] Cross-iteration testing: selected checkpoints {args.selected_cross_iter_checkpoints} "
                f"({n_sel}^2 = {n_sel**2} cross pairs per model)."
            )
        else:
            print(
                "[INFO] Cross-iteration testing: evaluating all available checkpoint pairs."
            )

    if args.is_plot_only:
        if jobs and jobs[0].is_cross_iter_test:
            print(
                "[INFO] Plot-only mode for cross-iteration testing: generating summary figures and tables."
            )
            figure_paths = save_cross_iter_heatmaps(
                root_path,
                jobs,
                keywords,
                usetex=args.usetex,
                usetex_font=args.usetex_font,
            )
            figure_directory = get_figure_directory(root_path, keywords)
            print(
                f"[INFO] Saved {len(figure_paths)} cross-iteration heatmap(s) to "
                f"{display_path(figure_directory)}"
            )
        else:
            print(
                "[INFO] Plot-only mode: generating summary figures and evaluation tables."
            )
            figure_paths = save_summary_barplots(
                root_path,
                jobs,
                keywords,
                usetex=args.usetex,
                usetex_font=args.usetex_font,
            )
            figure_directory = get_figure_directory(root_path, keywords)
            print(
                f"[INFO] Saved {len(figure_paths)} summary barplot(s) to "
                f"{display_path(figure_directory)}"
            )
        table_paths = save_summary_tables(
            root_path,
            jobs,
            keywords,
            precision=args.precision,
            summary_output_override=args.summary_output,
        )
        print(
            f"[INFO] Saved {len(table_paths)} summary table file(s) to "
            f"{display_path(root_path)} and {display_path(get_figure_directory(root_path, keywords))}"
        )
        return

    print(f"[INFO] Scheduling {len(jobs)} total rollout job(s).")

    if args.is_hpc:
        jobs_per_model = len(jobs) // len(model_paths)
        print(
            f"[INFO] HPC mode: submitting {len(model_paths)} job(s), "
            f"with {jobs_per_model} rollout setting(s) per model."
        )
        submit_hpc_jobs(model_paths, args)
        print("[INFO] All HPC model-job submissions completed.")
        return

    failures = run_batch(
        jobs,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        save_video_enabled=args.save_video,
        save_out_td=args.save_out_td,
        is_skip=args.is_skip,
        is_sequential=args.is_hpc_worker,
    )

    table_paths = save_summary_tables(
        root_path,
        jobs,
        keywords,
        precision=args.precision,
        summary_output_override=args.summary_output,
    )
    print(
        f"[INFO] Saved {len(table_paths)} summary table file(s) to "
        f"{display_path(root_path)} and {display_path(get_figure_directory(root_path, keywords))}"
    )

    if jobs and jobs[0].is_cross_iter_test:
        figure_paths = save_cross_iter_heatmaps(
            root_path,
            jobs,
            keywords,
            usetex=args.usetex,
            usetex_font=args.usetex_font,
        )
        figure_directory = get_figure_directory(root_path, keywords)
        print(
            f"[INFO] Saved {len(figure_paths)} cross-iteration heatmap(s) to "
            f"{display_path(figure_directory)}"
        )
    else:
        figure_paths = save_summary_barplots(
            root_path,
            jobs,
            keywords,
            usetex=args.usetex,
            usetex_font=args.usetex_font,
        )
        figure_directory = get_figure_directory(root_path, keywords)
        print(
            f"[INFO] Saved {len(figure_paths)} summary barplot(s) to "
            f"{display_path(figure_directory)}"
        )

    if failures:
        print(f"[WARNING] {len(failures)} rollout job(s) failed.", file=sys.stderr)


if __name__ == "__main__":
    main()
