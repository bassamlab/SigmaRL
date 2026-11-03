#!/usr/bin/env python3
"""Generate the YAML configuration files.

The method names and parameters are defined in this script.  The generated
files keep the same stable key order and values as the current benchmark
matrix, so every output YAML specifies the same number of parameters.

Example:

    python scripts/generate_configs.py


"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import yaml

from sigmarl.constants import ACTIVE_AGENT_PROFILE


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = "checkpoints/{date_stamp}"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "configs" / "{date_stamp}"
SCENARIO_TYPES = [
    # "cpm_mixed",
    "round_location0",
    "ind_location4",
    # "cpm_entire",
    # "ball_passage",
    # "discovery",
    # "dropout",
    # "sampling",
    # "transport",
]
SEEDS = (1, 2, 3, 4, 5)
N_AGENTS_BY_SCENARIO = {
    "cpm_mixed": 6,
    "round_location0": 16,  # T-ITS 2026 (sigmarl/evaluation_tits26/settings.py)
    "ind_location4": 12,
    "cpm_entire": 12,
    "ball_passage": 2,
    "discovery": 6,
    "dropout": 6,
    "sampling": 5,
    "transport": 4,
}
MAX_STEPS_BY_SCENARIO = {
    "ball_passage": 1024,
    "discovery": 512,
    "dropout": 256,
    "sampling": 512,
    "transport": 1600,
}
REWARD_GROUP_THRESHOLDS = {
    "cpm_mixed": 0.602,
    "cpm_entire": 1.204,
    "ball_passage": 0.566,
    "discovery": 0.566,
    "dropout": 0.566,
    "sampling": 0.552,
    "transport": 0.684,
}

# Keep this insertion order: it is the key order used by the current output
# YAML files.  Scenario, seed, and where_to_save are overwritten per run below.
COMMON_PARAMETERS: dict[str, Any] = {
    "scenario_type": "cpm_mixed",
    "n_agents": 6,
    "algorithm": "mappo",
    "n_iters": 500,
    "lr": 1.0e-04,
    "lr_min": 1.0e-05,
    "is_continue_train": True,
    "dt": 0.05,
    "max_steps": 256,
    "num_vmas_envs": 32,
    "num_epochs": 5,
    "minibatch_size": 512,
    "is_multi_timescale_learning": False,
    "lr_slow": 2.0e-05,
    "lr_slow_min": 2.0e-06,
    "multi_timescale_switching_period": 50,
    "is_share_params_actor": True,
    "where_to_save": "",
    "is_using_opponent_modeling": False,
    "is_using_prioritized_marl": False,
    "is_priority_conditioned_motion": False,
    "is_even_reward_distribution": False,
    "execution_mode": "reward_propagation",
    "reward_distribution_alpha_start": 0.2,
    "reward_distribution_alpha_end": 0.2,
    "reward_distribution_mode": "linear",
    "is_add_noise": False,
    "seed": 1,
    "prioritization_method": "random",
    "random_priority_fixed_episode": None,
    "entropy_eps": 5.0e-03,
    "entropy_eps_priority": 0.01,
    "clip_epsilon": 0.15,
    "max_grad_norm": 0.5,
    "lr_priority": 2.0e-05,
    "lr_priority_min": 2.0e-06,
    "prioritized_reward_group_threshold": 0.0,
    "scale_map": 1.0,
    "velocity_mask_margin": 0.5,
    "is_use_mtv_distance": False,
    "penalty_near_collision_reset": -0.5,
    "n_nearing_agents_observed": 3,
    "is_observe_boundary_points": False,
    "n_points_nearing_boundary": 8,
    "is_use_simple_road_distance": True,
    "is_opponent_modeling_separate_predictor": False,
    "reward_progress": 0.10,
    "rew_method": "cbfmargin",  # "cbfmargin", "ttc_sparse", "distance_sparse"
    "cbf_margin_reward_condition": "backup_horizon",  # "one_step" or "backup_horizon" (BH-TTCBF), for rew_method "cbfmargin"
    "agent_profile": ACTIVE_AGENT_PROFILE,
    "penalty_near_collision_reset": 0.0,
    "reset_agent_surrounding_distance": 0,
    # Policy speed scale shared by all maps: the highest speed limit among them
    # (rounD and inD: 50 km/h)
    "speed_normalization": 50.0 / 3.6,
    "rl_obs_range": None,  # None: resolved from the speed scale (Parameters.resolve_speed_defaults)
    "cbf_coupling_range": None,  # None: resolved from the speed limit (Parameters.resolve_speed_defaults)
    "is_center_centerline": False,
    "must_save_checkpoints": [0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500],
    # Fixed CBF decay rate of the CBF margin rewards (unused by other rewards)
    "cbf_lambda": 0.2,
}

METHOD_OVERRIDES: dict[str, dict[str, Any]] = {
    # "vanilla": {},
    # T-ITS 2026, Stage 1 pre-training: the same task reward (progress and the
    # sparse terms) with one of three safety rewards
    "stage1_ttc": {
        "dt": 0.1,
        "n_iters": 500,
        "max_steps": 128,
        "rew_method": "ttc_sparse",
        "cbf_margin_reward_condition": "one_step",  # Unused by this reward
    },
    "stage1_onestep": {
        "dt": 0.1,
        "n_iters": 500,
        "max_steps": 128,
        "rew_method": "cbfmargin_sparse",
        "cbf_margin_reward_condition": "one_step",
    },
    "stage1_bh": {
        "dt": 0.1,
        "n_iters": 500,
        "max_steps": 128,
        "rew_method": "cbfmargin_sparse",
        "cbf_margin_reward_condition": "backup_horizon",
    },
    # 2026-10-14: one-step vs. BH margin reward (rew_method "cbfmargin" has no
    # sparse terms; the condition is now set explicitly per method)
    # "fullscale_dt0.1_iter250_step128_cbfmargin": {
    #     "dt": 0.1,
    #     "n_iters": 250,
    #     "max_steps": 128,
    #     "rew_method": "cbfmargin",
    #     "cbf_margin_reward_condition": "one_step",
    # },
    # "fullscale_dt0.1_iter250_step128_cbfmarginbh": {
    #     "dt": 0.1,
    #     "n_iters": 250,
    #     "max_steps": 128,
    #     "rew_method": "cbfmargin",
    #     "cbf_margin_reward_condition": "backup_horizon",
    # },
    # "vanilla_dt0.1_iter250_step128_ttc_sparse": {
    #     "dt": 0.1,
    #     "n_iters": 250,
    #     "max_steps": 128,
    #     "rew_method": "ttc_sparse",
    # },
    # "mtppo_motion": {
    #     "is_multi_timescale_learning": True,
    #     "is_using_prioritized_marl": False,
    #     "is_priority_conditioned_motion": False,
    #     "is_share_params_actor": False,
    #     "lr_slow": 1.0e-05,
    #     "lr_slow_min": 1.0e-06,
    #     "multi_timescale_switching_period": 100,
    #     "n_iters": 2000,
    # },
    # "mtppo_motion_v2": {
    #     "is_multi_timescale_learning": True,
    #     "is_using_prioritized_marl": False,
    #     "is_priority_conditioned_motion": False,
    #     "is_share_params_actor": False,
    #     "lr_slow": 2.0e-05,
    #     "lr_slow_min": 2.0e-06,
    #     "multi_timescale_switching_period": 50,
    # },
    # "mtppo_motion_v3": {
    #     "is_multi_timescale_learning": True,
    #     "is_using_prioritized_marl": False,
    #     "is_priority_conditioned_motion": False,
    #     "is_share_params_actor": False,
    #     "lr_slow": 2.0e-05,
    #     "lr_slow_min": 2.0e-06,
    #     "multi_timescale_switching_period": 100,
    # },
    # "mtppo_motion_v4": {
    #     "n_iters": 1000,
    #     "is_multi_timescale_learning": True,
    #     "is_using_prioritized_marl": False,
    #     "is_priority_conditioned_motion": False,
    #     "is_share_params_actor": False,
    #     "lr_slow": 1.0e-05,
    #     "lr_slow_min": 1.0e-06,
    #     "multi_timescale_switching_period": 100,
    # },
    # "mtppo_motion_v5": {
    #     "n_iters": 1000,
    #     "is_multi_timescale_learning": True,
    #     "is_using_prioritized_marl": False,
    #     "is_priority_conditioned_motion": False,
    #     "is_share_params_actor": False,
    #     "lr_slow": 1.0e-05,
    #     "lr_slow_min": 1.0e-06,
    #     "multi_timescale_switching_period": 200,
    # },
    # "mtppo_motion_v6": {
    #     "n_iters": 1000,
    #     "is_multi_timescale_learning": True,
    #     "is_using_prioritized_marl": False,
    #     "is_priority_conditioned_motion": False,
    #     "is_share_params_actor": False,
    #     "lr_slow": 2.0e-05,
    #     "lr_slow_min": 2.0e-06,
    #     "multi_timescale_switching_period": 100,
    # },
    # "our_motion_only": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "reward_distribution_alpha_start": 0.0,
    #     "reward_distribution_alpha_end": 0.0,
    #     "prioritization_method": "marl",
    # },
    # "our_rr_motion": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "marl",
    # },
    # "our_rr_motion_random": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "random",
    # },
    # "our_rr_motion_random_episode1": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "random",
    #     "random_priority_fixed_episode": 1,
    # },
    # "our_rr_motion_random_episode10": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "random",
    #     "random_priority_fixed_episode": 10,
    # },
    # "our_rr_motion_random_episode50": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "random",
    #     "random_priority_fixed_episode": 50,
    # },
    # "our_rr_motion_random_episode100": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "random",
    #     "random_priority_fixed_episode": 100,
    # },
    # "our_rr_motion_fixed": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "fixed",
    # },
    # "our_rr_motion_taskprogress": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "taskprogress",
    # },
    # "our_rr_motion_taskprogress_group": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "taskprogress",
    # },
    # "our_rr_motion_taskprogress_permute": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "taskprogress_permute",
    # },
    # "our_rr_motion_group": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "prioritization_method": "marl",
    # },
    # "our_rr_only": {
    #     "is_using_prioritized_marl": True,
    #     "prioritization_method": "marl",
    # },
    # "our_rr_only_group": {
    #     "is_using_prioritized_marl": True,
    #     "prioritization_method": "marl",
    # },
    # "om_dedicated": {
    #     "is_using_opponent_modeling": True,
    #     "is_opponent_modeling_separate_predictor": True,
    # },
    # "om_shared": {
    #     "is_using_opponent_modeling": True,
    # },
    # "our_motion_only_taskprogress": {
    #     "is_using_prioritized_marl": True,
    #     "is_priority_conditioned_motion": True,
    #     "reward_distribution_alpha_start": 0.0,
    #     "reward_distribution_alpha_end": 0.0,
    #     "prioritization_method": "taskprogress",
    # },
    # "our_rr_only_taskprogress": {
    #     "is_using_prioritized_marl": True,
    #     "prioritization_method": "taskprogress",
    # },
}

if (
    "agent_profile" in COMMON_PARAMETERS
    and COMMON_PARAMETERS["agent_profile"] != ACTIVE_AGENT_PROFILE
):
    raise ValueError(
        "COMMON_PARAMETERS['agent_profile'] is inconsistent with "
        f"ACTIVE_AGENT_PROFILE: {COMMON_PARAMETERS['agent_profile']!r} != "
        f"{ACTIVE_AGENT_PROFILE!r}."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory receiving generated YAML files.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Root used for each YAML where_to_save value.",
    )
    parser.add_argument(
        "--date-stamp",
        default=None,
        help="Eight-digit date used in output paths (default: today's date).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated paths without writing files.",
    )
    return parser.parse_args()


def _date_stamp(value: str | None) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError("`--date-stamp` must be an eight-digit YYYYMMDD value.")
    return value


def _build_configs(
    checkpoint_root: Path,
) -> list[tuple[str, str, int, dict[str, Any]]]:
    configs = []
    for method, overrides in METHOD_OVERRIDES.items():
        for scenario_type in SCENARIO_TYPES:
            for seed in SEEDS:
                config = {**COMMON_PARAMETERS, **overrides}
                # CBF-based rewards need the CBF-QP during training.
                is_cbf_reward = "cbf" in config["rew_method"]
                config["is_using_cbf_training"] = is_cbf_reward
                config["is_using_centralized_cbf"] = is_cbf_reward
                # The margin reward only needs the CBF values, not the QP solve.
                config["is_solve_qp"] = "cbfmargin" not in config["rew_method"]
                config["scenario_type"] = scenario_type
                config["n_agents"] = N_AGENTS_BY_SCENARIO[scenario_type]
                config["max_steps"] = MAX_STEPS_BY_SCENARIO.get(
                    scenario_type, config["max_steps"]
                )
                if "_group" in method:
                    config[
                        "prioritized_reward_group_threshold"
                    ] = REWARD_GROUP_THRESHOLDS[scenario_type]
                config["seed"] = seed
                config["where_to_save"] = str(
                    checkpoint_root / scenario_type / f"{method}_seed_{seed}"
                )
                configs.append((method, scenario_type, seed, config))
    return configs


def _output_path(output_root: Path, method: str, scenario_type: str, seed: int) -> Path:
    return output_root / scenario_type / f"{method}_seed_{seed}.yaml"


def _display_path(path: Path) -> str:
    """Return repository paths relative to the project root for console output."""

    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = _parse_args()
    date_stamp = _date_stamp(args.date_stamp)
    output_root = args.output_dir or Path(
        str(DEFAULT_OUTPUT_ROOT).format(date_stamp=date_stamp)
    )
    checkpoint_root = args.checkpoint_root or Path(
        DEFAULT_CHECKPOINT_ROOT.format(date_stamp=date_stamp)
    )

    configs = _build_configs(checkpoint_root)

    output_entries = []
    for method, scenario_type, seed, config in configs:
        output_entries.append(
            (_output_path(output_root, method, scenario_type, seed), config)
        )

    key_counts = {len(config) for _path, config in output_entries}
    if len(key_counts) != 1:
        raise ValueError(
            f"Generated YAML files do not have one key count: {key_counts}"
        )

    for output_path, config in output_entries:
        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, sort_keys=False, default_flow_style=None)
        print(_display_path(output_path))

    action = "Would generate" if args.dry_run else "Generated"
    print(
        f"{action} {len(output_entries)} YAML files with {next(iter(key_counts))} "
        f"parameters each."
    )


if __name__ == "__main__":
    main()
