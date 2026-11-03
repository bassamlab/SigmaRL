"""Launch target-architecture MAPPO from legacy flags or a benchmark YAML."""

import argparse
import contextlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

import yaml

from sigmarl.constants import SCENARIOS
from sigmarl.env_factory import canonical_scenario_name
from sigmarl.helper_common import Parameters, Tee, start_run_log


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = REPO_ROOT / "sigmarl" / "config.json"


def _float_or_none(value: str):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"none", "null", ""}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--h_nom", type=_float_or_none, default=None)
    parser.add_argument("--reward_progress", type=float, default=0.1)
    parser.add_argument("--cbf_lambda_rew_scale", type=float, default=0.1)
    parser.add_argument("--cbf_lambda_rew_scale_abs", type=float, default=1.0)
    parser.add_argument("--penalty_deviate_from_cbf_vel", type=float, default=-0.05)
    parser.add_argument("--penalty_deviate_from_cbf_steer", type=float, default=-0.05)
    parser.add_argument("--rew_method", type=str, default="distance_sparse")
    args, _unknown = parser.parse_known_args()
    return args


def _load_mapping(path: Path, *, yaml_file: bool) -> dict:
    path = path.expanduser()
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) if yaml_file else json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration file does not exist: {path}") from error
    except (yaml.YAMLError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not parse configuration file {path}: {error}"
        ) from error
    if not isinstance(data, dict):
        kind = "YAML" if yaml_file else "JSON"
        raise ValueError(f"{kind} configuration must contain a mapping: {path}")
    return data


def _normalize_yaml_config(config: dict) -> dict:
    normalized = dict(config)
    aliases = {
        "seed": "random_seed",
        "is_add_noise": "is_obs_noise",
    }
    for old_name, new_name in aliases.items():
        if old_name in normalized:
            if new_name in normalized and normalized[new_name] != normalized[old_name]:
                raise ValueError(
                    f"Conflicting YAML values for `{old_name}` and `{new_name}`."
                )
            normalized[new_name] = normalized.pop(old_name)

    algorithm = str(normalized.pop("algorithm", "mappo")).lower()
    if algorithm != "mappo":
        raise ValueError("Only MAPPO benchmark configurations are supported.")

    normalized.pop("total_frames", None)
    return normalized


def load_parameters(yaml_config: Path) -> Parameters:
    """Load and validate a benchmark YAML against target Parameters."""

    overrides = _normalize_yaml_config(_load_mapping(Path(yaml_config), yaml_file=True))
    scenario_type = canonical_scenario_name(overrides.get("scenario_type", "cpm_mixed"))
    overrides["scenario_type"] = scenario_type

    config = {
        **_load_mapping(DEFAULT_JSON, yaml_file=False),
        **overrides,
        "n_agents": overrides.get("n_agents", SCENARIOS[scenario_type]["n_agents"]),
    }
    if "frames_per_batch" in config:
        frames_per_batch = config.pop("frames_per_batch")
        # Legacy benchmark YAMLs derived this field against the benchmark
        # horizon (512) when they did not spell out max_steps themselves.
        max_steps = overrides.get("max_steps", 512)
        config["max_steps"] = max_steps
        if (
            isinstance(frames_per_batch, bool)
            or not isinstance(frames_per_batch, int)
            or frames_per_batch <= 0
        ):
            raise ValueError("`frames_per_batch` must be a positive integer.")
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
            or frames_per_batch % max_steps
        ):
            raise ValueError(
                "`frames_per_batch` must be divisible by positive `max_steps`."
            )
        config["num_vmas_envs"] = frames_per_batch // max_steps
    valid_keys = set(inspect.signature(Parameters.__init__).parameters) - {"self"}
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        raise ValueError(
            "Unknown Parameters configuration field(s): " + ", ".join(unknown_keys)
        )
    parameters = Parameters(**config)
    parameters.algorithm = "mappo"
    return parameters


def build_legacy_parameters(args: argparse.Namespace) -> Parameters:
    """Preserve the existing higher-speed non-YAML training behavior."""

    parameters = Parameters.from_json(DEFAULT_JSON)
    parameters.scenario_type = "cpm_mixed"
    parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
    parameters.n_iters = 1000
    parameters.random_seed = args.seed
    parameters.is_using_centralized_cbf = False
    parameters.is_using_cbf_training = False
    parameters.is_using_cbf_testing = False
    parameters.is_grouping_agents = False
    parameters.max_group_size = 1
    parameters.is_using_prioritized_marl = False
    parameters.is_continue_train = True
    parameters.is_load_model = False
    parameters.is_apply_cbf_action = False
    parameters.is_solve_qp = False
    parameters.h_nom = args.h_nom
    parameters.is_testing_mode = False
    parameters.lane_width = 0.25
    parameters.reward_progress = args.reward_progress
    parameters.rew_method = args.rew_method
    parameters.is_learn_cbf_lambda = False
    parameters.cbf_lambda_rew_scale = args.cbf_lambda_rew_scale
    parameters.cbf_lambda_rew_scale_abs = args.cbf_lambda_rew_scale_abs
    parameters.adaptive_lambda = False
    parameters.nom_controller_type = "rl"
    parameters.is_alternating_action_lambda_training = False
    parameters.alternating_train_action_iters = 100
    parameters.alternating_train_lambda_iters = 100
    parameters.pretrained_motion_policy_path = ""
    parameters.pretrained_lambda_policy_path = ""
    parameters.is_load_pretrained_motion_policy = False
    parameters.is_load_pretrained_lambda_policy = False
    parameters.is_train_motion_policy = True
    parameters.is_train_lambda_policy = False
    parameters.penalty_deviate_from_cbf_vel = args.penalty_deviate_from_cbf_vel
    parameters.penalty_deviate_from_cbf_steer = args.penalty_deviate_from_cbf_steer
    parameters.where_to_save = "outputs/test/"
    return parameters


def run_training(parameters: Parameters):
    from sigmarl.mappo_cavs import mappo_cavs

    return mappo_cavs(parameters=parameters)


def main() -> None:
    if os.getenv("HPC_ENVIRONMENT", "false").lower() == "true":
        os.environ["PYGLET_HEADLESS"] = "True"
    args = parse_args()
    if args.yaml_config is not None:
        try:
            parameters = load_parameters(args.yaml_config)
        except Exception as error:
            print(f"[INFO] Failed to read YAML config {args.yaml_config}: {error}")
            raise
        print(f"[INFO] Successfully read YAML config: {args.yaml_config}")
    else:
        parameters = build_legacy_parameters(args)
    save_directory = Path(parameters.where_to_save)
    save_directory.mkdir(parents=True, exist_ok=True)
    start_run_log(parameters.where_to_save, "training")
    log_path = save_directory / "training_logs.txt"

    with log_path.open("a", encoding="utf-8") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                print(
                    f"[INFO] Scenario: {parameters.scenario_type}; "
                    f"agents: {parameters.n_agents}; seed: {parameters.random_seed}"
                )
                print(f"[INFO] Saving outputs to: {parameters.where_to_save}")
                run_training(parameters)
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
