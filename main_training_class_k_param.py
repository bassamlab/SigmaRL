import os
import sys
from datetime import datetime
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.helper_common import start_run_log
from sigmarl.helper_training import Parameters
from sigmarl.constants import SCENARIOS
import argparse

# ===============================
# Check if running in HPC environment
# ===============================
hpc_environment = os.getenv("HPC_ENVIRONMENT", "false").lower() == "true"
if hpc_environment:
    print(
        "HPC environment detected. Setting the environment variable PYGLET_HEADLESS to True."
    )
    os.environ["PYGLET_HEADLESS"] = "True"  # Enable if run in HPC


def _float_or_none(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if s.lower() in {"none", "null", ""}:
        return None
    return float(s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--h_nom", type=_float_or_none, default=0.1)
    parser.add_argument("--reward_progress", type=float, default=0.1)
    parser.add_argument(
        "--rew_method",
        type=str,
        default="distance_sparse",  # Combination of keywords "cbf", "ttc", "distance", "sparse", "cbfcorrection", "cbfmargin", "cbflambda" to indicate the reward method. For example, "cbfcorrection_sparse" means using the CBF correction reward and sparse reward; "cbfmargin_cbflambda" means using the CBF margin reward and CBF lambda reward.
    )

    # If you sometimes pass extra flags and do not want errors:
    args, _unknown = parser.parse_known_args()
    return args


args = parse_args()

print(f"[INFO] Using random seed = {args.seed}")
print(f"[INFO] Using h_nom = {args.h_nom}")
print(f"[INFO] Using reward_progress = {args.reward_progress}")
print(f"[INFO] Using rew_method = {args.rew_method}")

# ===============================
# Training configuration
# ===============================
scenario_type = "cpm_mixed"  # One of "cpm_mixed", "cpm_entire", "intersection_1", etc.
config_file = "sigmarl/config.json"  # Adjust parameters therein

parameters = Parameters.from_json(config_file)

# Set parameters
parameters.scenario_type = scenario_type
parameters.n_agents = 5

parameters.n_iters = 500
parameters.random_seed = args.seed
parameters.is_using_centralized_cbf = False
parameters.is_using_cbf_training = True
parameters.is_using_cbf_testing = False
parameters.is_grouping_agents = False
parameters.is_continue_train = True
parameters.is_load_model = False
parameters.is_solve_qp = True
parameters.is_apply_cbf_action = True
parameters.h_nom = args.h_nom
parameters.is_testing_mode = False
parameters.reward_progress = args.reward_progress
parameters.rew_method = args.rew_method
parameters.adaptive_lambda = False
parameters.nom_controller_type = "clf"
parameters.is_learn_cbf_lambda = True
# The backup-horizon rows support only a fixed lambda
parameters.is_backup_horizon_cbf = False
parameters.is_train_motion_policy = False
parameters.is_train_lambda_policy = True
parameters.is_alternating_action_lambda_training = False
parameters.pretrained_motion_policy_path = ""
parameters.pretrained_lambda_policy_path = ""
parameters.is_load_pretrained_motion_policy = (
    True if parameters.pretrained_motion_policy_path else False
)
parameters.is_load_pretrained_lambda_policy = (
    True if parameters.pretrained_lambda_policy_path else False
)
parameters.is_observe_distance_to_boundaries = True
parameters.is_observe_distance_to_center_line = False
parameters.is_observe_boundary_points = False
parameters.is_observe_ref_path_other_agents = False
parameters.n_nearing_agents_observed = 2
parameters.n_points_nearing_boundary = 8
parameters.n_points_short_term = 3
parameters.is_smooth_sampling = True
parameters.is_use_simple_road_distance = False
parameters.is_exact_backup_shield = True
parameters.nom_controller_type = "clf"

parameters.dt = 0.1
parameters.max_steps = 128

print(f"[INFO] Using reward method = {parameters.rew_method}")

date_stamp = datetime.now().strftime("%Y%m%d")
parameters.where_to_save = f"checkpoints/{date_stamp}-class-k-param/{parameters.rew_method}/reward_progress{parameters.reward_progress}/seed{parameters.random_seed}"

os.makedirs(parameters.where_to_save, exist_ok=True)

print(f"[INFO] All files will be saved in {parameters.where_to_save}")
start_run_log(parameters.where_to_save, "training")


# ===============================
# Run training
# ===============================
(
    env,
    decision_making_module,
    optimization_module,
    priority_module,
    cbf_controllers,
    parameters,
) = mappo_cavs(parameters=parameters)
