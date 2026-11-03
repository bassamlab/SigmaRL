"""
python main_training_cbf_learned_ref_point_and_target_speed.py \
    --seed 1 \
    --h-nom 0.1 \
    --reward-progress 0.1 \
    --rew-method cbf \
    --cbf-lambda-rew-scale 0.1 \
    --cbf-lambda-rew-scale-abs 1.0
"""
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
    parser.add_argument("--h-nom", type=_float_or_none, default=0.1)
    parser.add_argument("--reward-progress", type=float, default=0.1)
    parser.add_argument("--cbf-lambda-rew-scale", type=float, default=0.1)
    parser.add_argument("--cbf-lambda-rew-scale-abs", type=float, default=1.0)
    parser.add_argument("--penalty-deviate-from-cbf-vel", type=float, default=-0.05)
    parser.add_argument("--penalty-deviate-from-cbf-steer", type=float, default=-0.05)
    parser.add_argument(
        "--rl-action-type", type=str, default="speed_steer"
    )  # "position_speed" or "speed_steer"
    parser.add_argument(
        "--rew-method",
        type=str,
        default="cbfmargin",  # Combination of keywords "cbf", "ttc", "distance", "sparse", "cbfcorrection", "cbfmargin", "cbflambda" to indicate the reward method. For example, "cbfcorrection_sparse" means using the CBF correction reward and sparse reward; "cbfmargin_cbflambda" means using the CBF margin reward and CBF lambda reward.
    )

    # If you sometimes pass extra flags and do not want errors:
    args, _unknown = parser.parse_known_args()
    return args


args = parse_args()

random_seed = args.seed
h_nom = args.h_nom
reward_progress = args.reward_progress
rl_action_type = args.rl_action_type
rew_method = args.rew_method
cbf_lambda_rew_scale = args.cbf_lambda_rew_scale
cbf_lambda_rew_scale_abs = args.cbf_lambda_rew_scale_abs
penalty_deviate_from_cbf_vel = args.penalty_deviate_from_cbf_vel
penalty_deviate_from_cbf_steer = args.penalty_deviate_from_cbf_steer

print(f"[INFO] Using random seed = {random_seed}")
print(f"[INFO] Using h_nom = {h_nom}")
print(f"[INFO] Using reward_progress = {reward_progress}")
print(f"[INFO] Using rew_method = {rew_method}")
print(f"[INFO] Using cbf_lambda_rew_scale = {cbf_lambda_rew_scale}")

# ===============================
# Training configuration
# ===============================
scenario_type = "cpm_entire"  # One of "cpm_mixed", "cpm_entire", "intersection_1", etc.
config_file = "sigmarl/config.json"  # Adjust parameters therein

parameters = Parameters.from_json(config_file)

# Set parameters
parameters.scenario_type = scenario_type
parameters.n_agents = 10

parameters.n_iters = 1000
parameters.random_seed = random_seed
parameters.is_using_centralized_cbf = False
parameters.is_using_cbf_training = True
parameters.is_using_cbf_testing = False
parameters.is_grouping_agents = False
parameters.max_group_size = 1
parameters.is_using_prioritized_marl = False
parameters.is_continue_train = True
parameters.is_load_model = False
parameters.is_solve_qp = False
parameters.is_apply_cbf_action = False
parameters.h_nom = h_nom
parameters.is_testing_mode = False
parameters.lane_width = 0.25  # lane width in meter (except cpm_mixed and cpm_entire)
parameters.reward_progress = reward_progress
parameters.rl_action_type = rl_action_type
parameters.rew_method = rew_method
parameters.is_learn_cbf_lambda = False
parameters.cbf_lambda_rew_scale = cbf_lambda_rew_scale
parameters.cbf_lambda_rew_scale_abs = cbf_lambda_rew_scale_abs
parameters.adaptive_lambda = False
parameters.nom_controller_type = "clf"
parameters.is_alternating_action_lambda_training = False
parameters.alternating_train_action_iters = 100
parameters.alternating_train_lambda_iters = 100
parameters.pretrained_motion_policy_path = ""
parameters.pretrained_lambda_policy_path = ""
parameters.is_load_pretrained_motion_policy = (
    True if parameters.pretrained_motion_policy_path else False
)
parameters.is_load_pretrained_lambda_policy = (
    True if parameters.pretrained_lambda_policy_path else False
)
parameters.is_train_motion_policy = True
parameters.is_train_lambda_policy = False
parameters.penalty_deviate_from_cbf_vel = penalty_deviate_from_cbf_vel
parameters.penalty_deviate_from_cbf_steer = penalty_deviate_from_cbf_steer


date_stamp = datetime.now().strftime("%Y%m%d")
parameters.where_to_save = f"checkpoints/{date_stamp}-jx-path-mode-lateral-offset/{parameters.nom_controller_type}/{parameters.rl_action_type}/{parameters.rew_method}/seed{random_seed}"

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
