import os
import sys
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.helper_training import Parameters
from sigmarl.constants import SCENARIOS, AGENTS
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
    parser.add_argument("--h-nom", type=_float_or_none, default=None)
    parser.add_argument("--reward-progress", type=float, default=10.0)
    parser.add_argument(
        "--rew-method",
        type=str,
        choices=["cbf", "ttc", "default", "sparse"],
        default=None,
    )

    # If you sometimes pass extra flags and do not want errors:
    args, _unknown = parser.parse_known_args()
    return args


args = parse_args()

random_seed = args.seed
h_nom = args.h_nom
reward_progress = args.reward_progress
rew_method = args.rew_method

print(f"[INFO] Using random seed = {random_seed}")
print(f"[INFO] Using h_nom = {h_nom}")
print(f"[INFO] Using reward_progress = {reward_progress}")
print(f"[INFO] Using rew_method = {rew_method}")

# ===============================
# Training configuration
# ===============================
scenario_type = "cpm_mixed"  # One of "cpm_mixed", "cpm_entire", "intersection_1", etc.
config_file = "sigmarl/config.json"  # Adjust parameters therein

parameters = Parameters.from_json(config_file)

# Set parameters
parameters.scenario_type = scenario_type
# parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
parameters.n_agents = 4

# parameters.where_to_save = "outputs/marl_cbf_fixed_group_2_episode_100_rl/"
parameters.n_iters = 1000
parameters.random_seed = random_seed
parameters.is_using_centralized_cbf = True
parameters.is_using_cbf_testing = False
parameters.is_using_prioritized_marl = False
parameters.is_continue_train = True
parameters.is_load_model = False
parameters.is_apply_cbf_action = False
parameters.is_solve_qp = False
parameters.h_nom = h_nom
parameters.is_testing_mode = False
parameters.lane_width = 0.25  # lane width in meter (except cpm_mixed and cpm_entire)
parameters.reward_progress = reward_progress

if parameters.h_nom is None:
    parameters.is_using_cbf_training = False
    if rew_method is not None:
        parameters.rew_method = rew_method
    else:
        print("[INFO] No reward method specified. Using default reward method.")
        parameters.rew_method = (
            "default"  # Reward method: {"default", "cbf", "ttc", "sparse"}
        )
else:
    parameters.is_using_cbf_training = True
    parameters.rew_method = "cbf"


print(f"[INFO] Using reward method = {parameters.rew_method}")
parameters.where_to_save = f"checkpoints/itsc26/{scenario_type}/rew_method_{parameters.rew_method}/reward_progress{reward_progress}/seed{random_seed}/"

# ===============================
# Save parameters and AGENTS
# ===============================
os.makedirs(parameters.where_to_save, exist_ok=True)

info_path = os.path.join(parameters.where_to_save, "info.txt")

with open(info_path, "w") as f:
    f.write("===== PARAMETERS =====\n")

    # Parameters is a class instance. Use __dict__ for readable key-value export.
    for key, value in vars(parameters).items():
        f.write(f"{key}: {value}\n")

    f.write("\n===== AGENTS =====\n")

    # AGENTS is typically a dict or list defined in sigmarl.constants
    if isinstance(AGENTS, dict):
        for key, value in AGENTS.items():
            f.write(f"{key}: {value}\n")
    else:
        f.write(str(AGENTS))

print(f"[INFO] Saved parameters and AGENTS to: {info_path}")

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
