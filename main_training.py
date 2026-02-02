import os
import sys
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.helper_training import Parameters
from sigmarl.constants import SCENARIOS, AGENTS

# ===============================
# Check if running in HPC environment
# ===============================
hpc_environment = os.getenv("HPC_ENVIRONMENT", "false").lower() == "true"
if hpc_environment:
    print(
        "HPC environment detected. Setting the environment variable PYGLET_HEADLESS to True."
    )
    os.environ["PYGLET_HEADLESS"] = "True"  # Enable if run in HPC

# ===============================
# Handle command-line arguments
# ===============================
# Usage:
#   python main_training.py [seed] [h_nom]
# Defaults:
#   seed = 1
#   h_nom = None
random_seed = 1
h_nom = None

if len(sys.argv) > 1:
    print(f"sys.argv: {sys.argv}")

    # Parse seed
    try:
        random_seed = int(sys.argv[1])
    except ValueError:
        print(f"[WARNING] Invalid seed '{sys.argv[1]}', falling back to default = 1")
        random_seed = 1

    # Parse h_nom (optional)
    if len(sys.argv) > 2:
        try:
            h_nom = float(sys.argv[2])
        except ValueError:
            if sys.argv[2].lower() in {"none", "null"}:
                h_nom = None
            else:
                print(
                    f"[WARNING] Invalid h_nom '{sys.argv[2]}', falling back to default = None"
                )
                h_nom = None
else:
    print(f"sys.argv: {sys.argv}")

print(f"[INFO] Using random seed = {random_seed}")
print(f"[INFO] Using h_nom = {h_nom}")


# ===============================
# Training configuration
# ===============================
scenario_type = "cpm_mixed"  # One of "cpm_mixed", "cpm_entire", "intersection_1", etc.
config_file = "sigmarl/config.json"  # Adjust parameters therein

parameters = Parameters.from_json(config_file)

# Set parameters
parameters.scenario_type = scenario_type
parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]

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

if parameters.h_nom is None:
    parameters.is_using_cbf_training = False
else:
    parameters.is_using_cbf_training = True

parameters.where_to_save = f"outputs/cbf_informed_marl/not-agil/{scenario_type}/seed{random_seed}/h{parameters.h_nom}/"

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
