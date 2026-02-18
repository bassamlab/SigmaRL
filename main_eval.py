import time
from typing import Optional
import torch
from sigmarl.helper_common import get_name_suffix, save_video, trim_td
from sigmarl.helper_training import Parameters, SaveData
import os
import sys
import json
import argparse

from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.constants import SCENARIOS


# ===============================
# Argument parsing
# ===============================
parser = argparse.ArgumentParser(description="Run MARL-CBF evaluation")

parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs/marl_cbf_test/",
)
parser.add_argument("--random_seed", type=int, default=1)
# parser.add_argument("--is_grouping_agents", action="store_true", default=None)  # Default None
parser.add_argument(
    "--is_grouping_agents",
    action="store_true",
    default=False,
)
parser.add_argument("--max_group_size", type=int, default=2)
parser.add_argument("--nom_controller_type", type=str, default="rl")
parser.add_argument("--scenario_type", type=str, default="interchange_3")
parser.add_argument("--n_agents", type=int, default=8)

parser.add_argument(
    "--is_using_cbf_testing",
    action="store_true",
    default=False,
)

args = parser.parse_args()

print("[INFO] Parsed arguments:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")

path = args.output_dir
os.makedirs(path, exist_ok=True)


# ===============================
# Load parameters from JSON
# ===============================

# Find json file
path_to_json_file: Optional[str] = None
for file in os.listdir(path):
    if file.endswith(".json") and file.startswith("reward"):
        path_to_json_file = os.path.join(path, file)
        break

# Load if exists, otherwise initialize Parameters
if path_to_json_file is not None and os.path.isfile(path_to_json_file):
    with open(path_to_json_file, "r") as file:
        data = json.load(file)
    saved_data = SaveData.from_dict(data)
    parameters: Parameters = saved_data.parameters
else:
    print(
        "[WARNING] No JSON file found in output folder. Initializing default Parameters."
    )
    parameters = Parameters()

# ===============================
# Adjust parameters
# ===============================
parameters.is_testing_mode = True
parameters.is_real_time_rendering = False
parameters.is_save_eval_results = False
parameters.is_load_final_model = False
parameters.is_load_out_td = False
parameters.is_continue_train = False

parameters.max_steps = 300
parameters.num_vmas_envs = 1

parameters.scenario_type = args.scenario_type
if args.n_agents is not None:
    parameters.n_agents = args.n_agents
else:
    parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]

parameters.dt = 0.1

parameters.is_visualize_short_term_path = True
parameters.is_visualize_lane_boundary = False
parameters.is_visualize_extra_info = True

parameters.is_using_cbf_testing = args.is_using_cbf_testing

parameters.is_using_cbf_training = False
parameters.is_using_prioritized_marl = False
parameters.is_using_centralized_cbf = True

parameters.is_grouping_agents = args.is_grouping_agents

if parameters.is_grouping_agents:
    parameters.max_group_size = args.max_group_size
else:
    parameters.max_group_size = None

parameters.nom_controller_type = args.nom_controller_type

if parameters.nom_controller_type == "rl":
    parameters.is_load_model = True
else:
    parameters.is_load_model = False

parameters.observation_range = 0.5
parameters.n_circles_approximate_vehicle = 3
parameters.adaptive_lambda = True
parameters.random_seed = args.random_seed
parameters.rew_method = "sparse"

if parameters.random_seed == 1:
    parameters.is_save_simulation_video = True
else:
    parameters.is_save_simulation_video = False

# ===============================
# Build system
# ===============================
(
    env,
    decision_making_module,
    optimization_module,
    priority_module,
    cbf_controllers,
    parameters,
) = mappo_cavs(parameters=parameters)

# ===============================
# Rollout
# ===============================
t_0 = time.time()
out_td, frame_list = env.rollout(
    max_steps=parameters.max_steps - 1,
    policy=decision_making_module.policy,
    priority_module=priority_module,
    cbf_controllers=cbf_controllers,
    callback=lambda env, _: env.render(mode="rgb_array", visualize_when_rgb=False),
    auto_cast_to_device=True,
    break_when_any_done=False,
    is_save_simulation_video=parameters.is_save_simulation_video,
)
com_t = time.time() - t_0
print(f"Simulation time: {com_t:.2f} seconds")

# ===============================
# Filename encoding
# ===============================

print(f"parameters.is_using_cbf_testing = {parameters.is_using_cbf_testing}")
name_suffix = get_name_suffix(
    parameters.is_grouping_agents,
    parameters.is_using_cbf_testing,
    parameters.n_agents,
    parameters.random_seed,
    parameters.max_group_size,
    parameters.nom_controller_type,
    parameters.scenario_type.lower(),
)

out_td_filename = f"out_td_{name_suffix}.td"
video_basename = f"video_{name_suffix}"
t_filename = f"computation_t_{name_suffix}.json"

is_trim_td = True
if is_trim_td:
    out_td = trim_td(out_td)
torch.save(out_td, os.path.join(path, out_td_filename))
print(f"Saved: {out_td_filename}")

if parameters.is_save_simulation_video:
    save_video(
        os.path.join(path, video_basename),
        frame_list,
        fps=1 / parameters.dt,
        fmt="mp4",
        quality="low",
    )
    print(f"Saved: {video_basename}.mp4")

if parameters.is_using_cbf_testing:
    # Save computation time for solving the CBF-QP
    with open(os.path.join(path, t_filename), "w") as f:
        json.dump(cbf_controllers[0].hist.qp_solving_t, f)
    print(f"Saved: {t_filename}")
