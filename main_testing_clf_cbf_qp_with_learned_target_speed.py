# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Optional
import argparse
import torch
from sigmarl.helper_common import (
    start_run_log,
    save_video,
    visu_cbf_boundary_distances,
    visu_td_results,
)
from sigmarl.helper_training import (
    Parameters,
    SaveData,
    save_evaluation_summary,
)
import os

import json

from sigmarl.mappo_cavs import mappo_cavs

from sigmarl.constants import SCENARIOS

path = "checkpoints/20260529-backup-cbf/ttc/seed1"  # Adjust this to the path of the folder of your saved model

# ===============================
# Handle command-line arguments
# ===============================
def parse_optional_int(value: str) -> Optional[int]:
    """Parse an integer CLI value while accepting 'none' for disabled grouping."""

    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def parse_args():
    """Parse command-line arguments for the CLF-CBF-QP test rollout."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--scenario_type",
        type=str,
        default="cpm_mixed",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario type to test. See sigmarl/constants.py for available scenarios.",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=4,
        help="Number of agents in the test scenario.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="Number of rollout steps configured for the environment.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=path,
        help="Directory where TD outputs, figures, and videos are saved.",
    )
    parser.add_argument(
        "--is_save_simulation_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the rollout video.",
    )
    parser.add_argument(
        "--is_visu_result",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the TD profile visualization.",
    )
    parser.add_argument(
        "--is_real_time_rendering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause rendering to match real time during rollout.",
    )
    parser.add_argument(
        "--visu_vehicles",
        type=str,
        default="all",
        help="Vehicle indices to visualize in the TD plot: 'all', '0', or '0,2,3'.",
    )
    parser.add_argument(
        "--visu_env_index",
        type=int,
        default=0,
        help="Environment index from the rollout batch used for TD visualization.",
    )
    parser.add_argument(
        "--visu_show",
        action="store_true",
        help="Display the exported TD profile figure after saving it.",
    )
    return parser.parse_args()


args = parse_args()
random_seed = args.seed
scenario_type = args.scenario_type
n_agents = args.n_agents
visu_vehicles = args.visu_vehicles
visu_env_index = args.visu_env_index
visu_show = args.visu_show
path = args.output_dir

print(f"[INFO] Using random seed = {random_seed}")


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


parameters.where_to_save = path
start_run_log(path, "testing")

# Adjust parameters
parameters.is_testing_mode = True
parameters.is_real_time_rendering = args.is_real_time_rendering
parameters.is_save_eval_results = False
parameters.is_load_model = False
parameters.is_load_final_model = False
parameters.is_load_out_td = False
parameters.max_steps = args.max_steps  # 1200 -> 1 min
if parameters.is_load_out_td:
    parameters.num_vmas_envs = 32
else:
    parameters.num_vmas_envs = 1

parameters.scenario_type = scenario_type  # One of "cpm_mixed", "cpm_entire", "intersection_1", "on_ramp_1", "roundabout_1", etc. See sigmarl/constants.py for more scenario types
parameters.n_agents = n_agents
parameters.lane_width = 0.25

parameters.is_save_simulation_video = args.is_save_simulation_video
parameters.is_visualize_short_term_path = True
parameters.is_visualize_lane_boundary = False
parameters.is_visualize_extra_info = True
parameters.random_seed = random_seed
parameters.is_continue_train = False
parameters.rew_method = "sparse"
# CBF parameters
parameters.is_using_centralized_cbf = True
parameters.is_using_cbf_testing = True
parameters.is_using_cbf_training = False

parameters.is_train_motion_policy = False
parameters.is_train_lambda_policy = False

(
    env,
    decision_making_module,
    optimization_module,
    priority_module,
    cbf_controllers,
    parameters,
) = mappo_cavs(parameters=parameters)

render_callback = (
    (lambda env, _: env.render(mode="rgb_array", visualize_when_rgb=True))
    if parameters.is_save_simulation_video
    else None
)

t_0 = time.time()
with torch.inference_mode():
    out_td, frame_list = env.rollout(
        max_steps=parameters.max_steps - 1,
        policy=decision_making_module.policy,
        priority_module=priority_module,
        cbf_controllers=cbf_controllers,
        callback=render_callback,  # mode \in {"human", "rgb_array"}
        auto_cast_to_device=True,
        break_when_any_done=False,
        is_save_simulation_video=parameters.is_save_simulation_video,
    )
print(f"Simulation time: {time.time() - t_0:.2f} seconds")
# Save simulation outputs
requested_group_size = parameters.max_group_size

name_suffix = f"{parameters.scenario_type}"
name_suffix += f"_agents{parameters.n_agents}"
name_suffix += (
    f"_max_group_size{requested_group_size}_{parameters.grouping_method}"
    if requested_group_size is not None
    else "_max_group_sizeNone"
)
name_suffix += f"_proxsuite" if parameters.is_proxsuite_qp else "_cvxpy"
name_suffix += f"_seed{parameters.random_seed}"

out_td_filename = f"out_td_{name_suffix}.td"
video_basename = f"video_{name_suffix}"
fig_basename = f"td_results_profiles_{name_suffix}"
cbf_boundary_fig_basename = f"cbf_boundary_distances_{name_suffix}"

torch.save(out_td, os.path.join(path, out_td_filename))
print(f"Simulation outputs saved to {os.path.join(path, out_td_filename)}")
save_evaluation_summary(
    out_td,
    path,
    name_suffix,
    random_seed=parameters.random_seed,
)
if parameters.is_save_simulation_video:
    save_video(
        os.path.join(path, video_basename),
        frame_list,
        fps=1 / parameters.dt,
        fmt="mp4",
        quality="high",
    )
    print(f"Video saved to {os.path.join(path, video_basename)}.mp4")

if args.is_visu_result:
    visu_td_results(
        out_td=out_td,
        parameters=parameters,
        vehicle_ids=visu_vehicles,
        env_index=visu_env_index,
        save_dir=path,
        fig_name=fig_basename,
        show=visu_show,
    )
    visu_cbf_boundary_distances(
        out_td=out_td,
        parameters=parameters,
        vehicle_ids=visu_vehicles,
        env_index=visu_env_index,
        save_dir=path,
        fig_name=cbf_boundary_fig_basename,
        show=visu_show,
    )
