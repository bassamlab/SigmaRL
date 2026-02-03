# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Optional
import torch
from sigmarl.helper_common import save_video
from sigmarl.helper_training import Parameters, SaveData
import os
import sys

import json

from sigmarl.mappo_cavs import mappo_cavs

from sigmarl.constants import SCENARIOS

# path = "outputs/marl_cbf_fixed_group_2_episode_100_rl/"
path = "outputs/cbf_informed_marl_with_cbf_4_only_cbf_reward/"
# path = "outputs/cbf_informed_marl_with_cbf_2/"

# ===============================
# Handle command-line arguments
# ===============================
# Usage: python main_training.py [seed]
# Default seed = 1
if len(sys.argv) > 1:
    print(f"sys.argv: {sys.argv}")
    try:
        random_seed = int(sys.argv[1])
    except ValueError:
        print(f"[WARNING] Invalid seed '{sys.argv[1]}', falling back to default = 1")
        random_seed = 1
else:
    random_seed = 1

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


# Load parameters from the saved json file
with open(path_to_json_file, "r") as file:
    data = json.load(file)
    saved_data = SaveData.from_dict(data)
    parameters: Parameters = saved_data.parameters

    # Adjust parameters
    parameters.is_testing_mode = True
    parameters.is_real_time_rendering = False
    parameters.is_save_eval_results = False
    parameters.is_load_model = True
    parameters.is_load_final_model = False
    parameters.is_load_out_td = False
    parameters.max_steps = 2000  # 1200 -> 1 min
    if parameters.is_load_out_td:
        parameters.num_vmas_envs = 32
    else:
        parameters.num_vmas_envs = 1

    parameters.scenario_type = "CPM_mixed"  # One of "CPM_mixed", "CPM_entire", "intersection_1", "on_ramp_1", "roundabout_1", etc. See sigmarl/constants.py for more scenario types
    # parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
    parameters.n_agents = 4
    parameters.dt = 0.1
    parameters.lane_width = 0.25

    parameters.is_save_simulation_video = False
    parameters.is_visualize_short_term_path = True
    parameters.is_visualize_lane_boundary = False
    parameters.is_visualize_extra_info = True
    parameters.is_using_cbf_testing = False
    parameters.is_using_cbf_training = False
    parameters.is_using_prioritized_marl = False
    parameters.is_using_centralized_cbf = True
    parameters.is_grouping_agents = False
    parameters.max_group_size = 2
    parameters.observation_range = 0.5
    parameters.nom_controller_type = "clf"  # "rl" or "clf"
    parameters.n_circles_approximate_vehicle = 3
    parameters.adaptive_lambda = True
    parameters.random_seed = random_seed
    parameters.is_continue_train = False
    parameters.is_solve_qp = False

    (
        env,
        decision_making_module,
        optimization_module,
        priority_module,
        cbf_controllers,
        parameters,
    ) = mappo_cavs(parameters=parameters)

    t_0 = time.time()
    out_td, frame_list = env.rollout(
        max_steps=parameters.max_steps - 1,
        policy=decision_making_module.policy,
        priority_module=priority_module,
        cbf_controllers=cbf_controllers,
        callback=lambda env, _: env.render(
            mode="rgb_array", visualize_when_rgb=True
        ),  # mode \in {"human", "rgb_array"}
        auto_cast_to_device=True,
        break_when_any_done=False,
        is_save_simulation_video=parameters.is_save_simulation_video,
    )
    print(f"Simulation time: {time.time() - t_0:.2f} seconds")
    # Save simulation outputs
    name_suffix = "_grouping_on" if parameters.is_grouping_agents else "_grouping_off"
    name_suffix += f"_seed_{parameters.random_seed}"

    out_td_filename = f"out_td{name_suffix}.td"
    video_basename = f"video{name_suffix}"

    torch.save(out_td, os.path.join(path, out_td_filename))
    print(f"Simulation outputs saved to {os.path.join(path, out_td_filename)}")
    if parameters.is_save_simulation_video:
        save_video(
            os.path.join(path, video_basename),
            frame_list,
            fps=1 / parameters.dt,
            fmt="mp4",
            quality="high",
        )
        print(f"Video saved to {os.path.join(path, video_basename)}.mp4")
