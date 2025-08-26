# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from sigmarl.helper_training import Parameters, SaveData, reduce_out_td
import os

from vmas.simulator.utils import save_video
import json

from sigmarl.mappo_cavs import mappo_cavs

import pandas as pd
import re

# Load the Excel file
df = pd.read_csv("checkpoints/at25/sigmarl/poses.csv")

# Loop over rows
count = 0
for _, row in df.iterrows():
    count += 1
    print("------------------------------------")
    print(f"----------------{count} / {len(df)}----------------")
    print("------------------------------------")
    # Extract seed number from the "file" column (assumes 'seed#' appears in the string)
    seed_str = [part for part in row["file"].split("/") if part.startswith("seed")][0]
    seed_num = seed_str.replace("seed", "")

    # Construct path
    path = f"checkpoints/at25/sigmarl/seed{seed_num}/"

    td_name = [part for part in row["file"].split("/") if part.startswith("init")][0]
    out_td_path = path + td_name
    video_path = out_td_path.replace(".td", "")

    match = re.search(r"(\d+)(?!.*\d)", td_name)
    if match:
        try_idx = int(match.group(1))
    else:
        raise ValueError(f"No number found in file path: {row['file']}")

    random_seed = try_idx
    print(f"Use random_seed {random_seed}")

    # Convert ref_path_ids to list of integers
    predefined_ref_path_idx = [int(x) for x in str(row["ref_path_ids"]).split("|")]

    # Collect x, y, theta for v1, v2, v3
    init_state = [
        [row["v1_x"], row["v1_y"], row["v1_theta"]],
        [row["v2_x"], row["v2_y"], row["v2_theta"]],
        [row["v3_x"], row["v3_y"], row["v3_theta"]],
    ]

    # Example: print or use them
    print(f"path: {path}")
    print(f"predefined_ref_path_idx: {predefined_ref_path_idx}")
    print(f"init_state: {init_state}")

    try:
        path_to_json_file = next(
            os.path.join(path, file)
            for file in os.listdir(path)
            if file.endswith(".json")
        )  # Find the first json file in the folder
        # Load parameters from the saved json file
        with open(path_to_json_file, "r") as file:
            data = json.load(file)
            saved_data = SaveData.from_dict(data)
            parameters = saved_data.parameters

            # parameters.predefined_ref_path_idx = predefined_ref_path_idx  # Specify the indices of the predefined reference paths to be used
            # parameters.init_state = init_state
            parameters.random_seed = random_seed  # Random seed for training

            # Adjust parameters
            parameters.is_testing_mode = False
            parameters.is_real_time_rendering = False
            parameters.is_save_eval_results = False
            parameters.is_load_model = True
            parameters.is_load_final_model = False
            parameters.is_load_out_td = False
            parameters.max_steps = 18_0000

            if parameters.is_load_out_td:
                parameters.num_vmas_envs = 32
            else:
                parameters.num_vmas_envs = 1

            parameters.scenario_type = "CPM_mixed"  # One of "CPM_mixed", "CPM_entire", "intersection_1", "on_ramp_1", "roundabout_1", etc. See sigmarl/ constants.py for more scenario types
            # parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
            parameters.n_agents = 3

            parameters.is_save_simulation_video = True
            parameters.is_visualize_short_term_path = True
            parameters.is_visualize_lane_boundary = False
            parameters.is_visualize_extra_info = True

            (
                env,
                decision_making_module,
                optimization_module,
                priority_module,
                cbf_controllers,
                parameters,
            ) = mappo_cavs(parameters=parameters)

            out_td, frame_list = env.rollout(
                max_steps=parameters.max_steps - 1,
                policy=decision_making_module.policy,
                priority_module=priority_module,
                callback=lambda env, _: env.render(
                    mode="rgb_array", visualize_when_rgb=True
                ),  # mode \in {"human", "rgb_array"}
                auto_cast_to_device=True,
                break_when_any_done=False,
                is_save_simulation_video=parameters.is_save_simulation_video,
            )
            # Save simulation outputs
            is_save_reduced_td = True
            if is_save_reduced_td:
                print("out_td data reduced")
                out_td = reduce_out_td(out_td)
            torch.save(out_td, out_td_path)
            print(f"Simulation outputs saved to {out_td_path}")
            if parameters.is_save_simulation_video:
                save_video(video_path, frame_list, fps=1 / parameters.dt)
                print(f"Video saved to {video_path}.mp4")
    except StopIteration:
        raise FileNotFoundError("No json file found.")
