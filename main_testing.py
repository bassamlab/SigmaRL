# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import select
import sys
import time
from typing import Optional

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from sigmarl.helper_common import VideoWriter, trim_td, start_run_log
from sigmarl.helper_training import (
    Parameters,
    SaveData,
    save_evaluation_summary,
)

from sigmarl.mappo_cavs import mappo_cavs

DEFAULT_PATH = "path/to/checkpoint/folder"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the saved-model test rollout."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_PATH,
        help="Path to the folder containing the saved model and reward JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory where TD outputs, summaries, and videos are saved. If "
            "omitted, the checkpoint folder (--path) is used."
        ),
    )
    parser.add_argument(
        "--scenario_type",
        type=str,
        default=None,
        help="Scenario type (defaults to the value saved in the JSON file).",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=None,
        help="Number of agents (defaults to the value saved in the JSON file).",
    )
    parser.add_argument(
        "--num_vmas_envs",
        type=int,
        default=None,
        help="Number of VMAS environments (defaults to the value saved in the JSON file).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum rollout steps (defaults to the value saved in the JSON file).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed (defaults to the value saved in the JSON file).",
    )
    parser.add_argument(
        "--scale_map",
        type=float,
        default=None,
        help=(
            "Scale map coordinates around their global center. "
            "Defaults to the value saved in the JSON file."
        ),
    )
    parser.add_argument(
        "--lane_width",
        type=float,
        default=None,
        help=("Width of each lane in the simulation. "),
    )
    parser.add_argument(
        "--is_center_centerline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Center reference paths between shared boundaries "
            "(--no-is_center_centerline to disable). Defaults to true."
        ),
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Save a simulation video (default: false).",
    )
    parser.add_argument(
        "--fast_inference",
        "--fast-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fast inference mode (default: false).",
    )
    return parser.parse_args()


args = parse_args()
checkpoint_path = args.path
# Models are loaded from `checkpoint_path`; outputs go to `path`
path = args.output_dir if args.output_dir is not None else checkpoint_path
os.makedirs(path, exist_ok=True)
start_run_log(path, "testing")

# Find json file
path_to_json_file: Optional[str] = None
for file in os.listdir(checkpoint_path):
    if file.endswith(".json") and file.startswith("reward"):
        path_to_json_file = os.path.join(checkpoint_path, file)
        break

# Load if exists, otherwise initialize Parameters
if path_to_json_file is not None and os.path.isfile(path_to_json_file):
    with open(path_to_json_file, "r") as file:
        data = json.load(file)
    saved_data = SaveData.from_dict(data)
    parameters: Parameters = saved_data.parameters
else:
    raise FileNotFoundError(f"No JSON file found in folder {checkpoint_path}.")
    parameters = Parameters()  # Optionally, initialize Parameters with default values

# Apply command-line overrides after loading the saved parameters.
for parameter_name in (
    "scenario_type",
    "n_agents",
    "num_vmas_envs",
    "max_steps",
    "random_seed",
    "scale_map",
    "lane_width",
):
    value = getattr(args, parameter_name)
    if value is not None:
        setattr(parameters, parameter_name, value)
parameters.is_save_simulation_video = args.video
parameters.is_fast_inference_mode = args.fast_inference
if args.is_center_centerline is not None:
    parameters.is_center_centerline = args.is_center_centerline
parameters.lane_width = args.lane_width if args.lane_width is not None else 0.3
random_seed = parameters.random_seed

print(f"[INFO] Using random seed = {random_seed}")

# Adjust parameters
parameters.where_to_save = checkpoint_path
parameters.is_testing_mode = True
parameters.is_enable_debug_diagnostics = False
parameters.is_save_detailed_info = True
parameters.is_real_time_rendering = False
parameters.is_save_eval_results = False
parameters.is_load_model = True
parameters.is_load_final_model = False
parameters.is_load_out_td = False
parameters.observed_agents_render_ego_index = 0
parameters.is_joint_task = False

parameters.is_visualize_short_term_path = True
parameters.is_visualize_lane_boundary = False
parameters.is_visualize_extra_info = True
parameters.is_visualize_observed_agents = True

parameters.is_continue_train = False
parameters.rew_method = "sparse"
parameters.is_using_cbf_training = False

print(
    f"[INFO] Model path: {checkpoint_path}\n"
    f"[INFO] Output path: {path}\n"
    f"[INFO] Scenario: {parameters.scenario_type}; "
    f"agents: {parameters.n_agents}; "
    f"max steps: {parameters.max_steps}; "
    f"num VMAS envs: {parameters.num_vmas_envs}; "
    f"random seed: {parameters.random_seed}; "
    f"scale map: {parameters.scale_map}; "
    f"lane width: {parameters.lane_width}; "
    f"center centerline: {parameters.is_center_centerline}; "
    f"fast inference: {parameters.is_fast_inference_mode}; "
    f"save video: {parameters.is_save_simulation_video}"
)

(
    env,
    decision_making_module,
    optimization_module,
    priority_module,
    cbf_controllers,
    parameters,
) = mappo_cavs(parameters=parameters)

# The scenario sizes the viewer for the map (`set_render_resolution`).
scenario = env.scenario
# With a learned CBF lambda head, show the lambdas in a panel right of the map
scenario.add_render_side_panel(scenario.cbf_lambda_panel_width())


def render_frame():
    """Render the current state for normal and pre-reset video frames."""
    return env.render(mode="rgb_array", visualize_when_rgb=False)


render_callback = (
    (lambda _, __: render_frame()) if parameters.is_save_simulation_video else None
)
scenario.set_pre_reset_render_callback(
    render_frame if parameters.is_save_simulation_video else None
)
# Output names
name_suffix = f"_{parameters.scenario_type}_seed_{parameters.random_seed}"

out_td_filename = f"out_td{name_suffix}.td"

if parameters.is_visualize_observed_agents:
    video_basename = (
        f"video_ego{parameters.observed_agents_render_ego_index}{name_suffix}"
    )
else:
    video_basename = f"video{name_suffix}"

# Stream frames to the video file instead of keeping them in memory
video_writer = (
    VideoWriter(
        os.path.join(path, video_basename),
        fps=1 / parameters.dt,
        quality="very_high",
    )
    if parameters.is_save_simulation_video
    else None
)

t_0 = time.time()
try:
    with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC):
        out_td, _ = env.rollout(
            max_steps=parameters.max_steps - 1,
            policy=decision_making_module.policy,
            priority_module=priority_module,
            cbf_controllers=cbf_controllers,
            callback=render_callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=parameters.is_save_simulation_video,
            frame_sink=video_writer,
        )
except BaseException:
    if video_writer is not None:
        video_writer.abort()
    raise
simulation_time = time.time() - t_0
num_frames = out_td.numel()
simulation_speed = num_frames / simulation_time if simulation_time > 0 else 0.0
real_time_factor = (
    num_frames * parameters.dt / simulation_time if simulation_time > 0 else 0.0
)
print(f"Simulation time: {simulation_time:.2f} s")
print(f"Simulation speed: {simulation_speed:.2f} FPS")
print(f"Real-time factor: {real_time_factor:.2f}x")

# Keep only the rollout fields needed for saving and evaluation summaries.
out_td = trim_td(out_td, parameters=parameters)

# Save simulation outputs
torch.save(out_td, os.path.join(path, out_td_filename))
print(f"Simulation outputs saved to {os.path.join(path, out_td_filename)}")
num_task_tries = int(scenario.num_task_tries.sum().item())
task_success_times = int(scenario.task_success_times.sum().item())
if not parameters.is_fast_inference_mode:
    save_evaluation_summary(
        out_td,
        path,
        name_suffix,
        random_seed=parameters.random_seed,
        task_performance={
            "is_joint_task": parameters.is_joint_task,
            "num_task_tries": num_task_tries,
            "task_success_times": task_success_times,
        },
        computation_efficiency={
            "simulation_time": simulation_time,
            "simulation_speed": simulation_speed,
            "real_time_factor": real_time_factor,
        },
    )
if parameters.is_save_simulation_video:
    print(f"Video saved to {video_writer.close()}")
