import time
import torch
from sigmarl.helper_common import save_video
from sigmarl.helper_training import Parameters, SaveData
import os
import sys
import json
import argparse

from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.constants import SCENARIOS

# ===============================
# Paths
# ===============================
path = "outputs/marl_cbf_0/"

# ===============================
# Argument parsing
# ===============================
parser = argparse.ArgumentParser(description="Run MARL-CBF evaluation")

parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--is_grouping_agents", action="store_true", default=None)
parser.add_argument("--max_group_size", type=int, default=2)
parser.add_argument("--nom_controller_type", type=str, default="clf")
parser.add_argument("--scenario_type", type=str, default="CPM_entire")
parser.add_argument("--n_agents", type=int, default=8)

args = parser.parse_args()

print("[INFO] Parsed arguments:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")


from tensordict import TensorDict


def trim_td(out_td: TensorDict) -> TensorDict:
    """
    Trim a TensorDict to keep only selected fields.

    Args:
        out_td: Original TensorDict.

    Returns:
        A new TensorDict containing only the specified keys.
    """
    keys_to_keep = [
        ("agents", "info", "pos"),
        ("agents", "info", "rot"),
        ("agents", "info", "vel"),
        ("agents", "info", "ref"),
        ("agents", "info", "ref_lanelet_ids"),
        ("agents", "info", "is_collision_with_agents"),
        ("agents", "info", "is_collision_with_lanelets"),
    ]

    trimmed = TensorDict(
        {},
        batch_size=out_td.batch_size,
        device=out_td.device,
    )

    for key in keys_to_keep:
        trimmed.set(key, out_td.get(key))

    return trimmed


# ===============================
# Load parameters from JSON
# ===============================
try:
    path_to_json_file = next(
        os.path.join(path, file)
        for file in os.listdir(path)
        if (file.endswith(".json") and file.startswith("reward"))
    )

    with open(path_to_json_file, "r") as file:
        data = json.load(file)
        saved_data = SaveData.from_dict(data)
        parameters: Parameters = saved_data.parameters

    # ===============================
    # Adjust parameters
    # ===============================
    parameters.is_testing_mode = True
    parameters.is_real_time_rendering = False
    parameters.is_save_eval_results = False
    parameters.is_load_model = True
    parameters.is_load_final_model = False
    parameters.is_load_out_td = False

    parameters.max_steps = 1000
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

    parameters.is_using_cbf_testing = True
    parameters.is_using_cbf_training = False
    parameters.is_using_prioritized_marl = False
    parameters.is_using_centralized_cbf = True

    parameters.is_grouping_agents = args.is_grouping_agents
    parameters.max_group_size = args.max_group_size
    parameters.nom_controller_type = args.nom_controller_type

    parameters.observation_range = 0.5
    parameters.n_circles_approximate_vehicle = 3
    parameters.adaptive_lambda = True
    parameters.random_seed = args.random_seed

    if parameters.random_seed == 0:
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
    name_suffix = (
        f"_agents_{parameters.n_agents}"
        f"_seed_{parameters.random_seed}"
        f"_grouping_{'on' if parameters.is_grouping_agents else 'off'}"
        f"_maxgroup_{parameters.max_group_size}"
        f"_nom_{parameters.nom_controller_type}"
        f"_scenario_{parameters.scenario_type.lower()}"
    )

    out_td_filename = f"out_td{name_suffix}.td"
    video_basename = f"video{name_suffix}"
    t_filename = f"computation_t{name_suffix}.json"

    is_trim_td = True
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

    # Save computation time for solving the CBF-QP
    with open(os.path.join(path, t_filename), "w") as f:
        json.dump(cbf_controllers[0].hist.qp_solving_t, f)
    print(f"Saved: {t_filename}")

except StopIteration:
    raise FileNotFoundError("No JSON file found in output folder.")
