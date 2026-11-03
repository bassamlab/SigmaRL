# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""""
python main_testing_clf_cbf_qp.py --scenario_type cpm_entire --n_agents 12 --lane_distance_type softmin --is_proxsuite_qp
python main_testing_clf_cbf_qp.py --scenario_type cpm_entire --n_agents 12 --lane_distance_type softmin --no-is_proxsuite_qp
python main_testing_clf_cbf_qp.py --scenario_type cpm_entire --n_agents 12 --lane_distance_type pseudo --is_proxsuite_qp
python main_testing_clf_cbf_qp.py --scenario_type cpm_entire --n_agents 12 --lane_distance_type pseudo --no-is_proxsuite_qp
"""

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
from sigmarl.utils.plot_reward_components_td import plot_reward_components_td

import json

from sigmarl.mappo_cavs import mappo_cavs

from sigmarl.constants import SCENARIOS

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
    parser.add_argument("--seed", type=int, default=1)
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
        "--ref_speed",
        type=float,
        default=None,
        help=(
            "Reference speed [m/s] of the CLF nominal controller. If omitted, the "
            "Parameters default is used (scaled with the agent profile)."
        ),
    )
    parser.add_argument(
        "--clf_nominal_mode",
        type=str,
        default=None,
        choices=["pure_pursuit", "heading_rate"],
        help=(
            "Steering law of the CLF nominal controller. If omitted, the "
            "Parameters default is used."
        ),
    )
    parser.add_argument(
        "--lane_hessian_mode",
        type=str,
        default=None,
        choices=["geometric", "stencil"],
        help=(
            "Hessian of the lane-distance surrogate. If omitted, the Parameters "
            "default is used."
        ),
    )
    parser.add_argument(
        "--cross_cbf_responsibility",
        type=str,
        default=None,
        choices=["half", "follower"],
        help=(
            "Split of decentralized pair CBFs between the two agents. If omitted, "
            "the Parameters default is used."
        ),
    )
    parser.add_argument(
        "--is_cbf_limit_acc_to_nominal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Let the CBF-QP only brake relative to the nominal acceleration. If "
            "omitted, the Parameters default is used."
        ),
    )
    parser.add_argument(
        "--lane_distance_type",
        type=str,
        default="softmin",
        choices=["pseudo", "softmin"],
        help="Lane distance type used by the CBF-QP.",
    )
    parser.add_argument(
        "--lane_cbf_footprint",
        type=str,
        default="rectangle_sat",
        choices=["circle", "rectangle", "rectangle_sat"],
        help=(
            "Vehicle footprint of the lane CBF: covering circles, or certified "
            "rectangle segments with a signed-distance barrier, curvature margin, "
            "and exact heading coupling (rectangle needs softmin), or the exact "
            "rectangle-polyline (separating-axis) barrier with the exact two-step "
            "Euler map (rectangle_sat)."
        ),
    )
    parser.add_argument(
        "--is_proxsuite_qp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ProxSuite for solving the CBF-QP.",
    )
    parser.add_argument(
        "--veh_veh_cbf_footprint",
        type=str,
        default="rectangle_sat",
        choices=["circle", "mtv", "rectangle_sat"],
        help=(
            "Vehicle footprint of the vehicle-vehicle CBF: covering circles (see "
            "--is_aggregate_veh_veh_safety), the learned MTV safety margin, or the "
            "exact rectangle-rectangle (separating-axis) barrier with the exact "
            "two-step Euler map (rectangle_sat, needs --lane_cbf_footprint "
            "rectangle_sat)."
        ),
    )
    parser.add_argument(
        "--is_backup_horizon_cbf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Backup-horizon TTCBF rows, the proposed condition "
            "(sigmarl/backup_horizon_cbf.py). Default: on with the rectangle_sat "
            "footprints and ProxSuite, off otherwise."
        ),
    )
    parser.add_argument(
        "--is_aggregate_veh_veh_safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggregate circle-pair vehicle-vehicle CBF constraints per neighbor (circle footprint only).",
    )
    parser.add_argument(
        "--is_exact_backup_shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the exact backup shield as the last filter after the CBF-QP: a "
            "control is kept only if the joint emergency-braking rollout after it "
            "stays collision-free (only with --lane_cbf_footprint rectangle_sat)."
        ),
    )
    parser.add_argument(
        "--is_standstill_escape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Let a vehicle that its lane CBF keeps stopped steer at standstill and "
            "creep forward, or backward if the agent profile allows negative "
            "speeds (only with --lane_cbf_footprint rectangle_sat)."
        ),
    )
    parser.add_argument(
        "--is_visu_shield_rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Visualize the latest backup-shield rollout trajectories in rendering.",
    )
    parser.add_argument(
        "--max_group_size",
        type=parse_optional_int,
        default=1,  # Set to None for no grouping (full centralized CBF), or set to 1 for fully decentralized CBF
        help=(
            "Maximum group size for grouped CBF-QP. If omitted, grouping is "
            "disabled and the full centralized CBF is used."
        ),
    )
    parser.add_argument(
        "--grouping_method",
        type=str,
        default="interaction_risk",
        choices=["interaction_risk", "k_nearest"],
        help="Grouping method used when --max_group_size is provided.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=601,
        help="Number of rollout steps configured for the environment.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory where TD outputs, figures, and videos are saved. If "
            "omitted, 'outputs/yyyymmdd-hhmmss' is used."
        ),
    )
    parser.add_argument(
        "--scale_map",
        type=float,
        default=1.0,
        help="Scale map coordinates around their global center (e.g., 20 for cpm maps with the full_scale profile).",
    )
    parser.add_argument(
        "--is_center_centerline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center the centerlines.",
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
        "--cbf_qp_failure_fallback",
        type=str,
        default=None,
        choices=["nominal", "brake"],
        help="Control of an agent whose QP is not solved (default: the Parameters default).",
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
lane_distance_type = args.lane_distance_type
is_proxsuite_qp = args.is_proxsuite_qp
is_aggregate_veh_veh_safety = args.is_aggregate_veh_veh_safety
is_exact_backup_shield = (
    args.is_exact_backup_shield and args.lane_cbf_footprint == "rectangle_sat"
)
is_standstill_escape = (
    args.is_standstill_escape and args.lane_cbf_footprint == "rectangle_sat"
)
max_group_size = args.max_group_size
grouping_method = args.grouping_method
visu_vehicles = args.visu_vehicles
visu_env_index = args.visu_env_index
visu_show = args.visu_show
path = (
    args.output_dir
    if args.output_dir is not None
    else os.path.join("outputs", time.strftime("%Y%m%d-%H%M%S"))
)

print(f"[INFO] Using random seed = {random_seed}")

parameters = Parameters()  # Initialize with default values if no JSON file is found

parameters.dt = 0.10
parameters.where_to_save = path
start_run_log(path, "testing")

# Adjust parameters
parameters.is_testing_mode = True
parameters.is_real_time_rendering = args.is_real_time_rendering
parameters.is_save_eval_results = False
parameters.is_load_model = False
parameters.is_load_final_model = False
parameters.is_load_out_td = False
parameters.max_steps = args.max_steps
if parameters.is_load_out_td:
    parameters.num_vmas_envs = 32
else:
    parameters.num_vmas_envs = 1

parameters.scenario_type = scenario_type  # One of "cpm_mixed", "cpm_entire", "intersection_1", "on_ramp_1", "roundabout_1", etc. See sigmarl/constants.py for more scenario types
parameters.n_agents = n_agents
parameters.is_center_centerline = args.is_center_centerline
parameters.scale_map = args.scale_map

parameters.is_save_simulation_video = args.is_save_simulation_video
parameters.is_visualize_short_term_path = True
parameters.is_visualize_lane_boundary = False
parameters.is_visualize_extra_info = True
parameters.is_visualize_observed_agents = False
parameters.is_using_prioritized_marl = False
parameters.random_seed = random_seed
parameters.is_continue_train = False
parameters.rew_method = "sparse"
parameters.is_use_sampled_lane_boundary_for_cbf = True
# CBF parameters
parameters.is_using_centralized_cbf = True
parameters.is_using_cbf_testing = True
parameters.is_using_cbf_training = False
parameters.is_solve_qp = True
parameters.is_apply_cbf_action = True
parameters.adaptive_lambda = False
parameters.lane_distance_type = lane_distance_type  # "pseudo" or "softmin"
parameters.lane_cbf_footprint = args.lane_cbf_footprint
parameters.is_proxsuite_qp = (
    is_proxsuite_qp  # Whether to use ProxSuite for solving the CBF-QP
)
parameters.is_grouping_agents = max_group_size is not None
parameters.max_group_size = max_group_size if max_group_size is not None else 2
parameters.grouping_method = grouping_method  # {"interaction_risk", "k_nearest"}
parameters.grouping_use_hysteresis = True
if parameters.is_grouping_agents and parameters.max_group_size == 1:
    parameters.is_using_centralized_cbf = False
    print("[INFO] Using fully decentralized CBF since max_group_size=1.")
parameters.nom_controller_type = "clf"  # "rl" or "clf"
if args.ref_speed is not None:
    parameters.ref_speed = args.ref_speed
if args.clf_nominal_mode is not None:
    parameters.clf_nominal_mode = args.clf_nominal_mode
if args.lane_hessian_mode is not None:
    parameters.lane_hessian_mode = args.lane_hessian_mode
if args.cross_cbf_responsibility is not None:
    parameters.cross_cbf_responsibility = args.cross_cbf_responsibility
if args.cbf_qp_failure_fallback is not None:
    parameters.cbf_qp_failure_fallback = args.cbf_qp_failure_fallback
if args.is_cbf_limit_acc_to_nominal is not None:
    parameters.is_cbf_limit_acc_to_nominal = args.is_cbf_limit_acc_to_nominal
parameters.n_circles_approximate_vehicle = 3
parameters.adaptive_lambda = True
parameters.is_learn_cbf_lambda = False
# The backup-horizon rows need the rectangle_sat footprints (ProxSuite only)
parameters.is_backup_horizon_cbf = (
    args.is_backup_horizon_cbf
    if args.is_backup_horizon_cbf is not None
    else args.lane_cbf_footprint == "rectangle_sat"
    and args.veh_veh_cbf_footprint == "rectangle_sat"
    and is_proxsuite_qp
)
parameters.is_log_cbf_boundary_distance = True
parameters.is_aggregate_veh_veh_safety = is_aggregate_veh_veh_safety
parameters.is_aggregate_lane_veh_safety = False
parameters.veh_veh_cbf_footprint = args.veh_veh_cbf_footprint
parameters.is_exact_backup_shield = is_exact_backup_shield
parameters.is_standstill_escape = is_standstill_escape
parameters.is_visu_shield_rollout = args.is_visu_shield_rollout

parameters.is_alternating_action_lambda_training = False
parameters.pretrained_motion_policy_path = (
    ""  # Adjust this to the path of your motion policy
)
parameters.pretrained_lambda_policy_path = ""
parameters.is_load_pretrained_motion_policy = (
    True if parameters.pretrained_motion_policy_path else False
)
parameters.is_load_pretrained_lambda_policy = (
    True if parameters.pretrained_lambda_policy_path else False
)
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
if parameters.is_proxsuite_qp:
    from sigmarl.cbf_qp_proxsuite import escape_statistics, qp_statistics

    n_qp_solves, qp_failures = qp_statistics(cbf_controllers)
    print(
        f"[INFO] QP solves: {n_qp_solves}, failures: {sum(qp_failures.values())} "
        f"{dict(qp_failures)} (fallback: {parameters.cbf_qp_failure_fallback})"
    )
    if parameters.is_standstill_escape:
        print(f"[INFO] Standstill escape: {dict(escape_statistics(cbf_controllers))}")
# Save simulation outputs
requested_group_size = max_group_size

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

out_td_path = os.path.join(path, out_td_filename)
torch.save(out_td, out_td_path)
print(f"Simulation outputs saved to {out_td_path}")
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
    # visu_cbf_boundary_distances(
    #     out_td=out_td,
    #     parameters=parameters,
    #     vehicle_ids=visu_vehicles,
    #     env_index=visu_env_index,
    #     save_dir=path,
    #     fig_name=cbf_boundary_fig_basename,
    #     show=visu_show,
    # )
    # plot_reward_components_td(td_path=out_td_path)
