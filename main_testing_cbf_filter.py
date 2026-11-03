# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test the CBF-QP safety filter on top of a CLF or a trained-RL nominal controller.

# CLF nominal controller (default); no checkpoint needed
python main_testing_cbf_filter.py --nominal clf --scenario_type cpm_mixed --n_agents 8 --lane_cbf_footprint rectangle_sat --veh_veh_cbf_footprint rectangle_sat --max_steps 601 --max_group_size 1 --is_save_simulation_video --no-is_center_centerline

SIGMARL_AGENT_PROFILE=full_scale python main_testing_cbf_filter.py --nominal clf --scenario_type round_location0 --n_agents 8 --lane_cbf_footprint rectangle_sat --veh_veh_cbf_footprint rectangle_sat --max_steps 601 --max_group_size 1 --is_save_simulation_video --no-is_center_centerline

# RL nominal controller; scenario, n_agents, and dt default to the checkpoint's values
SIGMARL_AGENT_PROFILE=full_scale python main_testing_cbf_filter.py --nominal rl --path checkpoints/round_location0/vanilla_seed_1 --scenario_type round_location0 --n_agents 10 --lane_cbf_footprint rectangle_sat --veh_veh_cbf_footprint rectangle_sat --max_steps 601 --max_group_size 1 --is_save_simulation_video

python main_testing_cbf_filter.py --nominal rl --path checkpoints/20261011/cpm_mixed/vanilla_dt0.1_iter250_step128_ttc_sparse_seed_1 --scenario_type cpm_mixed --n_agents 6 --max_steps 601 --max_group_size 1 --is_save_simulation_video

# RL nominal controller with a trained pair-level CBF lambda head (cbf_lambda_head="pair")
SIGMARL_AGENT_PROFILE=full_scale python main_testing_cbf_filter.py --nominal rl --path checkpoints/round_location0/pair_lambda_seed_1 --is_pair_lambda --scenario_type round_location0 --n_agents 10 --lane_cbf_footprint rectangle_sat --veh_veh_cbf_footprint rectangle_sat --max_steps 601 --max_group_size 1 --is_save_simulation_video

# CLF nominal controller with a trained per-agent CBF lambda head (cbf_lambda_head="agent");
# scenario, n_agents, dt, and the CLF settings default to the checkpoint's values
SIGMARL_AGENT_PROFILE=full_scale python main_testing_cbf_filter.py --nominal clf --path checkpoints/round_location0/clf_agent_lambda_seed_1 --is_agent_lambda --scenario_type round_location0 --n_agents 10 --lane_cbf_footprint rectangle_sat --veh_veh_cbf_footprint rectangle_sat --max_steps 601 --max_group_size 1 --is_save_simulation_video --no-is_center_centerline

# CLF nominal controller with a trained pair-level CBF lambda head
SIGMARL_AGENT_PROFILE=full_scale python main_testing_cbf_filter.py --nominal clf --path checkpoints/round_location0/clf_pair_lambda_seed_1 --is_pair_lambda --scenario_type round_location0 --n_agents 10 --lane_cbf_footprint rectangle_sat --veh_veh_cbf_footprint rectangle_sat --max_steps 601 --max_group_size 1 --is_save_simulation_video --no-is_center_centerline

# Lambda head from an intermediate checkpoint (checkpoints_cross_iter/policy_iter150.pth),
# e.g., while the run is still training; works with --is_pair_lambda as well
SIGMARL_AGENT_PROFILE=full_scale python main_testing_cbf_filter.py --nominal clf --path checkpoints/round_location0/clf_agent_lambda_seed_1 --is_agent_lambda --lambda_iter 150 --scenario_type round_location0 --n_agents 10 --lane_cbf_footprint rectangle_sat --veh_veh_cbf_footprint rectangle_sat --max_steps 601 --max_group_size 1 --is_save_simulation_video --no-is_center_centerline
"""

import time
from typing import Optional
import argparse
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from sigmarl.helper_common import (
    VideoWriter,
    start_run_log,
    visu_cbf_boundary_distances,
    visu_td_results,
)
from sigmarl.helper_training import (
    Parameters,
    SaveData,
    get_path_to_save_model,
    save_evaluation_summary,
)
import os
from sigmarl.utils.plot_reward_components_td import plot_reward_components_td

import json

from sigmarl.mappo_cavs import mappo_cavs

from sigmarl.constants import SCENARIOS
from sigmarl.cbf_baselines import CONDITION_BASELINES, LANE_BASELINES, PAIR_BASELINES

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
        "--nominal",
        type=str,
        default="clf",
        choices=["clf", "rl"],
        help="Nominal controller filtered by the CBF-QP: CLF or a trained RL policy.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Checkpoint folder: the RL policy (required with --nominal rl) or, with "
            "--is_agent_lambda or --is_pair_lambda, a lambda run with a CLF nominal "
            "controller."
        ),
    )
    parser.add_argument(
        "--is_load_final_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load final_*.pth instead of the highest-reward model (requires --path).",
    )
    parser.add_argument(
        "--scenario_type",
        type=str,
        default=None,
        choices=sorted(SCENARIOS.keys()),
        help=(
            "Scenario type to test. See sigmarl/constants.py. Default: cpm_mixed "
            "for CLF, the checkpoint's scenario for RL."
        ),
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=None,
        help="Number of agents. Default: 4 for CLF, the checkpoint's value for RL.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "CLF without --path only: sampling time [s] (default 0.1). With --path, "
            "the checkpoint's dt is used."
        ),
    )
    parser.add_argument(
        "--ref_speed",
        type=float,
        default=None,
        help=(
            "CLF only: reference speed [m/s] of the nominal controller. If omitted, the "
            "scenario's reference speed is used, else the Parameters default "
            "(scaled with the agent profile)."
        ),
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=None,
        help=(
            "CLF only: speed limit [m/s] of the agents. If omitted, the scenario's "
            "speed limit is used (see SCENARIOS in sigmarl/constants.py), else the "
            "agent profile's max_speed."
        ),
    )
    parser.add_argument(
        "--clf_nominal_mode",
        type=str,
        default=None,
        choices=["pure_pursuit", "heading_rate"],
        help=(
            "CLF only: steering law of the nominal controller. If omitted, the "
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
        default=False,
        help="Let the CBF-QP only brake relative to the nominal acceleration.",
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
        "--cbf_lambda",
        type=float,
        default=None,
        help=(
            "Fixed CBF decay rate lambda in (0, 1] (higher is more aggressive). If "
            "omitted, the checkpoint's value or the Parameters default is used: the "
            "tuned value of BH-TTCBF or the selected baseline "
            "(sigmarl/cbf_baselines/tuned_defaults.json), else 0.5."
        ),
    )
    parser.add_argument(
        "--lane_cbf_baseline",
        type=str,
        default="none",
        choices=["none", *LANE_BASELINES],
        help=(
            "SOTA baseline that replaces the rectangle_sat lane-boundary rows "
            "(comparison only, see sigmarl/cbf_baselines); 'none' keeps ours."
        ),
    )
    parser.add_argument(
        "--veh_veh_cbf_baseline",
        type=str,
        default="none",
        choices=["none", *PAIR_BASELINES],
        help=(
            "SOTA baseline that replaces the rectangle_sat inter-vehicle rows "
            "(comparison only, see sigmarl/cbf_baselines); 'none' keeps ours."
        ),
    )
    parser.add_argument(
        "--is_backup_horizon_cbf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Backup-horizon TTCBF rows, the proposed condition "
            "(sigmarl/backup_horizon_cbf.py): every lane and pair constraint must "
            "stay safe along the backup rollout after the QP control. Default: on, "
            "unless a CBF baseline, --no-is_cbf_filter, a learned lambda, or a "
            "footprint other than rectangle_sat is selected. Sets "
            "--cbf_qp_failure_fallback brake unless given."
        ),
    )
    parser.add_argument(
        "--is_two_step_rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With the backup-horizon rows: also keep the two-step rectangle_sat "
            "lane and pair rows in the QP (ablation). Without them, the two-step "
            "rows are the CBF condition and always used."
        ),
    )
    parser.add_argument(
        "--condition_cbf_baseline",
        type=str,
        default="none",
        choices=["none", *CONDITION_BASELINES],
        help=(
            "SOTA baseline that replaces the condition of the backup-horizon "
            "rows (comparison only, see sigmarl/cbf_baselines); implies "
            "--is_backup_horizon_cbf. 'none' keeps ours."
        ),
    )
    parser.add_argument(
        "--is_cbf_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "--no-is_cbf_filter applies the nominal control unfiltered: both row "
            "types become 'unfiltered' (no rows), and the shield and the standstill "
            "escape are off."
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
            "stays collision-free (needs --lane_cbf_footprint rectangle_sat)."
        ),
    )
    parser.add_argument(
        "--is_standstill_escape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Let a vehicle that its lane CBF keeps stopped steer at standstill and "
            "creep forward, or backward if the agent profile allows negative "
            "speeds (needs --lane_cbf_footprint rectangle_sat)."
        ),
    )
    parser.add_argument(
        "--is_log_backup_value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store the joint backup value of every agent at every step in the "
            "output TD (diagnostic of the certified set, see "
            "CBFQP.backup_values_all_envs; needs the rectangle_sat footprints)."
        ),
    )
    parser.add_argument(
        "--is_zero_initial_speed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Spawn the vehicles at rest (also for an RL checkpoint trained with "
            "random initial speeds), so that spawns are recoverable."
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
        "--scale_map",
        type=float,
        default=None,
        help="Scale OSM map coordinates around their global center. If omitted, unchanged.",
    )
    parser.add_argument(
        "--lane_width",
        type=float,
        default=None,
        help="Width of each lane. If omitted, unchanged.",
    )
    parser.add_argument(
        "--is_visualize_observed_agents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="RL only: render the agents observed by ego agent 0.",
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
        default=False,
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
    parser.add_argument(
        "--is_pair_lambda",
        action="store_true",
        help=(
            "--path is a pair-lambda training run (cbf_lambda_head='pair'): use its "
            "pair head (final_lambda_policy.pth with --is_load_final_model, else the "
            "best-reward reward*_lambda_policy.pth) and, with --nominal rl, its "
            "frozen motion policy."
        ),
    )
    parser.add_argument(
        "--is_agent_lambda",
        action="store_true",
        help=(
            "--path is a lambda training run with a CLF nominal controller "
            "(cbf_lambda_head='agent'): use its per-agent lambda head "
            "(final_lambda_policy.pth with --is_load_final_model, else the "
            "best-reward reward*_lambda_policy.pth)."
        ),
    )
    parser.add_argument(
        "--lambda_iter",
        type=int,
        default=None,
        help=(
            "With --is_pair_lambda or --is_agent_lambda: load the lambda head from "
            "<path>/checkpoints_cross_iter/policy_iter<N>.pth instead, e.g., while "
            "the run is still training."
        ),
    )
    parser.add_argument(
        "--policy_iter",
        type=int,
        default=None,
        help=(
            "With --nominal rl: deploy the motion policy of iteration N "
            "(<path>/checkpoints_cross_iter/policy_iter<N>.pth) instead of the final "
            "or best-reward one."
        ),
    )
    args = parser.parse_args()
    if not args.is_cbf_filter:
        if args.lane_cbf_baseline != "none" or args.veh_veh_cbf_baseline != "none":
            parser.error("--no-is_cbf_filter excludes the CBF baselines.")
        args.lane_cbf_baseline = "unfiltered"
        args.veh_veh_cbf_baseline = "unfiltered"
        args.is_exact_backup_shield = False
        args.is_standstill_escape = False
    if args.condition_cbf_baseline != "none":
        if args.is_backup_horizon_cbf is False:
            parser.error("--condition_cbf_baseline needs the backup-horizon rows.")
        args.is_backup_horizon_cbf = True
    # Configurations the backup-horizon rows do not support
    bh_unsupported = (
        not args.is_cbf_filter
        or args.lane_cbf_baseline != "none"
        or args.veh_veh_cbf_baseline != "none"
        or args.is_agent_lambda
        or args.is_pair_lambda
        or args.lane_cbf_footprint != "rectangle_sat"
        or args.veh_veh_cbf_footprint != "rectangle_sat"
    )
    if args.is_backup_horizon_cbf is None:
        args.is_backup_horizon_cbf = not bh_unsupported
    if args.is_backup_horizon_cbf:
        if bh_unsupported:
            parser.error(
                "--is_backup_horizon_cbf needs the rectangle_sat footprints, a fixed "
                "lambda, and no CBF baseline or --no-is_cbf_filter."
            )
        if args.cbf_qp_failure_fallback is None:
            args.cbf_qp_failure_fallback = "brake"
    if args.is_agent_lambda and args.nominal != "clf":
        parser.error("--is_agent_lambda requires --nominal clf.")
    if args.lambda_iter is not None:
        if not (args.is_pair_lambda or args.is_agent_lambda):
            parser.error(
                "--lambda_iter requires --is_pair_lambda or --is_agent_lambda."
            )
        if args.is_load_final_model:
            parser.error("--lambda_iter and --is_load_final_model are exclusive.")
    if args.policy_iter is not None and (
        args.nominal != "rl" or args.is_pair_lambda or args.is_agent_lambda
    ):
        parser.error("--policy_iter requires --nominal rl without a lambda head.")
    if args.nominal == "rl":
        if args.path is None:
            parser.error("--path is required with --nominal rl.")
        clf_only = {
            "--dt": args.dt,
            "--ref_speed": args.ref_speed,
            "--max_speed": args.max_speed,
            "--clf_nominal_mode": args.clf_nominal_mode,
        }
        for flag, value in clf_only.items():
            if value is not None:
                parser.error(f"{flag} is only supported with --nominal clf.")
    elif args.is_agent_lambda or args.is_pair_lambda:
        if args.path is None:
            parser.error(
                "--path is required with --is_agent_lambda or --is_pair_lambda."
            )
        if args.dt is not None:
            parser.error(
                "--dt is not supported with --path; the checkpoint's dt is used."
            )
        if args.is_visualize_observed_agents:
            parser.error(
                "--is_visualize_observed_agents is only supported with --nominal rl."
            )
    else:
        if args.path is not None:
            parser.error(
                "--path is only supported with --nominal rl, --is_agent_lambda, "
                "or --is_pair_lambda."
            )
        if args.is_load_final_model:
            parser.error("--is_load_final_model requires --path.")
        if args.is_visualize_observed_agents:
            parser.error(
                "--is_visualize_observed_agents is only supported with --nominal rl."
            )
    return args


def load_checkpoint_parameters(checkpoint_dir: str) -> Parameters:
    """Load the Parameters saved with an RL checkpoint (reward*.json)."""

    json_files = sorted(
        f
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".json") and f.startswith("reward")
    )
    if not json_files:
        raise FileNotFoundError(f"No reward*.json found in '{checkpoint_dir}'.")
    with open(os.path.join(checkpoint_dir, json_files[0]), "r") as file:
        return SaveData.from_dict(json.load(file)).parameters


args = parse_args()
nominal = args.nominal
random_seed = args.seed
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

if args.path is not None:
    parameters = load_checkpoint_parameters(args.path)
    run_motion_policy_path = parameters.pretrained_motion_policy_path
    # Models are loaded from where_to_save; outputs go to `path`
    parameters.where_to_save = args.path
    parameters.is_load_model = True
    parameters.is_load_final_model = args.is_load_final_model
    print(
        f"[INFO] {nominal.upper()} nominal controller with the checkpoint from "
        f"{args.path}; dt = {parameters.dt}."
    )
else:
    parameters = Parameters()
    parameters.dt = args.dt if args.dt is not None else 0.10
    parameters.where_to_save = path
    parameters.is_load_model = False
    parameters.is_load_final_model = False
os.makedirs(path, exist_ok=True)

# Save the command and mirror the output to <path>/command_testing.txt and <path>/log_testing.txt
start_run_log(path, "testing")

# Adjust parameters
parameters.is_testing_mode = True
parameters.is_real_time_rendering = args.is_real_time_rendering
parameters.is_save_eval_results = False
parameters.is_load_out_td = False
parameters.max_steps = args.max_steps
if parameters.is_load_out_td:
    parameters.num_vmas_envs = 32
else:
    parameters.num_vmas_envs = 1

if args.scenario_type is not None:
    if args.path is not None and args.scenario_type != parameters.scenario_type:
        # The map's speed limit and CBF ranges; the policy's speed scale stays
        parameters.retarget_scenario(args.scenario_type)
    else:
        parameters.scenario_type = args.scenario_type
elif args.path is None:
    parameters.scenario_type = "cpm_mixed"
if args.n_agents is not None:
    parameters.n_agents = args.n_agents
elif args.path is None:
    parameters.n_agents = 4
parameters.is_center_centerline = args.is_center_centerline
parameters.is_zero_initial_speed = args.is_zero_initial_speed

parameters.is_save_simulation_video = args.is_save_simulation_video
parameters.is_visualize_short_term_path = True
parameters.is_visualize_lane_boundary = False
parameters.is_visualize_extra_info = True
parameters.is_visualize_observed_agents = args.is_visualize_observed_agents
parameters.observed_agents_render_ego_index = 0
if args.scale_map is not None:
    parameters.scale_map = args.scale_map
if args.lane_width is not None:
    parameters.lane_width = args.lane_width
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
parameters.nom_controller_type = nominal
if args.ref_speed is not None:
    parameters.ref_speed = args.ref_speed
if args.max_speed is not None:
    parameters.max_speed = args.max_speed
if args.clf_nominal_mode is not None:
    parameters.clf_nominal_mode = args.clf_nominal_mode
if args.lane_hessian_mode is not None:
    parameters.lane_hessian_mode = args.lane_hessian_mode
if args.cross_cbf_responsibility is not None:
    parameters.cross_cbf_responsibility = args.cross_cbf_responsibility
if args.cbf_qp_failure_fallback is not None:
    parameters.cbf_qp_failure_fallback = args.cbf_qp_failure_fallback
parameters.is_cbf_limit_acc_to_nominal = args.is_cbf_limit_acc_to_nominal
parameters.n_circles_approximate_vehicle = 3
parameters.adaptive_lambda = True
parameters.is_learn_cbf_lambda = False
parameters.is_log_cbf_boundary_distance = True
parameters.is_aggregate_veh_veh_safety = is_aggregate_veh_veh_safety
parameters.is_aggregate_lane_veh_safety = False
parameters.veh_veh_cbf_footprint = args.veh_veh_cbf_footprint
parameters.lane_cbf_baseline = args.lane_cbf_baseline
if args.cbf_lambda is not None:
    parameters.cbf_lambda = args.cbf_lambda
parameters.veh_veh_cbf_baseline = args.veh_veh_cbf_baseline
parameters.is_backup_horizon_cbf = args.is_backup_horizon_cbf
parameters.condition_cbf_baseline = args.condition_cbf_baseline
parameters.is_two_step_rows = args.is_two_step_rows
parameters.is_exact_backup_shield = is_exact_backup_shield
parameters.is_standstill_escape = is_standstill_escape
parameters.is_visu_shield_rollout = args.is_visu_shield_rollout
parameters.is_visu_sampled_boundary_points = True

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
# No training happens here; the RL motion policy is flagged only so that the
# policy-training-case validation accepts an RL nominal controller.
parameters.is_train_motion_policy = nominal == "rl"
parameters.is_train_lambda_policy = False
if args.is_pair_lambda or args.is_agent_lambda:
    # Lambda-only case: the pair head comes with the run's frozen motion policy
    # (pretrained_motion_policy_path); the per-agent head with the CLF nominal
    parameters.is_train_motion_policy = False
    parameters.is_train_lambda_policy = True
    parameters.is_learn_cbf_lambda = True
    parameters.cbf_lambda_head = "pair" if args.is_pair_lambda else "agent"
    parameters.is_load_pretrained_motion_policy = nominal == "rl"
    parameters.pretrained_motion_policy_path = (
        run_motion_policy_path if nominal == "rl" else ""
    )
    parameters.is_load_pretrained_lambda_policy = True
    # Cross-iteration, final, or best-reward intermediate head
    if args.lambda_iter is not None:
        parameters.pretrained_lambda_policy_path = os.path.join(
            args.path, "checkpoints_cross_iter", f"policy_iter{args.lambda_iter}.pth"
        )
    elif parameters.is_load_final_model:
        parameters.pretrained_lambda_policy_path = os.path.join(
            args.path, "final_lambda_policy.pth"
        )
    else:
        parameters.pretrained_lambda_policy_path = get_path_to_save_model(
            parameters=parameters
        )[3]
    if not os.path.isfile(parameters.pretrained_lambda_policy_path):
        raise FileNotFoundError(
            f"No lambda head at '{parameters.pretrained_lambda_policy_path}'."
        )
parameters.is_joint_task = False
(
    env,
    decision_making_module,
    optimization_module,
    priority_module,
    cbf_controllers,
    parameters,
) = mappo_cavs(parameters=parameters)

if args.policy_iter is not None:
    policy_iter_path = os.path.join(
        args.path, "checkpoints_cross_iter", f"policy_iter{args.policy_iter}.pth"
    )
    decision_making_module.policy.load_state_dict(
        torch.load(policy_iter_path, weights_only=True)
    )
    print(f"[INFO] Loaded the motion policy '{policy_iter_path}'")

# The scenario sizes the viewer for the map (`set_render_resolution`).
scenario = env.scenario
# With a learned CBF lambda head, show the lambdas in a panel right of the map
scenario.add_render_side_panel(scenario.cbf_lambda_panel_width())


def render_frame():
    """Render the current state for normal and pre-reset video frames."""
    return env.render(mode="rgb_array", visualize_when_rgb=True)


render_callback = (
    (lambda _, __: render_frame()) if parameters.is_save_simulation_video else None
)
scenario.set_pre_reset_render_callback(
    render_frame if parameters.is_save_simulation_video else None
)

# Output names
requested_group_size = max_group_size

name_suffix = f"nom{nominal}_{parameters.scenario_type}"
name_suffix += f"_agents{parameters.n_agents}"
name_suffix += (
    f"_max_group_size{requested_group_size}_{parameters.grouping_method}"
    if requested_group_size is not None
    else "_max_group_sizeNone"
)
name_suffix += f"_proxsuite" if parameters.is_proxsuite_qp else "_cvxpy"
if not args.is_cbf_filter:
    name_suffix += "_unfiltered"
else:
    if parameters.lane_cbf_baseline != "none":
        name_suffix += f"_lane-{parameters.lane_cbf_baseline}"
    if parameters.veh_veh_cbf_baseline != "none":
        name_suffix += f"_veh-{parameters.veh_veh_cbf_baseline}"
    if parameters.condition_cbf_baseline != "none":
        name_suffix += f"_cond-{parameters.condition_cbf_baseline}"
    elif parameters.is_backup_horizon_cbf:
        name_suffix += "_bh"
    if parameters.is_backup_horizon_cbf and parameters.is_two_step_rows:
        name_suffix += "_twostep"
name_suffix += f"_seed{parameters.random_seed}"

out_td_filename = f"out_td_{name_suffix}.td"
video_basename = (
    f"video_ego{parameters.observed_agents_render_ego_index}_{name_suffix}"
    if parameters.is_visualize_observed_agents
    else f"video_{name_suffix}"
)
fig_basename = f"td_results_profiles_{name_suffix}"
cbf_boundary_fig_basename = f"cbf_boundary_distances_{name_suffix}"

# Stream frames to the video file instead of keeping them in memory
video_writer = (
    VideoWriter(
        os.path.join(path, video_basename), fps=1 / parameters.dt, quality="high"
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
            callback=render_callback,  # mode \in {"human", "rgb_array"}
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=parameters.is_save_simulation_video,
            frame_sink=video_writer,
            is_log_backup_value=(
                args.is_log_backup_value
                and parameters.lane_cbf_footprint == "rectangle_sat"
                and parameters.veh_veh_cbf_footprint == "rectangle_sat"
            ),
        )
except BaseException:
    if video_writer is not None:
        video_writer.abort()
    raise
simulation_time = time.time() - t_0
num_frames = out_td.numel()
simulation_speed = num_frames / simulation_time
real_time_factor = num_frames * parameters.dt / simulation_time
print(f"Simulation time: {simulation_time:.2f} s")
print(f"Simulation speed: {simulation_speed:.2f} FPS")
print(f"Real-time factor: {real_time_factor:.2f}x")
if parameters.is_proxsuite_qp:
    from sigmarl.cbf_qp_proxsuite import TIME_PARTS, escape_statistics, qp_statistics

    TIME_PART_KEYS = {"total": "time_cbf_parallel"}
    TIME_PART_KEYS.update({part: f"time_cbf_{part}_parallel" for part in TIME_PARTS})

    n_qp_solves, qp_failures = qp_statistics(cbf_controllers)
    print(
        f"[INFO] QP solves: {n_qp_solves}, failures: {sum(qp_failures.values())} "
        f"{dict(qp_failures)} (fallback: {parameters.cbf_qp_failure_fallback})"
    )
    if parameters.is_standstill_escape:
        print(f"[INFO] Standstill escape: {dict(escape_statistics(cbf_controllers))}")

    def summed_stats(counters_of):
        """Counters (e.g., shield_stats) summed over all controllers."""
        flat = [
            ctrl
            for item in cbf_controllers
            for ctrl in (item if isinstance(item, list) else [item])
        ]
        total = {}
        for ctrl in flat:
            for key, value in counters_of(ctrl).items():
                total[key] = total.get(key, 0) + int(value)
        return total

    def time_stats(key):
        """Mean, median, and p95 [ms] of a per-step timing field of out_td."""
        values = out_td[key][0].numpy() * 1e3
        return {
            "mean_ms": float(values.mean()),
            "median_ms": float(torch.tensor(values).median()),
            "p95_ms": float(torch.tensor(values).quantile(0.95)),
        }

    cbf_statistics = {
        "lane_cbf_baseline": parameters.lane_cbf_baseline,
        "veh_veh_cbf_baseline": parameters.veh_veh_cbf_baseline,
        "is_backup_horizon_cbf": parameters.is_backup_horizon_cbf,
        "condition_cbf_baseline": parameters.condition_cbf_baseline,
        "is_two_step_rows": parameters.is_two_step_rows,
        "is_using_centralized_cbf": parameters.is_using_centralized_cbf,
        "is_exact_backup_shield": parameters.is_exact_backup_shield,
        "is_standstill_escape": parameters.is_standstill_escape,
        "qp_solves": n_qp_solves,
        "qp_failures": dict(qp_failures),
        "escape_stats": dict(escape_statistics(cbf_controllers)),
        "shield_stats": summed_stats(lambda ctrl: ctrl.shield_stats),
        "lane_sat_stats": summed_stats(lambda ctrl: ctrl.lane_sat_stats),
        # QPs that failed or needed CBF slack, and the steps with any such QP
        "slack_stats": summed_stats(lambda ctrl: ctrl.slack_stats),
        "slack_active_steps": int(out_td["cbf_slack_active"][0].sum()),
        "lane_baseline_stats": summed_stats(
            lambda ctrl: ctrl.lane_baseline.stats if ctrl.lane_baseline else {}
        ),
        "pair_baseline_stats": summed_stats(
            lambda ctrl: ctrl.pair_baseline.stats if ctrl.pair_baseline else {}
        ),
        # Sequential: actual wall-clock loop time; parallelized: per-agent
        # parallel-execution estimate (equal to sequential when centralized)
        "time_cbf_sequential": time_stats("time_cbf"),
        "time_cbf_parallelized": time_stats("time_cbf_parallel"),
        # Per-step times [ms] of the parallelized estimate and its parts (the
        # slowest agent's when fully decentralized; see CBFQP.time_parts)
        "time_cbf_per_step_ms": {
            name: (out_td[key][0].numpy() * 1e3).tolist()
            for name, key in TIME_PART_KEYS.items()
            if key in out_td.keys()
        },
    }
out_td_path = os.path.join(path, out_td_filename)
torch.save(out_td, out_td_path)
print(f"Simulation outputs saved to {out_td_path}")
save_evaluation_summary(
    out_td,
    path,
    name_suffix,
    random_seed=parameters.random_seed,
    task_performance={
        "is_joint_task": parameters.is_joint_task,
        "num_task_tries": int(scenario.num_task_tries.sum().item()),
        "task_success_times": int(scenario.task_success_times.sum().item()),
    },
    computation_efficiency={
        "simulation_time": simulation_time,
        "simulation_speed": simulation_speed,
        "real_time_factor": real_time_factor,
    },
)
if parameters.is_proxsuite_qp:
    cbf_statistics_path = os.path.join(path, f"cbf_statistics_{name_suffix}.json")
    with open(cbf_statistics_path, "w") as file:
        json.dump(cbf_statistics, file, indent=2)
    print(f"CBF statistics saved to {cbf_statistics_path}")
if parameters.is_save_simulation_video:
    print(f"Video saved to {video_writer.close()}")

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
