# Copyright (c) 2025, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import shlex
import subprocess
import sys
import warnings
import numpy as np
from vmas.simulator.core import AgentState, Agent
from vmas.simulator.utils import TorchUtils, override
from sigmarl.constants import (
    ACTIVE_AGENT_PROFILE,
    AGENTS,
    H_NOM,
    LENGTH_SCALE,
    POSITION_SPEED_LATERAL_OFFSET_BOUND,
    POSITION_SPEED_LOOKAHEAD_DIST,
    SCENARIOS,
    SPEED_SCALE,
    validate_scenario_agent_profile,
)

import torch
from torch import Tensor
from tensordict import TensorDict

import typing
from matplotlib import colormaps


def get_model_name(parameters):
    model_name = f"reward{parameters.episode_reward_mean_current:.2f}"

    return model_name


def compute_applied_action_control(
    action_u: Tensor,
    speed: Tensor,
    steering: Tensor,
    dt: float,
    *,
    dynamics=None,
    rl_action_type: str | None = None,
    lookahead_dist: float | None = None,
    l_wb: float | None = None,
    position_speed_lateral_max: float | None = None,
    pos: Tensor | None = None,
    rot: Tensor | None = None,
    path_tracking_ref_point: Tensor | None = None,
    target_action_override: Tensor | None = None,
    target_speed_min: float | None = None,
    target_speed_max: float | None = None,
    target_steering_min: float | None = None,
    target_steering_max: float | None = None,
    min_acc: float | None = None,
    max_acc: float | None = None,
    min_steering_rate: float | None = None,
    max_steering_rate: float | None = None,
):
    """
    Convert an environment action into the target speed/steering and low-level
    acceleration/steering-rate command that WorldCustom applies.
    """

    action_u = action_u.clone()
    if action_u.ndim == 1:
        action_u = action_u.unsqueeze(0)
        is_batch = False
    else:
        is_batch = True

    device = action_u.device
    dtype = action_u.dtype

    def _as_vector(value):
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.unsqueeze(0)
        if value.ndim == 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value

    def _resolve(name, explicit, default):
        if explicit is not None:
            return explicit
        if dynamics is not None and hasattr(dynamics, name):
            return getattr(dynamics, name)
        return default

    def _as_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    speed = _as_vector(speed)
    steering = _as_vector(steering)

    rl_action_type = _resolve("rl_action_type", rl_action_type, "speed_steer")
    lookahead_dist = _as_float(
        _resolve("lookahead_dist", lookahead_dist, POSITION_SPEED_LOOKAHEAD_DIST)
    )
    l_wb = _as_float(_resolve("l_wb", l_wb, AGENTS["l_wb"]))
    position_speed_lateral_max = _as_float(
        _resolve(
            "position_speed_lateral_max",
            position_speed_lateral_max,
            POSITION_SPEED_LATERAL_OFFSET_BOUND,
        )
    )

    target_speed_max = _as_float(
        _resolve("max_speed", target_speed_max, AGENTS["max_speed"])
    )
    target_speed_min = _as_float(
        _resolve("min_speed", target_speed_min, -target_speed_max)
    )
    target_steering_max = _as_float(
        _resolve("max_steering", target_steering_max, AGENTS["max_steering"])
    )
    target_steering_min = _as_float(
        target_steering_min if target_steering_min is not None else -target_steering_max
    )
    min_acc = _as_float(_resolve("min_acc", min_acc, AGENTS["min_acc"]))
    max_acc = _as_float(_resolve("max_acc", max_acc, AGENTS["max_acc"]))
    min_steering_rate = _as_float(
        _resolve("min_steering_rate", min_steering_rate, AGENTS["min_steering_rate"])
    )
    max_steering_rate = _as_float(
        _resolve("max_steering_rate", max_steering_rate, AGENTS["max_steering_rate"])
    )

    if rl_action_type == "speed":
        target_speed = action_u[:, 0]
        if pos is None or rot is None or path_tracking_ref_point is None:
            target_steering = steering
        else:
            target_steering = compute_path_tracking_target_steering(
                pos=pos,
                rot=rot,
                ref_point=path_tracking_ref_point,
                l_wb=l_wb,
            ).to(device=device, dtype=dtype)
            if target_steering.ndim == 0:
                target_steering = target_steering.unsqueeze(0)
        target_action = torch.stack([target_speed, target_steering], dim=1)
    elif rl_action_type == "position_speed":
        lateral_offset = torch.clamp(
            action_u[:, 0],
            min=-position_speed_lateral_max,
            max=position_speed_lateral_max,
        )
        target_speed = action_u[:, 1]
        ld = action_u.new_tensor(lookahead_dist)
        wheelbase = action_u.new_tensor(l_wb)
        target_steering = torch.atan(
            2.0 * wheelbase * lateral_offset / (ld**2 + lateral_offset**2)
        )
        target_action = torch.stack([target_speed, target_steering], dim=1)
    else:
        target_action = action_u

    if target_action_override is not None:
        target_action_override = target_action_override.to(
            device=device, dtype=dtype
        ).clone()
        if target_action_override.ndim == 1:
            target_action_override = target_action_override.unsqueeze(0)
        override_mask = torch.isfinite(target_action_override).all(dim=1)
        target_action[override_mask] = target_action_override[override_mask]

    target_action[:, 0] = torch.clamp(
        target_action[:, 0], target_speed_min, target_speed_max
    )
    target_action[:, 1] = torch.clamp(
        target_action[:, 1], target_steering_min, target_steering_max
    )

    u_acc = (target_action[:, 0] - speed) / dt
    u_steering_rate = (target_action[:, 1] - steering) / dt
    u = torch.stack(
        [
            torch.clamp(u_acc, min_acc, max_acc),
            torch.clamp(u_steering_rate, min_steering_rate, max_steering_rate),
        ],
        dim=1,
    )

    if not is_batch:
        target_action = target_action.squeeze(0)
    return target_action, u


def select_short_term_reference_point(
    short_term_ref: Tensor,
    preferred_index: int = 2,
) -> Tensor:
    """
    Select a point from a short-term reference path.

    The default index is 2, i.e. the third short-term point when available.
    If fewer points are configured, the last available point is used.
    """
    if short_term_ref.shape[-1] != 2:
        raise ValueError(
            "short_term_ref is expected to end with coordinate dimension 2."
        )
    n_points = short_term_ref.shape[-2]
    if n_points < 1:
        raise ValueError("short_term_ref must contain at least one point.")
    ref_index = min(preferred_index, n_points - 1)
    return short_term_ref[..., ref_index, :]


def compute_path_tracking_target_steering(
    pos: Tensor,
    rot: Tensor,
    ref_point: Tensor,
    l_wb: float,
) -> Tensor:
    """
    Pure-pursuit target steering to track a global reference point.
    """
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
        is_batch = False
    else:
        is_batch = True

    if rot.ndim == 0:
        rot = rot.unsqueeze(0)
    if rot.ndim == 2 and rot.shape[-1] == 1:
        rot = rot.squeeze(-1)

    if ref_point.ndim == 1:
        ref_point = ref_point.unsqueeze(0)

    device = pos.device
    dtype = pos.dtype
    ref_point = ref_point.to(device=device, dtype=dtype)
    rot = rot.to(device=device, dtype=dtype)

    dx = ref_point[:, 0] - pos[:, 0]
    dy = ref_point[:, 1] - pos[:, 1]
    cos_psi = torch.cos(rot)
    sin_psi = torch.sin(rot)
    x_local = cos_psi * dx + sin_psi * dy
    y_local = -sin_psi * dx + cos_psi * dy
    dist_sq = torch.clamp(x_local**2 + y_local**2, min=1e-8)
    wheelbase = pos.new_tensor(float(l_wb))
    target_steering = torch.atan(2.0 * wheelbase * y_local / dist_sq)

    if not is_batch:
        target_steering = target_steering.squeeze(0)
    return target_steering


class Parameters:
    POLICY_TRAINING_CASES = ("motion_only", "lambda_only", "joint", "none")

    """
    This class stores parameters for training and testing.
    """

    def __init__(
        self,
        # General parameters
        n_agents: int = 4,  # Number of agents
        dt: float = 0.05,  # [s] sample time
        device: str = "cpu",  # Default tensor device
        scenario_name: str = "road_traffic",  # Scenario name
        # Training parameters
        n_iters: int = 250,  # Number of training iterations
        num_epochs: int = 30,  # Optimization steps per batch of data collected
        minibatch_size: int = 512,  # Size of the mini-batches in each optimization step (2**9 - 2**12?)
        lr: float = 2e-4,  # Learning rate
        lr_min: float = 1e-5,  # Minimum learning rate (used for scheduling of learning rate)
        lr_slow: float = 2e-5,  # Slow motion-policy learning rate for multi-timescale learning
        lr_slow_min: float = 2e-6,  # Minimum slow motion-policy learning rate
        multi_timescale_switching_period: int = 100,  # Motion PPO optimizer steps between fast-agent rotations
        lr_priority: float = 2e-5,  # Learning rate for the learned priority policy
        lr_priority_min: float = 2e-6,  # Minimum learned-priority learning rate
        max_grad_norm: float = 1.0,  # Maximum norm for the gradients
        clip_epsilon: float = 0.2,  # Clip value for PPO loss
        gamma: float = 0.99,  # Discount factor from 0 to 1. A greater value corresponds to a better farsight
        lmbda: float = 0.9,  # lambda for generalised advantage estimation
        entropy_eps: float = 1e-4,  # Controls the trade-off between trying new actions (exploration) and optimizing known good actions (exploitation). Higher entropy_coeff encourages more exploration by favoring stochastic (less certain) policies.
        entropy_eps_priority: float = 0.01,  # Entropy coefficient for the learned priority policy
        max_steps: int = 128,  # Episode steps before done
        num_vmas_envs: int = 32,  # Number of vectorized environments
        scenario_type: str = "intersection_1",  # One of {"cpm_entire", "cpm_mixed", "intersection_1", ...}. See SCENARIOS in utilities/constants.py for more scenarios.
        # "cpm_entire": Entire map of the CPM Lab
        # "cpm_mixed": Intersection, merge-in, and merge-out of the CPM Lab. Probability defined in `cpm_scenario_probabilities`
        # "intersection_1": Intersection with ID 1
        agent_profile: str = None,  # Named AGENTS profile used to train a saved model. See sigmarl/constants.py for available profiles.
        episode_reward_mean_current: float = 0.00,  # Achieved mean episode reward (total/n_agents)
        episode_reward_intermediate: float = -1e3,  # A arbitrary, small initial value
        is_prb: bool = False,  # Whether to enable prioritized replay buffer
        reset_at_each_iter: bool = False,  # Whether the data collector resets the env at the start of each iteration (recommended for the lab backend so no transition spans the pause between iterations)
        is_challenging_initial_state_buffer=False,  # Whether to enable challenging initial state buffer
        cpm_scenario_probabilities=[
            1.0,
            0.0,
            0.0,
        ],  # Probabilities of training agents in intersection, merge-in, or merge-out scenario
        n_steps_stored: int = 10,  # Store previous `n_steps_stored` steps of states
        # Observation
        n_points_short_term: int = 3,  # Number of points that build a short-term reference path
        short_term_ref_path_spacing_m: float = None,  # Distance between smooth short-term reference path points
        n_points_nearing_boundary: int = 8,  # Number of nearing-boundary points to observe
        is_smooth_sampling: bool = True,  # Whether to sample local reference and boundary points by continuous route progress
        is_use_simple_road_distance: bool = True,  # Whether to compute road distances from sampled local points
        is_partial_observation: bool = True,  # Whether to enable partial observation
        n_nearing_agents_observed: int = 3,  # Number of nearing agents to be observed (consider limited sensor range)
        # Parameters for ablation studies
        is_ego_view: bool = True,  # Ego view or bird view
        is_apply_distance_mask: bool = True,  # Whether to mask distant agents
        is_apply_velocity_mask: bool = True,  # Whether to mask agents whose swept footprints do not intersect
        velocity_mask_margin: float = 0.5,  # Additional disk-radius margin factor for velocity-based masking
        rl_obs_range: float = None,  # Range within which other agents are visible to the RL observation; None scales 1.0 * LENGTH_SCALE with the squared speed limit (see resolve_speed_defaults)
        visible_agent_dropout_probability: float = 0.0,  # Training-time probability of randomly masking a visible observed agent
        is_observe_distance_to_agents: bool = True,  # Whether to observe the distance to other agents
        is_observe_distance_to_boundaries: bool = True,  # Whether to additionally observe distances to lanelet boundaries
        is_observe_boundary_points: bool = True,  # Whether to observe the coordinates of boundary points
        is_observe_distance_to_center_line: bool = True,  # Whether to observe the distance to reference path
        is_observe_vertices: bool = True,  # Whether to observe the vertices of other agents (or center point)
        is_obs_noise: bool = False,  # Whether to add noise to observations
        obs_noise_level: float = 0.05,  # Defines the variance of the normal distribution modeling the noise.
        is_observe_ref_path_other_agents: bool = False,  # Whether to observe the reference paths of other agents
        is_use_mtv_distance: bool = True,  # Whether to use mtv-based (Minimum Translation Vector) distance or c2c-based (center-to-center) distance.
        # Visu
        is_visualize_short_term_path: bool = True,  # Whether to visualize short-term reference paths
        is_visualize_lane_boundary: bool = False,  # Whether to visualize lane boundary
        is_visu_sampled_boundary_points: bool = False,  # Whether to visualize sampled boundary points
        is_visu_applied_action: bool = False,  # Whether to visualize the action applied to the environment
        is_visu_nom_action: bool = False,  # Whether to visualize the nominal action
        is_visu_shield_rollout: bool = False,  # Whether to visualize the backup-shield rollout trajectories
        is_visu_driving_direction: bool = False,  # Whether to draw chevrons showing the driving direction of lanes
        is_visu_veh_direction: bool = True,  # Whether to draw a chevron on each vehicle showing its heading
        is_real_time_rendering: bool = False,  # Simulation will be paused at each time step for a certain duration to enable real-time rendering
        is_visualize_extra_info: bool = True,  # Whether to render extra information such time and time step
        is_visualize_observed_agents: bool = True,  # Whether to visualize observed agents for a selected ego agent
        observed_agents_render_ego_index: int = 0,  # Ego agent index for observed-agent visualization
        cross_iter_ego_indices: typing.Optional[
            typing.Sequence[int]
        ] = None,  # Ego agent indices for cross-iteration testing
        render_title: str = "",  # The title to be rendered
        # Save/Load
        is_save_intermediate_model: bool = True,  # Whether to save intermediate model (also called checkpoint) with the hightest episode reward
        is_load_model: bool = False,  # Whether to load saved model
        is_load_final_model: bool = False,  # Whether to load the final model (last iteration)
        model_name: str = None,
        where_to_save: str = "outputs/",  # Define where to save files such as intermediate models
        is_continue_train: bool = False,  # Whether to continue training after loading an offline model
        must_save_checkpoints: list = None,  # Checkpoints to save at fixed iterations
        is_cross_iter_test: bool = False,  # Whether to run cross-iteration testing
        is_save_eval_results: bool = True,  # Whether to save evaluation results such as figures and evaluation outputs
        is_load_out_td: bool = False,  # Whether to load evaluation outputs
        is_save_detailed_info: bool = False,  # Whether to retain detailed per-step diagnostic info fields
        is_testing_mode: bool = False,  # In testing mode, collisions do not terminate the current simulation
        is_enable_debug_diagnostics: bool = False,  # Whether to emit expensive per-step diagnostic messages
        is_joint_task: bool = False,  # In testing mode, require all agents to complete the joint task before resetting
        joint_task_complete_duration: float = 12.0,  # Survival duration for loop-path joint tasks [s]
        is_save_simulation_video: bool = False,  # Whether to save simulation videos
        # extensions
        is_using_opponent_modeling: bool = False,  # Whether to use opponent modeling to predict the actions of other agents
        is_opponent_modeling_separate_predictor: bool = False,  # Whether to use a separate frozen opponent-prediction policy
        is_multi_timescale_learning: bool = False,  # Whether to use multi-timescale learning (only for motion policy)
        is_using_prioritized_marl: bool = False,  # Whether to use prioritized reward-propagation MARL.
        is_obs_vehicle_id_in_priority_policy: bool = False,  # Whether the priority policy observes normalized vehicle IDs.
        is_obs_vehicle_id_in_motion_policy: bool = False,  # Whether the motion policy observes normalized vehicle IDs.
        is_global_agent_id: bool = True,  # Whether observed vehicle IDs use global IDs instead of ego-relative rankings.
        is_priority_conditioned_motion: bool = False,  # Whether detached local priority features condition motion actions.
        execution_mode: str = "reward_propagation",
        prioritization_method: str = "marl",  # Which method to use for generating priority ranks (options: {"marl", "random", "fixed", "taskprogress", "taskprogress_permute"}; "taskprogress_permute" assigns dynamic route progress via an episode-fixed random agent permutation). Applicable only for prioritized MARL scenarios.
        random_priority_fixed_episode: int
        | None = None,  # Number of episodes for which random priorities remain fixed. None or 0 preserves per-step randomization.
        is_own_priority_obs: bool = False,
        is_communication_noise: str = False,  # Whether to inject communication noise to propagated actions
        communication_noise_level: float = 0.1,  # Defines the variance of the normal distribution modeling the noise.
        reward_distribution_alpha_start: float = 0.2,
        reward_distribution_alpha_end: float = 0.2,
        reward_distribution_mode: str = "linear",
        reward_distribution_alpha_coeff: float = 10.0,
        is_even_reward_distribution: bool = False,
        even_reward_group_threshold: float = 0.0,
        prioritized_reward_group_threshold: float = 0.0,
        is_weighted_reward_sharing: bool = False,
        priority_weight_beta: float = 1.0,
        is_using_cbf_testing: bool = False,  # Whether to use Control Barrier Function (CBF) during testing
        is_using_cbf_training: bool = False,  # Whether to use Control Barrier Function (CBF) during training
        is_using_centralized_cbf: bool = False,  # Whether to use centralized solving for CBF-constrained MARL
        is_apply_cbf_action: bool = False,  # Whether to apply CBF action (use when is_using_cbf_training is True, deciding if CBF action or RL action should be applied)
        is_use_sampled_lane_boundary_for_cbf: bool = True,  # Whether to use sampled lane boundary points as constraints for CBF (if False, the whole boundaries will be used)
        is_store_rl_action_under_cbf: bool = False,  # Shielded-PPO mode: execute the CBF (applied) action in the environment, but store the raw RL action + its log-prob in the transition so PPO learns from the policy's own action. Only meaningful with is_apply_cbf_action=True (else the stored action already equals the RL action).
        is_keep_nominal_when_safe: bool = False,  # CBF filter (centralized): when the min-deviation QP leaves an agent's action unchanged (its nominal already satisfied the CBF), keep the RL action verbatim instead of writing back the ~equal QP solution. Only agents the QP actually moved get overwritten. No-op unless is_apply_cbf_action=True.
        cbf_qp_failure_fallback: str = None,  # CBF-QP (ProxSuite): control of an agent whose QP is not solved (solver failure, iteration cap, or a solution outside the box constraints). One of {"nominal" (the unfiltered nominal control), "brake" (braking with route-tracking steering, the backup policy)}. None: "brake" with is_backup_horizon_cbf (required there), else "nominal"
        is_solve_qp: bool = True,  # Whether to solve QP (if False, use the nominal action to access the constraint violation/value)
        experiment_type: str = "simulation",  # One of {"simulation", "lab"}. If you only use simulation, you do not need to worry about "lab".
        physics_backend: str = "vmas",  # One of {"vmas", "lab", "mock"}. "vmas" integrates the kinematic bicycle model in-process (WorldCustom); "lab" delegates physics to the CPM lab over ROS 2 (WorldLab + LabBridge); "mock" uses the lab pipeline with an in-process loopback (WorldLab + MockBridge), mainly for tests.
        without_lanelet_reset: bool = False,  # Whether to disable lanelet respawn on agent reset (for lab/mock backends)
        is_fast_inference_mode: bool = False,  # Whether to enable fast inference mode (e.g. skip reward computation)
        lab_vehicle_ids: list = None,  # Lab vehicle IDs, one per agent (parallel to agent indices). Required for physics_backend="lab"; defaults to [1..n_agents] otherwise.
        lab_period_ms: int = 100,  # Period of the lab bridge's VehicleStateList publication in ms; must equal dt*1000 for lab/mock backends.
        lab_state_timeout: float = 30.0,  # [s] How long a lab world step waits for a fresh VehicleStateList before raising (generous to ride out mid-step respawns).
        lab_manual_reset: bool = False,  # Human-in-the-loop reset for the REAL lab (where vehicles cannot be teleported). On every reset the fleet is stopped, the operator places the vehicles by hand and presses ENTER, and reference paths are assigned from the MEASURED poses (never sampled). False keeps the sampled-pose /lab/respawn_vehicles teleport used by the digital twin and the mock. Only meaningful for physics_backend in {"lab", "mock"}.
        lab_mode: str = "speed_curvature",  # Lab control mode: one of {"direct", "path_tracking", "speed_curvature", "trajectory"}. "direct" and "path_tracking" are not implemented.
        lab_traj_horizon: int = 3,  # Horizon in steps for policy-generated virtual trajectory (relevant only for lab_mode="trajectory").
        is_obs_steering: bool = False,  # Whether to observe the steering angle of other agents
        predefined_ref_path_idx: list[
            int
        ] = None,  # A list of integers specify the index of the predefined reference path for each. They will be used when initializing/resetting the agents. Set to None to use a randomly selected ones.
        init_state: list[
            float
        ] = None,  # Initial state of the agents in the form of [x, y, rot, speed]. If None, the initial state will be randomly sampled.
        is_zero_initial_speed: bool = True,  # Whether to set the initial speed of the agents to zero (spawns are then checked as at rest, see spawn_clearances in world_state_rt_sim.py). If False, the initial speed is sampled in [0, 20% max speed].
        random_seed: int = 0,  # Random seed,
        n_circles_approximate_vehicle: int = 3,  # Number of circles to approximate the vehicle shape
        lane_width=0.30 * LENGTH_SCALE,  # For custom scenarios only
        is_sample_equal_distance: bool = True,  # Whether to resample CPM reference paths
        reset_agent_fixed_duration: int = 0,  # Reset agents after fixed duration in seconds. Set to 0 if not used.
        is_reset_all_agents_in_test_mode_when_collision: bool = True,  # Reset the whole testing sub-environment after an agent-agent or lanelet collision.
        reset_agent_surrounding_distance: float = None,  # Distance for collision-neighborhood resets; None selects an automatic geometry-based value, 0 disables the feature.
        is_grouping_agents: bool = False,  # Whether to use grouping agents
        max_group_size: int = 2,  # Maximum number of agents in a group
        cbf_coupling_range: float = None,  # Range within which other agents are included in CBF-QP coupling; None scales 1.0 * LENGTH_SCALE with the squared speed limit (see resolve_speed_defaults)
        grouping_method: str = "interaction_risk",  # {"interaction_risk", "k_nearest"}
        grouping_use_hysteresis: bool = True,  # Whether to bias grouping toward previous groups
        grouping_interaction_range: float = None,  # If None, use max(cbf_coupling_range, vehicle-size heuristic)
        grouping_hysteresis_bonus: float = 0.25,  # Pair-risk bonus for previously grouped nearby agents
        grouping_hysteresis_keep_range_factor: float = 1.25,  # Keep-bonus range multiplier
        grouping_risk_threshold: float = 0.0,  # Minimum risk for optional group expansion
        grouping_closing_speed_scale: float = None,  # Closing-speed normalizer for pair risk; None scales 0.5 * SPEED_SCALE with the speed limit (see resolve_speed_defaults)
        nom_controller_type: str = "rl",  # Type of the nominal controller: {"rl", "clf"}
        max_speed: float = None,  # [m/s] Speed limit of the agents; None uses the scenario's limit, else the agent profile's max_speed
        speed_normalization: float = None,  # [m/s] Speed scale of the policy: speeds in the observations are divided by it, and the speed action ranges over [-speed_normalization, speed_normalization] (the dynamics clip it to max_speed). A common value lets one policy run on maps with different speed limits. None: max_speed
        ref_speed: float = None,  # [m/s] Reference speed of the CLF nominal controller; None uses the scenario's, else 1.0 * SPEED_SCALE
        clf_nominal_mode: str = "pure_pursuit",  # CLF nominal steering law: {"pure_pursuit", "heading_rate"} (legacy: steering rate proportional to heading error)
        lane_hessian_mode: str = "geometric",  # Hessian of the lane-distance surrogate in the lane CBFs: {"geometric", "stencil"} (legacy finite differences, distorted by the softmin blending)
        cross_cbf_responsibility: str = "follower",  # Split of decentralized pair CBFs between the two agents: {"half", "follower"} (the agent driving toward the other takes the larger share)
        is_cbf_limit_acc_to_nominal: bool = True,  # Whether the CBF-QP may only brake relative to the nominal acceleration (never accelerate beyond max(nominal, 0))
        adaptive_lambda: bool = False,  # Whether to use adaptive lambda for CBF-QP
        is_aggregate_veh_veh_safety: bool = True,  # Whether to aggregate circle-pair vehicle CBF constraints per neighbor (veh_veh_cbf_footprint="circle" only)
        is_aggregate_lane_veh_safety: bool = False,  # Whether to aggregate lane CBF constraints over vehicle footprint circles
        aggregate_safety_softmin_beta: float = 400.0
        / LENGTH_SCALE**2,  # Softmin beta for aggregated safety margins in m^2
        veh_veh_cbf_footprint: str = "rectangle_sat",  # Vehicle-vehicle CBF footprint: covering "circle"s (see is_aggregate_veh_veh_safety), learned "mtv" safety margin, or exact rectangle pair "rectangle_sat" (needs lane_cbf_footprint="rectangle_sat")
        is_exact_backup_shield: bool = False,  # Exact backup-rollout shield after the CBF-QP, in all QP modes (rectangle_sat only)
        is_shield_priority: bool = True,  # Exact backup shield: resolve conflicts by priority (arrival order), so the agent that came first keeps its control, and let the centralized shield release yielded agents where the joint rollout allows it; False: the centralized shield brakes all violating agents, the decentralized one prioritizes lower indices
        is_standstill_escape: bool = False,  # Let vehicles stopped by their lane CBF steer at standstill and creep forward or backward out of it (rectangle_sat only)
        h_nom: float = H_NOM,  # Unitless factor of the CBF-margin reward normalizers (1: closing on a barrier at the speed limit gives -1), see CBFQP.cbf_margin_reward_scales
        cbf_margin_reward_condition: str = "backup_horizon",  # CBF condition whose nominal-action margins give the "cbfmargin" reward: "one_step" (two-step TTCBF condition) or "backup_horizon" (BH-TTCBF rows along the backup rollout, see CBFQP._update_nominal_bh_margin_rewards_all_envs; centralized CBF, rectangle_sat footprints, and is_backup_horizon_cbf only)
        rew_method: str = "ttc_sparse",  # Reward method: {"distance", "cbf", "ttc", "sparse", "distance_sparse", "cbf_sparse", "ttc_sparse"}
        is_schedule_rew_coefficient: bool = False,  # Whether to schedule reward and penalty coefficients during training
        reward_progress: float = 0.10,  # Reward for progress along the reference path
        threshold_near_boundary_high: float = 0.005 * LENGTH_SCALE,
        threshold_near_boundary_low: float = 0,
        threshold_near_other_agents_c2c_high: float = 0.2 * LENGTH_SCALE,
        threshold_near_other_agents_c2c_low: float = 0,
        ttc_low: float = 0,
        ttc_high: float = 5.0,
        penalty_near_boundary: float = -0.2,
        penalty_near_other_agents: float = -0.2,
        is_deadlock_enabled: bool = True,
        deadlock_duration: float = 3.0,
        deadlock_progress_threshold: float = None,
        penalty_deadlock: float = -1.0,
        penalty_near_collision_reset: float = -0.5,  # One-time penalty for a non-colliding agent reset because it is near a collision participant.
        is_learn_cbf_lambda: bool = False,  # Whether to learn the lambda of the CBF function
        cbf_lambda_rew_scale: float = 0.1,  # Reward scale for the CBF lambda (use when is_learn_cbf_lambda is True)
        cbf_lambda_rew_scale_abs: float = 1.0,  # Reward scale for the CBF lambda magnitude (use when is_learn_cbf_lambda is True)
        is_alternating_action_lambda_training: bool = False,  # Whether to use alternating training for motion and lambda
        alternating_train_action_iters: int = 50,  # Number of iterations to train the motion policy in each alternation (use when is_alternating_action_lambda_training is True)
        alternating_train_lambda_iters: int = 50,  # Number of iterations to train the lambda policy in each alternation (use when is_alternating_action_lambda_training is True)
        is_load_pretrained_motion_policy: bool = False,
        is_load_pretrained_lambda_policy: bool = False,
        pretrained_motion_policy_path: str = "",
        pretrained_lambda_policy_path: str = "",
        is_train_motion_policy: bool = True,
        is_train_lambda_policy: bool = False,
        cbf_lambda_head: str = "agent",  # Learned CBF lambda head: {"agent" (one lambda per agent; pair rows average the two), "pair" (one lambda per observed (ego, neighbor) pair; the nominal controller is a CLF or a frozen pretrained motion policy)}
        penalty_deviate_from_cbf_vel: float = -0.05,  # Penalty for deviating from the CBF-safe velocity
        penalty_deviate_from_cbf_steer: float = -0.05,  # Penalty for deviating from the CBF-safe steering
        penalty_jerk_acceleration: float = -0.02,  # Penalty for quick acceleration changes
        penalty_jerk_steering_rate: float = -0.02,  # Penalty for quick steering-rate changes
        penalty_high_acceleration: float = -0.02,  # Penalty near acceleration limits
        penalty_high_steering_rate: float = -0.02,  # Penalty near steering-rate limits
        high_control_threshold: float = 0.8,  # Fraction of control limit where high-control penalty starts
        lane_distance_type: str = "softmin",  # One of pseudo, softmin. Used only by the "circle" and "rectangle" lane-CBF footprints; ignored by "rectangle_sat"
        lane_cbf_footprint: str = "rectangle_sat",  # Lane-CBF footprint: "circle", certified "rectangle" (needs softmin), or exact polyline "rectangle_sat"
        cbf_lambda: float = None,  # Fixed decay rate lambda in (0, 1] of the CBF conditions h(x_{k+2}) >= (1 - lambda) h(x_{k+1}) (unless lambda is learned); higher is more aggressive. None: the tuned value of the method (sigmarl/cbf_baselines/tuned_defaults.json, see tuned_cbf_lambda): BH-TTCBF, the one-step condition alone with the rectangle_sat footprints, or the selected SOTA baseline; else 0.5 (B6, other footprints)
        lane_cbf_baseline: str = "none",  # SOTA baseline that replaces the rectangle_sat lane rows (comparison only, see sigmarl/cbf_baselines): "none", "unfiltered" (no rows), "b1_lse_sat", "b2_frenet_point", "b2_frenet_heading", "b5_convex_containment", or "t2_dual_distance"
        veh_veh_cbf_baseline: str = "none",  # SOTA baseline that replaces the rectangle_sat pair rows (comparison only, see sigmarl/cbf_baselines): "none", "unfiltered" (no rows), "b1_lse_sat", "b3_disk", "b4_collision_cone", or "t2_dual_distance"
        is_backup_horizon_cbf: bool = True,  # Backup-horizon TTCBF, the proposed CBF-QP condition: every rectangle_sat lane and pair constraint must stay safe along the backup rollout after the QP control (sigmarl/backup_horizon_cbf.py; recursive feasibility in the QP). Needs the rectangle_sat footprints, the ProxSuite QP, a fixed lambda, no CBF baseline, and the "brake" failure fallback; set False for the two-step condition alone, the CBF baselines, other footprints, or a learned lambda
        is_two_step_rows: bool = False,  # With is_backup_horizon_cbf: also keep the two-step rectangle_sat lane and pair rows in the QP (ablation); by default only the backup-horizon tail rows. Without is_backup_horizon_cbf, the two-step rows are the CBF condition and always used
        condition_cbf_baseline: str = "none",  # SOTA baseline that replaces the condition of the backup-horizon rows (comparison only, needs is_backup_horizon_cbf): "none" (ours) or "b6_backup_cbf" (per-flow-point decay, sigmarl/cbf_baselines/b6_backup_cbf.py)
        rl_action_type: str = "speed_steer",  # "speed_steer", "position_speed", or "speed"
        lookahead_dist: float = POSITION_SPEED_LOOKAHEAD_DIST,  # [m] pure-pursuit lookahead distance (position_speed mode only)
        is_proxsuite_qp: bool = True,  # Whether to use ProxSuite for solving the CBF-QP
        is_log_cbf_boundary_distance: bool = False,  # Whether to log the distance to the boundary defined by the CBF constraint (for analysis of safety margin)
        is_centralized_critic: bool = True,  # Whether to use a centralized critic for MARL
        is_share_params_actor: bool = True,  # Whether to share parameters between agents
        scale_map: float = 1.0,  # Scale OSM map coordinates around their global center
        parameter_independent_reward_intermediate: float = None,  # Parameter-independent reward corresponding to the best intermediate model
        is_center_centerline: bool = True,  # Whether to center reference paths between shared boundaries
    ):

        self.n_agents = n_agents
        self.dt = dt

        self.device = device
        self.scenario_name = scenario_name

        # Sampling
        self.n_iters = n_iters

        # Training
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.lr_min = lr_min
        self.lr_slow = lr_slow
        self.lr_slow_min = lr_slow_min
        self.multi_timescale_switching_period = multi_timescale_switching_period
        self.lr_priority = lr_priority
        self.lr_priority_min = lr_priority_min
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.entropy_eps_priority = entropy_eps_priority
        self.max_steps = max_steps

        self.scenario_type = scenario_type
        self.agent_profile = agent_profile

        self.num_vmas_envs = num_vmas_envs

        self.is_save_intermediate_model = is_save_intermediate_model
        self.is_load_model = is_load_model
        self.is_load_final_model = is_load_final_model

        self.episode_reward_mean_current = episode_reward_mean_current
        self.episode_reward_intermediate = episode_reward_intermediate
        self.parameter_independent_reward_intermediate = (
            parameter_independent_reward_intermediate
        )
        self.where_to_save = where_to_save
        self.is_continue_train = is_continue_train
        if must_save_checkpoints is None:
            must_save_checkpoints = []
        self.must_save_checkpoints = list(must_save_checkpoints)
        self.is_cross_iter_test = bool(is_cross_iter_test)

        self.n_points_short_term = n_points_short_term
        self.short_term_ref_path_spacing_m = short_term_ref_path_spacing_m
        self.n_points_nearing_boundary = n_points_nearing_boundary
        self.is_smooth_sampling = is_smooth_sampling
        self.is_use_simple_road_distance = is_use_simple_road_distance
        # Observation
        self.is_partial_observation = is_partial_observation
        self.n_steps_stored = n_steps_stored
        self.n_nearing_agents_observed = n_nearing_agents_observed
        self.is_observe_distance_to_agents = is_observe_distance_to_agents

        self.is_testing_mode = is_testing_mode
        self.is_enable_debug_diagnostics = is_enable_debug_diagnostics
        self.is_joint_task = is_joint_task
        if joint_task_complete_duration < 0.0:
            raise ValueError("`joint_task_complete_duration` must be non-negative.")
        self.joint_task_complete_duration = float(joint_task_complete_duration)
        self.is_save_simulation_video = is_save_simulation_video
        self.is_visualize_short_term_path = is_visualize_short_term_path
        self.is_visualize_lane_boundary = is_visualize_lane_boundary
        self.is_visu_sampled_boundary_points = is_visu_sampled_boundary_points
        self.is_visu_applied_action = is_visu_applied_action
        self.is_visu_nom_action = is_visu_nom_action
        self.is_visu_shield_rollout = is_visu_shield_rollout
        self.is_visu_driving_direction = is_visu_driving_direction
        self.is_visu_veh_direction = is_visu_veh_direction

        self.is_ego_view = is_ego_view
        self.is_apply_distance_mask = is_apply_distance_mask
        self.is_apply_velocity_mask = is_apply_velocity_mask
        self.velocity_mask_margin = velocity_mask_margin
        self.rl_obs_range = rl_obs_range
        self.visible_agent_dropout_probability = visible_agent_dropout_probability
        self.is_use_mtv_distance = is_use_mtv_distance
        self.is_observe_distance_to_boundaries = is_observe_distance_to_boundaries
        self.is_observe_boundary_points = is_observe_boundary_points
        self.is_observe_distance_to_center_line = is_observe_distance_to_center_line
        self.is_observe_vertices = is_observe_vertices
        self.is_obs_noise = is_obs_noise
        self.is_add_noise = is_obs_noise  # Backward-compatible copied-scenario alias
        self.obs_noise_level = obs_noise_level
        self.is_observe_ref_path_other_agents = is_observe_ref_path_other_agents

        self.is_save_eval_results = is_save_eval_results
        self.is_load_out_td = is_load_out_td
        self.is_save_detailed_info = is_save_detailed_info

        self.is_real_time_rendering = is_real_time_rendering
        self.is_visualize_extra_info = is_visualize_extra_info
        self.is_visualize_observed_agents = is_visualize_observed_agents
        self.observed_agents_render_ego_index = observed_agents_render_ego_index
        self.cross_iter_ego_indices = (
            tuple(cross_iter_ego_indices)
            if cross_iter_ego_indices is not None
            else None
        )
        self.render_title = render_title

        self.is_prb = is_prb
        self.reset_at_each_iter = reset_at_each_iter
        self.is_challenging_initial_state_buffer = is_challenging_initial_state_buffer

        self.cpm_scenario_probabilities = cpm_scenario_probabilities

        self.is_using_opponent_modeling = is_using_opponent_modeling
        self.is_opponent_modeling_separate_predictor = (
            is_opponent_modeling_separate_predictor
        )
        self.is_multi_timescale_learning = is_multi_timescale_learning
        self.is_using_prioritized_marl = is_using_prioritized_marl
        self.is_obs_vehicle_id_in_priority_policy = is_obs_vehicle_id_in_priority_policy
        self.is_obs_vehicle_id_in_motion_policy = is_obs_vehicle_id_in_motion_policy
        self.is_global_agent_id = is_global_agent_id
        self.is_priority_conditioned_motion = is_priority_conditioned_motion
        if self.is_priority_conditioned_motion and not self.is_using_prioritized_marl:
            raise ValueError(
                "`is_priority_conditioned_motion=True` requires "
                "`is_using_prioritized_marl=True`."
            )
        if self.is_priority_conditioned_motion and not self.is_partial_observation:
            raise ValueError(
                "`is_priority_conditioned_motion=True` currently requires "
                "`is_partial_observation=True`."
            )
        self.execution_mode = execution_mode.lower()
        if self.execution_mode != "reward_propagation":
            raise ValueError(
                "`execution_mode` must be 'reward_propagation' in this implementation."
            )
        self.prioritization_method = prioritization_method.lower()
        if self.prioritization_method not in {
            "marl",
            "random",
            "fixed",
            "taskprogress",
            "taskprogress_permute",
        }:
            raise ValueError(
                "`prioritization_method` must be 'marl', 'random', 'fixed', 'taskprogress', or 'taskprogress_permute'."
            )
        if random_priority_fixed_episode is None:
            self.random_priority_fixed_episode = None
        else:
            if (
                isinstance(random_priority_fixed_episode, bool)
                or int(random_priority_fixed_episode) != random_priority_fixed_episode
                or random_priority_fixed_episode < 0
            ):
                raise ValueError(
                    "`random_priority_fixed_episode` must be None or a non-negative integer."
                )
            self.random_priority_fixed_episode = int(random_priority_fixed_episode)
        if self.is_multi_timescale_learning:
            self.multi_timescale_switching_period = int(
                self.multi_timescale_switching_period
            )
            if self.multi_timescale_switching_period <= 0:
                raise ValueError("multi_timescale_switching_period must be positive.")
            if self.lr_slow < 0.0 or self.lr_slow_min < 0.0:
                raise ValueError(
                    "Slow multi-timescale learning rates must be non-negative."
                )
            if self.lr_slow_min > self.lr_slow:
                raise ValueError("lr_slow_min must not exceed lr_slow.")
            if self.is_using_prioritized_marl:
                raise ValueError(
                    "Multi-timescale learning is motion-only and cannot use prioritized MARL."
                )
            if self.is_priority_conditioned_motion:
                raise ValueError(
                    "Multi-timescale learning cannot use priority-conditioned motion."
                )
            if is_share_params_actor:
                raise ValueError(
                    "Multi-timescale learning requires is_share_params_actor=False."
                )
            if not is_train_motion_policy or is_train_lambda_policy:
                raise ValueError(
                    "Multi-timescale learning requires motion-only policy training."
                )
        if is_own_priority_obs:
            raise ValueError("`is_own_priority_obs=True` is no longer supported.")
        self.is_own_priority_obs = False
        self.is_communication_noise = is_communication_noise
        self.communication_noise_level = communication_noise_level

        self.reward_distribution_mode = reward_distribution_mode.lower()
        if self.reward_distribution_mode not in {"linear", "adaptive"}:
            raise ValueError(
                "`reward_distribution_mode` must be 'linear' or 'adaptive'."
            )
        if not 0.0 <= reward_distribution_alpha_start <= 1.0:
            raise ValueError("`reward_distribution_alpha_start` must be in [0, 1].")
        if not 0.0 <= reward_distribution_alpha_end <= 1.0:
            raise ValueError("`reward_distribution_alpha_end` must be in [0, 1].")
        if reward_distribution_alpha_coeff < 0.0:
            raise ValueError("`reward_distribution_alpha_coeff` must be non-negative.")
        if even_reward_group_threshold < 0.0:
            raise ValueError("`even_reward_group_threshold` must be non-negative.")
        if prioritized_reward_group_threshold < 0.0:
            raise ValueError(
                "`prioritized_reward_group_threshold` must be non-negative."
            )
        if priority_weight_beta <= 0.0:
            raise ValueError("`priority_weight_beta` must be positive.")
        self.reward_distribution_alpha_start = reward_distribution_alpha_start
        self.reward_distribution_alpha_end = reward_distribution_alpha_end
        self.reward_distribution_alpha = reward_distribution_alpha_start
        self.reward_distribution_alpha_coeff = reward_distribution_alpha_coeff
        self.is_even_reward_distribution = is_even_reward_distribution
        self.even_reward_group_threshold = even_reward_group_threshold
        self.prioritized_reward_group_threshold = prioritized_reward_group_threshold
        self.is_weighted_reward_sharing = is_weighted_reward_sharing
        self.priority_weight_beta = priority_weight_beta

        self.is_using_cbf_testing = is_using_cbf_testing
        self.is_using_cbf_training = is_using_cbf_training
        self.is_using_centralized_cbf = is_using_centralized_cbf
        self.is_apply_cbf_action = is_apply_cbf_action
        self.is_use_sampled_lane_boundary_for_cbf = is_use_sampled_lane_boundary_for_cbf
        self.is_store_rl_action_under_cbf = is_store_rl_action_under_cbf
        self.is_keep_nominal_when_safe = is_keep_nominal_when_safe
        assert cbf_qp_failure_fallback in (
            None,
            "nominal",
            "brake",
        ), f"Unknown cbf_qp_failure_fallback '{cbf_qp_failure_fallback}'."
        self.cbf_qp_failure_fallback = cbf_qp_failure_fallback
        self.is_solve_qp = is_solve_qp

        self.experiment_type = experiment_type
        self.physics_backend = physics_backend
        self.without_lanelet_reset = without_lanelet_reset
        self.is_fast_inference_mode = is_fast_inference_mode
        self.lab_vehicle_ids = lab_vehicle_ids
        self.lab_period_ms = lab_period_ms
        self.lab_state_timeout = lab_state_timeout
        self.lab_manual_reset = lab_manual_reset
        if lab_mode in ["direct", "path_tracking"]:
            raise NotImplementedError(
                f"lab_mode '{lab_mode}' is not implemented yet. Supported modes are 'speed_curvature' and 'trajectory'."
            )
        if lab_mode not in ["speed_curvature", "trajectory"]:
            raise ValueError(
                f"Unknown lab_mode '{lab_mode}'. Expected one of: 'direct', 'path_tracking', 'speed_curvature', 'trajectory'."
            )
        if lab_traj_horizon < 1:
            raise ValueError(f"lab_traj_horizon must be >= 1, got {lab_traj_horizon}.")
        self.lab_mode = lab_mode
        self.lab_traj_horizon = lab_traj_horizon
        self.is_obs_steering = is_obs_steering

        self.predefined_ref_path_idx = predefined_ref_path_idx
        self.init_state = init_state
        self.is_zero_initial_speed = is_zero_initial_speed

        self.random_seed = random_seed

        self.n_circles_approximate_vehicle = n_circles_approximate_vehicle

        self.lane_width = lane_width
        self.is_sample_equal_distance = is_sample_equal_distance
        if not math.isfinite(scale_map) or scale_map <= 0.0:
            raise ValueError("`scale_map` must be a finite number greater than 0.")
        self.scale_map = float(scale_map)
        self.is_center_centerline = is_center_centerline

        self.reset_agent_fixed_duration = reset_agent_fixed_duration
        self.is_reset_all_agents_in_test_mode_when_collision = (
            is_reset_all_agents_in_test_mode_when_collision
        )
        if reset_agent_surrounding_distance is not None and (
            reset_agent_surrounding_distance < 0
        ):
            raise ValueError(
                "`reset_agent_surrounding_distance` must be non-negative or None."
            )
        self.reset_agent_surrounding_distance = reset_agent_surrounding_distance
        self.is_grouping_agents = is_grouping_agents
        self.max_group_size = max_group_size
        self.cbf_coupling_range = cbf_coupling_range
        self.grouping_method = grouping_method
        self.grouping_use_hysteresis = grouping_use_hysteresis
        self.grouping_interaction_range = grouping_interaction_range
        self.grouping_hysteresis_bonus = grouping_hysteresis_bonus
        self.grouping_hysteresis_keep_range_factor = (
            grouping_hysteresis_keep_range_factor
        )
        self.grouping_risk_threshold = grouping_risk_threshold
        self.grouping_closing_speed_scale = grouping_closing_speed_scale
        self.penalty_near_collision_reset = penalty_near_collision_reset
        self.nom_controller_type = nom_controller_type
        self.adaptive_lambda = adaptive_lambda
        self.is_aggregate_veh_veh_safety = is_aggregate_veh_veh_safety
        self.is_aggregate_lane_veh_safety = is_aggregate_lane_veh_safety
        self.aggregate_safety_softmin_beta = aggregate_safety_softmin_beta
        self.veh_veh_cbf_footprint = veh_veh_cbf_footprint
        self.is_exact_backup_shield = is_exact_backup_shield
        self.is_shield_priority = is_shield_priority
        self.is_standstill_escape = is_standstill_escape
        self.max_speed = max_speed
        self.speed_normalization = speed_normalization
        self.ref_speed = ref_speed
        self.clf_nominal_mode = clf_nominal_mode
        self.is_cbf_limit_acc_to_nominal = is_cbf_limit_acc_to_nominal
        self.lane_hessian_mode = lane_hessian_mode
        self.cross_cbf_responsibility = cross_cbf_responsibility
        self.h_nom = h_nom
        self.cbf_margin_reward_condition = cbf_margin_reward_condition
        self.rew_method = rew_method
        self.is_schedule_rew_coefficient = is_schedule_rew_coefficient
        self.reward_progress = reward_progress

        self.threshold_near_boundary_high = threshold_near_boundary_high
        self.threshold_near_boundary_low = threshold_near_boundary_low
        self.threshold_near_other_agents_c2c_high = threshold_near_other_agents_c2c_high
        self.threshold_near_other_agents_c2c_low = threshold_near_other_agents_c2c_low
        self.ttc_high = ttc_high
        self.ttc_low = ttc_low
        self.penalty_near_boundary = penalty_near_boundary
        self.penalty_near_other_agents = penalty_near_other_agents
        self.is_deadlock_enabled = is_deadlock_enabled
        self.deadlock_duration = deadlock_duration
        self.deadlock_progress_threshold = deadlock_progress_threshold
        self.penalty_deadlock = penalty_deadlock
        self.is_learn_cbf_lambda = is_learn_cbf_lambda
        self.cbf_lambda_rew_scale = cbf_lambda_rew_scale
        self.cbf_lambda_rew_scale_abs = cbf_lambda_rew_scale_abs

        self.is_alternating_action_lambda_training = (
            is_alternating_action_lambda_training
        )
        self.alternating_train_action_iters = alternating_train_action_iters
        self.alternating_train_lambda_iters = alternating_train_lambda_iters

        self.is_load_pretrained_motion_policy = is_load_pretrained_motion_policy
        self.is_load_pretrained_lambda_policy = is_load_pretrained_lambda_policy
        self.pretrained_motion_policy_path = pretrained_motion_policy_path
        self.pretrained_lambda_policy_path = pretrained_lambda_policy_path
        self.is_train_motion_policy = is_train_motion_policy
        self.is_train_lambda_policy = is_train_lambda_policy
        self.cbf_lambda_head = cbf_lambda_head
        self.penalty_deviate_from_cbf_vel = penalty_deviate_from_cbf_vel
        self.penalty_deviate_from_cbf_steer = penalty_deviate_from_cbf_steer
        self.penalty_jerk_acceleration = penalty_jerk_acceleration
        self.penalty_jerk_steering_rate = penalty_jerk_steering_rate
        self.penalty_high_acceleration = penalty_high_acceleration
        self.penalty_high_steering_rate = penalty_high_steering_rate
        self.high_control_threshold = high_control_threshold
        self.lane_distance_type = lane_distance_type
        self.lane_cbf_footprint = lane_cbf_footprint
        self.cbf_lambda = cbf_lambda
        self.lane_cbf_baseline = lane_cbf_baseline
        self.veh_veh_cbf_baseline = veh_veh_cbf_baseline
        self.is_backup_horizon_cbf = is_backup_horizon_cbf
        self.is_two_step_rows = is_two_step_rows
        self.condition_cbf_baseline = condition_cbf_baseline
        self.rl_action_type = rl_action_type
        self.lookahead_dist = lookahead_dist
        self.is_proxsuite_qp = is_proxsuite_qp
        self.is_log_cbf_boundary_distance = is_log_cbf_boundary_distance

        self.is_centralized_critic = is_centralized_critic
        self.is_share_params_actor = is_share_params_actor

        if (model_name is None) and (scenario_name is not None):
            self.model_name = get_model_name(self)

    @property
    def frames_per_batch(self):
        """
        Number of team frames collected per training iteration
            num_envs = frames_per_batch / max_steps
            total_frames = frames_per_batch * n_iters
            sub_batch_size = frames_per_batch // minibatch_size
        """
        return self.num_vmas_envs * self.max_steps

    @property
    def total_frames(self):
        """
        Total number of frames collected for training
        """
        return self.frames_per_batch * self.n_iters

    def to_dict(self):
        # Create a dictionary representation of the instance
        return self.__dict__

    @classmethod
    def from_dict(cls, dict_data):
        """
        Create an instance of the class from a dictionary.
        """
        code = cls.__init__.__code__
        valid_keys = set(
            code.co_varnames[1 : code.co_argcount + code.co_kwonlyargcount]
        )
        filtered_data = {
            key: value for key, value in dict_data.items() if key in valid_keys
        }
        # Legacy flag, replaced by veh_veh_cbf_footprint
        if "veh_veh_cbf_footprint" not in dict_data and dict_data.get(
            "is_use_mtv_for_veh_veh", False
        ):
            filtered_data["veh_veh_cbf_footprint"] = "mtv"
        return cls(**filtered_data)

    @classmethod
    def from_json(cls, config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
            return cls.from_dict(config)

    def resolve_speed_defaults(self):
        """Fill the speed-dependent parameters left as None from the scenario.

        The scenario sets the speed limit `max_speed` of its road (e.g., highways
        are faster than roundabouts), falling back to the agent profile's. The
        defaults of `rl_obs_range`, `cbf_coupling_range`, and
        `grouping_closing_speed_scale` were tuned at the profile's max_speed;
        both ranges grow with the squared speed like the braking distance, the
        closing-speed scale linearly, and none shrinks below its tuned value.
        The CBF coupling range and the closing-speed scale follow the speed limit
        `max_speed`; the RL observation range, an input of the policy, follows
        the policy's speed scale `speed_normalization` (default: max_speed), so
        that it does not change with the map. Values already set, e.g., by the
        command line or a checkpoint, are kept (see retarget_scenario).
        """
        validate_scenario_agent_profile(self.scenario_type)
        scenario = SCENARIOS[self.scenario_type]
        scenario_max_speed = scenario.get("max_speed", AGENTS["max_speed"])
        if self.max_speed is None:
            self.max_speed = scenario_max_speed
        elif self.max_speed != scenario_max_speed:
            # Observations and rewards are normalized by max_speed, so a policy
            # trained under one speed limit sees differently scaled inputs under
            # another. Training one policy across maps with different speed
            # limits is not supported yet.
            warnings.warn(
                f"max_speed = {self.max_speed:.2f} m/s differs from the speed limit "
                f"{scenario_max_speed:.2f} m/s of scenario '{self.scenario_type}'. "
                "Policies are not trained across different speed limits.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.max_speed <= 0.0:
            raise ValueError(f"max_speed = {self.max_speed} m/s must be positive.")
        if self.speed_normalization is None:
            self.speed_normalization = self.max_speed
        if self.speed_normalization < self.max_speed:
            warnings.warn(
                f"speed_normalization = {self.speed_normalization:.2f} m/s is below "
                f"the speed limit {self.max_speed:.2f} m/s: the policy cannot "
                "command speeds above speed_normalization.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.ref_speed is None:
            self.ref_speed = scenario.get("ref_speed", 1.0 * SPEED_SCALE)
        if self.ref_speed > self.max_speed:
            warnings.warn(
                f"ref_speed = {self.ref_speed} m/s exceeds max_speed = "
                f"{self.max_speed} m/s.",
                RuntimeWarning,
                stacklevel=2,
            )
        speed_ratio = max(1.0, self.max_speed / AGENTS["max_speed"])
        if self.rl_obs_range is None:
            policy_ratio = max(1.0, self.speed_normalization / AGENTS["max_speed"])
            self.rl_obs_range = 1.0 * LENGTH_SCALE * policy_ratio**2
        if self.cbf_coupling_range is None:
            self.cbf_coupling_range = 1.0 * LENGTH_SCALE * speed_ratio**2
        if self.grouping_closing_speed_scale is None:
            self.grouping_closing_speed_scale = 0.5 * SPEED_SCALE * speed_ratio

    def retarget_scenario(self, scenario_type: str):
        """Move the (checkpoint) parameters of a policy to another map: the speed
        limit and the speed-dependent ranges of the CBF and the grouping are
        resolved again from the new map (see resolve_speed_defaults), while the
        policy's speed scale and observation range are kept. A checkpoint from
        before speed_normalization keeps its speed limit as its speed scale.
        """
        if self.speed_normalization is None:
            self.speed_normalization = self.max_speed
        self.scenario_type = scenario_type
        self.max_speed = None
        self.ref_speed = None
        self.cbf_coupling_range = None
        self.grouping_closing_speed_scale = None

    def update_reward_distribution_alpha(self, progress: float):
        """Update the current redistribution factor for progress in ``[0, 1]``."""

        if not 0.0 <= progress <= 1.0:
            raise ValueError("`progress` must be in [0, 1].")
        if self.reward_distribution_mode == "linear":
            self.reward_distribution_alpha = (
                self.reward_distribution_alpha_start
                + (
                    self.reward_distribution_alpha_end
                    - self.reward_distribution_alpha_start
                )
                * progress
            )
        else:
            self.reward_distribution_alpha = self.reward_distribution_alpha_start

    def update_reward_distribution_alpha_after_iteration(
        self, completed_iterations: int
    ):
        """Set alpha for the next batch after an iteration has completed."""

        progress = min(
            max(completed_iterations, 0) / max(self.n_iters - 1, 1),
            1.0,
        )
        self.update_reward_distribution_alpha(progress)

    def validate_policy_training_case(self):
        if self.is_lambda_only:
            # if self.nom_controller_type != "clf":
            #     raise ValueError(
            #         "lambda_only requires nom_controller_type='clf' because there is "
            #         "no motion policy to supply RL nominal actions."
            #     )
            # if not self.is_using_cbf_training:
            #     raise ValueError(
            #         "lambda_only requires is_using_cbf_training=True so the learned "
            #         "lambda actually affects the rollout."
            #     )
            if not self.is_apply_cbf_action:
                raise ValueError(
                    "lambda_only requires is_apply_cbf_action=True so the CBF/CLF "
                    "action is applied to the environment."
                )
            if not self.is_solve_qp:
                raise ValueError(
                    "lambda_only requires is_solve_qp=True so the learned lambda "
                    "affects the solved safe action."
                )

        if self.cbf_lambda_head not in ("agent", "pair"):
            raise ValueError(
                f"cbf_lambda_head must be 'agent' or 'pair', got {self.cbf_lambda_head!r}."
            )
        if self.cbf_lambda_head == "pair":
            self._validate_pair_lambda_head()

        if self.is_no_policy_training:
            if self.nom_controller_type != "clf":
                raise ValueError(
                    "policy_training_case='none' requires nom_controller_type='clf' "
                    "because no learned motion policy will provide nominal actions."
                )
            if self.is_continue_train:
                raise ValueError(
                    "policy_training_case='none' cannot be used with "
                    "is_continue_train=True because no policy is being optimized."
                )
            if (
                self.is_load_pretrained_motion_policy
                or self.is_load_pretrained_lambda_policy
            ):
                raise ValueError(
                    "policy_training_case='none' is incompatible with loading "
                    "pretrained policy modules."
                )
            if self.is_learn_cbf_lambda:
                raise ValueError(
                    "policy_training_case='none' requires is_learn_cbf_lambda=False."
                )
            if not (self.is_using_cbf_training or self.is_using_cbf_testing):
                raise ValueError(
                    "policy_training_case='none' requires CBF rollout to be enabled "
                    "because no policy-generated action will exist otherwise."
                )
            if not self.is_apply_cbf_action:
                raise ValueError(
                    "policy_training_case='none' requires is_apply_cbf_action=True so "
                    "the CLF/CBF action is written back to the environment."
                )
            if not self.is_solve_qp:
                raise ValueError(
                    "policy_training_case='none' requires is_solve_qp=True so the "
                    "safe action is explicitly solved and written back."
                )

        if self.is_joint_training:
            if self.nom_controller_type != "rl":
                raise ValueError(
                    "joint training requires nom_controller_type='rl' so the motion "
                    "policy action is the nominal action corrected by the CBF-QP."
                )

        if "cbf" in self.rew_method:
            if not self.is_using_cbf_training and not self.is_using_cbf_testing:
                raise ValueError(
                    "Using CBF-related reward requires is_using_cbf_training=True."
                )
        if (
            "cbfmargin" in self.rew_method
            and self.is_continue_train
            and self.is_solve_qp
        ):
            raise ValueError(
                "rew_method='cbfmargin' requires is_solve_qp=False: the nominal "
                "CBF margins are computed only when no CBF-QP is solved, so the "
                "margin reward would stay zero."
            )
        if self.cbf_margin_reward_condition not in {"one_step", "backup_horizon"}:
            raise ValueError(
                "`cbf_margin_reward_condition` must be 'one_step' or 'backup_horizon'."
            )
        if self.cbf_margin_reward_condition == "backup_horizon" and (
            "cbfmargin" in self.rew_method
        ):
            if not self.is_using_centralized_cbf:
                raise ValueError(
                    "cbf_margin_reward_condition='backup_horizon' supports only "
                    "the centralized CBF (is_using_centralized_cbf=True)."
                )
            if (
                not self.is_backup_horizon_cbf
                or self.lane_cbf_footprint != "rectangle_sat"
                or self.veh_veh_cbf_footprint != "rectangle_sat"
                or self.is_learn_cbf_lambda
            ):
                raise ValueError(
                    "cbf_margin_reward_condition='backup_horizon' needs "
                    "is_backup_horizon_cbf=True, the rectangle_sat lane and "
                    "vehicle-vehicle footprints, and a fixed lambda."
                )

        if self.is_store_rl_action_under_cbf:
            if not self.is_using_cbf_training:
                raise ValueError(
                    "is_store_rl_action_under_cbf=True requires is_using_cbf_training=True "
                    "so the CBF-constrained collector runs and caches an RL action to store; "
                    "otherwise the flag silently does nothing."
                )
            if not self.is_apply_cbf_action:
                raise ValueError(
                    "is_store_rl_action_under_cbf=True requires is_apply_cbf_action=True; "
                    "without it the CBF action is never applied, so the stored action "
                    "already equals the RL action and the flag is a no-op."
                )

        if (
            self.is_grouping_agents
            and (self.max_group_size != 1)
            and (self.is_using_cbf_training or self.is_using_cbf_testing)
        ):
            if not self.is_using_centralized_cbf:
                raise ValueError(
                    "Grouping agents requires is_using_centralized_cbf to be True."
                )

        return self

    def _validate_pair_lambda_head(self):
        """Check the settings the pair-level CBF lambda head supports.

        The head trains only the lambda policy on top of a CLF nominal
        controller or a frozen pretrained motion policy, reads per-neighbor
        blocks from the ego-view partial observation, and feeds the centralized
        ProxSuite CBF-QP.
        """
        requirements = {
            "is_learn_cbf_lambda=True": self.is_learn_cbf_lambda,
            "lambda-only training (is_train_motion_policy=False, "
            "is_train_lambda_policy=True)": self.is_lambda_only,
            "nom_controller_type='clf', or 'rl' with is_load_pretrained_motion_policy=True": (
                self.nom_controller_type == "clf"
                or (
                    self.nom_controller_type == "rl"
                    and self.is_load_pretrained_motion_policy
                )
            ),
            "is_load_pretrained_motion_policy=False with nom_controller_type='clf'": (
                self.nom_controller_type != "clf"
                or not self.is_load_pretrained_motion_policy
            ),
            "is_using_centralized_cbf=True": self.is_using_centralized_cbf,
            "is_grouping_agents=False": not self.is_grouping_agents,
            "is_proxsuite_qp=True": self.is_proxsuite_qp,
            "is_ego_view=True": self.is_ego_view,
            "is_partial_observation=True": self.is_partial_observation,
            "is_using_opponent_modeling=False": not self.is_using_opponent_modeling,
            "is_using_prioritized_marl=False": not self.is_using_prioritized_marl,
            "is_obs_vehicle_id_in_motion_policy=False": (
                not self.is_obs_vehicle_id_in_motion_policy
            ),
            "'cbflambda' not in rew_method": "cbflambda" not in self.rew_method,
        }
        missing = [name for name, is_met in requirements.items() if not is_met]
        if missing:
            raise ValueError(
                "cbf_lambda_head='pair' requires: " + "; ".join(missing) + "."
            )

    @property
    def policy_training_case(self) -> str:
        if self.is_train_motion_policy and self.is_train_lambda_policy:
            return "joint"
        if self.is_train_motion_policy and not self.is_train_lambda_policy:
            return "motion_only"
        if not self.is_train_motion_policy and self.is_train_lambda_policy:
            return "lambda_only"
        return "none"

    @property
    def is_motion_only(self) -> bool:
        return self.policy_training_case == "motion_only"

    @property
    def is_lambda_only(self) -> bool:
        return self.policy_training_case == "lambda_only"

    @property
    def is_joint_training(self) -> bool:
        return self.policy_training_case == "joint"

    @property
    def is_no_policy_training(self) -> bool:
        return self.policy_training_case == "none"

    @property
    def uses_motion_policy(self) -> bool:
        return self.policy_training_case in ("motion_only", "joint")

    @property
    def uses_lambda_policy(self) -> bool:
        return self.policy_training_case in ("lambda_only", "joint")


class VehicleState(AgentState):
    def __init__(self):
        """
        Initialize the VehicleState with all attributes from AgentState
        and add some new attributes.
        """
        super().__init__()
        self._speed = None  # Speed (magnitude, positive for forward moving, negative for backward moving)
        self._steering = None
        self._sideslip_angle = None
        self._acceleration = None
        self._steering_rate = None

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._speed = value.to(self._device)

    @property
    def steering(self):
        return self._steering

    @steering.setter
    def steering(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._steering = value.to(self._device)

    @property
    def sideslip_angle(self):
        return self._sideslip_angle

    @sideslip_angle.setter
    def sideslip_angle(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._sideslip_angle = value.to(self._device)

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._acceleration = value.to(self._device)

    @property
    def steering_rate(self):
        return self._steering_rate

    @steering_rate.setter
    def steering_rate(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._steering_rate = value.to(self._device)

    @override(AgentState)
    def _reset(self, env_index: typing.Optional[int]):
        for attr_name in [
            "speed",
            "steering",
            "sideslip_angle",
            "acceleration",
            "steering_rate",
        ]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                if env_index is None:
                    self.__setattr__(attr_name, torch.zeros_like(attr))
                else:
                    self.__setattr__(
                        attr_name,
                        TorchUtils.where_from_index(env_index, 0, attr),
                    )
        super()._reset(env_index)

    @override(AgentState)
    def zero_grad(self):
        for attr_name in [
            "speed",
            "steering",
            "sideslip_angle",
            "acceleration",
            "steering_rate",
        ]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                self.__setattr__(attr_name, attr.detach())
        super().zero_grad()

    @override(AgentState)
    def _spawn(self, dim_c: int, dim_p: int):
        self.speed = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.steering = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.sideslip_angle = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.acceleration = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.steering_rate = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        super()._spawn(dim_c, dim_p)


def make_heading_chevron(length, width, translation, rotation, color) -> list:
    """Return a chevron in the front half of a vehicle body, pointing forward.

    It is sized by the body width and drawn as two convex quads that share only
    the edge on the body axis, so an opaque color shows no seam.

    Args:
        length: Body length.
        width: Body width.
        translation: Body center (x, y).
        rotation: Body heading [rad].
        color: RGB color of the chevron.
    """
    from vmas.simulator import rendering

    tip = length / 2 - 0.15 * width
    depth = 0.35 * width  # Along the body axis
    half_span = 0.3 * width
    thickness = 0.12 * width
    geoms = []
    for side in (1, -1):
        tail = (tip - depth, side * half_span)
        geom = rendering.FilledPolygon(
            [
                tail,
                (tip, 0.0),
                (tip - thickness, 0.0),
                (tail[0] - thickness, tail[1]),
            ],
            draw_border=False,
        )
        xform = rendering.Transform()
        geom.add_attr(xform)
        xform.set_translation(*translation)
        xform.set_rotation(rotation)
        geom.set_color(*color)
        geoms.append(geom)
    return geoms


class Vehicle(Agent):
    def __init__(self, *args, is_render_direction: bool = False, **kwargs):
        """
        Initialize the Vehicle by calling the parent Agent's initializer
        and setting up the VehicleState.

        Redefine _state attribute to be VehicleState.

        Args:
            is_render_direction (bool): Whether to draw a chevron at the front
                of the vehicle body showing its heading.
        """
        super().__init__(*args, **kwargs)

        # Replace the default AgentState with VehicleState
        self._state = VehicleState()
        self.is_render_direction = is_render_direction

    def render(self, env_index: int = 0):
        """Render the vehicle body, plus its heading chevron if enabled."""
        geoms = super().render(env_index)
        if self.is_render_direction and geoms:
            geoms += self._render_direction_chevron(env_index)
        return geoms

    def _render_direction_chevron(self, env_index: int):
        """Return the heading chevron of the body in an opaque, darker body color."""
        return make_heading_chevron(
            self.shape.length,
            self.shape.width,
            self.state.pos[env_index],
            self.state.rot[env_index],
            [0.6 * channel for channel in self.color],
        )

    def set_speed(self, speed: Tensor, batch_index: int = None):
        """
        Set the speed state of the vehicle.

        Args:
            speed (Tensor): The new speed value.
            batch_index (int, optional): The index in the batch to set the speed for.
                                         If None, all batches are updated.
        """
        self._set_state_property(VehicleState.speed, self.state, speed, batch_index)

    def set_steering(self, steering: Tensor, batch_index: int = None):
        """
        Set the steering state of the vehicle.

        Args:
            steering (Tensor): The new steering value.
            batch_index (int, optional): The index in the batch to set the steering for.
                                         If None, all batches are updated.
        """
        self._set_state_property(
            VehicleState.steering, self.state, steering, batch_index
        )

    def set_sideslip_angle(self, sideslip_angle: Tensor, batch_index: int = None):
        """
        Set the sideslip angle of the vehicle, i.e., the angle between the velocity and the chasis.

        Args:
            sideslip_angle (Tensor): The new sideslip_angle value.
            batch_index (int, optional): The index in the batch to set the sideslip_angle for.
                                         If None, all batches are updated.
        """
        self._set_state_property(
            VehicleState.sideslip_angle, self.state, sideslip_angle, batch_index
        )

    def set_acceleration(self, acceleration: Tensor, batch_index: int = None):
        self._set_state_property(
            VehicleState.acceleration, self.state, acceleration, batch_index
        )

    def set_steering_rate(self, steering_rate: Tensor, batch_index: int = None):
        self._set_state_property(
            VehicleState.steering_rate, self.state, steering_rate, batch_index
        )


def is_latex_available() -> bool:
    try:
        # latex executable exists?
        subprocess.run(
            ["latex", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # required style file exists?
        subprocess.run(
            ["kpsewhich", "type1cm.sty"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # type1ec is often also needed by matplotlib's tex preamble on some setups
        subprocess.run(
            ["kpsewhich", "type1ec.sty"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class VideoWriter:
    """Stream RGB frames to FFmpeg so that a video is never held in memory.

    Frames (H x W x 3, uint8) are encoded as they are appended; FFmpeg starts at
    the first frame. `close()` finalizes the file and returns its name. Use it
    as a context manager to remove a partial file if an exception occurs.
    It can be passed to `env.rollout(frame_sink=...)` in place of a frame list.

    Args:
        name (str): Base name of the video file (without extension).
        fps (float): Frames per second for the output video.
        fmt (str, optional): Output format, one of {"mp4", "avi"}. Default is "mp4".
        quality (str, optional): MP4 quality level, one of {"low", "medium",
            "high", "very_high", "lossless"}. Default is "medium".
    """

    _CRF = {
        "low": "28",
        "medium": "23",
        "high": "18",
        "very_high": "14",
        "lossless": "0",
    }

    def __init__(
        self, name: str, fps: float, fmt: str = "mp4", quality: str = "medium"
    ):
        self.fmt = fmt.lower()
        self.quality = quality.lower()
        if self.fmt not in {"mp4", "avi"}:
            raise ValueError("Unsupported format. Use 'mp4' or 'avi'.")
        if self.fmt == "mp4" and self.quality not in self._CRF:
            raise ValueError(f"Unsupported quality. Use one of {list(self._CRF)}.")
        self.file_name = f"{name}.{self.fmt}"
        self.fps = fps
        self.shape = None
        self.n_frames = 0
        self._process = None

    def __len__(self):
        return self.n_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False

    def append(self, frame: np.ndarray):
        """Encode one RGB frame; all frames must have the shape of the first."""
        frame = np.asarray(frame)
        if self.shape is None:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected an H x W x 3 frame; got {frame.shape}.")
            self.shape = frame.shape
            self._process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        if frame.shape != self.shape:
            self.abort()
            raise ValueError(
                f"Frame {self.n_frames} has shape {frame.shape}; expected {self.shape}."
            )
        try:
            self._process.stdin.write(
                np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
            )
        except BrokenPipeError:
            self._process.stdin = None
            self.close()  # Raises with FFmpeg's error message
            raise
        self.n_frames += 1

    def close(self) -> str:
        """Finish encoding and return the file name."""
        if self._process is None:
            raise ValueError("No frames were appended. Cannot save video.")
        process, self._process = self._process, None
        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        _, stderr = process.communicate()
        if process.returncode != 0:
            if os.path.exists(self.file_name):
                os.remove(self.file_name)
            error = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed to save {self.file_name}: {error}")
        return self.file_name

    def abort(self):
        """Stop FFmpeg and remove the partial file."""
        if self._process is not None:
            self._process.kill()
            self._process.communicate()
            self._process = None
        if os.path.exists(self.file_name):
            os.remove(self.file_name)

    def _command(self) -> list:
        """FFmpeg command reading raw RGB frames from stdin."""
        import shutil

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            conda_ffmpeg = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
            if os.path.isfile(conda_ffmpeg) and os.access(conda_ffmpeg, os.X_OK):
                ffmpeg = conda_ffmpeg
            else:
                raise FileNotFoundError(
                    "FFmpeg is required to save videos but was not found on PATH "
                    "or beside the active Python executable."
                )
        h, w = self.shape[:2]
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{w}x{h}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
            "-an",
        ]
        if self.fmt == "mp4":
            command.extend(
                (
                    "-c:v",
                    "libx264",
                    "-crf",
                    self._CRF[self.quality],
                    "-preset",
                    "slow",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                )
            )
        else:
            command.extend(("-c:v", "ffv1", "-pix_fmt", "rgb24"))
        command.append(self.file_name)
        return command


def save_video(
    name: str,
    frame_list: "list[np.ndarray] | VideoWriter",
    fps: float,
    fmt: str = "mp4",
    quality: str = "medium",
):
    """
    Save a video from a list of frames with user-defined format and quality.

    Args:
        name (str): Base name of the video file (without extension).
        frame_list (list[np.ndarray] | VideoWriter): List of RGB frames
            (H x W x 3) as NumPy arrays, or a `VideoWriter` that already
            streamed them (e.g., from `env.rollout(frame_sink=...)`), which is
            then closed; its name, fps, format, and quality must match.
        fps (float): Frames per second for the output video.
        fmt (str, optional): Output format, one of {"mp4", "avi"}. Default is "mp4".
        quality (str, optional): MP4 quality level, one of {"low", "medium",
            "high", "very_high", "lossless"}. Default is "medium".

    Returns:
        str: The name of the saved file.
    """
    if isinstance(frame_list, VideoWriter):
        expected = (f"{name}.{fmt.lower()}", fps, quality.lower())
        actual = (frame_list.file_name, frame_list.fps, frame_list.quality)
        if expected != actual:
            frame_list.abort()
            raise ValueError(
                f"VideoWriter settings {actual} do not match save_video {expected}."
            )
        return frame_list.close()
    if len(frame_list) == 0:
        raise ValueError("frame_list is empty. Cannot save video.")
    with VideoWriter(name, fps, fmt=fmt, quality=quality) as writer:
        for frame in frame_list:
            writer.append(frame)
    return writer.file_name


def get_n_colors_cmap(n, cmap_name="tab20"):
    cmap = colormaps.get_cmap(cmap_name).resampled(n)
    return [cmap(i)[:3] for i in range(n)]  # RGB in [0,1]


def get_name_suffix(
    grouping, is_using_cbf_testing, n_agent, seed, max_group, nom, scenario
):
    cbf_tag = "cbf_on" if is_using_cbf_testing else "cbf_off"
    group_tag = "group_on" if grouping else "group_off"

    parts = [
        f"agents_{n_agent}",
        f"seed_{seed}",
        cbf_tag,
        group_tag,
        f"nom_{nom}",
        f"scenario_{scenario.lower()}",
    ]

    if grouping:
        parts.insert(4, f"maxgroup_{max_group}")  # right after group tag (optional)

    return "_".join(parts)


def trim_td(out_td: TensorDict, keys_to_keep=None, parameters=None) -> TensorDict:
    """
    Trim a TensorDict to keep only selected fields.

    Args:
        out_td: Original TensorDict.
        keys_to_keep: Custom list of keys to keep (overrides default key selection).
        parameters: Optional experiment configuration parameters. When specified,
            priority relation analysis keys are conditionally preserved if
            is_using_prioritized_marl is True, prioritization_method == "taskprogress",
            and is_save_detailed_info is True.

    Returns:
        A new TensorDict containing only the specified keys.
    """
    if keys_to_keep is None:
        keys_to_keep = [
            ("agents", "info", "pos"),
            ("agents", "info", "rot"),
            ("agents", "info", "vel"),
            ("agents", "info", "speed"),
            ("agents", "info", "acceleration"),
            ("agents", "info", "steering"),
            ("agents", "info", "steering_rate"),
            ("agents", "info", "act_vel"),
            ("agents", "info", "act_steer"),
            ("agents", "info", "applied_action_vel"),
            ("agents", "info", "applied_action_steer"),
            ("agents", "info", "nominal_action_vel"),
            ("agents", "info", "nominal_action_steer"),
            ("agents", "info", "priority", "scores"),
            ("agents", "info", "priority", "rank"),
            ("agents", "info", "is_collision_with_agents"),
            ("agents", "info", "is_collision_with_lanelets"),
            ("agents", "info", "is_reach_goal"),
            ("agents", "info", "is_deadlocked"),
            ("agents", "info", "is_joint_task_success"),
            ("agents", "info", "is_joint_task_failure"),
            ("agents", "info", "is_joint_task_try"),
            ("agents", "info", "is_joint_task_on_hold"),
            ("agents", "info", "num_task_tries"),
            ("agents", "info", "task_success_times"),
            ("agents", "info", "rew_parameter_independent"),
            ("agents", "info", "reward_progress"),
            ("agents", "info", "penalty_near_other_agents"),
            ("agents", "info", "penalty_near_boundary"),
            ("agents", "info", "penalty_deadlock"),
            ("agents", "info", "distance_ref"),
            ("agents", "info", "distance_left_b"),
            ("agents", "info", "distance_right_b"),
            ("agents", "info", "step_time"),
            "time_cbf",
        ]

        is_priority_relation_active = False
        if parameters is not None:
            is_priority_relation_active = (
                bool(getattr(parameters, "is_using_prioritized_marl", False))
                and getattr(parameters, "prioritization_method", None) == "taskprogress"
                and bool(getattr(parameters, "is_save_detailed_info", False))
            )
        else:
            try:
                is_priority_relation_active = (
                    "priority_score" in out_td.get(("agents", "info")).keys()
                )
            except Exception:
                is_priority_relation_active = False

        if is_priority_relation_active:
            keys_to_keep.extend(
                [
                    ("agents", "info", "route_progress"),
                    ("agents", "info", "route_total_length"),
                    ("agents", "info", "progress_fraction"),
                    ("agents", "info", "priority_score"),
                    ("agents", "info", "priority_rank"),
                    ("agents", "info", "priority_weight"),
                    ("agents", "info", "min_distance_to_agents"),
                    ("agents", "info", "nearest_agent_index"),
                    ("agents", "info", "nearest_agent_distance"),
                    ("agents", "info", "nearest_agent_priority_score"),
                    ("agents", "info", "nearest_agent_priority_rank"),
                    ("agents", "info", "nearest_agent_priority_weight"),
                    ("agents", "info", "nearest_agent_speed"),
                    ("agents", "info", "nearest_agent_acceleration"),
                    ("agents", "info", "priority_difference_nearest"),
                    ("agents", "info", "rank_difference_nearest"),
                    ("agents", "info", "speed_difference_nearest"),
                    ("agents", "info", "acc_difference_nearest"),
                ]
            )

    trimmed = TensorDict(
        {},
        batch_size=out_td.batch_size,
        device=out_td.device,
    )

    optional_keys = {
        "time_cbf",
        ("agents", "info", "is_joint_task_success"),
        ("agents", "info", "is_joint_task_failure"),
        ("agents", "info", "is_joint_task_try"),
        ("agents", "info", "is_joint_task_on_hold"),
        ("agents", "info", "num_task_tries"),
        ("agents", "info", "task_success_times"),
        ("agents", "info", "speed"),
        ("agents", "info", "acceleration"),
        ("agents", "info", "steering"),
        ("agents", "info", "steering_rate"),
        ("agents", "info", "act_vel"),
        ("agents", "info", "act_steer"),
        ("agents", "info", "applied_action_vel"),
        ("agents", "info", "applied_action_steer"),
        ("agents", "info", "nominal_action_vel"),
        ("agents", "info", "nominal_action_steer"),
        ("agents", "info", "route_progress"),
        ("agents", "info", "route_total_length"),
        ("agents", "info", "progress_fraction"),
        ("agents", "info", "priority_score"),
        ("agents", "info", "priority_rank"),
        ("agents", "info", "priority_weight"),
        ("agents", "info", "min_distance_to_agents"),
        ("agents", "info", "nearest_agent_index"),
        ("agents", "info", "nearest_agent_distance"),
        ("agents", "info", "nearest_agent_priority_score"),
        ("agents", "info", "nearest_agent_priority_rank"),
        ("agents", "info", "nearest_agent_priority_weight"),
        ("agents", "info", "nearest_agent_speed"),
        ("agents", "info", "nearest_agent_acceleration"),
        ("agents", "info", "priority_difference_nearest"),
        ("agents", "info", "rank_difference_nearest"),
        ("agents", "info", "speed_difference_nearest"),
        ("agents", "info", "acc_difference_nearest"),
        ("agents", "info", "priority", "scores"),
        ("agents", "info", "priority", "rank"),
        ("agents", "info", "reward_progress"),
        ("agents", "info", "penalty_near_other_agents"),
        ("agents", "info", "penalty_near_boundary"),
        ("agents", "info", "penalty_deadlock"),
        ("agents", "info", "distance_ref"),
        ("agents", "info", "distance_left_b"),
        ("agents", "info", "distance_right_b"),
    }
    for key in keys_to_keep:
        try:
            value = out_td.get(key)
        except Exception as e:
            if key in optional_keys:
                warnings.warn(
                    f"[WARN] Optional key '{key}' raised an exception ({e}) and was skipped.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            warnings.warn(
                f"[WARN] Key '{key}' raised an exception ({e}).",
                UserWarning,
                stacklevel=2,
            )
            raise

        if value is None:
            if parameters.is_fast_inference_mode:
                continue
            if key in optional_keys:
                warnings.warn(
                    f"[WARN] Optional key '{key}' not found in out_td, skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            else:
                warnings.warn(
                    f"[WARN] Key '{key}' not found in out_td.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

        trimmed.set(key, value)

    return trimmed


def visu_td_results(
    out_td: TensorDict,
    parameters: Parameters = None,
    vehicle_ids: typing.Union[str, int, typing.Sequence[int]] = "all",
    env_index: int = 0,
    save_dir: str = None,
    fig_name: str = "td_results_profiles",
    show: bool = False,
):
    """
    Visualize rollout profiles stored in a TensorDict and export them as one figure.

    The exported figure contains subplots for speed, steering, acceleration,
    steering rate, the two control-input channels, events, reward, and CBF
    computation time. The function primarily uses `out_td["agents", "info", ...]`
    and falls back to other fields when needed. Missing fields produce warnings
    instead of exceptions.

    Args:
        out_td: Rollout output TensorDict with batch shape [n_envs, time].
        parameters: Optional Parameters instance used for dt, save path, and
            interpreting `rl_action_type`.
        vehicle_ids: `"all"`, a single int, or a sequence / comma-separated
            string of vehicle indices.
        env_index: Batch environment index to visualize.
        save_dir: Directory where the figure is saved.
        fig_name: Output figure name. If no extension is provided, `.png` is
            appended and the selected env / vehicles are encoded in the name.
        show: If True, display the figure after saving.

    Returns:
        Path to the saved figure, or None if visualization could not be built.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from sigmarl.constants import AGENTS, POSITION_SPEED_LATERAL_OFFSET_BOUND

    # This figure contains plain-text summaries assembled at runtime.
    # Disable TeX rendering here to avoid LaTeX parsing failures on strings
    # such as "m/s^2", underscores, or other non-TeX-safe content.
    plt.rcParams["text.usetex"] = False

    def _warn(msg: str):
        print(f"[WARNING] visu_td_results: {msg}")

    def _get_tensor(key):
        try:
            return out_td.get(key)
        except Exception:
            return None

    def _infer_n_envs():
        if len(out_td.batch_size) >= 1:
            return int(out_td.batch_size[0])
        for key in (
            ("agents", "info", "pos"),
            ("agents", "info", "vel"),
            ("agents", "action"),
        ):
            tensor = _get_tensor(key)
            if tensor is not None and tensor.ndim >= 1:
                return int(tensor.shape[0])
        return 1

    def _infer_n_agents():
        for key in (
            ("agents", "info", "pos"),
            ("agents", "info", "vel"),
            ("agents", "info", "act_vel"),
            ("agents", "action"),
        ):
            tensor = _get_tensor(key)
            if tensor is not None and tensor.ndim >= 3:
                return int(tensor.shape[2])
        return None

    def _normalize_vehicle_ids(n_agents: int):
        if n_agents is None or n_agents <= 0:
            return []

        if vehicle_ids is None:
            return list(range(n_agents))

        if isinstance(vehicle_ids, str):
            vehicle_ids_str = vehicle_ids.strip().lower()
            if vehicle_ids_str in {"all", ""}:
                return list(range(n_agents))
            vehicle_list = [item.strip() for item in vehicle_ids.split(",")]
            parsed = []
            for item in vehicle_list:
                if not item:
                    continue
                try:
                    parsed.append(int(item))
                except ValueError:
                    _warn(f"Ignore invalid vehicle index '{item}'.")
            vehicle_list = parsed
        elif isinstance(vehicle_ids, int):
            vehicle_list = [vehicle_ids]
        else:
            vehicle_list = [int(i) for i in vehicle_ids]

        valid_vehicle_ids = []
        for idx in vehicle_list:
            if 0 <= idx < n_agents:
                valid_vehicle_ids.append(idx)
            else:
                _warn(f"Ignore out-of-range vehicle index {idx}.")

        if not valid_vehicle_ids:
            _warn("No valid vehicle indices remain. Fall back to all vehicles.")
            return list(range(n_agents))
        return valid_vehicle_ids

    def _select_env_agents(tensor, field_name: str):
        if tensor is None:
            return None

        tensor = torch.as_tensor(tensor).detach().cpu()
        if tensor.ndim < 3:
            _warn(
                f"Field '{field_name}' has unexpected shape {tuple(tensor.shape)} and will be ignored."
            )
            return None

        return tensor[env_index, :, selected_vehicle_ids]

    def _prepare_scalar_series(tensor, field_name: str):
        tensor = _select_env_agents(tensor, field_name)
        if tensor is None:
            return None

        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        elif tensor.ndim != 2:
            _warn(
                f"Field '{field_name}' cannot be reduced to a scalar time series. Shape={tuple(tensor.shape)}"
            )
            return None
        return tensor.to(torch.float32)

    def _prepare_rollout_scalar_series(tensor, field_name: str):
        if tensor is None:
            return None

        tensor = torch.as_tensor(tensor).detach().cpu()
        if tensor.ndim == 1:
            series = tensor
        elif tensor.ndim >= 2:
            if env_index >= tensor.shape[0]:
                _warn(
                    f"Field '{field_name}' does not contain env_index={env_index}. Shape={tuple(tensor.shape)}"
                )
                return None
            series = tensor[env_index]
        else:
            _warn(
                f"Field '{field_name}' has unexpected shape {tuple(tensor.shape)} and will be ignored."
            )
            return None

        while series.ndim > 1 and series.shape[-1] == 1:
            series = series.squeeze(-1)

        if series.ndim != 1:
            _warn(
                f"Field '{field_name}' cannot be reduced to a rollout scalar time series. Shape={tuple(tensor.shape)}"
            )
            return None
        return series.to(torch.float32)

    def _choose_first_available(candidates, label: str, warn_if_missing: bool = True):
        for key, fallback_note in candidates:
            tensor = _prepare_scalar_series(_get_tensor(key), "/".join(key))
            if tensor is not None:
                if fallback_note:
                    _warn(fallback_note)
                return tensor
        if warn_if_missing:
            _warn(f"No usable field found for {label}.")
        return None

    def _choose_first_available_rollout_scalar(
        candidates, label: str, warn_if_missing: bool = True
    ):
        for key, fallback_note in candidates:
            field_name = "/".join(key) if isinstance(key, tuple) else str(key)
            tensor = _prepare_rollout_scalar_series(_get_tensor(key), field_name)
            if tensor is not None:
                if fallback_note:
                    _warn(fallback_note)
                return tensor
        if warn_if_missing:
            _warn(f"No usable field found for {label}.")
        return None

    def _prepare_event_series(tensor, field_name: str):
        tensor = _select_env_agents(tensor, field_name)
        if tensor is None:
            return None

        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        elif tensor.ndim != 2:
            _warn(
                f"Field '{field_name}' cannot be reduced to an event time series. Shape={tuple(tensor.shape)}"
            )
            return None
        return tensor.to(torch.bool)

    def _finite_difference(series: torch.Tensor, label: str):
        if series is None:
            return None
        series_np = series.numpy()
        if series_np.shape[0] == 0:
            _warn(f"Cannot compute {label}: empty time axis.")
            return None
        if series_np.shape[0] == 1:
            return torch.zeros_like(series)
        return torch.from_numpy(np.gradient(series_np, dt, axis=0)).to(torch.float32)

    def _recover_lateral_offset_from_steer(steer_series: torch.Tensor):
        if steer_series is None:
            return None

        lookahead_dist = (
            float(getattr(parameters, "lookahead_dist", POSITION_SPEED_LOOKAHEAD_DIST))
            if parameters is not None
            else POSITION_SPEED_LOOKAHEAD_DIST
        )
        l_wb = float(
            getattr(parameters, "l_wb", AGENTS["l_f"] + AGENTS["l_r"])
            if parameters is not None
            else AGENTS["l_f"] + AGENTS["l_r"]
        )
        ld = steer_series.new_tensor(lookahead_dist)
        l_wb_tensor = steer_series.new_tensor(l_wb)
        kappa = torch.tan(steer_series) / l_wb_tensor

        y_max = steer_series.new_tensor(float(POSITION_SPEED_LATERAL_OFFSET_BOUND))
        max_kappa = 2.0 * y_max / (ld**2 + y_max**2) if y_max.item() > 0 else 0.0
        kappa = torch.clamp(kappa, min=-max_kappa, max=max_kappa)

        discr = torch.clamp(1.0 - (kappa * ld) ** 2, min=0.0)
        lateral_offset = torch.where(
            kappa.abs() < 1e-8,
            torch.zeros_like(kappa),
            (kappa * (ld**2)) / (1.0 + torch.sqrt(discr)),
        )
        if y_max is not None:
            lateral_offset = torch.clamp(lateral_offset, min=-y_max, max=y_max)
        return lateral_offset

    def _plot_series(ax, series: torch.Tensor, title: str, ylabel: str):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)

        if series is None:
            ax.text(
                0.5,
                0.5,
                "Unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="grey",
            )
            return

        series_np = series.numpy()
        if series_np.ndim == 1:
            series_np = series_np[:, None]

        time_axis_local = np.arange(series_np.shape[0], dtype=np.float32) * dt
        for idx_vehicle, vehicle_idx in enumerate(selected_vehicle_ids):
            ax.plot(
                time_axis_local,
                series_np[:, idx_vehicle],
                linewidth=1.8,
                color=plot_colors[idx_vehicle],
                label=f"veh {vehicle_idx}",
            )

    def _plot_timing_series(ax, series: torch.Tensor, title: str):
        ax.set_title(title)
        ax.set_ylabel("Time [ms]")
        ax.grid(True, linestyle="--", alpha=0.35)

        if series is None:
            ax.text(
                0.5,
                0.5,
                "Unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="grey",
            )
            return

        series_np = series.numpy() * 1e3
        time_axis_local = np.arange(series_np.shape[0], dtype=np.float32) * dt
        ax.plot(time_axis_local, series_np, linewidth=1.8, color="tab:red")

        finite_values = series_np[np.isfinite(series_np)]
        if finite_values.size > 0:
            ax.text(
                0.99,
                0.93,
                f"mean={finite_values.mean():.2f} ms | max={finite_values.max():.2f} ms",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "lightgrey", "alpha": 0.85},
            )

    def _apply_ylim(ax, y_limits):
        if y_limits is None:
            return
        y_min, y_max = y_limits
        if y_min is None or y_max is None or y_min >= y_max:
            return
        ax.set_ylim(float(y_min), float(y_max))

    def _plot_events(
        ax,
        collision_agents,
        collision_lanelets,
        reach_goal,
        deadlocked,
    ):
        ax.set_title("Events")
        ax.set_ylabel("Vehicle")
        ax.grid(True, linestyle="--", alpha=0.35, axis="x")

        n_rows = len(selected_vehicle_ids)
        y_positions = np.arange(n_rows, dtype=np.float32)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"veh {idx}" for idx in selected_vehicle_ids])
        ax.set_ylim(-0.5, max(0.5, n_rows - 0.5))

        n_steps_local = None
        for event_series in (
            collision_agents,
            collision_lanelets,
            reach_goal,
            deadlocked,
        ):
            if event_series is not None:
                n_steps_local = event_series.shape[0]
                break
        if n_steps_local is None and speed is not None:
            n_steps_local = speed.shape[0]

        if n_steps_local is None:
            ax.text(
                0.5,
                0.5,
                "Unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="grey",
            )
            return

        time_axis_local = np.arange(n_steps_local, dtype=np.float32) * dt
        has_any_event = False

        for row_idx in range(n_rows):
            if collision_agents is not None:
                mask_agents = collision_agents[:, row_idx].numpy()
                if mask_agents.any():
                    has_any_event = True
                    ax.scatter(
                        time_axis_local[mask_agents],
                        np.full(mask_agents.sum(), row_idx),
                        marker="o",
                        s=42,
                        facecolors="none",
                        edgecolors=[plot_colors[row_idx]],
                        linewidths=1.6,
                    )

            if collision_lanelets is not None:
                mask_lanelets = collision_lanelets[:, row_idx].numpy()
                if mask_lanelets.any():
                    has_any_event = True
                    ax.scatter(
                        time_axis_local[mask_lanelets],
                        np.full(mask_lanelets.sum(), row_idx),
                        marker="x",
                        s=42,
                        color=[plot_colors[row_idx]],
                        linewidths=1.6,
                    )

            if reach_goal is not None:
                mask_goal = reach_goal[:, row_idx].numpy()
                if mask_goal.any():
                    has_any_event = True
                    ax.scatter(
                        time_axis_local[mask_goal],
                        np.full(mask_goal.sum(), row_idx),
                        marker="^",
                        s=48,
                        facecolors="none",
                        edgecolors=[plot_colors[row_idx]],
                        linewidths=1.6,
                    )

            if deadlocked is not None:
                mask_deadlock = deadlocked[:, row_idx].numpy()
                if mask_deadlock.any():
                    has_any_event = True
                    ax.scatter(
                        time_axis_local[mask_deadlock],
                        np.full(mask_deadlock.sum(), row_idx),
                        marker="s",
                        s=38,
                        facecolors="none",
                        edgecolors=[plot_colors[row_idx]],
                        linewidths=1.6,
                    )

        if not has_any_event:
            ax.text(
                0.5,
                0.5,
                "No event detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="grey",
            )

        event_legend_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="black",
                color="black",
                markersize=7,
                label="Collision with agents",
            ),
            Line2D(
                [],
                [],
                marker="x",
                linestyle="None",
                color="black",
                markersize=7,
                label="Collision with lanelets",
            ),
            Line2D(
                [],
                [],
                marker="^",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="black",
                color="black",
                markersize=7,
                label="Goal reached",
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="black",
                color="black",
                markersize=7,
                label="Deadlocked",
            ),
        ]
        ax.legend(handles=event_legend_handles, loc="upper right", frameon=False)

    def _abs_mean_per_vehicle(series: torch.Tensor):
        if series is None:
            return None
        return torch.mean(torch.abs(series), dim=0)

    def _format_abs_mean(value, unit: str):
        if value is None:
            return "N/A"
        unit = unit.strip()
        return f"{float(value):.3f}" if not unit else f"{float(value):.3f} {unit}"

    n_envs = _infer_n_envs()
    if n_envs <= 0:
        _warn("Unable to infer the number of environments from out_td.")
        return None

    if env_index < 0 or env_index >= n_envs:
        _warn(
            f"Requested env_index={env_index} is out of range [0, {n_envs - 1}]. Fall back to env 0."
        )
        env_index = 0

    n_agents = _infer_n_agents()
    if n_agents is None:
        _warn("Unable to infer the number of agents from out_td.")
        return None

    selected_vehicle_ids = _normalize_vehicle_ids(n_agents)
    if not selected_vehicle_ids:
        _warn("No vehicle selected for visualization.")
        return None

    dt = float(getattr(parameters, "dt", 1.0)) if parameters is not None else 1.0
    rl_action_type = (
        getattr(parameters, "rl_action_type", "speed_steer")
        if parameters is not None
        else "speed_steer"
    )

    plot_colors = get_n_colors_cmap(len(selected_vehicle_ids), cmap_name="tab10")

    speed = _choose_first_available(
        [
            (("agents", "info", "speed"), None),
        ],
        label="speed",
        warn_if_missing=False,
    )

    if speed is None:
        vel = _select_env_agents(
            _get_tensor(("agents", "info", "vel")), "agents/info/vel"
        )
        rot = _select_env_agents(
            _get_tensor(("agents", "info", "rot")), "agents/info/rot"
        )

        if vel is not None:
            if vel.ndim == 3 and vel.shape[-1] >= 2:
                if rot is not None:
                    if rot.ndim == 3 and rot.shape[-1] == 1:
                        rot = rot.squeeze(-1)
                    elif rot.ndim != 2:
                        _warn(
                            f"Field 'agents/info/rot' has unexpected shape {tuple(rot.shape)}. Using velocity magnitude for speed."
                        )
                        rot = None
                if rot is not None:
                    speed = vel[..., 0] * torch.cos(rot) + vel[..., 1] * torch.sin(rot)
                else:
                    _warn(
                        "Field 'agents/info/rot' is unavailable. Using velocity magnitude for speed."
                    )
                    speed = torch.linalg.norm(vel[..., :2], dim=-1)
            else:
                _warn(
                    f"Field 'agents/info/vel' has unexpected shape {tuple(vel.shape)} and cannot be used for speed."
                )

    if speed is None:
        speed = _choose_first_available(
            [
                (
                    ("agents", "info", "applied_action_vel"),
                    "Speed state is unavailable. Using applied_action_vel instead.",
                ),
                (
                    ("agents", "info", "act_vel"),
                    "Speed state is unavailable. Using act_vel instead.",
                ),
                (
                    ("agents", "info", "nominal_action_vel"),
                    "Speed state is unavailable. Using nominal_action_vel instead.",
                ),
            ],
            label="speed",
        )

    steering = _choose_first_available(
        [
            (("agents", "info", "steering"), None),
            (
                ("agents", "info", "applied_action_steer"),
                "Steering state is unavailable. Using applied_action_steer instead.",
            ),
            (
                ("agents", "info", "act_steer"),
                "Steering state is unavailable. Using act_steer instead.",
            ),
            (
                ("agents", "info", "nominal_action_steer"),
                "Steering state is unavailable. Using nominal_action_steer instead.",
            ),
        ],
        label="steering",
    )

    acceleration = _choose_first_available(
        [
            (("agents", "info", "acceleration"), None),
            (("agents", "info", "acc"), None),
        ],
        label="acceleration",
        warn_if_missing=False,
    )
    if acceleration is None:
        if speed is not None:
            _warn(
                "Acceleration field is unavailable. Using finite difference of the speed profile."
            )
            acceleration = _finite_difference(speed, label="acceleration")
        else:
            _warn("Acceleration field is unavailable and speed could not be recovered.")

    steering_rate = _choose_first_available(
        [
            (("agents", "info", "steering_rate"), None),
            (("agents", "info", "steer_rate"), None),
        ],
        label="steering rate",
        warn_if_missing=False,
    )
    if steering_rate is None:
        if steering is not None:
            _warn(
                "Steering-rate field is unavailable. Using finite difference of the steering profile."
            )
            steering_rate = _finite_difference(steering, label="steering rate")
        else:
            _warn(
                "Steering-rate field is unavailable and steering could not be recovered."
            )

    reward_total = _choose_first_available(
        [
            (("agents", "info", "rew_total"), None),
        ],
        label="total reward",
        warn_if_missing=False,
    )
    if reward_total is None:
        _warn("Total reward field 'agents/info/rew_total' is unavailable.")

    collision_with_agents = _prepare_event_series(
        _get_tensor(("agents", "info", "is_collision_with_agents")),
        "agents/info/is_collision_with_agents",
    )
    if collision_with_agents is None:
        _warn("Collision field 'agents/info/is_collision_with_agents' is unavailable.")

    collision_with_lanelets = _prepare_event_series(
        _get_tensor(("agents", "info", "is_collision_with_lanelets")),
        "agents/info/is_collision_with_lanelets",
    )
    if collision_with_lanelets is None:
        _warn(
            "Collision field 'agents/info/is_collision_with_lanelets' is unavailable."
        )

    reach_goal = _prepare_event_series(
        _get_tensor(("agents", "info", "is_reach_goal")),
        "agents/info/is_reach_goal",
    )
    if reach_goal is None:
        _warn("Event field 'agents/info/is_reach_goal' is unavailable.")

    deadlocked = _prepare_event_series(
        _get_tensor(("agents", "info", "is_deadlocked")),
        "agents/info/is_deadlocked",
    )
    if deadlocked is None:
        _warn("Event field 'agents/info/is_deadlocked' is unavailable.")

    time_cbf = _choose_first_available_rollout_scalar(
        [
            ("time_cbf", None),
            (("collector", "time_cbf"), None),
        ],
        label="CBF computation time",
    )

    action = _select_env_agents(_get_tensor(("agents", "action")), "agents/action")
    control_input_0 = None
    control_input_1 = None

    if rl_action_type == "speed":
        if action is not None and action.ndim == 3 and action.shape[-1] >= 1:
            control_input_0 = action[..., 0].to(torch.float32)
        else:
            control_input_0 = _choose_first_available(
                [
                    (("agents", "info", "act_vel"), None),
                    (("agents", "info", "applied_action_vel"), None),
                    (("agents", "info", "nominal_action_vel"), None),
                ],
                label="speed target speed",
            )
        control_input_1 = _choose_first_available(
            [
                (("agents", "info", "act_steer"), None),
                (("agents", "info", "applied_action_steer"), None),
                (("agents", "info", "nominal_action_steer"), None),
            ],
            label="speed path-tracking steering",
            warn_if_missing=False,
        )
    elif rl_action_type == "position_speed":
        if action is not None and action.ndim == 3 and action.shape[-1] >= 2:
            control_input_0 = action[..., 0].to(torch.float32)
            control_input_1 = action[..., 1].to(torch.float32)
        else:
            if action is not None:
                _warn(
                    f"Field 'agents/action' has unexpected shape {tuple(action.shape)}. Falling back to other fields for control inputs."
                )
            control_input_1 = _choose_first_available(
                [
                    (("agents", "info", "act_vel"), None),
                    (("agents", "info", "applied_action_vel"), None),
                    (("agents", "info", "nominal_action_vel"), None),
                ],
                label="position_speed target speed",
            )
            steering_for_recovery = _choose_first_available(
                [
                    (("agents", "info", "act_steer"), None),
                    (("agents", "info", "applied_action_steer"), None),
                    (("agents", "info", "nominal_action_steer"), None),
                ],
                label="position_speed lateral offset recovery",
            )
            if steering_for_recovery is not None:
                control_input_0 = _recover_lateral_offset_from_steer(
                    steering_for_recovery
                )
                _warn(
                    "Raw position_speed action is unavailable. Reconstructed lateral offset from steering."
                )
    else:
        control_input_0 = _choose_first_available(
            [
                (("agents", "info", "act_vel"), None),
                (("agents", "info", "applied_action_vel"), None),
                (("agents", "info", "nominal_action_vel"), None),
            ],
            label="speed_steer speed command",
            warn_if_missing=False,
        )
        control_input_1 = _choose_first_available(
            [
                (("agents", "info", "act_steer"), None),
                (("agents", "info", "applied_action_steer"), None),
                (("agents", "info", "nominal_action_steer"), None),
            ],
            label="speed_steer steering command",
            warn_if_missing=False,
        )
        if (
            (control_input_0 is None or control_input_1 is None)
            and action is not None
            and action.ndim == 3
            and action.shape[-1] >= 2
        ):
            if control_input_0 is None:
                control_input_0 = action[..., 0].to(torch.float32)
            if control_input_1 is None:
                control_input_1 = action[..., 1].to(torch.float32)
            _warn(
                "Simulator-reported speed_steer command is unavailable. Using agents/action instead."
            )
        elif action is not None and not (action.ndim == 3 and action.shape[-1] >= 2):
            _warn(
                f"Field 'agents/action' has unexpected shape {tuple(action.shape)} and was ignored for control inputs."
            )

        if control_input_0 is None and control_input_1 is None:
            _warn("No usable field found for speed_steer control inputs.")

    if rl_action_type == "position_speed":
        control_input_0_title = "Control Input 1"
        control_input_0_ylabel = "Lateral Offset [m]"
        control_input_1_title = "Control Input 2"
        control_input_1_ylabel = "Target Speed [m/s]"
    elif rl_action_type == "speed":
        control_input_0_title = "Control Input 1"
        control_input_0_ylabel = "Target Speed [m/s]"
        control_input_1_title = "Path Tracking"
        control_input_1_ylabel = "Target Steering [rad]"
    else:
        control_input_0_title = "Control Input 1"
        control_input_0_ylabel = "Target Speed [m/s]"
        control_input_1_title = "Control Input 2"
        control_input_1_ylabel = "Target Steering [rad]"

    speed_limits = (
        AGENTS.get("min_speed"),
        (
            parameters.max_speed
            if parameters is not None and parameters.max_speed is not None
            else AGENTS.get("max_speed")
        ),
    )
    steering_limits = (AGENTS.get("min_steering"), AGENTS.get("max_steering"))
    acceleration_limits = (AGENTS.get("min_acc"), AGENTS.get("max_acc"))
    steering_rate_limits = (
        AGENTS.get("min_steering_rate"),
        AGENTS.get("max_steering_rate"),
    )
    if rl_action_type == "position_speed":
        control_input_0_limits = (
            -float(POSITION_SPEED_LATERAL_OFFSET_BOUND),
            float(POSITION_SPEED_LATERAL_OFFSET_BOUND),
        )
        control_input_1_limits = speed_limits
    elif rl_action_type == "speed":
        control_input_0_limits = speed_limits
        control_input_1_limits = steering_limits
    else:
        control_input_0_limits = speed_limits
        control_input_1_limits = steering_limits

    fig = plt.figure(figsize=(14, 16))
    grid_spec = fig.add_gridspec(5, 2)
    axes = np.empty((4, 2), dtype=object)
    shared_axis = None
    for row in range(4):
        for col in range(2):
            axes[row, col] = fig.add_subplot(
                grid_spec[row, col],
                sharex=shared_axis,
            )
            if shared_axis is None:
                shared_axis = axes[row, col]
    time_cbf_ax = fig.add_subplot(grid_spec[4, :], sharex=shared_axis)

    _plot_series(axes[0, 0], speed, "Speed", "Speed [m/s]")
    _plot_series(axes[0, 1], steering, "Steering", "Steering [rad]")
    _plot_series(axes[1, 0], acceleration, "Acceleration", "Acceleration [m/s²]")
    _plot_series(axes[1, 1], steering_rate, "Steering Rate", "Steering Rate [rad/s]")
    _plot_series(
        axes[2, 0], control_input_0, control_input_0_title, control_input_0_ylabel
    )
    _plot_series(
        axes[2, 1], control_input_1, control_input_1_title, control_input_1_ylabel
    )
    _plot_series(axes[3, 0], reward_total, "Total Reward", "Reward [-]")
    _plot_events(
        axes[3, 1],
        collision_with_agents,
        collision_with_lanelets,
        reach_goal,
        deadlocked,
    )
    _plot_timing_series(time_cbf_ax, time_cbf, "CBF Computation Time")

    _apply_ylim(axes[0, 0], speed_limits)
    _apply_ylim(axes[0, 1], steering_limits)
    _apply_ylim(axes[1, 0], acceleration_limits)
    _apply_ylim(axes[1, 1], steering_rate_limits)
    _apply_ylim(axes[2, 0], control_input_0_limits)
    _apply_ylim(axes[2, 1], control_input_1_limits)

    time_cbf_ax.set_xlabel("Time [s]")

    vehicle_tag = (
        "all"
        if len(selected_vehicle_ids) == n_agents
        else "-".join(str(idx) for idx in selected_vehicle_ids)
    )
    fig.suptitle(
        f"TD rollout profiles | env {env_index} | vehicles {vehicle_tag} | rl_action_type={rl_action_type}",
        fontsize=13,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if labels:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, 0.98),
            frameon=False,
        )

    abs_mean_speed = _abs_mean_per_vehicle(speed)
    abs_mean_steering = _abs_mean_per_vehicle(steering)
    abs_mean_acceleration = _abs_mean_per_vehicle(acceleration)
    abs_mean_steering_rate = _abs_mean_per_vehicle(steering_rate)
    abs_mean_control_input_0 = _abs_mean_per_vehicle(control_input_0)
    abs_mean_control_input_1 = _abs_mean_per_vehicle(control_input_1)
    abs_mean_reward_total = _abs_mean_per_vehicle(reward_total)
    abs_mean_metric_items = [
        ("|speed|", abs_mean_speed, "m/s"),
        ("|steering|", abs_mean_steering, "rad"),
        ("|acc|", abs_mean_acceleration, "m/s^2"),
        ("|steer_rate|", abs_mean_steering_rate, "rad/s"),
        (
            "|u1|",
            abs_mean_control_input_0,
            control_input_0_ylabel.split("[")[-1].rstrip("]")
            if "[" in control_input_0_ylabel
            else "",
        ),
        (
            "|u2|",
            abs_mean_control_input_1,
            control_input_1_ylabel.split("[")[-1].rstrip("]")
            if "[" in control_input_1_ylabel
            else "",
        ),
        ("|rew_total|", abs_mean_reward_total, ""),
    ]

    summary_lines = []
    mean_time_cbf_ms = None
    if time_cbf is not None:
        time_cbf_ms = time_cbf.numpy() * 1e3
        finite_time_cbf_ms = time_cbf_ms[np.isfinite(time_cbf_ms)]
        if finite_time_cbf_ms.size > 0:
            mean_time_cbf_ms = finite_time_cbf_ms.mean()
            summary_lines.append(
                "CBF computation time: "
                f"mean={mean_time_cbf_ms:.3f} ms | "
                f"max={finite_time_cbf_ms.max():.3f} ms | "
                f"steps={finite_time_cbf_ms.size}"
            )

    summary_lines.append("Absolute mean values per selected vehicle:")
    for idx_vehicle, vehicle_idx in enumerate(selected_vehicle_ids):
        summary_lines.append(
            " | ".join(
                [f"veh {vehicle_idx}"]
                + [
                    f"{metric_label}={_format_abs_mean(values[idx_vehicle] if values is not None else None, unit)}"
                    for metric_label, values, unit in abs_mean_metric_items
                ]
            )
        )

    summary_lines.append("Mean across selected vehicles:")
    summary_lines.append(
        " | ".join(
            ["vehicles mean"]
            + [
                f"{metric_label}={_format_abs_mean(values.mean() if values is not None else None, unit)}"
                for metric_label, values, unit in abs_mean_metric_items
            ]
        )
    )

    summary_text = "\n".join(summary_lines)
    fig.text(
        0.5,
        0.01,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "lightgrey", "alpha": 0.9},
    )

    fig.tight_layout(rect=[0, 0.12, 1, 0.94])

    if save_dir is None:
        save_dir = (
            getattr(parameters, "where_to_save", ".") if parameters is not None else "."
        )
    os.makedirs(save_dir, exist_ok=True)

    file_root, file_ext = os.path.splitext(fig_name)
    if not file_ext:
        fig_name = f"{file_root}_env{env_index}_veh{vehicle_tag}.png"

    fig_path = os.path.join(save_dir, fig_name)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved TD profile figure to: {fig_path}")
    if mean_time_cbf_ms is not None:
        print(f"[INFO] Mean CBF computation time: {mean_time_cbf_ms:.3f} ms")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


def visu_cbf_boundary_distances(
    out_td: TensorDict,
    parameters: Parameters = None,
    vehicle_ids: typing.Union[str, int, typing.Sequence[int]] = "all",
    env_index: int = 0,
    save_dir: str = None,
    fig_name: str = "cbf_boundary_distances",
    show: bool = False,
):
    """
    Visualize the boundary distance diagnostics used by the CBF-QP.

    The raw fields are the values returned by the configured lane distance
    module. For softmin they are squared-distance surrogate values; for pseudo
    they are pseudo distances. The margin fields subtract the circle
    approximation radius term in the same units used by the lane CBF.
    """
    import os
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False

    def _warn(msg: str):
        print(f"[WARNING] visu_cbf_boundary_distances: {msg}")

    def _get_tensor(key):
        try:
            return out_td.get(key)
        except Exception:
            return None

    def _infer_n_agents():
        for key in (
            ("agents", "info", "cbf_boundary_left_raw"),
            ("agents", "info", "pos"),
            ("agents", "action"),
        ):
            tensor = _get_tensor(key)
            if tensor is not None and tensor.ndim >= 3:
                return int(tensor.shape[2])
        return None

    def _normalize_vehicle_ids(n_agents: int):
        if n_agents is None or n_agents <= 0:
            return []
        if vehicle_ids is None:
            return list(range(n_agents))
        if isinstance(vehicle_ids, str):
            vehicle_ids_str = vehicle_ids.strip().lower()
            if vehicle_ids_str == "all":
                return list(range(n_agents))
            ids = [int(part.strip()) for part in vehicle_ids.split(",") if part.strip()]
        elif isinstance(vehicle_ids, int):
            ids = [vehicle_ids]
        else:
            ids = [int(idx) for idx in vehicle_ids]
        invalid = [idx for idx in ids if idx < 0 or idx >= n_agents]
        if invalid:
            raise ValueError(
                f"Vehicle id(s) {invalid} are outside valid range [0, {n_agents - 1}]."
            )
        return ids

    def _select_env_agents(tensor: torch.Tensor, label: str):
        if tensor is None:
            _warn(f"{label} is unavailable.")
            return None
        if env_index < 0 or env_index >= int(tensor.shape[0]):
            raise ValueError(
                f"env_index={env_index} is outside valid range [0, {int(tensor.shape[0]) - 1}] for {label}."
            )
        selected = tensor.detach().cpu()[env_index, :, selected_vehicle_ids]
        if selected.ndim == 3 and selected.shape[-1] == 1:
            selected = selected.squeeze(-1)
        return selected

    n_agents = _infer_n_agents()
    selected_vehicle_ids = _normalize_vehicle_ids(n_agents)
    if not selected_vehicle_ids:
        _warn("No vehicles selected.")
        return None

    left_raw = _select_env_agents(
        _get_tensor(("agents", "info", "cbf_boundary_left_raw")),
        "cbf_boundary_left_raw",
    )
    right_raw = _select_env_agents(
        _get_tensor(("agents", "info", "cbf_boundary_right_raw")),
        "cbf_boundary_right_raw",
    )
    left_margin = _select_env_agents(
        _get_tensor(("agents", "info", "cbf_boundary_left_margin")),
        "cbf_boundary_left_margin",
    )
    right_margin = _select_env_agents(
        _get_tensor(("agents", "info", "cbf_boundary_right_margin")),
        "cbf_boundary_right_margin",
    )

    if all(
        series is None for series in (left_raw, right_raw, left_margin, right_margin)
    ):
        _warn("No CBF boundary diagnostic fields found in the rollout.")
        return None

    dt = float(getattr(parameters, "dt", 1.0) if parameters is not None else 1.0)
    lane_distance_type = getattr(parameters, "lane_distance_type", "unknown")
    lane_cbf_footprint = (
        parameters.lane_cbf_footprint if parameters is not None else "unknown"
    )
    barrier_name = f"{lane_cbf_footprint} ({lane_distance_type})"
    if lane_cbf_footprint == "rectangle_sat":
        barrier_name = lane_cbf_footprint
        raw_ylabel = "Unbuffered SAT barrier [m]"
        margin_ylabel = "SAT barrier with buffers [m]"
    elif lane_distance_type == "softmin":
        raw_ylabel = "Raw softmin squared value"
        margin_ylabel = "Softmin safety margin"
    elif lane_distance_type == "pseudo":
        raw_ylabel = "Raw pseudo distance [m]"
        margin_ylabel = "Pseudo safety margin [m]"
    else:
        raw_ylabel = "Raw CBF boundary value"
        margin_ylabel = "CBF safety margin"

    plot_colors = colormaps.get_cmap("tab10")
    plot_colors = [plot_colors(i % 10) for i in range(len(selected_vehicle_ids))]

    def _plot_series(ax, series: torch.Tensor, title: str, ylabel: str):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        if series is None:
            ax.text(
                0.5,
                0.5,
                "Unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
            )
            return
        series_np = series.numpy()
        time_axis = np.arange(series_np.shape[0], dtype=np.float32) * dt
        for idx_vehicle, vehicle_idx in enumerate(selected_vehicle_ids):
            ax.plot(
                time_axis,
                series_np[:, idx_vehicle],
                linewidth=1.6,
                color=plot_colors[idx_vehicle],
                label=f"veh {vehicle_idx}",
            )
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    _plot_series(axes[0, 0], left_raw, "Left Boundary Raw", raw_ylabel)
    _plot_series(axes[0, 1], right_raw, "Right Boundary Raw", raw_ylabel)
    _plot_series(axes[1, 0], left_margin, "Left Boundary Margin", margin_ylabel)
    _plot_series(axes[1, 1], right_margin, "Right Boundary Margin", margin_ylabel)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 1].set_xlabel("Time [s]")

    vehicle_tag = (
        "all"
        if len(selected_vehicle_ids) == n_agents
        else "-".join(str(idx) for idx in selected_vehicle_ids)
    )
    fig.suptitle(
        f"CBF boundary diagnostics | env {env_index} | vehicles {vehicle_tag} | lane barrier: {barrier_name}",
        fontsize=13,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if labels:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(5, len(labels)),
            bbox_to_anchor=(0.5, 0.97),
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if save_dir is None:
        save_dir = (
            getattr(parameters, "where_to_save", ".") if parameters is not None else "."
        )
    os.makedirs(save_dir, exist_ok=True)

    file_root, file_ext = os.path.splitext(fig_name)
    if not file_ext:
        fig_name = f"{file_root}_env{env_index}_veh{vehicle_tag}.png"

    fig_path = os.path.join(save_dir, fig_name)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved CBF boundary diagnostics figure to: {fig_path}")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


class Tee:
    """Write output to several streams (e.g., the terminal and a log file) at once."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(stream.isatty() for stream in self._streams)


def start_run_log(save_dir: str, mode: str) -> None:
    """Save the command line to `command_<mode>.txt` and mirror output to `log_<mode>.txt`.

    `mode` is `"training"` or `"testing"`, so that training and testing runs that share
    a model folder do not overwrite each other's files. A repeated run of the same mode
    overwrites the previous files of that mode.

    The command is prefixed with `SIGMARL_AGENT_PROFILE=<active profile>`: the variable
    is not part of `sys.argv` but determines the vehicle profile.

    Both files are written to `save_dir` (created if missing). Output printed before
    this call is not captured. The log stays open until the process exits.
    """
    if mode not in ("training", "testing"):
        raise ValueError(f"mode must be 'training' or 'testing', got {mode!r}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"command_{mode}.txt"), "w") as file:
        file.write(
            f"SIGMARL_AGENT_PROFILE={ACTIVE_AGENT_PROFILE} "
            + shlex.join(["python"] + sys.argv)
            + "\n"
        )
    log_file = open(os.path.join(save_dir, f"log_{mode}.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
