# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from collections import deque
import math
import numpy as np
import time
import os
from termcolor import colored, cprint

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Dict

from sigmarl.scenarios.observations.observation_provider_rt import (
    RoadTrafficObservationProviderSimulation,
    ObservationProviderParametersRT,
)

from sigmarl.scenarios.world_state.world_state_rt.world_state_rt import (
    WorldStateRTParameters,
)
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt_sim import (
    WorldStateRTSimulation,
    WorldStateRTSimParameters,
)

cicd_testing = os.getenv("CICD_TESTING", "false").lower() == "true"

_RESET_RENDER_INFO_DURATION = 3.0
_RESET_RENDER_INFO_FADE_DURATION = 1.0

try:
    if not cicd_testing:
        from sigmarl.render_backend import import_vmas_rendering

        rendering = import_vmas_rendering()
    else:
        rendering = None
except Exception:
    # Covers missing OpenGL or other backend errors
    rendering = None

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, World
from vmas.simulator.scenario import BaseScenario

from sigmarl.dynamics import KinematicBicycleModel

from sigmarl.colors import Color, colors

from sigmarl.helper_common import (
    Parameters,
    Vehicle,
    get_n_colors_cmap,
    make_heading_chevron,
    select_short_term_reference_point,
)
from sigmarl.helper_training import WorldCustom, schedule_rew_coefficient
from sigmarl.rectangle_approximation import RectangleCircleApproximation

from sigmarl.helper_scenario import (
    Normalizers,
    Penalties,
    RewardInfo,
    Rewards,
    Thresholds,
    Timer,
    Constants,
    StateBuffer,
    InitialStateBuffer,
    decreasing_fcn,
    angle_eliminate_two_pi,
    get_progress_on_polyline,
)

from sigmarl.map_manager import MapManager
from sigmarl.render_backend import (
    RENDER_SIZE_MULTIPLE,
    RENDER_VEHICLE_PX,
    compute_text_scale,
    compute_viewer_size,
)

from sigmarl.constants import (
    SCENARIOS,
    AGENTS,
    LENGTH_SCALE,
    POSITION_SPEED_LATERAL_OFFSET_BOUND,
    THRESHOLD,
    validate_scenario_agent_profile,
)

from sigmarl.pseudo_distance import PseudoDistance
from sigmarl.softmin_distance import SoftminDistance
from sigmarl.lane_sat_barrier import LaneSatBarrier
from sigmarl.vehicle_params import VEHICLE_PARAM_KEYS, VehicleParams


# Reward terms in the info of every step, also without is_save_detailed_info:
# the parameter-independent reward and the task terms that the training task
# metrics are computed from (MAPPOCAVs._compute_task_metrics)
TASK_METRIC_REWARDS = (
    "rew_parameter_independent",
    "rew_progress",
    "rew_reach_goal",
    "rew_collide_other_agents",
    "rew_collide_lane",
    "rew_deadlock",
)


class ScenarioRoadTraffic(BaseScenario):
    """
    This scenario aims to design an MARL framework with information-dense observation design to enable fast training and to empower agents the ability to generalize to unseen scenarios.

    We propose five observation-design strategies. They correspond to five parameters in this file, and their default
    values are True.
        - is_ego_view: Whether to use ego view (otherwise bird view)
        - is_observe_distance_to_agents: Whether to observe the distance to other agents
        - is_observe_distance_to_boundaries: Whether to additionally observe distances to lanelet boundaries
        - is_observe_distance_to_center_line: Whether to observe the distance to reference path (otherwise None)
        - is_observe_vertices: Whether to observe the vertices of other agents (otherwise center points)

    In addition, there are some commonly used parameters you may want to adjust to suit your case:
        - n_agents: Number of agents
        - dt: Sample time in seconds
        - scenario_type: One of {"cpm_entire", "cpm_mixed", "intersection_1", ...}. See SCENARIOS in utilities/constants.py for more scenarios.
                         "cpm_entire": the entire CPM map will be used
                         "cpm_mixed": a specific part of the CPM map (intersection, merge-in, or merge-out) will be used for each env when making or resetting it. You can control the probability of using each of them by the parameter `scenario_probabilities`. It is an array with three values. The first value corresponds to the probability of using intersection. The second and the third values correspond to merge-in and merge-out, respectively. If you only want to use one specific part of the map for all parallel envs, you can set the other two values to zero. For example, if you want to train a RL policy only for intersection, they can set `scenario_probabilities` to [1.0, 0.0, 0.0].
                         "intersection_1": the intersection scenario with ID 1
        - is_partial_observation: Whether to enable partial observation (to model partially observable MDP)
        - n_nearing_agents_observed: Number of nearing agents to be observed (consider limited sensor range)

        is_testing_mode: Testing mode is designed to test the learned policy.
                         In non-testing mode, once a collision occurs, all agents will be reset with random initial states.
                         To ensure these initial states are feasible, the initial positions are conservatively large (1.2*diagonalLengthOfAgent).
                         This ensures agents are initially safe and avoids putting agents in an immediate dangerous situation at the beginning of a new scenario.
                         During testing, the
                         `is_reset_all_agents_in_test_mode_when_collision`
                         parameter controls whether a collision resets the
                         complete sub-environment or only the colliding and
                         configured nearby agents.
                         This may allow for more effective testing.

    For other parameters, see the class Parameter defined in this file.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self._init_params(batch_dim, device, **kwargs)
        world = self._init_world(batch_dim, device)
        # VMAS would draw the world boundary below `extra_render`, i.e., below
        # the road surface; `_render_lanelets` draws it instead
        self.visualize_semidims = False
        world.parameters = self.parameters
        self._init_agents(world)
        self.world_state.init_world(world)
        # Per-agent vehicle parameters [n_agents, len(VEHICLE_PARAM_KEYS)]
        # (see set_vehicle_params); every agent starts with the nominal ones
        self.vehicle_param_table = np.tile(
            VehicleParams.nominal().row(), (self.n_agents, 1)
        )
        self.is_heterogeneous_vehicles = False
        return world

    def _init_params(self, batch_dim, device, **kwargs):
        """
        Initialize parameters.
        """
        scenario_type = kwargs.pop(
            "scenario_type",
            "roundabout_2",  # TODO: currently this is hard coded. Do not simply change it to other scenarios. The observation design may not be compatible with other scenarios. For example, lane_width is used as a normalization factor when observing the distance to lanelet boundaries. If the lane_width is different in the testing scenarios, the observation no longer makes sense.
        )  # Scenario type such as cpm_entire, cpm_mixed, cpm_intersection_extended, interchange_1, intersection_1, etc. See all scenario types in SCENARIOS in utilities/constants
        self.agent_width = AGENTS["width"]  # The width of the agent in [m]
        self.agent_length = AGENTS["length"]  # The length of the agent in [m]

        # Reward
        r_p_normalizer = (
            100  # This parameter normalizes rewards and penalties to [-1, 1].
        )
        # This is useful for RL algorithms with an actor-critic architecture where the critic's
        # output is limited to [-1, 1] (e.g., due to tanh activation function).

        if hasattr(self, "parameters"):
            is_sample_equal_distance = getattr(
                self.parameters, "is_sample_equal_distance", True
            )
            reward_progress = (
                self.parameters.reward_progress
                if self.parameters.reward_progress is not None
                else 0.1
            )
            threshold_near_boundary_high = (
                self.parameters.threshold_near_boundary_high
                if self.parameters.threshold_near_boundary_high is not None
                else 0.02
            )
            threshold_near_boundary_low = (
                self.parameters.threshold_near_boundary_low
                if self.parameters.threshold_near_boundary_low is not None
                else 0
            )
            threshold_near_other_agents_c2c_high = (
                self.parameters.threshold_near_other_agents_c2c_high
                if self.parameters.threshold_near_other_agents_c2c_high is not None
                else 0.3
            )
            threshold_near_other_agents_c2c_low = (
                self.parameters.threshold_near_other_agents_c2c_low
                if self.parameters.threshold_near_other_agents_c2c_low is not None
                else 0
            )
            ttc_low = (
                self.parameters.ttc_low if self.parameters.ttc_low is not None else 0
            )
            ttc_high = (
                self.parameters.ttc_high
                if self.parameters.ttc_high is not None
                else 3.75
            )
            penalty_near_boundary = (
                self.parameters.penalty_near_boundary
                if self.parameters.penalty_near_boundary is not None
                else -0.2
            )
            penalty_near_other_agents = (
                self.parameters.penalty_near_other_agents
                if self.parameters.penalty_near_other_agents is not None
                else -0.2
            )
            penalty_deviate_from_cbf_vel = (
                self.parameters.penalty_deviate_from_cbf_vel
                if self.parameters.penalty_deviate_from_cbf_vel is not None
                else -0.05
            )
            penalty_deviate_from_cbf_steer = (
                self.parameters.penalty_deviate_from_cbf_steer
                if self.parameters.penalty_deviate_from_cbf_steer is not None
                else -0.05
            )
            penalty_deadlock = (
                self.parameters.penalty_deadlock
                if getattr(self.parameters, "penalty_deadlock", None) is not None
                else -1.0
            )
            penalty_near_collision_reset = (
                self.parameters.penalty_near_collision_reset
                if getattr(self.parameters, "penalty_near_collision_reset", None)
                is not None
                else -0.5
            )
            penalty_jerk_acceleration = (
                self.parameters.penalty_jerk_acceleration
                if getattr(self.parameters, "penalty_jerk_acceleration", None)
                is not None
                else -0.02
            )
            penalty_jerk_steering_rate = (
                self.parameters.penalty_jerk_steering_rate
                if getattr(self.parameters, "penalty_jerk_steering_rate", None)
                is not None
                else -0.02
            )
            penalty_high_acceleration = (
                self.parameters.penalty_high_acceleration
                if getattr(self.parameters, "penalty_high_acceleration", None)
                is not None
                else -0.02
            )
            penalty_high_steering_rate = (
                self.parameters.penalty_high_steering_rate
                if getattr(self.parameters, "penalty_high_steering_rate", None)
                is not None
                else -0.02
            )
        else:
            is_sample_equal_distance = kwargs.pop("is_sample_equal_distance", True)
            reward_progress = (
                kwargs.pop("reward_progress", 10) / r_p_normalizer
            )  # Reward for moving along reference paths

            threshold_near_boundary_high = kwargs.pop(
                "threshold_near_boundary_high",
                self.agent_width / 5,
            )  # Threshold beneath which agents will started be
            # Penalized for being too close to lanelet boundaries
            threshold_near_boundary_low = kwargs.pop(
                "threshold_near_boundary_low", 0
            )  # Threshold above which agents will be penalized for being too close to lanelet boundaries

            threshold_near_other_agents_c2c_high = kwargs.pop(
                "threshold_near_other_agents_c2c_high",
                self.agent_length + self.agent_width,
            )  # Threshold beneath which agents will started be
            # Penalized for being too close to other agents (for center-to-center distance)
            threshold_near_other_agents_c2c_low = kwargs.pop(
                "threshold_near_other_agents_c2c_low",
                (self.agent_length + self.agent_width) / 2,
            )  # Threshold above which agents will be penalized (for center-to-center distance,
            ttc_low = kwargs.pop(
                "ttc_low",
                0,
            )  # Threshold above which agents will be penalized
            ttc_high = kwargs.pop(
                "ttc_high",
                3.75,
            )  # Threshold above which agents will be penalized
            penalty_near_boundary = kwargs.pop(
                "penalty_near_boundary", -20 / r_p_normalizer
            )  # Penalty for being too close to lanelet boundaries
            penalty_near_other_agents = kwargs.pop(
                "penalty_near_other_agents", -20 / r_p_normalizer
            )  # Penalty for being too close to other agents
            penalty_deviate_from_cbf_vel = kwargs.pop(
                "penalty_deviate_from_cbf_vel", -5 / r_p_normalizer
            )
            penalty_deviate_from_cbf_steer = kwargs.pop(
                "penalty_deviate_from_cbf_steer", -5 / r_p_normalizer
            )
            penalty_deadlock = kwargs.get("penalty_deadlock", -1.0)
            penalty_near_collision_reset = kwargs.pop(
                "penalty_near_collision_reset", -0.5
            )
            penalty_jerk_acceleration = kwargs.get("penalty_jerk_acceleration", -0.02)
            penalty_jerk_steering_rate = kwargs.get("penalty_jerk_steering_rate", -0.02)
            penalty_high_acceleration = kwargs.get("penalty_high_acceleration", -0.02)
            penalty_high_steering_rate = kwargs.get("penalty_high_steering_rate", -0.02)
            is_visualize_observed_agents = kwargs.pop(
                "is_visualize_observed_agents", True
            )  # Whether to visualize observed agents for a selected ego agent

        reward_vel = (
            kwargs.pop("reward_vel", 5) / r_p_normalizer
        )  # Reward for moving in high velocities.
        reward_reach_goal = (
            kwargs.pop("reward_reach_goal", 100) / r_p_normalizer
        )  # Goal-reaching reward

        # Penalty
        penalty_deviate_from_ref_path = kwargs.pop(
            "penalty_deviate_from_ref_path", -2 / r_p_normalizer
        )  # Penalty for deviating from reference paths
        penalty_collide_with_agents = kwargs.pop(
            "penalty_collide_with_agents", -100 / r_p_normalizer
        )  # Penalty for colliding with other agents
        penalty_collide_with_boundaries = kwargs.pop(
            "penalty_collide_with_boundaries", -100 / r_p_normalizer
        )  # Penalty for colliding with lanelet boundaries
        penalty_change_steering = kwargs.pop(
            "penalty_change_steering", -2 / r_p_normalizer
        )  # Penalty for changing steering too quick
        penalty_time = kwargs.pop(
            "penalty_time", 5 / r_p_normalizer
        )  # Penalty for losing time

        threshold_deviate_from_ref_path = kwargs.pop(
            "threshold_deviate_from_ref_path", self.agent_width / 4
        )  # Use for penalizing of deviating from reference path

        threshold_reach_goal = kwargs.pop(
            "threshold_reach_goal", self.agent_width / 2
        )  # Threshold less than which agents are considered at their goal positions

        threshold_change_steering = kwargs.pop(
            "threshold_change_steering", 10
        )  # Threshold above which agents will be penalized for changing steering too quick [degree]

        threshold_no_reward_if_too_close_to_boundaries = kwargs.pop(
            "threshold_no_reward_if_too_close_to_boundaries", self.agent_width / 10
        )
        threshold_no_reward_if_too_close_to_other_agents = kwargs.pop(
            "threshold_no_reward_if_too_close_to_other_agents", self.agent_width / 6
        )

        threshold_near_other_agents_MTV_low = kwargs.pop(
            "threshold_near_other_agents_MTV_low", 0
        )

        threshold_near_other_agents_MTV_high = kwargs.pop(
            "threshold_near_other_agents_MTV_high", self.agent_length
        )

        # Reference path
        short_term_ref_path_spacing_m = kwargs.pop(
            "short_term_ref_path_spacing_m",
            getattr(self.parameters, "short_term_ref_path_spacing_m", None)
            if hasattr(self, "parameters")
            else None,
        )
        if short_term_ref_path_spacing_m is None:
            short_term_ref_path_spacing_m = self.agent_length
        is_smooth_sampling = kwargs.pop(
            "is_smooth_sampling",
            getattr(self.parameters, "is_smooth_sampling", True)
            if hasattr(self, "parameters")
            else True,
        )
        is_use_simple_road_distance = kwargs.pop(
            "is_use_simple_road_distance",
            getattr(self.parameters, "is_use_simple_road_distance", True)
            if hasattr(self, "parameters")
            else True,
        )
        max_ref_path_points = kwargs.pop(
            "max_ref_path_points", 200
        )  # The estimated maximum points on the reference path

        n_stored_steps = kwargs.pop(
            "n_stored_steps",
            5,  # The number of steps to store (include the current step). At least one
        )
        n_observed_steps = kwargs.pop(
            "n_observed_steps", 1
        )  # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`

        self.max_steering = kwargs.pop(
            "max_steering",
            torch.tensor(AGENTS["max_steering"], device=device, dtype=torch.float32),
        )  # Maximum allowed steering angle in degree
        max_speed = kwargs.pop("max_speed", None)  # Overrides parameters.max_speed
        self.min_speed = kwargs.pop(
            "min_speed", AGENTS["min_speed"]
        )  # Minimum allowed speed in [m/s]

        n_points_nearing_boundary = kwargs.pop(
            "n_points_nearing_boundary",
            getattr(self.parameters, "n_points_nearing_boundary", 8)
            if hasattr(self, "parameters")
            else 8,
        )  # The number of points on nearing boundaries to be observed

        probability_record = kwargs.pop("probability_record", 1.0)
        probability_use_recording = kwargs.pop("probability_use_recording", 0.2)
        buffer_size = kwargs.pop("buffer_size", 100)

        if not hasattr(self, "parameters"):
            self.parameters = Parameters(
                n_agents=kwargs.pop("n_agents", SCENARIOS[scenario_type]["n_agents"]),
                scenario_type=scenario_type,
                is_partial_observation=kwargs.pop("is_partial_observation", True),
                max_steps=kwargs.pop("max_steps", 1024),
                is_testing_mode=kwargs.pop("is_testing_mode", True),
                is_joint_task=kwargs.pop("is_joint_task", False),
                joint_task_complete_duration=kwargs.pop(
                    "joint_task_complete_duration", 12.0
                ),
                is_visualize_short_term_path=kwargs.pop(
                    "is_visualize_short_term_path", True
                ),
                n_nearing_agents_observed=kwargs.pop("n_nearing_agents_observed", 2),
                is_real_time_rendering=kwargs.pop("is_real_time_rendering", False),
                n_steps_stored=kwargs.pop("n_steps_stored", 10),
                n_points_short_term=kwargs.pop("n_points_short_term", 3),
                short_term_ref_path_spacing_m=short_term_ref_path_spacing_m,
                n_points_nearing_boundary=n_points_nearing_boundary,
                is_smooth_sampling=is_smooth_sampling,
                is_use_simple_road_distance=is_use_simple_road_distance,
                dt=kwargs.pop("dt", 0.05),
                is_ego_view=kwargs.pop("is_ego_view", True),
                is_observe_vertices=kwargs.pop("is_observe_vertices", True),
                is_observe_distance_to_agents=kwargs.pop(
                    "is_observe_distance_to_agents", True
                ),
                is_observe_distance_to_boundaries=kwargs.pop(
                    "is_observe_distance_to_boundaries", True
                ),
                is_observe_distance_to_center_line=kwargs.pop(
                    "is_observe_distance_to_center_line", True
                ),
                is_apply_distance_mask=kwargs.pop("is_apply_distance_mask", True),
                is_apply_velocity_mask=kwargs.pop("is_apply_velocity_mask", True),
                rl_obs_range=kwargs.pop("rl_obs_range", None),
                visible_agent_dropout_probability=kwargs.pop(
                    "visible_agent_dropout_probability", 0.05
                ),
                is_challenging_initial_state_buffer=kwargs.pop(
                    "is_challenging_initial_state_buffer", False
                ),
                cpm_scenario_probabilities=kwargs.pop(
                    "cpm_scenario_probabilities", [1.0, 0.0, 0.0]
                ),  # Probabilities of training agents in intersection, merge-in, and merge-out scenario
                is_obs_noise=kwargs.pop("is_obs_noise", True),
                obs_noise_level=kwargs.pop(
                    "obs_noise_level", 0.2 * self.agent_width
                ),  # Noise will be generated by the standary normal distribution. This parameter controls the noise level
                is_observe_ref_path_other_agents=kwargs.pop(
                    "is_observe_ref_path_other_agents", False
                ),
                is_visualize_extra_info=kwargs.pop("is_visualize_extra_info", False),
                render_title=kwargs.pop(
                    "render_title",
                    "Multi-Agent Reinforcement Learning for Road Traffic (CPM Lab Scenario)",
                ),
                is_using_opponent_modeling=kwargs.pop(
                    "is_using_opponent_modeling", False
                ),
                is_use_mtv_distance=kwargs.pop("is_use_mtv_distance", False),
                is_using_cbf_testing=kwargs.pop("is_using_cbf_testing", False),
                is_using_cbf_training=kwargs.pop("is_using_cbf_training", False),
                is_using_centralized_cbf=kwargs.pop("is_using_centralized_cbf", False),
                experiment_type=kwargs.pop("experiment_type", "simulation"),
                is_obs_steering=kwargs.pop("is_obs_steering", False),
                is_visualize_lane_boundary=kwargs.pop(
                    "is_visualize_lane_boundary", True
                ),
                is_visu_sampled_boundary_points=kwargs.pop(
                    "is_visu_sampled_boundary_points", False
                ),
                is_visu_applied_action=kwargs.pop("is_visu_applied_action", True),
                is_visu_nom_action=kwargs.pop("is_visu_nom_action", True),
                is_visu_shield_rollout=kwargs.pop("is_visu_shield_rollout", True),
                reset_agent_fixed_duration=kwargs.pop("reset_agent_fixed_duration", 0),
                is_reset_all_agents_in_test_mode_when_collision=kwargs.pop(
                    "is_reset_all_agents_in_test_mode_when_collision", True
                ),
                reset_agent_surrounding_distance=kwargs.pop(
                    "reset_agent_surrounding_distance", None
                ),
                is_deadlock_enabled=kwargs.pop("is_deadlock_enabled", False),
                deadlock_duration=kwargs.pop("deadlock_duration", 3.0),
                deadlock_progress_threshold=kwargs.pop(
                    "deadlock_progress_threshold", None
                ),
                penalty_deadlock=kwargs.pop("penalty_deadlock", -1.0),
                penalty_near_collision_reset=penalty_near_collision_reset,
                is_schedule_rew_coefficient=kwargs.pop(
                    "is_schedule_rew_coefficient", True
                ),
                is_sample_equal_distance=is_sample_equal_distance,
                is_center_centerline=kwargs.pop("is_center_centerline", True),
                scale_map=kwargs.pop("scale_map", 1.0),
            )

        if max_speed is not None:
            self.parameters.max_speed = max_speed
        self.parameters.resolve_speed_defaults()
        self.max_speed = self.parameters.max_speed  # Maximum allowed speed in [m/s]
        print(
            f"[INFO] Speed limit {self.max_speed:.1f} m/s, policy speed scale "
            f"{self.parameters.speed_normalization:.1f} m/s, CLF reference speed "
            f"{self.parameters.ref_speed:.1f} m/s, RL observation range "
            f"{self.parameters.rl_obs_range:.1f} m, CBF coupling range "
            f"{self.parameters.cbf_coupling_range:.1f} m."
        )

        self.parameters.short_term_ref_path_spacing_m = short_term_ref_path_spacing_m
        self.parameters.n_points_nearing_boundary = n_points_nearing_boundary
        self.parameters.is_smooth_sampling = is_smooth_sampling
        self.parameters.is_use_simple_road_distance = is_use_simple_road_distance
        self.n_agents = self.parameters.n_agents

        # Optional video-only hook. When configured, done() captures a
        # collision frame before resetting the world; it never changes the
        # transition or reset logic.
        self._pre_reset_render_callback = None
        self._pre_reset_render_frame = None
        # Learned CBF lambdas of the last CBF-QP step, per env, for rendering.
        # Each row is the ego's list of (agent index, lambda) entries.
        self._cbf_lambda_render_rows = {}
        self._shield_rollout_render = {}
        # Width (px) of a white panel added right of the map for text overlays
        self.render_panel_width = 0
        self._render_panel_world_box = None
        self._warned_fast_inference_non_loop = False
        if getattr(self.parameters, "deadlock_progress_threshold", None) is None:
            self.parameters.deadlock_progress_threshold = 5e-3 * LENGTH_SCALE
        if self.n_agents > len(colors):
            self.colors = get_n_colors_cmap(self.n_agents)
        else:
            self.colors = colors

        # Logs
        if self.parameters.is_testing_mode:
            print(colored(f"[INFO] Testing mode", "red"))
        if self.parameters.is_fast_inference_mode:
            print(colored(f"[INFO] Fast inference mode", "blue"))
        print(colored(f"[INFO] Scenario type: {self.parameters.scenario_type}", "red"))
        if self.parameters.is_prb:
            print(colored("[INFO] Enable prioritized replay buffer", "red"))
        if self.parameters.is_challenging_initial_state_buffer:
            print(colored("[INFO] Enable challenging initial state buffer", "red"))
        if self.parameters.is_using_opponent_modeling:
            print(colored("[INFO] Using opponent modeling", "red"))
        if self.parameters.is_using_prioritized_marl:
            if self.parameters.prioritization_method == "marl":
                print(
                    colored(
                        "[INFO] Using prioritized MARL with MARL-generated priorities",
                        "red",
                    )
                )
            elif self.parameters.prioritization_method == "random":
                print(
                    colored(
                        "[INFO] Using prioritized MARL with random priorities", "red"
                    )
                )
            elif self.parameters.prioritization_method == "fixed":
                print(
                    colored(
                        "[INFO] Using prioritized MARL with fixed priorities", "red"
                    )
                )
            elif self.parameters.prioritization_method == "taskprogress":
                print(
                    colored(
                        "[INFO] Using prioritized MARL with taskprogress priorities",
                        "red",
                    )
                )
            elif self.parameters.prioritization_method == "taskprogress_permute":
                print(
                    colored(
                        "[INFO] Using prioritized MARL with permuted taskprogress priorities",
                        "red",
                    )
                )
            else:
                raise ValueError(
                    f"The given prioritization method is not supported. Obtained: {self.parameters.prioritization_method}. Expected: 'marl', 'random', 'fixed', 'taskprogress', or 'taskprogress_permute'."
                )

        cbf_type_1 = (
            "centralized"
            if self.parameters.is_using_centralized_cbf
            else "decentralized"
        )

        if (
            self.parameters.is_using_cbf_training
            and self.parameters.is_using_cbf_testing
        ):
            cbf_type_2 = "training and testing"
        elif (
            self.parameters.is_using_cbf_training
            and not self.parameters.is_using_cbf_testing
        ):
            cbf_type_2 = "training"
        elif (
            not self.parameters.is_using_cbf_training
            and self.parameters.is_using_cbf_testing
        ):
            cbf_type_2 = "testing"
        else:
            cbf_type_2 = None

        if (
            self.parameters.is_using_cbf_training
            or self.parameters.is_using_cbf_testing
        ):
            print(
                colored(
                    f"[INFO] Using CBF-constrained MARL with {cbf_type_1} solving during {cbf_type_2}",
                    "red",
                )
            )

        if self.parameters.is_exact_backup_shield:
            print(colored("[INFO] Using the exact backup shield", "red"))

        if self.parameters.reset_agent_fixed_duration > 0:
            print(
                colored(
                    f"[INFO] Resetting agents after each {self.parameters.reset_agent_fixed_duration} seconds!!!",
                    "red",
                )
            )

        if self.parameters.n_nearing_agents_observed == 0:
            cprint("[INFO] Not observing other agents.", "blue")
        self.n_agents = self.parameters.n_agents

        # Timer for the first env
        self.timer = Timer(
            start=time.time(),
            end=0,
            step=torch.zeros(
                batch_dim, device=device, dtype=torch.int32
            ),  # Each environment has its own time step
            step_duration=torch.zeros(
                self.parameters.max_steps, device=device, dtype=torch.float32
            ),
            step_begin=time.time(),
            render_begin=0,
        )

        # Get map data
        validate_scenario_agent_profile(self.parameters.scenario_type)
        print(f"[INFO] lane_width: {self.parameters.lane_width}")
        self.map = MapManager(
            scenario_type=self.parameters.scenario_type,
            device=device,
            lane_width=self.parameters.lane_width,  # For custom scenarios only
            is_sample_equal_distance=self.parameters.is_sample_equal_distance,
            is_center_centerline=self.parameters.is_center_centerline,
            scale_map=getattr(self.parameters, "scale_map", 1.0),
        )
        cprint(f"[INFO] Map of {self.parameters.scenario_type} parsed.", "blue")

        # World dimensions
        self.world_x_dim = self.map.parser.bounds["world_x_dim"]
        self.world_y_dim = self.map.parser.bounds["world_y_dim"]

        self.render_origin = [
            self.world_x_dim / 2,
            self.world_y_dim / 2,
        ]

        self.set_render_resolution()

        # Initialize the lane-boundary geometry used by the lane CBF. The
        # rectangle_sat barrier uses the map polyline directly and needs no
        # pseudo or softmin distance.
        self.map_pseudo_distance = None
        self.lane_sat_barrier = None
        if (
            self.parameters.is_using_cbf_testing
            or self.parameters.is_using_cbf_training
        ):
            if self.parameters.lane_cbf_footprint == "rectangle_sat":
                self.lane_sat_barrier = LaneSatBarrier(
                    length=AGENTS["length"],
                    width=AGENTS["width"],
                    safety_buffer=0.01 * AGENTS["width"],  # As CBFQP.safety_buffer
                )
            elif self.parameters.lane_distance_type == "pseudo":
                self.map_pseudo_distance = PseudoDistance(
                    scenario_type=self.parameters.scenario_type, map=self.map
                )
            elif self.parameters.lane_distance_type == "softmin":
                self.map_pseudo_distance = SoftminDistance(
                    scenario_type=self.parameters.scenario_type,
                    map=self.map,
                    parameters=self.parameters,
                )
            else:
                raise ValueError(
                    "Unsupported lane_distance_type: "
                    f"{self.parameters.lane_distance_type}"
                )

        # Determine the maximum number of points on the reference path
        if self.parameters.scenario_type == "cpm_intersection_extended":
            reference_paths_for_max = (
                self.map.parser.reference_paths_intersection_extended
            )
        elif "cpm_mixed" in self.parameters.scenario_type:
            # Mixed scenarios including intersection, merge in, and merge out
            reference_paths_for_max = (
                self.map.parser.reference_paths_intersection
                + self.map.parser.reference_paths_merge_in
                + self.map.parser.reference_paths_merge_out
            )
        else:
            # Single scenario
            reference_paths_for_max = self.map.parser.reference_paths
        max_ref_path_points = (
            max(ref_p["center_line"].shape[0] for ref_p in reference_paths_for_max)
            + self.parameters.n_points_short_term
            + 2
        )  # Append a smaller buffer

        """
        --------------------------------- REWARDS ---------------------------------
        """

        weighting_ref_directions = torch.linspace(
            1,
            0.2,
            steps=self.parameters.n_points_short_term,
            device=device,
            dtype=torch.float32,
        )
        weighting_ref_directions /= weighting_ref_directions.sum()

        self.rewards = Rewards(
            progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions,
            # Progress in the weighted directions (directions indicating by closer short-term reference points have higher weights)
            higth_v=torch.tensor(reward_vel, device=device, dtype=torch.float32),
            reach_goal=torch.tensor(
                reward_reach_goal, device=device, dtype=torch.float32
            ),
        )

        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        self.reward_info = RewardInfo.zeros(batch_dim, self.n_agents, device)
        self._collision_participant_mask = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.bool
        )
        self._near_collision_reset_mask = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.bool
        )

        self.penalties = Penalties(
            deviate_from_ref_path=torch.tensor(
                penalty_deviate_from_ref_path, device=device, dtype=torch.float32
            ),
            near_boundary=torch.tensor(
                penalty_near_boundary, device=device, dtype=torch.float32
            ),
            near_other_agents=torch.tensor(
                penalty_near_other_agents, device=device, dtype=torch.float32
            ),
            collide_with_agents=torch.tensor(
                penalty_collide_with_agents, device=device, dtype=torch.float32
            ),
            near_collision_reset=torch.tensor(
                penalty_near_collision_reset, device=device, dtype=torch.float32
            ),
            collide_with_boundaries=torch.tensor(
                penalty_collide_with_boundaries, device=device, dtype=torch.float32
            ),
            change_steering=torch.tensor(
                penalty_change_steering, device=device, dtype=torch.float32
            ),
            time=torch.tensor(penalty_time, device=device, dtype=torch.float32),
            deviate_from_cbf_vel=torch.tensor(
                penalty_deviate_from_cbf_vel, device=device, dtype=torch.float32
            ),
            deviate_from_cbf_steer=torch.tensor(
                penalty_deviate_from_cbf_steer, device=device, dtype=torch.float32
            ),
            deadlock=torch.tensor(penalty_deadlock, device=device, dtype=torch.float32),
            jerk_acceleration=torch.tensor(
                penalty_jerk_acceleration, device=device, dtype=torch.float32
            ),
            jerk_steering_rate=torch.tensor(
                penalty_jerk_steering_rate, device=device, dtype=torch.float32
            ),
            high_acceleration=torch.tensor(
                penalty_high_acceleration, device=device, dtype=torch.float32
            ),
            high_steering_rate=torch.tensor(
                penalty_high_steering_rate, device=device, dtype=torch.float32
            ),
        )

        self.previous_acceleration = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.float32
        )
        self.previous_steering_rate = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.float32
        )
        self.previous_control_valid = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.bool
        )

        self.normalizers = Normalizers(
            pos=torch.tensor(
                [self.agent_length * 6, self.agent_length * 6],
                device=device,
                dtype=torch.float32,
            ),
            pos_world=torch.tensor(
                [self.world_x_dim, self.world_y_dim], device=device, dtype=torch.float32
            ),
            # Speeds are normalized by the policy's speed scale, which may exceed
            # the speed limit, so that one policy sees the same scaling on maps
            # with different speed limits
            v=torch.tensor(
                self.parameters.speed_normalization, device=device, dtype=torch.float32
            ),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            steering=self.max_steering,
            distance_lanelet=torch.tensor(
                self.agent_width * 2, device=device, dtype=torch.float32
            ),
            distance_ref=torch.tensor(
                self.agent_width * 2, device=device, dtype=torch.float32
            ),
            distance_agent=torch.tensor(
                self.agent_length * 6, device=device, dtype=torch.float32
            ),
        )

        # Distances to boundaries and reference path, and also the closest point on the reference paths of agents
        if self.parameters.is_use_mtv_distance:
            self.distance_type = "mtv"  # One of {"c2c", "mtv"}
        else:
            self.distance_type = "c2c"  # One of {"c2c", "mtv"}
        # print(colored("[INFO] Distance type: ", "black"), colored(distance_type, "blue"))

        configured_reset_distance = getattr(
            self.parameters, "reset_agent_surrounding_distance", 0.0
        )
        if configured_reset_distance is None:
            self.reset_agent_surrounding_distance = (
                4.0 if self.distance_type == "c2c" else 2.0
            ) * THRESHOLD["disk_radius"]
        else:
            self.reset_agent_surrounding_distance = float(configured_reset_distance)

        self.thresholds = Thresholds(
            reach_goal=torch.tensor(
                threshold_reach_goal, device=device, dtype=torch.float32
            ),
            deviate_from_ref_path=torch.tensor(
                threshold_deviate_from_ref_path, device=device, dtype=torch.float32
            ),
            near_boundary_low=torch.tensor(
                threshold_near_boundary_low, device=device, dtype=torch.float32
            ),
            near_boundary_high=torch.tensor(
                threshold_near_boundary_high, device=device, dtype=torch.float32
            ),
            ttc_low=torch.tensor(ttc_low, device=device, dtype=torch.float32),
            ttc_high=torch.tensor(ttc_high, device=device, dtype=torch.float32),
            near_other_agents_low=torch.tensor(
                (
                    threshold_near_other_agents_c2c_low
                    if self.distance_type == "c2c"
                    else threshold_near_other_agents_MTV_low
                ),
                device=device,
                dtype=torch.float32,
            ),
            near_other_agents_high=torch.tensor(
                (
                    threshold_near_other_agents_c2c_high
                    if self.distance_type == "c2c"
                    else threshold_near_other_agents_MTV_high
                ),
                device=device,
                dtype=torch.float32,
            ),
            change_steering=torch.tensor(
                threshold_change_steering, device=device, dtype=torch.float32
            ).deg2rad(),
            no_reward_if_too_close_to_boundaries=torch.tensor(
                threshold_no_reward_if_too_close_to_boundaries,
                device=device,
                dtype=torch.float32,
            ),
            no_reward_if_too_close_to_other_agents=torch.tensor(
                threshold_no_reward_if_too_close_to_other_agents,
                device=device,
                dtype=torch.float32,
            ),
            rl_obs_range=self.parameters.rl_obs_range,
        )

        self.constants = Constants(
            env_idx_broadcasting=torch.arange(
                batch_dim, device=device, dtype=torch.int32
            ).unsqueeze(-1),
            empty_action_vel=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            empty_action_steering=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor(
                self.agent_length**2 + self.agent_width**2,
                device=device,
                dtype=torch.float32,
            ).sqrt()
            * 1.5,
        )

        if self.parameters.is_challenging_initial_state_buffer:
            self.initial_state_buffer = InitialStateBuffer(
                # Used only when self.parameters.is_challenging_initial_state_buffer is True
                probability_record=torch.tensor(
                    probability_record, device=device, dtype=torch.float32
                ),
                probability_use_recording=torch.tensor(
                    probability_use_recording, device=device, dtype=torch.float32
                ),
                buffer=torch.zeros(
                    (buffer_size, self.n_agents, 8), device=device, dtype=torch.float32
                ),  # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id]
            )

        # Store the states of agents at previous several time steps
        self.state_buffer = StateBuffer(
            buffer=torch.zeros(
                (self.parameters.n_steps_stored, batch_dim, self.n_agents, 8),
                device=device,
                dtype=torch.float32,
            ),  # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id],
        )

        self._init_deadlock_tracker(batch_dim=batch_dim, device=device)

        # Store the computed observations of each agent for reuse in `info()`
        # to generate `base_obs` and `priority_obs`, which are used for prioritized MARL.
        self.stored_observations = [None] * self.n_agents

        self.world_state = self._make_world_state(
            WorldStateRTSimParameters(
                batch_dim=batch_dim,
                n_agents=self.n_agents,
                device=device,
                distance_type=self.distance_type,
                max_ref_path_points=max_ref_path_points,
                n_points_nearing_boundary=self.parameters.n_points_nearing_boundary,
                n_points_short_term=self.parameters.n_points_short_term,
                short_term_ref_path_spacing_m=short_term_ref_path_spacing_m,
                is_smooth_sampling=self.parameters.is_smooth_sampling,
                is_use_simple_road_distance=self.parameters.is_use_simple_road_distance,
                observe_distance_to_boundaries=self.parameters.is_observe_distance_to_boundaries,
                scenario_type=self.parameters.scenario_type,
                cpm_scenario_probabilities=self.parameters.cpm_scenario_probabilities,
                random_seed=self.parameters.random_seed,
                is_testing_mode=self.parameters.is_testing_mode,
                is_enable_debug_diagnostics=self.parameters.is_enable_debug_diagnostics,
                is_zero_initial_speed=self.parameters.is_zero_initial_speed,
                is_fast_inference_mode=self.parameters.is_fast_inference_mode,
                is_observe_vertices=self.parameters.is_observe_vertices,
            ),
        )
        self.world_state.state_buffer = self.state_buffer

        self.observation_provider = RoadTrafficObservationProviderSimulation(
            ObservationProviderParametersRT(
                batch_dim=batch_dim,
                n_agents=self.n_agents,
                device=device,
                n_stored_steps=n_stored_steps,
                n_observed_steps=n_observed_steps,
                n_points_nearing_boundary=self.parameters.n_points_nearing_boundary,
                n_nearing_agents_observed=self.parameters.n_nearing_agents_observed,
                n_points_short_term=self.parameters.n_points_short_term,
                is_observe_distance_to_boundaries=self.parameters.is_observe_distance_to_boundaries,
                is_observe_boundary_points=self.parameters.is_observe_boundary_points,
                is_ego_view=self.parameters.is_ego_view,
                is_using_opponent_modeling=self.parameters.is_using_opponent_modeling,
                is_apply_distance_mask=self.parameters.is_apply_distance_mask,
                is_apply_velocity_mask=getattr(
                    self.parameters, "is_apply_velocity_mask", True
                ),
                velocity_mask_margin=getattr(
                    self.parameters, "velocity_mask_margin", 0.5
                ),
                rl_obs_range=self.parameters.rl_obs_range,
                visible_agent_dropout_probability=getattr(
                    self.parameters, "visible_agent_dropout_probability", 0.05
                ),
                is_testing_mode=self.parameters.is_testing_mode,
                is_obs_noise=self.parameters.is_obs_noise,
                obs_noise_level=self.parameters.obs_noise_level,
                is_partial_observation=self.parameters.is_partial_observation,
                is_obs_steering=self.parameters.is_obs_steering,
                is_observe_vertices=self.parameters.is_observe_vertices,
                is_observe_distance_to_agents=self.parameters.is_observe_distance_to_agents,
                is_observe_ref_path_other_agents=self.parameters.is_observe_ref_path_other_agents,
                is_observe_distance_to_center_line=self.parameters.is_observe_distance_to_center_line,
                is_fast_inference_mode=self.parameters.is_fast_inference_mode,
            ),
            self.constants,
            self.normalizers,
            self.thresholds,
            self.map,
            self.world_state,
        )

        self.num_task_tries = torch.zeros(
            batch_dim, device=device, dtype=torch.int32
        )  # The total number of tries to complete tasks, initialized to n_agents
        self.task_success_times = torch.zeros(
            batch_dim, device=device, dtype=torch.int32
        )  # The number of successful tasks
        self._last_reset_render_info = [None] * batch_dim
        self._last_reset_render_time = [0.0] * batch_dim
        self._last_reset_render_color = [None] * batch_dim
        self._static_lanelet_geoms = None
        self._joint_task_finished_agents = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.bool
        )
        self._joint_task_exit_event = torch.zeros_like(self._joint_task_finished_agents)
        self._joint_task_success_event = torch.zeros(
            batch_dim, device=device, dtype=torch.bool
        )
        self._joint_task_failure_event = torch.zeros_like(
            self._joint_task_success_event
        )
        self._joint_task_try_event = torch.zeros_like(self._joint_task_success_event)
        self._joint_task_collision_agents_event = torch.zeros_like(
            self._joint_task_finished_agents
        )
        self._joint_task_collision_lanelets_event = torch.zeros_like(
            self._joint_task_finished_agents
        )
        # This is a per-task clock. ``timer.step`` may continue across early
        # joint-task resets so that the overall evaluation horizon is stable.
        self._joint_task_elapsed_steps = torch.zeros(
            batch_dim, device=device, dtype=torch.int32
        )
        # A joint-task terminal event is reset immediately in ``done()`` and
        # then reset once more by VMAS/TorchRL. Preserve the horizon clock
        # across both resets when the task ended before the horizon.
        self._preserve_timer_on_next_reset = torch.zeros(
            batch_dim, device=device, dtype=torch.bool
        )
        self._transition_state_prepared = False
        # Optional `LogReplay` (sigmarl/log_replay.py) that spawns, replays, and parks agents according to
        # recorded tracks. None keeps the usual random spawning and resetting.
        self.log_replay = None

    def _make_world_state(self, params: WorldStateRTSimParameters):
        """Factory seam for the world state.

        Subclasses that need different *reset* semantics override this and
        return their own `WorldStateRTSimulation` subclass. The lab backend
        uses it to derive reference paths from the measured poses of
        hand-placed vehicles instead of sampling them. Everything else
        (collisions, deadlock, state buffer) is shared.
        """
        return WorldStateRTSimulation(params, self.map)

    def _init_world(self, batch_dim: int, device: torch.device):
        # Make world
        world = WorldCustom(
            batch_dim,
            device,
            x_semidim=torch.tensor(
                self.world_x_dim, device=device, dtype=torch.float32
            ),
            y_semidim=torch.tensor(
                self.world_y_dim, device=device, dtype=torch.float32
            ),
            dt=self.parameters.dt,
        )
        return world

    def _init_agents(self, world: World):
        if self.parameters.rl_action_type:
            print(
                colored(
                    f"[INFO] RL action type: {self.parameters.rl_action_type}", "blue"
                )
            )
        # Create agents
        for i in range(self.n_agents):
            agent = Vehicle(
                name=f"agent_{i}",
                shape=Box(length=AGENTS["length"], width=AGENTS["width"]),
                color=tuple(self.colors[i]),
                collide=False,
                render_action=False,
                is_render_direction=self.parameters.is_visu_veh_direction,
                # Control command serves as velocity command, ranging over the
                # policy's speed scale; the dynamics clip it to the speed limit
                u_range=(
                    [self.parameters.speed_normalization]
                    if self.parameters.rl_action_type == "speed"
                    else (
                        [
                            POSITION_SPEED_LATERAL_OFFSET_BOUND,
                            self.parameters.speed_normalization,
                        ]
                        if self.parameters.rl_action_type == "position_speed"
                        else [self.parameters.speed_normalization, self.max_steering]
                    )
                ),
                u_multiplier=(
                    [1] if self.parameters.rl_action_type == "speed" else [1, 1]
                ),
                max_speed=self.max_speed,
                dynamics=KinematicBicycleModel(  # Use the kinematic bicycle model for each agent
                    l_f=AGENTS["l_f"],
                    l_r=AGENTS["l_r"],
                    max_speed=self.max_speed,
                    min_speed=self.min_speed,
                    max_steering=AGENTS["max_steering"],
                    min_steering=AGENTS["min_steering"],
                    max_acc=AGENTS["max_acc"],
                    min_acc=AGENTS["min_acc"],
                    max_steering_rate=AGENTS["max_steering_rate"],
                    min_steering_rate=AGENTS["min_steering_rate"],
                    device=world.device,
                    rl_action_type=self.parameters.rl_action_type,
                    lookahead_dist=self.parameters.lookahead_dist,
                ),
            )
            world.add_agent(agent)

    def process_action(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        on_hold_agents = self.get_on_hold_agents_mask()
        if on_hold_agents is not None:
            agent.dynamics.joint_task_hold_mask = on_hold_agents[:, agent_index].clone()
        else:
            agent.dynamics.joint_task_hold_mask = None

        if self.parameters.rl_action_type != "speed" or agent.action.u is None:
            return

        agent.dynamics.path_tracking_ref_point = select_short_term_reference_point(
            self.world_state.ref_paths_agent_related.short_term[:, agent_index]
        )

    def _is_deadlock_enabled(self) -> bool:
        return bool(getattr(self.parameters, "is_deadlock_enabled", True))

    def _deadlock_position_threshold(self) -> float:
        threshold = getattr(self.parameters, "deadlock_progress_threshold", None)
        return 1e-3 * LENGTH_SCALE if threshold is None else float(threshold)

    def _init_deadlock_tracker(self, batch_dim: int, device: torch.device):
        deadlock_duration = float(getattr(self.parameters, "deadlock_duration", 3.0))
        dt = float(getattr(self.parameters, "dt", 0.05))
        self.deadlock_window_size = max(1, math.ceil(deadlock_duration / dt))
        self.deadlock_position_delta_buffer = torch.zeros(
            (self.deadlock_window_size, batch_dim, self.n_agents),
            device=device,
            dtype=torch.float32,
        )
        self.deadlock_position_delta_count = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.int32
        )
        self.deadlock_position_delta_pointer = 0
        self.is_deadlocked_agents = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.bool
        )
        self.deadlock_reset_agents = torch.zeros_like(self.is_deadlocked_agents)

    def _is_joint_task_enabled(self) -> bool:
        return bool(self.parameters.is_joint_task)

    def get_on_hold_agents_mask(self):
        """Agents [batch_dim, n_agents] that stand still and are hidden from observations and CBFs, or None.

        These are the agents parked by a log replay, or the agents that reached their exits in a joint task.
        """
        if self.log_replay is not None:
            return self.log_replay.parked
        if self._is_joint_task_enabled():
            return self._joint_task_finished_agents
        return None

    def get_uncontrolled_agents_mask(self):
        """Agents [batch_dim, n_agents] that follow recorded tracks (log replay) and ignore actions, or None."""
        if self.log_replay is not None:
            return self.log_replay.replayed
        return None

    def _is_debug_diagnostics_enabled(self) -> bool:
        """Return whether optional per-step diagnostic output is enabled."""
        return self.parameters.is_enable_debug_diagnostics

    def _validate_joint_task_reference_paths(self, env_index: int) -> None:
        """Validate the reference-path layout supported by joint tasks."""
        if not self._is_joint_task_enabled():
            return

        is_loop = self.world_state.ref_paths_agent_related.is_loop[env_index]
        if is_loop.any() and not is_loop.all():
            raise NotImplementedError(
                "Joint testing tasks require either all loop or all non-loop "
                "reference paths; mixed layouts are not implemented."
            )

    def _validate_fast_inference_reference_paths(self) -> None:
        """Warn once in red if fast inference mode is active with non-loop reference paths."""
        if (
            self.parameters.is_fast_inference_mode
            and not self._warned_fast_inference_non_loop
        ):
            if not self.world_state.ref_paths_agent_related.is_loop.all():
                print(
                    colored(
                        "[WARN] Fast inference mode is enabled with non-loop reference paths! "
                        "Agents will not be reset once they reach their exits.",
                        "red",
                    )
                )
                self._warned_fast_inference_non_loop = True

    def _reset_joint_task_state(self, env_index=slice(None), agent_index=None):
        """Clear held-agent state after a whole or partial world reset.

        Collision-event latches intentionally survive this reset so VMAS can
        report the failed transition through its reset-time ``info`` query.
        They are cleared when the next joint transition starts.
        """
        if not self._is_joint_task_enabled():
            return

        if agent_index is None:
            self._joint_task_finished_agents[env_index] = False
            self._joint_task_elapsed_steps[env_index] = 0
        else:
            self._joint_task_finished_agents[env_index, agent_index] = False

    def _joint_task_collision_masks(self):
        """Return collision masks after removing agents that have reached exits."""
        active_agents = ~self._joint_task_finished_agents
        # Keep agents that reached an exit on the current transition active for
        # its collision penalty; they are held only from the next transition.
        active_agents = active_agents | self._joint_task_exit_event
        with_agents = self.world_state.collisions.with_agents
        with_agents = (
            with_agents & active_agents.unsqueeze(1) & active_agents.unsqueeze(2)
        )
        with_lanelets = self.world_state.collisions.with_lanelets & active_agents
        return with_agents, with_lanelets

    def _update_joint_task_state(self):
        """Record joint-task events before VMAS collects the current ``info``."""
        if not self._is_joint_task_enabled():
            return

        self._joint_task_exit_event.zero_()
        self._joint_task_success_event.zero_()
        self._joint_task_failure_event.zero_()
        self._joint_task_try_event.zero_()
        self._joint_task_collision_agents_event.zero_()
        self._joint_task_collision_lanelets_event.zero_()
        is_loop_task = self.world_state.ref_paths_agent_related.is_loop.all(dim=-1)
        # Exit-based tasks do not consume the duration parameter.
        self._joint_task_elapsed_steps += is_loop_task.to(torch.int32)

        active_agents = ~self._joint_task_finished_agents
        with_agents, with_lanelets = self._joint_task_collision_masks()
        self._joint_task_collision_agents_event.copy_(with_agents.any(dim=-1))
        self._joint_task_collision_lanelets_event.copy_(with_lanelets)
        is_collision = with_agents.any(dim=(-1, -2)) | with_lanelets.any(dim=-1)

        new_exit = self.world_state.collisions.with_exit_segments & active_agents
        self._joint_task_exit_event.copy_(new_exit)
        self._joint_task_finished_agents |= new_exit

        is_duration_complete = is_loop_task & (
            self._joint_task_elapsed_steps.to(torch.float32) * self.parameters.dt
            >= self.parameters.joint_task_complete_duration
        )
        is_exit_task_complete = self._joint_task_finished_agents.all(dim=-1)
        is_success = (
            torch.where(
                is_loop_task,
                is_duration_complete,
                is_exit_task_complete,
            )
            & ~is_collision
        )

        if self.parameters.max_steps is None:
            is_max_steps_reached = torch.zeros_like(is_success)
        else:
            is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)
        if self.parameters.reset_agent_fixed_duration > 0:
            t = self.timer.step * self.parameters.dt
            is_fixed_duration_reset = (
                t % self.parameters.reset_agent_fixed_duration == 0
            ) & (t != 0)
        else:
            is_fixed_duration_reset = torch.zeros_like(is_success)

        is_failure = is_collision | (
            (is_max_steps_reached | is_fixed_duration_reset) & ~is_success
        )
        is_try = is_success | is_failure

        self._joint_task_success_event.copy_(is_success)
        self._joint_task_failure_event.copy_(is_failure)
        self._joint_task_try_event.copy_(is_try)
        self.task_success_times[is_success] += 1
        self.num_task_tries[is_try] += 1

    def _joint_task_collision_info(self, agent_index: int):
        """Return collision info, preserving the current failed transition."""
        (
            collision_with_agents,
            collision_with_lanelets,
        ) = self._joint_task_collision_masks()
        is_collision_with_agents = collision_with_agents[:, agent_index].any(dim=-1)
        is_collision_with_lanelets = collision_with_lanelets[:, agent_index]

        # VMAS resets a completed joint task before its reset-time ``info``
        # query. Use the snapshot from the failed transition in that query;
        # otherwise the reset clears the collision tensors and the summary
        # reports a task failure with no corresponding collision.
        is_failure = self._joint_task_failure_event
        is_collision_with_agents = torch.where(
            is_failure,
            self._joint_task_collision_agents_event[:, agent_index],
            is_collision_with_agents,
        )
        is_collision_with_lanelets = torch.where(
            is_failure,
            self._joint_task_collision_lanelets_event[:, agent_index],
            is_collision_with_lanelets,
        )
        return is_collision_with_agents, is_collision_with_lanelets

    def _reset_deadlock_tracker(
        self, env_index=slice(None), agent_index=None, clear_current_mask: bool = True
    ):
        if not hasattr(self, "deadlock_position_delta_buffer"):
            return

        if agent_index is None:
            self.deadlock_position_delta_buffer[:, env_index, :] = 0
            self.deadlock_position_delta_count[env_index, :] = 0
            self.deadlock_reset_agents[env_index, :] = False
            if clear_current_mask:
                self.is_deadlocked_agents[env_index, :] = False
        else:
            self.deadlock_position_delta_buffer[:, env_index, agent_index] = 0
            self.deadlock_position_delta_count[env_index, agent_index] = 0
            self.deadlock_reset_agents[env_index, agent_index] = False
            if clear_current_mask:
                self.is_deadlocked_agents[env_index, agent_index] = False

    def _reset_control_history(self, env_index=slice(None), agent_index=None):
        if not hasattr(self, "previous_acceleration"):
            return

        if torch.is_tensor(agent_index):
            agent_index = int(agent_index.item())

        if agent_index is None:
            self.previous_acceleration[env_index, :] = 0
            self.previous_steering_rate[env_index, :] = 0
            self.previous_control_valid[env_index, :] = False
        else:
            self.previous_acceleration[env_index, agent_index] = 0
            self.previous_steering_rate[env_index, agent_index] = 0
            self.previous_control_valid[env_index, agent_index] = False

    @staticmethod
    def _as_float(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _limit_proximity_factor(self, value: Tensor, min_limit, max_limit) -> Tensor:
        min_limit = self._as_float(min_limit)
        max_limit = self._as_float(max_limit)
        threshold_fraction = float(
            getattr(self.parameters, "high_control_threshold", 0.8)
        )
        threshold_fraction = min(max(threshold_fraction, 0.0), 0.999)

        factor = torch.zeros_like(value)
        if max_limit > 0:
            upper_threshold = threshold_fraction * max_limit
            upper_denominator = max(max_limit - upper_threshold, 1e-6)
            factor = torch.maximum(
                factor,
                torch.clamp(
                    (value - upper_threshold) / upper_denominator,
                    min=0.0,
                    max=1.0,
                ),
            )
        if min_limit < 0:
            lower_threshold = threshold_fraction * min_limit
            lower_denominator = max(lower_threshold - min_limit, 1e-6)
            factor = torch.maximum(
                factor,
                torch.clamp(
                    (lower_threshold - value) / lower_denominator,
                    min=0.0,
                    max=1.0,
                ),
            )

        return factor

    def _update_deadlock_tracker(
        self, agent_index: int, move_vec: torch.Tensor
    ) -> torch.Tensor:
        if not self._is_deadlock_enabled():
            self.is_deadlocked_agents[:, agent_index] = False
            self.deadlock_reset_agents[:, agent_index] = False
            return torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.bool
            )

        position_delta = torch.linalg.norm(move_vec, dim=-1)

        self.deadlock_position_delta_buffer[
            self.deadlock_position_delta_pointer, :, agent_index
        ] = position_delta
        self.deadlock_position_delta_count[:, agent_index] = torch.clamp(
            self.deadlock_position_delta_count[:, agent_index] + 1,
            max=self.deadlock_window_size,
        )

        window_position_delta = self.deadlock_position_delta_buffer[
            :, :, agent_index
        ].sum(dim=0)
        is_window_full = (
            self.deadlock_position_delta_count[:, agent_index]
            >= self.deadlock_window_size
        )
        is_deadlocked = is_window_full & (
            window_position_delta < self._deadlock_position_threshold()
        )

        self.is_deadlocked_agents[:, agent_index] = is_deadlocked
        self.deadlock_reset_agents[:, agent_index] = is_deadlocked
        return is_deadlocked

    def _advance_deadlock_tracker(self):
        if not hasattr(self, "deadlock_position_delta_buffer"):
            return
        self.deadlock_position_delta_pointer = (
            self.deadlock_position_delta_pointer + 1
        ) % self.deadlock_window_size

    def _get_current_state_tensor(self) -> torch.Tensor:
        return torch.cat(
            (
                torch.stack([a.state.pos for a in self.world.agents], dim=1),
                torch.stack([a.state.rot for a in self.world.agents], dim=1),
                torch.stack([a.state.vel for a in self.world.agents], dim=1),
                self.world_state.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                self.world_state.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                self.world_state.ref_paths_agent_related.point_id[:].unsqueeze(-1),
            ),
            dim=-1,
        )

    def _sync_state_buffer_after_reset(
        self,
        state_add: torch.Tensor,
        env_index=slice(None),
        agent_index=None,
    ):
        if self.state_buffer.valid_size == 0:
            self.state_buffer.add(state_add)
            return

        latest_index = (self.state_buffer.pointer - 1) % self.state_buffer.buffer_size
        if agent_index is None:
            self.state_buffer.buffer[latest_index, env_index, :, :] = state_add[
                env_index, :, :
            ]
        else:
            self.state_buffer.buffer[
                latest_index, env_index, agent_index, :
            ] = state_add[env_index, agent_index, :]

    def place_agent(
        self,
        env_index: int,
        agent_index: int,
        pos: Tensor,
        rot: float,
        speed: float,
        steering: float,
        path_id: int = None,
    ):
        """Put one agent at a given state, e.g., a recorded one, without resetting the others.

        Args:
            pos: Position [2] of the vehicle center.
            rot: Yaw [rad].
            speed: Speed [m/s].
            steering: Steering angle [rad].
            path_id: Index of the reference path to assign. None keeps the current path, so only the
                state changes (e.g., to replay a recorded vehicle or to park it).
        """
        agent = self.world.agents[agent_index]
        device = self.world.device
        sideslip = math.atan(
            agent.dynamics.l_r / agent.dynamics.l_wb * math.tan(steering)
        )
        agent.set_pos(
            torch.as_tensor(pos, device=device, dtype=torch.float32),
            batch_index=env_index,
        )
        agent.set_rot(
            torch.tensor([rot], device=device, dtype=torch.float32),
            batch_index=env_index,
        )
        agent.set_speed(
            torch.tensor([speed], device=device, dtype=torch.float32),
            batch_index=env_index,
        )
        agent.set_steering(
            torch.tensor([steering], device=device, dtype=torch.float32),
            batch_index=env_index,
        )
        agent.set_sideslip_angle(
            torch.tensor([sideslip], device=device, dtype=torch.float32),
            batch_index=env_index,
        )
        agent.set_vel(
            torch.tensor(
                [speed * math.cos(rot + sideslip), speed * math.sin(rot + sideslip)],
                device=device,
                dtype=torch.float32,
            ),
            batch_index=env_index,
        )
        if path_id is None:
            return

        (
            ref_paths_scenario,
            extended_points,
        ) = self.world_state._reset_scenario_related_ref_paths(
            env_index=env_index, agent_index=agent_index
        )
        self.world_state.ref_paths_agent_related.path_id[
            env_index, agent_index
        ] = path_id
        self.world_state._reset_agent_related_ref_path(
            env_index,
            agent_index,
            ref_paths_scenario[path_id],
            path_id,
            extended_points,
        )
        self.world_state.reset_init_distances_and_short_term_ref_path(
            self.world_state.get_agent_state_list(), env_index, agent_index
        )
        self._reset_deadlock_tracker(env_index=env_index, agent_index=agent_index)
        self._reset_control_history(env_index=env_index, agent_index=agent_index)

    def set_vehicle_params(self, agent_index: int, params: VehicleParams = None):
        """Give one agent the footprint, axle distances, and acceleration limits of a vehicle (e.g., a
        recorded one; None: the nominal parameters of the agent profile).

        The parameters apply to all envs. The dynamics (plant), the footprint of the collision checks,
        observations, and rendering, and the CBF-QP (`vehicle_param_rows`) use them from now on. Call it
        before `place_agent`, whose sideslip angle depends on the axle distances.
        """
        if params is None:
            params = VehicleParams.nominal()
        agent = self.world.agents[agent_index]
        agent.shape._length = params.length
        agent.shape._width = params.width
        agent.dynamics.l_f = params.l_f
        agent.dynamics.l_r = params.l_r
        agent.dynamics.l_wb = params.l_wb
        agent.dynamics.min_acc = params.min_acc
        agent.dynamics.max_acc = params.max_acc
        self.world_state.agent_lengths[agent_index] = params.length
        self.world_state.agent_widths[agent_index] = params.width
        self.vehicle_param_table[agent_index] = params.row()
        self.is_heterogeneous_vehicles = bool(
            (self.vehicle_param_table != VehicleParams.nominal().row()).any()
        )

    def vehicle_param_rows(self, agents) -> dict:
        """Parameters of the given agents as arrays [len(agents)] by key (VEHICLE_PARAM_KEYS), or None
        while every agent has the nominal parameters (then the profile's scalars apply)."""
        if not self.is_heterogeneous_vehicles:
            return None
        rows = self.vehicle_param_table[np.asarray(agents, dtype=int)]
        return {key: rows[:, k] for k, key in enumerate(VEHICLE_PARAM_KEYS)}

    def sync_agent_history(self, env_index: int, agent_index: int):
        """Overwrite the latest stored state of one agent with its current state.

        Call after an agent jumped (spawned or parked) so that the anti-tunneling lane check and the
        deadlock tracker do not see the jump as motion.
        """
        self._sync_state_buffer_after_reset(
            state_add=self._get_current_state_tensor(),
            env_index=env_index,
            agent_index=agent_index,
        )

    @staticmethod
    def _collision_neighborhood_masks(
        collision_matrix: torch.Tensor,
        pairwise_distances: torch.Tensor,
        surrounding_distance: float,
        collision_with_lanelets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return collision participants and nearby non-colliding agents.

        The masks are derived from one pre-reset snapshot. This is important
        because a single-agent reset clears collision flags for the complete
        sub-environment and recomputes pairwise distances. Lanelet collisions
        are included as collision sources when their mask is provided.
        """
        collision_participants = collision_matrix.any(dim=-1) | collision_matrix.any(
            dim=-2
        )
        if collision_with_lanelets is not None:
            collision_participants = collision_participants | collision_with_lanelets
        nearby_agents = torch.zeros_like(collision_participants)

        if surrounding_distance > 0:
            near_collision_participants = pairwise_distances <= surrounding_distance
            nearby_agents = (
                near_collision_participants & collision_participants.unsqueeze(-1)
            ).any(dim=1) & ~collision_participants

        return collision_participants, nearby_agents

    def _update_collision_neighborhood_masks(self):
        """Snapshot collision participants and their nearby agents."""
        (
            self._collision_participant_mask,
            self._near_collision_reset_mask,
        ) = self._collision_neighborhood_masks(
            self.world_state.collisions.with_agents,
            self.world_state.distances.agents,
            self.reset_agent_surrounding_distance,
            self.world_state.collisions.with_lanelets,
        )

    def reset_world_at(
        self, env_index: int = None, agent_index: int = None, preserve_timer=None
    ):
        """
        This function resets the world at the specified env_index and the specified agent_index.
        If env_index is given as None, the majority part of computation will be done in a vectorized manner.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed
        :param agent_index: index of the agent to reset. If None all agents in the specified environment will be reset.
        :param preserve_timer: Preserve the cumulative horizon timer for this
            task reset. If None, consume the pending preservation latch used by
            the VMAS/TorchRL reset that follows a joint-task terminal event.
        """
        if env_index is None and agent_index is None:
            self._last_reset_render_info = [None] * self.world.batch_dim
            self._last_reset_render_time = [0.0] * self.world.batch_dim
            self._last_reset_render_color = [None] * self.world.batch_dim

        self._transition_state_prepared = False
        is_reset_single_agent = agent_index is not None

        if is_reset_single_agent:
            assert env_index is not None

        for env_i in (
            [env_index] if env_index is not None else range(self.world.batch_dim)
        ):
            if preserve_timer is None:
                pending_timer_preservation = getattr(
                    self, "_preserve_timer_on_next_reset", None
                )
                preserve_timer_for_env = (
                    pending_timer_preservation is not None
                    and bool(pending_timer_preservation[env_i].item())
                )
                if pending_timer_preservation is not None:
                    pending_timer_preservation[env_i] = False
            else:
                preserve_timer_for_env = bool(preserve_timer)

            # Beginning of a new simulation (only record for the first env).
            # A task reset keeps the cumulative horizon step but starts a new
            # wall-clock measurement for the next simulator transition.
            if preserve_timer_for_env:
                self.timer.step_begin = time.time()
            else:
                if env_i == 0:
                    self.timer.step_duration[:] = 0
                    self.timer.start = time.time()
                    self.timer.step_begin = time.time()
                    self.timer.end = 0

            if self.parameters.predefined_ref_path_idx is not None:
                predefined_ref_path_idx = self.parameters.predefined_ref_path_idx
                if is_reset_single_agent:
                    initial_state = None
                else:
                    # Execute this only when initializing the whole world (not a single angnt)
                    initial_state = torch.tensor(
                        self.parameters.init_state,
                        device=self.parameters.device,
                        dtype=torch.float32,
                    )  # [[x1, y1, rot1], ..., [xn, yn, rotn]]
            else:
                predefined_ref_path_idx = None
                initial_state = None

                if (
                    self.parameters.is_challenging_initial_state_buffer
                    and (
                        torch.rand(1)
                        < self.initial_state_buffer.probability_use_recording
                    )
                    and (self.initial_state_buffer.valid_size >= 1)
                ):
                    # Use initial state buffer
                    initial_state = (
                        self.world_state.init_ref_paths_agent_related_from_buffer(
                            env_i, self.initial_state_buffer
                        )
                    )
                    # print(colored(f"[LOG] Reset with path ids: {initial_state[:, -2]}", "red"))
                else:
                    initial_state = None

            if not is_reset_single_agent and not preserve_timer_for_env:
                # Each time step of a simulation
                self.timer.step[env_i] = 0

            if self.parameters.is_testing_mode and not is_reset_single_agent:
                reset_index = int(self.num_task_tries[env_i].item())
            else:
                reset_index = None

            self.world_state.reset(
                self.world_state.get_agent_state_list(),
                env_index=env_i,
                agent_index=agent_index,
                predefined_ref_path_idx=predefined_ref_path_idx,
                initial_state=initial_state,
                initial_state_buffer=self.state_buffer,
                reset_index=reset_index,
            )
            self._validate_joint_task_reference_paths(env_i)
        self._validate_fast_inference_reference_paths()

        reset_indices = (
            range(self.n_agents) if agent_index is None else agent_index.unsqueeze(0)
        )

        env_j = slice(None)

        if env_index is not None:
            env_j = env_index

        self._reset_joint_task_state(env_index=env_j, agent_index=agent_index)

        for i_agent in reset_indices:
            self.world_state.reset_init_distances_and_short_term_ref_path(
                self.world_state.get_agent_state_list(), env_j, i_agent
            )

        # Compute mutual distances between agents
        self.world_state.update_mutual_distances(
            self.world_state.get_agent_state_list(), env_index=env_j
        )

        self.world_state.reset_collisions(env_j)
        self._reset_deadlock_tracker(env_index=env_j, agent_index=agent_index)
        self._reset_control_history(env_index=env_j, agent_index=agent_index)

        state_add = self._get_current_state_tensor()
        if env_index is None and agent_index is None:
            self.state_buffer.reset()
            self.state_buffer.add(state_add)
        else:
            self._sync_state_buffer_after_reset(
                state_add=state_add,
                env_index=env_j,
                agent_index=agent_index,
            )

        if self.log_replay is not None and agent_index is None:
            # Replace the random initial states by the recorded ones
            for env_i in (
                [env_index] if env_index is not None else range(self.world.batch_dim)
            ):
                self.log_replay.on_reset(self, env_i)
            self.world_state.update_mutual_distances(
                self.world_state.get_agent_state_list(), env_index=env_j
            )

    def post_step(self):
        """Prepare shared transition state before VMAS collects rewards.

        VMAS invokes this hook exactly once after the world dynamics have been
        advanced and before it calls ``reward`` for each agent.  The reward
        method remains the public per-agent API, while shared collision,
        distance, timer, and joint-task bookkeeping is now explicitly
        separated from scalar reward calculation.
        """
        self._transition_state_prepared = False
        if self.log_replay is not None:
            # Replayed agents jump to their recorded states and agents are spawned or parked before the
            # shared state (distances, collisions) of this transition is computed
            self.log_replay.on_post_step(self)
        self._prepare_transition_state()

    def _prepare_transition_state(self):
        """Update state shared by reward, observation, info, and termination."""
        if self._transition_state_prepared:
            return

        # Preserve the original order from the first reward call.
        self.reward_info.reset()
        self.rew[:] = 0
        self.timer.step_duration[self.timer.step] = time.time() - self.timer.step_begin
        self.timer.step_begin = time.time()
        self.timer.step += 1

        # Skip route progress snapshot if fast inference is on (only used by progress reward delta).
        if not self.parameters.is_fast_inference_mode:
            self.world_state.snapshot_route_progress()

        self.world_state.update_state_after_step()
        if not self.parameters.is_fast_inference_mode:
            self._update_deadlocks()
            self._update_collision_neighborhood_masks()
            self._update_joint_task_state()
            self._advance_deadlock_tracker()
            # Skip state buffer updates if fast inference is on (only read by reward and deadlock tracking).
            self.state_buffer.add(self._get_current_state_tensor())
        self._transition_state_prepared = True

    def _update_deadlocks(self):
        """Update deadlock tracking for all agents."""
        latest_state = self.state_buffer.get_latest(n=1)
        for agent_index, agent in enumerate(self.world.agents):
            move_vec_current = agent.state.pos - latest_state[:, agent_index, 0:2]
            is_deadlocked = self._update_deadlock_tracker(agent_index, move_vec_current)
            if self._is_joint_task_enabled():
                is_held = self._joint_task_finished_agents[:, agent_index]
                self.is_deadlocked_agents[:, agent_index] &= ~is_held
                self.deadlock_reset_agents[:, agent_index] &= ~is_held

    def reward(self, agent: Agent):
        """Prepare shared transition state and return one agent's reward."""
        # VMAS calls this public method once per agent.  Shared state is already
        # prepared by post_step() in normal execution; the helper retains a
        # safe fallback for direct scenario.reward() callers.
        self._prepare_transition_state()
        if self.parameters.is_fast_inference_mode:
            return torch.zeros(
                self.world.batch_dim, dtype=torch.float32, device=self.world.device
            )
        self.rew[:] = 0
        return self._compute_reward(agent)

    def _compute_reward(self, agent: Agent):
        """
        Compute the reward for the given agent in all environments.
            Positive Rewards:
                Moving forward (become negative if the projection of the moving direction to its reference path is negative)
                Moving forward with high speed (become negative if the projection of the moving direction to its reference path is negative)
                Reaching goal (optional)

            Negative Rewards (penalties):
                Too close to lane boundaries
                Too close to other agents
                Deviating from reference paths
                Changing steering too quick
                Colliding with other agents
                Colliding with lane boundaries

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            A tensor with shape [batch_dim].
        """
        # Get the index of the current agent
        agent_index = self.world.agents.index(agent)

        ##################################################
        ## [reward] forward movement
        ##################################################
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec_current = agent.state.pos - latest_state[:, agent_index, 0:2]

        route_progress_cache_valid = getattr(
            self.world_state, "_route_progress_cache_valid", None
        )
        if route_progress_cache_valid is not None and bool(
            route_progress_cache_valid[:, agent_index].all()
        ):
            current_route_progress = self.world_state._route_progress[:, agent_index]
            previous_route_progress = self.world_state._route_progress_previous[
                :, agent_index
            ]
            route_total_length = self.world_state._route_progress_total_lengths[
                :, agent_index
            ]
        else:
            _, current_route_progress, route_total_length = get_progress_on_polyline(
                point=agent.state.pos,
                polyline=self.world_state.ref_paths_agent_related.long_term[
                    :, agent_index
                ],
                n_points_long_term=self.world_state.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
            )
            _, previous_route_progress, _ = get_progress_on_polyline(
                point=latest_state[:, agent_index, 0:2],
                polyline=self.world_state.ref_paths_agent_related.long_term[
                    :, agent_index
                ],
                n_points_long_term=self.world_state.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
            )
        route_progress_delta = current_route_progress - previous_route_progress
        half_route_length = 0.5 * route_total_length
        is_loop = self.world_state.ref_paths_agent_related.is_loop[:, agent_index]
        route_progress_delta = torch.where(
            is_loop & (route_progress_delta < -half_route_length),
            route_progress_delta + route_total_length,
            route_progress_delta,
        )
        route_progress_delta = torch.where(
            is_loop & (route_progress_delta > half_route_length),
            route_progress_delta - route_total_length,
            route_progress_delta,
        )

        reward_movement = (
            route_progress_delta
            / (agent.max_speed * self.world.dt)
            * self.rewards.progress
        )
        self.reward_info.rew_progress[:, agent_index] = reward_movement

        ##################################################
        ## [penalty] deadlock
        ##################################################
        is_deadlocked = self.is_deadlocked_agents[:, agent_index]
        if self._is_debug_diagnostics_enabled() and is_deadlocked.any():
            print(
                colored(
                    f"[LOG] Deadlock detected for agent {agent_index} in envs {torch.where(is_deadlocked)[0].tolist()} at step {self.timer.step[0].item()}",
                    "red",
                )
            )
        penalty_deadlock = is_deadlocked * self.penalties.deadlock
        self.reward_info.rew_deadlock[:, agent_index] = penalty_deadlock

        #################################################
        # [reward] reach goal
        #################################################
        if self._is_joint_task_enabled():
            is_reach_goal = self._joint_task_exit_event[:, agent_index]
            reward_goal = self._joint_task_success_event * self.rewards.reach_goal
        else:
            is_reach_goal = self.world_state.collisions.with_exit_segments[
                :, agent_index
            ]
            reward_goal = is_reach_goal * self.rewards.reach_goal
            self.task_success_times[:] += is_reach_goal.to(
                torch.int32
            )  # Each time an agent reaches the goal (i.e., the exit in this case), increment the task success times by 1
        self.reward_info.rew_reach_goal[:, agent_index] = reward_goal

        # ##################################################
        # ## [penalty] colliding with other agents
        # ##################################################
        if self._is_joint_task_enabled():
            (
                collision_with_agents,
                collision_with_lanelets,
            ) = self._joint_task_collision_masks()
            is_collide_with_agents = collision_with_agents[:, agent_index]
            is_collide_with_lanelets = collision_with_lanelets[:, agent_index]
        else:
            is_collide_with_agents = self.world_state.collisions.with_agents[
                :, agent_index
            ]  # [B, N]
            is_collide_with_lanelets = self.world_state.collisions.with_lanelets[
                :, agent_index
            ]  # [B]
        penalty_collide_other_agents = (
            is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        )
        self.reward_info.rew_collide_other_agents[
            :, agent_index
        ] = penalty_collide_other_agents

        ##################################################
        ## [penalty] colliding with lanelet boundaries
        ##################################################
        penalty_collide_lanelet = (
            is_collide_with_lanelets * self.penalties.collide_with_boundaries
        )
        self.reward_info.rew_collide_lane[:, agent_index] = penalty_collide_lanelet

        penalty_near_collision_reset = (
            self._near_collision_reset_mask[:, agent_index].to(
                dtype=penalty_collide_lanelet.dtype
            )
            * self.penalties.near_collision_reset
        )
        self.reward_info.rew_near_collision_reset[
            :, agent_index
        ] = penalty_near_collision_reset

        if not self._is_joint_task_enabled():
            # Increment the number of task attempts if a collision occurs
            is_increment_task = (
                is_collide_with_agents.any(dim=-1)
                | is_collide_with_lanelets
                | is_reach_goal
                | is_deadlocked
            )
            self.num_task_tries[is_increment_task] += 1

        ##################################################
        ## [penalty] close to lanelet boundaries
        ##################################################
        penalty_close_to_lanelets = (
            decreasing_fcn(
                x=self.world_state.distances.boundaries[:, agent_index],
                x0=self.thresholds.near_boundary_low,
                x1=self.thresholds.near_boundary_high,
                type="linear",
            )
            * self.penalties.near_boundary
        )

        current_acceleration = agent.state.acceleration.squeeze(-1)
        current_steering_rate = agent.state.steering_rate.squeeze(-1)
        min_acc = getattr(agent.dynamics, "min_acc", AGENTS["min_acc"])
        max_acc = getattr(agent.dynamics, "max_acc", AGENTS["max_acc"])
        min_steering_rate = getattr(
            agent.dynamics, "min_steering_rate", AGENTS["min_steering_rate"]
        )
        max_steering_rate = getattr(
            agent.dynamics, "max_steering_rate", AGENTS["max_steering_rate"]
        )
        acc_span = current_acceleration.new_tensor(
            max(self._as_float(max_acc) - self._as_float(min_acc), 1e-6)
        )
        steering_rate_span = current_steering_rate.new_tensor(
            max(
                self._as_float(max_steering_rate) - self._as_float(min_steering_rate),
                1e-6,
            )
        )

        self.reward_info.rew_parameter_independent[:, agent_index] = (
            penalty_deadlock
            + reward_goal
            + penalty_collide_other_agents
            + penalty_collide_lanelet
            + penalty_near_collision_reset
        )

        if not "noprog" in self.parameters.rew_method:
            self.rew += reward_movement  # Shared by all reward methods

        if "sparse" in self.parameters.rew_method:
            # Sparse reward that only cares about the final outcome of the task, without any reward shaping.
            self.rew += penalty_deadlock
            self.rew += reward_goal
            self.rew += penalty_collide_other_agents
            self.rew += penalty_collide_lanelet
            self.rew += penalty_near_collision_reset

        if "ttc" in self.parameters.rew_method:
            penalty_close_to_agents = self._apply_ttc_near_agent_penalty(
                agent_index=agent_index,
                decreasing_fcn=decreasing_fcn,
                ttc_low=self.thresholds.ttc_low,
                ttc_high=self.thresholds.ttc_high,
            )

            self.reward_info.rew_near_other_agents[
                :, agent_index
            ] = penalty_close_to_agents

            self.rew += penalty_close_to_agents
            self.rew += penalty_close_to_lanelets

        if "distance" in self.parameters.rew_method:
            ##################################################
            ## [penalty] close to other agents
            ##################################################
            mutual_distance_exp_fcn = decreasing_fcn(
                x=self.world_state.distances.agents[:, agent_index, :],
                x0=self.thresholds.near_other_agents_low,
                x1=self.thresholds.near_other_agents_high,
                type="linear",
            )
            penalty_close_to_agents = (
                torch.sum(mutual_distance_exp_fcn, dim=1)
                * self.penalties.near_other_agents
            )

            self.reward_info.rew_near_other_agents[
                :, agent_index
            ] = penalty_close_to_agents

            self.rew += penalty_close_to_agents
            self.rew += penalty_close_to_lanelets

        if "cbfcorrection" in self.parameters.rew_method:
            ##################################################
            ## [penalty] deviating from CBF safe action
            ##################################################
            assert (
                self.parameters.is_solve_qp
            ), "CBF correction reward can only be applied when CBF-constrained MARL with solving QP is used."
            # print("DEBUG CBF penalty calculation")
            nominal_action_vel = self.world_state.nominal_action_vel[
                :, agent_index
            ]  # RL action that is modified by CBF
            nominal_action_steer = self.world_state.nominal_action_steer[
                :, agent_index
            ]  # RL action that is modified by CBF
            current_action_vel = agent.action.u[:, 0]
            current_action_steer = agent.action.u[:, 1]

            cbf_vel_penalty = self.penalties.deviate_from_cbf_vel * (
                (current_action_vel - nominal_action_vel).abs() / self.max_speed
            )
            cbf_steer_penalty = self.penalties.deviate_from_cbf_steer * (
                (current_action_steer - nominal_action_steer).abs() / self.max_steering
            )

            self.rew += cbf_vel_penalty + cbf_steer_penalty

        if "cbfmargin" in self.parameters.rew_method:
            cbf_rew = (
                self.reward_info.rew_near_left_lane[:, agent_index]
                + self.reward_info.rew_near_right_lane[:, agent_index]
                + self.reward_info.rew_near_other_agents[:, agent_index]
            )
            self.rew += cbf_rew

        if "cbflambda" in self.parameters.rew_method:
            # Penalize sharp change of lambda
            self.rew += self.reward_info.rew_cbf_lambda[:, agent_index]
            # Raise error if all entries are consistently zero
            if (self.reward_info.rew_cbf_lambda == 0).all():
                raise ValueError(
                    "All entries of CBF lambda reward are zero. Please check if CBF lambda reward is computed or not."
                )

        if "jerk" in self.parameters.rew_method:
            valid_control_history = self.previous_control_valid[:, agent_index].to(
                current_acceleration.dtype
            )
            jerk_acceleration_factor = (
                torch.clamp(
                    (
                        current_acceleration
                        - self.previous_acceleration[:, agent_index]
                    ).abs()
                    / acc_span,
                    max=1.0,
                )
                * valid_control_history
            )
            jerk_steering_factor = (
                torch.clamp(
                    (
                        current_steering_rate
                        - self.previous_steering_rate[:, agent_index]
                    ).abs()
                    / steering_rate_span,
                    max=1.0,
                )
                * valid_control_history
            )

            jerk_acceleration_penalty = (
                jerk_acceleration_factor * self.penalties.jerk_acceleration
            )
            jerk_steering_penalty = (
                jerk_steering_factor * self.penalties.jerk_steering_rate
            )
            self.reward_info.rew_jerk_acceleration[
                :, agent_index
            ] = jerk_acceleration_penalty
            self.reward_info.rew_jerk_steering[:, agent_index] = jerk_steering_penalty
            self.rew += jerk_acceleration_penalty + jerk_steering_penalty
            if self._is_debug_diagnostics_enabled():
                print(
                    f"DEBUG Jerk penalty for agent {agent_index}: acceleration penalty {jerk_acceleration_penalty.mean().item():.4f}, steering penalty {jerk_steering_penalty.mean().item():.4f}"
                )

        if "acceleration" in self.parameters.rew_method:
            high_acceleration_factor = self._limit_proximity_factor(
                current_acceleration, min_acc, max_acc
            )
            high_steering_rate_factor = self._limit_proximity_factor(
                current_steering_rate, min_steering_rate, max_steering_rate
            )

            high_acceleration_penalty = (
                high_acceleration_factor * self.penalties.high_acceleration
            )
            high_steering_rate_penalty = (
                high_steering_rate_factor * self.penalties.high_steering_rate
            )
            self.reward_info.rew_energy_acceleration[
                :, agent_index
            ] = high_acceleration_penalty
            self.reward_info.rew_energy_steering[
                :, agent_index
            ] = high_steering_rate_penalty
            self.rew += high_acceleration_penalty + high_steering_rate_penalty
            if self._is_debug_diagnostics_enabled():
                print(
                    f"DEBUG Acceleration penalty for agent {agent_index}: acceleration penalty {high_acceleration_penalty.mean().item():.4f}, steering penalty {high_steering_rate_penalty.mean().item():.4f}"
                )

        if "speed" in self.parameters.rew_method:
            speed_reward = (
                torch.clamp(agent.state.speed.squeeze(-1) / agent.max_speed, -1.0, 1.0)
                * self.rewards.higth_v
            )
            self.reward_info.rew_speed[:, agent_index] = speed_reward
            self.rew += speed_reward

        if "time" in self.parameters.rew_method:
            if self.parameters.is_schedule_rew_coefficient:
                current_iter = getattr(self, "i_iter", 100)
                time_coefficient = schedule_rew_coefficient(
                    t=current_iter,
                    y2=self.penalties.time,
                )
            else:
                time_coefficient = self.penalties.time
            time_penalty = -time_coefficient
            self.reward_info.rew_time[:, agent_index] = time_penalty
            self.rew += time_penalty

        self.previous_acceleration[:, agent_index] = current_acceleration.detach()
        self.previous_steering_rate[:, agent_index] = current_steering_rate.detach()
        self.previous_control_valid[:, agent_index] = True

        assert not self.rew.isnan().any(), "Rewards contain nan."
        assert not self.rew.isinf().any(), "Rewards contain inf."

        # Clamed the reward to avoid abs(reward) being too large
        rew_clamed = torch.clamp(self.rew, min=-2, max=2)

        self.reward_info.rew_total[:, agent_index] = rew_clamed

        return rew_clamed

    def _apply_ttc_near_agent_penalty(
        self,
        agent_index: int,
        decreasing_fcn: Callable,
        ttc_low: float,
        ttc_high: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        # Near-agent reward/penalty
        ##################################################
        ## [penalty] close to other agents (TTC-based, 2D)
        ##################################################
        # Positions and velocities of all agents: [B, N, 2]
        pos_all = torch.stack([a.state.pos for a in self.world.agents], dim=1)[..., 0:2]
        vel_all = torch.stack([a.state.vel for a in self.world.agents], dim=1)[..., 0:2]

        pos_i = pos_all[:, agent_index, :]  # [B, 2]
        vel_i = vel_all[:, agent_index, :]  # [B, 2]

        # Relative states for all j: p_ij = p_j - p_i, v_ij = v_j - v_i
        p_rel = pos_all - pos_i.unsqueeze(1)  # [B, N, 2]
        v_rel = vel_all - vel_i.unsqueeze(1)  # [B, N, 2]

        # Safety distance (disc approximation).
        # In your codebase you already have near_other_agents_low/high for distance shaping.
        # Using near_other_agents_low as the "contact + buffer" distance is consistent with your previous shaping.
        d_safe = float(self.thresholds.near_other_agents_low)

        # Optional distance gating to avoid penalizing far-away agents even if TTC is finite.
        # This is helpful in dense scenes with many agents.
        d_gate = float(self.thresholds.near_other_agents_high)

        # Quadratic coefficients: ||p + v t||^2 = d_safe^2
        a = (v_rel * v_rel).sum(dim=-1)  # [B, N]
        b = 2.0 * (p_rel * v_rel).sum(dim=-1)  # [B, N]
        c = (p_rel * p_rel).sum(dim=-1) - (d_safe * d_safe)  # [B, N]

        disc = b * b - 4.0 * a * c  # [B, N]
        disc_clamped = torch.clamp(disc, min=0.0)
        sqrt_disc = torch.sqrt(disc_clamped)

        # Current distances for gating and "already unsafe" handling
        dist = torch.sqrt(torch.clamp((p_rel * p_rel).sum(dim=-1), min=0.0))  # [B, N]

        # Valid approaching interactions:
        # - positive relative speed (a > eps)
        # - real roots (disc > 0)
        # - approaching initially (b < 0)
        valid = (a > eps) & (disc > 0.0) & (b < 0.0)

        # Earliest intersection time (entry time)
        ttc = torch.full_like(a, float("inf"))  # [B, N]
        ttc_candidate = (-b - sqrt_disc) / (2.0 * a + eps)  # [B, N]

        ttc = torch.where(valid & (ttc_candidate > 0.0), ttc_candidate, ttc)

        # If already within safety distance, TTC = 0
        ttc = torch.where(dist <= d_safe, torch.zeros_like(ttc), ttc)

        # Mask out self-interaction j == i
        ttc[:, agent_index] = float("inf")

        # Distance gating (recommended)
        ttc = torch.where(dist <= d_gate, ttc, torch.full_like(ttc, float("inf")))

        # Map TTC to a bounded risk in [0,1]
        # Smaller TTC => larger penalty weight; large TTC/inf => ~0.
        ttc_for_shape = torch.clamp(ttc, max=ttc_high)
        risk_per_agent = decreasing_fcn(
            x=ttc_for_shape, x0=ttc_low, x1=ttc_high, type="linear"
        )  # [B, N]

        # Aggregate across other agents. Normalize to reduce dependence on N.
        risk = risk_per_agent.sum(dim=1) / max(1, self.n_agents - 1)  # [B]

        penalty_close_to_agents = risk * self.penalties.near_other_agents  # [B]

        return penalty_close_to_agents

    def observation(self, agent: Agent):
        """
        Generate an observation for the given agent in all envs.

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            The observation for the given agent in all envs, which consists of the observation of this agent itself and possibly the observation of its surrounding agents.
                The observation of this agent itself includes
                    position (in case of using bird view),
                    rotation (in case of using bird view),
                    velocity,
                    short-term reference path,
                    distance to its reference path (optional), and
                    lane boundary points,
                    distances to lane boundaries (optional).
                The observation of its surrounding agents includes their
                    vertices (or positions and rotations),
                    velocities,
                    distances to them (optional), and
                    reference paths (optional).
        """
        agent_index = self.world.agents.index(agent)

        self.observation_provider.set_on_hold_agents_mask(
            self.get_on_hold_agents_mask()
        )

        if agent_index == 0:  # Avoid repeated computations
            self.observation_provider.update_state(self.world)

        obs = self.observation_provider.get_observation(agent_index)

        if self.parameters.is_obs_vehicle_id_in_motion_policy:
            obs = self._append_vehicle_ids_to_observation(obs, agent_index)

        # Store observation for reuse in `info()`, only relevant for prioritized MARL
        self.stored_observations[agent_index] = obs

        return obs

    def get_priority_neighbor_metadata(self):
        """Return neighbor indices and validity masks for priority-conditioned motion."""
        return (
            self.observation_provider.observations.nearing_agents_indices,
            self.observation_provider.get_neighbor_valid(),
        )

    def _append_vehicle_ids_to_observation(
        self, observation: torch.Tensor, agent_index: int
    ) -> torch.Tensor:
        """Add normalized global or ego-relative IDs to observation blocks."""
        is_randomize_veh_ID = True
        is_global_agent_id = getattr(self.parameters, "is_global_agent_id", True)
        n_agents = max(int(self.n_agents), 1)
        vehicle_indices = torch.arange(
            n_agents, device=observation.device, dtype=observation.dtype
        )
        global_vehicle_indices = vehicle_indices.to(dtype=torch.long)
        if n_agents > 1:
            normalized_vehicle_ids = 2.0 * vehicle_indices / (n_agents - 1) - 1.0
        else:
            normalized_vehicle_ids = torch.zeros_like(vehicle_indices)

        ego_global_id = global_vehicle_indices[agent_index].expand(
            observation.shape[0], 1
        )
        ego_vehicle_id = normalized_vehicle_ids[agent_index].expand(
            observation.shape[0], 1
        )
        is_partial_observation = self.parameters.is_partial_observation
        if is_partial_observation:
            diagnostics = (
                self.observation_provider.get_observed_agent_render_diagnostics(
                    agent_index
                )
            )
            if diagnostics is None:
                raise RuntimeError(
                    "Vehicle-ID priority observations require neighbor metadata "
                    "before scenario info is collected."
                )
            neighbor_indices = diagnostics["neighbor_indices"].to(
                device=observation.device, dtype=torch.long
            )
            neighbor_global_ids = global_vehicle_indices[neighbor_indices]
            neighbor_vehicle_ids = normalized_vehicle_ids[neighbor_global_ids]
            padded_mask = diagnostics["padded_mask"].to(
                device=observation.device, dtype=torch.bool
            )
            neighbor_vehicle_ids = neighbor_vehicle_ids.masked_fill(padded_mask, 0.0)
        else:
            # Full observations use the provider's fixed all-agent ordering.
            neighbor_global_ids = global_vehicle_indices.expand(
                observation.shape[0], -1
            )
            neighbor_vehicle_ids = normalized_vehicle_ids.expand(
                observation.shape[0], -1
            )
            padded_mask = torch.zeros(
                neighbor_vehicle_ids.shape,
                device=observation.device,
                dtype=torch.bool,
            )

        if not is_global_agent_id:
            observed_global_ids = torch.cat(
                (ego_global_id, neighbor_global_ids), dim=-1
            )
            observed_valid_mask = torch.cat(
                (
                    torch.ones_like(ego_global_id, dtype=torch.bool),
                    ~padded_mask,
                ),
                dim=-1,
            )
            present_id_counts = torch.zeros(
                (observation.shape[0], n_agents),
                device=observation.device,
                dtype=torch.long,
            )
            present_id_counts.scatter_add_(
                dim=1,
                index=observed_global_ids,
                src=observed_valid_mask.to(dtype=torch.long),
            )
            present_ids = present_id_counts > 0
            rank_by_global_id = present_ids.cumsum(dim=-1) - present_ids.to(
                dtype=torch.long
            )
            relative_ranks = rank_by_global_id.gather(1, observed_global_ids)
            n_present_ids = present_ids.sum(dim=-1, keepdim=True)
            relative_vehicle_ids = torch.where(
                n_present_ids > 1,
                2.0
                * relative_ranks.to(dtype=observation.dtype)
                / (n_present_ids - 1).clamp_min(1).to(dtype=observation.dtype)
                - 1.0,
                torch.zeros_like(relative_ranks, dtype=observation.dtype),
            )
            relative_vehicle_ids = relative_vehicle_ids.masked_fill(
                ~observed_valid_mask, 0.0
            )
            ego_vehicle_id = relative_vehicle_ids[:, :1]
            neighbor_vehicle_ids = relative_vehicle_ids[:, 1:]

        observed_vehicle_ids = torch.cat((ego_vehicle_id, neighbor_vehicle_ids), dim=-1)
        if is_randomize_veh_ID:
            noise_magnitude = 0.05 * (2.0 / (n_agents - 1)) if n_agents > 1 else 0.0
            noise = (2.0 * torch.rand_like(observed_vehicle_ids) - 1.0) * (
                noise_magnitude
            )
            real_id_mask = torch.cat(
                (
                    torch.ones_like(ego_vehicle_id, dtype=torch.bool),
                    ~padded_mask,
                ),
                dim=-1,
            )
            noise = noise.masked_fill(~real_id_mask, 0.0)
            observed_vehicle_ids = torch.clamp(
                observed_vehicle_ids + noise, min=-1.0, max=1.0
            )

        n_neighbor_slots = (
            int(self.parameters.n_nearing_agents_observed)
            if is_partial_observation
            else n_agents
        )
        neighbor_feature_widths = [
            8 if self.parameters.is_observe_vertices else 5,
            2,
            1 if self.parameters.is_obs_steering else 0,
            1 if self.parameters.is_observe_distance_to_agents else 0,
            (
                2 * self.parameters.n_points_short_term
                if self.parameters.is_observe_ref_path_other_agents
                else 0
            ),
            (
                1
                if is_partial_observation
                and (
                    self.parameters.is_apply_distance_mask
                    or self.parameters.is_apply_velocity_mask
                )
                else 0
            ),
        ]
        neighbor_feature_widths = [
            width for width in neighbor_feature_widths if width > 0
        ]
        neighbor_feature_width = n_neighbor_slots * sum(neighbor_feature_widths)
        # The observation provider stores the raw observation. The action
        # placeholder is appended once later in ``info()``.
        neighbor_start = observation.shape[-1] - neighbor_feature_width
        if neighbor_start < 0:
            raise RuntimeError(
                "Could not identify the neighbor feature block in the priority "
                "observation."
            )
        neighbor_end = neighbor_start + neighbor_feature_width
        neighbor_observation = observation[:, neighbor_start:neighbor_end]

        neighbor_feature_groups = []
        feature_offset = 0
        for feature_width in neighbor_feature_widths:
            feature_end = feature_offset + n_neighbor_slots * feature_width
            neighbor_feature_groups.append(
                neighbor_observation[:, feature_offset:feature_end].reshape(
                    observation.shape[0], n_neighbor_slots, feature_width
                )
            )
            feature_offset = feature_end

        if feature_offset != neighbor_observation.shape[-1]:
            raise RuntimeError(
                "The configured neighbor feature layout does not match the "
                "priority observation width."
            )
        neighbor_features = torch.cat(
            (*neighbor_feature_groups, neighbor_vehicle_ids.unsqueeze(-1)), dim=-1
        ).reshape(observation.shape[0], -1)
        return torch.cat(
            (
                observation[:, :neighbor_start],
                ego_vehicle_id,
                neighbor_features,
                observation[:, neighbor_end:],
            ),
            dim=-1,
        )

    def set_pre_reset_render_callback(self, callback):
        """Configure an optional video-only callback for collision frames."""
        self._pre_reset_render_callback = callback
        self._pre_reset_render_frame = None

    def consume_pre_reset_render_frame(self):
        """Return and clear the latest pre-reset collision frame."""
        frame = getattr(self, "_pre_reset_render_frame", None)
        self._pre_reset_render_frame = None
        return frame

    def _capture_pre_reset_collision_frame(self, collision_envs):
        """Capture env 0 before a collision reset when video capture is enabled."""
        callback = getattr(self, "_pre_reset_render_callback", None)
        if callback is None or not bool(collision_envs[0].item()):
            return
        self._pre_reset_render_frame = callback()

    @staticmethod
    def _format_render_agent_indices(agent_mask: Tensor) -> str:
        """Format a one-dimensional agent mask using the internal agent IDs."""
        indices = torch.where(agent_mask)[0].tolist()
        return f"[{', '.join(str(index) for index in indices)}]"

    def _format_env_reset_render_info(
        self,
        *,
        reset_time: float,
        success,
        failure,
        collision_agents: Tensor,
        deadlock_agents: Tensor,
        max_steps_reached,
        fixed_duration_reset,
    ) -> str:
        """Format a whole-environment reset reason from pre-reset event state."""
        failure_details = []
        if bool(collision_agents.any().item()):
            failure_details.append(
                "collisions of "
                f"{self._format_render_agent_indices(collision_agents)}"
            )
        if bool(deadlock_agents.any().item()):
            failure_details.append(
                "deadlocks of " f"{self._format_render_agent_indices(deadlock_agents)}"
            )
        if bool(max_steps_reached):
            failure_details.append("maximum steps reached")
        if bool(fixed_duration_reset):
            failure_details.append("fixed-duration reset reached")

        if bool(success) and not failure_details and not bool(failure):
            return f"(task succeeded at {reset_time:.2f} s)"

        reasons = []
        if bool(success):
            reasons.append(f"task succeeded at {reset_time:.2f} s")
        if bool(failure) or failure_details:
            if not failure_details:
                failure_details.append("unspecified failure")
            reasons.append(
                f"task failed at {reset_time:.2f} s: {'; '.join(failure_details)}"
            )
        if not reasons:
            reasons.append(f"reset at {reset_time:.2f} s: explicit reset")
        return f"({'; '.join(reasons)})"

    def _format_agent_reset_render_info(
        self,
        *,
        reset_time: float,
        reset_agents: Tensor,
        collision_agents: Tensor,
        deadlock_agents: Tensor,
        success_agents: Tensor,
        entry_agents: Tensor,
        nearby_collision_agents: Tensor,
    ) -> str:
        """Format a selective agent-reset reason from pre-reset event state."""
        failure_details = []
        collision_agents = reset_agents & collision_agents
        deadlock_agents = reset_agents & deadlock_agents
        if bool(collision_agents.any().item()):
            failure_details.append(
                "collisions of "
                f"{self._format_render_agent_indices(collision_agents)}"
            )
        if bool(deadlock_agents.any().item()):
            failure_details.append(
                "deadlocks of " f"{self._format_render_agent_indices(deadlock_agents)}"
            )

        reasons = []
        if failure_details:
            reasons.append(f"task failure: {'; '.join(failure_details)}")

        success_agents = reset_agents & success_agents
        if bool(success_agents.any().item()):
            reasons.append(
                "task success: " f"{self._format_render_agent_indices(success_agents)}"
            )

        entry_agents = reset_agents & entry_agents
        if bool(entry_agents.any().item()):
            reasons.append(
                "leaving entry segment: "
                f"{self._format_render_agent_indices(entry_agents)}"
            )

        nearby_collision_agents = reset_agents & nearby_collision_agents
        if bool(nearby_collision_agents.any().item()):
            reasons.append(
                "nearby collision: "
                f"{self._format_render_agent_indices(nearby_collision_agents)}"
            )

        if not reasons:
            reasons.append("explicit reset")
        return f"(agent(s) reset at {reset_time:.2f} s: {'; '.join(reasons)})"

    def _record_reset_render_info(
        self,
        *,
        env_reset_indices,
        agent_reset_indices,
        agents_reset: Tensor,
        collision_participant_mask: Tensor,
        deadlock_agents_reset: Tensor,
        success_agents: Tensor,
        entry_agents: Tensor,
        nearby_collision_agents: Tensor,
        is_joint_task: bool,
        is_collision_reset_env: Tensor,
        is_max_steps_reached: Tensor,
        is_fixed_duration_reset: Tensor,
    ):
        """Latch formatted reset text before reset state is cleared."""
        has_env_reset = env_reset_indices.numel() > 0
        has_agent_reset = agent_reset_indices[0].numel() > 0
        if has_env_reset or has_agent_reset:
            reset_wall_time = time.monotonic()
            reset_sim_times = (
                (self.timer.step * self.parameters.dt).detach().cpu().tolist()
            )
        else:
            reset_wall_time = None
            reset_sim_times = None

        for env_idx in env_reset_indices:
            env_i = int(env_idx.item())
            if is_joint_task:
                collision_agents = (
                    self._joint_task_collision_agents_event[env_idx]
                    | self._joint_task_collision_lanelets_event[env_idx]
                )
                failure = self._joint_task_failure_event[env_idx]
                success = self._joint_task_success_event[env_idx]
            else:
                collision_agents = collision_participant_mask[env_idx]
                failure = (
                    is_collision_reset_env[env_idx]
                    | is_max_steps_reached[env_idx]
                    | is_fixed_duration_reset[env_idx]
                )
                success = False

            reset_sim_time = float(reset_sim_times[env_i])
            self._last_reset_render_info[env_i] = self._format_env_reset_render_info(
                reset_time=reset_sim_time,
                success=success,
                failure=failure,
                collision_agents=collision_agents,
                deadlock_agents=deadlock_agents_reset[env_idx],
                max_steps_reached=is_max_steps_reached[env_idx],
                fixed_duration_reset=is_fixed_duration_reset[env_idx],
            )
            has_failure = (
                bool(failure)
                or bool(collision_agents.any().item())
                or bool(deadlock_agents_reset[env_idx].any().item())
                or bool(is_max_steps_reached[env_idx])
                or bool(is_fixed_duration_reset[env_idx])
            )
            self._last_reset_render_color[env_i] = (
                Color.red100
                if has_failure
                else Color.blue100
                if bool(success)
                else None
            )
            self._last_reset_render_time[env_i] = reset_wall_time

        reset_envs = set()
        for env_idx in agent_reset_indices[0]:
            env_i = int(env_idx.item())
            if env_i not in reset_envs:
                reset_envs.add(env_i)
                self._last_reset_render_info[
                    env_i
                ] = self._format_agent_reset_render_info(
                    reset_time=float(reset_sim_times[env_i]),
                    reset_agents=agents_reset[env_idx],
                    collision_agents=collision_participant_mask[env_idx],
                    deadlock_agents=deadlock_agents_reset[env_idx],
                    success_agents=success_agents[env_idx],
                    entry_agents=entry_agents[env_idx],
                    nearby_collision_agents=nearby_collision_agents[env_idx],
                )
                reset_agents = agents_reset[env_idx]
                has_failure = bool(
                    (reset_agents & collision_participant_mask[env_idx]).any().item()
                    or (reset_agents & deadlock_agents_reset[env_idx]).any().item()
                    or (reset_agents & nearby_collision_agents[env_idx]).any().item()
                )
                has_success = bool(
                    (reset_agents & success_agents[env_idx]).any().item()
                )
                self._last_reset_render_color[env_i] = (
                    Color.red100
                    if has_failure
                    else Color.blue100
                    if has_success
                    else None
                )
                self._last_reset_render_time[env_i] = reset_wall_time

    def done(self):
        """This function computes the done flag for each env in a vectorized way.

        The returned tensor should contain the ``done`` for all envs and should have
        shape ``(n_envs)`` and dtype ``torch.bool``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        By default, this function returns all ``False`` s.

        The scenario can still be done if ``max_steps`` has been set at envirtonment construction.

        Returns:
            torch.Tensor: done tensor of shape ``(self.world.batch_dim)``
        """
        if self.log_replay is not None:
            # The log replay spawns and parks agents itself; never reset
            return self.timer.step == (self.parameters.max_steps - 1)

        if self.parameters.is_fast_inference_mode:
            if self.parameters.reset_agent_fixed_duration > 0:
                t = self.timer.step * self.parameters.dt
                is_fixed_duration_reset = (
                    t % self.parameters.reset_agent_fixed_duration == 0
                ) & (t != 0)
            else:
                is_fixed_duration_reset = torch.zeros(
                    self.world.batch_dim, dtype=torch.bool, device=self.world.device
                )
            is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)
            return is_max_steps_reached | is_fixed_duration_reset

        # Snapshot these masks before any reset changes collision flags or
        # pairwise distances. The same neighborhood was also used for the
        # current transition's reward when the reward function ran first.
        self._update_collision_neighborhood_masks()
        collision_participant_mask = self._collision_participant_mask.clone()
        near_collision_reset_mask = self._near_collision_reset_mask.clone()

        if self.parameters.reset_agent_fixed_duration > 0:
            # Reset agents at fixed time intervals
            t = self.timer.step * self.parameters.dt  # Time
            is_fixed_duration_reset = (
                t % self.parameters.reset_agent_fixed_duration == 0
            ) & (t != 0)
        else:
            is_fixed_duration_reset = torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            )

        is_collision_with_agents = self.world_state.collisions.with_agents.view(
            self.world.batch_dim, -1
        ).any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.world_state.collisions.with_lanelets.any(
            dim=-1
        )
        if self.parameters.without_lanelet_reset:
            is_collision_reset_env = is_collision_with_agents
        else:
            is_collision_reset_env = (
                is_collision_with_agents | is_collision_with_lanelets
            )
        is_leaving_entry_segment = self.world_state.collisions.with_entry_segments.any(
            dim=-1
        ) & (self.timer.step >= 20)
        is_any_agents_leaving_exit_segment = (
            self.world_state.collisions.with_exit_segments.any(dim=-1)
        )
        is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)
        deadlock_agents_reset = (
            self.deadlock_reset_agents
            if self._is_deadlock_enabled()
            else torch.zeros_like(self.deadlock_reset_agents)
        )
        deadlock_info_snapshot = self.is_deadlocked_agents.clone()

        if (
            self.parameters.is_challenging_initial_state_buffer
        ):  # Record challenging initial states
            if torch.rand(1) > (
                1 - self.initial_state_buffer.probability_record
            ):  # Only a certain probability to record
                for env_collide in torch.where(is_collision_with_agents)[0]:
                    self.initial_state_buffer.add(
                        self.state_buffer.get_latest(n=self.parameters.n_steps_stored)[
                            env_collide
                        ]
                    )
                    # print(colored(f"[LOG] Record states with path ids: {self.ref_paths_agent_related.path_id[env_collide]}.", "blue"))

        is_joint_task = bool(getattr(self.parameters, "is_joint_task", False))
        if self.parameters.is_testing_mode:
            is_done = (
                is_max_steps_reached | is_fixed_duration_reset
            )  # In test mode, we only reset the whole env if the maximum time steps are reached or the fixed duration is reached (if configured)

            if is_joint_task:
                joint_task_terminal = (
                    self._joint_task_success_event | self._joint_task_failure_event
                )
                is_done = is_done | joint_task_terminal
                if not hasattr(self, "_preserve_timer_on_next_reset"):
                    self._preserve_timer_on_next_reset = torch.zeros_like(
                        joint_task_terminal
                    )
                preserve_timer = joint_task_terminal & ~(
                    is_max_steps_reached | is_fixed_duration_reset
                )
                env_reset_mask = is_done
                agents_reset = torch.zeros_like(self._joint_task_finished_agents)
            elif self.parameters.is_reset_all_agents_in_test_mode_when_collision:
                # A collision is a team-level failure in this mode. Reset all
                # agents in the affected sub-environment, while leaving other
                # parallel environments untouched.
                env_reset_mask = is_done | is_collision_reset_env

                # Preserve single-agent resets for non-collision events.
                agents_reset = ~is_collision_reset_env.unsqueeze(-1) & (
                    self.world_state.collisions.with_entry_segments
                    | self.world_state.collisions.with_exit_segments
                    | deadlock_agents_reset
                )
            else:
                # Preserve the previous selective collision-neighborhood
                # reset behavior when team-level resetting is disabled.
                env_reset_mask = is_done
                agents_reset = (
                    collision_participant_mask
                    | self.world_state.collisions.with_lanelets
                    | self.world_state.collisions.with_entry_segments
                    | self.world_state.collisions.with_exit_segments
                    | deadlock_agents_reset
                    | near_collision_reset_mask
                )
        else:
            # If not in testing mode, we reset all agents in an env if any agent collides with others or lanelets
            is_done = (
                is_max_steps_reached
                | is_collision_with_agents
                | is_collision_with_lanelets
                | is_fixed_duration_reset
            )
            if (
                self.parameters.scenario_type != "cpm_entire"
            ):  # This part only applies to the map that have loop-shaped paths
                # Reset the whole system only when collisions occur. Reset a single agents if it leaves an entry or an exit

                # Reset single agnet
                agents_reset = (
                    self.world_state.collisions.with_entry_segments
                    | self.world_state.collisions.with_exit_segments
                    | deadlock_agents_reset
                )
            else:
                # Reset the whole system once collisions occur. There is no entry or exit in this scenario.
                assert not is_leaving_entry_segment.any()
                assert not is_any_agents_leaving_exit_segment.any()
                agents_reset = deadlock_agents_reset

            env_reset_mask = is_done

        # Agents in an environment that is already done are reset by the
        # subsequent environment reset, not by the selective reset loop below.
        agents_reset = agents_reset & ~is_done.unsqueeze(-1)
        env_reset_indices = torch.where(env_reset_mask)[0]
        agents_reset_indices = torch.where(agents_reset)
        self._record_reset_render_info(
            env_reset_indices=env_reset_indices,
            agent_reset_indices=agents_reset_indices,
            agents_reset=agents_reset,
            collision_participant_mask=collision_participant_mask,
            deadlock_agents_reset=deadlock_agents_reset,
            success_agents=self.world_state.collisions.with_exit_segments,
            entry_agents=self.world_state.collisions.with_entry_segments,
            nearby_collision_agents=near_collision_reset_mask,
            is_joint_task=is_joint_task,
            is_collision_reset_env=is_collision_reset_env,
            is_max_steps_reached=is_max_steps_reached,
            is_fixed_duration_reset=is_fixed_duration_reset,
        )

        # Capture only the rendered environment's collision state. This runs
        # after the reason is latched but before any reset and is consumed by
        # the video rollout callback.
        self._capture_pre_reset_collision_frame(is_collision_reset_env)

        if self.parameters.is_testing_mode and is_joint_task:
            for env_idx in env_reset_indices:
                preserve_timer_for_env = bool(preserve_timer[env_idx].item())
                self._preserve_timer_on_next_reset[env_idx] = preserve_timer_for_env
                self.reset_world_at(
                    env_index=env_idx,
                    agent_index=None,
                    preserve_timer=preserve_timer_for_env,
                )
        elif (
            self.parameters.is_testing_mode
            and self.parameters.is_reset_all_agents_in_test_mode_when_collision
        ):
            for env_idx in torch.where(is_collision_reset_env & ~is_done)[0]:
                if self.parameters.is_enable_debug_diagnostics:
                    print("[DEBUG] Resetting all agents in envs with collisions...")
                self.reset_world_at(env_index=env_idx, agent_index=None)

        for env_idx, agent_idx in zip(agents_reset_indices[0], agents_reset_indices[1]):
            self.reset_world_at(env_index=env_idx, agent_index=agent_idx)

        if not self.parameters.is_testing_mode:
            assert not (is_collision_with_agents & (self.timer.step == 0)).any()
            assert not (is_collision_with_lanelets & (self.timer.step == 0)).any()
            assert not (is_leaving_entry_segment & (self.timer.step == 0)).any()
            assert not (is_max_steps_reached & (self.timer.step == 0)).any()
            assert not (
                is_any_agents_leaving_exit_segment & (self.timer.step == 0)
            ).any()

        self.is_deadlocked_agents |= deadlock_info_snapshot

        return is_done

    def _get_cbf_circle_geometry(self):
        n_circles = int(self.parameters.n_circles_approximate_vehicle)
        cache = getattr(self, "_cbf_boundary_diag_geometry", None)
        if cache is not None and cache["n_circles"] == n_circles:
            return cache["centers"], cache["radius"]

        approximation = RectangleCircleApproximation(
            AGENTS["length"], AGENTS["width"], n_circles
        )
        centers = torch.tensor(
            approximation.centers,
            device=self.world.device,
            dtype=torch.float32,
        )
        radius = float(approximation.radius)
        self._cbf_boundary_diag_geometry = {
            "n_circles": n_circles,
            "centers": centers,
            "radius": radius,
        }
        return centers, radius

    def _compute_cbf_boundary_diagnostics(
        self, agent: Agent, agent_index: int
    ) -> Dict[str, Tensor]:
        batch_dim = self.world.batch_dim
        device = self.world.device
        nan_values = {
            name: torch.full((batch_dim,), float("nan"), device=device)
            for name in ("left_raw", "right_raw", "left_margin", "right_margin")
        }

        if not getattr(self.parameters, "is_log_cbf_boundary_distance", False):
            return nan_values

        if self.lane_sat_barrier is not None:
            return self._compute_lane_sat_boundary_diagnostics(agent, agent_index)

        if self.map_pseudo_distance is None:
            return nan_values

        local_centers, circle_radius = self._get_cbf_circle_geometry()
        yaw = agent.state.rot.squeeze(-1)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        left_raw = nan_values["left_raw"].clone()
        right_raw = nan_values["right_raw"].clone()
        left_margin = nan_values["left_margin"].clone()
        right_margin = nan_values["right_margin"].clone()

        for env_i in range(batch_dim):
            rotation = torch.stack(
                [
                    torch.stack([cos_yaw[env_i], -sin_yaw[env_i]]),
                    torch.stack([sin_yaw[env_i], cos_yaw[env_i]]),
                ]
            )
            circle_centers = local_centers @ rotation.T + agent.state.pos[env_i]
            path_id = int(
                self.world_state.ref_paths_agent_related.path_id[
                    env_i, agent_index
                ].item()
            )

            try:
                left_values, right_values = self.map_pseudo_distance.get_distance(
                    path_id, circle_centers.detach().cpu()
                )
            except Exception as exc:
                if not getattr(self, "_warned_cbf_boundary_diag_failure", False):
                    print(
                        "[WARNING] Failed to compute CBF boundary diagnostics: "
                        f"{exc}"
                    )
                    self._warned_cbf_boundary_diag_failure = True
                continue

            left_tensor = torch.as_tensor(
                left_values, device=device, dtype=torch.float32
            )
            right_tensor = torch.as_tensor(
                right_values, device=device, dtype=torch.float32
            )
            left_raw[env_i] = left_tensor.min()
            right_raw[env_i] = right_tensor.min()

            if self.parameters.lane_distance_type == "softmin":
                margin_offset = circle_radius**2
            else:
                margin_offset = circle_radius
            left_margin[env_i] = left_raw[env_i] - margin_offset
            right_margin[env_i] = right_raw[env_i] - margin_offset

        return {
            "left_raw": left_raw,
            "right_raw": right_raw,
            "left_margin": left_margin,
            "right_margin": right_margin,
        }

    def _compute_lane_sat_boundary_diagnostics(
        self, agent: Agent, agent_index: int
    ) -> Dict[str, Tensor]:
        """Boundary diagnostics of the rectangle_sat lane barrier.

        "raw" is the barrier without buffers, a lower bound on the distance
        between the vehicle rectangle and the map boundary polyline [m].
        "margin" is the barrier with the safety and resampling buffers, i.e.,
        the value the CBF keeps nonnegative [m].
        """
        poses = torch.cat([agent.state.pos, agent.state.rot], dim=-1)
        raw = torch.empty((self.world.batch_dim, 2), dtype=torch.float32)
        margin = torch.empty_like(raw)
        for env_i, pose in enumerate(poses.detach().cpu().tolist()):
            for values, is_buffered in ((raw, False), (margin, True)):
                values[env_i] = torch.from_numpy(
                    self.lane_sat_barrier.side_margins(
                        self.world_state, env_i, agent_index, pose, is_buffered
                    )
                )
        raw = raw.to(self.world.device)
        margin = margin.to(self.world.device)
        return {
            "left_raw": raw[:, 0],
            "right_raw": raw[:, 1],
            "left_margin": margin[:, 0],
            "right_margin": margin[:, 1],
        }

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        This function computes the info dict for "agent" in a vectorized way
        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape (n_envs, info_size)

        Implementors can access the world at "self.world"

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        if self.parameters.is_fast_inference_mode:
            return {
                "pos": agent.state.pos,
                "rot": angle_eliminate_two_pi(agent.state.rot),
                "vel": agent.state.vel,
            }

        agent_index = self.world.agents.index(agent)  # Index of the current agent

        is_action_empty = agent.action.u is None

        if self._is_joint_task_enabled():
            (
                is_collision_with_agents,
                is_collision_with_lanelets,
            ) = self._joint_task_collision_info(agent_index)
            is_reach_goal = self._joint_task_exit_event[:, agent_index]
            is_joint_task_on_hold = self._joint_task_finished_agents[:, agent_index]
        else:
            is_collision_with_agents = self.world_state.collisions.with_agents[
                :, agent_index
            ].any(
                dim=-1
            )  # [batch_dim]
            is_collision_with_lanelets = self.world_state.collisions.with_lanelets[
                :, agent_index
            ]
            is_reach_goal = self.world_state.collisions.with_exit_segments[
                :, agent_index
            ]
            is_joint_task_on_hold = torch.zeros_like(is_reach_goal)

        joint_task_success = (
            self._joint_task_success_event
            if self._is_joint_task_enabled()
            else torch.zeros_like(is_reach_goal)
        )
        joint_task_failure = (
            self._joint_task_failure_event
            if self._is_joint_task_enabled()
            else torch.zeros_like(is_reach_goal)
        )
        joint_task_try = (
            self._joint_task_try_event
            if self._is_joint_task_enabled()
            else torch.zeros_like(is_reach_goal)
        )
        is_cbf_active = bool(
            self.parameters.is_using_cbf_testing
            or self.parameters.is_using_cbf_training
        )
        needs_observation_without_actions = (
            self.parameters.is_save_detailed_info
            or self.parameters.is_using_opponent_modeling
            or self.parameters.is_using_prioritized_marl
            or self.parameters.is_obs_vehicle_id_in_motion_policy
            or self.parameters.is_priority_conditioned_motion
        )
        needs_policy_observation = (
            needs_observation_without_actions
            or self.parameters.is_using_prioritized_marl
            or self.parameters.is_using_opponent_modeling
        )

        stored_observation = (
            self.stored_observations[agent_index].clone()
            if needs_policy_observation
            else None
        )
        diagnostics = None
        opponent_modeling_neighbor_valid = None
        if self.parameters.is_using_opponent_modeling:
            diagnostics = (
                self.observation_provider.get_observed_agent_render_diagnostics(
                    agent_index
                )
            )
            if diagnostics is None:
                opponent_modeling_neighbor_valid = stored_observation.new_ones(
                    (
                        stored_observation.shape[0],
                        int(self.parameters.n_nearing_agents_observed),
                    )
                )
            else:
                opponent_modeling_neighbor_valid = (~diagnostics["final_mask"]).to(
                    dtype=stored_observation.dtype
                )

        priority_obs = None
        observation_without_actions = None
        base_obs = None
        if needs_policy_observation:
            # Observation of the policy in the priority-assignment module.
            priority_obs = stored_observation.clone()
            if (
                self.parameters.is_using_prioritized_marl
                and self.parameters.is_obs_vehicle_id_in_priority_policy
            ):
                priority_obs = self._append_vehicle_ids_to_observation(
                    stored_observation, agent_index
                )

            if needs_observation_without_actions:
                observation_without_actions = stored_observation.clone()
                if self.parameters.is_obs_vehicle_id_in_motion_policy:
                    observation_without_actions = (
                        self._append_vehicle_ids_to_observation(
                            stored_observation, agent_index
                        )
                    )

            if (
                self.parameters.is_save_detailed_info
                or self.parameters.is_using_opponent_modeling
            ):
                # Zero-padding as a placeholder for actions of surrounding agents.
                base_obs = F.pad(
                    observation_without_actions.clone(),
                    (
                        0,
                        self.parameters.n_nearing_agents_observed * AGENTS["n_actions"],
                    ),
                )

        priority_neighbor_indices = None
        priority_neighbor_valid = None
        if self.parameters.is_priority_conditioned_motion:
            if diagnostics is None:
                diagnostics = (
                    self.observation_provider.get_observed_agent_render_diagnostics(
                        agent_index
                    )
                )
            if diagnostics is None:
                raise RuntimeError(
                    "Priority-conditioned motion requires observed-neighbor "
                    "metadata before scenario info is collected."
                )
            priority_neighbor_indices = diagnostics["neighbor_indices"]
            priority_neighbor_valid = (~diagnostics["final_mask"]).to(
                dtype=priority_obs.dtype
            )

        if (not self.parameters.is_using_cbf_testing) and (
            not self.parameters.is_using_cbf_training
        ):
            # Nominal and applied actions are considered the same if not using CBF, since there is no modification on the original action from the policy
            self.world_state.nominal_action_vel[:, agent_index] = (
                agent.action.u[:, 0]
                if not is_action_empty
                else self.constants.empty_action_vel[:, agent_index]
            )
            self.world_state.nominal_action_steer[:, agent_index] = (
                agent.action.u[:, 1]
                if not is_action_empty
                else self.constants.empty_action_steering[:, agent_index]
            )
            self.world_state.applied_action_vel[:, agent_index] = (
                agent.action.u[:, 0]
                if not is_action_empty
                else self.constants.empty_action_vel[:, agent_index]
            )
            self.world_state.applied_action_steer[:, agent_index] = (
                agent.action.u[:, 1]
                if not is_action_empty
                else self.constants.empty_action_steering[:, agent_index]
            )
        # print agent speed
        # print(f"Agent {agent_index} speed: {agent.state.speed}")
        info = {
            "pos": agent.state.pos,
            "rot": angle_eliminate_two_pi(agent.state.rot),
            "vel": agent.state.vel,
            "is_collision_with_agents": is_collision_with_agents,
            "is_collision_with_lanelets": is_collision_with_lanelets,
            "is_reach_goal": is_reach_goal,
            "is_deadlocked": self.is_deadlocked_agents[:, agent_index],
            "is_joint_task_success": joint_task_success,
            "is_joint_task_failure": joint_task_failure,
            "is_joint_task_try": joint_task_try,
            "is_joint_task_on_hold": is_joint_task_on_hold,
            "num_task_tries": self.num_task_tries.clone(),
            "task_success_times": self.task_success_times.clone(),
            "step_time": agent.state.pos.new_full(
                (self.world.batch_dim,), float(self.parameters.dt)
            ),
        }

        if is_cbf_active:
            info.update(
                {
                    "ref": self.world_state.ref_paths_agent_related.short_term[
                        :, agent_index
                    ].reshape(self.world.batch_dim, -1),
                    "path_id": self.world_state.ref_paths_agent_related.path_id[
                        :, agent_index
                    ],
                }
            )

        if self.parameters.is_save_detailed_info:
            # Only detailed info stores the boundary diagnostics
            cbf_boundary_diag = self._compute_cbf_boundary_diagnostics(
                agent=agent, agent_index=agent_index
            )
            info.update(
                {
                    "pos_nom": agent.state.pos / self.normalizers.pos_world,
                    "rot_nom": angle_eliminate_two_pi(agent.state.rot)
                    / self.normalizers.rot,
                    "speed": agent.state.speed,
                    "speed_nom": agent.state.speed / self.normalizers.v,
                    "steering": agent.state.steering,
                    "steering_nom": agent.state.steering / self.normalizers.steering,
                    "acceleration": agent.state.acceleration,
                    "acceleration_nom": agent.state.acceleration / AGENTS["max_acc"],
                    "steering_rate": agent.state.steering_rate,
                    "steering_rate_nom": (
                        agent.state.steering_rate / AGENTS["max_steering_rate"]
                    ),
                    "vel_nom": agent.state.vel / self.normalizers.v,
                    "act_vel": (
                        agent.action.u[:, 0]
                        if not is_action_empty
                        else self.constants.empty_action_vel[:, agent_index]
                    ),
                    "act_vel_nom": (
                        (agent.action.u[:, 0] / self.normalizers.v)
                        if not is_action_empty
                        else self.constants.empty_action_vel[:, agent_index]
                    ),
                    "act_steer": (
                        agent.action.u[:, 1]
                        if not is_action_empty
                        else self.constants.empty_action_steering[:, agent_index]
                    ),
                    "act_steer_nom": (
                        (agent.action.u[:, 1] / self.normalizers.steering)
                        if not is_action_empty
                        else self.constants.empty_action_steering[:, agent_index]
                    ),
                    "ref": self.world_state.ref_paths_agent_related.short_term[
                        :, agent_index
                    ].reshape(self.world.batch_dim, -1),
                    "ref_nom": (
                        self.world_state.ref_paths_agent_related.short_term[
                            :, agent_index
                        ]
                        / self.normalizers.pos_world
                    ).reshape(self.world.batch_dim, -1),
                    "distance_ref": self.world_state.distances.ref_paths[
                        :, agent_index
                    ],
                    "distance_ref_nom": self.world_state.distances.ref_paths[
                        :, agent_index
                    ]
                    / self.normalizers.distance_ref,
                    "distance_left_b": self.world_state.distances.left_boundaries[
                        :, agent_index
                    ].min(dim=-1)[0],
                    "distance_left_b_nom": self.world_state.distances.left_boundaries[
                        :, agent_index
                    ].min(dim=-1)[0]
                    / self.normalizers.distance_lanelet,
                    "distance_right_b": self.world_state.distances.right_boundaries[
                        :, agent_index
                    ].min(dim=-1)[0],
                    "distance_right_b_nom": self.world_state.distances.right_boundaries[
                        :, agent_index
                    ].min(dim=-1)[0]
                    / self.normalizers.distance_lanelet,
                    "cbf_boundary_left_raw": cbf_boundary_diag["left_raw"],
                    "cbf_boundary_right_raw": cbf_boundary_diag["right_raw"],
                    "cbf_boundary_left_margin": cbf_boundary_diag["left_margin"],
                    "cbf_boundary_right_margin": cbf_boundary_diag["right_margin"],
                    "ref_lanelet_ids": self.world_state.ref_paths_agent_related.ref_lanelet_ids[
                        :, agent_index
                    ],
                    "path_id": self.world_state.ref_paths_agent_related.path_id[
                        :, agent_index
                    ],
                    "applied_action_vel": self.world_state.applied_action_vel[
                        :, agent_index
                    ],
                    "applied_action_steer": self.world_state.applied_action_steer[
                        :, agent_index
                    ],
                    "nominal_action_vel": self.world_state.nominal_action_vel[
                        :, agent_index
                    ],
                    "nominal_action_steer": self.world_state.nominal_action_steer[
                        :, agent_index
                    ],
                }
            )

            if (
                self.parameters.is_using_prioritized_marl
                and self.parameters.prioritization_method == "taskprogress"
            ):
                eps = torch.finfo(self.world_state._route_progress.dtype).eps
                current_progress = self.world_state._route_progress[:, agent_index]
                total_route_length = self.world_state._route_progress_total_lengths[
                    :, agent_index
                ]
                progress_frac = torch.nan_to_num(
                    current_progress / total_route_length.clamp_min(eps),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)

                all_progress_frac = torch.nan_to_num(
                    self.world_state._route_progress
                    / self.world_state._route_progress_total_lengths.clamp_min(eps),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)

                if (
                    hasattr(self, "priority_scores")
                    and self.priority_scores is not None
                ):
                    priority_scores_all = self.priority_scores
                    if priority_scores_all.ndim > 2:
                        priority_scores_all = priority_scores_all.squeeze(-1)
                else:
                    priority_scores_all = all_progress_frac

                ordered_indices = torch.argsort(
                    priority_scores_all, dim=-1, descending=True, stable=True
                )
                all_ranks = torch.argsort(ordered_indices, dim=-1).to(
                    dtype=priority_scores_all.dtype
                )
                all_weights = torch.softmax(priority_scores_all, dim=-1)

                ego_priority_score = priority_scores_all[:, agent_index]
                ego_priority_rank = all_ranks[:, agent_index]
                ego_priority_weight = all_weights[:, agent_index]

                ego_speed = agent.state.speed.reshape(self.world.batch_dim)
                ego_acc = agent.state.acceleration.reshape(self.world.batch_dim)

                n_agents = len(self.world.agents)
                if n_agents > 1:
                    distances_all = self.world_state.distances.agents[
                        :, agent_index, :
                    ].clone()
                    distances_all[:, agent_index] = float("inf")
                    min_dist_to_agents, nearest_agent_idx = distances_all.min(dim=-1)
                    nearest_agent_idx_expanded = nearest_agent_idx.unsqueeze(-1)

                    nearest_priority_score = torch.gather(
                        priority_scores_all, dim=-1, index=nearest_agent_idx_expanded
                    ).squeeze(-1)
                    nearest_priority_rank = torch.gather(
                        all_ranks, dim=-1, index=nearest_agent_idx_expanded
                    ).squeeze(-1)
                    nearest_priority_weight = torch.gather(
                        all_weights, dim=-1, index=nearest_agent_idx_expanded
                    ).squeeze(-1)

                    all_speeds = torch.stack(
                        [
                            a.state.speed.reshape(self.world.batch_dim)
                            for a in self.world.agents
                        ],
                        dim=-1,
                    )
                    nearest_agent_speed = torch.gather(
                        all_speeds, dim=-1, index=nearest_agent_idx_expanded
                    ).squeeze(-1)

                    all_accs = torch.stack(
                        [
                            a.state.acceleration.reshape(self.world.batch_dim)
                            for a in self.world.agents
                        ],
                        dim=-1,
                    )
                    nearest_agent_acc = torch.gather(
                        all_accs, dim=-1, index=nearest_agent_idx_expanded
                    ).squeeze(-1)

                    priority_difference_nearest = (
                        ego_priority_score - nearest_priority_score
                    )
                    rank_difference_nearest = nearest_priority_rank - ego_priority_rank
                    speed_difference_nearest = ego_speed - nearest_agent_speed
                    acc_difference_nearest = ego_acc - nearest_agent_acc
                else:
                    zeros = torch.zeros(
                        self.world.batch_dim,
                        dtype=ego_speed.dtype,
                        device=ego_speed.device,
                    )
                    min_dist_to_agents = torch.full(
                        (self.world.batch_dim,),
                        float("inf"),
                        dtype=ego_speed.dtype,
                        device=ego_speed.device,
                    )
                    nearest_agent_idx = torch.zeros(
                        self.world.batch_dim,
                        dtype=torch.long,
                        device=agent.state.pos.device,
                    )
                    nearest_priority_score = ego_priority_score
                    nearest_priority_rank = ego_priority_rank
                    nearest_priority_weight = ego_priority_weight
                    nearest_agent_speed = ego_speed
                    nearest_agent_acc = ego_acc
                    priority_difference_nearest = zeros
                    rank_difference_nearest = zeros
                    speed_difference_nearest = zeros
                    acc_difference_nearest = zeros

                info.update(
                    {
                        "route_progress": current_progress,
                        "route_total_length": total_route_length,
                        "progress_fraction": progress_frac,
                        "priority_score": ego_priority_score,
                        "priority_rank": ego_priority_rank,
                        "priority_weight": ego_priority_weight,
                        "min_distance_to_agents": min_dist_to_agents,
                        "nearest_agent_index": nearest_agent_idx,
                        "nearest_agent_distance": min_dist_to_agents,
                        "nearest_agent_priority_score": nearest_priority_score,
                        "nearest_agent_priority_rank": nearest_priority_rank,
                        "nearest_agent_priority_weight": nearest_priority_weight,
                        "nearest_agent_speed": nearest_agent_speed,
                        "nearest_agent_acceleration": nearest_agent_acc,
                        "priority_difference_nearest": priority_difference_nearest,
                        "rank_difference_nearest": rank_difference_nearest,
                        "speed_difference_nearest": speed_difference_nearest,
                        "acc_difference_nearest": acc_difference_nearest,
                    }
                )

            info.update(
                {
                    "base_observation": base_obs,
                    "priority_observation": priority_obs,
                    "observation_with_actions": base_obs,
                    "observation_without_actions": observation_without_actions,
                }
            )
        else:
            if self.parameters.is_using_prioritized_marl:
                info["priority_observation"] = priority_obs
            if needs_observation_without_actions:
                info["observation_without_actions"] = observation_without_actions
            if self.parameters.is_using_opponent_modeling:
                info["observation_with_actions"] = base_obs
        if opponent_modeling_neighbor_valid is not None:
            info["opponent_modeling_neighbor_valid"] = opponent_modeling_neighbor_valid
        if self.parameters.is_priority_conditioned_motion:
            info.update(
                {
                    "priority_neighbor_indices": priority_neighbor_indices,
                    "priority_neighbor_valid": priority_neighbor_valid,
                }
            )

        if self.parameters.is_save_detailed_info:
            reward_info_agent = {
                f.name: getattr(self.reward_info, f.name)[:, agent_index]
                for f in fields(RewardInfo)
            }
            info.update(reward_info_agent)
        else:
            for name in TASK_METRIC_REWARDS:
                info[name] = getattr(self.reward_info, name)[:, agent_index]

        return info

    def extra_render(self, env_index: int = 0):
        if rendering is None:
            return []

        if self.parameters.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0  # Not sure how long should the simulation be paused at time step 0, so rather 0
            else:
                pause_duration = self.world.dt - (time.time() - self.timer.render_begin)
            if pause_duration > 0:
                time.sleep(pause_duration)
            # print(f"Paused for {pause_duration} sec.")

            self.timer.render_begin = time.time()  # Update

        # Static map first: its filled road surface is the bottom layer
        geoms = list(self._get_static_lanelet_geoms())

        # self._render_action_propagation_direction(env_index, geoms) # Discard this function because action propagation is no longer used
        self._render_cbf_action(env_index, geoms)
        self._render_shield_rollout(env_index, geoms)

        self._render_on_hold_agents(env_index, geoms)

        self._render_human_overlay(env_index, geoms)

        self._render_extra_info(geoms, env_index)

        self._render_cbf_lambdas(env_index, geoms)

        self._render_position_speed_targets(env_index, geoms)

        self._render_reference_paths_and_boundaries(env_index, geoms)

        self._render_observed_agents(
            env_index,
            geoms,
            ego_agent_index=getattr(
                self.parameters, "observed_agents_render_ego_index", 0
            ),
        )

        self._render_entry_exit(env_index, geoms)

        self._render_groups(geoms)

        self._render_side_panel_background(geoms)

        return geoms

    def _render_on_hold_agents(self, env_index: int, geoms):
        """Render held vehicles with a more transparent replacement geometry."""
        if self.log_replay is not None:
            # Parked agents are off the map
            for agent_index, agent in enumerate(self.world.agents):
                agent.is_rendering[env_index] = not bool(
                    self.log_replay.parked[env_index, agent_index].item()
                )
            return
        if not self._is_joint_task_enabled():
            return

        for agent_index, agent in enumerate(self.world.agents):
            is_on_hold = bool(
                self._joint_task_finished_agents[env_index, agent_index].item()
            )
            agent.is_rendering[env_index] = not is_on_hold
            if not is_on_hold:
                continue

            geom = agent.shape.get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(*agent.state.pos[env_index])
            xform.set_rotation(agent.state.rot[env_index])
            geom.set_color(*self.colors[agent_index], alpha=0.25)
            geoms.append(geom)

    def _get_static_lanelet_geoms(self):
        """Return cached static lanelet geometries, initializing on first render."""
        if self._static_lanelet_geoms is None:
            geoms = []
            self._render_lanelets(geoms)
            self._static_lanelet_geoms = geoms
        return self._static_lanelet_geoms

    def _render_lanelets(self, geoms):
        lanelets = self._get_rendered_lanelets()
        # Fill the drivable area first so no boundary line is covered
        for lanelet in lanelets.values():
            self._render_lanelet_surface(lanelet, geoms)
        if self.parameters.is_visu_driving_direction:
            self._render_driving_directions(lanelets, geoms)
        self._render_world_boundary(geoms)
        for lanelet in lanelets.values():
            for side in ("left_boundary", "right_boundary"):
                geom = rendering.PolyLine(
                    v=lanelet[side],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*Color.black100)
                geoms.append(geom)

    @staticmethod
    def _render_lanelet_surface(lanelet, geoms):
        """Fill a lanelet with one opaque quad per boundary segment.

        Quads keep each polygon convex (OpenGL's `GL_POLYGON` cannot fill
        concave ones), and an opaque color keeps overlapping lanelets uniform.
        """
        left = lanelet["left_boundary"].tolist()
        right = lanelet["right_boundary"].tolist()
        for i in range(len(left) - 1):
            geom = rendering.FilledPolygon(
                [left[i], left[i + 1], right[i + 1], right[i]], draw_border=False
            )
            geom.set_color(0.98, 0.98, 0.98)  # Lighter than `Color.black10`
            geoms.append(geom)

    def _render_driving_directions(self, lanelets, geoms):
        """Draw chevrons pointing in the driving direction along the lanelets.

        Size and spacing scale with the median lane width, so the chevrons fit
        maps of any scale.
        """
        widths = [
            (lanelet["left_boundary"] - lanelet["right_boundary"]).norm(dim=-1).median()
            for lanelet in lanelets.values()
        ]
        lane_width = torch.stack(widths).median().item()
        spacing = 6.0 * lane_width
        length = 0.4 * lane_width  # Along the driving direction
        half_span = 0.3 * lane_width
        thickness = 0.12 * lane_width

        placed = []
        for position, direction in self._driving_direction_anchors(lanelets, spacing):
            normal = torch.stack([-direction[1], direction[0]])
            # Overlapping lanelets (forks, merges, roundabouts) would stack
            # chevrons; keep one per lane and spacing
            if any(
                abs(torch.dot(position - p, d)) < spacing / 2
                and abs(torch.dot(position - p, n)) < lane_width / 2
                for p, d, n in placed
            ):
                continue
            placed.append((position, direction, normal))

            tip = position + direction * length / 2
            for side in (1, -1):
                tail = tip - direction * length + side * half_span * normal
                # One convex quad per arm (`GL_POLYGON` cannot fill concave shapes)
                arm = [
                    tail,
                    tip,
                    tip - direction * thickness,
                    tail - direction * thickness,
                ]
                geom = rendering.FilledPolygon(
                    [point.tolist() for point in arm], draw_border=False
                )
                geom.set_color(0.95, 0.95, 0.95)
                geoms.append(geom)

    @staticmethod
    def _driving_direction_anchors(lanelets, spacing):
        """Yield (position, unit direction) every `spacing` along the centerlines.

        Lanelets are often shorter than `spacing`, so the distance since the
        last anchor carries over to successors (breadth-first from lanelets
        without predecessors; closed loops are seeded in ID order).
        """
        start_distance = {}
        queue = deque()
        seeds = sorted(lanelets, key=lambda i: bool(lanelets[i]["predecessor"]))
        for seed in seeds:
            if seed in start_distance:
                continue
            start_distance[seed] = spacing / 2
            queue.append(seed)
            while queue:
                lanelet_id = queue.popleft()
                lanelet = lanelets[lanelet_id]
                center = (lanelet["left_boundary"] + lanelet["right_boundary"]) / 2
                segments = center.diff(dim=0)
                segment_lengths = segments.norm(dim=-1)
                arc = torch.cat(
                    [segment_lengths.new_zeros(1), segment_lengths.cumsum(0)]
                )

                s = spacing - start_distance[lanelet_id]
                while s < arc[-1]:
                    # Segment containing `s`; it has a positive length
                    i = int(torch.searchsorted(arc, s, right=True)) - 1
                    direction = segments[i] / segment_lengths[i]
                    yield center[i] + direction * (s - arc[i]), direction
                    s += spacing
                end_distance = (start_distance[lanelet_id] + arc[-1].item()) % spacing

                for successor in lanelet["successor"]:
                    if successor in lanelets and successor not in start_distance:
                        start_distance[successor] = end_distance
                        queue.append(successor)

    def _render_world_boundary(self, geoms):
        """Draw the world boundary as VMAS `Environment.plot_boundary` does."""
        x_semi = float(self.world.x_semidim)
        y_semi = float(self.world.y_semidim)
        corners = [
            (-x_semi, y_semi),
            (x_semi, y_semi),
            (x_semi, -y_semi),
            (-x_semi, -y_semi),
        ]
        for i, start in enumerate(corners):
            line = rendering.Line(start, corners[(i + 1) % len(corners)], width=0.7)
            line.set_color(0.25, 0.25, 0.25)  # VMAS `Color.GRAY`
            geoms.append(line)

    def _get_rendered_lanelets(self):
        """Return the lanelets drawn on the map, keyed by their IDs (1..N)."""
        lanelets = {}
        for i in range(len(self.map.parser.lanelets_all)):
            if "cpm" in self.parameters.scenario_type:
                if self.parameters.scenario_type == "cpm_mixed":
                    # Assume cpm_scenario_probabilities = [1.0, 0.5, 0.5]
                    if not any(
                        (i + 1) in path for path in self.map.parser.path_intersection
                    ):
                        continue
                # elif self.parameters.scenario_type == "cpm_intersection_extended":
                #     # TODO: visualize shared boundaries
                #     if not any(
                #         (i + 1) in path
                #         for path in self.map.parser.path_intersection_extended
                #     ):
                #         continue

            lanelets[i + 1] = self.map.parser.lanelets_all[i]
        return lanelets

    def _append_render_line(self, geoms, start, end, color, width):
        line = rendering.Line(
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            width=width,
        )
        xform = rendering.Transform()
        line.add_attr(xform)
        line.set_color(*color)
        geoms.append(line)

    def _append_render_dashed_line(
        self,
        geoms,
        start,
        end,
        color,
        width=3,
        dash_length=0.12 * LENGTH_SCALE,
        gap_length=0.08 * LENGTH_SCALE,
    ):
        start_x, start_y = float(start[0]), float(start[1])
        end_x, end_y = float(end[0]), float(end[1])
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return

        ux = dx / length
        uy = dy / length
        distance_along_line = 0.0
        while distance_along_line < length:
            segment_end = min(distance_along_line + dash_length, length)
            segment_start_point = (
                start_x + ux * distance_along_line,
                start_y + uy * distance_along_line,
            )
            segment_end_point = (
                start_x + ux * segment_end,
                start_y + uy * segment_end,
            )
            self._append_render_line(
                geoms,
                segment_start_point,
                segment_end_point,
                color,
                width,
            )
            distance_along_line += dash_length + gap_length

    def _render_observed_agents(self, env_index: int, geoms, ego_agent_index: int = 0):
        """Render the selected agent's effective observed-neighbor relationships."""
        if not self.parameters.is_visualize_observed_agents:
            return

        if not 0 <= ego_agent_index < self.n_agents:
            return

        diagnostics = self.observation_provider.get_observed_agent_render_diagnostics(
            ego_agent_index
        )
        if diagnostics is None:
            return

        neighbor_indices = diagnostics["neighbor_indices"]
        if env_index >= neighbor_indices.shape[0]:
            return

        ego_pos = self.world.agents[ego_agent_index].state.pos[env_index]
        n_slots = neighbor_indices.shape[1]
        on_hold_agents = None
        if self._is_joint_task_enabled():
            on_hold_agents = self._joint_task_finished_agents[env_index]
            if bool(on_hold_agents[ego_agent_index].item()):
                return

        is_prioritized_rendering = self.parameters.is_using_prioritized_marl
        observed_agent_line_width = 12
        priority_by_agent = None
        if is_prioritized_rendering:
            if not hasattr(self, "priority_rank"):
                return

            priority_rank = self.priority_rank
            if priority_rank.ndim == 1:
                priority_order = priority_rank
            elif priority_rank.ndim == 2 and env_index < priority_rank.shape[0]:
                priority_order = priority_rank[env_index]
            else:
                return

            priority_order = priority_order.detach().cpu().tolist()
            if sorted(priority_order) != list(range(self.n_agents)):
                return

            priority_by_agent = [0] * self.n_agents
            for rank, agent_index in enumerate(priority_order):
                priority_by_agent[agent_index] = rank

        for slot_idx in range(n_slots):
            is_padded = bool(diagnostics["padded_mask"][env_index, slot_idx].item())
            is_distance_masked = bool(
                diagnostics["distance_mask"][env_index, slot_idx].item()
            )
            is_velocity_masked = bool(
                diagnostics["velocity_mask"][env_index, slot_idx].item()
            )
            is_dropout_masked = bool(
                diagnostics["dropout_mask"][env_index, slot_idx].item()
            )
            is_final_masked = bool(
                diagnostics["final_mask"][env_index, slot_idx].item()
            )

            if is_padded or is_distance_masked or is_dropout_masked:
                continue

            neighbor_index = int(neighbor_indices[env_index, slot_idx].item())
            if not 0 <= neighbor_index < self.n_agents:
                continue
            if on_hold_agents is not None and bool(
                on_hold_agents[neighbor_index].item()
            ):
                continue

            neighbor_pos = self.world.agents[neighbor_index].state.pos[env_index]
            if is_prioritized_rendering:
                higher_priority_agent = (
                    ego_agent_index
                    if priority_by_agent[ego_agent_index]
                    < priority_by_agent[neighbor_index]
                    else neighbor_index
                )
                line_color = self.colors[higher_priority_agent]
            else:
                line_color = Color.black100

            if not is_final_masked:
                self._append_render_line(
                    geoms,
                    ego_pos,
                    neighbor_pos,
                    line_color,
                    width=observed_agent_line_width,
                )
            elif is_velocity_masked:
                self._append_render_line(
                    geoms,
                    ego_pos,
                    neighbor_pos,
                    line_color,
                    width=observed_agent_line_width,
                )

    def _render_entry_exit(self, env_index: int, geoms):
        rendered_entry_keys = set()
        rendered_exit_keys = set()

        for agent_i in range(self.n_agents):
            if self.world_state.ref_paths_agent_related.is_loop[env_index, agent_i]:
                continue

            entry_segment = self.world_state.ref_paths_agent_related.entry[
                env_index, agent_i
            ]
            if not torch.allclose(entry_segment[0], entry_segment[1]):
                entry_key = tuple(
                    round(float(value), 6) for value in entry_segment.flatten()
                )
                if entry_key not in rendered_entry_keys:
                    rendered_entry_keys.add(entry_key)

                    geom = rendering.PolyLine(
                        v=entry_segment,
                        close=False,
                    )
                    geom.set_linewidth(4)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*Color.black50)
                    geoms.append(geom)

                    for entry_point in entry_segment:
                        circle = rendering.make_circle(
                            radius=0.015 * LENGTH_SCALE, filled=True
                        )
                        xform = rendering.Transform()
                        circle.add_attr(xform)
                        xform.set_translation(entry_point[0], entry_point[1])
                        circle.set_color(*Color.black50)
                        geoms.append(circle)

            exit_segment = self.world_state.ref_paths_agent_related.exit[
                env_index, agent_i
            ]
            if torch.allclose(exit_segment[0], exit_segment[1]):
                continue

            exit_key = tuple(round(float(value), 6) for value in exit_segment.flatten())
            if exit_key in rendered_exit_keys:
                continue
            rendered_exit_keys.add(exit_key)

            geom = rendering.PolyLine(
                v=exit_segment,
                close=False,
            )
            geom.set_linewidth(4)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*Color.red100)
            geoms.append(geom)

            for exit_point in exit_segment:
                circle = rendering.make_circle(radius=0.015 * LENGTH_SCALE, filled=True)
                xform = rendering.Transform()
                circle.add_attr(xform)
                xform.set_translation(exit_point[0], exit_point[1])
                circle.set_color(*Color.red100)
                geoms.append(circle)

    def _render_extra_info(self, geoms, env_index: int = 0):
        if self.parameters.is_visualize_extra_info:
            font_size, _, line_height, margin = self._render_text_layout()
            x = margin
            y = self.viewer_size[1] - margin - line_height

            # Title
            if self.parameters.render_title:
                geom = rendering.TextLine(
                    text=self.parameters.render_title,
                    x=x,
                    y=y,
                    font_size=font_size,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geoms.append(geom)
                y -= line_height

            # Time and optional reset reason on the same line
            time_text = f"t={self.timer.step[env_index] * self.parameters.dt:.2f} s"
            reset_render_info = self._last_reset_render_info[env_index]
            reset_render_color = self._last_reset_render_color[env_index]
            reset_render_alpha = self._get_reset_render_info_alpha(env_index)

            has_active_reset = (
                reset_render_info is not None
                and len(reset_render_info) > 0
                and reset_render_alpha > 0.0
            )
            display_text = (
                f"{time_text} {reset_render_info}" if has_active_reset else time_text
            )

            geom = rendering.TextLine(
                text=display_text,
                x=x,
                y=y,
                font_size=font_size,
            )
            if has_active_reset and reset_render_color is not None:
                self._set_render_text_color(geom, reset_render_color)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

    def set_cbf_lambda_render_rows(self, env_index: int, rows):
        """Store the learned CBF lambdas of one env for the next rendered frame.

        Args:
            env_index: Environment index.
            rows: One list per ego agent of ``(agent_j, lambda)`` entries: the
                ego's own lambda (per-agent head) or the pair lambdas with its
                observed neighbors (pair head).
        """
        self._cbf_lambda_render_rows[env_index] = rows

    def _render_text_layout(self, font_size: float = 14):
        """Return the font size, char width, line height, and margin (px) of
        rendered text, enlarged by `render_text_scale`. Pyglet and the
        Matplotlib backend draw `TextLine` in a monospace font (advance 0.6 em)
        at 96 DPI."""
        font_size = font_size * self.render_text_scale
        char_width = 0.6 * font_size * 96 / 72
        line_height = 1.5 * font_size * 96 / 72
        margin = 12 * self.render_text_scale
        return font_size, char_width, line_height, margin

    def _world_to_render_px(self, x, y):
        """Return the pixel position (from the bottom left) of world point (x, y)."""
        map_width = self.viewer_size[0] - self.render_panel_width
        center_x, center_y = self._render_map_center
        return (
            map_width / 2 + (float(x) - center_x) * self.resolution_factor,
            self.viewer_size[1] / 2 + (float(y) - center_y) * self.resolution_factor,
        )

    def _cbf_lambda_text_columns(self):
        """Return (header, id width, token width in chars) of the lambda panel."""
        id_width = len(str(self.n_agents - 1))
        if self.parameters.cbf_lambda_head == "pair":
            # "j=.42 " per observed neighbor
            return "CBF λ_ij", id_width, id_width + 5
        # ".42" per agent
        return "CBF λ_i", id_width, 4

    def cbf_lambda_panel_width(self) -> int:
        """Return the panel width (px) needed to render learned CBF lambdas, or
        0 if no CBF-QP with learned lambda runs at test time."""
        if not (
            self.parameters.is_learn_cbf_lambda and self.parameters.is_using_cbf_testing
        ):
            return 0
        _, char_width, _, margin = self._render_text_layout()
        header, id_width, token_width = self._cbf_lambda_text_columns()
        n_tokens = (
            self.parameters.n_nearing_agents_observed
            if self.parameters.cbf_lambda_head == "pair"
            else 1
        )
        n_chars = max(len(header), id_width + 2 + n_tokens * token_width)
        return int(math.ceil((n_chars * char_width + 2 * margin) / 2) * 2)

    def set_render_resolution(self, vehicle_px: float = RENDER_VEHICLE_PX, **limits):
        """Size the viewer so that a vehicle is about `vehicle_px` pixels long.

        The scale is set per vehicle length, so lab-scale (1:18) and full-scale
        maps get similar on-screen vehicles. `compute_viewer_size` caps the
        frame (4K by default), pads thin maps, and rounds both sides to codec
        friendly multiples; `limits` overrides its keyword arguments. Call
        before the first render and before `add_render_side_panel`.
        """
        if self.render_panel_width > 0:
            raise RuntimeError("Set the render resolution before adding a panel.")
        bounds = self.map.parser.bounds
        self.render_origin = [self.world_x_dim / 2, self.world_y_dim / 2]
        x_half = max(
            abs(bounds["min_x"] - self.render_origin[0]),
            abs(bounds["max_x"] - self.render_origin[0]),
        )
        y_half = max(
            abs(bounds["min_y"] - self.render_origin[1]),
            abs(bounds["max_y"] - self.render_origin[1]),
        )
        margin = 1.1  # Visible area around the map
        width, height, px_per_unit = compute_viewer_size(
            2 * x_half * margin,
            2 * y_half * margin,
            vehicle_px / self.agent_length,
            **limits,
        )
        self.viewer_size = (width, height)
        self.resolution_factor = px_per_unit
        self.render_text_scale = compute_text_scale(width, height)
        self._render_map_center = list(self.render_origin)
        # VMAS shows a half-height of zoom**2 * max(1, 1/a) for aspect ratio a
        half_height = height / (2 * px_per_unit)
        self.viewer_zoom = (half_height * min(1.0, width / height)) ** 0.5
        print(
            f"[INFO] Render size: {width} x {height} px, "
            f"vehicle length {px_per_unit * self.agent_length:.0f} px, "
            f"text scale {self.render_text_scale:.2f}."
        )

    def add_render_side_panel(self, width: int):
        """Widen the viewer by a white panel of `width` px right of the map.

        Call before the first render. The width is rounded up to a multiple of
        `RENDER_SIZE_MULTIPLE`. The camera keeps the map at the same pixel scale
        and position, so pixel-placed labels stay aligned. VMAS sets the camera
        half-extent to ``zoom**2 * [max(a, 1), max(1, 1/a)]`` for viewer aspect
        ratio ``a``.
        """
        if width <= 0:
            return
        width = int(math.ceil(width / RENDER_SIZE_MULTIPLE) * RENDER_SIZE_MULTIPLE)
        map_w, map_h = self.viewer_size
        aspect_old = map_w / map_h
        aspect_new = (map_w + width) / map_h
        half_height = self.viewer_zoom**2 * max(1.0, 1.0 / aspect_old)
        self.viewer_zoom = (half_height * min(1.0, aspect_new)) ** 0.5
        self.render_origin = [
            self.render_origin[0] + half_height * (aspect_new - aspect_old),
            self.render_origin[1],
        ]
        self.viewer_size = (map_w + width, map_h)
        self.render_panel_width = width
        # World-frame box of the panel, used to blank map geometry under it
        panel_left = self.render_origin[0] + half_height * (
            aspect_new - 2 * width / map_h
        )
        self._render_panel_world_box = (
            panel_left,
            self.render_origin[0] + half_height * aspect_new,
            self.render_origin[1] - half_height,
            self.render_origin[1] + half_height,
        )

    def _render_side_panel_background(self, geoms):
        """Cover map geometry reaching into the side panel with white."""
        if self.render_panel_width <= 0:
            return
        left, right, bottom, top = self._render_panel_world_box
        panel = rendering.make_polygon(
            [(left, bottom), (right, bottom), (right, top), (left, top)],
            draw_border=False,
        )
        panel.set_color(1.0, 1.0, 1.0)
        geoms.append(panel)

    def _render_cbf_lambdas(self, env_index: int, geoms):
        """Render learned CBF lambdas as text in the side panel.

        One line per ego agent: per-agent head ``i: .42``; pair head
        ``i: j=.42 k=.37`` over the ego's observed neighbors (the pair's
        lambda used in the CBF-QP). Agent ids and values take the color of the
        agent they belong to.
        """
        rows = self._cbf_lambda_render_rows.get(env_index)
        if rows is None or self.render_panel_width <= 0:
            return
        font_size, char_width, line_height, margin = self._render_text_layout()
        header, id_width, token_width = self._cbf_lambda_text_columns()
        is_pair = self.parameters.cbf_lambda_head == "pair"
        x0 = self.viewer_size[0] - self.render_panel_width + margin
        y = self.viewer_size[1] - margin - line_height

        def add_text(text, column, color=None):
            geom = rendering.TextLine(
                text=text, x=x0 + column * char_width, y=y, font_size=font_size
            )
            if color is not None:
                self._set_render_text_color(geom, color)
            geom.add_attr(rendering.Transform())
            geoms.append(geom)

        add_text(header, 0)
        for agent_i, row in enumerate(rows):
            y -= line_height
            add_text(f"{agent_i:>{id_width}}:", 0, self._text_color(agent_i))
            for k, (agent_j, value) in enumerate(row):
                value_text = f"{value:.2f}".lstrip("0")
                add_text(
                    f"{agent_j}={value_text}" if is_pair else value_text,
                    id_width + 2 + k * token_width,
                    self._text_color(agent_j),
                )

    def _text_color(self, agent_index: int):
        """Agent color darkened as needed to stay legible on white."""
        color = self.colors[agent_index]
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        scale = min(1.0, 0.55 / max(luminance, 1e-6))
        return tuple(channel * scale for channel in color)

    @staticmethod
    def _set_render_text_color(geom, rgb_color):
        """Apply a normalized RGB color to a VMAS/Pyglet text geometry."""
        label = getattr(geom, "label", None)
        if label is None or not hasattr(label, "color"):
            return
        label_color = label.color
        label.color = (
            *(int(round(channel * 255)) for channel in rgb_color),
            label_color[3],
        )

    def _get_reset_render_info_alpha(self, env_index: int) -> float:
        """Return the current opacity of the selected environment's reset text."""
        elapsed = time.monotonic() - self._last_reset_render_time[env_index]
        if elapsed >= _RESET_RENDER_INFO_DURATION:
            return 0.0
        fade_start = _RESET_RENDER_INFO_DURATION - _RESET_RENDER_INFO_FADE_DURATION
        if elapsed <= fade_start:
            return 1.0
        return max(
            0.0,
            (_RESET_RENDER_INFO_DURATION - elapsed) / _RESET_RENDER_INFO_FADE_DURATION,
        )

    @staticmethod
    def _set_render_text_alpha(geom, alpha: float):
        """Apply opacity to a VMAS/Pyglet text geometry when supported."""
        label = getattr(geom, "label", None)
        if label is None or not hasattr(label, "color"):
            return
        color = label.color
        label.color = (*color[:3], int(round(color[3] * alpha)))

    def _get_position_speed_target(self, env_index: int, agent_i: int):
        if (
            self.parameters.is_using_cbf_testing
            or self.parameters.is_using_cbf_training
        ):
            target_speed = self.world_state.applied_action_vel[env_index, agent_i]
            target_steer = self.world_state.applied_action_steer[env_index, agent_i]
        else:
            target_speed = self.world.agents[agent_i].action.u[env_index, 0]
            target_steer = self.world.agents[agent_i].action.u[env_index, 1]

        ld = target_speed.new_tensor(float(self.parameters.lookahead_dist))
        y_max = target_speed.new_tensor(float(POSITION_SPEED_LATERAL_OFFSET_BOUND))
        l_wb = target_speed.new_tensor(float(self.world.agents[agent_i].dynamics.l_wb))

        # Recover the lateral offset that corresponds to the rendered speed/steer pair.
        kappa = torch.tan(target_steer) / l_wb
        max_kappa = 2.0 * y_max / (ld**2 + y_max**2) if y_max.item() > 0 else 0.0
        kappa = torch.clamp(kappa, min=-max_kappa, max=max_kappa)

        discr = torch.clamp(1.0 - (kappa * ld) ** 2, min=0.0)
        lateral_offset = torch.where(
            kappa.abs() < 1e-8,
            torch.zeros_like(kappa),
            (kappa * (ld**2)) / (1.0 + torch.sqrt(discr)),
        )
        lateral_offset = torch.clamp(lateral_offset, min=-y_max, max=y_max)

        pos = self.world.agents[agent_i].state.pos[env_index]
        heading = self.world.agents[agent_i].state.rot[env_index]
        target_pos = torch.stack(
            [
                pos[0] + torch.cos(heading) * ld - torch.sin(heading) * lateral_offset,
                pos[1] + torch.sin(heading) * ld + torch.cos(heading) * lateral_offset,
            ]
        )
        return target_pos, target_speed

    def _render_position_speed_targets(self, env_index: int, geoms):
        if self.parameters.rl_action_type != "position_speed":
            return

        for agent_i in range(self.n_agents):
            if self.world.agents[agent_i].action.u is None:
                continue

            start = (
                self.world.agents[agent_i].state.pos[env_index, 0],
                self.world.agents[agent_i].state.pos[env_index, 1],
            )
            target_pos, target_speed = self._get_position_speed_target(
                env_index, agent_i
            )
            end = (target_pos[0], target_pos[1])

            line = rendering.Line(start, end, width=3)
            xform = rendering.Transform()
            line.add_attr(xform)
            line.set_color(*self.colors[agent_i])
            geoms.append(line)

            circle = rendering.make_circle(radius=0.02 * LENGTH_SCALE, filled=True)
            xform = rendering.Transform()
            circle.add_attr(xform)
            xform.set_translation(target_pos[0], target_pos[1])
            circle.set_color(*self.colors[agent_i])
            geoms.append(circle)

            label_x, label_y = self._world_to_render_px(target_pos[0], target_pos[1])
            font_size, _, _, _ = self._render_text_layout(font_size=12)
            geom = rendering.TextLine(
                text=f"{float(target_speed.item()):.2f} m/s",
                x=label_x,
                y=label_y + 12 * self.render_text_scale,
                font_size=font_size,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

    def set_shield_rollout_render(self, env_index: int, rollouts: dict):
        """Store the backup-shield rollouts of one env for the next rendered frame.

        Args:
            env_index: Environment index.
            rollouts: Center positions [T, 2] of each checked agent's accepted
                rollout, keyed by agent; index 0 is the predicted next state.
        """
        self._shield_rollout_render[env_index] = rollouts

    def _render_shield_rollout(self, env_index: int, geoms):
        """Draw the latest backup-shield rollout of each vehicle."""
        if not self.parameters.is_visu_shield_rollout:
            return
        for agent_i, points in self._shield_rollout_render.get(env_index, {}).items():
            if points.shape[0] < 2:
                continue

            geom = rendering.PolyLine(v=points, close=False)
            geom.set_linewidth(3)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*self.colors[agent_i])
            geoms.append(geom)

            endpoint = points[-1]
            circle = rendering.make_circle(radius=0.015 * LENGTH_SCALE, filled=True)
            xform = rendering.Transform()
            circle.add_attr(xform)
            xform.set_translation(endpoint[0], endpoint[1])
            circle.set_color(*self.colors[agent_i])
            geoms.append(circle)

    def _render_human_overlay(self, env_index: int, geoms):
        """Outline the recorded humans of the controlled vehicles (`LogReplay.is_overlay_humans`) in the
        color of their vehicle, linked to it by a line while it is on the map; gray while it is not active.
        With `is_visu_veh_direction`, a chevron in the outline color shows each human's heading."""
        if self.log_replay is None or not self.log_replay.is_overlay_humans:
            return
        for corners, slot in self.log_replay.human_overlay():
            color = Color.black50 if slot is None else self.colors[slot]
            geom = rendering.PolyLine(v=corners.tolist(), close=True)
            geom.set_linewidth(2)
            geom.add_attr(rendering.Transform())
            geom.set_color(*color)
            geoms.append(geom)
            if self.parameters.is_visu_veh_direction:
                # Corners: front left, rear left, rear right, front right
                heading = corners[0] - corners[1]
                geoms += make_heading_chevron(
                    math.hypot(*heading),
                    math.hypot(*(corners[0] - corners[3])),
                    corners.mean(axis=0).tolist(),
                    math.atan2(heading[1], heading[0]),
                    color,
                )
            if slot is not None and not self.log_replay.parked[env_index, slot]:
                self._append_render_line(
                    geoms,
                    corners.mean(axis=0),
                    self.world.agents[slot].state.pos[env_index].tolist(),
                    color,
                    width=1,
                )

    def _render_reference_paths_and_boundaries(self, env_index, geoms):
        for agent_i in range(self.n_agents):
            if (
                self.log_replay is not None
                and self.log_replay.parked[env_index, agent_i]
            ):
                continue  # Parked agents are off the map
            # Visualize short-term reference paths of agents
            if self.parameters.is_visualize_short_term_path:
                geom = rendering.PolyLine(
                    v=self.world_state.ref_paths_agent_related.short_term[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.colors[agent_i])
                geoms.append(geom)

                for i_p in self.world_state.ref_paths_agent_related.short_term[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(
                        radius=0.01 * LENGTH_SCALE, filled=True
                    )
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.colors[agent_i])
                    geoms.append(circle)

            # Visualize nearing points on boundaries
            if self.parameters.is_visu_sampled_boundary_points:
                geom = rendering.PolyLine(
                    v=self.world_state.ref_paths_agent_related.nearing_points_left_boundary[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.colors[agent_i])
                geoms.append(geom)

                for (
                    i_p
                ) in self.world_state.ref_paths_agent_related.nearing_points_left_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(
                        radius=0.01 * LENGTH_SCALE, filled=True
                    )
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.colors[agent_i])
                    geoms.append(circle)

                geom = rendering.PolyLine(
                    v=self.world_state.ref_paths_agent_related.nearing_points_right_boundary[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.colors[agent_i])
                geoms.append(geom)

                for (
                    i_p
                ) in self.world_state.ref_paths_agent_related.nearing_points_right_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(
                        radius=0.01 * LENGTH_SCALE, filled=True
                    )
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.colors[agent_i])
                    geoms.append(circle)

            # Agent IDs (if there are more than one agent)
            if self.parameters.n_agents > 1:
                label_x, label_y = self._world_to_render_px(
                    *self.world.agents[agent_i].state.pos[env_index]
                )
                font_size, char_width, _, _ = self._render_text_layout()
                text = f"{agent_i}"
                # Center on the vehicle (`TextLine` anchors its bottom left),
                # clear of the heading chevron at the front
                geom = rendering.TextLine(
                    text=text,
                    x=label_x - len(text) * char_width / 2,
                    y=label_y - 0.6 * font_size * 96 / 72,
                    font_size=font_size,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geoms.append(geom)

            # Lanelet boundaries of agents' reference path
            if self.parameters.is_visualize_lane_boundary:
                if agent_i == 0:
                    is_visualize_lane_boundary_points = True

                    # Left boundary
                    left_boundary = (
                        self.world_state.ref_paths_agent_related.left_boundary[
                            env_index, agent_i
                        ]
                    )
                    geom = rendering.PolyLine(
                        v=left_boundary,
                        close=False,
                    )
                    geom.set_linewidth(4)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.colors[agent_i])
                    geoms.append(geom)
                    if is_visualize_lane_boundary_points:
                        for i_p in left_boundary:
                            circle = rendering.make_circle(
                                radius=0.01 * LENGTH_SCALE, filled=True
                            )
                            xform = rendering.Transform()
                            circle.add_attr(xform)
                            xform.set_translation(i_p[0], i_p[1])
                            circle.set_color(*self.colors[agent_i])
                            geoms.append(circle)

                    # Right boundary
                    right_boundary = (
                        self.world_state.ref_paths_agent_related.right_boundary[
                            env_index, agent_i
                        ]
                    )
                    geom = rendering.PolyLine(
                        v=right_boundary,
                        close=False,
                    )
                    geom.set_linewidth(4)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.colors[agent_i])
                    geoms.append(geom)
                    if is_visualize_lane_boundary_points:
                        for i_p in right_boundary:
                            circle = rendering.make_circle(
                                radius=0.01 * LENGTH_SCALE, filled=True
                            )
                            xform = rendering.Transform()
                            circle.add_attr(xform)
                            xform.set_translation(i_p[0], i_p[1])
                            circle.set_color(*self.colors[agent_i])
                            geoms.append(circle)

    def _render_groups(self, geoms):
        """
        Render group relationships:
        - Solid lines: intra-group (inter_groups)
        - Dashed lines: cross-group (cross_groups)
        """
        # ---------- Intra-group (solid lines) ----------
        if hasattr(self, "inter_groups"):
            for group in self.inter_groups:
                if len(group) < 2:
                    continue
                # Fully connect agents inside the same group
                for idx_i in range(len(group)):
                    for idx_j in range(idx_i + 1, len(group)):
                        i = group[idx_i]
                        j = group[idx_j]

                        line = rendering.Line(
                            (
                                self.world.agents[i].state.pos[0, 0],
                                self.world.agents[i].state.pos[0, 1],
                            ),
                            (
                                self.world.agents[j].state.pos[0, 0],
                                self.world.agents[j].state.pos[0, 1],
                            ),
                            width=6,
                        )

                        xform = rendering.Transform()
                        line.add_attr(xform)

                        # Use a neutral color or one agent’s color
                        line.set_color(*Color.blue100)
                        geoms.append(line)

        # ---------- Cross-group (dashed lines) ----------
        if hasattr(self, "cross_groups"):
            for i, nbrs in self.cross_groups.items():
                for j in nbrs:
                    line = rendering.Line(
                        (
                            self.world.agents[i].state.pos[0, 0],
                            self.world.agents[i].state.pos[0, 1],
                        ),
                        (
                            self.world.agents[j].state.pos[0, 0],
                            self.world.agents[j].state.pos[0, 1],
                        ),
                        width=4,
                    )

                    xform = rendering.Transform()
                    line.add_attr(xform)

                    line.set_color(*Color.black50)
                    geoms.append(line)

    # def _render_action_propagation_direction(self, env_index, geoms):
    #     """
    #     Render the direction of action propagation between agents in a prioritized MARL setting.

    #     This function visualizes the action propagation directions, which are from higher-priority to lower-priority agents,
    #     by drawing lines between them. The lines are colored based on the higher-priority agent's color.

    #     Args:
    #         env_index (int): The index of the current environment.
    #         geoms (list): A list of geometric objects to which the new visualizations will be added.
    #     """
    #     # Check if prioritized MARL is being used and if observable higher priority agents exist
    #     if self.parameters.is_using_prioritized_marl and hasattr(
    #         self, "observable_higher_priority_agents"
    #     ):
    #         # self._recolor_agents() # Uncomment if agent recoloring is needed

    #         # Iterate through all agents to draw action propagation lines
    #         for i in range(self.n_agents):
    #             # Get the list of observable higher priority agents for the current agent
    #             observable_higher_priority_agents = (
    #                 self.observable_higher_priority_agents[env_index][i]
    #             )

    #             # Draw lines from each higher priority agent to the current agent
    #             for j in observable_higher_priority_agents:
    #                 # Create a line object representing the action propagation
    #                 line = rendering.Line(
    #                     # Starting point: position of the higher priority agent (j)
    #                     (
    #                         self.world.agents[j].state.pos[env_index, 0],
    #                         self.world.agents[j].state.pos[env_index, 1],
    #                     ),
    #                     # Ending point: position of the current agent (i)
    #                     (
    #                         self.world.agents[i].state.pos[env_index, 0],
    #                         self.world.agents[i].state.pos[env_index, 1],
    #                     ),
    #                     width=10,  # Set line width
    #                 )

    #                 # Add transform attribute to the line
    #                 xform = rendering.Transform()
    #                 line.add_attr(xform)

    #                 # Set the line color to match the higher priority agent's color
    #                 line.set_color(*self.colors[j])

    #                 # Add the line to the list of geometric objects to be rendered
    #                 geoms.append(line)

    def _render_cbf_action(self, env_index, geoms):
        """
        Render applied and nominal actions as arrows for each agent.

        Args:
            env_index (int): The index of the current environment.
            geoms (list): A list of geometric objects to which the new visualizations will be added.

        Description:
            For each agent, this method can visualize:
                - The applied action (shown in the agent's color)
                - The nominal action (shown in semi-transparent black when CBF is used)
            Each action is rendered as a directional arrow, where the direction reflects the steering angle,
            and the length indicates the magnitude of the velocity.
        """

        if not (
            self.parameters.is_visu_applied_action or self.parameters.is_visu_nom_action
        ):
            return

        is_using_cbf = (
            self.parameters.is_using_cbf_testing
            or self.parameters.is_using_cbf_training
        )

        def _append_action_arrow(
            agent_idx, heading, action_vel, action_steer, color, width
        ):
            start = (
                self.world.agents[agent_idx].state.pos[env_index, 0],
                self.world.agents[agent_idx].state.pos[env_index, 1],
            )
            end = (
                start[0] + torch.cos(heading + action_steer) * action_vel * 0.5,
                start[1] + torch.sin(heading + action_steer) * action_vel * 0.5,
            )
            arrow_end_1 = (
                end[0]
                + torch.cos(heading + action_steer + torch.pi * 8 / 9)
                * torch.sign(action_vel)
                * 0.07
                * LENGTH_SCALE,
                end[1]
                + torch.sin(heading + action_steer + torch.pi * 8 / 9)
                * torch.sign(action_vel)
                * 0.07
                * LENGTH_SCALE,
            )
            arrow_end_2 = (
                end[0]
                + torch.cos(heading + action_steer - torch.pi * 8 / 9)
                * torch.sign(action_vel)
                * 0.07
                * LENGTH_SCALE,
                end[1]
                + torch.sin(heading + action_steer - torch.pi * 8 / 9)
                * torch.sign(action_vel)
                * 0.07
                * LENGTH_SCALE,
            )

            line = rendering.Line(start, end, width=width)
            arrow_line_1 = rendering.Line(end, arrow_end_1, width=width)
            arrow_line_2 = rendering.Line(end, arrow_end_2, width=width)

            xform = rendering.Transform()
            for geom in (line, arrow_line_1, arrow_line_2):
                geom.add_attr(xform)
                geom.set_color(*color)
                geoms.append(geom)

        for i in range(self.n_agents):
            heading = self.world.agents[i].state.rot[env_index]

            if self.parameters.is_visu_applied_action:
                applied_action_vel = self.world_state.applied_action_vel[env_index, i]
                applied_action_steer = self.world_state.applied_action_steer[
                    env_index, i
                ]
                _append_action_arrow(
                    i,
                    heading,
                    applied_action_vel,
                    applied_action_steer,
                    self.colors[i],
                    5,
                )

            if is_using_cbf and self.parameters.is_visu_nom_action:
                nominal_action_vel = self.world_state.nominal_action_vel[env_index, i]
                nominal_action_steer = self.world_state.nominal_action_steer[
                    env_index, i
                ]
                _append_action_arrow(
                    i,
                    heading,
                    nominal_action_vel,
                    nominal_action_steer,
                    Color.black75,
                    3,
                )


if __name__ == "__main__":
    scenario = ScenarioRoadTraffic()
    render_interactively(
        scenario=scenario,
        control_two_agents=False,
        shared_reward=False,
    )
