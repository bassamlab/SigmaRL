# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
from termcolor import colored, cprint

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict

from sigmarl.scenarios.observations.observation_provider_rt import RoadTrafficObservationProviderSimulation, \
    ObservationProviderParametersRT

from sigmarl.scenarios.world_state.world_state_rt.world_state_rt import WorldStateRTParameters
from sigmarl.scenarios.world_state.world_state_rt.world_state_rt_sim import WorldStateRTSimulation, \
    WorldStateRTSimParameters

cicd_testing = os.getenv("CICD_TESTING", "false").lower() == "true"
if not cicd_testing:
    # During CI/CD testing, the OpenGL library is missing in the Docker container, so we need to import the rendering module conditionally
    from vmas.simulator import rendering

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, World
from vmas.simulator.scenario import BaseScenario

from sigmarl.dynamics import KinematicBicycleModel

from sigmarl.colors import Color, colors

from sigmarl.helper_training import Parameters, WorldCustom, Vehicle

from sigmarl.helper_scenario import (
    Normalizers,
    Penalties,
    ReferencePathsMapRelated,
    Rewards,
    Thresholds,
    Timer,
    Constants,
    StateBuffer,
    InitialStateBuffer,
    exponential_decreasing_fcn,
    angle_eliminate_two_pi,
)

from sigmarl.map_manager import MapManager

from sigmarl.constants import SCENARIOS, AGENTS


class ScenarioRoadTraffic(BaseScenario):
    """
    This scenario aims to design an MARL framework with information-dense observation design to enable fast training and to empower agents the ability to generalize to unseen scenarios.

    We propose five observation-design strategies. They correspond to five parameters in this file, and their default
    values are True.
        - is_ego_view: Whether to use ego view (otherwise bird view)
        - is_observe_distance_to_agents: Whether to observe the distance to other agents
        - is_observe_distance_to_boundaries: Whether to observe the distance to labelet boundaries (otherwise the points on lanelet boundaries)
        - is_observe_distance_to_center_line: Whether to observe the distance to reference path (otherwise None)
        - is_observe_vertices: Whether to observe the vertices of other agents (otherwise center points)

    In addition, there are some commonly used parameters you may want to adjust to suit your case:
        - n_agents: Number of agents
        - dt: Sample time in seconds
        - scenario_type: One of {"CPM_entire", "CPM_mixed", "intersection_1", ...}. See SCENARIOS in utilities/constants.py for more scenarios.
                         "CPM_entire": the entire CPM map will be used
                         "CPM_mixed": a specific part of the CPM map (intersection, merge-in, or merge-out) will be used for each env when making or resetting it. You can control the probability of using each of them by the parameter `scenario_probabilities`. It is an array with three values. The first value corresponds to the probability of using intersection. The second and the third values correspond to merge-in and merge-out, respectively. If you only want to use one specific part of the map for all parallel envs, you can set the other two values to zero. For example, if you want to train a RL policy only for intersection, they can set `scenario_probabilities` to [1.0, 0.0, 0.0].
                         "intersection_1": the intersection scenario with ID 1
        - is_partial_observation: Whether to enable partial observation (to model partially observable MDP)
        - n_nearing_agents_observed: Number of nearing agents to be observed (consider limited sensor range)

        is_testing_mode: Testing mode is designed to test the learned policy.
                         In non-testing mode, once a collision occurs, all agents will be reset with random initial states.
                         To ensure these initial states are feasible, the initial positions are conservatively large (1.2*diagonalLengthOfAgent).
                         This ensures agents are initially safe and avoids putting agents in an immediate dangerous situation at the beginning of a new scenario.
                         During testing, only colliding agents will be reset, without changing the states of other agents, who are possibly interacting with other agents.
                         This may allow for more effective testing.

    For other parameters, see the class Parameter defined in this file.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self._init_params(batch_dim, device, **kwargs)
        world = self._init_world(batch_dim, device)
        world.parameters = self.parameters
        self._init_agents(world)
        self.world_state.init_world(world)
        return world

    def _init_params(self, batch_dim, device, **kwargs):
        """
        Initialize parameters.
        """
        scenario_type = kwargs.pop(
            "scenario_type", "CPM_entire"
        )  # Scenario type. See all scenario types in SCENARIOS in utilities/constants
        self.agent_width = AGENTS["width"]  # The width of the agent in [m]
        self.agent_length = AGENTS["length"]  # The length of the agent in [m]
        lane_width = SCENARIOS[scenario_type][
            "lane_width"
        ]  # The (rough) width of each lane in [m]

        # Reward
        r_p_normalizer = (
            100  # This parameter normalizes rewards and penalties to [-1, 1].
        )
        # This is useful for RL algorithms with an actor-critic architecture where the critic's
        # output is limited to [-1, 1] (e.g., due to tanh activation function).

        reward_progress = (
                kwargs.pop("reward_progress", 10) / r_p_normalizer
        )  # Reward for moving along reference paths
        reward_vel = (
                kwargs.pop("reward_vel", 5) / r_p_normalizer
        )  # Reward for moving in high velocities.
        reward_reach_goal = (
                kwargs.pop("reward_reach_goal", 0) / r_p_normalizer
        )  # Goal-reaching reward

        # Penalty
        penalty_deviate_from_ref_path = kwargs.pop(
            "penalty_deviate_from_ref_path", -2 / r_p_normalizer
        )  # Penalty for deviating from reference paths
        penalty_near_boundary = kwargs.pop(
            "penalty_near_boundary", -20 / r_p_normalizer
        )  # Penalty for being too close to lanelet boundaries
        penalty_near_other_agents = kwargs.pop(
            "penalty_near_other_agents", -20 / r_p_normalizer
        )  # Penalty for being too close to other agents
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
            "threshold_deviate_from_ref_path", (lane_width - self.agent_width) / 2
        )  # Use for penalizing of deviating from reference path

        threshold_reach_goal = kwargs.pop(
            "threshold_reach_goal", self.agent_width / 2
        )  # Threshold less than which agents are considered at their goal positions

        threshold_change_steering = kwargs.pop(
            "threshold_change_steering", 10
        )  # Threshold above which agents will be penalized for changing steering too quick [degree]

        threshold_near_boundary_high = kwargs.pop(
            "threshold_near_boundary_high", (lane_width - self.agent_width) / 2 * 0.9
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to lanelet boundaries
        threshold_near_boundary_low = kwargs.pop(
            "threshold_near_boundary_low", 0
        )  # Threshold above which agents will be penalized for being too close to lanelet boundaries

        threshold_near_other_agents_c2c_high = kwargs.pop(
            "threshold_near_other_agents_c2c_high", self.agent_length + self.agent_width
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to other agents (for center-to-center distance)
        threshold_near_other_agents_c2c_low = kwargs.pop(
            "threshold_near_other_agents_c2c_low",
            (self.agent_length + self.agent_width) / 2,
        )  # Threshold above which agents will be penalized (for center-to-center distance,
        # If a c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another penalty)

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

        # Visualization
        self.resolution_factor = kwargs.pop("resolution_factor", 200)  # Default 200

        # Reference path
        sample_interval_ref_path = kwargs.pop(
            "sample_interval_ref_path", 2
        )  # Integer, sample interval from the long-term reference path for the short-term reference paths
        max_ref_path_points = kwargs.pop(
            "max_ref_path_points", 200
        )  # The estimated maximum points on the reference path

        noise_level = kwargs.pop(
            "noise_level", 0.2 * self.agent_width
        )  # Noise will be generated by the standary normal distribution. This parameter controls the noise level

        n_stored_steps = kwargs.pop(
            "n_stored_steps",
            5,  # The number of steps to store (include the current step). At least one
        )
        n_observed_steps = kwargs.pop(
            "n_observed_steps", 1
        )  # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`

        # World dimensions
        self.world_x_dim = SCENARIOS[scenario_type]["world_x_dim"]
        self.world_y_dim = SCENARIOS[scenario_type]["world_y_dim"]

        self.render_origin = [
            self.world_x_dim / 2,
            self.world_y_dim / 2,
        ]

        self.viewer_size = (
            int(self.world_x_dim * self.resolution_factor),
            int(self.world_y_dim * self.resolution_factor),
        )
        self.viewer_zoom = SCENARIOS[scenario_type]["viewer_zoom"]

        self.max_steering = kwargs.pop(
            "max_steering",
            torch.tensor(AGENTS["max_steering"], device=device, dtype=torch.float32),
        )  # Maximum allowed steering angle in degree
        self.max_speed = kwargs.pop(
            "max_speed", AGENTS["max_speed"]
        )  # Maximum allowed speed in [m/s]

        n_points_nearing_boundary = kwargs.pop(
            "n_points_nearing_boundary", 5
        )  # The number of points on nearing boundaries to be observed

        probability_record = kwargs.pop("probability_record", 1.0)
        probability_use_recording = kwargs.pop("probability_use_recording", 0.2)
        buffer_size = kwargs.pop("buffer_size", 100)

        if not hasattr(self, "parameters"):
            self.parameters = Parameters(
                n_agents=kwargs.pop("n_agents", SCENARIOS[scenario_type]["n_agents"]),
                scenario_type=scenario_type,
                is_partial_observation=kwargs.pop("is_partial_observation", True),
                is_testing_mode=kwargs.pop("is_testing_mode", False),
                is_visualize_short_term_path=kwargs.pop(
                    "is_visualize_short_term_path", True
                ),
                n_nearing_agents_observed=kwargs.pop("n_nearing_agents_observed", 2),
                is_real_time_rendering=kwargs.pop("is_real_time_rendering", False),
                n_steps_stored=kwargs.pop("n_steps_stored", 10),
                n_points_short_term=kwargs.pop("n_points_short_term", 3),
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
                is_apply_mask=kwargs.pop("is_apply_mask", False),
                is_challenging_initial_state_buffer=kwargs.pop(
                    "is_challenging_initial_state_buffer", True
                ),
                cpm_scenario_probabilities=kwargs.pop(
                    "cpm_scenario_probabilities", [1.0, 0.0, 0.0]
                ),  # Probabilities of training agents in intersection, merge-in, and merge-out scenario
                is_add_noise=kwargs.pop("is_add_noise", True),
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
                is_using_cbf=kwargs.pop("is_using_cbf", False),
                experiment_type=kwargs.pop("experiment_type", "simulation"),
                is_obs_steering=kwargs.pop("is_obs_steering", False),
            )

        self.n_agents = self.parameters.n_agents
        self.colors = colors

        # Logs
        if self.parameters.is_testing_mode:
            print(colored(f"[INFO] Testing mode", "red"))
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
            else:
                raise ValueError(
                    f"The given prioritization method is not supported. Obtained: {self.parameters.prioritization_method}. Expected: 'marl' or 'random'."
                )

        self.parameters.n_nearing_agents_observed = min(
            self.parameters.n_nearing_agents_observed, self.parameters.n_agents - 1
        )
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
        self.map = MapManager(
            scenario_type=self.parameters.scenario_type,
            device=device,
        )

        cprint(f"[INFO] Map of {self.parameters.scenario_type} parsed.", "blue")
        # Determine the maximum number of points on the reference path
        if "CPM_mixed" in self.parameters.scenario_type:
            # Mixed scenarios including intersection, merge in, and merge out
            max_ref_path_points = (
                    max(
                        [
                            ref_p["center_line"].shape[0]
                            for ref_p in self.map.parser.reference_paths_intersection
                                         + self.map.parser.reference_paths_merge_in
                                         + self.map.parser.reference_paths_merge_out
                        ]
                    )
                    + self.parameters.n_points_short_term * sample_interval_ref_path
                    + 2
            )  # Append a smaller buffer
        else:
            # Single scenario
            max_ref_path_points = (
                    max(
                        [
                            ref_p["center_line"].shape[0]
                            for ref_p in self.map.parser.reference_paths
                        ]
                    )
                    + self.parameters.n_points_short_term * sample_interval_ref_path
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
            collide_with_boundaries=torch.tensor(
                penalty_collide_with_boundaries, device=device, dtype=torch.float32
            ),
            change_steering=torch.tensor(
                penalty_change_steering, device=device, dtype=torch.float32
            ),
            time=torch.tensor(penalty_time, device=device, dtype=torch.float32),
        )

        self.normalizers = Normalizers(
            pos=torch.tensor(
                [self.agent_length * 10, self.agent_length * 10],
                device=device,
                dtype=torch.float32,
            ),
            pos_world=torch.tensor(
                [self.world_x_dim, self.world_y_dim], device=device, dtype=torch.float32
            ),
            v=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            steering=self.max_steering,
            distance_lanelet=torch.tensor(
                lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_ref=torch.tensor(
                lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_agent=torch.tensor(
                self.agent_length * 10, device=device, dtype=torch.float32
            ),
        )

        # Distances to boundaries and reference path, and also the closest point on the reference paths of agents
        if self.parameters.is_use_mtv_distance:
            self.distance_type = "mtv"  # One of {"c2c", "mtv"}
        else:
            self.distance_type = "c2c"  # One of {"c2c", "mtv"}
        # print(colored("[INFO] Distance type: ", "black"), colored(distance_type, "blue"))

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
            distance_mask_agents=self.agent_length * 5,
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
                AGENTS["length"] ** 2 + self.agent_width ** 2,
                device=device,
                dtype=torch.float32,
            ).sqrt()
                                     * 1.2,
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

        # Store the computed observations of each agent for reuse in `info()`
        # to generate `base_obs` and `prio_obs`, which are used for prioritized MARL.
        self.stored_observations = [None] * self.n_agents

        self.world_state = WorldStateRTSimulation(WorldStateRTSimParameters(
            batch_dim=batch_dim,
            n_agents=self.n_agents,
            device=device,
            distance_type=self.distance_type,
            max_ref_path_points=max_ref_path_points,
            n_points_nearing_boundary=n_points_nearing_boundary,
            n_points_short_term=self.parameters.n_points_short_term,
            sample_interval_ref_path=sample_interval_ref_path,
            observe_distance_to_boundaries=self.parameters.is_observe_distance_to_boundaries,
            scenario_type=self.parameters.scenario_type,
            cpm_scenario_probabilities=self.parameters.cpm_scenario_probabilities,
            reset_agent_min_distance=self.constants.reset_agent_min_distance),
            self.map
        )

        self.observation_provider = RoadTrafficObservationProviderSimulation(
            ObservationProviderParametersRT(
                batch_dim=batch_dim,
                n_agents=self.n_agents,
                device=device,
                n_stored_steps=n_stored_steps,
                n_observed_steps=n_observed_steps,
                n_points_nearing_boundary=n_points_nearing_boundary,
                n_nearing_agents_observed=self.parameters.n_nearing_agents_observed,
                n_points_short_term=self.parameters.n_points_short_term,
                is_observe_distance_to_boundaries=self.parameters.is_observe_distance_to_boundaries,
                is_ego_view=self.parameters.is_ego_view,
                is_using_opponent_modeling=self.parameters.is_using_opponent_modeling,
                is_apply_mask=self.parameters.is_apply_mask,
                is_add_noise=self.parameters.is_add_noise,
                noise_level=noise_level,
                is_partial_observation=self.parameters.is_partial_observation,
                is_obs_steering=self.parameters.is_obs_steering,
                is_observe_vertices=self.parameters.is_observe_vertices,
                is_observe_distance_to_agents=self.parameters.is_observe_distance_to_agents,
                is_observe_ref_path_other_agents=self.parameters.is_observe_ref_path_other_agents,
                is_observe_distance_to_center_line=self.parameters.is_observe_distance_to_center_line
            ),
            self.constants,
            self.normalizers,
            self.thresholds,
            self.map,
            self.world_state
        )

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
        # Create agents
        for i in range(self.n_agents):
            agent = Vehicle(
                name=f"agent_{i}",
                shape=Box(length=AGENTS["length"], width=AGENTS["width"]),
                color=tuple(self.colors[i]),
                collide=False,
                render_action=False,
                u_range=[
                    self.max_speed,
                    self.max_steering,
                ],  # Control command serves as velocity command
                u_multiplier=[1, 1],
                max_speed=self.max_speed,
                dynamics=KinematicBicycleModel(  # Use the kinematic bicycle model for each agent
                    l_f=AGENTS["l_f"],
                    l_r=AGENTS["l_r"],
                    max_speed=AGENTS["max_speed"],
                    max_steering=AGENTS["max_steering"],
                    max_acc=AGENTS["max_acc"],
                    min_acc=AGENTS["min_acc"],
                    max_steering_rate=AGENTS["max_steering_rate"],
                    min_steering_rate=AGENTS["min_steering_rate"],
                    device=world.device,
                ),
            )
            world.add_agent(agent)

    def reset_world_at(self, env_index: int = None, agent_index: int = None):
        """
        This function resets the world at the specified env_index and the specified agent_index.
        If env_index is given as None, the majority part of computation will be done in a vectorized manner.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed
        :param agent_index: index of the agent to reset. If None all agents in the specified environment will be reset.
        """
        agents = self.world.agents

        is_reset_single_agent = agent_index is not None

        if is_reset_single_agent:
            assert env_index is not None

        for env_i in ([env_index] if env_index is not None else range(self.world.batch_dim)):
            # Beginning of a new simulation (only record for the first env)
            if env_i == 0:
                self.timer.step_duration[:] = 0
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0

            if (
                self.parameters.is_challenging_initial_state_buffer and
                (torch.rand(1) < self.initial_state_buffer.probability_use_recording) and
                (self.initial_state_buffer.valid_size >= 1)
            ):
                # Use initial state buffer
                is_use_state_buffer = True
                initial_state = self.world_state.init_ref_paths_agent_related_from_buffer(env_i,
                                                                                          self.initial_state_buffer)
                # print(colored(f"[LOG] Reset with path ids: {initial_state[:, -2]}", "red"))
            else:
                initial_state = None
                is_use_state_buffer = False

            if not is_reset_single_agent:
                # Each time step of a simulation
                self.timer.step[env_i] = 0

            self.world_state.reset(self.world_state.get_agent_state_list(),
                                   env_index=env_i,
                                   agent_index=agent_index,
                                   initial_state=initial_state,
                                   initial_state_buffer=self.state_buffer)

        reset_indices = range(self.n_agents) if agent_index is None else agent_index.unsqueeze(0)
        env_j = slice(None)

        if env_index:
            env_j = env_index

        for i_agent in reset_indices:
            self.world_state.reset_init_distances_and_short_term_ref_path(self.world_state.get_agent_state_list(),
                                                                          env_j,
                                                                          i_agent)

        # Compute mutual distances between agents
        self.world_state.update_mutual_distances(self.world_state.get_agent_state_list(), env_index=env_j)

        self.world_state.reset_collisions(env_j)

        # Reset the state buffer
        self.state_buffer.reset()

        state_add = torch.cat(
            (
                torch.stack([a.state.pos for a in agents], dim=1),
                torch.stack([a.state.rot for a in agents], dim=1),
                torch.stack([a.state.vel for a in agents], dim=1),
                self.world_state.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                self.world_state.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                self.world_state.ref_paths_agent_related.point_id[:].unsqueeze(-1),
            ),
            dim=-1,
        )
        self.state_buffer.add(state_add)  # Add new state

    def reward(self, agent: Agent):
        """
        Issue rewards for the given agent in all envs.
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
        # Initialize
        self.rew[:] = 0

        # Get the index of the current agent
        agent_index = self.world.agents.index(agent)

        if agent_index == 0:  # Avoid repeated computations
            # Timer
            self.timer.step_duration[self.timer.step] = (
                time.time() - self.timer.step_begin
            )
            self.timer.step_begin = (
                time.time()
            )  # Set to the current time as the begin of the current time step
            self.timer.step += 1  # Increment step by 1
            # print(self.timer.step)

        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        self.world_state.update_state_before_rewarding(agent_index)

        ##################################################
        ## [reward] forward movement
        ##################################################
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(
            1
        )  # Vector of the current movement

        ref_points_vecs = self.world_state.ref_paths_agent_related.short_term[
                          :, agent_index
                          ] - latest_state[:, agent_index, 0:2].unsqueeze(
            1
        )  # Vectors from the previous position to the points on the short-term reference path
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(
            move_projected, self.rewards.weighting_ref_directions
        )  # Put more weights on nearing reference points

        reward_movement = (
                move_projected_weighted
                / (agent.max_speed * self.world.dt)
                * self.rewards.progress
        )
        self.rew += reward_movement  # Relative to the maximum possible movement

        ##################################################
        ## [reward] high velocity
        ##################################################
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(
            -1
        )
        factor_moving_direction = torch.where(
            v_proj > 0, 1, 2
        )  # Get penalty if move in negative direction

        reward_vel = (
                factor_moving_direction * v_proj / agent.max_speed * self.rewards.higth_v
        )
        self.rew += reward_vel

        ##################################################
        ## [reward] reach goal
        ##################################################
        reward_goal = (
                self.world_state.collisions.with_exit_segments[:, agent_index] * self.rewards.reach_goal
        )
        self.rew += reward_goal

        ##################################################
        ## [penalty] close to lanelet boundaries
        ##################################################
        penalty_close_to_lanelets = (
                exponential_decreasing_fcn(
                    x=self.world_state.distances.boundaries[:, agent_index],
                    x0=self.thresholds.near_boundary_low,
                    x1=self.thresholds.near_boundary_high,
                )
                * self.penalties.near_boundary
        )
        self.rew += penalty_close_to_lanelets

        ##################################################
        ## [penalty] close to other agents
        ##################################################
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.world_state.distances.agents[:, agent_index, :],
            x0=self.thresholds.near_other_agents_low,
            x1=self.thresholds.near_other_agents_high,
        )
        penalty_close_to_agents = (
                torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        )
        self.rew += penalty_close_to_agents

        ##################################################
        ## [penalty] deviating from reference path
        ##################################################
        self.rew += (
                self.world_state.distances.ref_paths[:, agent_index]
                / self.thresholds.deviate_from_ref_path
                * self.penalties.deviate_from_ref_path
        )

        ##################################################
        ## [penalty] changing steering too quick
        ##################################################
        steering_current = self.observation_provider.observations.past_action_steering.get_latest(n=1)[
                           :, agent_index
                           ]
        steering_past = self.observation_provider.observations.past_action_steering.get_latest(n=2)[
                        :, agent_index
                        ]

        steering_change = torch.clamp(
            (steering_current - steering_past).abs() * self.normalizers.steering
            - self.thresholds.change_steering,  # Not forget to denormalize
            min=0,
        )
        steering_change_reward_factor = steering_change / (
                2 * agent.u_range[1] - 2 * self.thresholds.change_steering
        )
        penalty_change_steering = (
                steering_change_reward_factor * self.penalties.change_steering
        )
        self.rew += penalty_change_steering

        # ##################################################
        # ## [penalty] colliding with other agents
        # ##################################################
        is_collide_with_agents = self.world_state.collisions.with_agents[:, agent_index]
        penalty_collide_other_agents = (
                is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        )
        self.rew += penalty_collide_other_agents

        ##################################################
        ## [penalty] colliding with lanelet boundaries
        ##################################################
        is_collide_with_lanelets = self.world_state.collisions.with_lanelets[:, agent_index]
        penalty_collide_lanelet = (
                is_collide_with_lanelets * self.penalties.collide_with_boundaries
        )
        self.rew += penalty_collide_lanelet

        ##################################################
        ## [penalty/reward] time
        ##################################################
        # Get time reward if moving in positive direction; otherwise get time penalty
        time_reward = (
                torch.where(v_proj > 0, 1, -1)
                * agent.state.vel.norm(dim=-1)
                / agent.max_speed
                * self.penalties.time
        )
        self.rew += time_reward

        if agent_index == (self.n_agents - 1):  # Avoid repeated updating
            state_add = torch.cat(
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
            self.state_buffer.add(state_add)

        # [update] previous positions and short-term reference paths
        self.world_state.update_state_after_rewarding(agent_index)

        assert not self.rew.isnan().any(), "Rewards contain nan."
        assert not self.rew.isinf().any(), "Rewards contain inf."

        # Clamed the reward to avoid abs(reward) being too large
        rew_clamed = torch.clamp(self.rew, min=-1, max=1)

        return rew_clamed

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
                    lane boundaries (or distances to them).
                The observation of its surrounding agents includes their
                    vertices (or positions and rotations),
                    velocities,
                    distances to them (optional), and
                    reference paths (optional).
        """
        agent_index = self.world.agents.index(agent)

        if agent_index == 0:  # Avoid repeated computations
            self.observation_provider.update_state(self.world)

        obs = self.observation_provider.get_observation(agent_index)

        # Store observation for reuse in `info()`, only relevant for prioritized MARL
        self.stored_observations[agent_index] = obs

        return obs

    def done(self):
        # print("[DEBUG] done()")
        is_collision_with_agents = self.world_state.collisions.with_agents.view(
            self.world.batch_dim, -1
        ).any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.world_state.collisions.with_lanelets.any(dim=-1)
        is_leaving_entry_segment = self.world_state.collisions.with_entry_segments.any(dim=-1) & (
                self.timer.step >= 20
        )
        is_any_agents_leaving_exit_segment = self.world_state.collisions.with_exit_segments.any(
            dim=-1
        )
        is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)

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

        if self.parameters.is_testing_mode:
            is_done = is_max_steps_reached  # In test mode, we only reset the whole env if the maximum time steps are reached

            # Reset single agent
            agents_reset = (
                    self.world_state.collisions.with_agents.any(dim=-1)
                    | self.world_state.collisions.with_lanelets
                    | self.world_state.collisions.with_entry_segments
                    | self.world_state.collisions.with_exit_segments
            )
            agents_reset_indices = torch.where(agents_reset)
            for env_idx, agent_idx in zip(
                    agents_reset_indices[0], agents_reset_indices[1]
            ):
                if not is_done[env_idx]:
                    self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
        else:
            is_done = (
                    is_max_steps_reached
                    | is_collision_with_agents
                    | is_collision_with_lanelets
            )
            if (
                    self.parameters.scenario_type != "CPM_entire"
            ):  # This part only applies to the map that have loop-shaped paths
                # Reset the whole system only when collisions occur. Reset a single agents if it leaves an entry or an exit

                # Reset single agnet
                agents_reset = (
                        self.world_state.collisions.with_entry_segments
                        | self.world_state.collisions.with_exit_segments
                )
                agents_reset_indices = torch.where(agents_reset)
                for env_idx, agent_idx in zip(
                        agents_reset_indices[0], agents_reset_indices[1]
                ):
                    if not is_done[env_idx]:
                        # Skip envs with done flag since later they will be reset anyway
                        self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
                        # print(f"Reset agent {agent_idx} in env {env_idx}")
            else:
                # Reset the whole system once collisions occur. There is no entry or exit in this scenario.
                assert not is_leaving_entry_segment.any()
                assert not is_any_agents_leaving_exit_segment.any()

            assert not (is_collision_with_agents & (self.timer.step == 0)).any()
            assert not (is_collision_with_lanelets & (self.timer.step == 0)).any()
            assert not (is_leaving_entry_segment & (self.timer.step == 0)).any()
            assert not (is_max_steps_reached & (self.timer.step == 0)).any()
            assert not (
                    is_any_agents_leaving_exit_segment & (self.timer.step == 0)
            ).any()

        # Logs
        # if is_collision_with_agents.any():
        #     print("Collide with other agents.")
        # if is_collision_with_lanelets.any():
        #     print("Collide with lanelet.")
        # if is_leaving_entry_segment.any():
        #     print("At least one agent is leaving its entry segment.")
        # if is_max_steps_reached.any():
        #     print("The number of the maximum steps is reached.")
        # if is_any_agents_leaving_exit_segment.any():
        #     print("At least one agent is leaving its exit segment.")

        return is_done

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
        agent_index = self.world.agents.index(agent)  # Index of the current agent

        is_action_empty = agent.action.u is None

        is_collision_with_agents = self.world_state.collisions.with_agents[:, agent_index].any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.world_state.collisions.with_lanelets.any(dim=-1)

        # Zero-padding as a placeholder for actions of surrounding agents
        base_obs = F.pad(
            self.stored_observations[agent_index].clone(),
            (0, self.parameters.n_nearing_agents_observed * AGENTS["n_actions"]),
        )

        # Observation of the policy in the priority-assignment module
        prio_obs = self.stored_observations[agent_index].clone()

        # Observation of the policy in the CBF module, which includes:
        # self-observation: speed
        # self-observation: distance to left lane boundary
        # self-observation: distance to right lane boundary
        # others-observation: vertices of surrounding, observable agents
        # others-observation: velocities of surrounding, observable agents
        # others-observation: jaw angles of surrounding, observable agents
        # others-observation: short-term reference path of surrounding, observable agents
        cbf_obs = None

        info = {
            "pos": agent.state.pos / self.normalizers.pos_world,
            "rot": angle_eliminate_two_pi(agent.state.rot) / self.normalizers.rot,
            "vel": agent.state.vel / self.normalizers.v,
            "act_vel": (
                (agent.action.u[:, 0] / self.normalizers.v)
                if not is_action_empty
                else self.constants.empty_action_vel[:, agent_index]
            ),
            "act_steer": (
                (agent.action.u[:, 1] / self.normalizers.steering)
                if not is_action_empty
                else self.constants.empty_action_steering[:, agent_index]
            ),
            "ref": (
                    self.world_state.ref_paths_agent_related.short_term[:, agent_index]
                    / self.normalizers.pos_world
            ).reshape(self.world.batch_dim, -1),
            "distance_ref": self.world_state.distances.ref_paths[:, agent_index]
                            / self.normalizers.distance_ref,
            "distance_left_b": self.world_state.distances.left_boundaries[:, agent_index].min(
                dim=-1
            )[0]
                               / self.normalizers.distance_lanelet,
            "distance_right_b": self.world_state.distances.right_boundaries[:, agent_index].min(
                dim=-1
            )[0]
                                / self.normalizers.distance_lanelet,
            "is_collision_with_agents": is_collision_with_agents,
            "is_collision_with_lanelets": is_collision_with_lanelets,
            **(
                {"base_observation": base_obs}
                if self.parameters.is_using_prioritized_marl
                else {}
            ),
            **(
                {"priority_observation": prio_obs}
                if self.parameters.is_using_prioritized_marl
                else {}
            ),
            **({"cbf_observation": cbf_obs} if self.parameters.is_using_cbf else {}),
        }

        return info

    def extra_render(self, env_index: int = 0):
        if cicd_testing:
            # Rendering is not needed during CI/CD testing
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

        geoms = []

        self._render_action_propagation_direction(env_index, geoms)

        self._render_lanelets(geoms)

        self._render_extra_info(geoms)

        self._render_reference_paths_and_boundaries(env_index, geoms)

        return geoms

    def _render_lanelets(self, geoms):
        # Visualize all lanelets
        for i in range(len(self.map.parser.lanelets_all)):
            lanelet = self.map.parser.lanelets_all[i]

            geom = rendering.PolyLine(
                v=lanelet["left_boundary"],
                close=False,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*Color.black100)
            geoms.append(geom)

            geom = rendering.PolyLine(
                v=lanelet["right_boundary"],
                close=False,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*Color.black100)
            geoms.append(geom)

    def _render_extra_info(self, geoms):
        if self.parameters.is_visualize_extra_info:
            hight_a = -0.10
            hight_b = -0.20
            hight_c = -0.30

            # Title
            geom = rendering.TextLine(
                text=self.parameters.render_title,
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_a) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            # Time and time step
            geom = rendering.TextLine(
                text=f"t: {self.timer.step[0] * self.parameters.dt:.2f} sec",
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_b) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            geom = rendering.TextLine(
                text=f"n: {self.timer.step[0]}",
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_c) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

    def _render_reference_paths_and_boundaries(self, env_index, geoms):
        for agent_i in range(self.n_agents):
            # Visualize short-term reference paths of agents
            if self.parameters.is_visualize_short_term_path:
                geom = rendering.PolyLine(
                    v=self.world_state.ref_paths_agent_related.short_term[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.colors[agent_i])
                geoms.append(geom)

                for i_p in self.world_state.ref_paths_agent_related.short_term[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.colors[agent_i])
                    geoms.append(circle)

            # Visualize nearing points on boundaries
            if not self.parameters.is_observe_distance_to_boundaries:
                # Left boundary
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

                for i_p in self.world_state.ref_paths_agent_related.nearing_points_left_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.colors[agent_i])
                    geoms.append(circle)

                # Right boundary
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

                for i_p in self.world_state.ref_paths_agent_related.nearing_points_right_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.colors[agent_i])
                    geoms.append(circle)

            # Agent IDs
            geom = rendering.TextLine(
                text=f"{agent_i}",
                x=(
                          self.world.agents[agent_i].state.pos[env_index, 0]
                          / self.world.x_semidim
                  )
                  * self.viewer_size[0],
                y=(
                          self.world.agents[agent_i].state.pos[env_index, 1]
                          / self.world.y_semidim
                  )
                  * self.viewer_size[1],
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            # Lanelet boundaries of agents' reference path
            if self.parameters.is_visualize_lane_boundary:
                if agent_i == 0:
                    # Left boundary
                    geom = rendering.PolyLine(
                        v=self.world_state.ref_paths_agent_related.left_boundary[
                            env_index, agent_i
                        ],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.colors[agent_i])
                    geoms.append(geom)
                    # Right boundary
                    geom = rendering.PolyLine(
                        v=self.world_state.ref_paths_agent_related.right_boundary[
                            env_index, agent_i
                        ],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.colors[agent_i])
                    geoms.append(geom)
                    # Entry
                    geom = rendering.PolyLine(
                        v=self.world_state.ref_paths_agent_related.entry[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.colors[agent_i])
                    geoms.append(geom)
                    # Exit
                    geom = rendering.PolyLine(
                        v=self.world_state.ref_paths_agent_related.exit[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.colors[agent_i])
                    geoms.append(geom)

    def _render_action_propagation_direction(self, env_index, geoms):
        """
        Render the direction of action propagation between agents in a prioritized MARL setting.

        This function visualizes the action propagation directions, which are from higher-priority to lower-priority agents,
        by drawing lines between them. The lines are colored based on the higher-priority agent's color.

        Args:
            env_index (int): The index of the current environment.
            geoms (list): A list of geometric objects to which the new visualizations will be added.
        """
        # Check if prioritized MARL is being used and if observable higher priority agents exist
        if self.parameters.is_using_prioritized_marl and hasattr(
                self, "observable_higher_priority_agents"
        ):
            # self._recolor_agents() # Uncomment if agent recoloring is needed

            # Iterate through all agents to draw action propagation lines
            for i in range(self.n_agents):
                # Get the list of observable higher priority agents for the current agent
                observable_higher_priority_agents = (
                    self.observable_higher_priority_agents[env_index][i]
                )

                # Draw lines from each higher priority agent to the current agent
                for j in observable_higher_priority_agents:
                    # Create a line object representing the action propagation
                    line = rendering.Line(
                        # Starting point: position of the higher priority agent (j)
                        (
                            self.world.agents[j].state.pos[env_index, 0],
                            self.world.agents[j].state.pos[env_index, 1],
                        ),
                        # Ending point: position of the current agent (i)
                        (
                            self.world.agents[i].state.pos[env_index, 0],
                            self.world.agents[i].state.pos[env_index, 1],
                        ),
                        width=10,  # Set line width
                    )

                    # Add transform attribute to the line
                    xform = rendering.Transform()
                    line.add_attr(xform)

                    # Set the line color to match the higher priority agent's color
                    line.set_color(*self.colors[j])

                    # Add the line to the list of geometric objects to be rendered
                    geoms.append(line)

    def _recolor_agents(self):
        """
        This function re-colors agents based on their priorities.
        """
        if (self.priority_rank.ndim == 2) and (self.priority_rank.shape[0] == 1):
            self.priority_rank = self.priority_rank.squeeze(0)
        # Re-order the colors to code priorities
        colors_copy = self.colors.copy()
        self.colors = [colors_copy[self.priority_rank[i]] for i in range(self.n_agents)]

        # Adjust agent colors
        for i in range(self.n_agents):
            self.world.agents[i]._color = self.colors[i]


if __name__ == "__main__":
    scenario = ScenarioRoadTraffic()
    render_interactively(
        scenario=scenario,
        control_two_agents=False,
        shared_reward=False,
    )
