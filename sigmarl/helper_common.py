# Copyright (c) 2025, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import numpy as np
from vmas.simulator.core import AgentState, Agent
from vmas.simulator.utils import TorchUtils, override

import torch
from torch import Tensor
from tensordict import TensorDict

import typing
from matplotlib import colormaps


def get_model_name(parameters):
    model_name = f"reward{parameters.episode_reward_mean_current:.2f}"

    return model_name


class Parameters:
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
        max_grad_norm: float = 1.0,  # Maximum norm for the gradients
        clip_epsilon: float = 0.2,  # Clip value for PPO loss
        gamma: float = 0.99,  # Discount factor from 0 to 1. A greater value corresponds to a better farsight
        lmbda: float = 0.9,  # lambda for generalised advantage estimation
        entropy_eps: float = 1e-4,  # Controls the trade-off between trying new actions (exploration) and optimizing known good actions (exploitation). Higher entropy_coef encourages more exploration by favoring stochastic (less certain) policies.
        max_steps: int = 128,  # Episode steps before done
        num_vmas_envs: int = 32,  # Number of vectorized environments
        scenario_type: str = "intersection_1",  # One of {"cpm_entire", "cpm_mixed", "intersection_1", ...}. See SCENARIOS in utilities/constants.py for more scenarios.
        # "cpm_entire": Entire map of the CPM Lab
        # "cpm_mixed": Intersection, merge-in, and merge-out of the CPM Lab. Probability defined in `cpm_scenario_probabilities`
        # "intersection_1": Intersection with ID 1
        episode_reward_mean_current: float = 0.00,  # Achieved mean episode reward (total/n_agents)
        episode_reward_intermediate: float = -1e3,  # A arbitrary, small initial value
        is_prb: bool = False,  # Whether to enable prioritized replay buffer
        is_challenging_initial_state_buffer=False,  # Whether to enable challenging initial state buffer
        cpm_scenario_probabilities=[
            1.0,
            0.0,
            0.0,
        ],  # Probabilities of training agents in intersection, merge-in, or merge-out scenario
        n_steps_stored: int = 10,  # Store previous `n_steps_stored` steps of states
        # Observation
        n_points_short_term: int = 3,  # Number of points that build a short-term reference path
        is_partial_observation: bool = True,  # Whether to enable partial observation
        n_nearing_agents_observed: int = 2,  # Number of nearing agents to be observed (consider limited sensor range)
        # Parameters for ablation studies
        is_ego_view: bool = True,  # Ego view or bird view
        is_apply_mask: bool = True,  # Whether to mask distant agents
        is_observe_distance_to_agents: bool = True,  # Whether to observe the distance to other agents
        is_observe_distance_to_boundaries: bool = True,  # Whether to observe points on lanelet boundaries or observe the distance to labelet boundaries
        is_observe_distance_to_center_line: bool = True,  # Whether to observe the distance to reference path
        is_observe_vertices: bool = True,  # Whether to observe the vertices of other agents (or center point)
        is_obs_noise: bool = True,  # Whether to add noise to observations
        obs_noise_level: float = 0.05,  # Defines the variance of the normal distribution modeling the noise.
        is_observe_ref_path_other_agents: bool = False,  # Whether to observe the reference paths of other agents
        is_use_mtv_distance: bool = True,  # Whether to use mtv-based (Minimum Translation Vector) distance or c2c-based (center-to-center) distance.
        # Visu
        is_visualize_short_term_path: bool = True,  # Whether to visualize short-term reference paths
        is_visualize_lane_boundary: bool = False,  # Whether to visualize lane boundary
        is_real_time_rendering: bool = False,  # Simulation will be paused at each time step for a certain duration to enable real-time rendering
        is_visualize_extra_info: bool = True,  # Whether to render extra information such time and time step
        render_title: str = "",  # The title to be rendered
        # Save/Load
        is_save_intermediate_model: bool = True,  # Whether to save intermediate model (also called checkpoint) with the hightest episode reward
        is_load_model: bool = False,  # Whether to load saved model
        is_load_final_model: bool = False,  # Whether to load the final model (last iteration)
        model_name: str = None,
        where_to_save: str = "outputs/",  # Define where to save files such as intermediate models
        is_continue_train: bool = False,  # Whether to continue training after loading an offline model
        is_save_eval_results: bool = True,  # Whether to save evaluation results such as figures and evaluation outputs
        is_load_out_td: bool = False,  # Whether to load evaluation outputs
        is_testing_mode: bool = False,  # In testing mode, collisions do not terminate the current simulation
        is_save_simulation_video: bool = False,  # Whether to save simulation videos
        # extensions
        is_using_opponent_modeling: bool = False,  # Whether to use opponent modeling to predict the actions of other agents
        is_using_prioritized_marl: bool = False,  # Whether to use prioritized MARL and action propagation.
        prioritization_method: str = "marl",  # Which method to use for generating priority ranks (options: {"marl", "random"}). Applicable only for prioritized MARL scenarios.
        is_communication_noise: str = False,  # Whether to inject communication noise to propagated actions
        communication_noise_level: float = 0.1,  # Defines the variance of the normal distribution modeling the noise.
        is_using_cbf_testing: bool = False,  # Whether to use Control Barrier Function (CBF) during testing
        is_using_cbf_training: bool = False,  # Whether to use Control Barrier Function (CBF) during training
        is_using_centralized_cbf: bool = False,  # Whether to use centralized solving for CBF-constrained MARL
        is_apply_cbf_action: bool = False,  # Whether to apply CBF action (use when is_using_cbf_training is True, deciding if CBF action or RL action should be applied)
        is_solve_qp: bool = True,  # Whether to solve QP (if False, use the nominal action to access the constraint violation/value)
        experiment_type: str = "simulation",  # One of {"simulation", "lab"}. If you only use simulation, you do not need to worry about "lab".
        is_obs_steering: bool = False,  # Whether to observe the steering angle of other agents
        predefined_ref_path_idx: list[
            int
        ] = None,  # A list of integers specify the index of the predefined reference path for each. They will be used when initializing/resetting the agents. Set to None to use a randomly selected ones.
        init_state: list[
            float
        ] = None,  # Initial state of the agents in the form of [x, y, rot, speed]. If None, the initial state will be randomly sampled.
        random_seed: int = 0,  # Random seed,
        is_using_pseudo_distance: bool = False,  # Whether to use pseudo distance
        n_circles_approximate_vehicle: int = 3,  # Number of circles to approximate the vehicle shape
        lane_width=0.25,  # For custom scenarios only
        reset_agent_fixed_duration: int = 0,  # Reset agents after fixed duration in seconds. Set to 0 if not used.
        is_grouping_agents: bool = False,  # Whether to use grouping agents
        max_group_size: int = 2,  # Maximum number of agents in a group
        observation_range: float = 0.5,  # Observation range for each agent in meters
        nom_controller_type: str = "rl",  # Type of the nominal controller: {"rl", "clf"}
        adaptive_lambda: bool = False,  # Whether to use adaptive lambda for CBF-QP
        rs: float = 0.5,  # (0,1), responsibility score for CBF-QP (the higher, the more responsible that an agent is for avoiding collisions)
        h_nom: float = 0.2,  # Nominizer of the CBF function (use when is_using_cbf_training True)
        rew_method: str = "distance",  # Reward method: {"distance", "cbf", "ttc", "sparse", "distance_sparse", "cbf_sparse", "ttc_sparse"}
        reward_progress: float = 10,  # Reward for progress along the reference path
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
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.max_steps = max_steps

        self.scenario_type = scenario_type

        self.num_vmas_envs = num_vmas_envs

        self.is_save_intermediate_model = is_save_intermediate_model
        self.is_load_model = is_load_model
        self.is_load_final_model = is_load_final_model

        self.episode_reward_mean_current = episode_reward_mean_current
        self.episode_reward_intermediate = episode_reward_intermediate
        self.where_to_save = where_to_save
        self.is_continue_train = is_continue_train

        self.n_points_short_term = n_points_short_term
        # Observation
        self.is_partial_observation = is_partial_observation
        self.n_steps_stored = n_steps_stored
        self.n_nearing_agents_observed = n_nearing_agents_observed
        self.is_observe_distance_to_agents = is_observe_distance_to_agents

        self.is_testing_mode = is_testing_mode
        self.is_save_simulation_video = is_save_simulation_video
        self.is_visualize_short_term_path = is_visualize_short_term_path
        self.is_visualize_lane_boundary = is_visualize_lane_boundary

        self.is_ego_view = is_ego_view
        self.is_apply_mask = is_apply_mask
        self.is_use_mtv_distance = is_use_mtv_distance
        self.is_observe_distance_to_boundaries = is_observe_distance_to_boundaries
        self.is_observe_distance_to_center_line = is_observe_distance_to_center_line
        self.is_observe_vertices = is_observe_vertices
        self.is_obs_noise = is_obs_noise
        self.obs_noise_level = obs_noise_level
        self.is_observe_ref_path_other_agents = is_observe_ref_path_other_agents

        self.is_save_eval_results = is_save_eval_results
        self.is_load_out_td = is_load_out_td

        self.is_real_time_rendering = is_real_time_rendering
        self.is_visualize_extra_info = is_visualize_extra_info
        self.render_title = render_title

        self.is_prb = is_prb
        self.is_challenging_initial_state_buffer = is_challenging_initial_state_buffer

        self.cpm_scenario_probabilities = cpm_scenario_probabilities

        self.is_using_opponent_modeling = is_using_opponent_modeling
        self.is_using_prioritized_marl = is_using_prioritized_marl

        self.prioritization_method = prioritization_method
        self.is_communication_noise = is_communication_noise
        self.communication_noise_level = communication_noise_level

        self.is_using_cbf_testing = is_using_cbf_testing
        self.is_using_cbf_training = is_using_cbf_training
        self.is_using_centralized_cbf = is_using_centralized_cbf
        self.is_apply_cbf_action = is_apply_cbf_action
        self.is_solve_qp = is_solve_qp

        self.experiment_type = experiment_type
        self.is_obs_steering = is_obs_steering

        self.predefined_ref_path_idx = predefined_ref_path_idx
        self.init_state = init_state

        self.random_seed = random_seed

        self.is_using_pseudo_distance = is_using_pseudo_distance
        self.n_circles_approximate_vehicle = n_circles_approximate_vehicle

        self.lane_width = lane_width

        self.reset_agent_fixed_duration = reset_agent_fixed_duration
        self.is_grouping_agents = is_grouping_agents
        self.max_group_size = max_group_size
        self.observation_range = observation_range
        self.nom_controller_type = nom_controller_type
        self.adaptive_lambda = adaptive_lambda
        self.rs = rs
        self.h_nom = h_nom
        self.rew_method = rew_method
        self.reward_progress = reward_progress

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
        # Create an instance of the class from a dictionary
        return cls(**dict_data)

    @classmethod
    def from_json(cls, config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
            return cls(**config)


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

    @override(AgentState)
    def _reset(self, env_index: typing.Optional[int]):
        for attr_name in ["speed", "steering", "sideslip_angle"]:
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
        for attr_name in ["speed", "steering", "sideslip_angle"]:
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
        super()._spawn(dim_c, dim_p)


class Vehicle(Agent):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Vehicle by calling the parent Agent's initializer
        and setting up the VehicleState.

        Redefine _state attribute to be VehicleState.
        """
        super().__init__(*args, **kwargs)

        # Replace the default AgentState with VehicleState
        self._state = VehicleState()

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


def save_video(
    name: str,
    frame_list: list[np.ndarray],
    fps: int,
    fmt: str = "mp4",
    quality: str = "medium",
):
    """
    Save a video from a list of frames with user-defined format and quality.

    Args:
        name (str): Base name of the video file (without extension).
        frame_list (list[np.ndarray]): List of RGB frames (H x W x 3) as NumPy arrays.
        fps (int): Frames per second for the output video.
        fmt (str, optional): Output format, one of {"mp4", "avi"}. Default is "mp4".
        quality (str, optional): Video quality level, one of {"low", "medium", "high"}, only applied to mp4.
                                 Default is "medium".
    """
    import cv2
    import subprocess
    import os
    import uuid

    if len(frame_list) == 0:
        raise ValueError("frame_list is empty. Cannot save video.")

    h, w = frame_list[0].shape[:2]
    fmt = fmt.lower()
    quality = quality.lower()

    if fmt not in {"mp4", "avi"}:
        raise ValueError("Unsupported format. Use 'mp4' or 'avi'.")

    # Write temporary file
    tmp_file = f"{name}_{uuid.uuid4().hex}_tmp.{fmt}"
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if fmt == "mp4" else "MJPG"))
    video = cv2.VideoWriter(tmp_file, fourcc, fps, (w, h))
    for f in frame_list:
        video.write(cv2.cvtColor(f.astype("uint8"), cv2.COLOR_RGB2BGR))
    video.release()

    # Post-process for quality
    if fmt == "mp4":
        crf_map = {"low": "28", "medium": "23", "high": "18"}
        crf = crf_map.get(quality, "23")
        final_name = f"{name}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_file,
                "-c:v",
                "libx264",
                "-crf",
                crf,
                "-preset",
                "slow",
                "-pix_fmt",
                "yuv420p",
                final_name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    elif fmt == "avi":
        # Lossless re-encoding preserving RGB (pure white background)
        final_name = f"{name}.avi"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_file,
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "rgb24",
                final_name,
            ],
            check=False,
        )

    # Clean up
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    return final_name


def get_n_colors_cmap(n, cmap_name="tab20"):
    cmap = colormaps.get_cmap(cmap_name).resampled(n)
    return [cmap(i)[:3] for i in range(n)]  # RGB in [0,1]


def get_name_suffix(
    grouping, is_using_cbf_testing, n_agent, seed, max_group, nom, scenario
):
    if grouping:
        if is_using_cbf_testing:
            name_suffix = (
                f"agents_{n_agent}_seed_{seed}_"
                f"grouping_on_maxgroup_{max_group}_"
                f"nom_{nom}_scenario_{scenario.lower()}"
            )
        else:
            name_suffix = (
                f"agents_{n_agent}_seed_{seed}_"
                f"nom_{nom}_only_scenario_{scenario.lower()}"
            )

    else:
        if is_using_cbf_testing:
            name_suffix = (
                f"agents_{n_agent}_seed_{seed}_"
                f"grouping_off_"
                f"nom_{nom}_scenario_{scenario.lower()}"
            )
        else:
            name_suffix = (
                f"agents_{n_agent}_seed_{seed}_"
                f"nom_{nom}_only_scenario_{scenario.lower()}"
            )
    return name_suffix


def trim_td(out_td: TensorDict, keys_to_keep=None) -> TensorDict:
    """
    Trim a TensorDict to keep only selected fields.

    Args:
        out_td: Original TensorDict.

    Returns:
        A new TensorDict containing only the specified keys.
    """
    if keys_to_keep is None:
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
