# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from termcolor import cprint
import torch
from torch import Tensor

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

from tensordict.nn import TensorDictModule

from tensordict.nn.distributions import NormalParamExtractor

from tensordict.tensordict import TensorDict

import matplotlib

import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch

# Set up font
matplotlib.rcParams["pdf.fonttype"] = 42  # Use Type 1 fonts (vector fonts)
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams.update({"font.size": 11})  # Set global font size
plt.rcParams["text.usetex"] = True

from sigmarl.mtv_based_sm_predictor import SafetyMarginEstimatorModule

from sigmarl.dynamics import KinematicBicycleModel

from sigmarl.helper_scenario import (
    Normalizers,
    get_perpendicular_distances,
    get_distances_between_agents,
)

from sigmarl.constants import AGENTS

import random
import time

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class ConstantController:
    """
    Controller that returns constant zero actions (no movement).
    """

    def __init__(self, device="cpu"):
        self.device = device

    def get_actions(self):
        """
        Return constant zero actions.
        """
        return torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device)


class CBF:
    def __init__(self, **kwargs):
        # Initialize simulation parameters
        self.initialize_params(**kwargs)

        # Load safety margin estimator module
        self.load_safety_margin_estimator()

        self.compute_rectangles()

        # Initialize Kinematic Bicycle Model
        self.kbm = KinematicBicycleModel(
            l_f=self.l_f,
            l_r=self.l_r,
            max_speed=AGENTS["max_speed"],
            max_steering=AGENTS["max_steering"],
            max_acc=AGENTS["max_acc"],
            min_acc=AGENTS["min_acc"],
            max_steering_rate=AGENTS["max_steering_rate"],
            min_steering_rate=AGENTS["min_steering_rate"],
            device=self.device,
        )

        self.initialize_norminal_controller()

        # Setup plot
        self.setup_plot()

    def initialize_params(self, **kwargs):
        """
        Initialize and return all simulation and vehicle parameters.
        """
        # General
        self.device = "cpu"
        self.dt = 0.05  # Sample time (50 ms)
        self.length = AGENTS["length"]  # Length of each rectangle (m)
        self.width = AGENTS["width"]  # Width of each rectangle (m)
        self.l_wb = AGENTS["l_wb"]  # Wheelbase
        self.l_f = AGENTS["l_f"]  # Front wheelbase (m)
        self.l_r = AGENTS["l_r"]  # Rear wheelbase (m)
        self.v_max = torch.tensor(
            AGENTS["max_speed"], device=self.device, dtype=torch.float32
        )
        self.v_min = torch.tensor(
            AGENTS["min_speed"], device=self.device, dtype=torch.float32
        )
        self.steering_max = torch.tensor(
            AGENTS["max_steering"], device=self.device, dtype=torch.float32
        )
        self.steering_min = torch.tensor(
            AGENTS["min_steering"], device=self.device, dtype=torch.float32
        )
        self.a_max = AGENTS["max_acc"]  # Max acceleration (m/s^2)
        self.a_min = AGENTS["min_acc"]  # Max deceleration (m/s^2)
        self.steering_rate_max = AGENTS[
            "max_steering_rate"
        ]  # Max steering rate (rad/s)
        self.steering_rate_min = AGENTS["min_steering_rate"]

        self.lane_width = self.width * 1.8  # Lane width

        self.is_save_video = kwargs.get("is_save_video", False)
        self.is_visu_ref_path = kwargs.get("is_visu_ref_path", False)
        self.is_visu_footprint = kwargs.get("is_visu_footprint", True)
        self.is_visu_nominal_action = kwargs.get("is_visu_nominal_action", False)
        self.is_visu_cbf_action = kwargs.get("is_visu_cbf_action", False)
        self.is_visu_time = kwargs.get("is_visu_time", True)
        self.is_visu_cost = kwargs.get("is_visu_cost", False)
        self.is_save_eval_result = kwargs.get("is_save_eval_result", True)

        self.font_size_video = 16

        # Safety margin
        # Radius of the circle that convers the vehicle
        self.radius = np.sqrt(self.length**2 + self.width**2) / 2

        # On which lane is an agent's center of gravity (1 for the top lane; 2 for the bottom lane)
        self.lane_i = "2"
        self.lane_j = "2"

        self.list_lane_i = [self.lane_i]
        self.list_lane_j = [self.lane_j]

        self.within_lane_threshold = (
            20  # Number of steps to determine if an agent is within a lane
        )

        self.is_overtake = False  # Whether agent i is overtaking agent j
        self.list_is_overtake = [self.is_overtake]

        self.is_obstructe = False  # Whether agent j is obstructing agent i
        self.list_is_obstructe = [self.is_obstructe]
        self.obstructe_times_max = 3  # Number of times agent j obstructes agent i
        self.n_success_obstructe = (
            0  # Number of times agent j successfully obstructes agent i
        )

        self.safety_buffer = 0
        self.add_longitudinal_safety_buffer = False

        self.overtake_target_lane = None  # Agent 1's target lane for overtaking
        self.obstructe_target_lane = None  # Agent 2's target lane for bypassing

        # y-coordinate of lane boundaries
        self.y_lane_top_bound = self.lane_width
        self.y_lane_center_line = 0
        self.y_lane_bottom_bound = -self.lane_width

        self.y_lane_1 = (self.y_lane_center_line + self.y_lane_top_bound) / 2
        self.y_lane_2 = (self.y_lane_center_line + self.y_lane_bottom_bound) / 2

        self.threshold_success_overtake = (
            self.length
        )  # Agent i successfully overtakes agent j if its longitudinal coordinate is larger than agent j plus a threshold

        self.threshold_within_lane = self.width / 2
        self.threshold_overtake = (
            3 * self.length
        )  # If agents' longitudinal distance is less than this value, one agent will overtake another agent, given they are within the same lane

        self.threshold_obstructe = (
            2 * self.length
        )  # If agents' longitudinal distance is less than this value, agent j will obstructe agent i

        # Two scenarios are available:
        # (1) overtaking: the ego agent, controlled by a greedy RL policy with CBF verification,
        # needs to overtake its precede agent that moves slowly
        # (2) bypassing: two agents, both controlled by a greedy RL policy with CBF verification,
        # needs to bypass each other within a confined space
        self.scenario_type = kwargs.get(
            "scenario_type", "overtaking"
        )  # One of "overtaking" and "bypassing"

        # Two types of safety margin: center-center-based safety margin, Minimum Translation Vector (MTV)-based safety margin
        self.sm_type = kwargs.get("sm_type", "mtv")  # One of "c2c" and "mtv"

        # CBF
        if self.scenario_type.lower() == "overtaking":
            # Initialize states ([x, y, psi, v, delta]) and goal states ([x, y])
            self.state_i = torch.tensor(
                [-1.2, self.y_lane_2, 0, 1.0, 0.0], dtype=torch.float32
            )
            self.state_j = torch.tensor(
                [-0.4, self.y_lane_2, 0.0, 0.4, 0.0], dtype=torch.float32
            )
            self.goal_i = torch.tensor(
                [10, self.y_lane_2], device=self.device, dtype=torch.float32
            )
            self.goal_j = torch.tensor(
                [10, self.y_lane_2], device=self.device, dtype=torch.float32
            )

            # Plot limits
            self.plot_x_min = min(self.state_i[0].item(), self.state_j[0].item()) - 0.25
            self.plot_x_max = 3.5
            self.plot_y_min = self.y_lane_bottom_bound - 0.2
            self.plot_y_max = self.y_lane_top_bound + 0.2

            self.visu_step_interval = 12

            # Irrelevant
            self.evasion_step_start = None
            self.evasion_step_end = None
            self.evasive_offset = None

            if self.sm_type.lower() == "c2c":
                # Overtaking & C2C
                self.total_time = 7.0  # Total simulation time
                self.lambda_cbf = 2  # Design parameter for CBF
                self.w_acc = 0.1  # Weight for acceleration in QP
                self.w_steer = 0.1  # Weight for steering rate in QP
                self.Q = np.diag(
                    [self.w_acc, self.w_steer]
                )  # Weights for acceleration and steering rate
            else:
                # Overtaking & MTV
                self.total_time = 7.0  # Total simulation time
                self.lambda_cbf = 2  # Design parameter for CBF
                self.w_acc = 0.1  # Weight for acceleration in QP
                self.w_steer = 0.1  # Weight for steering rate in QP
                self.Q = np.diag(
                    [self.w_acc, self.w_steer]
                )  # Weights for acceleration and steering rate
        else:
            # Initialize states ([x, y, psi, v, delta]) and goal states ([x, y])
            # In the bypassing scenario, two agents are facing each other
            self.state_i = torch.tensor(
                [-1.2, self.y_lane_center_line, 0.0, 1.5, 0.0], dtype=torch.float32
            )
            self.state_j = torch.tensor(
                [1.2, self.y_lane_center_line, -np.pi, 1.0, 0.0], dtype=torch.float32
            )
            self.goal_i = self.state_j[0:2].clone()
            self.goal_i[0] += 2
            self.goal_j = self.state_i[0:2].clone()
            self.goal_j[0] -= 2

            # Plot limits
            self.plot_x_min = min(self.state_i[0].item(), self.state_j[0].item()) - 0.25
            self.plot_x_max = max(self.state_i[0].item(), self.state_j[0].item()) + 0.25
            self.plot_y_min = self.y_lane_bottom_bound - 0.15
            self.plot_y_max = self.y_lane_top_bound + 0.15

            self.visu_step_interval = 6

            if self.sm_type.lower() == "c2c":
                # Bypassing & C2C
                self.total_time = 3.8  # Total simulation time

                self.lambda_cbf = 3  # Design parameter for CBF
                self.evasion_step_start = 10
                self.evasion_step_end = 50
                self.evasive_offset = self.radius * 1.3

                self.w_acc = 1  # Weight for acceleration in QP
                self.w_steer = 1  # Weight for steering rate in QP
                self.Q = np.diag(
                    [self.w_acc, self.w_steer]
                )  # Weights for acceleration and steering rate
            else:
                # Bypassing & MTV
                self.total_time = 3.0  # Total simulation time

                self.lambda_cbf = 6  # Design parameter for CBF
                self.evasion_step_start = 15
                self.evasion_step_end = 40
                self.evasive_offset = 0.5 * self.lane_width

                self.w_acc = 1  # Weight for acceleration in QP
                self.w_steer = 1  # Weight for steering rate in QP
                self.Q = np.diag(
                    [self.w_acc, self.w_steer]
                )  # Weights for acceleration and steering rate

        self.num_steps = int(self.total_time / self.dt)  # Total simulation time steps

        self.list_state_i = [self.state_i.clone().numpy()]
        self.list_state_j = [self.state_j.clone().numpy()]

        self.color_i = "tab:blue"
        self.color_j = "tab:green"

        self.color_psi_0 = "tab:blue"  # CBF value (safety margin)
        self.color_psi_1 = "tab:orange"  # CBF condition 1
        self.color_psi_2 = "tab:green"  # CBF condition 2

        self.color_cost_acc = "tab:blue"
        self.color_cost_steer = "tab:orange"
        self.color_cost_total = "tab:green"

        # RL policy
        self.rl_policy_path = "checkpoints/ecc25/nominal_controller.pth"
        self.rl_observation_key = ("agents", "observation")
        self.rl_action_key = ("agents", "action")
        self.rl_policy_n_in_feature = 9  # Number of input size
        self.rl_policy_n_out_feature = 4  # Number of output size: 2 * n_actions
        self.rl_n_points_ref = 3  # Number of points on the short-term reference path
        self.rl_is_add_noise = False  # Whether add noise to observation
        self.rl_noise_level = 0.1  # Noise will be generated by the standary normal distribution. This parameter controls the noise level
        self.rl_distance_between_points_ref_path = (
            self.length
        )  # Distance between the points in the short-term reference paths

        # Initialize optimization variable placeholder for vehicle i
        self.u_placeholder = torch.tensor(
            [0.0, 0.0], dtype=torch.float32
        )  # Do not update its value

        self.list_h_ji = []
        self.list_dot_h_ji = []
        self.list_ddot_h_ji = []
        self.list_cbf_condition_1_ji = []
        self.list_cbf_condition_2_ji = []
        self.list_h_ij = []
        self.list_dot_h_ij = []
        self.list_ddot_h_ij = []
        self.list_cbf_condition_1_ij = []
        self.list_cbf_condition_2_ij = []

        self.list_time = []

        self.list_rectangles_i = []
        self.list_rectangles_j = []

        self.list_opt_duration = []

        self.list_cost_acc = []
        self.list_cost_steer = []
        self.list_cost_total = []
        self.list_is_solve_success = []

        self.step = 0  # Simulation step counter

    def load_safety_margin_estimator(self):
        """
        Load the safety margin estimator module.

        Args:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.

        Returns:
            SafetyMarginEstimatorModule: safety margin estimator module.
        """
        self.SME = SafetyMarginEstimatorModule(
            length=self.length,
            width=self.width,
            path_nn="checkpoints/ecc25/sm_predictor.pth",
        )

        self.SME.load_model()  # Neural network for estimating safety margin

    def load_rl_policy(self):
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=self.rl_policy_n_in_feature,  # n_obs_per_agent
                n_agent_outputs=self.rl_policy_n_out_feature,  # 2 * n_actions
                n_agents=1,
                centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
                share_params=True,  # sharing parameters means that agents will all share the same policy, which will allow them to benefit from each other’s experiences, resulting in faster training. On the other hand, it will make them behaviorally homogenous, as they will share the same model
                device=self.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a `loc` and a non-negative `scale``, used as parameters for a normal distribution (mean and standard deviation)
        )

        # print("policy_net:", policy_net, "\n")

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[self.rl_observation_key],
            out_keys=[
                ("agents", "loc"),
                ("agents", "scale"),
            ],  # Represents the parameters of the policy distribution for each agent
        )

        # Use a probabilistic actor allows for exploration
        policy = ProbabilisticActor(
            module=policy_module,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[self.rl_action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": torch.tensor(
                    [self.v_min, self.steering_min],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "max": torch.tensor(
                    [self.v_max, self.steering_max],
                    device=self.device,
                    dtype=torch.float32,
                ),
            },
            return_log_prob=True,
            log_prob_key=(
                "agents",
                "sample_log_prob",
            ),  # log probability favors numerical stability and gradient calculation
        )  # we'll need the log-prob for the PPO loss

        if os.path.exists(self.rl_policy_path):
            policy.load_state_dict(torch.load(self.rl_policy_path, weights_only=True))
        else:
            raise FileNotFoundError(
                f"Model file at {self.rl_policy_path} not found. See README.md for more details regarding where you can download the pre-trained model."
            )

        cprint(f"[INFO] Loaded the nominal model '{self.rl_policy_path}'", "blue")

        return policy

    def initialize_norminal_controller(self):
        self.nomi_cont_i = self.load_rl_policy()
        # Initialize tensordicts for RL policy call
        self.tensordict_i = TensorDict(
            {
                self.rl_observation_key: torch.zeros(
                    (1, self.rl_policy_n_in_feature),
                    device=self.device,
                    dtype=torch.float32,
                )
            },
            batch_size=[1],
        )

        # In the bapassing scenario, agent j is controlled by a greedy RL policy with CBF verification
        self.nomi_cont_j = self.load_rl_policy()
        self.tensordict_j = TensorDict(
            {
                self.rl_observation_key: torch.zeros(
                    (1, self.rl_policy_n_in_feature),
                    device=self.device,
                    dtype=torch.float32,
                )
            },
            batch_size=[1],
        )

        # Initialize normalizers for preparing observations for the RL policy
        self.rl_normalizers = Normalizers(
            pos=torch.tensor(
                [
                    self.rl_distance_between_points_ref_path * self.rl_n_points_ref,
                    self.rl_distance_between_points_ref_path * self.rl_n_points_ref,
                ],
                device=self.device,
                dtype=torch.float32,
            ),
            v=self.v_max,
            rot=torch.tensor(2 * torch.pi, device=self.device, dtype=torch.float32),
            steering=self.steering_max,
            distance_ref=self.width * 2,
        )

    @staticmethod
    def compute_relative_poses(
        x_ego, y_ego, psi_ego, x_target, y_target, psi_target, sm_type
    ):
        """
        Compute the relative poses (positions and heading) between the ego vehicle and the target.

        Returns:
            tuple: Relative positions and heading in vehicle i's coordinate system.
        """

        dx = x_target - x_ego
        dy = y_target - y_ego
        psi_relative = psi_target - psi_ego  # Relative heading
        psi_relative = (psi_relative + torch.pi) % (
            2 * torch.pi
        ) - torch.pi  # Normalize psi_relative to [-pi, pi]

        # Transform to vehicle i's coordinate system
        distance = torch.sqrt(dx**2 + dy**2)
        psi_relative_coordinate = torch.atan2(dy, dx) - psi_ego
        x_relative = distance * torch.cos(psi_relative_coordinate)
        y_relative = distance * torch.sin(psi_relative_coordinate)

        # return dx, dy, psi_relative, distance
        return x_relative, y_relative, psi_relative, distance

    def estimate_safety_margin(self, x_relative, y_relative, psi_relative):
        """
        Estimate the safety margin using the neural network.

        Returns:
            tuple: Estimated safety margin and its gradient w.r.t inputs.
        """
        if self.sm_type.lower() == "c2c":
            sm, grad, hessian = self.c2c_based_sm(x_relative, y_relative, psi_relative)
        elif self.sm_type.lower() == "mtv":
            # In case of MTV-based safety margin, C2C-based safety margin will be used if two agents are distant.
            if (
                (x_relative > self.SME.x_min)
                and (x_relative < self.SME.x_max)
                and (y_relative > self.SME.y_min)
                and (y_relative < self.SME.y_max)
            ):
                # Use MTV-based safety margin, which considers the headings, to estimate safety margin if the surrounding objective is geometrically inside the allowed ranges
                sm, grad, hessian = self.mtv_based_sm(
                    x_relative, y_relative, psi_relative
                )

            else:
                # If the surrounding objective is outside the ranges, using c2c-based safety margin, which does not consider headings, to estimate safety margin.
                # This is allowable since the objective if far and the heading information is unessential.
                sm, grad, hessian = self.c2c_based_sm(
                    x_relative, y_relative, psi_relative
                )
        else:
            raise ValueError(
                f"Safety margin must be one of 'c2c' and 'mtv'. Got: {self.sm_type}."
            )

        return sm, grad, hessian

    def c2c_based_sm(self, x_ji, y_ji, psi_ji):
        """
        Center-to-Center (c2c)-based safety margin.

        Computes the c2c distance along with its first and second-order partial derivatives
        with respect to the inputs x_ji, y_ji, and psi_ji.

        For validation:
            ∂d/∂x = x / d
            ∂d/∂y = y / d
            ∂d/∂psi = 0
            ∂d^2/∂x^2 = y^2 / d^3
            ∂d^2/∂y^2 = x^2 / d^3
            ∂d^2/∂psi^2 = 0
            ∂d^2/∂x∂y = -x * y / d^3

        Args:
            x_ji (float or torch.Tensor): The x-coordinate of the objective in vehicle i's ego coordinate system.
            y_ji (float or torch.Tensor): The y-coordinate of the objective in vehicle i's ego coordinate system.
            psi_ji (float or torch.Tensor): The orientation (psi) of vehicle i (not used in distance calculation).

        Returns:
            tuple:
                - d_actual (float): The computed center-to-center distance.
                - grad_d_actual (numpy.ndarray): The first-order partial derivatives [∂d/∂x, ∂d/∂y, ∂d/∂psi].
                - hessian (numpy.ndarray): The second-order partial derivatives (Hessian matrix) of shape [3, 3].
        """
        # Create input tensor with proper gradient tracking
        # Assuming x_ji, y_ji, psi_ji are scalars or single-element tensors
        inputs = torch.tensor(
            [x_ji, y_ji, psi_ji], dtype=torch.float32, requires_grad=True
        )

        # Compute center-to-center distance d = sqrt(x^2 + y^2)
        d = torch.sqrt(inputs[0] ** 2 + inputs[1] ** 2)
        sm = (
            d - 2 * self.radius
        )  # Safety margin equals c2c distance minus two times radius

        # First-order derivatives: grad_d = [∂d/∂x, ∂d/∂y, ∂d/∂psi]
        grad_d_1st_order = torch.autograd.grad(
            outputs=d,
            inputs=inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Second-order derivatives (Hessian matrix)
        second_derivatives = []
        for i in range(inputs.size(0)):
            # Compute second derivative of d w.r.t. each input dimension
            grad2 = torch.autograd.grad(
                outputs=grad_d_1st_order[i],
                inputs=inputs,
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
            )[0]
            second_derivatives.append(grad2)

        # Stack second derivatives to form the Hessian matrix (shape [3, 3])
        hessian = torch.stack(
            second_derivatives
        )  # Each row corresponds to d^2 d / d input_i d input_j

        # Validation (consider numerical inaccuracies)
        assert torch.allclose(grad_d_1st_order[0], x_ji / d, rtol=1e-3, atol=1e-3)

        assert torch.allclose(grad_d_1st_order[1], y_ji / d, rtol=1e-3, atol=1e-3)

        assert torch.allclose(hessian[1, 1], x_ji**2 / d**3, rtol=1e-3, atol=1e-3)

        assert torch.allclose(hessian[0, 0], y_ji**2 / d**3, rtol=1e-3, atol=1e-3)

        assert torch.allclose(
            hessian[0, 1], -x_ji * y_ji / d**3, rtol=1e-3, atol=1e-3
        )

        # Detach gradients to prevent further computation and convert to numpy
        return (
            sm.detach().item(),
            grad_d_1st_order.detach().numpy(),
            hessian.detach().numpy(),
        )

    def mtv_based_sm(self, x_ji, y_ji, psi_ji):
        """
        Minimum Translation Vector (MTV)-based safety margin.

        Computes the NN output along with its first and second-order partial derivatives
        with respect to the inputs x_ji, y_ji, and psi_ji.
        """
        feature_norm = self.SME.feature_normalizer.squeeze(0)
        label_norm = self.SME.label_normalizer

        psi_ji = (psi_ji + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi]

        # Create input tensor with proper gradient tracking
        # Assuming x_ji, y_ji, psi_ji are scalars or single-element tensors
        inputs = torch.tensor(
            [x_ji, y_ji, psi_ji], dtype=torch.float32, device=self.device
        )
        normalized_inputs = inputs / feature_norm
        normalized_inputs.requires_grad_(True)  # Enable gradient tracking

        # Forward pass through the neural network
        sm_normalized = self.SME.net(normalized_inputs)  # Assuming scalar output

        # Denormalize the output
        sm_predicted = sm_normalized * label_norm  # Scalar

        # First-order derivatives
        grad_sm_1st_order = torch.autograd.grad(
            sm_normalized,
            normalized_inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Denormalize the first-order gradients
        grad_sm_predicted = (grad_sm_1st_order * label_norm) / feature_norm

        # Second-order derivatives (Hessian)
        second_derivatives = []
        for i in range(normalized_inputs.size(0)):
            # Compute second derivative of sm_normalized w.r.t. each input dimension
            grad2 = torch.autograd.grad(
                grad_sm_1st_order[i],
                normalized_inputs,
                retain_graph=True,
                create_graph=True,
                only_inputs=True,
            )[0]
            # Denormalize the second derivatives
            grad2_denorm = (grad2 * label_norm) / (feature_norm[i] ** 2)
            second_derivatives.append(grad2_denorm)

        # Stack second derivatives to form the Hessian matrix (shape [3, 3])
        hessian_predicted = torch.stack(
            second_derivatives
        )  # Each row corresponds to d^2 sm / d input_i d input_j

        # Detach gradients to prevent further computation
        sm_predicted = sm_predicted.detach().item()
        grad_sm_predicted = grad_sm_predicted.detach().numpy()
        hessian_predicted = hessian_predicted.detach().numpy()

        return sm_predicted, grad_sm_predicted, hessian_predicted

    def compute_cbf_conditions(
        self, dstate_time, ddstate_time, sm, grad_sm, hessian_sm
    ):
        """
        Compute the first and second order Control Barrier Function (CBF) conditions.

        Args:
            grad_sm (np.ndarray): Gradient of sm w.r.t inputs [dsm_dxji, dsm_dyji, dsm_dpsiji].
            state_derivatives_i (torch.Tensor): Derivatives of state_i [dx_i, dy_i, dpsi_i, dv_i, ddelta_i].
            state_derivatives_j (torch.Tensor): Derivatives of state_j [dx_j, dy_j, dpsi_j, dv_j, ddelta_j].

        Returns:
            tuple: h, dot_h, ddot_h, cbf_condition_1, cbf_condition_2
        """
        h = sm - self.safety_buffer

        # Compute dot_h
        dot_h = grad_sm @ dstate_time

        # Compute ddot_h
        ddot_h = grad_sm @ ddstate_time + dstate_time.T @ hessian_sm @ dstate_time

        # First-order CBF condition
        cbf_condition_1 = dot_h + self.lambda_cbf * h

        # Second-order CBF condition
        cbf_condition_2 = (
            ddot_h + 2 * self.lambda_cbf * dot_h + self.lambda_cbf**2 * h
        )

        if self.step == 1:
            # The first-order CBF condition and the CBF value at the first step should be greater than 0 to use the forward invariance property of CBF
            assert cbf_condition_1 > 0
            assert h > 0

        return h, dot_h, ddot_h, cbf_condition_1, cbf_condition_2

    def compute_state_time_derivatives(self, u_i, u_j):
        """
        Compute first and second order time derivatives of the relative states between vehicles i and j using global coordinate system.

        Args:
            u_i: Control inputs [acceleration, steering rate] for vehicle i
            u_j: Control inputs [acceleration, steering rate] for vehicle j

        Returns:
            dstate_time_ji: First derivatives of relative states j wrt i [dx, dy, dpsi]
            ddstate_time_ji: Second derivatives of relative states j wrt i [ddx, ddy, ddpsi]
            dstate_time_ij: First derivatives of relative states i wrt j [dx, dy, dpsi]
            ddstate_time_ij: Second derivatives of relative states i wrt j [ddx, ddy, ddpsi]
        """
        # Compute state time derivatives
        self.dstate_time_i = self.kbm.ode(
            None, self.state_i, self.u_placeholder
        )  # Since the time derivatives of x, y, and heading only depend on the states, we can simply ignore actions
        self.dstate_time_j = self.kbm.ode(None, self.state_j, self.u_placeholder)

        dstate_time_ji_global = (self.dstate_time_j - self.dstate_time_i)[0:3].numpy()

        ddx_i, ddy_i, ddpsi_i = self.compute_dstate_2nd_time(
            u_i, self.state_i, self.dstate_time_i
        )
        ddx_j, ddy_j, ddpsi_j = self.compute_dstate_2nd_time(
            u_j, self.state_j, self.dstate_time_j
        )

        [dx_ji_global, dy_ji_global, dpsi_ji_global] = dstate_time_ji_global
        # Compute relative second derivatives
        ddx_ji_global = ddx_j - ddx_i
        ddy_ji_global = ddy_j - ddy_i
        ddpsi_ji_global = ddpsi_j - ddpsi_i

        # Convert from the global coordinate system to the ego coordinate system
        cos_psi_i = np.cos(self.state_i[2].item())
        sin_psi_i = np.sin(self.state_i[2].item())

        d_psi_i = self.dstate_time_i[2]

        x_ji_global, y_ji_global, psi_ji_global = (
            self.state_j[0:3] - self.state_i[0:3]
        ).numpy()

        # Relative states in the ego coordinate system
        x_ji_ego = cos_psi_i * x_ji_global + sin_psi_i * y_ji_global  # Eq. (1)
        y_ji_ego = -sin_psi_i * x_ji_global + cos_psi_i * y_ji_global  # Eq. (2)

        # Take time derivatives of Eq. (1)
        dx_ji_ego = (
            cos_psi_i * dx_ji_global
            - x_ji_global * sin_psi_i * d_psi_i
            + sin_psi_i * dy_ji_global
            + y_ji_global * cos_psi_i * d_psi_i
        )  # Eq. (3)

        # Take time derivatives of Eq. (2)
        dy_ji_ego = (
            cos_psi_i * dy_ji_global
            - y_ji_global * sin_psi_i * d_psi_i
            - sin_psi_i * dx_ji_global
            - x_ji_global * cos_psi_i * d_psi_i
        )  # Eq. (4)

        # Take time derivatives of Eq. (3)
        ddx_ji_ego = (
            cos_psi_i * ddx_ji_global
            - 2 * dx_ji_global * sin_psi_i * d_psi_i
            - x_ji_global * cos_psi_i * d_psi_i**2
            - x_ji_global * sin_psi_i * ddpsi_i
            + sin_psi_i * ddy_ji_global
            + 2 * dy_ji_global * cos_psi_i * d_psi_i
            - y_ji_global * sin_psi_i * d_psi_i**2
            + y_ji_global * cos_psi_i * ddpsi_i
        )  # Eq. (5)

        # Take time derivatives of Eq. (4)
        ddy_ji_ego = (
            cos_psi_i * ddy_ji_global
            - 2 * dy_ji_global * sin_psi_i * d_psi_i
            - y_ji_global * cos_psi_i * d_psi_i**2
            - y_ji_global * sin_psi_i * ddpsi_i
            - sin_psi_i * ddx_ji_global
            - 2 * dx_ji_global * cos_psi_i * d_psi_i
            + x_ji_global * sin_psi_i * d_psi_i**2
            - x_ji_global * cos_psi_i * ddpsi_i
        )  # Eq. (6)

        dpsi_ji_ego = dpsi_ji_global
        ddpsi_ji_ego = ddpsi_ji_global

        dstate_time_ji_ego = np.array([dx_ji_ego, dy_ji_ego, dpsi_ji_ego])
        ddstate_time_ji_ego = [ddx_ji_ego, ddy_ji_ego, ddpsi_ji_ego]

        # Add negative sign to get the time derivatives of the relative states in another vehicle's ego perspective
        dstate_time_ij_ego = -dstate_time_ji_ego
        ddstate_time_ij_ego = [-expr for expr in ddstate_time_ji_ego]

        return (
            dstate_time_ji_ego,
            ddstate_time_ji_ego,
            dstate_time_ij_ego,
            ddstate_time_ij_ego,
        )

    def compute_dstate_2nd_time(self, u, state, dstate_time):
        u_1, u_2 = u  # Acceleration and steering rate

        # Compute beta and its derivative
        k = self.l_r / self.l_wb
        dpsi = dstate_time[2]
        tan_delta = np.tan(state[4].item())
        beta = np.arctan(k * tan_delta)
        sec_delta_sq = 1 / np.cos(state[4].item()) ** 2
        tan_beta = k * tan_delta
        cos_beta = 1 / np.sqrt(1 + tan_beta**2)
        sin_beta = tan_beta * cos_beta

        dbeta = (k * sec_delta_sq * u_2) / (1 + (k * tan_delta) ** 2)

        # Compute second derivatives
        ddx = u_1 * np.cos(state[2].item() + beta) - state[3].item() * np.sin(
            state[2].item() + beta
        ) * (dpsi + dbeta)
        ddy = u_1 * np.sin(state[2].item() + beta) + state[3].item() * np.cos(
            state[2].item() + beta
        ) * (dpsi + dbeta)

        ddpsi = (
            (u_1 / self.l_wb) * cos_beta * tan_delta
            + (state[3].item() / self.l_wb) * cos_beta * sec_delta_sq * u_2
            - (state[3].item() / self.l_wb) * sin_beta * tan_delta * dbeta
        )

        return ddx, ddy, ddpsi

    def eval_cbf_conditions(
        self,
        dstate_time_relative_opt,
        ddstate_time_relative_opt,
        x_relative_opt,
        y_relative_opt,
        psi_relative_opt,
    ):
        """
        Substitute the optimal control values to evaluate the second-order CBF condition.

        Args:
            u_i_opt (np.ndarray): Optimal control inputs [acceleration, steering rate].
            h (torch.Tensor): Control Barrier Function value.
            dot_h (float): First derivative of CBF.
            grad_sm (np.ndarray): Gradient of sm.

        Returns:
            tuple: ddot_h_opt, cbf_condition_2_opt
        """

        sm_opt, grad_sm_opt, hessian_sm_opt = self.estimate_safety_margin(
            x_relative_opt, y_relative_opt, psi_relative_opt
        )
        (
            h_opt,
            dot_h_opt,
            ddot_h_opt,
            cbf_condition_1_opt,
            cbf_condition_2_opt,
        ) = self.compute_cbf_conditions(
            dstate_time_relative_opt,
            ddstate_time_relative_opt,
            sm_opt,
            grad_sm_opt,
            hessian_sm_opt,
        )

        return (
            h_opt,
            dot_h_opt,
            ddot_h_opt,
            cbf_condition_1_opt,
            cbf_condition_2_opt,
        )

    def generate_reference_path(self, cur_pos, orig_pos, goal_pos, agent_idx):
        """
        Generate a fixed number of points along a reference path starting from the agent's current position
        and pointing towards the goal. The points are spaced at a fixed distance apart.
        """
        if self.scenario_type.lower() == "overtaking":
            path_points = torch.zeros(
                (self.rl_n_points_ref, 2), device=self.device, dtype=torch.float32
            )
            if agent_idx == 0:
                # Agent i
                ref_points_x = (
                    self.state_i[0]
                    + torch.arange(1, self.rl_n_points_ref + 1, device=self.device)
                    * self.rl_distance_between_points_ref_path
                )
                if self.is_overtake:
                    # Switch reference path to another lane to encourage overtaking
                    # Switch to target lane
                    ref_points_y = (
                        self.y_lane_1
                        if self.overtake_target_lane == "1"
                        else self.y_lane_2
                    )
                else:
                    # Stay in the current lane
                    ref_points_y = (
                        self.y_lane_1 if self.lane_i == "1" else self.y_lane_2
                    )
            else:
                # Agent j
                ref_points_x = (
                    self.state_j[0]
                    + torch.arange(1, self.rl_n_points_ref + 1, device=self.device)
                    * self.rl_distance_between_points_ref_path
                )
                if self.is_obstructe:
                    print("Agent j is obstructing agent i!")
                    # Switch to the target lane of agent i
                    ref_points_y = (
                        self.y_lane_1
                        if self.obstructe_target_lane == "1"
                        else self.y_lane_2
                    )
                else:
                    # Stay in the current lane
                    ref_points_y = (
                        self.y_lane_1 if self.lane_j == "1" else self.y_lane_2
                    )
            path_points[:, 0] = ref_points_x
            path_points[:, 1] = ref_points_y
        else:
            # Adjust the goal to encourage evasion
            if (self.step >= self.evasion_step_start) and (
                self.step <= self.evasion_step_end
            ):
                if agent_idx == 0:
                    goal_pos[1] += self.evasive_offset  # Introduce a small perturbation
                    orig_pos[1] += self.evasive_offset
                else:
                    goal_pos[1] -= self.evasive_offset  # Introduce a small perturbation
                    orig_pos[1] -= self.evasive_offset

            direction = goal_pos - cur_pos

            # Normalize the direction vector
            direction_norm = direction / direction.norm(dim=-1, keepdim=True)

            # Create a range of distances for the points along the path
            distances = (
                torch.arange(1, self.rl_n_points_ref + 1, device=self.device).view(
                    -1, 1
                )
                * self.rl_distance_between_points_ref_path
            )

            # Generate the points along the path
            path_points = cur_pos + direction_norm * distances

            path_points = self.project_to_line(path_points, goal_pos, orig_pos)

        return path_points

    @staticmethod
    def project_to_line(path_points, goal_pos, orig_pos):
        # Project points to the line connecting the initial position and the goal
        direction_orig = goal_pos - orig_pos

        # Normalize the direction vectors to unit vectors
        direction_norm_orig = direction_orig / direction_orig.norm(dim=-1, keepdim=True)

        # Calculate vectors from original positions to the short-term reference points
        vectors_to_points = path_points - orig_pos

        # Project the vectors_to_points onto the direction_norm_orig
        proj_length = (
            (vectors_to_points * direction_norm_orig).sum(dim=-1).unsqueeze(-1)
        )
        projected_points = orig_pos + proj_length * direction_norm_orig

        return projected_points

    @staticmethod
    def observe_ego_view(pos: Tensor, heading: Tensor, obs: Tensor):
        vec = obs - pos
        vec_rotation = torch.atan2(vec[:, 1], vec[:, 0])
        vec_norm = vec.norm(dim=-1)
        rot_relative = vec_rotation - heading

        obs_ego = torch.vstack(
            [vec_norm * torch.cos(rot_relative), vec_norm * torch.sin(rot_relative)]
        ).transpose(dim0=1, dim1=0)

        return obs_ego

    def update(self, frame):
        """
        Update function for each frame of the animation.

        Args:
            frame (int): Current frame number.

        Returns:
            list: Updated plot elements.
        """
        self.list_time.append(self.step * self.dt)
        self.step += 1
        print(
            f"------------Step: {self.step}--Time: {frame * self.dt:.2f}s------------"
        )

        if self.scenario_type.lower() == "overtaking":
            # Update flags such as lane, overtaking, obstructing, etc., for the overtaking scenario
            self.update_flags_prior()

        # RL actions for agent i as its nominal actions
        obs_i, self.ref_points_i, self.ref_points_ego_view_i = self.observation(
            self.state_i,
            self.list_state_i[0][0:2],
            self.goal_i.clone(),
            agent_idx=0,
        )
        self.tensordict_i.set(
            self.rl_observation_key, obs_i.unsqueeze(0)
        )  # Update tensordict for later policy call
        self.rl_actions_i = (  # Get nominal control inputs
            self.nomi_cont_i(self.tensordict_i)
            .get(self.rl_action_key)
            .squeeze(0)
            .detach()
        )  # Speed and steering

        u_nominal_i = self.rl_acrion_to_u(
            self.rl_actions_i, self.state_i[3], self.state_i[4]
        )  # Acceleration and steering rate

        # RL actions for agent j as its nominal actions
        obs_j, self.ref_points_j, self.ref_points_ego_view_j = self.observation(
            self.state_j,
            self.list_state_j[0][0:2],
            self.goal_j.clone(),
            agent_idx=1,
        )
        self.tensordict_j.set(
            self.rl_observation_key, obs_j.unsqueeze(0)
        )  # Update tensordict for later policy call
        self.rl_actions_j = (  # Get nominal control inputs
            self.nomi_cont_j(self.tensordict_j)
            .get(self.rl_action_key)
            .squeeze(0)
            .detach()
        )
        if self.scenario_type.lower() == "overtaking":
            # Purposely lowers the speed of agent j
            self.rl_actions_j[0] = torch.clamp(self.rl_actions_j[0], 0, self.v_max / 2)
        u_nominal_j = self.rl_acrion_to_u(
            self.rl_actions_j, self.state_j[3], self.state_j[4]
        )

        # Initialize optimization variables for vehicle i and possibly j
        u_i = cp.Variable(2)
        if self.scenario_type.lower() == "overtaking":
            u_j = u_nominal_j
        else:
            u_j = cp.Variable(2)

        # State derivative to time in global coordinate system
        (
            dstate_time_ji,
            ddstate_time_ji,
            dstate_time_ij,
            ddstate_time_ij,
        ) = self.compute_state_time_derivatives(u_i, u_j)

        # Compute relative poses between vehicles i and j in ego coordinate system
        x_ji, y_ji, psi_ji, _ = self.compute_relative_poses(
            self.state_i[0],
            self.state_i[1],
            self.state_i[2],
            self.state_j[0],
            self.state_j[1],
            self.state_j[2],
            self.sm_type,
        )
        # Estimate safety margin and gradients
        sm_ji, grad_sm_ji, hessian_sm_ji = self.estimate_safety_margin(
            x_ji, y_ji, psi_ji
        )

        # Compute CBF conditions for the optimization problem
        (
            h_ji,
            dot_h_ji,
            ddot_h_ji,
            cbf_condition_1_ji,
            cbf_condition_2_ji,
        ) = self.compute_cbf_conditions(
            dstate_time_ji, ddstate_time_ji, sm_ji, grad_sm_ji, hessian_sm_ji
        )

        if self.scenario_type.lower() == "overtaking":
            # Objective: Minimize weighted squared deviation from nominal control inputs
            objective = cp.Minimize(cp.quad_form(u_i - u_nominal_i, self.Q))
            constraints = [
                cbf_condition_2_ji >= 0,
                self.a_min <= u_i[0],
                u_i[0] <= self.a_max,
                self.steering_rate_min <= u_i[1],
                u_i[1] <= self.steering_rate_max,
            ]

        else:
            x_ij, y_ij, psi_ij, _ = self.compute_relative_poses(
                self.state_j[0],
                self.state_j[1],
                self.state_j[2],
                self.state_i[0],
                self.state_i[1],
                self.state_i[2],
                self.sm_type,
            )
            sm_ij, grad_sm_ij, hessian_sm_ij = self.estimate_safety_margin(
                x_ij, y_ij, psi_ij
            )
            (
                h_ij,
                dot_h_ij,
                ddot_h_ij,
                cbf_condition_1_ij,
                cbf_condition_2_ij,
            ) = self.compute_cbf_conditions(
                dstate_time_ij, ddstate_time_ij, sm_ij, grad_sm_ij, hessian_sm_ij
            )

            # Objective: Minimize weighted squared deviation from nominal control inputs
            objective = cp.Minimize(
                cp.quad_form(u_i - u_nominal_i, self.Q)
                + cp.quad_form(u_j - u_nominal_j, self.Q)
            )
            constraints = [
                cbf_condition_2_ji >= 0,
                self.a_min <= u_i[0],
                u_i[0] <= self.a_max,
                self.steering_rate_min <= u_i[1],
                u_i[1] <= self.steering_rate_max,
                # cbf_condition_2_ij >= 0,  # This is unnecessary because it is the same as cbf_condition_2_ji
                self.a_min <= u_j[0],
                u_j[0] <= self.a_max,
                self.steering_rate_min <= u_j[1],
                u_j[1] <= self.steering_rate_max,
            ]

        # Solve QP to get optimal control inputs for vehicle i
        # Formulate and solve the QP with custom solver settings
        prob = cp.Problem(objective, constraints)

        t_start = time.time()
        prob.solve(
            solver=cp.OSQP,  # DCP, DQCP
            verbose=False,  # Set to True for solver details
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=1000,
        )

        opt_duration = time.time() - t_start
        self.list_opt_duration.append(opt_duration)

        # print(f"Cost: {prob.value:.4f}")
        if self.scenario_type.lower() == "overtaking":
            if prob.status != cp.OPTIMAL:
                print(f"Warning: QP not solved optimally. Status: {prob.status}")
                u_i_opt = u_nominal_i
            else:
                u_i_opt = u_i.value

            u_j_opt = u_nominal_j
            (
                dstate_time_ji_opt,
                ddstate_time_ji_opt,
                _,
                _,
            ) = self.compute_state_time_derivatives(u_i_opt, u_j_opt)

            # Recompute CBF conditions with actual control actions (just for debugging)
            (
                h_ji_opt,
                dot_h_ji_opt,
                ddot_h_ji_opt,
                cbf_condition_1_ji_opt,
                cbf_condition_2_ji_opt,
            ) = self.eval_cbf_conditions(
                dstate_time_ji_opt, ddstate_time_ji_opt, x_ji, y_ji, psi_ji
            )

            # Append to lists
            if self.add_longitudinal_safety_buffer:
                self.list_h_ji.append(h_ji_opt + self.length / 4)
            else:
                self.list_h_ji.append(h_ji_opt)
            self.list_dot_h_ji.append(dot_h_ji_opt)
            self.list_ddot_h_ji.append(ddot_h_ji_opt)
            self.list_cbf_condition_1_ji.append(cbf_condition_1_ji_opt)
            self.list_cbf_condition_2_ji.append(cbf_condition_2_ji_opt)

            # The variables on the left side should be the same as the ones on the right side as they do not depend on the control actions
            assert h_ji_opt == h_ji
            assert dot_h_ji_opt == dot_h_ji
            assert cbf_condition_1_ji_opt == cbf_condition_1_ji

        else:
            if prob.status != cp.OPTIMAL:
                print(f"Warning: QP not solved optimally. Status: {prob.status}")
                u_i_opt = u_nominal_i
                u_j_opt = u_nominal_j
            else:
                u_i_opt = u_i.value
                u_j_opt = u_j.value

            # Recompute CBF conditions with actual control actions
            (
                dstate_time_ji_opt,
                ddstate_time_ji_opt,
                dstate_time_ij_opt,
                ddstate_time_ij_opt,
            ) = self.compute_state_time_derivatives(u_i_opt, u_j_opt)

            # Recompute CBF conditions with actual control actions
            (
                h_ji_opt,
                dot_h_ji_opt,
                ddot_h_ji_opt,
                cbf_condition_1_ji_opt,
                cbf_condition_2_ji_opt,
            ) = self.eval_cbf_conditions(
                dstate_time_ji_opt, ddstate_time_ji_opt, x_ji, y_ji, psi_ji
            )
            (
                h_ij_opt,
                dot_h_ij_opt,
                ddot_h_ij_opt,
                cbf_condition_1_ij_opt,
                cbf_condition_2_ij_opt,
            ) = self.eval_cbf_conditions(
                dstate_time_ij_opt, ddstate_time_ij_opt, x_ij, y_ij, psi_ij
            )

            # Append to lists for later plotting
            self.list_h_ji.append(h_ji_opt)
            self.list_dot_h_ji.append(dot_h_ji_opt)
            self.list_ddot_h_ji.append(ddot_h_ji_opt)
            self.list_cbf_condition_1_ji.append(cbf_condition_1_ji_opt)
            self.list_cbf_condition_2_ji.append(cbf_condition_2_ji_opt)
            self.list_h_ij.append(h_ij_opt)
            self.list_dot_h_ij.append(dot_h_ij_opt)
            self.list_ddot_h_ij.append(ddot_h_ij_opt)
            self.list_cbf_condition_1_ij.append(cbf_condition_1_ij_opt)
            self.list_cbf_condition_2_ij.append(cbf_condition_2_ij_opt)

            # The variables on the left side should be the same as the ones on the right side as they do not depend on the control actions
            assert h_ji_opt == h_ji
            assert dot_h_ji_opt == dot_h_ji
            assert cbf_condition_1_ji_opt == cbf_condition_1_ji

            assert h_ij_opt == h_ij
            assert dot_h_ij_opt == dot_h_ij
            assert cbf_condition_1_ij_opt == cbf_condition_1_ij

        u_1_cost = (u_i_opt[0] - u_nominal_i[0]) ** 2 * self.Q[0, 0]
        u_2_cost = (u_i_opt[1] - u_nominal_i[1]) ** 2 * self.Q[1, 1]
        self.list_cost_acc.append(u_1_cost)
        self.list_cost_steer.append(u_2_cost)
        self.list_cost_total.append(u_1_cost + u_2_cost)

        if prob.status == cp.OPTIMAL:
            self.list_is_solve_success.append(1)  # 1: success
            assert abs(prob.value - (u_1_cost + u_2_cost)) < 1e-3
        else:
            self.list_is_solve_success.append(0)  # 0: fail

        # Save for later plotting
        self.target_v_i = self.state_i[3].item() + u_i_opt[0] * self.dt
        self.target_rotation_i = (
            self.state_i[4].item() + u_i_opt[1] * self.dt + self.state_i[2].item()
        )
        self.target_rotation_i = (self.target_rotation_i + np.pi) % (
            2 * np.pi
        ) - np.pi  # Normalize to [-pi, pi]
        self.target_v_j = self.state_j[3].item() + u_j_opt[0] * self.dt
        self.target_rotation_j = (
            self.state_j[4].item() + u_j_opt[1] * self.dt + self.state_j[2].item()
        )
        self.target_rotation_j = (self.target_rotation_j + np.pi) % (
            2 * np.pi
        ) - np.pi  # Normalize to [-pi, pi]

        # Update state
        self.state_i, _, _ = self.kbm.step(
            self.state_i.clone(),
            torch.tensor(u_i_opt, device=self.device, dtype=torch.float32),
            self.dt,
        )
        self.list_state_i.append(self.state_i.clone().numpy())

        self.state_j, _, _ = self.kbm.step(
            self.state_j.clone(),
            torch.tensor(u_j_opt, device=self.device, dtype=torch.float32),
            self.dt,
        )
        self.list_state_j.append(self.state_j.clone().numpy())

        # Compute rectangles for later plotting
        self.compute_rectangles()

        # Update success and fail flags for the overtaking scenario
        if self.scenario_type.lower() == "overtaking":
            self.update_flags_posterior()

        # Update plot elements including arrows
        self.update_plot_elements()

        if (self.step == self.num_steps + 1) and not self.is_save_video:
            print("Close the plot window to continue:")

        return [
            self.fig,
            self.ax1,
            self.ax2,
            self.vehicle_i_rect,
            self.vehicle_j_rect,
            self.visu_footprint_i,
            self.visu_footprint_j,
            self.ref_line_i,
            self.ref_line_j,
            self.visu_ref_points_i,
            self.visu_ref_points_j,
            self.nominal_action_arrow_i,
            self.nominal_action_arrow_j,
            self.cbf_action_arrow_i,
            self.cbf_action_arrow_j,
            self.visu_goal_point_i,
            self.visu_goal_point_j,
            self.visu_circle_i,
            self.visu_circle_j,
            self.visu_psi_0,
            self.visu_psi_1,
            self.visu_psi_2,
        ]

    def update_flags_prior(self):
        # If an agent is previously on a transition, it should finish this transition
        self.lane_i = "2" if self.state_i[1] <= self.y_lane_center_line else "1"
        self.lane_j = "2" if self.state_j[1] <= self.y_lane_center_line else "1"

        self.list_lane_i.append(self.lane_i)
        self.list_lane_j.append(self.lane_j)

        # An agent is considered within a lane if its CoG is inside a lane for at least a certain duration
        if len(self.list_lane_i) < self.within_lane_threshold:
            within_lane_1_i = False
            within_lane_2_i = False
        else:
            within_lane_1_i = all(
                lane == "1" for lane in self.list_lane_i[-self.within_lane_threshold :]
            )
            within_lane_2_i = all(
                lane == "2" for lane in self.list_lane_i[-self.within_lane_threshold :]
            )
        if len(self.list_lane_j) < self.within_lane_threshold:
            within_lane_1_j = False
            within_lane_2_j = False
        else:
            within_lane_1_j = all(
                lane == "1" for lane in self.list_lane_j[-self.within_lane_threshold :]
            )
            within_lane_2_j = all(
                lane == "2" for lane in self.list_lane_j[-self.within_lane_threshold :]
            )

        # Agent i will contibue overtaking if it was previously in an overtaking state, or it starts overtaking if the following conditions are satisfied
        # Condition 1: both agents are within same lane
        self.overtake_condition_1 = (within_lane_1_i and within_lane_1_j) or (
            within_lane_2_i and within_lane_2_j
        )
        # Condition 2: longitudinal distance is less than a threshold
        self.overtake_condition_2 = (
            self.state_i[0] - self.state_j[0]
        ).abs().item() < self.threshold_overtake
        if self.list_is_overtake[-1]:
            # Continue overtaking
            self.is_overtake = True
        else:
            self.is_overtake = self.overtake_condition_1 and self.overtake_condition_2

        if self.overtake_target_lane is None:
            self.overtake_target_lane = (
                "2" if (within_lane_1_i and within_lane_1_j) else "1"
            )

        # Before actual overtaking, purposely use a larger safety buffer to ensure enough longitudinal distance; when actual overtaking, switch to a smaller safety buffer, which equals the upper bound of the safety margin estimation error
        if (
            self.scenario_type.lower() == "overtaking"
            and self.sm_type.lower() == "mtv"
            and self.n_success_obstructe < self.obstructe_times_max
        ):
            self.safety_buffer = self.SME.error_upper_bound + self.length / 4
            self.add_longitudinal_safety_buffer = True
        else:
            self.safety_buffer = self.SME.error_upper_bound
            self.add_longitudinal_safety_buffer = False

        # Agent i will continue obstructing agent j if it was previously in an obstructing state, or it starts obstructing if the following conditions are satisfied
        # Condition 1: agent i is on an overtaking
        obstructe_condition_1 = self.is_overtake
        # Condition 2: agent j will not obstructe agent i if it has obstructed agent i for more than a certain number of times
        if self.n_success_obstructe >= self.obstructe_times_max:
            obstructe_condition_2 = False
        else:
            obstructe_condition_2 = True
        # Condition 3: their longitudinal distance is less than a threshold
        obstructe_condition_3 = (
            self.state_i[0] - self.state_j[0]
        ).abs().item() < self.threshold_obstructe
        if self.list_is_obstructe[-1]:
            # Continue obstructing
            self.is_obstructe = True
        else:
            self.is_obstructe = (
                obstructe_condition_1
                and obstructe_condition_2
                and obstructe_condition_3
            )
        if self.obstructe_target_lane is None:
            self.obstructe_target_lane = self.overtake_target_lane

        print(f"Lane of agent i: {self.lane_i}; Lane of agent j: {self.lane_j}")

        self.list_lane_i.append(self.lane_i)
        self.list_lane_j.append(self.lane_j)

    def update_flags_posterior(self):
        # Agent i successfully overtakes agent j if its longitudinal coordinate is larger than agent j plus a threshold
        self.success_overtake = (
            self.state_i[0] > self.state_j[0] + self.threshold_success_overtake
        )

        if self.success_overtake:
            print("Agent i successfully overtakes agent j!")
            self.fail_overtake = False
            self.is_overtake = False
            self.is_obstructe = False
            self.overtake_target_lane = None
            self.obstructe_target_lane = None
        else:
            # Agent i fails to overtake if the following conditions are satisfied
            # Condition 1: it used to be in an overtaking state at least for a certain duration
            fail_condition_1 = len(self.list_is_overtake) >= 10 and all(
                self.list_is_overtake[-10:]
            )
            # Condition 2: It is (again) on the same lane as agent j
            fail_condition_2 = self.overtake_condition_1
            self.fail_overtake = fail_condition_1 and fail_condition_2

            # Reset overtaking and obstructing flags if agent i fails
            if self.fail_overtake:
                print("Agent i fails to overtake agent j!")
                self.success_overtake = False
                self.is_overtake = False
                self.is_obstructe = False
                self.overtake_target_lane = None
                self.obstructe_target_lane = None
                self.n_success_obstructe += 1

        self.list_is_overtake.append(self.is_overtake)
        self.list_is_obstructe.append(self.is_obstructe)

    def compute_rectangles(self):
        """
        Compute the vertices of the rectangles for both agents.
        """
        rect_i_vertices = torch.tensor(
            self.SME.get_rectangle_vertices(
                self.state_i[0].item(),
                self.state_i[1].item(),
                self.state_i[2].item(),
            ),
            device=self.device,
            dtype=torch.float32,
        )
        rect_j_vertices = torch.tensor(
            self.SME.get_rectangle_vertices(
                self.state_j[0].item(),
                self.state_j[1].item(),
                self.state_j[2].item(),
            ),
            device=self.device,
            dtype=torch.float32,
        )
        vertices = torch.stack([rect_i_vertices, rect_j_vertices], dim=0).unsqueeze(0)

        actual_sm = get_distances_between_agents(
            vertices, distance_type="mtv", is_set_diagonal=False
        )[0, 0, 1].item()

        # Append for later plotting
        self.list_rectangles_i.append(rect_i_vertices)
        self.list_rectangles_j.append(rect_j_vertices)

    def observation(self, state, orig_pos, goal_pos, agent_idx):
        """
        Generate the observation for the RL agent.
        """
        if not isinstance(orig_pos, torch.Tensor):
            orig_pos = torch.tensor(orig_pos, device=self.device, dtype=torch.float32)

        # Store the generated path points
        ref_points = self.generate_reference_path(
            cur_pos=state[0:2],
            orig_pos=orig_pos,
            goal_pos=goal_pos,
            agent_idx=agent_idx,
        )

        # Observe short-term reference path using ego view
        ref_points_ego_view = self.observe_ego_view(state[0:2], state[2], ref_points)

        # Distance from the center of gravity (CG) of the agent to its reference path
        rl_distance_to_ref, _ = get_perpendicular_distances(
            point=state[0:2],
            polyline=torch.stack(
                [
                    orig_pos,
                    goal_pos,
                ],
                dim=0,
            ),
        )

        obs = torch.hstack(
            [
                state[3] / self.rl_normalizers.v,  # Speed
                state[4] / self.rl_normalizers.steering,  # Steering
                (ref_points_ego_view / self.rl_normalizers.pos).reshape(
                    -1
                ),  # Short-term reference path
                rl_distance_to_ref / self.rl_normalizers.distance_ref,
            ]
        )
        if self.rl_is_add_noise:
            # Add sensor noise if required
            obs = obs + (
                self.rl_noise_level
                * torch.rand_like(obs, device=self.device, dtype=torch.float32)
            )

        return obs, ref_points, ref_points_ego_view

    def rl_acrion_to_u(self, rl_actions, v, steering):
        """
        Convert from RL actions (speed and steering) to acceleration [m/s^2] and steering rate [rad/s] used in the kinematic bicycle model.
        """
        # Assume linear acceleration and steering change
        u_acc = (rl_actions[0] - v) / self.dt
        u_steering_rate = (rl_actions[1] - steering) / self.dt
        u_nominal = torch.tensor(
            [u_acc, u_steering_rate], device=self.device, dtype=torch.float32
        ).numpy()

        return u_nominal

    def setup_plot(self):
        """
        Set up the matplotlib figure and plot elements for visualization.

        Args:
            params (dict): Simulation parameters.

        Returns:
            tuple: Tuple containing figure, axes, vehicle rectangles, path lines, and quivers.
        """
        xy_ratio = (self.plot_y_max - self.plot_y_min) / (
            self.plot_x_max - self.plot_x_min
        )

        if self.is_visu_cost:
            # The first figure for vehicles' movements, the second figure for CBF values, and the third figure for cost values
            x_fig_size = 20
            y_fig_size = 12
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(x_fig_size, y_fig_size))
        else:
            # The first figure for vehicles' movements, and the second figure for CBF values
            x_fig_size = 20
            y_fig_size = 8
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(x_fig_size, y_fig_size))

        ax1.set_xlim(self.plot_x_min, self.plot_x_max)
        ax1.set_ylim(self.plot_y_min, self.plot_y_max)
        ax1.set_aspect("equal")
        ax1.set_xlabel(r"$x$ [m]", fontsize=self.font_size_video)
        ax1.set_ylabel(r"$y$ [m]", fontsize=self.font_size_video)
        # Set tick size
        ax1.tick_params(axis="both", which="major", labelsize=self.font_size_video)
        xticks = np.arange(self.plot_x_min, self.plot_x_max, 0.4)
        xticks = np.round(xticks, decimals=1)
        ax1.set_xticks(xticks)
        ax1.set_yticks(
            [round(self.y_lane_bottom_bound, 2), 0, round(self.y_lane_top_bound, 2)]
        )
        plt.tight_layout()

        # Add static horizontal lines as lane boundaries
        ax1.axhline(
            y=self.y_lane_bottom_bound, color="k", linestyle="-", lw=1.0
        )  # Bottom lane boundary
        ax1.axhline(
            y=self.y_lane_center_line, color="k", linestyle="--", lw=1.0
        )  # Center line
        ax1.axhline(
            y=self.y_lane_top_bound, color="k", linestyle="-", lw=1.0
        )  # Top lane boundary

        # Initialize time
        if self.is_visu_time:
            visu_time = ax1.text(
                (self.plot_x_min + self.plot_x_max) / 2,
                self.plot_y_max - 0.05,
                f"Step: {self.step:.0f} (Time: {self.step*self.dt:.2f} s)",
                fontsize=self.font_size_video,
                ha="center",
            )
        else:
            visu_time = None

        # Initialize vehicle rectangles
        vehicle_i_rect = Rectangle(
            (self.state_i[0].item(), self.state_i[1].item()),
            self.length,
            self.width,
            angle=self.state_i[2].item(),
            color=self.color_i,
            alpha=0.5,
            label=r"Vehicle $i$",
        )
        vehicle_j_rect = Rectangle(
            (self.state_j[0].item(), self.state_j[1].item()),
            self.length,
            self.width,
            angle=self.state_j[2].item(),
            color=self.color_j,
            alpha=0.5,
            label=r"Vehicle $j$",
        )
        ax1.add_patch(vehicle_i_rect)
        ax1.add_patch(vehicle_j_rect)

        # Initialize path lines
        (visu_footprint_i,) = ax1.plot(
            [],
            [],
            color=self.color_i,
            linestyle="-",
            lw=1.0,
            # label=r"Footprint $i$",
        )
        (visu_footprint_j,) = ax1.plot(
            [],
            [],
            color=self.color_j,
            linestyle="-",
            lw=1.0,
            # label=r"Footprint $j$",
        )

        # Initialize points for short-term reference path
        if self.is_visu_ref_path:
            # Initialize polyline for short-term reference path
            (ref_line_i,) = ax1.plot(
                [], [], color=self.color_i, linestyle="-", lw=0.5, alpha=0.5
            )
            (ref_line_j,) = ax1.plot(
                [], [], color=self.color_j, linestyle="-", lw=0.5, alpha=0.5
            )
            (visu_ref_points_i,) = ax1.plot(
                [],
                [],
                color=self.color_i,
                marker="o",
                markersize=4,
                label=r"Short-term ref. points $i$",
            )
            (visu_ref_points_j,) = ax1.plot(
                [],
                [],
                color=self.color_j,
                marker="o",
                markersize=4,
                label=r"Short-term ref. points $j$",
            )
        else:
            ref_line_i = None
            ref_line_j = None
            visu_ref_points_i = None
            visu_ref_points_j = None

        # Initialize arrow to visualize nominal actions
        if self.is_visu_nominal_action:
            nominal_action_arrow_i = FancyArrowPatch(
                posA=(self.state_i[0].item(), self.state_i[1].item()),
                posB=(
                    self.state_i[0].item(),
                    self.state_i[1].item(),
                ),
                mutation_scale=10,
                mutation_aspect=1,
                color="black",
                alpha=0.2,
                label="Nominal actions",
            )
            ax1.add_patch(nominal_action_arrow_i)

            nominal_action_arrow_j = FancyArrowPatch(
                posA=(self.state_j[0].item(), self.state_j[1].item()),
                posB=(
                    self.state_j[0].item(),
                    self.state_j[1].item(),
                ),
                mutation_scale=10,
                mutation_aspect=1,
                color="black",
                alpha=0.2,
            )
            ax1.add_patch(nominal_action_arrow_j)

        else:
            nominal_action_arrow_i = None
            nominal_action_arrow_j = None

        # Initialize arrow to visualize CBF actions
        if self.is_visu_cbf_action:
            cbf_action_arrow_i = FancyArrowPatch(
                posA=(self.state_i[0].item(), self.state_i[1].item()),
                posB=(
                    self.state_i[0].item(),
                    self.state_i[1].item(),
                ),
                mutation_scale=10,
                mutation_aspect=1,
                color="black",
                alpha=0.8,
                label="CBF actions",
            )
            ax1.add_patch(cbf_action_arrow_i)

            if self.scenario_type.lower() == "bypassing":
                cbf_action_arrow_j = FancyArrowPatch(
                    posA=(self.state_j[0].item(), self.state_j[1].item()),
                    posB=(
                        self.state_j[0].item(),
                        self.state_j[1].item(),
                    ),
                    mutation_scale=10,
                    mutation_aspect=1,
                    color="black",
                    alpha=0.8,
                    # label="CBF action",
                )
                ax1.add_patch(cbf_action_arrow_j)
            else:
                cbf_action_arrow_j = None
        else:
            cbf_action_arrow_i = None
            cbf_action_arrow_j = None

        # Initialize point for goal
        (visu_goal_point_i,) = ax1.plot(
            [],
            [],
            color=self.color_i,
            marker="o",
            markersize=8,
        )
        (visu_goal_point_j,) = ax1.plot(
            [],
            [],
            color=self.color_j,
            marker="o",
            markersize=8,
        )

        # Create Circle patches
        visu_circle_i = plt.Circle(
            (self.state_i[0].item(), self.state_i[1].item()),
            self.radius,
            color=self.color_i,
            linestyle="--",
            lw=1.0,
            fill=False,
        )
        visu_circle_j = plt.Circle(
            (self.state_j[0].item(), self.state_j[1].item()),
            self.radius,
            color=self.color_j,
            linestyle="--",
            lw=1.0,
            fill=False,
        )
        ax1.add_patch(visu_circle_i)
        ax1.add_patch(visu_circle_j)

        # Subfigure 2: CBF values
        ax2.set_xlabel(r"$t$ [s]", fontsize=self.font_size_video)
        ax2.set_ylabel("CBF and CBF conditions", fontsize=self.font_size_video)
        ax2.tick_params(axis="x", labelsize=self.font_size_video)
        ax2.tick_params(axis="y", labelsize=self.font_size_video)
        ax2.grid(True)
        ax2.set_xlim(0, self.total_time)
        if self.scenario_type.lower() == "overtaking":
            ax2.set_ylim((-0.2, 2.0))
        else:
            ax2.set_ylim((-0.2, 5.0))
        plt.tight_layout()

        # Visualize CBF values
        (visu_psi_0,) = ax2.plot(
            [],
            [],
            color=self.color_psi_0,
            lw=1.5,
            ls="-",
            label=r"$\Phi_{0}$ (safety margin)",
        )
        (visu_psi_1,) = ax2.plot(
            [],
            [],
            color=self.color_psi_1,
            lw=1.5,
            ls="-.",
            label=r"$\Phi_{1}$ (CBF condition 1)",
        )
        (visu_psi_2,) = ax2.plot(
            [],
            [],
            color=self.color_psi_2,
            lw=1.5,
            ls="--",
            label=r"$\Phi_{2}$ (CBF condition 2)",
        )
        ax1.legend(loc="upper left", fontsize=self.font_size_video)
        ax2.legend(loc="upper right", fontsize=self.font_size_video)

        # Subfigure 3: Cost values
        if self.is_visu_cost:
            ax3.set_xlabel(r"$t$ [s]", fontsize=self.font_size_video)
            ax3.set_ylabel("Cost", fontsize=self.font_size_video)
            ax3.tick_params(axis="x", labelsize=self.font_size_video)
            ax3.tick_params(axis="y", labelsize=self.font_size_video)
            ax3.grid(True)
            ax3.set_xlim(0, self.total_time)
            # ax3.set_ylim((-2, 50.0))
            plt.tight_layout()

            (visu_cost_acc,) = ax3.plot(
                [],
                [],
                color=self.color_cost_acc,
                lw=1.5,
                ls="--",
                label="Perturbation cost of nominal acceleration",
            )
            (visu_cost_steer,) = ax3.plot(
                [],
                [],
                color=self.color_cost_steer,
                lw=1.5,
                ls="-.",
                label="Perturbation cost of nominal steering rate",
            )
            (visu_cost_total,) = ax3.plot(
                [],
                [],
                color=self.color_cost_total,
                lw=1.5,
                ls="-",
                label="Total cost",
            )
            visu_is_solve_success = ax3.scatter(
                [],
                [],
                color="black",
                marker="x",
                label="Solver failed",
            )
            ax3.legend(loc="upper right", fontsize=self.font_size_video)
            ax3.set_ylim([-1.2, 50])
        else:
            ax3 = None
            visu_cost_acc = None
            visu_cost_steer = None
            visu_cost_total = None
            visu_is_solve_success = None

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.visu_time = visu_time
        self.vehicle_i_rect = vehicle_i_rect
        self.vehicle_j_rect = vehicle_j_rect
        self.visu_footprint_i = visu_footprint_i
        self.visu_footprint_j = visu_footprint_j
        self.ref_line_i = ref_line_i
        self.ref_line_j = ref_line_j
        self.visu_ref_points_i = visu_ref_points_i
        self.visu_ref_points_j = visu_ref_points_j
        self.nominal_action_arrow_i = nominal_action_arrow_i
        self.nominal_action_arrow_j = nominal_action_arrow_j
        self.cbf_action_arrow_i = cbf_action_arrow_i
        self.cbf_action_arrow_j = cbf_action_arrow_j
        self.visu_goal_point_i = visu_goal_point_i
        self.visu_goal_point_j = visu_goal_point_j
        self.visu_circle_i = visu_circle_i
        self.visu_circle_j = visu_circle_j
        self.visu_psi_0 = visu_psi_0
        self.visu_psi_1 = visu_psi_1
        self.visu_psi_2 = visu_psi_2
        self.visu_cost_acc = visu_cost_acc
        self.visu_cost_steer = visu_cost_steer
        self.visu_cost_total = visu_cost_total
        self.visu_is_solve_success = visu_is_solve_success

    def update_plot_elements(self):
        """
        Update the live plot elements.
        """
        # Update time
        if self.is_visu_time:
            self.visu_time.set_text(
                f"Step: {self.step:.0f} (Time: {self.step*self.dt:.2f} s)"
            )

        # Update vehicle rectangles
        for rect, state in zip(
            [self.vehicle_i_rect, self.vehicle_j_rect], [self.state_i, self.state_j]
        ):
            x, y, psi = state[0].item(), state[1].item(), state[2].item()
            cos_angle = np.cos(psi)
            sin_angle = np.sin(psi)
            bottom_left_x = (
                x - (self.length / 2) * cos_angle + (self.width / 2) * sin_angle
            )
            bottom_left_y = (
                y - (self.length / 2) * sin_angle - (self.width / 2) * cos_angle
            )
            rect.set_xy((bottom_left_x, bottom_left_y))
            rect.angle = np.degrees(psi)

        # Update footprints
        if self.is_visu_footprint:
            self.visu_footprint_i.set_data(
                [s[0] for s in self.list_state_i], [s[1] for s in self.list_state_i]
            )
            self.visu_footprint_j.set_data(
                [s[0] for s in self.list_state_j], [s[1] for s in self.list_state_j]
            )

        # Update short-term reference path
        if self.is_visu_ref_path:
            ref_x_i = self.ref_points_i[:, 0].numpy()
            ref_y_i = self.ref_points_i[:, 1].numpy()
            self.ref_line_i.set_data(ref_x_i, ref_y_i)
            # Update short-term reference path points
            self.visu_ref_points_i.set_data(ref_x_i, ref_y_i)

        # Update goal point
        goal_x_i = self.goal_i[0].item()
        goal_y_i = self.goal_i[1].item()
        self.visu_goal_point_i.set_data([goal_x_i], [goal_y_i])

        # Update short-term reference path
        if self.is_visu_ref_path:
            ref_x_j = self.ref_points_j[:, 0].numpy()
            ref_y_j = self.ref_points_j[:, 1].numpy()
            self.ref_line_j.set_data(ref_x_j, ref_y_j)
            # Update short-term reference path points
            self.visu_ref_points_j.set_data(ref_x_j, ref_y_j)

        # Update nominal action arrow
        if self.is_visu_nominal_action:
            arrow_length_i = self.rl_actions_i[0].item() / 10
            arror_angle_i = self.state_i[2].item() + self.rl_actions_i[1].item()
            self.nominal_action_arrow_i.set_positions(
                posA=(self.state_i[0].item(), self.state_i[1].item()),
                posB=(
                    self.state_i[0].item() + np.cos(arror_angle_i) * arrow_length_i,
                    self.state_i[1].item() + np.sin(arror_angle_i) * arrow_length_i,
                ),
            )

            arrow_length_j = self.rl_actions_j[0].item() / 8
            arror_angle_j = self.state_j[2].item() + self.rl_actions_j[1].item()
            self.nominal_action_arrow_j.set_positions(
                posA=(self.state_j[0].item(), self.state_j[1].item()),
                posB=(
                    self.state_j[0].item() + np.cos(arror_angle_j) * arrow_length_j,
                    self.state_j[1].item() + np.sin(arror_angle_j) * arrow_length_j,
                ),
            )

        # Update actual action arrow
        if self.is_visu_cbf_action:
            arrow_length_i = self.target_v_i / 10
            arror_angle_i = self.target_rotation_i
            self.cbf_action_arrow_i.set_positions(
                posA=(self.state_i[0].item(), self.state_i[1].item()),
                posB=(
                    self.state_i[0].item() + np.cos(arror_angle_i) * arrow_length_i,
                    self.state_i[1].item() + np.sin(arror_angle_i) * arrow_length_i,
                ),
            )

            if self.scenario_type.lower() == "bypassing":
                arrow_length_j = self.target_v_j / 8
                arror_angle_j = self.target_rotation_j
                self.cbf_action_arrow_j.set_positions(
                    posA=(self.state_j[0].item(), self.state_j[1].item()),
                    posB=(
                        self.state_j[0].item() + np.cos(arror_angle_j) * arrow_length_j,
                        self.state_j[1].item() + np.sin(arror_angle_j) * arrow_length_j,
                    ),
                )

        goal_x_j = self.goal_j[0].item()
        goal_y_j = self.goal_j[1].item()
        self.visu_goal_point_j.set_data([goal_x_j], [goal_y_j])

        # Update circles with a radius of self.radius for C2C safety margin visualization
        self.visu_circle_i.center = (self.state_i[0].item(), self.state_i[1].item())
        self.visu_circle_j.center = (self.state_j[0].item(), self.state_j[1].item())

        self.visu_psi_0.set_data(self.list_time, self.list_h_ji)
        self.visu_psi_1.set_data(self.list_time, self.list_cbf_condition_1_ji)
        self.visu_psi_2.set_data(self.list_time, self.list_cbf_condition_2_ji)

        if self.is_visu_cost:
            self.visu_cost_acc.set_data(self.list_time, self.list_cost_acc)
            self.visu_cost_steer.set_data(self.list_time, self.list_cost_steer)
            self.visu_cost_total.set_data(self.list_time, self.list_cost_total)
            # Plot only the time steps where the solve failed
            zero_indices = [
                i for i, val in enumerate(self.list_is_solve_success) if val == 0
            ]
            zero_times = [self.list_time[i] for i in zero_indices]
            zero_values = [self.list_is_solve_success[i] for i in zero_indices]
            self.visu_is_solve_success.set_offsets(np.c_[zero_times, zero_values])

    def plot_cbf_curve(self):
        """
        Plot the simulation results including:
        1. The trajectories of both agents represented by rectangles with fading colors
        2. CBF-related data including barrier functions and CBF conditions
        """
        # Create figure with 2x1 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), height_ratios=[1, 1])

        # Plot lane boundaries and center line
        ax1.axhline(
            y=self.y_lane_bottom_bound, color="k", linestyle="-", lw=1.0
        )  # Bottom lane boundary
        ax1.axhline(
            y=self.y_lane_center_line, color="gray", linestyle="--", lw=1.0, alpha=0.4
        )  # Center line
        ax1.axhline(
            y=self.y_lane_top_bound, color="k", linestyle="-", lw=1.0
        )  # Top lane boundary

        # Create a list of time steps to be visualized
        time_steps_to_visualize = np.arange(
            0, len(self.list_h_ji), self.visu_step_interval
        ).tolist()
        if self.scenario_type.lower() == "overtaking":
            # Include the final time step
            time_steps_to_visualize.append(len(self.list_h_ji) - 1)
        # Find the time step where the CBF value is minimum and add it to the time steps to be visualized
        min_cbf_value = min(self.list_h_ji)
        t_min_cbf_idx = self.list_h_ji.index(min_cbf_value)

        # Add the time step where the CBF value is minimum and delete the time step that is most near the time step where the CBF value is minimum
        near_min_cbf_idx = np.argmin(
            np.abs(np.array(time_steps_to_visualize) - t_min_cbf_idx)
        )
        time_steps_to_visualize.remove(time_steps_to_visualize[near_min_cbf_idx])

        time_steps_to_visualize.append(t_min_cbf_idx)
        # Rearrange time_steps_to_visualize in ascending order
        time_steps_to_visualize.sort()

        # Hide a fixed number of time texts for rectangles before the minimum CBF timestep
        if self.scenario_type.lower() == "overtaking":
            if self.sm_type.lower() == "c2c":
                n_time_texts_to_hide = 0
            else:
                n_time_texts_to_hide = 0
        else:
            if self.sm_type.lower() == "c2c":
                n_time_texts_to_hide = 3
            else:
                n_time_texts_to_hide = 2

        # Find the index of the time step to hide
        # find where is min_cbf_idx in time_steps_to_visualize
        min_cbf_idx = time_steps_to_visualize.index(t_min_cbf_idx)
        t_hide_text = time_steps_to_visualize[
            min_cbf_idx - n_time_texts_to_hide : min_cbf_idx
        ]

        # Plot rectangles for each timestep with fading colors
        n_steps = len(self.list_rectangles_i)
        for t_idx in range(len(time_steps_to_visualize)):
            t = time_steps_to_visualize[t_idx]
            # Calculate alpha (transparency) based on timestep
            alpha = 0.2 + 0.8 * (t / n_steps)

            alpha_text = np.clip(alpha * 1.5, 0.0, 1.0)

            # Get vertices for both agents at timestep t
            vertices_i = self.list_rectangles_i[t].numpy()
            vertices_j = self.list_rectangles_j[t].numpy()

            # Create polygon patches
            poly_i = plt.Polygon(
                vertices_i, facecolor=self.color_i, alpha=alpha, edgecolor="none"
            )
            poly_j = plt.Polygon(
                vertices_j, facecolor=self.color_j, alpha=alpha, edgecolor="none"
            )

            # Add patches to plot
            ax1.add_patch(poly_i)
            ax1.add_patch(poly_j)

            center_i = self.list_state_i[t][0:2]
            center_j = self.list_state_j[t][0:2]

            # Visualize circles with a radius of self.radius for C2C safety margin visualization
            linestyle = "-" if t == t_min_cbf_idx else "--"
            lw = 1.0 if t == t_min_cbf_idx else 1.0
            alpha = 0.5 if t == t_min_cbf_idx else 0.3

            for center, color in [(center_i, self.color_i), (center_j, self.color_j)]:
                circle = plt.Circle(
                    center,
                    self.radius,
                    color=color,
                    fill=False,
                    linestyle=linestyle,
                    lw=lw,
                    alpha=alpha,
                )
                ax1.add_patch(circle)

            if self.scenario_type.lower() == "overtaking":
                text_y_pos_i = self.y_lane_bottom_bound - 0.07
                text_y_pos_j = self.y_lane_top_bound + 0.05
                text_y_pos_i_0 = self.y_lane_bottom_bound - 0.068
                text_y_pos_j_0 = self.y_lane_top_bound + 0.05
            else:
                text_y_pos_i = self.y_lane_top_bound + 0.05
                text_y_pos_j = self.y_lane_bottom_bound - 0.07
                text_y_pos_i_0 = self.y_lane_top_bound + 0.05
                text_y_pos_j_0 = self.y_lane_bottom_bound - 0.068

            # Add timestep text near each rectangle
            if t == 0:
                ax1.text(
                    self.plot_x_min + 0.05,
                    text_y_pos_i_0,
                    "t:",
                    color=self.color_i,
                    horizontalalignment="left",
                    verticalalignment="center",
                    fontsize=9,
                )
                ax1.text(
                    self.plot_x_min + 0.05,
                    text_y_pos_j_0,
                    "t:",
                    color=self.color_j,
                    horizontalalignment="left",
                    verticalalignment="center",
                    fontsize=9,
                )

            # Text properties
            text_props = {
                "horizontalalignment": "center",
                "verticalalignment": "center",
                "alpha": alpha_text,
                "fontsize": 9,
            }

            # Add fontweight='bold' if this is the min CBF timestep
            if t == t_min_cbf_idx:
                text_props["fontweight"] = "bold"

            if t not in t_hide_text:
                # Add timestep text for both agents
                ax1.text(
                    center_i[0],
                    text_y_pos_i,
                    f"{t*self.dt:.1f} s",
                    color=self.color_i,
                    **text_props,
                )
                ax1.text(
                    center_j[0],
                    text_y_pos_j,
                    f"{t*self.dt:.1f} s",
                    color=self.color_j,
                    **text_props,
                )

        # Set equal aspect ratio and limits
        ax1.set_aspect("equal")
        ax1.set_xlim(self.plot_x_min, self.plot_x_max)
        ax1.set_ylim(self.plot_y_min, self.plot_y_max)
        xticks = np.arange(self.list_state_i[0][0], self.list_state_i[-1][0], 0.4)
        xticks = np.round(xticks, decimals=1)
        ax1.set_xticks(xticks)
        ax1.set_yticks(
            [round(self.y_lane_bottom_bound, 2), 0, round(self.y_lane_top_bound, 2)]
        )
        ax1.set_xlabel(r"$x$ [m]")
        ax1.set_ylabel(r"$y$ [m]")
        ax1.grid(False)

        # Second subplot: CBF data
        x_i = [s[0] for s in self.list_state_i]  # x-positions as x-axis
        # Time as x-axis
        self.list_time = np.arange(0, len(self.list_h_ji)) * self.dt
        # self.list_time = x_i[0:-1]

        # Plot CBF-related curves
        ax2.plot(
            self.list_time,
            self.list_h_ji,
            label=r"$\Psi_0$ ($h$)",
            linestyle="-",
            lw=2.0,
        )
        ax2.plot(
            self.list_time,
            self.list_cbf_condition_1_ji,
            label=r"$\Psi_1$",
            linestyle="--",
            lw=2.0,
        )
        ax2.plot(
            self.list_time,
            self.list_cbf_condition_2_ji,
            label=r"$\Psi_2$",
            linestyle="-.",
            lw=2.0,
        )

        ax2.set_xlabel(r"$t$ [s]")
        ax2.set_ylabel("CBF and CBF Conditions")
        ax2.legend(loc="upper right")
        ax2.grid(True)
        ax2.set_xlim(0, self.list_time[-1])
        if self.scenario_type.lower() == "overtaking":
            ax2.set_ylim((-0.2, 2.0))
        else:
            ax2.set_ylim((-0.2, 5.0))

        opt_time_str = (
            f"Mean optimization time per step: {np.mean(self.list_opt_duration):.4f} s."
        )
        print(opt_time_str)

        # Save computation time and possible maximum deviation to a txt file
        if self.is_save_eval_result:
            if self.scenario_type.lower() == "bypassing":
                # Calculate and display the maximum lateral distance each agent deviates from the center line
                y_i = [s[1] for s in self.list_state_i]
                y_j = [s[1] for s in self.list_state_j]
                max_deviation_i = max(abs(np.array(y_i) - self.y_lane_center_line))
                max_deviation_j = max(abs(np.array(y_j) - self.y_lane_center_line))
                print_str = (
                    f"Maximum deviation in y-direction with {self.sm_type.upper()}-based safety margin:"
                    f" Vehicle i: {max_deviation_i:.4f} m ({max_deviation_i / self.width * 100:.2f}%); "
                    f"Vehicle j: {max_deviation_j:.4f} m ({max_deviation_j / self.width * 100:.2f}%); "
                    f"Mean: {(max_deviation_i + max_deviation_j) / 2:.4f} m ({((max_deviation_i + max_deviation_j) / 2 / self.width * 100):.2f}%)"
                )
                print(print_str)
                file_name = f"eval_cbf_bypassing_{self.sm_type}.txt"
                with open(file_name, "w") as file:
                    file.write(print_str + "\n")
                    file.write(opt_time_str + "\n")
                print(f"A text file has been saved to {file_name}.")
            else:
                # Save optimization time to file
                file_name = f"eval_cbf_overtaking_{self.sm_type}.txt"
                with open(file_name, "w") as file:
                    file.write(opt_time_str + "\n")
                print(f"A text file has been saved to {file_name}.")

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 1])

        if self.is_save_eval_result:
            fig_name = f"eva_cbf_{self.scenario_type}_{self.sm_type}.pdf"
            plt.savefig(fig_name, bbox_inches="tight", dpi=450)
            print(f"A figure has been saved to {fig_name}.")
        else:
            plt.show()

    def run(self):
        """
        Run the simulation, display the animation, and save the video.
        """
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.num_steps,
            blit=False,  # Set to True to optimize the rendering by updating only parts of the frame that have changed (may rely heavily on the capabilities of the underlying GUI backend and the system’s graphical stack)
            interval=self.dt,
            repeat=False,
        )

        if self.is_save_video:
            # Save the animation as a video file
            video_name = f"eva_cbf_{self.scenario_type}_{self.sm_type}.mp4"
            ani.save(video_name, writer="ffmpeg", fps=30)
            print(f"A video has been saved to {video_name}.")
        else:
            plt.show()


def main(
    scenario_type: str,
    sm_type: str,
    is_save_video: bool = False,
    is_visu_ref_path: bool = False,
    is_visu_nominal_action: bool = True,
    is_visu_cbf_action: bool = True,
    is_visu_footprint: bool = True,
    is_visu_time: bool = True,
    is_visu_cost: bool = False,
    is_save_eval_result: bool = False,
):
    simulation = CBF(
        scenario_type=scenario_type,
        sm_type=sm_type,
        is_save_video=is_save_video,
        is_visu_ref_path=is_visu_ref_path,
        is_visu_nominal_action=is_visu_nominal_action,
        is_visu_cbf_action=is_visu_cbf_action,
        is_visu_footprint=is_visu_footprint,
        is_visu_time=is_visu_time,
        is_visu_cost=is_visu_cost,
        is_save_eval_result=is_save_eval_result,
    )
    simulation.run()
    simulation.plot_cbf_curve()


if __name__ == "__main__":
    main(
        scenario_type="overtaking",  # One of "overtaking" and "bypassing"
        sm_type="mtv",  # One of "c2c" and "mtv"
        is_save_video=False,  # If True, video will be saved without live visualization
        is_visu_ref_path=True,
        is_visu_footprint=True,
        is_visu_nominal_action=True,
        is_visu_cbf_action=True,
        is_visu_time=True,
        is_visu_cost=True,
        is_save_eval_result=True,
    )
