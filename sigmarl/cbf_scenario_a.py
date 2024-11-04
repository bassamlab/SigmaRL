# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from termcolor import cprint
import torch
from torch import Tensor

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

from tensordict.nn import TensorDictModule

from tensordict.nn.distributions import NormalParamExtractor

from tensordict.tensordict import TensorDict

import matplotlib

# matplotlib.use('TkAgg')  # or another interactive backend like 'Qt5Agg'

import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.patches import Rectangle
from matplotlib import animation

from sigmarl.distance_nn import SafetyMarginEstimatorModule

from sigmarl.dynamics import KinematicBicycleModel

from sigmarl.helper_scenario import (
    Normalizers,
    get_perpendicular_distances,
    get_distances_between_agents,
)

from sigmarl.constants import AGENTS

from sigmarl.colors import colors

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
    def __init__(self):
        # Initialize simulation parameters
        self.initialize_params()

        # Load safety margin estimator module
        self.load_safety_margin_estimator()
        self.d_min = (
            self.SME.error_upper_bound
        )  # Minimum safety distance (m), considering uncertainties such as safety-margin prediction errors

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

    def initialize_params(self):
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

        # Safety margin
        # Radius of the circle that convers the vehicle
        self.radius = np.sqrt(self.length**2 + self.width**2) / 2

        # On which lane is an agent on (1 for the top lane; 2 for the bottom lane; and 0 for a transition state between two lanes). Mainly used in the overtaking scenario.
        self.lane_i = None
        self.lane_j = None

        self.list_lane_i = []
        self.list_lane_j = []

        # y-coordinate of lane boundaries
        self.y_lane_top_bound = self.lane_width
        self.y_lane_center_line = 0
        self.y_lane_bottom_bound = -self.lane_width

        self.y_lane_1 = (self.y_lane_center_line + self.y_lane_top_bound) / 2
        self.y_lane_2 = (self.y_lane_center_line + self.y_lane_bottom_bound) / 2

        self.threshold_within_lane = self.width / 2
        self.threshold_overtake = (
            self.length
        )  # If agents' longitudinal distance is less than this value, one agent will overtake another agent, given they are within the same lane

        # Two scenarios are available:
        # (1) overtaking: the ego agent, controlled by a greedy RL policy with CBF verification,
        # needs to overtake its precede agent that moves slowly
        # (2) bypassing: two agents, both controlled by a greedy RL policy with CBF verification,
        # needs to bypass each other within a confined space
        self.scenario_type = "overtaking"  # One of "overtaking" and "bypassing"

        # Two types of safety margin: center-center-based safety margin, Minimum Translation Vector (MTV)-based safety margin
        self.sm_type = "mtv"  # One of "c2c" and "mtv"

        # CBF
        if self.scenario_type.lower() == "overtaking":
            # Set simulation duration
            self.total_time = 6.0  # Total simulation time
            self.num_steps = int(
                self.total_time / self.dt
            )  # Total simulation time steps

            # Initialize states ([x, y, psi, v, delta]) and goal states ([x, y])
            self.state_i = torch.tensor(
                [-1.2, self.y_lane_2, 0, 1.0, 0.0], dtype=torch.float32
            )
            self.state_j = torch.tensor(
                [0.2, self.y_lane_2, 0.0, 0.3, 0.0], dtype=torch.float32
            )
            self.goal_i = torch.tensor(
                [5, self.y_lane_2], device=self.device, dtype=torch.float32
            )
            self.goal_j = None

            self.switch_step = 20  # Time step to encourage one agent to switch lane to overtake another agent

            # Irrelevant
            self.evasion_step_start = None
            self.evasion_step_end = None
            self.evasive_offset = None

            if self.sm_type.lower() == "c2c":
                # Overtaking & C2C
                self.lambda_cbf = 3  # Design parameter for CBF
                self.w_acc = 1  # Weight for acceleration in QP
                self.w_steer = 1  # Weight for steering rate in QP
                self.Q = np.diag(
                    [self.w_acc, self.w_steer]
                )  # Weights for acceleration and steering rate
            else:
                # Overtaking & MTV
                self.lambda_cbf = 3  # Design parameter for CBF
                self.w_acc = 1  # Weight for acceleration in QP
                self.w_steer = 1  # Weight for steering rate in QP
                self.Q = np.diag(
                    [self.w_acc, self.w_steer]
                )  # Weights for acceleration and steering rate
        else:
            # Initialize states ([x, y, psi, v, delta]) and goal states ([x, y])
            # In the bypassing scenario, two agents are facing each other
            self.state_i = torch.tensor(
                [-1.2, self.y_lane_2, 0.0, 1.0, 0.0], dtype=torch.float32
            )
            self.state_j = torch.tensor(
                [1.2, self.y_lane_2, -np.pi, 1.0, 0.0], dtype=torch.float32
            )
            self.goal_i = self.state_j[0:2].clone()
            self.goal_i[0] += 2
            self.goal_j = self.state_i[0:2].clone()
            self.goal_j[0] -= 2

            self.switch_step = None

            if self.sm_type.lower() == "c2c":
                # Bypassing & C2C
                self.total_time = 3.0  # Total simulation time
                self.num_steps = int(
                    self.total_time / self.dt
                )  # Total simulation time steps

                self.lambda_cbf = 3  # Design parameter for CBF
                self.evasion_step_start = 15
                self.evasion_step_end = 25
                self.evasive_offset = 1 * self.lane_width

                self.w_acc = 1  # Weight for acceleration in QP
                self.w_steer = 0.3  # Weight for steering rate in QP
                self.Q = np.diag(
                    [self.w_acc, self.w_steer]
                )  # Weights for acceleration and steering rate
            else:
                # Bypassing & MTV
                self.total_time = 3.0  # Total simulation time
                self.num_steps = int(
                    self.total_time / self.dt
                )  # Total simulation time steps

                self.lambda_cbf = 4  # Design parameter for CBF
                self.evasion_step_start = 15
                self.evasion_step_end = 40
                self.evasive_offset = 0.6 * self.lane_width

                self.w_acc = 1  # Weight for acceleration in QP
                self.w_steer = 0.8  # Weight for steering rate in QP
                self.Q = np.diag(
                    [self.w_acc, self.w_steer]
                )  # Weights for acceleration and steering rate

        self.list_state_i = [self.state_i.clone().numpy()]
        self.list_state_j = [self.state_j.clone().numpy()]

        self.color_i = colors[0]
        self.color_j = colors[1]

        # RL policy
        self.rl_policy_path = "checkpoints/ecc25/higher_ref_penalty_5.pth"
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

        self.list_opt_duration = []

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
            path_nn="sigmarl/assets/nn_sm_predictors/sm_predictor_41.pth",
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
            ],  # represents the parameters of the policy distribution for each agent
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

        policy.load_state_dict(torch.load(self.rl_policy_path))

        policy.module[0].module[0].agent_networks[0].in_features

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

        if self.scenario_type.lower() == "overtaking":
            # In the overtaking scenario, agent j moves with constant speed
            self.nomi_cont_j = ConstantController(self.device)
            self.tensordict_j = None
        else:
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
            v=torch.tensor(self.v_max, device=self.device, dtype=torch.float32),
            rot=torch.tensor(2 * torch.pi, device=self.device, dtype=torch.float32),
            steering=self.steering_max,
            distance_ref=self.width * 2,
        )

    @staticmethod
    def compute_relative_poses(x_ego, y_ego, psi_ego, x_target, y_target, psi_target):
        """
        Compute the relative poses (positions and heading) between the ego vehicle and the target.

        Returns:
            tuple: Relative positions and heading in vehicle i's coordinate system.
        """

        dx = x_target - x_ego
        dy = y_target - y_ego
        psi_relative = psi_target - psi_ego  # Relative heading
        psi_relative = (psi_relative + np.pi) % (
            2 * np.pi
        ) - np.pi  # Normalize psi_relative to [-pi, pi]

        # Transform to vehicle i's coordinate system
        distance = np.sqrt(dx**2 + dy**2)
        psi_relative_coordinate = np.arctan2(dy, dx) - psi_ego
        x_relative = distance * np.cos(psi_relative_coordinate)
        y_relative = distance * np.sin(psi_relative_coordinate)

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
                # assert not (
                #     (x_relative > self.SME.excl_x_min)
                #     and (x_relative < self.SME.excl_x_max)
                #     and (y_relative > self.SME.excl_y_min)
                #     and (y_relative < self.SME.excl_y_max)
                # )
                assert (psi_relative >= self.SME.heading_min) and (
                    psi_relative <= self.SME.heading_max
                )

                # Use MTV-based distance, which considers the headings, to estimate safety margin if the surrounding objective is geometrically inside the allowed ranges
                print(f"********************************Use MTV-based safety margin.")
                sm, grad, hessian = self.mtv_based_sm(
                    x_relative, y_relative, psi_relative
                )

                # Compute actual safety margin
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
                vertices = torch.stack(
                    [rect_i_vertices, rect_j_vertices], dim=0
                ).unsqueeze(0)

                actual_sm = get_distances_between_agents(
                    vertices, distance_type="mtv", is_set_diagonal=False
                )[0, 0, 1].item()

                # print(f"Predicted safety margin: {sm:.6f} m")
                # print(f"Actual safety margin (absolute vertices): {(actual_sm):.6f} m")

                error_prediction = abs(actual_sm - sm)

                print(f"Safety-margin prediction error: {error_prediction:.6f}")

                assert error_prediction <= self.SME.error_upper_bound
            else:
                # If the surrounding objective is outside the ranges, using center-to-center based distance, which does not consider headings, to estimate safety margin.
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
        Safety margin based on center-to-center (c2c) distance.

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
        # assert(torch.allclose(grad_d_1st_order[0], x_ji / d))
        # assert(torch.allclose(grad_d_1st_order[1], y_ji / d))
        # assert(torch.allclose(hessian[1, 1], x_ji ** 2 / d ** 3))
        # assert(torch.allclose(hessian[0, 0], y_ji ** 2 / d ** 3))
        # assert(torch.allclose(hessian[0, 1], - x_ji * y_ji / d ** 3))

        # Detach gradients to prevent further computation and convert to numpy
        return (
            sm.detach().item(),
            grad_d_1st_order.detach().numpy(),
            hessian.detach().numpy(),
        )

    def mtv_based_sm(self, x_ji, y_ji, psi_ji):
        """
        Safety margin based on Minimum Translation Vector (MTV)-based distance.

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
        h = sm - self.d_min

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

        return h, dot_h, ddot_h, cbf_condition_1, cbf_condition_2

    def compute_state_time_derivatives(self, u_i, u_j):
        # Compute state time derivatives
        self.dstate_time_i = self.kbm.ode(None, self.state_i, self.u_placeholder)
        self.dstate_time_j = self.kbm.ode(None, self.state_j, self.u_placeholder)

        dstate_time_ji = (self.dstate_time_j - self.dstate_time_i)[0:3].numpy()

        ddx_i, ddy_i, ddpsi_i = self.compute_dstate_2nd_time(
            u_i, self.state_i, self.dstate_time_i
        )
        ddx_j, ddy_j, ddpsi_j = self.compute_dstate_2nd_time(
            u_j, self.state_j, self.dstate_time_j
        )

        # Compute relative second derivatives
        ddx_ji = ddx_j - ddx_i
        ddy_ji = ddy_j - ddy_i
        ddpsi_ji = ddpsi_j - ddpsi_i

        # Compute second derivatives for vehicle i
        # Extract control inputs
        ddstate_time_ji = [ddx_ji, ddy_ji, ddpsi_ji]

        dstate_time_ij = -dstate_time_ji
        ddstate_time_ij = [-expr for expr in ddstate_time_ji]

        return dstate_time_ji, ddstate_time_ji, dstate_time_ij, ddstate_time_ij

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
            - (state[3].item() / self.l_wb) * sin_beta * tan_delta * dbeta
            + (state[3].item() / self.l_wb) * cos_beta * sec_delta_sq * u_2
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
            # x-coordinates
            path_points[:, 0] = (
                self.state_i[0]
                + torch.arange(1, self.rl_n_points_ref + 1, device=self.device)
                * self.rl_distance_between_points_ref_path
            )

            # Conditions for overtaking: (a) Both agents are on the same lane, (b) None of them is on a transition from one lane to another, and (c) their longitudinal distance is less than a threshold
            is_conduct_overtaking = (
                (self.lane_i == self.lane_j)
                and (len(self.lane_i) == 1)
                and (
                    (self.state_i[0] - self.state_j[0]).abs() < self.threshold_overtake
                )
            )
            if is_conduct_overtaking:
                # Switch reference path to another lane to encourage overtaking
                # y-coordinates
                path_points[:, 1] = (
                    self.y_lane_2 if self.lane_i == "1" else self.y_lane_1
                )
            else:
                # y-coordinates
                path_points[:, 1] = (
                    self.y_lane_1 if self.lane_i == "1" else self.y_lane_2
                )
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

    def update_goal_i(self):
        pass

    def update_goal_j(self):
        pass

    def update(self, frame):
        """
        Update function for each frame of the animation.

        Args:
            frame (int): Current frame number.

        Returns:
            list: Updated plot elements.
        """
        self.step += 1
        print(
            f"------------Step: {self.step}--Time: {frame * self.dt:.2f}s------------"
        )
        # Initialize optimization variables for vehicle i and possibly j
        u_i = cp.Variable(2)
        u_j = (
            self.nomi_cont_j.get_actions().numpy()
            if self.scenario_type.lower() == "overtaking"
            else cp.Variable(2)
        )

        # Update lane information
        self.lane_i = (
            "1"
            if (self.state_i[1] - self.y_lane_1).abs() <= self.threshold_within_lane
            else (
                "2"
                if (self.state_i[1] - self.y_lane_2).abs() <= self.threshold_within_lane
                else None
            )
        )
        self.lane_j = (
            "1"
            if (self.state_j[1] - self.y_lane_1).abs() <= self.threshold_within_lane
            else (
                "2"
                if (self.state_j[1] - self.y_lane_2).abs() <= self.threshold_within_lane
                else None
            )
        )
        if self.lane_i == None:
            # "12" mean on a transition from lane 1 to lane 2; vice versa
            self.lane_i = (
                "12"
                if self.list_lane_i[-1] == "1"
                else ("21" if self.list_lane_i[-1] == "2" else self.list_lane_i[-1])
            )
        if self.lane_j == None:
            self.lane_j = (
                "12"
                if self.list_lane_j[-1] == "1"
                else ("21" if self.list_lane_j[-1] == "2" else self.list_lane_j[-1])
            )

        self.list_lane_i.append(self.lane_i)
        self.list_lane_j.append(self.lane_j)

        # State derivative to time
        (
            dstate_time_ji,
            ddstate_time_ji,
            dstate_time_ij,
            ddstate_time_ij,
        ) = self.compute_state_time_derivatives(u_i, u_j)

        x_ji, y_ji, psi_ji, _ = self.compute_relative_poses(
            self.state_i[0],
            self.state_i[1],
            self.state_i[2],
            self.state_j[0],
            self.state_j[1],
            self.state_j[2],
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

        # Get RL observations
        obs_i, self.ref_points_i, self.ref_points_ego_view_i = self.observation(
            self.state_i, self.list_state_i[0][0:2], self.goal_i.clone(), agent_idx=0
        )
        # Update tensordict for later policy call
        self.tensordict_i.set(self.rl_observation_key, obs_i.unsqueeze(0))

        # Get nominal control inputs
        rl_actions_i = (
            self.nomi_cont_i(self.tensordict_i)
            .get(self.rl_action_key)
            .squeeze(0)
            .detach()
        )
        rl_actions_i[0] = torch.clamp(rl_actions_i[0], 0, self.v_max / 2)
        print(f"Reduce rl_actions_i: {rl_actions_i}")
        u_nominal_i = self.rl_acrion_to_u(
            rl_actions_i, self.state_i[3], self.state_i[4]
        )

        if self.scenario_type.lower() == "overtaking":
            u_nominal_j = self.nomi_cont_j.get_actions()

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

            # Get RL observations
            obs_j, self.ref_points_j, self.ref_points_ego_view_j = self.observation(
                self.state_j,
                self.list_state_j[0][0:2],
                self.goal_j.clone(),
                agent_idx=1,
            )
            # Update tensordict for later policy call
            self.tensordict_j.set(self.rl_observation_key, obs_j.unsqueeze(0))

            # Get nominal control inputs
            rl_actions_j = (
                self.nomi_cont_j(self.tensordict_j)
                .get(self.rl_action_key)
                .squeeze(0)
                .detach()
            )
            u_nominal_j = self.rl_acrion_to_u(
                rl_actions_j, self.state_j[3], self.state_j[4]
            )

            # Objective: Minimize weighted squared deviation from nominal control inputs
            objective = cp.Minimize(
                cp.quad_form(u_i - u_nominal_i, self.Q)
                + cp.quad_form(u_j - u_nominal_j, self.Q)
            )
            constraints = [
                cbf_condition_2_ji >= 0,
                # self.a_min <= u_i[0],
                # u_i[0] <= self.a_max,
                # self.steering_rate_min <= u_i[1],
                # u_i[1] <= self.steering_rate_max,
                cbf_condition_2_ij >= 0,
                # self.a_min <= u_j[0],
                # u_j[0] <= self.a_max,
                # self.steering_rate_min <= u_j[1],
                # u_j[1] <= self.steering_rate_max,
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

            u_j_opt = u_nominal_j.numpy()
            (
                dstate_time_ji_opt,
                ddstate_time_ji_opt,
                _,
                _,
            ) = self.compute_state_time_derivatives(u_i_opt, u_j_opt)
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
            self.list_h_ji.append(h_ji_opt)
            self.list_dot_h_ji.append(dot_h_ji_opt)
            self.list_ddot_h_ji.append(ddot_h_ji_opt)
            self.list_cbf_condition_1_ji.append(cbf_condition_1_ji_opt)
            self.list_cbf_condition_2_ji.append(cbf_condition_2_ji_opt)

            print(f"Nominal actions i: {u_nominal_i}")
            print(f"Optimized actions i: {u_i.value}")

            # Print CBF details for debugging
            # Recompute CBF conditions with actual control actions
            assert h_ji_opt == h_ji
            assert dot_h_ji_opt == dot_h_ji
            assert cbf_condition_1_ji_opt == cbf_condition_1_ji

            print(f"h_ji_opt: {h_ji_opt:.4f} (should >= 0)")
            print(f"dot_h_ji_opt: {dot_h_ji_opt:.4f}")
            print(f"ddot_h_ji_opt: {ddot_h_ji_opt:.4f}")
            print(f"cbf_condition_1_ji_opt: {cbf_condition_1_ji_opt:.4f} (should >= 0)")
            print(f"cbf_condition_2_ji_opt: {cbf_condition_2_ji_opt:.4f} (should >= 0)")
        else:
            if prob.status != cp.OPTIMAL:
                print(f"Warning: QP not solved optimally. Status: {prob.status}")
                u_i_opt = u_nominal_i
                u_j_opt = u_nominal_j
            else:
                u_i_opt = u_i.value
                u_j_opt = u_j.value

            (
                dstate_time_ji_opt,
                ddstate_time_ji_opt,
                dstate_time_ij_opt,
                ddstate_time_ij_opt,
            ) = self.compute_state_time_derivatives(u_i_opt, u_j_opt)
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

            # Append to lists
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

            # Update state
            print(f"Nominal actions i: {u_nominal_i}")
            print(f"Optimized actions i: {u_i.value}")
            print(f"Nominal actions j: {u_nominal_j}")
            print(f"Optimized actions j: {u_j.value}")

            # Print CBF details for debugging
            # Recompute CBF conditions with actual control actions
            assert h_ji_opt == h_ji
            assert dot_h_ji_opt == dot_h_ji
            assert cbf_condition_1_ji_opt == cbf_condition_1_ji

            assert h_ij_opt == h_ij
            assert dot_h_ij_opt == dot_h_ij
            assert cbf_condition_1_ij_opt == cbf_condition_1_ij

            print(f"h_ji_opt: {h_ji_opt:.4f} (should >= 0)")
            print(f"dot_h_ji_opt: {dot_h_ji_opt:.4f}")
            print(f"ddot_h_ji_opt: {ddot_h_ji_opt:.4f}")
            print(f"cbf_condition_1_ji_opt: {cbf_condition_1_ji_opt:.4f} (should >= 0)")
            print(f"cbf_condition_2_ji_opt: {cbf_condition_2_ji_opt:.4f} (should >= 0)")
            print(f"h_ij_opt: {h_ij_opt:.4f} (should >= 0)")
            print(f"dot_h_ij_opt: {dot_h_ij_opt:.4f}")
            print(f"ddot_h_ij_opt: {ddot_h_ij_opt:.4f}")
            print(f"cbf_condition_1_ij_opt: {cbf_condition_1_ij_opt:.4f} (should >= 0)")
            print(f"cbf_condition_2_ij_opt: {cbf_condition_2_ij_opt:.4f} (should >= 0)")

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

        # Update plot elements including arrows
        self.update_plot_elements()

        # Update traget
        self.update_goal_i()
        self.update_goal_j()

        return [
            self.vehicle_i_rect,
            self.vehicle_j_rect,
            self.line_i,
            self.line_j,
            self.visu_goal_point_i,
            self.visu_goal_point_j,
            self.visu_ref_points_i,
            self.visu_ref_points_j,
        ]

    def observation(self, state, orig_pos, goal_pos, agent_idx):
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
        # Clamp
        rl_actions[0] = torch.clamp(rl_actions[0], self.v_min, self.v_max)
        rl_actions[1] = torch.clamp(rl_actions[1], self.steering_min, self.steering_max)

        # Convert from RL actions (speed and steering) to acceleration and steering rate used in the kinematic bicycle model needs
        # Assume linear acceleration and steering change
        u_acc = (rl_actions[0] - v) / self.dt
        u_steering_rate = (rl_actions[1] - steering) / self.dt
        u_acc = torch.clamp(u_acc, self.a_min, self.a_max)
        u_steering_rate = torch.clamp(
            u_steering_rate, self.steering_rate_min, self.steering_rate_max
        )
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
        x_min = -2
        x_max = 2
        y_min = -0.2
        y_max = 0.2
        xy_ratio = (y_max - y_min) / (x_max - x_min)

        x_fig_size = 20
        fig, ax = plt.subplots(figsize=(x_fig_size, x_fig_size * xy_ratio + 2))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_title("Trajectories and Control Actions of Vehicles i and j")
        # ax.grid(True)
        plt.tight_layout()

        # Add static horizontal lines as lane boundaries
        ax.axhline(
            y=self.y_lane_bottom_bound, color="k", linestyle="-", lw=1.0
        )  # Bottom lane boundary
        ax.axhline(
            y=self.y_lane_center_line, color="k", linestyle="--", lw=1.0
        )  # Center line
        ax.axhline(
            y=self.y_lane_top_bound, color="k", linestyle="-", lw=1.0
        )  # Top lane boundary

        # Initialize vehicle rectangles
        vehicle_i_rect = Rectangle(
            (0, 0),
            self.length,
            self.width,
            angle=0,
            color=self.color_i,
            alpha=0.5,
            label="Vehicle i",
        )
        vehicle_j_rect = Rectangle(
            (0, 0),
            self.length,
            self.width,
            angle=0,
            color=self.color_j,
            alpha=0.5,
            label="Vehicle j",
        )
        ax.add_patch(vehicle_i_rect)
        ax.add_patch(vehicle_j_rect)

        # Initialize path lines
        (line_i,) = ax.plot(
            [],
            [],
            color=self.color_i,
            linestyle="-",
            lw=1.0,
            label="Vehicle i footprint",
        )
        (line_j,) = ax.plot(
            [],
            [],
            color=self.color_j,
            linestyle="-",
            lw=1.0,
            label="Vehicle j footprint",
        )

        # Initialize polyline for short-term reference path
        (ref_line_i,) = ax.plot(
            [], [], color=self.color_i, linestyle="-", lw=0.5, alpha=0.5
        )
        (ref_line_j,) = ax.plot(
            [], [], color=self.color_j, linestyle="-", lw=0.5, alpha=0.5
        )

        # Initialize points for short-term reference path
        (visu_ref_points_i,) = ax.plot(
            [], [], color=self.color_i, marker="o", markersize=4
        )
        (visu_ref_points_j,) = ax.plot(
            [], [], color=self.color_j, marker="o", markersize=4
        )

        # Initialize point for goal
        (visu_goal_point_i,) = ax.plot(
            [],
            [],
            color=self.color_i,
            marker="o",
            markersize=8,
            label="Vehicle i: goal",
        )
        (visu_goal_point_j,) = ax.plot(
            [],
            [],
            color=self.color_j,
            marker="o",
            markersize=8,
            label="Vehicle j: goal",
        )

        ax.legend(loc="upper right")

        self.fig = fig
        self.ax = ax
        self.vehicle_i_rect = vehicle_i_rect
        self.vehicle_j_rect = vehicle_j_rect
        self.line_i = line_i
        self.line_j = line_j
        self.ref_line_i = ref_line_i
        self.ref_line_j = ref_line_j
        self.visu_ref_points_i = visu_ref_points_i
        self.visu_ref_points_j = visu_ref_points_j
        self.visu_goal_point_i = visu_goal_point_i
        self.visu_goal_point_j = visu_goal_point_j

    def update_plot_elements(self):
        """
        Update the positions and orientations of the vehicle rectangles and path lines,
        and update the action arrows for visualization.
        """
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

        # Update path lines
        self.line_i.set_data(
            [s[0] for s in self.list_state_i], [s[1] for s in self.list_state_i]
        )
        self.line_j.set_data(
            [s[0] for s in self.list_state_j], [s[1] for s in self.list_state_j]
        )

        # Update short-term reference path polylinex
        ref_x_i = self.ref_points_i[:, 0].numpy()
        ref_y_i = self.ref_points_i[:, 1].numpy()
        self.ref_line_i.set_data(ref_x_i, ref_y_i)
        # Update short-term reference path points
        self.visu_ref_points_i.set_data(ref_x_i, ref_y_i)

        # Update goal point
        goal_x_i = self.goal_i[0].item()
        goal_y_i = self.goal_i[1].item()
        self.visu_goal_point_i.set_data([goal_x_i], [goal_y_i])

        if self.scenario_type.lower() == "bypassing":
            # Update short-term reference path polylinex
            ref_x_j = self.ref_points_j[:, 0].numpy()
            ref_y_j = self.ref_points_j[:, 1].numpy()
            self.ref_line_j.set_data(ref_x_j, ref_y_j)
            # Update short-term reference path points
            self.visu_ref_points_j.set_data(ref_x_j, ref_y_j)

            goal_x_j = self.goal_j[0].item()
            goal_y_j = self.goal_j[1].item()
            self.visu_goal_point_j.set_data([goal_x_j], [goal_y_j])

        # plt.pause(0.0001)

    def plot_cbf_curve(self) -> None:
        """
        Plot data related to CBF.
        """
        steps = range(1, len(self.list_h_ji) + 1)
        plt.figure(figsize=(5, 3))

        plt.plot(steps, self.list_h_ji, label="h_ji", linestyle="-", lw="2.0")
        # plt.plot(steps, self.list_dot_h_ji, label="dot_h_ji", linestyle="--", lw="2.0")
        # plt.plot(steps, self.list_ddot_h_ji, label="ddot_h_ji", linestyle="-.", lw="2.0")
        plt.plot(
            steps,
            self.list_cbf_condition_1_ji,
            label="cbf_condition_1_ji",
            linestyle="--",
            lw="2.0",
        )
        plt.plot(
            steps,
            self.list_cbf_condition_2_ji,
            label="cbf_condition_2_ji",
            linestyle="-.",
            lw="2.0",
        )

        if self.scenario_type.lower() == "bypassing":
            plt.plot(steps, self.list_h_ij, label="h_ij", linestyle=":", lw="2.0")
            # plt.plot(steps, self.list_dot_h_ij, label="dot_h_ij", linestyle="--", lw="2.0")
            # plt.plot(steps, self.list_ddot_h_ij, label="ddot_h_ij", linestyle="-.", lw="2.0")
            plt.plot(
                steps,
                self.list_cbf_condition_1_ij,
                label="cbf_condition_1_ij",
                linestyle="--",
                lw="2.0",
            )
            plt.plot(
                steps,
                self.list_cbf_condition_2_ij,
                label="cbf_condition_2_ij",
                linestyle="-.",
                lw="2.0",
            )

        plt.xlabel("Steps")
        plt.ylabel("CBF Data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Run the simulation and display the animation.
        """
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.num_steps,
            blit=False,  # Set to True to optimize the rendering by updating only parts of the frame that have changed (may rely heavily on the capabilities of the underlying GUI backend and the system’s graphical stack)
            interval=self.dt * 1000,
            repeat=False,
        )
        plt.show()


def main():
    simulation = CBF()
    simulation.run()
    simulation.plot_cbf_curve()
    print(
        f"Mean optimization time per step: {np.mean(simulation.list_opt_duration)} s."
    )


if __name__ == "__main__":
    main()
