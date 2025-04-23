# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import cvxpy as cp
import time
import math

from sigmarl.constants import SCENARIOS, AGENTS
from sigmarl.dynamics import KinematicBicycleModel
from sigmarl.mtv_based_sm_predictor import SafetyMarginEstimatorModule
from sigmarl.rectangle_approximation import RectangleCircleApproximation
from sigmarl.helper_training import Parameters


class CBFQP:
    def __init__(self, env=None, env_idx: int = None, agent_idx: int = None, **kwargs):

        self.env = env
        self.env_idx = env_idx  # Index of the environment
        self.agent_idx = agent_idx  # Index of the agent
        self.map_pseudo_distance = (
            self.env.base_env.scenario_name.map_pseudo_distance
        )  # Pseudo distance information for distance to bpundary
        self.parameters: Parameters = (
            self.env.base_env.scenario_name.parameters
        )  # Parameters from the environment
        self.step = 0  # Step for current environment of current agent
        self.is_using_centralized_cbf = (
            self.parameters.is_using_centralized_cbf
        )  # CBF solving strategy

        # Initialize the parameters
        self.initialize_params(**kwargs)

        # Kinematics model for the agent
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

        self.rec_cir_approx = RectangleCircleApproximation(
            self.length, self.width, self.parameters.n_circles_approximate_vehicle
        )  # Use circles to approximate agent

        # Load safety margin estimator module to estimate agent to agent safety margin
        self.load_safety_margin_estimator()

    def initialize_params(self, **kwargs):
        self.device = self.parameters.device
        self.dt = self.parameters.dt  # Sample time in seconds
        self.dx = 0.02
        self.dy = 0.02  # Sample interval for derivatives approximation

        self.length = AGENTS["length"]  # Length of each rectangle (m)
        self.width = AGENTS["width"]  # Width of each rectangle (m)
        self.l_wb = AGENTS["l_wb"]  # Wheelbase
        self.l_f = AGENTS["l_f"]  # Front wheelbase (m)
        self.l_r = AGENTS["l_r"]  # Rear wheelbase (m)

        self.v_max = torch.tensor(
            AGENTS["max_speed"], device=self.device, dtype=torch.float32
        )  # Max velocity (m/s)
        self.v_min = torch.tensor(
            AGENTS["min_speed"], device=self.device, dtype=torch.float32
        )  # Min velocity (m/s)
        self.steering_max = torch.tensor(
            AGENTS["max_steering"], device=self.device, dtype=torch.float32
        )  # Max steering (rad)
        self.steering_min = torch.tensor(
            AGENTS["min_steering"], device=self.device, dtype=torch.float32
        )  # Min steering (rad)
        self.a_max = AGENTS["max_acc"]  # Max acceleration (m/s^2)
        self.a_min = AGENTS["min_acc"]  # Min deceleration (m/s^2)
        self.steering_rate_max = AGENTS[
            "max_steering_rate"
        ]  # Max steering rate (rad/s)
        self.steering_rate_min = AGENTS[
            "min_steering_rate"
        ]  # Min steering rate (rad/s)

        self.w_acc = 30  # Weight for acceleration in weight metrix Q
        self.w_steer = 1  # Weight for steering rate in weight metrix Q
        self.Q = np.diag(
            [self.w_acc, self.w_steer]
        )  # Weights for acceleration and steering rate

        self.small_circle_radius = (
            0.05  # Vehicle approximation with a set of circles (agent-to-boundary)
        )
        self.big_circle_radius = (
            np.sqrt(self.length**2 + self.width**2) / 2
        )  # Vehicle approximation with one circle (agent-to-agent)

        # Distance type for agent-to-agent distance
        self.agent_2_agent_sm_type = kwargs.pop("agent_2_agent_sm_type", "mtv")

        self.is_inject_noise = kwargs.pop("is_inject_noise", False)
        self.noise_percentage = kwargs.pop(
            "noise_percentage", 1
        )  # Noise percentage with respect to the original data

        self.safety_buffer = 0.01  # saftey buffer

        # Parameters for CBF-constrained MARL
        self.lambda_cbf_a2l_1 = 0.2
        self.lambda_cbf_a2l_2 = 0.2
        self.lambda_cbf_a2a_1 = 0.2
        self.lambda_cbf_a2a_2 = 0.2

        # Parameters for PACBF-constrained MARL
        self.p1 = 0.2  # initial state of p1
        self.p1_0 = 0.2  # p1*, value to be stabilized
        self.p2_0 = 0.2  # p2*, value to be stabilized

        # Count for number of failure solving
        self.num_fail = 0
        # Whether to use adaptive CBF
        self.use_pacbf = True

    def load_safety_margin_estimator(self):
        """
        Load the safety margin estimator module for agent to agent distance

        Args:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.

        Returns:
            SafetyMarginEstimatorModule: safety margin estimator module.
        """
        self.SME = SafetyMarginEstimatorModule(
            length=self.length,
            width=self.width,
            path_nn="CBF_evaluation/CBF_constrained_MARL/sm_predictor.pth",
        )

        self.SME.load_model()  # Neural network for estimating safety margin

    def load_observation(
        self,
        tensordict,
        mask_higher_priority_agents=None,
    ):
        """
        Load the necessary data for CBF-QP solving from cbf observation.
        """
        # Get all of cbf observation
        cbf_observation_key = ("agents", "info", "cbf_observation")
        cbf_obs = tensordict[cbf_observation_key].clone()

        # Number of the environment and agent in each environment
        self.n_envs, self.n_agents = (
            cbf_obs.shape[0],
            cbf_obs.shape[1],
        )

        # Number of observable nearing agents for one agent
        self.num_nearing_agents = self.parameters.n_nearing_agents_observed

        # Prepare mask for higher priority agent
        self.mask_higher_priority_agents = mask_higher_priority_agents

        # List for storing cbf observation
        self.state_agent_list = [None] * self.n_agents
        self.lanelet_id_list = [None] * self.n_agents
        self.path_id_list = [None] * self.n_agents
        self.state_others_list = [None] * self.n_agents
        self.u_others = None

        # List for storing nominal RL actions and converted control actions u
        self.rl_actions_nominal_agent_list = [None] * self.n_agents
        self.u_nominal_agent_list = [None] * self.n_agents
        self.rl_actions_nominal_others_list = [None] * self.n_agents
        self.u_nominal_others_list = [None] * self.n_agents

        if self.is_using_centralized_cbf:
            self.nearing_agents_indices = (
                self.env.base_env.scenario_name.observations.nearing_agents_indices
            )
            for i in range(self.n_agents):
                self.single_agent_observation(tensordict, cbf_obs, i)
        else:
            self.single_agent_observation(tensordict, cbf_obs, self.agent_idx)

        # Initialize optimization variable placeholder for derivatives computation
        self.u_placeholder = torch.tensor(
            [0.0, 0.0], dtype=torch.float32
        )  # Do not update its value

    def single_agent_observation(self, tensordict, cbf_obs, agent_idx):
        """
        Load the necessary data for single agent.
        """
        # Self observation of certain agent in certain environment
        agent_cbs_obs = cbf_obs[self.env_idx, agent_idx]
        state_agent = agent_cbs_obs[0:5]  # [x, y, psi, velocity, steering]
        lanelet_id = agent_cbs_obs[5]  # Lanelet ID
        path_id = agent_cbs_obs[6]  # Path ID

        # Others observation of the agent
        obs_others = agent_cbs_obs[7:]

        if self.is_using_centralized_cbf:
            # For centralized solving, all the nearing agents will be considered
            # No action observation from higher priority agents
            state_others = obs_others.reshape(
                self.num_nearing_agents, -1
            )  # (num_nearing_agents, 5)

            # Certain agent's nearing agent indices
            nearing_indices = self.nearing_agents_indices[
                self.env_idx, agent_idx
            ]  # (num_nearing_agents, )

            # Store other agents's nominal action from RL policy to be optimized
            agent_rl_actions_nominal_others_list = []
            agent_u_nominal_others_list = []
            for i in range(len(nearing_indices)):
                agent_rl_actions_nominal_others = tensordict[("agents", "action")][
                    self.env_idx, nearing_indices[i]
                ].clone()
                agent_u_nominal_others = self.rl_action_to_u(
                    rl_actions=agent_rl_actions_nominal_others,
                    v=state_others[i][3],
                    steering=state_others[i][4],
                )  # (action_dim, )
                agent_rl_actions_nominal_others_list.append(
                    agent_rl_actions_nominal_others
                )
                agent_u_nominal_others_list.append(agent_u_nominal_others)

            self.rl_actions_nominal_others_list[
                agent_idx
            ] = agent_rl_actions_nominal_others_list
            self.u_nominal_others_list[agent_idx] = agent_u_nominal_others_list
        else:
            # For decentralized solution, only the higher priority agent will be considered
            # Actions observation from higher priority agents need to be splited from the cbf state observation
            if self.num_nearing_agents == 0:
                state_others = None
                self.num_higher_prio_nearing_agents = 0
            else:
                all_state_others = obs_others[: (-self.num_nearing_agents * 2)]
                all_state_others = all_state_others.reshape(
                    self.num_nearing_agents, -1
                )  # (n_nearing_agents, 6)

                cbf_masked_agents = all_state_others[
                    :, -1
                ]  # value 0: observable, 1: non observable
                cbf_masked_agents = (
                    cbf_masked_agents > 0
                )  # mask for observable nearing agent

                all_state_others = all_state_others[:, :-1]  # (n_nearing_agents, 5)
                all_action_others = obs_others[(-self.num_nearing_agents * 2) :]
                all_action_others = all_action_others.reshape(
                    self.num_nearing_agents, -1
                )  # (n_nearing_agents, 2)

                mask = (
                    self.mask_higher_priority_agents & ~cbf_masked_agents
                )  # mask for observable higher priority agents

                # Mask the surrounding higher priority agents for decentralized solution
                state_others = all_state_others[mask[self.env_idx, agent_idx]].reshape(
                    -1, all_state_others.shape[-1]
                )  # (n_higher_prio_nearing_agents, 5), state of each agent:[x, y, psi, velocity, steering]

                action_others = all_action_others[
                    mask[self.env_idx, agent_idx]
                ].reshape(
                    -1, all_action_others.shape[-1]
                )  # (n_higher_prio_nearing_agents, 2), RL action of each agent: [velocity, steering]

                # Number of higher priority nearing agents of the agent
                self.num_higher_prio_nearing_agents = state_others.shape[0]

                # If there are higher priority agents
                if self.num_higher_prio_nearing_agents > 0:
                    # Convert from RL actions speed [m/s] and steering [rad] to acceleration [m/s^2] and steering rate [rad/s]
                    action_others, self.u_others = self.rl_action_to_u(
                        rl_actions=action_others,
                        v=state_others[:, 3],
                        steering=state_others[:, 4],
                    )  # (n_higher_prio_nearing_agents, 2), control action u of each agent: [acceleration, steering rate]

        # Agent's nominal action from RL policy
        rl_actions_nominal_agent = tensordict[("agents", "action")][
            self.env_idx, agent_idx
        ].clone()
        if self.is_inject_noise:
            rl_action_noise = (
                torch.rand_like(rl_actions_nominal_agent) * self.noise_percentage
            )
            rl_actions_nominal_agent += rl_action_noise

        # Convert from RL actions speed [m/s] and steering [rad] to acceleration [m/s^2] and steering rate [rad/s]
        rl_actions_nominal_agent, u_nominal_agent = self.rl_action_to_u(
            rl_actions=rl_actions_nominal_agent,
            v=state_agent[3],
            steering=state_agent[4],
        )

        # Store state information
        self.state_agent_list[agent_idx] = state_agent
        self.state_others_list[agent_idx] = state_others
        # Lanelet ID and Reference ID
        self.lanelet_id_list[agent_idx] = int(lanelet_id)
        self.path_id_list[agent_idx] = int(path_id)

        # RL action and control action u
        self.rl_actions_nominal_agent_list[agent_idx] = rl_actions_nominal_agent
        self.u_nominal_agent_list[agent_idx] = u_nominal_agent

    def rl_action_to_u(
        self, rl_actions: torch.tensor, v: torch.tensor, steering: torch.tensor
    ):
        """
        Convert from RL actions speed [m/s] and steering [rad] to
        acceleration [m/s^2] and steering rate [rad/s] used in
        the kinematic bicycle model.
        """
        # Transform for btach calcultion
        if rl_actions.ndim == 1 or v.ndim == 0 or steering.ndim == 0:
            rl_actions = rl_actions.unsqueeze(0)
            v = v.unsqueeze(0)
            steering = steering.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        # Assume linear acceleration and steering change
        rl_actions[:, 0] = torch.clamp(
            rl_actions[:, 0], min=self.v_min, max=self.v_max
        )  # Limit speed
        rl_actions[:, 1] = torch.clamp(
            rl_actions[:, 1], min=self.steering_min, max=self.steering_max
        )  # Limit steering angle

        # Calculate acceleration and steering rate
        u_acc = (rl_actions[:, 0] - v) / self.dt
        u_steering_rate = (rl_actions[:, 1] - steering) / self.dt

        u_acc = torch.clamp(u_acc, min=self.a_min, max=self.a_max)  # Limit acceleration
        u_steering_rate = torch.clamp(
            u_steering_rate, min=self.steering_rate_min, max=self.steering_rate_max
        )  # Limit steering rate

        # Construct control action u
        u = torch.stack((u_acc, u_steering_rate), dim=1)

        if not is_batch:
            u.squeeze(0)

        return rl_actions, u

    def u_to_rl_action(self, u, state_v, state_steering):
        """
        Convert from acceleration [m/s^2] and steering rate [rad/s]
        used in the kinematic bicycle model to RL actions
        speed [m/s] and steering [rad].
        """
        # Transform for btach calcultion
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)
        if (
            u.ndimension() == 1
            or state_v.ndimension() == 0
            or state_steering.ndimension() == 0
        ):
            u = u.unsqueeze(0)  # Add batch dimension if needed
            state_v = state_v.unsqueeze(0)
            state_steering = state_steering.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        # Calculate the speed and steering angle
        v = state_v + u[:, 0] * self.dt  # Acceleration to speed
        steering = state_steering + u[:, 1] * self.dt  # Steering rate to steering angle

        # Normalize steering angle to [-pi, pi]
        steering = (steering + torch.pi) % (2 * torch.pi) - torch.pi

        # Apply clamping to limit v and steering within the desired range
        v = torch.clamp(v, min=self.v_min, max=self.v_max)  # Limit speed
        steering = torch.clamp(
            steering, min=self.steering_min, max=self.steering_max
        )  # Limit steering angle

        # Combine speed and steering into a tensor
        rl_action = torch.stack([v, steering], dim=1)

        # Remove batch dimension if needed
        if not is_batch:
            rl_action = rl_action.squeeze(0)

        return rl_action.to(self.device)

    @staticmethod
    def compute_relative_poses(
        x_ego,
        y_ego,
        psi_ego,
        x_target,
        y_target,
        psi_target,
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
        return x_relative.item(), y_relative.item(), psi_relative.item(), distance

    def get_circle_centers(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the global positions and states of circle centers used to
        approximate a vehicle's shape, based on the vehicle's current state,
        vehicle length, and the desired number of circles.

        The circles are distributed symmetrically along the vehicle's longitudinal
        axis (its local x-axis), centered around the vehicle's reference point
        (typically the rear axle or geometric center, as defined by the 'state' x,y).
        The spacing between adjacent circle centers along the local x-axis is
        explicitly defined as vehicle_length / n_circles.

        Args:
            state: A torch.Tensor representing the vehicle's state.
                Expected minimum structure: [x, y, yaw]. Additional elements
                (e.g., velocity, steering) are preserved and copied to each
                circle's state.
            n_circles: The number of circles to use for the approximation.
                    Must be an integer greater than or equal to 1.
            vehicle_length: The known length of the vehicle. Must be non-negative.
                            Used to determine the spacing between adjacent circles.

        Returns:
            A torch.Tensor of shape (n_circles, state.shape[0]), where each row
            represents the state of a circle center. The state of each circle
            includes its global (x, y) position and the vehicle's yaw and
            'other_state' elements from the input state tensor.

        Raises:
            ValueError: If n_circles is less than 1, if the state tensor
                        does not have at least 3 elements, or if vehicle_length
                        is negative.
            TypeError: If the input state is not a torch.Tensor.
        """
        # --- Input Validation ---
        if not isinstance(state, torch.Tensor):
            raise TypeError("Input 'state' must be a torch.Tensor.")

        n_circles = self.parameters.n_circles_approximate_vehicle
        if n_circles < 1:
            raise ValueError("Number of circles (n_circles) must be at least 1.")
        if state.ndim == 0 or state.shape[0] < 3:
            raise ValueError("State tensor must contain at least [x, y, yaw].")

        # Extract vehicle state components
        vehicle_center_global = state[0:2]
        vehicle_yaw = state[2]
        other_state = state[3:]

        relative_centers_local = torch.tensor(
            self.rec_cir_approx.centers, dtype=state.dtype, device=state.device
        )  # Shape: (n_circles, 2)

        # Rotate relative centers to global frame
        cos_yaw = torch.cos(vehicle_yaw)
        sin_yaw = torch.sin(vehicle_yaw)
        rotation_matrix_global = torch.tensor(
            [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
            dtype=state.dtype,
            device=state.device,
        )  # Shape: (2, 2)

        rotated_vectors_global = torch.matmul(
            rotation_matrix_global, relative_centers_local.T
        ).T
        # Shape: (n_circles, 2)

        # Translate rotated centers to global positions
        global_circle_centers = (
            rotated_vectors_global + vehicle_center_global.unsqueeze(0)
        )
        # Shape: (n_circles, 2)

        # Construct full circle states
        vehicle_yaw_expanded = (
            vehicle_yaw.unsqueeze(0).unsqueeze(0).expand(n_circles, 1)
        )
        other_state_expanded = other_state.unsqueeze(0).expand(n_circles, -1)

        circle_states = torch.cat(
            (global_circle_centers, vehicle_yaw_expanded, other_state_expanded), dim=1
        )  # Shape: (n_circles, state.shape[0])

        return circle_states

    def estimate_agent_2_lane_safety_margin(
        self, agent_pos: torch.tensor, lanelet_id, path_id
    ):
        """
        Estimate the safety margin between the agent and the lane boundary.
        Safety margin: sm = d - radius
        First-order derivatives [∂d/∂x, ∂d/∂y]
        Second-order derivatives
        [
            [∂d^2/∂x^2, ∂d^2/∂x∂y],
            [∂d^2/∂y∂x, ∂d^2/y^2]
        ]
        """
        # Offset points for the calculation of derivatives
        offsets = torch.tensor(
            [
                [0, 0],
                [self.dx, 0],
                [0, self.dy],
                [-self.dx, 0],
                [0, -self.dy],
                [self.dx, self.dy],
                [self.dx, -self.dy],
                [-self.dx, self.dy],
                [-self.dx, -self.dy],
            ],
            device=self.device,
        )
        query_points = agent_pos.unsqueeze(0) + offsets

        time_pseudo_dis_start = time.time()
        dleft_results, dright_results = self.map_pseudo_distance.get_distance(
            path_id, lanelet_id, query_points
        )
        self.time_pseudo_dis += (
            time.time() - time_pseudo_dis_start
        )  # Time for pseudo distance calculation

        # Get distance to boundary value for all query points
        (
            dleft,
            dleft_xplus,
            dleft_yplus,
            dleft_xminus,
            dleft_yminus,
            dleft_xplus_yplus,
            dleft_xplus_yminus,
            dleft_xminus_yplus,
            dleft_xminus_yminus,
        ) = dleft_results
        (
            dright,
            dright_xplus,
            dright_yplus,
            dright_xminus,
            dright_yminus,
            dright_xplus_yplus,
            dright_xplus_yminus,
            dright_xminus_yplus,
            dright_xminus_yminus,
        ) = dright_results

        # Approximate the first derivatives
        dleft_dx = (dleft_xplus - dleft) / self.dx
        dleft_dy = (dleft_yplus - dleft) / self.dy
        dright_dx = (dright_xplus - dright) / self.dx
        dright_dy = (dright_yplus - dright) / self.dy

        # Approximate the second derivatives
        dleft2_dx2 = (dleft_xplus - 2 * dleft + dleft_xminus) / (self.dx**2)
        dleft2_dy2 = (dleft_yplus - 2 * dleft + dleft_yminus) / (self.dy**2)
        dright2_dx2 = (dright_xplus - 2 * dright + dright_xminus) / (self.dx**2)
        dright2_dy2 = (dright_yplus - 2 * dright + dright_yminus) / (self.dy**2)
        dleft2_dxdy = (
            dleft_xplus_yplus
            - dleft_xplus_yminus
            - dleft_xminus_yplus
            + dleft_xminus_yminus
        ) / (4 * self.dx * self.dy)
        dright2_dxdy = (
            dright_xplus_yplus
            - dright_xplus_yminus
            - dright_xminus_yplus
            + dright_xminus_yminus
        ) / (4 * self.dx * self.dy)

        # Safety margin for left and right boundary
        sm_left = dleft - self.small_circle_radius
        sm_right = dright - self.small_circle_radius

        # First-order derivatives
        grad_left = np.array([dleft_dx, dleft_dy])
        grad_right = np.array([dright_dx, dright_dy])

        # Second-order derivatives (Hessian)
        hessian_left = np.array(
            [
                [dleft2_dx2, dleft2_dxdy],
                [dleft2_dxdy, dleft2_dy2],
            ]
        )
        hessian_right = np.array(
            [
                [dright2_dx2, dright2_dxdy],
                [dright2_dxdy, dright2_dy2],
            ]
        )

        return sm_left, grad_left, hessian_left, sm_right, grad_right, hessian_right

    def estimate_agent_2_agent_safety_margin(
        self, x_relative, y_relative, psi_relative
    ):
        """
        Estimate the safety margin using the neural network.

        Returns:
            tuple: Estimated safety margin and its gradient w.r.t inputs.
        """
        if self.agent_2_agent_sm_type == "c2c":
            sm, grad, hessian = self.c2c_based_sm(x_relative, y_relative, psi_relative)
        elif self.agent_2_agent_sm_type == "mtv":
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
        torch.set_grad_enabled(True)
        inputs = torch.tensor(
            [x_ji, y_ji, psi_ji],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        # Compute center-to-center distance d = sqrt(x^2 + y^2)
        x_squared = inputs[0] ** 2
        y_squared = inputs[1] ** 2
        d = torch.sqrt(x_squared + y_squared)

        sm = (
            d - 2 * self.big_circle_radius
        )  # Safety margin equals c2c distance minus two times big_circle_radius

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
        torch.set_grad_enabled(False)
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
        torch.set_grad_enabled(True)
        feature_norm = self.SME.feature_normalizer.squeeze(0)
        label_norm = self.SME.label_normalizer

        psi_ji = (psi_ji + torch.pi) % (2 * torch.pi) - torch.pi  # [-pi, pi]

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
        torch.set_grad_enabled(False)
        return sm_predicted, grad_sm_predicted, hessian_predicted

    def compute_dstate_2nd_time(self, u, state, dstate_time):
        """
        Compute the state second time dervatives.
        """
        u_1, u_2 = u  # Acceleration and steering rate

        # Compute beta and its derivative
        k = self.l_r / self.l_wb
        dpsi = dstate_time[2].item()
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

    def compute_relative_state_time_derivatives(
        self, state_agent, dstate_time_agent, ddx, ddy, ddpsi, u_j, state_j
    ):
        """
        Compute first and second order time derivatives of the relative states between vehicles i and j using global coordinate system.

        Args:
            u_j: Control inputs [acceleration, steering rate] for vehicle j

        Returns:
            dstate_time_ji: First derivatives of relative states j wrt i [dx, dy, dpsi]
            ddstate_time_ji: Second derivatives of relative states j wrt i [ddx, ddy, ddpsi]
            dstate_time_ij: First derivatives of relative states i wrt j [dx, dy, dpsi]
            ddstate_time_ij: Second derivatives of relative states i wrt j [ddx, ddy, ddpsi]
        """
        # Vehicle i is ego vehicle
        state_i = state_agent.detach().numpy()  # (5, )
        # The first and second derivatives
        dstate_time_i = dstate_time_agent.detach().numpy()  # (5, )
        ddx_i, ddy_i, ddpsi_i = ddx, ddy, ddpsi  # cp

        # Compute the first and second derivatives of vehicle j
        dstate_time_j = (
            self.kbm.ode(None, state_j, self.u_placeholder).detach().numpy()
        )  # (5, )
        ddx_j, ddy_j, ddpsi_j = self.compute_dstate_2nd_time(
            u_j, state_j, dstate_time_j
        )

        if isinstance(ddx_j, torch.Tensor):
            ddx_j = ddx_j.detach().numpy()
        if isinstance(ddy_j, torch.Tensor):
            ddy_j = ddy_j.detach().numpy()
        if isinstance(ddpsi_j, torch.Tensor):
            ddpsi_j = ddpsi_j.detach().numpy()

        # Compute relative first dervaitives
        dstate_time_ji_global = (dstate_time_j - dstate_time_i)[0:3]
        [dx_ji_global, dy_ji_global, dpsi_ji_global] = dstate_time_ji_global  # cp

        # Compute relative second derivatives
        ddx_ji_global = ddx_j - ddx_i
        ddy_ji_global = ddy_j - ddy_i
        ddpsi_ji_global = ddpsi_j - ddpsi_i  # cp

        # Convert from the global coordinate system to the ego coordinate system
        cos_psi_i = np.cos(state_i[2].item())
        sin_psi_i = np.sin(state_i[2].item())

        d_psi_i = dstate_time_i[2]

        x_ji_global, y_ji_global, psi_ji_global = (
            state_j[0:3].detach().numpy() - state_i[0:3]
        )

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

    def compute_agent_state_time_derivatives(self, state_agent, u_agent):
        """
        Compute agent's state first and second time derivatives: [dx, dy, dpsi, dv, ddelta] and [ddx, ddy, ddpsi].
        """
        # First time derivatives
        dstate_time_agent = self.kbm.ode(
            None, state_agent, self.u_placeholder
        )  # torch.tensor(dx, dy, dpsi, dv, ddelta)
        # Since the time derivatives of x, y, and heading only depend on the states, we can simply ignore actions

        # Second time derivatives
        ddx, ddy, ddpsi = self.compute_dstate_2nd_time(
            u_agent,  # CP variable(2)
            state_agent,  # torch.tensor
            dstate_time_agent,  # torch.tensor
        )

        return dstate_time_agent, ddx, ddy, ddpsi

    def compute_cbf_conditions(
        self, dstate_time, ddstate_time, sm, grad_sm, hessian_sm, p2, p1, v1, a2a
    ):
        """
        Compute the first and second order Control Barrier Function (CBF) conditions.

        Args:
            dstate_time: First-order time derivative of the state
            ddstate_time: Second-order time derivative of the state
            sm: Safety margin
            grad_sm: First-order derivative of the safety margin
            hessian_sm: Second-order derivative of the safety margin

        Returns:
            tuple: h, dot_h, ddot_h, cbf_condition_1, cbf_condition_2
        """
        if self.use_pacbf:
            # Compute h
            if a2a:
                h = sm - self.safety_buffer
            else:
                h = sm
            # Compute dot_h
            dot_h = grad_sm @ dstate_time

            # Compute ddot_h
            ddot_h = grad_sm @ ddstate_time + dstate_time.T @ hessian_sm @ dstate_time

            # First-order CBF condition
            cbf_condition_1 = dot_h * self.dt + p1 * h
            # Second-order CBF condition
            cbf_condition_2 = (
                ddot_h * self.dt + (p2 + p1) * dot_h * self.dt + (p1 * p2 + v1) * h
            )
        else:
            if a2a:
                lambda_cbf_1 = self.lambda_cbf_a2a_1
                lambda_cbf_2 = self.lambda_cbf_a2a_2
                safety_buffer = self.safety_buffer
            else:
                lambda_cbf_1 = self.lambda_cbf_a2l_1
                lambda_cbf_2 = self.lambda_cbf_a2l_2
                safety_buffer = 0
            # Compute h
            h = sm - safety_buffer

            # Compute dot_h
            dot_h = grad_sm @ dstate_time

            # Compute ddot_h
            ddot_h = grad_sm @ ddstate_time + dstate_time.T @ hessian_sm @ dstate_time

            # First-order CBF condition
            cbf_condition_1 = dot_h * self.dt + lambda_cbf_1 * h
            # Second-order CBF condition
            cbf_condition_2 = (
                ddot_h * self.dt
                + (lambda_cbf_1 + lambda_cbf_2) * dot_h * self.dt
                + (lambda_cbf_1 * lambda_cbf_2) * h
            )

        return h, dot_h, ddot_h, cbf_condition_1, cbf_condition_2

    def compute_center_state_time_derivatives(
        self, psi, dstate_time_agent, ddx, ddy, ddpsi, idx
    ):
        """
        Compute the time derivatives of a shifted center point (front/center/rear)
        on the agent, given the agent's center first and second state derivatives.

        The shifted center is defined by an offset (delta_x, delta_y).

        The transformation of derivatives is:

        First-order derivatives:
            dxc = dx - delta_x·sin(ψ)·dψ - delta_y·cos(ψ)·dψ
            dyc = dy + delta_x·cos(ψ)·dψ - delta_y·sin(ψ)·dψ

        Second-order derivatives:
            ddxc = dx - delta_x·(sin(ψ)·ddψ + cos(ψ)·dψ²) - delta_y·(cos(ψ)·ddψ - sin(ψ)·dψ²)
            ddyc = dy + delta_x·(cos(ψ)·ddψ - sin(ψ)·dψ²) - delta_y·(sin(ψ)·ddψ + cos(ψ)·dψ²)

        Args:
            psi (float): agent heading ψ
            dstate_time_agent (tensor): [dot_x, dot_y, dot_ψ] first time derivative at agent center
            ddx (float): second derivative of x at agent center
            ddy (float): second derivative of y at agent center
            ddpsi (float): second derivative of heading ψ at agent center
            idx (int): index of the sub-center to compute

        Returns:
            dx_center (float): first time derivative of x at the shifted center
            dy_center (float): first time derivative of y at the shifted center
            ddx_center (float): second time derivative of x at the shifted center
            ddy_center (float): second time derivative of y at the shifted center
        """
        psi = psi.item()
        # First time derivative of x, y, and psi
        dx = dstate_time_agent[0].item()
        dy = dstate_time_agent[1].item()
        dpsi = dstate_time_agent[2].item()

        # Offset of the sub-center (e.g., front, center, and rear) relative to the agent center
        delta_x, delta_y = self.rec_cir_approx.centers[idx]

        # First time derivative
        dx_center = dx - delta_x * np.sin(psi) * dpsi - delta_y * np.cos(psi) * dpsi
        dy_center = dy + delta_x * np.cos(psi) * dpsi - delta_y * np.sin(psi) * dpsi

        # Second time derivative
        ddx_center = (
            ddx
            - delta_x * (np.sin(psi) * ddpsi + np.cos(psi) * dpsi * dpsi)
            - delta_y * (np.cos(psi) * ddpsi - np.sin(psi) * dpsi * dpsi)
        )
        ddy_center = (
            ddy
            + delta_x * (np.cos(psi) * ddpsi - np.sin(psi) * dpsi * dpsi)
            - delta_y * (np.sin(psi) * ddpsi + np.cos(psi) * dpsi * dpsi)
        )

        dpsi_center = dpsi
        ddpsi_center = ddpsi

        return dx_center, dy_center, dpsi_center, ddx_center, ddy_center, ddpsi_center

    def compute_auxiliary_state(self, v1):
        # Update the auxiliary state according to auxiliary system dynamics
        self.p1 = self.p1 + v1

    def solve_decentralized_cbf_qp(self, tensordict, mask_higher_priority_agents):
        self.step += 1
        # Load cbf observation for cbf-qp solving
        self.load_observation(tensordict, mask_higher_priority_agents)

        # Store the original RL action in the tensor, which may be changed later
        tensordict[("agents", "info", "nominal_action")][
            self.env_idx, self.agent_idx
        ] = self.rl_actions_nominal_agent_list[self.agent_idx]

        # Optimization variable control action [accleration, steering rate]
        u_agent = cp.Variable(2)

        # Auxiliary variable
        v1 = cp.Variable(1)
        p2 = cp.Variable(1)
        delta1 = cp.Variable(1)

        # Agent state
        state_agent = self.state_agent_list[self.agent_idx]

        # Auxiliary state
        p1 = self.p1

        # Compute agent's state first and second time derivatives.
        dstate_time_agent, ddx, ddy, ddpsi = self.compute_agent_state_time_derivatives(
            state_agent, u_agent
        )

        # Compute center circles which will be considered for distance to boundary
        circle_centers = self.get_circle_centers(state_agent)  # (num_circles, 2)

        # List for CBF conditions with left and right boundary
        cbf_condition_2_left_b_list = []
        cbf_condition_2_right_b_list = []

        # Time for pseudo distance calculation
        self.time_pseudo_dis = 0

        # Iterate over all circle centers
        for i in range(circle_centers.shape[0]):
            # Compute circle center states' first and second time derivatives.
            (
                dx_circle,
                dy_circle,
                dpsi_circle,
                ddx_circle,
                ddy_circle,
                ddpsi_circle,
            ) = self.compute_center_state_time_derivatives(
                state_agent[2], dstate_time_agent, ddx, ddy, ddpsi, i
            )

            # Estimate safety margin between agent and lane boundary
            (
                sm_left_b,
                grad_left_b,
                hessian_left_b,
                sm_right_b,
                grad_right_b,
                hessian_right_b,
            ) = self.estimate_agent_2_lane_safety_margin(
                circle_centers[i][0:2],
                self.lanelet_id_list[self.agent_idx],
                self.path_id_list[self.agent_idx],
            )

            # Compute CBF conditions for distance to left boundary
            (
                h_left_b,
                dot_h_left_b,
                ddot_h_left_b,
                cbf_condition_1_left_b,
                cbf_condition_2_left_b,
            ) = self.compute_cbf_conditions(
                np.array([dx_circle, dy_circle]),
                [ddx_circle, ddy_circle],
                sm_left_b,
                grad_left_b,
                hessian_left_b,
                p2,
                p1,
                v1,
                a2a=False,
            )

            # Compute CBF conditions for distance to right boundary
            (
                h_right_b,
                dot_h_right_b,
                ddot_h_right_b,
                cbf_condition_1_right_b,
                cbf_condition_2_right_b,
            ) = self.compute_cbf_conditions(
                np.array([dx_circle, dy_circle]),
                [ddx_circle, ddy_circle],
                sm_right_b,
                grad_right_b,
                hessian_right_b,
                p2,
                p1,
                v1,
                a2a=False,
            )
            if self.step == 1:
                assert h_left_b > 0
                assert h_right_b > 0
                assert cbf_condition_1_left_b > 0
                assert cbf_condition_1_right_b > 0

            cbf_condition_2_left_b_list.append(cbf_condition_2_left_b)
            cbf_condition_2_right_b_list.append(cbf_condition_2_right_b)

        if self.num_higher_prio_nearing_agents > 0:
            # List for CBF conditions with each considered higher priority surounding angents
            # h_ji_list = []
            # cbf_condition_1_ji_list = []
            cbf_condition_2_ji_list = []

            # State of other agents
            state_others = self.state_others_list[
                self.agent_idx
            ]  # (n_higher_prio_nearing_agents, 5)

            # Iterate over all considered higher order nearing agents
            for j in range(self.num_higher_prio_nearing_agents):
                # Compute relative state time derivatives
                (
                    dstate_time_ji_ego,
                    ddstate_time_ji_ego,
                    dstate_time_ij_ego,
                    ddstate_time_ij_ego,
                ) = self.compute_relative_state_time_derivatives(
                    state_agent,
                    dstate_time_agent,
                    ddx,
                    ddy,
                    ddpsi,
                    self.u_others[j],
                    state_others[j],
                )

                # Compute relative poses between vehicles i and j in ego coordinate system
                x_ji, y_ji, psi_ji, _ = self.compute_relative_poses(
                    state_agent[0],
                    state_agent[1],
                    state_agent[2],
                    state_others[j, 0],
                    state_others[j, 1],
                    state_others[j, 2],
                )

                # Estimate safety margin between agents
                (
                    sm_ji,
                    grad_sm_ji,
                    hessian_sm_ji,
                ) = self.estimate_agent_2_agent_safety_margin(x_ji, y_ji, psi_ji)

                # Compute CBF conditions for distance to other agents
                (
                    h_ji,
                    dot_h_ji,
                    ddot_h_ji,
                    cbf_condition_1_ji,
                    cbf_condition_2_ji,
                ) = self.compute_cbf_conditions(
                    dstate_time_ji_ego,
                    ddstate_time_ji_ego,
                    sm_ji,
                    grad_sm_ji,
                    hessian_sm_ji,
                    p2,
                    p1,
                    v1,
                    a2a=True,
                )
                if self.step == 1:
                    assert h_ji > 0
                    assert cbf_condition_1_ji > 0
                cbf_condition_2_ji_list.append(cbf_condition_2_ji)

        # CBF Quaratic problem objective: Minimize weighted squared deviation from nominal control inputs
        if self.use_pacbf:
            objective = cp.Minimize(
                math.exp(-12)
                * cp.quad_form(
                    u_agent[:]
                    - self.u_nominal_agent_list[self.agent_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten(),
                    self.Q,
                )
                + cp.square(v1)
                + cp.square(delta1)
                + cp.square(p2 - self.p2_0)
            )
        else:
            objective = cp.Minimize(
                cp.quad_form(
                    u_agent[:]
                    - self.u_nominal_agent_list[self.agent_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten(),
                    self.Q,
                )
            )

        # Pysical limit constraints
        constraints = [
            self.a_min <= u_agent[0],
            u_agent[0] <= self.a_max,
            self.steering_rate_min <= u_agent[1],
            u_agent[1] <= self.steering_rate_max,
        ]

        # Constraints for distance to boundaries
        constraints += [
            cbf_condition_2_left_b_list[i] >= 0
            for i in range(len(cbf_condition_2_left_b_list))
        ]
        constraints += [
            cbf_condition_2_right_b_list[i] >= 0
            for i in range(len(cbf_condition_2_right_b_list))
        ]

        if self.num_higher_prio_nearing_agents > 0:
            # Constraints for distance to other agents
            constraints += [
                cbf_condition_2_ji_list[i] >= 0
                for i in range(len(cbf_condition_2_ji_list))
            ]

        if self.use_pacbf:
            # Constraints for feasibility improvement in PACBF
            cbf_condition_0_p2 = p2
            cbf_condition_1_p1_min = v1 + 0.2 * (p1 - 1e-6)
            cbf_condition_1_p1_max = -v1 + 0.5 * (1 - (1e-6) - p1)
            clf_condtion_1_p1 = (
                (p1 + v1 - self.p1_0) ** 2 - (p1 - self.p1_0) ** 2 + (p1) ** 2
            )

            constraints_auxi = [
                cbf_condition_0_p2 >= 1e-6,
                cbf_condition_0_p2 <= 1 - (1e-6),
                cbf_condition_1_p1_min >= 1e-6,
                cbf_condition_1_p1_max >= 1e-6,
                clf_condtion_1_p1 <= delta1,
            ]

            constraints += constraints_auxi

        # Formulate the optimization problem
        prob = cp.Problem(objective, constraints)

        # Solve optimization problem
        t_start = time.time()
        prob.solve(
            solver=cp.SCS,
            verbose=False,
        )
        opt_duration = time.time() - t_start

        if prob.status == cp.OPTIMAL and self.use_pacbf:
            # Update auxiliary state
            self.compute_auxiliary_state(v1.value.item())

        # Auxiliary variable value, which can be exported
        self.out_p2 = p2.value
        self.out_p1 = p1
        self.out_v1 = v1.value

        if prob.status != cp.OPTIMAL:
            print(f"Warning: QP not solved optimally. Status: {prob.status}")
            u_cbf = self.u_nominal_agent_list[self.agent_idx].flatten()
            self.num_fail += 1
            # If there is no valid solution for the optimization problem, take the original action
            tensordict[("agents", "action")][
                self.env_idx, self.agent_idx
            ] = self.rl_actions_nominal_agent_list[self.agent_idx]
        elif prob.status == cp.OPTIMAL:
            u_cbf = u_agent.value
            rl_actions_safe = self.u_to_rl_action(
                u_cbf, state_agent[3], state_agent[4]
            ).detach()
            # Rewrite the safe action
            tensordict[("agents", "action")][
                self.env_idx, self.agent_idx
            ] = rl_actions_safe

    def solve_centralized_cbf_qp(self, tensordict, policy):
        pass  # To be implemented
