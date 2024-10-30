import numpy as np
import torch

from abc import ABC, abstractmethod


class Controller(ABC):
    """
    Base class for controllers.
    """

    def __init__(
        self, dt, a_min, a_max, steering_rate_min, steering_rate_max, device="cpu"
    ):
        self.dt = dt
        self.a_min = a_min
        self.a_max = a_max
        self.steering_rate_min = steering_rate_min
        self.steering_rate_max = steering_rate_max
        self.device = device
        self.u = torch.zeros((2), device=self.device, dtype=torch.float32)

    @abstractmethod
    def get_actions(self, state, target_speed, target_pos):
        """
        Abstract method to compute control actions.
        Args:
            state: [x, y, yaw, speed, steering]
            target_speed: Target speed
            target_pos: [x, y]
        Returns:
            torch.Tensor: Control actions (acceleration, steering_rate)
        """
        raise NotImplementedError()


class PIDController(Controller):
    """
    PID controller for controlling speed and steering rate.
    """

    def __init__(
        self, dt, a_min, a_max, steering_rate_min, steering_rate_max, device="cpu"
    ):
        super().__init__(dt, a_min, a_max, steering_rate_min, steering_rate_max, device)

        # PID gains for speed control
        self.kp_speed = 1.0
        self.ki_speed = 0.1
        self.kd_speed = 0.05

        # PID gains for steering control
        self.kp_steering = 1.0
        self.ki_steering = 0.1
        self.kd_steering = 0.05

        # Integral and previous error terms
        self.integral_speed = 0.0
        self.previous_error_speed = 0.0
        self.integral_steering = 0.0
        self.previous_error_steering = 0.0

    def reset(self):
        """
        Reset the integral and previous error terms for the PID controllers.
        """
        self.integral_speed = 0.0
        self.previous_error_speed = 0.0
        self.integral_steering = 0.0
        self.previous_error_steering = 0.0

    def get_actions(self, state, target_speed, target_pos):
        """
        PID control to set acceleration and steering rate to reach target speed and steer towards target position.
        Args:
            state: [x, y, yaw, speed, steering]
            target_speed: Target speed
            target_pos: [x, y]
        Returns:
            torch.Tensor: Control actions (acceleration, steering_rate)
        """
        current_speed = state[3]

        # PID control for speed
        error_speed = target_speed - current_speed
        self.integral_speed += error_speed * self.dt
        derivative_speed = (error_speed - self.previous_error_speed) / self.dt
        acceleration = (
            self.kp_speed * error_speed
            + self.ki_speed * self.integral_speed
            + self.kd_speed * derivative_speed
        )
        self.previous_error_speed = error_speed

        # Clamp acceleration to a_min and a_max
        acceleration = np.clip(acceleration, self.a_min, self.a_max)

        # Compute heading to the target
        dx = target_pos[0] - state[0]
        dy = target_pos[1] - state[1]
        heading_to_target = np.arctan2(dy, dx)
        current_steering = state[4]

        # PID control for steering
        error_steering = heading_to_target - current_steering
        error_steering = (error_steering + np.pi) % (
            2 * np.pi
        ) - np.pi  # Normalize to [-pi, pi]
        self.integral_steering += error_steering * self.dt
        derivative_steering = (error_steering - self.previous_error_steering) / self.dt
        steering_rate = (
            self.kp_steering * error_steering
            + self.ki_steering * self.integral_steering
            + self.kd_steering * derivative_steering
        )
        self.previous_error_steering = error_steering

        # Clamp steering_rate to steering_rate limits
        steering_rate = np.clip(
            steering_rate, self.steering_rate_min, self.steering_rate_max
        )

        # Return actions as a torch tensor
        self.u[:] = torch.tensor(
            [acceleration, steering_rate], dtype=torch.float32, device=self.device
        )


class ConstantController(Controller):
    """
    Controller that returns constant zero actions (no movement).
    """

    def __init__(
        self, dt, a_min, a_max, steering_rate_min, steering_rate_max, device="cpu"
    ):
        super().__init__(dt, a_min, a_max, steering_rate_min, steering_rate_max, device)

    def get_actions(self):
        """
        Return constant zero actions.
        Args:
            state: [x, y, yaw, speed, steering]
            target_speed: Target speed (unused)
            target_pos: [x, y] (unused)
        Returns:
            torch.Tensor: Control actions (acceleration, steering_rate)
        """
        self.u[:] = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device)


class SimpleTargetFollowingController(Controller):
    """
    Simple controller for following a target position.
    """

    def __init__(
        self, dt, a_min, a_max, steering_rate_min, steering_rate_max, device="cpu"
    ):
        super().__init__(dt, a_min, a_max, steering_rate_min, steering_rate_max, device)

    def get_actions(self, state, target_speed, target_pos):
        """
        Compute control actions to follow a target position at a given speed.
        Args:
            state: [x, y, yaw, speed, steering]
            target_speed: Target speed
            target_pos: [x, y]
        Returns:
            torch.Tensor: Control actions (acceleration, steering_rate)
        """
        current_speed = state[3]
        acceleration = (target_speed - current_speed) / self.dt

        # Compute heading to the target
        dx = target_pos[0] - state[0]
        dy = target_pos[1] - state[1]
        heading_to_target = np.arctan2(dy, dx)
        current_steering = state[4]

        d_steering = heading_to_target - current_steering
        d_steering = (d_steering + np.pi) % (
            2 * np.pi
        ) - np.pi  # Normalize to [-pi, pi]
        steering_rate = d_steering / self.dt

        # Clamp steering_rate and acceleration
        steering_rate = np.clip(
            steering_rate, self.steering_rate_min, self.steering_rate_max
        )
        acceleration = np.clip(acceleration, self.a_min, self.a_max)

        # Return actions as a torch tensor
        self.u[:] = torch.tensor(
            [acceleration, steering_rate], dtype=torch.float32, device=self.device
        )


# Example usage
if __name__ == "__main__":
    controller = PIDController(
        dt=0.5,
        a_min=-4.0,
        a_max=4.0,
        steering_rate_min=-2 * np.pi,
        steering_rate_max=2 * np.pi,
        device="cpu",
    )
    state = [0, 0, 0, 1.0, 0]  # [x, y, yaw, speed, steering]
    target_speed = 2.0  # m/s
    target_pos = [1, 1]  # [x, y]
    controller.get_actions(state, target_speed, target_pos)
    print("Nominal control input:", controller.u)
