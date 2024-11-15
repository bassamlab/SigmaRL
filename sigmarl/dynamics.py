# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchdiffeq import odeint
from vmas.simulator.dynamics.common import Dynamics


class KinematicBicycleModel(Dynamics):
    """
    Kinematic Bicycle Model implemented using PyTorch for batch processing across different environments.
    """

    def __init__(
        self,
        l_f=0.08,
        l_r=0.08,
        max_speed=1.0,
        min_speed=-0.5,
        max_steering=torch.pi / 4,
        min_steering=-torch.pi / 4,
        max_acc=4.0,
        min_acc=-4.0,
        max_steering_rate=2 * torch.pi,
        min_steering_rate=-2 * torch.pi,
        device="cpu",
    ):
        """
        Initialize the KinematicBicycleModel.

        Parameters:
            l_f (float): Front wheelbase, distance from the center of gravity to the front axle [m].
            l_r (float): Rear wheelbase, distance from the center of gravity to the rear axle [m].
            max_speed (float): Maximum speed of the vehicle [m/s].
            max_steering (float): Maximum steering angle [radians].
            device (str): Device to store torch variables, e.g., "cpu" or "cuda".
        """
        super().__init__()

        self.l_f = l_f
        self.l_r = l_r
        self.l_wb = l_f + l_r  # Wheelbase

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steering = max_steering
        self.min_steering = min_steering

        self.max_acc = max_acc
        self.min_acc = min_acc
        self.max_steering_rate = max_steering_rate
        self.min_steering_rate = min_steering_rate

        self.device = device  # Where to store torch variable

    @property
    def needed_action_size(self) -> int:
        return 2

    def ode(self, _, x, u):
        """
        Defines the ODE for the Kinematic Bicycle Model to compute time derivatives for batch processing across different environments.

        Parameters:
            _ (float): Placeholder for time (unused, as the model is time-invariant).
            x (torch.Tensor): Current state tensor of shape [batch_size, 5] where batch_size is the number of environments.
                x[:, 0] = x position [m]
                x[:, 1] = y position [m]
                x[:, 2] = yaw angle [radians]
                x[:, 3] = speed [m/s]
                x[:, 4] = steering angle [radians]
            u (torch.Tensor): Input tensor of shape [batch_size, 2] where batch_size is the number of environments.
                u[:, 0] = acceleration [m/s²]
                u[:, 1] = steering rate [radians/s]

        Returns:
            dx (torch.Tensor): Time derivative of the state tensor of shape [batch_size, 5].
        """
        # The first dimension is for the batch size (even if the batch size is one)
        if u.ndim == 1:
            u = u.unsqueeze(0)
            is_batch_size_none = True
        else:
            is_batch_size_none = False

        if x.ndim == 1:
            x = x.unsqueeze(0)
            is_batch_size_none = True
        else:
            is_batch_size_none = False

        # Apply state limits
        # x[:, 3] = torch.clamp(x[:, 3], self.min_speed, self.max_speed)
        # x[:, 4] = torch.clamp(x[:, 4], self.min_steering, self.max_steering)

        # Apply action limits
        # u[:, 0] = torch.clamp(u[:, 0], self.min_acc, self.max_acc)
        # u[:, 1] = torch.clamp(u[:, 1], self.min_steering_rate, self.max_steering_rate)

        # Calculate parameters
        beta = torch.atan(self.l_r / self.l_wb * torch.tan(x[:, 4]))

        # Compute state derivatives
        dx = torch.zeros_like(x, device=self.device)
        dx[:, 0] = x[:, 3] * torch.cos(x[:, 2] + beta)  # dx/dt
        dx[:, 1] = x[:, 3] * torch.sin(x[:, 2] + beta)  # dy/dt
        dx[:, 2] = (
            (x[:, 3] / self.l_wb) * torch.tan(x[:, 4]) * torch.cos(beta)
        )  # dpsi/dt
        dx[:, 3] = u[:, 0]  # dv/dt
        dx[:, 4] = u[:, 1]  # ddelta/dt

        if is_batch_size_none:
            return dx.squeeze(0)  # Remove the first dimension
        else:
            return dx

    def step(self, x0, u, dt, tick_per_step=5):
        """
        Perform a discrete integration step using the ODE solver for batch processing across different environments.

        Parameters:
            x0 (torch.Tensor): Initial state tensor of shape [batch_size, 5] where batch_size is the number of environments.
            u (torch.Tensor): Input tensor of shape [batch_size, 2] where batch_size is the number of environments.
            dt (float): Sample time [s].
            tick_per_step (int): Number of integration ticks within the step.

        Returns:
            x (torch.Tensor): Final state tensor after integration of shape [batch_size, 5].
            v (torch.Tensor): Velocity tensor of shape [batch_size, 2] where v[:, 0] is velocity in x and v[:, 1] is velocity in y.
        """
        # The first dimension is for the batch size (even if the batch size is one)
        if u.ndim == 1:
            u = u.unsqueeze(0)
            is_batch_size_none = True
        else:
            is_batch_size_none = False

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)
            is_batch_size_none = True
        else:
            is_batch_size_none = False

        # Create a time tensor from 0 to dt with (tick_per_step + 1) points
        t = torch.linspace(0, dt, steps=tick_per_step + 1, device=self.device)

        # Define a closure to pass the input u to the ODE function
        def model(t, x):
            return self.ode(t, x, u)

        # Integrate the ODE
        x = odeint(model, x0, t, rtol=1e-8, atol=1e-8)

        x[..., 4] = (x[..., 4] + torch.pi) % (2 * torch.pi) - torch.pi  # [-pi, pi]

        # Calculate beta (also called sideslip angle) for the final state
        beta = torch.atan(self.l_r / self.l_wb * torch.tan(x[-1][:, 4]))

        # Compute velocity components
        course_angle = x[-1][:, 2] + beta
        velocity_x = x[-1][:, 3] * torch.cos(course_angle)  # Velocity in x direction
        velocity_y = x[-1][:, 3] * torch.sin(course_angle)  # Velocity in y direction
        velocity = torch.stack(
            (velocity_x, velocity_y), dim=1
        )  # Stack to create [batch_size, 2] tensor

        # Return the final state and velocity
        if is_batch_size_none:
            return (
                x[-1].squeeze(0),
                beta.squeeze(0),
                velocity.squeeze(0),
            )  # Remove the first dimension
        else:
            return x[-1], beta, velocity

    def process_action(self):
        pass


if __name__ == "__main__":
    # Simple example with three vehicles
    import torch

    # Initialize the BicycleModel with l_f = l_r = 0.08 meters
    model = KinematicBicycleModel(l_f=0.08, l_r=0.08)

    # Define simulation parameters
    tick_per_step = 5
    dt_seconds = 0.05  # 50 milliseconds
    batch_size = 3  # Number of environments to simulate

    # Initial state tensor: [batch_size, 5] where 5 is [x, y, yaw, speed, steering]
    x0 = torch.zeros(batch_size, 5)
    x0[:, 3] = 1.0  # Set initial speed to 1.0 for all environments
    x0[0, 3] = -1.0

    # Number of simulation steps
    num_steps = 100

    # Define inputs: [batch_size, 2] where 2 is [acceleration, steering_rate]
    acceleration = torch.zeros(batch_size, 1)  # meters per second squared
    steering_rate = torch.zeros(batch_size, 1) * 0.1  # radians per second
    steering_rate[1] += 0.1  # radians per second
    steering_rate[2] += 0.2  # radians per second
    u = torch.cat([acceleration, steering_rate], dim=1)

    # Initialize a list to store states
    states = [x0]

    # Current state
    current_state = x0

    # Simulate over multiple steps
    for step_idx in range(num_steps):
        print(f"Step: {step_idx}")
        current_state, _, _ = model.step(current_state, u, dt_seconds, tick_per_step)
        states.append(current_state)

    # Convert list of states to a tensor for easier handling
    states = torch.stack(states)

    # Example: Print the state at the final step for each environment
    final_states = states[-1]

    for i in range(batch_size):
        print(f"\nFinal state after {num_steps} steps for Environment {i}:")
        print(f"x position: {final_states[i, 0].item():.4f} m")
        print(f"y position: {final_states[i, 1].item():.4f} m")
        print(f"yaw angle: {final_states[i, 2].item():.4f} rad")
        print(f"speed: {final_states[i, 3].item():.4f} m/s")
        print(f"steering angle: {final_states[i, 4].item():.4f} rad")

    import matplotlib.pyplot as plt

    # Plot the trajectory for each environment
    plt.figure(figsize=(8, 6))
    for i in range(batch_size):
        x_positions = states[:, i, 0].numpy()
        y_positions = states[:, i, 1].numpy()
        plt.plot(
            x_positions, y_positions, marker="o", markersize=2, label=f"Environment {i}"
        )

    plt.title("Vehicle Trajectories")
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()
