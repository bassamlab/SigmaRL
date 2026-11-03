# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields
import torch


##################################################
## Custom Classes
##################################################
class Normalizers:
    def __init__(
        self,
        pos=None,
        pos_world=None,
        v=None,
        rot=None,
        steering=None,
        distance_lanelet=None,
        distance_agent=None,
        distance_ref=None,
    ):
        self.pos = pos
        self.pos_world = pos_world
        self.v = v
        self.rot = rot
        self.steering = steering
        self.distance_lanelet = distance_lanelet
        self.distance_agent = distance_agent
        self.distance_ref = distance_ref


class Rewards:
    def __init__(
        self,
        progress=None,
        weighting_ref_directions=None,
        higth_v=None,
        reach_goal=None,
        reach_intermediate_goal=None,
    ):
        self.progress = progress
        self.weighting_ref_directions = weighting_ref_directions
        self.higth_v = higth_v
        self.reach_goal = reach_goal
        self.reach_intermediate_goal = reach_intermediate_goal


class Penalties:
    def __init__(
        self,
        deviate_from_ref_path=None,
        deviate_from_goal=None,
        near_boundary=None,
        near_other_agents=None,
        collide_with_agents=None,
        near_collision_reset=None,
        collide_with_boundaries=None,
        collide_with_obstacles=None,
        leave_world=None,
        time=None,
        change_steering=None,
        deviate_from_cbf_vel=None,
        deviate_from_cbf_steer=None,
        deadlock=None,
        jerk_acceleration=None,
        jerk_steering_rate=None,
        high_acceleration=None,
        high_steering_rate=None,
    ):
        self.deviate_from_ref_path = (
            deviate_from_ref_path  # Penalty for deviating from reference path
        )
        self.deviate_from_goal = (
            deviate_from_goal  # Penalty for deviating from goal position
        )
        self.near_boundary = (
            near_boundary  # Penalty for being too close to lanelet boundaries
        )
        self.near_other_agents = (
            near_other_agents  # Penalty for being too close to other agents
        )
        self.collide_with_agents = (
            collide_with_agents  # Penalty for colliding with other agents
        )
        self.near_collision_reset = (
            near_collision_reset
            # One-time penalty for nearby agents reset after a collision.
        )
        self.collide_with_boundaries = (
            collide_with_boundaries  # Penalty for colliding with lanelet boundaries
        )
        self.collide_with_obstacles = (
            collide_with_obstacles  # Penalty for colliding with obstacles
        )
        self.leave_world = leave_world  # Penalty for leaving the world
        self.time = time  # Penalty for losing time
        self.change_steering = (
            change_steering  # Penalty for changing steering direction
        )
        self.deviate_from_cbf_vel = (
            deviate_from_cbf_vel  # Penalty for deviating from CBF suggested speed
        )
        self.deviate_from_cbf_steer = (
            deviate_from_cbf_steer  # Penalty for deviating from CBF suggested steering
        )
        self.deadlock = deadlock  # Penalty for deadlock
        self.jerk_acceleration = jerk_acceleration
        self.jerk_steering_rate = jerk_steering_rate
        self.high_acceleration = high_acceleration
        self.high_steering_rate = high_steering_rate


@dataclass
class RewardInfo:
    rew_progress: torch.Tensor
    rew_reach_goal: torch.Tensor
    rew_speed: torch.Tensor
    rew_time: torch.Tensor
    rew_centerline: torch.Tensor
    rew_collide_other_agents: torch.Tensor
    rew_collide_lane: torch.Tensor
    rew_energy_acceleration: torch.Tensor
    rew_energy_steering: torch.Tensor
    rew_jerk_acceleration: torch.Tensor
    rew_jerk_steering: torch.Tensor
    rew_deadlock: torch.Tensor
    rew_near_collision_reset: torch.Tensor
    rew_parameter_independent: torch.Tensor
    rew_total: torch.Tensor
    # The fields below are possibly computed ourside the reward function and will not be reset by the reset() function
    rew_near_other_agents: torch.Tensor
    rew_near_left_lane: torch.Tensor
    rew_near_right_lane: torch.Tensor
    rew_cbf_lambda: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int, n_agents: int, device=None, dtype=torch.float32):
        """
        Automatically create zero tensors for all fields.
        """
        kwargs = {}
        for f in fields(cls):
            kwargs[f.name] = torch.zeros(
                (batch_size, n_agents), device=device, dtype=dtype
            )
        return cls(**kwargs)

    def reset(self):
        """
        Reset all tensors to zero in place.
        """
        for f in fields(self):
            if f.name in [
                "rew_progress",
                "rew_reach_goal",
                "rew_speed",
                "rew_time",
                "rew_centerline",
                "rew_collide_other_agents",
                "rew_collide_lane",
                "rew_energy_acceleration",
                "rew_energy_steering",
                "rew_jerk_acceleration",
                "rew_jerk_steering",
                "rew_deadlock",
                "rew_near_collision_reset",
                "rew_parameter_independent",
                "rew_total",
            ]:
                getattr(self, f.name).zero_()


class Thresholds:
    def __init__(
        self,
        deviate_from_ref_path=None,
        near_boundary_low=None,
        near_boundary_high=None,
        ttc_low=None,
        ttc_high=None,
        near_other_agents_low=None,
        near_other_agents_high=None,
        reach_goal=None,
        reach_intermediate_goal=None,
        change_steering=None,
        no_reward_if_too_close_to_boundaries=None,
        no_reward_if_too_close_to_other_agents=None,
        rl_obs_range=None,
        distance_mask_agents=None,
    ):
        self.deviate_from_ref_path = deviate_from_ref_path
        self.near_boundary_low = near_boundary_low
        self.near_boundary_high = near_boundary_high
        self.ttc_low = ttc_low
        self.ttc_high = ttc_high
        self.near_other_agents_low = near_other_agents_low
        self.near_other_agents_high = near_other_agents_high
        self.reach_goal = reach_goal  # Threshold less than which agents are considered at their goal positions
        self.reach_intermediate_goal = reach_intermediate_goal  # Threshold less than which agents are considered at their intermediate goal positions
        self.change_steering = change_steering
        self.no_reward_if_too_close_to_boundaries = no_reward_if_too_close_to_boundaries  # Agents get no reward if they are too close to lanelet boundaries
        self.no_reward_if_too_close_to_other_agents = no_reward_if_too_close_to_other_agents  # Agents get no reward if they are too close to other agents
        self.rl_obs_range = (
            rl_obs_range  # RL observation range above which nearing agents are masked
        )
        self.distance_mask_agents = distance_mask_agents


class ReferencePathsMapRelated:
    def __init__(
        self,
        long_term_all=None,
        long_term_intersection=None,
        long_term_merge_in=None,
        long_term_merge_out=None,
        point_extended_all=None,
        point_extended_intersection=None,
        point_extended_merge_in=None,
        point_extended_merge_out=None,
        long_term_vecs_normalized=None,
        point_extended=None,
        sample_interval=None,
        long_term_intersection_extended=None,
        point_extended_intersection_extended=None,
    ):
        self.long_term_all = long_term_all  # All long-term reference paths
        self.long_term_intersection = long_term_intersection  # Long-term reference paths for the intersection scenario
        self.long_term_intersection_extended = long_term_intersection_extended
        self.long_term_merge_in = (
            long_term_merge_in  # Long-term reference paths for the mergin in scenario
        )
        self.long_term_merge_out = (
            long_term_merge_out  # Long-term reference paths for the merge out scenario
        )
        self.point_extended_all = point_extended_all  # Extend the long-term reference paths by one point at the end
        self.point_extended_intersection = point_extended_intersection  # Extend the long-term reference paths by one point at the end
        self.point_extended_intersection_extended = point_extended_intersection_extended
        self.point_extended_merge_in = point_extended_merge_in  # Extend the long-term reference paths by one point at the end
        self.point_extended_merge_out = point_extended_merge_out  # Extend the long-term reference paths by one point at the end

        self.long_term_vecs_normalized = long_term_vecs_normalized  # Normalized vectors of the line segments on the long-term reference path
        self.point_extended = point_extended  # Extended point for a non-loop reference path (address the oscillations near the goal)
        self.sample_interval = sample_interval  # Integer, sample interval from the long-term reference path for the short-term reference paths
        # TODO Instead of an integer, sample with a fixed distance from the long-term reference paths when reading the map data


class ReferencePathsAgentRelated:
    def __init__(
        self,
        long_term: torch.Tensor = None,
        long_term_vec_normalized: torch.Tensor = None,
        point_extended: torch.Tensor = None,
        left_boundary: torch.Tensor = None,
        right_boundary: torch.Tensor = None,
        entry: torch.Tensor = None,
        exit: torch.Tensor = None,
        n_points_long_term: torch.Tensor = None,
        n_points_left_b: torch.Tensor = None,
        n_points_right_b: torch.Tensor = None,
        is_loop: torch.Tensor = None,
        n_points_nearing_boundary: torch.Tensor = None,
        nearing_points_left_boundary: torch.Tensor = None,
        nearing_points_right_boundary: torch.Tensor = None,
        short_term: torch.Tensor = None,
        short_term_indices: torch.Tensor = None,
        scenario_id: torch.Tensor = None,
        path_id: torch.Tensor = None,
        point_id: torch.Tensor = None,
        ref_lanelet_ids: torch.Tensor = None,
        n_ref_lanelet_ids: torch.Tensor = None,
        ref_lanelet_segment_points: torch.Tensor = None,
    ):
        self.long_term = long_term  # Actual long-term reference paths of agents
        self.long_term_vec_normalized = (
            long_term_vec_normalized  # Normalized vectories on the long-term trajectory
        )
        self.point_extended = point_extended
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.entry = entry  # [for non-loop path only] Line segment of entry
        self.exit = exit  # [for non-loop path only] Line segment of exit
        self.is_loop = is_loop  # Whether the reference path is a loop
        self.n_points_long_term = (
            n_points_long_term  # The number of points on the long-term reference paths
        )
        self.n_points_left_b = n_points_left_b  # The number of points on the left boundary of the long-term reference paths
        self.n_points_right_b = n_points_right_b  # The number of points on the right boundary of the long-term reference paths
        self.short_term = short_term  # Short-term reference path
        self.short_term_indices = short_term_indices  # Indices that indicate which part of the long-term reference path is used to build the short-term reference path
        self.n_points_nearing_boundary = n_points_nearing_boundary  # Number of points on nearing boundaries to be observed
        self.nearing_points_left_boundary = (
            nearing_points_left_boundary  # Nearing left boundary
        )
        self.nearing_points_right_boundary = (
            nearing_points_right_boundary  # Nearing right boundary
        )

        self.scenario_id = scenario_id  # Which scenarios agents are (current implementation includes (1) intersection, (2) merge-in, and (3) merge-out)
        self.path_id = path_id  # Which paths agents are
        self.point_id = point_id  # Which points agents are

        self.ref_lanelet_ids = ref_lanelet_ids  # Lanelet IDs of the reference path
        self.n_ref_lanelet_ids = (
            n_ref_lanelet_ids  # Number of lanelets in the reference path
        )
        self.ref_lanelet_segment_points = (
            ref_lanelet_segment_points  # Connection points between lanelets
        )


class Distances:
    def __init__(
        self,
        type=None,
        agents=None,
        left_boundaries=None,
        right_boundaries=None,
        boundaries=None,
        ref_paths=None,
        closest_point_on_ref_path=None,
        closest_point_on_left_b=None,
        closest_point_on_right_b=None,
        goal=None,
        obstacles=None,
    ):
        if (type is not None) & (type not in ["c2c", "mtv"]):
            raise ValueError("Invalid distance type. Must be 'c2c' or 'mtv'.")
        self.type = type  # Distances between agents
        self.agents = agents  # Distances between agents
        self.left_boundaries = left_boundaries  # Distances between agents and the left boundaries of their current lanelets (for each vertex of each agent)
        self.right_boundaries = right_boundaries  # Distances between agents and the right boundaries of their current lanelets (for each vertex of each agent)
        self.boundaries = boundaries  # The minimum distances between agents and the boundaries of their current lanelets
        self.ref_paths = ref_paths  # Distances between agents and the center line of their current lanelets
        self.closest_point_on_ref_path = (
            closest_point_on_ref_path  # Index of the closest point on reference path
        )
        self.closest_point_on_left_b = (
            closest_point_on_left_b  # Index of the closest point on left boundary
        )
        self.closest_point_on_right_b = (
            closest_point_on_right_b  # Index of the closest point on right boundary
        )
        self.goal = goal  # Distances to goal positions
        self.obstacles = obstacles  # Distances to obstacles


class Evaluation:
    # This class stores the data relevant to evaluation of the system-wide performance, which necessitates the information being in the gloabl coordinate system.
    def __init__(
        self,
        pos_traj=None,
        v_traj=None,
        rot_traj=None,
        deviation_from_ref_path=None,
        path_tracking_error_mean=None,
    ):
        self.pos_traj = pos_traj  # Position trajectory
        self.v_traj = v_traj  # Velocity trajectory
        self.rot_traj = rot_traj  # Rotation trajectory
        self.deviation_from_ref_path = deviation_from_ref_path
        self.path_tracking_error_mean = path_tracking_error_mean  # [relevant to path-tracking scenarios] Calculated when an agent reached its goal. The goal reward could be adjusted according to this variable


class Obstacles:
    # This class stores the data relevant to static and dynamic obstacles.
    def __init__(
        self,
        n=None,
        pos=None,
        vertices=None,
        vel=None,
        rot=None,
        length=None,
        width=None,
        center_points=None,
        lengths=None,
        yaws=None,
    ):
        self.n = n  # The number of obstacles
        self.pos = pos  # Position
        self.vertices = vertices  # vertices
        self.vel = vel  # Velocity
        self.rot = rot  # Rotation
        self.length = length  # Length of the dynamic obstacle
        self.width = width  # Width of the dynamic obstacle


class Timer:
    # This class stores the data relevant to static and dynamic obstacles.
    def __init__(
        self,
        step=None,
        start=None,
        end=None,
        step_duration=None,
        step_begin=None,
        render_begin=None,
    ):
        self.step = step  # Count the current time step
        self.start = start  # Time point of simulation start
        self.end = end  # Time point of simulation end
        self.step_duration = step_duration  # Duration of each time step
        self.step_begin = step_begin  # Time when the current time step begins
        self.render_begin = (
            render_begin  # Time when the rendering of the current time step begins
        )


class Collisions:
    def __init__(
        self,
        with_obstacles: torch.Tensor = None,
        with_agents: torch.Tensor = None,
        with_lanelets: torch.Tensor = None,
        with_entry_segments: torch.Tensor = None,
        with_exit_segments: torch.Tensor = None,
    ):
        self.with_agents = with_agents  # Whether collide with agents
        self.with_obstacles = with_obstacles  # Whether collide with obstacles
        self.with_lanelets = with_lanelets  # Whether collide with lanelet boundaries
        self.with_entry_segments = (
            with_entry_segments  # Whether collide with entry segments
        )
        self.with_exit_segments = (
            with_exit_segments  # Whether collide with exit segments
        )


class Constants:
    # Predefined constants that may be used during simulations
    def __init__(
        self,
        env_idx_broadcasting: torch.Tensor = None,
        empty_action_vel: torch.Tensor = None,
        empty_action_steering: torch.Tensor = None,
        mask_pos: torch.Tensor = None,
        mask_vel: torch.Tensor = None,
        mask_rot: torch.Tensor = None,
        mask_zero: torch.Tensor = None,
        mask_one: torch.Tensor = None,
        reset_agent_min_distance: torch.Tensor = None,
    ):
        self.env_idx_broadcasting = env_idx_broadcasting
        self.empty_action_vel = empty_action_vel
        self.empty_action_steering = empty_action_steering
        self.mask_pos = mask_pos
        self.mask_zero = mask_zero
        self.mask_one = mask_one
        self.mask_vel = mask_vel
        self.mask_rot = mask_rot
        self.reset_agent_min_distance = reset_agent_min_distance  # The minimum distance between agents when being reset


class Prioritization:
    def __init__(
        self,
        values: torch.Tensor = None,
    ):
        self.values = values


class CircularBuffer:
    def __init__(
        self,
        buffer: torch.Tensor = None,
    ):
        """Initializes a circular buffer to store initial states."""
        self.buffer = buffer  # Buffer
        self.buffer_size = buffer.shape[0]  # Buffer size
        self.pointer = 0  # Point to the index where the new entry should be stored
        self.valid_size = 0  # Valid size of the buffer, maximum being `buffer_size`

    def add(self, recording: torch.Tensor = None):
        """Adds a new recording to the buffer, overwriting the oldest recording if the buffer is full.

        Args:
            recording: A recording tensor to add to the buffer.
        """
        self.buffer[self.pointer] = recording
        self.pointer = (
            self.pointer + 1
        ) % self.buffer_size  # Increment, loop back to 0 if full
        self.valid_size = min(
            self.valid_size + 1, self.buffer_size
        )  # Increment up to the maximum size

    def get_latest(self, n=1):
        """Returns the n-th latest recording from the buffer.

        Args:
            n: Specifies which latest recording to retrieve (1-based index: 1 is the most recent).

        Return:
            The n-th latest recording. If n is larger than valid_size, returns the first recording.
        """
        if n > self.valid_size:
            index = 0
        else:
            index = (self.pointer - n) % self.buffer_size

        return self.buffer[index]

    def reset(self):
        """Reset the buffer."""
        self.buffer[:] = 0
        self.pointer = 0
        self.valid_size = 0


class StateBuffer(CircularBuffer):
    def __init__(self, buffer: torch.Tensor = None):
        """Initializes a circular buffer to store initial states."""
        super().__init__(buffer=buffer)  # Properly initialize the parent class
        self.idx_scenario = 5
        self.idx_path = 6
        self.idx_point = 7


class InitialStateBuffer(CircularBuffer):
    def __init__(
        self,
        buffer: torch.Tensor = None,
        probability_record: torch.Tensor = None,
        probability_use_recording: torch.Tensor = None,
    ):
        """Initializes a circular buffer to store initial states."""
        super().__init__(buffer=buffer)  # Properly initialize the parent class
        self.probability_record = probability_record
        self.probability_use_recording = probability_use_recording
        self.idx_scenario = 5
        self.idx_path = 6
        self.idx_point = 7

    def get_random(self):
        """Returns a randomly selected recording from the buffer.

        Return:
            A randomly selected recording tensor. If the buffer is empty, returns None.
        """
        if self.valid_size == 0:
            return None
        else:
            # Random index based on the current size of the buffer
            random_index = torch.randint(0, self.valid_size, ())

            return self.buffer[random_index]


class Observations:
    def __init__(
        self,
        n_nearing_agents=None,
        nearing_agents_indices=None,
        obs_noise_level=None,
        noise_level=None,
        n_stored_steps=None,
        n_observed_steps=None,
        past_pos: CircularBuffer = None,
        past_rot: CircularBuffer = None,
        past_steering: CircularBuffer = None,
        past_vertices: CircularBuffer = None,
        past_vel: CircularBuffer = None,
        past_short_term_ref_points: CircularBuffer = None,
        past_action_vel: CircularBuffer = None,
        past_action_steering: CircularBuffer = None,
        past_distance_to_ref_path: CircularBuffer = None,
        past_distance_to_boundaries: CircularBuffer = None,
        past_distance_to_left_boundary: CircularBuffer = None,
        past_distance_to_right_boundary: CircularBuffer = None,
        past_distance_to_agents: CircularBuffer = None,
        past_lengths: CircularBuffer = None,
        past_widths: CircularBuffer = None,
        past_left_boundary: CircularBuffer = None,
        past_right_boundary: CircularBuffer = None,
        past_id: CircularBuffer = None,
    ):
        self.n_nearing_agents = n_nearing_agents
        self.nearing_agents_indices = nearing_agents_indices
        if obs_noise_level is None:
            obs_noise_level = noise_level
        self.obs_noise_level = obs_noise_level  # Observation-noise magnitude
        self.noise_level = obs_noise_level  # Backward-compatible scenario alias
        self.n_stored_steps = n_stored_steps  # Number of past steps to store
        self.n_observed_steps = n_observed_steps  # Number of past steps to observe

        self.past_pos = past_pos  # Past positions
        self.past_rot = past_rot  # Past rotations
        self.past_steering = past_steering  # Past steering angles
        self.past_vertices = past_vertices  # Past vertices
        self.past_vel = past_vel  # Past velocites

        self.past_short_term_ref_points = (
            past_short_term_ref_points  # Past short-term reference points
        )
        self.past_left_boundary = past_left_boundary  # Past left lanelet boundary
        self.past_right_boundary = past_right_boundary  # Past right lanelet boundary

        self.past_action_vel = past_action_vel  # Past velocity action
        self.past_action_steering = past_action_steering  # Past steering action
        self.past_distance_to_ref_path = (
            past_distance_to_ref_path  # Past distance to refrence path
        )
        self.past_distance_to_boundaries = (
            past_distance_to_boundaries  # Past distance to lanelet boundaries
        )
        self.past_distance_to_left_boundary = (
            past_distance_to_left_boundary  # Past distance to left lanelet boundary
        )
        self.past_distance_to_right_boundary = (
            past_distance_to_right_boundary  # Past distance to right lanelet boundary
        )
        self.past_distance_to_agents = (
            past_distance_to_agents  # Past mutual distance between agents
        )
        self.past_lengths = past_lengths  # Past lengths of agents (although they do not change, but for the reason of keeping consistence)
        self.past_widths = past_widths  # Past widths of agents (although they do not change, but for the reason of keeping consistence)
        self.past_id = past_id  # Past lanelet IDs


class Noise:
    def __init__(
        self,
        vel: torch.Tensor = None,
        ref: torch.Tensor = None,
        dis_ref: torch.Tensor = None,
        dis_lanelets: torch.Tensor = None,
        other_agents_pos: torch.Tensor = None,
        other_agents_rot: torch.Tensor = None,
        other_agents_vel: torch.Tensor = None,
        other_agents_dis: torch.Tensor = None,
        level_vel: torch.Tensor = None,
        level_pos: torch.Tensor = None,
        level_rot: torch.Tensor = None,
        level_dis: torch.Tensor = None,
    ):
        self.vel = vel
        self.ref = ref
        self.dis_ref = dis_ref
        self.dis_lanelets = dis_lanelets
        self.other_agents_pos = other_agents_pos
        self.other_agents_rot = other_agents_rot
        self.other_agents_vel = other_agents_vel
        self.other_agents_dis = other_agents_dis
        self.level_vel = level_vel
        self.level_pos = level_pos
        self.level_rot = level_rot
        self.level_dis = level_dis


##################################################
## Helper Functions
##################################################
def get_rectangle_vertices(
    center: torch.Tensor,
    yaw: torch.Tensor,
    width: float,
    length: float,
    is_close_shape: bool = True,
):
    """Compute the vertices of rectangles for a batch of agents given their centers, yaws (rotations),
    widths, and lengths, using PyTorch tensors.

    Args:
        center: [batch_dim, 2] or [2] center positions of the rectangles. In the case of the latter, batch_dim is deemed to be 1.
        yaw: [batch_dim, 1] or [1] or [] Rotation angles in radians.
        width: [scalar] Width of the rectangles.
        length: [scalar] Length of the rectangles.

    Return:
        [batch_dim, 4, 2] vertex points of the rectangles for each agent.
    """
    if center.ndim == 1:
        center = center.unsqueeze(0)

    if yaw.ndim == 0:
        yaw = yaw.unsqueeze(0).unsqueeze(0)
    elif yaw.ndim == 1:
        yaw = yaw.unsqueeze(0)

    batch_dim = center.shape[0]

    width_half = width / 2
    length_half = length / 2

    # vertex points relative to the center
    if is_close_shape:
        vertices = torch.tensor(
            [
                [length_half, width_half],
                [length_half, -width_half],
                [-length_half, -width_half],
                [-length_half, width_half],
                [length_half, width_half],
            ],
            dtype=center.dtype,
            device=center.device,
        )  # Repeat the first vertex to close the shape
    else:
        vertices = torch.tensor(
            [
                [length_half, width_half],
                [length_half, -width_half],
                [-length_half, -width_half],
                [-length_half, width_half],
            ],
            dtype=center.dtype,
            device=center.device,
        )

    # Expand vertices to match batch size
    vertices = vertices.unsqueeze(0).repeat(batch_dim, 1, 1)

    # Create rotation matrices for each agent
    cos_yaw = torch.cos(yaw).squeeze(1)
    sin_yaw = torch.sin(yaw).squeeze(1)

    # Rotation matrix for each agent
    rot_matrix = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=1),
            torch.stack([sin_yaw, cos_yaw], dim=1),
        ],
        dim=1,
    )

    # Apply rotation to vertices
    vertices_rotated = torch.matmul(rot_matrix, vertices.transpose(1, 2)).transpose(
        1, 2
    )

    # Add center positions to the rotated vertices
    vertices_global = vertices_rotated + center.unsqueeze(1)

    return vertices_global


def get_rectangle_vertices(
    center: torch.Tensor,
    yaw: torch.Tensor,
    width: float,
    length: float,
    is_close_shape: bool = True,
    num_point_length_side: int = 0,
    num_point_width_side: int = 0,
):
    """
    Compute the vertices (and optionally intermediate points) of rectangles for a batch of agents
    given their centers, yaws (rotations), widths, and lengths, using PyTorch tensors.
    It can include uniformly distributed intermediate points along the length and width sides.

    Args:
        center: [batch_dim, 2] or [2] center positions of the rectangles. If [2], batch_dim is 1.
        yaw: [batch_dim, 1] or [1] or [] Rotation angles in radians.
        width: scalar width of the rectangles.
        length: scalar length of the rectangles.
        is_close_shape: if True, the returned vertices will close the shape by repeating the first vertex.
        num_point_length_side: number of uniformly distributed intermediate points on each length side.
        num_point_width_side: number of uniformly distributed intermediate points on each width side.

    Returns:
        Tensor of shape [batch_dim, N, 2] containing vertex (and intermediate) points of the rectangles
        for each agent, where N depends on the number of requested intermediate points.
    """

    # Ensure center has batch dimension
    if center.ndim == 1:
        center = center.unsqueeze(0)

    # Normalize yaw dimensions to [batch_dim, 1]
    if yaw.ndim == 0:
        yaw = yaw.unsqueeze(0).unsqueeze(0)
    elif yaw.ndim == 1:
        yaw = yaw.unsqueeze(0)

    if num_point_length_side != 0 or num_point_width_side != 0:
        # Closed shape is required for the intermediate points
        is_close_shape = True

    batch_dim = center.shape[0]
    width_half = width / 2
    length_half = length / 2

    # Define base vertices relative to the center in local coordinates
    if is_close_shape:
        base_vertices = torch.tensor(
            [
                [length_half, width_half],
                [length_half, -width_half],
                [-length_half, -width_half],
                [-length_half, width_half],
                [length_half, width_half],  # repeat first to close the shape
            ],
            dtype=center.dtype,
            device=center.device,
        )
    else:
        base_vertices = torch.tensor(
            [
                [length_half, width_half],
                [length_half, -width_half],
                [-length_half, -width_half],
                [-length_half, width_half],
            ],
            dtype=center.dtype,
            device=center.device,
        )

    # If no intermediate points are requested, use original vertices
    if num_point_length_side == 0 and num_point_width_side == 0:
        vertices = base_vertices
    else:
        # List to collect sampled points along each edge
        edge_points = []
        # Iterate over each consecutive pair of vertices to sample points on edges
        for i in range(len(base_vertices) - 1):
            start = base_vertices[i]
            end = base_vertices[i + 1]
            # Determine if edge is horizontal or vertical in local frame
            if torch.isclose(start[1], end[1]):  # horizontal edge: constant y, vary x
                n_points = num_point_length_side + 2
            elif torch.isclose(start[0], end[0]):  # vertical edge: constant x, vary y
                n_points = num_point_width_side + 2
            else:
                # This should not happen for axis-aligned rectangle edges
                n_points = 2

            # Create linearly spaced points along the edge
            # linspace returns points including both endpoints
            t = torch.linspace(
                0, 1, steps=n_points, device=center.device, dtype=center.dtype
            ).unsqueeze(1)
            pts = start + t * (end - start)

            # Avoid duplicate vertices by dropping the first point after the first edge
            if i > 0:
                pts = pts[1:]
            edge_points.append(pts)

        # Concatenate all sampled points along edges to form the full set of vertices for one rectangle
        vertices = torch.cat(edge_points, dim=0)

    # Expand vertices to match batch size: shape becomes [batch_dim, num_vertices, 2]
    vertices = vertices.unsqueeze(0).repeat(batch_dim, 1, 1)

    # Create rotation matrices for each agent
    # Squeeze yaw to shape [batch_dim] for trigonometric computations
    cos_yaw = torch.cos(yaw).squeeze(-1)
    sin_yaw = torch.sin(yaw).squeeze(-1)

    # Build rotation matrices of shape [batch_dim, 2, 2]
    rot_matrix = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1),
        ],
        dim=1,
    )

    # Apply rotation: use batch matrix multiplication
    # vertices.transpose(1, 2) shape [batch_dim, 2, num_vertices], then multiply, then transpose back
    vertices_rotated = torch.matmul(rot_matrix, vertices.transpose(1, 2)).transpose(
        1, 2
    )

    # Add center positions to rotated vertices: broadcasting center to match vertices shape
    vertices_global = vertices_rotated + center.unsqueeze(1)

    return vertices_global


def get_perpendicular_distances(
    point: torch.Tensor, polyline: torch.Tensor, n_points_long_term: torch.Tensor = None
):
    """Calculate minimum perpendicular distances from points to polylines.

    Args:
        point: Point positions with shape ``[2]`` or ``[B, 2]``.
        polyline: Polyline points with shape ``[P, 2]`` or ``[B, P, 2]``.
        n_points_long_term: Number of valid points per polyline, with shape
            ``[]``, ``[1]``, or ``[B]``. Padded points after this count are ignored.

    Returns:
        Tuple of ``(perpendicular_distances, indices_closest_points)`` with both
        tensors shaped ``[B]``. The closest indices keep the historical behavior of
        pointing to the next polyline point after the closest segment.
    """
    if point.ndim == 1:
        point = point.unsqueeze(0)

    batch_size = point.shape[0]

    # Expand the polyline points to match the batch size
    if polyline.ndim == 2:
        polyline_expanded = polyline.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, n_points, 2]
    else:
        polyline_expanded = polyline

    # Split the polyline into line segments
    line_starts = polyline_expanded[:, :-1, :]
    line_ends = polyline_expanded[:, 1:, :]

    # Create vectors for each line segment and for the point to the start of each segment
    point_expanded = point.unsqueeze(1)  # Shape: [batch_size, 1, 2]
    line_vecs = line_ends - line_starts
    point_vecs = point_expanded - line_starts

    # Project point_vecs onto line_vecs
    line_lens_squared = torch.sum(line_vecs**2, dim=2)
    line_lens_squared_safe = torch.clamp(
        line_lens_squared, min=torch.finfo(polyline.dtype).eps
    )
    projected_lengths = (
        torch.sum(point_vecs * line_vecs, dim=2) / line_lens_squared_safe
    )

    # Clamp the projections to lie within the line segments
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)

    # Find the closest points on the line segments to the given points
    closest_points = line_starts + (line_vecs * clamped_lengths.unsqueeze(2))

    # Calculate the distances from the given points to these closest points
    distances = torch.norm(closest_points - point_expanded, dim=2)

    if n_points_long_term is not None:
        if n_points_long_term.ndim == 0:
            n_points_long_term = n_points_long_term.unsqueeze(0)
        if n_points_long_term.numel() == 1 and batch_size > 1:
            n_points_long_term = n_points_long_term.expand(batch_size)
        n_points_long_term = n_points_long_term.to(
            device=polyline.device, dtype=torch.long
        )
        segment_indices = torch.arange(distances.shape[1], device=polyline.device)
        valid_segments = segment_indices.unsqueeze(0) < (
            n_points_long_term - 1
        ).unsqueeze(1)
        last_valid_segment = torch.clamp(n_points_long_term - 2, min=0)
        distance_to_end_point = distances[
            torch.arange(batch_size, device=polyline.device), last_valid_segment
        ]
        distances = torch.where(
            valid_segments,
            distances,
            distance_to_end_point.unsqueeze(1),
        )

    assert not distances.isnan().any()

    perpendicular_distances, indices_closest_points = torch.min(distances, dim=1)

    indices_closest_points[
        :
    ] += 1  # Force the nearest point to lie always in the future

    return perpendicular_distances, indices_closest_points


def get_progress_on_polyline(
    point: torch.Tensor, polyline: torch.Tensor, n_points_long_term: torch.Tensor = None
):
    """Compute closest distance and arc-length progress of points along a polyline."""
    if point.ndim == 1:
        point = point.unsqueeze(0)

    batch_size = point.shape[0]

    if polyline.ndim == 2:
        polyline_expanded = polyline.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        polyline_expanded = polyline

    if n_points_long_term is None:
        n_points_long_term = torch.full(
            (batch_size,),
            polyline_expanded.shape[-2],
            device=polyline_expanded.device,
            dtype=torch.int32,
        )
    elif n_points_long_term.ndim == 0:
        n_points_long_term = n_points_long_term.unsqueeze(0)
    if n_points_long_term.numel() == 1 and batch_size > 1:
        n_points_long_term = n_points_long_term.expand(batch_size)

    line_starts = polyline_expanded[:, :-1, :]
    line_ends = polyline_expanded[:, 1:, :]
    line_vecs = line_ends - line_starts
    segment_lengths = torch.norm(line_vecs, dim=-1)
    line_lens_squared = torch.sum(line_vecs**2, dim=-1)
    line_lens_safe = torch.clamp(line_lens_squared, min=torch.finfo(polyline.dtype).eps)

    point_vecs = point.unsqueeze(1) - line_starts
    projected_lengths = torch.sum(point_vecs * line_vecs, dim=-1) / line_lens_safe
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)
    closest_points = line_starts + line_vecs * clamped_lengths.unsqueeze(-1)
    distances = torch.norm(closest_points - point.unsqueeze(1), dim=-1)

    cumulative_lengths = torch.cat(
        (
            torch.zeros((batch_size, 1), device=polyline.device, dtype=polyline.dtype),
            torch.cumsum(segment_lengths, dim=-1),
        ),
        dim=-1,
    )

    segment_indices = torch.arange(
        line_starts.shape[1], device=polyline.device
    ).unsqueeze(0)
    valid_segments = segment_indices < (n_points_long_term - 1).unsqueeze(-1)
    valid_segments = valid_segments & (
        segment_lengths > torch.finfo(polyline.dtype).eps
    )
    distances = torch.where(
        valid_segments,
        distances,
        torch.full_like(distances, torch.finfo(polyline.dtype).max),
    )

    perpendicular_distances, closest_segment_idx = torch.min(distances, dim=-1)
    progress = (
        cumulative_lengths[
            torch.arange(batch_size, device=polyline.device), closest_segment_idx
        ]
        + clamped_lengths[
            torch.arange(batch_size, device=polyline.device), closest_segment_idx
        ]
        * segment_lengths[
            torch.arange(batch_size, device=polyline.device), closest_segment_idx
        ]
    )
    total_lengths = cumulative_lengths[
        torch.arange(batch_size, device=polyline.device), n_points_long_term - 1
    ]

    return perpendicular_distances, progress, total_lengths


def sample_polyline_by_progress(
    polyline: torch.Tensor,
    progress_values: torch.Tensor,
    n_points_long_term: torch.Tensor = None,
    is_polyline_a_loop: torch.Tensor = False,
    is_extend_non_loop: bool = False,
):
    """
    Sample points from batched polylines at requested arc-length progress values.

    Args:
        polyline: Polyline points with shape ``[P, 2]`` or ``[B, P, 2]``.
        progress_values: Arc-length positions to sample with shape ``[K]`` or ``[B, K]``.
        n_points_long_term: Number of valid points in each polyline, with shape
            ``[]``, ``[1]``, or ``[B]``. Padded points after this count are ignored.
        is_polyline_a_loop: Loop flags with shape ``[]``, ``[1]``, or ``[B]``.
            Loop progress values are wrapped by the valid polyline length.
        is_extend_non_loop: If ``True``, samples outside non-loop polyline bounds are
            linearly extrapolated from the first or last valid segment.

    Returns:
        Sampled points with shape ``[B, K, 2]``.
    """
    if polyline.ndim == 2:
        polyline = polyline.unsqueeze(0)
    if progress_values.ndim == 1:
        progress_values = progress_values.unsqueeze(0)
    if polyline.shape[0] == 1 and progress_values.shape[0] > 1:
        polyline = polyline.expand(progress_values.shape[0], -1, -1)
    if not isinstance(is_polyline_a_loop, torch.Tensor):
        is_polyline_a_loop = torch.tensor(
            is_polyline_a_loop, device=polyline.device, dtype=torch.bool
        )
    if is_polyline_a_loop.ndim == 0:
        is_polyline_a_loop = is_polyline_a_loop.unsqueeze(0)
    if is_polyline_a_loop.numel() == 1 and progress_values.shape[0] > 1:
        is_polyline_a_loop = is_polyline_a_loop.expand(progress_values.shape[0])
    if n_points_long_term is None:
        n_points_long_term = torch.full(
            (polyline.shape[0],),
            polyline.shape[-2],
            device=polyline.device,
            dtype=torch.int32,
        )
    elif n_points_long_term.ndim == 0:
        n_points_long_term = n_points_long_term.unsqueeze(0)
    if n_points_long_term.numel() == 1 and progress_values.shape[0] > 1:
        n_points_long_term = n_points_long_term.expand(progress_values.shape[0])

    batch_size, n_samples = progress_values.shape
    n_points_total = polyline.shape[-2]
    n_segments_total = n_points_total - 1
    n_points_long_term = n_points_long_term.to(device=polyline.device, dtype=torch.long)

    if n_segments_total <= 0:
        return polyline[:, :1, :].expand(batch_size, n_samples, polyline.shape[-1])

    segment_vecs = polyline[:, 1:, :] - polyline[:, :-1, :]
    segment_lengths = torch.linalg.vector_norm(segment_vecs, dim=-1)
    segment_indices_all = torch.arange(n_segments_total, device=polyline.device)
    valid_segments = segment_indices_all.unsqueeze(0) < (
        n_points_long_term - 1
    ).unsqueeze(1)
    segment_lengths = torch.where(
        valid_segments,
        segment_lengths,
        torch.zeros((), device=polyline.device, dtype=polyline.dtype),
    )

    cumulative_lengths = torch.cat(
        (
            torch.zeros((batch_size, 1), device=polyline.device, dtype=polyline.dtype),
            torch.cumsum(segment_lengths, dim=-1),
        ),
        dim=-1,
    )
    batch_indices = torch.arange(batch_size, device=polyline.device)
    last_point_indices = torch.clamp(n_points_long_term - 1, min=0)
    total_lengths = cumulative_lengths[batch_indices, last_point_indices]
    eps = torch.finfo(polyline.dtype).eps

    progress_values = progress_values.to(polyline.dtype)
    total_lengths_safe = torch.clamp(total_lengths, min=eps)
    progress_wrapped = torch.where(
        is_polyline_a_loop.unsqueeze(-1),
        progress_values % total_lengths_safe.unsqueeze(-1),
        progress_values,
    )
    before_start = progress_wrapped < 0
    after_end = progress_wrapped > total_lengths.unsqueeze(-1)
    progress_clamped = torch.minimum(
        torch.clamp(progress_wrapped, min=0),
        total_lengths.unsqueeze(-1),
    )

    segment_indices = torch.searchsorted(cumulative_lengths, progress_clamped)
    upper_segment_indices = torch.clamp(n_points_long_term - 1, min=1).unsqueeze(-1)
    segment_indices = torch.minimum(
        torch.clamp(segment_indices, min=1), upper_segment_indices
    )

    segment_starts = torch.gather(cumulative_lengths, 1, segment_indices - 1)
    segment_ends = torch.gather(cumulative_lengths, 1, segment_indices)
    segment_denoms = torch.clamp(segment_ends - segment_starts, min=eps)
    segment_ratios = (progress_clamped - segment_starts) / segment_denoms

    gather_start_idx = (
        (segment_indices - 1).unsqueeze(-1).expand(-1, -1, polyline.shape[-1])
    )
    gather_end_idx = segment_indices.unsqueeze(-1).expand(-1, -1, polyline.shape[-1])
    start_points = torch.gather(polyline, 1, gather_start_idx)
    end_points = torch.gather(polyline, 1, gather_end_idx)
    sampled_points = start_points + segment_ratios.unsqueeze(-1) * (
        end_points - start_points
    )

    degenerate_polylines = total_lengths <= eps
    if torch.any(degenerate_polylines):
        sampled_points = torch.where(
            degenerate_polylines.view(batch_size, 1, 1),
            polyline[:, :1, :].expand(-1, n_samples, -1),
            sampled_points,
        )

    if is_extend_non_loop:
        non_loop = ~is_polyline_a_loop
        needs_before = before_start & non_loop.unsqueeze(-1)
        needs_after = after_end & non_loop.unsqueeze(-1)
        if torch.any(needs_before):
            first_lengths = torch.clamp(segment_lengths[:, 0], min=eps)
            before_points = (
                polyline[:, :1, :]
                + (progress_wrapped / first_lengths.unsqueeze(-1)).unsqueeze(-1)
                * segment_vecs[:, :1, :]
            )
            sampled_points = torch.where(
                needs_before.unsqueeze(-1), before_points, sampled_points
            )
        if torch.any(needs_after):
            last_segment_indices = torch.clamp(n_points_long_term - 2, min=0)
            last_lengths = torch.clamp(
                segment_lengths[batch_indices, last_segment_indices], min=eps
            )
            last_points = polyline[batch_indices, last_point_indices, :]
            last_vecs = segment_vecs[batch_indices, last_segment_indices, :]
            after_points = last_points.unsqueeze(1) + (
                (progress_wrapped - total_lengths.unsqueeze(-1))
                / last_lengths.unsqueeze(-1)
            ).unsqueeze(-1) * last_vecs.unsqueeze(1)
            sampled_points = torch.where(
                needs_after.unsqueeze(-1), after_points, sampled_points
            )

    return sampled_points


def get_route_progress_aligned_sampled_points(
    polyline: torch.Tensor,
    route_progress: torch.Tensor,
    route_total_length: torch.Tensor,
    n_points_to_return: int,
    n_points_long_term: torch.Tensor,
    is_polyline_a_loop: torch.Tensor,
    sample_interval: int = 1,
    n_points_shift: int = 0,
    sample_spacing=None,
    is_extend_non_loop: bool = False,
):
    """Sample local polyline points by normalized progress along the reference route."""
    n_points_to_return = int(n_points_to_return)
    if route_progress.ndim == 0:
        route_progress = route_progress.unsqueeze(0)
    if route_total_length.ndim == 0:
        route_total_length = route_total_length.unsqueeze(0)

    _, _, polyline_total_length = get_progress_on_polyline(
        point=polyline[..., 0, :],
        polyline=polyline,
        n_points_long_term=n_points_long_term,
    )
    route_total_safe = torch.clamp(
        route_total_length, min=torch.finfo(polyline.dtype).eps
    )
    progress_ratio = torch.clamp(route_progress / route_total_safe, 0, 1)
    base_progress = progress_ratio * polyline_total_length
    offsets = (
        torch.arange(n_points_to_return, device=polyline.device, dtype=polyline.dtype)
        * sample_interval
        + n_points_shift
    )
    if sample_spacing is None:
        spacing = polyline_total_length / torch.clamp(
            n_points_long_term.to(polyline.dtype) - 1,
            min=torch.ones((), device=polyline.device, dtype=polyline.dtype),
        )
    else:
        spacing = torch.as_tensor(
            sample_spacing, device=polyline.device, dtype=polyline.dtype
        )
        if spacing.ndim == 0:
            spacing = spacing.unsqueeze(0)
        if spacing.numel() == 1 and base_progress.shape[0] > 1:
            spacing = spacing.expand(base_progress.shape[0])
    progress_values = base_progress.unsqueeze(-1) + spacing.unsqueeze(-1) * offsets

    return sample_polyline_by_progress(
        polyline=polyline,
        progress_values=progress_values,
        n_points_long_term=n_points_long_term,
        is_polyline_a_loop=is_polyline_a_loop,
        is_extend_non_loop=is_extend_non_loop,
    )


def get_reference_progress_aligned_sampled_points(
    reference_polyline: torch.Tensor,
    aligned_polyline: torch.Tensor,
    route_progress: torch.Tensor,
    n_points_to_return: int,
    n_points_reference: torch.Tensor,
    is_polyline_a_loop: torch.Tensor,
    sample_interval: int = 1,
    n_points_shift: int = 0,
    sample_spacing=None,
    is_extend_non_loop: bool = False,
):
    """Sample an aligned polyline using the reference polyline's progress grid.

    Args:
        reference_polyline: Reference points with shape ``[P, 2]`` or
            ``[B, P, 2]``. Its cumulative arc length defines the station grid.
        aligned_polyline: Points aligned by index to ``reference_polyline`` with
            shape ``[P, 2]`` or ``[B, P, 2]``.
        route_progress: Current arc-length progress on ``reference_polyline`` with
            shape ``[]`` or ``[B]``.
        n_points_to_return: Number of local points to sample.
        n_points_reference: Number of valid aligned reference points with shape
            ``[]``, ``[1]``, or ``[B]``.
        is_polyline_a_loop: Loop flags with shape ``[]``, ``[1]``, or ``[B]``.
        sample_interval: Multiplier for default reference-point spacing.
        n_points_shift: Sample offset in units of the selected spacing.
        sample_spacing: Optional metric spacing on the reference polyline. If
            omitted, the mean valid reference segment length is used.
        is_extend_non_loop: If ``True``, samples outside non-loop reference bounds
            are linearly extrapolated using the first or last aligned segment.

    Returns:
        Sampled aligned-polyline points with shape ``[B, n_points_to_return, 2]``.
    """
    if reference_polyline.ndim == 2:
        reference_polyline = reference_polyline.unsqueeze(0)
    if aligned_polyline.ndim == 2:
        aligned_polyline = aligned_polyline.unsqueeze(0)
    if route_progress.ndim == 0:
        route_progress = route_progress.unsqueeze(0)
    if n_points_reference.ndim == 0:
        n_points_reference = n_points_reference.unsqueeze(0)
    if not isinstance(is_polyline_a_loop, torch.Tensor):
        is_polyline_a_loop = torch.tensor(
            is_polyline_a_loop, device=reference_polyline.device, dtype=torch.bool
        )
    if is_polyline_a_loop.ndim == 0:
        is_polyline_a_loop = is_polyline_a_loop.unsqueeze(0)

    batch_size = max(
        reference_polyline.shape[0],
        aligned_polyline.shape[0],
        route_progress.shape[0],
        n_points_reference.shape[0],
        is_polyline_a_loop.shape[0],
    )
    if reference_polyline.shape[0] == 1 and batch_size > 1:
        reference_polyline = reference_polyline.expand(batch_size, -1, -1)
    if aligned_polyline.shape[0] == 1 and batch_size > 1:
        aligned_polyline = aligned_polyline.expand(batch_size, -1, -1)
    if route_progress.shape[0] == 1 and batch_size > 1:
        route_progress = route_progress.expand(batch_size)
    if n_points_reference.shape[0] == 1 and batch_size > 1:
        n_points_reference = n_points_reference.expand(batch_size)
    if is_polyline_a_loop.shape[0] == 1 and batch_size > 1:
        is_polyline_a_loop = is_polyline_a_loop.expand(batch_size)

    n_points_to_return = int(n_points_to_return)
    n_points_reference = n_points_reference.to(
        device=reference_polyline.device, dtype=torch.long
    )
    n_points_reference = torch.minimum(
        n_points_reference,
        torch.full_like(n_points_reference, reference_polyline.shape[-2]),
    )
    n_points_reference = torch.minimum(
        n_points_reference,
        torch.full_like(n_points_reference, aligned_polyline.shape[-2]),
    )
    n_segments_total = min(reference_polyline.shape[-2], aligned_polyline.shape[-2]) - 1
    if n_segments_total <= 0:
        return aligned_polyline[:, :1, :].expand(
            batch_size, n_points_to_return, aligned_polyline.shape[-1]
        )

    eps = torch.finfo(reference_polyline.dtype).eps
    reference_vecs = reference_polyline[:, 1:, :] - reference_polyline[:, :-1, :]
    aligned_vecs = aligned_polyline[:, 1:, :] - aligned_polyline[:, :-1, :]
    reference_segment_lengths = torch.linalg.vector_norm(reference_vecs, dim=-1)
    segment_indices_all = torch.arange(
        reference_segment_lengths.shape[-1], device=reference_polyline.device
    )
    valid_segments = segment_indices_all.unsqueeze(0) < (
        n_points_reference - 1
    ).unsqueeze(1)
    reference_segment_lengths = torch.where(
        valid_segments,
        reference_segment_lengths,
        torch.zeros(
            (), device=reference_polyline.device, dtype=reference_polyline.dtype
        ),
    )
    cumulative_lengths = torch.cat(
        (
            torch.zeros(
                (batch_size, 1),
                device=reference_polyline.device,
                dtype=reference_polyline.dtype,
            ),
            torch.cumsum(reference_segment_lengths, dim=-1),
        ),
        dim=-1,
    )
    batch_indices = torch.arange(batch_size, device=reference_polyline.device)
    last_point_indices = torch.clamp(n_points_reference - 1, min=0)
    total_lengths = cumulative_lengths[batch_indices, last_point_indices]
    total_lengths_safe = torch.clamp(total_lengths, min=eps)

    route_progress = route_progress.to(reference_polyline.dtype)
    progress_wrapped = torch.where(
        is_polyline_a_loop.unsqueeze(-1),
        route_progress.unsqueeze(-1) % total_lengths_safe.unsqueeze(-1),
        route_progress.unsqueeze(-1),
    )
    offsets = (
        torch.arange(
            n_points_to_return,
            device=reference_polyline.device,
            dtype=reference_polyline.dtype,
        )
        * sample_interval
        + n_points_shift
    )
    if sample_spacing is None:
        spacing = total_lengths / torch.clamp(
            n_points_reference.to(reference_polyline.dtype) - 1,
            min=torch.ones(
                (), device=reference_polyline.device, dtype=reference_polyline.dtype
            ),
        )
    else:
        spacing = torch.as_tensor(
            sample_spacing,
            device=reference_polyline.device,
            dtype=reference_polyline.dtype,
        )
        if spacing.ndim == 0:
            spacing = spacing.unsqueeze(0)
        if spacing.numel() == 1 and batch_size > 1:
            spacing = spacing.expand(batch_size)
    progress_values = progress_wrapped + spacing.unsqueeze(-1) * offsets

    before_start = progress_values < 0
    after_end = progress_values > total_lengths.unsqueeze(-1)
    progress_clamped = torch.minimum(
        torch.clamp(progress_values, min=0),
        total_lengths.unsqueeze(-1),
    )
    segment_indices = torch.searchsorted(cumulative_lengths, progress_clamped)
    upper_segment_indices = torch.clamp(n_points_reference - 1, min=1).unsqueeze(-1)
    segment_indices = torch.minimum(
        torch.clamp(segment_indices, min=1), upper_segment_indices
    )

    segment_starts = torch.gather(cumulative_lengths, 1, segment_indices - 1)
    segment_ends = torch.gather(cumulative_lengths, 1, segment_indices)
    segment_denoms = torch.clamp(segment_ends - segment_starts, min=eps)
    segment_ratios = (progress_clamped - segment_starts) / segment_denoms

    gather_start_idx = (
        (segment_indices - 1).unsqueeze(-1).expand(-1, -1, aligned_polyline.shape[-1])
    )
    gather_end_idx = segment_indices.unsqueeze(-1).expand(
        -1, -1, aligned_polyline.shape[-1]
    )
    start_points = torch.gather(aligned_polyline, 1, gather_start_idx)
    end_points = torch.gather(aligned_polyline, 1, gather_end_idx)
    sampled_points = start_points + segment_ratios.unsqueeze(-1) * (
        end_points - start_points
    )

    degenerate_polylines = total_lengths <= eps
    if torch.any(degenerate_polylines):
        sampled_points = torch.where(
            degenerate_polylines.view(batch_size, 1, 1),
            aligned_polyline[:, :1, :].expand(-1, n_points_to_return, -1),
            sampled_points,
        )

    if is_extend_non_loop:
        non_loop = ~is_polyline_a_loop
        needs_before = before_start & non_loop.unsqueeze(-1)
        needs_after = after_end & non_loop.unsqueeze(-1)
        if torch.any(needs_before):
            first_lengths = torch.clamp(reference_segment_lengths[:, 0], min=eps)
            before_points = (
                aligned_polyline[:, :1, :]
                + (progress_values / first_lengths.unsqueeze(-1)).unsqueeze(-1)
                * aligned_vecs[:, :1, :]
            )
            sampled_points = torch.where(
                needs_before.unsqueeze(-1), before_points, sampled_points
            )
        if torch.any(needs_after):
            last_segment_indices = torch.clamp(n_points_reference - 2, min=0)
            last_lengths = torch.clamp(
                reference_segment_lengths[batch_indices, last_segment_indices], min=eps
            )
            last_points = aligned_polyline[batch_indices, last_point_indices, :]
            last_vecs = aligned_vecs[batch_indices, last_segment_indices, :]
            after_points = last_points.unsqueeze(1) + (
                (progress_values - total_lengths.unsqueeze(-1))
                / last_lengths.unsqueeze(-1)
            ).unsqueeze(-1) * last_vecs.unsqueeze(1)
            sampled_points = torch.where(
                needs_after.unsqueeze(-1), after_points, sampled_points
            )

    return sampled_points


def get_short_term_reference_path(
    polyline: torch.Tensor,
    index_closest_point: torch.Tensor,
    n_points_to_return: int,
    device=torch.device("cpu"),
    is_polyline_a_loop: torch.Tensor = False,
    n_points_long_term: torch.Tensor = None,
    sample_interval: int = 2,
    n_points_shift: int = 1,
    is_extend_non_loop: bool = False,
):
    """

    Args:
        polyline:                   [batch_size, num_points, 2] or [num_points, 2]. In the case of the latter, batch_dim is deemed as 1.
        index_closest_point:        [batch_size, 1] or [1] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_to_return:         [1] or []. In the case of the latter, batch_dim is deemed as 1.
        is_polyline_a_loop:         [batch_size] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_long_term:         [batch_size] or []. In the case of the latter, batch_dim is deemed as 1.
        sample_interval:            Sample interval to match specific purposes;
                                    set to 2 when using this function to get the short-term reference path;
                                    set to 1 when using this function to get the nearing boundary points.
        n_points_shift:             Number of points to be shifted to match specific purposes;
                                    set to 1 when using this function to get the short-term reference path to "force" the first point of the short-term reference path being in front of the agent;
                                    set to -2 when using this function to get the nearing boundary points to consider the points behind the agent.
        is_extend_non_loop:         Whether to extrapolate samples outside the valid range of a non-loop polyline using the first or last valid segment.
    """

    def warn_if_overlapping_points(points: torch.Tensor):
        """Print a warning if consecutive sampled points overlap."""
        if points.shape[-2] < 2:
            return
        overlapping = torch.diff(points, dim=-2).norm(dim=-1) <= 1e-6
        if torch.any(overlapping):
            print(
                "[WARN] get_short_term_reference_path returned overlapping consecutive points."
            )

    if polyline.ndim == 2:
        polyline = polyline.unsqueeze(0)
    if index_closest_point.ndim == 1:
        index_closest_point = index_closest_point.unsqueeze(1)
    elif index_closest_point.ndim == 0:
        index_closest_point = index_closest_point.unsqueeze(0).unsqueeze(1)
    if not isinstance(is_polyline_a_loop, torch.Tensor):
        is_polyline_a_loop = torch.tensor(
            is_polyline_a_loop, device=device, dtype=torch.bool
        )
    if is_polyline_a_loop.ndim == 0:
        is_polyline_a_loop = is_polyline_a_loop.unsqueeze(0)
    if n_points_long_term is None:
        n_points_long_term = torch.full(
            (index_closest_point.shape[0],),
            polyline.shape[-2],
            device=device,
            dtype=torch.int32,
        )
    elif n_points_long_term.ndim == 0:
        n_points_long_term = n_points_long_term.unsqueeze(0)

    batch_size = index_closest_point.shape[0]

    future_points_idx = (
        torch.arange(n_points_to_return, device=device) * sample_interval
        + index_closest_point
        + n_points_shift
    )

    for env_i in range(batch_size):
        n_long_term_point = n_points_long_term[env_i]
        if is_polyline_a_loop[env_i]:
            # Apply modulo to handle the case that each agent's reference path is a loop
            future_points_idx[env_i] = future_points_idx[env_i] % (
                n_long_term_point - 1
            )  # Skip the last point since it overlaps with the first point.

    if is_extend_non_loop:
        short_term_path = torch.empty(
            (batch_size, n_points_to_return, polyline.shape[-1]),
            device=polyline.device,
            dtype=polyline.dtype,
        )
        for env_i in range(batch_size):
            n_long_term_point = int(n_points_long_term[env_i].item())
            indices_i = future_points_idx[env_i]
            if is_polyline_a_loop[env_i] or n_long_term_point < 2:
                short_term_path[env_i] = polyline[env_i, indices_i]
                continue

            last_valid_idx = n_long_term_point - 1
            indices_clamped = torch.clamp(indices_i, 0, last_valid_idx)
            short_term_path[env_i] = polyline[env_i, indices_clamped]

            before_start = indices_i < 0
            if torch.any(before_start):
                first_segment = polyline[env_i, 1] - polyline[env_i, 0]
                short_term_path[env_i, before_start] = (
                    polyline[env_i, 0]
                    + indices_i[before_start].unsqueeze(1).to(polyline.dtype)
                    * first_segment
                )

            after_end = indices_i > last_valid_idx
            if torch.any(after_end):
                last_segment = (
                    polyline[env_i, last_valid_idx]
                    - polyline[env_i, last_valid_idx - 1]
                )
                short_term_path[env_i, after_end] = (
                    polyline[env_i, last_valid_idx]
                    + (indices_i[after_end] - last_valid_idx)
                    .unsqueeze(1)
                    .to(polyline.dtype)
                    * last_segment
                )

        warn_if_overlapping_points(short_term_path)
        return short_term_path, future_points_idx

    # Extract
    short_term_path = polyline[
        torch.arange(batch_size, device=device, dtype=torch.int).unsqueeze(
            1
        ),  # For broadcasting
        future_points_idx,
    ]

    warn_if_overlapping_points(short_term_path)
    return short_term_path, future_points_idx


def decreasing_fcn(x, x0, x1, type: str = "linear"):
    """
    Decreasing function defined on [x0, x1] with output in [0, 1].

    Inputs:
        x: torch.Tensor
        x0: float or torch scalar
        x1: float or torch scalar
        type: "linear" or "exponential" (default: "linear")

    Behavior:
        - x is clamped to [x0, x1]
        - y(x0) = 1, y(x1) = 0

    Linear:
        y(x) = 1 - (x - x0) / (x1 - x0)

    Exponential:
        y(x) = (exp(-(x - x0) / (x1 - x0)) - e^-1) / (1 - e^-1)
    """
    # Ensure x stays inside [x0, x1]
    x = torch.clamp(x, min=x0, max=x1)

    denom = x1 - x0
    if denom < 0:
        raise ValueError("x1 must be strictly non-negative.")

    if type == "linear":
        y = 1.0 - (x - x0) / denom

    elif type == "exponential":
        y = (torch.exp(-(x - x0) / denom) - 1.0 / torch.e) / (1.0 - 1.0 / torch.e)

    else:
        raise ValueError(f'Unsupported type "{type}". Use "linear" or "exponential".')

    return y


def get_distances_between_agents(
    data, distance_type, is_set_diagonal=False, x_semidim=None, y_semidim=None
):
    """This function calculates the mutual distances between agents.
        Currently, the calculation of two types of distances is supported ('c2c' and 'mtv'):
            c2c: center-to-center distance
            mtv: minimum translation vector (mtv)-based distance
    Args:
        data: positions of the agents with dimension [batch_dim, n_agents, 2] if `distance_type == "c2c"` or their vertices with dimension [batch_dim, n_agents, 4 or 5, 2] if `distance_type == "mtv"`
        distance_type: one of {c2c, mtv}
        is_set_diagonal: whether to set the diagonal elements (distance from an agent to this agent itself) from zero to a high value
    TODO: Add the posibility to calculate the mutual distances between agents in a single env (`reset_world` sometime only needs to resets a single env)
    """
    if distance_type == "c2c":
        # Collect positions for all agents across all batches, shape [n_agents, batch_size, 2]

        # Reshape from [n_agents, batch_size, 2] to [batch_size, n_agents, 2]
        positions_reshaped = data.transpose(0, 1)

        # Reshape for broadcasting: shape becomes [batch_size, n_agents, 1, 2] and [batch_size, 1, n_agents, 2]
        pos1 = positions_reshaped.unsqueeze(2)
        pos2 = positions_reshaped.unsqueeze(1)

        # Calculate squared differences, shape [batch_size, n_agents, n_agents, 2]
        squared_diff = (pos1 - pos2) ** 2

        # Sum over the last dimension to get squared distances, shape [batch_size, n_agents, n_agents]
        squared_distances = squared_diff.sum(-1)

        # Take the square root to get actual distances, shape [batch_size, n_agents, n_agents]
        mutual_distances = torch.sqrt(squared_distances)
    elif distance_type == "mtv":
        batch_dim, n_agents, _, _ = data.shape
        device = data.device

        # Initialize
        mutual_distances = torch.zeros(
            (batch_dim, n_agents, n_agents),
            device=device,
            dtype=torch.float32,
        )

        # Calculate the normal axes of the four edges of each rectangle (Note that each rectangle has two normal axes)
        axes_all = torch.diff(data[:, :, 0:3, :], dim=2)
        axes_norm_all = axes_all / torch.norm(axes_all, dim=-1).unsqueeze(
            -1
        )  # Normalize

        if n_agents > 1:
            # Compute each unordered agent pair once
            pair_indices = torch.triu_indices(
                n_agents,
                n_agents,
                offset=1,
                device=device,
            )
            vertices_i = data[:, pair_indices[0], 0:4, :]
            vertices_j = data[:, pair_indices[1], 0:4, :]
            axes_norm_i = axes_norm_all[:, pair_indices[0]]
            axes_norm_j = axes_norm_all[:, pair_indices[1]]

            def project(vertices, axes):
                return (vertices.unsqueeze(-2) * axes.unsqueeze(-3)).sum(dim=-1)

            # Project both rectangles in every pair onto the two local axes of
            # each rectangle. For example, ij projects i onto j's axes.
            projection_jj = project(vertices_j, axes_norm_j)
            projection_ij = project(vertices_i, axes_norm_j)
            projection_ii = project(vertices_i, axes_norm_i)
            projection_ji = project(vertices_j, axes_norm_i)

            max_jj, _ = torch.max(projection_jj, dim=-2)
            min_jj, _ = torch.min(projection_jj, dim=-2)
            max_ij, _ = torch.max(projection_ij, dim=-2)
            min_ij, _ = torch.min(projection_ij, dim=-2)
            max_ii, _ = torch.max(projection_ii, dim=-2)
            min_ii, _ = torch.min(projection_ii, dim=-2)
            max_ji, _ = torch.max(projection_ji, dim=-2)
            min_ji, _ = torch.min(projection_ji, dim=-2)

            # For each vertex, measure its gaps to the other rectangle's
            # projection intervals. A gap is zero when the projection lies
            # inside an interval; the norm across both axes is its distance.
            mtvs_ij_positive = (projection_ij - min_jj.unsqueeze(-2)) * (
                projection_ij <= min_jj.unsqueeze(-2)
            ) + (max_jj.unsqueeze(-2) - projection_ij) * (
                projection_ij >= max_jj.unsqueeze(-2)
            )
            mtvs_ji_positive = (projection_ji - min_ii.unsqueeze(-2)) * (
                projection_ji <= min_ii.unsqueeze(-2)
            ) + (max_ii.unsqueeze(-2) - projection_ji) * (
                projection_ji >= max_ii.unsqueeze(-2)
            )
            mtvs_ij_positive = torch.norm(mtvs_ij_positive, dim=-1)
            mtvs_ji_positive = torch.norm(mtvs_ji_positive, dim=-1)

            # Projection overlaps on each axis. For overlapping rectangles,
            # the distance is minus the smallest overlap, i.e., the shortest
            # translation needed to separate them.
            overlap_j = torch.min(max_jj, max_ij) - torch.max(min_jj, min_ij)
            overlap_i = torch.min(max_ii, max_ji) - torch.max(min_ii, min_ji)

            # The rectangle distance is the smallest of the eight vertex
            # distances (four vertices in each direction).
            pair_distances, _ = torch.min(
                torch.cat((mtvs_ij_positive, mtvs_ji_positive), dim=-1),
                dim=-1,
            )
            # By the separating axis theorem, the rectangles overlap iff their
            # projections overlap on all four axes. This also covers crossing
            # overlaps where no vertex lies inside the other rectangle.
            is_negative_distance = (overlap_j > 0).all(dim=-1) & (overlap_i > 0).all(
                dim=-1
            )
            negative_pair_distances = -torch.min(
                overlap_j.min(dim=-1)[0],
                overlap_i.min(dim=-1)[0],
            )
            pair_distances = torch.where(
                is_negative_distance,
                negative_pair_distances,
                pair_distances,
            )

            # Scatter the unordered-pair results into both halves of the
            # symmetric distance matrix.
            mutual_distances[:, pair_indices[0], pair_indices[1]] = pair_distances
            mutual_distances[:, pair_indices[1], pair_indices[0]] = pair_distances

    if is_set_diagonal:
        mutual_distances.diagonal(dim1=-2, dim2=-1).fill_(
            torch.sqrt(x_semidim**2 + y_semidim**2)
        )

    return mutual_distances


def interX(L1, L2, is_return_points=False):
    """Calculate the intersections of batches of curves.
        Adapted from https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections
    Args:
        L1: [batch_size, num_points, 2]
        L2: [batch_size, num_points, 2]
        is_return_points: bool. Whether to return the intersecting points.
    """
    # L1[:,:,0] -= 0.35
    # L1[:,:,1] -= 0.05
    batch_dim = L1.shape[0]
    collision_index = torch.zeros(
        batch_dim, dtype=torch.bool, device=L1.device
    )  # Initialize

    # Handle empty inputs
    if L1.numel() == 0 or L2.numel() == 0:
        return torch.empty((0, 2), device=L1.device) if is_return_points else False

    # Extract x and y coordinates
    x1, y1 = L1[..., 0], L1[..., 1]
    x2, y2 = L2[..., 0], L2[..., 1]

    # Compute differences
    dx1, dy1 = torch.diff(x1, dim=1), torch.diff(y1, dim=1)
    dx2, dy2 = torch.diff(x2, dim=1), torch.diff(y2, dim=1)

    # Determine 'signed distances'
    S1 = dx1 * y1[..., :-1] - dy1 * x1[..., :-1]
    S2 = dx2 * y2[..., :-1] - dy2 * x2[..., :-1]

    # Helper function for computing D
    def D(x, y):
        return (x[..., :-1] - y) * (x[..., 1:] - y)

    C1 = (
        D(
            dx1.unsqueeze(2) * y2.unsqueeze(1) - dy1.unsqueeze(2) * x2.unsqueeze(1),
            S1.unsqueeze(2),
        )
        < 0
    )
    C2 = (
        D(
            (
                y1.unsqueeze(2) * dx2.unsqueeze(1) - x1.unsqueeze(2) * dy2.unsqueeze(1)
            ).transpose(1, 2),
            S2.unsqueeze(2),
        )
        < 0
    ).transpose(1, 2)

    # Obtain the segments where an intersection is expected
    batch_indices, i, j = torch.where(C1 & C2)
    batch_indices_pruned = torch.sort(torch.unique(batch_indices))[0]
    collision_index[batch_indices_pruned] = True

    if is_return_points:
        # In case of collisions, return collision points; else return empty points
        if batch_indices.numel() == 0:
            return torch.empty((0, 2), device=L1.device)
        else:
            # Process intersections for each batch item
            intersections = []
            for b in batch_indices.unique():
                L = dy2[b, j] * dx1[b, i] - dy1[b, i] * dx2[b, j]
                nonzero = L != 0
                i_nz, j_nz, L_nz = i[nonzero], j[nonzero], L[nonzero]

                P = torch.stack(
                    (
                        (dx2[b, j_nz] * S1[b, i_nz] - dx1[b, i_nz] * S2[b, j_nz])
                        / L_nz,
                        (dy2[b, j_nz] * S1[b, i_nz] - dy1[b, i_nz] * S2[b, j_nz])
                        / L_nz,
                    ),
                    dim=-1,
                )
                intersections.append(P)
            # Combine intersections from all batches
            return torch.cat(intersections, dim=0)
    else:
        # Simply return whether collisions occur or not
        return collision_index


def remove_overlapping_points(polyline: torch.Tensor, threshold: float = 1e-4):
    remove = polyline.diff(dim=0).norm(dim=1) <= threshold
    remove = torch.hstack(
        (remove, torch.zeros(1, dtype=torch.bool))
    )  # Always keep the last point
    # Filter out overlapping points
    return polyline[~remove]


def transform_from_global_to_local_coordinate(
    pos_i: torch.Tensor, pos_j: torch.Tensor, rot_i
):
    """
    Args:
        pos_i: torch.Size([batch_size, 2])
        pos_j: torch.Size([batch_size, num_points, 2]) or torch.Size([num_points, 2])
        rot_i: torch.Size([batch_size, 1])
    """
    # Prepare for vectorized ccomputation
    if pos_j.ndim == 3:
        pos_i_extended = pos_i.unsqueeze(1)
    else:
        pos_i_extended = pos_i.unsqueeze(1)
        # Check if the last point overlaps with the first point
        if (pos_j[0, :] - pos_j[-1, :]).norm() == 0:
            pos_j = pos_j[0:-1, :].unsqueeze(0)
        else:
            pos_j = pos_j.unsqueeze(0)

    pos_vec = pos_j - pos_i_extended
    pos_vec_abs = pos_vec.norm(dim=2)
    rot_rel = torch.atan2(pos_vec[:, :, 1], pos_vec[:, :, 0]) - rot_i

    pos_rel = torch.stack(
        (
            torch.cos(rot_rel) * pos_vec_abs,
            torch.sin(rot_rel) * pos_vec_abs,
        ),
        dim=2,
    )

    return pos_rel


def angle_eliminate_two_pi(angle):
    """
    Normalize an angle to be within the range -pi to pi.

    Args:
        angle (torch.Tensor): The angle to normalize in radians. Can be a tensor of any shape.

    Returns:
        torch.Tensor: Normalized angle between -pi and pi.
    """
    two_pi = 2 * torch.pi
    angle = angle % two_pi  # Normalize angle to be within 0 and 2*pi
    angle[angle > torch.pi] -= two_pi  # Shift to -pi to pi range
    return angle


def get_current_lanelet_id(
    point: torch.Tensor,
    ref_lanelet_segment_point: torch.Tensor,
    n_ref_lanelet_ids: torch.Tensor,
    ref_lanelet_ids: torch.Tensor,
):
    """
    Get the current lanelet ID for each agent based on their location.
    This function computes the closest lanelet to each agent by projecting
    their location onto the lanelet's polyline segments.

    Parameters:
    ----------
    point : torch.Tensor
        A tensor containing the coordinates of the points (agents' locations).
    ref_lanelet_segment_point : torch.Tensor
        A tensor containing the start and end points of each segment of the reference path polyline.
    n_ref_lanelet_ids : torch.Tensor
        A tensor containing the number of lanelets for each path.
    ref_lanelet_ids : torch.Tensor
        A tensor containing the IDs of the reference path.

    Returns:
    -------
    current_id : torch.Tensor
        The lanelet ID of the closest lanelet to the agent's location.
    """

    # Expand and repeat for broadcasting
    max_num_lanelets = ref_lanelet_segment_point.shape[2] - 1

    # Mask valid lanelets based on n_ref_lanelet_ids
    mask = (
        torch.arange(max_num_lanelets + 1 - 1, device=n_ref_lanelet_ids.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    mask = mask < (n_ref_lanelet_ids).unsqueeze(-1)

    # Split the polyline into line segments
    line_starts = ref_lanelet_segment_point[:, :, :-1, :]
    line_ends = ref_lanelet_segment_point[:, :, 1:, :]

    # Expand point for broadcasting
    point_expanded = point.unsqueeze(2)

    # Create vectors for each line segment and for the point to the start of each segment
    line_vecs = line_ends - line_starts
    point_vecs = point_expanded - line_starts

    # Project point_vecs onto line_vecs
    line_lens_squared = torch.sum(line_vecs**2, dim=-1)
    projected_lengths = torch.sum(point_vecs * line_vecs, dim=-1) / line_lens_squared

    # Clamp the projections to lie within the line segments
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)

    # Find the closest points on the line segments to the given points
    closest_points = line_starts + (line_vecs * clamped_lengths.unsqueeze(-1))

    # Calculate the distances from the given points to these closest points
    distances = torch.norm(closest_points - point_expanded, dim=-1)

    # Apply mask to distances
    distances[~mask] = float("inf")  # Ignore invalid lanelets

    # Get the index of the nearest lanelet
    _, nearest_idx = torch.min(distances, dim=-1)

    # Gather the corresponding lanelet IDs
    current_id = torch.gather(
        ref_lanelet_ids, dim=2, index=nearest_idx.unsqueeze(-1)
    ).squeeze(-1)

    return current_id


def compute_pseudo_tangent_vector(points: torch.Tensor) -> torch.Tensor:
    """
    Approximate the tangent vector at each point of a 2D polyline.

    The tangent at a given point is estimated as follows:
    - For the first point: forward difference with the next point.
    - For the last point: backward difference with the previous point.
    - For all interior points: central difference using the next and previous points.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing a polyline with N points in 2D space.

    Returns:
        torch.Tensor: Tensor of shape (N, 2) where each row is the approximate tangent vector at the corresponding point.
    """
    num_points = points.size(0)
    # Allocate tensor for storing the tangent vectors
    tangent_vector = torch.zeros_like(points)

    if num_points >= 2:
        # Approximate the tangent vector at the first point
        tangent_vector[0] = points[1] - points[0]
        # Approximate the tangent vector at the last point
        tangent_vector[-1] = points[-1] - points[-2]

    if num_points >= 3:
        # For interior points, use central differences (next - previous)
        # Vectorized computation from index 1 to N-2
        tangent_vector[1:-1] = points[2:] - points[:-2]

    return tangent_vector
