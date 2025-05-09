# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        collide_with_boundaries=None,
        collide_with_obstacles=None,
        leave_world=None,
        time=None,
        change_steering=None,
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


class Thresholds:
    def __init__(
        self,
        deviate_from_ref_path=None,
        near_boundary_low=None,
        near_boundary_high=None,
        near_other_agents_low=None,
        near_other_agents_high=None,
        reach_goal=None,
        reach_intermediate_goal=None,
        change_steering=None,
        no_reward_if_too_close_to_boundaries=None,
        no_reward_if_too_close_to_other_agents=None,
        distance_mask_agents=None,
    ):
        self.deviate_from_ref_path = deviate_from_ref_path
        self.near_boundary_low = near_boundary_low
        self.near_boundary_high = near_boundary_high
        self.near_other_agents_low = near_other_agents_low
        self.near_other_agents_high = near_other_agents_high
        self.reach_goal = reach_goal  # Threshold less than which agents are considered at their goal positions
        self.reach_intermediate_goal = reach_intermediate_goal  # Threshold less than which agents are considered at their intermediate goal positions
        self.change_steering = change_steering
        self.no_reward_if_too_close_to_boundaries = no_reward_if_too_close_to_boundaries  # Agents get no reward if they are too close to lanelet boundaries
        self.no_reward_if_too_close_to_other_agents = no_reward_if_too_close_to_other_agents  # Agents get no reward if they are too close to other agents
        self.distance_mask_agents = (
            distance_mask_agents  # Threshold above which nearing agents will be masked
        )


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
    ):
        self.long_term_all = long_term_all  # All long-term reference paths
        self.long_term_intersection = long_term_intersection  # Long-term reference paths for the intersection scenario
        self.long_term_merge_in = (
            long_term_merge_in  # Long-term reference paths for the mergin in scenario
        )
        self.long_term_merge_out = (
            long_term_merge_out  # Long-term reference paths for the merge out scenario
        )
        self.point_extended_all = point_extended_all  # Extend the long-term reference paths by one point at the end
        self.point_extended_intersection = point_extended_intersection  # Extend the long-term reference paths by one point at the end
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
    ):
        self.n_nearing_agents = n_nearing_agents
        self.nearing_agents_indices = nearing_agents_indices
        self.noise_level = noise_level  # Whether to add noise to observations
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
    center: torch.Tensor, yaw, width, length, is_close_shape: bool = True
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


def get_perpendicular_distances(
    point: torch.Tensor, polyline: torch.Tensor, n_points_long_term: torch.Tensor = None
):
    """Calculate the minimum perpendicular distance from the given point(s) to the given polyline.

    Args:
        point: torch.Size([batch_size, 2]) or torch.Size([2]), position of the point. In the case of the latter, the batch_size is deemed to be 1.
        polyline: torch.Size([num_points, 2]) or torch.Size([batch_size, num_points, 2]) x- and y-coordinates of the points on the polyline.
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
    projected_lengths = torch.sum(point_vecs * line_vecs, dim=2) / line_lens_squared

    # Clamp the projections to lie within the line segments
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)

    # Find the closest points on the line segments to the given points
    closest_points = line_starts + (line_vecs * clamped_lengths.unsqueeze(2))

    # Calculate the distances from the given points to these closest points
    distances = torch.norm(closest_points - point_expanded, dim=2)

    if n_points_long_term is not None:
        if n_points_long_term.ndim == 0:
            n_points_long_term = n_points_long_term.unsqueeze(0)
        for env_idx, n_long_term_point in enumerate(n_points_long_term):
            distance_to_end_point = distances[env_idx, n_long_term_point - 2]
            distances[env_idx, n_long_term_point - 1 :] = distance_to_end_point

    assert not distances.isnan().any()

    perpendicular_distances, indices_closest_points = torch.min(distances, dim=1)

    indices_closest_points[
        :
    ] += 1  # Force the nearest point to lie always in the future

    return perpendicular_distances, indices_closest_points


def get_short_term_reference_path(
    polyline: torch.Tensor,
    index_closest_point: torch.Tensor,
    n_points_to_return: int,
    device=torch.device("cpu"),
    is_polyline_a_loop: torch.Tensor = False,
    n_points_long_term: torch.Tensor = None,
    sample_interval: int = 2,
    n_points_shift: int = 1,
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
    """
    if polyline.ndim == 2:
        polyline = polyline.unsqueeze(0)
    if index_closest_point.ndim == 1:
        index_closest_point = index_closest_point.unsqueeze(1)
    elif index_closest_point.ndim == 0:
        index_closest_point = index_closest_point.unsqueeze(0).unsqueeze(1)
    if is_polyline_a_loop.ndim == 0:
        is_polyline_a_loop = is_polyline_a_loop.unsqueeze(0)
    if n_points_long_term.ndim == 0:
        n_points_long_term = n_points_long_term.unsqueeze(0)

    batch_size = index_closest_point.shape[0]

    future_points_idx = (
        torch.arange(n_points_to_return, device=device) * sample_interval
        + index_closest_point
        + n_points_shift
    )

    if n_points_long_term is None:
        n_points_long_term = polyline.shape[-2]

    for env_i in range(batch_size):
        n_long_term_point = n_points_long_term[env_i]
        if is_polyline_a_loop[env_i]:
            # Apply modulo to handle the case that each agent's reference path is a loop
            future_points_idx[env_i] = torch.where(
                future_points_idx[env_i] >= n_long_term_point - 1,
                (future_points_idx[env_i] + 1) % n_long_term_point,
                future_points_idx[env_i],
            )  # Use "+ 1" to skip the last point since it overlaps with the first point

    # Extract
    short_term_path = polyline[
        torch.arange(batch_size, device=device, dtype=torch.int).unsqueeze(
            1
        ),  # For broadcasting
        future_points_idx,
    ]

    return short_term_path, future_points_idx


def exponential_decreasing_fcn(x, x0, x1):
    """
    Exponential function y(x) = (e^( -(x-x0) / (x1-x0) ) - e^-1) / (1 - e^-1), so that y decreases exponentially from 1 to 0 when x increases from x0 to x1, where
    x = max(min(x, x1), x0),
    x1 = threshold_near_boundary, and
    x0 = agent.shape.width/2.
    """
    x = torch.clamp(x, min=x0, max=x1)  # x stays inside [x0, x1]
    y = (torch.exp(-(x - x0) / (x1 - x0)) - 1 / torch.e) / (1 - 1 / torch.e)

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

        for i in range(n_agents):
            vertices_i = data[:, i, 0:4, :]
            axes_norm_i = axes_norm_all[:, i]
            for j in range(i + 1, n_agents):
                vertices_j = data[:, j, 0:4, :]
                axes_norm_j = axes_norm_all[:, j]

                # 1. Project each of the four vertices of rectangle i and all the four vertices of rectangle j to each of the two axes of rectangle j.
                # 2. The distance from a vertex of rectangle i to rectangle j is calculated by taking the Euclidean distance of the "gaps" on the two axes of rectangle j between the projected point of this vertex on the axes and the projected points of rectangle j. If the projected point of this vertex lies inside the projection of rectangle j, the gap is consider zero. If the projections of a vertex to both axes of rectangle j are inside the projection of rectangle j, this vertex is considered inside rectangle j. In this case, the distance is a negative value indicating the shortest distance that this vertex should move to be outside rectangle j, while its other three vertices are still outside rectangle j.
                # 3. Steps 1 and 2 give us four distances. Repeat these two step for rectangle j, which give us another four distances.
                # 4. The mtv-based distance between the two rectangles is the smallest distance among the eight distances.

                # Project rectangle j to its own axes
                projection_jj = (
                    vertices_j.unsqueeze(2) * axes_norm_j.unsqueeze(1)
                ).sum(dim=3)
                max_jj, _ = torch.max(projection_jj, dim=1)
                min_jj, _ = torch.min(projection_jj, dim=1)
                # Project rectangle i to the axes of rectangle j
                projection_ij = (
                    vertices_i.unsqueeze(2) * axes_norm_j.unsqueeze(1)
                ).sum(dim=3)
                max_ij, _ = torch.max(projection_ij, dim=1)
                min_ij, _ = torch.min(projection_ij, dim=1)

                MTVs_ij_positive = (projection_ij - min_jj.unsqueeze(1)) * (
                    projection_ij <= min_jj.unsqueeze(1)
                ) + (max_jj.unsqueeze(1) - projection_ij) * (
                    projection_ij >= max_jj.unsqueeze(1)
                )
                MTVs_ij_Euclidean_positive = torch.norm(MTVs_ij_positive, dim=2)
                # Overlap of the projections on the axes of rectangle j
                overlap_j = torch.min(max_jj, max_ij) - torch.max(min_jj, min_ij)
                MTVs_ij_Euclidean_negative = -overlap_j.min(dim=-1)[0].unsqueeze(-1) * (
                    (projection_ij > min_jj.unsqueeze(1))
                    * (projection_ij < max_jj.unsqueeze(1))
                ).all(dim=-1)
                # MTVs_ij_Euclidean = MTVs_ij_Euclidean_positive + MTVs_ij_Euclidean_negative

                # Project rectangle i to its own axes
                projection_ii = (
                    vertices_i.unsqueeze(2) * axes_norm_i.unsqueeze(1)
                ).sum(dim=3)
                max_ii, _ = torch.max(projection_ii, dim=1)
                min_ii, _ = torch.min(projection_ii, dim=1)
                # Project rectangle j to the axes of rectangle i
                projection_ji = (
                    vertices_j.unsqueeze(2) * axes_norm_i.unsqueeze(1)
                ).sum(dim=3)
                max_ji, _ = torch.max(projection_ji, dim=1)
                min_ji, _ = torch.min(projection_ji, dim=1)
                MTVs_ji_positive = (projection_ji - min_ii.unsqueeze(1)) * (
                    projection_ji <= min_ii.unsqueeze(1)
                ) + (max_ii.unsqueeze(1) - projection_ji) * (
                    projection_ji >= max_ii.unsqueeze(1)
                )
                MTVs_ji_Euclidean_positive = torch.norm(MTVs_ji_positive, dim=2)
                overlap_i = torch.min(max_ii, max_ji) - torch.max(min_ii, min_ji)

                MTVs_ji_Euclidean_negative = -overlap_i.min(dim=1)[0].unsqueeze(-1) * (
                    (projection_ji > min_ii.unsqueeze(1))
                    * (projection_ji < max_ii.unsqueeze(1))
                ).all(dim=-1)

                # The distance from rectangle j to rectangle i is calculated as the Euclidean distance of the two lengths of the MTVs on the two axes of rectangle i
                distance_ji, _ = torch.min(
                    torch.hstack(
                        (MTVs_ij_Euclidean_positive, MTVs_ji_Euclidean_positive)
                    ),
                    dim=1,
                )
                is_negative_distance = (MTVs_ij_Euclidean_negative.abs() > 0).any(
                    dim=-1
                ) | (MTVs_ji_Euclidean_negative.abs() > 0).any(dim=-1)
                distance_ji[is_negative_distance] = -torch.min(
                    overlap_j.min(dim=1)[0], overlap_i.min(dim=1)[0]
                )[is_negative_distance]

                # # Check rectangles are overlapping
                # is_projection_overlapping = torch.hstack(
                #     (
                #         (max_ii >= min_ji) & (min_ii <= max_ji),
                #         (max_jj >= min_ij) & (min_jj <= max_ij),
                #     )
                # )
                # is_overlapping = torch.all(is_projection_overlapping, dim=1)
                # distance_ji[is_overlapping] = 0  # Rectangles are overlapping

                mutual_distances[
                    :, i, j
                ] = distance_ji  # The smaller one among the distance from rectangle i to j and the distance from rectangle j to i is define as the distance between two rectangles
                mutual_distances[:, j, i] = distance_ji

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
    collision_index = torch.zeros(batch_dim, dtype=torch.bool)  # Initialize

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
