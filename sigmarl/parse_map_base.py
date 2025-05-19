# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from sigmarl.colors import (
    Color,
)

from sigmarl.constants import SCENARIOS, AGENTS, THRESHOLD

from sigmarl.helper_scenario import (
    compute_pseudo_tangent_vector,
    get_rectangle_vertices,
)


class ParseMapBase(ABC):
    """
    Base class for map parse.
    """

    def __init__(self, scenario_type, device, **kwargs):
        self._scenario_type = scenario_type  # Path to the map data
        self._device = device  # Torch device

        self._get_map_path()

        self._is_visualize_map = kwargs.pop("is_visualize_map", False)
        self._is_save_fig = kwargs.pop("is_save_fig", False)
        self._is_plt_show = kwargs.pop("is_plt_show", False)
        self._is_visu_lane_ids = kwargs.pop("is_visu_lane_ids", False)
        self._is_visualize_random_agents = kwargs.pop(
            "is_visualize_random_agents", False
        )
        self._n_agents_visu = kwargs.pop("n_agents_visu", None)
        self._is_visualize_intersection = kwargs.pop(
            "is_visualize_intersection", False
        )  # For the CPM Scenario only
        self._is_visualize_entry_direction = kwargs.pop(
            "is_visualize_entry_direction", False
        )  # Use an arrow to indicate the direction of entries

        self._width = kwargs.pop("lane_width", None)

        if self._width is None:
            # Load the lane width from the scenario configuration if not provided
            self._width = SCENARIOS[self._scenario_type][
                "lane_width"
            ]  # Width of the lane

        self._scale = SCENARIOS[self._scenario_type]["scale"]  # Scale the map

        self._is_share_lanelets = kwargs.pop(
            "is_share_lanelets", False
        )  # Whether agents can move to nearing lanelets

        self._is_show_axis = kwargs.pop(
            "is_show_axis", False
        )  # Whether to show the x- and y- axes of the map

        self.bounds = {
            "min_x": float("inf"),
            "min_y": float("inf"),
            "max_x": float("-inf"),
            "max_y": float("-inf"),
            "world_x_dim": float("inf"),
            "world_y_dim": float("inf"),
        }  # Bounds of the map

        self.lanelets_all = (
            []
        )  # A list of dict. Each dict stores relevant data of a lane such as its center line, left boundary, and right boundary
        self.neighboring_lanelets_idx = []  # Neighboring lanelets of each lanelet

        self.reference_paths = []
        self.reference_paths_intersection = []
        self.reference_paths_merge_in = []
        self.reference_paths_merge_out = []

        self._intersection_info = []  # For the CPM Scenario only

        self._linewidth = 0.5
        self._fontsize = 9

        # Use the same name for data to be stored as the scenario type
        self._scenario_type = self._scenario_type

    def _get_map_path(self):
        # Get the path to the corresponding map for the given scenario type
        self._map_path = SCENARIOS[self._scenario_type]["map_path"]

    def _compute_pseudo_tangent_vectors(self):
        for ref_path in self.reference_paths:
            ref_path["left_boundary_pseudo_vector"] = compute_pseudo_tangent_vector(
                ref_path["left_boundary"]
            )
            ref_path["right_boundary_pseudo_vector"] = compute_pseudo_tangent_vector(
                ref_path["right_boundary"]
            )
            ref_path[
                "left_boundary_shared_pseudo_vector"
            ] = compute_pseudo_tangent_vector(ref_path["left_boundary_shared"])
            ref_path[
                "right_boundary_shared_pseudo_vector"
            ] = compute_pseudo_tangent_vector(ref_path["right_boundary_shared"])

    def _visualize_random_agents(self, ax, reference_paths):
        # Get the number of agents to be visualized

        initial_distance_threshold = THRESHOLD["initial_distance"]
        width = AGENTS["width"]
        length = AGENTS["length"]

        positions, rotations = self.generate_random_states(
            reference_paths,
            self._device,
            self._n_agents_visu,
            initial_distance_threshold,
        )

        vertices = get_rectangle_vertices(
            center=positions,
            yaw=rotations,
            width=width,
            length=length,
            is_close_shape=True,
        )  # Get the vertices of the agents

        for i_agent in range(self._n_agents_visu):
            # plt.fill(vertices[i_agent, :, 0], vertices[i_agent, :, 1], color="grey")
            polygon = plt.Polygon(
                vertices[i_agent],
                closed=True,
                edgecolor="black",
                linewidth=0.4,
                facecolor="tab:blue",
                zorder=3,
            )
            ax.add_patch(polygon)

    @staticmethod
    def get_center_length_yaw_polyline(polyline: torch.Tensor):
        """
        This function calculates the center points, lengths, and yaws of all line segments of the given polyline.
        """

        center_points = polyline.unfold(0, 2, 1).mean(dim=2)

        polyline_vecs = polyline.diff(dim=0)
        lengths = polyline_vecs.norm(dim=1)
        yaws = torch.atan2(polyline_vecs[:, 1], polyline_vecs[:, 0])

        return center_points, lengths, yaws, polyline_vecs

    @staticmethod
    def generate_random_states(
        reference_paths, device, n_agents, initial_distance_threshold
    ):
        positions = torch.zeros((n_agents, 2), device=device, dtype=torch.float32)
        rotations = torch.zeros((n_agents, 1), device=device, dtype=torch.float32)

        for i_agent in range(n_agents):
            # Get random states
            is_feasible_initial_position_found = False
            random_count = 0
            while not is_feasible_initial_position_found:
                if random_count >= 20:
                    print(f"Reset agent(s): random_count = {random_count}.")
                random_count += 1

                path_id = torch.randint(
                    0, len(reference_paths), (1,)
                ).item()  # Select randomly a path
                ref_path = reference_paths[path_id]

                num_points = ref_path["center_line"].shape[0]

                start_point_idx = 6  # Do not set to an overly small value to make sure agents are fully inside its lane
                end_point_idx = num_points - 3

                random_point_id = torch.randint(
                    start_point_idx, end_point_idx, (1,)
                ).item()

                positions[i_agent, :] = ref_path["center_line"][random_point_id]

                # Check if the initial position is feasible
                if i_agent == 0:
                    # The initial position of the first agent is always feasible
                    is_feasible_initial_position_found = True
                else:
                    diff_sq = (
                        positions[i_agent, :] - positions[0:i_agent, :]
                    ) ** 2  # Calculate pairwise squared distances between the current agent and the agents that have already obtained feasible initial positions
                    initial_mutual_distances_sq = torch.sum(diff_sq, dim=-1)
                    min_distance_sq = torch.min(initial_mutual_distances_sq)

                    is_feasible_initial_position_found = min_distance_sq >= (
                        initial_distance_threshold**2
                    )

            rotations[i_agent] = ref_path["center_line_yaw"][random_point_id]

        return positions, rotations

    @abstractmethod
    def _parse_map_file():
        raise NotImplementedError()
