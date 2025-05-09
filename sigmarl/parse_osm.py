# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import xml.etree.ElementTree as ET
from importlib import resources

import torch
import matplotlib.pyplot as plt
import matplotlib  # Set up font

matplotlib.rcParams["pdf.fonttype"] = 42  # Use Type 1 fonts (vector fonts)
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Georgia"]
matplotlib.rcParams.update({"font.size": 11})  # Set global font size

from sigmarl.parse_map_base import ParseMapBase

from sigmarl.colors import Color

from sigmarl.constants import SCENARIOS

import numpy as np


class ParseOSM(ParseMapBase):
    """
    ParseOSM is a class to parse OSM files generated by JOSM software (https://josm.openstreetmap.de/), which provides coordinates of center lines in GPS system without left and right boundaries.
    Therefore, this class also provides function to calculate left and right boundaries of the given center lines and the given lane width.

    There are a few points worth notin:
    - OSM files generated by JOSM store data in earth coordinate system, i.e., longitudinal and lateral coordinates. We map longitudinal coordinates to x coordinates and lateral coordinates to y coordinates. They will be simply scaled by a given factor `scale`.
    - When designing road network in JOSM, please assign each "way" (JOSM calls a lanelet a way) a distinct, integer tag. This servers as ID of the way.
    """

    def __init__(self, scenario_type, device, **kwargs):
        super().__init__(scenario_type, device, **kwargs)  # Initialize base class

        try:
            self._reference_paths_ids = SCENARIOS[scenario_type][
                "reference_paths_ids"
            ]  # A list of lists. Each sub-list stores the IDs of lanelets building a reference path
            self._neighboring_lanelet_ids = SCENARIOS[scenario_type][
                "neighboring_lanelet_ids"
            ]
            self._fig_title = SCENARIOS[scenario_type]["name"]
        except (KeyError):
            raise KeyError(
                f"Scenario '{scenario_type}' does not exist. If you have added new scenarios, please include them in `SCENARIOS` in utilities/constants.py."
            )

        self._nodes = {}
        self._ways = {}

        self._parse_map_file()
        self._process_map_data()
        self._get_reference_paths()
        self._compute_pseudo_tangent_vectors()
        self._determine_neighboring_lanelets()

        self._get_map_dimension()

        if self._is_visualize_map:
            self.visualize_map()

    def _parse_map_file(self):
        """Parse the OSM file and extract bounds, nodes, and ways."""
        with resources.open_binary(
            "sigmarl.scenarios.assets.maps", self._map_path
        ) as map_file:
            tree = ET.parse(map_file)
        root = tree.getroot()

        # Extract nodes
        node_data = {}
        for node in root.findall("node"):
            node_id = int(node.get("id"))
            lat = float(node.get("lat"))
            lon = float(node.get("lon"))
            node_data[node_id] = (lat, lon)

            # Update bounds
            if lat < self.bounds["min_x"]:
                self.bounds["min_x"] = lat
            if lon < self.bounds["min_y"]:
                self.bounds["min_y"] = lon
            if lat > self.bounds["max_x"]:
                self.bounds["max_x"] = lat
            if lon > self.bounds["max_y"]:
                self.bounds["max_y"] = lon

        # Scale and shift nodes
        for node_id, (lat, lon) in node_data.items():
            lat = (
                lat - self.bounds["min_x"]
            ) * self._scale + self._width * 1.2  # Plus a small margin to to ensure the coordinates of left and right boundaries will still be positive
            lon = (lon - self.bounds["min_y"]) * self._scale + self._width * 1.2
            self._nodes[node_id] = (lat, lon)

        # Extract ways
        for way in root.findall("way"):
            way_id = int(way.get("id"))
            node_refs = [int(nd.get("ref")) for nd in way.findall("nd")]
            tag = way.find("tag[@k='lanes']")
            lanes = int(tag.get("v")) if tag is not None else None
            self._ways[way_id] = {"nodes": node_refs, "lanes": lanes}

    def _process_map_data(self):
        """
        Calculate the relevant data for each center line and store them in self.lanelets_all.
        """
        lanelets_all_tmp = []
        # self._lanelet_id_to_index = {}  # Dictionary to store the mapping

        for _, way in self._ways.items():
            lanelet_id = way["lanes"]
            if lanelet_id is not None:
                # Only lanelets with IDs will be considered, i.e., only ways in JOSM that have a tag will be considered
                if not isinstance(lanelet_id, int):
                    raise ValueError(
                        "At least one lanelet has non-integer tag. Please check your OSM file."
                    )
                center_line_points = [self._nodes[node_id] for node_id in way["nodes"]]
                center_line = torch.tensor(
                    center_line_points, device=self._device, dtype=torch.float32
                )
                (
                    center_line_yaw,
                    center_line_vec_normalized,
                    center_line_vec_mean_length,
                ) = self._compute_center_line_info(center_line)
                left_boundary, right_boundary = self._compute_boundaries(center_line)

                # Find predecessors
                (
                    predecessors,
                    successors,
                ) = self._find_direct_predecessors_and_successors(lanelet_id)

                lanelets_all_tmp.append(
                    {
                        str(lanelet_id): {
                            "center_line": center_line,
                            "center_line_yaw": center_line_yaw,
                            "center_line_vec_normalized": center_line_vec_normalized,
                            "center_line_vec_mean_length": center_line_vec_mean_length,
                            "left_boundary": left_boundary,
                            "right_boundary": right_boundary,
                            "left_boundary_shared": left_boundary,  # Should be calculated if they are not the same
                            "right_boundary_shared": right_boundary,  # Should be calculated if they are not the same
                            "predecessor": predecessors,  # Predecessor lanelets
                            "successor": successors,  # Successor lanelets
                        }
                    }
                )
            else:
                print(
                    "At least one lanelet does not have a tag. These lanelets will not be considered."
                )

        # sorted_list = sorted(lanelets_all_tmp, key=lambda x: next(iter(x)))

        # Extracting the values from the list of dictionaries
        self.lanelets_all = [list(d.values())[0] for d in lanelets_all_tmp]

        # Convert to a dict
        tmp_dict = {int(k): v for d in lanelets_all_tmp for k, v in d.items()}

        # Create a sorted list where index = key - 1, and keep only the value
        max_index = max(tmp_dict.keys())
        self.lanelets_all = [tmp_dict[i + 1] for i in range(max_index)]

        # # Creating the mapping from way["lanes"] to the index after sorting
        # for index, lanelet_dict in enumerate(lanelets_all_tmp):
        #     lanelet_id = next(iter(lanelet_dict))
        #     self._lanelet_id_to_index[lanelet_id] = index

    def _find_direct_predecessors_and_successors(self, lanelet_id: int):
        target_id = str(lanelet_id)

        predecessors = set()
        successors = set()

        for path in self._reference_paths_ids:
            for i in range(len(path)):
                if path[i] == target_id:
                    # Check direct predecessor
                    if i > 0:
                        predecessors.add(int(path[i - 1]))
                    # Check direct successor
                    if i < len(path) - 1:
                        successors.add(int(path[i + 1]))

        return list(predecessors), list(successors)

    def _get_reference_paths(self):
        """
        Get the reference paths based on reference paths IDs.

        Returns:
            List of dict: Each dict contains information about a reference path.
        """
        for ref_path_ids in self._reference_paths_ids:
            center_line_points = []

            # Check if the reference path is a loop
            is_loop = len(ref_path_ids) > 1 and ref_path_ids[0] == ref_path_ids[-1]

            for path_idx in range(len(ref_path_ids)):
                way_id = str(ref_path_ids[path_idx])
                # way_idx = self._lanelet_id_to_index[way_id]
                way_idx = (
                    int(way_id) - 1
                )  # Convert lanelet ID to index (lanelet ID = index + 1

                way_data = self.lanelets_all[way_idx]
                nodes_to_add = way_data["center_line"]

                if (
                    path_idx > 0
                ):  # Avoid repeated nodes, since the first node of one lane overlaps with the last node of its predecessor lane
                    nodes_to_add = nodes_to_add[1:]

                center_line_points.extend(nodes_to_add)

            if is_loop and center_line_points:
                # Delete the last node since it overlaps with the first node in case of a loop-shaped reference path
                center_line_points.pop()

            if center_line_points:
                center_line = torch.stack(center_line_points)
                (
                    center_line_yaw,
                    center_line_vec_normalized,
                    center_line_vec_mean_length,
                ) = self._compute_center_line_info(center_line)
                left_boundary, right_boundary = self._compute_boundaries(center_line)

                ref_path_ids_int = [int(k) - 1 for k in ref_path_ids]
                self.reference_paths.append(
                    {
                        "lanelet_IDs": ref_path_ids_int,
                        "center_line": center_line,
                        "center_line_yaw": center_line_yaw,
                        "center_line_vec_normalized": center_line_vec_normalized,
                        "center_line_vec_mean_length": center_line_vec_mean_length,
                        "left_boundary": left_boundary,
                        "right_boundary": right_boundary,
                        "left_boundary_shared": left_boundary,
                        "right_boundary_shared": right_boundary,
                        "is_loop": is_loop,
                    }
                )

    def _determine_neighboring_lanelets(self):
        # Convert to list of lists of int(n) - 1, where n is the lanelet ID
        max_index = max(int(k) for k in self._neighboring_lanelet_ids.keys())
        self.neighboring_lanelets_idx = [
            [int(n) - 1 for n in self._neighboring_lanelet_ids[str(i + 1)]]
            for i in range(max_index)
        ]

    def _compute_center_line_info(self, center_line):
        """Compute additional information about the center line."""
        center_line_vec = torch.diff(
            center_line, dim=0
        )  # Vectors connecting each pair of neighboring points on the center line
        center_line_vec_length = torch.norm(
            center_line_vec, dim=1
        )  # The lengths of the vectors
        center_line_vec_mean_length = torch.mean(
            center_line_vec_length
        )  # The mean length of the vectors
        center_line_vec_normalized = center_line_vec / center_line_vec_length.unsqueeze(
            1
        )

        center_line_yaw = torch.atan2(center_line_vec[:, 1], center_line_vec[:, 0])

        return center_line_yaw, center_line_vec_normalized, center_line_vec_mean_length

    def _compute_boundaries(self, center_line):
        """Compute left and right boundaries of the given center line."""

        def normalize(v):
            norm = torch.norm(v)
            return v if norm == 0 else v / norm

        left_boundary = []
        right_boundary = []

        for i in range(len(center_line) - 1):
            p1 = center_line[i]
            p2 = center_line[i + 1]
            direction = p2 - p1
            perp_direction = torch.tensor([-direction[1], direction[0]])
            perp_direction = normalize(perp_direction)

            left_boundary.append(p1 + perp_direction * self._width / 2)
            right_boundary.append(p1 - perp_direction * self._width / 2)

        left_boundary.append(center_line[-1] + perp_direction * self._width / 2)
        right_boundary.append(center_line[-1] - perp_direction * self._width / 2)

        return torch.stack(left_boundary), torch.stack(right_boundary)

    def _get_map_dimension(self):
        # Collect all x and y coordinates to determine limits
        all_x = []
        all_y = []

        for path in self.reference_paths:
            center_line = path["center_line"]
            left_boundary = path["left_boundary"]
            right_boundary = path["right_boundary"]

            all_x.extend(center_line[:, 0].tolist())
            all_y.extend(center_line[:, 1].tolist())
            all_x.extend(left_boundary[:, 0].tolist())
            all_y.extend(left_boundary[:, 1].tolist())
            all_x.extend(right_boundary[:, 0].tolist())
            all_y.extend(right_boundary[:, 1].tolist())

        # Determine limits
        x_lim_min = min(all_x)
        x_lim_max = max(all_x)
        y_lim_min = min(all_y)
        y_lim_max = max(all_y)

        self.bounds["min_x"] = x_lim_min
        self.bounds["max_x"] = x_lim_max
        self.bounds["min_y"] = y_lim_min
        self.bounds["max_y"] = y_lim_max
        self.bounds["world_x_dim"] = x_lim_max + x_lim_min
        self.bounds["world_y_dim"] = y_lim_max + y_lim_min

    def visualize_map(self):
        """
        Visualize the map.
        """

        # Set up the plot
        aspect_ratio = (self.bounds["max_y"] - self.bounds["min_y"]) / (
            self.bounds["max_x"] - self.bounds["min_x"]
        )
        figsize_x = SCENARIOS[self._scenario_type]["figsize_x"]
        fig, ax = plt.subplots(
            figsize=(figsize_x, figsize_x * aspect_ratio), constrained_layout=True
        )
        ax.set_aspect("equal")

        for path in self.reference_paths:
            center_line = path["center_line"]
            left_boundary = path["left_boundary"]
            right_boundary = path["right_boundary"]
            center_line_vec_normalized = path["center_line_vec_normalized"]

            ax.plot(
                center_line[:, 0],
                center_line[:, 1],
                color=Color.black50,
                linestyle="--",
                linewidth=self._linewidth,
                label="Center Line",
                zorder=1,
            )
            ax.plot(
                left_boundary[:, 0],
                left_boundary[:, 1],
                color=Color.black50,
                linestyle="-",
                linewidth=self._linewidth,
                label="Left Boundary",
                zorder=1,
            )
            ax.plot(
                right_boundary[:, 0],
                right_boundary[:, 1],
                color=Color.black50,
                linestyle="-",
                linewidth=self._linewidth,
                label="Right Boundary",
                zorder=1,
            )

            # Fill the area between boundaries
            curve_close = torch.vstack([left_boundary, right_boundary.flip(0)])
            # ax.fill(curve_close[:,0], curve_close[:,1], color="lightgrey", alpha=0.4)

            # Add arrows to indicate direction
            if self._is_visualize_entry_direction:
                p_start = center_line[0]
                direction_start = center_line_vec_normalized[0]
                ax.quiver(
                    p_start[0],
                    p_start[1],
                    direction_start[0],
                    direction_start[1],
                    angles="xy",
                    scale_units="xy",
                    scale=3,
                    color="black",
                    zorder=2,
                )

            if self._is_visu_lane_ids:
                ax.text(
                    center_line[int(len(center_line) / 2), 0],
                    center_line[int(len(center_line) / 2), 1],
                    str(path["lanelet_IDs"]),
                    fontsize=self._fontsize,
                    zorder=2,
                )

        if self._is_visualize_random_agents:
            if self._n_agents_visu is None:
                self._n_agents_visu = SCENARIOS[self._scenario_type]["n_agents"]
            self._visualize_random_agents(ax, self.reference_paths)

        if self._is_show_axis:
            ax.set_xlabel(r"$x$ [m]", fontsize=self._fontsize)
            ax.set_ylabel(r"$y$ [m]", fontsize=self._fontsize)
            ax.set_xticks(
                np.arange(self.bounds["min_x"], self.bounds["max_x"] + 0.05, 1.0)
            )
            ax.set_yticks(
                np.arange(self.bounds["min_y"], self.bounds["max_y"] + 0.05, 1.0)
            )

            # Set all spines (outer box lines) to gray
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_color("gray")  # or use a specific shade, e.g., '#888888'
            # Set tick marks and labels to gray
            ax.tick_params(axis="both", colors="gray")  # both ticks and tick labels
            ax.xaxis.label.set_color("gray")
            ax.yaxis.label.set_color("gray")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove the outer box
            for spine in ax.spines.values():
                spine.set_visible(False)

        ax.set_xlim((self.bounds["min_x"], self.bounds["max_x"]))
        ax.set_ylim((self.bounds["min_y"], self.bounds["max_y"]))

        ax.grid(False)

        ax.autoscale()

        if self._is_save_fig:
            # Save fig
            fig_name = "map_" + self._scenario_type + ".pdf"
            plt.savefig(fig_name, format="pdf", bbox_inches="tight")
            print(f"A fig is saved at {fig_name}")

        if self._is_plt_show:
            plt.show()

        return fig, ax


if __name__ == "__main__":
    scenario_types = [
        "interchange_1",
        "interchange_2",
        "interchange_3",
        "intersection_1",
        "intersection_2",
        "intersection_3",
        "intersection_4",
        "intersection_5",
        "intersection_6",
        "intersection_7",
        "intersection_8",
        "on_ramp_1",
        "on_ramp_2_multilane",
        "roundabout_1",
        "roundabout_2",
        # "pseudo_distance_example",
    ]  # See sigmarl/constants.py for all available maps

    for scenario_type in scenario_types:
        # scenario_type = "intersection_2"
        print("---------------------------------------------------")
        print(f"---------------- Scenario: {scenario_type} -------------------")
        print("---------------------------------------------------")
        parser = ParseOSM(
            scenario_type=scenario_type,
            lane_width=0.2 if scenario_types == "intersection_3" else 0.3,
            device="cpu" if not torch.cuda.is_available() else "cuda:0",
            is_share_lanelets=False,
            is_visualize_map=True,
            is_visualize_random_agents=True,
            n_agents_visu=1,
            is_save_fig=True,
            is_plt_show=False,
            is_visu_lane_ids=False,
        )
