# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import xml.etree.ElementTree as ET
from importlib import resources

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

import numpy as np

from sigmarl.parse_map_base import ParseMapBase

from sigmarl.colors import Color

from sigmarl.constants import SCENARIOS, AGENTS, THRESHOLD


class ParseXML(ParseMapBase):
    """
    This class is used to parse the XML file of the map of the CPM Scenario.
    See https://github.com/bassamlab/assets/blob/main/sigmarl/media/cpm_scenario_all_lanelets_only_center_line.pdf for an visualization of the map.
    """

    def __init__(self, scenario_type, device="cpu", **kwargs):
        self._scale_map = kwargs.pop("scale_map", 1.0)
        if not math.isfinite(self._scale_map) or self._scale_map <= 0.0:
            raise ValueError("`scale_map` must be a finite number greater than 0.")
        # Point-matching tolerance [m]; grows with the map so that points that
        # coincide in the unscaled map still coincide after scaling
        self._point_tol = 1e-4 * self._scale_map
        self._is_sample_equal_distance = kwargs.pop("is_sample_equal_distance", True)
        self._is_align_boundary_points = kwargs.pop("is_align_boundary_points", True)
        self._is_center_centerline = kwargs.pop("is_center_centerline", True)
        super().__init__(scenario_type, device, **kwargs)  # Initialize base class

        self._intersection_info = []

        # Define reference paths
        # self._reference_paths_ids = [
        #     [4, 6, 8, 60, 58, 56, 54, 80, 82, 84, 86, 34, 32, 30, 28, 2],  # Loop 1
        #     [1, 3, 23, 10, 12, 17, 43, 38, 36, 49, 29, 27],  # Loop 2
        #     [64, 62, 75, 55, 53, 79, 81, 101, 88, 90, 95, 69],  # Loop 3
        #     [40, 45, 97, 92, 94, 100, 83, 85, 33, 31, 48, 42],  # Loop 4
        #     [5, 7, 59, 57, 74, 68, 66, 71, 19, 14, 16, 22],  # Loop 5
        #     [41, 39, 20, 63, 61, 57, 55, 67, 65, 98, 37, 35, 31, 29],  # Loop 6
        #     [3, 5, 9, 11, 72, 91, 93, 81, 83, 87, 89, 46, 13, 15],  # Loop 7
        # ]

        # All "inner" loops
        self._reference_paths_ids = [
            [
                1,
                3,
                5,
                7,
                59,
                57,
                55,
                53,
                79,
                81,
                83,
                85,
                33,
                31,
                29,
                27,
            ],  # Loop 1 (Big "o" loop)
            [
                1,
                3,
                23,
                10,
                12,
                17,
                43,
                38,
                36,
                49,
                29,
                27,
            ],  # Loop 2 (Upper horizontal loop)
            [
                64,
                62,
                75,
                55,
                53,
                79,
                81,
                101,
                88,
                90,
                95,
                69,
            ],  # Loop 3 (Lower horizontal loop)
            [
                40,
                45,
                97,
                92,
                94,
                100,
                83,
                85,
                33,
                31,
                48,
                42,
            ],  # Loop 4 (Left vertical loop)
            [
                5,
                7,
                59,
                57,
                74,
                68,
                66,
                71,
                19,
                14,
                16,
                22,
            ],  # Loop 5 (Right vertical loop)
            [
                48,
                42,
                40,
                21,
                64,
                62,
                75,
                74,
                68,
                66,
                99,
                38,
                36,
                49,
            ],  # Loop 6 (135 degree "8" loop)
            [
                14,
                16,
                22,
                23,
                10,
                12,
                73,
                92,
                94,
                100,
                101,
                88,
                90,
                47,
            ],  # Loop 7 (45 degree "8" loop)
        ]

        self._intersection_outer_boundary_ids = [
            18,
            44,
            96,
            70,
        ]  # Take the right boundaries of these lanelets to form the outer boundaries of the intersection

        # Hard code the size of the map in the CPM Lab
        self.bounds["min_x"] = SCENARIOS[self._scenario_type]["x_dim_min"]
        self.bounds["max_x"] = SCENARIOS[self._scenario_type]["x_dim_max"]
        self.bounds["min_y"] = SCENARIOS[self._scenario_type]["y_dim_min"]
        self.bounds["max_y"] = SCENARIOS[self._scenario_type]["y_dim_max"]

        self.bounds["world_x_dim"] = self.bounds["min_x"] + self.bounds["max_x"]
        self.bounds["world_y_dim"] = self.bounds["min_y"] + self.bounds["max_y"]

        map_min_x = self.bounds["min_x"]
        map_max_x = self.bounds["max_x"]
        map_min_y = self.bounds["min_y"]
        map_max_y = self.bounds["max_y"]
        self._map_center = (
            (map_min_x + map_max_x) / 2.0,
            (map_min_y + map_max_y) / 2.0,
        )
        self._map_translation = (0.0, 0.0)
        if self._scale_map != 1.0:
            print(f"[INFO] Scaling XML map by factor {self._scale_map:g}.")
            scaled_min_x = (
                self._map_center[0]
                + (map_min_x - self._map_center[0]) * self._scale_map
            )
            scaled_min_y = (
                self._map_center[1]
                + (map_min_y - self._map_center[1]) * self._scale_map
            )
            self._map_translation = (
                map_min_x - scaled_min_x,
                map_min_y - scaled_min_y,
            )
            self.bounds["max_x"] = map_min_x + (map_max_x - map_min_x) * self._scale_map
            self.bounds["max_y"] = map_min_y + (map_max_y - map_min_y) * self._scale_map
            self.bounds["world_x_dim"] = self.bounds["max_x"] - self.bounds["min_x"]
            self.bounds["world_y_dim"] = self.bounds["max_y"] - self.bounds["min_y"]

        self._parse_map_file()

        self._get_reference_paths(is_show_each_ref_path=False)

        if self._is_sample_equal_distance:
            self._resample_reference_paths_equal_distance()

        if self._is_align_boundary_points or self._is_center_centerline:
            self._align_reference_path_triplets()

        if self._is_center_centerline:
            self._center_reference_path_centerlines()

        self._compute_pseudo_tangent_vectors()

        if self._is_visualize_map:
            self.visualize_map(
                is_visu_intersection_only=self._is_visu_intersection_only
            )

    def _get_reference_path_collections(self):
        """Return all XML reference-path collections processed by shared helpers."""
        return (
            self.reference_paths,
            self.reference_paths_intersection,
            self.reference_paths_intersection_extended,
            self.reference_paths_merge_in,
            self.reference_paths_merge_out,
        )

    def _parse_map_file(self):
        with resources.path(
            "sigmarl.scenarios.assets.maps", self._map_path
        ) as map_path:
            tree = ET.parse(map_path)
        root = tree.getroot()
        lanelets = []
        for child in root:
            if child.tag == "lanelet":
                lanelets.append(self._parse_lanelet(child))
            elif child.tag == "intersection":
                self._parse_intersections(child)

        # Storing all the map data
        self.lanelets_all = lanelets

    def _parse_lanelet(self, element):
        """Parses a lanelet element to extract detailed information."""
        lanelet_data = {
            "id": int(element.get("id")),
            "left_boundary": [],
            "left_boundary_center_points": [],
            "left_boundary_lengths": [],
            "left_boundary_yaws": [],
            "left_line_marking": None,
            "right_boundary": [],
            "right_boundary_center_points": [],
            "right_boundary_lengths": [],
            "right_boundary_yaws": [],
            "right_line_marking": None,
            "center_line": [],
            "center_line_center_points": [],
            "center_line_lengths": [],
            "center_line_yaws": [],
            "center_line_marking": "dashed",
            "predecessor": [],
            "successor": [],
            "adjacent_left": None,
            "adjacent_right": None,
            "lanelet_type": None,
        }
        for child in element:
            if child.tag == "leftBound":
                (
                    lanelet_data["left_boundary"],
                    lanelet_data["left_line_marking"],
                ) = self._parse_bound(child)
            elif child.tag == "rightBound":
                (
                    lanelet_data["right_boundary"],
                    lanelet_data["right_line_marking"],
                ) = self._parse_bound(child)
            elif child.tag == "predecessor":
                lanelet_data["predecessor"].append(int(child.get("ref")))
            elif child.tag == "successor":
                lanelet_data["successor"].append(int(child.get("ref")))
            elif child.tag == "adjacentLeft":
                lanelet_data["adjacent_left"] = {
                    "id": int(child.get("ref")),
                    "drivingDirection": child.get("drivingDir"),
                }
            elif child.tag == "adjacentRight":
                lanelet_data["adjacent_right"] = {
                    "id": int(child.get("ref")),
                    "drivingDirection": child.get("drivingDir"),
                }
            elif child.tag == "lanelet_type":
                lanelet_data["lanelet_type"] = child.text

        lanelet_data["center_line"] = (
            lanelet_data["left_boundary"] + lanelet_data["right_boundary"]
        ) / 2

        (
            lanelet_data["center_line_center_points"],
            lanelet_data["center_line_lengths"],
            lanelet_data["center_line_yaws"],
            _,
        ) = ParseMapBase.get_center_length_yaw_polyline(
            polyline=lanelet_data["center_line"]
        )
        (
            lanelet_data["left_boundary_center_points"],
            lanelet_data["left_boundary_lengths"],
            lanelet_data["left_boundary_yaws"],
            _,
        ) = ParseMapBase.get_center_length_yaw_polyline(
            polyline=lanelet_data["left_boundary"]
        )
        (
            lanelet_data["right_boundary_center_points"],
            lanelet_data["right_boundary_lengths"],
            lanelet_data["right_boundary_yaws"],
            _,
        ) = ParseMapBase.get_center_length_yaw_polyline(
            polyline=lanelet_data["right_boundary"]
        )

        return lanelet_data

    # Function to parse intersections
    def _parse_intersections(self, element):
        for incoming in element.findall("incoming"):
            incoming_info = {
                "incomingLanelet": int(
                    incoming.find("incomingLanelet").get("ref")
                ),  # The starting lanelet of a part of the intersection
                "successorsRight": int(
                    incoming.find("successorsRight").get("ref")
                ),  # The successor right lanelet of the incoming lanelet
                "successorsStraight": [
                    int(s.get("ref")) for s in incoming.findall("successorsStraight")
                ],  # The successor lanelet(s) of the incoming lanelet
                "successorsLeft": int(
                    incoming.find("successorsLeft").get("ref")
                ),  # The successor left lanelet of the incoming lanelet
            }
            self._intersection_info.append(incoming_info)

    def _parse_bound(self, element):
        """Parses a bound (left boundary or right boundary) element to extract points and line marking."""
        points = [self._parse_point(point) for point in element.findall("point")]
        points = torch.vstack(points)
        line_marking = (
            element.find("lineMarking").text
            if element.find("lineMarking") is not None
            else None
        )
        return points, line_marking

    def _parse_point(self, element):
        """Parses a point element to extract x and y coordinates."""
        x = float(element.find("x").text) if element.find("x") is not None else None
        y = float(element.find("y").text) if element.find("y") is not None else None
        if self._scale_map != 1.0:
            x = (
                self._map_center[0]
                + (x - self._map_center[0]) * self._scale_map
                + self._map_translation[0]
            )
            y = (
                self._map_center[1]
                + (y - self._map_center[1]) * self._scale_map
                + self._map_translation[1]
            )
        return torch.tensor([x, y], device=self._device)

    def visualize_map(self, is_visu_intersection_only: bool = False):
        if self._scenario_type == "cpm_intersection_extended":
            intersection_reference_paths = self.reference_paths_intersection_extended
        else:
            intersection_reference_paths = self.reference_paths_intersection

        if is_visu_intersection_only and not intersection_reference_paths:
            raise RuntimeError(
                "is_visu_intersection_only=True, but the selected intersection "
                f"reference-path collection is empty for scenario '{self._scenario_type}'."
            )

        figsize_x = SCENARIOS[self._scenario_type]["figsize_x"]
        aspect_ratio = self.bounds["max_y"] / self.bounds["max_x"]

        # Set up the plot
        fig, ax = plt.subplots(
            figsize=(figsize_x, figsize_x * aspect_ratio), constrained_layout=True
        )  # Size in inches, adjusted for 4.0m x 4.5m dimensions

        ax.margins(x=0.0, y=0.0)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        ax.set_aspect("equal")  # Ensures equal scaling

        ax.tick_params(axis="both", direction="in")

        intersection_lanelet_ids = None
        if is_visu_intersection_only:
            intersection_lanelet_ids = set()

            # Include all lanelets used by the selected intersection paths.
            for p in intersection_reference_paths:
                intersection_lanelet_ids.update(p["lanelet_IDs"])

            # Also include outer-boundary lanelets used for shading the intersection
            intersection_lanelet_ids.update(self._intersection_outer_boundary_ids)

        for lanelet in self.lanelets_all:
            if (
                intersection_lanelet_ids is not None
                and lanelet["id"] not in intersection_lanelet_ids
            ):
                continue

            # Extract coordinates for left, right, and center lines
            left_bound = lanelet["left_boundary"]
            right_bound = lanelet["right_boundary"]
            center_line = lanelet["center_line"]

            # Extract line markings
            left_line_marking = lanelet["left_line_marking"]
            right_line_marking = lanelet["right_line_marking"]

            is_use_random_color = False
            # Choose color
            color = (
                np.random.rand(
                    3,
                )
                if is_use_random_color
                else "grey"
            )

            # Plot left boundary
            ax.plot(
                left_bound[:, 0],
                left_bound[:, 1],
                linestyle="--" if left_line_marking == "dashed" else "-",
                color=color,
                linewidth=self._linewidth,
            )
            # Plot right boundary
            ax.plot(
                right_bound[:, 0],
                right_bound[:, 1],
                linestyle="--" if right_line_marking == "dashed" else "-",
                color=color,
                linewidth=self._linewidth,
            )
            # Plot center line
            # ax.plot(center_line[:, 0], center_line[:, 1], linestyle="--" if center_line_marking == "dashed" else "-", color=color, linewidth=self._linewidth)
            # Adding lanelet ID as text
            if self._is_visu_lane_ids:
                ax.text(
                    center_line[int(len(center_line) / 2), 0],
                    center_line[int(len(center_line) / 2), 1],
                    str(lanelet["id"]),
                    color=color,
                    fontsize=self._fontsize,
                )

        # Add arrows to indicate entry and exit directions
        if self._is_visu_arrows:
            if is_visu_intersection_only or self._scenario_type in (
                "cpm_mixed",
                "cpm_intersection_extended",
            ):
                ref_paths_for_arrows = intersection_reference_paths
            else:
                ref_paths_for_arrows = self.reference_paths

            seen_entries = []
            for path in ref_paths_for_arrows:
                if path.get("is_loop", False):
                    continue
                p_start = path["center_line"][0]
                direction_start = path["center_line_vec_normalized"][0]
                if any(
                    torch.norm(p_start - p) < 500 * self._point_tol
                    for p in seen_entries
                ):
                    continue
                seen_entries.append(p_start)
                ax.quiver(
                    p_start[0],
                    p_start[1],
                    direction_start[0],
                    direction_start[1],
                    angles="xy",
                    scale_units="xy",
                    scale=6.0,
                    color="grey",
                    zorder=2,
                    pivot="tail",
                )

            seen_exits = []
            for path in ref_paths_for_arrows:
                if path.get("is_loop", False):
                    continue
                p_end = path["center_line"][-1]
                direction_end = path["center_line_vec_normalized"][-1]
                if any(
                    torch.norm(p_end - p) < 0.05 * self._scale_map for p in seen_exits
                ):
                    continue
                seen_exits.append(p_end)
                ax.quiver(
                    p_end[0],
                    p_end[1],
                    direction_end[0],
                    direction_end[1],
                    angles="xy",
                    scale_units="xy",
                    scale=6.0,
                    color="grey",
                    zorder=2,
                    pivot="tip",
                )

        if self._is_visualize_intersection_area:
            self._visualize_intersection()

        if self._is_visualize_random_agents:
            if self._n_agents_visu is None:
                self._n_agents_visu = SCENARIOS[self._scenario_type]["n_agents"]

            if is_visu_intersection_only or self._scenario_type in (
                "cpm_mixed",
                "cpm_intersection_extended",
            ):
                ref_paths_for_agents = intersection_reference_paths
            else:
                ref_paths_for_agents = self.reference_paths

            if not ref_paths_for_agents:
                raise RuntimeError(
                    "Cannot visualize random agents because the selected "
                    f"reference-path collection is empty for scenario '{self._scenario_type}'."
                )

            self._visualize_random_agents(ax, ref_paths_for_agents)

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

        # ax.set_xlim((self.bounds["min_x"], self.bounds["max_x"]))
        # ax.set_ylim((self.bounds["min_y"], self.bounds["max_y"]))
        ax.grid(False)

        # Save fig
        if self._is_save_fig:
            path = os.path.join("map_" + self._scenario_type + ".pdf")
            plt.savefig(path, format="pdf", bbox_inches="tight")
            print(f"A fig is saved at {path}")

        if self._is_plt_show:
            plt.show()

        return fig, ax

    def _visualize_intersection(self):
        intersection_boundary = []

        for j in self._intersection_outer_boundary_ids:
            lanelet = self.lanelets_all[j - 1]
            right_bound = lanelet["right_boundary"]
            intersection_boundary.append(right_bound)

            # # Plot right boundary
            # plt.plot(right_bound[:, 0], right_bound[:, 1], linestyle="--" if right_line_marking == "dashed" else "-", color=color, linewidth=line_width)

        intersection_boundary = torch.vstack([b for b in intersection_boundary])

        plt.fill(
            intersection_boundary[:, 0],
            intersection_boundary[:, 1],
            color="grey",
            alpha=0.2,
        )

    def visualize_reference_path_centerlines(
        self, output_path: str = None, is_show: bool = False
    ):
        """
        Visualize the centerline of each reference path in self._reference_paths_ids
        and export the figure as a PDF.
        """
        if output_path is None:
            output_path = f"reference_path_centerlines_{self._scenario_type}.pdf"

        figsize_x = SCENARIOS[self._scenario_type]["figsize_x"] * 3
        aspect_ratio = self.bounds["max_y"] / self.bounds["max_x"]

        fig, ax = plt.subplots(
            figsize=(figsize_x, figsize_x * aspect_ratio), constrained_layout=True
        )
        ax.set_aspect("equal")

        colors = plt.cm.tab10(np.linspace(0, 1, len(self._reference_paths_ids)))

        for path_idx, reference_path_ids in enumerate(self._reference_paths_ids):
            centerline = None

            for lanelet_id in reference_path_ids:
                lanelet_centerline = self.lanelets_all[lanelet_id - 1]["center_line"]

                if centerline is None:
                    centerline = lanelet_centerline
                elif (
                    torch.norm(centerline[-1, :] - lanelet_centerline[0, :])
                    < self._point_tol
                ):
                    centerline = torch.cat(
                        (centerline, lanelet_centerline[1:, :]), dim=0
                    )
                else:
                    centerline = torch.cat((centerline, lanelet_centerline), dim=0)

            centerline = centerline.detach().cpu().numpy()
            ax.plot(
                centerline[:, 0],
                centerline[:, 1],
                color=colors[path_idx],
                linewidth=self._linewidth * 4.0,
                linestyle="-.",
                label=f"Ref. path {path_idx + 1}",
            )

        ax.set_xlabel(r"$x$ [m]", fontsize=self._fontsize)
        ax.set_ylabel(r"$y$ [m]", fontsize=self._fontsize)
        ax.set_xlim((self.bounds["min_x"], self.bounds["max_x"]))
        ax.set_ylim((self.bounds["min_y"], self.bounds["max_y"]))
        ax.tick_params(axis="both", direction="in")
        ax.grid(False)
        ax.legend(fontsize=self._fontsize)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"Saved reference path centerlines figure to {output_path}")

        if is_show:
            plt.show()

        return fig, ax

    def _get_reference_paths(self, is_show_each_ref_path):
        # Mapping agent_id to loop index and starting lanelet: agent_id: (loop_index, starting_lanelet)
        # path_to_loop = {
        #     1: (1, 4),
        #     2: (2, 1),
        #     3: (3, 64),
        #     4: (4, 42),
        #     5: (5, 22),
        #     6: (6, 39),
        #     7: (7, 15),
        #     8: (1, 8),
        #     9: (2, 10),
        #     10: (3, 75),
        #     11: (4, 45),
        #     12: (5, 59),
        #     13: (6, 61),
        #     14: (7, 5),
        #     15: (1, 58),
        #     16: (2, 17),
        #     17: (3, 79),
        #     18: (4, 92),
        #     19: (5, 68),
        #     20: (6, 55),
        #     21: (7, 11),
        #     22: (1, 54),
        #     23: (2, 38),
        #     24: (3, 88),
        #     25: (4, 100),
        #     26: (5, 19),
        #     27: (6, 65),
        #     28: (7, 93),
        #     29: (1, 82),
        #     30: (2, 49),
        #     31: (3, 95),
        #     32: (4, 33),
        #     33: (5, 14),
        #     34: (6, 35),
        #     35: (7, 83),
        #     36: (1, 86),
        #     37: (6, 29),
        #     38: (7, 89),
        #     39: (1, 32),
        #     40: (1, 28),
        # }

        # For all "inner" loops
        path_to_loop = {
            1: (1, 1),
            2: (2, 23),
            3: (3, 64),
            4: (4, 40),
            5: (5, 5),
            6: (6, 48),
            7: (7, 14),
            8: (1, 3),
            9: (2, 10),
            10: (3, 62),
            11: (4, 45),
            12: (5, 7),
            13: (6, 42),
            14: (7, 16),
            15: (1, 5),
            16: (2, 12),
            17: (3, 75),
            18: (4, 97),
            19: (5, 59),
            20: (6, 40),
            21: (7, 22),
            22: (1, 7),
            23: (2, 17),
            24: (3, 55),
            25: (4, 92),
            26: (5, 57),
            27: (6, 21),
            28: (7, 23),
            29: (1, 59),
            30: (2, 43),
            31: (3, 53),
            32: (4, 94),
            33: (5, 74),
            34: (6, 64),
            35: (7, 10),
            36: (1, 57),
            37: (2, 38),
            38: (3, 79),
            39: (4, 100),
            40: (5, 68),
            41: (6, 62),
            42: (7, 12),
        }
        # Some lanelets share the same boundary (such as adjacent left and adjacent right lanelets)
        lanelets_share_same_boundaries_list = [
            [4, 3, 22],
            [6, 5, 23],
            [8, 7],
            [60, 59],
            [58, 57, 75],
            [56, 55, 74],
            [54, 53],
            [80, 79],
            [82, 81, 100],
            [84, 83, 101],
            [86, 85],
            [34, 33],
            [32, 31, 49],
            [30, 29, 48],
            [28, 27],
            [2, 1],  # outer circle (urban)
            [13, 14],
            [15, 16],
            [9, 10],
            [11, 12],  # inner circle (top right)
            [63, 64],
            [61, 62],
            [67, 68],
            [65, 66],  # inner circle (bottom right)
            [91, 92],
            [93, 94],
            [87, 88],
            [89, 90],  # inner circle (bottom left)
            [37, 38],
            [35, 36],
            [41, 42],
            [39, 40],  # inner circle (top left)
            [25, 18],
            [26, 17],
            [52, 43],
            [72, 73],  # intersection: incoming 1 and incoming 2
            [51, 44],
            [50, 45],
            [102, 97],
            [20, 21],  # intersection: incoming 3 and incoming 4
            [103, 96],
            [104, 95],
            [78, 69],
            [46, 47],  # intersection: incoming 5 and incoming 6
            [77, 70],
            [76, 71],
            [24, 19],
            [98, 99],  # intersection: incoming 7 and incoming 8
        ]

        # Do not consider merge-in and merge-out lanelets
        lanelets_share_same_boundaries_list_without_merges = [
            [4, 3],
            [22, 3],
            [6, 5],
            [23, 5],
            [8, 7],
            [60, 59],
            [58, 57],
            [75, 57],
            [56, 55],
            [74, 55],
            [54, 53],
            [80, 79],
            [82, 81],
            [100, 81],
            [84, 83],
            [101, 83],
            [86, 85],
            [34, 33],
            [32, 31],
            [31, 49],
            [30, 29],
            [29, 48],
            [28, 27],
            [2, 1],  # outer circle (urban)
            [13, 14],
            [15, 16],
            [9, 10],
            [11, 12],  # inner circle (top right)
            [63, 64],
            [61, 62],
            [67, 68],
            [65, 66],  # inner circle (bottom right)
            [91, 92],
            [93, 94],
            [87, 88],
            [89, 90],  # inner circle (bottom left)
            [37, 38],
            [35, 36],
            [41, 42],
            [39, 40],  # inner circle (top left)
            [25, 18],
            [26, 17],
            [52, 43],
            [72, 73],  # intersection: incoming 1 and incoming 2
            [51, 44],
            [50, 45],
            [102, 97],
            [20, 21],  # intersection: incoming 3 and incoming 4
            [103, 96],
            [104, 95],
            [78, 69],
            [46, 47],  # intersection: incoming 5 and incoming 6
            [77, 70],
            [76, 71],
            [24, 19],
            [98, 99],  # intersection: incoming 7 and incoming 8
        ]

        self.path_intersection = [
            [11, 25, 13],
            [11, 26, 52, 37],
            [11, 72, 91],
            [12, 18, 14],
            [12, 17, 43, 38],
            [12, 73, 92],
            [39, 51, 37],
            [39, 50, 102, 91],
            [39, 20, 63],
            [40, 44, 38],
            [40, 45, 97, 92],
            [40, 21, 64],
            [89, 103, 91],
            [89, 104, 78, 63],
            [89, 46, 13],
            [90, 96, 92],
            [90, 95, 69, 64],
            [90, 47, 14],
            [65, 77, 63],
            [65, 76, 24, 13],
            [65, 98, 37],
            [66, 70, 64],
            [66, 71, 19, 14],
            [66, 99, 38],
        ]

        # The extended intersection reference paths include the following three types of paths:
        # 1. Straight paths through the intersection (always go straight through the intersection and then loop back to the intersection)
        # 2. Left-turn paths through the intersection (always turn left at the intersection and then loop back to the intersection)
        # 3. Right-turn paths through the intersection (always turn right at the intersection and then loop back to the intersection)
        # In total, the CPM Lab scenario has 1 straight path, 2 left-turn paths, and 4 right-turn paths.
        self.path_intersection_extended = [
            # Straight paths through the intersection
            [
                12,
                17,
                43,
                38,
                36,
                49,
                48,
                42,
                40,
                45,
                97,
                92,
                94,
                100,
                101,
                88,
                90,
                95,
                69,
                64,
                62,
                75,
                74,
                68,
                66,
                71,
                19,
                14,
                16,
                22,
                23,
                10,
            ],
            # Left-turn paths through the intersection
            [12, 73, 92, 94, 100, 101, 88, 90, 47, 14, 16, 22, 23, 10],
            [66, 99, 38, 36, 49, 48, 42, 40, 21, 64, 62, 75, 74, 68],
            # Right-turn paths through the intersection
            [12, 18, 14, 16, 22, 23, 10],
            [66, 70, 64, 62, 75, 74, 68],
            [90, 96, 92, 94, 100, 101, 88],
            [40, 44, 38, 36, 49, 48, 42],
        ]
        self.path_merge_in = [
            [34, 32],
            [33, 31],
            [35, 31],
            [36, 49],
        ]
        self.path_merge_out = [
            [6, 8],
            [5, 7],
            [5, 9],
            [23, 10],
        ]

        num_paths_all = len(path_to_loop)
        for ref_path_id in range(num_paths_all):
            reference_lanelets_index = self._get_reference_lanelet_index(
                ref_path_id + 1, path_to_loop
            )  # Path ID starts from 1
            reference_path = self._get_reference_path(
                reference_lanelets_index,
                lanelets_share_same_boundaries_list,
                is_show_each_ref_path,
            )
            self.reference_paths.append(reference_path)

        for reference_lanelets_index in self.path_intersection:
            reference_path = self._get_reference_path(
                reference_lanelets_index,
                lanelets_share_same_boundaries_list,
                is_show_each_ref_path,
            )
            self.reference_paths_intersection.append(reference_path)

        for reference_lanelets_index in self.path_intersection_extended:
            reference_path = self._get_reference_path(
                reference_lanelets_index,
                lanelets_share_same_boundaries_list,
                is_show_each_ref_path,
            )
            self.reference_paths_intersection_extended.append(reference_path)

        for reference_lanelets_index in self.path_merge_in:
            reference_path = self._get_reference_path(
                reference_lanelets_index,
                lanelets_share_same_boundaries_list,
                is_show_each_ref_path,
            )
            self.reference_paths_merge_in.append(reference_path)

        for reference_lanelets_index in self.path_merge_out:
            reference_path = self._get_reference_path(
                reference_lanelets_index,
                lanelets_share_same_boundaries_list,
                is_show_each_ref_path,
            )
            self.reference_paths_merge_out.append(reference_path)

    def _get_reference_path(
        self,
        reference_lanelets_index,
        lanelets_share_same_boundaries_list,
        is_show_each_ref_path,
    ):
        # Initialize
        left_boundaries = None
        right_boundaries = None
        left_boundaries_shared = None
        right_boundaries_shared = None
        center_lines = None

        for lanelet in reference_lanelets_index:
            lanelets_share_same_boundaries = next(
                (
                    group
                    for group in lanelets_share_same_boundaries_list
                    if lanelet in group
                ),
                None,
            )
            assert lanelets_share_same_boundaries != None

            # Extracting left and right boundaries
            left_bound = self.lanelets_all[lanelet - 1][
                "left_boundary"
            ]  # Lanelet IDs start from 1, while the index of a list in Python starts from 0
            right_bound = self.lanelets_all[lanelet - 1]["right_boundary"]
            left_bound_shared = self.lanelets_all[
                lanelets_share_same_boundaries[0] - 1
            ]["left_boundary"]
            right_bound_shared = self.lanelets_all[
                lanelets_share_same_boundaries[-1] - 1
            ]["right_boundary"]

            if left_boundaries is None:
                left_boundaries = left_bound
                right_boundaries = right_bound
                left_boundaries_shared = left_bound_shared
                right_boundaries_shared = right_bound_shared
            else:
                # Concatenate boundary data while avoiding duplicated adta at the connection point
                if (
                    torch.norm(left_boundaries[-1, :] - left_bound[0, :])
                    < self._point_tol
                ):
                    left_boundaries = torch.cat(
                        (left_boundaries, left_bound[1:, :]), dim=0
                    )
                else:
                    left_boundaries = torch.cat((left_boundaries, left_bound), dim=0)
                    print(
                        f"[WARNING] The left boundaries of lanelet {lanelet} are not connected to the previous lanelet."
                    )
                if (
                    torch.norm(left_boundaries_shared[-1, :] - left_bound_shared[0, :])
                    < self._point_tol
                ):
                    left_boundaries_shared = torch.cat(
                        (left_boundaries_shared, left_bound_shared[1:, :]), dim=0
                    )
                else:
                    # Enable smooth transition by linear interpolation
                    left_boundaries_shared = self.smooth_concatenate(
                        left_boundaries_shared, left_bound_shared, overlap=4
                    )

                if (
                    torch.norm(right_boundaries[-1, :] - right_bound[0, :])
                    < self._point_tol
                ):
                    right_boundaries = torch.cat(
                        (right_boundaries, right_bound[1:, :]), dim=0
                    )
                else:
                    right_boundaries = torch.cat((right_boundaries, right_bound), dim=0)
                    print(
                        f"[WARNING] The right boundaries of lanelet {lanelet} are not connected to the previous lanelet."
                    )
                if (
                    torch.norm(
                        right_boundaries_shared[-1, :] - right_bound_shared[0, :]
                    )
                    < self._point_tol
                ):
                    right_boundaries_shared = torch.cat(
                        (right_boundaries_shared, right_bound_shared[1:, :]), dim=0
                    )
                else:
                    # Enable smooth transition by linear interpolation
                    right_boundaries_shared = self.smooth_concatenate(
                        right_boundaries_shared, right_bound_shared, overlap=4
                    )

        center_lines = (left_boundaries + right_boundaries) / 2

        center_lines_vec = torch.diff(
            center_lines, dim=0
        )  # Vectors connecting each pair of neighboring points on the center lines
        center_lines_vec_length = torch.norm(
            center_lines_vec, dim=1
        )  # The lengths of the vectors
        center_lines_vec_mean_length = torch.mean(
            center_lines_vec_length
        )  # The mean length of the vectors
        center_lines_vec_normalized = (
            center_lines_vec / center_lines_vec_length.unsqueeze(1)
        )

        center_line_yaw = torch.atan2(center_lines_vec[:, 1], center_lines_vec[:, 0])

        # Check if the center line is a loop
        is_loop = (center_lines[0, :] - center_lines[-1, :]).norm() <= self._point_tol

        if lanelet == 2:
            print(f"[INFO] The center line is a loop: {is_loop}")
        if is_loop:
            # Check if the first and last points are the same
            if (
                torch.norm(left_boundaries_shared[0, :] - left_boundaries_shared[-1, :])
                > 0.1 * self._scale_map
            ):
                left_boundaries_shared = self.smooth_loop_boundary(
                    left_boundaries_shared
                )
            if (
                torch.norm(
                    right_boundaries_shared[0, :] - right_boundaries_shared[-1, :]
                )
                > 0.1 * self._scale_map
            ):
                right_boundaries_shared = self.smooth_loop_boundary(
                    right_boundaries_shared
                )

            # plt.plot(right_boundaries_shared[:, 0], right_boundaries_shared[:, 1])

        assert left_boundaries.shape == left_boundaries.shape
        assert left_boundaries.shape[1] == 2  # Must be a two-column array

        if is_show_each_ref_path:
            # Mainly used for debugging
            plt.plot(
                left_boundaries[:, 0],
                left_boundaries[:, 1],
                color="black",
                linewidth=0.5,
                linestyle="-",
            )
            plt.plot(
                right_boundaries[:, 0],
                right_boundaries[:, 1],
                color="black",
                linewidth=0.5,
                linestyle="-",
            )

            # Filling between boundaries
            x = torch.cat(
                (
                    left_boundaries_shared[:, 0],
                    torch.flip(right_boundaries_shared[:, 0], dims=[0]),
                )
            )
            y = torch.cat(
                (
                    left_boundaries_shared[:, 1],
                    torch.flip(right_boundaries_shared[:, 1], dims=[0]),
                )
            )

            plt.fill(
                x, y, color="lightgrey", alpha=0.5
            )  # "alpha" controls the transparency

            plt.plot(
                center_lines[:, 0],
                center_lines[:, 1],
                color="grey",
                linewidth=0.5,
                linestyle="--",
            )

            # Set equal aspect ratio for the plot
            plt.gca().set_aspect("equal", adjustable="box")

            plt.show()

        reference_path = {
            "lanelet_IDs": reference_lanelets_index,
            "left_boundary": left_boundaries,
            "right_boundary": right_boundaries,
            "left_boundary_shared": left_boundaries_shared,
            "right_boundary_shared": right_boundaries_shared,
            "center_line": center_lines,  # Center lines are calculated based on left and right boundaries (instead of shared left and right boundaries)
            "center_line_yaw": center_line_yaw,  # Yaw angle of each point on the center lines
            "center_line_vec_normalized": center_lines_vec_normalized,  # Normalized vectors connecting each pair of neighboring points on the center lines
            "center_line_vec_mean_length": center_lines_vec_mean_length,
            "is_loop": is_loop,
        }
        return reference_path

    def _resample_reference_paths_equal_distance(self):
        """Overwrite reference path geometry with equal-distance point samples."""
        for reference_path in self._iter_unique_reference_paths():
            self._resample_reference_path_equal_distance(reference_path)

    def _resample_reference_path_equal_distance(self, reference_path):
        geometry_keys = (
            "center_line",
            "left_boundary",
            "right_boundary",
            "left_boundary_shared",
            "right_boundary_shared",
        )

        for key in geometry_keys:
            polyline = reference_path[key]
            reference_path[key] = self._resample_polyline_equal_distance(
                polyline=polyline,
                num_points=polyline.shape[0],
            )

        (
            reference_path["center_line_yaw"],
            reference_path["center_line_vec_normalized"],
            reference_path["center_line_vec_mean_length"],
        ) = self._compute_center_line_info(reference_path["center_line"])
        reference_path["is_loop"] = (
            reference_path["center_line"][0, :] - reference_path["center_line"][-1, :]
        ).norm() <= self._point_tol

    def _align_reference_path_triplets(self):
        """Overwrite shared boundaries so their indices align with centerline points."""
        for reference_path in self._iter_unique_reference_paths():
            self._align_reference_path_triplet(reference_path)

    def _center_reference_path_centerlines(self):
        """Center reference paths between their aligned shared boundaries."""
        for reference_path in self._iter_unique_reference_paths():
            reference_path["center_line"] = (
                reference_path["left_boundary_shared"]
                + reference_path["right_boundary_shared"]
            ) / 2
            (
                reference_path["center_line_yaw"],
                reference_path["center_line_vec_normalized"],
                reference_path["center_line_vec_mean_length"],
            ) = self._compute_center_line_info(reference_path["center_line"])
            reference_path["is_loop"] = (
                reference_path["center_line"][0, :]
                - reference_path["center_line"][-1, :]
            ).norm() <= self._point_tol

    def _align_reference_path_triplet(self, reference_path):
        """Project centerline stations onto shared boundaries for index alignment.

        Args:
            reference_path: Reference path dictionary containing ``center_line``,
                ``left_boundary_shared``, and ``right_boundary_shared`` tensors with
                shape ``[N, 2]`` or ``[M, 2]``.

        The method keeps the existing ``center_line`` samples and overwrites the
        existing shared boundary fields. Afterward, all three tensors have the same
        first dimension, and boundary point ``i`` is the closest projected point on
        the corresponding shared boundary for centerline point ``i`` among boundary
        segments with similar path progress.
        """
        center_line = reference_path["center_line"]
        center_progress = self._get_polyline_progress_ratios(center_line)
        is_loop = reference_path["is_loop"]
        reference_path["left_boundary_shared"] = self._project_points_to_polyline(
            points=center_line,
            polyline=reference_path["left_boundary_shared"],
            target_progress_ratios=center_progress,
            is_loop=is_loop,
        )
        reference_path["right_boundary_shared"] = self._project_points_to_polyline(
            points=center_line,
            polyline=reference_path["right_boundary_shared"],
            target_progress_ratios=center_progress,
            is_loop=is_loop,
        )
        (
            reference_path["center_line_yaw"],
            reference_path["center_line_vec_normalized"],
            reference_path["center_line_vec_mean_length"],
        ) = self._compute_center_line_info(reference_path["center_line"])
        reference_path["is_loop"] = (
            reference_path["center_line"][0, :] - reference_path["center_line"][-1, :]
        ).norm() <= self._point_tol

    @staticmethod
    def _project_points_to_polyline(
        points: torch.Tensor,
        polyline: torch.Tensor,
        target_progress_ratios: torch.Tensor | None = None,
        is_loop: bool | torch.Tensor = False,
        progress_window_ratio: float = 0.15,
    ) -> torch.Tensor:
        """Project points to nearby-progress closest points on a polyline.

        Args:
            points: Query points with shape ``[N, 2]``.
            polyline: Ordered polyline vertices with shape ``[M, 2]``.
            target_progress_ratios: Optional target progress values with shape
                ``[N]`` and values in ``[0, 1]``. When provided, each point is
                projected only to boundary segments with similar progress when such
                segments exist.
            is_loop: Whether the polyline should use circular progress distance near
                the first/last segment seam.
            progress_window_ratio: Progress half-window used to reject segments from
                unrelated parts of a self-intersecting or shared boundary.

        Returns:
            Tensor with shape ``[N, 2]`` containing the closest point on any valid
            polyline segment for each query point. Degenerate zero-length segments
            are ignored; if no valid segment exists, the first polyline point is
            repeated for every query point.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape [N, 2].")
        if polyline.ndim != 2 or polyline.shape[1] != 2:
            raise ValueError("polyline must have shape [M, 2].")
        if polyline.shape[0] == 0:
            raise ValueError("polyline must contain at least one point.")
        if polyline.shape[0] == 1:
            return polyline[0].expand(points.shape[0], -1).clone()

        segment_starts = polyline[:-1]
        segment_vecs = polyline[1:] - segment_starts
        segment_lengths_sq = torch.sum(segment_vecs * segment_vecs, dim=-1)
        segment_lengths = torch.sqrt(segment_lengths_sq)
        valid_segments = segment_lengths_sq > torch.finfo(polyline.dtype).eps

        if not torch.any(valid_segments):
            return polyline[0].expand(points.shape[0], -1).clone()

        cumulative_lengths = torch.cat(
            (
                torch.zeros(1, device=polyline.device, dtype=polyline.dtype),
                torch.cumsum(segment_lengths, dim=0),
            )
        )
        total_length = cumulative_lengths[-1]
        if total_length > torch.finfo(polyline.dtype).eps:
            segment_mid_progress = (
                (cumulative_lengths[:-1] + cumulative_lengths[1:]) * 0.5
            ) / total_length
        else:
            segment_mid_progress = torch.zeros_like(segment_lengths)

        segment_starts = segment_starts[valid_segments]
        segment_vecs = segment_vecs[valid_segments]
        segment_lengths_sq = segment_lengths_sq[valid_segments]
        segment_mid_progress = segment_mid_progress[valid_segments]

        point_offsets = points.unsqueeze(1) - segment_starts.unsqueeze(0)
        projection_ratios = torch.sum(
            point_offsets * segment_vecs.unsqueeze(0), dim=-1
        ) / segment_lengths_sq.unsqueeze(0)
        projection_ratios = torch.clamp(projection_ratios, min=0.0, max=1.0)
        projected_points = segment_starts.unsqueeze(0) + projection_ratios.unsqueeze(
            -1
        ) * segment_vecs.unsqueeze(0)
        distances_sq = torch.sum(
            (points.unsqueeze(1) - projected_points) ** 2,
            dim=-1,
        )
        if target_progress_ratios is not None:
            target_progress_ratios = target_progress_ratios.to(
                device=points.device, dtype=points.dtype
            )
            if target_progress_ratios.shape != (points.shape[0],):
                raise ValueError("target_progress_ratios must have shape [N].")
            progress_delta = torch.abs(
                target_progress_ratios.unsqueeze(1) - segment_mid_progress.unsqueeze(0)
            )
            if bool(torch.as_tensor(is_loop).item()):
                progress_delta = torch.minimum(progress_delta, 1.0 - progress_delta)
            window_ratio = max(
                progress_window_ratio,
                min(1.0, 4.0 / max(segment_mid_progress.shape[0], 1)),
            )
            progress_mask = progress_delta <= window_ratio
            has_progress_candidates = torch.any(progress_mask, dim=-1, keepdim=True)
            distances_sq = torch.where(
                has_progress_candidates & progress_mask,
                distances_sq,
                torch.full_like(distances_sq, torch.inf),
            )
            distances_sq = torch.where(
                has_progress_candidates,
                distances_sq,
                torch.sum((points.unsqueeze(1) - projected_points) ** 2, dim=-1),
            )
        closest_segment_indices = torch.argmin(distances_sq, dim=-1)
        return projected_points[
            torch.arange(points.shape[0], device=points.device),
            closest_segment_indices,
        ]

    @staticmethod
    def _get_polyline_progress_ratios(polyline: torch.Tensor) -> torch.Tensor:
        """Compute cumulative arc-length progress ratios for polyline vertices.

        Args:
            polyline: Ordered polyline vertices with shape ``[N, 2]``.

        Returns:
            Tensor with shape ``[N]`` containing normalized cumulative arc length in
            ``[0, 1]``. If the polyline is degenerate, evenly spaced ratios are
            returned so downstream alignment still has a stable ordering signal.
        """
        if polyline.ndim != 2 or polyline.shape[1] != 2:
            raise ValueError("polyline must have shape [N, 2].")
        if polyline.shape[0] == 0:
            raise ValueError("polyline must contain at least one point.")
        if polyline.shape[0] == 1:
            return torch.zeros(1, device=polyline.device, dtype=polyline.dtype)

        segment_lengths = torch.norm(torch.diff(polyline, dim=0), dim=-1)
        cumulative_lengths = torch.cat(
            (
                torch.zeros(1, device=polyline.device, dtype=polyline.dtype),
                torch.cumsum(segment_lengths, dim=0),
            )
        )
        total_length = cumulative_lengths[-1]
        if total_length <= torch.finfo(polyline.dtype).eps:
            return torch.linspace(
                0,
                1,
                steps=polyline.shape[0],
                device=polyline.device,
                dtype=polyline.dtype,
            )
        return cumulative_lengths / total_length

    @staticmethod
    def _compute_center_line_info(center_line):
        center_line_vec = torch.diff(center_line, dim=0)
        center_line_vec_length = torch.norm(center_line_vec, dim=1)
        center_line_vec_mean_length = torch.mean(center_line_vec_length)
        center_line_vec_normalized = center_line_vec / center_line_vec_length.unsqueeze(
            1
        )
        center_line_yaw = torch.atan2(center_line_vec[:, 1], center_line_vec[:, 0])

        return center_line_yaw, center_line_vec_normalized, center_line_vec_mean_length

    @staticmethod
    def _resample_polyline_equal_distance(
        polyline: torch.Tensor, num_points: int
    ) -> torch.Tensor:
        if polyline.ndim != 2 or polyline.shape[1] != 2:
            raise ValueError("Polyline must have shape [N, 2].")
        if num_points != polyline.shape[0]:
            raise ValueError("num_points must preserve the original polyline length.")
        return ParseXML._resample_polyline_to_num_points(polyline, num_points)

    @staticmethod
    def _resample_polyline_to_num_points(
        polyline: torch.Tensor, num_points: int
    ) -> torch.Tensor:
        """Resample a polyline to a requested number of equal-distance points."""
        if polyline.ndim != 2 or polyline.shape[1] != 2:
            raise ValueError("Polyline must have shape [N, 2].")
        if num_points < 2:
            return polyline.clone()

        segment_vecs = torch.diff(polyline, dim=0)
        segment_lengths = torch.norm(segment_vecs, dim=1)
        valid_segments = segment_lengths > torch.finfo(polyline.dtype).eps

        if not torch.any(valid_segments):
            return polyline.clone()

        points = torch.cat((polyline[:1], polyline[1:][valid_segments]), dim=0)
        segment_vecs = torch.diff(points, dim=0)
        segment_lengths = torch.norm(segment_vecs, dim=1)

        if points.shape[0] < 2:
            return polyline.clone()

        cumulative_lengths = torch.cat(
            (
                torch.zeros(1, device=polyline.device, dtype=polyline.dtype),
                torch.cumsum(segment_lengths, dim=0),
            )
        )
        total_length = cumulative_lengths[-1]
        if total_length <= torch.finfo(polyline.dtype).eps:
            return polyline.clone()

        target_lengths = torch.linspace(
            0,
            total_length,
            steps=num_points,
            device=polyline.device,
            dtype=polyline.dtype,
        )

        segment_indices = torch.searchsorted(cumulative_lengths, target_lengths)
        segment_indices = torch.clamp(
            segment_indices, 1, cumulative_lengths.shape[0] - 1
        )

        segment_starts = cumulative_lengths[segment_indices - 1]
        segment_ends = cumulative_lengths[segment_indices]
        segment_ratios = (target_lengths - segment_starts) / (
            segment_ends - segment_starts
        )

        start_points = points[segment_indices - 1]
        end_points = points[segment_indices]
        resampled = start_points + segment_ratios.unsqueeze(1) * (
            end_points - start_points
        )
        resampled[0] = polyline[0]
        resampled[-1] = polyline[-1]

        return resampled

    def _get_reference_lanelet_index(self, ref_path_id, path_to_loop):
        """
        Get loop of lanelets used for reference_path_struct.

        Args:
        ref_path_id (int): Path ID.

        Returns:
        list: List of lanelets indices.
        """
        # Get loop index and starting lanelet for the given agent
        assert (ref_path_id >= 1) & (
            ref_path_id <= len(path_to_loop)
        ), f"Reference ID should be in the range [1, {len(path_to_loop)}]"
        loop_index, starting_lanelet = path_to_loop.get(ref_path_id, (None, None))

        if loop_index is not None:
            # Take loop from all defined loops
            reference_lanelets_loop = self._reference_paths_ids[
                loop_index - 1
            ]  # Adjust for 0-based index
            # Find index of defined starting lanelet
            index_starting_lanelet = reference_lanelets_loop.index(starting_lanelet)
            # Shift loop according to starting lanelet
            lanelets_index = (
                reference_lanelets_loop[index_starting_lanelet:]
                + reference_lanelets_loop[:index_starting_lanelet]
            )
            return lanelets_index
        else:
            return []  # Return empty list if ref_path_id is not found

    @staticmethod
    def smooth_concatenate(
        left_boundaries_shared: torch.Tensor, left_bound: torch.Tensor, overlap: int = 4
    ) -> torch.Tensor:
        """
        Concatenate two tensors smoothly by interpolating between the last few points
        of the first tensor and the first few points of the second tensor.

        Parameters:
        - left_boundaries_shared (torch.Tensor): Tensor of shape (N1, 2)
        - left_bound (torch.Tensor): Tensor of shape (N2, 2)
        - overlap (int): Number of points to interpolate for smooth transition

        Returns:
        - torch.Tensor: Concatenated tensor with smooth transition
        """

        # Get the end point of the first tensor and start point of the second tensor for interpolation
        start_point = left_boundaries_shared[-overlap]
        end_point = left_bound[overlap - 1]

        # Create interpolation weights
        t = torch.linspace(
            0, 1, steps=2 * overlap, device=left_boundaries_shared.device
        ).unsqueeze(
            1
        )  # Shape: (2*overlap, 1)

        # Linear interpolation
        interpolated = (1 - t) * start_point + t * end_point  # Shape: (2*overlap, 2)

        # Concatenate:
        # - everything in `left_boundaries_shared` except the last `overlap` points
        # - interpolated transition
        # - everything in `left_bound` except the first `overlap` points
        result = torch.cat(
            [left_boundaries_shared[:-overlap], interpolated, left_bound[overlap:]],
            dim=0,
        )

        return result

    @staticmethod
    def smooth_loop_boundary(boundary: torch.Tensor, overlap: int = 4) -> torch.Tensor:
        """
        Smooth a looped boundary by replacing the first and last `overlap` points
        with interpolated points between the end and start of the boundary.

        Parameters:
        - boundary (torch.Tensor): Tensor of shape (N, 2), assumed to be a closed loop.
        - overlap (int): Number of points to replace at both ends.

        Returns:
        - torch.Tensor: Modified boundary with smooth loop transition.
        """
        # Define interpolation start and end
        start_point = boundary[-overlap]  # A bit before the end
        end_point = boundary[overlap - 1]  # A bit after the beginning

        # Generate interpolation weights
        t = torch.linspace(0, 1, steps=2 * overlap, device=boundary.device).unsqueeze(
            1
        )  # Shape: (2*overlap, 1)

        # Linear interpolation from end to start
        interpolated = (1 - t) * start_point + t * end_point  # Shape: (2*overlap, 2)

        # Replace:
        # - first `overlap` points with first half of interpolated
        # - last `overlap` points with second half of interpolated
        modified_boundary = boundary.clone()
        modified_boundary[:overlap] = interpolated[overlap:]
        modified_boundary[-overlap:] = interpolated[:overlap]

        # Append the first point to the end to close the loop
        modified_boundary = torch.cat([modified_boundary, modified_boundary[:1]], dim=0)

        return modified_boundary


if __name__ == "__main__":
    import argparse

    cli_parser = argparse.ArgumentParser(description="Parse and visualize XML maps.")
    cli_parser.add_argument(
        "--is_visu_arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to visualize entry and exit direction arrows (default: True).",
    )
    cli_args, _ = cli_parser.parse_known_args()

    parser = ParseXML(
        scenario_type="cpm_mixed",  # One of ["cpm_mixed", "cpm_intersection_extended", "cpm_entire"]
        device="cpu" if not torch.cuda.is_available() else "cuda:0",
        is_visualize_map=True,
        is_save_fig=True,
        is_plt_show=False,
        is_visu_lane_ids=False,
        is_visualize_random_agents=True,
        is_visu_arrows=cli_args.is_visu_arrows,
        n_agents_visu=6,
        is_visu_intersection_only=True,
        is_visualize_intersection_area=False,
        scale_map=1.0,
    )

    # parser.visualize_reference_path_centerlines()

    # print(parser.lanelets_all)
    # print(parser.reference_paths)
