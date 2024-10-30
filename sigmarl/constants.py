# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

SCENARIOS = {
    "CPM_entire": {
        "map_path": "cpm.xml",
        "n_agents": 15,
        "name": "CPM Map",
        "x_dim_min": 0,  # Min x-coordinate
        "x_dim_max": 4.5,  # Max x-coordinate
        "y_dim_min": 0,
        "y_dim_max": 4.0,
        "world_x_dim": 4.5,  # Environment x-dimension. (0, 0) is assumed to be the origin
        "world_y_dim": 4.0,  # Environment y-dimension. (0, 0) is assumed to be the origin
        "figsize_x": 3,  # For evaluation figs
        "viewer_zoom": 1.44,  # For VMAS render
        "lane_width": 0.15,  # [m] Lane width
        "scale": 1.0,  # Scale the map
    },
    "CPM_mixed": {
        "map_path": "cpm.xml",
        "n_agents": 4,
        "name": "CPM Map",
        "x_dim_min": 0,  # Min x-coordinate
        "x_dim_max": 4.5,  # Max x-coordinate
        "y_dim_min": 0,
        "y_dim_max": 4.0,
        "world_x_dim": 4.5,  # Environment x-dimension. (0, 0) is assumed to be the origin
        "world_y_dim": 4.0,  # Environment y-dimension. (0, 0) is assumed to be the origin
        "figsize_x": 3,  # For evaluation figs
        "viewer_zoom": 1.44,  # For VMAS render
        "lane_width": 0.15,  # [m] Lane width
        "scale": 1.0,  # Scale the map
    },
    "intersection_1": {
        "map_path": "intersection_1.osm",
        "n_agents": 6,
        "name": "Intersection 1",
        "x_dim_min": 0.180,
        "x_dim_max": 2.198,
        "y_dim_min": 0.179,
        "y_dim_max": 3.381,
        "world_x_dim": 2.5,
        "world_y_dim": 3.7,
        "reference_paths_ids": [
            ["1", "2"],
            ["1", "3"],
            ["1", "4"],
            ["5"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "3", "4"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "4", "5"],  # Lanelet with ID "2"
            "3": ["1", "2", "3", "5"],  # Lanelet with ID "3"
            "4": ["1", "2", "3", "4", "5"],  # Lanelet with ID "4"
            "5": ["2", "3", "4", "5"],  # Lanelet with ID "5"
        },
        "figsize_x": 2.5,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # A scale converts data from geographic coordinate system (used in JOSM) to Cartesian coordinate system
    },
    "intersection_2": {
        "map_path": "intersection_2.osm",
        "n_agents": 6,
        "name": "Intersection 2",
        "x_dim_min": 0.180,
        "x_dim_max": 2.583,
        "y_dim_min": 0.179,
        "y_dim_max": 3.381,
        "world_x_dim": 2.90,
        "world_y_dim": 3.70,
        "reference_paths_ids": [
            ["1", "2", "5", "10"],
            ["1", "2", "6", "11"],
            ["1", "3"],
            ["1", "4"],
            ["8", "9", "11"],
            ["8", "7", "10"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "3", "4"],
            "2": ["1", "2", "3", "4", "5", "6"],
            "3": ["1", "2", "3", "4", "8"],
            "4": ["1", "2", "3", "4", "11"],
            "5": ["2", "5", "6", "7", "9"],
            "6": ["5", "6", "9", "11"],
            "7": ["5", "7", "8", "9", "10"],
            "8": ["3", "7", "8", "9"],
            "9": ["5", "6", "7", "8", "9", "11"],
            "10": ["5", "7", "10"],
            "11": ["4", "6", "11"],
        },
        "figsize_x": 2.0,
        "viewer_zoom": 1.15,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "intersection_3": {
        "map_path": "intersection_3.osm",
        "n_agents": 8,
        "name": "Intersection 3",
        "x_dim_min": 0.178,
        "x_dim_max": 3.860,
        "y_dim_min": 0.105,
        "y_dim_max": 2.311,
        "world_x_dim": 4.2,
        "world_y_dim": 2.6,
        "reference_paths_ids": [
            ["1"],
            ["2", "3"],
            ["2", "7"],
            ["4"],
            ["5", "6"],
            ["8", "6"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1"],  # Lanelet with ID "1"
            "2": ["2", "3", "7"],  # Lanelet with ID "2"
            "3": ["3", "2"],  # Lanelet with ID "3"
            "4": ["4", "7"],  # Lanelet with ID "4"
            "5": ["5", "6", "7", "8"],  # Lanelet with ID "5"
            "6": ["6", "5", "8"],  # Lanelet with ID "6"
            "7": ["7", "2", "4", "5"],  # Lanelet with ID "7"
            "8": ["8", "5", "6"],  # Lanelet with ID "8"
        },
        "figsize_x": 2.5,
        "viewer_zoom": 1.15,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "on_ramp_1": {
        "map_path": "on_ramp_1.osm",
        "n_agents": 8,
        "name": "Intersection 3",
        "x_dim_min": 0.152,
        "x_dim_max": 3.803,
        "y_dim_min": 0.110,
        "y_dim_max": 1.537,
        "world_x_dim": 4.1,
        "world_y_dim": 1.8,
        "reference_paths_ids": [
            ["1", "3", "5", "7"],
            ["2", "3", "5", "7"],
            ["4", "5", "7"],
            ["6", "7"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "3"],  # Lanelet with ID "1"
            "2": ["1", "2", "3"],  # Lanelet with ID "2"
            "3": ["1", "2", "3", "4", "5"],  # Lanelet with ID "3"
            "4": ["3", "4", "5"],  # Lanelet with ID "4"
            "5": ["3", "4", "5", "6", "7"],  # Lanelet with ID "5"
            "6": ["5", "6", "7"],  # Lanelet with ID "6"
            "7": ["5", "6", "7"],  # Lanelet with ID "7"
        },
        "figsize_x": 3.5,
        "viewer_zoom": 0.95,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "roundabout_1": {
        "map_path": "roundabout_1.osm",
        "n_agents": 8,
        "name": "Intersection 3",
        "x_dim_min": 0.180,
        "x_dim_max": 3.638,
        "y_dim_min": 0.105,
        "y_dim_max": 2.019,
        "world_x_dim": 4.0,
        "world_y_dim": 2.3,
        "reference_paths_ids": [
            ["1", "2", "9"],
            ["1", "3", "7", "9"],
            ["1", "4", "8", "9"],
            ["5", "7", "9"],
            ["6", "8", "9"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "3", "4"],  # Lanelet with ID "1"
            "2": ["1", "2", "7", "8", "9"],  # Lanelet with ID "2"
            "3": ["3", "5", "7"],  # Lanelet with ID "3"
            "4": ["4", "6", "8"],  # Lanelet with ID "4"
            "5": ["3", "5", "7"],  # Lanelet with ID "5"
            "6": ["4", "6", "8"],  # Lanelet with ID "6"
            "7": ["2", "7", "8", "9"],  # Lanelet with ID "7"
            "8": ["2", "7", "8", "9"],  # Lanelet with ID "8"
            "9": ["2", "7", "8", "9"],  # Lanelet with ID "9"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "goal_reaching_1": {
        "n_agents": 1,
        "name": "",
        "x_dim_min": 0.0,
        "x_dim_max": 20.0,
        "y_dim_min": 0.0,
        "y_dim_max": 20.0,
        "world_x_dim": 20.0,
        "world_y_dim": 20.0,
        "reference_paths_ids": [],
        "neighboring_lanelet_ids": {},
        "figsize_x": 3.0,
        "viewer_zoom": 3.5,  # For VMAS render
        "resolution_factor": 50,
        "lane_width": 0.20,  # [m] Lane width
        "scale": 0.0,  # Scale the map
    },
}

AGENTS = {
    "width": 0.08,  # [m]
    "length": 0.16,  # [m]
    "l_f": 0.08,  # [m] Front wheelbase
    "l_r": 0.08,  # [m] Rear wheelbase
    "l_wb": 0.16,  # [m] Wheelbase
    "max_speed": 1.0,  # [m/s]
    "min_speed": -0.5,  # [m/s]
    "max_speed_achievable": 1.0,  # [m/s]
    "max_steering": math.pi / 4,  # [radian]
    "min_steering": -math.pi / 4,  # [radian]
    "max_acc": 4.0,  # [m/s^2]
    "min_acc": -4.0,  # [m/s^2]
    "max_steering_rate": 2 * math.pi,  # [radian/s]
    "min_steering_rate": -2 * math.pi,  # [radian/s]
    "n_actions": 2,
}

THRESHOLD = {
    "initial_distance": 1.2 * math.sqrt(AGENTS["width"] ** 2 + AGENTS["length"] ** 2),
    "reach_goal": AGENTS[
        "width"
    ],  # An agent is considered having reached its goal if its distance to the goal is less than its width
}
