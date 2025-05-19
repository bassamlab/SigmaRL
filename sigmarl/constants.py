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
    "interchange_1": {
        "map_path": "interchange_1.osm",
        "n_agents": 8,
        "name": "Scenario 3",
        "reference_paths_ids": [
            ["1", "2", "3"],
            ["1", "7", "6"],
            ["4", "5", "6"],
            ["4", "8", "3"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "7"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "7", "8"],  # Lanelet with ID "2"
            "3": ["2", "3", "8"],  # Lanelet with ID "3"
            "4": ["4", "5", "8"],  # Lanelet with ID "4"
            "5": ["4", "5", "6", "7", "8"],  # Lanelet with ID "5"
            "6": ["5", "6", "7"],  # Lanelet with ID "6"
            "7": ["1", "2", "5", "6", "7"],  # Lanelet with ID "7"
            "8": ["2", "3", "4", "5", "8"],  # Lanelet with ID "8"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.25,  # For VMAS render
        "lane_width": 0.30,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "interchange_2": {
        "map_path": "interchange_2.osm",
        "n_agents": 10,
        "name": "Scenario 6",
        "reference_paths_ids": [
            ["9", "1", "10"],
            ["9", "1", "2", "3", "12"],
            ["9", "1", "2", "3", "4", "5", "15"],
            ["9", "1", "2", "3", "4", "5", "6", "7", "14"],
            ["11", "3", "12"],
            ["11", "3", "4", "5", "15"],
            ["11", "3", "4", "5", "6", "7", "14"],
            ["11", "3", "4", "5", "6", "7", "8", "1", "10"],
            ["16", "5", "15"],
            ["16", "5", "6", "7", "14"],
            ["16", "5", "6", "7", "8", "1", "10"],
            ["16", "5", "6", "7", "8", "1", "2", "3", "12"],
            ["13", "7", "14"],
            ["13", "7", "8", "1", "10"],
            ["13", "7", "8", "1", "2", "3", "12"],
            ["13", "7", "8", "1", "2", "3", "4", "5", "15"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "8", "9", "10"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "10", "11"],  # Lanelet with ID "2"
            "3": ["2", "3", "4", "11", "12"],  # Lanelet with ID "3"
            "4": ["3", "4", "5", "12", "16"],  # Lanelet with ID "4"
            "5": ["4", "5", "6", "16", "15"],  # Lanelet with ID "5"
            "6": ["5", "6", "7", "13", "15"],  # Lanelet with ID "6"
            "7": ["6", "7", "8", "13", "14"],  # Lanelet with ID "7"^
            "8": ["1", "7", "8", "9", "14"],  # Lanelet with ID "8"
            "9": ["1", "8", "9"],  # Lanelet with ID "9"
            "10": ["1", "2", "10"],  # Lanelet with ID "10"
            "11": ["2", "3", "11"],  # Lanelet with ID "11"
            "12": ["3", "4", "12"],  # Lanelet with ID "12"
            "13": ["6", "7", "13"],  # Lanelet with ID "13"
            "14": ["7", "8", "14"],  # Lanelet with ID "14"
            "15": ["5", "6", "15"],  # Lanelet with ID "15"
            "16": ["4", "5", "16"],  # Lanelet with ID "16"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "interchange_3": {
        "map_path": "interchange_3.osm",
        "n_agents": 10,
        "name": "Scenario 9",
        "reference_paths_ids": [
            ["1", "2", "3"],
            ["1", "4", "5", "6"],
            ["1", "4", "5", "7", "9", "10", "3"],
            ["1", "4", "5", "7", "9", "22", "15", "16"],
            ["1", "4", "5", "7", "9", "22", "15", "17", "19", "20", "13"],
            ["8", "9", "10", "3"],
            ["8", "9", "22", "15", "16"],
            ["8", "9", "22", "15", "17", "19", "20", "13"],
            ["8", "9", "22", "15", "17", "19", "21", "5", "6"],
            ["11", "12", "13"],
            ["11", "14", "15", "16"],
            ["11", "14", "15", "17", "19", "20", "13"],
            ["11", "14", "15", "17", "19", "21", "5", "6"],
            ["11", "14", "15", "17", "19", "21", "5", "7", "9", "10", "3"],
            ["18", "19", "20", "13"],
            ["18", "19", "21", "5", "6"],
            ["18", "19", "21", "5", "7", "9", "10", "3"],
            ["18", "19", "21", "5", "7", "9", "22", "15", "16"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "4"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "4", "10"],  # Lanelet with ID "2"
            "3": ["2", "3", "10"],  # Lanelet with ID "3"
            "4": ["1", "2", "4", "5", "21"],  # Lanelet with ID "4"
            "5": ["4", "5", "6", "7", "21"],  # Lanelet with ID "5"
            "6": ["5", "6", "7"],  # Lanelet with ID "6"
            "7": ["5", "6", "7", "8", "9"],  # Lanelet with ID "7"
            "8": ["7", "8", "9"],  # Lanelet with ID "8"
            "9": ["7", "8", "9", "10", "22"],  # Lanelet with ID "9"
            "10": ["2", "3", "9", "10", "22"],  # Lanelet with ID "10"
            "11": ["11", "12", "14"],  # Lanelet with ID "11"
            "12": ["11", "12", "13", "14"],  # Lanelet with ID "12"
            "13": ["12", "13", "20"],  # Lanelet with ID "13"
            "14": ["11", "12", "14", "15", "22"],  # Lanelet with ID "14"
            "15": ["14", "15", "16", "17", "22"],  # Lanelet with ID "15"
            "16": ["15", "16", "17"],  # Lanelet with ID "16"
            "17": ["15", "16", "17", "18", "19"],  # Lanelet with ID "17"
            "18": ["17", "18", "19"],  # Lanelet with ID "18"
            "19": ["17", "18", "19", "20", "21"],  # Lanelet with ID "19"
            "20": ["12", "13", "19", "20", "21"],  # Lanelet with ID "20"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 0.8e5,  # Scale the map
    },
    "intersection_1": {
        "map_path": "intersection_1.osm",
        "n_agents": 6,
        "name": "Intersection 1",
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
        "lane_width": 0.30,  # [m] Lane width
        "scale": 1e5,  # A scale converts data from geographic coordinate system (used in JOSM) to Cartesian coordinate system
    },
    "intersection_2": {
        "map_path": "intersection_2.osm",
        "n_agents": 6,
        "name": "Intersection 2",
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
    "intersection_4": {
        "map_path": "intersection_4.osm",
        "n_agents": 8,
        "name": "Scenario 7",
        "reference_paths_ids": [
            ["1", "2", "3"],
            ["1", "13", "4"],
            ["1", "17", "9"],
            ["7", "8", "9"],
            ["7", "14", "3"],
            ["7", "18", "12"],
            ["10", "11", "12"],
            ["10", "15", "9"],
            ["10", "19", "4"],
            ["6", "5", "4"],
            ["6", "16", "12"],
            ["6", "20", "3"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "13", "17"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "13", "14", "17", "20"],  # Lanelet with ID "2"
            "3": ["2", "3", "14", "20"],  # Lanelet with ID "3"
            "4": ["4", "5", "13", "19"],  # Lanelet with ID "4"
            "5": ["4", "5", "6", "13", "16", "19", "20"],  # Lanelet with ID "5"
            "6": ["5", "6", "16", "20"],  # Lanelet with ID "6"
            "7": ["7", "8", "14", "18"],  # Lanelet with ID "7"
            "8": ["7", "8", "9", "14", "15", "17", "18"],  # Lanelet with ID "8"
            "9": ["8", "9", "15", "17"],  # Lanelet with ID "9"
            "10": ["10", "11", "15", "19"],  # Lanelet with ID "10"
            "11": ["10", "11", "12", "15", "16", "18", "19"],  # Lanelet with ID "11"
            "12": ["11", "12", "16", "18"],  # Lanelet with ID "12"
            "13": ["1", "2", "4", "5", "13", "19"],  # Lanelet with ID "13"
            "14": ["2", "3", "7", "8", "14", "18", "20"],  # Lanelet with ID "14"
            "15": ["8", "9", "10", "11", "15", "17", "19"],  # Lanelet with ID "15"
            "16": ["5", "6", "11", "12", "16", "18", "20"],  # Lanelet with ID "16"
            "17": ["1", "2", "8", "9", "13", "15", "17"],  # Lanelet with ID "17"
            "18": ["7", "8", "11", "12", "14", "16", "18"],  # Lanelet with ID "18"
            "19": ["4", "5", "10", "11", "13", "15", "19"],  # Lanelet with ID "19"
            "20": ["2", "3", "5", "6", "14", "16", "20"],  # Lanelet with ID "20"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.30,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "intersection_5": {
        "map_path": "intersection_5.osm",
        "n_agents": 10,
        "name": "Scenario 8",
        "reference_paths_ids": [
            ["1", "2", "3", "4"],
            ["1", "5", "10"],
            ["13", "14", "15", "16"],
            ["13", "17", "22"],
            ["19", "6", "4"],
            ["19", "20", "21", "22"],
            ["19", "20", "18", "15", "16"],
            ["7", "11", "16"],
            ["7", "8", "9", "10"],
            ["7", "8", "12", "3", "4"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "5"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "5", "12"],  # Lanelet with ID "2"
            "3": ["2", "3", "4", "6", "12"],  # Lanelet with ID "3"
            "4": ["3", "4", "6"],  # Lanelet with ID "4"
            "5": ["1", "2", "5", "9", "10"],  # Lanelet with ID "5"
            "6": ["3", "4", "6", "19", "20"],  # Lanelet with ID "6"
            "7": ["7", "8", "11"],  # Lanelet with ID "7"
            "8": ["7", "8", "9", "11", "12"],  # Lanelet with ID "8"
            "9": ["5", "8", "9", "10", "12"],  # Lanelet with ID "9"
            "10": ["5", "9", "10"],  # Lanelet with ID "10"
            "11": ["7", "8", "11", "15", "16"],  # Lanelet with ID "11"
            "12": ["2", "3", "8", "9", "12"],  # Lanelet with ID "12"
            "13": ["13", "14", "17"],  # Lanelet with ID "13"
            "14": ["13", "14", "15", "17", "18"],  # Lanelet with ID "14"
            "15": ["11", "14", "15", "16", "18"],  # Lanelet with ID "15"
            "16": ["11", "15", "16"],  # Lanelet with ID "16"
            "17": ["13", "14", "17", "21", "22"],  # Lanelet with ID "17"
            "18": ["14", "15", "18", "20", "21"],  # Lanelet with ID "18"
            "19": ["6", "19", "20"],  # Lanelet with ID "19"
            "20": ["6", "18", "19", "20", "21"],  # Lanelet with ID "20"
            "21": ["17", "18", "20", "21", "22"],  # Lanelet with ID "21"
            "22": ["17", "21", "22"],  # Lanelet with ID "22"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 0.7e5,  # Scale the map
    },
    "intersection_6": {
        "map_path": "intersection_6.osm",
        "n_agents": 10,
        "name": "Scenario 10",
        "reference_paths_ids": [
            ["1", "2"],
            ["4", "6", "7", "2"],
            ["4", "5"],
            ["3"],
            ["8", "7", "2"],
            ["9", "10", "11"],
            ["22", "21"],
            ["18", "17"],
            ["13", "11"],
            ["19", "17"],
            ["20", "21"],
            ["14", "15"],
            ["14", "16", "17"],
            ["12", "10", "11"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "7"],  # Lanelet with ID "1"
            "2": ["1", "2", "7"],  # Lanelet with ID "2"
            "3": ["3"],  # Lanelet with ID "3"
            "4": ["4", "5", "6"],  # Lanelet with ID "4"
            "5": ["4", "5", "6"],  # Lanelet with ID "5"
            "6": ["4", "5", "6", "7", "8"],  # Lanelet with ID "6"
            "7": ["2", "6", "7", "8"],  # Lanelet with ID "7"
            "8": ["6", "7", "8"],  # Lanelet with ID "8"
            "9": ["9", "10", "12"],  # Lanelet with ID "9"
            "10": ["9", "10", "11", "12", "13"],  # Lanelet with ID "10"
            "11": ["10", "11", "13"],  # Lanelet with ID "11"
            "12": ["9", "10", "12"],  # Lanelet with ID "12"
            "13": ["10", "11", "13"],  # Lanelet with ID "13"
            "14": ["14", "15", "16"],  # Lanelet with ID "14"
            "15": ["14", "15", "16"],  # Lanelet with ID "15"
            "16": ["14", "15", "16", "17", "18", "19"],  # Lanelet with ID "16"
            "17": ["16", "17", "18", "19"],  # Lanelet with ID "17"
            "18": ["16", "17", "18", "19"],  # Lanelet with ID "18"
            "19": ["16", "17", "18", "19"],  # Lanelet with ID "19"
            "20": ["20", "21", "22"],  # Lanelet with ID "20"
            "21": ["20", "21", "22"],  # Lanelet with ID "21"
            "22": ["20", "21", "22"],  # Lanelet with ID "22"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "intersection_7": {
        "map_path": "intersection_7.osm",
        "n_agents": 8,
        "name": "Scenario 2",
        "reference_paths_ids": [
            ["1", "2", "3", "4"],
            ["1", "2", "3", "12", "6", "7", "8"],
            ["1", "9", "10"],
            ["5", "6", "7", "8"],
            ["5", "6", "11", "10"],
            ["5", "6", "7", "13", "3", "4"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "9"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "9", "12"],  # Lanelet with ID "2"
            "3": ["2", "3", "4", "12", "13"],  # Lanelet with ID "3"
            "4": ["3", "4", "13"],  # Lanelet with ID "4"
            "5": ["5", "6", "12"],  # Lanelet with ID "5"
            "6": ["5", "6", "7", "11", "12"],  # Lanelet with ID "6"
            "7": ["6", "7", "8", "11", "13"],  # Lanelet with ID "7"
            "8": ["7", "8", "13"],  # Lanelet with ID "8"
            "9": ["1", "2", "9", "10", "11"],  # Lanelet with ID "9"
            "10": ["9", "10", "11"],  # Lanelet with ID "10"
            "11": ["6", "7", "9", "10", "11"],  # Lanelet with ID "11"
            "12": ["2", "3", "5", "6", "12"],  # Lanelet with ID "12"
            "13": ["3", "4", "7", "8", "13"],  # Lanelet with ID "13"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "intersection_8": {
        "map_path": "intersection_8.osm",
        "n_agents": 10,
        "name": "Scenario 5",
        "reference_paths_ids": [
            ["1", "2", "3", "4"],
            ["1", "2", "12"],
            ["1", "5", "6"],
            ["1", "5", "7", "9", "11"],
            ["8", "9", "11"],
            [
                "10",
                "11",
            ],
            ["13", "4"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "5"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "12"],  # Lanelet with ID "2"
            "3": ["2", "3", "4", "13"],  # Lanelet with ID "3"
            "4": ["3", "4", "13"],  # Lanelet with ID "4"
            "5": ["1", "2", "5", "6", "7"],  # Lanelet with ID "5"
            "6": ["5", "6", "7"],  # Lanelet with ID "6"
            "7": ["5", "6", "7", "8", "9"],  # Lanelet with ID "7"
            "8": ["7", "8", "9"],  # Lanelet with ID "8"
            "9": ["7", "8", "9", "10", "11"],  # Lanelet with ID "9"
            "10": ["9", "10", "11"],  # Lanelet with ID "10"
            "11": ["9", "10", "11"],  # Lanelet with ID "11"
            "12": ["2", "3", "12"],  # Lanelet with ID "12"
            "13": ["3", "4", "13"],  # Lanelet with ID "13"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "on_ramp_1": {
        "map_path": "on_ramp_1.osm",
        "n_agents": 8,
        "name": "Intersection 3",
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
    "on_ramp_2_multilane": {
        "map_path": "on_ramp_2_multilane.osm",
        "n_agents": 10,
        "name": "Scenario 4",
        "reference_paths_ids": [
            ["1", "2", "3", "4"],
            ["1", "2", "5", "6", "7", "8"],
            ["1", "2", "5", "6", "7", "15", "4"],
            ["1", "9", "11", "12"],
            ["1", "9", "11", "16", "7", "8"],
            ["1", "9", "11", "16", "7", "15", "4"],
            ["1", "2", "17", "18", "19", "20"],
            ["1", "2", "17", "18", "19", "10", "4"],
            ["1", "21", "23", "24"],
            ["1", "21", "23", "22", "19", "20"],
            ["1", "21", "23", "22", "19", "10", "4"],
            ["14", "11", "12"],
            ["14", "11", "16", "7", "8"],
            ["14", "11", "16", "7", "15", "4"],
            ["13", "6", "7", "8"],
            ["13", "6", "7", "15", "4"],
            ["26", "23", "24"],
            ["26", "23", "22", "19", "20"],
            ["26", "23", "22", "19", "10", "4"],
            ["25", "18", "19", "20"],
            ["25", "18", "19", "10", "4"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "9", "21"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "5", "9", "17", "21"],  # Lanelet with ID "2"
            "3": ["2", "3", "4", "5", "15", "17", "10"],  # Lanelet with ID "3"
            "4": ["3", "4", "15", "10"],  # Lanelet with ID "4"
            "5": ["2", "3", "5", "6", "13", "17"],  # Lanelet with ID "5"
            "6": ["5", "6", "7", "13", "16"],  # Lanelet with ID "6"
            "7": ["6", "7", "8", "15", "16"],  # Lanelet with ID "7"
            "8": ["7", "8", "15"],  # Lanelet with ID "8"
            "9": ["1", "2", "9", "11", "14"],  # Lanelet with ID "9"
            "10": ["3", "4", "15", "19", "20", "10"],  # Lanelet with ID "10"
            "11": ["9", "11", "12", "14", "16"],  # Lanelet with ID "11"
            "12": ["11", "12", "16"],  # Lanelet with ID "12"
            "13": ["5", "6", "13"],  # Lanelet with ID "13"
            "14": ["9", "11", "14"],  # Lanelet with ID "14"
            "15": ["3", "4", "7", "8", "15"],  # Lanelet with ID "15"
            "16": ["6", "7", "11", "12", "16"],  # Lanelet with ID "16"
            "17": ["2", "3", "5", "17", "18", "25"],  # Lanelet with ID "17"
            "18": ["17", "18", "19", "25", "22"],  # Lanelet with ID "18"
            "19": ["18", "19", "20", "10", "22"],  # Lanelet with ID "19"
            "20": ["19", "20", "10"],  # Lanelet with ID "20"
            "21": ["1", "2", "9", "21", "23"],  # Lanelet with ID "21"
            "22": ["18", "19", "23", "24", "22"],  # Lanelet with ID "22"
            "23": ["21", "23", "24", "26", "22"],  # Lanelet with ID "23"
            "24": ["23", "24", "22"],  # Lanelet with ID "24"
            "25": ["17", "18", "25"],  # Lanelet with ID "25"
            "26": ["21", "23", "26"],  # Lanelet with ID "26"
        },
        "figsize_x": 3.0,
        "viewer_zoom": 1.1,  # For VMAS render
        "lane_width": 0.20,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "roundabout_1": {
        "map_path": "roundabout_1.osm",
        "n_agents": 8,
        "name": "Intersection 3",
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
    "roundabout_2": {
        "map_path": "roundabout_2.osm",
        "n_agents": 10,
        "name": "Scenario 1",
        "reference_paths_ids": [
            ["8", "1", "9"],
            ["8", "1", "2", "10", "11"],
            ["8", "1", "2", "10", "12"],
            ["8", "1", "2", "3", "4", "5", "6", "16", "17"],
            ["8", "1", "2", "3", "4", "5", "6", "16", "18"],
            ["13", "4", "5", "6", "16", "17"],
            ["13", "4", "5", "6", "16", "18"],
            ["13", "4", "5", "6", "7", "1", "9"],
            ["13", "4", "5", "6", "7", "1", "2", "10", "11"],
            ["13", "4", "5", "6", "7", "1", "2", "10", "12"],
            ["14", "5", "6", "16", "17"],
            ["14", "5", "6", "16", "18"],
            ["14", "5", "6", "7", "1", "9"],
            ["14", "5", "6", "7", "1", "2", "10", "11"],
            ["14", "5", "6", "7", "1", "2", "10", "12"],
            ["15", "6", "16", "17"],
            ["15", "6", "16", "18"],
            ["15", "6", "7", "1", "9"],
            ["15", "6", "7", "1", "2", "10", "11"],
            ["15", "6", "7", "1", "2", "10", "12"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2", "7", "8", "9"],  # Lanelet with ID "1"
            "2": ["1", "2", "3", "9", "10"],  # Lanelet with ID "2"
            "3": ["2", "3", "4", "10", "13"],  # Lanelet with ID "3"
            "4": ["3", "4", "5", "13", "14"],  # Lanelet with ID "4"
            "5": ["4", "5", "6", "14", "15"],  # Lanelet with ID "5"
            "6": ["5", "6", "7", "15", "16"],  # Lanelet with ID "6"
            "7": ["1", "6", "7", "8", "16"],  # Lanelet with ID "7"
            "8": ["1", "7", "8"],  # Lanelet with ID "8"
            "9": ["1", "2", "9"],  # Lanelet with ID "9"
            "10": ["2", "3", "10", "11", "12"],  # Lanelet with ID "10"
            "11": ["10", "11", "12"],  # Lanelet with ID "11"
            "12": ["10", "11", "12"],  # Lanelet with ID "12"
            "13": ["3", "4", "13"],  # Lanelet with ID "13"
            "14": ["4", "5", "14"],  # Lanelet with ID "14"
            "15": ["5", "6", "15"],  # Lanelet with ID "15"
            "16": ["6", "7", "16", "17", "18"],  # Lanelet with ID "16"
            "17": ["16", "17", "18"],  # Lanelet with ID "17"
            "18": ["16", "17", "18"],  # Lanelet with ID "18"
        },
        "figsize_x": 2.0,
        "viewer_zoom": 1.0,  # For VMAS render
        "lane_width": 0.2,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "pseudo_distance_example": {
        "map_path": "pseudo_distance_example.osm",
        "n_agents": 1,
        "name": "pseudo_distance_example",
        "reference_paths_ids": [
            ["1"],
            ["2"],
        ],
        "neighboring_lanelet_ids": {
            "1": ["1", "2"],  # Lanelet with ID "1"
            "2": ["1", "2"],  # Lanelet with ID "2"
        },
        "figsize_x": 2.0,
        "viewer_zoom": 1.0,  # For VMAS render
        "lane_width": 0.2,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
}

AGENTS = {
    "width": 0.08,  # [m]
    "length": 0.16,  # [m]
    "l_f": 0.08,  # [m] Front wheelbase
    "l_r": 0.08,  # [m] Rear wheelbase
    "l_wb": 0.16,  # [m] Wheelbase
    "max_speed": 0.5,  # [m/s]
    "min_speed": -0.5,  # [m/s]
    "max_speed_achievable": 0.5,  # [m/s]
    "max_steering": math.pi / 4,  # [radian]
    "min_steering": -math.pi / 4,  # [radian]
    "max_acc": 5.0,  # [m/s^2]
    "min_acc": -5.0,  # [m/s^2]
    "max_steering_rate": 5 * math.pi,  # [radian/s]
    "min_steering_rate": -5 * math.pi,  # [radian/s]
    "n_actions": 2,
}

THRESHOLD = {
    "initial_distance": 1.2 * math.sqrt(AGENTS["width"] ** 2 + AGENTS["length"] ** 2),
    "reach_goal": AGENTS[
        "width"
    ],  # An agent is considered having reached its goal if its distance to the goal is less than its width
}
