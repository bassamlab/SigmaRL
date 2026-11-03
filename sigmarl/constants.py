# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import warnings

SCENARIOS = {
    "cpm_entire": {
        "map_path": "cpm.xml",
        "n_agents": 12,
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
    # cpm_mixed scenario is mainly used as an intersection scenario with cpm_scenario_probabilities = [1.0, 0, 0] in sigmarl/helper_common.py
    "cpm_mixed": {
        "map_path": "cpm.xml",
        "n_agents": 6,
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
    "cpm_intersection_extended": {
        "map_path": "cpm.xml",
        "n_agents": 8,
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
        "lane_width": 0.30,  # [m] Lane width
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
        "lane_width": 0.30,  # [m] Lane width
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
        "lane_width": 0.30,  # [m] Lane width
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
        "lane_width": 0.30,  # [m] Lane width
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
        "viewer_zoom": 1.45,  # For VMAS render
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
        "lane_width": 0.30,  # [m] Lane width
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
        "lane_width": 0.30,  # [m] Lane width
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
        "lane_width": 0.30,  # [m] Lane width
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
        "lane_width": 0.30,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    "on_ramp_1": {
        "map_path": "on_ramp_1.osm",
        "n_agents": 6,
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
        "lane_width": 0.30,  # [m] Lane width
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
        "lane_width": 0.30,  # [m] Lane width
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
        "lane_width": 0.30,  # [m] Lane width
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
        "viewer_zoom": 1.4,  # For VMAS render
        "lane_width": 0.30,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
    # rounD location 0 as a Lanelet2 map (from drone-dataset-tools), scaled 1:18 to the size of the CPM Lab
    # vehicles. The reference paths are all entry-to-exit routes, computed by ParseLanelet2.
    "round_location0_1to18": {
        "map_path": "round_location0.osm",
        "map_format": "lanelet2",
        "n_agents": 10,
        "name": "rounD Location 0 (1:18)",
        "utm_origin": [
            301221.365,
            5641501.341,
        ],  # xUtmOrigin, yUtmOrigin of the rounD recordings at location 0
        "figsize_x": 3.0,
        "viewer_zoom": 1.4,  # For VMAS render
        "lane_width": 0.24,  # [m] Median lane width (4.3 m / 18)
        "point_spacing": 0.08,  # [m] Max. distance between boundary points (1.5 m / 18)
        "scale": 1 / 18,  # Map units per meter of the real road
    },
    "round_location0": {
        "map_path": "round_location0.osm",
        "map_format": "lanelet2",
        "agent_profile": "full_scale",  # Agent profile this scenario is sized for
        "n_agents": 10,
        "name": "rounD Location 0",
        "utm_origin": [
            301221.365,
            5641501.341,
        ],  # xUtmOrigin, yUtmOrigin of the rounD recordings at location 0
        "figsize_x": 3.0,
        "viewer_zoom": 1.4,  # For VMAS render
        "lane_width": 0.24 * 18,  # [m] Median lane width
        "point_spacing": 0.08 * 18,  # [m] Max. distance between boundary points
        "scale": 1.0,  # Real size, 18 times the size of round_location0_1to18
        "max_speed": 50 / 3.6,  # [m/s] Speed limit (urban 50 km/h)
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
        "lane_width": 0.30,  # [m] Lane width
        "scale": 1e5,  # Scale the map
    },
}

# Lanelet2 maps of the inD (intersections) and exiD (highway entries and exits) locations, fixed and cut to the
# recorded area by `src/fix_lanelet_map.py` in drone-dataset-tools (like round_location0.osm), and of the highD
# (highway) locations, at real size for the `full_scale` agent profile. The reference paths are all entry-to-exit
# routes, computed by ParseLanelet2.
DRONE_DATASET_LANELET2_MAPS = {
    # Scenario: (name, xUtmOrigin and yUtmOrigin of the recordings, median lane width [m])
    "ind_location1": ("inD Bendplatz", [293487.12240, 5629711.58163], 3.3),
    "ind_location1_construction": (
        "inD Bendplatz (construction site)",
        [293487.12240, 5629711.58163],
        3.3,
    ),
    "ind_location2": ("inD Frankenburg", [295620.95750, 5628102.04258], 3.1),
    "ind_location3": ("inD Heckstrasse", [300127.08530, 5629091.05870], 3.2),
    "ind_location4": ("inD Aseag", [297631.31870, 5629917.34465], 3.5),
    "exid_location0": ("exiD Cologne Butzweiler", [352146.6, 5651141.9], 3.8),
    "exid_location1": ("exiD Cologne Fortiib", [351816.7, 5651523.4], 3.7),
    "exid_location2": ("exiD Aachen Brand", [298089.6, 5626462.9], 3.9),
    "exid_location3": ("exiD Bergheim Roemer", [334905.3, 5645453.9], 3.9),
    "exid_location4": ("exiD Cologne Klettenberg", [353013.5, 5640508.6], 3.7),
    "exid_location5": ("exiD Aachen Laurensberg", [293790.2, 5632271.9], 4.0),
    "exid_location6": ("exiD Merzenich Rather", [325298.1, 5635819.6], 3.6),
    # The highD maps are in local coordinates; 3.75 m is the German motorway lane width
    **{
        f"highd_location{i}": (f"highD location {i}", [0.0, 0.0], 3.75)
        for i in range(1, 7)
    },
}
# Dataset: (speed limit [m/s], CLF reference speed [m/s] or None for the default).
# inD follows the maps' speed_limit tags (highest per map), exiD 120 km/h, and
# highD the 130 km/h advisory speed on German motorways.
DRONE_DATASET_SPEEDS = {
    "ind": (50 / 3.6, None),
    "exid": (120 / 3.6, 25.0),
    "highd": (130 / 3.6, 30.0),
}
DRONE_DATASET_SPEED_LIMIT_OVERRIDES = {"ind_location2": 30 / 3.6}  # 30 km/h zone
for _scenario, (_name, _utm_origin, _lane_width) in DRONE_DATASET_LANELET2_MAPS.items():
    _max_speed, _ref_speed = DRONE_DATASET_SPEEDS[_scenario.split("_")[0]]
    SCENARIOS[_scenario] = {
        "map_path": f"{_scenario}.osm",
        "map_format": "lanelet2",
        "agent_profile": "full_scale",  # Agent profile this scenario is sized for
        "n_agents": 10,
        "name": _name,
        "utm_origin": _utm_origin,  # xUtmOrigin, yUtmOrigin of the recordings
        "figsize_x": 3.0,
        "viewer_zoom": 1.4,  # For VMAS render
        "lane_width": _lane_width,  # [m] Median lane width
        "point_spacing": 0.08 * 18,  # [m] Max. distance between boundary points
        "scale": 1.0,  # Real size
        # [m/s] Speed limit (see Parameters.resolve_speed_defaults)
        "max_speed": DRONE_DATASET_SPEED_LIMIT_OVERRIDES.get(_scenario, _max_speed),
    }
    if _ref_speed is not None:
        SCENARIOS[_scenario]["ref_speed"] = _ref_speed  # [m/s] CLF reference speed

# VMAS scenarios that do not use the road-traffic map stack. These entries are
# kept separate so the higher-speed road scenario definitions above remain the
# source of truth.
NON_ROAD_SCENARIOS = {
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
        "viewer_zoom": 3.5,
        "resolution_factor": 50,
        "lane_width": 0.30,
        "scale": 0.0,
    },
    "balance": {"n_agents": 4},
    "ball_passage": {"n_agents": 2},
    "ball_trajectory": {"n_agents": 2},
    "buzz_wire": {"n_agents": 2},
    "discovery": {"n_agents": 6},
    "dispersion": {"n_agents": 8},
    "dropout": {"n_agents": 6},
    "flocking": {"n_agents": 6},
    "football": {"n_agents": 8},
    "navigation": {"n_agents": 6},
    "sampling": {"n_agents": 5},
    "wheel": {"n_agents": 4},
    "wind_flocking": {"n_agents": 2},
    "give_way": {"n_agents": 2},
    "joint_passage_size": {"n_agents": 2},
    "joint_passage": {"n_agents": 2},
    "multi_give_way": {"n_agents": 4},
    "reverse_transport": {"n_agents": 4},
    "transport": {"n_agents": 4},
}
SCENARIOS.update(NON_ROAD_SCENARIOS)
ROAD_TRAFFIC_SCENARIOS = set(SCENARIOS).difference(NON_ROAD_SCENARIOS)

# `max_speed` of a profile is the default speed limit of the agents; a scenario
# may set its own (see "max_speed" in SCENARIOS and Parameters.resolve_speed_defaults).
AGENT_PROFILES = {
    # Smaller agents with higher acceleration and steering-rate limits.
    "agile": {
        "width": 0.08,
        "length": 0.16,
        "l_f": 0.08,
        "l_r": 0.08,
        "l_wb": 0.16,
        "max_speed": 1.0,
        "min_speed": -0.5,
        "max_steering": math.pi / 4,
        "min_steering": -math.pi / 4,
        "max_acc": 20.0,
        "min_acc": -20.0,
        "max_steering_rate": 5 * math.pi,
        "min_steering_rate": -5 * math.pi,
        "n_actions": 2,
    },
    # Lab dynamics with the smaller geometry used for simulation training.
    "semi_real_dyn": {
        "width": 0.08,
        "length": 0.16,
        "l_f": 0.075,
        "l_r": 0.075,
        "l_wb": 0.15,
        "max_speed": 1.0,
        "min_speed": -1.0,
        "max_steering": 24 * math.pi / 180,
        "min_steering": -24 * math.pi / 180,
        "max_acc": 2.5,
        "min_acc": -2.5,
        "max_steering_rate": 1.75,
        "min_steering_rate": -1.75,
        "n_actions": 2,
    },
    # Lab dynamics and the physical vehicle geometry.
    "real_dyn": {
        "width": 0.107,
        "length": 0.22,
        "l_f": 0.075,
        "l_r": 0.075,
        "l_wb": 0.15,
        "max_speed": 1.0,
        "min_speed": -1.0,
        "max_steering": 24 * math.pi / 180,
        "min_steering": -24 * math.pi / 180,
        "max_acc": 2.5,
        "min_acc": -2.5,
        "max_steering_rate": 1.75,
        "min_steering_rate": -1.75,
        "n_actions": 2,
    },
    "real_dyn_2": {
        "width": 0.107,
        "length": 0.22,
        "l_f": 0.075,
        "l_r": 0.075,
        "l_wb": 0.15,
        "max_speed": 1.0,
        "min_speed": -1.0,
        "max_steering": 24 * math.pi / 180,
        "min_steering": -24 * math.pi / 180,
        "max_acc": 4.0,
        "min_acc": -4.0,
        "max_steering_rate": 1.75,
        "min_steering_rate": -1.75,
        "n_actions": 2,
    },
    # Full-size passenger car, based on the cars in the rounD dataset (median
    # 2.07 m x 4.57 m; speed median 6.9 m/s, 99th percentile 12.6 m/s).
    "full_scale": {
        "width": 2.0,
        "length": 4.6,
        "l_f": 1.35,
        "l_r": 1.35,
        "l_wb": 2.7,
        "max_speed": 14.0,
        "min_speed": 0.0,
        "max_steering": 35 * math.pi / 180,
        "min_steering": -35 * math.pi / 180,
        "max_acc": 3.0,
        "min_acc": -6.0,
        # Fast for a car; with 0.6 rad/s the CLF-CBF-QP left the lane more often
        "max_steering_rate": 1.0,
        "min_steering_rate": -1.0,
        "n_actions": 2,
        # Factors by which lengths and typical speeds exceed those of the lab
        # vehicles. Defaults of parameters given in absolute meters or meters per
        # second are multiplied by them (see LENGTH_SCALE and SPEED_SCALE below).
        "length_scale": 20.0,
        "speed_scale": 8.0,
    },
}

# The profile is selected before any module copies its values, so it is read
# from the environment variable SIGMARL_AGENT_PROFILE (default: "real_dyn_2").
ACTIVE_AGENT_PROFILE = os.environ.get(
    "SIGMARL_AGENT_PROFILE",
    "real_dyn_2",  # one of {"agile", "semi_real_dyn", "real_dyn" "real_dyn_2", "full_scale"}
)
if ACTIVE_AGENT_PROFILE not in AGENT_PROFILES:
    raise ValueError(
        f"Unknown agent profile '{ACTIVE_AGENT_PROFILE}' in SIGMARL_AGENT_PROFILE. "
        f"Available profiles: {sorted(AGENT_PROFILES)}."
    )
AGENTS = AGENT_PROFILES[ACTIVE_AGENT_PROFILE]
# Scale factors for parameters in absolute units that were tuned for the lab
# vehicles (1.0 for all lab profiles, so their defaults are unchanged).
LENGTH_SCALE = AGENTS.get("length_scale", 1.0)  # For [m] (squared for [m^2])
SPEED_SCALE = AGENTS.get("speed_scale", 1.0)  # For [m/s]
ACCELERATION_SCALE = SPEED_SCALE**2 / LENGTH_SCALE  # For [m/s^2]
# Unitless factor of the CBF-margin reward normalizers: with 1, a reward channel
# saturates at -1 when an agent closes on its barrier at the speed limit while
# the barrier requires holding (see CBFQP.cbf_margin_reward_scales).
H_NOM = 1.0
print(
    f"[INFO] Using agent profile '{ACTIVE_AGENT_PROFILE}' with parameters:\n {AGENTS}"
)


def validate_agent_profile(saved_profile_name: str | None) -> None:
    """Validate checkpoint agent metadata against the active profile."""
    if saved_profile_name is None:
        warnings.warn(
            "The checkpoint predates agent-profile metadata; compatibility with "
            f"the active profile '{ACTIVE_AGENT_PROFILE}' cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if saved_profile_name != ACTIVE_AGENT_PROFILE:
        raise ValueError(
            "Agent profile mismatch: checkpoint was trained with "
            f"'{saved_profile_name}', but '{ACTIVE_AGENT_PROFILE}' is active. "
            f"Set the environment variable SIGMARL_AGENT_PROFILE={saved_profile_name}."
        )


def validate_scenario_agent_profile(scenario_type: str) -> None:
    """Raise if a scenario is sized for an agent profile other than the active one."""
    required_profile = SCENARIOS[scenario_type].get("agent_profile")
    if required_profile is not None and required_profile != ACTIVE_AGENT_PROFILE:
        raise ValueError(
            f"Scenario '{scenario_type}' requires agent profile '{required_profile}', "
            f"but '{ACTIVE_AGENT_PROFILE}' is active. Set the environment variable "
            f"SIGMARL_AGENT_PROFILE={required_profile}."
        )


# Used when parameters.rl_action_type is "position_speed"
POSITION_SPEED_LOOKAHEAD_DIST = AGENTS["length"]  # [m]
POSITION_SPEED_LATERAL_OFFSET_BOUND = AGENTS["width"] / 2.0  # [m]

THRESHOLD = {
    "initial_distance": 1.2 * math.sqrt(AGENTS["width"] ** 2 + AGENTS["length"] ** 2),
    "reach_goal": AGENTS[
        "width"
    ],  # An agent is considered having reached its goal if its distance to the goal is less than its width
    "disk_radius": 0.5
    * math.sqrt(
        AGENTS["width"] ** 2 + AGENTS["length"] ** 2
    ),  # [m] The radius of the disk that fully encloses the agent's shape
}
