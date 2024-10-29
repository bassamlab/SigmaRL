# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.functional import pad

from sigmarl.parse_osm import ParseOSM
from sigmarl.parse_xml import ParseXML


class MapManager:
    def __init__(self, **kwargs):
        self._scenario_type = kwargs.pop(
            "scenario_type", "CPM_entire"
        )  # One of {"CPM_entire", "CPM_mixed", "intersection_1", "design you own map and name it here"}

        self._device = kwargs.pop("device", "cpu")
        self.current_lanelet_idx = []

        self._parse_map_file()

    def _parse_map_file(self):
        # Get map data
        if "CPM" in self._scenario_type:
            # Load the map data of the CPM Lab
            map_parser = ParseXML(
                scenario_type=self._scenario_type,
                device=self._device,
            )
        else:
            map_parser = ParseOSM(
                scenario_type=self._scenario_type,
                device=self._device,
            )

        self.parser = map_parser

    def determine_current_lanelet(self, agent_pos):
        """
        Determines the lanelet IDs for multiple agents based on their current positions.

        Args:
            agent_pos (torch.Tensor): A tensor of shape (batch_dim, 2) representing the positions of multiple agents,
                                      where the first column corresponds to x-coordinates and the second to y-coordinates.

        Returns:
            torch.Tensor: A tensor of shape (batch_dim, 1) containing the lanelet IDs for each agent.
        """
        # Determine the maximum length of center_line tensors
        max_length = max(
            len(lanelet["center_line"]) for lanelet in self.parser.lanelets_all
        )

        # Pad all center_line tensors to the same length
        lanelet_centers_padded = [
            pad(
                lanelet["center_line"],
                (0, 0, 0, max_length - len(lanelet["center_line"])),
            )
            for lanelet in self.parser.lanelets_all
        ]
        lanelet_centers_padded = torch.stack(
            lanelet_centers_padded
        )  # Shape (num_lanelets, max_length, 2)

        # Expand agent positions and lanelet centers for distance calculation
        agent_pos_expanded = agent_pos.unsqueeze(2).unsqueeze(
            3
        )  # Shape (batch_size, num_agents, 1, 1, 2)
        lanelet_centers_expanded = lanelet_centers_padded.unsqueeze(0).unsqueeze(
            0
        )  # Shape (1, 1, num_lanelets, max_length, 2)

        # Compute squared distances between each agent and each lanelet center point
        dists = torch.sum((agent_pos_expanded - lanelet_centers_expanded) ** 2, dim=4)

        # Find the minimum distance for each agent across all center points of each lanelet
        min_dists, _ = torch.min(
            dists, dim=3
        )  # Shape (batch_size, num_agents, num_lanelets)

        # Find the lanelet with the minimum distance for each agent
        self.current_lanelet_idx = torch.argmin(min_dists, dim=2).unsqueeze(
            2
        )  # Shape (batch_size, num_agents, 1)

    def determine_masked_agents_by_lanelets(self, agent_idx, nearing_agents_indices):
        """
        Determines which nearing agents should be masked by each ego agent because of that their lanelets are not neighboring lanelets.

        Args:
            agent_idx: the index of the current agent.
            nearing_agents_indices (torch.Tensor): A tensor of shape (batch_dim, n_nearing_agents_observed) representing the indices of nearing agents for each ego agent.

        Returns:
            torch.Tensor: A tensor of shape (batch_dim, n_nearing_agents_observed) containing boolean values where True indicates that the nearing agent should be masked and False otherwise.
        """
        # Prepare a tensor to store the mask results
        mask = torch.zeros(nearing_agents_indices.shape, dtype=torch.bool)

        # Iterate through each agent and its nearing agents
        for env_idx in range(self.current_lanelet_idx.size(0)):
            ego_lanelet_idx = self.current_lanelet_idx[env_idx, agent_idx].item()
            for near_idx in range(nearing_agents_indices.size(1)):
                near_agent_idx = nearing_agents_indices[env_idx, near_idx].item()
                near_lanelet = self.current_lanelet_idx[env_idx, near_agent_idx].item()
                if (
                    near_lanelet
                    not in self.parser.neighboring_lanelets_idx[ego_lanelet_idx]
                ):
                    mask[env_idx, near_idx] = True

        return mask
