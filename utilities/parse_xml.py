import xml.etree.ElementTree as ET
import torch
import matplotlib.pyplot as plt
import numpy as np


import os
import sys
script_dir = os.path.dirname(__file__) # Directory of the current script
project_root = os.path.dirname(script_dir) # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.map_parse_base import MapParseBase

from utilities.colors import Color


class ParseXML(MapParseBase):
    def __init__(self, map_path, device="cpu"):
        super().__init__(map_path, device)  # Initialize base class
        
        self.bounds["min_x"] = 0
        self.bounds["max_x"] = 4.5
        self.bounds["min_y"] = 0
        self.bounds["max_y"] = 4.0
        
        # Visualization
        self.is_visu_lane_ids = False  # Whether to visualize the IDs of the lanes
        self._linewidth = 1.0
        self._fontsize = 12
        
        self.mean_lane_width = None
        self.intersection_info = None
        
        self._parse_map_file()
        self.visualize_map(is_save_fig=True, is_show=False, is_show_an_agent=False)
        self._get_reference_paths(is_show_each_ref_path=False)
        
    def _parse_map_file(self):
        tree = ET.parse(self._map_path)
        root = tree.getroot()
        lanelets = []
        for child in root:
            if child.tag == "lanelet":
                lanelets.append(self._parse_lanelet(child))
            elif child.tag == "intersection":
                self.intersection_info = self._parse_intersections(child)

        # Storing all the map data
        self.map_data = lanelets
        
        # Calculate the mean lane width
        self.mean_lane_width = torch.mean(torch.norm(torch.vstack([lanelets[i]["left_boundary"] for i in range(len(lanelets))]) - torch.vstack([lanelets[i]["right_boundary"] for i in range(len(lanelets))]), dim=1))
        
    def _parse_lanelet(self, element):
        """ Parses a lanelet element to extract detailed information. """
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
            "lanelet_type": None
        }
        for child in element:
            if child.tag == "leftBound":
                lanelet_data["left_boundary"], lanelet_data["left_line_marking"] = self._parse_bound(child)
            elif child.tag == "rightBound":
                lanelet_data["right_boundary"], lanelet_data["right_line_marking"] = self._parse_bound(child)
            elif child.tag == "predecessor":
                lanelet_data["predecessor"].append(int(child.get("ref")))
            elif child.tag == "successor":
                lanelet_data["successor"].append(int(child.get("ref")))
            elif child.tag == "adjacentLeft":
                lanelet_data["adjacent_left"] = {
                    "id": int(child.get("ref")),
                    "drivingDirection": child.get("drivingDir")
                }
            elif child.tag == "adjacentRight":
                lanelet_data["adjacent_right"] = {
                    "id": int(child.get("ref")),
                    "drivingDirection": child.get("drivingDir")
                }
            elif child.tag == "lanelet_type":
                lanelet_data["lanelet_type"] = child.text

        lanelet_data["center_line"] = (lanelet_data["left_boundary"] + lanelet_data["right_boundary"]) / 2
        
        lanelet_data["center_line_center_points"], lanelet_data["center_line_lengths"], lanelet_data["center_line_yaws"], _ = MapParseBase.get_center_length_yaw_polyline(polyline=lanelet_data["center_line"])
        lanelet_data["left_boundary_center_points"], lanelet_data["left_boundary_lengths"], lanelet_data["left_boundary_yaws"], _ = MapParseBase.get_center_length_yaw_polyline(polyline=lanelet_data["left_boundary"])
        lanelet_data["right_boundary_center_points"], lanelet_data["right_boundary_lengths"], lanelet_data["right_boundary_yaws"], _ = MapParseBase.get_center_length_yaw_polyline(polyline=lanelet_data["right_boundary"])
        
        return lanelet_data
    
    def _parse_bound(self, element):
        """ Parses a bound (left boundary or right boundary) element to extract points and line marking. """
        points = [self._parse_point(point) for point in element.findall("point")]
        points = torch.vstack(points)
        line_marking = element.find("lineMarking").text if element.find("lineMarking") is not None else None
        return points, line_marking

    def _parse_point(self, element):
        """ Parses a point element to extract x and y coordinates. """
        x = float(element.find("x").text) if element.find("x") is not None else None
        y = float(element.find("y").text) if element.find("y") is not None else None
        return torch.tensor([x, y], device=self._device)
    
    def _parse_intersections(self, element):
        """Function to parse intersections. """
        intersection_info = []

        for incoming in element.findall("incoming"):
            incoming_info = {
                "incomingLanelet": int(incoming.find("incomingLanelet").get("ref")), # The starting lanelet of a part of the intersection
                "successorsRight": int(incoming.find("successorsRight").get("ref")), # The successor right lanelet of the incoming lanelet
                "successorsStraight": [int(s.get("ref")) for s in incoming.findall("successorsStraight")], # The successor lanelet(s) of the incoming lanelet
                "successorsLeft": int(incoming.find("successorsLeft").get("ref")), # The successor left lanelet of the incoming lanelet
            }
            intersection_info.append(incoming_info)

        return intersection_info
    
    def visualize_map(self, is_save_fig, is_show, is_show_an_agent):
        x_lim = 4.5 # [m] Dimension in x-direction 
        y_lim = 4.0 # [m] Dimension in y-direction 

        # Set up the plot
        plt.figure(figsize=(x_lim*3, y_lim*3))  # Size in inches, adjusted for 4.0m x 4.5m dimensions
        plt.axis("equal")  # Ensure x and y dimensions are equally scaled

        for lanelet in self.map_data:
            # Extract coordinates for left, right, and center lines
            left_bound = lanelet["left_boundary"]
            right_bound = lanelet["right_boundary"]
            center_line = lanelet["center_line"]

            # Extract line markings
            left_line_marking = lanelet["left_line_marking"]
            right_line_marking = lanelet["right_line_marking"]
            center_line_marking = lanelet["center_line_marking"]

            is_use_random_color = False
            # Choose color
            color = np.random.rand(3,) if is_use_random_color else "grey"

            # Plot left boundary
            plt.plot(left_bound[:, 0], left_bound[:, 1], linestyle="--" if left_line_marking == "dashed" else "-", color=color, linewidth=self._linewidth)
            # Plot right boundary
            plt.plot(right_bound[:, 0], right_bound[:, 1], linestyle="--" if right_line_marking == "dashed" else "-", color=color, linewidth=self._linewidth)
            # Plot center line
            # plt.plot(center_line[:, 0], center_line[:, 1], linestyle="--" if center_line_marking == "dashed" else "-", color=color, linewidth=self._linewidth)
            # Adding lanelet ID as text
            if self.is_visu_lane_ids:
                plt.text(center_line[int(len(center_line)/2), 0], center_line[int(len(center_line)/2), 1], str(lanelet["id"]), color=color, fontsize=self._fontsize)
                
        if is_show_an_agent:
            w = 0.1 # Width
            l = 0.2 # Length
            
            rec_x = [0.208, 0.208, 0.208 + w, 0.208 + w, 0.208]
            rec_y = [2.1, 2.1 - l, 2.1 - l, 2.1, 2.1]
            plt.fill(rec_x, rec_y, color=Color.blue100)

        plt.xlabel(r"$x$ [m]", fontsize=18)
        plt.ylabel(r"$y$ [m]", fontsize=18)
        plt.xlim((0, x_lim))
        plt.ylim((0, y_lim))
        plt.xticks(np.arange(0, x_lim+0.05, 0.5), fontsize=self._fontsize)
        plt.yticks(np.arange(0, y_lim+0.05, 0.5), fontsize=self._fontsize)
        plt.title("CPM Map Visualization", fontsize=18)

        # Save fig
        if is_save_fig:
            plt.tight_layout() # Set the layout to be tight to minimize white space
            plt.savefig(self._file_name + ".pdf", format="pdf", bbox_inches="tight")
            
        if is_show:
            plt.show()

    def _get_reference_paths(self, is_show_each_ref_path):        
        # Mapping agent_id to loop index and starting lanelet: agent_id: (loop_index, starting_lanelet)
        path_to_loop = {
            1: (1, 4), 2: (2, 1), 3: (3, 64), 4: (4, 42), 5: (5, 22), 6: (6, 39), 7: (7, 15), 
            8: (1, 8), 9: (2, 10), 10: (3, 75), 11: (4, 45), 12: (5, 59), 13: (6, 61), 14: (7, 5), 
            15: (1, 58), 16: (2, 17), 17: (3, 79), 18: (4, 92), 19: (5, 68), 20: (6, 55), 21: (7, 11), 
            22: (1, 54), 23: (2, 38), 24: (3, 88), 25: (4, 100), 26: (5, 19), 27: (6, 65), 28: (7, 93), 
            29: (1, 82), 30: (2, 49), 31: (3, 95), 32: (4, 33), 33: (5, 14), 34: (6, 35), 35: (7, 83), 
            36: (1, 86), 37: (6, 29), 38: (7, 89), 39: (1, 32), 40: (1, 28)
        }
        # Some lanelets share the same boundary (such as adjacent left and adjacent right lanelets)
        lanelets_share_same_boundaries_list = [
            [4, 3, 22], [6, 5, 23], [8, 7], [60, 59], [58, 57, 75], [56, 55, 74], [54, 53], [80, 79], [82, 81, 100], [84, 83, 101], [86, 85], [34, 33], [32, 31, 49], [30, 29, 48], [28, 27], [2, 1], # outer circle (urban)
            [13, 14], [15, 16], [9, 10], [11, 12], # inner circle (top right) 
            [63, 64], [61, 62], [67, 68], [65, 66], # inner circle (bottom right)
            [91, 92], [93, 94], [87, 88], [89, 90], # inner circle (bottom left)
            [37, 38], [35, 36], [41, 42], [39, 40], # inner circle (top left)
            [25, 18], [26, 17], [52, 43], [72, 73], # intersection: incoming 1 and incoming 2
            [51, 44], [50, 45], [102, 97], [20, 21], # intersection: incoming 3 and incoming 4
            [103, 96], [104, 95], [78, 69], [46, 47], # intersection: incoming 5 and incoming 6
            [77, 70], [76, 71], [24, 19], [98, 99], # intersection: incoming 7 and incoming 8
        ]
        path_intersection = [
            [11, 25, 13],
            [11, 26, 52, 37],
            [11, 72, 91],
            [12, 18, 14],
            [12, 17 ,43, 38],
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
        path_merge_in = [
            [34, 32],
            [33, 31],
            [35, 31],
            [36, 49],
        ]
        path_merge_out = [
            [6, 8],
            [5, 7],
            [5, 9],
            [23, 10],
        ]
        
        num_paths_all = len(path_to_loop)
        for ref_path_id in range(num_paths_all):
            reference_lanelets_index = self._get_reference_lanelet_index(ref_path_id + 1, path_to_loop) # Path ID starts from 1
            reference_path = self._get_reference_path(reference_lanelets_index, lanelets_share_same_boundaries_list, is_show_each_ref_path)
            self.reference_paths.append(reference_path)

        for reference_lanelets_index in path_intersection:
            reference_path = self._get_reference_path(reference_lanelets_index, lanelets_share_same_boundaries_list, is_show_each_ref_path)
            self.reference_paths_intersection.append(reference_path)
            
        for reference_lanelets_index in path_merge_in:
            reference_path = self._get_reference_path(reference_lanelets_index, lanelets_share_same_boundaries_list, is_show_each_ref_path)
            self.reference_paths_merge_in.append(reference_path)

        for reference_lanelets_index in path_merge_out:
            reference_path = self._get_reference_path(reference_lanelets_index, lanelets_share_same_boundaries_list, is_show_each_ref_path)
            self.reference_paths_merge_out.append(reference_path)

    def _get_reference_path(self, reference_lanelets_index, lanelets_share_same_boundaries_list, is_show_each_ref_path):
        # Initialize
        left_boundaries = None
        right_boundaries = None
        left_boundaries_shared = None
        right_boundaries_shared = None
        center_lines = None

        for lanelet in reference_lanelets_index:
            lanelets_share_same_boundaries = next((group for group in lanelets_share_same_boundaries_list if lanelet in group), None)
            assert(lanelets_share_same_boundaries != None)
            
            # Extracting left and right boundaries
            left_bound = self.map_data[lanelet - 1]["left_boundary"] # Lanelet IDs start from 1, while the index of a list in Python starts from 0 
            right_bound = self.map_data[lanelet - 1]["right_boundary"]
            left_bound_shared = self.map_data[lanelets_share_same_boundaries[0] - 1]["left_boundary"]
            right_bound_shared = self.map_data[lanelets_share_same_boundaries[-1] - 1]["right_boundary"]

            if left_boundaries is None:
                left_boundaries = left_bound
                right_boundaries = right_bound
                left_boundaries_shared = left_bound_shared
                right_boundaries_shared = right_bound_shared
            else:
                # Concatenate boundary data while avoiding duplicated adta at the connection point
                if torch.norm(left_boundaries[-1,:] - left_bound[0,:]) < 1e-4:
                    left_boundaries = torch.cat((left_boundaries, left_bound[1:,:]), dim=0)
                    left_boundaries_shared = torch.cat((left_boundaries_shared, left_bound_shared[1:,:]), dim=0)
                else:
                    left_boundaries = torch.cat((left_boundaries, left_bound), dim=0)
                    left_boundaries_shared = torch.cat((left_boundaries_shared, left_bound_shared), dim=0)

                if torch.norm(right_boundaries[-1,:] - right_bound[0,:]) < 1e-4:
                    right_boundaries = torch.cat((right_boundaries, right_bound[1:,:]), dim=0)
                    right_boundaries_shared = torch.cat((right_boundaries_shared, right_bound_shared[1:,:]), dim=0)
                else:
                    right_boundaries = torch.cat((right_boundaries, right_bound), dim=0)
                    right_boundaries_shared = torch.cat((right_boundaries_shared, right_bound_shared), dim=0)


        center_lines = (left_boundaries + right_boundaries) / 2
        
        # Check if the center line is a loop
        is_loop = (center_lines[0, :] - center_lines[-1, :]).norm() <= 1e-4
        
        center_lines_vec = torch.diff(center_lines, dim=0) # Vectors connecting each pair of neighboring points on the center lines
        center_lines_vec_length = torch.norm(center_lines_vec, dim=1) # The lengths of the vectors
        center_lines_vec_mean_length = torch.mean(center_lines_vec_length) # The mean length of the vectors
        center_lines_vec_normalized = center_lines_vec / center_lines_vec_length.unsqueeze(1)
        
        center_line_yaw = torch.atan2(center_lines_vec[:,1], center_lines_vec[:,0])

        assert(left_boundaries.shape == left_boundaries.shape)
        assert(left_boundaries.shape[1] == 2) # Must be a two-column array

        if is_show_each_ref_path:
            # Mainly used for debugging
            plt.plot(left_boundaries[:,0], left_boundaries[:,1], color="black", linewidth=0.5, linestyle="-")
            plt.plot(right_boundaries[:,0], right_boundaries[:,1], color="black", linewidth=0.5, linestyle="-")

            # Filling between boundaries
            x = torch.cat((left_boundaries_shared[:, 0], torch.flip(right_boundaries_shared[:, 0], dims=[0])))
            y = torch.cat((left_boundaries_shared[:, 1], torch.flip(right_boundaries_shared[:, 1], dims=[0])))

            plt.fill(x, y, color="lightgrey", alpha=0.5)  # "alpha" controls the transparency

            plt.plot(center_lines[:,0], center_lines[:,1], color="grey", linewidth=0.5, linestyle="--")
            
            plt.show()
            
        reference_path = {   
            "reference_lanelets": reference_lanelets_index,
            "left_boundary": left_boundaries,
            "right_boundary": right_boundaries,
            "left_boundary_shared": left_boundaries_shared,
            "right_boundary_shared": right_boundaries_shared,
            "center_line": center_lines, # Center lines are calculated based on left and right boundaries (instead of shared left and right boundaries)
            "center_line_yaw": center_line_yaw, # Yaw angle of each point on the center lines
            "center_line_vec_normalized": center_lines_vec_normalized, # Normalized vectors connecting each pair of neighboring points on the center lines
            "center_line_vec_mean_length": center_lines_vec_mean_length,
            "is_loop": is_loop,
        }
        return reference_path
    
    def _get_reference_lanelet_index(self, ref_path_id, path_to_loop):
        """
        Get loop of lanelets used for reference_path_struct.

        Args:
        ref_path_id (int): Path ID.

        Returns:
        list: List of lanelets indices.
        """
        # Define loops of paths (successive lanelets)
        reference_lanelets_loops = [
            [4, 6, 8, 60, 58, 56, 54, 80, 82, 84, 86, 34, 32, 30, 28, 2], # Loop 1
            [1, 3, 23, 10, 12, 17, 43, 38, 36, 49, 29, 27], # Loop 2
            [64, 62, 75, 55, 53, 79, 81, 101, 88, 90, 95, 69], # Loop 3
            [40, 45, 97, 92, 94, 100, 83, 85, 33, 31, 48, 42], # Loop 4
            [5, 7, 59, 57, 74, 68, 66, 71, 19, 14, 16, 22], # Loop 5
            [41, 39, 20, 63, 61, 57, 55, 67, 65, 98, 37, 35, 31, 29], # Loop 6
            [3, 5, 9, 11, 72, 91, 93, 81, 83, 87, 89, 46, 13, 15], # Loop 7
        ]

        # Get loop index and starting lanelet for the given agent
        assert (ref_path_id >= 1) & (ref_path_id <= len(path_to_loop)), f"Reference ID should be in the range [1, {len(path_to_loop)}]"
        loop_index, starting_lanelet = path_to_loop.get(ref_path_id, (None, None))

        if loop_index is not None:
            # Take loop from all defined loops
            reference_lanelets_loop = reference_lanelets_loops[loop_index - 1]  # Adjust for 0-based index
            # Find index of defined starting lanelet
            index_starting_lanelet = reference_lanelets_loop.index(starting_lanelet)
            # Shift loop according to starting lanelet
            lanelets_index = reference_lanelets_loop[index_starting_lanelet:] + reference_lanelets_loop[:index_starting_lanelet]
            return lanelets_index
        else:
            return []  # Return empty list if ref_path_id is not found



if __name__ == "__main__":
    parse_xml = ParseXML(
        map_path="assets/maps/cpm_lab_map.xml",
        device="cpu",
    )
    
    print(parse_xml.map_data)
    print(parse_xml.reference_paths)
