# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from matplotlib.patches import PathPatch
import torch

from sigmarl.map_manager import MapManager
from sigmarl.constants import SCENARIOS

import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt

from sigmarl.helper_scenario import compute_pseudo_tangent_vector

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "serif",
        "text.usetex": True,
    }
)


class PseudoDistance:
    """
    Class for calculating pseudo distance.
    """

    def __init__(self, scenario_type, map: MapManager):
        # Initialize scenario map
        self.scenario_type: str = scenario_type
        self.lane_width = SCENARIOS[scenario_type]["lane_width"]

        # Prepare for distance calculation
        self.initialize_map(map)
        self.prepare_for_calculation()

    def initialize_map(self, map: MapManager):
        self.map = map
        # Initialize map information
        self.lanelets = self.map.parser.lanelets_all
        # Mapping reference path id to loop index and starting lanelet: ref_path_id: (loop_index, starting_lanelet)
        if "cpm" in self.scenario_type.lower():
            self.path_to_loop = {
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 7,
                8: 1,
                9: 2,
                10: 3,
                11: 4,
                12: 5,
                13: 6,
                14: 7,
                15: 1,
                16: 2,
                17: 3,
                18: 4,
                19: 5,
                20: 6,
                21: 7,
                22: 1,
                23: 2,
                24: 3,
                25: 4,
                26: 5,
                27: 6,
                28: 7,
                29: 1,
                30: 2,
                31: 3,
                32: 4,
                33: 5,
                34: 6,
                35: 7,
                36: 1,
                37: 6,
                38: 7,
                39: 1,
                40: 1,
            }
            # In loop 6 or 7, the two adjacent lanelets share the same right boundary in the merging area
            self.lanelets_share_same_right_boundaries_map = {
                5: 23,
                3: 22,
                81: 100,
                83: 101,
                57: 75,
                55: 74,
                29: 48,
                31: 49,
            }
            # List of two adjacent lanelets sharing the same left and right boundaries
            self.lanelets_share_same_boundaries_list = [
                [None, 22],  # The adjacent left lanelet has no solid boundary
                [4, 3],
                [6, 5],
                [None, 23],
                [8, 7],
                [60, 59],
                [58, 57],
                [None, 75],
                [56, 55],
                [None, 74],
                [54, 53],
                [80, 79],
                [82, 81],
                [None, 100],
                [84, 83],
                [None, 101],
                [86, 85],
                [34, 33],
                [32, 31],
                [None, 49],
                [30, 29],
                [None, 48],
                [28, 27],
                [2, 1],
                [13, 14],
                [15, 16],
                [9, 10],
                [11, 12],
                [63, 64],
                [61, 62],
                [67, 68],
                [65, 66],
                [91, 92],
                [93, 94],
                [87, 88],
                [89, 90],
                [37, 38],
                [35, 36],
                [41, 42],
                [39, 40],
                [25, 18],
                [26, 17],
                [52, 43],
                [72, 73],
                [51, 44],
                [50, 45],
                [102, 97],
                [20, 21],
                [103, 96],
                [104, 95],
                [78, 69],
                [46, 47],
                [77, 70],
                [76, 71],
                [24, 19],
                [98, 99],
            ]

    def prepare_for_calculation(self):
        """
        This function is used to prepare the data for pseudo distance calculation.
        """
        # Prepare left and right boundary data for pseudo distance calculation.
        # Initialize lists to store boundary point coordinates and their tangent vectors.
        self.left_boundary = []
        self.right_boundary = []
        self.left_tangent_vector = []
        self.right_tangent_vector = []

        if "cpm" not in self.scenario_type.lower():
            # Assume other scenarios do not have lanelets sharing the same boundaries
            for cur_idx in range(len(self.lanelets)):
                lan = self.lanelets[cur_idx]
                left_boundary = lan["left_boundary"]
                right_boundary = lan["right_boundary"]

                self.left_boundary.append(left_boundary)
                self.right_boundary.append(right_boundary)
                left_tangent_vec = compute_pseudo_tangent_vector(left_boundary)
                self.left_tangent_vector.append(left_tangent_vec)
                right_tangent_vec = compute_pseudo_tangent_vector(right_boundary)
                self.right_tangent_vector.append(right_tangent_vec)
        else:
            # Iterate through all lanelets in the map.
            for cur_idx in range(1, len(self.lanelets) + 1):
                # Find the lanelet group that shares boundaries with the current lanelet.
                lanelets_share_same_boundaries = next(
                    (
                        group
                        for group in self.lanelets_share_same_boundaries_list
                        if cur_idx in group
                    ),
                    None,
                )
                # Store the id of the lanelet whose left or right boundary serves as the left or right boundary of the current lanelet: cur_idx
                left_boundary_id = lanelets_share_same_boundaries[0]
                right_boundary_id = lanelets_share_same_boundaries[1]

                # Retrieve right boundary points
                right_boundary = self.lanelets[right_boundary_id - 1][
                    "right_boundary"
                ]  # Right boundary point coordinates
                predecessor_id = self.lanelets[right_boundary_id - 1]["predecessor"][0]
                successor_id = self.lanelets[right_boundary_id - 1]["successor"][0]

                # Extend the right boundary using its predecessor and successor
                # to improve calculation accuracy near the boundary ends.
                right_boundary = torch.cat(
                    (
                        self.lanelets[predecessor_id - 1]["right_boundary"][-5:-1, :],
                        right_boundary,
                        self.lanelets[successor_id - 1]["right_boundary"][1:5, :],
                    ),
                    dim=0,
                )
                self.right_boundary.append(right_boundary)

                # Compute tangent vectors for the right boundary
                right_tangent_vec = compute_pseudo_tangent_vector(right_boundary)

                self.right_tangent_vector.append(right_tangent_vec)

                if left_boundary_id is not None:
                    # Retrieve left boundary points
                    left_boundary = self.lanelets[left_boundary_id - 1][
                        "left_boundary"
                    ]  # Left boundary point coordinates
                    predecessor_id = self.lanelets[left_boundary_id - 1]["predecessor"][
                        0
                    ]
                    successor_id = self.lanelets[left_boundary_id - 1]["successor"][0]

                    # Extend the left boundary for better accuracy near endpoints
                    left_boundary = torch.cat(
                        (
                            self.lanelets[predecessor_id - 1]["left_boundary"][
                                -5:-1, :
                            ],
                            left_boundary,
                            self.lanelets[successor_id - 1]["left_boundary"][1:5, :],
                        ),
                        dim=0,
                    )
                    self.left_boundary.append(left_boundary)

                    # Compute tangent vectors for the left boundary
                    left_tangent_vec = compute_pseudo_tangent_vector(left_boundary)
                    self.left_tangent_vector.append(left_tangent_vec)
                else:
                    # No solid left boundary exists for this lanelet.
                    self.left_boundary.append(None)
                    self.left_tangent_vector.append(None)

    def transform_from_global_to_line_coordiante(
        self,
        vec: torch.Tensor,  # Shape [#, 1, 2]
        p_b: torch.Tensor,  # Shape [1, #, 2]
        p_t: torch.Tensor,  # Shape [1, #, 2]
        base_transformed: bool,
    ):
        # The x direction of new coordinate system is aligned with the line segment vector p_bt, which is from p_b to p_t.
        p_bt = p_t - p_b
        # Calculate the angle (theta) of the line segment relative to the global x-axis.
        theta = torch.atan2(p_bt[:, :, 1], p_bt[:, :, 0])
        # Create a 2D rotation matrix for aligning the global coordinate system with the line segment.
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        rotation_matrix = torch.stack(
            [
                torch.stack([cos_theta, sin_theta], dim=-1),
                torch.stack([-sin_theta, cos_theta], dim=-1),
            ],
            dim=-2,
        )

        if base_transformed:
            # Tangent vector has already been transformed to the origin of local coordinate system.
            result = torch.matmul(rotation_matrix, vec.unsqueeze(-1))
        else:
            # Translate the input vector (vec) to the base of the line segment.
            vec_b = vec - p_b
            rotation_matrix = rotation_matrix.expand(vec_b.shape[0], -1, -1, -1)
            result = torch.matmul(rotation_matrix, vec_b.unsqueeze(-1))

        return result

    def get_pseudo_distance_to_segment(
        self,
        position: torch.Tensor,
        l: torch.Tensor,
        t_b: torch.Tensor,
        t_t: torch.Tensor,
    ):
        """
        Calculate the pseudo distance of from one point to one line segment.

        Args:
            position (torch.Tensor): Coordinate (x, y) of the position in the local line segment coordiante system.
            l (torch.Tensor): Lengths of the line segments
            t_b (torch.Tensor): Tangent vectors at the base points of segments in local coordinates.
            t_t (torch.Tensor): Tangent vectors at the tip points of segments in local coordinates.

        Returns:
            pseudo_distance, direction, lamda: The pseudo distance of the point to the line segment, direction angle in the local coordinate system, projection point on the line segment.
        """
        # Extracpoint coordinate
        x = position[:, :, 0]
        y = position[:, :, 1]
        l = l.unsqueeze(-1)

        # Translate the tangent vector for calculation simplification
        m_t = torch.where(
            t_t[:, :, 0] != 0,
            t_t[:, :, 1] / t_t[:, :, 0],
            torch.full_like(t_t[:, :, 0], 1e-8),
        )
        m_b = torch.where(
            t_b[:, :, 0] != 0,
            t_b[:, :, 1] / t_b[:, :, 0],
            torch.full_like(t_b[:, :, 0], 1e-8),
        )

        # Find the corresponding point p_lamda on the line segment
        lambda_factor = (x + y * m_b) / (l - y * (m_t - m_b))
        p_lambda = torch.cat(
            [lambda_factor * l, torch.zeros_like(lambda_factor * l)], dim=-1
        )

        # Calcluate the pseuda distance vector from p_lamda to point position
        n_lambda = position.squeeze(-1) - p_lambda
        # Get the magnitude and direction of the pseudo distance vector
        direction = torch.atan2(n_lambda[:, :, 1], n_lambda[:, :, 0])
        pseudo_distance = torch.norm(n_lambda, dim=-1).unsqueeze(-1)

        return (
            pseudo_distance.to(torch.float16),
            direction.to(torch.float16),
            lambda_factor,
        )

    def get_pseudo_distance(
        self,
        tangent_vector: torch.Tensor,
        p_vector: torch.Tensor,
        position: torch.Tensor,
    ):
        """
        Calculate the pseudo distance and direction of a point relative to a polyline.

        The pseudo distance is defined as the shortest pseudo distance from the point to
        the valid projection range of all segments in the polyline (i.e., between adjacent vertices).
        Only projections that lie within a segment (0 <= projection factor < 1) are considered valid.


        Args:
            tangent_vector (torch.Tensor): Tangent vectors along the polyline.
                                        Each element corresponds to a segment of the polyline.
            p_vector (torch.Tensor): Vertices of the polyline represented as a sequence of points.
            position (torch.Tensor): The point for which the pseudo distance is to be calculated.
            id (int, optional): The lanelet ID for determining road direction and adjusting
                                the pseudo distance and direction. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - float: Minimum pseudo distance.
                - float: Direction of the pseudo distance.
        """
        # Add batch dimension
        tangent_vector = tangent_vector.unsqueeze(0)
        p_vector = p_vector.unsqueeze(0)

        # Separate tangent vectors for adjacent segments
        tangent_vector_i = tangent_vector[:, :-1, :]
        tangent_vector_i_plus = tangent_vector[:, 1:, :]

        # Separate position vectors for adjacent segments
        p_vector_i = p_vector[:, :-1, :]
        p_vector_i_plus = p_vector[:, 1:, :]

        # Transform position and tangent vectors into local coordinates
        position_local = self.transform_from_global_to_line_coordiante(
            position, p_vector_i, p_vector_i_plus, False
        )
        tangent_vector_b_local = self.transform_from_global_to_line_coordiante(
            tangent_vector_i, p_vector_i, p_vector_i_plus, True
        )
        tangent_vector_t_local = self.transform_from_global_to_line_coordiante(
            tangent_vector_i_plus, p_vector_i, p_vector_i_plus, True
        )

        # Compute the vector between consecutive polyline vertices
        p_bt = p_vector_i_plus - p_vector_i

        # Compute the length of each polyline segment
        segment_length = torch.norm(p_bt, dim=-1)

        # Calculate the pseudo distance to each segment
        pseudo_distance, direction, proj_factor = self.get_pseudo_distance_to_segment(
            position_local,
            segment_length,
            tangent_vector_b_local,
            tangent_vector_t_local,
        )
        # Apply a mask to ignore segments outside the valid projection range
        large_number = 1000
        mask = (proj_factor[:, :, 0] >= 0) & (proj_factor[:, :, 0] < 1)
        mask = mask.unsqueeze(-1)
        inf_tensor = torch.full_like(pseudo_distance, large_number)

        # Replace invalid distances with a large value
        pseudo_distance = torch.where(mask, pseudo_distance, inf_tensor)
        pseudo_distance = pseudo_distance.squeeze(-1)

        # Get the minimum pseudo distance and its corresponding direction
        pseudo_distance, idx = torch.min(pseudo_distance, dim=-1)
        direction = torch.gather(
            direction, dim=1, index=idx.unsqueeze(-1).expand(-1, direction.shape[1])
        ).squeeze(1)[:, 0]

        return pseudo_distance, direction

    def get_distance(self, ref_id, lanelet_id, pos):
        """
        Compute the pseudo distances from the given position to the left and right
        boundaries of a lanelet.

        Special handling is applied for lanelets in the merging areas of loop 6 and 7,
        where the right boundary is replaced with that of another lanelet to ensure
        a continuous connection between the boundaries of neighboring lanelets.
        Additionally, when no solid left boundary exists, a virtual one is inferred
        using twice the lane width to maintain continuity.

        Args:
            ref_id (Tensor): Reference path ID. Tensor integer.
            lanelet_id (Tensor): Lanelet ID. Tensor integer.
            pos (Tensor or array-like): Position at which the distances are computed. Shape [num_points, 2] in case of Tensor

        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_distance, right_distance) with shape [(num_points,), (num_points,)]
        """
        # Ensure pos is a tensor with the correct shape
        pos = pos if isinstance(pos, torch.Tensor) else torch.tensor(pos)
        pos = pos.unsqueeze(1)

        left_boundary = self.map.parser.reference_paths[ref_id]["left_boundary_shared"]
        right_boundary = self.map.parser.reference_paths[ref_id][
            "right_boundary_shared"
        ]
        left_boundary_pseudo_vector = self.map.parser.reference_paths[ref_id][
            "left_boundary_shared_pseudo_vector"
        ]
        right_boundary_pseudo_vector = self.map.parser.reference_paths[ref_id][
            "right_boundary_shared_pseudo_vector"
        ]

        left_distance, _ = self.get_pseudo_distance(
            left_boundary_pseudo_vector, left_boundary, pos
        )
        right_distance, _ = self.get_pseudo_distance(
            right_boundary_pseudo_vector, right_boundary, pos
        )

        return left_distance.detach().numpy(), right_distance.detach().numpy()

        # TODO: Delete
        # Get the loop id from the reference path id
        loop_id = self.path_to_loop[ref_id.item() + 1]

        # Check whether the lanelet is in the merging area of loop 6 or 7
        if (lanelet_id in self.lanelets_share_same_right_boundaries_map) and (
            loop_id == 6 or loop_id == 7
        ):
            # Use the shared right boundary for the merging area
            right_boundary_id = self.lanelets_share_same_right_boundaries_map[
                lanelet_id
            ]
            right_boundary = self.right_boundary[
                right_boundary_id - 1
            ]  # id starts from 1, but list index starts from 0
            right_boundary_tangent_vector = self.right_tangent_vector[
                right_boundary_id - 1
            ]

            # Compute pseudo distance to the right boundary
            right_distance, _ = self.get_pseudo_distance(
                right_boundary_tangent_vector, right_boundary, pos
            )
            # No solid left boundary in this case
            left_distance = None
        else:
            # Retrieve the left and right boundaries and their tangent vectors
            left_boundary = self.left_boundary[lanelet_id - 1]
            left_tangent_vector = self.left_tangent_vector[lanelet_id - 1]
            right_boundary = self.right_boundary[lanelet_id - 1]
            right_tangent_vector = self.right_tangent_vector[lanelet_id - 1]

            # Compute pseudo distance to the left boundary (if available)
            if left_boundary is not None:
                left_distance, _ = self.get_pseudo_distance(
                    left_tangent_vector, left_boundary, pos
                )
            else:
                left_distance = None

            right_distance, _ = self.get_pseudo_distance(
                right_tangent_vector, right_boundary, pos
            )

        # If no solid left boundary, approximate the left distance using the right one
        if left_distance is None:
            left_distance = 2 * self.lane_width - right_distance

        return left_distance.detach().numpy(), right_distance.detach().numpy()

    def visualize(self, ref_id=0, grid_resolution=0.005):
        """
        Build three colormaps over the full loop road (all lanelets in one):
        1. distance to left boundary
        2. distance to right boundary
        3. final distance = min(left, right)
        """
        # All lanelets that are part of the reference path (including the ones that share the same boundaries)
        (
            self.map.parser._reference_paths_ids[0]
            + self.map.parser._reference_paths_ids[1]
        )

        # Overall bounding box
        min_x, min_y = (
            self.map.parser.bounds["min_x"],
            self.map.parser.bounds["min_y"],
        )
        max_x, max_y = (
            self.map.parser.bounds["max_x"],
            self.map.parser.bounds["max_y"],
        )

        # Sample a uniform grid over the bounding box
        xs = np.arange(min_x, max_x, grid_resolution)
        ys = np.arange(min_y, max_y, grid_resolution)
        X, Y = np.meshgrid(xs, ys)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        # Prepare arrays to hold distances for all grid points
        shape = X.shape
        left_map = np.full(shape, np.nan)
        right_map = np.full(shape, np.nan)
        final_map = np.full(shape, np.nan)

        # For each lanelet, find which grid points lie inside and compute distances
        # for lid in lanelets_id_aux:
        # Get boundary points
        # left_pts = self.map.parser.lanelets_all[lid - 1]["left_boundary"].numpy()
        # right_pts = self.map.parser.lanelets_all[lid - 1]["right_boundary"].numpy()
        # verts = np.vstack([left_pts, right_pts[::-1]])

        left_pts = self.map.parser.reference_paths[ref_id]["left_boundary_shared"]
        right_pts = self.map.parser.reference_paths[ref_id]["right_boundary_shared"]

        verts = np.vstack([left_pts, right_pts.flip(dims=[0])])

        path = Path(verts)

        # Mask grid points inside this lanelet
        inside = path.contains_points(grid_points)
        pts_inside = grid_points[inside]

        # if pts_inside.size == 0:
        #     # This lanelet has no overlap with grid, skip
        #     continue

        pos_tensor = torch.from_numpy(pts_inside).float()

        left_list, right_list = self.get_distance(
            ref_id=torch.tensor(ref_id, dtype=torch.int32),
            lanelet_id=torch.tensor(0, dtype=torch.int32),  # TODO Delete
            pos=pos_tensor,
        )

        left_vals = np.array(left_list)
        right_vals = np.array(right_list)

        # Compute final distance as the minimum of left and right
        final_vals = np.minimum(left_vals, right_vals)

        # Fill into the full maps (flattened indices)
        left_map.ravel()[inside] = left_vals
        right_map.ravel()[inside] = right_vals
        final_map.ravel()[inside] = final_vals

        # Remove outliers from final_map
        valid_data = final_map[~np.isnan(final_map)]
        # Compute IQR statistics
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Build mask of outliers (preserving nans)
        outlier_mask = (final_map < lower_bound) | (final_map > upper_bound)
        if np.any(outlier_mask):
            print(f"Outliers detected: {np.sum(outlier_mask)}")

        # Set outliers to np.nan
        final_map[outlier_mask] = np.nan

        self.map.parser._is_save_fig = False
        self.map.parser._is_plt_show = False
        self.map.parser._is_visualize_intersection = False
        self.map.parser._is_visualize_random_agents = True
        self.map.parser._n_agents_visu = 1
        self.map.parser._is_show_axis = False

        fig, ax = self.map.parser.visualize_map()

        # Visualize the lanelet boundary
        patch = PathPatch(path, facecolor="none", edgecolor="tab:blue", lw=2)
        ax.add_patch(patch)

        vmax = np.nanmax(final_map)
        # mesh = ax.pcolormesh(xs, ys, data, shading='auto', vmin=0, vmax=vmax, cmap='Blues')
        data_masked = np.ma.masked_invalid(final_map)
        mesh = ax.pcolormesh(
            xs, ys, data_masked, shading="auto", vmin=0, vmax=vmax, cmap="Blues"
        )
        # Add colorbar
        cbar = fig.colorbar(mesh, ax=ax, label=r"$d_{\mathrm{pseudo}}$ [m]", shrink=0.6)

        tick_vals = np.linspace(0, vmax, 4)
        cbar.set_ticks(tick_vals)
        cbar.ax.set_yticklabels([f"{v:.2f}" for v in tick_vals])

        mesh.set_rasterized(True)

        fig_name = f"fig_pseudo_distance_{self.scenario_type}_ref_{ref_id}.pdf"
        fig.savefig(fig_name, dpi=450)
        plt.close(fig)

        print(f"Fig has been saved at {fig_name}")


if __name__ == "__main__":
    scenario_type = "interchange_1"  # CPM_entire, interchange_1, intersection_1, on_ramp_1, roundabout_1, etc., see sigmarl/constants.py for more scenario types
    map = MapManager(scenario_type=scenario_type, device="cpu")
    pseudo_distance = PseudoDistance(scenario_type, map)
    pseudo_distance.visualize()
