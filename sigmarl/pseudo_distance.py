# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from sigmarl.map_manager import MapManager
from sigmarl.constants import SCENARIOS

import numpy as np


class PseudoDistance:
    """
    Class for calculating pseudo distance.
    """

    def __init__(self, scenario_type, map: MapManager):
        # Initialize scenario map
        self.scenario_type: str = scenario_type
        self.lane_width = SCENARIOS[scenario_type]["lane_width"]

        # Prepare for distance calculation
        # self.initialize_map(map)
        # self.prepare_for_calculation()
        self.map = map
        if "CPM_mixed" == self.scenario_type:
            # Intersection scenario (TODO: consider actual mixed scenarios consisting of intersection, merge-in, and merge-out)
            self.reference_paths = self.map.parser.reference_paths_intersection
        else:
            # All reference paths
            self.reference_paths = self.map.parser.reference_paths

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

    def get_distance(self, ref_id, pos):
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
            pos (Tensor or array-like): Position at which the distances are computed. Shape [num_points, 2] in case of Tensor

        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_distance, right_distance) with shape [(num_points,), (num_points,)]
        """
        # Ensure pos is a tensor with the correct shape
        pos = pos if isinstance(pos, torch.Tensor) else torch.tensor(pos)
        pos = pos.unsqueeze(1)

        left_boundary = self.reference_paths[ref_id]["left_boundary_shared"]
        right_boundary = self.reference_paths[ref_id]["right_boundary_shared"]
        left_boundary_pseudo_vector = self.reference_paths[ref_id][
            "left_boundary_shared_pseudo_vector"
        ]
        right_boundary_pseudo_vector = self.reference_paths[ref_id][
            "right_boundary_shared_pseudo_vector"
        ]

        left_distance, _ = self.get_pseudo_distance(
            left_boundary_pseudo_vector, left_boundary, pos
        )
        right_distance, _ = self.get_pseudo_distance(
            right_boundary_pseudo_vector, right_boundary, pos
        )

        return left_distance.detach().numpy(), right_distance.detach().numpy()

        # TODO: Delete below (DEBUG)
        plt.plot(left_boundary[:, 0], left_boundary[:, 1])
        plt.plot(right_boundary[:, 0], right_boundary[:, 1])
        plt.scatter(pos[0, 0, 0], pos[1, 0, 1])

    def visualize(self, ref_id=0, grid_resolution=0.005):
        """
        Build three colormaps over the full loop road (all lanelets in one):
        1. distance to left boundary
        2. distance to right boundary
        3. final distance = min(left, right)
        """
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        import matplotlib.pyplot as plt
        from sigmarl.helper_common import is_latex_available

        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.labelsize": 9,
                "axes.titlesize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "font.family": "serif",
                "text.usetex": is_latex_available(),
            }
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

        # For pseudo_distance_example only
        if self.scenario_type == "pseudo_distance_example":
            left_pts = self.reference_paths[0]["center_line"]
            right_pts = self.reference_paths[1]["center_line"]
            grid_resolution = 0.1
        else:
            left_pts = self.reference_paths[ref_id]["left_boundary_shared"]
            right_pts = self.reference_paths[ref_id]["right_boundary_shared"]

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

        ax.autoscale()

        fig_name = f"fig_pseudo_distance_{self.scenario_type}_ref_{ref_id}.pdf"
        fig.savefig(fig_name, dpi=450)
        plt.close(fig)

        print(f"Fig has been saved at {fig_name}")


if __name__ == "__main__":
    scenario_types = [
        "CPM_entire",
        # "interchange_1",
        # "interchange_2",
        # "interchange_3",
        # "intersection_1",
        # "intersection_2",
        # "intersection_3",
        # "intersection_4",
        # "intersection_5",
        # "intersection_6",
        # "intersection_7",
        # "intersection_8",
        # "on_ramp_1",
        # "on_ramp_2_multilane",
        # "roundabout_1",
        # "roundabout_2",
        # "pseudo_distance_example",
    ]  # See sigmarl/constants.py for all available maps

    for scenario_type in scenario_types:
        map = MapManager(scenario_type=scenario_type, device="cpu", lane_width=0.3)
        pseudo_distance = PseudoDistance(scenario_type, map)

        for ref_id in range(len(map.parser._reference_paths_ids)):
            pseudo_distance.visualize(ref_id=ref_id, grid_resolution=0.005)
