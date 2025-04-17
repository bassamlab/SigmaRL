import torch

from sigmarl.map_manager import MapManager
from sigmarl.constants import SCENARIOS


class PseudoDistance:
    """
    Class for calculating pseudo distance.
    """

    def __init__(self, scenario_type, map: MapManager):
        # Initialize scenario map
        self.scenario_type = scenario_type
        self.lane_width = SCENARIOS[scenario_type]["lane_width"]
        self.map = map

        # Prepare for distance calculation
        self.initialize_map()
        self.prepare_for_calculation()

    def initialize_map(self):
        # Initialize map information
        self.lanelets = self.map.parser.lanelets_all
        # Mapping reference path id to loop index and starting lanelet: ref_path_id: (loop_index, starting_lanelet)
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
        # Prepare left and right boundary data for pseudo distance calculation.
        # Initialize lists to store boundary point coordinates and their tangent vectors.
        self.left_boundary = []
        self.right_boundary = []
        self.left_tangent_vector = []
        self.right_tangent_vector = []

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
            right_tangent_vec = self.compute_tangent_vector(right_boundary)
            self.right_tangent_vector.append(right_tangent_vec)

            if left_boundary_id is not None:
                # Retrieve left boundary points
                left_boundary = self.lanelets[left_boundary_id - 1][
                    "left_boundary"
                ]  # Left boundary point coordinates
                predecessor_id = self.lanelets[left_boundary_id - 1]["predecessor"][0]
                successor_id = self.lanelets[left_boundary_id - 1]["successor"][0]

                # Extend the left boundary for better accuracy near endpoints
                left_boundary = torch.cat(
                    (
                        self.lanelets[predecessor_id - 1]["left_boundary"][-5:-1, :],
                        left_boundary,
                        self.lanelets[successor_id - 1]["left_boundary"][1:5, :],
                    ),
                    dim=0,
                )
                self.left_boundary.append(left_boundary)

                # Compute tangent vectors for the left boundary
                left_tangent_vec = self.compute_tangent_vector(left_boundary)
                self.left_tangent_vector.append(left_tangent_vec)
            else:
                # No solid left boundary exists for this lanelet.
                self.left_boundary.append(None)
                self.left_tangent_vector.append(None)

    def compute_tangent_vector(self, points: torch.Tensor):
        """
        Compute the approximated tangent vector for each point of the polyline.

        Args:
            points (torch.Tensor): A tensor containing the coordinates of the polyline points.
            Shape: (N, 2) where N is the number of points, and each point is a 2D coordinate (x, y).

        Returns:
            tangent_vector(torch.Tensor): A tensor containing the approximated tangent vector for each point of the polyline.
        """
        # Compute the tangent vector of all points along the polyline
        tangent_vector = torch.zeros_like(points)
        # The first and last tangent vector are approximated with the position vectors of the nearest two points
        tangent_vector[0] = points[1] - points[0]
        tangent_vector[-1] = points[-1] - points[-2]
        # The other tangent vectors are approxomated with the position vectors of the previous and next point
        if points.size(0) >= 3:
            for i in range(1, points.size(0) - 1):
                tangent_vector[i] = points[i + 1] - points[i - 1]

        return tangent_vector

    def transform_from_global_to_line_coordiante(
        self,
        vec: torch.Tensor,
        p_b: torch.Tensor,
        p_t: torch.Tensor,
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
            ref_id (Tensor): Reference path ID.
            lanelet_id (Tensor): Lanelet ID.
            pos (Tensor or array-like): Position at which the distances are computed.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_distance, right_distance)
        """
        # Ensure pos is a tensor with the correct shape
        pos = pos if isinstance(pos, torch.Tensor) else torch.tensor(pos)
        pos = pos.unsqueeze(1)

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


if __name__ == "__main__":
    map = MapManager(_scenario_type="CPM_entire", _device="cpu")
    pseudo_distance = PseudoDistance("CPM_entire", map)
