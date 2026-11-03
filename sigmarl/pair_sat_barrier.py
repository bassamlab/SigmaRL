# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Exact rectangle-rectangle vehicle barrier (veh_veh_cbf_footprint="rectangle_sat").

By the separating axis theorem, two rectangles are disjoint iff their
projections onto one of the four edge normals are disjoint. Along the axis e
of the owner rectangle o (pose p_o) with sign s, the gap of the other
rectangle is a min over its corners q_c of the smooth atoms
  s e . (q_c - p_o) - half_o(e) - buffer,
where half_o(e) is the owner's half length or half width. With 8 directions
(owner a or b, longitudinal or lateral axis, sign + or -), the barrier
h = max_direction min_corners atom is >= 0 iff the rectangles keep the buffer
along some edge normal, which implies that they do not intersect. h equals
the backup shield's pair gap minus the buffer.

The geometry is used by the CBF-QP (``sigmarl/cbf_qp_proxsuite.py``), which
constrains the atoms of one direction per pair.
"""

import numpy as np


class PairSatBarrier:
    """Rectangle-rectangle barrier of two vehicles with the same footprint.

    Direction d = 4 owner + 2 axis + sign_index, with owner 0 (a) or 1 (b),
    axis 0 (longitudinal) or 1 (lateral), and sign_index 0 (+) or 1 (-).

    Args:
        length: vehicle length [m].
        width: vehicle width [m].
        safety_buffer: clearance the rectangles must keep [m].
    """

    N_DIRECTIONS = 8

    def __init__(self, length: float, width: float, safety_buffer: float):
        self.half = np.array([0.5 * float(length), 0.5 * float(width)])
        self.corners = (
            np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]) * self.half
        )  # [4, 2] in the vehicle frame
        self.safety_buffer = float(safety_buffer)

    @staticmethod
    def axes(poses: np.ndarray) -> np.ndarray:
        """Unit axes (longitudinal, lateral) [P, 2, 2] of poses [P, 3]."""
        cos_psi, sin_psi = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        return np.stack(
            [np.stack([cos_psi, sin_psi], -1), np.stack([-sin_psi, cos_psi], -1)], 1
        )

    def corner_offsets(self, poses: np.ndarray) -> np.ndarray:
        """Corners [P, 4, 2] of poses [P, 3] relative to their centers."""
        return np.einsum("ck,pkd->pcd", self.corners, self.axes(poses))

    def _owner_atoms(self, owner: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Atoms [P, 4 directions, 4 corners] along the axes of the owner."""
        rel = other[:, None, 0:2] + self.corner_offsets(other) - owner[:, None, 0:2]
        proj = np.einsum("pad,pcd->pac", self.axes(owner), rel)  # [P, 2, 4]
        atoms = np.stack([proj, -proj], axis=2)  # [P, axis, sign, 4]
        atoms -= self.half[None, :, None, None] + self.safety_buffer
        return atoms.reshape(-1, 4, 4)

    def atoms(self, poses_a: np.ndarray, poses_b: np.ndarray) -> np.ndarray:
        """Atom values [P, 8, 4] of pairs of poses (x, y, psi) [P, 3] [m]."""
        return np.concatenate(
            [self._owner_atoms(poses_a, poses_b), self._owner_atoms(poses_b, poses_a)],
            axis=1,
        )

    def values(self, poses_a: np.ndarray, poses_b: np.ndarray) -> np.ndarray:
        """Barrier values [P] (max over directions of min over corners) [m]."""
        return self.atoms(poses_a, poses_b).min(axis=2).max(axis=1)

    def atom_gradients(self, pose_a, pose_b, direction: int) -> tuple:
        """Gradients of the 4 atoms of one direction with respect to both poses.

        Returns:
            d atoms / d pose_a [4, 3] and d atoms / d pose_b [4, 3].
        """
        poses = np.asarray([pose_a, pose_b], dtype=np.float64)
        owner_idx, axis, sign_idx = direction // 4, (direction // 2) % 2, direction % 2
        owner, other = poses[owner_idx], poses[1 - owner_idx]
        sign = 1.0 if sign_idx == 0 else -1.0
        frames = self.axes(poses[[owner_idx]])[0]
        e = sign * frames[axis]
        # d e / d psi: the longitudinal axis turns into the lateral one, the
        # lateral one into minus the longitudinal one
        de = sign * (frames[1] if axis == 0 else -frames[0])
        offsets = self.corner_offsets(other[None])[0]  # [4, 2]
        rel = other[0:2] + offsets - owner[0:2]
        grad_owner = np.column_stack([np.tile(-e, (4, 1)), rel @ de])
        # d corner / d psi of the other rectangle: its offset turned by 90 deg
        turned = np.column_stack([-offsets[:, 1], offsets[:, 0]])
        grad_other = np.column_stack([np.tile(e, (4, 1)), turned @ e])
        if owner_idx == 0:
            return grad_owner, grad_other
        return grad_other, grad_owner
