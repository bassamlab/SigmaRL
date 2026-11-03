# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Variants of the backup-horizon rows (condition baselines and ablations).

All variants keep the tails, linearization, exact check, repair, and fallback
of ours (sigmarl/backup_horizon_cbf.py) and change one ingredient:

- ``bh_half_split`` (ablation of the decentralized split): our margins, but a
  cross pair is split as in Chen, Singletary, Ames (L-CSS 2021): each vehicle
  takes half of the decay of the pair's backup value, with the other vehicle
  on its backup tail,
    m_e(Psi_t(u_i), Psi^b_{j,t}) >= (1 - lambda / 2) B_e(X_k),  t = 1..N.
  The two halves add up to the joint decay only if the changes of the pair
  margin are additive (the chain rule in continuous time); over a finite step,
  the mixed term remains, so these rows carry no guarantee, unlike our corner
  budgets.
- ``b1_bh`` and ``b3_bh`` (approximated geometry with the backup-horizon
  condition): the margins of B1 (log-sum-exp smoothed separating-axis
  barriers of the lane and the pair) or of B3 (center distance of the
  circumscribed disks, without B3's braking term, which the braking tail
  replaces; lane rows ours) evaluated along the same tails, with the half split
  of the baselines' papers in the decentralized mode.
- ``bh_capsule`` (ablation of the geometry): ours with every vehicle covered
  by a capsule (the hull of three disks), for the lane and the pair margins,
  and the corner budgets applied to the capsule axes.
- ``mps`` (model predictive shielding, Bastani 2021; Zhang et al., MAMPS): no
  QP. The nominal control is kept unless the exact check of the rows with
  lambda = 1 fails, i.e., unless the backup after it leaves a margin negative;
  then the vehicles of the violated constraints switch to their backup
  controls until the check passes (BackupHorizonCBF.fallback). In the
  decentralized mode, the rows with lambda = 1 are our corner budgets, so that
  the test stays valid without exchanging controls (MAMPS is centralized).
"""

import math

import numpy as np

from sigmarl.backup_horizon_cbf import BackupHorizonCBF
from sigmarl.cbf_baselines.b1_lse_sat import (
    lse_lane_padded_values,
    lse_pair_values,
    resolve_kappa,
)


class BackupHorizonHalfSplit(BackupHorizonCBF):
    """Backup-horizon rows whose cross pairs take half of the decay of the
    pair's backup value (bh_half_split)."""

    KINDS = BackupHorizonCBF.KINDS + ("sdpair",)
    # Share of the decay of a cross pair carried by each vehicle
    CROSS_SHARE = 0.5

    def _build_cross_entries(self, block, ie, io, cross):
        """Rows of the ego on the pair margin to the other vehicle's backup
        tail, with half of the decay of the pair's backup value."""
        ref = block["ref"]
        n_t = ref.shape[0]
        lam = self.decay_rate(np.array([float(c[2]["lam"]) for c in cross]))
        margin = self.pair_values(
            ref[:, ie].reshape(-1, 3), ref[:, io].reshape(-1, 3)
        ).reshape(n_t, -1)
        backup_value = margin.min(axis=0)
        rhs = np.repeat(
            ((1.0 - self.CROSS_SHARE * lam) * backup_value)[:, None], n_t, axis=1
        )
        self._add_entries(
            block,
            "sdpair",
            ie[:, None],
            [c[2]["slack_idx"] for c in cross],
            rhs,
            ego=ie,
            other_ref=ref[:, io],
        )
        self.stats["bh_pairs_unrecoverable"] += int((backup_value < 0.0).sum())

    def _kind_values(self, block, poses, kind: str, sub, variant_a=0, variant_b=0):
        """Values [len(sub), T]; "sdpair" is the pair margin of the ego's tail
        and the other vehicle's backup tail."""
        if kind != "sdpair":
            return super()._kind_values(block, poses, kind, sub, variant_a, variant_b)
        data = block[kind]
        ego = poses[:, data["ego"][sub], variant_a]
        values = self.pair_values(
            ego.reshape(-1, 3), data["other_ref"][:, sub].reshape(-1, 3)
        )
        return values.reshape(ego.shape[0], -1).T


class LseSatBackupHorizon(BackupHorizonHalfSplit):
    """B1's smoothed lane and pair barriers on the backup-horizon rows (b1_bh)."""

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.kappa = resolve_kappa(ctrl.parameters, "b1_bh")

    def pair_values(self, poses_a, poses_b, half_a=None, half_b=None) -> np.ndarray:
        assert half_a is None, "b1_bh supports only the nominal footprint."
        return lse_pair_values(self.ctrl.pair_sat, poses_a, poses_b, self.kappa)

    def _lane_values(self, lane: dict, poses: np.ndarray, sub=None) -> np.ndarray:
        n_t, n_lanes = poses.shape[0], poses.shape[1]
        boundary = np.arange(lane["agent"].shape[0]) if sub is None else sub
        boundary = np.tile(boundary, n_t)
        values = lse_lane_padded_values(
            self.ctrl.lane_sat,
            {key: value[boundary] for key, value in lane["facets"].items()},
            lane["valid"][boundary],
            lane["side"][boundary],
            poses.reshape(-1, 3),
            self.kappa,
        )
        return values.reshape(n_t, n_lanes)

    def coupling_ranges(self, speeds, agent, others) -> np.ndarray:
        """Ours plus the largest underestimate of the smoothed pair barrier
        (ln 4 / kappa per soft minimum over corners, ln 8 / kappa for the soft
        maximum over directions)."""
        return (
            super().coupling_ranges(speeds, agent, others) + math.log(32.0) / self.kappa
        )


class DiskBackupHorizon(BackupHorizonHalfSplit):
    """B3's circumscribed disks as the pair margin of the backup-horizon rows
    (b3_bh); the lane rows are ours."""

    def __init__(self, ctrl):
        super().__init__(ctrl)
        radius = 0.5 * math.hypot(float(ctrl.length), float(ctrl.width))
        self.clearance = 2.0 * radius + float(ctrl.pair_safety_buffer)

    def pair_values(self, poses_a, poses_b, half_a=None, half_b=None) -> np.ndarray:
        assert half_a is None, "b3_bh supports only the nominal footprint."
        return (
            np.linalg.norm(poses_a[:, 0:2] - poses_b[:, 0:2], axis=1) - self.clearance
        )


def segment_endpoints(poses: np.ndarray, half_axis: float) -> np.ndarray:
    """End points [..., 2, 2] of the segments of half length half_axis along
    the heading at poses [..., 3]."""
    e_lon = np.stack([np.cos(poses[..., 2]), np.sin(poses[..., 2])], axis=-1)
    offset = half_axis * e_lon[..., None, :] * np.array([[1.0], [-1.0]])
    return poses[..., None, 0:2] + offset


def point_segment_vectors(points: np.ndarray, start: np.ndarray, end: np.ndarray):
    """Vectors [K, 2] from the closest points of segments [K, 2] to points [K, 2]."""
    segment = end - start
    tau = np.clip(
        np.sum((points - start) * segment, axis=1)
        / np.maximum(np.sum(segment * segment, axis=1), 1e-24),
        0.0,
        1.0,
    )
    return points - (start + tau[:, None] * segment)


def segment_distance(poses_a: np.ndarray, poses_b: np.ndarray, half_axis: float):
    """Distance [K] of the axis segments of pairs at poses [K, 3] and the unit
    direction [K, 2] from the closest point of b to that of a (the center
    offset for crossing segments, whose distance is zero)."""
    ends_a = segment_endpoints(poses_a, half_axis)
    ends_b = segment_endpoints(poses_b, half_axis)
    vectors = np.stack(
        [
            point_segment_vectors(ends_a[:, 0], ends_b[:, 0], ends_b[:, 1]),
            point_segment_vectors(ends_a[:, 1], ends_b[:, 0], ends_b[:, 1]),
            -point_segment_vectors(ends_b[:, 0], ends_a[:, 0], ends_a[:, 1]),
            -point_segment_vectors(ends_b[:, 1], ends_a[:, 0], ends_a[:, 1]),
        ],
        axis=1,
    )  # [K, 4, 2], from b to a
    lengths = np.linalg.norm(vectors, axis=2)
    closest = lengths.argmin(axis=1)
    k = np.arange(poses_a.shape[0])
    distance, vector = lengths[k, closest], vectors[k, closest]

    def cross(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    # Proper crossing of the two segments (distance zero)
    d_a = ends_a[:, 1] - ends_a[:, 0]
    d_b = ends_b[:, 1] - ends_b[:, 0]
    crossing = (
        cross(d_a, ends_b[:, 0] - ends_a[:, 0]) * cross(d_a, ends_b[:, 1] - ends_a[:, 0])
        < 0.0
    ) & (
        cross(d_b, ends_a[:, 0] - ends_b[:, 0]) * cross(d_b, ends_a[:, 1] - ends_b[:, 0])
        < 0.0
    )
    distance = np.where(crossing, 0.0, distance)
    vector = np.where(
        (crossing | (distance < 1e-12))[:, None], poses_a[:, 0:2] - poses_b[:, 0:2], vector
    )
    direction = vector / np.maximum(np.linalg.norm(vector, axis=1), 1e-12)[:, None]
    return distance, direction


class CapsuleBudgets(BackupHorizonCBF):
    """Ablation of the geometry (bh_capsule): every vehicle is the capsule
    that covers its rectangle (the convex hull of the standard cover by three
    disks: a segment of half length 2 l_lon / 3 along the heading, inflated by
    the radius (l_lon^2 / 9 + l_lat^2)^(1/2)), for the lane and the pair
    margins, with everything else ours, including the corner budgets of the
    decentralized mode. A single circumscribed disk is wider than a lane.

    The pair margin is the distance of the two segments minus both radii and
    the pair buffer. The lane margin is our lane barrier of the segment (a
    rectangle of zero width) minus the radius. The capsules are convex, so the
    budget rows bound the shift of the ego's closest segment end point along
    the witness, the direction between the closest points of the two braking
    tails (the radius cancels in the shift), and the budgets of both vehicles
    imply the pair condition as for rectangles.
    """

    def __init__(self, ctrl):
        super().__init__(ctrl)
        half_length, half_width = self.half
        self.half_axis = 2.0 * half_length / 3.0
        self.radius = float(np.hypot(half_length / 3.0, half_width))

    def pair_values(self, poses_a, poses_b, half_a=None, half_b=None) -> np.ndarray:
        assert half_a is None, "bh_capsule supports only the nominal footprint."
        distance, _ = segment_distance(poses_a, poses_b, self.half_axis)
        return distance - 2.0 * self.radius - float(self.ctrl.pair_safety_buffer)

    def pair_distance_witness(self, poses_a, poses_b, half, is_a_first, half_b=None):
        assert half_b is None, "bh_capsule supports only the nominal footprint."
        first = np.where(is_a_first[:, None], poses_a, poses_b)
        second = np.where(is_a_first[:, None], poses_b, poses_a)
        distance, witness = segment_distance(first, second, self.half_axis)
        return (
            distance - 2.0 * self.radius,
            np.where(is_a_first[:, None], witness, -witness),
        )

    def support_points(self, poses, half) -> np.ndarray:
        """The segment end points (the capsule's support is their smallest
        projection minus the radius)."""
        return segment_endpoints(poses, self.half_axis)

    def coupling_ranges(self, speeds, agent, others) -> np.ndarray:
        """Ours with the capsules' circumradius instead of the rectangles'."""
        return (
            super().coupling_ranges(speeds, agent, others)
            + 2.0 * (self.half_axis + self.radius)
            - self.diagonal
        )

    def _lane_values(self, lane: dict, poses: np.ndarray, sub=None) -> np.ndarray:
        assert lane["half"] is None, "bh_capsule supports only the nominal footprint."
        n_t, n_lanes = poses.shape[0], poses.shape[1]
        boundary = np.arange(lane["agent"].shape[0]) if sub is None else sub
        boundary = np.tile(boundary, n_t)
        segment = np.tile([self.half_axis, 0.0], (boundary.shape[0], 1))
        values = self.ctrl.lane_sat.padded_values(
            {key: value[boundary] for key, value in lane["facets"].items()},
            lane["valid"][boundary],
            lane["side"][boundary],
            poses.reshape(-1, 3),
            segment,
        )
        return values.reshape(n_t, n_lanes) - self.radius


class ModelPredictiveShield(BackupHorizonCBF):
    """Model predictive shielding with our backup and margins (mps)."""

    IS_ACCEPT_OR_REPLACE = True

    def decay_rate(self, lam):
        """lambda = 1: the backup after the control keeps every margin
        nonnegative."""
        return np.ones_like(np.asarray(lam, dtype=np.float64))
