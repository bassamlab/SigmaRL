# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B1: log-sum-exp smoothed SAT barriers (Wu et al., T-Cyb 2025, with the
Boolean composition of Molnar & Ames, 2023).

Wu et al. smooth the polygon-polygon SDF lower bound max_direction min_corner
(the rectangle_sat pair barrier) by log-sum-exp, and subtract ln(n)/kappa at
every soft maximum, so that the smoothed set is an inner approximation of the
safe set. The lane version composes the rectangle_sat lane atoms in the same
way (a boundary segment as a degenerate polygon, which their Assumption 1
excludes): softmin over facets of softmax over SAT axes of softmin over atoms.
The smooth barrier gets one row per side or per pair, linearized once
(barrier swap, see common.py); the continuous-time condition of the paper
does not apply to the acceleration and steering-rate inputs (relative
degree 2).

Checked against the papers: the 8 directions and 32 atoms are Wu et al.'s
Eqs. 29-30 for two rectangles; nesting soft max over soft mins with one kappa
and subtracting ln(number of directions) / kappa is their Eq. 33 with the
smallest offset of Lemma 3 (b = ln 8), and for the lane Molnar & Ames'
Theorem 5 (b = ln 4 for the OR over 4 axes).
"""

import numpy as np

from sigmarl.cbf_baselines import tuned_env_value
from sigmarl.cbf_baselines.common import (
    LaneBaseline,
    PairBaseline,
    softmax_lower,
    softmin,
)
from sigmarl.constants import LENGTH_SCALE

# Smoothing parameter kappa [1/m], given at lab scale as kappa * LENGTH_SCALE
# (100 is 5 /m at full scale, the value of Wu et al.). The tuned value per block
# and mode is in tuned_defaults.json; the environment variable overrides it.
KAPPA_VARIABLE = "SIGMARL_B1_LSE_KAPPA"


def resolve_kappa(parameters, method: str = "b1_lse_sat") -> float:
    """Smoothing parameter kappa [1/m] of a method (b1_lse_sat or b1_bh) for
    the active block and mode."""
    return float(tuned_env_value(parameters, method, KAPPA_VARIABLE)) / LENGTH_SCALE


def lane_atoms_and_gradients(lane_sat, facets: dict, side: int, pose) -> tuple:
    """Lane SAT atoms [F, 4 axes, 4] (+inf padded) and their pose gradients
    [F, 4, 4, 3] of one boundary (see LaneSatBarrier.atom_tables)."""
    n_facets = facets["start"].shape[0]
    cos_psi, sin_psi = np.cos(pose[2]), np.sin(pose[2])
    e_lon = np.array([cos_psi, sin_psi])
    e_lat = np.array([-sin_psi, cos_psi])
    position = np.asarray(pose[0:2], dtype=np.float64)
    offsets = (
        lane_sat.corners[:, 0:1] * e_lon + lane_sat.corners[:, 1:2] * e_lat
    )  # [4, 2]
    turned = np.stack([-offsets[:, 1], offsets[:, 0]], axis=1)  # d offset / d psi

    atoms = np.full((n_facets, 4, 4), np.inf)
    grads = np.zeros((n_facets, 4, 4, 3))
    normal = facets["normal"]
    atoms[:, 0, :] = normal @ (position + offsets).T - facets["offset"][:, None]
    grads[:, 0, :, 0:2] = normal[:, None, :]
    grads[:, 0, :, 2] = normal @ turned.T

    rel = np.stack([facets["start"], facets["end"]], axis=1) - position  # [F, 2, 2]
    lon = rel @ e_lon
    lat_sign = 1.0 if side == 0 else -1.0
    buffer = facets["buffer"][:, None]
    atoms[:, 1, 0:2] = lon - lane_sat.half_length - buffer
    grads[:, 1, 0:2, 0:2] = -e_lon
    grads[:, 1, 0:2, 2] = rel @ e_lat
    atoms[:, 2, 0:2] = -lon - lane_sat.half_length - buffer
    grads[:, 2, 0:2, 0:2] = e_lon
    grads[:, 2, 0:2, 2] = -(rel @ e_lat)
    atoms[:, 3, 0:2] = lat_sign * (rel @ e_lat) - lane_sat.half_width - buffer
    grads[:, 3, 0:2, 0:2] = -lat_sign * e_lat
    grads[:, 3, 0:2, 2] = -lat_sign * (rel @ e_lon)
    return atoms, grads


def lse_lane_value_and_gradient(lane_sat, facets, side, pose, kappa):
    """Smoothed lane barrier of one boundary at a pose and its pose gradient."""
    atoms, grads = lane_atoms_and_gradients(lane_sat, facets, side, pose)
    axis_values, w_atoms = softmin(atoms, kappa)  # [F, 4]
    facet_values, w_axes = softmax_lower(axis_values, kappa, 4)  # [F]
    value, w_facets = softmin(facet_values, kappa)
    weights = w_facets[:, None, None] * w_axes[:, :, None] * w_atoms  # [F, 4, 4]
    return float(value), np.einsum("fam,famk->k", weights, grads)


def lse_lane_padded_values(lane_sat, facets: dict, valid, side, poses, kappa):
    """Smoothed lane barriers [S] of S boundaries at one pose each [S, 3]
    (vectorized lse_lane_value_and_gradient without the gradient).

    Args:
        facets: zero-padded facets [S, F, ...] (see LaneSatBarrier.padded_values).
        valid: [S, F] mask of the real facets (every boundary has one).
        side: side [S] of each boundary (0 left, 1 right).
    """
    cos_psi, sin_psi = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    e_lon = np.stack([cos_psi, sin_psi], axis=1)  # [S, 2]
    e_lat = np.stack([-sin_psi, cos_psi], axis=1)
    position = poses[:, None, 0:2]  # [S, 1, 2]
    corners = position + np.matmul(
        lane_sat.corners, np.stack([e_lon, e_lat], axis=1)
    )  # [S, 4, 2]
    n_sides, n_facets = valid.shape
    atoms = np.full((n_sides, n_facets, 4, 4), np.inf)
    atoms[:, :, 0, :] = (
        np.matmul(facets["normal"], corners.transpose(0, 2, 1))
        - facets["offset"][:, :, None]
    )
    rel = np.stack([facets["start"], facets["end"]], axis=2) - position[:, :, None]
    lon = np.einsum("sfkd,sd->sfk", rel, e_lon)  # [S, F, 2]
    lat = np.where((side == 0)[:, None, None], 1.0, -1.0) * np.einsum(
        "sfkd,sd->sfk", rel, e_lat
    )
    buffer = facets["buffer"][:, :, None]
    atoms[:, :, 1, 0:2] = lon - lane_sat.half_length - buffer
    atoms[:, :, 2, 0:2] = -lon - lane_sat.half_length - buffer
    atoms[:, :, 3, 0:2] = lat - lane_sat.half_width - buffer
    axis_values, _ = softmin(atoms, kappa)  # [S, F, 4]
    facet_values, _ = softmax_lower(axis_values, kappa, 4)  # [S, F]
    values, _ = softmin(np.where(valid, facet_values, np.inf), kappa)
    return values


def lse_pair_values(pair_sat, poses_a, poses_b, kappa):
    """Smoothed pair barriers [K] of pose pairs [K, 3] (vectorized
    lse_pair_value_and_gradients without the gradients)."""
    atoms = pair_sat.atoms(poses_a, poses_b)  # [K, 8, 4]
    direction_values, _ = softmin(atoms, kappa)
    values, _ = softmax_lower(direction_values, kappa, atoms.shape[1])
    return values


def lse_pair_value_and_gradients(pair_sat, pose_a, pose_b, kappa):
    """Smoothed pair barrier of two poses and its gradients w.r.t. both."""
    atoms = pair_sat.atoms(np.asarray(pose_a)[None], np.asarray(pose_b)[None])[0]
    direction_values, w_corners = softmin(atoms, kappa)  # [8]
    value, w_directions = softmax_lower(direction_values, kappa, atoms.shape[0])
    grad_a, grad_b = np.zeros(3), np.zeros(3)
    for direction in range(atoms.shape[0]):
        g_a, g_b = pair_sat.atom_gradients(pose_a, pose_b, direction)
        weights = w_directions[direction] * w_corners[direction]  # [4]
        grad_a += weights @ g_a
        grad_b += weights @ g_b
    return float(value), grad_a, grad_b


class LseSatLaneBaseline(LaneBaseline):
    """B1 lane rows: one smoothed barrier per side."""

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.kappa = resolve_kappa(ctrl.parameters)

    def side_constraints(self, env_idx, agent_idx, pose_y, pose_z):
        lane_sat = self.ctrl.lane_sat
        boundaries = lane_sat.local_boundaries(
            self.ctrl.scenario.world_state, env_idx, agent_idx
        )
        out = []
        for side, points in enumerate(boundaries):
            facets = lane_sat.facets(points, side)
            if facets["start"].shape[0] == 0:
                continue
            h_y, _ = lse_lane_value_and_gradient(
                lane_sat, facets, side, pose_y, self.kappa
            )
            h_z, grad = lse_lane_value_and_gradient(
                lane_sat, facets, side, pose_z, self.kappa
            )
            out.append((side, h_y, [(h_z, grad)]))
        return out


class LseSatPairBaseline(PairBaseline):
    """B1 pair rows: one smoothed barrier per pair."""

    # Wu et al. split the decay between the two agents equally (Eq. 34b)
    CROSS_SHARE = 0.5

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.kappa = resolve_kappa(ctrl.parameters)

    def value(self, x_a, x_b) -> float:
        return self.value_and_gradients(x_a, x_b)[0]

    def value_and_gradients(self, x_a, x_b) -> tuple:
        return lse_pair_value_and_gradients(self.ctrl.pair_sat, x_a, x_b, self.kappa)
