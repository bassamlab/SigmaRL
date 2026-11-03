# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B5: convex free-space containment lane barrier (polytope agent in a convex
polytope, Molnar et al., CCTA 2025; also Gonzalez-Garcia et al., 2026).

The free space around the vehicle is the convex polygon cut out by the
half-planes {q : n_j . (q - a_j) >= buffer_j} of the boundary facets j near
the vehicle (within CONVEX_RADIUS of its center at k+1). The rectangle lies in
this polygon iff every corner m satisfies every half-plane, so
  h = min_j min_m n_j . (c_m - a_j) - buffer_j,
which is the normal-axis part of the rectangle_sat lane atoms without the SAT
maximum over axes. On curved, non-convex boundaries, a half-plane of a facet
beside the vehicle can cut the lane in front of it, which is the weakness this
baseline exposes.
"""

import numpy as np

from sigmarl.cbf_baselines.common import LaneBaseline
from sigmarl.constants import LENGTH_SCALE


class ConvexContainmentLaneBaseline(LaneBaseline):
    """B5 lane rows: one row per nearly active (facet, corner) atom."""

    # Facets within this distance [m] of the vehicle center (one vehicle
    # length) form the local convex free space
    CONVEX_RADIUS_FACTOR = 1.0
    # Atoms whose margin at the linearization control exceeds this band [m]
    # get no row (the constraint changes by centimeters over the input box)
    ROW_BAND = 0.1 * LENGTH_SCALE

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.radius = self.CONVEX_RADIUS_FACTOR * float(ctrl.length)

    def atoms_and_gradients(self, facets: dict, pose) -> tuple:
        """Corner atoms [F, 4] of the facets' half-planes and gradients [F, 4, 3]."""
        lane_sat = self.ctrl.lane_sat
        cos_psi, sin_psi = np.cos(pose[2]), np.sin(pose[2])
        e_lon = np.array([cos_psi, sin_psi])
        e_lat = np.array([-sin_psi, cos_psi])
        offsets = lane_sat.corners[:, 0:1] * e_lon + lane_sat.corners[:, 1:2] * e_lat
        turned = np.stack([-offsets[:, 1], offsets[:, 0]], axis=1)
        normal = facets["normal"]
        atoms = normal @ (np.asarray(pose[0:2]) + offsets).T - facets["offset"][:, None]
        grads = np.zeros(atoms.shape + (3,))
        grads[..., 0:2] = normal[:, None, :]
        grads[..., 2] = normal @ turned.T
        return atoms, grads

    def local_facets(self, facets: dict, center) -> dict:
        """Facets whose segment is within the convex radius of a point."""
        start, end = facets["start"], facets["end"]
        seg = end - start
        t = np.clip(
            np.sum((center - start) * seg, axis=1) / np.sum(seg * seg, axis=1), 0, 1
        )
        dist = np.linalg.norm(start + t[:, None] * seg - center, axis=1)
        keep = dist <= self.radius
        return {name: values[keep] for name, values in facets.items()}

    def side_constraints(self, env_idx, agent_idx, pose_y, pose_z):
        lane_sat = self.ctrl.lane_sat
        decay = 1.0 - self.ctrl.lambda_ttcbf_predefined
        boundaries = lane_sat.local_boundaries(
            self.ctrl.scenario.world_state, env_idx, agent_idx
        )
        out = []
        for side, points in enumerate(boundaries):
            facets = self.local_facets(
                lane_sat.facets(points, side), np.asarray(pose_y[0:2])
            )
            if facets["start"].shape[0] == 0:
                continue
            h_y = float(self.atoms_and_gradients(facets, pose_y)[0].min())
            atoms_z, grads_z = self.atoms_and_gradients(facets, pose_z)
            active = np.argwhere(atoms_z - decay * h_y < self.ROW_BAND)
            out.append(
                (
                    side,
                    h_y,
                    [(atoms_z[j, m], grads_z[j, m]) for j, m in active],
                )
            )
        return out
