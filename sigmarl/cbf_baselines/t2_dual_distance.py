# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""T2: optimization-based exact polytope distance CBF (Thirugnanam et al.,
duality-based CBF, ACC 2022; and their discrete-time NMPC version, ICRA 2022).

The barrier is the Euclidean distance between two convex polygons,
  d = min_{p, q} |p - q|  s.t.  A_1 p <= b_1,  A_2 q <= b_2,
obtained from a QP per pair (ProxSuite, as the CBF-QP). Its optimal
primal-dual solution also gives the gradient: by the envelope theorem, a rigid
motion of polygon 1 changes d by n . (dt_1 + dpsi_1 perp(p* - c_1)), with
n = (p* - q*) / d and the witness points p*, q*. Lane boundary facets are
thickened outward by SEGMENT_THICKNESS into polygons, since the method needs
obstacles with nonempty interior. As the distance of overlapping polygons is
zero, the barrier carries no penetration information (h = -buffer).

Where the witness points are not unique (parallel edges, e.g., cars side by
side or along a lane boundary), every closest vertex-edge pair gets its own
row, so that the rotation cannot be exploited through an arbitrary witness.

Deviations from the papers, to state: the barrier is the distance (the paper
uses the squared distance minus a small epsilon), it is enforced by the
common discrete-time condition with the envelope-theorem gradient of the
distance, whereas the paper adds the dual-variable rates to the QP and
enforces a lower bound of the barrier's rate (their Eq. 18), and the
decentralized mode, which the paper does not have, uses a reciprocal split.
"""

import numpy as np
import proxsuite

from sigmarl.cbf_baselines.common import LaneBaseline, PairBaseline
from sigmarl.constants import LENGTH_SCALE

# Outward thickness of lane boundary facets [m] (1 cm at full scale)
SEGMENT_THICKNESS = 5e-4 * LENGTH_SCALE
# Tolerance [m] of non-unique witness points
WITNESS_TOL = 1e-4 * LENGTH_SCALE
# Solver accuracy of the distance QP [m]
DISTANCE_QP_EPS = 1e-9 * LENGTH_SCALE
# Distance QP Hessian of |p - q|^2 in the variables (p, q)
_H = 2.0 * np.block([[np.eye(2), -np.eye(2)], [-np.eye(2), np.eye(2)]])
_G = np.zeros(4)
_L = np.full(8, -1e20)


def rectangle_halfspaces(pose, half) -> tuple:
    """Half-spaces (A [4, 2], b [4]) of a rectangle with half sizes (l, w)."""
    cos_psi, sin_psi = np.cos(pose[2]), np.sin(pose[2])
    e_lon = np.array([cos_psi, sin_psi])
    e_lat = np.array([-sin_psi, cos_psi])
    A = np.stack([e_lon, -e_lon, e_lat, -e_lat])
    center = np.asarray(pose[0:2], dtype=np.float64)
    b = A @ center + np.array([half[0], half[0], half[1], half[1]])
    return A, b


def segment_halfspaces(start, end, normal, thickness=SEGMENT_THICKNESS) -> tuple:
    """Half-spaces of a facet thickened away from the lane (normal points into
    the lane), i.e. the rectangle start, end, end - t n, start - t n."""
    tangent = (end - start) / np.linalg.norm(end - start)
    A = np.stack([normal, -normal, tangent, -tangent])
    b = np.array(
        [
            normal @ start,
            -(normal @ start) + thickness,
            tangent @ end,
            -(tangent @ start),
        ]
    )
    return A, b


def distance_qp(A1, b1, A2, b2, origin) -> tuple:
    """Distance of two polygons and its witness points (p*, q*).

    The QP is solved in coordinates relative to origin (e.g., the center of
    polygon 1), which keeps it well conditioned far from the map origin.
    """
    origin = np.asarray(origin, dtype=np.float64)
    C = np.zeros((8, 4))
    C[0:4, 0:2] = A1
    C[4:8, 2:4] = A2
    result = proxsuite.proxqp.dense.solve(
        _H,
        _G,
        None,
        None,
        C,
        _L,
        np.concatenate([b1 - A1 @ origin, b2 - A2 @ origin]),
        eps_abs=DISTANCE_QP_EPS,
        eps_rel=0.0,
    )
    p, q = result.x[0:2] + origin, result.x[2:4] + origin
    return float(np.linalg.norm(p - q)), p, q


def closest_pairs(points_a, edges_b, points_b, edges_a, distance, tol) -> list:
    """All witness pairs (p on A, q on B) within tol of the distance: the
    vertices of one body against the edges of the other (convex polygons or
    segments; edges as (start, end) pairs)."""
    pairs = {}

    def closest(point, start, end):
        seg = end - start
        t = np.clip((point - start) @ seg / (seg @ seg), 0.0, 1.0)
        return start + t * seg

    for vertex in points_a:
        for start, end in edges_b:
            q = closest(vertex, start, end)
            if np.linalg.norm(vertex - q) <= distance + tol:
                pairs[tuple(np.round(np.r_[vertex, q], 6))] = (vertex, q)
    for vertex in points_b:
        for start, end in edges_a:
            p = closest(vertex, start, end)
            if np.linalg.norm(p - vertex) <= distance + tol:
                pairs[tuple(np.round(np.r_[p, vertex], 6))] = (p, vertex)
    return list(pairs.values())


def rectangle_corners(pose, half) -> np.ndarray:
    """Corners [4, 2] of a rectangle in cyclic order."""
    cos_psi, sin_psi = np.cos(pose[2]), np.sin(pose[2])
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * half
    return np.asarray(pose[0:2]) + local @ np.array(
        [[cos_psi, sin_psi], [-sin_psi, cos_psi]]
    )


def edges_of(points: np.ndarray) -> list:
    """Edges of a closed polygon (or the single edge of a segment)."""
    if points.shape[0] == 2:
        return [(points[0], points[1])]
    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def rigid_gradient(direction, witness, pose) -> np.ndarray:
    """d distance / d pose (x, y, psi) of the body that carries the witness
    point, for the unit direction from the other witness point to it."""
    rel = witness - np.asarray(pose[0:2])
    return np.array(
        [direction[0], direction[1], direction @ np.array([-rel[1], rel[0]])]
    )


class DualDistancePairBaseline(PairBaseline):
    """T2 pair rows: one distance-QP barrier per pair."""

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.half = np.array([0.5 * float(ctrl.length), 0.5 * float(ctrl.width)])
        self.buffer = float(ctrl.pair_safety_buffer)

    # No published decentralized split (the paper is centralized); reciprocal
    CROSS_SHARE = 0.5

    def rows_at(self, x_a, x_b, is_all_witnesses: bool = True) -> list:
        """Rows at the distance QP's witness points, then one per further
        witness pair within WITNESS_TOL (if is_all_witnesses)."""
        distance, p, q = distance_qp(
            *rectangle_halfspaces(x_a, self.half),
            *rectangle_halfspaces(x_b, self.half),
            x_a[0:2],
        )
        self.stats["distance_qps"] += 1
        if distance <= 1e-9:  # Overlap: no gradient information, push apart
            delta = np.asarray(x_a[0:2]) - x_b[0:2]
            direction = delta / max(float(np.linalg.norm(delta)), 1e-9)
            witnesses = [(p, q)]
        else:
            direction = (p - q) / distance
            witnesses = [(p, q)]
            if is_all_witnesses:
                corners_a = rectangle_corners(x_a, self.half)
                corners_b = rectangle_corners(x_b, self.half)
                witnesses += closest_pairs(
                    corners_a,
                    edges_of(corners_b),
                    corners_b,
                    edges_of(corners_a),
                    distance,
                    WITNESS_TOL,
                )
        return [
            (
                distance - self.buffer,
                rigid_gradient(direction, wa, x_a),
                rigid_gradient(-direction, wb, x_b),
            )
            for wa, wb in witnesses
        ]

    def value_and_gradients(self, x_a, x_b) -> tuple:
        return self.rows_at(x_a, x_b, is_all_witnesses=False)[0]

    def value(self, x_a, x_b) -> float:
        return self.rows_at(x_a, x_b, is_all_witnesses=False)[0][0]


class DualDistanceLaneBaseline(LaneBaseline):
    """T2 lane rows: one distance-QP barrier per nearly active facet."""

    # Facets farther than this from the vehicle's circumscribed circle [m]
    # are skipped without a QP (a bounding-circle prefilter; 4 m at full scale)
    PREFILTER_DISTANCE = 0.2 * LENGTH_SCALE
    # Facets whose margin at the linearization control exceeds this band [m]
    # get no row
    ROW_BAND = 0.1 * LENGTH_SCALE

    def __init__(self, ctrl):
        super().__init__(ctrl)
        self.half = np.array([0.5 * float(ctrl.length), 0.5 * float(ctrl.width)])
        self.circumradius = float(np.linalg.norm(self.half))

    def facet_distances(self, facets: dict, keep, pose) -> tuple:
        """Barrier values (distance - buffer) [K] of the kept facets and their
        pose gradients [K, 3] (at the solver's witness points)."""
        values, grads, _ = self.facet_rows(facets, keep, pose)
        return values, grads

    def facet_rows(self, facets: dict, keep, pose) -> tuple:
        """As facet_distances, plus the rows [(value, gradient)] of every
        witness pair of each kept facet."""
        A_rect, b_rect = rectangle_halfspaces(pose, self.half)
        corners = rectangle_corners(pose, self.half)
        values, grads, rows = [], [], []
        for j in keep:
            distance, p, q = distance_qp(
                A_rect,
                b_rect,
                *segment_halfspaces(
                    facets["start"][j], facets["end"][j], facets["normal"][j]
                ),
                pose[0:2],
            )
            self.stats["distance_qps"] += 1
            direction = (p - q) / distance if distance > 1e-9 else facets["normal"][j]
            value = distance - facets["buffer"][j]
            values.append(value)
            grads.append(rigid_gradient(direction, p, pose))
            witnesses = [(p, q)]
            if distance > 1e-9:
                segment = np.stack([facets["start"][j], facets["end"][j]])
                witnesses += closest_pairs(
                    corners,
                    edges_of(segment),
                    segment,
                    edges_of(corners),
                    distance,
                    WITNESS_TOL,
                )
            rows.append(
                [(value, rigid_gradient(direction, w, pose)) for w, _ in witnesses]
            )
        return np.array(values), np.array(grads), rows

    def side_constraints(self, env_idx, agent_idx, pose_y, pose_z):
        lane_sat = self.ctrl.lane_sat
        decay = 1.0 - self.ctrl.lambda_ttcbf_predefined
        boundaries = lane_sat.local_boundaries(
            self.ctrl.scenario.world_state, env_idx, agent_idx
        )
        center = np.asarray(pose_y[0:2])
        out = []
        for side, points in enumerate(boundaries):
            facets = lane_sat.facets(points, side)
            start, seg = facets["start"], facets["end"] - facets["start"]
            if start.shape[0] == 0:
                continue
            t = np.clip(
                np.sum((center - start) * seg, axis=1) / np.sum(seg * seg, axis=1),
                0.0,
                1.0,
            )
            gap = np.linalg.norm(start + t[:, None] * seg - center, axis=1)
            keep = np.nonzero(gap <= self.circumradius + self.PREFILTER_DISTANCE)[0]
            if keep.size == 0:
                continue
            values_y, _ = self.facet_distances(facets, keep, pose_y)
            h_y = float(values_y.min())
            values_z, _, rows_z = self.facet_rows(facets, keep, pose_z)
            active = np.nonzero(values_z - decay * h_y < self.ROW_BAND)[0]
            out.append((side, h_y, [row for k in active for row in rows_z[k]]))
        return out
