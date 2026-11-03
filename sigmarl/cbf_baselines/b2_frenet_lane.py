# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B2: Frenet-frame lateral-offset lane barrier.

The standard lane-keeping CBF bounds the lateral offset d of the vehicle's
center of gravity from one reference path by the local half-widths of the lane:
  point variant:    h_L = w_L(s) - d - W/2,   h_R = w_R(s) + d - W/2,
the linear rows of Kalaria et al. (2023, Eq. 11) and BarrierNet (Xiao et al.,
2023, Eq. 30) with the bound shrunk by the half width (Kalaria Eq. 3); Yin et
al. (2023, Eq. 22) use the product form w^2 - e_y^2, whose zero set is the same.
  heading variant:  h_{L/R} = w_{L/R}(s) -+ d - (L/2 |sin dpsi| + W/2 |cos dpsi|),
dpsi the heading error to the path: the exact corner set of a rotated
rectangle on a straight lane (Jiang et al., 2024, Eqs. 8-9, moved from the rear
axle to the center and including the rear overhang), i.e., the footprint term
of Squires et al. (2025, Eq. 36) with the lever to the bumper (L/2) instead of
the front axle (l_f). Jiang et al.'s published CBF is an ellipse inscribed in
these conditions, and Squires et al.'s barrier is the minimum of this term
along an evasive maneuver (a backup CBF); both are not reproduced here. The reference path is the
agent's own route centerline, whose boundary vertices are aligned with it by
index, so that w_L and w_R interpolate the vertex distances along s. The pose
is projected onto the nearest centerline segment around the route progress.
"""

import math

import numpy as np

from sigmarl.cbf_baselines.common import LaneBaseline


class FrenetLaneBaseline(LaneBaseline):
    """B2 lane rows: one Frenet lateral-offset barrier per side."""

    # Centerline segments searched for the projection, around the route progress
    SEARCH_BACK = 5
    SEARCH_FRONT = 10

    def __init__(self, ctrl, is_heading_aware: bool):
        super().__init__(ctrl)
        self.is_heading_aware = is_heading_aware
        self.half_length = 0.5 * float(ctrl.length)
        self.half_width = 0.5 * float(ctrl.width)
        self.buffer = float(ctrl.safety_buffer)
        self._path_cache = {}

    def path_geometry(self, env_idx: int, agent_idx: int) -> dict:
        """Centerline segments and boundary half-widths of an agent's route."""
        ref_paths = self.ctrl.scenario.world_state.ref_paths_agent_related
        path_id = int(ref_paths.path_id[env_idx, agent_idx])
        geometry = self._path_cache.get(path_id)
        if geometry is not None:
            return geometry
        n_points = int(ref_paths.n_points_long_term[env_idx, agent_idx])

        def to_np(points):
            return points[:n_points].detach().cpu().numpy().astype(np.float64)

        center = to_np(ref_paths.long_term[env_idx, agent_idx])
        left = to_np(ref_paths.left_boundary[env_idx, agent_idx])
        right = to_np(ref_paths.right_boundary[env_idx, agent_idx])
        segments = np.diff(center, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        valid = lengths > 1e-9
        tangent = np.zeros_like(segments)
        tangent[valid] = segments[valid] / lengths[valid, None]
        geometry = dict(
            center=center,
            segments=segments,
            lengths=lengths,
            valid=valid,
            tangent=tangent,
            normal=np.stack([-tangent[:, 1], tangent[:, 0]], axis=1),  # Left
            heading=np.arctan2(tangent[:, 1], tangent[:, 0]),
            w_left=np.linalg.norm(left - center, axis=1),
            w_right=np.linalg.norm(right - center, axis=1),
            cum=np.concatenate(([0.0], np.cumsum(lengths))),
            is_loop=bool(ref_paths.is_loop[env_idx, agent_idx]),
        )
        self._path_cache[path_id] = geometry
        return geometry

    def project(self, geometry: dict, progress: float, point) -> tuple:
        """Nearest centerline segment i and fraction t of a point, searched
        around a route progress [m]."""
        cum = geometry["cum"]
        n_segments = geometry["lengths"].shape[0]
        if geometry["is_loop"]:
            progress %= cum[-1]
        i0 = int(np.searchsorted(cum, progress, side="right")) - 1
        idx = np.arange(i0 - self.SEARCH_BACK, i0 + self.SEARCH_FRONT + 1)
        idx = np.mod(idx, n_segments) if geometry["is_loop"] else idx
        idx = np.unique(idx[(idx >= 0) & (idx < n_segments)])
        idx = idx[geometry["valid"][idx]]
        rel = np.asarray(point, dtype=np.float64) - geometry["center"][idx]
        t = np.clip(
            np.sum(rel * geometry["tangent"][idx], axis=1) / geometry["lengths"][idx],
            0.0,
            1.0,
        )
        dist = np.linalg.norm(rel - t[:, None] * geometry["segments"][idx], axis=1)
        k = int(np.argmin(dist))
        return int(idx[k]), float(t[k])

    def side_values(self, geometry: dict, progress: float, pose) -> list:
        """Rows [(value, gradient), ...] of the left and the right side at a
        pose (x, y, psi); the side's barrier is the minimum of its rows.

        Point variant: one row per side. Heading variant: one smooth row per
        corner on the side (front and rear), whose minimum is
        w -+ d - (L/2 |sin dpsi| + W/2 |cos dpsi|) for |dpsi| < pi/2; separate
        rows avoid linearizing the kink of |sin dpsi| at dpsi = 0.
        """
        i, t = self.project(geometry, progress, pose[0:2])
        tangent, normal = geometry["tangent"][i], geometry["normal"][i]
        offset = float((np.asarray(pose[0:2]) - geometry["center"][i]) @ normal)
        dpsi = math.remainder(float(pose[2]) - geometry["heading"][i], 2.0 * math.pi)
        sin_d, cos_d = math.sin(dpsi), math.cos(dpsi)
        hl, hw = self.half_length, self.half_width
        out = []
        for name, sign in (("w_left", -1.0), ("w_right", 1.0)):
            widths = geometry[name]
            w = (1.0 - t) * widths[i] + t * widths[i + 1]
            dw_ds = (widths[i + 1] - widths[i]) / geometry["lengths"][i]
            grad_xy = dw_ds * tangent + sign * normal
            if self.is_heading_aware:
                # Lateral offsets of the front and rear corner toward the side
                # (left: +W/2, right: -W/2) and their derivatives in dpsi
                corners = [
                    (
                        lon * hl * sin_d - sign * hw * cos_d,
                        lon * hl * cos_d + sign * hw * sin_d,
                    )
                    for lon in (1.0, -1.0)
                ]
            else:
                corners = [(-sign * hw, 0.0)]
            rows = []
            for lateral, d_lateral in corners:
                # Left: w_L - (d + lateral); right: w_R + (d + lateral)
                value = w + sign * (offset + lateral) - self.buffer
                gradient = np.zeros(3)
                gradient[0:2] = grad_xy
                gradient[2] = sign * d_lateral
                rows.append((value, gradient))
            out.append(rows)
        return out

    def side_constraints(self, env_idx, agent_idx, pose_y, pose_z):
        geometry = self.path_geometry(env_idx, agent_idx)
        progress = float(
            self.ctrl.scenario.world_state._route_progress[env_idx, agent_idx]
        )
        rows_y = self.side_values(geometry, progress, pose_y)
        rows_z = self.side_values(geometry, progress, pose_z)
        return [
            (side, min(value for value, _ in rows_y[side]), rows_z[side])
            for side in range(2)
        ]
