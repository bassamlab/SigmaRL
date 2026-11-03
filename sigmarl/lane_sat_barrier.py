# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Exact rectangle-polyline lane barrier (lane_cbf_footprint="rectangle_sat").

The boundary is the map polyline itself. For every facet (segment) j, the
separation between the vehicle rectangle and the segment along an axis of the
separating axis theorem (SAT) is a min over a few smooth "atoms":
  axis 0, facet normal n_j:   n_j . (corner_m - a_j)           (4 corners)
  axis 1/2, +-heading e_lon:  +-e_lon . (x - p) - l            (2 endpoints x)
  axis 3, lateral e_lat:      sign e_lat . (x - p) - w         (2 endpoints x)
The barrier h = min_j max_axis min_atoms is >= 0 iff the rectangle (with
buffer) does not intersect any facet, with no approximation of the boundary.

The geometry is shared by the CBF-QP (``sigmarl/cbf_qp_proxsuite.py``), which
constrains the atoms, and the road-traffic scenario, which logs the barrier.
"""

import math

import numpy as np

from sigmarl.constants import LENGTH_SCALE


class LaneSatBarrier:
    """Rectangle-polyline lane barrier of one vehicle footprint.

    Args:
        length: vehicle length [m].
        width: vehicle width [m].
        safety_buffer: clearance every facet must keep [m].
        is_resampling_buffer: add the largest gap between the map polyline and
            the collision checker's resampled local boundary (see ``facets``).
        window_back: map boundary vertices behind the route progress.
        window_front: map boundary vertices ahead of the route progress.
        extension: straight extension of non-loop boundaries at their ends [m].
    """

    def __init__(
        self,
        length: float,
        width: float,
        safety_buffer: float,
        is_resampling_buffer: bool = True,
        window_back: int = 5,
        window_front: int = 6,
        extension: float = 1.0 * LENGTH_SCALE,
    ):
        self.half_length = 0.5 * float(length)
        self.half_width = 0.5 * float(width)
        self.corners = np.array(
            [
                [self.half_length, self.half_width],
                [self.half_length, -self.half_width],
                [-self.half_length, self.half_width],
                [-self.half_length, -self.half_width],
            ],
            dtype=np.float64,
        )
        self.safety_buffer = float(safety_buffer)
        self.is_resampling_buffer = is_resampling_buffer
        self.window_back = window_back
        self.window_front = window_front
        self.extension = extension
        self._path_cache = {}
        self._facet_table_cache = {}

    def _footprint(self, half):
        """Corners in the vehicle frame and half length and half width of the
        barrier's footprint (half None) or of other half sizes (length, width):
        [2] for one vehicle, or [S, 2] for one vehicle per row (then the corners
        are [S, 4, 2] and the half sizes [S, 1])."""
        if half is None:
            return self.corners, self.half_length, self.half_width
        half = np.asarray(half, dtype=np.float64)
        corners = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        if half.ndim == 1:
            return corners * half, float(half[0]), float(half[1])
        return corners[None] * half[:, None, :], half[:, 0:1], half[:, 1:2]

    def path_geometry(self, ref_paths, env_idx: int, agent_idx: int) -> dict:
        """Return cached centerline arc lengths and aligned boundaries of a path."""
        path_id = int(ref_paths.path_id[env_idx, agent_idx])
        geometry = self._path_cache.get(path_id)
        if geometry is not None:
            return geometry

        n_center = int(ref_paths.n_points_long_term[env_idx, agent_idx])
        n_left = int(ref_paths.n_points_left_b[env_idx, agent_idx])
        n_right = int(ref_paths.n_points_right_b[env_idx, agent_idx])
        if not n_center == n_left == n_right:
            raise ValueError(
                "The rectangle_sat lane barrier needs boundaries aligned with the "
                "centerline by index."
            )

        def to_np(points):
            return points.detach().cpu().numpy().astype(np.float64)

        center = to_np(ref_paths.long_term[env_idx, agent_idx, :n_center])
        segment_lengths = np.linalg.norm(np.diff(center, axis=0), axis=1)
        geometry = dict(
            cum=np.concatenate(([0.0], np.cumsum(segment_lengths))),
            left=to_np(ref_paths.left_boundary[env_idx, agent_idx, :n_left]),
            right=to_np(ref_paths.right_boundary[env_idx, agent_idx, :n_right]),
            is_loop=bool(ref_paths.is_loop[env_idx, agent_idx]),
        )
        self._path_cache[path_id] = geometry
        return geometry

    @staticmethod
    def remove_duplicates(points: np.ndarray) -> np.ndarray:
        """Remove consecutive duplicate polyline points."""
        keep = np.ones(points.shape[0], dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9
        return points[keep]

    def local_boundaries(
        self, world_state, env_idx: int, agent_idx: int, ahead_distance: float = 0.0
    ):
        """Return the map boundary vertices around an agent as [left, right].

        The vertices are the map's own (not resampled) boundary points around the
        agent's route progress. Non-loop boundaries are extended straight along
        their end segments, like the collision checker's local boundaries.

        Args:
            ahead_distance: extends the front of the window by this route
                distance [m], e.g., to cover a predicted trajectory.
        """
        geometry = self.path_geometry(
            world_state.ref_paths_agent_related, env_idx, agent_idx
        )
        progress = float(world_state._route_progress[env_idx, agent_idx])
        return self.boundaries_at(geometry, progress, ahead_distance)

    def boundaries_at(
        self, geometry: dict, progress: float, ahead_distance: float = 0.0
    ):
        """Return the map boundary vertices of a path around a route progress [m]
        as [left, right] (see ``local_boundaries``).

        Args:
            geometry: path geometry as returned by ``path_geometry``.
        """
        cum = geometry["cum"]
        n_points = cum.shape[0]
        if geometry["is_loop"]:
            progress %= cum[-1]
        i0 = self.window_start(geometry, progress)
        i1 = self.window_end(geometry, progress, ahead_distance)
        idx = np.arange(i0 - self.window_back, i1 + self.window_front + 1)

        boundaries = []
        for polyline in (geometry["left"], geometry["right"]):
            if geometry["is_loop"]:
                # The last point of a loop duplicates the first one
                points = polyline[np.mod(idx, n_points - 1)]
            else:
                points = polyline[np.clip(idx, 0, n_points - 1)]
            points = self.remove_duplicates(points)
            if not geometry["is_loop"]:
                if idx[0] < 0:
                    tangent = points[1] - points[0]
                    tangent /= np.linalg.norm(tangent)
                    points = np.vstack([points[0] - self.extension * tangent, points])
                if idx[-1] > n_points - 1:
                    tangent = points[-1] - points[-2]
                    tangent /= np.linalg.norm(tangent)
                    points = np.vstack([points, points[-1] + self.extension * tangent])
            boundaries.append(points)
        return boundaries

    def route_point_margins(
        self, geometry: dict, point_ids: np.ndarray, poses: np.ndarray, ahead_distance
    ) -> np.ndarray:
        """Lane margins [K] (minimum over both sides) of vehicles at poses
        [K, 3] whose route progress is the centerline vertex point_ids [K],
        each over the window of ``boundaries_at`` at that progress, in one
        batch per side.

        The facets are built once on the whole boundary (with the straight end
        extensions of a non-loop path) and gathered per window. The end facets
        of a window thus also get the resampling buffer of their outer
        neighbor, which can only lower the margin.

        Args:
            geometry: path geometry as returned by ``path_geometry``.
            ahead_distance: window extension [m] (see ``local_boundaries``).
        """
        cum = geometry["cum"]
        n_points = cum.shape[0]
        first = np.empty(point_ids.shape[0], dtype=int)
        last = np.empty(point_ids.shape[0], dtype=int)
        for k, point_id in enumerate(point_ids):
            first[k] = self.window_start(geometry, cum[point_id]) - self.window_back
            last[k] = (
                self.window_end(geometry, cum[point_id], ahead_distance)
                + self.window_front
            )
        values = np.full((point_ids.shape[0], 2), np.inf)
        for side, polyline in enumerate((geometry["left"], geometry["right"])):
            # Window vertex range [a, b] of each pose in `vertices`
            if geometry["is_loop"]:
                lo = int(first.min())
                vertices = polyline[
                    np.mod(np.arange(lo, int(last.max()) + 1), n_points - 1)
                ]
                a, b = first - lo, last - lo
            else:
                # End tangents of the distinct vertices, as in boundaries_at
                distinct = self.remove_duplicates(polyline)
                start_tangent = distinct[1] - distinct[0]
                end_tangent = distinct[-1] - distinct[-2]
                vertices = np.vstack(
                    [
                        polyline[0]
                        - self.extension
                        * start_tangent
                        / np.linalg.norm(start_tangent),
                        polyline,
                        polyline[-1]
                        + self.extension * end_tangent / np.linalg.norm(end_tangent),
                    ]
                )
                a = np.maximum(first, -1) + 1
                b = np.minimum(last, n_points) + 1
            # Facet j of the deduplicated vertices spans its vertices j, j + 1;
            # a duplicate vertex maps to the kept one before it
            keep = np.ones(vertices.shape[0], dtype=bool)
            keep[1:] = np.linalg.norm(np.diff(vertices, axis=0), axis=1) > 1e-9
            kept_index = np.cumsum(keep) - 1
            facets = self.facets(vertices[keep], side)
            facet_lo, facet_hi = kept_index[a], kept_index[b]
            n_window = int((facet_hi - facet_lo).max())
            idx = facet_lo[:, None] + np.arange(n_window)[None, :]
            valid = idx < facet_hi[:, None]
            idx = np.minimum(idx, facets["start"].shape[0] - 1)
            values[:, side] = self.padded_values(
                {
                    key: facets[key][idx]
                    for key in ("start", "end", "normal", "offset", "buffer")
                },
                valid,
                np.full(point_ids.shape[0], side),
                poses,
            )
        return values.min(axis=1)

    @staticmethod
    def window_start(geometry: dict, progress: float) -> int:
        """Index of the last centerline vertex at or behind a route progress [m]
        (wrapped for a loop); the local window of ``boundaries_at`` starts
        ``window_back`` vertices before it."""
        cum = geometry["cum"]
        if geometry["is_loop"]:
            progress %= cum[-1]
        return int(np.searchsorted(cum, progress, side="right")) - 1

    @classmethod
    def window_end(cls, geometry: dict, progress: float, ahead_distance: float) -> int:
        """Index of the last centerline vertex at or behind a route progress
        plus ``ahead_distance`` [m] (unwrapped for a loop), at least the window
        start; the local window of ``boundaries_at`` ends ``window_front``
        vertices after it."""
        i0 = cls.window_start(geometry, progress)
        if ahead_distance <= 0.0:
            return i0
        cum = geometry["cum"]
        if geometry["is_loop"]:
            progress %= cum[-1]
            cum = np.concatenate((cum, cum[-1] + cum[1:]))
        return max(
            i0,
            int(np.searchsorted(cum, progress + ahead_distance, side="right")) - 1,
        )

    def padded_local_facets(
        self, world_state, n_envs: int, n_agents: int, ahead_distance=None
    ) -> tuple:
        """Facets of the local boundaries of all agents, zero-padded to a common
        facet count F (see ``local_boundaries`` and ``facets``).

        The facets of a window depend only on the path and the window's start
        and end, so they are cached by them.

        Args:
            ahead_distance: per-agent extension [E, A] of the window fronts [m]
                (see ``local_boundaries``); None: no extension.

        Returns:
            facets: dict of "start", "end", "normal" [E, A, 2, F, 2] and
                "offset", "buffer" [E, A, 2, F], with side 0 left and 1 right.
            valid: [E, A, 2, F] mask of the real facets.
        """
        ref_paths = world_state.ref_paths_agent_related
        path_ids = ref_paths.path_id[:n_envs, :n_agents].cpu().numpy()
        progress = world_state._route_progress[:n_envs, :n_agents].cpu().numpy()
        tables = []
        for env_idx in range(n_envs):
            for agent_idx in range(n_agents):
                path_id = int(path_ids[env_idx, agent_idx])
                geometry = self._path_cache.get(path_id)
                if geometry is None:
                    geometry = self.path_geometry(ref_paths, env_idx, agent_idx)
                agent_progress = float(progress[env_idx, agent_idx])
                ahead = (
                    0.0
                    if ahead_distance is None
                    else float(ahead_distance[env_idx, agent_idx])
                )
                key = (
                    path_id,
                    self.window_start(geometry, agent_progress),
                    self.window_end(geometry, agent_progress, ahead),
                )
                table = self._facet_table_cache.get(key)
                if table is None:
                    table = [
                        self._facet_table(self.facets(points, side))
                        for side, points in enumerate(
                            self.boundaries_at(geometry, agent_progress, ahead)
                        )
                    ]
                    self._facet_table_cache[key] = table
                tables.extend(table)

        n_facets = max(1, max(table.shape[0] for table in tables))
        packed = np.zeros((len(tables), n_facets, 8), dtype=np.float64)
        valid = np.zeros((len(tables), n_facets), dtype=bool)
        for slot, table in enumerate(tables):
            packed[slot, : table.shape[0]] = table
            valid[slot, : table.shape[0]] = True
        shape = (n_envs, n_agents, 2, n_facets)
        facets = {
            name: np.ascontiguousarray(packed[..., columns]).reshape(*shape, -1)
            for name, columns in (
                ("start", slice(0, 2)),
                ("end", slice(2, 4)),
                ("normal", slice(4, 6)),
                ("offset", slice(6, 7)),
                ("buffer", slice(7, 8)),
            )
        }
        facets["offset"] = facets["offset"][..., 0]
        facets["buffer"] = facets["buffer"][..., 0]
        return facets, valid.reshape(shape)

    @staticmethod
    def _facet_table(facets: dict) -> np.ndarray:
        """Facets packed as rows (start [2], end [2], normal [2], offset, buffer)."""
        return np.column_stack(
            [
                facets["start"],
                facets["end"],
                facets["normal"],
                facets["offset"],
                facets["buffer"],
            ]
        )

    def facets(self, points: np.ndarray, side: int, is_buffered: bool = True) -> dict:
        """Return the facets of one boundary with lane-pointing unit normals.

        Each facet also gets a buffer: the safety buffer plus, optionally, the
        largest gap between the map polyline and the collision checker's local
        boundary. The checker samples each boundary segment at the same fraction
        f and connects the samples, so near a vertex that turns toward the lane
        its chords lie inside the lane by at most
        2 sin(dtheta / 2) L1 L2 / (sqrt(L1) + sqrt(L2))^2,
        the maximum over f of the triangle height at the vertex.

        Args:
            points: boundary vertices [N, 2].
            side: 0 for the left boundary, 1 for the right one.
            is_buffered: False gives zero buffers, so the barrier lower-bounds
                the plain rectangle-polyline distance.
        """
        start = points[:-1]
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        keep = lengths > 1e-9
        start, segments, lengths = start[keep], segments[keep], lengths[keep]
        tangent = segments / lengths[:, None]
        if side == 0:  # Left boundary: the lane lies to its right
            normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
        else:
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

        buffer = np.zeros(start.shape[0])
        if is_buffered:
            buffer += self.safety_buffer
            if self.is_resampling_buffer and start.shape[0] > 1:
                cos_turn = np.clip(
                    np.sum(tangent[:-1] * tangent[1:], axis=1), -1.0, 1.0
                )
                turn = np.arccos(cos_turn)
                turns_to_lane = np.sum(
                    (tangent[1:] - tangent[:-1]) * normal[:-1], axis=1
                )
                l1, l2 = lengths[:-1], lengths[1:]
                gap = (
                    2.0
                    * np.sin(0.5 * turn)
                    * l1
                    * l2
                    / (np.sqrt(l1) + np.sqrt(l2)) ** 2
                )
                gap = np.where(turns_to_lane > 0.0, gap, 0.0)
                extra = np.zeros_like(buffer)
                extra[:-1] = gap
                extra[1:] = np.maximum(extra[1:], gap)
                buffer += extra

        return dict(
            start=start,
            end=start + segments,
            normal=normal,
            offset=np.sum(normal * start, axis=1) + buffer,
            buffer=buffer,
        )

    def atom_tables(self, facets: dict, side: int, pose) -> tuple:
        """Atom values of all facets at a pose (x, y, psi), one array per axis.

        Returns (normal [F, 4], ahead [F, 2], behind [F, 2], lateral [F, 2]).
        """
        cos_psi, sin_psi = math.cos(pose[2]), math.sin(pose[2])
        position = np.array([pose[0], pose[1]], dtype=np.float64)
        corners = position + self.corners @ np.array(
            [[cos_psi, sin_psi], [-sin_psi, cos_psi]]
        )
        normal_atoms = facets["normal"] @ corners.T - facets["offset"][:, None]

        e_lon = np.array([cos_psi, sin_psi])
        e_lat = np.array([-sin_psi, cos_psi])
        rel_start = facets["start"] - position
        rel_end = facets["end"] - position
        lon = np.stack([rel_start @ e_lon, rel_end @ e_lon], axis=1)
        # The left boundary lies on the vehicle's left, the right one on its right
        lat = (1.0 if side == 0 else -1.0) * np.stack(
            [rel_start @ e_lat, rel_end @ e_lat], axis=1
        )
        buffer = facets["buffer"][:, None]
        return (
            normal_atoms,
            lon - self.half_length - buffer,
            -lon - self.half_length - buffer,
            lat - self.half_width - buffer,
        )

    def atom_gradient(
        self, facets: dict, side: int, axis: int, facet: int, atom: int, pose
    ) -> np.ndarray:
        """Gradient of one atom with respect to the pose (x, y, psi)."""
        cos_psi, sin_psi = math.cos(pose[2]), math.sin(pose[2])
        if axis == 0:
            normal = facets["normal"][facet]
            offset = self.corners[atom]
            rotated = np.array(
                [
                    cos_psi * offset[0] - sin_psi * offset[1],
                    sin_psi * offset[0] + cos_psi * offset[1],
                ]
            )
            return np.array(
                [normal[0], normal[1], normal @ np.array([-rotated[1], rotated[0]])]
            )

        if axis == 1:
            direction = np.array([cos_psi, sin_psi])
        elif axis == 2:
            direction = -np.array([cos_psi, sin_psi])
        else:
            direction = (1.0 if side == 0 else -1.0) * np.array([-sin_psi, cos_psi])
        point = facets["start"][facet] if atom == 0 else facets["end"][facet]
        rel = point - np.array([pose[0], pose[1]])
        return np.array(
            [
                -direction[0],
                -direction[1],
                np.array([-direction[1], direction[0]]) @ rel,
            ]
        )

    def value(self, facets: dict, side: int, pose) -> float:
        """Barrier value min_j max_axis min_atoms of one boundary at a pose [m]."""
        tables = self.atom_tables(facets, side, pose)
        axis_values = np.stack([table.min(axis=1) for table in tables], axis=1)
        return float(axis_values.max(axis=1).min())

    def values(
        self, facets: dict, side: int, poses: np.ndarray, half=None
    ) -> np.ndarray:
        """Barrier values of one boundary at many poses [P, 3] (as value) [m].

        half: half length and half width [2] of the vehicle; None: those of the
        barrier.
        """
        local, half_length, half_width = self._footprint(half)
        cos_psi, sin_psi = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        e_lon = np.stack([cos_psi, sin_psi], axis=1)  # [P, 2]
        e_lat = np.stack([-sin_psi, cos_psi], axis=1)
        position = poses[:, None, 0:2]  # [P, 1, 2]
        corners = (
            position
            + local[None, :, 0:1] * e_lon[:, None, :]
            + local[None, :, 1:2] * e_lat[:, None, :]
        )  # [P, 4, 2]
        normal_value = (
            np.einsum("fd,pmd->pfm", facets["normal"], corners)
            - facets["offset"][None, :, None]
        ).min(axis=2)

        rel_start = facets["start"][None] - position  # [P, F, 2]
        rel_end = facets["end"][None] - position
        lon = np.stack(
            [
                np.einsum("pfd,pd->pf", rel_start, e_lon),
                np.einsum("pfd,pd->pf", rel_end, e_lon),
            ],
            axis=2,
        )
        lat = (1.0 if side == 0 else -1.0) * np.stack(
            [
                np.einsum("pfd,pd->pf", rel_start, e_lat),
                np.einsum("pfd,pd->pf", rel_end, e_lat),
            ],
            axis=2,
        )
        buffer = facets["buffer"][None, :]
        axis_values = np.stack(
            [
                normal_value,
                lon.min(axis=2) - half_length - buffer,
                -lon.max(axis=2) - half_length - buffer,
                lat.min(axis=2) - half_width - buffer,
            ],
            axis=2,
        )  # [P, F, 4]
        return axis_values.max(axis=2).min(axis=1)

    def padded_values(
        self,
        facets: dict,
        valid: np.ndarray,
        side: np.ndarray,
        poses: np.ndarray,
        half=None,
    ) -> np.ndarray:
        """Barrier values [S] of S boundaries at one pose each [S, 3] [m], equal
        bit for bit to ``value`` per boundary (inf for one without facets).

        ``value`` computes its 2-term dot products with matrix products, which
        BLAS may fuse into multiply-adds, so this function uses matrix products
        of the same inner size instead of elementwise sums.

        Args:
            facets: zero-padded facets [S, F, ...] (see ``padded_local_facets``).
            valid: [S, F] mask of the real facets.
            side: side [S] of each boundary (0 left, 1 right).
            half: half length and half width [S, 2] of the vehicle of each
                pose; None: those of the barrier.
        """
        local, half_length, half_width = self._footprint(half)
        cos_psi, sin_psi = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        e_lon = np.stack([cos_psi, sin_psi], axis=1)  # [S, 2]
        e_lat = np.stack([-sin_psi, cos_psi], axis=1)
        position = poses[:, None, 0:2]  # [S, 1, 2]
        corners = position + np.matmul(
            local, np.stack([e_lon, e_lat], axis=1)
        )  # [S, 4, 2]
        normal_atoms = (
            np.matmul(facets["normal"], corners.transpose(0, 2, 1))
            - facets["offset"][:, :, None]
        )  # [S, F, 4]

        # Facet end points relative to the pose and projected on its axes
        # [S, F]. Rounding is monotone, so taking the min or max over the two
        # end points before the shifts below gives the same bits as after.
        rel_start = facets["start"] - position
        rel_end = facets["end"] - position

        def project(rel, axis):
            return np.matmul(rel, axis[:, :, None])[..., 0]

        lon_start, lon_end = project(rel_start, e_lon), project(rel_end, e_lon)
        lat_start, lat_end = project(rel_start, e_lat), project(rel_end, e_lat)
        lat_min = np.where(
            (side == 0)[:, None],
            np.minimum(lat_start, lat_end),
            -np.maximum(lat_start, lat_end),
        )
        buffer = facets["buffer"]
        value = np.maximum(
            np.maximum(
                normal_atoms.min(axis=2),
                np.minimum(lon_start, lon_end) - half_length - buffer,
            ),
            np.maximum(
                -np.maximum(lon_start, lon_end) - half_length - buffer,
                lat_min - half_width - buffer,
            ),
        )  # [S, F]
        return np.where(valid, value, np.inf).min(axis=1)

    def side_margins(
        self, world_state, env_idx: int, agent_idx: int, pose, is_buffered=True
    ) -> np.ndarray:
        """Barrier values [left, right] of one agent at a pose [m].

        The local boundaries are taken at the agent's current route progress.
        """
        boundaries = self.local_boundaries(world_state, env_idx, agent_idx)
        return np.array(
            [
                self.value(self.facets(points, side, is_buffered), side, pose)
                for side, points in enumerate(boundaries)
            ]
        )
