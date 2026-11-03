# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Initial-state repair: move recorded start poses whose simulated footprint crosses the route boundary.

About 6% of the controllable tracks start with the simulated footprint (`AGENTS` length x width, or the
vehicle's own, see `VehicleParams`) a few centimeters across a route boundary, mostly because of bounding-box noise and map inaccuracies. The lane
CBF treats such a start as an unsafe state and stops the vehicle, which says nothing about the controller
on a valid start. The repair moves the start pose by the smallest lateral shift along the route normal
(and, if that is not enough, the smallest heading change toward the route direction) that makes the
lane barrier of the CBF-QP (`LaneSatBarrier`, same footprint and buffers) at least `MARGIN` on both sides.
Speed, steering, and time stay as recorded. A start that needs a correction beyond `MAX_SHIFT` or
`MAX_YAW` is "failed" (a bad recording); `episodes.build_episode` removes such vehicles from the episode.

The repair checks the start pose only. A vehicle that enters at speed while heading toward the boundary can
have a pose with a nonnegative lane barrier whose braking backup still crosses the boundary, i.e., a start
outside the certified set. A certified entry check (`LogReplay.entry_repair`) then searches `candidates`,
the same corrections in the same order, for the smallest one whose braking backup also stays in the lane.
"""

import math
from dataclasses import dataclass

import numpy as np

from sigmarl.constants import AGENTS
from sigmarl.lane_sat_barrier import LaneSatBarrier

MARGIN = 0.01  # [m] Lane barrier value (on top of the CBF's buffers) the repaired pose must reach
MAX_SHIFT = 1.0  # [m] Repairs all starts of rounD location 0 (largest needed: 0.96 m)
MAX_YAW = math.radians(5.0)  # [rad]
SHIFT_STEP = 0.002  # [m] Resolution of the search
YAW_STEP = math.radians(0.1)  # [rad]


@dataclass
class StartRepair:
    """Outcome of the repair of one start pose."""

    xy: np.ndarray  # Start position [m] (repaired, or recorded if not repaired)
    yaw: float  # Start yaw [rad]
    barrier_recorded: float  # Lane barrier (min of both sides) at the recorded pose [m]
    barrier: float  # Lane barrier at the start pose [m]
    shift: float  # Lateral shift along the route's left normal [m]
    dyaw: float  # Heading change [rad]
    status: str  # "ok" (no repair needed), "repaired", or "failed" (correction beyond the limits)
    # Repaired at the entry so that the braking backup stays in the lane too (see `candidates`)
    is_backup_repair: bool = False


class StartPoseRepair:
    """Repairs start poses on the reference paths of a map parser (see module docstring).

    Args:
        reference_paths: `ParseLanelet2.reference_paths` (shared boundaries aligned with the center line).
    """

    def __init__(self, reference_paths):
        # Footprint and buffer of the CBF-QP's lane barrier (see `road_traffic.py`)
        self.barrier = LaneSatBarrier(
            length=AGENTS["length"],
            width=AGENTS["width"],
            safety_buffer=0.01 * AGENTS["width"],
        )
        self.geometries = []
        for path in reference_paths:
            center = path["center_line"].cpu().numpy().astype(np.float64)
            segment_lengths = np.linalg.norm(np.diff(center, axis=0), axis=1)
            self.geometries.append(
                dict(
                    cum=np.concatenate(([0.0], np.cumsum(segment_lengths))),
                    left=path["left_boundary_shared"].cpu().numpy().astype(np.float64),
                    right=path["right_boundary_shared"]
                    .cpu()
                    .numpy()
                    .astype(np.float64),
                    is_loop=bool(path["is_loop"]),
                )
            )

    def barrier_values(
        self, path_id: int, progress: float, poses: np.ndarray, params=None
    ):
        """Lane barrier min(left, right) at poses [P, 3] (x, y, yaw), boundaries taken at `progress` [m], of
        a vehicle with parameters `params` (`VehicleParams`; None: the nominal ones)."""
        boundaries = self.barrier.boundaries_at(self.geometries[path_id], progress)
        half = None if params is None else 0.5 * np.array([params.length, params.width])
        return np.min(
            [
                self.barrier.values(
                    self.barrier.facets(points, side), side, poses, half
                )
                for side, points in enumerate(boundaries)
            ],
            axis=0,
        )

    def corrections(self, xy, yaw: float, route_yaw: float):
        """Candidate start poses [K, 3] with their lateral shifts and heading changes [K], smallest heading
        change first and, for each heading, smallest shift first (the search order of `repair`)."""
        xy = np.asarray(xy, dtype=np.float64)
        normal = np.array([-math.sin(route_yaw), math.cos(route_yaw)])
        n_shift = int(round(MAX_SHIFT / SHIFT_STEP))
        shifts = SHIFT_STEP * np.arange(-n_shift, n_shift + 1)
        shifts = shifts[np.argsort(np.abs(shifts), kind="stable")]
        # Heading changes toward the route direction only, smallest first
        yaw_error = (route_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        max_dyaw = min(MAX_YAW, abs(yaw_error))
        dyaws = math.copysign(1.0, yaw_error) * np.append(
            np.arange(0.0, max_dyaw, YAW_STEP), max_dyaw
        )
        shift_grid = np.tile(shifts, dyaws.shape[0])
        dyaw_grid = np.repeat(dyaws, shifts.shape[0])
        poses = np.column_stack(
            [
                xy[0] + shift_grid * normal[0],
                xy[1] + shift_grid * normal[1],
                yaw + dyaw_grid,
            ]
        )
        return poses, shift_grid, dyaw_grid

    def candidates(
        self,
        path_id: int,
        progress: float,
        xy,
        yaw: float,
        route_yaw: float,
        params=None,
    ):
        """Start poses [K, 3] in the order of `repair` whose lane barrier is at least `MARGIN`, with their
        shifts and heading changes [K] (arguments as in `repair`)."""
        poses, shifts, dyaws = self.corrections(xy, yaw, route_yaw)
        feasible = self.barrier_values(path_id, progress, poses, params) >= MARGIN
        return poses[feasible], shifts[feasible], dyaws[feasible]

    def repair(
        self,
        path_id: int,
        progress: float,
        xy,
        yaw: float,
        route_yaw: float,
        params=None,
    ) -> StartRepair:
        """Repair one start pose.

        Args:
            path_id: reference path of the vehicle.
            progress: arc length of the pose along the path's center line [m].
            xy, yaw: recorded pose.
            route_yaw: direction of the center line at `progress` [rad].
            params: `VehicleParams` of the vehicle; None: the nominal ones.
        """
        xy = np.asarray(xy, dtype=np.float64)
        pose = np.array([[xy[0], xy[1], yaw]])
        b0 = float(self.barrier_values(path_id, progress, pose, params)[0])
        if b0 >= MARGIN:
            return StartRepair(xy, yaw, b0, b0, 0.0, 0.0, "ok")

        normal = np.array([-math.sin(route_yaw), math.cos(route_yaw)])
        n_shift = int(round(MAX_SHIFT / SHIFT_STEP))
        shifts = SHIFT_STEP * np.arange(-n_shift, n_shift + 1)
        # Heading changes toward the route direction only, smallest first
        yaw_error = (route_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        max_dyaw = min(MAX_YAW, abs(yaw_error))
        dyaws = math.copysign(1.0, yaw_error) * np.append(
            np.arange(0.0, max_dyaw, YAW_STEP), max_dyaw
        )
        for dyaw in dyaws:
            poses = np.column_stack(
                [
                    xy[0] + shifts * normal[0],
                    xy[1] + shifts * normal[1],
                    np.full(shifts.shape, yaw + dyaw),
                ]
            )
            values = self.barrier_values(path_id, progress, poses, params)
            feasible = np.flatnonzero(values >= MARGIN)
            if feasible.size:
                k = feasible[np.argmin(np.abs(shifts[feasible]))]
                return StartRepair(
                    poses[k, :2],
                    float(poses[k, 2]),
                    b0,
                    float(values[k]),
                    float(shifts[k]),
                    float(dyaw),
                    "repaired",
                )
        return StartRepair(xy, yaw, b0, b0, 0.0, 0.0, "failed")
