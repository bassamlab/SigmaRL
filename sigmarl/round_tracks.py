# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Recorded rounD tracks in map coordinates, matched to the reference paths of a Lanelet2 scenario.

The rounD data set (https://www.round-dataset.com/) stores one recording as `<NN>_tracks.csv`,
`<NN>_tracksMeta.csv`, and `<NN>_recordingMeta.csv`. Positions (`xCenter`, `yCenter`) are bounding-box
centers in meters relative to `xUtmOrigin`/`yUtmOrigin`; `heading` is in degrees, counterclockwise from
east. `ParseLanelet2.utm_to_map` maps UTM coordinates to the coordinates of the simulated map (shift and
scale only, so headings are unchanged).

`RouteMatcher` assigns each track to the reference path (route) of the scenario that it follows and
computes its arc-length progress along that route. `initial_state` gives the simulator state of a matched
track at a given time, e.g., to start agents at recorded positions, headings, and speeds.
"""

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Columns of <NN>_tracks.csv
_COL_TRACK_ID = 1
_COL_FRAME = 2
_COL_X = 4
_COL_Y = 5
_COL_HEADING = 6
_COL_VX = 9
_COL_VY = 10
_COL_LON_ACC = 15
_COL_LAT_ACC = 16

# Recordings of rounD location 0 (the Lanelet2 map `round_location0.osm`)
ROUND_LOCATION0_RECORDINGS = tuple(range(2, 24))

# Road-user classes of rounD
MOTOR_VEHICLE_CLASSES = ("car", "van", "truck", "bus", "trailer", "motorcycle")
VRU_CLASSES = ("bicycle", "pedestrian")


@dataclass
class Track:
    """One recorded road user in map coordinates."""

    recording_id: int
    track_id: int
    cls: str
    width: float  # [m]
    length: float  # [m]
    frames: np.ndarray  # [T] frame numbers
    t: np.ndarray  # [T] time since recording start [s]
    xy: np.ndarray  # [T, 2] bounding-box center in map coordinates
    heading: np.ndarray  # [T] yaw [rad], unwrapped
    speed: np.ndarray  # [T] [m/s]
    lon_acc: np.ndarray  # [T] [m/s^2]
    lat_acc: np.ndarray  # [T] [m/s^2]
    match: Optional["RouteMatch"] = field(default=None, repr=False)

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def state_at(self, t: float) -> np.ndarray:
        """Linearly interpolated [x, y, yaw, speed] at time t (clipped to the recorded interval)."""
        return np.array(
            [
                np.interp(t, self.t, self.xy[:, 0]),
                np.interp(t, self.t, self.xy[:, 1]),
                np.interp(t, self.t, self.heading),
                np.interp(t, self.t, self.speed),
            ]
        )


@dataclass
class Recording:
    recording_id: int
    location_id: int
    frame_rate: float
    duration: float  # [s]
    tracks: list


@dataclass
class RouteMatch:
    """Result of matching a track to a route."""

    route_idx: Optional[
        int
    ]  # Index into the scenario's reference paths; None if no route fits
    s: Optional[np.ndarray]  # [T] arc length along the route [m]
    lateral: Optional[
        np.ndarray
    ]  # [T] signed lateral offset from the route center line (left > 0) [m]
    mean_abs_lateral: float
    max_abs_lateral: float
    reason: str  # "ok" or why the track was rejected


def load_recording(data_dir: str, recording_id: int, map_parser) -> Recording:
    """Load one rounD recording and transform its tracks to map coordinates.

    Args:
        data_dir: Directory with the rounD csv files.
        recording_id: Recording number (e.g., 2 for `02_tracks.csv`).
        map_parser: `ParseLanelet2` of the scenario (provides `utm_to_map`).
    """
    prefix = os.path.join(data_dir, f"{recording_id:02d}")
    with open(f"{prefix}_recordingMeta.csv") as f:
        meta = next(csv.DictReader(f))
    with open(f"{prefix}_tracksMeta.csv") as f:
        tracks_meta = {int(row["trackId"]): row for row in csv.DictReader(f)}
    data = np.loadtxt(f"{prefix}_tracks.csv", delimiter=",", skiprows=1)

    frame_rate = float(meta["frameRate"])
    utm_origin = np.array([float(meta["xUtmOrigin"]), float(meta["yUtmOrigin"])])
    xy_map = map_parser.utm_to_map(data[:, [_COL_X, _COL_Y]] + utm_origin)

    tracks = []
    track_ids = data[:, _COL_TRACK_ID].astype(int)
    boundaries = np.flatnonzero(np.diff(track_ids)) + 1
    for rows in np.split(np.arange(len(data)), boundaries):
        row_meta = tracks_meta[int(track_ids[rows[0]])]
        frames = data[rows, _COL_FRAME].astype(int)
        tracks.append(
            Track(
                recording_id=recording_id,
                track_id=int(track_ids[rows[0]]),
                cls=row_meta["class"],
                width=float(row_meta["width"]),
                length=float(row_meta["length"]),
                frames=frames,
                t=frames / frame_rate,
                xy=xy_map[rows],
                heading=np.unwrap(np.radians(data[rows, _COL_HEADING])),
                speed=np.hypot(data[rows, _COL_VX], data[rows, _COL_VY]),
                lon_acc=data[rows, _COL_LON_ACC],
                lat_acc=data[rows, _COL_LAT_ACC],
            )
        )
    return Recording(
        recording_id=recording_id,
        location_id=int(meta["locationId"]),
        frame_rate=frame_rate,
        duration=float(meta["duration"]),
        tracks=tracks,
    )


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class RoutePolyline:
    """Center line of one route with arc length, segment directions, and helpers for projections."""

    def __init__(self, center_line: np.ndarray):
        self.points = np.asarray(center_line, dtype=float)
        seg = np.diff(self.points, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        keep = self.seg_len > 1e-6
        self.points = np.vstack([self.points[:1], self.points[1:][keep]])
        seg = np.diff(self.points, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_dir = seg / self.seg_len[:, None]
        self.seg_yaw = np.arctan2(self.seg_dir[:, 1], self.seg_dir[:, 0])
        self.s_points = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s_points[-1])

    def project(self, xy: np.ndarray, seg_lo: int = 0, seg_hi: Optional[int] = None):
        """Project points [N, 2] onto segments seg_lo..seg_hi-1.

        Returns arc length s, signed lateral offset (left > 0), segment index, and route yaw, each [N].
        """
        seg_hi = len(self.seg_len) if seg_hi is None else seg_hi
        a = self.points[seg_lo:seg_hi]
        d = self.seg_dir[seg_lo:seg_hi]
        rel = xy[:, None, :] - a[None]
        u = np.clip(
            np.einsum("nsk,sk->ns", rel, d), 0.0, self.seg_len[seg_lo:seg_hi][None]
        )
        foot = a[None] + u[..., None] * d[None]
        dist = np.linalg.norm(xy[:, None, :] - foot, axis=-1)
        k = np.argmin(dist, axis=1)
        rows = np.arange(len(xy))
        seg = seg_lo + k
        cross = d[k, 0] * rel[rows, k, 1] - d[k, 1] * rel[rows, k, 0]
        lateral = np.sign(cross) * dist[rows, k]
        return self.s_points[seg] + u[rows, k], lateral, seg, self.seg_yaw[seg]

    def yaw_at(self, s):
        seg = np.clip(
            np.searchsorted(self.s_points, s, side="right") - 1,
            0,
            len(self.seg_len) - 1,
        )
        return self.seg_yaw[seg]

    def xy_at(self, s):
        s = np.clip(s, 0.0, self.length)
        return np.stack(
            [
                np.interp(s, self.s_points, self.points[:, 0]),
                np.interp(s, self.s_points, self.points[:, 1]),
            ],
            axis=-1,
        )


class RouteMatcher:
    """Match recorded tracks to the reference paths (routes) of a scenario.

    A track matches a route if, sampled every `sample_dt` seconds, every point lies within
    `max_lateral` of the route's center line, its heading agrees with the route direction (within
    `max_heading_error` while moving), and its progress along the route never goes backward by more than
    `max_backward`. The progress is tracked sequentially, so routes that pass near the same place twice
    (e.g., U-turns) are handled. Among the matching routes, the one with the smallest mean lateral offset
    wins.
    """

    def __init__(
        self,
        reference_paths,
        max_lateral: float = 4.5,
        max_heading_error: float = math.radians(50),
        max_backward: float = 2.0,
        sample_dt: float = 0.4,
        min_moving_speed: float = 1.0,
    ):
        self.routes = [
            RoutePolyline(p["center_line"].cpu().numpy()) for p in reference_paths
        ]
        self.max_lateral = max_lateral
        self.max_heading_error = max_heading_error
        self.max_backward = max_backward
        self.sample_dt = sample_dt
        self.min_moving_speed = min_moving_speed

    def _track_route(self, route: RoutePolyline, xy, heading, speed):
        """Sequential projection of points onto one route; returns (s, lateral) or None if it does not fit."""
        moving = speed > self.min_moving_speed
        # Start: nearest heading-consistent segment over the whole route
        n_seg = len(route.seg_len)
        rel = xy[0][None] - route.points[:-1]
        u = np.clip(np.einsum("sk,sk->s", rel, route.seg_dir), 0, route.seg_len)
        dist0 = np.linalg.norm(rel - u[:, None] * route.seg_dir, axis=1)
        ok0 = dist0 <= self.max_lateral
        if moving[0]:
            ok0 &= np.abs(_wrap(route.seg_yaw - heading[0])) <= self.max_heading_error
        if not ok0.any():
            return None
        seg = int(np.flatnonzero(ok0)[np.argmin(dist0[ok0])])
        s_all = np.empty(len(xy))
        lat_all = np.empty(len(xy))
        for k in range(len(xy)):
            lo = max(0, seg - 3)
            hi = min(n_seg, seg + 12)
            s_k, lat_k, seg_k, yaw_k = route.project(xy[k : k + 1], lo, hi)
            if abs(lat_k[0]) > self.max_lateral:
                return None
            if moving[k] and abs(_wrap(yaw_k[0] - heading[k])) > self.max_heading_error:
                return None
            if k > 0 and s_k[0] < s_all[k - 1] - self.max_backward:
                return None
            s_all[k], lat_all[k], seg = s_k[0], lat_k[0], int(seg_k[0])
        return s_all, lat_all

    def match(self, track: Track) -> RouteMatch:
        """Find the route the track follows (see class docstring)."""
        stride = max(1, int(round(self.sample_dt / np.median(np.diff(track.t)))))
        idx = np.unique(np.r_[np.arange(0, len(track.t), stride), len(track.t) - 1])
        xy, heading, speed = track.xy[idx], track.heading[idx], track.speed[idx]
        best = None
        for r_idx, route in enumerate(self.routes):
            res = self._track_route(route, xy, heading, speed)
            if res is None:
                continue
            mean_abs = float(np.mean(np.abs(res[1])))
            if best is None or mean_abs < best[1]:
                best = (r_idx, mean_abs)
        if best is None:
            return RouteMatch(None, None, None, math.inf, math.inf, "no_route")
        r_idx = best[0]
        # Dense progress for all frames, seeded by the sampled progress
        route = self.routes[r_idx]
        res = self._track_route(route, track.xy, track.heading, track.speed)
        if (
            res is None
        ):  # Can only happen through the denser heading check; fall back to the samples
            s_sparse, lat_sparse = self._track_route(route, xy, heading, speed)
            s = np.interp(track.t, track.t[idx], s_sparse)
            lateral = np.interp(track.t, track.t[idx], lat_sparse)
        else:
            s, lateral = res
        return RouteMatch(
            route_idx=r_idx,
            s=s,
            lateral=lateral,
            mean_abs_lateral=float(np.mean(np.abs(lateral))),
            max_abs_lateral=float(np.max(np.abs(lateral))),
            reason="ok",
        )


def steering_from_yaw_rate(
    track: Track, k: int, wheelbase: float, max_steering: float
) -> float:
    """Kinematic-bicycle steering angle that reproduces the recorded yaw rate at frame index k."""
    lo, hi = max(0, k - 2), min(len(track.t) - 1, k + 2)
    if hi == lo:
        return 0.0
    yaw_rate = (track.heading[hi] - track.heading[lo]) / (track.t[hi] - track.t[lo])
    speed = max(float(track.speed[k]), 1e-3)
    if speed < 1.0:
        return 0.0
    # Rear-axle referenced curvature; the bicycle model's sideslip correction is small at these angles
    return float(
        np.clip(math.atan(wheelbase * yaw_rate / speed), -max_steering, max_steering)
    )


def initial_state(
    track: Track, t: float, wheelbase: float, max_steering: float
) -> dict:
    """Simulator state of a track at time t: position, yaw, speed, and steering angle."""
    x, y, yaw, speed = track.state_at(t)
    k = int(np.clip(np.searchsorted(track.t, t), 0, len(track.t) - 1))
    return {
        "pos": np.array([x, y]),
        "rot": float(_wrap(yaw)),
        "speed": float(speed),
        "steering": steering_from_yaw_rate(track, k, wheelbase, max_steering),
    }
