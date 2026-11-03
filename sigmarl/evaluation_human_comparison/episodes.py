# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Load the rounD location-0 recordings, match tracks to routes, and build log-replay episodes.

Three episode types:

- "replay": one human-driven car (the ego) is replaced by the controller under test; all other road users
  are replayed from the recording and do not react to it (non-reactive log replay).
- "closed_loop": all cars and vans that appear in a time window and match a route are replaced by the
  controller under test; trucks, buses, trailers, motorcycles, bicycles, and the cars and vans outside the
  window (or without a matching route) are replayed.
- "homogeneous": a maximal time interval in which every road user on the map is a controllable car or van
  (no truck, trailer, bus, motorcycle, bicycle, pedestrian, or car/van that is not controllable). All of
  them are controlled, nothing is replayed; vehicles already on the road at the interval start are
  controlled from their recorded states at that time. Only the vehicles whose whole recorded trip lies
  inside the interval are compared with the humans (`EpisodeSpec.evaluated_keys`); the others give traffic.

Pedestrians are ignored in "replay" and "closed_loop" (12 in the 22 recordings, all off the road).

Controlled vehicles have the nominal parameters of the agent profile, or, with recorded vehicles
(`recorded_vehicle_params`), the recorded footprint of their human's vehicle, axle distances scaled with
its length, and the acceleration limits of its class.
"""

import math
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sigmarl.constants import AGENTS
from sigmarl.log_replay import ControlledVehicle, ReplayedVehicle, split_into_boxes
from sigmarl.round_tracks import (
    ROUND_LOCATION0_RECORDINGS,
    RouteMatcher,
    load_recording,
)
from sigmarl.vehicle_params import VehicleParams

CONTROLLABLE_CLASSES = ("car", "van")
IGNORED_CLASSES = ("pedestrian",)

# Selection of controlled vehicles
MIN_ROUTE_DISTANCE = 20.0  # [m] Recorded distance along the route
MIN_MEAN_SPEED = 1.0  # [m/s] Excludes vehicles that stand most of the time
# Timeout of a controlled vehicle: TIMEOUT_FACTOR x human travel time + TIMEOUT_MARGIN
TIMEOUT_FACTOR = 2.0
TIMEOUT_MARGIN = 30.0  # [s]
# Homogeneous episodes: shortest interval (median recorded trip: 14 s)
MIN_HOMOGENEOUS_INTERVAL = 20.0  # [s]
# Acceleration limits (min, max) [m/s^2] of recorded vehicles by class: cars those of the agent
# profile; vans (median 6.2 m x 2.4 m in rounD) brake and accelerate less
CLASS_ACCELERATION = {
    "car": (AGENTS["min_acc"], AGENTS["max_acc"]),
    "van": (-5.0, 2.5),
}


@dataclass
class EpisodeSpec:
    """Everything needed to build a `LogReplay` for one episode."""

    kind: str  # "replay", "closed_loop", or "homogeneous"
    recording_id: int
    t0: float  # Recording time of simulation step 0 [s]
    controlled_keys: list  # (recording_id, track_id) of the controlled vehicles
    window: float  # Closed loop: length of the spawn window [s]. Homogeneous: length of the interval [s]
    name: str
    evaluated_keys: Optional[
        list
    ] = None  # Vehicles compared with the humans (None: all controlled ones)

    def is_evaluated(self, key) -> bool:
        return self.evaluated_keys is None or tuple(key) in self.evaluated_keys


def load_location0(
    data_dir, map_parser, cache_path=None, recordings=ROUND_LOCATION0_RECORDINGS
):
    """Load and route-match all recordings of location 0 (cached in `cache_path` if given)."""
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    matcher = RouteMatcher(map_parser.reference_paths)
    data = {}
    for rec_id in recordings:
        recording = load_recording(data_dir, rec_id, map_parser)
        assert recording.location_id == 0, f"Recording {rec_id} is not at location 0."
        for track in recording.tracks:
            if track.cls not in IGNORED_CLASSES:
                track.match = matcher.match(track)
        data[rec_id] = recording
        print(
            f"[INFO] Loaded and matched recording {rec_id:02d} ({len(recording.tracks)} tracks)."
        )
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    return data


def is_controllable(track, max_length: Optional[float] = None) -> bool:
    """Whether a track can be given to the controller under test (see module docstring); with
    `max_length` [m], longer vehicles are not (they count as heavy vehicles)."""
    if track.cls not in CONTROLLABLE_CLASSES or track.match is None:
        return False
    if max_length is not None and track.length > max_length:
        return False
    if track.match.route_idx is None:
        return False
    distance = track.match.s[-1] - track.match.s[0]
    duration = track.t_end - track.t_start
    return distance >= MIN_ROUTE_DISTANCE and distance / duration >= MIN_MEAN_SPEED


def matching_summary(data) -> dict:
    """Counts of tracks per class and matching outcome."""
    summary = {}
    for recording in data.values():
        for track in recording.tracks:
            entry = summary.setdefault(
                track.cls, {"tracks": 0, "matched": 0, "controllable": 0}
            )
            entry["tracks"] += 1
            if track.match is not None and track.match.route_idx is not None:
                entry["matched"] += 1
            entry["controllable"] += int(is_controllable(track))
    return summary


def nearest_route(matcher_routes, track) -> int:
    """Route closest to the first point of a track whose direction agrees with the track's heading."""
    best, best_cost = 0, math.inf
    for r_idx, route in enumerate(matcher_routes):
        _, lateral, _, yaw = route.project(track.xy[:1])
        heading_error = abs(
            (yaw[0] - track.heading[0] + math.pi) % (2 * math.pi) - math.pi
        )
        cost = abs(lateral[0]) + 10.0 * heading_error
        if cost < best_cost:
            best, best_cost = r_idx, cost
    return best


def recorded_vehicle_params(track) -> VehicleParams:
    """Parameters of the simulated vehicle of a recorded car or van: its recorded footprint, axle distances
    scaled with its length (the wheelbase is not recorded), and the acceleration limits of its class."""
    min_acc, max_acc = CLASS_ACCELERATION[track.cls]
    return VehicleParams.scaled(track.length, track.width, min_acc, max_acc)


def controlled_vehicle(
    track,
    routes,
    t_begin: Optional[float] = None,
    start_repair=None,
    is_recorded_vehicle: bool = False,
) -> ControlledVehicle:
    """Controlled vehicle of a track, controlled from its first recorded sample at or after `t_begin`
    (default: from its appearance).

    With a `StartPoseRepair` (`start_repair.py`), a start pose whose simulated footprint crosses the route
    boundary is corrected; the outcome is stored in `vehicle.start_repair`. With `is_recorded_vehicle`, the
    vehicle has the parameters of the recorded one (`recorded_vehicle_params`), else the nominal ones.
    """
    k_start = 0 if t_begin is None else int(np.searchsorted(track.t, t_begin - 1e-9))
    duration = track.t_end - float(track.t[k_start])
    route = routes[track.match.route_idx]
    vehicle = ControlledVehicle(
        track=track,
        path_id=track.match.route_idx,
        route=route,
        s_goal=float(track.match.s[-1]),
        max_duration=TIMEOUT_FACTOR * duration + TIMEOUT_MARGIN,
        k_start=k_start,
        params=recorded_vehicle_params(track) if is_recorded_vehicle else None,
    )
    if start_repair is not None:
        progress = float(track.match.s[k_start])
        outcome = start_repair.repair(
            vehicle.path_id,
            progress,
            track.xy[k_start],
            float(track.heading[k_start]),
            float(route.yaw_at(progress)),
            vehicle.params,
        )
        vehicle.start_xy, vehicle.start_yaw = outcome.xy, outcome.yaw
        vehicle.start_repair = outcome
    return vehicle


def build_replay_vehicles(tracks, routes) -> list:
    """Replayed vehicles (one per car-sized box) for the given tracks."""
    vehicles = []
    for track in tracks:
        if track.match is not None and track.match.route_idx is not None:
            path_id = track.match.route_idx
        else:
            path_id = nearest_route(routes, track)
        for offset in split_into_boxes(track, AGENTS["length"]):
            vehicles.append(
                ReplayedVehicle(track=track, offset=float(offset), path_id=path_id)
            )
    return vehicles


def replay_episodes(data, n_per_recording: int, seed: int) -> list:
    """One episode per sampled ego car; egos are drawn uniformly from the controllable tracks of each recording."""
    rng = np.random.default_rng(seed)
    episodes = []
    for rec_id, recording in sorted(data.items()):
        candidates = [t for t in recording.tracks if is_controllable(t)]
        chosen = rng.choice(
            len(candidates), size=min(n_per_recording, len(candidates)), replace=False
        )
        for idx in sorted(chosen):
            track = candidates[idx]
            episodes.append(
                EpisodeSpec(
                    kind="replay",
                    recording_id=rec_id,
                    t0=track.t_start,
                    controlled_keys=[(rec_id, track.track_id)],
                    window=0.0,
                    name=f"rec{rec_id:02d}_track{track.track_id:04d}",
                )
            )
    return episodes


def closed_loop_episodes(data, window: float, n_per_recording: int, seed: int) -> list:
    """Time windows of `window` seconds; all controllable vehicles that appear in a window are controlled."""
    rng = np.random.default_rng(seed)
    episodes = []
    for rec_id, recording in sorted(data.items()):
        latest_start = recording.duration - window - 60.0
        starts = np.sort(rng.uniform(30.0, latest_start, size=n_per_recording))
        for t0 in starts:
            keys = [
                (rec_id, t.track_id)
                for t in recording.tracks
                if is_controllable(t) and t0 <= t.t_start < t0 + window
            ]
            episodes.append(
                EpisodeSpec(
                    kind="closed_loop",
                    recording_id=rec_id,
                    t0=float(t0),
                    controlled_keys=keys,
                    window=window,
                    name=f"rec{rec_id:02d}_t{t0:06.1f}",
                )
            )
    return episodes


def homogeneous_intervals(recording, max_length: Optional[float] = None) -> list:
    """Maximal intervals (t_a, t_b) in which no road user that is not controllable is on the map
    (`max_length`: see `is_controllable`)."""
    t_rec_start = min(t.t_start for t in recording.tracks)
    t_rec_end = max(t.t_end for t in recording.tracks)
    blocked = sorted(
        (t.t_start, t.t_end)
        for t in recording.tracks
        if not is_controllable(t, max_length)
    )
    intervals, t_free = [], t_rec_start
    for t_a, t_b in blocked:
        if t_a > t_free:
            intervals.append((t_free, t_a))
        t_free = max(t_free, t_b)
    if t_rec_end > t_free:
        intervals.append((t_free, t_rec_end))
    return intervals


def homogeneous_episode(
    recording, t_a: float, t_b: float, max_length: Optional[float] = None
) -> EpisodeSpec:
    """All vehicles on the map during (t_a, t_b) are controlled; those whose recorded trips lie inside are
    evaluated (`max_length`: see `is_controllable`)."""
    rec_id = recording.recording_id
    present = [t for t in recording.tracks if t.t_end > t_a and t.t_start < t_b]
    assert all(
        is_controllable(t, max_length) for t in present
    ), "Not a homogeneous interval."
    return EpisodeSpec(
        kind="homogeneous",
        recording_id=rec_id,
        t0=float(t_a),
        controlled_keys=[(rec_id, t.track_id) for t in present],
        window=float(t_b - t_a),
        name=f"rec{rec_id:02d}_h{t_a:06.1f}",
        evaluated_keys=[
            (rec_id, t.track_id) for t in present if t.t_start >= t_a and t.t_end <= t_b
        ],
    )


def homogeneous_episodes(
    data,
    min_length: float = MIN_HOMOGENEOUS_INTERVAL,
    max_vehicle_length: Optional[float] = None,
) -> list:
    """Homogeneous episodes of all intervals of at least `min_length` seconds with an evaluated vehicle;
    with `max_vehicle_length` [m], longer cars and vans end the intervals like trucks and buses."""
    episodes = []
    for _, recording in sorted(data.items()):
        for t_a, t_b in homogeneous_intervals(recording, max_vehicle_length):
            if t_b - t_a < min_length:
                continue
            spec = homogeneous_episode(recording, t_a, t_b, max_vehicle_length)
            if spec.evaluated_keys:
                episodes.append(spec)
    return episodes


def build_episode(
    spec: EpisodeSpec,
    data,
    routes,
    start_repair=None,
    is_recorded_vehicles: bool = False,
):
    """Controlled and replayed vehicles of an episode, the latest time the episode can last, and the
    keys of the vehicles removed because of a bad start.

    `start_repair`: optional `StartPoseRepair` for the start poses of the controlled vehicles. A vehicle
    whose start cannot be repaired (bad recording) is removed from the episode: it is neither controlled
    nor replayed, and it is not compared with its human. `is_recorded_vehicles`: the controlled vehicles
    have the parameters of the recorded ones (see `controlled_vehicle`).
    """
    recording = data[spec.recording_id]
    by_key = {(spec.recording_id, t.track_id): t for t in recording.tracks}
    controlled, removed = [], []
    for k in spec.controlled_keys:
        # Homogeneous: vehicles already on the road are controlled from their recorded states at the
        # interval start
        t_begin = (
            max(spec.t0, by_key[k].t_start) if spec.kind == "homogeneous" else None
        )
        vehicle = controlled_vehicle(
            by_key[k],
            routes,
            t_begin=t_begin,
            start_repair=start_repair,
            is_recorded_vehicle=is_recorded_vehicles,
        )
        if vehicle.start_repair is not None and vehicle.start_repair.status == "failed":
            removed.append(k)
        else:
            controlled.append(vehicle)
    t_last = max(
        (v.t_begin + v.max_duration for v in controlled), default=float(spec.t0)
    )
    if spec.kind == "homogeneous":
        return controlled, [], t_last, removed
    controlled_set = set(spec.controlled_keys)
    others = [
        t
        for t in recording.tracks
        if t.cls not in IGNORED_CLASSES
        and (spec.recording_id, t.track_id) not in controlled_set
        and t.t_end >= spec.t0
        and t.t_start <= t_last
    ]
    replayed = build_replay_vehicles(others, routes)
    return controlled, replayed, t_last, removed


def max_concurrent_slots(spec: EpisodeSpec, data, routes, dt: float) -> int:
    """Upper bound of the agents needed by an episode (controlled vehicles count until their human left)."""
    controlled, replayed, t_last, _ = build_episode(spec, data, routes)
    times = np.arange(spec.t0, t_last + dt, dt)
    count = np.zeros(len(times), dtype=int)
    for v in replayed:
        count += (times >= v.track.t_start) & (times <= v.track.t_end)
    for v in controlled:
        # Controlled vehicles may be slower than the human; count them for twice the human duration
        duration = v.track.t_end - v.t_begin
        count += (times >= v.t_begin) & (times <= v.t_begin + 2 * duration + 10)
    return int(count.max())
