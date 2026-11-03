# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Per-vehicle metrics of controlled vehicles and of the human drivers they replace.

Both trajectories are evaluated with the same code at the simulation rate (10 Hz): the human trajectory is
the recorded track resampled to 10 Hz, the controller trajectory is the simulator log. Accelerations,
yaw rates, and jerks are finite differences of the 10 Hz speed and yaw for both.

Geometry: controlled vehicles have the simulated footprint (2.0 m x 4.6 m); human drivers and replayed
road users have their recorded footprints.
"""

import math

import numpy as np

from sigmarl.constants import AGENTS

DT = 0.1
# Roundabout geometry of rounD location 0 in map coordinates (fitted to the ring lanelets)
RING_CENTER = np.array([78.18, 55.64])
RING_ENTRY_RADIUS = 26.0  # [m] Outer edge of the circulatory roadway (yield line)
MERGE_RADIUS = 23.0  # [m] Center of the outer ring lane, where entering vehicles merge
CONFLICT_RADIUS = (
    3.5  # [m] Others passing this close to the merge point count for the PET
)
CONFLICT_MIN_HEADING_DIFF = math.radians(
    30
)  # Others with a similar heading are leaders, not conflicts
PET_CAP = 10.0  # [s]
TTC_HORIZON = 5.0  # [s]
TTC_CRITICAL = 1.5  # [s]
NEIGHBOR_RANGE = (
    40.0  # [m] Others farther than this are skipped in gap/TTC computations
)
STOP_SPEED = 0.5  # [m/s]
DEEP_DEPARTURE = (
    0.5  # [m] Boundary crossings deeper than this count as deep lane departures
)
HARSH_LON_ACC = 3.0  # [m/s^2] |longitudinal acceleration| above this counts as harsh
HARSH_LAT_ACC = 3.0  # [m/s^2]


# ---------------------------------------------------------------------------- geometry


def rect_corners(x, y, yaw, length, width):
    """Corners [..., 4, 2] of oriented rectangles (counterclockwise)."""
    x, y, yaw, length, width = np.broadcast_arrays(x, y, yaw, length, width)
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2, width / 2
    local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
    lx = local[:, 0] * hl[..., None]
    ly = local[:, 1] * hw[..., None]
    cx = x[..., None] + c[..., None] * lx - s[..., None] * ly
    cy = y[..., None] + s[..., None] * lx + c[..., None] * ly
    return np.stack([cx, cy], axis=-1)


def rects_overlap(a, b):
    """Separating-axis test for rectangles a, b [..., 4, 2]; returns bool [...]."""
    overlap = np.ones(a.shape[:-2], dtype=bool)
    for poly in (a, b):
        for k in range(2):
            edge = poly[..., k + 1, :] - poly[..., k, :]
            axis = np.stack([-edge[..., 1], edge[..., 0]], axis=-1)
            pa = np.einsum("...ij,...j->...i", a, axis)
            pb = np.einsum("...ij,...j->...i", b, axis)
            overlap &= (pa.max(-1) >= pb.min(-1)) & (pb.max(-1) >= pa.min(-1))
    return overlap


def _point_segment_distance(p, a, b):
    ab = b - a
    t = np.clip(
        np.einsum("...k,...k->...", p - a, ab)
        / np.maximum(np.einsum("...k,...k->...", ab, ab), 1e-12),
        0.0,
        1.0,
    )
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def rect_distance(a, b):
    """Minimum distance between rectangles a, b [..., 4, 2] (0 if they overlap)."""
    d = np.full(a.shape[:-2], np.inf)
    for p_poly, e_poly in ((a, b), (b, a)):
        for i in range(4):
            for k in range(4):
                d = np.minimum(
                    d,
                    _point_segment_distance(
                        p_poly[..., i, :],
                        e_poly[..., k, :],
                        e_poly[..., (k + 1) % 4, :],
                    ),
                )
    return np.where(rects_overlap(a, b), 0.0, d)


def segments_intersect(p1, p2, q1, q2):
    """Whether segments p1-p2 and q1-q2 intersect (broadcast over leading dims)."""

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
            b[..., 1] - a[..., 1]
        ) * (c[..., 0] - a[..., 0])

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return (o1 * o2 <= 0) & (o3 * o4 <= 0)


# ---------------------------------------------------------------------------- trajectories


def human_trajectory(track):
    """Recorded track resampled to 10 Hz: dict with t, x, y, yaw, speed."""
    t = np.arange(track.t_start, track.t_end + 1e-9, DT)
    return {
        "t": t,
        "x": np.interp(t, track.t, track.xy[:, 0]),
        "y": np.interp(t, track.t, track.xy[:, 1]),
        "yaw": np.interp(t, track.t, track.heading),
        "speed": np.interp(t, track.t, track.speed),
    }


def controller_trajectory(log):
    return {
        "t": log["t"],
        "x": log["x"],
        "y": log["y"],
        "yaw": np.unwrap(log["yaw"]),
        "speed": log["speed"],
    }


def kinematics(traj):
    """Longitudinal and lateral acceleration, longitudinal jerk, and yaw rate by finite differences."""
    if len(traj["t"]) < 3:
        z = np.zeros(len(traj["t"]))
        return {"lon_acc": z, "lat_acc": z, "jerk": z, "yaw_rate": z}
    lon_acc = np.gradient(traj["speed"], DT)
    yaw_rate = np.gradient(traj["yaw"], DT)
    return {
        "lon_acc": lon_acc,
        "lat_acc": traj["speed"] * yaw_rate,
        "jerk": np.gradient(lon_acc, DT),
        "yaw_rate": yaw_rate,
    }


# ---------------------------------------------------------------------------- others


class SceneOthers:
    """States of the other road users at the 10 Hz times of one trajectory.

    For human drivers: all other recorded road users (except pedestrians). For controlled vehicles: the
    episode snapshots (other controlled vehicles from the simulator, replayed ones from the record).
    """

    def __init__(self, times, others):
        # others: dict key -> dict(t [T_k], x, y, yaw, speed, length, width, is_controlled)
        self.times = times
        self.others = others

    @classmethod
    def from_recording(cls, recording, ego_key, times):
        others = {}
        t0, t1 = times[0], times[-1]
        for track in recording.tracks:
            key = (recording.recording_id, track.track_id)
            if key == ego_key or track.cls == "pedestrian":
                continue
            if track.t_end < t0 or track.t_start > t1:
                continue
            t = times[(times >= track.t_start - 1e-9) & (times <= track.t_end + 1e-9)]
            if len(t) == 0:
                continue
            others[key] = {
                "t": t,
                "x": np.interp(t, track.t, track.xy[:, 0]),
                "y": np.interp(t, track.t, track.xy[:, 1]),
                "yaw": np.interp(t, track.t, track.heading),
                "speed": np.interp(t, track.t, track.speed),
                "length": track.length,
                "width": track.width,
                "is_controlled": False,
            }
        return cls(times, others)

    @classmethod
    def from_snapshots(
        cls, snapshots, recording, ego_index, times, controlled_sizes=None
    ):
        """controlled_sizes: footprint (length, width) [m] of each controlled vehicle by its index; None:
        the nominal one for all."""
        sizes = {t.track_id: (t.length, t.width) for t in recording.tracks}
        rows = {}
        for t, entries in snapshots:
            for kind, idx, x, y, yaw, speed in entries:
                if kind == "c" and idx == ego_index:
                    continue
                rows.setdefault((kind, idx), []).append((t, x, y, yaw, speed))
        others = {}
        t_set = np.round(times / DT).astype(int)
        for (kind, idx), values in rows.items():
            arr = np.array(values)
            keep = np.isin(np.round(arr[:, 0] / DT).astype(int), t_set)
            arr = arr[keep]
            if len(arr) == 0:
                continue
            if kind == "c":
                length, width = (
                    (AGENTS["length"], AGENTS["width"])
                    if controlled_sizes is None
                    else controlled_sizes[idx]
                )
            else:
                length, width = sizes[idx]
            others[(kind, idx)] = {
                "t": arr[:, 0],
                "x": arr[:, 1],
                "y": arr[:, 2],
                "yaw": arr[:, 3],
                "speed": arr[:, 4],
                "length": length,
                "width": width,
                "is_controlled": kind == "c",
            }
        return cls(times, others)

    def aligned(self):
        """Arrays [K, T] of the others' states on the ego times (NaN where absent)."""
        keys = list(self.others)
        n_t = len(self.times)
        out = {k: np.full((len(keys), n_t), np.nan) for k in ("x", "y", "yaw", "speed")}
        length = np.zeros(len(keys))
        width = np.zeros(len(keys))
        is_controlled = np.zeros(len(keys), dtype=bool)
        for row, key in enumerate(keys):
            other = self.others[key]
            cols = np.round((other["t"] - self.times[0]) / DT).astype(int)
            valid = (cols >= 0) & (cols < n_t)
            for k in ("x", "y", "yaw", "speed"):
                out[k][row, cols[valid]] = other[k][valid]
            length[row], width[row] = other["length"], other["width"]
            is_controlled[row] = other["is_controlled"]
        return keys, out, length, width, is_controlled


# ---------------------------------------------------------------------------- metrics


def interaction_metrics(traj, ego_length, ego_width, scene: SceneOthers):
    """Collisions, minimum gap, TTC, and collision attribution against the other road users."""
    keys, o, o_len, o_wid, o_ctrl = scene.aligned()
    n_t = len(traj["t"])
    result = {
        "collision": False,
        "collision_at_fault": False,
        "collision_with_controlled": False,
        "collision_time": np.nan,
        "min_gap": np.inf,
        "min_ttc": np.inf,
        "critical_ttc_time": 0.0,
    }
    if not keys:
        return result
    ego = rect_corners(
        traj["x"], traj["y"], traj["yaw"], ego_length, ego_width
    )  # [T, 4, 2]
    present = ~np.isnan(o["x"])
    dist_c = np.hypot(o["x"] - traj["x"][None], o["y"] - traj["y"][None])
    near = present & (dist_c < NEIGHBOR_RANGE)
    if not near.any():
        return result
    rows, cols = np.nonzero(near)
    other = rect_corners(
        o["x"][rows, cols],
        o["y"][rows, cols],
        o["yaw"][rows, cols],
        o_len[rows],
        o_wid[rows],
    )
    gap = rect_distance(ego[cols], other)
    result["min_gap"] = float(gap.min())

    hit = gap <= 0.0
    if hit.any():
        first = np.argmin(np.where(hit, cols, n_t + 1))
        k, row = int(cols[first]), int(rows[first])
        result["collision"] = True
        result["collision_time"] = float(traj["t"][k] - traj["t"][0])
        result["collision_with_controlled"] = bool(o_ctrl[row])
        result["collision_at_fault"] = _is_at_fault(traj, k, o, row)

    # Time to collision with constant speed and heading of both, sampled every 0.1 s
    horizon = np.arange(DT, TTC_HORIZON + 1e-9, DT)
    ttc = np.full(len(rows), np.inf)
    ego_x = (
        traj["x"][cols][:, None]
        + traj["speed"][cols][:, None] * np.cos(traj["yaw"][cols])[:, None] * horizon
    )
    ego_y = (
        traj["y"][cols][:, None]
        + traj["speed"][cols][:, None] * np.sin(traj["yaw"][cols])[:, None] * horizon
    )
    ox = (
        o["x"][rows, cols][:, None]
        + o["speed"][rows, cols][:, None]
        * np.cos(o["yaw"][rows, cols])[:, None]
        * horizon
    )
    oy = (
        o["y"][rows, cols][:, None]
        + o["speed"][rows, cols][:, None]
        * np.sin(o["yaw"][rows, cols])[:, None]
        * horizon
    )
    ego_f = rect_corners(
        ego_x,
        ego_y,
        np.broadcast_to(traj["yaw"][cols][:, None], ego_x.shape),
        ego_length,
        ego_width,
    )
    oth_f = rect_corners(
        ox,
        oy,
        np.broadcast_to(o["yaw"][rows, cols][:, None], ox.shape),
        o_len[rows][:, None],
        o_wid[rows][:, None],
    )
    future_hit = rects_overlap(ego_f, oth_f) & ~hit[:, None]
    has_hit = future_hit.any(axis=1)
    ttc[has_hit] = horizon[np.argmax(future_hit[has_hit], axis=1)]
    if np.isfinite(ttc).any():
        result["min_ttc"] = float(ttc.min())
        ttc_per_step = np.full(n_t, np.inf)
        np.minimum.at(ttc_per_step, cols, ttc)
        result["critical_ttc_time"] = float(np.sum(ttc_per_step < TTC_CRITICAL) * DT)
    return result


def _is_at_fault(traj, k, o, row) -> bool:
    """Collision attribution: the ego is not at fault if it stood still or was hit from behind."""
    if traj["speed"][k] < STOP_SPEED:
        return False
    dx = o["x"][row, k] - traj["x"][k]
    dy = o["y"][row, k] - traj["y"][k]
    yaw = traj["yaw"][k]
    longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
    heading_diff = abs((o["yaw"][row, k] - yaw + math.pi) % (2 * math.pi) - math.pi)
    is_rear_end_by_other = longitudinal < 0 and heading_diff < math.radians(45)
    return not is_rear_end_by_other


def lane_departure(traj, ego_length, ego_width, left_boundary, right_boundary):
    """Time [s] during which the footprint crosses a boundary of the route (shared boundaries)."""
    corners = rect_corners(
        traj["x"], traj["y"], traj["yaw"], ego_length, ego_width
    )  # [T, 4, 2]
    edges_a = corners
    edges_b = np.roll(corners, -1, axis=1)
    crossing = np.zeros(len(traj["t"]), dtype=bool)
    for boundary in (left_boundary, right_boundary):
        q1, q2 = boundary[:-1], boundary[1:]  # [S, 2]
        # Only boundary segments near the vehicle
        center = np.stack([traj["x"], traj["y"]], axis=-1)
        mid = (q1 + q2) / 2
        near = np.linalg.norm(center[:, None] - mid[None], axis=-1) < 15.0  # [T, S]
        hit = segments_intersect(
            edges_a[:, :, None, :],
            edges_b[:, :, None, :],
            q1[None, None],
            q2[None, None],
        )  # [T, 4, S]
        crossing |= (hit & near[:, None, :]).any(axis=(1, 2))
    return float(crossing.sum() * DT)


def entry_metrics(traj, scene: SceneOthers):
    """Yielding behavior at the roundabout entry and post-encroachment time (PET) with circulating traffic.

    The PET is the time between the ego and another road user with a different heading (circulating
    traffic) passing the point where the ego merges into the outer ring lane, capped at PET_CAP.
    """
    dist = np.hypot(traj["x"] - RING_CENTER[0], traj["y"] - RING_CENTER[1])
    result = {
        "enters_ring": False,
        "entry_speed": np.nan,
        "approach_min_speed": np.nan,
        "stopped_before_entry": False,
        "wait_time_before_entry": np.nan,
        "entry_pet": np.nan,
    }
    inside = np.flatnonzero(dist <= RING_ENTRY_RADIUS)
    if len(inside) == 0 or inside[0] == 0:
        return result
    k = int(inside[0])
    result["enters_ring"] = True
    result["entry_speed"] = float(traj["speed"][k])
    approach = slice(max(0, k - int(5.0 / DT)), k + 1)
    result["approach_min_speed"] = float(traj["speed"][approach].min())
    result["stopped_before_entry"] = bool((traj["speed"][: k + 1] < STOP_SPEED).any())
    result["wait_time_before_entry"] = float(np.sum(traj["speed"][: k + 1] < 1.0) * DT)

    merge = np.flatnonzero(dist <= MERGE_RADIUS)
    if len(merge) == 0:
        return result
    k = int(merge[0])
    p_e = np.array([traj["x"][k], traj["y"][k]])
    t_e = traj["t"][k]
    yaw_e = traj["yaw"][k]
    pet = PET_CAP
    for other in scene.others.values():
        d = np.hypot(other["x"] - p_e[0], other["y"] - p_e[1])
        close = d <= CONFLICT_RADIUS
        if not close.any():
            continue
        heading_diff = np.abs(
            (other["yaw"][close] - yaw_e + np.pi) % (2 * np.pi) - np.pi
        )
        crossing = heading_diff >= CONFLICT_MIN_HEADING_DIFF
        if crossing.any():
            pet = min(pet, float(np.abs(other["t"][close][crossing] - t_e).min()))
    result["entry_pet"] = pet
    return result


def path_deviation(traj_c, traj_h):
    """Distances between the controller and the human trajectories.

    ADE/FDE: time-aligned (same time since the human appeared) over the common duration.
    Path deviation: mean distance from the controller's positions to the human's path (time-free).
    """
    n = min(len(traj_c["t"]), len(traj_h["t"]))
    pc = np.stack([traj_c["x"], traj_c["y"]], -1)
    ph = np.stack([traj_h["x"], traj_h["y"]], -1)
    d = np.linalg.norm(pc[:n] - ph[:n], axis=-1)
    seg_d = _point_segment_distance(
        pc[:, None, :], ph[None, :-1, :], ph[None, 1:, :]
    ).min(axis=1)
    return {
        "ade": float(d.mean()),
        "fde": float(d[-1]),
        "path_deviation": float(seg_d.mean()),
    }


def trajectory_metrics(
    traj, length, width, scene, route, left_boundary, right_boundary, s_goal
):
    """All per-vehicle metrics of one (human or controller) trajectory."""
    kin = kinematics(traj)
    _, lateral, _, _ = route.project(np.stack([traj["x"], traj["y"]], -1))
    m = {
        "mean_speed": float(np.mean(traj["speed"])),
        "max_speed": float(np.max(traj["speed"])),
        "rms_lon_acc": float(np.sqrt(np.mean(kin["lon_acc"] ** 2))),
        "rms_lat_acc": float(np.sqrt(np.mean(kin["lat_acc"] ** 2))),
        "rms_jerk": float(np.sqrt(np.mean(kin["jerk"] ** 2))),
        "p95_abs_lon_acc": float(np.percentile(np.abs(kin["lon_acc"]), 95)),
        "p95_abs_lat_acc": float(np.percentile(np.abs(kin["lat_acc"]), 95)),
        "p95_abs_jerk": float(np.percentile(np.abs(kin["jerk"]), 95)),
        "min_lon_acc": float(np.min(kin["lon_acc"])),
        "harsh_time": float(
            np.sum(
                (np.abs(kin["lon_acc"]) > HARSH_LON_ACC)
                | (np.abs(kin["lat_acc"]) > HARSH_LAT_ACC)
            )
            * DT
        ),
        "mean_abs_lateral_offset": float(np.mean(np.abs(lateral))),
        "lane_departure_time": lane_departure(
            traj, length, width, left_boundary, right_boundary
        ),
        # Footprint shrunk on every side: the vehicle crosses a boundary by more than DEEP_DEPARTURE
        "deep_lane_departure_time": lane_departure(
            traj,
            length - 2 * DEEP_DEPARTURE,
            width - 2 * DEEP_DEPARTURE,
            left_boundary,
            right_boundary,
        ),
        "stop_time": float(np.sum(traj["speed"] < STOP_SPEED) * DT),
    }
    m.update(interaction_metrics(traj, length, width, scene))
    m.update(entry_metrics(traj, scene))
    return m, kin
