# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Gap acceptance at the entries of the rounD roundabout (location 0): the recorded human drivers and the
fine-tuned policies with the decentralized BH-CBF filter of Stage 3 in the same 112 homogeneous windows.

Geometry. The ring is circulated counterclockwise. The conflict point of an entering vehicle is the merge
point of its route: the first route point within MERGE_RADIUS (23 m, the center of the outer ring lane) of
the ring center, at the ring angle theta*. A vehicle circulates while its route progress lies between the
merge and the exit point of its route (both at MERGE_RADIUS). The arc distance of a circulating vehicle j
to the conflict point is RING_RADIUS times the counterclockwise angle from j to theta*, and its time to
arrival (TTA) is the arc distance over its speed averaged over the last SPEED_WINDOW seconds. j is a true
conflict if its route reaches theta* before it leaves the ring, and a false conflict if it leaves the ring
at an exit before theta* (it approaches the entry but turns off; a human cannot know this in advance, the
BH-CBF filter knows it from the broadcast route).

Decisions. An entering vehicle i makes its first decision when its route progress is D_REF before its
merge point ("lag" event). The nearest circulating true conflict j (smallest arc distance, up to half a
ring) is the lag vehicle; i accepts the lag if it reaches its merge point before j reaches theta*. If i
rejects, its next decision is when j passes theta* ("gap" event), against the next true conflict, and so
on (Raff's lags and gaps). Each event also records the nearest false conflict and the kinematics of both
vehicles (stopping of i, speed loss and delay of j). Events after the window end are dropped, since the
recording then contains road users that the simulation does not have.

Stops. For every step at which an entering vehicle waits (speed below WAIT_SPEED within APPROACH m before
its merge point), the situation is classified: queued behind a vehicle of its own entry lane, a true
conflict within the TTA threshold, only false conflicts within the threshold, or none of them.

Trips. For every vehicle compared with its human in the paper (whole recorded trip inside the window),
the time before its merge point (from its first position) and after it (to its last position) show where
the policies gain and lose time. Both add up to the travel time of the trip.

Writes to <out>/: events.csv (one row per decision), waits.csv (one row per waiting step), trips.csv (one
row per evaluated trip through the ring), and geometry.json. The analysis and figures: stage3_gap_acceptance_plots.py.

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.stage3_gap_acceptance [--d_ref 10] [--speed_window 1.0]
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import pickle

import numpy as np

from sigmarl.evaluation_human_comparison.metrics import MERGE_RADIUS, RING_CENTER

RESULTS = "outputs/tits26/stage3_human_ft_recorded_maxlen6.5"
# Worlds: the recorded humans and the ten policies (pre-trained on the roundabout: "natural", on the
# intersection: "transfer"; fine-tuned on the roundabout with the BH-CBF filter in the loop)
POLICY_LABEL = (
    "{map}__stage1_ttc_seed_{seed}__iter100__ft-round_location0-ours-seed1__iter100__ours_dec"
)
POLICY_GROUPS = {"natural": "round_location0", "transfer": "ind_location4"}
DT = 0.1  # [s]
RING_RADIUS = 20.5  # [m] Mean radius of the ring part of the routes
MAX_ANGLE = math.pi  # Circulating vehicles up to half a ring upstream of the conflict point count
D_REF = 11.3  # [m] Route distance before the merge point at which the first decision is made (where the
# median speed of the recorded humans is lowest)
SPEED_WINDOW = 1.0  # [s] Speeds for the TTA are averaged over this past window
MIN_SPEED = 0.1  # [m/s] Lower bound of the averaged speed in the TTA
STOP_SPEED = 0.5  # [m/s] An entering vehicle below this speed has stopped
WAIT_SPEED = 1.0  # [m/s] Waiting steps (stop attribution)
APPROACH = 20.0  # [m] Route distance before the merge point in which waiting steps count
QUEUE_DISTANCE = 15.0  # [m] A vehicle of the same entry lane this far ahead makes i queued
EXIT_TOLERANCE = math.radians(10.0)  # Ring angle beyond the exit point that still counts as exiting


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--d_ref", type=float, default=D_REF)
    parser.add_argument("--speed_window", type=float, default=SPEED_WINDOW)
    parser.add_argument("--out", default="outputs/tits26/stage3_gap_acceptance")
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def ring_angle(x, y):
    return np.arctan2(np.asarray(y) - RING_CENTER[1], np.asarray(x) - RING_CENTER[0])


def ccw(a, b):
    """Counterclockwise angle in [0, 2 pi) from angle a to angle b."""
    return np.mod(b - a, 2 * np.pi)


def route_geometry(routes) -> dict:
    """Merge and exit point (progress, ring angle) of every route through the ring; bypasses are None."""
    geometry = {}
    for idx, route in enumerate(routes):
        s = np.arange(0.0, route.length, 0.1)
        xy = route.xy_at(s)
        inside = np.flatnonzero(np.hypot(*(xy - RING_CENTER).T) <= MERGE_RADIUS)
        if len(inside) == 0:
            geometry[idx] = None
            continue
        a, b = inside[0], inside[-1]
        geometry[idx] = dict(
            s_merge=float(s[a]),
            s_exit=float(s[b]),
            theta_merge=float(ring_angle(*xy[a])),
            theta_exit=float(ring_angle(*xy[b])),
        )
    # Entry lanes: routes with the same merge point share the entry lane
    lanes = sorted({round(math.degrees(g["theta_merge"])) for g in geometry.values() if g})
    for g in geometry.values():
        if g:
            g["lane"] = lanes.index(round(math.degrees(g["theta_merge"])))
    return geometry


def human_trajectories(spec, recording):
    """Recorded trajectories of the vehicles of a window on the simulation grid (k = (t - t0) / DT)."""
    tracks = {(spec.recording_id, t.track_id): t for t in recording.tracks}
    vehicles = []
    for key in spec.controlled_keys:
        track = tracks[key]
        k = np.arange(
            math.ceil((track.t_start - spec.t0) / DT - 1e-6),
            math.floor((track.t_end - spec.t0) / DT + 1e-6) + 1,
        )
        t = spec.t0 + k * DT
        vehicles.append(
            dict(
                key=key[1],
                route=track.match.route_idx,
                k=k,
                x=np.interp(t, track.t, track.xy[:, 0]),
                y=np.interp(t, track.t, track.xy[:, 1]),
                v=np.interp(t, track.t, track.speed),
                s=np.interp(t, track.t, track.match.s),
            )
        )
    return vehicles


def policy_trajectories(result):
    """Simulated trajectories of the controlled vehicles of an episode on the simulation grid."""
    spec = result["spec"]
    vehicles = []
    for vehicle in result["vehicles"]:
        log = vehicle["log"]
        if vehicle["t_spawn"] is None or len(log["t"]) < 2:
            continue
        vehicles.append(
            dict(
                key=vehicle["key"][1],
                route=vehicle["path_id"],
                k=np.round((np.asarray(log["t"]) - spec.t0) / DT).astype(int),
                x=np.asarray(log["x"], float),
                y=np.asarray(log["y"], float),
                v=np.asarray(log["speed"], float),
                s=np.asarray(log["s"], float),  # Progress along its route (as the human's match)
            )
        )
    return vehicles


def grid(vehicles, speed_window: float):
    """Arrays [n, K] (NaN where absent) of the vehicles on a common grid starting at k_min."""
    k_min = min(int(v["k"][0]) for v in vehicles)
    k_max = max(int(v["k"][-1]) for v in vehicles)
    n, n_k = len(vehicles), k_max - k_min + 1
    arrays = {name: np.full((n, n_k), np.nan) for name in ("x", "y", "v", "s", "va")}
    window = max(1, int(round(speed_window / DT)))
    for row, v in enumerate(vehicles):
        cols = v["k"] - k_min
        for name in ("x", "y", "v", "s"):
            arrays[name][row, cols] = v[name]
        # Causal mean speed over the last `window` samples
        cumsum = np.concatenate([[0.0], np.cumsum(v["v"])])
        idx = np.arange(len(v["v"]))
        lo = np.maximum(0, idx + 1 - window)
        arrays["va"][row, cols] = (cumsum[idx + 1] - cumsum[lo]) / (idx + 1 - lo)
    arrays["theta"] = ring_angle(arrays["x"], arrays["y"])
    # Unwrapped ring angle per vehicle (for the passing of a ring angle)
    theta_u = np.full((n, n_k), np.nan)
    for row in range(n):
        ok = ~np.isnan(arrays["theta"][row])
        theta_u[row, ok] = np.unwrap(arrays["theta"][row, ok])
    arrays["theta_u"] = theta_u
    return k_min, arrays


def first_index(mask, start: int = 0):
    idx = np.flatnonzero(mask[start:])
    return None if len(idx) == 0 else start + int(idx[0])


def episode_rows(world, spec, vehicles, geometry, d_ref, speed_window):
    """Decision events and waiting steps of all entering vehicles of one episode."""
    vehicles = [v for v in vehicles if geometry.get(v["route"]) is not None]
    if not vehicles:
        return [], [], []
    k_min, a = grid(vehicles, speed_window)
    n, n_k = a["x"].shape
    k_a = -k_min  # Grid column of the window start
    k_b = int(round(spec.window / DT)) - k_min  # ... and of its end
    geo = [geometry[v["route"]] for v in vehicles]
    s_merge = np.array([g["s_merge"] for g in geo])
    s_exit = np.array([g["s_exit"] for g in geo])
    theta_exit = np.array([g["theta_exit"] for g in geo])
    lane = np.array([g["lane"] for g in geo])
    present = ~np.isnan(a["x"])
    circulating = present & (a["s"] >= s_merge[:, None]) & (a["s"] <= s_exit[:, None])
    events, waits, trips = [], [], []
    evaluated = {key[1] for key in spec.evaluated_keys}

    def situation(i, k, theta_star):
        """Nearest circulating true and false conflict of i at column k: (row, angle) or (None, nan)."""
        rows = np.flatnonzero(circulating[:, k])
        rows = rows[rows != i]
        delta = ccw(a["theta"][rows, k], theta_star)
        remaining = ccw(a["theta"][rows, k], theta_exit[rows])
        # A vehicle slightly past the angle of its exit point (lateral offset) is about to exit
        remaining = np.where(remaining > 2 * np.pi - EXIT_TOLERANCE, 0.0, remaining)
        keep = delta <= MAX_ANGLE
        rows, delta, remaining = rows[keep], delta[keep], remaining[keep]
        out = []
        for is_true in (True, False):
            sel = (delta <= remaining) if is_true else (delta > remaining)
            if sel.any():
                m = int(np.argmin(np.where(sel, delta, np.inf)))
                out.append((int(rows[m]), float(delta[m])))
            else:
                out.append((None, math.nan))
        return out

    def tta(j, k, delta):
        return RING_RADIUS * delta / max(a["va"][j, k], MIN_SPEED)

    def queued(i, k):
        """Whether a vehicle of i's entry lane that has not merged yet is close ahead of i."""
        d_i = s_merge[i] - a["s"][i, k]
        d = s_merge - a["s"][:, k]
        ahead = present[:, k] & (lane == lane[i]) & (d >= 0) & (d < d_i) & (d_i - d < QUEUE_DISTANCE)
        ahead[i] = False
        return bool(ahead.any())

    for i in range(n):
        s_i = a["s"][i]
        ok = present[i]
        k_first = first_index(ok)
        if vehicles[i]["key"] in evaluated and s_i[k_first] < s_merge[i]:
            k_merge = first_index(ok & (s_i >= s_merge[i]))
            k_last = int(np.flatnonzero(ok)[-1])
            if k_merge is not None:
                trips.append(dict(
                    world=world, episode=spec.name, i=vehicles[i]["key"], route_i=vehicles[i]["route"],
                    before_time=(k_merge - k_first) * DT, after_time=(k_last - k_merge) * DT,
                    before_distance=s_merge[i] - s_i[k_first],
                ))  # fmt: skip
        if s_i[k_first] > s_merge[i] - d_ref:
            continue  # Already past its first decision when it appears (or taken over in the ring)
        k_ref = first_index(ok & (s_i >= s_merge[i] - d_ref))
        k_arr = first_index(ok & (s_i >= s_merge[i]))
        if k_ref is None or k_arr is None:
            continue
        theta_star = geo[i]["theta_merge"]
        base = dict(world=world, episode=spec.name, i=vehicles[i]["key"], route_i=vehicles[i]["route"],
                    lane_i=int(lane[i]))  # fmt: skip

        # Waiting steps in the approach zone
        approach = ok & (s_i >= s_merge[i] - APPROACH) & (s_i < s_merge[i])
        for k in np.flatnonzero(approach & (a["v"][i] < WAIT_SPEED)):
            if not k_a <= k <= k_b:
                continue
            (j, dj), (f, df) = situation(i, k, theta_star)
            waits.append(dict(
                base, t=round(spec.t0 + (k + k_min) * DT, 2), queued=queued(i, k),
                tta_true=tta(j, k, dj) if j is not None else math.nan,
                tta_false=tta(f, k, df) if f is not None else math.nan,
            ))  # fmt: skip

        # Decision events: the lag at k_ref, then the gaps after each rejected vehicle
        k_e, kind = k_ref, "lag"
        while k_e < k_arr:
            if not k_a <= k_e <= k_b:
                break
            (j, dj), (f, df) = situation(i, k_e, theta_star)
            row = dict(
                base, kind=kind, t=round(spec.t0 + (k_e + k_min) * DT, 2), queued=queued(i, k_e),
                v_i=a["v"][i, k_e], dist_i=s_merge[i] - s_i[k_e],
                stopped_i=bool(a["v"][i, k_e] < STOP_SPEED),
                t_arr_i=(k_arr - k_e) * DT,
                min_v_i=float(np.nanmin(a["v"][i, k_e:k_arr + 1])),
                stops_i=bool(np.nanmin(a["v"][i, k_e:k_arr + 1]) < STOP_SPEED),
            )  # fmt: skip
            if f is not None:
                k_exit = first_index(present[f] & (a["s"][f] > s_exit[f]), k_e)
                row.update(
                    f=vehicles[f]["key"], arc_f=RING_RADIUS * df, v_f=a["va"][f, k_e],
                    tta_f=tta(f, k_e, df),
                    t_exit_f=math.nan if k_exit is None else (k_exit - k_e) * DT,
                )  # fmt: skip
            if j is None:
                row.update(has_conflict=False)
                events.append(row)
                break
            # Arrival of j at theta*: its unwrapped ring angle has advanced by the angle to theta*
            passed = present[j] & (a["theta_u"][j] - a["theta_u"][j, k_e] >= dj - 1e-9)
            k_arr_j = first_index(passed, k_e)
            v_j = a["v"][j]
            k_end = k_arr_j if k_arr_j is not None else first_index(~present[j], k_e)
            k_end = n_k - 1 if k_end is None else k_end
            row.update(
                has_conflict=True, j=vehicles[j]["key"], route_j=vehicles[j]["route"],
                arc=RING_RADIUS * dj, v_j=v_j[k_e], va_j=a["va"][j, k_e],
                tta=tta(j, k_e, dj), tta_inst=RING_RADIUS * dj / max(v_j[k_e], MIN_SPEED),
                t_arr_j=math.nan if k_arr_j is None else (k_arr_j - k_e) * DT,
                speed_loss_j=float(v_j[k_e] - np.nanmin(v_j[k_e:k_end + 1])),
                accept=(k_arr_j is None) or (k_arr < k_arr_j),
                f_closer=(f is not None) and df < dj,
            )  # fmt: skip
            events.append(row)
            if k_arr_j is None or k_arr < k_arr_j:
                break
            k_e, kind = k_arr_j, "gap"
    return events, waits, trips


def load_world(job):
    """Rows of one world: ("human", None) or (group, label)."""
    world, group, label, d_ref, speed_window = job
    from sigmarl.parse_lanelet2 import ParseLanelet2
    from sigmarl.round_tracks import RouteMatcher

    parser = ParseLanelet2("round_location0", "cpu")
    routes = RouteMatcher(parser.reference_paths).routes
    geometry = route_geometry(routes)
    with open(os.path.join(RESULTS, "natural", "matched_tracks.pkl"), "rb") as file:
        data = pickle.load(file)
    # The windows (specs) are the same for all policies; take them from the first natural policy
    folder = os.path.join(RESULTS, group or "natural", "homogeneous",
                          label or POLICY_LABEL.format(map="round_location0", seed=1))  # fmt: skip
    events, waits, trips = [], [], []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".pkl"):
            continue
        with open(os.path.join(folder, name), "rb") as file:
            result = pickle.load(file)
        spec = result["spec"]
        if group is None:
            vehicles = human_trajectories(spec, data[spec.recording_id])
        else:
            vehicles = policy_trajectories(result)
        e, w, t = episode_rows(world, spec, vehicles, geometry, d_ref, speed_window)
        events += e
        waits += w
        trips += t
    return events, waits, trips


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    jobs = [("human", None, None, args.d_ref, args.speed_window)]
    for group, map_name in POLICY_GROUPS.items():
        for seed in range(1, 6):
            label = POLICY_LABEL.format(map=map_name, seed=seed)
            jobs.append((f"{group}_{seed}", group, label, args.d_ref, args.speed_window))
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(load_world, jobs)
    events = [row for e, _, _ in results for row in e]
    waits = [row for _, w, _ in results for row in w]
    trips = [row for _, _, t in results for row in t]
    os.makedirs(args.out, exist_ok=True)
    suffix = "" if (args.d_ref, args.speed_window) == (D_REF, SPEED_WINDOW) else (
        f"_d{args.d_ref:g}_w{args.speed_window:g}")  # fmt: skip
    write_csv(os.path.join(args.out, f"events{suffix}.csv"), events)
    write_csv(os.path.join(args.out, f"waits{suffix}.csv"), waits)
    write_csv(os.path.join(args.out, f"trips{suffix}.csv"), trips)
    from sigmarl.parse_lanelet2 import ParseLanelet2
    from sigmarl.round_tracks import RouteMatcher

    routes = RouteMatcher(ParseLanelet2("round_location0", "cpu").reference_paths).routes
    with open(os.path.join(args.out, "geometry.json"), "w") as file:
        json.dump({str(k): v for k, v in route_geometry(routes).items()}, file, indent=1)
    for world in dict.fromkeys(r["world"] for r in events):
        rows = [r for r in events if r["world"] == world]
        n_conf = sum(r["has_conflict"] for r in rows)
        print(f"[INFO] {world}: {len(rows)} events ({n_conf} with a conflict), "
              f"{sum(r['kind'] == 'lag' for r in rows)} entering vehicles, "
              f"{sum(w['world'] == world for w in waits)} waiting steps, "
              f"{sum(t['world'] == world for t in trips)} evaluated trips")  # fmt: skip


if __name__ == "__main__":
    main()
