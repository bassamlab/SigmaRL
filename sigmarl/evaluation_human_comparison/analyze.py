# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compute metrics for the results of `run.py` and compare the controllers with the human drivers.

Writes to `<out_dir>/<mode>/`:
- `vehicles.csv`: one row per controlled vehicle and controller (controller and human metrics);
- `summary.json` and `summary.md`: aggregated metrics per controller and for the human drivers;
- `fig_*.pdf/png`: distributions and paired comparisons.

Example:

    SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_human_comparison.analyze --controllers cbf_qp_dec
"""

import argparse
import csv
import json
import math
import os
import pickle

import numpy as np

CONTROLLER_LABELS = {
    "human": "Human",
    "marl": "MARL",
    "marl_cbf": "MARL + CBF (dec.)",
    "cbf_qp_dec": "CLF-CBF-QP (dec.)",
    "cbf_qp_cen": "CLF-CBF-QP (cen.)",
}
DECENTRALIZED = ("marl_cbf", "cbf_qp_dec")


def controller_label(controller: str) -> str:
    """Display name; other output folders (run.py --label) are shown as is."""
    return CONTROLLER_LABELS.get(controller, controller)


def is_decentralized(controller: str) -> bool:
    """Fully decentralized controllers, incl. run.py labels with "_dec"."""
    return controller in DECENTRALIZED or "_dec" in controller


# Row statuses of vehicles without a simulated trajectory
NOT_SIMULATED = ("not_spawned", "removed_bad_start")
# Simulation step of run.py [s]
DT = 0.1
# Per-step timing keys of run.py; the last two (joint backup value) are missing in older results
TIMING_KEYS = (
    "policy",
    "cbf_seq",
    "cbf_par",
    "n_controlled",
    "env_step",
    "backup_value_min",
    "n_outside_c",
)


def entry_delay(t_spawn: float, t_begin: float, t0: float) -> float:
    """Delay [s] of a spawn beyond the first simulation step at or after the recorded appearance `t_begin`
    (the spawn delay without the rounding to the step grid of the log replay)."""
    k_first = math.ceil((t_begin - 1e-6 - t0) / DT - 1e-9)
    return t_spawn - (t0 + k_first * DT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode",
        choices=("replay", "closed_loop", "homogeneous"),
        default="homogeneous",
        help="Episode type (default: homogeneous).",
    )
    parser.add_argument("--controllers", default="marl,marl_cbf,cbf_qp_dec,cbf_qp_cen")
    parser.add_argument("--out_dir", default="outputs/human_comparison")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--common_only",
        action="store_true",
        help="Only use episodes for which all controllers have results.",
    )
    return parser.parse_args(argv)


def _load_results(directory):
    results = {}
    if not os.path.isdir(directory):
        return results
    for name in sorted(os.listdir(directory)):
        if name.endswith(".pkl"):
            with open(os.path.join(directory, name), "rb") as f:
                results[name[:-4]] = pickle.load(f)
    return results


def controlled_size(vehicle: dict) -> tuple:
    """Footprint (length, width) [m] of a controlled vehicle of a result: its own (`--vehicles recorded`), or
    the nominal one of the agent profile (also for results without vehicle parameters)."""
    from sigmarl.constants import AGENTS

    params = vehicle.get("params")
    if params is None:
        return AGENTS["length"], AGENTS["width"]
    return params["length"], params["width"]


def evaluate(args, data=None):
    """Metrics of all results under `<args.out_dir>/<args.mode>/`; `data` are the matched recordings (loaded
    from the cache in `args.out_dir` if None)."""
    from sigmarl.parse_lanelet2 import ParseLanelet2
    from sigmarl.round_tracks import RouteMatcher
    from sigmarl.evaluation_human_comparison.episodes import load_location0
    from sigmarl.evaluation_human_comparison.metrics import (
        SceneOthers,
        controller_trajectory,
        human_trajectory,
        path_deviation,
        trajectory_metrics,
    )

    parser = ParseLanelet2("round_location0", "cpu")
    routes = RouteMatcher(parser.reference_paths).routes
    boundaries = [
        (
            p["left_boundary_shared"].numpy().astype(float),
            p["right_boundary_shared"].numpy().astype(float),
        )
        for p in parser.reference_paths
    ]
    if data is None:
        data = load_location0(
            args.data_dir, parser, os.path.join(args.out_dir, "matched_tracks.pkl")
        )
    mode_dir = os.path.join(args.out_dir, args.mode)
    controllers = [
        c
        for c in args.controllers.split(",")
        if os.path.isdir(os.path.join(mode_dir, c))
    ]
    results = {c: _load_results(os.path.join(mode_dir, c)) for c in controllers}
    if args.common_only and controllers:
        common = set.intersection(*(set(r) for r in results.values()))
        results = {
            c: {k: v for k, v in r.items() if k in common} for c, r in results.items()
        }

    human_cache = {}
    rows = []
    samples = {}
    timing = {}

    def add_samples(name, traj, kin):
        entry = samples.setdefault(
            name, {k: [] for k in ("speed", "lon_acc", "lat_acc", "jerk")}
        )
        entry["speed"].append(traj["speed"])
        for k in ("lon_acc", "lat_acc", "jerk"):
            entry[k].append(kin[k])

    for controller in controllers:
        n_eps = len(results[controller])
        print(f"[INFO] Evaluating {controller}: {n_eps} episodes.")
        tim = timing.setdefault(controller, {k: [] for k in TIMING_KEYS})
        for ep_name, result in results[controller].items():
            for k in tim:
                if k in result["timing"]:
                    tim[k].append(result["timing"][k])
            spec = result["spec"]
            recording = data[spec.recording_id]
            tracks = {(spec.recording_id, t.track_id): t for t in recording.tracks}
            for key in result["removed_bad_start"]:
                # Removed from the episode (start repair failed); traffic-only vehicles are listed too
                rows.append(
                    {
                        "mode": args.mode,
                        "controller": controller,
                        "episode": ep_name,
                        "recording": key[0],
                        "track": key[1],
                        "status": "removed_bad_start",
                        "evaluated": spec.is_evaluated(key),
                    }
                )
            for idx, v in enumerate(result["vehicles"]):
                key = tuple(v["key"])
                if not spec.is_evaluated(key):
                    continue  # Homogeneous episodes: traffic only, its human trip is not fully inside
                track = tracks[key]
                route = routes[v["path_id"]]
                left, right = boundaries[v["path_id"]]
                if key not in human_cache:
                    traj_h = human_trajectory(track)
                    scene_h = SceneOthers.from_recording(recording, key, traj_h["t"])
                    m_h, kin_h = trajectory_metrics(
                        traj_h,
                        track.length,
                        track.width,
                        scene_h,
                        route,
                        left,
                        right,
                        v["s_goal"],
                    )
                    m_h["travel_time"] = track.t_end - track.t_start
                    human_cache[key] = (traj_h, m_h, kin_h)
                traj_h, m_h, kin_h = human_cache[key]
                if v["t_spawn"] is None or len(v["log"]["t"]) < 2:
                    rows.append(
                        {
                            "mode": args.mode,
                            "controller": controller,
                            "episode": ep_name,
                            "recording": key[0],
                            "track": key[1],
                            "status": "not_spawned",
                        }
                    )
                    continue
                traj_c = controller_trajectory(v["log"])
                scene_c = SceneOthers.from_snapshots(
                    result["snapshots"],
                    recording,
                    idx,
                    traj_c["t"],
                    [controlled_size(other) for other in result["vehicles"]],
                )
                m_c, kin_c = trajectory_metrics(
                    traj_c,
                    *controlled_size(v),
                    scene_c,
                    route,
                    left,
                    right,
                    v["s_goal"],
                )
                add_samples(controller, traj_c, kin_c)
                row = {
                    "mode": args.mode,
                    "controller": controller,
                    "episode": ep_name,
                    "recording": key[0],
                    "track": key[1],
                    "class": track.cls,
                    "route": v["path_id"],
                    "route_distance": float(v["s_goal"] - track.match.s[0]),
                    "status": v["status"],
                    "success": v["status"] == "goal",
                    "spawn_delay": float(v["t_spawn"] - track.t_start),
                    # Evaluated vehicles start with their tracks (whole trip inside the window)
                    "entry_delay": entry_delay(
                        float(v["t_spawn"]), float(track.t_start), float(spec.t0)
                    ),
                    "travel_time": (
                        float(v["t_finish"] - track.t_start)
                        if v["status"] == "goal"
                        else math.nan
                    ),
                    "h_travel_time": m_h["travel_time"],
                }
                repair = v["start_repair"]
                if repair is not None:
                    # Initial-state repair (start_repair.py)
                    row.update(
                        {
                            "start_repair": repair["status"],
                            # Moved at the entry so that its braking backup stays in the lane
                            "start_backup_repair": bool(
                                repair.get("is_backup_repair", False)
                            ),
                            "start_barrier_recorded": repair["barrier_recorded"],
                            "start_shift": repair["shift"],
                            "start_dyaw_deg": math.degrees(repair["dyaw"]),
                        }
                    )
                row.update(path_deviation(traj_c, traj_h))
                row.update({f"c_{k}": val for k, val in m_c.items()})
                row.update(
                    {f"h_{k}": val for k, val in m_h.items() if k != "travel_time"}
                )
                rows.append(row)

    # Human samples once per vehicle
    for traj_h, _, kin_h in human_cache.values():
        add_samples("human", traj_h, kin_h)
    samples = {
        n: {k: np.concatenate(v) for k, v in s.items()} for n, s in samples.items()
    }
    timing = {
        c: {k: np.concatenate(v) if v else np.array([]) for k, v in t.items()}
        for c, t in timing.items()
    }
    return rows, samples, timing, controllers


def _write_csv(rows, path):
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _f(values):
    return np.asarray([v for v in values], dtype=float)


def summarize(rows, samples, timing, controllers):
    from scipy import stats

    summary = {}
    for controller in controllers:
        rs = [
            r
            for r in rows
            if r["controller"] == controller and r["status"] not in NOT_SIMULATED
        ]
        n_not_spawned = sum(
            r["controller"] == controller and r["status"] == "not_spawned" for r in rows
        )
        removed = [
            r
            for r in rows
            if r["controller"] == controller and r["status"] == "removed_bad_start"
        ]
        if not rs:
            continue
        ok = [r for r in rs if r["success"]]
        tt_c, tt_h = _f([r["travel_time"] for r in ok]), _f(
            [r["h_travel_time"] for r in ok]
        )
        diff = tt_c - tt_h
        s = {
            "n_vehicles": len(rs),
            "n_not_spawned": n_not_spawned,
            # Removed for a bad start: evaluated vehicles, and all controlled ones (incl. traffic only)
            "n_removed_bad_start": sum(r["evaluated"] for r in removed),
            "n_removed_bad_start_all": len(removed),
        }
        for prefix, name in (("c_", controller), ("h_", "human")):
            group = {}
            group["collision_rate"] = float(
                np.mean([r[prefix + "collision"] for r in rs])
            )
            group["at_fault_collision_rate"] = float(
                np.mean([r[prefix + "collision_at_fault"] for r in rs])
            )
            group["collision_with_controlled_rate"] = float(
                np.mean([r[prefix + "collision_with_controlled"] for r in rs])
            )
            group["lane_departure_rate"] = float(
                np.mean([r[prefix + "lane_departure_time"] > 0 for r in rs])
            )
            group["mean_lane_departure_time"] = float(
                np.mean([r[prefix + "lane_departure_time"] for r in rs])
            )
            group["deep_lane_departure_rate"] = float(
                np.mean([r[prefix + "deep_lane_departure_time"] > 0 for r in rs])
            )
            for k in (
                "mean_speed",
                "max_speed",
                "rms_lon_acc",
                "rms_lat_acc",
                "rms_jerk",
                "p95_abs_lon_acc",
                "p95_abs_lat_acc",
                "p95_abs_jerk",
                "mean_abs_lateral_offset",
                "stop_time",
            ):
                group[f"median_{k}"] = float(np.median(_f([r[prefix + k] for r in rs])))
            group["mean_harsh_time"] = float(
                np.mean([r[prefix + "harsh_time"] for r in rs])
            )
            gaps = _f([r[prefix + "min_gap"] for r in rs])
            group["median_min_gap"] = float(np.median(gaps))
            group["p5_min_gap"] = float(np.percentile(gaps, 5))
            group["min_gap_below_0.5m_rate"] = float(np.mean(gaps < 0.5))
            group["ttc_below_1.5s_rate"] = float(
                np.mean([r[prefix + "min_ttc"] < 1.5 for r in rs])
            )
            group["mean_critical_ttc_time"] = float(
                np.mean([r[prefix + "critical_ttc_time"] for r in rs])
            )
            entering = [r for r in rs if r[prefix + "enters_ring"]]
            if entering:
                pet = _f([r[prefix + "entry_pet"] for r in entering])
                pet = pet[
                    ~np.isnan(pet)
                ]  # Vehicles that leave the recording before the merge point
                group["n_entering"] = len(entering)
                group["stop_before_entry_rate"] = float(
                    np.mean([r[prefix + "stopped_before_entry"] for r in entering])
                )
                group["median_entry_speed"] = float(
                    np.median(_f([r[prefix + "entry_speed"] for r in entering]))
                )
                group["median_wait_before_entry"] = float(
                    np.median(
                        _f([r[prefix + "wait_time_before_entry"] for r in entering])
                    )
                )
                group["median_entry_pet"] = float(np.median(pet))
                group["entry_pet_below_1.5s_rate"] = float(np.mean(pet < 1.5))
            if prefix == "c_":
                group["success_rate"] = len(ok) / len(rs)
                group["timeout_rate"] = float(
                    np.mean([r["status"] == "timeout" for r in rs])
                )
                group["median_travel_time"] = (
                    float(np.median(tt_c)) if len(ok) else math.nan
                )
                group["median_spawn_delay"] = float(
                    np.median(_f([r["spawn_delay"] for r in rs]))
                )
                group["mean_spawn_delay"] = float(
                    np.mean(_f([r["spawn_delay"] for r in rs]))
                )
                spawn_delay = _f([r["spawn_delay"] for r in rs])
                group["p95_spawn_delay"] = float(np.percentile(spawn_delay, 95))
                group["max_spawn_delay"] = float(spawn_delay.max())
                # Entries delayed by the entry check (or by a lack of free agents)
                delays = _f([r["entry_delay"] for r in rs])
                is_delayed = delays > 0.5 * DT
                group["delayed_entry_rate"] = float(np.mean(is_delayed))
                group["n_delayed_entries"] = int(is_delayed.sum())
                if is_delayed.any():
                    group["median_entry_delay_of_delayed"] = float(
                        np.median(delays[is_delayed])
                    )
                    group["mean_entry_delay_of_delayed"] = float(
                        np.mean(delays[is_delayed])
                    )
                group["median_ade"] = float(np.median(_f([r["ade"] for r in rs])))
                group["median_fde"] = float(np.median(_f([r["fde"] for r in rs])))
                group["median_path_deviation"] = float(
                    np.median(_f([r["path_deviation"] for r in rs]))
                )
                repaired = [r for r in rs if "start_repair" in r]
                if repaired:
                    group["start_backup_repaired_rate"] = float(
                        np.mean([r["start_backup_repair"] for r in repaired])
                    )
                    group["start_repaired_rate"] = float(
                        np.mean([r["start_repair"] == "repaired" for r in repaired])
                    )
                if len(ok):
                    group["median_travel_time_diff"] = float(np.median(diff))
                    group["mean_travel_time_diff"] = float(np.mean(diff))
                    group["faster_than_human_rate"] = float(np.mean(diff < 0))
                    group["travel_time_ratio_median"] = float(np.median(tt_c / tt_h))
                    if len(ok) > 10 and np.any(diff != 0):
                        group["travel_time_wilcoxon_p"] = float(
                            stats.wilcoxon(tt_c, tt_h).pvalue
                        )
            else:
                group["median_travel_time"] = (
                    float(np.median(tt_h)) if len(ok) else math.nan
                )
                group["median_travel_time_all"] = float(
                    np.median(_f([r["h_travel_time"] for r in rs]))
                )
            s[name] = group
        # Distribution distances to the human samples of the same vehicles
        dist = {}
        for k in ("speed", "lon_acc", "lat_acc", "jerk"):
            c_s, h_s = samples[controller][k], samples["human"][k]
            if k != "speed":
                c_s, h_s = np.abs(c_s), np.abs(h_s)
            dist[k] = {
                "wasserstein": float(stats.wasserstein_distance(c_s, h_s)),
                "ks": float(stats.ks_2samp(c_s, h_s).statistic),
            }
        s["distribution_distance_to_human"] = dist
        tim = timing[controller]
        active = tim["n_controlled"] > 0
        n_ctrl = np.maximum(tim["n_controlled"][active], 1)
        t_s = {
            "policy_ms_mean": float(1e3 * tim["policy"][active].mean()),
            "env_step_ms_mean": float(1e3 * tim["env_step"][active].mean()),
            "mean_controlled_per_step": float(tim["n_controlled"][active].mean()),
        }
        if controller != "marl":
            t_s["cbf_sequential_ms_mean"] = float(1e3 * tim["cbf_seq"][active].mean())
            t_s["cbf_sequential_ms_median"] = float(
                1e3 * np.median(tim["cbf_seq"][active])
            )
            t_s["cbf_sequential_ms_p95"] = float(
                1e3 * np.percentile(tim["cbf_seq"][active], 95)
            )
            t_s["cbf_sequential_ms_per_controlled_agent"] = float(
                1e3 * np.mean(tim["cbf_seq"][active] / n_ctrl)
            )
        if is_decentralized(controller):
            t_s["cbf_parallelized_ms_mean"] = float(1e3 * tim["cbf_par"][active].mean())
            t_s["cbf_parallelized_ms_median"] = float(
                1e3 * np.median(tim["cbf_par"][active])
            )
            t_s["cbf_parallelized_ms_p95"] = float(
                1e3 * np.percentile(tim["cbf_par"][active], 95)
            )
        s["computation_time"] = t_s
        if len(tim["backup_value_min"]):
            # Joint backup value of the controlled vehicles (run.py, with our margins)
            has_controlled = tim["n_controlled"] > 0
            s["certified_set"] = {
                "n_agent_steps": int(tim["n_controlled"].sum()),
                "n_agent_steps_outside": int(tim["n_outside_c"].sum()),
                "n_steps_outside": int((tim["n_outside_c"] > 0).sum()),
                "min_backup_value": float(
                    tim["backup_value_min"][has_controlled].min(initial=np.inf)
                ),
            }
        summary[controller] = s
    return summary


def _fmt(value, digits=2, pct=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if pct:
        return f"{100 * value:.1f}%"
    return f"{value:.{digits}f}"


def summary_markdown(summary, mode):
    controllers = list(summary)
    if not controllers:
        return ""
    header = (
        "| Metric | Human | "
        + " | ".join(controller_label(c) for c in controllers)
        + " |"
    )
    lines = [f"## {mode}", "", header, "|" + "---|" * (len(controllers) + 2)]
    human = summary[controllers[0]][
        "human"
    ]  # Same vehicles for all controllers when run with --common_only

    def row(label, key, digits=2, pct=False, human_key=None):
        values = [
            _fmt(human.get(human_key or key), digits, pct) if human_key != "" else "–"
        ]
        values += [_fmt(summary[c][c].get(key), digits, pct) for c in controllers]
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.append(
        "| Vehicles | "
        + " | ".join(
            [str(summary[controllers[0]]["n_vehicles"])]
            + [str(summary[c]["n_vehicles"]) for c in controllers]
        )
        + " |"
    )
    lines.append(
        "| Removed (bad start): evaluated / all controlled | – | "
        + " | ".join(
            f"{summary[c]['n_removed_bad_start']} / {summary[c]['n_removed_bad_start_all']}"
            for c in controllers
        )
        + " |"
    )
    row("Success (goal reached)", "success_rate", pct=True, human_key="")
    row("Timeout", "timeout_rate", pct=True, human_key="")
    row("Collision (any)", "collision_rate", pct=True)
    row("Collision (at fault)", "at_fault_collision_rate", pct=True)
    row("Lane boundary touched", "lane_departure_rate", pct=True)
    row("Lane boundary crossed by > 0.5 m", "deep_lane_departure_rate", pct=True)
    row("Travel time, median [s] (successful)", "median_travel_time", 1)
    row("Travel time - human, median [s]", "median_travel_time_diff", 1, human_key="")
    row("Faster than the human", "faster_than_human_rate", pct=True, human_key="")
    lines.append(
        "| Never entered (entry delayed until the timeout) | – | "
        + " | ".join(str(summary[c]["n_not_spawned"]) for c in controllers)
        + " |"
    )
    row("Spawn delay, mean [s]", "mean_spawn_delay", 2, human_key="")
    row("Spawn delay, median [s]", "median_spawn_delay", 2, human_key="")
    row("Spawn delay, max. [s]", "max_spawn_delay", 1, human_key="")
    row("Entries delayed", "delayed_entry_rate", pct=True, human_key="")
    row(
        "Entry delay of the delayed, median [s]",
        "median_entry_delay_of_delayed",
        1,
        human_key="",
    )
    row("Start pose repaired", "start_repaired_rate", pct=True, human_key="")
    row(
        "Start pose repaired for the braking backup",
        "start_backup_repaired_rate",
        pct=True,
        human_key="",
    )
    row("Mean speed, median [m/s]", "median_mean_speed")
    row("RMS long. acceleration, median [m/s²]", "median_rms_lon_acc")
    row("RMS lat. acceleration, median [m/s²]", "median_rms_lat_acc")
    row("RMS jerk, median [m/s³]", "median_rms_jerk")
    row("Time with abs. acc. > 3 m/s², mean [s]", "mean_harsh_time")
    row("Min. gap, median [m]", "median_min_gap")
    row("Min. gap, 5th percentile [m]", "p5_min_gap")
    row("Min. TTC < 1.5 s", "ttc_below_1.5s_rate", pct=True)
    row("Stops before ring entry", "stop_before_entry_rate", pct=True)
    row("Entry speed, median [m/s]", "median_entry_speed")
    row("Entry PET, median [s]", "median_entry_pet")
    row("Entry PET < 1.5 s", "entry_pet_below_1.5s_rate", pct=True)
    row("Lateral offset to center line, median [m]", "median_mean_abs_lateral_offset")
    row("ADE to human, median [m]", "median_ade", human_key="")
    row("Path deviation from human, median [m]", "median_path_deviation", human_key="")
    lines.append("")
    lines.append("Distribution distance to the human samples (Wasserstein / KS):")
    lines.append("")
    lines.append(
        "| Controller | speed | abs. long. acc. | abs. lat. acc. | abs. jerk |"
    )
    lines.append("|---|---|---|---|---|")
    for c in controllers:
        d = summary[c]["distribution_distance_to_human"]
        lines.append(
            f"| {controller_label(c)} | "
            + " | ".join(
                f"{d[k]['wasserstein']:.2f} / {d[k]['ks']:.2f}"
                for k in ("speed", "lon_acc", "lat_acc", "jerk")
            )
            + " |"
        )
    certified = [c for c in controllers if "certified_set" in summary[c]]
    if certified:
        from sigmarl.evaluation_human_comparison.run import OUTSIDE_C_TOL

        lines.append("")
        lines.append(
            "Certified set C (joint backup value B of the controlled vehicles at each step; outside: "
            f"B < -{OUTSIDE_C_TOL:g} m):"
        )
        lines.append("")
        lines.append(
            "| Controller | agent-steps | agent-steps outside C | steps outside C | min. B [m] |"
        )
        lines.append("|---|---|---|---|---|")
        for c in certified:
            cs = summary[c]["certified_set"]
            lines.append(
                f"| {controller_label(c)} | {cs['n_agent_steps']} | {cs['n_agent_steps_outside']} | "
                f"{cs['n_steps_outside']} | {cs['min_backup_value']:.3f} |"
            )
    lines.append("")
    lines.append(
        "Computation time per step (sequential = wall-clock loop over agents; parallelized = shared "
        "cache + slowest controlled agent's QP with shield round 1 + slowest controlled agent's shield "
        "round 2, an estimate for parallel execution):"
    )
    lines.append("")
    lines.append(
        "| Controller | controlled agents | policy [ms] | CBF sequential mean / median [ms] | "
        "CBF parallelized (est.) mean / median [ms] |"
    )
    lines.append("|---|---|---|---|---|")
    for c in controllers:
        t = summary[c]["computation_time"]
        lines.append(
            f"| {controller_label(c)} | {t['mean_controlled_per_step']:.1f} | {t['policy_ms_mean']:.1f} | "
            f"{_fmt(t.get('cbf_sequential_ms_mean'), 1)} / {_fmt(t.get('cbf_sequential_ms_median'), 1)} | "
            f"{_fmt(t.get('cbf_parallelized_ms_mean'), 1)} / {_fmt(t.get('cbf_parallelized_ms_median'), 1)} |"
        )
    return "\n".join(lines) + "\n"


def plot(rows, samples, controllers, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "human": "#222222",
        "marl": "#1f77b4",
        "marl_cbf": "#2ca02c",
        "cbf_qp_dec": "#d62728",
        "cbf_qp_cen": "#9467bd",
    }
    # Other output folders (run.py --label) in a fixed order; gray beyond eight
    others = [c for c in controllers if c not in colors]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    palette += ["#6250d6", "#e34948"]
    for k, controller in enumerate(others):
        colors[controller] = palette[k] if k < len(palette) else "#8a8984"
    names = ["human"] + controllers

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (k, label) in zip(
        axes,
        (
            ("speed", "speed [m/s]"),
            ("lon_acc", "|long. acc.| [m/s²]"),
            ("lat_acc", "|lat. acc.| [m/s²]"),
            ("jerk", "|jerk| [m/s³]"),
        ),
    ):
        for n in names:
            x = samples[n][k]
            x = np.sort(x if k == "speed" else np.abs(x))
            ax.plot(
                x,
                np.linspace(0, 1, len(x)),
                color=colors[n],
                label=controller_label(n),
            )
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        if k == "jerk":
            ax.set_xlim(0, np.percentile(np.abs(samples["human"][k]), 99.5) * 3)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_kinematics_cdf.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "fig_kinematics_cdf.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(controllers), figsize=(4 * len(controllers), 3.8), squeeze=False
    )
    for ax, c in zip(axes[0], controllers):
        rs = [r for r in rows if r["controller"] == c and r.get("success")]
        h = _f([r["h_travel_time"] for r in rs])
        t = _f([r["travel_time"] for r in rs])
        ax.scatter(h, t, s=6, alpha=0.5, color=colors[c])
        lim = [0, max(60, np.nanmax(t) if len(t) else 60)]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("human travel time [s]")
        ax.set_ylabel(f"{controller_label(c)} travel time [s]")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_travel_time_paired.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "fig_travel_time_paired.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (k, label, lim) in zip(
        axes,
        (
            ("min_gap", "min. gap to others [m]", 10),
            ("min_ttc", "min. TTC [s]", 5),
            ("entry_pet", "entry PET [s]", 10),
        ),
    ):
        for n in names:
            if n == "human":
                rs = [
                    r
                    for r in rows
                    if r["controller"] == controllers[0]
                    and r["status"] not in NOT_SIMULATED
                ]
                x = _f([r["h_" + k] for r in rs])
            else:
                rs = [
                    r
                    for r in rows
                    if r["controller"] == n and r["status"] not in NOT_SIMULATED
                ]
                x = _f([r["c_" + k] for r in rs])
            x = np.sort(np.clip(x[~np.isnan(x)], 0, lim))
            if len(x):
                ax.plot(
                    x,
                    np.linspace(0, 1, len(x)),
                    color=colors[n],
                    label=controller_label(n),
                )
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_safety_cdf.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "fig_safety_cdf.pdf"))
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    assert (
        os.environ.get("SIGMARL_AGENT_PROFILE") == "full_scale"
    ), "Set SIGMARL_AGENT_PROFILE=full_scale."
    if args.data_dir is None:
        from sigmarl.evaluation_human_comparison.run import DEFAULT_DATA_DIR

        args.data_dir = DEFAULT_DATA_DIR
    rows, samples, timing, controllers = evaluate(args)
    out_dir = os.path.join(args.out_dir, args.mode)
    _write_csv(rows, os.path.join(out_dir, "vehicles.csv"))
    summary = summarize(rows, samples, timing, controllers)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = summary_markdown(summary, args.mode)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    plot(rows, samples, controllers, out_dir)
    print(md)


if __name__ == "__main__":
    main()
