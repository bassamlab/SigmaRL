# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run one controller on one episode of one rounD recording and compare it with the human drivers.

Homogeneous (`--mode homogeneous`, default): the episode is a homogeneous interval of the recording (only
controllable cars and vans on the map, all of them controlled, nothing replayed; see `episodes.py`): by
default the one with the most evaluated vehicles, or the one selected with `--interval` (`--list_intervals`
prints them). Closed loop (`--mode closed_loop`): all controllable vehicles that appear in a window are
controlled, all other road users are replayed; without `--t0`, the window is the one the batch evaluation
(`run.py`, same `--seed`) uses for this recording. Replay (`--mode replay --track <id>`): only that vehicle
is controlled.

Results (episode pickle, `vehicles.csv`, `summary.md`, and the videos) are written to
`<out_dir>/<episode>_<controller>/`. `--save-human-video` also saves a video of the recorded human drivers
of the episode, and `--overlay-human-video` outlines the humans in the controller's video. Example (run from
the repository root):

    SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_human_comparison.run_single \
        --controller cbf_qp_cen --recording 11 --interval 10 --save-video --save-human-video
"""

import argparse
import dataclasses
import os
import pickle
import time

from sigmarl.evaluation_human_comparison.run import (
    CONTROLLERS,
    CertifiedStartRepair,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL,
    ControllerRunner,
    _episode_result,
    add_entry_check_argument,
    add_max_vehicle_length_argument,
    add_vehicles_argument,
)

DEFAULT_TRACKS_CACHE = "outputs/human_comparison/matched_tracks.pkl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode",
        choices=("homogeneous", "closed_loop", "replay"),
        default="homogeneous",
        help="Episode type (default: homogeneous; replay needs --track).",
    )
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument(
        "--recording", type=int, required=True, help="Recording of location 0 (2-23)."
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=None,
        help="Closed loop: window start [s] (default: as in run.py).",
    )
    parser.add_argument(
        "--window", type=float, default=60.0, help="Closed loop: spawn window [s]."
    )
    parser.add_argument(
        "--track", type=int, default=None, help="Replay: the controlled track."
    )
    parser.add_argument(
        "--interval", type=int, default=None, help="Homogeneous: index of the interval."
    )
    parser.add_argument(
        "--list_intervals",
        action="store_true",
        help="Print the homogeneous intervals and exit.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the window sampling of run.py."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--tracks_cache",
        default=DEFAULT_TRACKS_CACHE,
        help="Cache of the matched tracks.",
    )
    parser.add_argument("--out_dir", default="outputs/human_comparison_single")
    parser.add_argument(
        "--n_slots",
        type=int,
        default=None,
        help="Agents of the env (default: from the episode).",
    )
    parser.add_argument(
        "--render-video",
        dest="render_video",
        action="store_true",
        help="Show the run in a window.",
    )
    parser.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        help="Save the run as a video.",
    )
    parser.add_argument(
        "--save-human-video",
        dest="save_human_video",
        action="store_true",
        help="Also save a video that replays the recorded human drivers of the episode.",
    )
    parser.add_argument(
        "--overlay-human-video",
        dest="overlay_human_video",
        action="store_true",
        help="Outline the recorded human drivers in the controller's video (and window), each in the color "
        "of the vehicle that replaces it.",
    )
    parser.add_argument(
        "--veh_veh_cbf_footprint",
        choices=("rectangle_sat", "circle"),
        default="rectangle_sat",
        help="Vehicle-vehicle CBF footprint of the CBF-QP controllers.",
    )
    parser.add_argument(
        "--no_shield_priority",
        action="store_true",
        help="CBF-QP controllers: resolve shield conflicts without priority (the centralized shield brakes "
        "all violating vehicles, the decentralized one prioritizes lower indices; default: arrival order).",
    )
    parser.add_argument(
        "--no_start_repair",
        action="store_true",
        help="Start at the recorded poses even if the simulated footprint crosses the route boundary "
        "(default: repair them, see start_repair.py).",
    )
    add_entry_check_argument(parser)
    add_vehicles_argument(parser)
    add_max_vehicle_length_argument(parser)
    return parser.parse_args(argv)


def episode_spec(args, data):
    """The episode of the selected window or track."""
    from sigmarl.evaluation_human_comparison.episodes import (
        EpisodeSpec,
        closed_loop_episodes,
        homogeneous_episode,
        homogeneous_episodes,
        homogeneous_intervals,
        is_controllable,
    )

    recording = data[args.recording]
    assert (
        args.interval is None or args.mode == "homogeneous"
    ), "--interval selects a homogeneous interval."
    assert (args.track is not None) == (
        args.mode == "replay"
    ), "--track is required in replay mode and only used there."
    assert (
        args.t0 is None or args.mode == "closed_loop"
    ), "--t0 is the window start of the closed-loop mode."
    if args.mode == "homogeneous":
        if args.interval is not None:
            intervals = homogeneous_intervals(recording, args.max_vehicle_length)
            assert (
                0 <= args.interval < len(intervals)
            ), f"Recording {args.recording} has {len(intervals)} intervals."
            return homogeneous_episode(
                recording, *intervals[args.interval], args.max_vehicle_length
            )
        specs = homogeneous_episodes(
            {args.recording: recording}, max_vehicle_length=args.max_vehicle_length
        )
        assert (
            specs
        ), f"Recording {args.recording} has no homogeneous interval long enough (see --list_intervals)."
        return max(specs, key=lambda s: len(s.evaluated_keys))
    if args.mode == "replay":
        track = next(t for t in recording.tracks if t.track_id == args.track)
        assert is_controllable(
            track
        ), f"Track {args.track} is not controllable (class, route, or speed)."
        return EpisodeSpec(
            kind="replay",
            recording_id=args.recording,
            t0=track.t_start,
            controlled_keys=[(args.recording, track.track_id)],
            window=0.0,
            name=f"rec{args.recording:02d}_track{track.track_id:04d}",
        )
    if args.t0 is None:
        specs = closed_loop_episodes(
            data, args.window, n_per_recording=1, seed=args.seed
        )
        return next(s for s in specs if s.recording_id == args.recording)
    keys = [
        (args.recording, t.track_id)
        for t in recording.tracks
        if is_controllable(t) and args.t0 <= t.t_start < args.t0 + args.window
    ]
    assert (
        keys
    ), f"No controllable vehicle appears in [{args.t0}, {args.t0 + args.window}) s."
    return EpisodeSpec(
        kind="closed_loop",
        recording_id=args.recording,
        t0=args.t0,
        controlled_keys=keys,
        window=args.window,
        name=f"rec{args.recording:02d}_t{args.t0:06.1f}",
    )


def print_intervals(recording, max_vehicle_length=None):
    """Table of the homogeneous intervals of a recording."""
    from sigmarl.evaluation_human_comparison.episodes import (
        MIN_HOMOGENEOUS_INTERVAL,
        homogeneous_episode,
        homogeneous_intervals,
    )

    print(
        f"Homogeneous intervals of recording {recording.recording_id:02d} (used by run.py: length >= "
        f"{MIN_HOMOGENEOUS_INTERVAL:.0f} s and at least one evaluated vehicle):"
    )
    print(" index    start      end   length  controlled  evaluated  used")
    for k, (t_a, t_b) in enumerate(
        homogeneous_intervals(recording, max_vehicle_length)
    ):
        spec = homogeneous_episode(recording, t_a, t_b, max_vehicle_length)
        n_eval = len(spec.evaluated_keys)
        is_used = t_b - t_a >= MIN_HOMOGENEOUS_INTERVAL and n_eval > 0
        print(
            f"{k:6d} {t_a:8.1f} {t_b:8.1f} {t_b - t_a:8.1f} {len(spec.controlled_keys):11d} {n_eval:10d}  "
            f"{'yes' if is_used else ''}"
        )


def human_video_frames(runner, spec, controlled, replayed, dt):
    """Frames of the recorded human drivers of an episode in the runner's environment.

    Call it after the controller's run. The humans of the controlled vehicles are outlined (unfilled, as in
    `--overlay-human-video`) in the colors their vehicles had in that run; the replayed road users (other
    modes than homogeneous) are replayed as in the run. From the episode start until the last of them leaves
    the recording.
    """
    import torch
    from torchrl.envs.utils import step_mdp

    from sigmarl.log_replay import LogReplay

    t_end = max(v.track.t_end for v in controlled + replayed)
    log_replay = LogReplay(
        t0=spec.t0,
        dt=dt,
        n_slots=runner.parameters.n_agents,
        controlled=controlled,
        replayed=[dataclasses.replace(v, slot=None) for v in replayed],
        is_record_snapshots=False,
        is_humans_only=True,
    )
    runner.scenario.log_replay = log_replay
    frames = []
    with torch.inference_mode():
        td = runner.env.reset()
        n_steps = 0
        while log_replay.t < t_end and n_steps < runner.parameters.max_steps - 2:
            obs = td.get(("agents", "observation"))
            # Parked and replayed agents ignore their actions
            td.set(
                ("agents", "action"), torch.zeros(*obs.shape[:-1], 2, dtype=obs.dtype)
            )
            td = step_mdp(runner.env.step(td))
            frames.append(runner._render(log_replay))
            n_steps += 1
    runner.scenario.log_replay = None
    if log_replay.n_replay_steps_dropped:
        print(
            f"[WARNING] Human video: {log_replay.n_replay_steps_dropped} vehicle steps not shown (too few agents)."
        )
    return frames


def main(argv=None):
    args = parse_args(argv)
    assert not args.overlay_human_video or (
        args.save_video or args.render_video
    ), "--overlay-human-video needs --save-video or --render-video."
    assert (
        os.environ.get("SIGMARL_AGENT_PROFILE") == "full_scale"
    ), "Set SIGMARL_AGENT_PROFILE=full_scale."
    from types import SimpleNamespace

    from sigmarl.evaluation_human_comparison import analyze
    from sigmarl.evaluation_human_comparison.episodes import (
        build_episode,
        load_location0,
        max_concurrent_slots,
    )
    from sigmarl.evaluation_human_comparison.start_repair import StartPoseRepair
    from sigmarl.helper_common import save_video
    from sigmarl.log_replay import LogReplay
    from sigmarl.parse_lanelet2 import ParseLanelet2
    from sigmarl.render_backend import resolve_renderer
    from sigmarl.round_tracks import ROUND_LOCATION0_RECORDINGS, RouteMatcher

    assert (
        args.recording in ROUND_LOCATION0_RECORDINGS
    ), "Location 0 has recordings 2-23."
    if args.render_video and resolve_renderer() == "matplotlib":
        print(
            "[WARNING] No display found (or SIGMARL_RENDERER=matplotlib): --render-video shows no window."
        )
    parser = ParseLanelet2("round_location0", "cpu")
    routes = RouteMatcher(parser.reference_paths).routes
    data = load_location0(args.data_dir, parser, args.tracks_cache)
    if args.list_intervals:
        print_intervals(data[args.recording], args.max_vehicle_length)
        return
    spec = episode_spec(args, data)
    start_repair = (
        None if args.no_start_repair else StartPoseRepair(parser.reference_paths)
    )
    controlled, replayed, t_last, removed = build_episode(
        spec, data, routes, start_repair, args.vehicles == "recorded"
    )
    if removed:
        print(
            f"[INFO] Removed for a bad start (start repair failed): tracks {[k[1] for k in removed]}."
        )
    dt = 0.1
    n_slots = args.n_slots or max_concurrent_slots(spec, data, routes, dt) + 2
    print(
        f"[INFO] Episode {spec.name} ({spec.kind}): {len(controlled)} controlled, "
        f"{len({v.track.track_id for v in replayed})} replayed road users, {n_slots} agents."
    )
    if spec.evaluated_keys is not None:
        print(
            f"[INFO] Interval [{spec.t0:.1f}, {spec.t0 + spec.window:.1f}] s: {len(spec.evaluated_keys)} "
            f"vehicles are compared with the humans, the others give traffic."
        )

    run_dir = os.path.join(args.out_dir, f"{spec.name}_{args.controller}")
    result_dir = os.path.join(run_dir, spec.kind, args.controller)
    os.makedirs(result_dir, exist_ok=True)

    runner = ControllerRunner(
        args.controller,
        args.model_path,
        n_slots,
        int((t_last - spec.t0) / dt) + 20,
        args.veh_veh_cbf_footprint,
        not args.no_shield_priority,
    )
    entry_check = runner.entry_check(args.entry_check)
    log_replay = LogReplay(
        t0=spec.t0,
        dt=dt,
        n_slots=n_slots,
        controlled=controlled,
        replayed=replayed,
        entry_check=entry_check,
        entry_repair=(
            None
            if entry_check is None or start_repair is None
            else CertifiedStartRepair(entry_check, start_repair)
        ),
        is_overlay_humans=args.overlay_human_video,
    )
    t_start = time.time()
    timing, frames = runner.run_episode(
        log_replay, is_video=args.save_video, is_show_window=args.render_video
    )
    result = _episode_result(spec, log_replay, timing, removed, args.vehicles)
    result["wall_time"] = time.time() - t_start
    with open(os.path.join(result_dir, f"{spec.name}.pkl"), "wb") as f:
        pickle.dump(result, f)
    if args.save_video:
        suffix = "_human_overlay" if args.overlay_human_video else ""
        video_path = os.path.join(
            run_dir, f"video_{spec.name}_{args.controller}{suffix}"
        )
        save_video(video_path, frames, fps=1 / dt)
        print(f"[INFO] Video saved to {video_path}.mp4")
    if args.save_human_video:
        video_path = os.path.join(run_dir, f"video_{spec.name}_human")
        save_video(
            video_path,
            human_video_frames(runner, spec, controlled, replayed, dt),
            fps=1 / dt,
        )
        print(f"[INFO] Human video saved to {video_path}.mp4")

    # Metrics of this run and of the human drivers it replaced
    eval_args = SimpleNamespace(
        out_dir=run_dir,
        mode=spec.kind,
        controllers=args.controller,
        common_only=False,
        data_dir=args.data_dir,
    )
    rows, samples, timing_all, controllers = analyze.evaluate(eval_args, data=data)
    analyze._write_csv(rows, os.path.join(run_dir, "vehicles.csv"))
    summary = analyze.summarize(rows, samples, timing_all, controllers)
    md = analyze.summary_markdown(summary, spec.kind)
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"[INFO] Wall time {result['wall_time']:.1f} s. Results in {run_dir}")


if __name__ == "__main__":
    main()
