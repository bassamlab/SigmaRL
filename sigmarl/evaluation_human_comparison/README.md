# Controllers vs. Human Drivers on rounD Location 0

Can our controllers drive the vehicles of the rounD recordings better than the
humans who drove them? The scripts here start controlled vehicles at the
recorded states of human drivers, give them the humans' routes and goals, and
compare the resulting trajectories with the recorded ones. Run all commands
from the repository root with `SIGMARL_AGENT_PROFILE=full_scale`.

The method is described in detail in `docs/human_comparison_methodology.md`.

## Data

- rounD v1.0, location 0 (recordings 02-23, 25 Hz). The dataset may not be
  redistributed; request it from [levelXdata](https://levelxdata.com/round-dataset/). The default data directory
  is `~/projects/drone-dataset-tools/rounD_v1.0/data`; override it with
  `--data_dir` or the environment variable `ROUND_DATA_DIR`.
- `sigmarl/round_tracks.py` loads a recording, transforms the tracks into map
  coordinates (`ParseLanelet2.utm_to_map`), and matches each track to one of the
  40 reference paths of `round_location0` (every point within 4.5 m of the
  route's center line, heading within 50°, progress never going backward by
  more than 2 m; smallest mean lateral offset wins). 97.8% of the cars match a
  route.
- The goal of a controlled vehicle is the arc length along its route where the
  human left the recording; its timeout is 2x the human travel time + 30 s.
- Controlled vehicles: cars and vans with a matched route, at least 20 m of
  recorded route, and a mean speed of at least 1 m/s. Trucks, buses, trailers,
  motorcycles, and bicycles are always replayed; long vehicles are covered by
  several car-sized boxes. Pedestrians (12 in total) are ignored.

## Episode Types

- `replay`: one human-driven car is replaced by the controller; all other road
  users are replayed and do not react to it (non-reactive log replay). 20 cars
  per recording, 440 episodes.
- `closed_loop`: every controllable vehicle that appears in a 60 s window is
  replaced by the controller (spawned at its recorded state and time, delayed
  if the spawn spot is occupied); everyone else is replayed. One window per
  recording, 22 windows.
- `homogeneous`: a maximal interval in which only controllable cars and vans
  are on the map (no truck, trailer, bus, motorcycle, bicycle, pedestrian, or
  car/van without a route match, shorter than 20 m, or slower than 1 m/s on
  average). Every vehicle on the map is controlled, nothing is replayed;
  vehicles already on the road at the interval start are placed at their
  recorded states then. Only vehicles whose whole recorded trip lies in the
  interval are compared with the humans; the others are traffic only. All
  intervals of at least 20 s with one such vehicle: 102 episodes in 20
  recordings (none in 2 and 13), 1171 evaluated and 2567 controlled vehicles.
  With `--max_vehicle_length L`, cars and vans longer than L m count as heavy
  vehicles, so they end the intervals like trucks and buses (default: no
  limit).

Start poses whose simulated footprint crosses a route boundary (about 6%,
mostly bounding-box and map inaccuracies where vehicles enter the field of
view) are moved into the lane by the smallest lateral shift (at most 1.0 m;
`start_repair.py`, `--no_start_repair` to switch off); otherwise the lane CBF
would stop these vehicles at the start. A vehicle whose start cannot be
repaired is a bad recording and is removed from the simulation and the
comparison (counted as "Removed (bad start)" in the summary). `--episodes` runs selected episodes
by name.

The environment mechanics (spawning, replaying, parking agents) are in
`sigmarl/log_replay.py`; the scenario never resets agents while a log replay is
attached. Replayed vehicles are non-reactive: the CBF-QP gives the controlled
vehicle the full responsibility for pair constraints with them and predicts
them with zero control. All CBF controllers (`marl_cbf`, `cbf_qp_dec`,
`cbf_qp_cen`) use the exact rectangle-polyline lane barrier
(`lane_cbf_footprint="rectangle_sat"`), the exact rectangle-pair vehicle
barrier (`veh_veh_cbf_footprint="rectangle_sat"`; `--veh_veh_cbf_footprint circle` selects
the covering circles instead), and the exact backup shield. The shield's
guarantee does not cover replayed vehicles: the centralized shield predicts
them with zero control for one step and braking afterward, and the
decentralized shield does not check them.

Vehicle parameters (`--vehicles`): by default (`nominal`), every controlled
vehicle is the 4.6 m x 2.0 m car of the agent profile. With `recorded`, each
controlled vehicle has the recorded footprint of the vehicle it replaces, axle
distances scaled with its length (the wheelbase is not recorded), and the
acceleration limits of its class (cars [-6, 3], vans [-5, 2.5] m/s^2;
`episodes.CLASS_ACCELERATION`). The parameters apply to the plant, the
collision checks, the observations, and the filter (`sigmarl/vehicle_params.py`,
`ScenarioRoadTraffic.set_vehicle_params`); a slot gets them when its vehicle
enters and the nominal ones when it is parked. Recorded vehicles need the
fully decentralized BH-CBF filter (`--controller marl_filter --method ours
--filter_mode dec`); other controllers raise an error. The T-ITS Stage 3 uses
`recorded` (`sigmarl/evaluation_tits26/stage3_human_comparison.py`).

## Commands

```bash
# Controllers: marl, marl_cbf (decentralized CBF filter on the MARL policy),
# cbf_qp_dec (fully decentralized CLF-CBF-QP), cbf_qp_cen (centralized CLF-CBF-QP)
# --mode defaults to homogeneous in run.py, run_single.py, and analyze.py
python -m sigmarl.evaluation_human_comparison.run --controller cbf_qp_dec --recordings 5,11,17,23 --workers 4
python -m sigmarl.evaluation_human_comparison.run --mode replay --controller marl --workers 4
python -m sigmarl.evaluation_human_comparison.run --mode closed_loop --controller cbf_qp_dec --n_per_recording 1 --workers 4

# One controller on one homogeneous interval of one recording (default: the interval with the most
# evaluated vehicles; --interval K selects one, --list_intervals prints them), shown in a window and saved
# as a video; results and metrics in outputs/human_comparison_single/<episode>_<controller>/
python -m sigmarl.evaluation_human_comparison.run_single --controller cbf_qp_cen --recording 11 --interval 10 --render-video --save-video
# --save-human-video: also a video of the recorded human drivers of the episode (unfilled, in the colors
#   their controlled vehicles had in the run; filled rectangles are always controlled vehicles)
# --overlay-human-video: outline the human drivers in the controller's video, in the color of their vehicle
# Other modes: --mode closed_loop [--t0 120 --window 30], or --mode replay --track 226

# Metrics, tables, and figures (outputs/human_comparison/<mode>/)
python -m sigmarl.evaluation_human_comparison.analyze --controllers cbf_qp_dec,cbf_qp_cen --common_only
```

Results are stored per episode in `outputs/human_comparison/<mode>/<controller>/`
(existing episodes are skipped, so runs can be resumed). `--n_videos N` saves
videos of the first N episodes of worker 0.

## Metrics (`metrics.py`)

Human and controller trajectories are evaluated with the same code at 10 Hz
(the recorded track is resampled). Controlled vehicles have their simulated
footprint (2.0 m x 4.6 m, or the recorded one with `--vehicles recorded`);
humans and replayed road users their recorded ones.

- Success, timeout, collisions (footprint overlap with any other road user),
  at-fault collisions (not at fault if the ego stood still or was hit from
  behind), lane departures (footprint crossing the route's shared boundaries).
- Travel time from the human's appearance to the goal (includes spawn delays),
  paired with the human travel time.
- Speed, longitudinal and lateral acceleration, jerk (finite differences),
  harsh events (|a| > 3 m/s²), and distribution distances (Wasserstein, KS).
- Minimum gap, minimum time to collision (constant speed and heading, 5 s
  horizon), stops and speed at the ring entry (26 m from the center), and
  post-encroachment time with crossing traffic at the merge point (23 m).
- ADE/FDE (time-aligned) and path deviation (time-free) from the human.
- Computation time per step. For decentralized controllers, *sequential* is the
  wall-clock time of the loop over all agents in the current implementation;
  *parallelized* is an estimate: the shared per-step cache plus the slowest
  controlled agent's QP with shield round 1, plus the slowest controlled
  agent's shield round 2.
