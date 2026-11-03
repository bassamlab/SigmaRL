# How Recorded Human Trajectories Are Used to Evaluate Controllers

This note describes how the rounD recordings of location 0 are turned into
test episodes for our controllers, and how the controlled trajectories are
compared with the recorded human ones. The code lives in
`sigmarl/round_tracks.py`, `sigmarl/log_replay.py`, and
`sigmarl/evaluation_human_comparison/` (`episodes.py`, `run.py`, `metrics.py`,
`analyze.py`).

The basic idea: a human-driven vehicle of the recording gives a start state
(where and how it entered the scene), a route and a goal (where it left the
scene), and a reference behavior (what the human did in between). We let a
controller drive from the same start to the same goal in the same traffic, and
evaluate both trajectories with the same metrics.

## 1. Data

- rounD v1.0 (drone recordings of German roundabouts), location 0 =
  recordings 02-23 (22 recordings, roughly 15-20 min each), 25 Hz.
- Per recording: `<NN>_tracks.csv` (per frame: bounding-box center `xCenter`,
  `yCenter` relative to the UTM origin, `heading` in degrees counterclockwise
  from east, `xVelocity`, `yVelocity`, longitudinal and lateral
  acceleration), `<NN>_tracksMeta.csv` (class, width, length), and
  `<NN>_recordingMeta.csv` (frame rate, UTM origin, location).
- Default directory `~/projects/drone-dataset-tools/rounD_v1.0/data`
  (`--data_dir` or `ROUND_DATA_DIR`).

## 2. From recording coordinates to the simulated map

`load_recording` (`round_tracks.py`) adds the UTM origin to the recorded
positions and maps them with `ParseLanelet2.utm_to_map`, the same transform
that built the simulated `round_location0` map from the Lanelet2 file. The
transform is a shift and a scale only, so recorded headings are used
unchanged (converted to radians and unwrapped). The speed is the norm of the
recorded velocity vector. Each road user becomes a `Track` with time stamps
`frame / frame_rate`.

`Track.state_at(t)` linearly interpolates position, heading, and speed. All
uses of the tracks at simulation times (spawning, replaying, metrics) go
through this interpolation, which converts 25 Hz recordings to the
simulation step of 0.1 s.

## 3. Matching tracks to routes

The simulated map has 40 reference paths (routes: entry arm -> exit arm).
A controller needs one of them, so every track is matched to a route by
`RouteMatcher`:

- The track is subsampled every 0.4 s and projected sequentially onto each
  route's center line (the projection searches a window of segments around
  the previous one, so routes that pass near the same place twice work).
- A route fits if every sample is within 4.5 m laterally of its center line,
  the heading agrees with the route direction within 50° (only checked while
  moving faster than 1 m/s), and the progress `s` never goes backward by more
  than 2 m.
- Among fitting routes, the smallest mean lateral offset wins. The match is
  then recomputed for all frames, giving the arc length `s(t)` and the signed
  lateral offset of the human along the route.

Matching rate: 10783 of 11020 cars (97.8%). Unmatched tracks are never
controlled, but they are still replayed.

The matched tracks of all 22 recordings are cached in
`outputs/human_comparison/matched_tracks.pkl` (delete it after changing the
map or the matcher).

## 4. Which vehicles are given to the controller

`is_controllable` (`episodes.py`):

- class `car` or `van` (the controllers are trained and tuned for passenger
  vehicles);
- a matched route;
- at least 20 m of recorded progress along the route, and a mean speed of at
  least 1 m/s (excludes parked or nearly parked vehicles).

10754 cars and 584 vans qualify. Everything else is replayed: trucks, buses,
trailers, motorcycles, bicycles, unmatched cars, and controllable vehicles
that are not selected for the episode. Pedestrians (12 in total, all off the
road) are ignored.

## 5. Start state, goal, and timeout of a controlled vehicle

- **Start**: the first recorded frame of the human, i.e., the moment the
  vehicle enters the drone's field of view. Position, heading, and speed are
  the recorded values; the steering angle is the kinematic-bicycle angle that
  reproduces the recorded yaw rate (`steering_from_yaw_rate`, central
  difference over +-2 frames, zero below 1 m/s, clipped to the steering
  limit). Accelerations start at zero.
- **Initial-state repair** (`start_repair.py`, on by default,
  `--no_start_repair` to switch off): about 6% of the start poses put the
  simulated footprint (4.6 m x 2.0 m) across a route boundary, where the lane
  CBF stops the vehicle. Such a pose is moved by the smallest lateral shift
  along the route normal (and, if needed, the smallest heading change toward
  the route direction) that gives a lane barrier (same footprint and buffers
  as the CBF-QP) of at least 1 cm; speed, steering, and time stay as
  recorded. The limits are 1.0 m and 5 deg; they repair all starts of rounD
  location 0 (the largest shift needed is 0.96 m). A vehicle whose start needs
  more (status `failed`, a bad recording) is removed from the episode, i.e.,
  from the simulation and from the comparison; the result lists it under
  `removed_bad_start`, and the summary reports the count ("Removed (bad
  start)": evaluated / all controlled vehicles). The outcome is stored per
  vehicle (`start_repair`, `start_shift`, `start_dyaw_deg` in
  `vehicles.csv`). In 93% of the crossing poses the recorded footprint
  crosses too, and 82% lie at the edge of the drone's field of view, where
  vehicles are only partly visible and their boxes are unreliable.
- **Spawn time**: the human's first time stamp. In closed loop, a spawn is
  delayed while another *controlled* vehicle is closer than
  `4.6 m + 1.0 s * speed` to the spawn position (a controller may still be
  there because it is slower than the human was). Replayed vehicles are not
  checked: they are exactly where they were when the human appeared. The
  delay counts toward the travel time.
- **Route**: the matched route (`path_id`); the policy's observations and the
  CBF lane constraints use this route.
- **Goal**: the arc length where the human left the field of view,
  `s_goal = s(t_end)` (capped at the route length minus 1 m). The vehicle
  reaches its goal when its projected progress exceeds this value; it is then
  removed (parked).
- **Timeout**: 2 x the human travel time + 30 s after the human's appearance.

The simulated vehicle has the fixed footprint of the full-scale agent profile
(4.6 m x 2.0 m), not the human vehicle's recorded size.

## 6. The simulation with log replay (`log_replay.py`)

The VMAS environment has a fixed number of agents, here called slots (the
maximum number of simultaneously present road users over the episodes of a
run plus 2, or `--n_slots`).
At every step each slot is in one of three roles:

| Role | State | Seen by observations | Seen by CBFs |
|---|---|---|---|
| parked | standing at (-2000 + 100 i, -2000) | no (on hold) | no |
| replayed | overwritten with the recorded state after every world step | yes | yes, as a non-reactive obstacle |
| controlled | integrated by the simulator from the controller's actions | yes | yes |

- **Replayed road users** follow `Track.state_at(t)`: position, heading,
  speed, and the steering angle from the yaw rate. They appear and disappear
  with the recording. A long vehicle (truck, bus, trailer) is covered by
  several car-sized boxes placed along its length (`split_into_boxes`), each
  in its own slot; its route (or nearest route) is assigned so its
  observations are defined.
- **Non-reactivity**: replayed vehicles do not respond to the controlled
  ones. In the CBF-QP they are predicted with zero control, and the
  controlled vehicle takes the full responsibility (share 1.0) for the pair
  constraint; their own lane constraints and QPs are skipped.
- **Order of operations per step**: controller computes actions -> world
  step -> `LogReplay.on_post_step` (log and park controlled vehicles that
  reached their goals or timed out, update replayed vehicles, spawn pending
  controlled vehicles) -> the scenario computes distances and collisions of
  the transition. The scenario never resets agents while a log replay is
  attached (collisions do not end the episode; they are evaluated
  afterward).
- **Recorded output** per episode: the full log of every controlled vehicle
  (t, x, y, yaw, speed, steering, acceleration, steering rate, s), spawn and
  finish times and status, and snapshots of all present road users at each
  step (controlled ones from the simulator, replayed ones from the record),
  plus per-step computation times.

## 7. Episode types

- **replay** (`replay_episodes`): one controllable car is the ego; every
  other road user is replayed. The episode starts at the ego's appearance.
  20 egos per recording, drawn uniformly without replacement with a seeded
  RNG -> 440 episodes. This is the cleanest pairwise comparison: the ego sees
  exactly the traffic the human saw, but that traffic ignores the ego (a
  follower can drive into a controller that is slower than the human was).
- **closed_loop** (`closed_loop_episodes`): a 60 s window per recording, start
  time drawn uniformly from [30 s, duration - 120 s]; every controllable
  vehicle that appears in the window is controlled, all others replayed.
  Controlled vehicles interact with each other reactively; they still share
  the road with non-reactive replayed traffic (trucks, buses, vehicles
  outside the window). The episode runs until all controlled vehicles have
  finished.

- **homogeneous** (`homogeneous_episodes`): an episode without replayed
  traffic, so that humans and controllers are compared in homogeneous
  car/van traffic.
  - *Interval.* For each recording, the time intervals in which no
    non-controllable road user is on the map: the complement of the union of
    the lifetimes of all trucks, trailers, buses, motorcycles, bicycles,
    pedestrians, and cars/vans that are not controllable (Section 4). Such
    users (1791) appear every 12 s on average, so the intervals are short (at most
    85 s). Intervals shorter than 20 s (the median trip lasts 14 s) or
    without an evaluated vehicle are dropped: 102 intervals in 20 recordings
    (recordings 2 and 13 have none), 61 min in total.
  - *Controlled vehicles.* Every vehicle on the map during the interval
    (all of them controllable by construction). Vehicles already on the road
    at the interval start are controlled from their first recorded sample at
    or after that time (`ControlledVehicle.k_start`), without the spawn gate
    (they are placed where the humans were); their timeout counts from there.
    Vehicles appearing during the interval spawn as in closed loop. Nothing
    appears after the interval end; the episode runs until all controlled
    vehicles have finished.
  - *Evaluated vehicles.* Only vehicles whose whole recorded trip lies inside
    the interval (`EpisodeSpec.evaluated_keys`), because only their human
    counterparts drove in purely homogeneous traffic and started at their
    real start. The others (present at the interval start or still present
    at its end) are traffic only and are excluded from all metrics. 1171
    evaluated and 2567 controlled vehicles.
  - *Remaining asymmetry.* A controller that is slower than the human may
    still be on the road after the interval end, where the human would have
    met newly arriving traffic; the simulation has none. All interactions of
    the evaluated humans are with cars/vans, but the traffic of the
    controllers thins out at the end of each interval.

Episode names (`rec<NN>_track<ID>`, `rec<NN>_t<t0>`, or `rec<NN>_h<t_start>`) depend only on the seed
and `--n_per_recording`; `--recordings` filters after sampling, so subsets
produce the same episodes as the full run.

## 8. Comparing the trajectories (`metrics.py`, `analyze.py`)

Both trajectories are evaluated by the same functions at 10 Hz:

- **Human trajectory**: the recorded track resampled to 10 Hz, with its
  recorded footprint. Its scene is the full recording (all other road users
  except pedestrians, at their recorded states).
- **Controller trajectory**: the simulator log, with the simulated footprint.
  Its scene is the episode snapshots (other controlled vehicles from the
  simulator, replayed ones from the record with their recorded footprints).

Derivatives (longitudinal acceleration, yaw rate, lateral acceleration =
speed x yaw rate, jerk) are finite differences of the 10 Hz speed and yaw for
both, so they are filtered identically. Note that rounD positions are already
smoothed by the dataset pipeline.

Metrics per vehicle:

| Group | Metric | Definition |
|---|---|---|
| Task | success / timeout | goal reached before the timeout |
| | travel time | goal time - human appearance time (includes spawn delay); human: `t_end - t_start` |
| | spawn delay | actual - recorded spawn time |
| Safety | collision | footprints overlap (separating-axis test) with any other road user |
| | at-fault collision | collision unless the ego stood still (< 0.5 m/s) or was hit from behind (other behind it, heading difference < 45°) |
| | min. gap | minimum footprint distance to any road user within 40 m |
| | min. TTC | constant speed and heading of both, 5 s horizon, 0.1 s resolution; "TTC < 1.5 s" counts vehicles with any such moment |
| | lane boundary touched / crossed > 0.5 m | footprint (or footprint shrunk by 0.5 m on each side) intersects the route's left or right shared boundary |
| Roundabout | stop before entry, entry speed | first time within 26 m of the ring center (yield line) |
| | entry PET | time gap at the point where the ego reaches the outer ring lane (23 m radius) to any road user with a heading difference >= 30° passing within 3.5 m; capped at 10 s |
| Comfort | RMS longitudinal / lateral acceleration, RMS jerk, time with abs. acceleration > 3 m/s² | from the finite differences |
| Similarity | ADE / FDE | time-aligned distance to the human (same time since appearance), over the common duration |
| | path deviation | mean distance from controller positions to the human's path (time-free) |
| | distribution distance | Wasserstein and Kolmogorov-Smirnov distances of the pooled speed, abs. acceleration, and abs. jerk samples |
| Compute | policy time, CBF time | per step; for decentralized CBF-QPs, *sequential* = wall-clock loop over all agents in the current implementation, *parallelized (est.)* = shared per-step cache + slowest controlled agent's QP with shield round 1 + slowest controlled agent's shield round 2 |

Aggregates: rates over vehicles, medians, paired travel-time differences
(controller - human, successful vehicles only). `--common_only` restricts all
controllers to episodes every controller finished, so all columns describe
the same vehicles. Each vehicle's human metrics are computed once and appear
as the "Human" column.

## 9. Known limitations of the method

- **Non-reactive replay.** Replayed vehicles ignore the controllers. In replay
  mode most not-at-fault collisions are recorded followers rear-ending a
  controller that is slower than the human. CBF controllers react to these
  followers (they have full responsibility) and may evade sideways, which
  inflates their lane departures.
- **Reference human scene differs from the controller scene.** The human is
  evaluated against the real, mutually reactive traffic; the controller
  against partly replayed traffic. Rates of interaction events (TTC, PET) are
  therefore not strictly like-for-like.
- **Footprints.** Controlled vehicles are 4.6 m x 2.0 m regardless of the
  human's vehicle; long replayed vehicles are approximated by car-sized boxes
  for the controllers' perception (the metrics use the recorded footprint).
- **Annotation noise.** Recorded bounding boxes occasionally overlap (0.2-0.5%
  of human "collisions") and humans touch the route's shared boundaries often
  (about 40%), because routes follow lane centers while humans cut curves and
  use slip lanes; the > 0.5 m variant is the more meaningful lane metric.
- **Route vs. human lane.** On multi-lane segments the route's center line may
  be a different lane than the one the human used, so the controller starts
  offset and moves over (side swipes with trucks were partly caused by this).
- **Vehicle model.** The kinematic bicycle model has no friction or
  lateral-acceleration limit; spawning at high speed in a curve can produce
  unrealistic lateral accelerations right after the spawn.
- **Goal definition.** The human's travel time ends at its last recorded
  frame; the controller's ends when its projected progress passes `s_goal`
  (capped at route length - 1 m), so the controller's distance can be up to
  about 1 m shorter.
- **Timing.** Computation times depend on load; runs with many parallel
  workers inflate them. Use a dedicated run with `--workers 1` for timing.

## 10. Reproducing

```bash
export SIGMARL_AGENT_PROFILE=full_scale
# Per controller (marl, marl_cbf, cbf_qp_dec, cbf_qp_cen) and mode
python -m sigmarl.evaluation_human_comparison.run --mode replay --controller marl --workers 6
python -m sigmarl.evaluation_human_comparison.run --mode closed_loop --controller marl --n_per_recording 1 --workers 6
python -m sigmarl.evaluation_human_comparison.run --mode homogeneous --controller marl --workers 6
# Tables and figures
python -m sigmarl.evaluation_human_comparison.analyze --mode replay --controllers marl,marl_cbf,cbf_qp_dec --common_only
python -m sigmarl.evaluation_human_comparison.analyze --mode closed_loop --controllers marl,marl_cbf,cbf_qp_dec --common_only
```

See `sigmarl/evaluation_human_comparison/README.md` for all options.
