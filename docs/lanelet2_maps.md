# Lanelet2 maps (rounD, inD, exiD, highD)

The maps of the drone datasets rounD, inD, exiD, and highD are not part of this
repository, since the datasets, including their maps, may not be redistributed.
Request the datasets from levelXdata ([rounD](https://levelxdata.com/round-dataset/),
[inD](https://levelxdata.com/ind-dataset/), [exiD](https://levelxdata.com/exid-dataset/),
[highD](https://levelxdata.com/highd-dataset/)) and generate the maps with the
scripts in `scripts/lanelet2_maps/`. They write `round_location0.osm`,
`ind_location<N>.osm` (inD location 1 also as `ind_location1_construction.osm`,
for recordings 11-17), `exid_location<N>.osm`, and `highd_location<N>.osm` to
`sigmarl/scenarios/assets/maps/`, where SigmaRL loads them. The scripts need
`pandas` and `opencv-python` (`pip install pandas opencv-python`).

```bash
# The two maps of the T-ITS 2026 evaluation (roundabout and intersection)
python scripts/lanelet2_maps/fix_lanelet_map.py --maps round_location0 ind_location4 \
    --round_dir <path>/rounD_v1.0 --ind_dir <path>/inD_v1.1
# All rounD, inD, and exiD maps
python scripts/lanelet2_maps/fix_lanelet_map.py --round_dir <path>/rounD_v1.0 \
    --ind_dir <path>/inD_v1.1 --exid_dir <path>/exiD_v2.0
# highD (built from the lane markings of the recordings; highD has no maps)
python scripts/lanelet2_maps/make_highd_lanelet_maps.py --dataset_dir <path>/highD_v1.0/data
```

Each dataset folder holds the recordings in `data/` and the untouched Lanelet2
maps in `lanelets/location<N>.osm`, as downloaded. `fix_lanelet_map.py` cuts
every map to the area that the background images and recorded tracks cover and
repairs what breaks routing (the repairs of each map are listed in its `MAPS`):
wrong-way lanelets and missing connections in the rounD circle, a duplicate
node that disconnects two exiD lanelets, two inD turning lanelets that end or
start short of the lanelet they continue, a junk sliver lanelet, and a walkway
outline that is no ring. It also extends the open map ends to the image border
and, for inD locations 2 and 3, rebuilds the lanes through the intersections
with smooth center lines. A scenario whose map is missing stops with an error
that points here.

## Scenario and parser

- Scenario `round_location0_1to18` in `SCENARIOS` (`sigmarl/constants.py`)
  loads the map at 1:18, the scale of the CPM Lab vehicles, so the current
  agent profiles fit (lanes are about 0.24 m wide). Scenario `round_location0`
  loads it at real size (lanes are about 4.3 m wide) for the `full_scale`
  agent profile (see below). `"map_format": "lanelet2"` makes `MapManager`
  use `ParseLanelet2` (`sigmarl/parse_lanelet2.py`).
- `ParseLanelet2` projects the map to UTM, subtracts `utm_origin` (the
  `xUtmOrigin`/`yUtmOrigin` of the rounD recordings at this location), scales
  by `scale` and `scale_map`, and shifts the result into the positive quadrant.
  `utm_to_map` applies the same transform to other UTM data, e.g. tracks.
- It keeps the real lane boundaries instead of offsetting a center line by a
  fixed width. Lanes next to each other across a dashed line widen each other's
  shared boundaries. Where the outer right bound is a virtual line (e.g. where
  a lane splits off), the shared boundary is moved out to the next physical
  border (road border, curb, or solid line), so it follows the real road edge.
- Scenarios `ind_location1`, `ind_location1_construction`, `ind_location2` to
  `ind_location4`, and `exid_location0` to `exid_location6`
  (`DRONE_DATASET_LANELET2_MAPS` in `sigmarl/constants.py`) load the inD and exiD
  maps at real size for the `full_scale` profile, with the `utm_origin` of the
  recordings at the location and the median lane width of the map.
- Only lanelets that cars drive on are parsed (`DRIVABLE_SUBTYPES`: `road`,
  `highway` in exiD, `play_street` in inD); bus lanes, emergency lanes, walkways
  and lanelets without subtype are left out, as Lanelet2's traffic rules do.
- Reference paths can be checked visually with
  `sigmarl/utils/plot_lanelet2_reference_paths.py`.
- Reference paths: if the scenario has `reference_paths_ids` (lists of OSM
  lanelet IDs), those are used. Otherwise all routes from a map entry to a map
  exit are computed at start-up from the Lanelet2 topology (a successor starts
  at both end nodes of its predecessor). Routes that pass the same place twice
  (more than one round) are dropped, and so are routes that start or end at a
  lane taper (a lanelet end narrower than a quarter lane width, where a lane
  begins or ends beside another one, e.g. exiD on- and off-ramps), since
  vehicles change lanes there. The rounD map gives 40 routes, the inD maps 9 to
  12, the exiD maps 6 to 10 (exiD routes follow one lane each; lane changes are
  not routes).
- Known limits of the inD/exiD maps: in the inD intersections the center line
  kinks where turning lanelets meet the next lanelet (the reference path plot
  reports turn radii below 5 m there), and exiD location 1 has one route
  (lanelet 2187, the end of an acceleration lane) that ends in the middle of
  the highway, since that lane ends without a taper.

## Full scale

- Agent profile `full_scale` (`AGENT_PROFILES` in `sigmarl/constants.py`): a
  passenger car of 2.0 m x 4.6 m, wheelbase 2.7 m, speed 0 to 14 m/s,
  acceleration -6 to 3 m/s^2, steering +-35 deg, steering rate +-1 rad/s,
  based on the cars in rounD (median 2.07 m x 4.57 m, median speed 6.9 m/s).
- The profile is chosen with the environment variable `SIGMARL_AGENT_PROFILE`
  (default `real_dyn_2`), because many modules copy the profile values at
  import time. Scenario `round_location0` has `"agent_profile": "full_scale"`
  and refuses to start with another profile; checkpoints store the profile and
  refuse to load with another one.
- Many defaults were tuned for the lab vehicles and are given in meters or
  m/s. The profile carries `length_scale` (20) and `speed_scale` (8), exposed
  as `LENGTH_SCALE`, `SPEED_SCALE`, and `ACCELERATION_SCALE` (=
  `SPEED_SCALE**2 / LENGTH_SCALE`) in `sigmarl/constants.py`; they are 1 for
  all lab profiles, so the lab behavior is unchanged. They scale:
  - `Parameters` defaults: `rl_obs_range`, `cbf_coupling_range`, `lane_width`,
    `threshold_near_boundary_high`, `threshold_near_other_agents_c2c_high`
    (meters); `h_nom` (m^2); `aggregate_safety_softmin_beta` (1/m^2);
    `ref_speed` (new, the CLF reference speed), `grouping_closing_speed_scale`
    (m/s).
  - Fallbacks in the code: the deadlock progress threshold, the softmin
    `eps`/`tau_sq`, the route-progress association margin, the finite-
    difference step and recovery thresholds of the CBF-QP, and render sizes.
  - The CBF-QP cost and the speed CLF row measure acceleration and speed in
    lab units, so the tracking weights keep their meaning.
  Defaults derived from the agent size (normalizers, reward thresholds, short-
  term reference path spacing, `deadlock`/`reset` distances) and the progress
  reward (normalized by `max_speed * dt`) scale by themselves. Time constants
  (`dt`, `ttc_*`, `deadlock_duration`) are not scaled.
- Check: with a profile that is the lab profile scaled exactly by 18 (lengths,
  speeds, accelerations), `round_location0` reproduces the trajectories of
  `round_location0_1to18` up to the accuracy of the QP solver.
- `rl_obs_range` and `cbf_coupling_range` default to `None` (also in
  `sigmarl/config.json` and `scripts/generate_configs.py`) and resolve to
  `1.0 * LENGTH_SCALE * max(1, max_speed / AGENTS["max_speed"])**2` in
  `Parameters.resolve_speed_defaults`. Explicit values in meters are kept
  unscaled.

## Checks

```bash
python -m sigmarl.parse_lanelet2 --is_visu_lane_ids   # writes map_round_location0_1to18.pdf
python -m pytest sigmarl/tests/test_parse_lanelet2.py
python main_testing_clf_cbf_qp.py --scenario_type round_location0_1to18 --n_agents 6 \
    --max_steps 301 --no-is_real_time_rendering --output_dir <scratch dir>
SIGMARL_AGENT_PROFILE=full_scale python main_testing_clf_cbf_qp.py \
    --scenario_type round_location0 --n_agents 6 --max_steps 301 \
    --no-is_real_time_rendering --output_dir <scratch dir>
SIGMARL_AGENT_PROFILE=full_scale python main_training.py \
    --yaml_config configs/round_location0/vanilla_seed_1.yaml
SIGMARL_AGENT_PROFILE=full_scale python main_testing_clf_cbf_qp.py \
    --scenario_type exid_location2 --n_agents 6 --max_steps 301 \
    --no-is_real_time_rendering --output_dir <scratch dir>
```

With 6 agents and 301 steps, the CLF-CBF-QP controller (decentralized) had no
collisions on the exiD maps; on the inD maps it had no agent-agent collisions,
1 (location 2) and 4 (location 3) lane collisions and 1 to 11 deadlocks, since
crossing traffic in the intersections has no right-of-way rules.

## Recorded tracks

Only for rounD so far: `sigmarl/round_tracks.py` loads rounD recordings, transforms the tracks with
`utm_to_map`, and matches each track to a reference path (`RouteMatcher`);
`initial_state` gives the recorded position, heading, speed, and steering angle
of a track. `sigmarl/log_replay.py` spawns agents at recorded states, replays
recorded road users, and parks unused agents (`scenario.log_replay`). The
comparison with the human drivers is in `sigmarl/evaluation_human_comparison/`.
