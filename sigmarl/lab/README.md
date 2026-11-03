# `sigmarl.lab` — CPM-lab physics backend

Swaps **only** the physics layer of the SigmaRL VMAS environment for the CPM
lab (treated as a high-fidelity simulator). Every RL component — data
collection, GAE, PPO, rewards, observations, collision/done logic,
checkpointing — runs unchanged sigmarl code through the normal
`Environment.step()` contract.

## How it works

`WorldCustom.step` (the VMAS world) converts each agent's RL action into a
clamped `(speed, steering)` target plus low-level control and integrates the
kinematic bicycle model. [`WorldLab.step`](world_lab.py) keeps that conversion
bit-identical (including the `action.u` / `acceleration` / `steering_rate`
bookkeeping that rewards and infos read) but delegates **integration** to a
bridge: it sends the targets to the lab and writes the *measured* next states
back into `entity.state`. Observations, rewards, collisions and done are
inherited and operate on those measured states.

```
MAPPOCAVs / env.rollout         (unchanged sigmarl RL)
  SyncDataCollectorCustom       (unchanged)
    Environment.step
      world.step  ->  WorldLab.step   ── apply_actions ─▶ bridge ─▶ lab
                                        ◀─ wait_for_state_list ── measured states
      reward / observation / done  (unchanged, read entity.state)
```

The only physics-layer-aware code is three small classes plus a bridge:

| file | role |
|---|---|
| [`bridge_base.py`](bridge_base.py) | ROS-free shared logic: lab↔SigmaRL frame transform, VehicleState parsing, SPEED_CURVATURE command building |
| [`lab_bridge.py`](lab_bridge.py) | the real lab over ROS 2 (rclpy; imported lazily) |
| [`mock_bridge.py`](mock_bridge.py) | in-process loopback over the same bicycle model (no ROS), for tests |
| [`world_lab.py`](world_lab.py) | `WorldLab(WorldCustom)` — the physics seam |
| [`world_state_lab.py`](world_state_lab.py) | `WorldStateLab` — reference paths from *measured* poses (manual reset only) |
| [`scenario_road_traffic_lab.py`](scenario_road_traffic_lab.py) | `ScenarioRoadTrafficLab` — overrides only `_init_world`, `_make_world_state` and `reset_world_at` |

Selection is by `Parameters.physics_backend` ∈ `{"vmas", "lab", "mock"}`
(default `"vmas"` → unchanged behaviour). `MAPPOCAVs._setup_environment`
picks the scenario class accordingly.

## Reset modes

Resetting means putting the fleet into a fresh start state. There are two ways,
chosen by `Parameters.lab_manual_reset` (`--manual-reset` on both lab drivers):

| | `lab_manual_reset = False` (default) | `lab_manual_reset = True` |
|---|---|---|
| for | digital twin (simulated vehicles), mock | **real hardware** |
| pose | **sampled** by SigmaRL, then teleported via `/lab/respawn_vehicles` | **measured** — the operator places the vehicles by hand |
| reference path | derived from the sampled pose | derived from the measured pose |
| operator | none, fully automatic | stops the fleet and waits for **ENTER** on *every* reset |
| world state | `WorldStateRTSimulation` | `WorldStateLab` |

Real vehicles cannot be teleported, which is what manual mode exists for. The
flow is: **stop the fleet → prompt → operator places the vehicles → read the
measured poses → `super().reset_world_at()` derives everything from them.**
Because the measured poses are written into `entity.state` *before* the base
reset runs, all of the base class's bookkeeping (path assignment, short-term
window, mutual distances, collision/deadlock reset, state buffer) is correct
with no duplicated code — the only override needed is
`WorldStateLab._reset_init_state`, which *assigns* the nearest matching
reference path instead of *inventing* a pose.

Notes for manual mode:

- Each prompt names the reason (collision, deadlock, episode end, staging) and
  the vehicle IDs. The fleet is actively held stopped by the bridge watchdog
  for the whole prompt.
- Startup performs a few bookkeeping resets back to back (VMAS construction +
  the collector), and `reset_at_each_iter` adds one per training iteration.
  Prompts where nothing has moved are flagged `[NO PHYSICS HAS ADVANCED …]`.
- Placements are validated (minimum inter-vehicle distance, on-map) and
  re-prompted if invalid.
- A per-agent line reports what each measured pose was matched to; it turns red
  when the lateral offset or heading error is large, i.e. the placement and the
  chosen path disagree.
- Incompatible with `predefined_ref_path_idx`, `is_challenging_initial_state_buffer`
  and `scenario_type="cpm_mixed"` (all asserted).

## Running

**Mock (no ROS)** — validates the whole pipeline:

```bash
source ~/venvs/sigmarl-py312/bin/activate
cd ~/marl_for_cavs
python -m pytest sigmarl/tests/test_lab_mock_equivalence.py -v   # mock == vmas
python main_eval_lab.py --backend mock --model-dir outputs/my_path_mode_models
python main_training_lab.py --backend mock --n-iters 2
```

**Real lab** — source ROS + the lab workspace, then run with the **venv
python** (Python 3.10, ABI-compatible with ROS Humble; the venv site-packages
and ROS dist-packages coexist on `sys.path`):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2-workspace-lab/install/setup.bash
~/venvs/sigmarl-py312/bin/python main_eval_lab.py --backend lab \
    --model-dir <dir> --vehicle-ids 1,2,3,4
```

In the RViz ControlCenter, deploy the bridge (period **100** ms, matching
`--period-ms`) and the simulated vehicles, then press **START**. The driver
blocks at `wait_for_first_state` until it does. Headless? add
`MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen` (the checkpoint-save path uses
matplotlib).

**Real vehicles** — add `--manual-reset` (see [Reset modes](#reset-modes)).
Nothing is teleported; you stage the fleet by hand and press ENTER at each
prompt. Prefer `--n-rollouts 1`, since each rollout rebuilds the environment
and so re-runs the startup prompts:

```bash
~/venvs/sigmarl-py312/bin/python main_eval_lab.py --backend lab --manual-reset \
    --model-dir <dir> --vehicle-ids 1,2,3,4 --n-rollouts 1
```

## Constraints and caveats

- **One fleet, one env.** `num_vmas_envs == 1` and `dt == lab_period_ms/1000`
  (asserted). `frames_per_batch` therefore equals `max_steps`; fine-tuning
  clamps `minibatch_size` to it.
- **Regime B (wall-clock).** The lab runs on wall time, so collection is
  real-time paced and a full from-scratch training run is impractical; this
  backend is for **evaluation and on-twin fine-tuning**. Stepped-time
  (faster-than-real, exact VMAS semantics) is a future lab-side change.
- **`reset_at_each_iter=True`** is recommended for fine-tuning so no transition
  spans the PPO-update pause (the bridge watchdog holds the fleet stopped
  during it).
- **Not bit-reproducible** against pure VMAS: the lab's vehicle model and
  timing differ (that's the point). Equivalence is asserted only for the mock,
  which shares the bicycle model.
- **Real hardware** needs `--manual-reset`: the respawn service teleports, which
  only simulated vehicles can do. See [Reset modes](#reset-modes). A state
  estimator is still an open hedge.
