# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

This repository contains SigmaRL, a Python package for multi-agent reinforcement
learning motion planning for connected and automated vehicles. The main package
is `sigmarl`; top-level `main_*.py` and `evaluation_*.py` scripts are experiment
entry points. `docs/` holds pipeline notes such as `docs/backup_cbf_pipeline.md`
and `docs/cpm_lab_bridge_and_modes.md`.

## Environment

The conda environment named `sigmarl-py312` is usually the preferred
project environment when it exists. If it is not available, inspect the local
Conda or virtualenv setup and choose another environment with a supported Python
version (3.10 to 3.12) and the project dependencies installed. The environment
is already set up: do not install or upgrade packages unless the user asks.

`sigmarl` is installed in editable mode (`pip install -e .`), so an environment
imports `sigmarl` from exactly one checkout. A machine may also have a secondary
git worktree for side features, named `<repo>-feat` (for example
`marl_for_cavs-feat`), paired with an environment named `<main env>-feat` (for
example `sigmarl-py312-feat`) whose editable install points at that worktree.
Before running code, pick the environment whose `sigmarl` resolves to the
current checkout:

```bash
git rev-parse --show-toplevel
conda run -n <env> --cwd / python -c "import sigmarl; print(sigmarl.__file__)"
```

The printed path must lie under the current checkout's root. Run the import
check outside the repository (hence `--cwd /`): from the repository root, the
local `sigmarl/` folder shadows the installed one and every environment appears
to match. If no environment
matches (for example, a worktree without its own environment), prefix commands
with `PYTHONPATH=<checkout root>` and re-run the check. Never run code from one
checkout against `sigmarl` imported from another: scripts outside the repository
root, such as `sigmarl/scenarios/road_traffic.py`, would silently use the other
checkout's code. Do not create, clone, or re-point environments unless the user
asks.

## Common Commands

Run all commands from the repository root inside the environment selected above
(for example, prefix them with `conda run -n sigmarl-py312`).

```bash
# Interactive environment check (arrow keys drive agents, Tab switches agents)
python sigmarl/scenarios/road_traffic.py

# Training: from a benchmark YAML, or legacy flags on top of sigmarl/config.json
python main_training.py --yaml_config <config>.yaml
python main_training.py --seed 1

# Test a saved model folder (contains the .pth files and reward JSON)
python main_testing.py --path <checkpoint_folder> [--scenario_type ... --n_agents ...]
python main_testing_batch_iterate.py --path <folder_of_model_folders>
```

Paper-specific reproduction commands live in per-paper scripts and
READMEs (for example `sigmarl/evaluation_icra27/README.md`).

## Architecture

- **Configuration.** `Parameters` (`sigmarl/helper_common.py`) holds every
  experiment flag. It is built from `sigmarl/config.json`, a benchmark YAML
  (unknown keys are rejected), or the JSON saved with a checkpoint. Entry
  scripts mostly mutate a `Parameters` object and call `mappo_cavs(parameters)`.
- **One driver for training and testing.** `MAPPOCAVs.train()`
  (`sigmarl/mappo_cavs.py`) builds the env, the policy, PPO, and priority
  modules, and optional CBF-QP controllers, and loads checkpoints. With
  `is_continue_train=False` it returns them without training; testing and
  evaluation scripts take that path and then call `env.rollout(...)`.
- **Environment.** `sigmarl/env_factory.py` maps `scenario_type` (keys of
  `SCENARIOS` in `sigmarl/constants.py`) to a scenario class; every road map uses
  `sigmarl/scenarios/road_traffic.py`. `Parameters.physics_backend`
  (`vmas`/`lab`/`mock`) can swap in `sigmarl/lab/`, which replaces only the
  physics integration. The env wrapper's custom `rollout` in
  `sigmarl/helper_training.py` is where prioritization and CBF-QP safety
  filtering are applied.
- **Road-traffic scenario.** It keeps reward, done, info, and rendering logic,
  and delegates state to `sigmarl/scenarios/world_state/` and observations to
  `sigmarl/scenarios/observations/` (`_rt` = road traffic, `_gr` = goal
  reaching). Maps come from `sigmarl/parse_xml.py`, `sigmarl/parse_osm.py`, and
  `sigmarl/scenarios/assets/maps/`.
- **Policies** (`sigmarl/modules/`). The motion policy, the learned CBF class-K
  (lambda) policy, or both; `Parameters.policy_training_case` selects
  motion-only, lambda-only, joint, or no policy training (pure CLF/CBF control).
- **CBF-QP.** `sigmarl/cbf_qp_proxsuite.py` is the active implementation;
  `sigmarl/cbf_qp.py` (CVXPY) will be discarded. Controllers run centralized per
  env (optionally grouped) or fully decentralized per agent.
- **Vehicle profile.** `ACTIVE_AGENT_PROFILE` in `sigmarl/constants.py` sets the
  vehicle geometry and dynamics limits. It is stamped into checkpoints and
  validated on load, so changing it invalidates existing models. It is read from
  the environment variable `SIGMARL_AGENT_PROFILE`. Always set
  `SIGMARL_AGENT_PROFILE=full_scale` when using full-scale maps (for example
  `SIGMARL_AGENT_PROFILE=full_scale python main_testing.py ...`).

## Code Style

- Follow the existing Python style and keep changes close to the surrounding
  patterns. Code is formatted with black (via pre-commit).
- Before committing, format with `pre-commit run --files <changed files>` (inside the project
  environment). Use `--all-files` only if the user asks, since it reformats unrelated files.
- Access expected object attributes and interface methods directly rather than using
  defensive checks or dynamic reflection (e.g., getattr/hasattr in Python). Do not use
  getattr/hasattr to bypass missing fields in mock namespaces or test fixtures; ensure
  test fixtures and objects provide expected attributes or let missing fields fail fast.
- Add proper docstrings and code comments for new public functions and nontrivial helper
  functions.
- Keep code comments short and useful; prefer clear names over explanatory comments.
- Avoid broad refactors while fixing experiment, plotting, CBF, or map-parsing
  behavior.

## Experiment and Data Hygiene

- Do not commit generated rollouts, videos, plots, checkpoints, or ad hoc output
  directories unless the user explicitly asks. `checkpoints/.gitkeep` is the only
  tracked checkpoint placeholder.
- Use a scratch location outside tracked source files for disposable figures,
  smoke-test outputs, and rollout diagnostics.
- Treat `sigmarl/config.json` and checkpoint paths as experiment state. Avoid
  permanent config changes unless the task is specifically about defaults.
- Prefer small deterministic smoke checks over long training runs when verifying
  implementation changes.
- Unless the user states otherwise, use `cpm_mixed` when the number of agents is
  less than 7, and use `cpm_entire` when the number of agents is 7 or greater.
- For fully decentralized methods, such as fully decentralized CBF-QP, clearly
  distinguish sequential computation time from parallelized computation time in
  benchmark tables, CSV files, and written summaries. Sequential time is the
  actual wall-clock loop time in the current implementation; parallelized time is
  the per-agent or otherwise parallel-execution estimate and must be labeled as
  such. For the fully decentralized backup shield, both rounds are per-agent
  parallelizable work, but round 2 starts only after every agent's round 1:
  add the shared per-step preparation in full, the slowest agent's QP with
  round 1, and the slowest agent's round 2. Any shared sequential finalizer is
  added in full.
- When implementing a new evaluation script or benchmark script with many runs,
  include a simple progress bar or progress indicator so users can see how much
  work has completed and what remains.

## Known Project Quirks

- In CLF/CBF rollouts, `agents/action` may be stale or constant. For actual
  applied controls, prefer simulator-reported fields such as
  `agents/info/act_vel`, `agents/info/act_steer`,
  `agents/info/applied_action_vel`, and `agents/info/applied_action_steer`.
- Map parser outputs can align `center_line`, `left_boundary_shared`, and
  `right_boundary_shared` by index. Runtime sampling for aligned boundaries
  should preserve the centerline progress grid instead of reparameterizing each
  boundary by its own arc length.
- For headless plotting or verification scripts, use a noninteractive Matplotlib
  backend such as `Agg`. On HPC, `main_training.py` sets `PYGLET_HEADLESS` when
  `HPC_ENVIRONMENT=true`.
- Rendering backend: `SIGMARL_RENDERER` (`auto` default, `pyglet`, `matplotlib`)
  selects how VMAS frames are drawn. `auto` falls back to an off-screen
  Matplotlib viewer (`sigmarl/render_backend.py`) when no display is available,
  e.g. over SSH on macOS, so `main_testing.py --video` works headless. Both
  backends draw the same `vmas.simulator.rendering` geoms, so put new
  visualizations in the scenario's `extra_render` (no renderer-specific code).
  `rendering.Image` (VMAS `plot_position_function`) is pyglet-only.
- If a saved model folder is moved, update `where_to_save` in its JSON file.
- The lab backend requires `num_vmas_envs=1`, `device="cpu"`, and
  `dt == lab_period_ms / 1000`; `rclpy` is imported lazily, so the `mock`
  backend works without ROS.

## Git Workflow

- Preserve unrelated user changes.
- Do not stage or revert unrelated modified files.
- In a secondary worktree, a branch checked out in another worktree cannot be
  checked out here. Do not add, remove, or prune worktrees unless the user asks.
- Gitignored data (checkpoints, outputs) is not shared between worktrees; refer
  to it by absolute path, and do not let two checkouts write to the same output
  folder.
