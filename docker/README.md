# Running SigmaRL in the CPM Lab with Docker

A trained SigmaRL policy runs on a **compute machine**, while the lab
infrastructure (Zenoh router, RViz ControlCenter, vehicles) runs on the **main
lab machine**. The two talk over Zenoh: the main machine hosts the router, the
SigmaRL container joins it as a client.

| Image | Built by | Runs on |
|---|---|---|
| `cpm-lab-dev` | `ros2-workspace-lab/docker/create_image.bash` | main lab machine |
| `sigmarl-dev` | [`create_image.bash`](create_image.bash) | compute machine |

**Prerequisites:** Docker on both machines, port `7447/tcp` reachable from the
compute machine, and an SSH key for `git.rwth-aachen.de` loaded in an agent
(`ssh-add ~/.ssh/id_ed25519`) — the SigmaRL image clones the lab workspace at
build time.

---

## Main machine

```bash
cd ~/ros2-workspace-lab

./docker/create_image.bash        # once — builds cpm-lab-dev
./docker/build_workspace.bash     # colcon build inside the image
./docker/run_router.bash          # terminal 1: zenoh router on :7447 (keep open)
./docker/run_control_center.bash  # terminal 2: RViz — press START here later
```

Note the IP of this machine (`hostname -I`), you need it below.

**Automatic respawn on collision — digital twin only.** Without
`--manual-reset`, colliding vehicles are teleported to fresh poses through the
`/lab/respawn_vehicles` service. This only works with simulated vehicles; real
cars cannot be teleported, so on the physical fleet use `--manual-reset` and
place them by hand. Both agent–agent and lane-boundary collisions trigger a
respawn; pass `--without-lanelet-reset` to respawn on agent–agent collisions only.
The service exists only on the `gh-path-mode` branch of four submodules — check
them out **before** `build_workspace.bash`:

```bash
cd ~/ros2-workspace-lab/src
for d in bridge mid-level-controller common-interfaces visualization; do
  git -C "$d" checkout gh-path-mode
done
```

Then run `./docker/build_workspace.bash` and `./docker/run_control_center.bash`.

## Compute machine

**1. Point Zenoh at the main machine** — edit
[`zenoh_session_config.json5`](zenoh_session_config.json5):

```json5
connect: { endpoints: ["tcp/<main-machine-ip>:7447"] }
```

**2. Build the image and the virtualenv:**

```bash
cd ~/marl_for_cavs
./docker/create_image.bash
```

Builds `sigmarl-dev` (ROS 2 Jazzy + `cpm_lab_lab_msgs`) and creates
`marl_for_cavs/.venv` on the host via `pip install -e .`. Re-running it
**deletes and rebuilds** the venv.

**3. Start the container:**

```bash
./docker/run_sigmarl_control.bash
```

Drops you into a shell with the ROS workspace sourced and the venv active. The
repo is mounted at `/workspace/sigmarl`, so `outputs/` and saved `.td` files are
shared with the host.

**4. Run the evaluation inside the container:**

```bash
python3 main_eval_lab.py \
  --model-dir outputs/model_name \
  --backend lab \
  --vehicle-ids 1,2,3,4 \
  --scenario-type cpm_entire \
  --period-ms 50 \
  --max-steps 2048 \
  --manual-reset
```

It blocks until you press **START** in RViz, then writes
`out_td_lab_<timestamp>_seed_<n>.td` into `--model-dir`.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--model-dir` | `outputs/my_path_mode_models/` | Dir with one `reward*.json` + matching `*_policy.pth` |
| `--backend` | `lab` | `lab` = real fleet; `mock` = in-process loopback, no hardware |
| `--vehicle-ids` | `1,2,3,4` | Lab vehicle IDs, one per agent; also sets `n_agents` |
| `--scenario-type` | `cpm_entire` | Any scenario in `sigmarl/constants.py` |
| `--period-ms` | `100` | Control period → `dt` |
| `--max-steps` | `600` | Steps per rollout |
| `--n-rollouts` | `3` | Env is rebuilt per rollout; use `1` with `--manual-reset` |
| `--manual-reset` | off | Stop the fleet on reset and place vehicles by hand, assigning paths from the measured poses — **required on real hardware**. Off = colliding vehicles are teleported via `/lab/respawn_vehicles`, digital twin only, needs the `gh-path-mode` submodules above |
| `--without-lanelet-reset` | off | If on then lane-boundary collisions are still recorded, but no longer reset the episode |
| `--load-final-model` | off | Pass the flag for `True`; **omit** for `False` |

Evaluation always runs with `rew_method="sparse"` and
CBF actuation off, regardless of what the model's JSON says.

Dry run without hardware: `--backend mock --n-rollouts 1`.

---

## Files

**`marl_for_cavs/docker/`**

| File | Description |
|---|---|
| [`Dockerfile`](Dockerfile) | `ros:jazzy` + desktop-full + Zenoh/CycloneDDS RMWs. Clones `ros2-workspace-lab` (`dev`), switches `common-interfaces` to `gh-path-mode`, builds `cpm_lab_lab_msgs`. |
| [`create_image.bash`](create_image.bash) | SSH-agent check → `docker build --ssh default` → recreate the host `.venv`. |
| [`run_sigmarl_control.bash`](run_sigmarl_control.bash) | Interactive `sigmarl-controller` container: host network, repo + Zenoh config + X11 + SSH-agent mounts, `ROS_DOMAIN_ID=1`, `rmw_zenoh_cpp`. |
| [`zenoh_session_config.json5`](zenoh_session_config.json5) | Zenoh client session — **edit the IP per deployment**. |

**`ros2-workspace-lab/docker/`**

| File | Description |
|---|---|
| [`Dockerfile`](../../ros2-workspace-lab/docker/Dockerfile) | `ros:jazzy` for amd64/arm64: desktop-full, lanelet2, cv_bridge, Basler pylon 8.1, both RMWs. |
| [`create_image.bash`](../../ros2-workspace-lab/docker/create_image.bash) | Builds `cpm-lab-dev`. Build context is the **current directory**. |
| [`build_workspace.bash`](../../ros2-workspace-lab/docker/build_workspace.bash) | Runs `make` (colcon Release) in the container; artifacts land on the host. |
| [`run_router.bash`](../../ros2-workspace-lab/docker/run_router.bash) | `rmw_zenohd` on port 7447. |
| [`run_control_center.bash`](../../ros2-workspace-lab/docker/run_control_center.bash) | RViz ControlCenter, `ROS_DOMAIN_ID=21`. |
| [`run_lane_control*.bash`](../../ros2-workspace-lab/docker/) | Lane-control node / launch variants, `ROS_DOMAIN_ID=1`. |
| [`run_terminal.bash`](../../ros2-workspace-lab/docker/run_terminal.bash) | Plain shell in the lab image. |

---

## Notes

- **No lab topics?** Check the router is up, the IP in
  `zenoh_session_config.json5` is current (`nc -vz <ip> 7447`), and both sides
  use `rmw_zenoh_cpp`.
- **`main_eval_lab.py` hangs at startup** — expected, it is waiting for START.
- **`cpm_lab_lab_msgs` not found** — the messages are built into the image, not
  the host mount; rebuild the image if they changed upstream.
