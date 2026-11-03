# ICRA 2027 Evaluation

Commands to reproduce the ICRA 2027 experiments: training, baseline and
ablation comparison, the cross-iteration test, behavioral analysis, and lab
deployment. Run all commands from the repository root.

## Checkpoint Folder Structure

All artifacts are stored under `checkpoints/icra27/`:

| Folder | Contents |
| --- | --- |
| `config-main/cpm_mixed/` | YAML configs for training the main models |
| `config-cross-iter-test/cpm_mixed/` | YAML configs for training the cross-iteration models |
| `main/cpm_mixed/` | Trained main models (baselines and ablation variants) |
| `cross-iter-test/cpm_mixed/` | Trained models for the cross-iteration test |
| `lab/` | Model deployed in the physical lab |
| `policy_for_video/` | Models used to generate the video demo |

## 1. Main Models

### Training

Train one model per YAML config in `config-main/cpm_mixed/`:

```bash
python main_training.py --yaml_config checkpoints/icra27/config-main/cpm_mixed/<config>.yaml
```

### Plot Episode Rewards

```bash
python sigmarl/evaluation_icra27/plot_episode_rewards.py --path checkpoints/icra27/main/cpm_mixed/
```

### Compare with Baselines and Ablation Variants

```bash
python main_testing_batch_iterate.py --path checkpoints/icra27/main/cpm_mixed/
```

## 2. Cross-Iteration Test

### Training

Train one model per YAML config in `config-cross-iter-test/cpm_mixed/`:

```bash
python main_training.py --yaml_config checkpoints/icra27/config-cross-iter-test/cpm_mixed/<config>.yaml
```

### Testing

```bash
python main_testing_batch_iterate.py \
    --path checkpoints/icra27/cross-iter-test/cpm_mixed/ \
    --is_cross_iter_test \
    --cross_iter_ego_indices "[0,2,4,6,8]"
```

## 3. Behavioral Analysis

Analyze how agents' priorities (derived from task progress) relate to their
egoistic and altruistic behaviors.

### Generate Rollout Data

Roll out one of the trained models:

```bash
python main_testing.py \
    --path checkpoints/icra27/main/cpm_mixed/our_rr_motion_taskprogress_seed_1 \
    --max_steps 2400 \
    --n_agents 6 \
    --scenario_type cpm_mixed \
    --num_vmas_envs 1 \
    --random_seed 100
```

This saves the rollout as `out_td_cpm_mixed_seed_100.td` in the model folder.

### Plot Priority–Behavior Relation

```bash
python sigmarl/evaluation_icra27/plot_priority_behavior_analysis.py \
    --model_dir checkpoints/icra27/main/cpm_mixed/our_rr_motion_taskprogress_seed_1
```

Figures are saved to `fig_relation_analysis/` in the model folder.

## 4. Lab Deployment

### Prerequisites

Lab deployment requires our CPM Lab digital twin (the ROS 2 lab workspace).
See the [CPM Lab ROS 2 documentation](https://bassamlab.github.io/cpm-lab-ros2-documentation/)
for installation instructions.

> **Note:** As of October 1, 2026, the full version of the digital twin has
> not been published yet. It may be released soon.

The policy runs in a Docker container on a **compute machine**. The lab
infrastructure (Zenoh router, RViz ControlCenter, and vehicles) runs on the
**main lab machine**. For the one-time setup of both machines (building the
Docker images, configuring the Zenoh endpoint, and starting the router and the
ControlCenter), follow [docker/README.md](../../docker/README.md).

### Run

On the compute machine, start the control container:

```bash
./docker/run_sigmarl_control.bash
```

Inside the container, run the policy on the seven lab vehicles:

```bash
python3 main_eval_lab.py \
    --backend lab \
    --lab-mode trajectory \
    --lab-traj-horizon 5 \
    --model-dir checkpoints/icra27/lab \
    --vehicle-ids 1,2,3,4,5,6,7 \
    --period-ms 50 \
    --manual-reset
```

The script waits until you press **START** in the RViz ControlCenter.

- `--manual-reset` is required on physical vehicles: on reset, the fleet stops
  and you place the vehicles by hand. Without this flag, colliding vehicles are
  teleported to new poses. Teleporting works only in the digital twin and
  requires the `gh-path-mode` submodule branches (see
  [docker/README.md](../../docker/README.md)).
- To test without any hardware, use `--backend mock`.
