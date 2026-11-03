# T-ITS 2026 evaluation (four stages)

Code of the evaluation of the paper "Safeguarding Multi-Agent Reinforcement
Learning for Connected and Automated Vehicles with Recursively Feasible Control
Barrier Functions". The paper's evaluation has four stages. The script names
follow an earlier numbering with three stages, in which zero-shot deployment was
part of Stage 1. The script names are kept so that output folders of earlier
runs stay valid. The tables below map the paper to the code.

## Paper to code

| Paper | Scripts | Output folder (`outputs/tits26/`) |
|---|---|---|
| Stage 1: pre-training with safety rewards (Fig. "training rollouts", Table "pre-trained policies") | `stage1_pretrain.py`, `stage1_plot_training.py`, `paper_fig_stage1_pretrain.py`, `stage1_zero_shot.py` (without a filter) | `stage1_pretrain`, `stage1_zero_shot` |
| Stage 2: zero-shot deployment with run-time filters (filter table, counterfactual footprints, headline figure) | `stage1_zero_shot.py --policy_iter 100`, `stage1_tables.py`, `stage1_counterfactual_rollouts.py`, `paper_fig_stage1_counterfactual.py`, `paper_fig_headline.py` | `stage1_filters_ttc_iter100`, `stage1_counterfactual` |
| Stage 3: fine-tuning with filters in the loop | `stage2_finetune_campaign.py`, `stage2_finetune_run.py`, `paper_fig_stage12_collisions.py` | `stage2_finetune` |
| Stage 4: comparison with human drivers | `stage3_human_comparison.py` (calls `sigmarl/evaluation_human_comparison/`), `paper_fig_stage3_footprints.py`, `stage3_gap_acceptance.py`, `stage3_gap_acceptance_plots.py`, `paper_fig_stage4_gap_acceptance.py` | `stage3_human_ft_recorded_maxlen6.5`, `stage3_gap_acceptance` |
| Computation time | `computation_time_scaling.py` | `computation_time` |
| Tables of the paper | `paper_tables_stage123.py` | `paper_tables` |

The paper names the filters and fine-tuning settings differently from the code
(`--method`, the keys of `METHODS` in `before_stages_filter_benchmark.py`):

| Paper | Code key | Folder in the released models |
|---|---|---|
| BH-CBF (ours) | `ours` (`--mode dec`) | `bh_cbf` |
| A-Geometry | `bh_capsule` | – |
| A-Condition | `ours_onestep` | – |
| A-Split | `bh_half_split` | – |
| A-Centralized | `ours` (`--mode cen`) | – |
| B-SAT | `b1_lse_sat` | – |
| B-Frenet | `b2_frenet_heading` | – |
| B-Disk | `b3_wang` | – |
| B-Dual | `t2_dual_distance` | `b_dual` |
| B-Backup | `b6_backup_cbf` | `b_backup` |
| B-MPS | `mps` | `b_mps` |
| Unfiltered / without filter, TTC reward | `unfiltered` (deployment), `none` (fine-tuning) | `no_filter_ttc_reward` |
| Without filter, BH-CBF reward (fine-tuning) | `none_bh` | `no_filter_bh_cbf_reward` |

The safety rewards of pre-training are TTC (`ttc`), one-step CBF (`onestep`),
and BH-CBF-informed (`bh`), as in the folder names `stage1_<reward>_seed_<k>`.
The other methods of `METHODS` (`b2_frenet_point`, `b3_disk`, `b4_collision_cone`,
`b5_convex_containment`, `b1_bh`, `b3_bh`) are not in the paper.

## Datasets

The rounD and inD datasets, including their maps, may not be redistributed,
so this repository includes neither their recordings nor their maps. Request
them from levelXdata: [rounD](https://levelxdata.com/round-dataset/) (the
roundabout, location 0) and [inD](https://levelxdata.com/ind-dataset/) (the
intersection, location 4). Then generate the two maps (see
`docs/lanelet2_maps.md`):

```bash
python scripts/lanelet2_maps/fix_lanelet_map.py --maps round_location0 ind_location4 \
    --round_dir <path>/rounD_v1.0 --ind_dir <path>/inD_v1.1
```

Stage 4 replays the rounD recordings of location 0; point `--data_dir` of
`sigmarl/evaluation_human_comparison/run.py` or the environment variable
`ROUND_DATA_DIR` to the dataset's `data` folder (see
`sigmarl/evaluation_human_comparison/README.md`).

## Trained models

The released models contain only the models that the paper uses, in two files
(`tits26_stage1_pretrain.zip` and `tits26_stage3_finetune.zip`). Unzip both into
`checkpoints/` at the repository root, so that the folders are
`checkpoints/tits26/stage1_pretrain/` and `checkpoints/tits26/stage3_finetune/`.
A full version, [`tits26_full.zip`](https://rwth-aachen.sciebo.de/s/Jo6g4qdWbPQsFqr) (about 920 MB), holds the same
models with the policies of every saved iteration (`checkpoints_cross_iter/`:
iterations 0 to 500 of pre-training and 0 to 100 of fine-tuning) and unzips to
the same folders.

- `stage1_pretrain/<map>/stage1_<reward>_seed_<k>/`: the 30 pre-trained
  policies of Stage 1 (three safety rewards, two maps, five seeds).
  `final_policy.pth` is the policy of iteration 500 (Stage 1). The TTC-reward
  policies also hold `checkpoints_cross_iter/policy_iter100.pth`, the unsafe
  source policies of Stages 2 and 3 and of the computation-time measurement.
  `reward*_data.json` holds the parameters and the training curves.
- `stage3_finetune/<source map>__stage1_ttc_seed_<k>__iter100/<target map>/<setting>/seed1/`:
  the 120 fine-tuned policies of Stage 3 (ten source policies, two target maps,
  six settings; folder names in the table above) with
  `checkpoints_cross_iter/policy_iter100.pth`, the training statistics
  (`train_stats.csv`), and the evaluations (`eval_*.jsonl`). Stage 4 uses the
  ten `bh_cbf` runs with the target map `round_location0`.

Examples (from the repository root):

```bash
# Stage 2: deploy the source policies with BH-CBF and B-Backup (fully decentralized)
SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_tits26.stage1_zero_shot \
    --checkpoints checkpoints/tits26/stage1_pretrain/*/stage1_ttc_seed_* --policy_iter 100 \
    --methods ours b6_backup_cbf --modes dec --seeds 201 202 203 \
    --out_root outputs/tits26/stage1_filters_ttc_iter100
# Stage 3: evaluate a fine-tuned policy with its filter
SIGMARL_AGENT_PROFILE=full_scale python sigmarl/evaluation_tits26/stage2_finetune_run.py eval \
    --run checkpoints/tits26/stage3_finetune/round_location0__stage1_ttc_seed_1__iter100/round_location0/bh_cbf/seed1 \
    --iter 100 --filter
# Stage 4: the fine-tuned BH-CBF policies against the human drivers of rounD
SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_tits26.stage3_human_comparison \
    --policies checkpoints/tits26/stage3_finetune/*/round_location0/bh_cbf/seed1 --iter 100
```

## Notes

Pre-training (Stage 1) runs `main_training.py` on the YAML files of
`scripts/generate_configs.py` (via `stage1_pretrain.py`). The human comparison
(Stage 4, rounD only) is in `sigmarl/evaluation_human_comparison/`; its
controlled vehicles have the recorded footprints and class acceleration limits
of the vehicles they replace (`stage3_human_comparison.py --vehicles recorded`,
default), while Stages 1 to 3 use the nominal 4.6 m x 2.0 m car. Vans longer
than 6.5 m count as heavy vehicles and end the windows
(`--max_vehicle_length 6.5`, default). `settings.py` holds the maps and vehicle
counts. Paths are date-free: `configs/tits26/stage1_pretrain`,
`checkpoints/tits26/stage1_pretrain`, `outputs/tits26/<stage>`.

The filters are the methods of `before_stages_filter_benchmark.py` (`METHODS`);
their decay rates and own parameters are the tuned values in
`sigmarl/cbf_baselines/tuned_defaults.json` (see below).

Speeds: all maps share the policy speed scale `Parameters.speed_normalization`
(50 km/h, the highest speed limit of the two maps, in the generated configs). Observations divide speeds
by it, and the speed action ranges over it; each map keeps its own speed limit
`max_speed`, to which the dynamics clip the action. A checkpoint deployed or
fine-tuned on another map takes that map's speed limit and CBF ranges
(`Parameters.retarget_scenario`) and keeps its speed scale and observation range.

## Before the stages: safety filters and their tuning

`before_stages_filter_benchmark.py` defines the safety filters (`METHODS`) that
all stages use and compares them with a CLF nominal controller (no MARL policy).
`before_stages_filter_tuning.py` tunes them; the paper summarizes this
tuning protocol.

The comparison is a **barrier swap**. Each baseline in `sigmarl/cbf_baselines/`
replaces one row type of the rectangle_sat CBF-QP: either the lane-boundary rows
(`--lane_cbf_baseline`) or the inter-vehicle rows (`--veh_veh_cbf_baseline`).
Everything else is identical for all methods:
- the CLF nominal controller;
- ProxSuite and the input box;
- the slacks (with a common weight; velocity barriers are scaled to meters);
- optionally, the exact backup shield and the standstill escape.

Each method's decay rate λ (`--cbf_lambda`) and its own parameter are tuned with
the same budget (`before_stages_filter_tuning.py`). The tuned values are the defaults
(`sigmarl/cbf_baselines/tuned_defaults.json`, per block and mode): a baseline
uses them when `Parameters.cbf_lambda` is None or the parameter's environment
variable is unset. Ours (BH-CBF) uses its value tuned without the shield
(`shield_off/ours/<mode>`, also with the shield on); ours_onestep and B6 keep
λ = 0.5 by default (see `tuned_cbf_lambda`). `before_stages_filter_benchmark.py` passes the
tuned λ of every method in the file (ours_onestep included). The final
comparison uses `--tuned '' --cbf_lambda 0.2` for all methods. In decentralized mode, each baseline uses
the split published with it (`PairBaseline.CROSS_SHARE`), and ours uses its
responsibility shares.

| Method | Lane-boundary rows | Inter-vehicle rows | Decentralized split |
|---|---|---|---|
| ours | BH-CBF, the proposed method: exact SAT (rectangle–polyline) along the backup tail (`--is_backup_horizon_cbf --no-is_two_step_rows`, sigmarl/backup_horizon_cbf.py) | exact SAT (rectangle–rectangle) along the backup tail | displacement budgets along the backup witness direction (Theorem 3) |
| ours_onestep | exact SAT (rectangle–polyline), one-step condition only (`--no-is_backup_horizon_cbf`) | exact SAT (rectangle–rectangle), one-step condition | responsibility shares |
| b6_backup_cbf | ours + backup-horizon tail rows with per-flow-point decay, backup CBF [Chen 2021] (`--condition_cbf_baseline b6_backup_cbf`, sigmarl/cbf_baselines/b6_backup_cbf.py) | ours + per-point tail rows | ½ split of the signed-distance decay, other vehicle on its backup tail [Chen, Singletary, Ames 2021] |
| b1_lse_sat | log-sum-exp smoothed SAT [Wu et al. 2025, Molnar & Ames 2023] | log-sum-exp smoothed SAT | ½ (Wu Eq. 34b) |
| b2_frenet_point | Frenet lateral offset, point footprint [Kalaria 2023, Xiao 2023, Yin 2023] | ours | — |
| b2_frenet_heading | Frenet lateral offset, exact corners of the rotated rectangle [Jiang 2024 Eqs. 8–9, Squires 2025 Eq. 36] | ours | — |
| b3_wang | ours | braking-aware disk certificate [Wang et al. 2017, Eq. 6] | ½ (Wang Eq. 12) |
| b3_disk | ours | disk distance, no braking term | ½ |
| b4_collision_cone | ours | collision cone, C3BF [Goswami et al. 2024] | 1 (other car as obstacle) |
| b5_convex_containment | corners inside the local convex free space [Molnar et al. 2025] | ours | — |
| t2_dual_distance | distance QP to facets [Thirugnanam et al. 2022] | distance QP between rectangles | ½ (centralized in the paper) |
| bh_half_split | ours along the backup tail | ours along the backup tail | ½ of the decay of the pair's backup value, other vehicle on its backup tail [Chen, Singletary, Ames 2021] (ablation of the corner budgets; sigmarl/cbf_baselines/bh_variants.py) |
| b1_bh | B1's smoothed SAT along the backup tail | B1's smoothed SAT along the backup tail | ½ as bh_half_split |
| b3_bh | ours along the backup tail | B3's disk distance (no braking term) along the backup tail | ½ as bh_half_split |
| mps | no QP: nominal kept unless the backup after it leaves a margin negative (rows with lambda = 1), else backup controls [Bastani 2021, Zhang et al. MAMPS] | same | corner budgets with lambda = 1 |
| unfiltered | none | none | — |

The deviations from each paper are listed in the module docstrings. The setup
is full-scale round_location0 and ind_location4 with 14 vehicles, a CLF
nominal controller, and 601 steps.
- **Modes.** Each method runs centralized (`cen`, `--max_group_size none`) and
  fully decentralized (`dec`, `--max_group_size 1`).
- **Blocks.**
  - `shield_off`: shield and standstill escape off. This is the barrier-only
    ablation.
  - `shield_on`: both on. This is the main result.

```bash
# 1. Tuning (output: outputs/tits26/before_stages_filter_tuning; seeds 101-103, round_location0 and ind_location4, 14 vehicles):
#    lambda per method, mode, and block, then each method's knob
SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_tits26.before_stages_filter_tuning --workers 4
#    Copy <out_root>/tuned.json to sigmarl/cbf_baselines/tuned_defaults.json to make
#    the result the defaults (keep its "_description" entry up to date).
# 2. Evaluation (seeds 1-5) with the tuned defaults (or --tuned <file>)
python -m sigmarl.evaluation_tits26.before_stages_filter_benchmark \
    --scenario_type round_location0 --n_agents 14 --workers 4 --out_root outputs/tits26/before_stages_filter_benchmark/round_location0
# Tables (summary_<block>.md, summary.csv)
SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_tits26.before_stages_filter_summary --out_root outputs/tits26/before_stages_filter_benchmark/round_location0
```

**Tuning rule.** Each method's settings are chosen on seeds 101–103 of both
round_location0 and ind_location4 (14 vehicles each); one setting per method,
mode, and block serves both maps. The rule is the same for every method:
1. A setting's goal score is its goals divided by the method's best goals on
   each map, averaged over the two maps.
2. Keep the settings that reach at least 75 % of the best goal score. A stalled
   fleet is trivially collision-free, so this filter comes first.
3. Among those, keep the settings whose mean total collisions per run (both
   maps) are at most 1 above the lowest.
4. Of these, select the one with the highest goal score.

Notes on the reported metrics:
- **CBF times.** Sequential is the actual wall-clock time of the CBF loop per
  step. Parallelized is the per-agent parallel-execution estimate: the shared
  preparation, plus the slowest agent's QP (including shield round 1), plus the
  slowest agent's shield round 2, plus the shared finalizer. For centralized
  controllers the two are equal.
- **Infeasible steps.** Steps in which a CBF-QP needed slack (more than 1 mm,
  in the barrier's unit) or failed.
- **Collisions after an infeasible step.** The share of collision onsets that
  had an infeasible step within the 15 steps before.
