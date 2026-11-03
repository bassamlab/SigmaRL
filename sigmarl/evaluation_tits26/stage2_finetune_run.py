# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Filter-guarded fine-tuning on a target map (Stage 2): train or evaluate one run.

Training (``train``) fine-tunes a MAPPO policy pre-trained on one map on a
target map (its own map or an unseen one) with one of the safety filters of
sigmarl/evaluation_tits26/before_stages_filter_benchmark.py (``--method``, fully
decentralized by default) in the data collector: the environment executes the
filtered action, PPO stores the policy's own action and log-prob, and the
reward is the task reward of the pre-training (progress, goal, collision, and
deadlock terms; rew_method "sparse") minus ``intervention_penalty`` per unit of
normalized filter correction. Without a filter (ablations), ``--method none``
fine-tunes with the source's pre-training reward, and ``--method none_bh`` with
the task reward plus the BH-CBF-informed reward (rew_method "cbfmargin_sparse"
with the backup-horizon condition, computed centrally without a QP). The source
model is the final one, a reward* model, or a training iteration
(``--source_model iter<N>``: checkpoints_cross_iter/policy_iter<N>.pth).

Per iteration, ``train_stats.csv`` records the training rollouts' collisions,
goals, deadlocks, and speed; the filter's interventions and mean correction,
braking fallbacks, env-steps with an infeasible (slack) QP, and env-steps
outside the certified set C with the smallest backup value (our margins, as in
Stage 1; ``--no-log_backup_value`` skips them); and the wall-clock time.

Evaluation (``eval``) rolls out one saved iteration checkpoint
deterministically with or without a filter (``--filter``: the run's filter, or
BH-CBF for a run without one; ``--eval_method`` overrides it) and appends one
JSON line. One checkpoint per process: a second rollout in the same process
starts from leftover scenario state, so its result would depend on what ran
before.

SIGMARL_AGENT_PROFILE=full_scale python sigmarl/evaluation_tits26/stage2_finetune_run.py train \
    --source checkpoints/<date>/round_location0/<run> --scenario_type ind_location4 \
    --method ours --out outputs/<date>-stage2/<run>/ind_location4/ours/seed1 --seed 1
SIGMARL_AGENT_PROFILE=full_scale python sigmarl/evaluation_tits26/stage2_finetune_run.py eval \
    --run outputs/<date>-stage2/<run>/ind_location4/ours/seed1 --iter 20 --filter
"""

import argparse
import csv
import glob
import json
import os
import re
import time

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

import sigmarl.helper_training as helper_training
from sigmarl.cbf_baselines import CONDITION_BASELINES
from sigmarl.evaluation_tits26.before_stages_filter_benchmark import METHODS
from sigmarl.helper_common import start_run_log
from sigmarl.helper_training import SaveData, cbf_filter_deviation
from sigmarl.mappo_cavs import MAPPOCAVs

CHECKPOINT_DIR = "checkpoints_cross_iter"
# Reward of fine-tuning with a filter: the task reward of the pre-training
# (progress + goal, collision, and deadlock terms); the intervention penalty is
# added in FinetuneMAPPO._process_tensordict_data
FILTER_REWARD_METHOD = "sparse"
# Reward of fine-tuning without a filter (None: the source's pre-training reward)
NO_FILTER_REWARD_METHODS = {"none": None, "none_bh": "cbfmargin_sparse"}
# The filter of the evaluation with a filter of a run without one
DEFAULT_EVAL_METHOD = "ours"

# Filter statistics accumulated by the patched `cbf_constrained_policy_main`
_FILTER_STATS = {
    "agent_steps": 0,
    "intervened": 0,
    "dev_vel": 0.0,
    "dev_steer": 0.0,
    "fallback_agents": 0,
    "env_steps": 0,
    "slack_env_steps": 0,
    "outside_c_env_steps": 0,
    "min_backup_value": float("inf"),
}
_INTERVENTION_TOL = 1e-3  # Normalized deviation above which a step counts as intervened
_UNRECOVERABLE_TOL = 1e-3  # Backup value below -tol: outside C (as in Stage 1)
# Whether the patched filter step also evaluates the joint backup values
_LOG_BACKUP_VALUE = [True]
# Per collected step: normalized filter deviation [n_envs, n_agents, 1], consumed
# by the intervention penalty
_DEV_BUFFER = []
_original_cbf_main = helper_training.cbf_constrained_policy_main


def _env_controllers(item) -> list:
    """Controllers of one env: the agents' (fully decentralized) or the one."""
    return item if isinstance(item, list) else [item]


def _fallback_agents(cbf_controllers) -> int:
    """Agents switched to the braking backup so far, over all controllers."""
    return sum(
        ctrl.lane_sat_stats["bh_fallback_agents"]
        for item in cbf_controllers
        for ctrl in _env_controllers(item)
    )


def _cbf_main_with_stats(tensordict, policy, cbf_controllers, **kwargs):
    """Run the filter and record how much it changed the policy's actions (see
    helper_training.cbf_filter_deviation), its fallbacks and infeasible QPs,
    and whether the joint state left the certified set."""
    first = cbf_controllers[0]
    first = first[0] if isinstance(first, list) else first
    if not first.parameters.is_apply_cbf_action:
        # No filter (the controllers only compute reward margins)
        return _original_cbf_main(tensordict, policy, cbf_controllers, **kwargs)
    fallback_before = _fallback_agents(cbf_controllers)
    out = _original_cbf_main(tensordict, policy, cbf_controllers, **kwargs)
    _FILTER_STATS["fallback_agents"] += _fallback_agents(cbf_controllers) - (
        fallback_before
    )
    _FILTER_STATS["env_steps"] += len(cbf_controllers)
    _FILTER_STATS["slack_env_steps"] += sum(
        any(ctrl.last_slack_active for ctrl in _env_controllers(item))
        for item in cbf_controllers
    )
    if _LOG_BACKUP_VALUE[0]:
        backup = first.backup_values_all_envs(cbf_controllers)
        _FILTER_STATS["outside_c_env_steps"] += int(
            (backup < -_UNRECOVERABLE_TOL).any(axis=1).sum()
        )
        _FILTER_STATS["min_backup_value"] = min(
            _FILTER_STATS["min_backup_value"], float(backup.min())
        )
    dev_vel, dev_steer = cbf_filter_deviation(cbf_controllers)
    _DEV_BUFFER.append((dev_vel + dev_steer).unsqueeze(-1))
    _FILTER_STATS["agent_steps"] += dev_vel.numel()
    _FILTER_STATS["intervened"] += int(
        ((dev_vel > _INTERVENTION_TOL) | (dev_steer > _INTERVENTION_TOL)).sum()
    )
    _FILTER_STATS["dev_vel"] += float(dev_vel.sum())
    _FILTER_STATS["dev_steer"] += float(dev_steer.sum())
    return out


helper_training.cbf_constrained_policy_main = _cbf_main_with_stats


def _reset_filter_stats():
    for key in _FILTER_STATS:
        _FILTER_STATS[key] = (
            float("inf")
            if key == "min_backup_value"
            else 0.0
            if key in ("dev_vel", "dev_steer")
            else 0
        )


def _filter_summary() -> dict:
    """Rates of the accumulated filter statistics (see _cbf_main_with_stats);
    None without a filter."""
    if _FILTER_STATS["agent_steps"] == 0:
        return dict.fromkeys(
            (
                "intervention_rate",
                "mean_dev_vel",
                "mean_dev_steer",
                "fallback_per_1000",
                "slack_env_step_rate",
                "outside_c_env_step_rate",
                "min_backup_value",
            )
        )
    steps = _FILTER_STATS["agent_steps"]
    env_steps = max(_FILTER_STATS["env_steps"], 1)
    is_backup = _LOG_BACKUP_VALUE[0] and _FILTER_STATS["env_steps"] > 0
    return {
        "intervention_rate": _FILTER_STATS["intervened"] / steps,
        "mean_dev_vel": _FILTER_STATS["dev_vel"] / steps,
        "mean_dev_steer": _FILTER_STATS["dev_steer"] / steps,
        "fallback_per_1000": 1e3 * _FILTER_STATS["fallback_agents"] / steps,
        "slack_env_step_rate": _FILTER_STATS["slack_env_steps"] / env_steps,
        "outside_c_env_step_rate": (
            _FILTER_STATS["outside_c_env_steps"] / env_steps if is_backup else None
        ),
        "min_backup_value": _FILTER_STATS["min_backup_value"] if is_backup else None,
    }


def source_policy_path(source: str, model_name: str) -> str:
    """Policy weights of a source model: final, reward*, or iter<N>."""
    if model_name.startswith("iter"):
        return os.path.join(source, CHECKPOINT_DIR, f"policy_{model_name}.pth")
    return os.path.join(source, f"{model_name}_policy.pth")


def best_source_model(source: str) -> str:
    """Name (e.g., "reward4.94") of the highest-reward model of a checkpoint."""
    names = [
        os.path.basename(path).removesuffix("_data.json")
        for path in glob.glob(os.path.join(source, "reward*_data.json"))
    ]
    if not names:
        raise FileNotFoundError(f"No reward*_data.json in '{source}'.")
    return max(names, key=lambda name: float(re.sub(r"^reward", "", name)))


def load_source_parameters(source: str, model_name: str):
    """Parameters saved with the source checkpoint ("final" and "iter<N>":
    those saved with its highest-reward model, the same run settings)."""
    data_name = (
        best_source_model(source)
        if model_name == "final" or model_name.startswith("iter")
        else model_name
    )
    with open(os.path.join(source, f"{data_name}_data.json"), "r") as file:
        return SaveData.from_dict(json.load(file)).parameters


def apply_method(parameters, method: str, mode: str):
    """Filter settings of a method of before_stages_filter_benchmark.METHODS, as
    main_testing_cbf_filter.py sets them (mode "dec": fully decentralized,
    "cen": centralized); lambda and the baselines' own parameters are the tuned
    values (cbf_lambda None, see sigmarl/cbf_baselines)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'; one of {sorted(METHODS)}.")
    is_decentralized = mode == "dec"
    parameters.is_grouping_agents = is_decentralized
    parameters.max_group_size = 1 if is_decentralized else 2
    parameters.is_using_centralized_cbf = not is_decentralized
    if METHODS[method] is None:
        lane, veh = "unfiltered", "unfiltered"
    else:
        lane, veh = METHODS[method]
    parameters.lane_cbf_baseline = lane
    parameters.veh_veh_cbf_baseline = veh
    parameters.condition_cbf_baseline = (
        method if method in CONDITION_BASELINES else "none"
    )
    parameters.is_backup_horizon_cbf = method == "ours" or method in CONDITION_BASELINES
    parameters.is_two_step_rows = False
    parameters.cbf_qp_failure_fallback = (
        "brake" if parameters.is_backup_horizon_cbf else None
    )
    parameters.cbf_lambda = None


def apply_target_settings(parameters, args):
    """Target map, agent count, and the CBF settings of the paper; the filter of
    args.method (none: the centralized BH-CBF margins of the reward only)."""
    if args.scenario_type != parameters.scenario_type:
        # The map's speed limit and CBF ranges; the policy's speed scale stays
        parameters.retarget_scenario(args.scenario_type)
    parameters.n_agents = args.n_agents
    parameters.scale_map = 1.0
    parameters.is_center_centerline = True
    parameters.random_seed = args.seed
    parameters.is_load_model = False
    parameters.is_load_final_model = False
    parameters.is_load_pretrained_motion_policy = False
    parameters.is_load_pretrained_lambda_policy = False
    parameters.is_using_prioritized_marl = False
    parameters.is_save_simulation_video = False
    parameters.is_real_time_rendering = False
    parameters.is_save_eval_results = False
    parameters.is_load_out_td = False
    parameters.is_visualize_observed_agents = False
    # Spawn at rest, so that braking keeps every spawn collision-free (the source
    # checkpoint may have been trained with random initial speeds)
    parameters.is_zero_initial_speed = True
    parameters.cbf_margin_reward_condition = "backup_horizon"

    parameters.lane_distance_type = "softmin"
    parameters.lane_cbf_footprint = "rectangle_sat"
    parameters.veh_veh_cbf_footprint = "rectangle_sat"
    parameters.is_proxsuite_qp = True
    parameters.grouping_method = "interaction_risk"
    parameters.grouping_use_hysteresis = True
    parameters.is_cbf_limit_acc_to_nominal = False
    parameters.n_circles_approximate_vehicle = 3
    parameters.adaptive_lambda = True
    parameters.is_learn_cbf_lambda = False
    parameters.is_aggregate_veh_veh_safety = True
    parameters.is_aggregate_lane_veh_safety = False
    if args.method in NO_FILTER_REWARD_METHODS:
        # The BH-CBF margins of the reward are computed centrally without a QP
        apply_method(parameters, DEFAULT_EVAL_METHOD, "cen")
        parameters.is_solve_qp = False
    else:
        apply_method(parameters, args.method, args.mode)
        parameters.is_solve_qp = True
    parameters.is_exact_backup_shield = False
    parameters.is_standstill_escape = False
    parameters.is_visu_shield_rollout = False
    parameters.is_use_sampled_lane_boundary_for_cbf = True
    parameters.is_train_motion_policy = True
    parameters.is_train_lambda_policy = False
    parameters.is_alternating_action_lambda_training = False
    parameters.nom_controller_type = "rl"
    parameters.is_joint_task = False


class FinetuneMAPPO(MAPPOCAVs):
    """MAPPO driver with policy initialization, critic warm-up, intervention
    penalty, and safety logging.

    Under the filter, PPO stores the policy's own action and log-prob, while
    reward and next state come from the filtered action. Since the filter
    prevents collisions, the collision penalty no longer tells the policy what
    is unsafe; the intervention penalty ``reward -= intervention_penalty *
    deviation`` does instead.
    """

    def __init__(self, parameters, init_policy_path, warmup_iters, stats_path, penalty):
        super().__init__(parameters)
        self.intervention_penalty = penalty
        self.init_policy_path = init_policy_path
        self.warmup_iters = warmup_iters
        self.stats_path = stats_path
        self.i_iter_done = 0
        self.t_start = time.time()
        self.cum = {"agent_collisions": 0, "lane_collisions": 0, "frames": 0}

    def _setup_decision_making_module(self, env):
        module = super()._setup_decision_making_module(env)
        module.policy.load_state_dict(
            torch.load(self.init_policy_path, weights_only=True)
        )
        print(f"[INFO] Initialized the policy from '{self.init_policy_path}'.")
        return module

    def _process_tensordict_data(self, tensordict_data, env):
        super()._process_tensordict_data(tensordict_data, env)
        if _DEV_BUFFER and self.intervention_penalty != 0.0:
            # Steps are appended in collection order; stack along time
            dev = torch.stack(_DEV_BUFFER, dim=1)
            key = ("next", "agents", "reward")
            reward = tensordict_data.get(key)
            if dev.shape != reward.shape:
                raise RuntimeError(f"Deviation {dev.shape} != reward {reward.shape}.")
            tensordict_data.set(key, reward - self.intervention_penalty * dev)
        _DEV_BUFFER.clear()

    def _train_on_batch(self, optimization_module, priority_module, batch, env, phase):
        # The source critic is centralized over a different agent count, so the
        # critic starts fresh; fit it before the actor moves.
        if self.i_iter_done >= self.warmup_iters:
            return super()._train_on_batch(
                optimization_module, priority_module, batch, env, phase
            )
        loss_critic = optimization_module.loss_module(batch)["loss_critic"]
        optimization_module.optim.zero_grad(set_to_none=True)
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(
            optimization_module.critic.parameters(), self.parameters.max_grad_norm
        )
        optimization_module.optim.step()

    def _log_and_save(self, tensordict_data, *args, **kwargs):
        super()._log_and_save(tensordict_data, *args, **kwargs)
        info = ("next", "agents", "info")
        col_agents = int(
            tensordict_data.get(info + ("is_collision_with_agents",)).sum()
        )
        col_lane = int(
            tensordict_data.get(info + ("is_collision_with_lanelets",)).sum()
        )
        goals = int(tensordict_data.get(info + ("is_reach_goal",)).sum())
        deadlocked = int(tensordict_data.get(info + ("is_deadlocked",)).sum())
        speed = tensordict_data.get(info + ("vel",)).norm(dim=-1).mean().item()
        agent_steps = tensordict_data.get(info + ("is_reach_goal",)).numel()
        self.cum["agent_collisions"] += col_agents
        self.cum["lane_collisions"] += col_lane
        self.cum["frames"] += tensordict_data.numel()
        row = {
            "iter": self.i_iter_done + 1,
            "frames": self.cum["frames"],
            "agent_steps": agent_steps,
            "wall_s": round(time.time() - self.t_start, 1),
            "episode_reward": args[3].episode_reward_mean_list[-1],
            "param_indep_reward": (
                args[3].parameter_independent_reward_mean_list or [None]
            )[-1],
            "agent_collision_steps": col_agents,
            "lane_collision_steps": col_lane,
            "cum_agent_collision_steps": self.cum["agent_collisions"],
            "cum_lane_collision_steps": self.cum["lane_collisions"],
            "goal_steps": goals,
            "deadlocked_steps": deadlocked,
            "mean_speed": round(speed, 3),
            **_filter_summary(),
            "warmup": int(self.i_iter_done < self.warmup_iters),
        }
        is_new = not os.path.exists(self.stats_path)
        with open(self.stats_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        _reset_filter_stats()
        _DEV_BUFFER.clear()
        self.i_iter_done += 1


def train(args):
    parameters = load_source_parameters(args.source, args.source_model)
    apply_target_settings(parameters, args)
    os.makedirs(args.out, exist_ok=True)
    start_run_log(args.out, "training")
    is_filter = args.method not in NO_FILTER_REWARD_METHODS
    _LOG_BACKUP_VALUE[0] = args.log_backup_value

    parameters.where_to_save = args.out
    parameters.is_testing_mode = False
    parameters.is_continue_train = True
    # Fixed-iteration checkpoints suffice; skipping the best-model figure avoids
    # Matplotlib image-cache races between parallel runs.
    parameters.is_save_intermediate_model = False
    parameters.episode_reward_intermediate = -1e3
    parameters.episode_reward_mean_current = -1e3
    parameters.n_iters = args.n_iters
    parameters.num_vmas_envs = args.num_vmas_envs
    parameters.max_steps = args.max_steps
    parameters.must_save_checkpoints = args.save_iters
    if args.rew_method is not None:
        parameters.rew_method = args.rew_method
    elif is_filter:
        parameters.rew_method = FILTER_REWARD_METHOD
    elif NO_FILTER_REWARD_METHODS[args.method] is not None:
        parameters.rew_method = NO_FILTER_REWARD_METHODS[args.method]
    print(f"[INFO] Fine-tuning reward: rew_method '{parameters.rew_method}'.")
    parameters.is_using_cbf_testing = False
    # Without a filter, the CBF controllers only compute the reward's margins
    parameters.is_using_cbf_training = is_filter or "cbfmargin" in parameters.rew_method
    parameters.is_apply_cbf_action = is_filter
    parameters.is_store_rl_action_under_cbf = is_filter

    with open(os.path.join(args.out, "run_config.json"), "w") as file:
        json.dump(vars(args), file, indent=2)
    trainer = FinetuneMAPPO(
        parameters,
        init_policy_path=source_policy_path(args.source, args.source_model),
        warmup_iters=args.warmup_iters,
        stats_path=os.path.join(args.out, "train_stats.csv"),
        penalty=args.intervention_penalty if is_filter else 0.0,
    )
    trainer.train()


def evaluate(args):
    """Deterministic rollout of one saved iteration checkpoint."""
    from sigmarl.mappo_cavs import mappo_cavs

    with open(os.path.join(args.run, "run_config.json"), "r") as file:
        run_config = argparse.Namespace(**json.load(file))
    run_config.seed = args.seed
    if args.filter:
        run_config.method = args.eval_method or (
            DEFAULT_EVAL_METHOD
            if run_config.method in NO_FILTER_REWARD_METHODS
            else run_config.method
        )
    _LOG_BACKUP_VALUE[0] = args.log_backup_value
    parameters = load_source_parameters(run_config.source, run_config.source_model)
    apply_target_settings(parameters, run_config)
    parameters.is_testing_mode = True
    parameters.is_continue_train = False
    parameters.is_save_detailed_info = False
    parameters.rew_method = "sparse"
    parameters.num_vmas_envs = args.num_vmas_envs
    parameters.max_steps = args.max_steps
    parameters.is_using_cbf_training = False
    parameters.is_using_cbf_testing = args.filter
    parameters.is_apply_cbf_action = args.filter
    parameters.is_store_rl_action_under_cbf = False

    # Build the env, policy, and filter (loading the source model the way
    # main_testing.py does), then load the checkpoint's weights.
    parameters.where_to_save = run_config.source
    parameters.is_load_model = True
    (
        env,
        decision_making_module,
        _,
        priority_module,
        cbf_controllers,
        parameters,
    ) = mappo_cavs(parameters=parameters)
    tag = "filter" if args.filter else "nofilter"
    if args.filter and args.eval_method is not None:
        tag += f"-{args.eval_method}"
    out_path = os.path.join(args.run, f"eval_{tag}_seed{args.seed}.jsonl")
    policy_path = os.path.join(args.run, CHECKPOINT_DIR, f"policy_iter{args.iter}.pth")
    decision_making_module.policy.load_state_dict(
        torch.load(policy_path, weights_only=True)
    )
    env.set_seed(args.seed)
    _reset_filter_stats()
    _DEV_BUFFER.clear()
    t_0 = time.time()
    with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC):
        out_td, _ = env.rollout(
            max_steps=parameters.max_steps - 1,
            policy=decision_making_module.policy,
            priority_module=priority_module,
            cbf_controllers=cbf_controllers,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )
    _DEV_BUFFER.clear()
    result = summarize_rollout(out_td, parameters)
    result.update(
        iter=int(args.iter),
        filter=args.filter,
        eval_method=run_config.method if args.filter else "none",
        seed=args.seed,
        num_envs=parameters.num_vmas_envs,
        wall_s=round(time.time() - t_0, 1),
        **_filter_summary(),
    )
    print(json.dumps(result))
    with open(out_path, "a") as file:
        file.write(json.dumps(result) + "\n")


def summarize_rollout(out_td, parameters):
    """Rollout statistics with the definitions of save_evaluation_summary."""
    stats = helper_training._build_rollout_statistics(out_td)
    collisions = stats["collisions"]
    return {
        "reward_per_agent_min": helper_training.compute_parameter_independent_reward_per_minute(
            out_td
        ),
        "collision_events": collisions["total"]["event_count"],
        "agent_agent_events": collisions["agent_agent"]["event_count"],
        "agent_lane_events": collisions["agent_lane"]["event_count"],
        "envs_with_collision": collisions["total"]["environments_affected"],
        "goal_events": stats["goal_reaching"]["agent_step_count"],
        "throughput_veh_s": stats["traffic_throughput"]["vehicles_per_second"],
        "deadlock_events": stats["deadlocks"]["event_count"],
        "mean_speed": stats["speed_mps"]["mean"],
        "duration_s": stats["timing"]["mean_simulated_duration_s"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train")
    tr.add_argument(
        "--method",
        required=True,
        choices=[*NO_FILTER_REWARD_METHODS, *METHODS],
        help="Safety filter in the loop (see before_stages_filter_benchmark.METHODS); none, none_bh: "
        "no filter, with the source's reward or the BH-CBF-informed reward.",
    )
    tr.add_argument(
        "--mode",
        default="dec",
        choices=["dec", "cen"],
        help="Fully decentralized or centralized filter.",
    )
    tr.add_argument("--out", required=True)
    tr.add_argument("--seed", type=int, default=1)
    tr.add_argument("--source", required=True, help="Pre-trained checkpoint folder.")
    tr.add_argument(
        "--source_model",
        default="final",
        help="Model name in --source: final (the last iteration, default), a "
        "reward* model, or iter<N> (checkpoints_cross_iter/policy_iter<N>.pth).",
    )
    tr.add_argument("--scenario_type", required=True, help="Target map.")
    tr.add_argument("--n_agents", type=int, required=True)
    tr.add_argument("--n_iters", type=int, default=100)
    tr.add_argument("--num_vmas_envs", type=int, default=16)
    tr.add_argument("--max_steps", type=int, default=128)
    tr.add_argument("--warmup_iters", type=int, default=5)
    tr.add_argument(
        "--intervention_penalty",
        type=float,
        default=3.0,
        help="With a filter: reward penalty per unit of normalized filter correction (speed + steering).",
    )
    tr.add_argument(
        "--rew_method",
        default=None,
        help=(
            f"Reward method (default: '{FILTER_REWARD_METHOD}' with a filter; "
            "without one, the source's reward (none) or "
            f"'{NO_FILTER_REWARD_METHODS['none_bh']}' (none_bh))."
        ),
    )
    tr.add_argument(
        "--save_iters", type=int, nargs="+", default=list(range(0, 101, 10))
    )
    ev = sub.add_parser("eval")
    for command in (tr, ev):
        command.add_argument(
            "--log_backup_value",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="With a filter: log the env-steps outside C and the smallest "
            "backup value (one batched backup rollout per step).",
        )
    ev.add_argument("--run", required=True)
    ev.add_argument("--iter", default="0", help="Iteration of the checkpoint")
    ev.add_argument("--filter", action=argparse.BooleanOptionalAction, default=False)
    ev.add_argument(
        "--eval_method",
        default=None,
        choices=list(METHODS),
        help="With --filter: this filter instead of the run's.",
    )
    ev.add_argument("--seed", type=int, default=100)
    ev.add_argument("--num_vmas_envs", type=int, default=8)
    ev.add_argument("--max_steps", type=int, default=601)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)
