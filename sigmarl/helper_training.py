# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from vmas.simulator.core import World

# Tensordict modules
from tensordict.tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule

from torchrl.collectors import SyncDataCollector

# Env
from torchrl.envs import TransformedEnv, StepCounter
from torchrl.envs.utils import (
    set_exploration_type,
    _terminated_or_truncated,
    step_mdp,
    ExplorationType,
)
from torchrl.envs.common import EnvBase

from sigmarl.helper_common import (
    Parameters,
    Vehicle,
    compute_applied_action_control,
    get_model_name,
    is_latex_available,
)


# TorchRL Utils
from torchrl._utils import (
    RL_WARNINGS,
)

# Losses

from torchrl.data.utils import DEVICE_TYPING


# Utils
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams["text.usetex"] = is_latex_available()

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import json
import math
import os
import re
import time

import numpy as np

DEFAULT_EXPLORATION_TYPE: ExplorationType = ExplorationType.RANDOM

import warnings

from sigmarl.backup_horizon_cbf import WRITE_BARRIER, rollout_requests
from sigmarl.constants import AGENTS

if TYPE_CHECKING:
    from sigmarl.cbf_qp import CBFQP as CBFQPCVXPY
    from sigmarl.cbf_qp_proxsuite import CBFQP as CBFQPProxSuite
else:
    CBFQPCVXPY = Any
    CBFQPProxSuite = Any


def schedule_rew_coefficient(t, t1=0, t2=100, y1=0, y2=None):
    """Linearly schedule a reward or penalty coefficient."""
    if y2 is None:
        raise ValueError("y2 must be provided.")
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1.")

    if t <= t1:
        return _match_scheduled_coefficient_type(y1, y2)
    if t >= t2:
        return _match_scheduled_coefficient_type(y2, y1)

    ratio = (t - t1) / (t2 - t1)
    return y1 + (y2 - y1) * ratio


def _match_scheduled_coefficient_type(value, reference):
    if torch.is_tensor(value) or not torch.is_tensor(reference):
        return value
    return torch.full_like(reference, value)


def redistribute_rewards(
    rewards: torch.Tensor,
    priority_scores: Optional[torch.Tensor],
    alpha: float = 0.2,
    *,
    mode: str = "linear",
    coeff: float = 1.0,
    even: bool = False,
    group_threshold: Optional[float] = None,
    distances: Optional[torch.Tensor] = None,
    prioritized_group_threshold: Optional[float] = None,
    weighted: bool = False,
    priority_beta: float = 1.0,
) -> torch.Tensor:
    """Redistribute reward and cost mass while conserving each separately.

    Positive rewards are redistributed toward higher-priority agents. Negative
    rewards are treated as nonnegative costs and redistributed using the inverse
    priority order, so higher-priority agents receive less pooled cost. This is
    a directional shaping guarantee: raw rewards can still determine the final
    absolute ordering between agents.
    """

    if mode not in {"linear", "adaptive"}:
        raise ValueError("`mode` must be either 'linear' or 'adaptive'.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("`alpha` must be in [0, 1].")
    if coeff < 0.0:
        raise ValueError("`coeff` must be non-negative.")
    if priority_beta <= 0.0:
        raise ValueError("`priority_beta` must be positive.")

    def squeeze_last(tensor):
        if tensor is not None and tensor.ndim == 3 and tensor.shape[-1] == 1:
            return tensor.squeeze(-1)
        return tensor

    def connected_groups(distance_matrix, threshold) -> List[List[int]]:
        n_agents = distance_matrix.shape[0]
        visited = torch.zeros(n_agents, dtype=torch.bool, device=distance_matrix.device)
        groups = []
        for root in range(n_agents):
            if visited[root]:
                continue
            stack = [root]
            visited[root] = True
            component = []
            while stack:
                current = stack.pop()
                component.append(current)
                neighbours = torch.nonzero(
                    distance_matrix[current] <= threshold, as_tuple=False
                ).flatten()
                for neighbour in neighbours.tolist():
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        stack.append(neighbour)
            groups.append(component)
        return groups

    def apply_prioritized_mass(mass_slice, score_slice):
        scores = (
            score_slice if torch.is_floating_point(score_slice) else score_slice.float()
        )
        soft = torch.softmax(scores, dim=-1)
        alpha_local = torch.as_tensor(
            alpha, dtype=mass_slice.dtype, device=mass_slice.device
        )
        if mode == "adaptive":
            variance = scores.var(dim=-1, keepdim=True, unbiased=False)
            alpha_local = (alpha_local + coeff * variance).clamp(0.0, 1.0)
        sacrifice = alpha_local * (1 - soft)
        retained = mass_slice * (1 - sacrifice)
        gain = (mass_slice * sacrifice).sum(dim=-1, keepdim=True)
        return retained + gain * soft

    def apply_prioritized(reward_slice, score_slice):
        positive_mass = reward_slice.clamp_min(0)
        cost_mass = (-reward_slice).clamp_min(0)
        redistributed_positive = apply_prioritized_mass(positive_mass, score_slice)
        redistributed_cost = apply_prioritized_mass(cost_mass, -score_slice)
        return redistributed_positive - redistributed_cost

    rewards_2d = squeeze_last(rewards)
    scores_2d = squeeze_last(priority_scores)
    if rewards_2d is None or rewards_2d.ndim != 2:
        raise ValueError("`rewards` must have shape [env, agent] or [env, agent, 1].")
    if scores_2d is not None and not torch.is_floating_point(scores_2d):
        scores_2d = scores_2d.float()

    batch_size, n_agents = rewards_2d.shape
    if (
        weighted
        and distances is not None
        and prioritized_group_threshold is not None
        and prioritized_group_threshold > 0.0
    ):
        if scores_2d is None:
            raise ValueError(
                "`priority_scores` must be provided for weighted redistribution."
            )
        result = rewards_2d.clone()
        threshold = prioritized_group_threshold
        for batch in range(batch_size):
            dist = distances[batch]
            distance_weights = torch.clamp(threshold - dist, min=0.0) / threshold
            diagonal = torch.arange(n_agents, device=dist.device)
            distance_weights[diagonal, diagonal] = 1.0

            def redistribute_weighted_mass(mass, scores):
                priority_weights = torch.softmax(scores * priority_beta, dim=-1)
                weights = distance_weights * priority_weights.unsqueeze(-1)
                weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
                neighbour_mass = weights @ mass
                return mass * (1 - alpha) + neighbour_mass * alpha

            positive_mass = rewards_2d[batch].clamp_min(0)
            cost_mass = (-rewards_2d[batch]).clamp_min(0)
            redistributed_positive = redistribute_weighted_mass(
                positive_mass, scores_2d[batch]
            )
            redistributed_cost = redistribute_weighted_mass(
                cost_mass, -scores_2d[batch]
            )
            result[batch] = redistributed_positive - redistributed_cost
        return result.unsqueeze(-1)

    if even:
        if (
            distances is not None
            and group_threshold is not None
            and group_threshold > 0.0
        ):
            result = rewards_2d.clone()
            for batch in range(batch_size):
                for group in connected_groups(distances[batch], group_threshold):
                    result[batch, group] = rewards_2d[batch, group].mean()
            return result.unsqueeze(-1)
        mean_reward = rewards_2d.mean(dim=-1, keepdim=True)
        return mean_reward.expand_as(rewards_2d).unsqueeze(-1)

    if scores_2d is None:
        raise ValueError(
            "`priority_scores` must be provided for prioritized redistribution."
        )
    if (
        distances is not None
        and prioritized_group_threshold is not None
        and prioritized_group_threshold > 0.0
    ):
        result = rewards_2d.clone()
        for batch in range(batch_size):
            for group in connected_groups(
                distances[batch], prioritized_group_threshold
            ):
                result[batch, group] = apply_prioritized(
                    rewards_2d[batch, group], scores_2d[batch, group]
                )
        return result.unsqueeze(-1)
    return apply_prioritized(rewards_2d, scores_2d).unsqueeze(-1)


def _scenario_parameters(env):
    return env.base_env.scenario_name.parameters


def _consume_pre_reset_render_frame(env):
    """Consume an optional frame captured before a scenario collision reset."""
    scenario = env.base_env.scenario_name
    consume = getattr(scenario, "consume_pre_reset_render_frame", None)
    if consume is None:
        return None
    return consume()


def _scenario_distances(env):
    """Return the latest, pre-reset pairwise agent distances."""

    scenario = env.base_env.scenario_name
    if hasattr(scenario, "distances"):
        return scenario.distances.agents
    return scenario.world_state.distances.agents


def _generate_behavior_action(
    tensordict, policy, priority_module, env, opponent_predictor=None
):
    """Generate priority metadata first, followed by the behavior action."""

    parameters = _scenario_parameters(env)
    scores = None
    if parameters.is_using_prioritized_marl:
        if priority_module is None:
            raise ValueError("Prioritized MARL requires a priority module.")
        priority_module(tensordict)
        scores = tensordict.get(priority_module.prefix_key + ("scores",)).squeeze(-1)
        env.base_env.scenario_name.priority_rank = tensordict.get(
            priority_module.prefix_key + ("rank",)
        )
        env.base_env.scenario_name.priority_scores = scores
    if parameters.is_using_opponent_modeling:
        opponent_modeling(
            tensordict=tensordict,
            policy=policy,
            n_nearing_agents_observed=parameters.n_nearing_agents_observed,
            nearing_agents_indices=(
                env.base_env.scenario_name.observations.nearing_agents_indices
                if hasattr(env.base_env.scenario_name, "observations")
                else env.base_env.scenario_name.observation_provider.observations.nearing_agents_indices
            ),
            neighbor_valid=env.base_env.scenario_name.observation_provider.get_neighbor_valid(),
            opponent_predictor=opponent_predictor,
        )
    policy(tensordict)
    return scores


def _notify_priority_episode_end(priority_module, tensordict):
    """Notify cached random priorities about environments that just finished."""
    if priority_module is None:
        return
    episode_end = tensordict.get(("next", "done"), None)
    if episode_end is not None:
        priority_module.notify_episode_end(episode_end)


def _redistribute_transition_reward(env, tensordict, scores):
    """Apply reward propagation before any completed environments are reset."""

    parameters = _scenario_parameters(env)
    if not (
        (parameters.is_using_prioritized_marl and scores is not None)
        or parameters.is_even_reward_distribution
    ):
        return
    env_reward_key = (
        (env.reward_key,) if isinstance(env.reward_key, str) else tuple(env.reward_key)
    )
    reward_key = ("next",) + env_reward_key
    rewards = tensordict.get(reward_key)
    tensordict.set(
        reward_key,
        redistribute_rewards(
            rewards,
            scores,
            parameters.reward_distribution_alpha,
            mode=parameters.reward_distribution_mode,
            coeff=parameters.reward_distribution_alpha_coeff,
            even=parameters.is_even_reward_distribution,
            group_threshold=parameters.even_reward_group_threshold,
            distances=_scenario_distances(env),
            prioritized_group_threshold=parameters.prioritized_reward_group_threshold,
            weighted=parameters.is_weighted_reward_sharing,
            priority_beta=parameters.priority_weight_beta,
        ),
    )


##################################################
## Legacy TorchRL code
##################################################
def _convert_exploration_type(*, exploration_mode, exploration_type):
    if exploration_mode is not None:
        return ExplorationType.from_str(exploration_mode)
    return exploration_type


##################################################
## Custom Classes
##################################################
class TransformedEnvCustom(TransformedEnv):
    """
    Slightly modify the function `rollout`, `_rollout_stop_early`, and `_rollout_nonstop` to enable returning a frame list to save evaluation video
    """

    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        callback: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = True,
        return_contiguous: bool = True,
        tensordict: Optional[TensorDictBase] = None,
        out=None,
        is_save_simulation_video: bool = False,
        priority_module=None,
        cbf_controllers=None,
        opponent_predictor=None,
        frame_sink=None,
        is_log_backup_value: bool = False,
    ):
        """Executes a rollout in the environment.

        The function will stop as soon as one of the contained environments
        returns done=True.

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using :obj:`env.rand_step()`
                default = None
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool, optional): if ``True``, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is ``True``.
            auto_cast_to_device (bool, optional): if ``True``, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is ``False``.
            break_when_any_done (bool): breaks if any of the done state is True. If False, a reset() is
                called on the sub-envs that are done. Default is True.
            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.
            frame_sink (optional): object with ``append(frame)`` that receives the
                video frames instead of a new list, e.g., a
                :class:`sigmarl.helper_common.VideoWriter` that streams them to
                disk. It is returned in place of the frame list.
            is_log_backup_value (bool): with the CBF filter, also store the joint
                backup value of every agent at every step in "backup_value"
                (see CBFQP.backup_values_all_envs; diagnostic, costs one batched
                braking rollout per step).

        Returns:
            TensorDict object containing the resulting trajectory. With the CBF
            filter, "cbf_dev_vel" and "cbf_dev_steer" [n_envs, T, n_agents] hold
            the filter's normalized changes of each agent's targets (see
            cbf_filter_deviation).

        The data returned will be marked with a "time" dimension name for the last
        dimension of the tensordict (at the ``env.ndim`` index).
        """

        try:
            policy_device = next(policy.parameters()).device
        except (StopIteration, AttributeError):
            policy_device = self.device

        env_device = self.device

        if auto_reset:
            if tensordict is not None:
                raise RuntimeError(
                    "tensordict cannot be provided when auto_reset is True"
                )
            tensordict = self.reset()
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")
        if policy is None:
            policy = self.rand_action

        kwargs = {
            "tensordict": tensordict,
            "auto_cast_to_device": auto_cast_to_device,
            "max_steps": max_steps,
            "policy": policy,
            "policy_device": policy_device,
            "env_device": env_device,
            "callback": callback,
            "is_save_simulation_video": is_save_simulation_video,
            "priority_module": priority_module,
            "cbf_controllers": cbf_controllers,
            "opponent_predictor": opponent_predictor,
            "frame_sink": frame_sink,
            "is_log_backup_value": is_log_backup_value,
        }
        self.time_rl = []
        self.time_cbf = []
        self.time_cbf_parallel = []
        # Parts of the parallelized CBF time (see cbf_constrained_policy_main)
        self.time_cbf_parts = []
        self.cbf_slack_active = []
        self.time_pseudo_dis = []
        self.cbf_dev_vel = []
        self.cbf_dev_steer = []
        self.backup_value = []
        if break_when_any_done:
            if is_save_simulation_video:
                tensordicts, frame_list = self._rollout_stop_early(
                    **kwargs
                )  # Modification
            else:
                tensordicts = self._rollout_stop_early(**kwargs)
        else:
            if is_save_simulation_video:
                tensordicts, frame_list = self._rollout_nonstop(
                    **kwargs
                )  # Modification
            else:
                tensordicts = self._rollout_nonstop(**kwargs)

        batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        out_td = torch.stack(tensordicts, len(batch_size), out=out)
        if return_contiguous:
            out_td = out_td.contiguous()
        out_td.refine_names(..., "time")
        self._set_rollout_timing_fields(out_td)

        if is_save_simulation_video:
            return out_td, frame_list  # Modification
        else:
            return out_td, None

    def _set_rollout_timing_fields(self, out_td: TensorDictBase):
        if len(out_td.batch_size) < 2:
            return

        n_envs = int(out_td.batch_size[0])
        n_steps = int(out_td.batch_size[-1])
        if n_steps <= 0:
            return

        device = out_td.device
        if device is None:
            try:
                device = out_td.get(("agents", "info", "pos")).device
            except Exception:
                device = torch.device("cpu")

        def _to_timing_tensor(values):
            values = list(values[:n_steps])
            if len(values) < n_steps:
                values.extend([float("nan")] * (n_steps - len(values)))

            timing = torch.tensor(values, dtype=torch.float32, device=device)
            return timing.unsqueeze(0).expand(n_envs, n_steps).clone()

        out_td.set("time_rl", _to_timing_tensor(self.time_rl))
        out_td.set("time_cbf", _to_timing_tensor(self.time_cbf))
        out_td.set("time_cbf_parallel", _to_timing_tensor(self.time_cbf_parallel))
        if self.time_cbf_parts:
            for part in self.time_cbf_parts[0]:
                out_td.set(
                    f"time_cbf_{part}_parallel",
                    _to_timing_tensor([parts[part] for parts in self.time_cbf_parts]),
                )
        # 1 if any CBF-QP of the step needed slack or failed
        out_td.set("cbf_slack_active", _to_timing_tensor(self.cbf_slack_active))
        out_td.set("time_pseudo_dis", _to_timing_tensor(self.time_pseudo_dis))
        # Per-agent fields [n_envs, T, n_agents] of the CBF filter (if it ran)
        for key, values in (
            ("cbf_dev_vel", self.cbf_dev_vel),
            ("cbf_dev_steer", self.cbf_dev_steer),
            ("backup_value", self.backup_value),
        ):
            if values:
                out_td.set(
                    key,
                    torch.as_tensor(
                        np.stack(values[:n_steps], axis=1),
                        dtype=torch.float32,
                        device=device,
                    ),
                )

    def _record_cbf_step(self, cbf_controllers, step_timing: dict):
        """Store the filter's deviation and, if logged, the backup values of a
        filtered step (see rollout)."""
        dev_vel, dev_steer = cbf_filter_deviation(cbf_controllers)
        self.cbf_dev_vel.append(dev_vel.detach().cpu().numpy())
        self.cbf_dev_steer.append(dev_steer.detach().cpu().numpy())
        if "backup_value" in step_timing:
            self.backup_value.append(step_timing["backup_value"])

    def _rollout_stop_early(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
        is_save_simulation_video,
        priority_module,
        cbf_controllers,
        opponent_predictor,
        frame_sink,
        is_log_backup_value,
    ):
        tensordicts = []

        if is_save_simulation_video:
            frame_list = frame_sink if frame_sink is not None else []

        for i in range(max_steps):
            if (i + 1) % 100 == 0:
                print(f"[INFO] k={i+1}/{max_steps+1}", end="\r", flush=True)
            if auto_cast_to_device:
                tensordict = tensordict.to(policy_device, non_blocking=True)

            time_cbf = 0.0
            time_pseudo_dis = 0.0
            step_timing = {"parallel": 0.0, "slack_active": False}
            if self.base_env.scenario_name.parameters.is_using_cbf_testing:
                (
                    time_rl,
                    time_cbf,
                    time_pseudo_dis,
                    tensordict,
                ) = cbf_constrained_policy_main(
                    tensordict,
                    policy,
                    cbf_controllers,
                    timing=step_timing,
                    is_log_backup_value=is_log_backup_value,
                )
                self._record_cbf_step(cbf_controllers, step_timing)
            else:
                time_rl_start = time.time()
                scores = _generate_behavior_action(
                    tensordict,
                    policy,
                    priority_module,
                    self,
                    opponent_predictor=opponent_predictor,
                )
                time_rl = time.time() - time_rl_start

            self.time_rl.append(time_rl)
            self.time_cbf.append(time_cbf)
            self.time_cbf_parallel.append(step_timing["parallel"])
            if "parts" in step_timing:
                self.time_cbf_parts.append(step_timing["parts"])
            self.cbf_slack_active.append(float(step_timing["slack_active"]))
            self.time_pseudo_dis.append(time_pseudo_dis)

            if auto_cast_to_device:
                tensordict = tensordict.to(env_device, non_blocking=True)
            tensordict = self.step(tensordict)
            if not self.base_env.scenario_name.parameters.is_using_cbf_testing:
                _redistribute_transition_reward(self, tensordict, scores)
            _notify_priority_episode_end(priority_module, tensordict)
            pre_reset_frame = (
                _consume_pre_reset_render_frame(self)
                if is_save_simulation_video
                else None
            )
            tensordicts.append(tensordict.clone(False))

            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break
            tensordict = step_mdp(
                tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=self.reward_keys,
                action_keys=self.action_keys,
                done_keys=self.done_keys,
            )
            # done and truncated are in done_keys
            # We read if any key is done.
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=self.output_spec["full_done_spec"],
                key=None,
            )
            if any_done:
                if (
                    callback is not None
                    and is_save_simulation_video
                    and pre_reset_frame is not None
                ):
                    frame_list.append(pre_reset_frame)
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = (
                        pre_reset_frame
                        if pre_reset_frame is not None
                        else callback(self, tensordict)
                    )
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)
        if is_save_simulation_video:
            return tensordicts, frame_list  # Modification
        else:
            return tensordicts

    def _rollout_nonstop(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
        is_save_simulation_video,
        priority_module,
        cbf_controllers,
        opponent_predictor,
        frame_sink,
        is_log_backup_value,
    ):
        tensordicts = []
        tensordict_ = tensordict

        if is_save_simulation_video:
            frame_list = frame_sink if frame_sink is not None else []

        for i in range(max_steps):
            # start_global_timer = time.perf_counter()
            if (i + 1) % 100 == 0:
                print(f"[INFO] k={i+1}/{max_steps+1}", end="\r", flush=True)
            if auto_cast_to_device:
                tensordict_ = tensordict_.to(policy_device, non_blocking=True)

            # Possibly predict the actions of surrounding agents using opponent modeling

            if self.base_env.scenario_name.parameters.is_using_cbf_training:
                raise NotImplementedError("Please set is_using_cbf_training to False.")

            time_cbf = 0.0
            time_pseudo_dis = 0.0
            step_timing = {"parallel": 0.0, "slack_active": False}
            if self.base_env.scenario_name.parameters.is_using_cbf_testing:
                (
                    time_rl,
                    time_cbf,
                    time_pseudo_dis,
                    tensordict_,
                ) = cbf_constrained_policy_main(
                    tensordict_,
                    policy,
                    cbf_controllers,
                    timing=step_timing,
                    is_log_backup_value=is_log_backup_value,
                )
                self._record_cbf_step(cbf_controllers, step_timing)
            else:
                time_rl_start = time.time()
                scores = _generate_behavior_action(
                    tensordict_,
                    policy,
                    priority_module,
                    self,
                    opponent_predictor=opponent_predictor,
                )
                time_rl = time.time() - time_rl_start

            # print(f"[DEBUG] Step {i+1}/{max_steps}: time_rl={time_rl:.4f} s, time_cbf={time_cbf:.4f} s")
            self.time_rl.append(time_rl)
            self.time_cbf.append(time_cbf)
            self.time_cbf_parallel.append(step_timing["parallel"])
            if "parts" in step_timing:
                self.time_cbf_parts.append(step_timing["parts"])
            self.cbf_slack_active.append(float(step_timing["slack_active"]))
            self.time_pseudo_dis.append(time_pseudo_dis)

            if auto_cast_to_device:
                tensordict_ = tensordict_.to(env_device, non_blocking=True)
            # time_before_step_and_maybe_reset = time.perf_counter()
            tensordict = self.step(tensordict_)
            if not self.base_env.scenario_name.parameters.is_using_cbf_testing:
                _redistribute_transition_reward(self, tensordict, scores)
            _notify_priority_episode_end(priority_module, tensordict)
            tensordict_ = self._step_mdp(tensordict)
            tensordict_ = self.maybe_reset(tensordict_)
            pre_reset_frame = (
                _consume_pre_reset_render_frame(self)
                if is_save_simulation_video
                else None
            )
            # time_after_step_and_maybe_reset = time.perf_counter()
            # print(f"[DEBUG] Step time {time_after_step_and_maybe_reset - time_before_step_and_maybe_reset} s, time before step {time_before_step_and_maybe_reset - start_global_timer}")
            tensordicts.append(tensordict)
            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = (
                        pre_reset_frame
                        if pre_reset_frame is not None
                        else callback(self, tensordict)
                    )
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)

        if is_save_simulation_video:
            return tensordicts, frame_list  # Modification
        else:
            return tensordicts


class SyncDataCollectorCustom(SyncDataCollector):
    """
    Slightly modify the function `rollout` to enable opponent modeling and prioritized action propagation.
    """

    def helper_init(
        self,
        create_env_fn: Union[
            EnvBase, Sequence[Callable[[], EnvBase]]  # noqa: F821
        ],  # noqa: F821
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ],
        *,
        frames_per_batch: int,
        total_frames: int,
        device: DEVICE_TYPING = None,
        storing_device: DEVICE_TYPING = None,
        env_device: DEVICE_TYPING = None,
        policy_device: DEVICE_TYPING = None,
        create_env_kwargs: Union[Dict, None] = None,  # Changed from dict | None
        max_frames_per_traj: Union[int, None] = None,  # Changed from int | None
        init_random_frames: Union[int, None] = None,  # Changed from int | None
        reset_at_each_iter: bool = False,
        postproc: Optional[
            Callable[[TensorDictBase], TensorDictBase]
        ] = None,  # Changed from Callable[...] | None
        split_trajs: Union[bool, None] = None,  # Changed from bool | None
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        exploration_mode=None,
        return_same_td: bool = False,
        reset_when_done: bool = True,
        interruptor=None,
    ):
        from torchrl.envs.batched_envs import _BatchedEnv

        self.closed = True

        exploration_type = _convert_exploration_type(
            exploration_mode=exploration_mode, exploration_type=exploration_type
        )
        if create_env_kwargs is None:
            create_env_kwargs = {}
        if not isinstance(create_env_fn, EnvBase):
            env = create_env_fn(**create_env_kwargs)
        else:
            env = create_env_fn
            if create_env_kwargs:
                if not isinstance(env, _BatchedEnv):
                    raise RuntimeError(
                        "kwargs were passed to SyncDataCollector but they can't be set "
                        f"on environment of type {type(create_env_fn)}."
                    )
                env.update_kwargs(create_env_kwargs)

        self.env: EnvBase = env
        self.closed = False
        if not reset_when_done:
            raise ValueError("reset_when_done is deprectated.")
        self.reset_when_done = reset_when_done
        self.n_env = self.env.batch_size.numel()

        (self.policy, self.get_weights_fn,) = self._get_policy_and_device(
            policy=policy,
            # observation_spec=self.env.observation_spec,
        )

        if isinstance(self.policy, nn.Module):
            self.policy_weights = TensorDict(dict(self.policy.named_parameters()), [])
            self.policy_weights.update(
                TensorDict(dict(self.policy.named_buffers()), [])
            )
        else:
            self.policy_weights = TensorDict({}, [])

        self.env: EnvBase = self.env.to(self.device)
        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )
        if self.max_frames_per_traj is not None and self.max_frames_per_traj > 0:
            # let's check that there is no StepCounter yet
            for key in self.env.output_spec.keys(True, True):
                if isinstance(key, str):
                    key = (key,)
                if "step_count" in key:
                    raise ValueError(
                        "A 'step_count' key is already present in the environment "
                        "and the 'max_frames_per_traj' argument may conflict with "
                        "a 'StepCounter' that has already been set. "
                        "Possible solutions: Set max_frames_per_traj to 0 or "
                        "remove the StepCounter limit from the environment transforms."
                    )
            env = self.env = TransformedEnv(
                self.env, StepCounter(max_steps=self.max_frames_per_traj)
            )

        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({frames_per_batch})."
                    f"This means {frames_per_batch - remainder} additional frames will be collected."
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )
        self.reset_at_each_iter = reset_at_each_iter
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames is not None else 0
        )
        if (
            init_random_frames is not None
            and init_random_frames % frames_per_batch != 0
            and RL_WARNINGS
        ):
            warnings.warn(
                f"init_random_frames ({init_random_frames}) is not exactly a multiple of frames_per_batch ({frames_per_batch}), "
                f" this results in more init_random_frames than requested"
                f" ({-(-init_random_frames // frames_per_batch) * frames_per_batch})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )

        self.postproc = postproc
        if self.postproc is not None and hasattr(self.postproc, "to"):
            self.postproc.to(self.storing_device)
        if frames_per_batch % self.n_env != 0 and RL_WARNINGS:
            warnings.warn(
                f"frames_per_batch ({frames_per_batch}) is not exactly divisible by the number of batched environments ({self.n_env}), "
                f" this results in more frames_per_batch per iteration that requested"
                f" ({-(-frames_per_batch // self.n_env) * self.n_env})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )
        self.requested_frames_per_batch = int(frames_per_batch)
        self.frames_per_batch = -(-frames_per_batch // self.n_env)
        self.exploration_type = (
            exploration_type if exploration_type else DEFAULT_EXPLORATION_TYPE
        )
        self.return_same_td = return_same_td

        self._tensordict = env.reset()
        traj_ids = torch.arange(self.n_env, device=env.device).view(self.env.batch_size)
        self._tensordict.set(
            ("collector", "traj_ids"),
            traj_ids,
        )

        with torch.no_grad():
            self._tensordict_out = self.env.fake_tensordict()
        # If the policy has a valid spec, we use it
        if (
            hasattr(self.policy, "spec")
            and self.policy.spec is not None
            and all(v is not None for v in self.policy.spec.values(True, True))
        ):
            if any(
                key not in self._tensordict_out.keys(isinstance(key, tuple))
                for key in self.policy.spec.keys(True, True)
            ):
                # if policy spec is non-empty, all the values are not None and the keys
                # match the out_keys we assume the user has given all relevant information
                # the policy could have more keys than the env:
                policy_spec = self.policy.spec
                if policy_spec.ndim < self._tensordict_out.ndim:
                    policy_spec = policy_spec.expand(self._tensordict_out.shape)
                for key, spec in policy_spec.items(True, True):
                    if key in self._tensordict_out.keys(isinstance(key, tuple)):
                        continue
                    self._tensordict_out.set(key, spec.zero())

        else:
            # otherwise, we perform a small number of steps with the policy to
            # determine the relevant keys with which to pre-populate _tensordict_out.
            # This is the safest thing to do if the spec has None fields or if there is
            # no spec at all.
            # See #505 for additional context.
            self._tensordict_out.update(self._tensordict)
            with torch.no_grad():
                self._tensordict_out = self.policy(self._tensordict_out.to(self.device))

        if self.env.base_env.scenario_name.parameters.is_using_prioritized_marl:
            # Create the TensorDict
            priority = TensorDict(
                {
                    "loc": torch.zeros(
                        self.env.base_env.scenario_name.parameters.num_vmas_envs,
                        self.env.base_env.scenario_name.parameters.n_agents,
                        1,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario_name.parameters.device,
                    ),
                    "sample_log_prob": torch.zeros(
                        self.env.base_env.scenario_name.parameters.num_vmas_envs,
                        self.env.base_env.scenario_name.parameters.n_agents,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario_name.parameters.device,
                    ),
                    "scale": torch.zeros(
                        self.env.base_env.scenario_name.parameters.num_vmas_envs,
                        self.env.base_env.scenario_name.parameters.n_agents,
                        1,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario_name.parameters.device,
                    ),
                    "scores": torch.zeros(
                        self.env.base_env.scenario_name.parameters.num_vmas_envs,
                        self.env.base_env.scenario_name.parameters.n_agents,
                        1,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario_name.parameters.device,
                    ),
                    "rank": torch.zeros(
                        self.env.base_env.scenario_name.parameters.num_vmas_envs,
                        self.env.base_env.scenario_name.parameters.n_agents,
                        dtype=torch.int64,
                        device=self.env.base_env.scenario_name.parameters.device,
                    ),
                    "state_value": torch.zeros(
                        self.env.base_env.scenario_name.parameters.num_vmas_envs,
                        self.env.base_env.scenario_name.parameters.n_agents,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario_name.parameters.device,
                    ),
                },
                batch_size=[self.env.base_env.scenario_name.parameters.num_vmas_envs],
            )
            self._tensordict_out["next", "agents", "info"].set("priority", priority)
            self._tensordict_out["agents", "info"].set("priority", priority)
            self._tensordict_out["next", "agents"].set("priority", priority.clone())
            self._tensordict_out["agents"].set("priority", priority.clone())

        self._tensordict_out = (
            self._tensordict_out.unsqueeze(-1)
            .expand(*env.batch_size, self.frames_per_batch)
            .clone()
            .zero_()
        )
        # in addition to outputs of the policy, we add traj_ids to
        # _tensordict_out which will be collected during rollout
        self._tensordict_out = self._tensordict_out.to(self.storing_device)
        self._tensordict_out.set(
            ("collector", "traj_ids"),
            torch.zeros(
                *self._tensordict_out.batch_size,
                dtype=torch.int64,
                device=self.storing_device,
            ),
        )

        self._tensordict_out.refine_names(..., "time")

        if split_trajs is None:
            split_trajs = False
        self.split_trajs = split_trajs
        self._exclude_private_keys = True
        self.interruptor = interruptor
        self._frames = 0
        self._iter = -1

    def __init__(
        self,
        env,
        policy,
        priority_module=None,
        cbf_controllers=None,
        opponent_predictor=None,
        **kwargs,
    ):
        self.priority_module = priority_module
        self.cbf_controllers = cbf_controllers
        self.opponent_predictor = opponent_predictor
        super().__init__(env, policy, **kwargs)
        self.base_env = self.env.base_env if hasattr(self.env, "base_env") else self.env
        self.parameters = self.base_env.scenario_name.parameters
        self.reward_key = env.reward_key
        self.helper_init(env, policy, **kwargs)

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        if self.reset_at_each_iter:
            self._tensordict.update(self.env.reset())

        # self._tensordict.fill_(("collector", "step_count"), 0)
        self._tensordict_out.fill_(("collector", "traj_ids"), -1)
        tensordicts = []
        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                scores = None

                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    self.env.rand_action(self._tensordict)
                else:
                    if self.env.base_env.scenario_name.parameters.is_using_cbf_training:
                        # print("[INFO] Using CBF-constrained policy during training.")
                        (
                            time_rl,
                            time_cbf,
                            time_pseudo_dis,
                            self._tensordict,
                        ) = cbf_constrained_policy_main(
                            self._tensordict,
                            self.policy,
                            self.cbf_controllers,
                        )
                    else:
                        scores = _generate_behavior_action(
                            self._tensordict,
                            self.policy,
                            self.priority_module,
                            self.env,
                            opponent_predictor=self.opponent_predictor,
                        )

                tensordict = self.env.step(self._tensordict)
                if not self.parameters.is_using_cbf_training:
                    _redistribute_transition_reward(self.env, tensordict, scores)
                _notify_priority_episode_end(self.priority_module, tensordict)
                tensordict_ = self.env._step_mdp(tensordict)
                tensordict_ = self.env.maybe_reset(tensordict_)

                # Shielded-PPO: the environment just executed the CBF (applied) action; overwrite the *stored* transition's action with the raw
                # RL action so PPO learns from the policy's own action (its log-prob in ("agents", "sample_log_prob") already matches). The reward /
                # next state in ("next", ...) remain the outcome of the applied CBF action, which is the intended shielding semantics.
                _params = self.env.base_env.scenario_name.parameters
                if (
                    getattr(_params, "is_using_cbf_training", False)
                    and getattr(_params, "is_store_rl_action_under_cbf", False)
                    and getattr(self.policy, "_rl_action_cache", None) is not None
                ):
                    tensordict.set(("agents", "action"), self.policy._rl_action_cache)

                self._tensordict = tensordict_.set(
                    "collector", tensordict.get("collector").clone(False)
                )
                tensordicts.append(
                    tensordict.to(self.storing_device, non_blocking=True)
                )

                self._update_traj_ids(tensordict)
                if (
                    self.interruptor is not None
                    and self.interruptor.collection_stopped()
                ):
                    try:
                        torch.stack(
                            tensordicts,
                            self._tensordict_out.ndim - 1,
                            out=self._tensordict_out[: t + 1],
                        )
                    except RuntimeError:
                        with self._tensordict_out.unlock_():
                            torch.stack(
                                tensordicts,
                                self._tensordict_out.ndim - 1,
                                out=self._tensordict_out[: t + 1],
                            )
                    break
            else:
                try:
                    self._tensordict_out = torch.stack(
                        tensordicts,
                        self._tensordict_out.ndim - 1,
                        out=self._tensordict_out,
                    )
                except RuntimeError:
                    with self._tensordict_out.unlock_():
                        self._tensordict_out = torch.stack(
                            tensordicts,
                            self._tensordict_out.ndim - 1,
                            out=self._tensordict_out,
                        )
        return self._tensordict_out


class WorldCustom(World):
    """
    The default step function update agents' states based on physical forces.
    Redefine step function to update agents' states based on speed and steering command.
    """

    def step(self):
        for entity in self.entities:
            hold_mask = getattr(entity.dynamics, "joint_task_hold_mask", None)
            target_action_override = None
            override_mask = getattr(
                entity.dynamics, "speed_target_action_override_mask", None
            )
            if override_mask is not None and override_mask.any():
                override = entity.dynamics.speed_target_action_override
                nan_override = torch.full_like(override, float("nan"))
                target_action_override = torch.where(
                    override_mask.unsqueeze(-1), override, nan_override
                )

            target_action, u = compute_applied_action_control(
                entity.action.u,
                entity.state.speed.squeeze(-1),
                entity.state.steering.squeeze(-1),
                self.dt,
                dynamics=entity.dynamics,
                pos=entity.state.pos,
                rot=entity.state.rot,
                path_tracking_ref_point=getattr(
                    entity.dynamics, "path_tracking_ref_point", None
                ),
                target_action_override=target_action_override,
            )
            if hold_mask is not None and hold_mask.any():
                target_action = target_action.clone()
                u = u.clone()
                target_action[hold_mask] = 0
                u[hold_mask] = 0
            if entity.action.u.shape[-1] < 2:
                entity.action.u = target_action.clone()
            else:
                entity.action.u[:, 0] = target_action[:, 0]
                entity.action.u[:, 1] = target_action[:, 1]

            if override_mask is not None:
                override_mask[:] = False

            entity.state.acceleration = u[:, 0].unsqueeze(-1)
            entity.state.steering_rate = u[:, 1].unsqueeze(-1)

            x0 = torch.cat(
                [
                    entity.state.pos,
                    entity.state.rot,
                    entity.state.speed,
                    entity.state.steering,
                ],
                dim=-1,
            )

            # Compute new states
            x1, sideslip_angle, velocity = entity.dynamics.step(
                x0=x0,  # [x, y, yaw, speed, steering]
                u=u,  #  [acceleration, steering_rate]
                dt=self.dt,  # Sample time in [s]
                tick_per_step=1,  # Ticks per time step controling the fedelity of the ODE
            )
            if hold_mask is not None and hold_mask.any():
                x1 = x1.clone()
                velocity = velocity.clone()
                x1[hold_mask, 0:3] = x0[hold_mask, 0:3]
                x1[hold_mask, 3] = 0
                x1[hold_mask, 4] = x0[hold_mask, 4]
                velocity[hold_mask] = 0
            # Update states
            entity.state.pos = x1[:, 0:2]
            entity.state.rot = x1[:, 2].unsqueeze(-1)
            entity.state.speed = x1[:, 3].unsqueeze(-1)
            entity.state.steering = x1[:, 4].unsqueeze(-1)
            entity.state.vel = velocity
            entity.state.sideslip_angle = sideslip_angle.unsqueeze(-1)


class SaveData:
    def __init__(
        self,
        parameters: Parameters,
        episode_reward_mean_list: list = None,
        parameter_independent_reward_mean_list: list = None,
        task_metrics_list: list = None,
    ):
        self.parameters = parameters
        self.episode_reward_mean_list = episode_reward_mean_list
        self.parameter_independent_reward_mean_list = (
            parameter_independent_reward_mean_list
        )
        # Per iteration: task metrics of the training rollouts that do not
        # depend on the training reward (MAPPOCAVs._compute_task_metrics)
        self.task_metrics_list = [] if task_metrics_list is None else task_metrics_list

    def to_dict(self):
        return {
            "parameters": self.parameters.to_dict(),  # Convert Parameters instance to dict
            "episode_reward_mean_list": self.episode_reward_mean_list,
            "parameter_independent_reward_mean_list": (
                self.parameter_independent_reward_mean_list
            ),
            "task_metrics_list": self.task_metrics_list,
        }

    @classmethod
    def from_dict(cls, dict_data):
        saved = dict_data["parameters"]
        if "is_backup_horizon_cbf" not in saved:
            # Saved before the backup-horizon rows existed: keep the two-step
            # condition the run used (the default is now True)
            saved = {**saved, "is_backup_horizon_cbf": False}
        if "cbf_lambda" not in saved:
            # Saved before cbf_lambda existed: keep the fixed lambda of that
            # time instead of the tuned defaults (tuned_cbf_lambda)
            saved = {**saved, "cbf_lambda": 0.5}
        parameters = Parameters.from_dict(saved)  # Convert dict to Parameters
        return cls(
            parameters,
            dict_data["episode_reward_mean_list"],
            dict_data.get("parameter_independent_reward_mean_list", []),
            dict_data.get("task_metrics_list", []),
        )


def _require_rollout_scalar_field(out_td, key):
    try:
        tensor = out_td.get(key)
    except Exception as exc:
        raise KeyError(f"Required rollout field {key} is missing.") from exc

    if tensor is None:
        raise KeyError(f"Required rollout field {key} is missing.")

    tensor = torch.as_tensor(tensor).detach().float()
    while tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected rollout field {key} to have shape [num_envs, num_steps, "
            f"num_agents] or [num_envs, num_steps, num_agents, 1], got "
            f"{tuple(tensor.shape)}."
        )

    return tensor


def _get_parameter_independent_reward_rollout_fields(out_td):
    reward = _require_rollout_scalar_field(
        out_td, ("agents", "info", "rew_parameter_independent")
    )
    step_time = _require_rollout_scalar_field(out_td, ("agents", "info", "step_time"))

    if reward.shape != step_time.shape:
        raise ValueError(
            "Parameter-independent reward and step_time rollout fields must have "
            f"the same shape, got {tuple(reward.shape)} and {tuple(step_time.shape)}."
        )

    return reward, step_time


def _find_rollout_scalar_field(out_td, key):
    """Return a rollout scalar field when present, in ``[env, time, agent]`` form."""
    try:
        return _require_rollout_scalar_field(out_td, key)
    except KeyError:
        return None


def _find_rollout_velocity_magnitude(out_td):
    """Return the planar velocity magnitude when a rollout stores ``vel``."""
    try:
        velocity = out_td.get(("agents", "info", "vel"))
    except Exception:
        return None

    if velocity is None:
        return None

    velocity = torch.as_tensor(velocity).detach().float()
    if velocity.ndim != 4 or velocity.shape[-1] < 2:
        raise ValueError(
            "Expected rollout field ('agents', 'info', 'vel') to have shape "
            f"[num_envs, num_steps, num_agents, 2+], got {tuple(velocity.shape)}."
        )
    return torch.linalg.vector_norm(velocity[..., :2], dim=-1)


def _finite_float(value):
    """Convert a scalar to a JSON-safe float, using ``None`` for non-finite data."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _finite_float_or_zero(value):
    """Convert a scalar to a finite float, defaulting to 0.0 if None or non-finite."""
    if value is None:
        return 0.0
    try:
        val = float(value)
        return val if math.isfinite(val) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _summarize_numeric_rollout(values):
    """Summarize a ``[env, time, agent]`` tensor without retaining its samples."""
    empty_summary = {
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "finite_sample_count": None,
        "per_agent_mean": None,
    }
    if values is None:
        return empty_summary

    values = torch.as_tensor(values).detach().float()
    if values.ndim != 3:
        raise ValueError(
            "Expected numeric rollout statistics to have shape "
            f"[num_envs, num_steps, num_agents], got {tuple(values.shape)}."
        )

    finite_mask = torch.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.numel() == 0:
        return empty_summary

    finite_counts = finite_mask.sum(dim=(0, 1))
    finite_sums = torch.where(finite_mask, values, torch.zeros_like(values)).sum(
        dim=(0, 1)
    )
    per_agent_mean = [
        _finite_float(finite_sums[agent_index] / finite_counts[agent_index])
        if finite_counts[agent_index] > 0
        else None
        for agent_index in range(values.shape[-1])
    ]

    return {
        "mean": _finite_float(finite_values.mean()),
        "std": _finite_float(finite_values.std(unbiased=False)),
        "min": _finite_float(finite_values.min()),
        "max": _finite_float(finite_values.max()),
        "finite_sample_count": int(finite_values.numel()),
        "per_agent_mean": per_agent_mean,
    }


def _summarize_boolean_rollout(flags):
    """Summarize environment-step events and agent-step incidences."""
    empty_summary = {
        "event_count": None,
        "agent_step_count": None,
        "environments_affected": None,
        "agents_affected": None,
        "per_agent_event_count": None,
    }
    if flags is None:
        return empty_summary

    flags = torch.as_tensor(flags).detach().bool()
    if flags.ndim != 3:
        raise ValueError(
            "Expected boolean rollout statistics to have shape "
            f"[num_envs, num_steps, num_agents], got {tuple(flags.shape)}."
        )

    event_mask = flags.any(dim=-1)
    per_agent_event_count = flags.sum(dim=(0, 1))
    return {
        "event_count": int(event_mask.sum().item()),
        "agent_step_count": int(flags.sum().item()),
        "environments_affected": int(event_mask.any(dim=1).sum().item()),
        "agents_affected": int(flags.any(dim=1).sum().item()),
        "per_agent_event_count": [
            int(count) for count in per_agent_event_count.tolist()
        ],
    }


def _collision_rates_from_summary(
    collision_summary, num_envs: int, num_steps: int, num_agents: int
):
    """Compute the requested step rate and its agent-normalized counterpart."""
    empty_rates = {
        "collision_rate_steps": None,
        "collision_rate_agents": None,
    }
    if collision_summary is None:
        return empty_rates

    event_count = collision_summary.get("event_count")
    total_time_steps = num_envs * num_steps
    total_agent_time_steps = total_time_steps * num_agents
    if event_count is None or total_time_steps <= 0 or total_agent_time_steps <= 0:
        return empty_rates

    return {
        "collision_rate_steps": _finite_float(event_count / total_time_steps),
        "collision_rate_agents": _finite_float(event_count / total_agent_time_steps),
    }


def _summarize_traffic_throughput(goal_flags, step_time):
    """Compute successful vehicle exits per aggregate simulated second."""
    empty_summary = {
        "vehicles_per_second": None,
        "successful_vehicle_count": None,
        "total_simulated_duration_s": None,
        "unit": "vehicles/s",
    }
    if goal_flags is None or step_time is None:
        return empty_summary

    if goal_flags.shape != step_time.shape:
        raise ValueError(
            "Goal-reaching and step_time rollout fields must have the same shape "
            f"for traffic throughput, got {tuple(goal_flags.shape)} and "
            f"{tuple(step_time.shape)}."
        )

    # step_time is repeated along the agent axis. Average that axis first so
    # each environment-time step contributes once to the denominator, then
    # sum all parallel environments' simulated durations.
    total_simulated_duration = step_time.mean(dim=-1).sum()
    if (
        not torch.isfinite(total_simulated_duration).item()
        or total_simulated_duration.item() <= 0
    ):
        raise ValueError(
            "Traffic throughput requires a positive finite simulated duration."
        )

    successful_vehicle_count = int(goal_flags.bool().sum().item())
    total_simulated_duration_s = _finite_float(total_simulated_duration)
    return {
        "vehicles_per_second": _finite_float(
            successful_vehicle_count / total_simulated_duration_s
        ),
        "successful_vehicle_count": successful_vehicle_count,
        "total_simulated_duration_s": total_simulated_duration_s,
        "unit": "vehicles/s",
    }


def _combine_boolean_rollouts(first, second):
    """Combine boolean rollout fields while allowing either field to be absent."""
    if first is None:
        return second
    if second is None:
        return first
    return first.bool() | second.bool()


def _build_rollout_statistics(out_td):
    """Build compact speed, safety, completion, and timing statistics."""
    speed = _find_rollout_scalar_field(out_td, ("agents", "info", "speed"))
    speed_source = "speed"
    if speed is None:
        speed = _find_rollout_velocity_magnitude(out_td)
        speed_source = "velocity_norm" if speed is not None else None
    speed_statistics = _summarize_numeric_rollout(
        speed.abs() if speed is not None else None
    )
    speed_statistics["source"] = speed_source

    agent_agent_collisions = _find_rollout_scalar_field(
        out_td, ("agents", "info", "is_collision_with_agents")
    )
    agent_lane_collisions = _find_rollout_scalar_field(
        out_td, ("agents", "info", "is_collision_with_lanelets")
    )
    total_collisions = _combine_boolean_rollouts(
        agent_agent_collisions, agent_lane_collisions
    )

    goal_flags = _find_rollout_scalar_field(out_td, ("agents", "info", "is_reach_goal"))
    step_time = _find_rollout_scalar_field(out_td, ("agents", "info", "step_time"))
    timing_statistics = _summarize_numeric_rollout(step_time)
    if step_time is None:
        mean_simulated_duration = None
    else:
        mean_simulated_duration = _finite_float(
            step_time.mean(dim=-1).sum(dim=1).mean()
        )

    return {
        "speed_mps": speed_statistics,
        "collisions": {
            "total": _summarize_boolean_rollout(total_collisions),
            "agent_agent": _summarize_boolean_rollout(agent_agent_collisions),
            "agent_lane": _summarize_boolean_rollout(agent_lane_collisions),
        },
        "goal_reaching": _summarize_boolean_rollout(goal_flags),
        "traffic_throughput": _summarize_traffic_throughput(goal_flags, step_time),
        "deadlocks": _summarize_boolean_rollout(
            _find_rollout_scalar_field(out_td, ("agents", "info", "is_deadlocked"))
        ),
        "timing": {
            "mean_step_time_s": timing_statistics["mean"],
            "mean_simulated_duration_s": mean_simulated_duration,
        },
    }


def compute_parameter_independent_reward_per_second(
    out_td, collision_penalty: Optional[float] = None
):
    """Compute unclamped parameter-independent reward per agent per second.

    When ``collision_penalty`` is provided, replace the original ``-1.0``
    agent-agent and agent-road collision contributions using the saved
    collision flags. The rollout itself is not modified.
    """
    reward, step_time = _get_parameter_independent_reward_rollout_fields(out_td)

    if collision_penalty is not None:
        collision_penalty = float(collision_penalty)
        if not math.isfinite(collision_penalty):
            raise ValueError("collision_penalty must be finite.")

        collision_with_agents = _require_rollout_scalar_field(
            out_td, ("agents", "info", "is_collision_with_agents")
        )
        collision_with_lanelets = _require_rollout_scalar_field(
            out_td, ("agents", "info", "is_collision_with_lanelets")
        )
        if (
            collision_with_agents.shape != reward.shape
            or collision_with_lanelets.shape != reward.shape
        ):
            raise ValueError(
                "Collision flags and parameter-independent reward rollout fields "
                "must have the same shape."
            )

        original_collision_penalty = -1.0
        collision_count = collision_with_agents.bool().to(reward.dtype) + (
            collision_with_lanelets.bool().to(reward.dtype)
        )
        reward = (
            reward + (collision_penalty - original_collision_penalty) * collision_count
        )

    duration = step_time.sum(dim=1)
    if not torch.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Rollout step_time must sum to positive finite durations.")

    reward_per_agent_per_second = reward.sum(dim=1) / duration
    if not torch.isfinite(reward_per_agent_per_second).all():
        raise ValueError("Parameter-independent reward contains non-finite values.")

    return reward_per_agent_per_second.mean().item()


def compute_parameter_independent_reward_per_minute(
    out_td, collision_penalty: Optional[float] = None
):
    """Compute unclamped parameter-independent reward per agent per minute."""
    return 60.0 * compute_parameter_independent_reward_per_second(
        out_td, collision_penalty=collision_penalty
    )


def _task_performance_from_rollout(out_td):
    """Recover cumulative task counters when they are present in a rollout."""
    num_task_tries = _find_rollout_scalar_field(
        out_td, ("agents", "info", "num_task_tries")
    )
    task_success_times = _find_rollout_scalar_field(
        out_td, ("agents", "info", "task_success_times")
    )
    if num_task_tries is None or task_success_times is None:
        return None

    # The scenario counter is repeated for every agent in ``info``. Use the
    # largest value across the rollout once per environment, then aggregate envs.
    num_task_tries = int(num_task_tries.amax(dim=(1, 2)).sum().item())
    task_success_times = int(task_success_times.amax(dim=(1, 2)).sum().item())
    return {
        "num_task_tries": num_task_tries,
        "task_success_times": task_success_times,
        "task_success_rate": (
            task_success_times / num_task_tries if num_task_tries else 0.0
        ),
    }


def _normalize_task_performance(task_performance):
    """Return JSON-safe task counters with a consistently defined rate."""
    normalized = dict(task_performance)
    normalized["num_task_tries"] = int(normalized["num_task_tries"])
    normalized["task_success_times"] = int(normalized["task_success_times"])
    normalized["task_success_rate"] = (
        normalized["task_success_times"] / normalized["num_task_tries"]
        if normalized["num_task_tries"]
        else 0.0
    )
    return normalized


def _normalize_computation_efficiency(
    computation_efficiency: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return JSON-safe computation efficiency statistics with zero defaults."""
    raw = computation_efficiency or {}
    return {
        "simulation_time": _finite_float_or_zero(raw.get("simulation_time")),
        "simulation_speed": _finite_float_or_zero(raw.get("simulation_speed")),
        "real_time_factor": _finite_float_or_zero(raw.get("real_time_factor")),
    }


def save_evaluation_summary(
    out_td,
    output_dir,
    name_suffix,
    random_seed: int | None = None,
    task_performance: Optional[Dict[str, Any]] = None,
    collision_penalty: Optional[float] = None,
    computation_efficiency: Optional[Dict[str, Any]] = None,
):
    """Print and save reward plus compact aggregate rollout statistics."""
    reward, _ = _get_parameter_independent_reward_rollout_fields(out_td)
    reward_value = compute_parameter_independent_reward_per_minute(
        out_td, collision_penalty=collision_penalty
    )
    statistics = _build_rollout_statistics(out_td)
    collision_rates = _collision_rates_from_summary(
        statistics["collisions"]["total"],
        num_envs=reward.shape[0],
        num_steps=reward.shape[1],
        num_agents=reward.shape[2],
    )
    normalized_suffix = name_suffix.strip("_")
    if random_seed is None:
        seed_match = re.search(r"(?:^|_)seed_?(-?\d+)(?:_|$)", normalized_suffix)
        random_seed = int(seed_match.group(1)) if seed_match else None

    summary = {
        "random_seed": random_seed,
        "parameter_independent_reward_per_agent_per_minute": reward_value,
        "unit": "reward/(agent*min)",
        **collision_rates,
        "num_envs": reward.shape[0],
        "num_steps": reward.shape[1],
        "num_agents": reward.shape[2],
        "computation_efficiency": _normalize_computation_efficiency(
            computation_efficiency
        ),
        "statistics_schema_version": 1,
        "statistics": statistics,
    }
    if task_performance is None:
        task_performance = _task_performance_from_rollout(out_td)
    if task_performance is not None:
        summary["task_performance"] = _normalize_task_performance(task_performance)

    os.makedirs(output_dir, exist_ok=True)
    summary_filename = (
        f"evaluation_summary_{normalized_suffix}.json"
        if normalized_suffix
        else "evaluation_summary.json"
    )
    path_summary = os.path.join(output_dir, summary_filename)
    with open(path_summary, "w") as outfile:
        json.dump(summary, outfile, indent=4)

    print("Parameter-independent reward = " f"{reward_value:.4f} reward/(agent*min)")
    print(f"Evaluation summary saved to {path_summary}")
    return summary


##################################################
## Helper Functions
##################################################
# TODO: remove
# def get_path_to_save_model(parameters: Parameters, absolute=True):
#     parameters.model_name = get_model_name(parameters=parameters)

#     uses_prioritized_marl = (
#         parameters.is_using_prioritized_marl
#         and parameters.prioritization_method.lower() == "marl"
#     )
#     file_ext = "pth"

#     where_to_save = parameters.where_to_save if absolute else ""

#     PATH_POLICY = os.path.join(
#         where_to_save, f"{parameters.model_name}_policy.{file_ext}"
#     )
#     PATH_CRITIC = os.path.join(
#         where_to_save, f"{parameters.model_name}_critic.{file_ext}"
#     )
#     PATH_FIG = os.path.join(
#         where_to_save, parameters.model_name + "_training_process.jpg"
#     )
#     PATH_JSON = os.path.join(where_to_save, parameters.model_name + "_data.json")
#     PATH_PRIORITY_POLICY = (
#         os.path.join(
#             where_to_save, f"{parameters.model_name}_priority_policy.{file_ext}"
#         )
#         if uses_prioritized_marl
#         else None
#     )
#     PATH_PRIORITY_CRITIC = (
#         os.path.join(
#             where_to_save, f"{parameters.model_name}_priority_critic.{file_ext}"
#         )
#         if uses_prioritized_marl
#         else None
#     )

#     return (
#         PATH_POLICY,
#         PATH_CRITIC,
#         PATH_PRIORITY_POLICY,
#         PATH_PRIORITY_CRITIC,
#         PATH_FIG,
#         PATH_JSON,
#     )


def get_path_to_save_model(parameters: Parameters, absolute=True):
    parameters.model_name = get_model_name(parameters=parameters)

    uses_prioritized_marl = (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    )
    file_ext = "pth"

    where_to_save = parameters.where_to_save if absolute else ""

    PATH_POLICY = os.path.join(
        where_to_save, f"{parameters.model_name}_policy.{file_ext}"
    )
    PATH_CRITIC = os.path.join(
        where_to_save, f"{parameters.model_name}_critic.{file_ext}"
    )
    PATH_MOTION_POLICY = os.path.join(
        where_to_save, f"{parameters.model_name}_motion_policy.{file_ext}"
    )
    PATH_LAMBDA_POLICY = os.path.join(
        where_to_save, f"{parameters.model_name}_lambda_policy.{file_ext}"
    )
    PATH_FIG = os.path.join(
        where_to_save, parameters.model_name + "_training_process.jpg"
    )
    PATH_JSON = os.path.join(where_to_save, parameters.model_name + "_data.json")
    PATH_PRIORITY_POLICY = (
        os.path.join(
            where_to_save, f"{parameters.model_name}_priority_policy.{file_ext}"
        )
        if uses_prioritized_marl
        else None
    )
    PATH_PRIORITY_CRITIC = (
        os.path.join(
            where_to_save, f"{parameters.model_name}_priority_critic.{file_ext}"
        )
        if uses_prioritized_marl
        else None
    )

    return (
        PATH_POLICY,
        PATH_CRITIC,
        PATH_MOTION_POLICY,
        PATH_LAMBDA_POLICY,
        PATH_PRIORITY_POLICY,
        PATH_PRIORITY_CRITIC,
        PATH_FIG,
        PATH_JSON,
    )


# TODO: remove
# def delete_files_with_lower_mean_reward(parameters: Parameters):
#     # Regular expression pattern to match and capture the float number
#     pattern = r"reward(-?[0-9]*\.?[0-9]+)_"

#     # Iterate over files in the directory
#     for file_name in os.listdir(parameters.where_to_save):
#         match = re.search(pattern, file_name)
#         if match:
#             # Get the achieved mean episode reward of the saved model
#             episode_reward_mean = float(match.group(1))
#             if episode_reward_mean < parameters.episode_reward_intermediate:
#                 # Delete the saved model if its performance is worse
#                 os.remove(os.path.join(parameters.where_to_save, file_name))


def delete_files_with_lower_mean_reward(parameters: Parameters):
    pattern = r"reward(-?[0-9]*\.?[0-9]+)_"

    for file_name in os.listdir(parameters.where_to_save):
        file_path = os.path.join(parameters.where_to_save, file_name)

        if not os.path.isfile(file_path):
            continue

        match = re.search(pattern, file_name)
        if match:
            episode_reward_mean = float(match.group(1))
            if episode_reward_mean < parameters.episode_reward_intermediate:
                os.remove(file_path)


def find_the_highest_reward_among_all_models(path):
    """This function returns the highest reward of the models stored in folder `parameters.where_to_save`"""
    # Initialize variables to track the highest reward and corresponding model
    highest_reward = float("-inf")

    pattern = r"reward(-?[0-9]*\.?[0-9]+)_"
    # Iterate through the files in the directory
    for filename in os.listdir(path):
        match = re.search(pattern, filename)
        if match:
            # Extract the reward and convert it to float
            episode_reward_mean = float(match.group(1))

            # Check if this reward is higher than the current highest
            if episode_reward_mean > highest_reward:
                highest_reward = episode_reward_mean  # Update

    return highest_reward


# TODO: remove
# def save(
#     parameters: Parameters,
#     save_data: SaveData,
#     decision_making_module,
#     optimization_module,
#     priority_module,
# ):
#     # Get paths
#     (
#         PATH_POLICY,
#         PATH_CRITIC,
#         PATH_PRIORITY_POLICY,
#         PATH_PRIORITY_CRITIC,
#         PATH_FIG,
#         PATH_JSON,
#     ) = get_path_to_save_model(parameters=parameters)

#     # Save parameters and mean episode reward list
#     json_object = json.dumps(save_data.to_dict(), indent=4)  # Serializing json
#     with open(PATH_JSON, "w") as outfile:  # Writing to sample.json
#         outfile.write(json_object)
#     # Example to how to open the saved json file
#     # with open('save_data.json', 'r') as file:
#     #     data = json.load(file)
#     #     loaded_parameters = Parameters.from_dict(data)

#     # Save figure
#     plt.clf()  # Clear the current figure to avoid drawing on the same figure in the next iteration
#     plt.plot(save_data.episode_reward_mean_list)
#     plt.xlabel("Iterations")
#     plt.ylabel("Episode mean reward")
#     # plt.tight_layout()
#     plt.savefig(PATH_FIG, format="jpg", dpi=300, bbox_inches="tight")

#     # Save models
#     if (
#         decision_making_module
#         and optimization_module
#         and decision_making_module.policy
#         and optimization_module.critic
#     ):
#         # Save current models
#         torch.save(decision_making_module.policy.state_dict(), PATH_POLICY)
#         torch.save(optimization_module.critic.state_dict(), PATH_CRITIC)

#         if (
#             parameters.is_using_prioritized_marl
#             and parameters.prioritization_method.lower() == "marl"
#             and priority_module
#             and priority_module.policy
#             and priority_module.critic
#         ):
#             # Save current models
#             torch.save(priority_module.policy.state_dict(), PATH_PRIORITY_POLICY)
#             torch.save(priority_module.critic.state_dict(), PATH_PRIORITY_CRITIC)

#         # Delete files with lower mean episode reward
#         delete_files_with_lower_mean_reward(parameters=parameters)

#     print(f"Saved model: {parameters.episode_reward_mean_current:.2f}.")


def save(
    parameters: Parameters,
    save_data: SaveData,
    decision_making_module,
    optimization_module,
    priority_module,
    emwa_alpha=0.2,
):
    # Get paths
    (
        PATH_POLICY,
        PATH_CRITIC,
        PATH_MOTION_POLICY,
        PATH_LAMBDA_POLICY,
        PATH_PRIORITY_POLICY,
        PATH_PRIORITY_CRITIC,
        PATH_FIG,
        PATH_JSON,
    ) = get_path_to_save_model(parameters=parameters)

    # Save parameters and mean episode reward list
    json_object = json.dumps(save_data.to_dict(), indent=4)
    with open(PATH_JSON, "w") as outfile:
        outfile.write(json_object)

    # Save figure
    fig, axes = plt.subplots(2, 1, figsize=(6, 12), sharex=True)
    axes[0].plot(save_data.episode_reward_mean_list, label="Episode mean reward")
    if save_data.episode_reward_mean_list:
        # Exponentially weighted moving average of the reward curve
        ewma_alpha = emwa_alpha
        ewma_list = []
        ewma_prev = save_data.episode_reward_mean_list[0]
        for reward in save_data.episode_reward_mean_list:
            ewma_prev = ewma_alpha * reward + (1 - ewma_alpha) * ewma_prev
            ewma_list.append(ewma_prev)
        axes[0].plot(
            ewma_list,
            color="red",
            label=f"EWMA (alpha={ewma_alpha})",
        )
        axes[0].legend()
    axes[0].set_ylabel("Episode mean reward")
    axes[1].plot(save_data.parameter_independent_reward_mean_list or [])
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Parameter-independent reward")
    fig.tight_layout()
    fig.savefig(PATH_FIG, format="jpg", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save models
    if (
        decision_making_module
        and optimization_module
        and decision_making_module.policy
        and optimization_module.critic
    ):
        # Save full actor and critic
        torch.save(decision_making_module.policy.state_dict(), PATH_POLICY)
        torch.save(optimization_module.critic.state_dict(), PATH_CRITIC)

        if hasattr(decision_making_module.policy, "motion_policy_state_dict"):
            torch.save(
                decision_making_module.policy.motion_policy_state_dict(),
                PATH_MOTION_POLICY,
            )
        if hasattr(decision_making_module.policy, "lambda_policy_state_dict"):
            torch.save(
                decision_making_module.policy.lambda_policy_state_dict(),
                PATH_LAMBDA_POLICY,
            )

        if (
            parameters.is_using_prioritized_marl
            and parameters.prioritization_method.lower() == "marl"
            and priority_module
            and priority_module.policy
            and priority_module.critic
        ):
            torch.save(priority_module.policy.state_dict(), PATH_PRIORITY_POLICY)
            torch.save(priority_module.critic.state_dict(), PATH_PRIORITY_CRITIC)

        delete_files_with_lower_mean_reward(parameters=parameters)

    parameter_independent_reward = parameters.parameter_independent_reward_intermediate
    if parameter_independent_reward is None:
        print(f"Saved model: {parameters.episode_reward_mean_current:.2f}.")
    else:
        print(
            f"Saved model: {parameters.episode_reward_mean_current:.2f}; "
            "corresp. param.-indep. rew.="
            f"{parameter_independent_reward:.2f}."
        )  # parameter-independent reward that corresponds to the best saving intermediate model


def compute_td_error(tensordict_data: TensorDict, gamma=0.9):
    """
    Computes TD error.

    Args:
        gamma: discount factor
    """
    current_state_values = tensordict_data["agents"]["state_value"]
    next_rewards = tensordict_data.get(("next", "agents", "reward"))
    next_state_values = tensordict_data.get(("next", "agents", "state_value"))
    done = tensordict_data.get(("next", "agents", "done"))

    # To mask out terminal states since TD error for terminal states should only consider the immediate reward without any future value
    not_done = ~done

    # See Eq. (2) of Section B EXPERIMENTAL DETAILS of paper https://doi.org/10.48550/arXiv.1511.05952
    td_error = (
        next_rewards + gamma * next_state_values * not_done - current_state_values
    )

    td_error = td_error.abs()  # Magnitude is more interesting than the actual TD error

    td_error_average_over_agents = td_error.mean(dim=-2)  # Cooperative agents

    # Normalize TD error to [0, 1] (priorities must be positive)
    td_min = td_error_average_over_agents.min()
    td_max = td_error_average_over_agents.max()
    td_error_range = td_max - td_min
    td_error_range = max(td_error_range, 1e-3)  # For numerical stability

    target_range = 10

    td_error_average_over_agents = (
        (td_error_average_over_agents - td_min) / td_error_range * target_range
    )
    td_error_average_over_agents = torch.clamp(
        td_error_average_over_agents, 1e-3, target_range
    )  # For numerical stability

    return td_error_average_over_agents


def _opponent_modeling_observation_key(policy):
    observation_with_actions_key = (
        "agents",
        "info",
        "observation_with_actions",
    )
    if observation_with_actions_key in getattr(policy, "in_keys", ()):
        return observation_with_actions_key
    return ("agents", "observation")


def _infer_opponent_modeling_action_dim(tensordict, policy):
    action = tensordict.get(("agents", "action"), None)
    if action is not None:
        return int(action.shape[-1])

    action_low = getattr(policy, "action_low", None)
    if action_low is not None:
        # Motion-policy bounds are stored per agent as [n_agents, n_actions].
        # Opponent modeling needs the action width for one agent only.
        return int(action_low.shape[-1])

    raise ValueError(
        "Opponent modeling could not infer the action dimension. The policy must "
        "expose an action tensor or an action_low bound."
    )


def _zero_opponent_modeling_action_slots(
    observation, n_nearing_agents_observed, n_actions
):
    action_block_width = int(n_nearing_agents_observed) * int(n_actions)
    if action_block_width == 0:
        return observation.clone()
    if observation.shape[-1] < action_block_width:
        raise ValueError(
            "Opponent-modeling observation is shorter than its configured action "
            "placeholder block."
        )

    observation_without_actions = observation.clone()
    observation_without_actions[..., -action_block_width:] = 0
    return observation_without_actions


def opponent_modeling(
    tensordict,
    policy,
    n_nearing_agents_observed,
    nearing_agents_indices,
    noise_percentage: float = 0,
    neighbor_valid=None,
    opponent_predictor=None,
):
    """
    Insert predicted neighboring actions into the behavior observation.

    By default, this retains the baseline self-modeling pass: ``policy`` first
    predicts tentative actions and is called again by the caller for the final
    behavior action. If ``opponent_predictor`` is supplied, the predictor sees
    zeroed action slots and the behavior policy is called only after its
    predictions have been inserted. Invalid or padded neighbor slots always
    receive zero actions.

    Reference
        [1] Raileanu, Roberta, et al. "Modeling others using oneself in multi-agent
        reinforcement learning." International conference on machine learning.
        PMLR, 2018.
    """
    observation_key = _opponent_modeling_observation_key(policy)

    if opponent_predictor is None:
        policy(tensordict)  # Baseline self-modeling prediction pass.
        prediction_tensordict = tensordict
    else:
        n_actions_hint = _infer_opponent_modeling_action_dim(
            tensordict, opponent_predictor
        )
        prediction_tensordict = tensordict.clone(False)
        prediction_observation = prediction_tensordict.get(observation_key)
        prediction_tensordict.set(
            observation_key,
            _zero_opponent_modeling_action_slots(
                prediction_observation,
                n_nearing_agents_observed,
                n_actions_hint,
            ),
        )
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            opponent_predictor(prediction_tensordict)

    actions_tentative = prediction_tensordict.get(("agents", "action"))
    if actions_tentative.ndim != 3:
        raise ValueError(
            "Opponent-modeling actions must have shape [batch, agents, actions]."
        )
    actions_tentative = actions_tentative.detach().clone()
    batch_dim, n_agents, n_actions = actions_tentative.shape
    device = actions_tentative.device

    observation = tensordict.get(observation_key)
    if observation.ndim != 3 or observation.shape[0] != batch_dim:
        raise ValueError(
            "Opponent-modeling observations must have shape [batch, agents, features]."
        )
    if observation.shape[1] != n_agents:
        raise ValueError(
            "Opponent-modeling observation and action agent dimensions differ."
        )

    if noise_percentage != 0:
        # Model prediction inaccuracy as a percentage of the maximum action value.
        noise_stds = torch.zeros(
            n_actions, device=device, dtype=actions_tentative.dtype
        )
        n_known_action_dims = min(n_actions, 2)
        noise_stds[:n_known_action_dims] = actions_tentative.new_tensor(
            [AGENTS["max_speed"], AGENTS["max_steering"]][:n_known_action_dims]
        )
        noise_stds = noise_stds * noise_percentage
        actions_tentative += torch.randn_like(actions_tentative) * noise_stds

    indices = torch.as_tensor(nearing_agents_indices, device=device, dtype=torch.long)
    if indices.ndim != 3 or indices.shape[:2] != (batch_dim, n_agents):
        raise ValueError(
            "Neighbor indices must have shape [batch, agents, neighbor_slots]."
        )

    n_neighbor_slots = int(n_nearing_agents_observed)
    n_available = min(indices.shape[-1], n_neighbor_slots)
    if neighbor_valid is None:
        valid = torch.ones(
            (batch_dim, n_agents, n_available), device=device, dtype=torch.bool
        )
    else:
        valid = torch.as_tensor(neighbor_valid, device=device, dtype=torch.bool)
        if valid.ndim != 3 or valid.shape[:2] != (batch_dim, n_agents):
            raise ValueError(
                "Neighbor validity must have shape [batch, agents, neighbor_slots]."
            )
        n_available = min(n_available, valid.shape[-1])
        valid = valid[..., :n_available]

    indices = indices[..., :n_available]
    if n_available > 0 and (indices.min() < 0 or indices.max() >= n_agents):
        raise ValueError("Neighbor indices are outside the agent dimension.")

    action_block_width = n_neighbor_slots * n_actions
    if observation.shape[-1] < action_block_width:
        raise ValueError(
            "Opponent-modeling observation is shorter than its configured action "
            "placeholder block."
        )
    if action_block_width == 0:
        return

    # Clear all slots first so stale predictions cannot survive a changed mask.
    observation[..., -action_block_width:] = 0
    batch_indices = torch.arange(batch_dim, device=device, dtype=torch.long)
    for ego_agent in range(n_agents):
        for neighbor_slot in range(n_available):
            surrounding_agent = indices[:, ego_agent, neighbor_slot]
            surrounding_action = actions_tentative[batch_indices, surrounding_agent]
            surrounding_action = surrounding_action.masked_fill(
                ~valid[:, ego_agent, neighbor_slot].unsqueeze(-1), 0.0
            )

            action_start = (
                observation.shape[-1] - action_block_width + neighbor_slot * n_actions
            )
            action_end = action_start + n_actions
            observation[:, ego_agent, action_start:action_end] = surrounding_action


def get_observation_key(parameters):
    if parameters.is_using_opponent_modeling:
        return ("agents", "info", "observation_with_actions")
    if parameters.is_using_prioritized_marl or getattr(
        parameters, "is_obs_vehicle_id_in_motion_policy", False
    ):
        return ("agents", "info", "observation_without_actions")
    return ("agents", "observation")


def get_priority_observation_key(parameters=None):
    if parameters is not None and getattr(
        parameters, "is_using_prioritized_marl", False
    ):
        return ("agents", "info", "priority_observation")
    return ("agents", "info", "observation_without_actions")


def get_cbf_observation_key():
    """The observation key of CBF"""
    return ("agents", "info", "cbf_observation")


# def prioritized_ap_policy(
#     tensordict,
#     policy,
#     priority_module,
#     nearing_agents_indices,
#     is_communication_noise,
#     communication_noise_level,
# ):
#     """
#     Implements prioritized action propagation (AP) for multiple agents.
#     The function first generates a priority rank using the provided priority module.
#     Then, agents are processed in this priority order, where each agent computes its action
#     and propagates it to lower-priority agents as part of their observation.

#     Since the policy call generates actions for all agents in all environments at once,
#     the function uses a mask to isolate the actions of the agent whose turn it is to act.
#     These actions are progressively combined to form the full action tensor.

#     Parameters:
#     -----------
#     tensordict : TensorDict
#         A dictionary-like object that stores the data for all agents and environments.
#     policy : Callable
#         The policy function used to compute the actions for the agents.
#     priority_module : Callable
#         A module that computes the priority rank for agents by wrapping the process
#         of generating priority scores and ranking agents according to these scores.
#     nearing_agents_indices : Tensor
#         A tensor indicating the neighboring agents for each agent in each environment.

#     Returns:
#     --------
#     tensordict : TensorDict
#         The updated tensordict with combined actions and observations after prioritized action propagation.
#     """
#     base_observation_key = ("agents", "info", "base_observation")

#     device = tensordict.device

#     # Clone original observation
#     original_obs = tensordict[base_observation_key].clone()

#     # Infer parameters
#     n_envs, n_agents, obs_dim, action_dim = (
#         original_obs.shape[0],
#         original_obs.shape[1],
#         original_obs.shape[2],
#         AGENTS["n_actions"],
#     )

#     # Generate priority rank using the priority module
#     priority_module(tensordict)
#     # Extract priority rank (shape: (n_envs, n_agents)) from tensordict
#     priority_rank = tensordict[priority_module.prefix_key + ("rank",)]

#     # Temporary tensors to store intermediate observations and combined results
#     temp_obs = torch.zeros(n_envs, n_agents, obs_dim)
#     combined_action = torch.zeros(n_envs, n_agents, action_dim)
#     combined_loc = torch.zeros(n_envs, n_agents, action_dim)
#     combined_sample_log_prob = torch.zeros(n_envs, n_agents)
#     combined_scale = torch.zeros(n_envs, n_agents, action_dim)
#     combined_obs = torch.zeros(n_envs, n_agents, obs_dim)

#     # Observable high-priority agents
#     observable_higher_priority_agents = [
#         [[] for _ in range(n_agents)] for _ in range(n_envs)
#     ]

#     time_rl = 0

#     # Loop through each step in the priority rank
#     for turn in range(n_agents):
#         # Reset the observation
#         tensordict[("agents", "info")].set("base_observation", original_obs)

#         # Get the list of agents for the current turn based on priority
#         current_turn_agents = priority_rank[:, turn]

#         # Create environment indices (from 0 to n_envs - 1)
#         envs = torch.arange(n_envs)

#         # Create a mask indicating which agents are acting in each environment
#         mask = torch.zeros(n_envs, n_agents, dtype=torch.bool)
#         mask[envs, current_turn_agents] = True

#         # Prepare input for the policy by modifying the observations
#         for env in range(n_envs):
#             agent_idx = current_turn_agents[env]
#             obs = tensordict[base_observation_key][env, agent_idx].clone()

#             # Get the indices of neighboring agents
#             current_turn_agent_neighbors = nearing_agents_indices[env, agent_idx].to(
#                 torch.int64
#             )

#             # Collect actions of the current agent's neighbors
#             actions_so_far = combined_action[env, current_turn_agent_neighbors].view(-1)

#             # Set of higher-priority agents
#             higher_priority_agents = priority_rank[env, :turn]

#             # Observable high-priority agents
#             are_observable_higher_priority_agents = torch.isin(
#                 current_turn_agent_neighbors, higher_priority_agents
#             )
#             observable_higher_priority_agents[env][
#                 agent_idx
#             ] = current_turn_agent_neighbors[are_observable_higher_priority_agents]

#             # Propagate the collected actions into the current agent's observation
#             if len(actions_so_far) > 0:
#                 if is_communication_noise:
#                     std_noise_speed = AGENTS["max_speed"] * communication_noise_level
#                     std_noise_steering = (
#                         AGENTS["max_steering"] * communication_noise_level
#                     )
#                     noise = torch.cat(
#                         [
#                             std_noise_speed * torch.randn(action_dim, device=device),
#                             std_noise_steering * torch.randn(action_dim, device=device),
#                         ],
#                         dim=-1,
#                     )
#                     obs[-len(actions_so_far) :] = actions_so_far + noise
#                 else:
#                     obs[-len(actions_so_far) :] = actions_so_far

#             # Store the updated observation for the current agent in temp_obs
#             temp_obs[env, agent_idx] = obs

#         # Update the base observation with temp_obs for policy execution
#         tensordict[("agents", "info")].set("base_observation", temp_obs)

#         # Call the policy to generate actions for the agents
#         time_rl_start = time.time()
#         policy(tensordict)
#         time_rl += time.time() - time_rl_start

#         # Use the mask to place the data into the correct positions in the combined tensors
#         combined_action[mask] = tensordict[("agents", "action")][mask]
#         combined_loc[mask] = tensordict[("agents", "loc")][mask]
#         combined_sample_log_prob[mask] = tensordict[("agents", "sample_log_prob")][mask]
#         combined_scale[mask] = tensordict[("agents", "scale")][mask]
#         combined_obs[mask] = tensordict[base_observation_key][mask]

#     # Write the combined actions back to the tensordict
#     tensordict[("agents", "action")] = combined_action
#     tensordict[("agents", "loc")] = combined_loc
#     tensordict[("agents", "sample_log_prob")] = combined_sample_log_prob
#     tensordict[("agents", "scale")] = combined_scale
#     tensordict[base_observation_key] = combined_obs

#     return time_rl, tensordict, priority_rank, observable_higher_priority_agents


def _update_fully_decentralized_qps_batched(tensordict, controllers):
    """Update the fully decentralized QPs of one env and serve the
    backup-horizon rollouts of all its agents jointly.

    Every agent runs CBFQP.update_qp_steps up to its next rollout request, and
    the requests of a round are served in one vectorized rollout, which gives
    each agent the poses of its own rollout bit for bit (see
    sigmarl/backup_horizon_cbf.py). Once all QPs are solved, the agents write
    their results in agent order (WRITE_BARRIER). Their QPs read only the
    shared step state and their own action, so the results equal those of
    updating the agents one after another.
    """
    steps = [controller.update_qp_steps(tensordict) for controller in controllers]
    replies = [None] * len(steps)
    running = list(range(len(steps)))
    at_barrier = []
    while running:
        requests = []
        for j in running:
            try:
                request = steps[j].send(replies[j])
            except StopIteration:
                continue
            if request is WRITE_BARRIER:
                at_barrier.append(j)
            else:
                requests.append((j, request))
        if requests:
            poses = rollout_requests([request for _, request in requests])
            for (j, _), poses_j in zip(requests, poses):
                replies[j] = poses_j
        running = [j for j, _ in requests]
    for j in sorted(at_barrier):
        try:
            steps[j].send(None)
        except StopIteration:
            continue
        raise RuntimeError(f"Agent {j} yielded after its WRITE_BARRIER.")


def cbf_filter_deviation(cbf_controllers) -> tuple:
    """Normalized changes [n_envs, n_agents] of the speed and steering targets
    by the CBF filter in the current step: |applied - nominal| / v_max and
    / steering_max, with the nominal targets clipped to the set reachable in
    one step and to the speed range (raw targets beyond the actuator limits
    are saturated by the dynamics anyway and do not count as changes)."""
    first = cbf_controllers[0]
    first = first[0] if isinstance(first, list) else first
    scenario = first.env.base_env.scenario_name
    world_state = scenario.world_state
    agents = scenario.world.agents
    dt = first.dt
    speed = torch.stack([a.state.speed.reshape(-1) for a in agents], dim=1)
    steering = torch.stack([a.state.steering.reshape(-1) for a in agents], dim=1)
    v_min, v_max = float(first.v_min), float(first.v_max)
    steer_min, steer_max = float(first.steering_min), float(first.steering_max)
    # Speed targets below the minimum speed (e.g., reversing at standstill) are
    # clipped by the dynamics as well
    nominal_vel = world_state.nominal_action_vel.clamp(v_min, v_max)
    if scenario.is_heterogeneous_vehicles:
        # Each vehicle's own acceleration limits
        a_min = torch.tensor([a.dynamics.min_acc for a in agents], dtype=speed.dtype)
        a_max = torch.tensor([a.dynamics.max_acc for a in agents], dtype=speed.dtype)
        nominal_vel = torch.minimum(
            torch.maximum(nominal_vel, speed + a_min * dt), speed + a_max * dt
        )
    else:
        nominal_vel = torch.minimum(
            torch.maximum(nominal_vel, speed + first.a_min * dt),
            speed + first.a_max * dt,
        )
    nominal_steer = world_state.nominal_action_steer.clamp(steer_min, steer_max)
    nominal_steer = torch.minimum(
        torch.maximum(nominal_steer, steering + first.steering_rate_min * dt),
        steering + first.steering_rate_max * dt,
    )
    dev_vel = (world_state.applied_action_vel - nominal_vel).abs() / v_max
    dev_steer = (world_state.applied_action_steer - nominal_steer).abs() / steer_max
    return dev_vel, dev_steer


def cbf_constrained_policy_main(
    tensordict,
    policy,
    cbf_controllers: Optional[
        Union[
            list[Union[CBFQPCVXPY, CBFQPProxSuite]],
            list[list[Union[CBFQPCVXPY, CBFQPProxSuite]]],
        ]
    ],
    timing: Optional[dict] = None,
    is_log_backup_value: bool = False,
):
    """
    This function certifies the safety of actions for multiple agents using Control Barrier Functions (CBFs).

    Args:
        timing: if given, receives "slack_active" (whether any CBF-QP of the
            step needed slack or failed) and "parallel", the parallelized
            estimate of the CBF time [s]. For fully decentralized controllers, it adds the
            shared per-step preparation in full, the slowest agent's QP (with
            shield round 1), the slowest agent's shield round 2, and the shared
            finalizer; otherwise it equals the sequential CBF time. Without
            it (training collector), the fully decentralized agent QPs of an
            env share their backup-horizon rollouts (see
            _update_fully_decentralized_qps_batched); the results are the same.
        is_log_backup_value: with timing, also store the joint backup value of
            every agent at the step's current states in timing["backup_value"]
            [n_envs, n_agents] (CBFQP.backup_values_all_envs; not timed).
    """
    time_pseudo_dis = 0

    # Call the policy to generate actions for the agents
    time_rl_start = time.time()
    policy(tensordict)
    time_rl = time.time() - time_rl_start  # Time for RL policy

    # Cache the raw RL action *before* the CBF controller may overwrite ("agents", "action") with the safe action. Used by the training collector's
    # shielded-PPO mode (is_store_rl_action_under_cbf) to store the RL action in the transition while the environment executes the CBF (applied) action.
    # The RL action's log-prob is already in ("agents", "sample_log_prob") and is not touched by the QP, so the stored (action, log-prob) pair stays consistent.
    # None when the active policy predicts only CBF lambda (no motion action yet).
    policy._rl_action_cache = (
        tensordict.get(("agents", "action")).clone()
        if ("agents", "action") in tensordict.keys(True, True)
        else None
    )

    # The environment always consumes ("agents", "action"), even when the active
    # policy only predicts CBF lambda. In that case, create a placeholder action
    # tensor and let the CBF controller overwrite it with the safe action.
    if ("agents", "action") not in tensordict.keys(True, True):
        if not hasattr(policy, "env"):
            raise KeyError(
                'key ("agents", "action") is missing and policy has no env to infer the action shape.'
            )

        try:
            action_spec = policy.env.full_action_spec_unbatched[policy.env.action_key]
        except AttributeError:
            action_spec = policy.env.unbatched_action_spec[policy.env.action_key]

        action_dim = action_spec.shape[-1]
        action_dtype = action_spec.dtype
        action_device = tensordict.device
        if action_device is None:
            obs = tensordict.get(("agents", "observation"))
            action_device = obs.device

        if ("agents", "observation") in tensordict.keys(True, True):
            batch_shape = tensordict.get(("agents", "observation")).shape[:-1]
        else:
            cbf_lambda = tensordict.get(("agents", "cbf", "lambda"))
            batch_shape = cbf_lambda.shape[:-1]

        tensordict.set(
            ("agents", "action"),
            torch.zeros(
                *batch_shape,
                action_dim,
                device=action_device,
                dtype=action_dtype,
            ),
        )

    time_cbf_start = time.time()
    n_envs = tensordict.batch_size[0]
    batched_nominal_update_time = None
    time_parallel = 0.0
    # Parts of the parallelized estimate (CBFQP.time_parts): the slowest
    # agent's in the fully decentralized mode
    time_parts = {}
    if n_envs > 0:
        # In the fully decentralized mode, each env has a list of agent controllers
        first_controller = (
            cbf_controllers[0][0]
            if isinstance(cbf_controllers[0], list)
            else cbf_controllers[0]
        )
        batched_update = getattr(
            first_controller, "update_nominal_cbf_margin_rewards_all_envs", None
        )
        if batched_update is not None:
            batched_nominal_update_time = batched_update(
                tensordict, controllers=cbf_controllers
            )
    time_parallel += time.time() - time_cbf_start

    if batched_nominal_update_time is not None:
        time_pseudo_dis += batched_nominal_update_time
    else:
        for i in range(n_envs):
            # cbf_controllers[i].solve_centralized_cbf_qp(tensordict)  # Solve QP at each time step
            # cbf_controllers[i].build_centralized_cbf_qp()  # Build QP at each time step
            # cbf_controllers[i].update_centralized_cbf_qp(tensordict)  # Build QP at each time step
            time_env_start = time.time()
            if isinstance(cbf_controllers[i], list):
                first_controller = cbf_controllers[i][0] if cbf_controllers[i] else None
                cache_prepared = False
                if first_controller is not None and hasattr(
                    first_controller, "prepare_fully_decentralized_step_cache"
                ):
                    first_controller.prepare_fully_decentralized_step_cache(
                        tensordict, cbf_controllers[i]
                    )
                    cache_prepared = True
                # Parallelized estimate: shared preparation, slowest agent per
                # round, and the shared finalizer
                time_shared = time.time() - time_env_start
                time_agents = [0.0]
                parts_agents = [{}]
                time_round2 = [0.0]
                try:
                    if (
                        timing is None
                        and first_controller is not None
                        and first_controller.parameters.is_proxsuite_qp
                    ):
                        # No per-agent time needed: batch the agents' rollouts
                        _update_fully_decentralized_qps_batched(
                            tensordict, cbf_controllers[i]
                        )
                        for controller in cbf_controllers[i]:
                            time_pseudo_dis += controller.time_pseudo_dis
                    else:
                        for j in range(len(cbf_controllers[i])):
                            time_agent_start = time.time()
                            cbf_controllers[i][j].update_qp(tensordict)
                            time_agents.append(time.time() - time_agent_start)
                            parts_agents.append(cbf_controllers[i][j].time_parts)
                            time_pseudo_dis += cbf_controllers[i][j].time_pseudo_dis
                    if (
                        first_controller is not None
                        and first_controller.parameters.is_exact_backup_shield
                    ):
                        time_shield_start = time.time()
                        shield_infos = (
                            first_controller.apply_fully_decentralized_exact_shield(
                                tensordict, cbf_controllers[i]
                            )
                        )
                        time_round2 = [0.0] + [
                            info["time"] for info in shield_infos.values()
                        ]
                        time_shared += (
                            time.time() - time_shield_start - sum(time_round2)
                        )
                finally:
                    time_clear_start = time.time()
                    if cache_prepared and hasattr(
                        first_controller, "clear_fully_decentralized_step_cache"
                    ):
                        first_controller.clear_fully_decentralized_step_cache(
                            cbf_controllers[i]
                        )
                    time_shared += time.time() - time_clear_start
                time_parallel += time_shared + max(time_agents) + max(time_round2)
                slowest = parts_agents[int(np.argmax(time_agents))]
                for part, value in slowest.items():
                    time_parts[part] = time_parts.get(part, 0.0) + value
            else:
                cbf_controllers[i].update_qp(tensordict)
                time_pseudo_dis += cbf_controllers[i].time_pseudo_dis
                time_parallel += time.time() - time_env_start
                for part, value in cbf_controllers[i].time_parts.items():
                    time_parts[part] = time_parts.get(part, 0.0) + value
    time_cbf = time.time() - time_cbf_start
    if timing is not None and is_log_backup_value and n_envs > 0:
        timing["backup_value"] = first_controller.backup_values_all_envs(
            cbf_controllers
        )
    if timing is not None:
        timing["slack_active"] = any(
            ctrl.last_slack_active
            for item in cbf_controllers
            for ctrl in (item if isinstance(item, list) else [item])
        )
        timing["parallel"] = (
            time_parallel if batched_nominal_update_time is None else time_cbf
        )
        if time_parts:
            timing["parts"] = time_parts

    return (
        time_rl,
        time_cbf,
        time_pseudo_dis,
        tensordict,
    )


def reduce_out_td(
    out_td: TensorDict,
    *,
    pos_predicted: torch.Tensor | None = None,  # (T, A, H, 1) if available
    vel_predicted: torch.Tensor | None = None,  # (T, A, H, 1) if available
    convert_collisions_to_bool: bool = True,
) -> TensorDict:
    # Safety: ensure expected keys exist
    def _get(*keypath):
        try:
            return out_td.get(keypath)
        except KeyError as e:
            raise KeyError(f"Missing key {keypath} in out_td") from e

    # Remove the leading batch dimension of size 1 -> (T, A, F)
    def _squeeze_b1(x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 4:
            raise ValueError(
                f"Expected a tensor with 4 dims (1,T,A,F), got shape {tuple(x.shape)}"
            )
        if x.shape[0] != 1:
            raise ValueError(f"Expected leading batch size 1, got {x.shape[0]}")
        return x.squeeze(0)

    pos = _squeeze_b1(_get("agents", "info", "pos"))  # (T, A, 2)
    rot = _squeeze_b1(_get("agents", "info", "rot"))  # (T, A, 1)
    rot = rot.squeeze(-1)
    vel = _squeeze_b1(_get("agents", "info", "vel"))  # (T, A, 2)

    col_agents = _squeeze_b1(
        _get("agents", "info", "is_collision_with_agents")
    )  # (T, A, 1)
    col_lane = _squeeze_b1(
        _get("agents", "info", "is_collision_with_lanelets")
    )  # (T, A, 1)

    if convert_collisions_to_bool:
        # Stored as float32; convert 0/1 to bool without changing shape
        col_agents = col_agents > 0.5
        col_lane = col_lane > 0.5

    # If you already have lists like self.pos_predicted_list, stack them before calling:
    # pos_predicted = torch.stack(self.pos_predicted_list, dim=0)  # (T, A, H, 1)
    # vel_predicted = torch.stack(self.vel_predicted_list, dim=0)  # (T, A, H, 1)

    td_content = {
        "pos": pos,  # (T, A, 2)
        "rot": rot,  # (T, A, 1)
        "vel": vel,  # (T, A, 2)
        "is_collision_with_agents": col_agents,  # (T, A, 1)
        "is_collision_with_lanelets": col_lane,  # (T, A, 1)
    }

    if pos_predicted is not None:
        td_content["pos_predicted_list"] = pos_predicted  # (T, A, H, 1)
    if vel_predicted is not None:
        td_content["vel_predicted_list"] = vel_predicted  # (T, A, H, 1)

    # You asked for batch_size=[] explicitly; that is fine for a container of leaf tensors.
    td = TensorDict(td_content, batch_size=[])
    return td


# def cbf_constrained_decentralized_policy(
#     tensordict,
#     policy,
#     cbf_controllers: list[list[CBFQP]],
# ):
#     """

#     This function is used to make action decisions for multiple agents at each time step,
#     based on their priority, and enforces safety constraints via Control Barrier Functions (CBFs).
#     Each agent makes decisions independently based on its observations
#     and the actions of its neighbors.

#     Parameters:
#     ----------
#     tensordict : TensorDict
#         A dictionary-like object that stores the data for all agents and environments.
#     policy : Callable
#         The policy function used to compute the actions for the agents.
#     priority_module : Callable
#         A module that computes the priority rank for agents by wrapping the process
#         of generating priority scores and ranking agents according to these scores.
#     cbf_controllers : List[List[CBFQP]]
#         A list of CBF-based controllers for each environment and agent, used to
#         correct the unsafe policy's actions.
#     nearing_agents_indices : Tensor
#         A tensor indicating the neighboring agents for each agent in each environment.

#     Returns:
#     ----------
#     tensordict : TensorDict
#         The updated tensordict with corrected safe actions with CBF.
#     """
#     base_observation_key = ("agents", "info", "base_observation")

#     # Clone original observation
#     original_obs = tensordict[base_observation_key].clone()

#     # Infer parameters
#     n_envs, n_agents, obs_dim, action_dim = (
#         original_obs.shape[0],
#         original_obs.shape[1],
#         original_obs.shape[2],
#         AGENTS["n_actions"],
#     )

#     time_cbf = 0
#     time_rl = 0
#     time_pseudo_dis = 0

#     # Call the policy to generate actions for the agents
#     time_rl_start = time.time()
#     policy(tensordict)
#     time_rl += time.time() - time_rl_start  # Time for RL policy

#     if "nominal_action" not in tensordict[("agents", "info")].keys():
#         nominal_action = torch.zeros(
#             n_envs,
#             n_agents,
#             action_dim,
#             dtype=torch.float32,
#             device=tensordict.device,
#         )
#         tensordict[("agents", "info")].set("nominal_action", nominal_action)

#     # CBF-QP solving
#     time_cbf_start = time.time()
#     for i in range(n_envs):
#         for j in range(n_agents):
#             cbf_controllers[i][j].solve_decentralized_cbf_qp(tensordict)
#             time_pseudo_dis += cbf_controllers[i][j].time_pseudo_dis
#     time_cbf += time.time() - time_cbf_start

#     return (
#         time_rl,
#         time_cbf,
#         time_pseudo_dis,
#         tensordict,
#     )


# def cbf_constrained_decentralized_policy_multi_agents(
#     tensordict,
#     policy,
#     priority_module,
#     cbf_controllers: list[list[CBFQP]],
#     nearing_agents_indices,
#     is_communication_noise,
#     communication_noise_level,
# ):
#     """

#     This function is used to make action decisions for multiple agents at each time step,
#     based on their priority, and enforces safety constraints via Control Barrier Functions (CBFs).
#     Each agent makes decisions independently based on its observations
#     and the actions of its neighbors.

#     Parameters:
#     ----------
#     tensordict : TensorDict
#         A dictionary-like object that stores the data for all agents and environments.
#     policy : Callable
#         The policy function used to compute the actions for the agents.
#     priority_module : Callable
#         A module that computes the priority rank for agents by wrapping the process
#         of generating priority scores and ranking agents according to these scores.
#     cbf_controllers : List[List[CBFQP]]
#         A list of CBF-based controllers for each environment and agent, used to
#         correct the unsafe policy's actions.
#     nearing_agents_indices : Tensor
#         A tensor indicating the neighboring agents for each agent in each environment.

#     Returns:
#     ----------
#     tensordict : TensorDict
#         The updated tensordict with corrected safe actions with CBF.
#     """
#     base_observation_key = ("agents", "info", "base_observation")
#     cbf_observation_key = ("agents", "info", "cbf_observation")

#     device = tensordict.device

#     # Clone original observation
#     original_obs = tensordict[base_observation_key].clone()
#     original_cbf_obs = tensordict[cbf_observation_key].clone()

#     # Infer parameters
#     n_envs, n_agents, obs_dim, action_dim = (
#         original_obs.shape[0],
#         original_obs.shape[1],
#         original_obs.shape[2],
#         AGENTS["n_actions"],
#     )

#     cbf_obs_dim = original_cbf_obs.shape[2]
#     n_nearing_agents = nearing_agents_indices.shape[-1]
#     # Generate priority rank using the priority module
#     priority_module(tensordict)
#     # Extract priority rank (shape: (n_envs, n_agents)) from tensordict
#     priority_rank = tensordict[priority_module.prefix_key + ("rank",)]

#     # Temporary tensors to store intermediate observations and combined results
#     temp_obs = torch.zeros(n_envs, n_agents, obs_dim)
#     combined_action = torch.zeros(n_envs, n_agents, action_dim)
#     combined_loc = torch.zeros(n_envs, n_agents, action_dim)
#     combined_sample_log_prob = torch.zeros(n_envs, n_agents)
#     combined_scale = torch.zeros(n_envs, n_agents, action_dim)
#     combined_obs = torch.zeros(n_envs, n_agents, obs_dim)

#     # Observable high-priority agents
#     observable_higher_priority_agents = [
#         [[] for _ in range(n_agents)] for _ in range(n_envs)
#     ]

#     time_cbf = 0
#     time_rl = 0
#     time_pseudo_dis = 0

#     # Loop through each step in the priority rank
#     for turn in range(n_agents):
#         # Reset the observation
#         tensordict[("agents", "info")].set("base_observation", original_obs)
#         tensordict[("agents", "info")].set("cbf_observation", original_cbf_obs)

#         # Get the list of agents for the current turn based on priority
#         current_turn_agents = priority_rank[:, turn]

#         # Create environment indices (from 0 to n_envs - 1)
#         envs = torch.arange(n_envs)

#         # Create a mask indicating which agents are acting in each environment
#         mask = torch.zeros(n_envs, n_agents, dtype=torch.bool)
#         mask[envs, current_turn_agents] = True

#         # Create a mask indication which higher priority agents are nearing the currrent agents
#         mask_higher_priority_agents = torch.zeros(
#             n_envs, n_agents, n_nearing_agents, dtype=torch.bool
#         )
#         temp_cbf_obs = torch.zeros(n_envs, n_agents, cbf_obs_dim)

#         # Prepare input for the policy by modifying the observations
#         for env in range(n_envs):
#             agent_idx = current_turn_agents[env]
#             obs = tensordict[base_observation_key][env, agent_idx].clone()
#             cbf_obs = tensordict[cbf_observation_key][env, agent_idx].clone()

#             # Get the indices of neighboring agents
#             current_turn_agent_neighbors = nearing_agents_indices[env, agent_idx].to(
#                 torch.int64
#             )

#             # Collect actions of the current agent's neighbors
#             actions_so_far = combined_action[env, current_turn_agent_neighbors].view(-1)

#             # Set of higher-priority agents
#             higher_priority_agents = priority_rank[env, :turn]

#             # Observable high-priority agents
#             are_observable_higher_priority_agents = torch.isin(
#                 current_turn_agent_neighbors, higher_priority_agents
#             )
#             observable_higher_priority_agents[env][
#                 agent_idx
#             ] = current_turn_agent_neighbors[are_observable_higher_priority_agents]

#             mask_higher_priority_agents[
#                 env, agent_idx, :
#             ] = are_observable_higher_priority_agents

#             # Propagate the collected actions into the current agent's observation
#             if len(actions_so_far) > 0:
#                 if is_communication_noise:
#                     std_noise_speed = AGENTS["max_speed"] * communication_noise_level
#                     std_noise_steering = (
#                         AGENTS["max_steering"] * communication_noise_level
#                     )
#                     noise = torch.cat(
#                         [
#                             std_noise_speed * torch.randn(action_dim, device=device),
#                             std_noise_steering * torch.randn(action_dim, device=device),
#                         ],
#                         dim=-1,
#                     )
#                     obs[-len(actions_so_far) :] = actions_so_far + noise
#                     cbf_obs[-len(actions_so_far) :] = actions_so_far + noise
#                 else:
#                     obs[-len(actions_so_far) :] = actions_so_far
#                     cbf_obs[-len(actions_so_far) :] = actions_so_far

#             # Store the updated observation for the current agent in temp_obs
#             temp_obs[env, agent_idx] = obs
#             temp_cbf_obs[env, agent_idx] = cbf_obs

#         # Update the base_observation with temp_obs for policy execution
#         tensordict[("agents", "info")].set("base_observation", temp_obs)
#         # Update the cbf_observation with temp_cbf_obs for policy execution
#         tensordict[("agents", "info")].set("cbf_observation", temp_cbf_obs)

#         # Call the policy to generate actions for the agents
#         time_rl_start = time.time()
#         policy(tensordict)
#         time_rl += time.time() - time_rl_start  # Time for RL policy

#         if "nominal_action" not in tensordict[("agents", "info")].keys():
#             nominal_action = torch.zeros(
#                 n_envs,
#                 n_agents,
#                 action_dim,
#                 dtype=torch.float32,
#                 device=tensordict.device,
#             )
#             tensordict[("agents", "info")].set("nominal_action", nominal_action)

#         # CBF-QP solving
#         time_cbf_start = time.time()
#         for i in range(n_envs):
#             for j in range(n_agents):
#                 if mask[i, j]:
#                     cbf_controllers[i][j].solve_decentralized_cbf_qp(
#                         tensordict, mask_higher_priority_agents
#                     )
#                     time_pseudo_dis += cbf_controllers[i][j].time_pseudo_dis
#         time_cbf += time.time() - time_cbf_start

#         # Use the mask to place the data into the correct positions in the combined tensors
#         combined_action[mask] = tensordict[("agents", "action")][mask]
#         combined_loc[mask] = tensordict[("agents", "loc")][mask]
#         combined_sample_log_prob[mask] = tensordict[("agents", "sample_log_prob")][mask]
#         combined_scale[mask] = tensordict[("agents", "scale")][mask]
#         combined_obs[mask] = tensordict[base_observation_key][mask]

#     tensordict[("agents", "action")] = combined_action
#     tensordict[("agents", "loc")] = combined_loc
#     tensordict[("agents", "sample_log_prob")] = combined_sample_log_prob
#     tensordict[("agents", "scale")] = combined_scale
#     tensordict[base_observation_key] = combined_obs
#     tensordict[cbf_observation_key] = original_cbf_obs

#     return (
#         time_rl,
#         time_cbf,
#         time_pseudo_dis,
#         tensordict,
#         priority_rank,
#         observable_higher_priority_agents,
#     )
