# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import sys
from torch import Tensor

import typing

from vmas.simulator.core import World, AgentState, Agent
from vmas.simulator.utils import TorchUtils, override

# Tensordict modules
from tensordict.tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

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
from torchrl.envs.libs.vmas import VmasEnv


# TorchRL Utils
from torchrl._utils import (
    RL_WARNINGS,
)

# Losses
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from torchrl.data.utils import DEVICE_TYPING

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Specs
from torchrl.data import Unbounded

# Utils
from matplotlib import pyplot as plt
from typing import Callable, Optional, Callable, Optional

import json
import os
import re
import time

from typing import Callable, Dict, Optional, Sequence, Union

DEFAULT_EXPLORATION_TYPE: ExplorationType = ExplorationType.RANDOM

import warnings

from sigmarl.constants import AGENTS


def get_model_name(parameters):
    # model_name = f"nags{parameters.n_agents}_it{parameters.n_iters}_fpb{parameters.frames_per_batch}_tfrms{parameters.total_frames}_neps{parameters.num_epochs}_mnbsz{parameters.minibatch_size}_lr{parameters.lr}_mgn{parameters.max_grad_norm}_clp{parameters.clip_epsilon}_gm{parameters.gamma}_lmbda{parameters.lmbda}_etp{parameters.entropy_eps}_mstp{parameters.max_steps}_nenvs{parameters.num_vmas_envs}"
    model_name = f"reward{parameters.episode_reward_mean_current:.2f}"

    return model_name


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

        Returns:
            TensorDict object containing the resulting trajectory.

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
        }
        self.time_rl = []
        self.time_cbf = []
        self.time_pseudo_dis = []
        self.num_fail = 0
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

        if is_save_simulation_video:
            return out_td, frame_list  # Modification
        else:
            return out_td, None

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
    ):
        tensordicts = []

        if is_save_simulation_video:
            frame_list = []

        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict = tensordict.to(policy_device, non_blocking=True)

            # Possibly predict the actions of surrounding agents using opponent modeling
            if self.base_env.scenario_name.parameters.is_using_opponent_modeling:
                time_rl = opponent_modeling(
                    tensordict=tensordict,
                    policy=policy,
                    n_nearing_agents_observed=self.base_env.scenario_name.parameters.n_nearing_agents_observed,
                    nearing_agents_indices=self.base_env.scenario_name.observations.nearing_agents_indices,
                )

            if (
                self.base_env.scenario_name.parameters.is_using_prioritized_marl
                and not self.base_env.scenario_name.parameters.is_using_cbf
            ):
                (
                    time_rl,
                    tensordict,
                    priority_rank,
                    observable_higher_priority_agents,
                ) = prioritized_ap_policy(
                    tensordict,
                    policy,
                    priority_module,
                    self.base_env.scenario_name.observations.nearing_agents_indices,
                    self.base_env.scenario_name.parameters.is_communication_noise,
                    self.base_env.scenario_name.parameters.communication_noise_level,
                )
                # Store priority_rank for later use
                self.base_env.scenario_name.priority_rank = priority_rank
                self.base_env.scenario_name.observable_higher_priority_agents = (
                    observable_higher_priority_agents
                )
            elif (
                self.base_env.scenario_name.parameters.is_using_cbf
                and not self.base_env.scenario_name.parameters.is_using_centralized_cbf
            ):
                (
                    time_rl,
                    time_cbf,
                    time_pseudo_dis,
                    tensordict,
                ) = cbf_constrained_decentralized_policy(
                    tensordict,
                    policy,
                    cbf_controllers,
                )
                # Store time for computational efficiency
                self.time_cbf.append(time_cbf)
                self.time_pseudo_dis.append(time_pseudo_dis)

            elif (
                self.base_env.scenario_name.parameters.is_using_cbf
                and self.base_env.scenario_name.parameters.is_using_centralized_cbf
            ):
                time_rl, tensordict = cbf_constrained_centralized_policy(
                    tensordict, policy, cbf_controllers
                )
            else:
                # Baseline
                time_rl_start = time.time()
                tensordict = policy(tensordict)
                time_rl += time.time() - time_rl_start  # Time for RL policy

            self.time_rl.append(time_rl)

            if auto_cast_to_device:
                tensordict = tensordict.to(env_device, non_blocking=True)
            tensordict = self.step(tensordict)
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
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = callback(self, tensordict)  # Modification
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)
        if self.base_env.scenario_name.parameters.is_using_cbf:
            num_fail = 0
            for i in range(len(cbf_controllers)):
                for j in range(len(cbf_controllers[i])):
                    num_fail += cbf_controllers[i][j].num_fail
            self.num_fail = num_fail
            # print("Total number of cbf failure solving is:", num_fail)
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
    ):
        tensordicts = []
        tensordict_ = tensordict

        if is_save_simulation_video:
            frame_list = []

        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict_ = tensordict_.to(policy_device, non_blocking=True)

            # Possibly predict the actions of surrounding agents using opponent modeling
            if self.base_env.scenario_name.parameters.is_using_opponent_modeling:
                time_rl = opponent_modeling(
                    tensordict=tensordict_,
                    policy=policy,
                    n_nearing_agents_observed=self.base_env.scenario_name.parameters.n_nearing_agents_observed,
                    nearing_agents_indices=self.base_env.scenario_name.observations.nearing_agents_indices,
                )

            if (
                self.base_env.scenario_name.parameters.is_using_prioritized_marl
                and not self.base_env.scenario_name.parameters.is_using_cbf
            ):
                (
                    time_rl,
                    tensordict_,
                    priority_rank,
                    observable_higher_priority_agents,
                ) = prioritized_ap_policy(
                    tensordict_,
                    policy,
                    priority_module,
                    self.base_env.scenario_name.observations.nearing_agents_indices,
                    self.base_env.scenario_name.parameters.is_communication_noise,
                    self.base_env.scenario_name.parameters.communication_noise_level,
                )
                # Store priority_rank for later use
                self.base_env.scenario_name.priority_rank = priority_rank
                self.base_env.scenario_name.observable_higher_priority_agents = (
                    observable_higher_priority_agents
                )
            elif (
                self.base_env.scenario_name.parameters.is_using_cbf
                and not self.base_env.scenario_name.parameters.is_using_centralized_cbf
            ):
                (
                    time_rl,
                    time_cbf,
                    time_pseudo_dis,
                    tensordict_,
                ) = cbf_constrained_decentralized_policy(
                    tensordict_,
                    policy,
                    cbf_controllers,
                )
                # Store time for computational efficiency
                self.time_cbf.append(time_cbf)
                self.time_pseudo_dis.append(time_pseudo_dis)
                self.base_env.scenario_name.nominal_action = tensordict[
                    "agents", "info", "nominal_action"
                ]
            elif (
                self.base_env.scenario_name.parameters.is_using_cbf
                and self.base_env.scenario_name.parameters.is_using_centralized_cbf
            ):
                time_rl, tensordict_ = cbf_constrained_centralized_policy(
                    tensordict_, policy, cbf_controllers
                )
            else:
                time_rl_start = time.time()
                tensordict_ = policy(tensordict_)
                time_rl = time.time() - time_rl_start

            self.time_rl.append(time_rl)

            if auto_cast_to_device:
                tensordict_ = tensordict_.to(env_device, non_blocking=True)
            tensordict, tensordict_ = self.step_and_maybe_reset(tensordict_)
            tensordicts.append(tensordict)
            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = callback(self, tensordict)  # Modification
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)
        if self.base_env.scenario_name.parameters.is_using_cbf:
            num_fail = 0
            for i in range(len(cbf_controllers)):
                for j in range(len(cbf_controllers[i])):
                    num_fail += cbf_controllers[i][j].num_fail
            self.num_fail = num_fail
            # print("Total number of cbf failure solving is:", num_fail)

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
            observation_spec=self.env.observation_spec,
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
        **kwargs,
    ):
        self.priority_module = priority_module
        self.cbf_controllers = cbf_controllers
        super().__init__(env, policy, **kwargs)
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
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    self.env.rand_action(self._tensordict)

                    # <Modification starts>
                    # Possibly predict the actions of surrounding agents using opponent modeling
                elif (
                    self.env.base_env.scenario_name.parameters.is_using_opponent_modeling
                ):
                    opponent_modeling(
                        tensordict=self._tensordict,
                        policy=self.policy,
                        n_nearing_agents_observed=self.env.base_env.scenario_name.parameters.n_nearing_agents_observed,
                        nearing_agents_indices=self.env.base_env.scenario_name.observations.nearing_agents_indices,
                    )
                    self.policy(self._tensordict)
                # <Modification ends>
                elif (
                    self.env.base_env.scenario_name.parameters.is_using_prioritized_marl
                    and not self.env.base_env.scenario_name.parameters.is_using_cbf
                ):
                    prioritized_ap_policy(
                        tensordict=self._tensordict,
                        policy=self.policy,
                        priority_module=self.priority_module,
                        nearing_agents_indices=self.env.base_env.scenario_name.observations.nearing_agents_indices,
                        is_communication_noise=self.env.base_env.scenario_name.parameters.is_communication_noise,
                        communication_noise_level=self.env.base_env.scenario_name.parameters.communication_noise_level,
                    )
                elif (
                    self.env.base_env.scenario_name.parameters.is_using_cbf
                    and not self.env.base_env.scenario_name.parameters.is_using_centralized_cbf
                ):
                    cbf_constrained_decentralized_policy(
                        tensordict=self._tensordict,
                        policy=self.policy,
                        cbf_controllers=self.cbf_controllers,
                    )
                elif (
                    self.env.base_env.scenario_name.parameters.is_using_cbf
                    and self.env.base_env.scenario_name.parameters.is_using_centralized_cbf
                ):
                    cbf_constrained_centralized_policy(
                        tensordict=self._tensordict,
                        policy=self.policy,
                        cbf_controllers=self.cbf_controllers,
                    )
                else:
                    self.policy(self._tensordict)

                tensordict, tensordict_ = self.env.step_and_maybe_reset(
                    self._tensordict
                )
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


class VehicleState(AgentState):
    def __init__(self):
        """
        Initialize the VehicleState with all attributes from AgentState
        and add some new attributes.
        """
        super().__init__()
        self._speed = None  # Speed (magnitude, positive for forward moving, negative for backward moving)
        self._steering = None
        self._sideslip_angle = None

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._speed = value.to(self._device)

    @property
    def steering(self):
        return self._steering

    @steering.setter
    def steering(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._steering = value.to(self._device)

    @property
    def sideslip_angle(self):
        return self._sideslip_angle

    @sideslip_angle.setter
    def sideslip_angle(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._sideslip_angle = value.to(self._device)

    @override(AgentState)
    def _reset(self, env_index: typing.Optional[int]):
        for attr_name in ["speed", "steering", "sideslip_angle"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                if env_index is None:
                    self.__setattr__(attr_name, torch.zeros_like(attr))
                else:
                    self.__setattr__(
                        attr_name,
                        TorchUtils.where_from_index(env_index, 0, attr),
                    )
        super()._reset(env_index)

    @override(AgentState)
    def zero_grad(self):
        for attr_name in ["speed", "steering", "sideslip_angle"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                self.__setattr__(attr_name, attr.detach())
        super().zero_grad()

    @override(AgentState)
    def _spawn(self, dim_c: int, dim_p: int):
        self.speed = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.steering = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.sideslip_angle = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        super()._spawn(dim_c, dim_p)


class Vehicle(Agent):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Vehicle by calling the parent Agent's initializer
        and setting up the VehicleState.

        Redefine _state attribute to be VehicleState.
        """
        super().__init__(*args, **kwargs)

        # Replace the default AgentState with VehicleState
        self._state = VehicleState()

    def set_speed(self, speed: Tensor, batch_index: int = None):
        """
        Set the speed state of the vehicle.

        Args:
            speed (Tensor): The new speed value.
            batch_index (int, optional): The index in the batch to set the speed for.
                                         If None, all batches are updated.
        """
        self._set_state_property(VehicleState.speed, self.state, speed, batch_index)

    def set_steering(self, steering: Tensor, batch_index: int = None):
        """
        Set the steering state of the vehicle.

        Args:
            steering (Tensor): The new steering value.
            batch_index (int, optional): The index in the batch to set the steering for.
                                         If None, all batches are updated.
        """
        self._set_state_property(
            VehicleState.steering, self.state, steering, batch_index
        )

    def set_sideslip_angle(self, sideslip_angle: Tensor, batch_index: int = None):
        """
        Set the sideslip angle of the vehicle, i.e., the angle between the velocity and the chasis.

        Args:
            sideslip_angle (Tensor): The new sideslip_angle value.
            batch_index (int, optional): The index in the batch to set the sideslip_angle for.
                                         If None, all batches are updated.
        """
        self._set_state_property(
            VehicleState.sideslip_angle, self.state, sideslip_angle, batch_index
        )


class WorldCustom(World):
    """
    The default step function update agents' states based on physical forces.
    Redefine step function to update agents' states based on speed and steering command.
    """

    def step(self):
        for entity in self.entities:
            # if entity.movable:
            # Model drag coefficient
            # if entity.drag is not None:
            #     entity.action.u[:, 0] = entity.action.u[:, 0] * (1 - entity.drag)
            # else:
            #     entity.action.u[:, 0] = entity.action.u[:, 0] * (1 - self._drag)
            # entity.action.u += torch.rand_like(entity.action.u) * 0.1  # Inject random noise to actions
            # Clamp the speed and steering angle to be within bounds
            if hasattr(entity.dynamics, "max_speed"):
                entity.action.u[:, 0] = torch.clamp(
                    entity.action.u[:, 0],
                    -entity.dynamics.max_speed,
                    entity.dynamics.max_speed,
                )
            if hasattr(entity.dynamics, "max_steering"):
                entity.action.u[:, 1] = torch.clamp(
                    entity.action.u[:, 1],
                    -entity.dynamics.max_steering,
                    entity.dynamics.max_steering,
                )

            # Assume linear acceleration and steering change
            u_acc = (entity.action.u[:, 0] - entity.state.speed.squeeze(-1)) / self.dt
            u_steering_rate = (
                entity.action.u[:, 1] - entity.state.steering.squeeze(-1)
            ) / self.dt

            u = torch.cat([u_acc.unsqueeze(-1), u_steering_rate.unsqueeze(-1)], dim=-1)

            # Apply action limits
            u[:, 0] = torch.clamp(
                u[:, 0], entity.dynamics.min_acc, entity.dynamics.max_acc
            )
            u[:, 1] = torch.clamp(
                u[:, 1],
                entity.dynamics.min_steering_rate,
                entity.dynamics.max_steering_rate,
            )

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
                tick_per_step=5,  # Ticks per time step controling the fedelity of the ODE
            )
            # Update states
            entity.state.pos = x1[:, 0:2]
            entity.state.rot = x1[:, 2].unsqueeze(-1)
            entity.state.speed = x1[:, 3].unsqueeze(-1)
            entity.state.steering = x1[:, 4].unsqueeze(-1)
            entity.state.vel = velocity
            entity.state.sideslip_angle = sideslip_angle.unsqueeze(-1)


class Parameters:
    """
    This class stores parameters for training and testing.
    """

    def __init__(
        self,
        # General parameters
        n_agents: int = 4,  # Number of agents
        dt: float = 0.05,  # [s] sample time
        device: str = "cpu",  # Default tensor device
        scenario_name: str = "road_traffic",  # Scenario name
        # Training parameters
        n_iters: int = 250,  # Number of training iterations
        num_epochs: int = 30,  # Optimization steps per batch of data collected
        minibatch_size: int = 512,  # Size of the mini-batches in each optimization step (2**9 - 2**12?)
        lr: float = 2e-4,  # Learning rate
        lr_min: float = 1e-5,  # Minimum learning rate (used for scheduling of learning rate)
        max_grad_norm: float = 1.0,  # Maximum norm for the gradients
        clip_epsilon: float = 0.2,  # Clip value for PPO loss
        gamma: float = 0.99,  # Discount factor from 0 to 1. A greater value corresponds to a better farsight
        lmbda: float = 0.9,  # lambda for generalised advantage estimation
        entropy_eps: float = 1e-4,  # Coefficient of the entropy term in the PPO loss
        max_steps: int = 128,  # Episode steps before done
        num_vmas_envs: int = 32,  # Number of vectorized environments
        scenario_type: str = "intersection_1",  # One of {"CPM_entire", "CPM_mixed", "intersection_1", ...}. See SCENARIOS in utilities/constants.py for more scenarios.
        # "CPM_entire": Entire map of the CPM Lab
        # "CPM_mixed": Intersection, merge-in, and merge-out of the CPM Lab. Probability defined in `cpm_scenario_probabilities`
        # "intersection_1": Intersection with ID 1
        episode_reward_mean_current: float = 0.00,  # Achieved mean episode reward (total/n_agents)
        episode_reward_intermediate: float = -1e3,  # A arbitrary, small initial value
        is_prb: bool = False,  # Whether to enable prioritized replay buffer
        is_challenging_initial_state_buffer=False,  # Whether to enable challenging initial state buffer
        cpm_scenario_probabilities=[
            1.0,
            0.0,
            0.0,
        ],  # Probabilities of training agents in intersection, merge-in, or merge-out scenario
        n_steps_stored: int = 10,  # Store previous `n_steps_stored` steps of states
        # Observation
        n_points_short_term: int = 3,  # Number of points that build a short-term reference path
        is_partial_observation: bool = True,  # Whether to enable partial observation
        n_nearing_agents_observed: int = 2,  # Number of nearing agents to be observed (consider limited sensor range)
        # Parameters for ablation studies
        is_ego_view: bool = True,  # Ego view or bird view
        is_apply_mask: bool = True,  # Whether to mask distant agents
        is_observe_distance_to_agents: bool = True,  # Whether to observe the distance to other agents
        is_observe_distance_to_boundaries: bool = True,  # Whether to observe points on lanelet boundaries or observe the distance to labelet boundaries
        is_observe_distance_to_center_line: bool = True,  # Whether to observe the distance to reference path
        is_observe_vertices: bool = True,  # Whether to observe the vertices of other agents (or center point)
        is_obs_noise: bool = True,  # Whether to add noise to observations
        obs_noise_level: float = 0.05,  # Defines the variance of the normal distribution modeling the noise.
        is_observe_ref_path_other_agents: bool = False,  # Whether to observe the reference paths of other agents
        is_use_mtv_distance: bool = True,  # Whether to use mtv-based (Minimum Translation Vector) distance or c2c-based (center-to-center) distance.
        # Visu
        is_visualize_short_term_path: bool = True,  # Whether to visualize short-term reference paths
        is_visualize_lane_boundary: bool = False,  # Whether to visualize lane boundary
        is_real_time_rendering: bool = False,  # Simulation will be paused at each time step for a certain duration to enable real-time rendering
        is_visualize_extra_info: bool = True,  # Whether to render extra information such time and time step
        render_title: str = "",  # The title to be rendered
        # Save/Load
        is_save_intermediate_model: bool = True,  # Whether to save intermediate model (also called checkpoint) with the hightest episode reward
        is_load_model: bool = False,  # Whether to load saved model
        is_load_final_model: bool = False,  # Whether to load the final model (last iteration)
        model_name: str = None,
        where_to_save: str = "outputs/",  # Define where to save files such as intermediate models
        is_continue_train: bool = False,  # Whether to continue training after loading an offline model
        is_save_eval_results: bool = True,  # Whether to save evaluation results such as figures and evaluation outputs
        is_load_out_td: bool = False,  # Whether to load evaluation outputs
        is_testing_mode: bool = False,  # In testing mode, collisions do not terminate the current simulation
        is_save_simulation_video: bool = False,  # Whether to save simulation videos
        is_using_opponent_modeling: bool = False,  # Whether to use opponent modeling to predict the actions of other agents
        is_using_prioritized_marl: bool = False,  # Whether to use prioritized MARL and action propagation.
        prioritization_method: str = "marl",  # Which method to use for generating priority ranks (options: {"marl", "random"}). Applicable only for prioritized MARL scenarios.
        is_communication_noise: str = False,  # Whether to inject communication noise to propagated actions
        communication_noise_level: float = 0.1,  # Defines the variance of the normal distribution modeling the noise.
        is_using_cbf: bool = False,  # Whether to use Control Barrier Function (CBF)
        is_using_centralized_cbf: bool = False,  # Whether to use centralized solving for CBF-constrained MARL
        experiment_type: str = "simulation",  # One of {"simulation", "lab"}. If you only use simulation, you do not need to worry about "lab".
        is_obs_steering: bool = False,  # Whether to observe the steering angle of other agents
        ref_path_idx: int = None,  # Specify the index of the predefined reference paths that should be used when initializing/resetting agents. Leave empty to use a randomly selected ones.
        random_seed: int = 0,  # Random seed,
        is_using_pseudo_distance: bool = False,  # Whether to use pseudo distance
        n_circles_approximate_vehicle: int = 3,  # Number of circles to approximate the vehicle shape
        lane_width=0.3,  # For custom scenarios only
        reset_agent_fixed_duration: int = 0,  # Reset agents after fixed duration in seconds. Set to 0 if not used.
    ):

        self.n_agents = n_agents
        self.dt = dt

        self.device = device
        self.scenario_name = scenario_name

        # Sampling
        self.n_iters = n_iters

        # Training
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.lr_min = lr_min
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.max_steps = max_steps

        self.scenario_type = scenario_type

        self.num_vmas_envs = num_vmas_envs

        self.is_save_intermediate_model = is_save_intermediate_model
        self.is_load_model = is_load_model
        self.is_load_final_model = is_load_final_model

        self.episode_reward_mean_current = episode_reward_mean_current
        self.episode_reward_intermediate = episode_reward_intermediate
        self.where_to_save = where_to_save
        self.is_continue_train = is_continue_train

        self.n_points_short_term = n_points_short_term
        # Observation
        self.is_partial_observation = is_partial_observation
        self.n_steps_stored = n_steps_stored
        self.n_nearing_agents_observed = n_nearing_agents_observed
        self.is_observe_distance_to_agents = is_observe_distance_to_agents

        self.is_testing_mode = is_testing_mode
        self.is_save_simulation_video = is_save_simulation_video
        self.is_visualize_short_term_path = is_visualize_short_term_path
        self.is_visualize_lane_boundary = is_visualize_lane_boundary

        self.is_ego_view = is_ego_view
        self.is_apply_mask = is_apply_mask
        self.is_use_mtv_distance = is_use_mtv_distance
        self.is_observe_distance_to_boundaries = is_observe_distance_to_boundaries
        self.is_observe_distance_to_center_line = is_observe_distance_to_center_line
        self.is_observe_vertices = is_observe_vertices
        self.is_obs_noise = is_obs_noise
        self.obs_noise_level = obs_noise_level
        self.is_observe_ref_path_other_agents = is_observe_ref_path_other_agents

        self.is_save_eval_results = is_save_eval_results
        self.is_load_out_td = is_load_out_td

        self.is_real_time_rendering = is_real_time_rendering
        self.is_visualize_extra_info = is_visualize_extra_info
        self.render_title = render_title

        self.is_prb = is_prb
        self.is_challenging_initial_state_buffer = is_challenging_initial_state_buffer

        self.cpm_scenario_probabilities = cpm_scenario_probabilities

        self.is_using_opponent_modeling = is_using_opponent_modeling
        self.is_using_prioritized_marl = is_using_prioritized_marl

        self.prioritization_method = prioritization_method
        self.is_communication_noise = is_communication_noise
        self.communication_noise_level = communication_noise_level

        self.is_using_cbf = is_using_cbf
        self.is_using_centralized_cbf = is_using_centralized_cbf

        self.experiment_type = experiment_type
        self.is_obs_steering = is_obs_steering

        self.ref_path_idx = ref_path_idx

        self.random_seed = random_seed

        self.is_using_pseudo_distance = is_using_pseudo_distance
        self.n_circles_approximate_vehicle = n_circles_approximate_vehicle

        self.lane_width = lane_width

        self.reset_agent_fixed_duration = reset_agent_fixed_duration

        if (model_name is None) and (scenario_name is not None):
            self.model_name = get_model_name(self)

    @property
    def frames_per_batch(self):
        """
        Number of team frames collected per training iteration
            num_envs = frames_per_batch / max_steps
            total_frames = frames_per_batch * n_iters
            sub_batch_size = frames_per_batch // minibatch_size
        """
        return self.num_vmas_envs * self.max_steps

    @property
    def total_frames(self):
        """
        Total number of frames collected for training
        """
        return self.frames_per_batch * self.n_iters

    def to_dict(self):
        # Create a dictionary representation of the instance
        return self.__dict__

    @classmethod
    def from_dict(cls, dict_data):
        # Create an instance of the class from a dictionary
        return cls(**dict_data)

    @classmethod
    def from_json(cls, config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
            return cls(**config)


class SaveData:
    def __init__(self, parameters: Parameters, episode_reward_mean_list: [] = None):
        self.parameters = parameters
        self.episode_reward_mean_list = episode_reward_mean_list

    def to_dict(self):
        return {
            "parameters": self.parameters.to_dict(),  # Convert Parameters instance to dict
            "episode_reward_mean_list": self.episode_reward_mean_list,
        }

    @classmethod
    def from_dict(cls, dict_data):
        parameters = Parameters.from_dict(
            dict_data["parameters"]
        )  # Convert dict back to Parameters instance
        return cls(parameters, dict_data["episode_reward_mean_list"])


class DecisionMakingModule:
    def __init__(self, env: TransformedEnvCustom = None, mappo: bool = True):
        """
        Initializes the DecisionMakingModule, which is responsible for generating decisions using a neural network policy.
        It also sets up a PPO loss module with an actor-critic architecture and GAE (Generalized Advantage Estimation) for RL optimization.

        Parameters:
        -----------
        env : TransformedEnvCustom
            The environment containing the observation specifications and other scenario parameters.
        mappo : bool, optional
            Flag to indicate whether to use centralised learning in the critic (MAPPO). Default is True.
        """
        parameters = env.scenario.parameters
        observation_key = get_observation_key(parameters)

        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec[observation_key].shape[
                    -1
                ],  # n_obs_per_agent
                n_agent_outputs=(
                    2 * env.action_spec.shape[-1]
                ),  # 2 * n_actions_per_agents
                n_agents=env.n_agents,
                centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
                share_params=True,  # sharing parameters means that agents will all share the same policy, which will allow them to benefit from each other’s experiences, resulting in faster training. On the other hand, it will make them behaviorally homogenous, as they will share the same model
                device=parameters.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a `loc` and a non-negative `scale``, used as parameters for a normal distribution (mean and standard deviation)
        )

        # print("policy_net:", policy_net, "\n")

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[observation_key],
            out_keys=[
                ("agents", "loc"),
                ("agents", "scale"),
            ],  # represents the parameters of the policy distribution for each agent
        )

        # Use a probabilistic actor allows for exploration
        policy = ProbabilisticActor(
            module=policy_module,
            spec=env.unbatched_action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": env.unbatched_action_spec[env.action_key].space.low,
                "high": env.unbatched_action_spec[env.action_key].space.high,
            },
            return_log_prob=True,
            log_prob_key=(
                "agents",
                "sample_log_prob",
            ),  # log probability favors numerical stability and gradient calculation
        )  # we'll need the log-prob for the PPO loss

        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[observation_key].shape[
                -1
            ],  # Number of observations
            n_agent_outputs=1,  # 1 value per agent
            n_agents=env.n_agents,
            centralised=mappo,  # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
            share_params=True,  # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
            device=parameters.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        # print("critic_net:", critic_net, "\n")

        critic = TensorDictModule(
            module=critic_net,
            in_keys=[
                observation_key
            ],  # Note that the critic in PPO only takes the same inputs (observations) as the actor
            out_keys=[("agents", "state_value")],
        )

        loss_module = ClipPPOLoss(
            actor=policy,
            critic=critic,
            clip_epsilon=parameters.clip_epsilon,
            entropy_coef=parameters.entropy_eps,
            normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
        )

        loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=env.reward_key,
            action=env.action_key,
            sample_log_prob=("agents", "sample_log_prob"),
            value=("agents", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=parameters.gamma, lmbda=parameters.lmbda
        )  # We build GAE
        GAE = loss_module.value_estimator  # Generalized Advantage Estimation

        optim = torch.optim.Adam(loss_module.parameters(), parameters.lr)

        self.policy = policy
        self.critic = critic

        self.loss_module = loss_module
        self.optim = optim
        self.GAE = GAE


class PriorityModule:
    def __init__(self, env: TransformedEnvCustom = None, mappo: bool = True):
        """
        Initializes the PriorityModule, which is responsible for computing the priority rank of agents
        and their scores using a neural network policy. It also sets up a PPO loss module with an actor-critic
        architecture and GAE (Generalized Advantage Estimation) for RL optimization.

        Parameters:
        -----------
        env : TransformedEnvCustom
            The environment containing the observation specifications and other scenario parameters.
        mappo : bool, optional
            Flag to indicate whether to use centralised learning in the critic (MAPPO). Default is True.
        """

        self.env = env
        self.parameters = self.env.scenario.parameters

        # Tuple containing the prefix keys relevant to the priority variables
        self.prefix_key = ("agents", "info", "priority")

        observation_key = get_priority_observation_key()

        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec[observation_key].shape[-1],
                n_agent_outputs=2 * 1,  # 2 * n_actions_per_agents
                n_agents=self.parameters.n_agents,
                centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
                share_params=True,
                device=self.parameters.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[observation_key],
            out_keys=[self.prefix_key + ("loc",), self.prefix_key + ("scale",)],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=Unbounded(),
            in_keys=[self.prefix_key + ("loc",), self.prefix_key + ("scale",)],
            out_keys=[self.prefix_key + ("scores",)],
            distribution_class=TanhNormal,
            distribution_kwargs={},
            return_log_prob=True,
            log_prob_key=self.prefix_key + ("sample_log_prob",),
        )  # we'll need the log-prob for the PPO loss

        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[observation_key].shape[
                -1
            ],  # Number of observations
            n_agent_outputs=1,  # 1 value per agent
            n_agents=self.parameters.n_agents,
            centralised=mappo,  # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
            share_params=True,  # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
            device=self.parameters.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        critic = TensorDictModule(
            module=critic_net,
            in_keys=[
                observation_key
            ],  # Note that the critic in PPO only takes the same inputs (observations) as the actor
            out_keys=[self.prefix_key + ("state_value",)],
        )

        self.policy = policy
        self.critic = critic

        if self.parameters.prioritization_method.lower() == "marl":
            loss_module = ClipPPOLoss(
                actor=policy,
                critic=critic,
                clip_epsilon=self.parameters.clip_epsilon,
                entropy_coef=self.parameters.entropy_eps,
                normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
            )

            # Comment out advantage and value_target keys to use the same advantage for both base and priority loss modules
            loss_module.set_keys(  # We have to tell the loss where to find the keys
                reward=env.reward_key,
                action=self.prefix_key + ("scores",),
                sample_log_prob=self.prefix_key + ("sample_log_prob",),
                value=self.prefix_key + ("state_value",),
                # These last 2 keys will be expanded to match the reward shape
                done=("agents", "done"),
                terminated=("agents", "terminated"),
                # advantage=self.prefix_key + ("advantage",),
                # value_target=self.prefix_key + ("value_target",),
            )

            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.parameters.gamma,
                lmbda=self.parameters.lmbda,
            )  # We build GAE
            GAE = loss_module.value_estimator  # Generalized Advantage Estimation

            optim = torch.optim.Adam(loss_module.parameters(), self.parameters.lr)

            self.GAE = GAE
            self.loss_module = loss_module
            self.optim = optim

    def rank_agents(self, scores):
        """
        Ranks agents based on their priority scores.

        The method returns the indices of agents in descending order based on their scores.

        Parameters:
        -----------
        scores : Tensor
            A tensor containing the priority scores for each agent.

        Returns:
        --------
        ordered_indices : Tensor
            A tensor containing the indices of agents ordered by their priority scores in descending order.
        """
        # Remove the last dimension of size 1
        scores = scores.squeeze(-1)

        # Get the indices that would sort the tensor along the last dimension in descending order
        ordered_indices = torch.argsort(scores, dim=-1, descending=True)

        return ordered_indices

    def __call__(self, tensordict):
        """
        Computes the priority rank of agents based on their scores and updates the tensordict.

        The method calls the priority actor to generate scores for each agent, ranks the agents
        based on those scores, and then updates the tensordict with the priority rank.

        Parameters:
        -----------
        tensordict : TensorDict
            A dictionary-like object containing the data for the agents.

        Returns:
        --------
        tensordict : TensorDict
            The updated tensordict with the priority rank of agents added.
        """

        if self.parameters.prioritization_method.lower() == "random":
            n_envs = self.parameters.num_vmas_envs
            n_agents = self.parameters.n_agents
            priority_rank = torch.stack(
                [torch.randperm(n_agents) for _ in range(n_envs)]
            )
        else:
            # Call the priority actor and extract the scores key from the resulting tensordict
            scores = self.policy(tensordict)[self.prefix_key + ("scores",)]

            # Generate the priority rank of agents
            priority_rank = self.rank_agents(scores)

        tensordict[self.prefix_key + ("rank",)] = priority_rank

        # Return the tensordict with the priority rank included
        return tensordict

    def compute_losses_and_optimize(self, data):
        """
        Computes the PPO loss (actor and critic losses) and performs backpropagation with gradient clipping.

        This method computes the combined loss (objective, critic, and entropy losses) from the loss module,
        checks for invalid gradients (NaN or infinite values), performs backpropagation, applies gradient clipping,
        and then steps the optimizer to update the model parameters.

        Parameters:
        -----------
        data : TensorDict
            A dictionary-like object containing the data for computing the losses.

        Returns:
        --------
        None
        """

        loss_vals = self.loss_module(data)

        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        assert not loss_value.isnan().any()
        assert not loss_value.isinf().any()

        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), self.parameters.max_grad_norm
        )  # Optional

        self.optim.step()
        self.optim.zero_grad()


##################################################
## Helper Functions
##################################################
def get_path_to_save_model(parameters: Parameters):
    parameters.model_name = get_model_name(parameters=parameters)

    PATH_POLICY = os.path.join(
        parameters.where_to_save, parameters.model_name + "_policy.pth"
    )
    PATH_CRITIC = os.path.join(
        parameters.where_to_save, parameters.model_name + "_critic.pth"
    )
    PATH_FIG = os.path.join(
        parameters.where_to_save, parameters.model_name + "_training_process.jpg"
    )
    PATH_JSON = os.path.join(
        parameters.where_to_save, parameters.model_name + "_data.json"
    )

    if (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    ):
        PATH_PRIORITY_POLICY = os.path.join(
            parameters.where_to_save, parameters.model_name + "_priority_policy.pth"
        )
        PATH_PRIORITY_CRITIC = os.path.join(
            parameters.where_to_save, parameters.model_name + "_priority_critic.pth"
        )

        return (
            PATH_POLICY,
            PATH_CRITIC,
            PATH_PRIORITY_POLICY,
            PATH_PRIORITY_CRITIC,
            PATH_FIG,
            PATH_JSON,
        )

    return (
        PATH_POLICY,
        PATH_CRITIC,
        PATH_FIG,
        PATH_JSON,
    )


def delete_files_with_lower_mean_reward(parameters: Parameters):
    # Regular expression pattern to match and capture the float number
    pattern = r"reward(-?[0-9]*\.?[0-9]+)_"

    # Iterate over files in the directory
    for file_name in os.listdir(parameters.where_to_save):
        match = re.search(pattern, file_name)
        if match:
            # Get the achieved mean episode reward of the saved model
            episode_reward_mean = float(match.group(1))
            if episode_reward_mean < parameters.episode_reward_intermediate:
                # Delete the saved model if its performance is worse
                os.remove(os.path.join(parameters.where_to_save, file_name))


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


def save(
    parameters: Parameters,
    save_data: SaveData,
    policy=None,
    critic=None,
    priority_policy=None,
    priority_critic=None,
):
    # Get paths
    paths = get_path_to_save_model(parameters=parameters)

    if (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    ):
        (
            PATH_POLICY,
            PATH_CRITIC,
            PATH_PRIORITY_POLICY,
            PATH_PRIORITY_CRITIC,
            PATH_FIG,
            PATH_JSON,
        ) = paths
    else:
        PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON = paths

    # Save parameters and mean episode reward list
    json_object = json.dumps(save_data.to_dict(), indent=4)  # Serializing json
    with open(PATH_JSON, "w") as outfile:  # Writing to sample.json
        outfile.write(json_object)
    # Example to how to open the saved json file
    # with open('save_data.json', 'r') as file:
    #     data = json.load(file)
    #     loaded_parameters = Parameters.from_dict(data)

    # Save figure
    plt.clf()  # Clear the current figure to avoid drawing on the same figure in the next iteration
    plt.plot(save_data.episode_reward_mean_list)
    plt.xlabel("Iterations")
    plt.ylabel("Episode mean reward")
    plt.tight_layout()  # Set the layout to be tight to minimize white space !!! deprecated
    plt.savefig(PATH_FIG, format="jpg", dpi=300, bbox_inches="tight")

    # Save models
    if (policy != None) & (critic != None):
        # Save current models
        torch.save(policy.state_dict(), PATH_POLICY)
        torch.save(critic.state_dict(), PATH_CRITIC)

        if (
            parameters.is_using_prioritized_marl
            and parameters.prioritization_method.lower() == "marl"
        ):
            if (priority_policy != None) & (priority_critic != None):
                # Save current models
                torch.save(priority_policy.state_dict(), PATH_PRIORITY_POLICY)
                torch.save(priority_critic.state_dict(), PATH_PRIORITY_CRITIC)

        # Delete files with lower mean episode reward
        delete_files_with_lower_mean_reward(parameters=parameters)

    print(f"Saved model: {parameters.episode_reward_mean_current:.2f}.")


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


def opponent_modeling(
    tensordict,
    policy,
    n_nearing_agents_observed,
    nearing_agents_indices,
    noise_percentage: float = 0,
):
    """
    This function implements opponent modeling, inspired by [1].
    Each ego agent uses its own policy to predict the tentative actions of its surrounding agents, aiming to mitigate the non-stationarity problem.
    The ego agent appends these tentative actions to its observation stored in the input tensordict.

    Reference
        [1] Raileanu, Roberta, et al. "Modeling others using oneself in multi-agent reinforcement learning." International conference on machine learning. PMLR, 2018.
    """
    time_rl_start = time.time()

    policy(tensordict)  # Run the policy to get tentative actions
    # Infer parameters
    n_agents = tensordict["agents"]["action"].shape[1]
    n_actions = tensordict["agents"]["action"].shape[2]

    batch_dim = tensordict.batch_size[0]
    device = tensordict.device

    actions_tentative = tensordict["agents"]["action"]

    if noise_percentage != 0:
        # Model inaccuracy to opponent modeling

        # A certain percentage of the maximum value as the noise standard diviation
        noise_std_speed = AGENTS["max_speed"] * noise_percentage
        noise_std_steering = AGENTS["max_steering"] * noise_percentage

        noise_actions = torch.cat(
            [
                torch.randn([batch_dim, n_agents, 1], device=actions_tentative.device)
                * noise_std_speed,
                torch.randn([batch_dim, n_agents, 1], device=actions_tentative.device)
                * noise_std_steering,
            ],
            dim=-1,
        )

        actions_tentative[:] += noise_actions

    for ego_agent in range(n_agents):
        for j in range(n_nearing_agents_observed):
            sur_agent = nearing_agents_indices[:, ego_agent, j]

            batch_indices = torch.arange(batch_dim, device=device, dtype=torch.int32)
            action_tentative_sur_agent = actions_tentative[batch_indices, sur_agent]

            # Update observation with tentative actions
            idx_action_start = (
                -(n_nearing_agents_observed - j) * 2
            )  # Start index of the action of surrounding agents in the observation (actions are appended at the end of the observation)
            idx_action_end = (
                idx_action_start + n_actions
            )  # End index of the action of surrounding agents in the observation (actions are appended at the end of the observation)
            if idx_action_end == 0:
                idx_action_end = None  # Avoid slicing with zero

            # Insert the tentative actions of the surrounding agents into each ago agent's observation
            tensordict["agents"]["observation"][
                :, ego_agent, idx_action_start:idx_action_end
            ] = action_tentative_sur_agent

    policy(tensordict)  # Run the policy to update actions
    time_rl = time.time() - time_rl_start

    return time_rl


def get_observation_key(parameters):
    return (
        ("agents", "observation")
        if not parameters.is_using_prioritized_marl
        else ("agents", "info", "base_observation")
    )


def get_priority_observation_key():
    return ("agents", "info", "priority_observation")


def get_cbf_observation_key():
    """The observation key of CBF"""
    return ("agents", "info", "cbf_observation")


def prioritized_ap_policy(
    tensordict,
    policy,
    priority_module,
    nearing_agents_indices,
    is_communication_noise,
    communication_noise_level,
):
    """
    Implements prioritized action propagation (AP) for multiple agents.
    The function first generates a priority rank using the provided priority module.
    Then, agents are processed in this priority order, where each agent computes its action
    and propagates it to lower-priority agents as part of their observation.

    Since the policy call generates actions for all agents in all environments at once,
    the function uses a mask to isolate the actions of the agent whose turn it is to act.
    These actions are progressively combined to form the full action tensor.

    Parameters:
    -----------
    tensordict : TensorDict
        A dictionary-like object that stores the data for all agents and environments.
    policy : Callable
        The policy function used to compute the actions for the agents.
    priority_module : Callable
        A module that computes the priority rank for agents by wrapping the process
        of generating priority scores and ranking agents according to these scores.
    nearing_agents_indices : Tensor
        A tensor indicating the neighboring agents for each agent in each environment.

    Returns:
    --------
    tensordict : TensorDict
        The updated tensordict with combined actions and observations after prioritized action propagation.
    """
    base_observation_key = ("agents", "info", "base_observation")

    device = tensordict.device

    # Clone original observation
    original_obs = tensordict[base_observation_key].clone()

    # Infer parameters
    n_envs, n_agents, obs_dim, action_dim = (
        original_obs.shape[0],
        original_obs.shape[1],
        original_obs.shape[2],
        AGENTS["n_actions"],
    )

    # Generate priority rank using the priority module
    priority_module(tensordict)
    # Extract priority rank (shape: (n_envs, n_agents)) from tensordict
    priority_rank = tensordict[priority_module.prefix_key + ("rank",)]

    # Temporary tensors to store intermediate observations and combined results
    temp_obs = torch.zeros(n_envs, n_agents, obs_dim)
    combined_action = torch.zeros(n_envs, n_agents, action_dim)
    combined_loc = torch.zeros(n_envs, n_agents, action_dim)
    combined_sample_log_prob = torch.zeros(n_envs, n_agents)
    combined_scale = torch.zeros(n_envs, n_agents, action_dim)
    combined_obs = torch.zeros(n_envs, n_agents, obs_dim)

    # Observable high-priority agents
    observable_higher_priority_agents = [
        [[] for _ in range(n_agents)] for _ in range(n_envs)
    ]

    time_rl = 0

    # Loop through each step in the priority rank
    for turn in range(n_agents):
        # Reset the observation
        tensordict[("agents", "info")].set("base_observation", original_obs)

        # Get the list of agents for the current turn based on priority
        current_turn_agents = priority_rank[:, turn]

        # Create environment indices (from 0 to n_envs - 1)
        envs = torch.arange(n_envs)

        # Create a mask indicating which agents are acting in each environment
        mask = torch.zeros(n_envs, n_agents, dtype=torch.bool)
        mask[envs, current_turn_agents] = True

        # Prepare input for the policy by modifying the observations
        for env in range(n_envs):
            agent_idx = current_turn_agents[env]
            obs = tensordict[base_observation_key][env, agent_idx].clone()

            # Get the indices of neighboring agents
            current_turn_agent_neighbors = nearing_agents_indices[env, agent_idx].to(
                torch.int64
            )

            # Collect actions of the current agent's neighbors
            actions_so_far = combined_action[env, current_turn_agent_neighbors].view(-1)

            # Set of higher-priority agents
            higher_priority_agents = priority_rank[env, :turn]

            # Observable high-priority agents
            are_observable_higher_priority_agents = torch.isin(
                current_turn_agent_neighbors, higher_priority_agents
            )
            observable_higher_priority_agents[env][
                agent_idx
            ] = current_turn_agent_neighbors[are_observable_higher_priority_agents]

            # Propagate the collected actions into the current agent's observation
            if len(actions_so_far) > 0:
                if is_communication_noise:
                    std_noise_speed = AGENTS["max_speed"] * communication_noise_level
                    std_noise_steering = (
                        AGENTS["max_steering"] * communication_noise_level
                    )
                    noise = torch.cat(
                        [
                            std_noise_speed * torch.randn(action_dim, device=device),
                            std_noise_steering * torch.randn(action_dim, device=device),
                        ],
                        dim=-1,
                    )
                    obs[-len(actions_so_far) :] = actions_so_far + noise
                else:
                    obs[-len(actions_so_far) :] = actions_so_far

            # Store the updated observation for the current agent in temp_obs
            temp_obs[env, agent_idx] = obs

        # Update the base observation with temp_obs for policy execution
        tensordict[("agents", "info")].set("base_observation", temp_obs)

        # Call the policy to generate actions for the agents
        time_rl_start = time.time()
        policy(tensordict)
        time_rl += time.time() - time_rl_start

        # Use the mask to place the data into the correct positions in the combined tensors
        combined_action[mask] = tensordict[("agents", "action")][mask]
        combined_loc[mask] = tensordict[("agents", "loc")][mask]
        combined_sample_log_prob[mask] = tensordict[("agents", "sample_log_prob")][mask]
        combined_scale[mask] = tensordict[("agents", "scale")][mask]
        combined_obs[mask] = tensordict[base_observation_key][mask]

    # Write the combined actions back to the tensordict
    tensordict[("agents", "action")] = combined_action
    tensordict[("agents", "loc")] = combined_loc
    tensordict[("agents", "sample_log_prob")] = combined_sample_log_prob
    tensordict[("agents", "scale")] = combined_scale
    tensordict[base_observation_key] = combined_obs

    return time_rl, tensordict, priority_rank, observable_higher_priority_agents


def cbf_constrained_decentralized_policy(
    tensordict,
    policy,
    cbf_controllers,
):
    """

    This function is used to make action decisions for multiple agents at each time step,
    based on their priority, and enforces safety constraints via Control Barrier Functions (CBFs).
    Each agent makes decisions independently based on its observations
    and the actions of its neighbors.

    Parameters:
    ----------
    tensordict : TensorDict
        A dictionary-like object that stores the data for all agents and environments.
    policy : Callable
        The policy function used to compute the actions for the agents.
    priority_module : Callable
        A module that computes the priority rank for agents by wrapping the process
        of generating priority scores and ranking agents according to these scores.
    cbf_controllers : List[List[CBFController]]
        A list of CBF-based controllers for each environment and agent, used to
        correct the unsafe policy's actions.
    nearing_agents_indices : Tensor
        A tensor indicating the neighboring agents for each agent in each environment.

    Returns:
    ----------
    tensordict : TensorDict
        The updated tensordict with corrected safe actions with CBF.
    """
    base_observation_key = ("agents", "info", "base_observation")

    # Clone original observation
    original_obs = tensordict[base_observation_key].clone()

    # Infer parameters
    n_envs, n_agents, obs_dim, action_dim = (
        original_obs.shape[0],
        original_obs.shape[1],
        original_obs.shape[2],
        AGENTS["n_actions"],
    )

    time_cbf = 0
    time_rl = 0
    time_pseudo_dis = 0

    # Call the policy to generate actions for the agents
    time_rl_start = time.time()
    policy(tensordict)
    time_rl += time.time() - time_rl_start  # Time for RL policy

    if "nominal_action" not in tensordict[("agents", "info")].keys():
        nominal_action = torch.zeros(
            n_envs,
            n_agents,
            action_dim,
            dtype=torch.float32,
            device=tensordict.device,
        )
        tensordict[("agents", "info")].set("nominal_action", nominal_action)

    # CBF-QP solving
    time_cbf_start = time.time()
    for i in range(n_envs):
        for j in range(n_agents):
            cbf_controllers[i][j].solve_decentralized_cbf_qp(tensordict)
            time_pseudo_dis += cbf_controllers[i][j].time_pseudo_dis
    time_cbf += time.time() - time_cbf_start

    return (
        time_rl,
        time_cbf,
        time_pseudo_dis,
        tensordict,
    )


def cbf_constrained_decentralized_policy_multi_agents(
    tensordict,
    policy,
    priority_module,
    cbf_controllers,
    nearing_agents_indices,
    is_communication_noise,
    communication_noise_level,
):
    """

    This function is used to make action decisions for multiple agents at each time step,
    based on their priority, and enforces safety constraints via Control Barrier Functions (CBFs).
    Each agent makes decisions independently based on its observations
    and the actions of its neighbors.

    Parameters:
    ----------
    tensordict : TensorDict
        A dictionary-like object that stores the data for all agents and environments.
    policy : Callable
        The policy function used to compute the actions for the agents.
    priority_module : Callable
        A module that computes the priority rank for agents by wrapping the process
        of generating priority scores and ranking agents according to these scores.
    cbf_controllers : List[List[CBFController]]
        A list of CBF-based controllers for each environment and agent, used to
        correct the unsafe policy's actions.
    nearing_agents_indices : Tensor
        A tensor indicating the neighboring agents for each agent in each environment.

    Returns:
    ----------
    tensordict : TensorDict
        The updated tensordict with corrected safe actions with CBF.
    """
    base_observation_key = ("agents", "info", "base_observation")
    cbf_observation_key = ("agents", "info", "cbf_observation")

    device = tensordict.device

    # Clone original observation
    original_obs = tensordict[base_observation_key].clone()
    original_cbf_obs = tensordict[cbf_observation_key].clone()

    # Infer parameters
    n_envs, n_agents, obs_dim, action_dim = (
        original_obs.shape[0],
        original_obs.shape[1],
        original_obs.shape[2],
        AGENTS["n_actions"],
    )

    cbf_obs_dim = original_cbf_obs.shape[2]
    n_nearing_agents = nearing_agents_indices.shape[-1]
    # Generate priority rank using the priority module
    priority_module(tensordict)
    # Extract priority rank (shape: (n_envs, n_agents)) from tensordict
    priority_rank = tensordict[priority_module.prefix_key + ("rank",)]

    # Temporary tensors to store intermediate observations and combined results
    temp_obs = torch.zeros(n_envs, n_agents, obs_dim)
    combined_action = torch.zeros(n_envs, n_agents, action_dim)
    combined_loc = torch.zeros(n_envs, n_agents, action_dim)
    combined_sample_log_prob = torch.zeros(n_envs, n_agents)
    combined_scale = torch.zeros(n_envs, n_agents, action_dim)
    combined_obs = torch.zeros(n_envs, n_agents, obs_dim)

    # Observable high-priority agents
    observable_higher_priority_agents = [
        [[] for _ in range(n_agents)] for _ in range(n_envs)
    ]

    time_cbf = 0
    time_rl = 0
    time_pseudo_dis = 0

    # Loop through each step in the priority rank
    for turn in range(n_agents):
        # Reset the observation
        tensordict[("agents", "info")].set("base_observation", original_obs)
        tensordict[("agents", "info")].set("cbf_observation", original_cbf_obs)

        # Get the list of agents for the current turn based on priority
        current_turn_agents = priority_rank[:, turn]

        # Create environment indices (from 0 to n_envs - 1)
        envs = torch.arange(n_envs)

        # Create a mask indicating which agents are acting in each environment
        mask = torch.zeros(n_envs, n_agents, dtype=torch.bool)
        mask[envs, current_turn_agents] = True

        # Create a mask indication which higher priority agents are nearing the currrent agents
        mask_higher_priority_agents = torch.zeros(
            n_envs, n_agents, n_nearing_agents, dtype=torch.bool
        )
        temp_cbf_obs = torch.zeros(n_envs, n_agents, cbf_obs_dim)

        # Prepare input for the policy by modifying the observations
        for env in range(n_envs):
            agent_idx = current_turn_agents[env]
            obs = tensordict[base_observation_key][env, agent_idx].clone()
            cbf_obs = tensordict[cbf_observation_key][env, agent_idx].clone()

            # Get the indices of neighboring agents
            current_turn_agent_neighbors = nearing_agents_indices[env, agent_idx].to(
                torch.int64
            )

            # Collect actions of the current agent's neighbors
            actions_so_far = combined_action[env, current_turn_agent_neighbors].view(-1)

            # Set of higher-priority agents
            higher_priority_agents = priority_rank[env, :turn]

            # Observable high-priority agents
            are_observable_higher_priority_agents = torch.isin(
                current_turn_agent_neighbors, higher_priority_agents
            )
            observable_higher_priority_agents[env][
                agent_idx
            ] = current_turn_agent_neighbors[are_observable_higher_priority_agents]

            mask_higher_priority_agents[
                env, agent_idx, :
            ] = are_observable_higher_priority_agents

            # Propagate the collected actions into the current agent's observation
            if len(actions_so_far) > 0:
                if is_communication_noise:
                    std_noise_speed = AGENTS["max_speed"] * communication_noise_level
                    std_noise_steering = (
                        AGENTS["max_steering"] * communication_noise_level
                    )
                    noise = torch.cat(
                        [
                            std_noise_speed * torch.randn(action_dim, device=device),
                            std_noise_steering * torch.randn(action_dim, device=device),
                        ],
                        dim=-1,
                    )
                    obs[-len(actions_so_far) :] = actions_so_far + noise
                    cbf_obs[-len(actions_so_far) :] = actions_so_far + noise
                else:
                    obs[-len(actions_so_far) :] = actions_so_far
                    cbf_obs[-len(actions_so_far) :] = actions_so_far

            # Store the updated observation for the current agent in temp_obs
            temp_obs[env, agent_idx] = obs
            temp_cbf_obs[env, agent_idx] = cbf_obs

        # Update the base_observation with temp_obs for policy execution
        tensordict[("agents", "info")].set("base_observation", temp_obs)
        # Update the cbf_observation with temp_cbf_obs for policy execution
        tensordict[("agents", "info")].set("cbf_observation", temp_cbf_obs)

        # Call the policy to generate actions for the agents
        time_rl_start = time.time()
        policy(tensordict)
        time_rl += time.time() - time_rl_start  # Time for RL policy

        if "nominal_action" not in tensordict[("agents", "info")].keys():
            nominal_action = torch.zeros(
                n_envs,
                n_agents,
                action_dim,
                dtype=torch.float32,
                device=tensordict.device,
            )
            tensordict[("agents", "info")].set("nominal_action", nominal_action)

        # CBF-QP solving
        time_cbf_start = time.time()
        for i in range(n_envs):
            for j in range(n_agents):
                if mask[i, j]:
                    cbf_controllers[i][j].solve_decentralized_cbf_qp(
                        tensordict, mask_higher_priority_agents
                    )
                    time_pseudo_dis += cbf_controllers[i][j].time_pseudo_dis
        time_cbf += time.time() - time_cbf_start

        # Use the mask to place the data into the correct positions in the combined tensors
        combined_action[mask] = tensordict[("agents", "action")][mask]
        combined_loc[mask] = tensordict[("agents", "loc")][mask]
        combined_sample_log_prob[mask] = tensordict[("agents", "sample_log_prob")][mask]
        combined_scale[mask] = tensordict[("agents", "scale")][mask]
        combined_obs[mask] = tensordict[base_observation_key][mask]

    tensordict[("agents", "action")] = combined_action
    tensordict[("agents", "loc")] = combined_loc
    tensordict[("agents", "sample_log_prob")] = combined_sample_log_prob
    tensordict[("agents", "scale")] = combined_scale
    tensordict[base_observation_key] = combined_obs
    tensordict[cbf_observation_key] = original_cbf_obs

    return (
        time_rl,
        time_cbf,
        time_pseudo_dis,
        tensordict,
        priority_rank,
        observable_higher_priority_agents,
    )


def cbf_constrained_centralized_policy(
    tensordict,
    policy,
    cbf_controllers,
):
    try:
        raise NotImplementedError(
            "Function cbf_constrained_centralized_policy is not yet implemented!"
        )
    except NotImplementedError as e:
        print(f"Error: {e}")
        sys.exit("Exiting program: Function not implemented.")

    return time_rl, tensordict


def cbf_constrained_centralized_policy(
    tensordict,
    policy,
    cbf_controllers,
):
    try:
        raise NotImplementedError(
            "Function cbf_constrained_centralized_policy is not yet implemented!"
        )
    except NotImplementedError as e:
        print(f"Error: {e}")
        sys.exit("Exiting program: Function not implemented.")

    return time_rl, tensordict
