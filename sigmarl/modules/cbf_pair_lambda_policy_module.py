# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping
from typing import Any

import torch
from tensordict.nn import NormalParamExtractor, TensorDictModuleBase
from torch.distributions import Distribution
from torchrl.envs.utils import ExplorationType, exploration_type, set_exploration_type
from torchrl.modules import TanhNormal
from torchrl.modules.models import MLP

from sigmarl.modules.motion_policy_module import MotionPolicyModule
from sigmarl.scenarios.observations.observation_provider_rt import (
    neighbor_observation_layout,
)

PAIR_LAMBDA_KEY = ("agents", "cbf", "pair_lambda")
PAIR_LAMBDA_LOC_KEY = ("agents", "cbf", "pair_lambda_loc")
PAIR_LAMBDA_SCALE_KEY = ("agents", "cbf", "pair_lambda_scale")
LAMBDA_LOG_PROB_KEY = ("agents", "cbf", "lambda_log_prob")


class MaskedPairLambdaDistribution(Distribution):
    """Per-slot TanhNormal over pair lambdas whose log-prob sums the valid slots.

    Samples have shape ``[..., n_agents, K, 1]``; ``log_prob`` returns
    ``[..., n_agents]`` so that each agent is credited with the lambdas it
    proposed for its observed neighbors. Masked (invalid) slots do not enter
    the CBF-QP and are excluded from the log-prob.
    """

    has_rsample = True

    def __init__(self, base: TanhNormal, valid: torch.Tensor):
        super().__init__(validate_args=False)
        self.base = base
        self.valid = valid

    @property
    def deterministic_sample(self):
        return self.base.deterministic_sample

    def sample(self, sample_shape=torch.Size()):
        return self.base.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base.rsample(sample_shape)

    def log_prob(self, value):
        log_prob = self.base.log_prob(value)
        if log_prob.shape == value.shape:
            log_prob = log_prob.sum(dim=-1)
        return (log_prob * self.valid).sum(dim=-1)

    def entropy(self):
        # TanhNormal has no closed-form entropy; PPO falls back to Monte Carlo
        raise NotImplementedError


class CBFPairLambdaPolicyModule(TensorDictModuleBase):
    """Learned CBF decay rate for each (ego, observed neighbor) pair.

    The ego-view partial observation ends with ``n_nearing_agents_observed``
    neighbor blocks. One MLP, shared across agents and neighbor slots, maps
    the ego's own features concatenated with one neighbor block to a
    TanhNormal over ``lambda`` in ``[lambda_low, lambda_high]``. The CBF-QP
    combines the two directed lambdas of a pair into the pair's decay rate.
    """

    def __init__(self, env, parameters, observation_key):
        super().__init__()
        self.env = env
        self.observation_key = observation_key

        obs_dim = env.observation_spec[observation_key].shape[-1]
        self.n_neighbors = parameters.n_nearing_agents_observed
        self.block_dim, self.has_valid_flag = neighbor_observation_layout(parameters)
        self.self_dim = obs_dim - self.n_neighbors * self.block_dim
        if self.self_dim <= 0:
            raise ValueError(
                f"Observation dimension {obs_dim} is too small for "
                f"{self.n_neighbors} neighbor blocks of size {self.block_dim}."
            )
        print(
            "[INFO] CBFPairLambdaPolicyModule input dimension: "
            f"{self.self_dim} (ego) + {self.block_dim} (neighbor), "
            f"{self.n_neighbors} neighbor slots"
        )

        self.param_net = torch.nn.Sequential(
            MLP(
                in_features=self.self_dim + self.block_dim,
                out_features=2,
                depth=3,
                num_cells=256,
                activation_class=torch.nn.Tanh,
                device=parameters.device,
            ),
            NormalParamExtractor(),
        )

        self.register_buffer(
            "lambda_low",
            torch.tensor(0.1, device=parameters.device),
            persistent=False,
        )
        self.register_buffer(
            "lambda_high",
            torch.tensor(0.99, device=parameters.device),
            persistent=False,
        )

    def split_observation(self, observation: torch.Tensor):
        """Split the observation into per-pair inputs and slot validity.

        Args:
            observation: ``[..., n_agents, obs_dim]``.

        Returns:
            Pair inputs ``[..., n_agents, K, self_dim + block_dim]`` and the
            validity ``[..., n_agents, K]`` (1 for a valid neighbor slot).
        """
        obs_self = observation[..., : self.self_dim]
        blocks = observation[..., self.self_dim :].reshape(
            *observation.shape[:-1], self.n_neighbors, self.block_dim
        )
        obs_self = obs_self.unsqueeze(-2).expand(
            *obs_self.shape[:-1], self.n_neighbors, self.self_dim
        )
        pair_input = torch.cat([obs_self, blocks], dim=-1)
        if self.has_valid_flag:
            valid = (blocks[..., -1] > 0.5).to(observation.dtype)
        else:
            valid = torch.ones_like(blocks[..., -1])
        return pair_input, valid

    def _build_lambda_dist(self, tensordict):
        pair_input, valid = self.split_observation(tensordict.get(self.observation_key))
        lambda_loc, lambda_scale = self.param_net(pair_input)
        base = TanhNormal(
            loc=lambda_loc,
            scale=lambda_scale,
            low=self.lambda_low,
            high=self.lambda_high,
        )
        return MaskedPairLambdaDistribution(base, valid), lambda_loc, lambda_scale

    @property
    def in_keys(self):
        return [self.observation_key]

    @property
    def out_keys(self):
        return [
            PAIR_LAMBDA_LOC_KEY,
            PAIR_LAMBDA_SCALE_KEY,
            PAIR_LAMBDA_KEY,
            LAMBDA_LOG_PROB_KEY,
            ("agents", "sample_log_prob"),
        ]

    @property
    def dist_sample_keys(self):
        return [PAIR_LAMBDA_KEY]

    @property
    def log_prob_keys(self):
        return [LAMBDA_LOG_PROB_KEY]

    def get_dist(self, tensordict):
        return self.build_dist(tensordict)

    def build_dist(self, tensordict):
        lambda_dist, _, _ = self._build_lambda_dist(tensordict)
        return lambda_dist

    def forward(self, tensordict):
        lambda_dist, lambda_loc, lambda_scale = self._build_lambda_dist(tensordict)

        if exploration_type() in {
            ExplorationType.DETERMINISTIC,
            ExplorationType.MODE,
            ExplorationType.MEAN,
            ExplorationType.MEDIAN,
        }:
            lambda_value = lambda_dist.deterministic_sample
        else:
            lambda_value = lambda_dist.rsample()
        lambda_log_prob = lambda_dist.log_prob(lambda_value)

        tensordict.set(PAIR_LAMBDA_LOC_KEY, lambda_loc)
        tensordict.set(PAIR_LAMBDA_SCALE_KEY, lambda_scale)
        tensordict.set(PAIR_LAMBDA_KEY, lambda_value)
        tensordict.set(LAMBDA_LOG_PROB_KEY, lambda_log_prob)
        tensordict.set(("agents", "sample_log_prob"), lambda_log_prob)
        return tensordict

    def set_train_mode(
        self,
        train_action: bool,
        train_lambda: bool,
        allow_action: bool = True,
        allow_lambda: bool = True,
    ):
        final_train_lambda = train_lambda and allow_lambda
        for p in self.parameters():
            p.requires_grad = final_train_lambda

    def load_lambda_policy(self, path: str, map_location=None):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)

    def lambda_policy_state_dict(self):
        return self.state_dict()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        # Lambda bounds come from the current module, not from checkpoints
        state_dict = dict(state_dict)
        state_dict.pop("lambda_low", None)
        state_dict.pop("lambda_high", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class FrozenMotionPairLambdaPolicyModule(TensorDictModuleBase):
    """A frozen pretrained motion policy followed by the trainable pair-lambda head.

    The motion policy always acts deterministically and without gradients,
    so its action is the nominal control the CBF-QP filters. Only the pair
    lambda head is trained.
    """

    def __init__(self, env, parameters, observation_key):
        super().__init__()
        self.env = env

        self.motion_policy = MotionPolicyModule(env, parameters, observation_key)
        self.lambda_policy = CBFPairLambdaPolicyModule(env, parameters, observation_key)
        for p in self.motion_policy.parameters():
            p.requires_grad = False

    @property
    def in_keys(self):
        return self.motion_policy.in_keys

    @property
    def out_keys(self):
        return self.motion_policy.out_keys + self.lambda_policy.out_keys

    @property
    def dist_sample_keys(self):
        return self.lambda_policy.dist_sample_keys

    @property
    def log_prob_keys(self):
        return self.lambda_policy.log_prob_keys

    def forward(self, tensordict):
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            tensordict = self.motion_policy(tensordict)
        return self.lambda_policy(tensordict)

    def get_dist(self, tensordict):
        return self.lambda_policy.build_dist(tensordict)

    def build_dist(self, tensordict):
        return self.lambda_policy.build_dist(tensordict)

    def set_train_mode(
        self,
        train_action: bool,
        train_lambda: bool,
        allow_action: bool = True,
        allow_lambda: bool = True,
    ):
        self.lambda_policy.set_train_mode(
            train_action=False,
            train_lambda=train_lambda,
            allow_action=False,
            allow_lambda=allow_lambda,
        )

    def load_motion_policy(self, path: str, map_location=None):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.motion_policy.load_state_dict(state_dict)

    def load_lambda_policy(self, path: str, map_location=None):
        """Load the pair head from a head-only or a whole-wrapper checkpoint.

        Cross-iteration checkpoints (policy_iter<N>.pth) save the whole wrapper,
        whose head parameters carry the ``lambda_policy.`` prefix.
        """
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        prefix = "lambda_policy."
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        self.lambda_policy.load_state_dict(state_dict)

    def motion_policy_state_dict(self):
        return self.motion_policy.state_dict()

    def lambda_policy_state_dict(self):
        return self.lambda_policy.state_dict()
