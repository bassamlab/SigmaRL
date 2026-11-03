from collections.abc import Mapping
from typing import Any

import torch
from tensordict.nn import TensorDictModule, TensorDictModuleBase, NormalParamExtractor
from torchrl.envs.utils import ExplorationType, exploration_type
from torchrl.modules import MultiAgentMLP, TanhNormal
from torchrl.modules.models import MLP


class IndependentAgentMLP(torch.nn.Module):
    """A motion actor with one MLP and parameter set per agent."""

    def __init__(
        self,
        *,
        n_agents: int,
        n_agent_inputs: int,
        n_agent_outputs: int,
        device,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.agent_networks = torch.nn.ModuleList(
            [
                MLP(
                    in_features=n_agent_inputs,
                    out_features=n_agent_outputs,
                    depth=3,
                    num_cells=256,
                    activation_class=torch.nn.Tanh,
                    device=device,
                )
                for _ in range(n_agents)
            ]
        )

    def forward(self, observation):
        if observation.shape[-2] != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} agent observations, got {observation.shape}."
            )
        outputs = [
            network(observation[..., agent_index, :])
            for agent_index, network in enumerate(self.agent_networks)
        ]
        return torch.stack(outputs, dim=-2)

    def parameters_by_agent(self):
        """Return independent parameter lists in agent-index order."""

        return [list(network.parameters()) for network in self.agent_networks]


class MotionPolicyModule(TensorDictModuleBase):
    PRIORITY_SCORES_KEY = ("agents", "info", "priority", "scores")
    PRIORITY_NEIGHBOR_INDICES_KEY = (
        "agents",
        "info",
        "priority_neighbor_indices",
    )
    PRIORITY_NEIGHBOR_VALID_KEY = ("agents", "info", "priority_neighbor_valid")
    PRIORITY_CONDITIONED_OBSERVATION_KEY = (
        "agents",
        "info",
        "priority_conditioned_motion_observation",
    )

    def __init__(self, env, parameters, observation_key):
        super().__init__()
        self.env = env
        self.observation_key = observation_key
        self.is_priority_conditioned_motion = parameters.is_priority_conditioned_motion
        self.n_nearing_agents_observed = parameters.n_nearing_agents_observed

        obs_dim = env.observation_spec[observation_key].shape[-1]
        policy_observation_key = observation_key
        self._input_keys = [observation_key]
        if self.is_priority_conditioned_motion:
            n_neighbors = parameters.n_nearing_agents_observed
            obs_dim += 1 + 3 * n_neighbors
            policy_observation_key = self.PRIORITY_CONDITIONED_OBSERVATION_KEY
            self._input_keys.extend(
                [
                    self.PRIORITY_SCORES_KEY,
                    self.PRIORITY_NEIGHBOR_INDICES_KEY,
                    self.PRIORITY_NEIGHBOR_VALID_KEY,
                ]
            )
        action_spec = self._get_unbatched_action_spec(env)
        action_dim = action_spec.shape[-1]
        print(f"[INFO] MotionPolicyModule observation dimension: {obs_dim}")
        print(f"[INFO] MotionPolicyModule output dimension: {2 * action_dim}")
        self._uses_independent_actor = parameters.is_multi_timescale_learning
        if parameters.is_multi_timescale_learning:
            actor_module = IndependentAgentMLP(
                n_agents=env.n_agents,
                n_agent_inputs=obs_dim,
                n_agent_outputs=2 * action_dim,
                device=parameters.device,
            )
        else:
            share_params = (
                parameters.is_share_params_actor and not parameters.is_cross_iter_test
            )
            actor_module = MultiAgentMLP(
                n_agent_inputs=obs_dim,
                n_agent_outputs=2 * action_dim,
                n_agents=env.n_agents,
                centralised=False,
                share_params=share_params,
                device=parameters.device,
                depth=3,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            )
        self.param_module = TensorDictModule(
            module=torch.nn.Sequential(
                actor_module,
                NormalParamExtractor(),
            ),
            in_keys=[policy_observation_key],
            out_keys=[
                ("agents", "action_loc"),
                ("agents", "action_scale"),
            ],
        )

        action_low = action_spec.space.low.clone().detach()
        action_high = action_spec.space.high.clone().detach()

        self.register_buffer("action_low", action_low, persistent=False)
        self.register_buffer(
            "action_high",
            action_high,
            persistent=False,
        )

    @staticmethod
    def _get_unbatched_action_spec(env):
        try:
            return env.full_action_spec_unbatched[env.action_key]
        except AttributeError:
            return env.unbatched_action_spec[env.action_key]

    def _build_action_dist(
        self,
        action_loc: torch.Tensor,
        action_scale: torch.Tensor,
    ) -> TanhNormal:
        return TanhNormal(
            loc=action_loc,
            scale=action_scale,
            low=self.action_low,
            high=self.action_high,
        )

    @property
    def in_keys(self):
        return self._input_keys

    @property
    def out_keys(self):
        return [
            ("agents", "action_loc"),
            ("agents", "action_scale"),
            self.env.action_key,
            ("agents", "action_log_prob"),
            ("agents", "sample_log_prob"),
        ]

    @property
    def dist_sample_keys(self):
        return [self.env.action_key]

    @property
    def log_prob_keys(self):
        return [("agents", "action_log_prob")]

    def get_dist(self, tensordict):
        return self.build_dist(tensordict)

    def forward(self, tensordict):
        self._set_policy_observation(tensordict)
        tensordict = self.param_module(tensordict)

        action_loc = tensordict.get(("agents", "action_loc"))
        action_scale = tensordict.get(("agents", "action_scale"))

        action_dist = self._build_action_dist(action_loc, action_scale)

        if exploration_type() in {
            ExplorationType.DETERMINISTIC,
            ExplorationType.MODE,
            ExplorationType.MEAN,
            ExplorationType.MEDIAN,
        }:
            action = action_dist.deterministic_sample
        else:
            action = action_dist.rsample()

        action_log_prob = action_dist.log_prob(action)
        if action_log_prob.shape == action.shape:
            action_log_prob = action_log_prob.sum(dim=-1)

        tensordict.set(self.env.action_key, action)
        tensordict.set(("agents", "action_log_prob"), action_log_prob)
        tensordict.set(("agents", "sample_log_prob"), action_log_prob)

        return tensordict

    def build_dist(self, tensordict):
        td = tensordict.clone(False)
        self._set_policy_observation(td)
        td = self.param_module(td)

        action_loc = td.get(("agents", "action_loc"))
        action_scale = td.get(("agents", "action_scale"))
        return self._build_action_dist(action_loc, action_scale)

    def _set_policy_observation(self, tensordict):
        if not self.is_priority_conditioned_motion:
            return

        observation = tensordict.get(self.observation_key)
        scores = tensordict.get(self.PRIORITY_SCORES_KEY, None)
        if scores is None:
            scores = tensordict.get(("agents", "priority", "scores"), None)
        if scores is None:
            # TorchRL calls the motion policy once during collector construction
            # to infer its output spec, before the runtime priority module runs.
            scores = observation.new_zeros(*observation.shape[:-1], 1)

        neighbor_indices = tensordict.get(self.PRIORITY_NEIGHBOR_INDICES_KEY, None)
        if neighbor_indices is None:
            neighbor_indices = tensordict.get(
                ("agents", "priority_neighbor_indices"), None
            )
        neighbor_valid = tensordict.get(self.PRIORITY_NEIGHBOR_VALID_KEY, None)
        if neighbor_valid is None:
            neighbor_valid = tensordict.get(("agents", "priority_neighbor_valid"), None)

        if neighbor_indices is None or neighbor_valid is None:
            if self.env is not None and self.env.scenario is not None:
                (
                    scen_indices,
                    scen_valid,
                ) = self.env.scenario.get_priority_neighbor_metadata()
                if neighbor_indices is None:
                    neighbor_indices = scen_indices
                if neighbor_valid is None:
                    neighbor_valid = scen_valid
            if neighbor_indices is None:
                neighbor_indices = observation.new_zeros(
                    *observation.shape[:-1],
                    self.n_nearing_agents_observed,
                    dtype=torch.long,
                )
            if neighbor_valid is None:
                neighbor_valid = observation.new_ones(
                    *observation.shape[:-1],
                    self.n_nearing_agents_observed,
                    dtype=torch.bool,
                )

        features = self.build_priority_features(
            scores=scores,
            neighbor_indices=neighbor_indices,
            neighbor_valid=neighbor_valid,
        )
        tensordict.set(
            self.PRIORITY_CONDITIONED_OBSERVATION_KEY,
            torch.cat((observation, features), dim=-1),
        )

    @staticmethod
    def build_priority_features(scores, neighbor_indices, neighbor_valid):
        """Build detached ego/neighbor priority features for the motion actor."""

        if scores.shape[-1] != 1:
            raise ValueError("Priority scores must have a singleton last dimension.")
        weights = torch.softmax(scores.squeeze(-1), dim=-1)
        n_agents = weights.shape[-1]
        indices = neighbor_indices.to(dtype=torch.long)
        if indices.shape[-2] != n_agents:
            raise ValueError("Neighbor indices must contain one row per ego agent.")
        if indices.numel() and (indices.min() < 0 or indices.max() >= n_agents):
            raise ValueError("Neighbor indices are outside the agent dimension.")

        weights_by_ego = weights.unsqueeze(-2).expand(
            *weights.shape[:-1], n_agents, n_agents
        )
        neighbor_weights = torch.gather(weights_by_ego, dim=-1, index=indices)
        ego_weights = weights.unsqueeze(-1)
        valid = neighbor_valid.to(dtype=weights.dtype)
        neighbor_weights = neighbor_weights * valid
        priority_differences = (ego_weights - neighbor_weights) * valid

        return torch.cat(
            (ego_weights, neighbor_weights, priority_differences, valid), dim=-1
        ).detach()

    def set_train_mode(
        self,
        train_action: bool,
        train_lambda: bool,
        allow_action: bool = True,
        allow_lambda: bool = True,
    ):
        final_train_action = train_action and allow_action
        for p in self.parameters():
            p.requires_grad = final_train_action

    def actor_parameters_by_agent(self):
        """Return motion actor parameters for multi-timescale optimization."""

        if not self._uses_independent_actor:
            raise RuntimeError(
                "Per-agent motion actor parameters are available only for "
                "multi-timescale learning."
            )
        return self.param_module.module[0].parameters_by_agent()

    def load_motion_policy(self, path: str, map_location=None):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)

    def motion_policy_state_dict(self):
        return self.state_dict()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        # Action bounds are derived from the current env and may change with n_agents,
        # so older checkpoints should not override them.
        state_dict = dict(state_dict)
        state_dict.pop("action_low", None)
        state_dict.pop("action_high", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
