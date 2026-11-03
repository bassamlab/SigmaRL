from collections.abc import Mapping
from typing import Any

import torch
from tensordict.nn import TensorDictModule, TensorDictModuleBase, NormalParamExtractor
from torchrl.envs.utils import ExplorationType, exploration_type
from torchrl.modules import MultiAgentMLP, TanhNormal


class CBFLambdaPolicyModule(TensorDictModuleBase):
    def __init__(self, env, parameters, observation_key):
        super().__init__()
        self.env = env
        self.observation_key = observation_key

        obs_dim = env.observation_spec[observation_key].shape[-1]

        self.param_module = TensorDictModule(
            module=torch.nn.Sequential(
                MultiAgentMLP(
                    n_agent_inputs=obs_dim,
                    n_agent_outputs=2,
                    n_agents=env.n_agents,
                    centralised=False,
                    share_params=True,
                    device=parameters.device,
                    depth=3,
                    num_cells=256,
                    activation_class=torch.nn.Tanh,
                ),
                NormalParamExtractor(),
            ),
            in_keys=[observation_key],
            out_keys=[
                ("agents", "cbf", "lambda_loc"),
                ("agents", "cbf", "lambda_scale"),
            ],
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

    def _build_lambda_dist(
        self,
        lambda_loc: torch.Tensor,
        lambda_scale: torch.Tensor,
    ) -> TanhNormal:
        return TanhNormal(
            loc=lambda_loc,
            scale=lambda_scale,
            low=self.lambda_low,
            high=self.lambda_high,
        )

    @property
    def in_keys(self):
        return self.param_module.in_keys

    @property
    def out_keys(self):
        return [
            ("agents", "cbf", "lambda_loc"),
            ("agents", "cbf", "lambda_scale"),
            ("agents", "cbf", "lambda"),
            ("agents", "cbf", "lambda_log_prob"),
            ("agents", "sample_log_prob"),
        ]

    @property
    def dist_sample_keys(self):
        return [("agents", "cbf", "lambda")]

    @property
    def log_prob_keys(self):
        return [("agents", "cbf", "lambda_log_prob")]

    def get_dist(self, tensordict):
        return self.build_dist(tensordict)

    def forward(self, tensordict):
        tensordict = self.param_module(tensordict)

        lambda_loc = tensordict.get(("agents", "cbf", "lambda_loc"))
        lambda_scale = tensordict.get(("agents", "cbf", "lambda_scale"))

        lambda_dist = self._build_lambda_dist(lambda_loc, lambda_scale)

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
        if lambda_log_prob.shape == lambda_value.shape:
            lambda_log_prob = lambda_log_prob.sum(dim=-1)

        tensordict.set(("agents", "cbf", "lambda"), lambda_value)
        tensordict.set(("agents", "cbf", "lambda_log_prob"), lambda_log_prob)
        tensordict.set(("agents", "sample_log_prob"), lambda_log_prob)

        return tensordict

    def build_dist(self, tensordict):
        td = tensordict.clone(False)
        td = self.param_module(td)

        lambda_loc = td.get(("agents", "cbf", "lambda_loc"))
        lambda_scale = td.get(("agents", "cbf", "lambda_scale"))
        return self._build_lambda_dist(lambda_loc, lambda_scale)

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
        # Lambda bounds are configuration-derived and should come from the
        # current module rather than being restored from older checkpoints.
        state_dict = dict(state_dict)
        state_dict.pop("lambda_low", None)
        state_dict.pop("lambda_high", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
