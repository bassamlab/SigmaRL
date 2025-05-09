from typing import Union, Any

import torch
from tensordict.nn import TensorDictModule, NormalParamExtractor
from torchrl.data import TensorSpec
from torchrl.envs import VmasEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

#from sigmarl.helper_training import Parameters, TransformedEnvCustom
from sigmarl.modules.module import Module

from dataclasses import dataclass

from sigmarl.scenarios.road_traffic import ScenarioRoadTraffic

class DecisionMakingModule(Module):
    def __init__(self,
                 env,
                 parameters):
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
        super().__init__()

        self.parameters = parameters

        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec[self.get_observation_key()].shape[-1],  # n_obs_per_agent
                n_agent_outputs=(
                    2 * env.action_spec.shape[-1]
                ),  # 2 * n_actions_per_agent
                n_agents=env.n_agents,
                centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
                share_params=True,  # sharing parameters means that agents will all share the same policy, which will allow them to benefit from each other’s experiences, resulting in faster training. On the other hand, it will make them behaviorally homogenous, as they will share the same model
                device=self.parameters.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a `loc` and a non-negative `scale``, used as parameters for a normal distribution (mean and standard deviation)
        )

        # print("policy_net:", policy_net, "\n")

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[self.get_observation_key()],
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
                "high": env.unbatched_action_spec[env.action_key].space.high
            },
            return_log_prob=True,
            log_prob_key=(
                "agents",
                "sample_log_prob",
            ),  # log probability favors numerical stability and gradient calculation
        )  # we'll need the log-prob for the PPO loss

        self.policy = policy

    @classmethod
    def from_env(cls, env, parameters):
        return cls(env, parameters)

    @classmethod
    def from_params(cls, parameters, n_agents):

        scenario = ScenarioRoadTraffic()

        env = VmasEnv(
            scenario=scenario,
            num_envs=1,
            continuous_actions=True,
            max_steps=parameters.max_steps,
            device=parameters.device,
            n_agents=n_agents,
        )

        return cls.from_env(env, parameters)

    def get_observation_key(self):

        return (
            ("agents", "observation")
            if not self.parameters.is_using_prioritized_marl
            else ("agents", "info", "base_observation")
        )
