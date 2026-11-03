# TODO: remove
# from typing import Union, Any

# import torch
# from tensordict.nn import TensorDictModule, NormalParamExtractor
# from torchrl.data import TensorSpec
# from torchrl.envs import VmasEnv
# from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# # from sigmarl.helper_training import Parameters, TransformedEnvCustom
# from sigmarl.modules.module import Module

# from dataclasses import dataclass

# from sigmarl.scenarios.road_traffic import ScenarioRoadTraffic


# class DecisionMakingModule(Module):
#     def __init__(self, env, parameters):
#         """
#         Initializes the DecisionMakingModule, which is responsible for generating decisions using a neural network policy.
#         It also sets up a PPO loss module with an actor-critic architecture and GAE (Generalized Advantage Estimation) for RL optimization.

#         Parameters:
#         -----------
#         env : TransformedEnvCustom
#             The environment containing the observation specifications and other scenario parameters.
#         mappo : bool, optional
#             Flag to indicate whether to use centralised learning in the critic (MAPPO). Default is True.
#         """
#         super().__init__()

#         self.parameters = parameters

#         policy_net = torch.nn.Sequential(
#             MultiAgentMLP(
#                 n_agent_inputs=env.observation_spec[self.get_observation_key()].shape[
#                     -1
#                 ],  # n_obs_per_agent
#                 n_agent_outputs=(
#                     2 * env.action_spec.shape[-1]
#                 ),  # 2 * n_actions_per_agent
#                 n_agents=env.n_agents,
#                 centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
#                 share_params=True,  # sharing parameters means that agents will all share the same policy, which will allow them to benefit from each other’s experiences, resulting in faster training. On the other hand, it will make them behaviorally homogenous, as they will share the same model
#                 device=self.parameters.device,
#                 depth=3,
#                 num_cells=256,
#                 activation_class=torch.nn.Tanh,
#             ),
#             NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a `loc` and a non-negative `scale``, used as parameters for a normal distribution (mean and standard deviation)
#         )

#         # print("policy_net:", policy_net, "\n")

#         policy_module = TensorDictModule(
#             policy_net,
#             in_keys=[self.get_observation_key()],
#             out_keys=[
#                 ("agents", "loc"),
#                 ("agents", "scale"),
#             ],  # represents the parameters of the policy distribution for each agent
#         )

#         # Use a probabilistic actor allows for exploration
#         policy = ProbabilisticActor(
#             module=policy_module,
#             spec=env.unbatched_action_spec,
#             in_keys=[("agents", "loc"), ("agents", "scale")],
#             out_keys=[env.action_key],
#             distribution_class=TanhNormal,
#             distribution_kwargs={
#                 "low": env.unbatched_action_spec[env.action_key].space.low,
#                 "high": env.unbatched_action_spec[env.action_key].space.high,
#             },
#             return_log_prob=True,
#             log_prob_key=(
#                 "agents",
#                 "sample_log_prob",
#             ),  # log probability favors numerical stability and gradient calculation
#         )  # we'll need the log-prob for the PPO loss

#         self.policy = policy

#     @classmethod
#     def from_env(cls, env, parameters):
#         return cls(env, parameters)

#     @classmethod
#     def from_params(cls, parameters, n_agents):

#         scenario = ScenarioRoadTraffic()

#         env = VmasEnv(
#             scenario=scenario,
#             num_envs=1,
#             continuous_actions=True,
#             max_steps=parameters.max_steps,
#             device=parameters.device,
#             n_agents=n_agents,
#         )

#         return cls.from_env(env, parameters)

#     def get_observation_key(self):

#         return (
#             ("agents", "observation")
#             if not self.parameters.is_using_prioritized_marl
#             else ("agents", "info", "base_observation")
#         )


import os
from typing import Union, Any

import torch
from tensordict.nn import TensorDictModule, NormalParamExtractor

# from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from sigmarl.helper_common import Parameters

from sigmarl.modules.module import Module

from sigmarl.modules.joint_policy_module import JointPolicyModule
from sigmarl.modules.cbf_lambda_policy_module import CBFLambdaPolicyModule
from sigmarl.modules.cbf_pair_lambda_policy_module import (
    CBFPairLambdaPolicyModule,
    FrozenMotionPairLambdaPolicyModule,
)
from sigmarl.modules.motion_policy_module import MotionPolicyModule


class NoPolicyModule(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self, tensordict):
        return tensordict

    def set_train_mode(
        self,
        train_action: bool,
        train_lambda: bool,
        allow_action: bool = True,
        allow_lambda: bool = True,
    ):
        return None


class DecisionMakingModule(Module):
    def __init__(self, env, parameters):
        super().__init__()
        self.parameters: Parameters = parameters

        if self.parameters.is_joint_training:
            self.policy = JointPolicyModule(
                env=env,
                parameters=parameters,
                observation_key=self.get_observation_key(),
            )
        elif (
            self.parameters.is_lambda_only
            and self.parameters.cbf_lambda_head == "pair"
            and self.parameters.nom_controller_type == "rl"
        ):
            self.policy = FrozenMotionPairLambdaPolicyModule(
                env=env,
                parameters=parameters,
                observation_key=self.get_observation_key(),
            )
        elif (
            self.parameters.is_lambda_only and self.parameters.cbf_lambda_head == "pair"
        ):
            # CLF nominal controller: the pair head alone
            self.policy = CBFPairLambdaPolicyModule(
                env=env,
                parameters=parameters,
                observation_key=self.get_observation_key(),
            )
        elif self.parameters.is_lambda_only:
            self.policy = CBFLambdaPolicyModule(
                env=env,
                parameters=parameters,
                observation_key=self.get_observation_key(),
            )
        elif self.parameters.is_no_policy_training:
            self.policy = NoPolicyModule(env=env)
        else:
            self.policy = MotionPolicyModule(
                env=env,
                parameters=parameters,
                observation_key=self.get_observation_key(),
            )

        self.opponent_predictor = None
        if self.parameters.is_using_opponent_modeling and getattr(
            self.parameters,
            "is_opponent_modeling_separate_predictor",
            False,
        ):
            if not self.parameters.uses_motion_policy:
                raise ValueError(
                    "A separate opponent-modeling predictor requires a motion policy."
                )
            self.opponent_predictor = MotionPolicyModule(
                env=env,
                parameters=parameters,
                observation_key=self.get_observation_key(),
            )
            self.sync_opponent_predictor()

        # if not self.parameters.is_train_rl_motion_policy:
        #     # Train a policy that directly outputs the CBF lambda without learning a separate RL motion policy. This is used when we only want to learn lambda and use a nominal controller (such as a CLF controller) for motion planning, without learning a RL motion policy.
        #     self.policy = LambdaPolicy(
        #         env=env,
        #         parameters=parameters,
        #         observation_key=self.get_observation_key(),
        #     )
        # else:
        #     if self.parameters.is_learn_cbf_lambda:
        #         # Train a joint policy that jointly outputs the actions for motion planning and CBF lambda
        #         self.policy = JointActionLambdaPolicy(
        #             env=env,
        #             parameters=parameters,
        #             observation_key=self.get_observation_key(),
        #         )
        #     else:
        #         # Train a regular RL policy for motion planning without learning lambda
        #         policy_net = torch.nn.Sequential(
        #             MultiAgentMLP(
        #                 n_agent_inputs=env.observation_spec[
        #                     self.get_observation_key()
        #                 ].shape[-1],
        #                 n_agent_outputs=2 * env.action_spec.shape[-1],
        #                 n_agents=env.n_agents,
        #                 centralised=False,
        #                 share_params=True,
        #                 device=self.parameters.device,
        #                 depth=3,
        #                 num_cells=256,
        #                 activation_class=torch.nn.Tanh,
        #             ),
        #             NormalParamExtractor(),
        #         )

        #         policy_module = TensorDictModule(
        #             policy_net,
        #             in_keys=[self.get_observation_key()],
        #             out_keys=[("agents", "loc"), ("agents", "scale")],
        #         )

        #         self.policy = ProbabilisticActor(
        #             module=policy_module,
        #             spec=env.unbatched_action_spec,
        #             in_keys=[("agents", "loc"), ("agents", "scale")],
        #             out_keys=[env.action_key],
        #             distribution_class=TanhNormal,
        #             distribution_kwargs={
        #                 "low": env.unbatched_action_spec[env.action_key].space.low,
        #                 "high": env.unbatched_action_spec[env.action_key].space.high,
        #             },
        #             return_log_prob=True,
        #             log_prob_key=("agents", "sample_log_prob"),
        #         )

    @classmethod
    def from_env(cls, env, parameters):
        return cls(env, parameters)

    @classmethod
    def from_params(cls, parameters, n_agents):
        # Env. The fleet size comes from the caller (e.g. ROS launch arguments),
        # not from the checkpoint, and the backend is pinned to "vmas": this
        # path only needs a scenario to shape the policy, never a bridge to the
        # real fleet, even when the checkpoint was fine-tuned against the lab.
        from sigmarl.env_factory import build_vmas_env

        env = build_vmas_env(
            parameters,
            physics_backend="vmas",
            n_agents=n_agents,
            num_envs=1,
        )
        return cls.from_env(env, parameters)

    def get_observation_key(self):
        if self.parameters.is_fast_inference_mode:
            return ("agents", "observation")
        if self.parameters.is_using_opponent_modeling:
            return ("agents", "info", "observation_with_actions")
        if (
            self.parameters.is_using_prioritized_marl
            or self.parameters.is_obs_vehicle_id_in_motion_policy
        ):
            return ("agents", "info", "observation_without_actions")
        return ("agents", "observation")

    def sync_opponent_predictor(self):
        """Hard-sync and freeze the optional opponent-prediction policy."""

        if self.opponent_predictor is None:
            return

        motion_policy = getattr(self.policy, "motion_policy", self.policy)
        self.opponent_predictor.load_state_dict(motion_policy.state_dict())
        self.opponent_predictor.eval()
        for parameter in self.opponent_predictor.parameters():
            parameter.requires_grad_(False)

    def load_pretrained_policy_modules(self):
        if self.parameters.is_load_pretrained_motion_policy:
            if not self.parameters.pretrained_motion_policy_path:
                print(
                    "[WARNING] pretrained_motion_policy_path is not specified. "
                    f"Searching in {self.parameters.where_to_save}."
                )
                with os.scandir(self.parameters.where_to_save) as entries:
                    for entry in entries:
                        if (
                            entry.is_file()
                            and entry.name.endswith(".pth")
                            and "motion_policy" in entry.name
                        ):
                            self.parameters.pretrained_motion_policy_path = entry.path
                            break
                if not self.parameters.pretrained_motion_policy_path:
                    raise ValueError("pretrained_motion_policy_path is empty.")
            if not hasattr(self.policy, "load_motion_policy"):
                raise ValueError(
                    "Current policy does not include a motion policy to load."
                )
            self.policy.load_motion_policy(
                self.parameters.pretrained_motion_policy_path,
                map_location=self.parameters.device,
            )
            print(
                f"[INFO] Loaded pretrained motion policy from {self.parameters.pretrained_motion_policy_path}"
            )

        if self.parameters.is_load_pretrained_lambda_policy:
            if not self.parameters.pretrained_lambda_policy_path:
                print(
                    "[WARNING] pretrained_lambda_policy_path is not specified. "
                    f"Searching in {self.parameters.where_to_save}."
                )
                with os.scandir(self.parameters.where_to_save) as entries:
                    for entry in entries:
                        if (
                            entry.is_file()
                            and entry.name.endswith(".pth")
                            and "lambda_policy" in entry.name
                        ):
                            self.parameters.pretrained_lambda_policy_path = entry.path
                            break
                if not self.parameters.pretrained_lambda_policy_path:
                    raise ValueError("pretrained_lambda_policy_path is empty.")
            if not hasattr(self.policy, "load_lambda_policy"):
                raise ValueError(
                    "Current policy does not include a lambda policy to load."
                )
            self.policy.load_lambda_policy(
                self.parameters.pretrained_lambda_policy_path,
                map_location=self.parameters.device,
            )
            print(
                f"[INFO] Loaded pretrained lambda policy from {self.parameters.pretrained_lambda_policy_path}"
            )

        self.sync_opponent_predictor()
