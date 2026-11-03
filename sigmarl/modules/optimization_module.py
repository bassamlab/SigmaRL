# TODO: remove
# import torch
# from tensordict.nn import TensorDictModule
# from torchrl.modules import MultiAgentMLP
# from torchrl.objectives import ClipPPOLoss, ValueEstimators

# from sigmarl.modules.decision_making_module import DecisionMakingModule
# from sigmarl.modules.module import Module


# class OptimizationModule(Module):
#     def __init__(self, env, decision_module: DecisionMakingModule, mappo=True):

#         super().__init__()

#         self.parameters = env.scenario.parameters
#         self.decision_module = decision_module

#         critic_net = MultiAgentMLP(
#             n_agent_inputs=env.observation_spec[self.get_observation_key()].shape[
#                 -1
#             ],  # Number of observations
#             n_agent_outputs=1,  # 1 value per agent
#             n_agents=env.n_agents,
#             centralised=mappo,
#             # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
#             share_params=True,
#             # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
#             device=self.parameters.device,
#             depth=3,
#             num_cells=256,
#             activation_class=torch.nn.Tanh,
#         )

#         # print("critic_net:", critic_net, "\n")

#         critic = TensorDictModule(
#             module=critic_net,
#             in_keys=[
#                 self.get_observation_key()
#             ],  # Note that the critic in PPO only takes the same inputs (observations) as the actor
#             out_keys=[("agents", "state_value")],
#         )

#         loss_module = ClipPPOLoss(
#             actor=self.decision_module.policy,
#             critic=critic,
#             clip_epsilon=self.parameters.clip_epsilon,
#             entropy_coef=self.parameters.entropy_eps,
#             normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
#         )

#         loss_module.set_keys(
#             reward=env.reward_key,
#             action=[env.action_key, ("agents", "cbf", "lambda")],
#             sample_log_prob=[
#                 ("agents", "action_log_prob"),
#                 ("agents", "cbf", "lambda_log_prob"),
#             ],
#             value=("agents", "state_value"),
#             done=("agents", "done"),
#             terminated=("agents", "terminated"),
#         )

#         loss_module.make_value_estimator(
#             ValueEstimators.GAE,
#             gamma=self.parameters.gamma,
#             lmbda=self.parameters.lmbda,
#         )  # We build GAE
#         GAE = loss_module.value_estimator  # Generalized Advantage Estimation

#         optim = torch.optim.Adam(loss_module.parameters(), self.parameters.lr)

#         self.critic = critic

#         self.loss_module = loss_module
#         self.optim = optim
#         self.GAE = GAE

#     def get_observation_key(self):
#         return (
#             ("agents", "observation")
#             if not self.parameters.is_using_prioritized_marl
#             else ("agents", "info", "base_observation")
#         )


import torch
import inspect
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from sigmarl.modules.cbf_pair_lambda_policy_module import PAIR_LAMBDA_KEY
from sigmarl.modules.decision_making_module import DecisionMakingModule
from sigmarl.modules.module import Module


class OptimizationModule(Module):
    ACTION_REWARD_KEY = ("agents", "reward_motion")
    LAMBDA_REWARD_KEY = ("agents", "reward_lambda")
    ACTION_VALUE_KEY = ("agents", "action_state_value")
    LAMBDA_VALUE_KEY = ("agents", "lambda_state_value")
    ACTION_ADVANTAGE_KEY = ("agents", "action_advantage")
    LAMBDA_ADVANTAGE_KEY = ("agents", "lambda_advantage")
    ACTION_VALUE_TARGET_KEY = ("agents", "action_value_target")
    LAMBDA_VALUE_TARGET_KEY = ("agents", "lambda_value_target")

    def __init__(self, env, decision_module: DecisionMakingModule, mappo=True):
        super().__init__()

        self.parameters = env.scenario.parameters
        self.decision_module = decision_module
        self._observation_key = self.get_observation_key()
        observation_dim = env.observation_spec[self._observation_key].shape[-1]
        print(f"[INFO] OptimizationModule observation dimension: {observation_dim}")
        print("[INFO] OptimizationModule output dimension: 1")

        self.critic = None
        self.loss_module = None
        self.optim = None
        self.GAE = None
        self.action_critic = None
        self.action_loss_module = None
        self.action_GAE = None
        self.lambda_critic = None
        self.lambda_loss_module = None
        self.lambda_GAE = None
        self.motion_actor_param_group_count = 0
        self.motion_critic_param_group_index = None

        if self.parameters.is_no_policy_training:
            return

        if self.parameters.is_motion_only:
            critic = self._build_critic(
                env=env,
                mappo=mappo,
                out_key=("agents", "state_value"),
            )
            loss_module = ClipPPOLoss(
                actor=self.decision_module.policy,
                critic=critic,
                clip_epsilon=self.parameters.clip_epsilon,
                entropy_coeff=self.parameters.entropy_eps,
                normalize_advantage=False,
            )
            loss_module.set_keys(
                reward=env.reward_key,
                action=env.action_key,
                sample_log_prob=("agents", "action_log_prob"),
                value=("agents", "state_value"),
                done=("agents", "done"),
                terminated=("agents", "terminated"),
            )
            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.parameters.gamma,
                lmbda=self.parameters.lmbda,
            )

            self.critic = critic
            self.loss_module = loss_module
            self.optim = self._build_motion_optimizer(loss_module)
            self.GAE = loss_module.value_estimator
            self.action_critic = critic
            self.action_loss_module = loss_module
            self.action_GAE = self.GAE
            self.lambda_critic = None
            self.lambda_loss_module = None
            self.lambda_GAE = None
        elif self.parameters.is_lambda_only:
            critic = self._build_critic(
                env=env,
                mappo=mappo,
                out_key=("agents", "state_value"),
            )
            if self.parameters.cbf_lambda_head == "pair":
                # Only the pair head is optimized; an RL nominal motion policy is frozen
                lambda_actor = (
                    self.decision_module.policy.lambda_policy
                    if self.parameters.nom_controller_type == "rl"
                    else self.decision_module.policy
                )
                lambda_action_key = PAIR_LAMBDA_KEY
            else:
                lambda_actor = self.decision_module.policy
                lambda_action_key = ("agents", "cbf", "lambda")
            loss_module = ClipPPOLoss(
                actor=lambda_actor,
                critic=critic,
                clip_epsilon=self.parameters.clip_epsilon,
                entropy_coeff=self.parameters.entropy_eps,
                normalize_advantage=False,
            )
            loss_module.set_keys(
                reward=env.reward_key,
                action=lambda_action_key,
                sample_log_prob=("agents", "cbf", "lambda_log_prob"),
                value=("agents", "state_value"),
                done=("agents", "done"),
                terminated=("agents", "terminated"),
            )
            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.parameters.gamma,
                lmbda=self.parameters.lmbda,
            )

            self.critic = critic
            self.loss_module = loss_module
            self.optim = torch.optim.Adam(loss_module.parameters(), self.parameters.lr)
            self.GAE = loss_module.value_estimator
            self.action_critic = None
            self.action_loss_module = None
            self.action_GAE = None
            self.lambda_critic = critic
            self.lambda_loss_module = loss_module
            self.lambda_GAE = self.GAE
        else:
            action_critic = self._build_critic(
                env=env,
                mappo=mappo,
                out_key=self.ACTION_VALUE_KEY,
            )
            lambda_critic = self._build_critic(
                env=env,
                mappo=mappo,
                out_key=self.LAMBDA_VALUE_KEY,
            )

            action_loss_module = ClipPPOLoss(
                actor=self.decision_module.policy.motion_policy,
                critic=action_critic,
                clip_epsilon=self.parameters.clip_epsilon,
                entropy_coeff=self.parameters.entropy_eps,
                normalize_advantage=False,
            )
            action_loss_module.set_keys(
                reward=self.ACTION_REWARD_KEY,
                action=env.action_key,
                sample_log_prob=("agents", "action_log_prob"),
                value=self.ACTION_VALUE_KEY,
                advantage=self.ACTION_ADVANTAGE_KEY,
                value_target=self.ACTION_VALUE_TARGET_KEY,
                done=("agents", "done"),
                terminated=("agents", "terminated"),
            )
            action_loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.parameters.gamma,
                lmbda=self.parameters.lmbda,
            )

            lambda_loss_module = ClipPPOLoss(
                actor=self.decision_module.policy.lambda_policy,
                critic=lambda_critic,
                clip_epsilon=self.parameters.clip_epsilon,
                entropy_coeff=self.parameters.entropy_eps,
                normalize_advantage=False,
            )
            lambda_loss_module.set_keys(
                reward=self.LAMBDA_REWARD_KEY,
                action=("agents", "cbf", "lambda"),
                sample_log_prob=("agents", "cbf", "lambda_log_prob"),
                value=self.LAMBDA_VALUE_KEY,
                advantage=self.LAMBDA_ADVANTAGE_KEY,
                value_target=self.LAMBDA_VALUE_TARGET_KEY,
                done=("agents", "done"),
                terminated=("agents", "terminated"),
            )
            lambda_loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.parameters.gamma,
                lmbda=self.parameters.lmbda,
            )

            optim_params = self._unique_parameters(
                action_loss_module.parameters(),
                lambda_loss_module.parameters(),
            )

            self.critic = action_critic
            self.loss_module = action_loss_module
            self.optim = torch.optim.Adam(optim_params, self.parameters.lr)
            self.GAE = action_loss_module.value_estimator
            self.action_critic = action_critic
            self.action_loss_module = action_loss_module
            self.action_GAE = action_loss_module.value_estimator
            self.lambda_critic = lambda_critic
            self.lambda_loss_module = lambda_loss_module
            self.lambda_GAE = lambda_loss_module.value_estimator

    def _build_critic(self, env, mappo, out_key):
        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[self._observation_key].shape[-1],
            n_agent_outputs=1,
            n_agents=env.n_agents,
            centralised=self.parameters.is_centralized_critic,
            share_params=True,
            device=self.parameters.device,
            depth=3,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        return TensorDictModule(
            module=critic_net,
            in_keys=[self._observation_key],
            out_keys=[out_key],
        )

    def _build_motion_optimizer(self, loss_module):
        """Build the motion optimizer, optionally with one group per agent."""

        if not self.parameters.is_multi_timescale_learning:
            return torch.optim.Adam(loss_module.parameters(), self.parameters.lr)

        actor_parameters_by_agent = (
            self.decision_module.policy.actor_parameters_by_agent()
        )
        if len(actor_parameters_by_agent) != self.parameters.n_agents:
            raise ValueError(
                "The multi-timescale motion actor must expose one parameter "
                "group per agent."
            )

        actor_parameter_ids = {
            id(parameter)
            for agent_parameters in actor_parameters_by_agent
            for parameter in agent_parameters
        }
        loss_parameter_ids = {id(parameter) for parameter in loss_module.parameters()}
        missing_actor_parameters = actor_parameter_ids - loss_parameter_ids
        if missing_actor_parameters:
            raise ValueError(
                "Motion actor parameters are missing from the PPO loss module."
            )

        optimizer_groups = [
            {"params": agent_parameters, "lr": self.parameters.lr}
            for agent_parameters in actor_parameters_by_agent
        ]
        critic_parameters = [
            parameter
            for parameter in loss_module.parameters()
            if id(parameter) not in actor_parameter_ids
        ]
        self.motion_actor_param_group_count = len(optimizer_groups)
        if critic_parameters:
            self.motion_critic_param_group_index = len(optimizer_groups)
            optimizer_groups.append(
                {"params": critic_parameters, "lr": self.parameters.lr}
            )
        return torch.optim.Adam(optimizer_groups, lr=self.parameters.lr)

    def set_motion_learning_rates(
        self,
        *,
        fast_learning_rate: float,
        slow_learning_rate: float,
        fast_agent: int,
        critic_learning_rate: float,
    ):
        """Set the current fast/slow actor rates for one PPO update."""

        if not self.parameters.is_multi_timescale_learning:
            return
        if not 0 <= fast_agent < self.motion_actor_param_group_count:
            raise ValueError(f"Invalid fast-agent index: {fast_agent}.")
        for agent_index in range(self.motion_actor_param_group_count):
            self.optim.param_groups[agent_index]["lr"] = (
                fast_learning_rate if agent_index == fast_agent else slow_learning_rate
            )
        if self.motion_critic_param_group_index is not None:
            self.optim.param_groups[self.motion_critic_param_group_index][
                "lr"
            ] = critic_learning_rate

    @staticmethod
    def _unique_parameters(*param_groups):
        unique = []
        seen = set()
        for params in param_groups:
            for param in params:
                if not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                unique.append(param)
        return unique

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
