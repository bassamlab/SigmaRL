import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from sigmarl.modules.decision_making_module import DecisionMakingModule
from sigmarl.modules.module import Module

class OptimizationModule(Module):
    def __init__(self, env, decision_module: DecisionMakingModule, mappo=True):

        super().__init__()

        self.parameters = env.scenario.parameters
        self.decision_module = decision_module

        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[self.get_observation_key()].shape[
                -1
            ],  # Number of observations
            n_agent_outputs=1,  # 1 value per agent
            n_agents=env.n_agents,
            centralised=mappo,
            # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
            share_params=True,
            # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
            device=self.parameters.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        # print("critic_net:", critic_net, "\n")

        critic = TensorDictModule(
            module=critic_net,
            in_keys=[
                self.get_observation_key()
            ],  # Note that the critic in PPO only takes the same inputs (observations) as the actor
            out_keys=[("agents", "state_value")],
        )

        loss_module = ClipPPOLoss(
            actor=self.decision_module.policy,
            critic=critic,
            clip_epsilon=self.parameters.clip_epsilon,
            entropy_coef=self.parameters.entropy_eps,
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
            ValueEstimators.GAE, gamma=self.parameters.gamma, lmbda=self.parameters.lmbda
        )  # We build GAE
        GAE = loss_module.value_estimator  # Generalized Advantage Estimation

        optim = torch.optim.Adam(loss_module.parameters(), self.parameters.lr)

        self.critic = critic

        self.loss_module = loss_module
        self.optim = optim
        self.GAE = GAE

    def get_observation_key(self):
        return (
            ("agents", "observation")
            if not self.parameters.is_using_prioritized_marl
            else ("agents", "info", "base_observation")
        )