import torch
from tensordict.nn import TensorDictModule, NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

#from sigmarl.helper_training import TransformedEnvCustom
from sigmarl.modules.module import Module


class DecisionMakingModule(Module):
    def __init__(self, env = None, mappo: bool = True):
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
        super().__init__(env, mappo)

        self.parameters = env.scenario.parameters
        observation_key = self.get_observation_key()

        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec[observation_key].shape[-1],  # n_obs_per_agent
                n_agent_outputs=(
                    2 * env.action_spec.shape[-1]
                ),  # 2 * n_actions_per_agents
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
            device=self.parameters.device,
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

        self.policy = policy
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
