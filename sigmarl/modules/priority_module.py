import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

#from sigmarl.helper_training import TransformedEnvCustom TODO OOP
from sigmarl.modules.module import Module


class PriorityModule(Module):
    def __init__(self, env = None, mappo: bool = True):
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

        super().__init__(env, mappo)

        self.env = env
        self.parameters = self.env.scenario.parameters

        # Tuple containing the prefix keys relevant to the priority variables
        self.prefix_key = ("agents", "info", "priority")

        observation_key = self.get_observation_key()

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
            spec=UnboundedContinuousTensorSpec(),
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

    def get_observation_key(self) -> tuple:
        return ("agents", "info", "priority_observation")

