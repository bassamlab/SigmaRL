import torch
import inspect
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# from sigmarl.helper_training import TransformedEnvCustom TODO OOP
from sigmarl.modules.module import Module


class PriorityModule(Module):
    def __init__(self, env=None, mappo: bool = True):
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

        super().__init__()

        self.env = env
        self.parameters = self.env.scenario.parameters
        self.reset_diagnostics()
        self._random_priority_scores = None
        self._random_priority_valid = None
        self._random_priority_episodes_since_refresh = None
        self._taskprogress_permutation = None
        self._taskprogress_permutation_valid = None

        # Tuple containing the prefix keys relevant to the priority variables
        self.prefix_key = ("agents", "info", "priority")

        observation_key = self.get_observation_key()
        observation_dim = env.observation_spec[observation_key].shape[-1]
        print(f"[INFO] PriorityModule observation dimension: {observation_dim}")
        print("[INFO] PriorityModule output dimension: 2")

        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=observation_dim,
                n_agent_outputs=2 * 1,  # 2 * n_actions_per_agents
                n_agents=self.parameters.n_agents,
                centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
                share_params=self.parameters.is_share_params_actor,  # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
                device=self.parameters.device,
                depth=3,
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
            n_agent_inputs=observation_dim,  # Number of observations
            n_agent_outputs=1,  # 1 value per agent
            n_agents=self.parameters.n_agents,
            centralised=self.parameters.is_centralized_critic,  # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
            share_params=True,  # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
            device=self.parameters.device,
            depth=3,
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
                entropy_coeff=self.parameters.entropy_eps_priority,  # torchrl==0.7.2 uses "entropy_coef" while torchrl==0.9.0 uses "entropy_coeff"
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
                advantage=self.prefix_key + ("advantage",),
                value_target=self.prefix_key + ("value_target",),
            )

            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.parameters.gamma,
                lmbda=self.parameters.lmbda,
            )  # We build GAE
            GAE = loss_module.value_estimator  # Generalized Advantage Estimation

            optim = torch.optim.Adam(
                loss_module.parameters(), self.parameters.lr_priority
            )

            self.GAE = GAE
            self.loss_module = loss_module
            self.optim = optim

    def _random_scores(self, observation):
        """Return random scores with optional per-environment episode caching."""
        random_scores = observation[..., :1]
        if not self.parameters.random_priority_fixed_episode:
            return torch.rand_like(random_scores)

        batch_shape = observation.shape[:-2]
        cache_is_compatible = (
            self._random_priority_scores is not None
            and self._random_priority_scores.shape == random_scores.shape
            and self._random_priority_scores.device == random_scores.device
            and self._random_priority_scores.dtype == random_scores.dtype
        )
        if not cache_is_compatible:
            self._random_priority_scores = torch.rand_like(random_scores)
            self._random_priority_valid = torch.ones(
                batch_shape, dtype=torch.bool, device=random_scores.device
            )
            self._random_priority_episodes_since_refresh = torch.zeros(
                batch_shape, dtype=torch.long, device=random_scores.device
            )
            return self._random_priority_scores

        invalid = ~self._random_priority_valid
        if invalid.any():
            fresh_scores = torch.rand_like(random_scores)
            invalid_mask = invalid.reshape(*batch_shape, 1, 1)
            self._random_priority_scores = torch.where(
                invalid_mask, fresh_scores, self._random_priority_scores
            )
            self._random_priority_valid.fill_(True)
        return self._random_priority_scores

    @staticmethod
    def _sample_agent_permutations(batch_shape, n_agents, device):
        """Sample one agent-index permutation for each vectorized environment."""
        random_keys = torch.rand(
            (*batch_shape, n_agents),
            device=device,
        )
        return torch.argsort(random_keys, dim=-1)

    def _permuted_taskprogress_scores(self, observation):
        """Return taskprogress scores assigned through an episode-fixed agent permutation."""
        taskprogress_scores = self._taskprogress_scores(observation)
        batch_shape = taskprogress_scores.shape[:-2]
        n_agents = taskprogress_scores.shape[-2]
        permutation_shape = (*batch_shape, n_agents)
        cache_is_compatible = (
            self._taskprogress_permutation is not None
            and self._taskprogress_permutation.shape == permutation_shape
            and self._taskprogress_permutation.device == taskprogress_scores.device
        )
        if not cache_is_compatible:
            self._taskprogress_permutation = self._sample_agent_permutations(
                batch_shape,
                n_agents,
                taskprogress_scores.device,
            )
            self._taskprogress_permutation_valid = torch.ones(
                batch_shape,
                dtype=torch.bool,
                device=taskprogress_scores.device,
            )

        invalid = ~self._taskprogress_permutation_valid
        if invalid.any():
            fresh_permutations = self._sample_agent_permutations(
                batch_shape,
                n_agents,
                taskprogress_scores.device,
            )
            invalid_mask = invalid.reshape(*batch_shape, 1)
            self._taskprogress_permutation = torch.where(
                invalid_mask,
                fresh_permutations,
                self._taskprogress_permutation,
            )
            self._taskprogress_permutation_valid.fill_(True)

        permuted_scores = torch.gather(
            taskprogress_scores.squeeze(-1),
            dim=-1,
            index=self._taskprogress_permutation,
        )
        return permuted_scores.unsqueeze(-1)

    @staticmethod
    def _normalize_episode_end(episode_end, batch_shape, device):
        """Normalize a vectorized environment done mask to its batch shape."""
        episode_end = episode_end.to(device=device, dtype=torch.bool)
        while episode_end.ndim > len(batch_shape) and episode_end.shape[-1] == 1:
            episode_end = episode_end.squeeze(-1)
        if episode_end.shape != batch_shape:
            raise ValueError(
                "Episode-end mask shape must match the priority batch shape: "
                f"got {tuple(episode_end.shape)}, expected {tuple(batch_shape)}."
            )
        return episode_end

    def notify_episode_end(self, episode_end):
        """Advance episode counters and refresh cached random priorities as needed."""
        if (
            self.parameters.random_priority_fixed_episode
            and self.parameters.random_priority_fixed_episode > 0
            and self._random_priority_scores is not None
        ):
            batch_shape = self._random_priority_scores.shape[:-2]
            normalized_episode_end = self._normalize_episode_end(
                episode_end,
                batch_shape,
                self._random_priority_scores.device,
            )
            self._random_priority_episodes_since_refresh += normalized_episode_end.to(
                torch.long
            )
            refresh = (
                self._random_priority_episodes_since_refresh
                >= self.parameters.random_priority_fixed_episode
            )
            if refresh.any():
                self._random_priority_valid[refresh] = False
                self._random_priority_episodes_since_refresh[refresh] = 0

        if self._taskprogress_permutation is not None:
            batch_shape = self._taskprogress_permutation.shape[:-1]
            normalized_episode_end = self._normalize_episode_end(
                episode_end,
                batch_shape,
                self._taskprogress_permutation.device,
            )
            self._taskprogress_permutation_valid[normalized_episode_end] = False

    def _fixed_scores(self, observation):
        """Return descending fixed scores for agent IDs 0 through N-1."""
        n_agents = observation.shape[-2]
        scores = torch.arange(
            n_agents - 1,
            -1,
            -1,
            device=observation.device,
            dtype=observation.dtype,
        )
        return scores.reshape(1, n_agents, 1).expand(*observation.shape[:-1], 1)

    def _taskprogress_scores(self, observation):
        """Return route-progress scores for the taskprogress priority ablation.

        The road-traffic world state tracks progress along each selected route.
        A vehicle farther along its route is treated as having higher priority;
        stable ranking then resolves equal-progress cases by agent ID.
        """
        world_state = self.env.base_env.scenario_name.world_state
        route_progress = world_state._route_progress.to(
            device=observation.device, dtype=observation.dtype
        )
        route_total_lengths = world_state._route_progress_total_lengths.to(
            device=observation.device, dtype=observation.dtype
        )
        route_total_lengths = route_total_lengths.clamp_min(
            torch.finfo(observation.dtype).eps
        )
        progress_fraction = torch.nan_to_num(
            route_progress / route_total_lengths,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        return progress_fraction.unsqueeze(-1)

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
        ordered_indices = torch.argsort(scores, dim=-1, descending=True, stable=True)

        return ordered_indices

    def __call__(self, tensordict):
        """
        Computes priority scores and ranks, then updates the tensordict.

        In MARL mode, scores come from the learned priority actor. In random
        mode, scores are sampled per call or cached for the configured number
        of episodes. Fixed mode uses agent IDs, while taskprogress modes use
        normalized route progress, with ``taskprogress_permute`` assigning those
        scores through an episode-fixed random agent permutation. All modes
        write scores and ranks through the same interface used by reward redistribution and
        priority-conditioned motion.

        Parameters:
        -----------
        tensordict : TensorDict
            A dictionary-like object containing the data for the agents.

        Returns:
        --------
        tensordict : TensorDict
            The updated tensordict with the priority rank of agents added.
        """

        prioritization_method = self.parameters.prioritization_method.lower()
        obs_key = self.get_observation_key()
        observation = tensordict.get(obs_key, None)
        if observation is None:
            observation = tensordict.get(("agents", "observation"), None)

        if prioritization_method == "random":
            scores = self._random_scores(observation)
            tensordict.set(self.prefix_key + ("scores",), scores)
        elif prioritization_method == "fixed":
            scores = self._fixed_scores(observation)
            tensordict.set(self.prefix_key + ("scores",), scores)
        elif prioritization_method == "taskprogress":
            scores = self._taskprogress_scores(observation)
            tensordict.set(self.prefix_key + ("scores",), scores)
        elif prioritization_method == "taskprogress_permute":
            scores = self._permuted_taskprogress_scores(observation)
            tensordict.set(self.prefix_key + ("scores",), scores)
        elif prioritization_method == "marl":
            # Call the priority actor and extract the learned scores.
            scores = self.policy(tensordict)[self.prefix_key + ("scores",)]
        else:
            raise ValueError(
                "Unsupported prioritization method: "
                f"{self.parameters.prioritization_method!r}."
            )

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

        diagnostics = self._compute_diagnostics(data, loss_vals)
        if not torch.isfinite(loss_value).all():
            diagnostics["gradient_norm"] = float("nan")
            self._diagnostic_records.append(diagnostics)
            print(self.format_diagnostics(prefix="[ERROR]"))
            raise ValueError("Non-finite priority PPO loss.")

        loss_value.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), self.parameters.max_grad_norm
        )  # Optional
        diagnostics["gradient_norm"] = self._as_float(gradient_norm)
        self._diagnostic_records.append(diagnostics)

        if not torch.isfinite(gradient_norm).all():
            print(self.format_diagnostics(prefix="[ERROR]"))
            self.optim.zero_grad()
            raise ValueError("Non-finite priority PPO gradient norm.")

        self.optim.step()
        self.optim.zero_grad()

    def reset_diagnostics(self):
        """Reset the priority PPO diagnostics accumulated for one iteration."""
        self._diagnostic_records = []

    def _compute_diagnostics(self, data, loss_vals):
        """Collect numerical diagnostics for the current priority PPO update."""
        with torch.no_grad():
            current_log_prob, distribution, _ = self.loss_module._get_cur_log_prob(data)
            old_log_prob = data.get(self.prefix_key + ("sample_log_prob",))
            log_ratio = current_log_prob - old_log_prob

        diagnostics = {
            "loss_objective": self._as_float(loss_vals["loss_objective"]),
            "loss_critic": self._as_float(loss_vals["loss_critic"]),
            "loss_entropy": self._as_float(loss_vals["loss_entropy"]),
            "policy_entropy": self._as_float(loss_vals["entropy"]),
            "approximate_kl": self._as_float(loss_vals["kl_approx"]),
            "clip_fraction": self._as_float(loss_vals["clip_fraction"]),
            "effective_sample_size": self._as_float(loss_vals["ESS"]),
            "old_log_prob_min": self._as_float(old_log_prob.min()),
            "old_log_prob_max": self._as_float(old_log_prob.max()),
            "new_log_prob_min": self._as_float(current_log_prob.min()),
            "new_log_prob_max": self._as_float(current_log_prob.max()),
            "log_ratio_min": self._as_float(log_ratio.min()),
            "log_ratio_max": self._as_float(log_ratio.max()),
            "loc_mean": self._as_float(distribution.loc.mean()),
            "loc_min": self._as_float(distribution.loc.min()),
            "loc_max": self._as_float(distribution.loc.max()),
            "scale_mean": self._as_float(distribution.scale.mean()),
            "scale_min": self._as_float(distribution.scale.min()),
            "scale_max": self._as_float(distribution.scale.max()),
        }
        if "explained_variance" in loss_vals.keys():
            diagnostics["explained_variance"] = self._as_float(
                loss_vals["explained_variance"]
            )
        return diagnostics

    @staticmethod
    def _as_float(value):
        """Convert a scalar tensor or number to a detached Python float."""
        if isinstance(value, torch.Tensor):
            value = value.detach().mean().item()
        return float(value)

    def diagnostics_summary(self):
        """Return summary statistics for priority updates in this iteration."""
        if not self._diagnostic_records:
            return {}

        mean_keys = (
            "loss_objective",
            "loss_critic",
            "loss_entropy",
            "policy_entropy",
            "approximate_kl",
            "clip_fraction",
            "effective_sample_size",
            "explained_variance",
            "gradient_norm",
            "loc_mean",
            "scale_mean",
        )
        summary = {"updates": len(self._diagnostic_records)}
        for key in mean_keys:
            values = [
                record[key] for record in self._diagnostic_records if key in record
            ]
            if values:
                summary[key] = sum(values) / len(values)

        for stem in ("old_log_prob", "new_log_prob", "log_ratio", "loc", "scale"):
            summary[f"{stem}_min"] = min(
                record[f"{stem}_min"] for record in self._diagnostic_records
            )
            summary[f"{stem}_max"] = max(
                record[f"{stem}_max"] for record in self._diagnostic_records
            )
        summary["gradient_norm_max"] = max(
            record["gradient_norm"] for record in self._diagnostic_records
        )
        return summary

    def format_diagnostics(self, iteration=None, prefix="[PRIORITY]"):
        """Format the accumulated priority PPO diagnostics as one log line."""
        summary = self.diagnostics_summary()
        if not summary:
            return f"{prefix} no priority PPO updates"

        iteration_text = f" iter={iteration}" if iteration is not None else ""
        learning_rate = self.optim.param_groups[0]["lr"]
        explained_variance = summary.get("explained_variance", float("nan"))
        return (
            f"{prefix}{iteration_text} updates={summary['updates']} lr={learning_rate:.3e} "
            f"loss(obj={summary['loss_objective']:.3e}, "
            f"critic={summary['loss_critic']:.3e}, "
            f"entropy={summary['loss_entropy']:.3e}) "
            f"policy(entropy={summary['policy_entropy']:.3e}, "
            f"kl={summary['approximate_kl']:.3e}, "
            f"clip={summary['clip_fraction']:.3f}, "
            f"ess={summary['effective_sample_size']:.3f}, "
            f"explained_var={explained_variance:.3e}) "
            f"grad_norm(mean={summary['gradient_norm']:.3e}, "
            f"max={summary['gradient_norm_max']:.3e}) "
            f"logp(old=[{summary['old_log_prob_min']:.3e},"
            f"{summary['old_log_prob_max']:.3e}], "
            f"new=[{summary['new_log_prob_min']:.3e},"
            f"{summary['new_log_prob_max']:.3e}], "
            f"ratio=[{summary['log_ratio_min']:.3e},"
            f"{summary['log_ratio_max']:.3e}]) "
            f"dist(loc_mean={summary['loc_mean']:.3e}, "
            f"loc=[{summary['loc_min']:.3e},{summary['loc_max']:.3e}], "
            f"scale_mean={summary['scale_mean']:.3e}, "
            f"scale=[{summary['scale_min']:.3e},{summary['scale_max']:.3e}])"
        )

    def get_observation_key(self) -> tuple:
        if getattr(self.parameters, "is_fast_inference_mode", False):
            return ("agents", "observation")
        if self.parameters.is_using_prioritized_marl:
            return ("agents", "info", "priority_observation")
        return ("agents", "info", "observation_without_actions")
