# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
import time

from termcolor import colored, cprint

import random

# Torch
import torch

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

# Data collection
from sigmarl.helper_training import SyncDataCollectorCustom
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data import TensorDictPrioritizedReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Utils
from tqdm import tqdm

import json
import os

from sigmarl.helper_common import Parameters
from sigmarl.env_factory import build_vmas_env
from sigmarl.helper_training import (
    SaveData,
    get_path_to_save_model,
    find_the_highest_reward_among_all_models,
    save,
    compute_td_error,
)
from sigmarl.modules.decision_making_module import DecisionMakingModule
from sigmarl.modules.optimization_module import OptimizationModule
from sigmarl.modules.priority_module import PriorityModule
from sigmarl.constants import ACTIVE_AGENT_PROFILE, validate_agent_profile


class MAPPOCAVs:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) for Connected and Autonomous Vehicles (CAVs).
    This class encapsulates the entire training process for MAPPO in a CAV environment.
    """

    def __init__(self, parameters: Parameters):
        """
        Initialize the MAPPOCAVs class.

        Args:
            parameters (Parameters): Configuration parameters for the training process.
        """
        self.parameters = parameters
        self._prepare_agent_profile_metadata()
        self.parameters.validate_policy_training_case()

        # Reproducibility
        torch.manual_seed(self.parameters.random_seed)
        random.seed(self.parameters.random_seed)
        self._motion_optimizer_step = 0
        self._motion_learning_rate = self.parameters.lr
        self._motion_slow_learning_rate = self.parameters.lr_slow

    def _prepare_agent_profile_metadata(self):
        """Stamp new runs and validate the metadata of loaded checkpoints."""
        if self.parameters.is_load_model:
            validate_agent_profile(self.parameters.agent_profile)
            if not self.parameters.is_continue_train:
                return

        self.parameters.agent_profile = ACTIVE_AGENT_PROFILE

    def train(self):
        """
        Execute the main training loop for MAPPO.

        Returns:
            tuple: Containing the environment, decision-making module, priority module (possibly), and updated parameters.
        """
        env = self._setup_environment()
        env.set_seed(self.parameters.random_seed)
        save_data = self._initialize_save_data()

        decision_making_module = self._setup_decision_making_module(env)

        optimization_module = self._setup_optimization_module(
            env, decision_making_module
        )
        priority_module = self._setup_priority_module(env)

        self._ensure_save_directory_exists()

        if (
            self.parameters.is_using_cbf_testing
            or self.parameters.is_using_cbf_training
        ):
            self.cbf_controllers = self._setup_cbf_qp_controller(env)
        else:
            self.cbf_controllers = None

        if (
            self.parameters.is_load_pretrained_motion_policy
            or self.parameters.is_load_pretrained_lambda_policy
        ):
            decision_making_module.load_pretrained_policy_modules()
        elif self.parameters.is_load_model:
            self._load_existing_model(
                decision_making_module, optimization_module, priority_module
            )
        else:
            if self.parameters.is_testing_mode:
                if self.parameters.nom_controller_type == "rl":
                    raise ValueError(
                        "[ERROR] Testing mode is ON but no model is loaded. The results may not be meaningful."
                    )

        # Loading or pretraining changes the behavior policy after the optional
        # predictor is constructed, so synchronize it before any rollout.
        decision_making_module.sync_opponent_predictor()

        if not self.parameters.is_continue_train:
            cprint("[INFO] Training will not continue.", "blue")
            return (
                env,
                decision_making_module,
                optimization_module,
                priority_module,
                self.cbf_controllers,
                self.parameters,
            )

        collector = self._setup_data_collector(
            env, decision_making_module, priority_module
        )
        replay_buffer = self._setup_replay_buffer()

        pbar = tqdm(total=self.parameters.n_iters, desc="epi_rew_mean = 0")

        t_start = time.time()
        i_iter = 0
        env.scenario.i_iter = i_iter  # To be used later in the scenario if needed
        if 0 in self.parameters.must_save_checkpoints:
            self._save_fixed_iteration_model(
                decision_making_module, optimization_module, priority_module, 0
            )
        for tensordict_data in collector:
            start_timer = time.perf_counter()
            i_iter += 1
            # print(f"[INFO] Starting iteration {i_iter}...")
            env.scenario.i_iter = i_iter  # To be used later in the scenario if needed
            self._process_tensordict_data(tensordict_data, env)
            self._compute_gae(tensordict_data, optimization_module, priority_module)
            self._update_priorities(tensordict_data)

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            phase = self._determine_training_phase(i_iter)

            # if i_iter % 10 == 0:
            # print(f"[INFO] Alternating training phase: {phase} at iteration {i_iter}")

            if phase == "joint":
                decision_making_module.policy.set_train_mode(
                    train_action=True,
                    train_lambda=True,
                    allow_action=self.parameters.uses_motion_policy,
                    allow_lambda=self.parameters.uses_lambda_policy,
                )
            elif phase == "action":
                decision_making_module.policy.set_train_mode(
                    train_action=True,
                    train_lambda=False,
                    allow_action=self.parameters.uses_motion_policy,
                    allow_lambda=self.parameters.uses_lambda_policy,
                )
            elif phase == "lambda":
                decision_making_module.policy.set_train_mode(
                    train_action=False,
                    train_lambda=True,
                    allow_action=self.parameters.uses_motion_policy,
                    allow_lambda=self.parameters.uses_lambda_policy,
                )

            if (
                priority_module
                and self.parameters.prioritization_method.lower() == "marl"
            ):
                priority_module.reset_diagnostics()

            self._train_epoch(
                optimization_module,
                priority_module,
                replay_buffer,
                env,
                phase,
            )

            if (
                priority_module
                and self.parameters.prioritization_method.lower() == "marl"
            ):
                pbar.write(priority_module.format_diagnostics(iteration=i_iter))

            # Keep the predictor frozen during collection and refresh it only
            # between optimization batches.
            decision_making_module.sync_opponent_predictor()
            collector.update_policy_weights_()

            self._log_and_save(
                tensordict_data,
                decision_making_module,
                optimization_module,
                priority_module,
                save_data,
                pbar,
            )
            if i_iter in self.parameters.must_save_checkpoints:
                self._save_fixed_iteration_model(
                    decision_making_module,
                    optimization_module,
                    priority_module,
                    i_iter,
                )
            self.parameters.update_reward_distribution_alpha_after_iteration(i_iter)
            self._update_learning_rate(optimization_module, priority_module, pbar)
            pbar.update()  # Increment the progress bar
            print(f"[MAPPO] training time {time.perf_counter() - start_timer}")

        self._save_final_model(
            decision_making_module, optimization_module, priority_module
        )
        self._print_training_summary(t_start)

        return (
            env,
            decision_making_module,
            optimization_module,
            priority_module,
            self.cbf_controllers,
            self.parameters,
        )

    def _setup_environment(self):
        """Set up the training environment."""
        return build_vmas_env(self.parameters)

    def _initialize_save_data(self):
        """Initialize the data structure for saving training progress."""
        if (
            self.parameters.is_load_model or self.parameters.is_load_final_model
        ) and self.parameters.is_continue_train:
            history_data = self._find_and_load_existing_save_data()
            if history_data is not None:
                return history_data
        return SaveData(
            parameters=self.parameters,
            episode_reward_mean_list=[],
            parameter_independent_reward_mean_list=[],
        )

    def _find_and_load_existing_save_data(self):
        """Attempt to restore prior training history from existing JSON files."""
        if not os.path.exists(self.parameters.where_to_save):
            return None

        highest_reward = find_the_highest_reward_among_all_models(
            self.parameters.where_to_save
        )
        candidates = []
        if highest_reward != float("-inf"):
            candidates.append(f"reward{highest_reward:.2f}_data.json")

        for fname in sorted(os.listdir(self.parameters.where_to_save)):
            if fname.endswith("_data.json") and fname not in candidates:
                candidates.append(fname)

        for fname in candidates:
            json_path = os.path.join(self.parameters.where_to_save, fname)
            if os.path.isfile(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as file:
                        raw_data = json.load(file)
                    episode_rewards = raw_data.get("episode_reward_mean_list", [])
                    param_indep_rewards = raw_data.get(
                        "parameter_independent_reward_mean_list", []
                    )
                    cprint(
                        f"[INFO] Restored training history ({len(episode_rewards)} iterations) from '{json_path}'",
                        "blue",
                    )
                    if (
                        self.parameters.parameter_independent_reward_intermediate
                        is None
                        and "parameters" in raw_data
                        and isinstance(raw_data["parameters"], dict)
                    ):
                        self.parameters.parameter_independent_reward_intermediate = (
                            raw_data["parameters"].get(
                                "parameter_independent_reward_intermediate"
                            )
                        )
                    return SaveData(
                        parameters=self.parameters,
                        episode_reward_mean_list=list(episode_rewards),
                        parameter_independent_reward_mean_list=list(param_indep_rewards)
                        if param_indep_rewards is not None
                        else [],
                        task_metrics_list=list(raw_data.get("task_metrics_list", [])),
                    )
                except Exception as e:
                    cprint(
                        f"[WARN] Failed to load prior training data from '{json_path}': {e}",
                        "yellow",
                    )
        return None

    def _setup_decision_making_module(self, env):
        """Set up the decision-making module."""
        return DecisionMakingModule.from_env(env, env.scenario.parameters)

    def _setup_optimization_module(self, env, decision_module: DecisionMakingModule):
        mappo = True
        return OptimizationModule(env=env, decision_module=decision_module, mappo=mappo)

    def _setup_priority_module(self, env):
        """Set up the priority module if prioritized MARL is enabled."""
        if self.parameters.is_using_prioritized_marl:
            return PriorityModule(env=env, mappo=True)
        else:
            return None

    def _determine_training_phase(self, i_iter: int):
        if not self.parameters.is_joint_training:
            if self.parameters.is_motion_only:
                return "action"
            if self.parameters.is_lambda_only:
                return "lambda"

        if self.parameters.is_alternating_action_lambda_training:
            action_iters = self.parameters.alternating_train_action_iters
            lambda_iters = self.parameters.alternating_train_lambda_iters
            cycle = action_iters + lambda_iters

            if cycle <= 0:
                raise ValueError("alternating training cycle length must be positive.")

            phase_idx = (i_iter - 1) % cycle

            if phase_idx < action_iters:
                return "action"
            return "lambda"
        return "joint"

    def _ensure_save_directory_exists(self):
        """Ensure the directory for saving models exists."""
        if not os.path.exists(self.parameters.where_to_save):
            os.makedirs(self.parameters.where_to_save)
            print(
                colored(
                    "[INFO] Created a new directory to save the trained model:", "black"
                ),
                colored(f"{self.parameters.where_to_save}", "blue"),
            )

    def _load_existing_model(
        self, decision_making_module, optimization_module, priority_module
    ):
        """Load an existing model if find one."""
        if getattr(self.parameters, "is_cross_iter_test", False):
            self._load_cross_iter_model(decision_making_module, priority_module)
            return

        highest_reward = find_the_highest_reward_among_all_models(
            self.parameters.where_to_save
        )
        self.parameters.episode_reward_mean_current = highest_reward
        if highest_reward is not float("-inf"):
            if (
                self.parameters.is_continue_train
                and self.parameters.episode_reward_intermediate == -1e3
            ):
                self.parameters.episode_reward_intermediate = highest_reward
            self._load_model(
                decision_making_module, optimization_module, priority_module
            )
        elif self.parameters.is_load_final_model:
            final_policy_path = os.path.join(
                self.parameters.where_to_save, "final_policy.pth"
            )
            if os.path.exists(final_policy_path):
                self._load_model(
                    decision_making_module, optimization_module, priority_module
                )
            else:
                raise ValueError(f"No final model found at '{final_policy_path}'.")
        else:
            raise ValueError("No valid model found in the specified directory.")

    def _load_model(self, decision_making_module, optimization_module, priority_module):
        """Load either the final or an intermediate model based on parameters."""
        if self.parameters.is_load_final_model:
            self._load_final_model(
                decision_making_module, optimization_module, priority_module
            )
        else:
            self._load_intermediate_model(
                decision_making_module, optimization_module, priority_module
            )

    def _load_final_model(
        self,
        decision_making_module: DecisionMakingModule,
        optimization_module: OptimizationModule,
        priority_module=None,
    ):
        """Load the final model."""
        final_policy_path = os.path.join(
            self.parameters.where_to_save, "final_policy.pth"
        )
        decision_making_module.policy.load_state_dict(
            torch.load(
                final_policy_path,
                weights_only=True,
            )
        )
        cprint(f"[INFO] Loaded the final model '{final_policy_path}'", "red")
        if priority_module and self.parameters.prioritization_method.lower() == "marl":
            final_priority_policy_path = os.path.join(
                self.parameters.where_to_save, "final_priority_policy.pth"
            )
            if os.path.exists(final_priority_policy_path):
                priority_module.policy.load_state_dict(
                    torch.load(
                        final_priority_policy_path,
                        weights_only=True,
                    )
                )
                cprint(
                    f"[INFO] Loaded the final priority model '{final_priority_policy_path}'",
                    "red",
                )

        if self.parameters.is_continue_train:
            final_critic_path = os.path.join(
                self.parameters.where_to_save, "final_critic.pth"
            )
            if os.path.exists(final_critic_path):
                optimization_module.critic.load_state_dict(
                    torch.load(final_critic_path, weights_only=True)
                )
                cprint(
                    f"[INFO] Loaded the final critic '{final_critic_path}' for continued training",
                    "red",
                )
            if (
                priority_module
                and self.parameters.prioritization_method.lower() == "marl"
            ):
                final_priority_critic_path = os.path.join(
                    self.parameters.where_to_save, "final_priority_critic.pth"
                )
                if os.path.exists(final_priority_critic_path):
                    priority_module.critic.load_state_dict(
                        torch.load(final_priority_critic_path, weights_only=True)
                    )
                    cprint(
                        f"[INFO] Loaded the final priority critic '{final_priority_critic_path}'",
                        "red",
                    )

    def _load_intermediate_model(
        self,
        decision_making_module: DecisionMakingModule,
        optimization_module: OptimizationModule,
        priority_module=None,
    ):
        """Load an intermediate model."""
        (
            PATH_POLICY,
            PATH_CRITIC,
            PATH_MOTION_POLICY,
            PATH_LAMBDA_POLICY,
            PATH_PRIORITY_POLICY,
            PATH_PRIORITY_CRITIC,
            PATH_FIG,
            PATH_JSON,
        ) = get_path_to_save_model(parameters=self.parameters)

        decision_making_module.policy.load_state_dict(
            torch.load(PATH_POLICY, weights_only=True)
        )
        cprint(f"[INFO] Loaded the intermediate model '{PATH_POLICY}'", "blue")

        if priority_module and self.parameters.prioritization_method.lower() == "marl":
            priority_module.policy.load_state_dict(
                torch.load(PATH_PRIORITY_POLICY, weights_only=True)
            )
            cprint(
                f"[INFO] Loaded the intermediate priority model '{PATH_PRIORITY_POLICY}'",
                "blue",
            )

        if self.parameters.is_continue_train:
            cprint("[INFO] Training will continue with the loaded model.", "red")
            optimization_module.critic.load_state_dict(
                torch.load(PATH_CRITIC, weights_only=True)
            )
            if (
                priority_module
                and self.parameters.prioritization_method.lower() == "marl"
            ):
                priority_module.critic.load_state_dict(
                    torch.load(PATH_PRIORITY_CRITIC, weights_only=True)
                )

    def _load_cross_iter_model(self, decision_making_module, priority_module=None):
        """Load ego checkpoint for cross_iter_ego_indices and peer checkpoint for other agents."""
        cross_iter_ego_indices = self.parameters.cross_iter_ego_indices
        if cross_iter_ego_indices is None:
            primary_ego = self.parameters.observed_agents_render_ego_index
            cross_iter_ego_indices = (primary_ego,)
        elif isinstance(cross_iter_ego_indices, int):
            cross_iter_ego_indices = (cross_iter_ego_indices,)
        else:
            cross_iter_ego_indices = tuple(cross_iter_ego_indices)
        ego_indices_set = set(cross_iter_ego_indices)
        primary_ego = cross_iter_ego_indices[0] if cross_iter_ego_indices else 0

        ego_policy_path = getattr(self.parameters, "ego_policy_path", None)
        peer_policy_path = getattr(self.parameters, "peer_policy_path", None)

        if ego_policy_path is None or peer_policy_path is None:
            ego_iter = getattr(self.parameters, "ego_iter", None)
            peer_iter = getattr(self.parameters, "peer_iter", None)
            if ego_iter is not None and peer_iter is not None:
                ego_policy_path = os.path.join(
                    self.parameters.where_to_save,
                    "checkpoints_cross_iter",
                    f"policy_iter{ego_iter}.pth",
                )
                peer_policy_path = os.path.join(
                    self.parameters.where_to_save,
                    "checkpoints_cross_iter",
                    f"policy_iter{peer_iter}.pth",
                )
            else:
                raise ValueError(
                    "Cross-iteration testing requires ego and peer policy paths or iteration numbers."
                )

        if not os.path.isfile(ego_policy_path):
            raise FileNotFoundError(
                f"Ego policy checkpoint not found at: {ego_policy_path}"
            )
        if not os.path.isfile(peer_policy_path):
            raise FileNotFoundError(
                f"Peer policy checkpoint not found at: {peer_policy_path}"
            )

        ego_state = torch.load(ego_policy_path, weights_only=True)
        peer_state = torch.load(peer_policy_path, weights_only=True)

        n_agents = self.parameters.n_agents
        current_state = decision_making_module.policy.state_dict()
        new_state = {}
        for k, v in current_state.items():
            if (
                isinstance(v, torch.Tensor)
                and k in ego_state
                and isinstance(ego_state[k], torch.Tensor)
                and k in peer_state
                and isinstance(peer_state[k], torch.Tensor)
            ):
                s_ego = ego_state[k]
                s_peer = peer_state[k]
                if v.ndim == s_ego.ndim + 1 and v.shape[0] == n_agents:
                    new_state[k] = torch.stack(
                        [
                            s_ego if i in ego_indices_set else s_peer
                            for i in range(n_agents)
                        ],
                        dim=0,
                    )
                else:
                    new_state[k] = s_ego if (0 in ego_indices_set) else s_peer
            else:
                new_state[k] = v

        decision_making_module.policy.load_state_dict(new_state)
        ego_desc = (
            f"agent {primary_ego}"
            if len(ego_indices_set) == 1
            else f"agents {sorted(ego_indices_set)} (primary: {primary_ego})"
        )
        cprint(
            f"[INFO] Loaded cross-iteration policy: {ego_desc} from '{ego_policy_path}', "
            f"peers from '{peer_policy_path}'",
            "blue",
        )

    def _setup_data_collector(self, env, decision_making_module, priority_module):
        """Set up the data collector for gathering experience."""
        return SyncDataCollectorCustom(
            env,
            decision_making_module.policy,
            priority_module=priority_module,
            cbf_controllers=self.cbf_controllers,
            opponent_predictor=decision_making_module.opponent_predictor,
            device=self.parameters.device,
            storing_device=self.parameters.device,
            frames_per_batch=self.parameters.frames_per_batch,
            total_frames=self.parameters.total_frames,
            reset_at_each_iter=getattr(self.parameters, "reset_at_each_iter", False),
        )

    def _setup_replay_buffer(self):
        """Set up the replay buffer for storing experiences."""
        if self.parameters.is_prb:
            return TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.6,
                storage=LazyTensorStorage(
                    self.parameters.frames_per_batch, device=self.parameters.device
                ),
                batch_size=self.parameters.minibatch_size,
                priority_key="td_error",
            )
        else:
            return ReplayBuffer(
                storage=LazyTensorStorage(
                    self.parameters.frames_per_batch, device=self.parameters.device
                ),
                sampler=SamplerWithoutReplacement(),
                batch_size=self.parameters.minibatch_size,
            )

    def _process_tensordict_data(self, tensordict_data, env):
        """Process the collected tensordict data."""
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        if self.parameters.is_joint_training:
            total_reward = tensordict_data.get(("next", "agents", "reward"))
            lambda_aux_reward = tensordict_data.get(
                ("next", "agents", "info", "rew_cbf_lambda")
            )
            tensordict_data.set(
                ("next",) + OptimizationModule.ACTION_REWARD_KEY,
                total_reward - lambda_aux_reward,
            )
            tensordict_data.set(
                ("next",) + OptimizationModule.LAMBDA_REWARD_KEY,
                total_reward,
            )

    def _compute_gae(
        self,
        tensordict_data,
        optimization_module: OptimizationModule,
        priority_module: PriorityModule,
    ):
        """Compute Generalized Advantage Estimation (GAE)."""
        with torch.no_grad():
            if self.parameters.is_joint_training:
                optimization_module.action_GAE(
                    tensordict_data,
                    params=optimization_module.action_loss_module.critic_network_params,
                    target_params=optimization_module.action_loss_module.target_critic_network_params,
                )
                optimization_module.lambda_GAE(
                    tensordict_data,
                    params=optimization_module.lambda_loss_module.critic_network_params,
                    target_params=optimization_module.lambda_loss_module.target_critic_network_params,
                )
            else:
                optimization_module.GAE(
                    tensordict_data,
                    params=optimization_module.loss_module.critic_network_params,
                    target_params=optimization_module.loss_module.target_critic_network_params,
                )
            if (
                priority_module
                and self.parameters.prioritization_method.lower() == "marl"
            ):
                priority_module.GAE(
                    tensordict_data,
                    params=priority_module.loss_module.critic_network_params,
                    target_params=priority_module.loss_module.target_critic_network_params,
                )

    def _update_priorities(self, tensordict_data):
        """Update priorities for prioritized replay buffer."""
        if self.parameters.is_prb:
            td_error = compute_td_error(tensordict_data, gamma=0.9)
            tensordict_data.set(("td_error"), td_error)
            assert (
                tensordict_data["td_error"].min() >= 0
            ), "TD error must be greater than 0"

    def _train_epoch(
        self,
        optimization_module: OptimizationModule,
        priority_module,
        replay_buffer,
        env,
        phase,
    ):
        """Train for one epoch."""
        for _ in range(self.parameters.num_epochs):
            for _ in range(
                self.parameters.frames_per_batch // self.parameters.minibatch_size
            ):
                mini_batch_data, info = replay_buffer.sample(return_info=True)
                self._train_on_batch(
                    optimization_module,
                    priority_module,
                    mini_batch_data,
                    env,
                    phase,
                )
                if self.parameters.is_prb:
                    self._update_priorities_after_training(
                        optimization_module,
                        priority_module,
                        mini_batch_data,
                        replay_buffer,
                    )

    # TODO: remove
    # def _train_on_batch(
    #     self, optimization_module: OptimizationModule, priority_module, mini_batch_data
    # ):
    #     # """Train on a single batch of data."""
    #     loss_vals = optimization_module.loss_module(mini_batch_data)
    #     loss_value = (
    #         loss_vals["loss_objective"]
    #         + loss_vals["loss_critic"]
    #         + loss_vals["loss_entropy"]
    #     )
    #     assert not loss_value.isnan().any() and not loss_value.isinf().any()

    #     optimization_module.optim.zero_grad()
    #     loss_value.backward()

    #     if (
    #         self.parameters.is_learn_cbf_lambda
    #         and self.parameters.is_alternating_action_lambda_training
    #     ):
    #         trainable_params = [
    #             p
    #             for p in optimization_module.loss_module.parameters()
    #             if p.requires_grad
    #         ]
    #         torch.nn.utils.clip_grad_norm_(
    #             trainable_params,
    #             self.parameters.max_grad_norm,
    #         )
    #     else:
    #         torch.nn.utils.clip_grad_norm_(
    #             optimization_module.loss_module.parameters(),
    #             self.parameters.max_grad_norm,
    #         )

    #     optimization_module.optim.step()

    def _train_on_batch(
        self,
        optimization_module: OptimizationModule,
        priority_module,
        mini_batch_data,
        env,
        phase,
    ):
        if self.parameters.is_joint_training:
            loss_tds = []
            if phase in {"joint", "action"}:
                loss_tds.append(optimization_module.action_loss_module(mini_batch_data))
            if phase in {"joint", "lambda"}:
                loss_tds.append(optimization_module.lambda_loss_module(mini_batch_data))

            if not loss_tds:
                raise ValueError(f"Unsupported training phase: {phase}")

            loss_value = 0.0
            for loss_td in loss_tds:
                loss_value = (
                    loss_value
                    + loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + loss_td["loss_entropy"]
                )
        else:
            loss_vals = optimization_module.loss_module(mini_batch_data)
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

        # TODO: remove after debugging
        # print("action_log_prob finite:",
        #     torch.isfinite(mini_batch_data.get(("agents", "action_log_prob"))).all().item())
        # print("lambda_log_prob finite:",
        #     torch.isfinite(mini_batch_data.get(("agents", "cbf", "lambda_log_prob"))).all().item())
        # print("action finite:",
        #     torch.isfinite(mini_batch_data.get(env.action_key)).all().item())
        # print("lambda finite:",
        #     torch.isfinite(mini_batch_data.get(("agents", "cbf", "lambda"))).all().item())
        # print("lambda min/max:",
        #     mini_batch_data.get(("agents", "cbf", "lambda")).min().item(),
        #     mini_batch_data.get(("agents", "cbf", "lambda")).max().item())

        if not torch.isfinite(loss_value).all():
            if self.parameters.is_joint_training:
                if phase in {"joint", "action"}:
                    action_loss_vals = optimization_module.action_loss_module(
                        mini_batch_data
                    )
                    print("action loss_objective:", action_loss_vals["loss_objective"])
                    print("action loss_critic:", action_loss_vals["loss_critic"])
                    print("action loss_entropy:", action_loss_vals["loss_entropy"])
                if phase in {"joint", "lambda"}:
                    lambda_loss_vals = optimization_module.lambda_loss_module(
                        mini_batch_data
                    )
                    print("lambda loss_objective:", lambda_loss_vals["loss_objective"])
                    print("lambda loss_critic:", lambda_loss_vals["loss_critic"])
                    print("lambda loss_entropy:", lambda_loss_vals["loss_entropy"])
            else:
                print("loss_objective:", loss_vals["loss_objective"])
                print("loss_critic:", loss_vals["loss_critic"])
                print("loss_entropy:", loss_vals["loss_entropy"])

            if ("agents", "action_log_prob") in mini_batch_data.keys(True, True):
                alp = mini_batch_data.get(("agents", "action_log_prob"))
                print(
                    "stored action_log_prob min/max:",
                    alp.min().item(),
                    alp.max().item(),
                )

            if ("agents", "cbf", "lambda_log_prob") in mini_batch_data.keys(True, True):
                llp = mini_batch_data.get(("agents", "cbf", "lambda_log_prob"))
                print(
                    "stored lambda_log_prob min/max:",
                    llp.min().item(),
                    llp.max().item(),
                )

            raise ValueError("Non-finite PPO loss.")

        assert not loss_value.isnan().any() and not loss_value.isinf().any()

        self._set_motion_learning_rates_for_step(optimization_module)
        optimization_module.optim.zero_grad()
        loss_value.backward()

        if self.parameters.is_joint_training:
            trainable_params = [
                p
                for p in optimization_module.optim.param_groups[0]["params"]
                if p.requires_grad
            ]
        else:
            trainable_params = [
                p
                for p in optimization_module.loss_module.parameters()
                if p.requires_grad
            ]
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            self.parameters.max_grad_norm,
        )

        optimization_module.optim.step()
        if self.parameters.is_multi_timescale_learning:
            self._motion_optimizer_step += 1

        if priority_module and self.parameters.prioritization_method.lower() == "marl":
            priority_module.compute_losses_and_optimize(mini_batch_data)

    def _update_priorities_after_training(
        self,
        optimization_module: OptimizationModule,
        priority_module: PriorityModule,
        mini_batch_data,
        replay_buffer,
    ):
        """Update priorities in the replay buffer after training."""
        with torch.no_grad():
            if self.parameters.is_joint_training:
                optimization_module.action_GAE(
                    mini_batch_data,
                    params=optimization_module.action_loss_module.critic_network_params,
                    target_params=optimization_module.action_loss_module.target_critic_network_params,
                )
                optimization_module.lambda_GAE(
                    mini_batch_data,
                    params=optimization_module.lambda_loss_module.critic_network_params,
                    target_params=optimization_module.lambda_loss_module.target_critic_network_params,
                )
            else:
                optimization_module.GAE(
                    mini_batch_data,
                    params=optimization_module.loss_module.critic_network_params,
                    target_params=optimization_module.loss_module.target_critic_network_params,
                )
            if (
                priority_module
                and self.parameters.prioritization_method.lower() == "marl"
            ):
                priority_module.GAE(
                    mini_batch_data,
                    params=priority_module.loss_module.critic_network_params,
                    target_params=priority_module.loss_module.target_critic_network_params,
                )
        new_td_errors = compute_td_error(mini_batch_data, gamma=0.9)
        mini_batch_data.set("td_error", new_td_errors)
        replay_buffer.update_tensordict_priority(mini_batch_data)

    def _log_and_save(
        self,
        tensordict_data,
        decision_making_module,
        optimization_module,
        priority_module,
        save_data,
        pbar,
    ):
        """Log training progress and save intermediate models."""
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean_current = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done]
            .mean()
            .item()
        )
        episode_reward_mean_current = round(episode_reward_mean_current, 2)
        save_data.episode_reward_mean_list.append(episode_reward_mean_current)

        parameter_independent_reward_mean_current = (
            self._compute_parameter_independent_reward_mean(tensordict_data, done)
        )
        description = f"Episode mean rew.={episode_reward_mean_current:.2f}"
        if parameter_independent_reward_mean_current is not None:
            parameter_independent_reward_mean_current = round(
                parameter_independent_reward_mean_current, 2
            )
            save_data.parameter_independent_reward_mean_list.append(
                parameter_independent_reward_mean_current
            )
            description += (
                ", param.-indep. rew.="
                f"{parameter_independent_reward_mean_current:.2f}"
            )
        pbar.set_description(description, refresh=False)
        task_metrics = self._compute_task_metrics(tensordict_data)
        if task_metrics is not None:
            save_data.task_metrics_list.append(task_metrics)

        if self.parameters.is_save_intermediate_model:
            self._save_intermediate_model(
                decision_making_module,
                optimization_module,
                priority_module,
                save_data,
                episode_reward_mean_current,
                parameter_independent_reward_mean_current,
            )

    def _compute_task_metrics(self, tensordict_data):
        """Task metrics of the training rollouts of one iteration, over all
        agent-steps of the batch. They do not depend on the training reward,
        so runs with different rewards on the same map can be compared:

        - task_reward: progress + goal + collision and deadlock terms of the
          `sparse` reward per agent and second (the parameter-independent
          reward plus the dense progress term);
        - speed_ratio: mean progress speed along the route over the speed
          limit;
        - goals, veh_collisions, lane_collisions: events per agent-minute
          (agent-steps with a goal or in collision; colliding agents are
          reset).

        Returns None without the reward terms in the info (old scenarios).
        """
        from sigmarl.scenarios.road_traffic import TASK_METRIC_REWARDS

        info_keys = {
            name: ("next", "agents", "info", name) for name in TASK_METRIC_REWARDS
        }
        available_keys = tensordict_data.keys(include_nested=True)
        if any(key not in available_keys for key in info_keys.values()):
            return None
        terms = {
            name: tensordict_data.get(key).float() for name, key in info_keys.items()
        }
        dt = float(self.parameters.dt)
        n_agent_steps = terms["rew_progress"].numel()
        agent_minutes = n_agent_steps * dt / 60.0
        task_reward = (
            terms["rew_progress"]
            + terms["rew_reach_goal"]
            + terms["rew_collide_other_agents"]
            + terms["rew_collide_lane"]
            + terms["rew_deadlock"]
        )
        metrics = {
            "task_reward": task_reward.mean().item() / dt,
            "speed_ratio": terms["rew_progress"].mean().item()
            / float(self.parameters.reward_progress),
            "goals": (terms["rew_reach_goal"] > 0).sum().item() / agent_minutes,
            "veh_collisions": (terms["rew_collide_other_agents"] < 0).sum().item()
            / agent_minutes,
            "lane_collisions": (terms["rew_collide_lane"] < 0).sum().item()
            / agent_minutes,
        }
        return {key: round(value, 5) for key, value in metrics.items()}

    def _compute_parameter_independent_reward_mean(self, tensordict_data, done):
        """Compute mean unclamped sparse reward per agent per second."""
        available_keys = tensordict_data.keys(include_nested=True)
        reward_info_key = ("next", "agents", "info", "rew_parameter_independent")
        step_time_key = ("next", "agents", "info", "step_time")
        if reward_info_key not in available_keys or step_time_key not in available_keys:
            return None
        sparse_reward = tensordict_data.get(reward_info_key).float()
        step_time = tensordict_data.get(step_time_key).float()

        if done.ndim == sparse_reward.ndim + 1 and done.shape[-1] == 1:
            done = done.squeeze(-1)
        if sparse_reward.ndim == done.ndim + 1 and sparse_reward.shape[-1] == 1:
            sparse_reward = sparse_reward.squeeze(-1)
            step_time = step_time.squeeze(-1)

        time_dim = len(tensordict_data.batch_size) - 1
        sparse_reward_by_time = sparse_reward.movedim(time_dim, 0)
        step_time_by_time = step_time.movedim(time_dim, 0)
        done_by_time = done.movedim(time_dim, 0)

        reward_accumulator = torch.zeros_like(sparse_reward_by_time[0])
        duration_accumulator = torch.zeros_like(step_time_by_time[0])
        completed_rewards = []

        eps = torch.finfo(step_time.dtype).eps
        for reward_t, step_time_t, done_t in zip(
            sparse_reward_by_time, step_time_by_time, done_by_time
        ):
            reward_accumulator = reward_accumulator + reward_t
            duration_accumulator = duration_accumulator + step_time_t
            if done_t.any():
                completed_rewards.append(
                    reward_accumulator[done_t]
                    / duration_accumulator[done_t].clamp_min(eps)
                )
                reward_accumulator = torch.where(
                    done_t, torch.zeros_like(reward_accumulator), reward_accumulator
                )
                duration_accumulator = torch.where(
                    done_t, torch.zeros_like(duration_accumulator), duration_accumulator
                )

        if not completed_rewards:
            return float("nan")
        return (
            torch.cat([reward.reshape(-1) for reward in completed_rewards])
            .mean()
            .item()
        )

    # TODO: remove
    # def _save_intermediate_model(
    #     self,
    #     decision_making_module,
    #     optimization_module,
    #     priority_module,
    #     save_data,
    #     episode_reward_mean_current,
    # ):
    #     """Save an intermediate model if it performs better than the previous best."""
    #     self.parameters.episode_reward_mean_current = episode_reward_mean_current

    #     if episode_reward_mean_current > self.parameters.episode_reward_intermediate:
    #         # Save intermediate model only when its episode mean reward is better than the saved one
    #         self.parameters.episode_reward_intermediate = episode_reward_mean_current
    #         save(
    #             self.parameters,
    #             save_data,
    #             decision_making_module,
    #             optimization_module,
    #             priority_module if priority_module else None,
    #         )
    #     else:
    #         # Otherwise only save parameters and episode mean reward values
    #         self.parameters.episode_reward_mean_current = (
    #             self.parameters.episode_reward_intermediate
    #         )
    #         save(self.parameters, save_data, None, None, None)

    def _save_intermediate_model(
        self,
        decision_making_module,
        optimization_module,
        priority_module,
        save_data,
        episode_reward_mean_current,
        parameter_independent_reward_mean_current,
    ):
        """Save an intermediate model if it performs better than the previous best."""
        self.parameters.episode_reward_mean_current = episode_reward_mean_current

        if episode_reward_mean_current > self.parameters.episode_reward_intermediate:
            self.parameters.episode_reward_intermediate = episode_reward_mean_current
            self.parameters.parameter_independent_reward_intermediate = (
                parameter_independent_reward_mean_current
            )
            save(
                self.parameters,
                save_data,
                decision_making_module,
                optimization_module,
                priority_module if priority_module else None,
            )
        else:
            self.parameters.episode_reward_mean_current = (
                self.parameters.episode_reward_intermediate
            )
            save(self.parameters, save_data, None, None, None)

    def _set_motion_learning_rates_for_step(self, optimization_module):
        """Apply the current multi-timescale rates before a motion PPO step."""

        if not self.parameters.is_multi_timescale_learning:
            return
        fast_agent = (
            self._motion_optimizer_step
            // self.parameters.multi_timescale_switching_period
        ) % self.parameters.n_agents
        optimization_module.set_motion_learning_rates(
            fast_learning_rate=self._motion_learning_rate,
            slow_learning_rate=self._motion_slow_learning_rate,
            fast_agent=fast_agent,
            critic_learning_rate=self._motion_learning_rate,
        )

    def _update_learning_rate(
        self,
        optimization_module: OptimizationModule,
        priority_module,
        pbar,
    ):
        """Apply the linear learning-rate schedule to every learned policy."""
        training_fraction_remaining = 1 - (pbar.n / self.parameters.n_iters)
        motion_learning_rate = (
            self.parameters.lr_min
            + (self.parameters.lr - self.parameters.lr_min)
            * training_fraction_remaining
        )

        if self.parameters.is_multi_timescale_learning:
            self._motion_learning_rate = motion_learning_rate
            self._motion_slow_learning_rate = (
                self.parameters.lr_slow_min
                + (self.parameters.lr_slow - self.parameters.lr_slow_min)
                * training_fraction_remaining
            )
            self._set_motion_learning_rates_for_step(optimization_module)
        else:
            for param_group in optimization_module.optim.param_groups:
                param_group["lr"] = motion_learning_rate

        priority_learning_rate = None
        if priority_module is not None and hasattr(priority_module, "optim"):
            priority_learning_rate = (
                self.parameters.lr_priority_min
                + (self.parameters.lr_priority - self.parameters.lr_priority_min)
                * training_fraction_remaining
            )
            for param_group in priority_module.optim.param_groups:
                param_group["lr"] = priority_learning_rate

        if pbar.n % 10 == 0:
            if self.parameters.is_multi_timescale_learning:
                fast_agent = (
                    self._motion_optimizer_step
                    // self.parameters.multi_timescale_switching_period
                ) % self.parameters.n_agents
                message = (
                    f"Learning rates updated: motion_fast={motion_learning_rate}, "
                    f"motion_slow={self._motion_slow_learning_rate}, "
                    f"fast_agent={fast_agent}"
                )
            else:
                message = f"Learning rates updated: motion={motion_learning_rate}"
            if priority_learning_rate is not None:
                message += f", priority={priority_learning_rate}"
            print(f"{message}.")

    def _save_final_model(
        self,
        decision_making_module: DecisionMakingModule,
        optimization_module: OptimizationModule,
        priority_module,
    ):
        """Save the final trained model."""
        torch.save(
            decision_making_module.policy.state_dict(),
            os.path.join(self.parameters.where_to_save, "final_policy.pth"),
        )
        torch.save(
            optimization_module.critic.state_dict(),
            os.path.join(self.parameters.where_to_save, "final_critic.pth"),
        )

        if self.parameters.is_joint_training:
            torch.save(
                decision_making_module.policy.state_dict(),
                os.path.join(self.parameters.where_to_save, "final_joint_policy.pth"),
            )
            torch.save(
                optimization_module.action_critic.state_dict(),
                os.path.join(self.parameters.where_to_save, "final_action_critic.pth"),
            )
            torch.save(
                optimization_module.lambda_critic.state_dict(),
                os.path.join(self.parameters.where_to_save, "final_lambda_critic.pth"),
            )
            torch.save(
                decision_making_module.policy.motion_policy_state_dict(),
                os.path.join(self.parameters.where_to_save, "final_motion_policy.pth"),
            )
            torch.save(
                decision_making_module.policy.lambda_policy_state_dict(),
                os.path.join(self.parameters.where_to_save, "final_lambda_policy.pth"),
            )
        elif self.parameters.is_motion_only and hasattr(
            decision_making_module.policy, "motion_policy_state_dict"
        ):
            torch.save(
                decision_making_module.policy.motion_policy_state_dict(),
                os.path.join(self.parameters.where_to_save, "final_motion_policy.pth"),
            )
        elif self.parameters.is_lambda_only and hasattr(
            decision_making_module.policy, "lambda_policy_state_dict"
        ):
            torch.save(
                decision_making_module.policy.lambda_policy_state_dict(),
                os.path.join(self.parameters.where_to_save, "final_lambda_policy.pth"),
            )

        if (
            self.parameters.is_using_prioritized_marl
            and self.parameters.prioritization_method.lower() == "marl"
        ):
            torch.save(
                priority_module.policy.state_dict(),
                os.path.join(
                    self.parameters.where_to_save, "final_priority_policy.pth"
                ),
            )
            torch.save(
                priority_module.critic.state_dict(),
                os.path.join(
                    self.parameters.where_to_save, "final_priority_critic.pth"
                ),
            )

        print(
            colored("[INFO] All files have been saved under:", "black"),
            colored(f"{self.parameters.where_to_save}", "red"),
        )

    def _save_fixed_iteration_model(
        self,
        decision_making_module: DecisionMakingModule,
        optimization_module: OptimizationModule,
        priority_module,
        i_iter: int,
    ):
        """Save checkpoints at specified iterations into checkpoints_cross_iter folder."""
        cross_iter_dir = os.path.join(
            self.parameters.where_to_save, "checkpoints_cross_iter"
        )
        os.makedirs(cross_iter_dir, exist_ok=True)

        policy_path = os.path.join(cross_iter_dir, f"policy_iter{i_iter}.pth")
        critic_path = os.path.join(cross_iter_dir, f"critic_iter{i_iter}.pth")
        torch.save(decision_making_module.policy.state_dict(), policy_path)
        torch.save(optimization_module.critic.state_dict(), critic_path)

        if (
            self.parameters.is_using_prioritized_marl
            and self.parameters.prioritization_method.lower() == "marl"
            and priority_module
            and priority_module.policy
            and priority_module.critic
        ):
            torch.save(
                priority_module.policy.state_dict(),
                os.path.join(cross_iter_dir, f"priority_policy_iter{i_iter}.pth"),
            )
            torch.save(
                priority_module.critic.state_dict(),
                os.path.join(cross_iter_dir, f"priority_critic_iter{i_iter}.pth"),
            )

        cprint(
            f"[INFO] Saved fixed iteration {i_iter} model to '{policy_path}'",
            "green",
        )

    def _print_training_summary(self, t_start):
        """Print a summary of the training process."""
        training_duration = (time.time() - t_start) / 3600
        cprint(f"[INFO] Training duration: {training_duration:.2f} hours.", "blue")

    def _setup_cbf_qp_controller(self, env):
        """Set up the cbf_qp_controller if cbf constrained MARL is enabled."""

        # elif self.parameters.is_using_cbf_testing:

        #     pass
        if self.parameters.is_proxsuite_qp:
            from sigmarl.cbf_qp_proxsuite import CBFQP

            print("[INFO] Using ProxSuite QP solver.")
        else:
            from sigmarl.cbf_qp import CBFQP

            print("[INFO] Using CVXPY solver.")

        if self.parameters.is_using_centralized_cbf:
            # Each environment has one CBF-based controller
            cbf_controllers = [
                CBFQP(env=env, env_idx=env_idx)
                for env_idx in range(self.parameters.num_vmas_envs)
            ]
            return cbf_controllers
        else:
            # Each agent in each environment has one fully decentralized CBF-QP.
            cbf_controllers = [
                [
                    CBFQP(env=env, env_idx=env_idx, agent_idx=agent_idx)
                    for agent_idx in range(self.parameters.n_agents)
                ]
                for env_idx in range(self.parameters.num_vmas_envs)
            ]
            return cbf_controllers

    def _load_existing_model_for_cbf(self, decision_making_module, priority_module):
        """Load RL decision making and priority policy model"""
        decision_making_module.policy.load_state_dict(
            torch.load(self.parameters.where_to_save + "final_policy.pth")
        )

        cprint(
            "[INFO] Loaded decision-making model for CBF-constrained MARL",
            "red",
        )

        if (
            self.parameters.is_using_prioritized_marl
            and self.parameters.prioritization_method.lower() == "marl"
        ):
            priority_module.policy.load_state_dict(
                torch.load(self.parameters.where_to_save + "final_priority_policy.pth")
            )
            cprint("[INFO] Loaded priority model for CBF-constrained MARL", "red")
        else:
            priority_module = None

            cprint(
                "[INFO] Use random priority model for priority-based solving",
                "red",
            )


def mappo_cavs(parameters: Parameters):
    """
    Main function to initialize and run the MAPPO training for CAVs.

    Args:
        parameters (Parameters): Configuration parameters for the training process.

    Returns:
        tuple: Containing the environment, decision-making module, priority module, and updated parameters.
    """
    trainer = MAPPOCAVs(parameters)
    return trainer.train()


if __name__ == "__main__":
    config_file = "config.json"
    parameters = Parameters.from_json(config_file)
    (
        env,
        decision_making_module,
        optimization_module,
        priority_module,
        cbf_controllers,
        parameters,
    ) = mappo_cavs(parameters=parameters)
