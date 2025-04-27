# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
import time

from termcolor import colored, cprint

# Torch
import torch

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

# Data collection
from sigmarl.helper_training import (
    SyncDataCollectorCustom
)
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data import TensorDictPrioritizedReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum
from torchrl.envs.utils import (
    check_env_specs,
)

# Utils
from tqdm import tqdm

import os

import matplotlib.pyplot as plt
import matplotlib

from sigmarl.modules.cbf_module import CBFModule
from sigmarl.modules.decision_making_module import DecisionMakingModule
from sigmarl.modules.priority_module import PriorityModule

# Set up font
matplotlib.rcParams["pdf.fonttype"] = 42  # Use Type 1 fonts (vector fonts)
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Georgia"]
matplotlib.rcParams.update({"font.size": 11})  # Set global font size

from torchrl.envs.libs.vmas import VmasEnv

# Import custom classes
from sigmarl.helper_training import (
    Parameters,
    SaveData,
    TransformedEnvCustom,
    get_path_to_save_model,
    find_the_highest_reward_among_all_models,
    save,
    compute_td_error
)

from sigmarl.scenarios.goal_reaching import GoalReaching

# Reproducibility
torch.manual_seed(0)


class PPOGoalReaching:
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

    def train(self):
        """
        Execute the main training loop for MAPPO.

        Returns:
            tuple: Containing the environment, decision-making module, priority module (possibly), and updated parameters.
        """
        env = self._setup_environment()
        save_data = self._initialize_save_data()
        decision_making_module = self._setup_decision_making_module(env)
        priority_module = self._setup_priority_module(env)
        cbf_module = self._setup_cbf_module(env)

        self._ensure_save_directory_exists()

        if self.parameters.is_load_model:
            self._load_existing_model(decision_making_module, priority_module)
            if not self.parameters.is_continue_train:
                cprint("[INFO] Training will not continue.", "blue")
                return (
                    env,
                    decision_making_module,
                    priority_module,
                    self.parameters,
                )

        collector = self._setup_data_collector(
            env, decision_making_module, priority_module
        )
        replay_buffer = self._setup_replay_buffer()

        pbar = tqdm(total=self.parameters.n_iters, desc="epi_rew_mean = 0")

        t_start = time.time()
        for tensordict_data in collector:
            self._process_tensordict_data(tensordict_data, env)
            self._compute_gae(tensordict_data, decision_making_module, priority_module)
            self._update_priorities(tensordict_data)

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            self._train_epoch(decision_making_module, priority_module, replay_buffer)

            collector.update_policy_weights_()

            self._log_and_save(
                tensordict_data,
                decision_making_module,
                priority_module,
                save_data,
                pbar,
            )
            self._update_learning_rate(decision_making_module, pbar)
            pbar.update()  # Increment the progress bar

        self._save_final_model(decision_making_module, priority_module)
        self._print_training_summary(t_start)

        return env, decision_making_module, priority_module, self.parameters

    def _setup_environment(self):
        """Set up the training environment."""
        scenario = GoalReaching()
        scenario.parameters = self.parameters
        env = VmasEnv(
            scenario=scenario,
            num_envs=self.parameters.num_vmas_envs,
            continuous_actions=True,
            max_steps=self.parameters.max_steps,
            device=self.parameters.device,
            n_agents=self.parameters.n_agents,
        )
        env = TransformedEnvCustom(
            env,
            RewardSum(
                in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]
            ),
        )
        return env

    def _initialize_save_data(self):
        """Initialize the data structure for saving training progress."""
        return SaveData(
            parameters=self.parameters,
            episode_reward_mean_list=[],
        )

    def _setup_decision_making_module(self, env):
        """Set up the decision-making module."""
        mappo = True
        return DecisionMakingModule(env=env, mappo=mappo)

    def _setup_priority_module(self, env):
        """Set up the priority module if prioritized MARL is enabled."""
        if self.parameters.is_using_prioritized_marl:
            return PriorityModule(env=env, mappo=True)
        return None

    def _setup_cbf_module(self, env):
        """Set up the Control Barrier Function (CBF) module if enabled."""
        if self.parameters.is_using_cbf:
            return CBFModule(env=env, mappo=True)
        return None

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

    def _load_existing_model(self, decision_making_module, priority_module):
        """Load an existing model if find one."""
        highest_reward = find_the_highest_reward_among_all_models(
            self.parameters.where_to_save
        )
        self.parameters.episode_reward_mean_current = highest_reward
        if highest_reward is not float("-inf"):
            self._load_model(decision_making_module, priority_module)
        else:
            raise ValueError("No valid model found in the specified directory.")

    def _load_model(self, decision_making_module, priority_module):
        """Load either the final or an intermediate model based on parameters."""
        if self.parameters.is_load_final_model:
            self._load_final_model(decision_making_module, priority_module)
        else:
            self._load_intermediate_model(decision_making_module, priority_module)

    def _load_final_model(self, decision_making_module, priority_module):
        """Load the final model."""
        decision_making_module.policy.load_state_dict(
            torch.load(
                self.parameters.where_to_save + "final_policy.pth", weights_only=True
            )
        )
        cprint("[INFO] Loaded the final model", "red")
        if priority_module and self.parameters.prioritization_method.lower() == "marl":
            priority_module.policy.load_state_dict(
                torch.load(
                    self.parameters.where_to_save + "final_priority_policy.pth",
                    weights_only=True,
                )
            )
            cprint("[INFO] Loaded the final priority model", "red")

    def _load_intermediate_model(self, decision_making_module, priority_module):
        """Load an intermediate model."""
        (
            PATH_POLICY,
            PATH_CRITIC,
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
            decision_making_module.critic.load_state_dict(
                torch.load(PATH_CRITIC, weights_only=True)
            )
            if (
                priority_module
                and self.parameters.prioritization_method.lower() == "marl"
            ):
                priority_module.critic.load_state_dict(
                    torch.load(PATH_PRIORITY_CRITIC, weights_only=True)
                )

    def _setup_data_collector(self, env, decision_making_module, priority_module):
        """Set up the data collector for gathering experience."""
        return SyncDataCollectorCustom(
            env,
            decision_making_module.policy,
            priority_module=priority_module,
            device=self.parameters.device,
            storing_device=self.parameters.device,
            frames_per_batch=self.parameters.frames_per_batch,
            total_frames=self.parameters.total_frames,
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

    def _compute_gae(self, tensordict_data, decision_making_module, priority_module):
        """Compute Generalized Advantage Estimation (GAE)."""
        with torch.no_grad():
            decision_making_module.GAE(
                tensordict_data,
                params=decision_making_module.loss_module.critic_network_params,
                target_params=decision_making_module.loss_module.target_critic_network_params,
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

    def _train_epoch(self, decision_making_module, priority_module, replay_buffer):
        """Train for one epoch."""
        for _ in range(self.parameters.num_epochs):
            for _ in range(
                self.parameters.frames_per_batch // self.parameters.minibatch_size
            ):
                mini_batch_data, info = replay_buffer.sample(return_info=True)
                self._train_on_batch(
                    decision_making_module, priority_module, mini_batch_data
                )
                if self.parameters.is_prb:
                    self._update_priorities_after_training(
                        decision_making_module,
                        priority_module,
                        mini_batch_data,
                        replay_buffer,
                    )

    def _train_on_batch(self, decision_making_module, priority_module, mini_batch_data):
        """Train on a single batch of data."""
        loss_vals = decision_making_module.loss_module(mini_batch_data)
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )
        assert not loss_value.isnan().any() and not loss_value.isinf().any()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(
            decision_making_module.loss_module.parameters(),
            self.parameters.max_grad_norm,
        )
        decision_making_module.optim.step()
        decision_making_module.optim.zero_grad()

        if priority_module and self.parameters.prioritization_method.lower() == "marl":
            priority_module.compute_losses_and_optimize(mini_batch_data)

    def _update_priorities_after_training(
        self, decision_making_module, priority_module, mini_batch_data, replay_buffer
    ):
        """Update priorities in the replay buffer after training."""
        with torch.no_grad():
            decision_making_module.GAE(
                mini_batch_data,
                params=decision_making_module.loss_module.critic_network_params,
                target_params=decision_making_module.loss_module.target_critic_network_params,
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
        self, tensordict_data, decision_making_module, priority_module, save_data, pbar
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
        pbar.set_description(
            f"Episode mean reward = {episode_reward_mean_current:.2f}", refresh=False
        )

        if self.parameters.is_save_intermediate_model:
            self._save_intermediate_model(
                decision_making_module,
                priority_module,
                save_data,
                episode_reward_mean_current,
            )

    def _save_intermediate_model(
        self,
        decision_making_module,
        priority_module,
        save_data,
        episode_reward_mean_current,
    ):
        """Save an intermediate model if it performs better than the previous best."""
        self.parameters.episode_reward_mean_current = episode_reward_mean_current

        if episode_reward_mean_current > self.parameters.episode_reward_intermediate:
            # Save intermediate model only when its episode mean reward is better than the saved one
            self.parameters.episode_reward_intermediate = episode_reward_mean_current
            save(
                self.parameters,
                save_data,
                decision_making_module,
                priority_module if priority_module else None
            )
        else:
            # Otherwise only save parameters and episode mean reward values
            self.parameters.episode_reward_mean_current = (
                self.parameters.episode_reward_intermediate
            )
            save(self.parameters, save_data, None, None)

    def _update_learning_rate(self, decision_making_module, pbar):
        """Update the learning rate based on training progress."""
        for param_group in decision_making_module.optim.param_groups:
            lr_decay = (self.parameters.lr - self.parameters.lr_min) * (
                1 - (pbar.n / self.parameters.n_iters)
            )
            param_group["lr"] = self.parameters.lr_min + lr_decay
            if pbar.n % 10 == 0:
                print(f"Learning rate updated to {param_group['lr']}.")

    def _save_final_model(self, decision_making_module, priority_module):
        """Save the final trained model."""
        torch.save(
            decision_making_module.policy.state_dict(),
            self.parameters.where_to_save + "final_policy.pth",
        )
        torch.save(
            decision_making_module.critic.state_dict(),
            self.parameters.where_to_save + "final_critic.pth",
        )

        if (
            self.parameters.is_using_prioritized_marl
            and self.parameters.prioritization_method.lower() == "marl"
        ):
            torch.save(
                priority_module.policy.state_dict(),
                self.parameters.where_to_save + "final_priority_policy.pth",
            )
            torch.save(
                priority_module.critic.state_dict(),
                self.parameters.where_to_save + "final_priority_critic.pth",
            )

        print(
            colored("[INFO] All files have been saved under:", "black"),
            colored(f"{self.parameters.where_to_save}", "red"),
        )

    def _print_training_summary(self, t_start):
        """Print a summary of the training process."""
        training_duration = (time.time() - t_start) / 3600
        cprint(f"[INFO] Training duration: {training_duration:.2f} hours.", "blue")


def ppo_goal_reaching(parameters: Parameters):
    """
    Main function to initialize and run the MAPPO training for CAVs.

    Args:
        parameters (Parameters): Configuration parameters for the training process.

    Returns:
        tuple: Containing the environment, decision-making module, priority module, and updated parameters.
    """
    trainer = PPOGoalReaching(parameters)
    return trainer.train()


if __name__ == "__main__":
    config_file = "sigmarl/config_goal_reaching.json"
    parameters = Parameters.from_json(config_file)
    env, decision_making_module, priority_module, parameters = ppo_goal_reaching(parameters=parameters)
