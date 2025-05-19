# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
import torch
import numpy as np
from termcolor import colored, cprint

from vmas.simulator.utils import save_video

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

import os
import time
import json

from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.helper_training import (
    SaveData,
    find_the_highest_reward_among_all_models,
    get_model_name,
    is_latex_available,
)

from sigmarl.constants import SCENARIOS, AGENTS

from sigmarl.colors import Color, colors

# Set up font
matplotlib.rcParams["pdf.fonttype"] = 42  # Use Type 1 fonts (vector fonts)
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Georgia"]
matplotlib.rcParams.update({"font.size": 11})  # Set global font size
matplotlib.rcParams["text.usetex"] = is_latex_available()


class Evaluation:
    """
    A class for handling the evaluation of simulation outputs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Evaluation object with model paths and titles for rendering.

        Args:
        model_paths (list): List of paths to the model directories.
        where_to_save_eva_results (str): Where to save evaluation results.
        render_titles (list): Titles for rendering the plots.
        video_names (list): Video names.
        """
        self.scenario_type = kwargs.pop("scenario_type", None)

        self.agent_width = AGENTS["width"]

        self.max_speed = AGENTS["max_speed_achievable"]  # Maximum achievable speed

        self.model_paths = kwargs.pop("model_paths", None)

        self.where_to_save_eva_results = kwargs.pop(
            "where_to_save_eva_results", "outputs"
        )

        self.is_show_different_collisions = kwargs.pop(
            "is_show_different_collisions", True
        )

        self.x_tick_label_rotation = kwargs.pop("x_tick_label_rotation", 0)

        self.fig_sizes = kwargs.pop(
            "fig_sizes",
            {
                "episode_reward": (3.8, 4.3),
                "collision_rate": (3.5, 2.0),
                "centerline_deviation": (3.5, 2.0),
                "average_speed": (3.5, 2.0),
            },
        )

        self.y_limits = kwargs.pop(
            "y_limits",
            {
                "episode_reward": [-2, 10],
                "collision_rate": [0, 100],
                "centerline_deviation": [0, 100],
                "average_speed": [0, 100],
            },
        )

        self.where_to_save_logging = kwargs.pop("where_to_save_logging", "log.txt")

        self.fitst_model_index = kwargs.pop("fitst_model_index", 1)

        self.legends = kwargs.pop("legends", None)
        self.render_titles = kwargs.pop("render_titles", None)
        self.num_simulations_per_model = kwargs.pop("num_simulations_per_model", 32)
        self.n_agents = kwargs.pop("n_agents", 15)
        self.simulation_steps = kwargs.pop("simulation_steps", 15)

        self.is_render = kwargs.pop("is_render", False)
        self.is_save_simulation_video = kwargs.pop("is_save_simulation_video", False)

        self.is_real_time_rendering = kwargs.pop("is_real_time_rendering", False)

        if self.is_save_simulation_video:
            self.is_render = True

        self.video_names = kwargs.pop("video_names", None)

        self.idx_our = next(
            (i for i, path in enumerate(self.model_paths) if "our" in path), -1
        )  # Index of our model

        if self.idx_our in {-1, None}:
            self.idx_our = kwargs.pop("idx_our", 0)

        print(f"Index of our model: {self.idx_our}")

        self.parameters = None  # This will be set when loading model parameters

        self.saved_data = None

        self.num_models = len(self.model_paths)

        self.x_ticks = kwargs.pop(
            "x_ticks", [f"$M_{{{idx}}}$" for idx in range(0, self.num_models)]
        )

        self.labels = [m.split("/")[-2] for m in self.model_paths]

    def _load_parameters(self):
        """
        Loads parameters from a JSON file located in the model path.
        """
        try:
            path_to_json_file = next(
                os.path.join(self.model_i_path, file)
                for file in os.listdir(self.model_i_path)
                if file.endswith(".json")
            )  # Find the first json file in the folder
            # Load parameters from the saved json file
            with open(path_to_json_file, "r") as file:
                data = json.load(file)
                self.saved_data = SaveData.from_dict(data)
                self.parameters = self.saved_data.parameters
        except StopIteration:
            raise FileNotFoundError("No json file found.")

    def _adjust_parameters(self):
        # Adjust parameters
        self.parameters.scenario_type = self.scenario_type
        self.parameters.is_testing_mode = True
        self.parameters.is_real_time_rendering = self.is_real_time_rendering
        self.parameters.is_save_eval_results = True
        self.parameters.is_load_model = True
        self.parameters.is_load_final_model = False
        self.parameters.is_load_out_td = True
        self.parameters.n_agents = self.n_agents
        self.parameters.max_steps = self.simulation_steps

        self.parameters.is_save_simulation_video = self.is_save_simulation_video

        if self.is_render or self.parameters.is_save_simulation_video:
            # Only one env is needed if video should be shown or saved
            self.parameters.num_vmas_envs = 1
            self.parameters.is_load_out_td = False
        else:
            # Otherwise run multiple parallel envs and average the evaluation results
            self.parameters.num_vmas_envs = self.num_simulations_per_model

        # The two parameters below are only used in training. Set them to False so that they do not effect testing
        self.parameters.is_prb = False
        self.parameters.is_challenging_initial_state_buffer = False

        self.parameters.is_visualize_short_term_path = False
        self.parameters.is_visualize_lane_boundary = False
        self.parameters.is_visualize_extra_info = False
        self.parameters.render_title = self.render_titles[self.model_i]

    def _evaluate_model_i(self):
        """
        Evaluate a model.
        """
        out_td = self._get_simulation_outputs()

        if not self.parameters.is_save_simulation_video:
            positions = out_td["agents", "info", "pos"]
            velocities = out_td["agents", "info", "vel"]
            is_collision_with_agents = out_td[
                "agents", "info", "is_collision_with_agents"
            ].bool()
            is_collision_with_lanelets = out_td[
                "agents", "info", "is_collision_with_lanelets"
            ].bool()
            distance_ref = out_td["agents", "info", "distance_ref"]

            is_collide = is_collision_with_agents | is_collision_with_lanelets

            num_steps = positions.shape[1]

            self.average_speed[self.model_idx, :] = velocities.norm(dim=-1).mean(
                dim=(-2, -1)
            )
            self.collision_rate_with_agents[self.model_idx, :] = (
                is_collision_with_agents.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
            )
            self.collision_rate_with_lanelets[self.model_idx, :] = (
                is_collision_with_lanelets.squeeze(-1).any(dim=-1).sum(dim=-1)
                / num_steps
            )
            self.distance_ref_average[self.model_idx, :] = distance_ref.squeeze(
                -1
            ).mean(dim=(-2, -1))

    def _get_simulation_outputs(self):
        # Load the model with the highest reward
        self.parameters.episode_reward_mean_current = (
            find_the_highest_reward_among_all_models(self.model_i_path)
        )
        self.parameters.model_name = get_model_name(parameters=self.parameters)
        path_eval_out_td = (
            self.parameters.where_to_save
            + self.parameters.model_name
            + f"_out_td_{self.parameters.scenario_type}.pth"
        )
        # Load simulation outputs if exist; otherwise run simulation
        if os.path.exists(path_eval_out_td) & (not self.is_render):
            cprint("[INFO] Simulation outputs exist and will be loaded.", "grey")
            out_td = torch.load(path_eval_out_td)
        else:
            (
                env,
                decision_making_module,
                optimization_module,
                priority_module,
                _,
            ) = mappo_cavs(parameters=self.parameters)

            cprint("[INFO] Run simulation...", "grey")
            sim_begin = time.time()

            if self.parameters.is_save_simulation_video:
                with torch.no_grad():
                    out_td, frame_list = env.rollout(
                        max_steps=self.parameters.max_steps - 1,
                        policy=decision_making_module.policy,
                        priority_module=priority_module,
                        callback=lambda env, _: env.render(
                            mode="rgb_array", visualize_when_rgb=False
                        ),  # mode \in {"human", "rgb_array"}
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                        is_save_simulation_video=self.parameters.is_save_simulation_video,
                    )
                sim_end = time.time() - sim_begin

                save_video(
                    f"{self.model_i_path}{self.video_names[self.model_i]}_{self.scenario_type}",
                    frame_list,
                    fps=1 / self.parameters.dt,
                )
                print(
                    colored(f"[INFO] Video saved under ", "grey"),
                    colored(f"{self.model_i_path}.", "blue"),
                )
            else:
                with torch.no_grad():
                    out_td = env.rollout(
                        max_steps=self.parameters.max_steps - 1,
                        policy=decision_making_module.policy,
                        priority_module=priority_module,
                        callback=(lambda env, _: env.render(mode="human"))
                        if self.parameters.num_vmas_envs == 1
                        else None,  # mode should be one of {"human", "rgb_array"}
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                        is_save_simulation_video=False,
                    )
                sim_end = time.time() - sim_begin

            # Save simulation outputs
            torch.save(out_td, path_eval_out_td)
            print(colored("[INFO] Simulation outputs saved.", "grey"))

            # print(
            #     colored(
            #         f"[INFO] Total execution time for {self.parameters.num_vmas_envs} simulations (each has {self.parameters.max_steps} steps): {sim_end:.3f} sec.",
            #         "blue",
            #     )
            # )
            # print(
            #     colored(
            #         f"[INFO] One-step execution time {(sim_end / self.parameters.num_vmas_envs / self.parameters.max_steps):.4f} sec.",
            #         "blue",
            #     )
            # )

        return out_td

    @staticmethod
    def smooth_data(data, window_size=5):
        """
        Smooths the data using a simple moving average.

        Args:
            data: The input data to smooth.
            window_size (int): The size of the smoothing window.

        Returns:
            numpy.ndarray: The smoothed data.
        """
        smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
        return np.concatenate((data[: window_size - 1], smoothed))

    def plot(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Check if the directory exists
        if not os.path.exists(self.where_to_save_eva_results):
            os.makedirs(self.where_to_save_eva_results)
            print(f"[INFO] Directory '{self.where_to_save_eva_results}' was created.")
        else:
            print(
                f"[INFO] Directory '{self.where_to_save_eva_results}' already exists."
            )
        ###############################
        ## Fig 1 - Episode reward
        ###############################
        data_np = self.episode_reward.numpy()
        plt.clf()
        plt.figure(figsize=self.fig_sizes["episode_reward"])

        for i in range(data_np.shape[0]):
            # Original data with transparency
            plt.plot(
                data_np[i, :], color=colors[i], alpha=0.2, linestyle="-", linewidth=0.2
            )

            # Smoothed data
            smoothed_reward = self.smooth_data(data_np[i, :])
            plt.plot(
                smoothed_reward,
                label=self.legends[i],
                color=colors[i],
                linestyle="-",
                linewidth=0.8,
            )

        plt.xlim([0, data_np.shape[1]])
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.legend(
            bbox_to_anchor=(0.5, 1.88),
            loc="upper center",
            fontsize=9,
            ncol=1,
            handleheight=0.7,
            handletextpad=0.5,
            labelspacing=0.3,
        )
        # plt.legend(loc='lower right', fontsize="small", ncol=4)
        # plt.legend(bbox_to_anchor=(1, 0.5), loc='center right', fontsize=fontsize)

        plt.ylim(
            self.y_limits["episode_reward"],
        )
        plt.tight_layout(
            rect=[0, 0, 1, 1]
        )  # Adjust the layout to make space for the legend on the right
        plt.subplots_adjust(top=0.4)  # Adjust top value to make space for the legend

        # Save figure
        if self.parameters.is_save_eval_results:
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_episode_reward.pdf"
            plt.savefig(
                path_save_eval_fig,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_episode_reward.png"
            plt.savefig(
                path_save_eval_fig,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            print(
                colored(f"[INFO] A fig has been saved under", "black"),
                colored(f"{path_save_eval_fig}", "blue"),
            )

        plt.clf()

        ###############################
        ## Fig 2 - collision rate [%]
        ###############################
        if not self.is_show_different_collisions:
            self._boxplot(
                figsize=self.fig_sizes["collision_rate"],
                data=self.collision_rate_sum,
                y_limits=self.y_limits["collision_rate"],
                y_label=r"Collision rate [$\%$]",
                fig_name="collision_rate",
            )
        else:
            plt.clf()
            fig, ax = plt.subplots(figsize=self.fig_sizes["collision_rate"])
            data_np = self.collision_rate_sum.numpy()
            data_with_agents_np = self.collision_rate_with_agents.numpy()
            data_with_lanelets_np = self.collision_rate_with_lanelets.numpy()

            # Positions of the violin plots (adjust as needed to avoid overlap)
            positions = np.arange(1, self.num_models + 1)
            offset = 0.2  # Offset for positioning the violins side by side

            # Plot a horizontal line indicating the median of our model
            median_our = np.median(data_np[self.idx_our])
            max_our = np.max(data_np[self.idx_our])
            plt.axhline(y=median_our, color="black", linestyle="--", alpha=0.3)
            ax.text(
                self.idx_our + 1,
                max_our,
                f"Our: {median_our:.2f}$\%$",
                ha="center",
                va="bottom",
                fontsize="small",
                color="grey",
            )

            # Plot each dataset with different colors
            parts1 = ax.violinplot(
                dataset=data_np.T,
                positions=positions - offset,
                showmeans=False,
                showmedians=True,
                widths=0.2,
            )
            parts2 = ax.violinplot(
                dataset=data_with_agents_np.T,
                positions=positions,
                showmeans=False,
                showmedians=True,
                widths=0.2,
            )
            parts3 = ax.violinplot(
                dataset=data_with_lanelets_np.T,
                positions=positions + offset,
                showmeans=False,
                showmedians=True,
                widths=0.2,
            )

            # Set colors for each violin plot
            self.custom_violinplot_color(parts1, Color.red100, Color.black100, 0.5)
            self.custom_violinplot_color(parts2, Color.blue100, Color.black100, 0.15)
            self.custom_violinplot_color(parts3, Color.green100, Color.black100, 0.15)

            # Setting ticks and labels
            ax.set_xticks(np.arange(len(self.x_ticks)))
            ax.set_xticklabels(
                self.x_ticks,
                rotation=self.x_tick_label_rotation,
                ha="right",
                fontsize="small",
            )
            ax.set_ylim(self.y_limits["collision_rate"])
            ax.set_ylabel(r"Collision rate [$\%$]")
            # ax.set_ylim([0, 2.0]) # [%]
            ax.yaxis.set_major_formatter(
                ticker.FormatStrFormatter("%.2f")
            )  # Set y-axis tick labels to have two digits after the comma
            ax.xaxis.set_minor_locator(
                ticker.NullLocator()
            )  # Make minor x-ticks invisible
            ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.1)
            ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.1)

            # # Add text to shown the median of M1
            # median_M1 = np.median(data_np[0])
            # ax.text(positions[0], 0.2, f'Median\n(total):\n{median_M1:.2f}$\%$', ha='center', va='bottom', fontsize='small', color='grey')

            # Adding legend
            ax.legend(
                [parts1["bodies"][0], parts2["bodies"][0], parts3["bodies"][0]],
                ["Total", "Agent-agent", "Agent-boundary"],
                loc="upper right",
                fontsize=8,
            )

            plt.tight_layout()
            # Save figure
            if self.parameters.is_save_eval_results:
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_collision_rate.pdf"
                plt.savefig(
                    path_save_eval_fig,
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.01,
                    dpi=300,
                )
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_collision_rate.png"
                plt.savefig(
                    path_save_eval_fig,
                    format="png",
                    bbox_inches="tight",
                    pad_inches=0.01,
                    dpi=600,
                )
                print(
                    colored(f"[INFO] A fig has been saved under", "black"),
                    colored(f"{path_save_eval_fig}", "blue"),
                )

        ###############################
        ## Fig 3 - relative centerline deviation [%]
        ###############################
        self._boxplot(
            figsize=self.fig_sizes["centerline_deviation"],
            data=self.distance_ref_average,
            y_limits=self.y_limits["centerline_deviation"],
            y_label=r"Relative centerline deviation [$\%$]",
            fig_name="deviation_average",
        )

        ###############################
        ## Fig 4 - relative average velocity [%]
        ###############################
        self._boxplot(
            figsize=self.fig_sizes["average_speed"],
            data=self.average_speed,
            y_limits=self.y_limits["average_speed"],
            y_label=r"Rel. avg. spd. [$\%$]",
            fig_name="average_speed",
        )

    def _boxplot(self, figsize, data, y_label, fig_name, y_limits: None):
        plt.clf()
        fig, ax = plt.subplots(figsize=figsize)

        data_np = data.numpy()

        # Plot a horizontal line indicating the median of our model
        median_our = np.median(data_np[self.idx_our])
        max_our = np.max(data_np[self.idx_our])
        plt.axhline(y=median_our, color="black", linestyle="--", alpha=0.3)
        ax.text(
            self.idx_our + 1,
            max_our,
            f"Our: {median_our:.2f}$\%$",
            ha="center",
            va="bottom",
            fontsize="small",
            color="grey",
        )

        ax.violinplot(dataset=data_np.T, showmeans=False, showmedians=True)
        ax.set_xticks(np.arange(len(self.x_ticks)))
        ax.set_xticklabels(
            self.x_ticks,
            rotation=self.x_tick_label_rotation,
            ha="right",
            fontsize="small",
        )
        ax.set_ylabel(y_label)

        if y_limits:
            ax.set_ylim(y_limits)

        ax.xaxis.set_minor_locator(ticker.NullLocator())  # Make minor x-ticks invisible
        ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.1)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.1)

        # Save figure
        plt.tight_layout()
        if self.parameters.is_save_eval_results:
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_{fig_name}.pdf"
            plt.savefig(
                path_save_eval_fig,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.01,
                dpi=300,
            )
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_{fig_name}.png"
            plt.savefig(
                path_save_eval_fig,
                format="png",
                bbox_inches="tight",
                pad_inches=0.01,
                dpi=600,
            )
            print(
                colored(f"[INFO] A fig has been saved under", "black"),
                colored(f"{path_save_eval_fig}", "blue"),
            )

    def _init_eva_matrices(self):
        """
        Initialize evaluation matrices.
        """
        self.average_speed = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.collision_rate_with_agents = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.collision_rate_with_lanelets = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.distance_ref_average = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.episode_reward = torch.zeros(
            (self.num_models, self.parameters.n_iters),
            device=self.parameters.device,
            dtype=torch.float32,
        )

    def run_evaluation(self):
        """
        Main method to run evaluation over all provided model paths.
        """
        self.model_idx = 0

        for model_i in range(self.num_models):
            print("------------------------------------------")
            print(
                colored("-- [INFO] Model ", "black"),
                colored(f"{model_i + 1}", "blue"),
                colored(f"({self.legends[model_i]})", color="grey"),
            )
            print("------------------------------------------")

            self.model_i = model_i
            self.model_i_path = self.model_paths[model_i]

            self._load_parameters()
            self._adjust_parameters()

            if self.model_idx == 0:
                self._init_eva_matrices()  # Only need to be done once

            self.episode_reward[self.model_idx, :] = torch.tensor(
                [self.saved_data.episode_reward_mean_list]
            )

            self._evaluate_model_i()

            self.model_idx += 1

        if not self.parameters.is_save_simulation_video:
            self.compute_performance_metrics()
            self.plot()

    def compute_performance_metrics(
        self, is_remove_max_min: bool = True, is_relative_values: bool = True
    ):
        """
        This function computes the three performance metrices related to safety, lane following, and traffic efficienty.
        """
        # Remove the best and the worst to eliminate the influence of the stochastic nature of the randomness

        if is_remove_max_min:
            self.collision_rate_with_agents = self.remove_max_min(
                self.collision_rate_with_agents
            )
            self.collision_rate_with_lanelets = self.remove_max_min(
                self.collision_rate_with_lanelets
            )

            self.distance_ref_average = self.remove_max_min(self.distance_ref_average)

            self.average_speed = self.remove_max_min(self.average_speed)

        if is_relative_values:
            # Use relative values
            self.distance_ref_average = (
                self.distance_ref_average / self.agent_width * 100
            )
            self.average_speed = self.average_speed / self.max_speed * 100

        self.collision_rate_with_agents = self.collision_rate_with_agents * 100
        self.collision_rate_with_lanelets = self.collision_rate_with_lanelets * 100

        self.collision_rate_sum = (
            self.collision_rate_with_agents[:] + self.collision_rate_with_lanelets[:]
        )

        # Collision rate [%]
        self.CR_AA_avg = self.collision_rate_with_agents.mean(dim=-1)
        self.CR_AL_avg = self.collision_rate_with_lanelets.mean(dim=-1)
        self.CR_total_avg = self.collision_rate_sum.mean(dim=-1)

        # Relative centerline deviation [%]
        self.CD_avg = self.distance_ref_average.mean(dim=-1)

        # Relative average speed [%]
        self.AS_avg = self.average_speed.mean(dim=-1)

        # Compute composite score
        w1 = 1 / self.CR_total_avg.mean()
        w2 = 1 / self.CD_avg.mean()
        w3 = 1 / self.AS_avg.mean()
        # composite score = - w1 * CR_total_avg - w2 * CD_avg + w3 * AS_avg
        self.composite_score = (
            -w1 * self.CR_total_avg - w2 * self.CD_avg + w3 * self.AS_avg
        )

        print(f"composite_score: {self.composite_score}")

        self._log()

    def _log(self):
        # Collision rate [%]
        self.CR_AA_median = self.collision_rate_with_agents.median(dim=-1).values
        self.CR_AL_median = self.collision_rate_with_lanelets.median(dim=-1).values
        self.CR_total_median = self.collision_rate_sum.median(dim=-1).values

        # Relative centerline deviation [%]
        self.CD_median = self.distance_ref_average.median(dim=-1).values

        # Relative average speed [%]
        self.AS_median = self.average_speed.median(dim=-1).values

        # Log messages
        log_CR_AA_median = f"[LOG] Agent-agent collision rate [%]: {self.format_array(self.CR_AA_median.numpy())}"
        log_CR_AL_median = f"[LOG] Agent-lanelet collision rate [%]: {self.format_array(self.CR_AL_median.numpy())}"
        log_CR_total_median = f"[LOG] Total collision rate [%]: {self.format_array(self.CR_total_median.numpy())}"
        log_CD_median = f"[LOG] Relative centerline deviation [%]: {self.format_array(self.CD_median.numpy())}"
        log_AS_median = f"[LOG] Relative average speed [%]: {self.format_array(self.AS_median.numpy())}"

        log_CS = (
            f"[LOG] Composite scores: {self.format_array(self.composite_score.numpy())}"
        )

        cprint(log_CR_total_median, "black")
        cprint(log_CR_AA_median, "black")
        cprint(log_CR_AL_median, "black")
        cprint(log_CD_median, "black")
        cprint(log_AS_median, "black")
        cprint(log_CS)

        # Save the evaluation results to a txt file
        with open(self.where_to_save_logging, "a") as file:
            file.write(f"Scenario: {self.parameters.scenario_type}\n")
            file.write(f"{log_CR_AA_median}\n")
            file.write(f"{log_CR_AL_median}\n")
            file.write(f"{log_CR_total_median}\n")
            file.write(f"{log_CD_median}\n")
            file.write(f"{log_AS_median}\n")
            file.write(f"{log_CS}\n")
            file.write("=========================================")
            file.write("\n")

    @staticmethod
    def remove_max_min(tensor):
        """
        Remove the maximum and minimum values from each row of the tensor.

        Args:
            tensor: A 2D tensor with shape [a, b]

        Returns:
            A 2D tensor with the max and min values removed from each row.
        """
        # Find the indices of the max and min in each row
        max_vals, max_indices = torch.max(tensor, dim=1, keepdim=True)
        min_vals, min_indices = torch.min(tensor, dim=1, keepdim=True)

        # Deal with the case in which one whole row has the same value
        is_row_same_value = max_indices == min_indices
        is_row_different_values = max_indices != min_indices

        row_same_value, _ = torch.where(is_row_same_value)
        row_different_values, _ = torch.where(is_row_different_values)

        # Replace max and min values with inf and -inf
        tensor[row_same_value.unsqueeze(-1), 0] = float("inf")
        tensor[row_same_value.unsqueeze(-1), 1] = float("-inf")

        tensor[
            row_different_values.unsqueeze(-1),
            max_indices[is_row_different_values].unsqueeze(-1),
        ] = float("inf")
        tensor[
            row_different_values.unsqueeze(-1),
            min_indices[is_row_different_values].unsqueeze(-1),
        ] = float("-inf")

        # Remove the inf and -inf values
        mask = (tensor != float("inf")) & (tensor != float("-inf"))
        filtered_tensor = tensor[mask].view(tensor.size(0), -1)

        return filtered_tensor

    # Function to add custom median markers
    @staticmethod
    def custom_violinplot_color(parts, color_face, color_lines, alpha):
        for pc in parts["bodies"]:
            pc.set_facecolor(color_face)
            pc.set_edgecolor(color_face)
            pc.set_alpha(alpha)

        parts["cmedians"].set_colors(color_lines)
        parts["cmedians"].set_alpha(alpha)
        parts["cmaxes"].set_colors(color_lines)
        parts["cmaxes"].set_alpha(alpha)
        parts["cmins"].set_colors(color_lines)
        parts["cmins"].set_alpha(alpha)
        parts["cbars"].set_colors(color_lines)
        parts["cbars"].set_alpha(alpha)

    @staticmethod
    def format_array(arr):
        return ", ".join([f"{x:.2f}" for x in arr])

    @staticmethod
    def min_max_normalize(tensor: torch.Tensor, **kwargs):
        """
        This function min-max normalize the given tensor.
        """
        min_val = kwargs.pop("min", tensor.min())
        max_val = kwargs.pop("max", tensor.max())
        return (tensor - min_val) / (max_val - min_val)
