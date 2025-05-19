import os
import json
import time
from termcolor import cprint

import torch
from tensordict.tensordict import TensorDict

from vmas.simulator.utils import save_video

from sigmarl.helper_training import SaveData, is_latex_available
from sigmarl.map_manager import MapManager
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.constants import AGENTS
from sigmarl.rectangle_approximation import RectangleCircleApproximation

import pyglet

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.family": "serif",
        "text.usetex": is_latex_available(),
    }
)


class CBF_MARL_Evaluation:
    def __init__(
        self,
        path: str,
        n_agents: int = 1,
        is_using_cbf: bool = True,
        scenario_type: str = "CPM_entire",
        random_seed=0,
        n_circles_approximate_vehicle: int = 3,
        is_load_out_td: bool = False,
        obs_noise_level=None,
    ):
        self.path = path
        path_to_json_file = next(
            os.path.join(self.path, file)
            for file in os.listdir(self.path)
            if file.endswith(".json") and "data" in file
        )

        with open(path_to_json_file, "r") as file:
            data = json.load(file)
            saved_data = SaveData.from_dict(data)
            parameters = saved_data.parameters
            parameters.dt = 0.05  # Sampling period
            parameters.is_testing_mode = True
            parameters.is_load_model = True
            parameters.is_load_final_model = True
            parameters.is_real_time_rendering = False
            parameters.is_save_eval_results = False
            parameters.is_load_out_td = is_load_out_td
            parameters.max_steps = 600
            parameters.reset_agent_fixed_duration = (
                15  # Duration in seconds. Set to 0 to disable
            )
            parameters.num_vmas_envs = 1
            parameters.scenario_type = scenario_type
            parameters.n_agents = n_agents  # Number of agents
            parameters.is_save_simulation_video = True
            parameters.is_visualize_short_term_path = True
            parameters.is_visualize_lane_boundary = False
            parameters.is_visualize_extra_info = True
            parameters.random_seed = random_seed
            parameters.n_circles_approximate_vehicle = n_circles_approximate_vehicle
            parameters.is_visualize_lane_boundary = True
            parameters.lane_width = 0.3

            if obs_noise_level is not None and obs_noise_level > 0:
                parameters.is_obs_noise = True
                parameters.obs_noise_level = obs_noise_level
            else:
                parameters.is_obs_noise = False
                parameters.obs_noise_level = 0.0

            # Parameter settings for CBF-constrained MARL
            parameters.is_using_cbf = is_using_cbf
            parameters.is_using_prioritized_marl = (
                True  # Use priority-based solving strategy
            )
            parameters.prioritization_method = "random"
            parameters.is_using_centralized_cbf = False

            self.parameters = parameters

        # Specify file names
        if self.parameters.is_using_cbf:
            self.out_td_path = os.path.join(
                self.path,
                f"out_td_rl_cbf_seed_scenario_{parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}.pth",
            )
            self.performance_metric_path = os.path.join(
                self.path,
                f"peformance_metric_rl_cbf_seed_scenario_{parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}.json",
            )
            self.video_path = os.path.join(
                self.path,
                f"video_rl_cbf_seed_scenario_{parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}",
            )
            self.fig_footprints_path = os.path.join(
                self.path,
                f"fig_footprints_rl_cbf_seed_scenario_{parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}.pdf",
            )
        else:
            self.out_td_path = os.path.join(
                self.path,
                f"out_td_rl_seed_scenario_{parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}.pth",
            )
            self.performance_metric_path = os.path.join(
                self.path,
                f"peformance_metric_rl_seed_scenario_{parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}.json",
            )
            self.video_path = os.path.join(
                self.path,
                f"video_rl_seed_scenario_{parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}",
            )
            self.fig_footprints_path = os.path.join(
                self.path,
                f"fig_footprints_rl_seed_scenario_{parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}.pdf",
            )

    def evaluate(self):
        if self.parameters.is_load_out_td:
            if os.path.exists(self.out_td_path):
                out_td = torch.load(self.out_td_path, weights_only=False)
                cprint(f"[INFO] Loaded out_td from {self.out_td_path}", "green")
                time.sleep(0.05)
                return
            else:
                cprint(
                    f"[WARNING] File {self.out_td_path} does not exist. Run simulation to get out_td.",
                    "green",
                )

        # Initialize environment
        env, policy, priority_module, cbf_controllers, self.parameters = mappo_cavs(
            parameters=self.parameters
        )

        out_td, frame_list = env.rollout(
            max_steps=self.parameters.max_steps - 1,
            policy=policy,
            priority_module=priority_module,
            callback=lambda env, _: env.render(
                mode="rgb_array", visualize_when_rgb=True
            ),
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=self.parameters.is_save_simulation_video,
            cbf_controllers=cbf_controllers,
        )

        minimal_out_td = TensorDict(
            {
                "pos": out_td["agents"]["info"]["pos"].clone(),  # shape [B, T, N, 2]
                "vel": out_td["agents"]["info"]["vel"].clone(),  # shape [B, T, N, 2]
                "rot": out_td["agents"]["info"]["rot"].clone(),  # shape [B, T, N, 1]
                "is_collision_with_agents": out_td["agents"]["info"][
                    "is_collision_with_agents"
                ].clone(),  # shape [B, T, N, 1]
                "is_collision_with_lanelets": out_td["agents"]["info"][
                    "is_collision_with_lanelets"
                ].clone(),  # shape [B, T, N, 1]
                "nominal_action_vel": out_td["agents"]["info"][
                    "nominal_action_vel"
                ].clone(),  # shape [B, T, N, 1]
                "nominal_action_steer": out_td["agents"]["info"][
                    "nominal_action_steer"
                ].clone(),  # shape [B, T, N, 1]
                "cbf_action_vel": out_td["agents"]["info"][
                    "cbf_action_vel"
                ].clone(),  # shape [B, T, N, 1]
                "cbf_action_steer": out_td["agents"]["info"][
                    "cbf_action_steer"
                ].clone(),  # shape [B, T, N, 1]
                "ref_lanelet_ids": out_td["agents"]["info"][
                    "ref_lanelet_ids"
                ].clone(),  # shape [B, T, N, 1]
                "path_id": out_td["agents"]["info"][
                    "path_id"
                ].clone(),  # shape [B, T, N, 1]
            },
            batch_size=out_td["agents"]["info"]["pos"].shape[:3],
        )  # [B, T, N]

        torch.save(minimal_out_td, self.out_td_path)

        self.save_data(minimal_out_td, env)
        save_video(self.video_path, frame_list, fps=1 / self.parameters.dt)
        self.visualize_footprint_of_agent_i(i=0)

    def save_data(self, out_td, env):
        # Collect metrics as before
        is_collision_with_agents = out_td["is_collision_with_agents"].bool()
        is_collision_with_lanelets = out_td["is_collision_with_lanelets"].bool()
        is_collide = is_collision_with_agents | is_collision_with_lanelets

        velocities = out_td["vel"]

        num_steps = is_collision_with_agents.shape[1]

        collision_a2a_count = (
            is_collision_with_agents.squeeze(-1).any(dim=-1).sum(dim=-1)
        )
        collision_rate_a2a = collision_a2a_count / num_steps
        collision_a2l_count = (
            is_collision_with_lanelets.squeeze(-1).any(dim=-1).sum(dim=-1)
        )
        collision_rate_a2l = collision_a2l_count / num_steps
        collision_total_count = is_collide.squeeze(-1).any(dim=-1).sum(dim=-1)
        collision_rate = collision_total_count / num_steps

        average_speed = velocities.norm(dim=-1).mean(dim=(-2, -1))

        time_rl = sum(env.time_rl) / len(env.time_rl) if env.time_rl else 0
        time_cbf = sum(env.time_cbf) / len(env.time_cbf) if env.time_cbf else 0
        time_pseudo_dis = (
            sum(env.time_pseudo_dis) / len(env.time_pseudo_dis)
            if env.time_pseudo_dis
            else 0
        )

        num_fail = env.num_fail
        failure_rate = num_fail / (num_steps * self.parameters.n_agents)

        # Create structured dictionary
        data = {
            "collision": {
                "a2a_count": collision_a2a_count.item(),
                "a2a_rate": round(collision_rate_a2a.item(), 4),
                "a2l_count": collision_a2l_count.item(),
                "a2l_rate": round(collision_rate_a2l.item(), 4),
                "total_count": collision_total_count.item(),
                "total_rate": round(collision_rate.item(), 4),
            },
            "computation_time": {
                "mean_rl": round(time_rl, 5),
                "mean_cbf": round(time_cbf, 4),
                "mean_pseudo_dis": round(time_pseudo_dis, 5),
                # "raw_rl": env.time_rl,  # this will be saved as list
                # "raw_cbf": env.time_cbf,
                # "raw_pseudo_dis": env.time_pseudo_dis
            },
            "performance": {
                "mean_speed": round(average_speed.item(), 3),
                "infeasibility_rate": round(failure_rate, 3),
            },
        }

        # Write to JSON file
        with open(self.performance_metric_path, "w") as f:
            json.dump(data, f, indent=4)

    def visualize_footprint_of_agent_i(self, i=0):
        map = MapManager(
            scenario_type=self.parameters.scenario_type,
            device=self.parameters.device,
            lane_width=self.parameters.lane_width,
        )
        fig, ax = map.parser.visualize_map()

        # Agent size
        length = AGENTS["length"]
        width = AGENTS["width"]
        half_l = length / 2.0
        half_w = width / 2.0

        out_td = torch.load(self.out_td_path, weights_only=False)
        print(f"Loaded out_td from {self.out_td_path}")

        # Extract relevant tensors
        pos = out_td["pos"][0, :, i].numpy()  # [T, 2]
        rot = out_td["rot"][0, :, i, 0].numpy()  # [T]
        coll1 = out_td["is_collision_with_agents"][0, :, i, 0]  # [T]
        coll2 = out_td["is_collision_with_lanelets"][0, :, i, 0]  # [T]

        coll_mask = (coll1.bool() | coll2.bool()).numpy()  # [T]

        # Local corners (relative to center)
        corners = np.array(
            [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]]
        )  # [4, 2]

        # Group into episodes
        num_steps = pos.shape[0]
        start_idx = 0
        for t in range(num_steps):
            if coll_mask[t] or t == num_steps - 1:
                end_idx = t

                # Handle empty segments
                if end_idx < start_idx:
                    start_idx = t + 1
                    continue

                seg_len = end_idx - start_idx + 1
                for k, step in enumerate(range(start_idx, end_idx + 1)):
                    center = pos[step]
                    theta = rot[step]
                    alpha = 0.2 + 0.6 * (
                        k / max(seg_len - 1, 1)
                    )  # Fade from 0.2 to 0.8

                    # Rotation matrix
                    R = np.array(
                        [
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)],
                        ]
                    )
                    transformed = (R @ corners.T).T + center

                    if step == end_idx and coll_mask[t]:
                        facecolor = "tab:green"
                        edgecolor = "black"  # Colliding footprint
                    else:
                        facecolor = "tab:blue"
                        edgecolor = "black"
                    patch = Polygon(
                        transformed,
                        closed=True,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        alpha=alpha,
                        linewidth=0.4,
                        zorder=10,
                    )

                    ax.add_patch(patch)

                start_idx = t + 1  # Reset start index

        # Save the figure
        fig.savefig(
            self.fig_footprints_path, format="pdf", dpi=450, bbox_inches="tight"
        )
        print(f"Figure saved as {self.fig_footprints_path}")
        plt.close(fig)

        return fig, ax

    def visualize_action_of_agent_i(
        self, i=0, step=2, visu_boundary=True, show_legend=True, legend_loc="best"
    ):
        """
        :param i: agent index
        :param step: time step to visualize
        """

        map = MapManager(
            scenario_type=self.parameters.scenario_type,
            device=self.parameters.device,
            lane_width=self.parameters.lane_width,
        )
        map.parser._is_visualize_entry_direction = False
        fig, ax = map.parser.visualize_map()

        length = AGENTS["length"]
        width = AGENTS["width"]
        half_l = length / 2.0
        half_w = width / 2.0

        out_td = torch.load(self.out_td_path, weights_only=False)
        print(f"Loaded out_td from {self.out_td_path}")

        # Extract relevant tensors
        pos = out_td["pos"][0, step, i].numpy()  # [T, 2]
        rot = out_td["rot"][0, step, i, 0].numpy()  # [T]
        nominal_action_vel = (
            out_td["nominal_action_vel"][0, step, i, 0].detach().numpy()
        )  # [T]
        nominal_action_steer = (
            out_td["nominal_action_steer"][0, step, i, 0].detach().numpy()
        )  # [T]
        cbf_action_vel = out_td["cbf_action_vel"][0, step, i, 0].numpy()  # [T]
        cbf_action_steer = out_td["cbf_action_steer"][0, step, i, 0].numpy()  # [T]

        path_id = out_td["path_id"][0, step, i, 0].numpy().astype(int)  # [T]

        if visu_boundary:
            ref_path = map.parser.reference_paths[path_id]
            left_bound = ref_path["left_boundary_shared"]
            right_bound = ref_path["right_boundary_shared"]
            ax.plot(
                left_bound[:, 0],
                left_bound[:, 1],
                color="black",
                linewidth=1.0,
                marker="o",
            )
            ax.plot(
                right_bound[:, 0],
                right_bound[:, 1],
                color="black",
                linewidth=1.0,
                marker="o",
            )

        # Local corners (relative to center)
        corners = np.array(
            [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]]
        )  # [4, 2]

        range_x = 0.5
        range_y = 0.4
        ax.set_xlim(pos[0] - range_x, pos[0] + range_x)
        ax.set_ylim(pos[1] - range_y, pos[1] + range_y)

        #
        new_aspect_ratio = range_y / range_x

        figsize_x = 3
        fig.set_size_inches(figsize_x, figsize_x * new_aspect_ratio)

        R = np.array(
            [
                [np.cos(rot), -np.sin(rot)],
                [np.sin(rot), np.cos(rot)],
            ]
        )
        transformed = (R @ corners.T).T + pos

        patch = Polygon(
            transformed,
            closed=True,
            facecolor="tab:blue",
            edgecolor="black",
            alpha=0.8,
            linewidth=0.4,
            zorder=10,
        )

        ax.add_patch(patch)

        cbf_action_vec = np.array(
            [
                0.5 * cbf_action_vel * np.cos(rot + cbf_action_steer),
                0.5 * cbf_action_vel * np.sin(rot + cbf_action_steer),
            ]
        )

        cbf_action_arrow = FancyArrowPatch(
            posA=pos,
            posB=pos + cbf_action_vec,
            arrowstyle="-|>",
            mutation_scale=30,
            color="tab:blue",
            linewidth=2,
            linestyle="-",
            zorder=10,
        )
        ax.add_patch(cbf_action_arrow)

        nominal_action_vec = np.array(
            [
                0.5 * nominal_action_vel * np.cos(rot + nominal_action_steer),
                0.5 * nominal_action_vel * np.sin(rot + nominal_action_steer),
            ]
        )

        nominal_action_arrow = FancyArrowPatch(
            posA=pos,
            posB=pos + nominal_action_vec,
            arrowstyle="-|>",
            mutation_scale=30,
            color="gray",
            linewidth=1.5,
            linestyle="--",
            zorder=10,
        )
        ax.add_patch(nominal_action_arrow)

        if show_legend:

            legend_proxy_cbf = Line2D(
                [0],
                [0],
                color="tab:blue",
                linestyle="-",
                linewidth=2,
                label="CBF action (arrow)",
            )
            legend_proxy_nominal = Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                linewidth=2,
                label="Nominal action (arrow)",
            )

            ax.legend(
                handles=[legend_proxy_cbf, legend_proxy_nominal],
                loc=legend_loc,
            )

        fig_name = os.path.join(
            self.path,
            f"fig_cbf_action_rl_cbf_seed_scenario_{self.parameters.scenario_type}_{self.parameters.random_seed}_circles_{self.parameters.n_circles_approximate_vehicle}_step_{step}.pdf",
        )
        fig.savefig(fig_name, format="pdf", dpi=450)
        print(f"Figure saved as {fig_name}")
        plt.close(fig)

    def load_pefromance_metrics(self):
        if not os.path.exists(self.performance_metric_path):
            raise FileNotFoundError(f"No file found at {self.performance_metric_path}")

        with open(self.performance_metric_path, "r") as f:
            data = json.load(f)
        mean_rl = data["computation_time"]["mean_rl"]
        mean_cbf = data["computation_time"]["mean_cbf"]
        mean_pseudo_dis = data["computation_time"]["mean_pseudo_dis"]
        mean_speed = data["performance"]["mean_speed"]
        infeasibility_rate = data["performance"]["infeasibility_rate"]
        collision_count = data["collision"]["total_count"]
        collision_rate = data["collision"]["total_rate"]

        return (
            mean_rl,
            mean_cbf,
            mean_pseudo_dis,
            mean_speed,
            infeasibility_rate,
            collision_count,
            collision_rate,
        )


def run_simulations(
    policy_path,
    is_using_cbfs,
    random_seeds,
    n_circles_approximate_vehicle_list,
    scenario_types,
    is_load_out_td,
):

    mean_computation_time_rl_list = np.zeros(
        (
            len(is_using_cbfs),
            len(scenario_types),
            len(n_circles_approximate_vehicle_list),
            len(random_seeds),
        )
    )
    mean_computation_time_cbf_list = np.zeros_like(mean_computation_time_rl_list)
    mean_computation_time_pseudo_dis_list = np.zeros_like(mean_computation_time_rl_list)
    mean_speed_list = np.zeros_like(mean_computation_time_rl_list)
    infeasibility_rate_list = np.zeros_like(mean_computation_time_rl_list)
    collision_count_list = np.zeros_like(mean_computation_time_rl_list)

    total_simulations = (
        len(is_using_cbfs)
        * len(scenario_types)
        * len(n_circles_approximate_vehicle_list)
        * len(random_seeds)
    )

    simulation_count = 0

    for i_cbf, is_using_cbf in enumerate(is_using_cbfs):
        for i_scenario, scenario_type in enumerate(scenario_types):
            for i_cir, n_circles_approximate_vehicle in enumerate(
                n_circles_approximate_vehicle_list
            ):
                for i_seed, random_seed in enumerate(random_seeds):
                    simulation_count += 1
                    print(
                        "-------------------------------------------------------------"
                    )
                    print(
                        f"----------------- Simulation {simulation_count}/{total_simulations} --------------------"
                    )
                    print(
                        "-------------------------------------------------------------"
                    )
                    print(
                        f"Random seed: {random_seed}, using CBF: {is_using_cbf}, number of circles approximating vehicle: {n_circles_approximate_vehicle}"
                    )

                    time_start = time.time()

                    evaluation = CBF_MARL_Evaluation(
                        path=policy_path,
                        n_agents=1,
                        is_using_cbf=is_using_cbf,
                        scenario_type=scenario_type,
                        random_seed=random_seed,
                        n_circles_approximate_vehicle=n_circles_approximate_vehicle,
                        is_load_out_td=is_load_out_td,
                        obs_noise_level=0.0,
                    )
                    evaluation.evaluate()

                    (
                        mean_rl,
                        mean_cbf,
                        mean_pseudo_dis,
                        mean_speed,
                        infeasibility_rate,
                        collision_count,
                        collision_rate,
                    ) = evaluation.load_pefromance_metrics()

                    mean_computation_time_rl_list[
                        i_cbf, i_scenario, i_cir, i_seed
                    ] = mean_rl
                    mean_computation_time_cbf_list[
                        i_cbf, i_scenario, i_cir, i_seed
                    ] = mean_cbf
                    mean_computation_time_pseudo_dis_list[
                        i_cbf, i_scenario, i_cir, i_seed
                    ] = mean_pseudo_dis
                    mean_speed_list[i_cbf, i_scenario, i_cir, i_seed] = mean_speed
                    infeasibility_rate_list[
                        i_cbf, i_scenario, i_cir, i_seed
                    ] = infeasibility_rate
                    collision_count_list[
                        i_cbf, i_scenario, i_cir, i_seed
                    ] = collision_count

                    # Close all pyglet windows
                    for window in list(pyglet.app.windows):
                        window.close()
                    plt.close()

                    print(f"All data saved to {evaluation.path}.")
                    print(
                        f"Total simulation time: {time.time() - time_start:.2f} seconds"
                    )

    cprint(
        "[INFO] Evaluation for CBF-constrained MARL completed successfully.",
        "green",
    )
    idx_using_cbf = is_using_cbfs.index(True)

    mean_computation_time_cbf = mean_computation_time_cbf_list[idx_using_cbf].mean(
        axis=(0, 2)
    )  # Average over scenarios and random seed. Shape (n_circles_approximate_vehicle_list)
    mean_computation_time_pseudo_dis = mean_computation_time_pseudo_dis_list[
        idx_using_cbf
    ].mean(
        axis=(0, 2)
    )  # Average over scenarios and random seed. Shape (n_circles_approximate_vehicle_list)
    mean_computation_time_rl = mean_computation_time_rl_list[idx_using_cbf].mean(
        axis=(0, 2)
    )  # Average over scenarios and random seed. Shape (n_circles_approximate_vehicle_list)
    mean_speed = mean_speed_list[idx_using_cbf].mean(axis=(0, 2))
    infeasibility_rate = infeasibility_rate_list[idx_using_cbf].mean(
        axis=(2)
    )  # Average over random seed. Shape (n_scenarios, n_circles_approximate_vehicle_list)

    collision_count_rl_cbf = collision_count_list[idx_using_cbf].mean(
        axis=(2)
    )  # Average over random seed. Shape (n_scenarios, n_circles_approximate_vehicle_list)

    if len(is_using_cbfs) == 2:
        idx_not_using_cbf = is_using_cbfs.index(False)
        collision_count_rl = collision_count_list[idx_not_using_cbf].mean(
            axis=(2)
        )  # Average over random seed. Shape (n_scenarios, n_circles_approximate_vehicle_list)

    else:
        collision_count_rl = None

    return (
        mean_computation_time_cbf,
        mean_computation_time_pseudo_dis,
        mean_computation_time_rl,
        collision_count_rl_cbf,
        collision_count_rl,
    )


def plot_collision_count(
    collision_count_rl,
    collision_count_rl_cbf,
    n_circles_approximate_vehicle_list,
    fig_path,
):
    """
    Plots collision counts and the diameter-to-width ratio against the number of circles.

    Args:
        collision_count_rl (float): Mean collision count without safety filter (RL only).
        collision_count_rl_cbf (list): List of mean collision counts with safety filter (RL+CBF)
                                       for each number of circles.
        n_circles_approximate_vehicle_list (list): List of number of circles used for approximation.
        fig_path (str): Path to save the output figure (e.g., "figure.pdf").
    """

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.axhline(
        y=collision_count_rl,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Collision count: without safety filter (RL only)",
    )

    ax1.plot(
        n_circles_approximate_vehicle_list,
        collision_count_rl_cbf,
        marker="o",
        markersize=6,
        linestyle="-",
        color="black",
        linewidth=1.5,
        label="Collision count: with safety filter (RL + CBF)",
    )

    diameter_width_ratio = []
    for n in n_circles_approximate_vehicle_list:

        rectangle_circle_approximation = RectangleCircleApproximation(
            AGENTS["length"], AGENTS["width"], n
        )

        ratio = rectangle_circle_approximation.radius * 2 / AGENTS["width"]
        diameter_width_ratio.append(ratio)

    ax2 = ax1.twinx()

    ax2.plot(
        n_circles_approximate_vehicle_list,
        diameter_width_ratio,
        marker="s",
        markersize=6,
        linestyle="-.",
        color="tab:red",
        linewidth=1.5,
        label="Ratio: circle diameter / vehicle width",
    )

    ax1.set_xlabel("Number of circles used to approximate vehicle")
    ax1.set_ylabel("Collision count")
    ax2.set_ylabel("Ratio: circle diameter / vehicle width")

    ax1.set_xticks(n_circles_approximate_vehicle_list)

    ax1.set_ylim(0, max(max(collision_count_rl_cbf), collision_count_rl) * 1.2)
    ax2.set_ylim(1, max(diameter_width_ratio) * 1.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.tick_params(axis="both", direction="in")
    ax2.tick_params(axis="y", direction="in")

    ax1.grid(True, axis="y", linestyle="--", alpha=0.6, linewidth=0.5)

    fig.tight_layout(pad=0.3)

    fig.savefig(fig_path, format="pdf", dpi=450, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    # plt.show()


def plot_computation_time(
    mean_computation_time_cbf,
    mean_computation_time_pseudo_dis,
    mean_computation_time_rl,
    n_circles_approximate_vehicle_list,
    diameter_width_ratio,
    fig_path,
):
    fig, ax1 = plt.subplots(figsize=(6, 3))

    bottom_pseudo_dis = np.array(mean_computation_time_cbf)
    bottom_rl = bottom_pseudo_dis + np.array(mean_computation_time_pseudo_dis)

    bar_width = 0.5  # Width of the bars

    ax1.bar(
        n_circles_approximate_vehicle_list,
        mean_computation_time_cbf,
        bar_width,
        label="Solving CBF-QP",
        color="tab:blue",
    )
    ax1.bar(
        n_circles_approximate_vehicle_list,
        mean_computation_time_pseudo_dis,
        bar_width,
        bottom=bottom_pseudo_dis,
        label="Computing pseudo-distance",
        color="tab:orange",
    )
    ax1.bar(
        n_circles_approximate_vehicle_list,
        mean_computation_time_rl,
        bar_width,
        bottom=bottom_rl,
        label="Executing RL policy",
        color="tab:green",
    )

    ax1.set_xticks(n_circles_approximate_vehicle_list)
    ax1.set_xlim(
        n_circles_approximate_vehicle_list[0] - 0.5,
        n_circles_approximate_vehicle_list[-1] + 0.5,
    )

    ax1.set_ylim(0, 0.06)
    ax1.set_xlabel("Number of circles approximating the vehicle")
    ax1.set_ylabel("Computation time per step [s]")
    ax1.tick_params(
        axis="both",
        direction="in",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.6, linewidth=0.5)

    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        n_circles_approximate_vehicle_list,
        diameter_width_ratio,
        color="tab:red",
        marker="o",
        linestyle="-",
        label="Diameter-to-width ratio",
    )
    ax2.set_ylabel("Diameter-to-width ratio")
    ax2.tick_params(axis="y", direction="in")

    ax2.legend(loc="upper right")
    ax2.set_ylim(1, max(diameter_width_ratio) * 1.2)

    fig.tight_layout(pad=0.3)

    plt.savefig(fig_path, format="pdf", dpi=450, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    plt.show()


def plot_actions(scenario_type="CPM_entire", steps=None, path=None, show_legend=True):
    """
    Visualize the actions of agents in a scenario.
    Args:
        scenario_type (str): The type of scenario to visualize, e.g., CPM_entire, interchange_3, intersection_4, intersection_6.
        steps (list): List of steps to visualize. If None, default steps for the scenario type will be used.
    """
    evaluation = CBF_MARL_Evaluation(
        path=path if path else "checkpoints/itsc25/undertrained_policy_1",
        n_agents=1,
        is_using_cbf=True,
        scenario_type=scenario_type if scenario_type else "CPM_entire",
        random_seed=1,
        n_circles_approximate_vehicle=3,
        is_load_out_td=True,
        obs_noise_level=0.0,
    )
    evaluation.evaluate()

    if steps is None:
        # Pre-defined steps for each scenario type
        if scenario_type == "CPM_entire":
            steps = [87, 177, 278, 598]
        elif scenario_type == "interchange_3":
            steps = [113, 303, 322, 358]
        elif scenario_type == "intersection_4":
            steps = [56, 129, 200, 400]
        elif scenario_type == "intersection_6":
            steps = [41, 61, 115, 552]

    for step in steps:
        evaluation.visualize_action_of_agent_i(
            step=step, show_legend=show_legend, legend_loc="upper center"
        )  # best, upper center, lower center, upper left, upper right, lower left, lower right


if __name__ == "__main__":
    policy_path = "checkpoints/itsc25"
    is_load_out_td = False

    print("What do you want to evaluate?")
    print("1. Computationally efficiency")
    print("2. Safety")
    user_choice = int(input("Enter your choice: "))
    if user_choice == 1:
        is_using_cbfs = [True]
        random_seeds = [1]
        n_circles_approximate_vehicle_list = [1, 2, 3, 4, 5]
        scenario_types = [
            "CPM_entire",
        ]

        (
            mean_computation_time_cbf,
            mean_computation_time_pseudo_dis,
            mean_computation_time_rl,
            collision_count_rl_cbf,
            collision_count_rl,
        ) = run_simulations(
            policy_path,
            is_using_cbfs,
            random_seeds,
            n_circles_approximate_vehicle_list,
            scenario_types,
            is_load_out_td,
        )

        fig_path = os.path.join(policy_path, "fig_computation_time.pdf")
        diameter_width_ratio = []
        for n in n_circles_approximate_vehicle_list:
            rectangle_circle_approximation = RectangleCircleApproximation(
                AGENTS["length"], AGENTS["width"], n
            )
            ratio = rectangle_circle_approximation.radius * 2 / AGENTS["width"]
            diameter_width_ratio.append(ratio)

        plot_computation_time(
            mean_computation_time_cbf,
            mean_computation_time_pseudo_dis,
            mean_computation_time_rl,
            n_circles_approximate_vehicle_list,
            diameter_width_ratio,
            fig_path,
        )

    elif user_choice == 2:
        is_using_cbfs = [True, False]
        random_seeds = [
            1
        ]  # Add more random seeds if needed. In our paper, we use [1, 2, 3, 4, 5].
        n_circles_approximate_vehicle_list = [3]
        scenario_types = [
            "CPM_entire",
            "interchange_3",
            "intersection_4",
            "intersection_6",
        ]

        (
            mean_computation_time_cbf,
            mean_computation_time_pseudo_dis,
            mean_computation_time_rl,
            collision_count_rl_cbf,
            collision_count_rl,
        ) = run_simulations(
            policy_path,
            is_using_cbfs,
            random_seeds,
            n_circles_approximate_vehicle_list,
            scenario_types,
            is_load_out_td,
        )

        print(f"Collision count with safety filter: \n {collision_count_rl_cbf}")

        print(f"Collision count without safety filter: \n {collision_count_rl}")

        print(f"Please check the figures and videos saved in {policy_path}")

        fig_path = os.path.join(policy_path, "fig_collision_count.pdf")
        plot_collision_count(
            collision_count_rl.mean(axis=0),
            collision_count_rl_cbf.mean(axis=0),
            n_circles_approximate_vehicle_list,
            fig_path,
        )
        user_choice = int(
            input(
                "Do you want to plot some snippets visualizing the safety filter being active and inactive? (1 for yes, 0 for no): "
            )
        )
        if user_choice == 1:
            scenario_types = [
                "CPM_entire",
                "interchange_3",
                "intersection_4",
                "intersection_6",
            ]
            for scenario_type in scenario_types:
                plot_actions(
                    scenario_type=scenario_type,
                    steps=None,
                    path=policy_path,
                    show_legend=False,
                )
