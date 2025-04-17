import os
import json
from termcolor import cprint

from vmas.simulator.utils import save_video

from sigmarl.helper_training import Parameters, SaveData
from sigmarl.mappo_cavs import mappo_cavs


class CBF_MARL_Evaluation:
    def evaluate(self, path, i):
        self.path = path
        path_to_json_file = next(
            os.path.join(self.path, file)
            for file in os.listdir(self.path)
            if file.endswith(".json")
        )

        with open(path_to_json_file, "r") as file:
            data = json.load(file)
            saved_data = SaveData.from_dict(data)
            parameters = saved_data.parameters
            parameters.is_testing_mode = True
            parameters.is_real_time_rendering = False
            parameters.is_save_eval_results = False
            parameters.is_load_out_td = False
            parameters.max_steps = 1000
            parameters.num_vmas_envs = 1
            parameters.scenario_type = "CPM_entire"
            parameters.n_agents = i  # Number of agents
            parameters.is_save_simulation_video = True
            parameters.is_visualize_short_term_path = True
            parameters.is_visualize_lane_boundary = False
            parameters.is_visualize_extra_info = True

            # Parameter settings for CBF-constrained MARL
            parameters.is_using_cbf = True
            # parameters.is_using_prioritized_marl = (
            # True  # Use priority-based solving strategy
            # )
            parameters.is_using_centralized_cbf = False
            parameters.is_using_cbf_in_testing = (
                True  # Incooperate CBF-constrained MARL in the execution phase.
            )

            # Initialize environment
            env, policy, priority_module, cbf_controllers, parameters = mappo_cavs(
                parameters=parameters
            )

            out_td, frame_list = env.rollout(
                max_steps=parameters.max_steps - 1,
                policy=policy,
                priority_module=priority_module,
                callback=lambda env, _: env.render(
                    mode="rgb_array", visualize_when_rgb=True
                ),
                auto_cast_to_device=True,
                break_when_any_done=False,
                is_save_simulation_video=parameters.is_save_simulation_video,
                cbf_controllers=cbf_controllers,
            )
            self.save_date(out_td, env, i)
            save_video(f"{self.path}{i}agents_video", frame_list, fps=1 / parameters.dt)

    def save_date(self, out_td, env, i):
        """Save evaluation related data"""
        # Retrieve data
        is_collision_with_agents = out_td[
            "agents", "info", "is_collision_with_agents"
        ].bool()
        is_collision_with_lanelets = out_td[
            "agents", "info", "is_collision_with_lanelets"
        ].bool()
        is_collide = is_collision_with_agents | is_collision_with_lanelets

        positions = out_td["agents", "info", "pos"]
        velocities = out_td["agents", "info", "vel"]

        num_steps = positions.shape[1]

        # Collision rate
        collision_rate_a2a = (
            is_collision_with_agents.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
        )
        collision_rate_a2l = (
            is_collision_with_lanelets.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
        )
        collision_rate = is_collide.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps

        # Avereage speed
        average_speed = velocities.norm(dim=-1).mean(dim=(-2, -1))

        # Time
        time_rl = sum(env.time_rl) / len(env.time_rl)
        time_cbf = sum(env.time_cbf) / len(env.time_cbf)
        time_pseudo_dis = sum(env.time_pseudo_dis) / len(env.time_pseudo_dis)

        # Number of failure solution:
        num_fail = env.num_fail
        failure_rate = num_fail / (num_steps * i)

        #  Write evaluation results
        with open(f"{self.path}{i}agents_output.txt", "w") as file:
            file.write(
                f"Collision rate for agent to agent: {collision_rate_a2a.item()}\n"
            )
            file.write(
                f"Collision rate for agent to lane: {collision_rate_a2l.item()}\n"
            )
            file.write(f"Total collision rate: {collision_rate.item()}\n")
            file.write(f"time_rl: {time_rl}\n")
            file.write(f"time_cbf: {time_cbf}\n")
            file.write(f"time_dis: {time_pseudo_dis}\n")
            file.write(f"average_speed: {average_speed}\n")
            file.write(f"Infeasibility rate: {failure_rate}\n")


if __name__ == "__main__":
    number_agent = 10
    # Path of policies
    rough_policy_path = "CBF_evaluation/CBF_constrained_MARL/Rough_policy/"
    greedy_policy_path = "CBF_evaluation/CBF_constrained_MARL/Greedy_policy/"

    # Evaluate the CBF-constrained MARL and save results
    try:
        evaluation = CBF_MARL_Evaluation()
        evaluation.evaluate(rough_policy_path, number_agent)
        cprint(
            "[INFO] Evaluation for CBF-constrained MARL completed successfully.",
            "green",
        )
    except FileNotFoundError as e:
        cprint(
            f"[ERROR] File not found: {e}. Please ensure the correct path for the policy files.",
            "red",
        )
