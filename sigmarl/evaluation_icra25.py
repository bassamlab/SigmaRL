# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from sigmarl.constants import SCENARIOS
from sigmarl.evaluation_base import Evaluation

model_paths = [
    "checkpoints/icra25/M1 (XP-MARL)/",
    "checkpoints/icra25/M2 (Baseline)/",
    "checkpoints/icra25/M3 (Baseline with perfect opponent modeling)/",
    "checkpoints/icra25/M4 (XP-MARL with random priorities)/",
    "checkpoints/icra25/M5 (XP-MARL with communication noise)/",
]

num_models = len(model_paths)
x_ticks = [f"$M_{{{idx}}}$" for idx in range(0, num_models)]

fig_sizes = {
    "episode_reward": (3.8, 3.6),
    "collision_rate": (3.5, 1.2),
    "centerline_deviation": (3.5, 2.0),
    "average_speed": (3.5, 1.2),
}

y_limits = {
    "episode_reward": [-1, 6],
    "collision_rate": [0, 2],
    "centerline_deviation": [0, 100],
    "average_speed": [70, 100],
}

idx_our = 0  # Index of our model (0-based index)

legends = [
    r"$M_{" + path.split("/")[-2][1] + r"}$ " + path.split("/")[-2][2:]
    for path in model_paths
]

legends[0] = "$M_{1}$  (XP-MARL, our)"

render_titles = [path.rsplit("/", 2)[-2] for path in model_paths]

video_names = [path.rsplit("/", 2)[-2][0:2] for path in model_paths]

scenario_types = [
    # "CPM_mixed",
    "CPM_entire",
    # "intersection_2",
    # "on_ramp_1",
    # "roundabout_1",
]

for i_scenario in scenario_types:
    print("*****************************************")
    print("*****************************************")
    print(f"[INFO] Scenario: {i_scenario}")
    print("*****************************************")
    print("*****************************************")

    n_agents = SCENARIOS[i_scenario]["n_agents"]

    evaluator = Evaluation(
        scenario_type=i_scenario,  # Specify which scenario should be used to do the evaluation. One of {"CPM_entire", "intersection_2", "on_ramp_1", "roundabout_1"}
        model_paths=model_paths,
        fitst_model_index=1,  # The index of the first model. If 1, then models are indexed as M1, M2, ...
        idx_our=idx_our,  # Index of our model (0-based index)
        x_tick_label_rotation=0,  # the rotation of x tick label in degrees
        fig_sizes=fig_sizes,
        y_limits=y_limits,
        is_show_different_collisions=False,  # Two types of collisions are distinguished: agent-agent collisions, agent-lanelet collisions. Set to false if only the total collision rate is of interest.
        n_agents=n_agents,  # Number of agents to be used in the evaluation
        simulation_steps=1200,  # Number of time steps of each simulation. 1200 -> 1 min if sample time is 50 ms
        where_to_save_eva_results=f"checkpoints/icra25/eva_{i_scenario}",
        where_to_save_logging=f"checkpoints/icra25/log.txt",
        models_selected=[],  # Leave empty if all the models should be evaluated
        legends=legends,
        render_titles=render_titles,
        num_simulations_per_model=32,
        is_render=False,
        is_real_time_rendering=True,
        is_save_simulation_video=False,
        video_names=video_names,
    )

    evaluator.run_evaluation()
