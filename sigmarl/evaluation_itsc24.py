# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from sigmarl.constants import SCENARIOS
from sigmarl.evaluation_base import Evaluation


fig_for = "paper"  # One of {"paper", "presentation"}

model_paths = [
    "checkpoints/itsc24/M0 (our)/",
    "checkpoints/itsc24/M1 (do not use an ego view)/",
    "checkpoints/itsc24/M2 (do not observe vertices of surrounding agents)/",
    "checkpoints/itsc24/M3 (do not observe distances to surrounding agents)/",
    "checkpoints/itsc24/M4 (do not observe distances to lane boundaries)/",
    "checkpoints/itsc24/M5 (do not observe distances to lane center lines)/",
]

num_models = len(model_paths)
x_ticks = [f"$M_{{{idx}}}$" for idx in range(0, num_models)]

y_limits = {
    "episode_reward": [-1, 6],
    "collision_rate": [0, 3],
    "centerline_deviation": [0, 100],
    "average_speed": [70, 100],
}

# Figures are different in the paper and in the presentation
if fig_for.lower() == "paper":
    fig_sizes = {
        "episode_reward": (3.8, 4.2),
        "collision_rate": (3.5, 2.0),
        "centerline_deviation": (3.5, 2.0),
        "average_speed": (3.5, 2.0),
    }

    legends = [
        r"$M_{" + path.split("/")[-2][1] + r"}$ " + path.split("/")[-2][2:]
        for path in model_paths
    ]
    is_show_different_collisions = True
else:
    # For presentation
    fig_sizes = {
        "episode_reward": (4.5, 4.4),
        "collision_rate": (3.5, 2.0),
        "centerline_deviation": (3.5, 2.0),
        "average_speed": (3.5, 2.0),
    }

    legends = [
        r"$M_{0}$ (our)",
        r"$M_{1}$ (bird-eye view instead of ego view)",
        r"$M_{2}$ (poses and dimensions instead of vertices)",
        r"$M_{3}$ (does not observe distances)",
        r"$M_{4}$ (sampled points from lane boundaries instead of distances)",
        r"$M_{5}$ (does not observe lane center lines)",
    ]
    is_show_different_collisions = False

render_titles = [path.rsplit("/", 2)[-2] for path in model_paths]

video_names = [path.rsplit("/", 2)[-2][0:2] for path in model_paths]

scenario_types = [
    "CPM_entire",
    "intersection_2",
    "on_ramp_1",
    "roundabout_1",
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
        fitst_model_index=0,  # The index of the first model. If 1, then models are indexed as M1, M2, ...
        n_agents=n_agents,  # Number of agents to be used in the evaluation
        fig_sizes=fig_sizes,
        y_limits=y_limits,
        simulation_steps=1200,  # Number of time steps of each simulation. 1200 -> 1 min if sample time is 50 ms
        is_show_different_collisions=is_show_different_collisions,
        x_ticks=x_ticks,
        where_to_save_eva_results=f"checkpoints/itsc24/eva_{i_scenario}",
        where_to_save_logging=f"checkpoints/itsc24/log.txt",
        models_selected=[],  # Leave empty if all the models should be evaluated
        legends=legends,
        render_titles=render_titles,
        num_simulations_per_model=32,
        is_render=False,
        is_save_simulation_video=False,
        video_names=video_names,
    )

    evaluator.run_evaluation()
