import os
import shutil
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.ppo_goal_reaching import ppo_goal_reaching
from sigmarl.helper_training import Parameters
from sigmarl.constants import SCENARIOS


def test_training():
    scenarios_to_test = ["goal_reaching_1", "CPM_mixed"]
    output_dir = "outputs/tmp_cicd/"

    for scenario in scenarios_to_test:
        print(f"Testing scenario: {scenario}")

        if scenario.lower() == "goal_reaching_1":
            config_file = "sigmarl/config_goal_reaching.json"
        else:
            config_file = "sigmarl/config.json"

        # Load parameters
        parameters = Parameters.from_json(config_file)
        parameters.scenario_type = scenario
        parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
        parameters.where_to_save = output_dir
        parameters.n_iters = 10  # Reduce the number of iterations for quick testing
        parameters.total_frames = parameters.frames_per_batch * parameters.n_iters

        # Ensure output directory exists
        os.makedirs(parameters.where_to_save, exist_ok=True)

        # Run training for the scenario
        if parameters.scenario_type.lower() == "goal_reaching_1":
            env, policy, priority_module, parameters = ppo_goal_reaching(
                parameters=parameters
            )
        else:
            env, policy, priority_module, parameters = mappo_cavs(parameters=parameters)

        print(f"Scenario {scenario} passed successfully!")

    # Cleanup after testing
    shutil.rmtree(output_dir, ignore_errors=True)
    print("Temporary outputs cleaned up.")


if __name__ == "__main__":
    test_training()
