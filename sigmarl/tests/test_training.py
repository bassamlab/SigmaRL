import os
import shutil
import unittest
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.ppo_goal_reaching import ppo_goal_reaching
from sigmarl.helper_training import Parameters
from sigmarl.constants import SCENARIOS


class TestTrainingScenarios(unittest.TestCase):
    def setUp(self):
        """Set up temporary output directory for tests."""
        self.output_dir = "outputs/cicd_test/"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_training_scenarios(self):
        """Test training for different scenarios."""
        scenarios_to_test = ["intersection_1", "goal_reaching_1", "CPM_mixed"]

        for scenario in scenarios_to_test:
            with self.subTest(scenario=scenario):
                print(f"Testing scenario: {scenario}")

                if scenario.lower() == "goal_reaching_1":
                    config_file = "sigmarl/config_goal_reaching.json"
                else:
                    config_file = "sigmarl/config.json"

                # Load parameters
                parameters = Parameters.from_json(config_file)
                parameters.scenario_type = scenario
                parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
                parameters.where_to_save = os.path.join(self.output_dir, scenario)
                os.makedirs(parameters.where_to_save, exist_ok=True)
                print(f"Folder {parameters.where_to_save} created or already exists.")
                parameters.n_iters = (
                    5  # Reduce the number of iterations for quick testing
                )
                parameters.total_frames = (
                    parameters.frames_per_batch * parameters.n_iters
                )

                # Run training for the scenario
                if parameters.scenario_type.lower() == "goal_reaching_1":
                    env, decision_making_module, priority_module, parameters = ppo_goal_reaching(
                        parameters=parameters
                    )
                else:
                    env, decision_making_module, optimization_module, priority_module, parameters = mappo_cavs(
                        parameters=parameters
                    )

                self.assertTrue(os.path.exists(parameters.where_to_save))
                self.assertTrue(len(os.listdir(parameters.where_to_save)) > 1)
                print(f"Scenario {scenario} passed successfully!")


if __name__ == "__main__":
    unittest.main()
