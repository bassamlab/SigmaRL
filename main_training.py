from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.ppo_goal_reaching import ppo_goal_reaching

from sigmarl.helper_training import Parameters
from sigmarl.constants import SCENARIOS

scenario_type = "CPM_mixed"  # Training scenario. One of "CPM_mixed" (default), "CPM_entire", "intersection_1", "on_ramp_1", "roundabout_1", "goal_reaching_1", etc. See sigmarl/constants.py for more scenario types
if scenario_type.lower() == "goal_reaching_1":
    config_file = "sigmarl/config_goal_reaching.json"  # Adjust parameters therein
else:
    config_file = "sigmarl/config.json"  # Adjust parameters therein

parameters = Parameters.from_json(config_file)

# Set parameters
parameters.scenario_type = scenario_type
parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]

if not parameters.where_to_save:
    parameters.where_to_save = "outputs/tmp/"

if parameters.scenario_type.lower() == "goal_reaching_1":
    env, decision_making_module, priority_module, parameters = ppo_goal_reaching(parameters=parameters)
else:
    env, decision_making_module, priority_module, parameters = mappo_cavs(parameters=parameters)
