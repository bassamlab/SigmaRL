from sigmarl.mappo_cavs import mappo_cavs

from sigmarl.helper_training import Parameters
from sigmarl.constants import SCENARIOS

scenario_type = "CPM_mixed"  # Training scenario. One of "CPM_mixed" (default), "CPM_entire", "intersection_1", "on_ramp_1", "roundabout_1", etc. See sigmarl/constants.py for more scenario types
config_file = "sigmarl/config.json"  # Adjust parameters therein

parameters = Parameters.from_json(config_file)

# Set parameters
parameters.scenario_type = scenario_type
parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]


parameters.where_to_save = "outputs/tmp/"
parameters.n_iters = 250
parameters.random_seed = 0
parameters.is_using_cbf = False

(
    env,
    decision_making_module,
    optimization_module,
    priority_module,
    parameters,
) = mappo_cavs(parameters=parameters)
