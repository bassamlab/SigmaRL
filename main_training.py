from sigmarl.constants import SCENARIOS
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.helper_training import Parameters
from sigmarl.ppo_goal_reaching import ppo_goal_reaching

config_file = "sigmarl/config.json"  # Adjust parameters therein
parameters = Parameters.from_json(config_file)
# parameters.n_iters = 250
# parameters.total_frames = parameters.frames_per_batch * parameters.n_iters
parameters.is_add_noise = True
# parameters.scenario_type = "road_traffic"
parameters.where_to_save = "outputs/test_tmp/"
# parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]

# ppo_goal_reaching(parameters=parameters)
mappo_cavs(parameters=parameters)
