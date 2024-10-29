from utilities.mappo_cavs import mappo_cavs
from utilities.ppo_goal_reaching import ppo_goal_reaching
from utilities.helper_training import Parameters
from utilities.constants import SCENARIOS

config_file = "config.json"  # Adjust parameters therein
parameters = Parameters.from_json(config_file)
parameters.n_iters = 200
parameters.total_frames = parameters.frames_per_batch * parameters.n_iters
parameters.is_add_noise = True
parameters.scenario_type = "goal_reaching_1"
parameters.where_to_save = "outputs/goal_reaching_v14/"
parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]

ppo_goal_reaching(parameters=parameters)
# mappo_cavs(parameters=parameters)
