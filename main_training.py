from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.helper_training import Parameters

config_file = "sigmarl/config.json"  # Adjust parameters therein
parameters = Parameters.from_json(config_file)

parameters.where_to_save = "outputs/tmp/"

mappo_cavs(parameters=parameters)
