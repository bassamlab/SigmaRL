import json
import os
import sys

import torch

from sigmarl.helper_training import SaveData
from sigmarl.mappo_cavs import mappo_cavs

from pathlib import Path

# make path relative to project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.append(str(project_root))

path_in = "outputs/interface/reward6.33_data.json"

dir_out = "outputs/interface_test"

if __name__ == "__main__":

    with open(path_in, "r") as file:
        data = json.load(file)
        saved_data = SaveData.from_dict(data)
        parameters = saved_data.parameters

        parameters.is_load_model = True
        parameters.is_load_final_model = False

        env, decision_making_module, priority_module, parameters = mappo_cavs(
            parameters=parameters
        )

        policy_path_out = os.path.join(dir_out, "policy.pt")
        data_path_out = os.path.join(dir_out, "data.json")

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        torch.save(decision_making_module.policy, policy_path_out)  # save policy

        json_object = json.dumps(saved_data.to_dict(), indent=4)  # Serializing json
        with open(
            data_path_out, "w"
        ) as outfile:  # Writing to json file in corresponding directory
            outfile.write(json_object)
