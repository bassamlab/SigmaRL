# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from sigmarl.cbf import main

scenario_types = ["overtaking", "bypassing"]
sm_types = ["c2c", "mtv"]

for scenario_type in scenario_types:
    for sm_type in sm_types:
        print(f"Scenario type: {scenario_type}; Safety measure type: {sm_type}")
        main(
            scenario_type=scenario_type,  # One of "overtaking" and "bypassing"
            sm_type=sm_type,  # One of "c2c" and "mtv"
            is_save_video=False,
            is_visu_ref_path=False,
            is_visu_nominal_action=True,
            is_visu_cbf_action=True,
            is_save_eval_result=True,
        )
