# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Settings shared by the stages of the T-ITS evaluation."""

# Maps of the evaluation (scenario types)
MAPS = ("round_location0", "ind_location4")
# Vehicles per map, the same in training and deployment
N_AGENTS = {"round_location0": 16, "ind_location4": 12}

# Stage 2 fine-tuning iterations whose lane-boundary contact steps are excluded from the reported
# collisions: one step each of B6 and MPS at the inD route-1 curb, where the boundary vertices lag their
# centerline index (an artifact of the HD map, not of the filters).
# Keys: (run folder relative to stage2_finetune, iteration).
STAGE2_EXCLUDED_LANE_STEPS = {
    (
        "ind_location4__stage1_ttc_seed_2__iter100/ind_location4/b6_backup_cbf/seed1",
        12,
    ): 1,
    (
        "ind_location4__stage1_ttc_seed_2__iter100/ind_location4/b6_backup_cbf/seed1",
        77,
    ): 1,
    ("ind_location4__stage1_ttc_seed_1__iter100/ind_location4/mps/seed1", 17): 1,
}
