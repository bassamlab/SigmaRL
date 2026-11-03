# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SOTA CBF baselines for the comparison with the exact rectangle_sat barriers.

Each baseline replaces one row type of the rectangle_sat CBF-QP ("barrier
swap"): the lane-boundary rows (``Parameters.lane_cbf_baseline``) or the
inter-vehicle rows (``Parameters.veh_veh_cbf_baseline``). B6
(``Parameters.condition_cbf_baseline``) instead keeps our barriers and replaces
the condition of our backup-horizon rows. Everything else
(nominal controller, QP, input box, decay rate, slacks, decentralized shares,
backup shield, standstill escape) stays that of the rectangle_sat pipeline.
Unlike ours, baseline rows are linearized once and neither repaired nor
certified.

One module per baseline, so that a baseline can be removed by deleting its
module and its name below:
  b1_lse_sat.py                 B1  log-sum-exp smoothed SAT barrier, lane and pair
  b2_frenet_lane.py             B2  Frenet lateral-offset lane barrier (point, heading)
  b3_disk_pair.py               B3  circumscribed-disk pair barriers: Wang et al.'s
                                    braking-aware certificate and the plain distance
  b4_collision_cone_pair.py     B4  collision-cone pair barrier (C3BF)
  b5_convex_containment_lane.py B5  convex free-space containment lane barrier
  t2_dual_distance.py           T2  optimization-based exact distance, lane and pair
  b6_backup_cbf.py              B6  backup CBF with per-flow-point decay (condition
                                    baseline of the backup-horizon rows)
  bh_variants.py                    variants of the backup-horizon rows: half split
                                    of cross pairs (bh_half_split), B1's and B3's
                                    margins on the tails (b1_bh, b3_bh), capsules with
                                    corner budgets (bh_capsule), and model
                                    predictive shielding (mps)
"unfiltered" adds no rows to either type, so that the QP returns the nominal
control (clipped to the input box) as the unfiltered reference.

tuned_defaults.json holds every method's tuned decay rate lambda and own
parameters per shield block and QP mode (see tuned_setting), including ours
(BH-TTCBF); they are the defaults when Parameters.cbf_lambda or the
parameter's environment variable is unset (see tuned_cbf_lambda).
"""

import json
import os

LANE_BASELINES = (
    "unfiltered",
    "b1_lse_sat",
    "b2_frenet_point",
    "b2_frenet_heading",
    "b5_convex_containment",
    "t2_dual_distance",
)
PAIR_BASELINES = (
    "unfiltered",
    "b1_lse_sat",
    "b3_wang",
    "b3_disk",
    "b4_collision_cone",
    "t2_dual_distance",
)
CONDITION_BASELINES = (
    "b6_backup_cbf",
    "bh_half_split",
    "b1_bh",
    "b3_bh",
    "bh_capsule",
    "mps",
)
TUNED_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "tuned_defaults.json")


def tuned_setting(parameters, method: str, block: str | None = None) -> dict:
    """Tuned setting {"cbf_lambda": ..., "env": {...}} of a method (a baseline
    name or "ours") for the active mode (fully decentralized if max_group_size
    is 1, else centralized) and block (default: the active one, exact backup
    shield on or off)."""
    if block is None:
        block = "shield_on" if parameters.is_exact_backup_shield else "shield_off"
    mode = "dec" if parameters.max_group_size == 1 else "cen"
    with open(TUNED_DEFAULTS_PATH) as file:
        return json.load(file)[f"{block}/{method}/{mode}"]


def tuned_cbf_lambda(parameters) -> float | None:
    """Tuned lambda of the selected method, or None if it has none.

    - A lane or pair baseline: its value for the active block and mode.
    - Ours (BH-TTCBF, is_backup_horizon_cbf without a condition baseline): its
      value tuned without the shield, which is not part of the method, also
      used with the shield on.
    - Our barriers with the one-step condition alone (ours_onestep: BH off,
      rectangle_sat footprints, fixed lambda): its value for the active block
      and mode.
    - A condition baseline (B6, the variants of bh_variants.py): its value
      tuned without the shield if tuned_defaults.json has one, else None.
    - None for other footprints, a learned lambda, and the unfiltered
      reference (no CBF rows).
    """
    names = {
        parameters.lane_cbf_baseline,
        parameters.veh_veh_cbf_baseline,
    } - {"none", "unfiltered"}
    if len(names) > 1:
        raise ValueError(
            f"No tuned lambda for the baseline mix {sorted(names)}; set "
            "Parameters.cbf_lambda."
        )
    if names:
        return tuned_setting(parameters, names.pop())["cbf_lambda"]
    if parameters.is_backup_horizon_cbf and parameters.condition_cbf_baseline == "none":
        return tuned_setting(parameters, "ours", block="shield_off")["cbf_lambda"]
    if parameters.is_backup_horizon_cbf:
        mode = "dec" if parameters.max_group_size == 1 else "cen"
        key = f"shield_off/{parameters.condition_cbf_baseline}/{mode}"
        with open(TUNED_DEFAULTS_PATH) as file:
            setting = json.load(file).get(key)
        return None if setting is None else setting["cbf_lambda"]
    if (
        not parameters.is_backup_horizon_cbf
        and not parameters.is_learn_cbf_lambda
        and parameters.lane_cbf_footprint == "rectangle_sat"
        and parameters.veh_veh_cbf_footprint == "rectangle_sat"
    ):
        return tuned_setting(parameters, "ours_onestep")["cbf_lambda"]
    return None


def tuned_env_value(parameters, method: str, variable: str) -> str:
    """Own parameter of a baseline: its environment variable if set (e.g., in
    a tuning sweep), else the tuned value."""
    return (
        os.environ.get(variable) or tuned_setting(parameters, method)["env"][variable]
    )


def make_backup_horizon(name: str, ctrl):
    """Backup-horizon rows of a CBFQP controller: ours (BH-TTCBF) for "none",
    else the condition baseline of that name."""
    if name == "none":
        from sigmarl.backup_horizon_cbf import BackupHorizonCBF

        return BackupHorizonCBF(ctrl)
    if name == "b6_backup_cbf":
        from sigmarl.cbf_baselines.b6_backup_cbf import BackupCBFPerPoint

        return BackupCBFPerPoint(ctrl)
    if name in ("bh_half_split", "b1_bh", "b3_bh", "bh_capsule", "mps"):
        from sigmarl.cbf_baselines import bh_variants

        variant = {
            "bh_half_split": bh_variants.BackupHorizonHalfSplit,
            "b1_bh": bh_variants.LseSatBackupHorizon,
            "b3_bh": bh_variants.DiskBackupHorizon,
            "bh_capsule": bh_variants.CapsuleBudgets,
            "mps": bh_variants.ModelPredictiveShield,
        }[name]
        return variant(ctrl)
    raise ValueError(
        f"Unknown condition_cbf_baseline '{name}'; one of {CONDITION_BASELINES}."
    )


def make_lane_baseline(name: str, ctrl):
    """Lane-boundary baseline of a CBFQP controller, or None for "none"."""
    if name == "none":
        return None
    if name == "unfiltered":
        from sigmarl.cbf_baselines.common import UnfilteredLaneBaseline

        return UnfilteredLaneBaseline(ctrl)
    if name == "b1_lse_sat":
        from sigmarl.cbf_baselines.b1_lse_sat import LseSatLaneBaseline

        return LseSatLaneBaseline(ctrl)
    if name in ("b2_frenet_point", "b2_frenet_heading"):
        from sigmarl.cbf_baselines.b2_frenet_lane import FrenetLaneBaseline

        return FrenetLaneBaseline(ctrl, is_heading_aware=name == "b2_frenet_heading")
    if name == "b5_convex_containment":
        from sigmarl.cbf_baselines.b5_convex_containment_lane import (
            ConvexContainmentLaneBaseline,
        )

        return ConvexContainmentLaneBaseline(ctrl)
    if name == "t2_dual_distance":
        from sigmarl.cbf_baselines.t2_dual_distance import DualDistanceLaneBaseline

        return DualDistanceLaneBaseline(ctrl)
    raise ValueError(f"Unknown lane_cbf_baseline '{name}'; one of {LANE_BASELINES}.")


def make_pair_baseline(name: str, ctrl):
    """Inter-vehicle baseline of a CBFQP controller, or None for "none"."""
    if name == "none":
        return None
    if name == "unfiltered":
        from sigmarl.cbf_baselines.common import UnfilteredPairBaseline

        return UnfilteredPairBaseline(ctrl)
    if name == "b1_lse_sat":
        from sigmarl.cbf_baselines.b1_lse_sat import LseSatPairBaseline

        return LseSatPairBaseline(ctrl)
    if name == "b3_wang":
        from sigmarl.cbf_baselines.b3_disk_pair import WangPairBaseline

        return WangPairBaseline(ctrl)
    if name == "b3_disk":
        from sigmarl.cbf_baselines.b3_disk_pair import DiskPairBaseline

        return DiskPairBaseline(ctrl)
    if name == "b4_collision_cone":
        from sigmarl.cbf_baselines.b4_collision_cone_pair import (
            CollisionConePairBaseline,
        )

        return CollisionConePairBaseline(ctrl)
    if name == "t2_dual_distance":
        from sigmarl.cbf_baselines.t2_dual_distance import DualDistancePairBaseline

        return DualDistancePairBaseline(ctrl)
    raise ValueError(f"Unknown veh_veh_cbf_baseline '{name}'; one of {PAIR_BASELINES}.")
