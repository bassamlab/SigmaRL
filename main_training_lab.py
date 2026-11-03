# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""On-twin fine-tuning of a trained SigmaRL policy against the CPM-lab backend.

Runs the unchanged MAPPO loop (collect -> GAE -> PPO) with the physics layer
delegated to the lab (or the in-process mock). Each iteration collects
``frames_per_batch`` (= max_steps, since num_vmas_envs == 1) transitions, then
the PPO update runs while the fleet is held stopped by the bridge watchdog;
``reset_at_each_iter`` respawns at the start of every iteration so no
transition spans the pause.

The pretrained model (reward*.json + *_policy.pth + *_critic.pth) is copied
into --out-dir first so new checkpoints never clobber the original.

Usage:
    python main_training_lab.py --backend mock --model-dir <dir> --n-iters 2
    python main_training_lab.py --backend lab --model-dir <dir> \
        --vehicle-ids 1,2,3,4 --n-iters 3
"""

import argparse
import json
import os
import shutil
from datetime import datetime

from sigmarl.helper_common import Parameters, start_run_log
from sigmarl.helper_training import SaveData
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.lab.bridge_base import LabStopped, LabStateTimeout

# ===============================
# Check if running in HPC environment
# ===============================
hpc_environment = os.getenv("HPC_ENVIRONMENT", "false").lower() == "true"
if hpc_environment:
    print(
        "HPC environment detected. Setting the environment variable PYGLET_HEADLESS to True."
    )
    os.environ["PYGLET_HEADLESS"] = "True"  # Enable if run in HPC
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Use offscreen rendering


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="outputs/my_path_mode_models/",
        help="Directory holding the pretrained reward*.json + *_policy.pth + *_critic.pth.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write fine-tuned checkpoints (default: <model-dir>/lab_finetune_<ts>).",
    )
    parser.add_argument(
        "--scenario-type",
        default="cpm_entire",
        help="One of the scenario types in sigmarl/constants.py",
    )
    parser.add_argument("--backend", choices=["lab", "mock"], default="lab")
    parser.add_argument("--vehicle-ids", default="1,2,3,4")
    parser.add_argument("--period-ms", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--load-final-model", action=argparse.BooleanOptionalAction
    )  # If you wrote --load-final-model it will be True don't write it in the terminal to be false
    parser.add_argument(
        "--rew-method",
        type=str,
        default=None,  # Combination of keywords "cbf", "ttc", "distance", "sparse", "cbfcorrection", "cbfmargin", "cbflambda" to indicate the reward method. For example, "cbfcorrection_sparse" means using the CBF correction reward and sparse reward; "cbfmargin_cbflambda" means using the CBF margin reward and CBF lambda reward.
    )
    parser.add_argument(
        "--manual-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Human-in-the-loop reset for REAL hardware: stop the fleet on "
        "every reset, wait for the operator to place the vehicles by hand, and "
        "assign reference paths from the measured poses. Default off (sampled "
        "poses teleported via /lab/respawn_vehicles, i.e. the digital twin). "
        "Expect a few startup prompts plus one per iteration "
        "(reset_at_each_iter is on).",
    )
    return parser.parse_args()


def find_reward_basename(model_dir: str) -> str:
    for name in os.listdir(model_dir):
        if name.endswith(".json") and name.startswith("reward"):
            return (
                name[: -len("_data.json")]
                if name.endswith("_data.json")
                else name[: -len(".json")]
            )
    raise FileNotFoundError(f"No reward*.json found in {model_dir}")


def stage_model(model_dir: str, out_dir: str) -> str:
    """Copy the pretrained model into out_dir so _load_existing_model finds it
    there and fine-tuned checkpoints never overwrite the original."""
    os.makedirs(out_dir, exist_ok=True)
    base = find_reward_basename(model_dir)
    copied_json = None
    for name in os.listdir(model_dir):
        if name.startswith(base) and (name.endswith(".json") or name.endswith(".pth")):
            shutil.copy2(os.path.join(model_dir, name), os.path.join(out_dir, name))
            if name.endswith(".json"):
                copied_json = os.path.join(out_dir, name)
    if copied_json is None:
        raise FileNotFoundError(f"No {base}*.json staged into {out_dir}")
    return copied_json


def main():
    args = parse_args()
    vehicle_ids = [int(v) for v in args.vehicle_ids.split(",")]
    out_dir = args.out_dir or os.path.join(
        args.model_dir, f"lab_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    staged_json = stage_model(args.model_dir, out_dir)
    with open(staged_json, "r") as f:
        parameters: Parameters = SaveData.from_dict(json.load(f)).parameters

    assert parameters.n_agents == len(vehicle_ids), (
        f"Model was trained for {parameters.n_agents} agents but "
        f"{len(vehicle_ids)} vehicle IDs were given."
    )
    parameters.is_zero_initial_speed = True  # Not important because the lab reset will override the initial state with the lab's actual state

    # Continue-training overrides: load the staged model and keep optimizing it.
    parameters.is_testing_mode = False
    parameters.is_load_model = True
    parameters.is_load_final_model = args.load_final_model
    parameters.is_continue_train = True
    parameters.where_to_save = out_dir + os.sep
    start_run_log(out_dir, "training")
    # The staged JSON carries the *base* model's best reward in episode_reward_intermediate (the save gate in mappo_cavs._save_intermediate_model).
    parameters.episode_reward_intermediate = -1e3
    parameters.num_vmas_envs = 1
    parameters.device = "cpu"
    parameters.scenario_type = args.scenario_type
    parameters.dt = args.period_ms / 1000.0
    parameters.max_steps = args.max_steps
    parameters.n_iters = args.n_iters
    parameters.num_epochs = args.num_epochs
    parameters.minibatch_size = args.minibatch_size
    parameters.random_seed = args.seed
    parameters.rew_method = (
        args.rew_method if args.rew_method is not None else parameters.rew_method
    )  # Use the command-line argument 'reward' if provided; otherwise, keep the pretrained model's reward method.
    parameters.reset_at_each_iter = True
    parameters.is_save_simulation_video = False
    parameters.is_real_time_rendering = False
    parameters.is_prb = False
    parameters.is_using_prioritized_marl = False
    ################################## CBF #####################################
    parameters.is_using_centralized_cbf = True
    parameters.is_using_cbf_training = True
    parameters.is_using_cbf_testing = False
    parameters.is_apply_cbf_action = True
    parameters.is_solve_qp = True
    parameters.is_keep_nominal_when_safe = False
    parameters.is_store_rl_action_under_cbf = True
    ############################################################################
    parameters.entropy_eps = 1e-3
    parameters.lr = 3e-5
    parameters.lr_min = 3e-6
    parameters.clip_epsilon = 0.05
    parameters.max_grad_norm = 0.5

    # One VMAS env => frames_per_batch == max_steps. The pretrained JSON's
    # minibatch_size (e.g. 512) would exceed it, so clamp.
    parameters.minibatch_size = min(
        parameters.minibatch_size, parameters.frames_per_batch
    )

    parameters.physics_backend = args.backend
    parameters.lab_vehicle_ids = vehicle_ids
    parameters.lab_period_ms = args.period_ms
    parameters.lab_manual_reset = args.manual_reset
    if args.manual_reset:
        # The pretrained JSON may carry pose-imposing settings; manual mode
        # must derive everything from the measured poses. The scenario asserts
        # on these too, so this is a convenience rather than the safety net.
        parameters.predefined_ref_path_idx = None
        parameters.init_state = None
        parameters.is_challenging_initial_state_buffer = False
        # Would otherwise prompt the operator on a wall-clock timer.
        parameters.reset_agent_fixed_duration = 0

    print(
        f"[INFO] Fine-tuning {staged_json} on backend={args.backend}, "
        f"{args.n_iters} iters x {parameters.frames_per_batch} frames "
        f"(minibatch {parameters.minibatch_size}); checkpoints -> {out_dir}"
    )

    env = None
    try:
        env, *_ = mappo_cavs(parameters=parameters)
        print(f"[INFO] Fine-tuning complete; checkpoints in {out_dir}")
    except (LabStopped, LabStateTimeout) as exc:
        print(f"[ERROR] Lab interrupted fine-tuning: {exc}")
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        if env is not None:
            env.scenario.bridge.close()


if __name__ == "__main__":
    main()
