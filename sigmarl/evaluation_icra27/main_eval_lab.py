# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate a trained SigmaRL policy against the CPM-lab physics backend.

Mirrors main_testing.py, but executes rollouts through the lab (or the
in-process mock) instead of the in-process VMAS integrator. Saves both the
rollout TensorDict (.td) and an evaluation summary JSON containing collision
rates, task performance, and computational efficiency metrics.

Supported Control Modes (--lab-mode):
    speed_curvature:
        Sends instantaneous (speed, curvature) commands (VehicleCommand type 3)
        to the lab mid-level controller each control step (default).
    trajectory:
        Sends multi-step trajectory waypoints (VehicleCommand type 4). The
        policy is iteratively unrolled for --lab-traj-horizon virtual steps
        using the kinematic bicycle model and padded with constant deceleration
        actions to meet the lab MPC lead-time requirement (>= 0.40 s).

Usage:
    # In-process loopback in speed_curvature mode (default):
    python main_eval_lab.py --backend mock --lab-mode speed_curvature --model-dir checkpoints/icra27/lab --period-ms 50

    # In-process loopback in trajectory mode:
    python main_eval_lab.py --backend mock --lab-mode trajectory --lab-traj-horizon 5 --model-dir checkpoints/icra27/lab --period-ms 50

    # Digital twin in ROS (speed_curvature mode):
    python main_eval_lab.py --backend lab --lab-mode speed_curvature --model-dir <dir> --vehicle-ids 1,2,3,4 --period-ms 50

    # Digital twin in ROS (trajectory mode):
    python main_eval_lab.py --backend lab --lab-mode trajectory --lab-traj-horizon 5 --model-dir <dir> --vehicle-ids 1,2,3,4 --period-ms 50

    # Physical lab hardware (manual vehicle positioning on reset):
    python main_eval_lab.py --backend lab --lab-mode trajectory --lab-traj-horizon 5 --model-dir checkpoints/icra27/lab --vehicle-ids 1,2,3,4 --period-ms 50 --manual-reset
"""

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from sigmarl.helper_common import Parameters, Tee
from sigmarl.helper_training import SaveData, save_evaluation_summary
from sigmarl.mappo_cavs import mappo_cavs
from sigmarl.lab.bridge_base import LabStopped, LabStateTimeout


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="outputs/my_path_mode_models/",
        help="Directory holding one reward*.json and matching *_policy.pth.",
    )
    parser.add_argument(
        "--backend",
        choices=["lab", "mock"],
        default="lab",
        help="'lab' = real ROS lab; 'mock' = in-process loopback.",
    )
    parser.add_argument(
        "--vehicle-ids",
        default="1,2,3,4",
        help="Comma-separated lab vehicle IDs, one per agent.",
    )
    parser.add_argument(
        "--scenario-type",
        default="cpm_intersection_extended",
        help="One of the scenario types in sigmarl/constants.py",
    )
    parser.add_argument(
        "--load-final-model", action=argparse.BooleanOptionalAction
    )  # If you wrote --load-final-model it will be True don't write it in the terminal to be false
    parser.add_argument("--period-ms", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument(
        "--manual-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Human-in-the-loop reset for REAL hardware: stop the fleet on "
        "every reset, wait for the operator to place the vehicles by hand, and "
        "assign reference paths from the measured poses. Default off (sampled "
        "poses teleported via /lab/respawn_vehicles, i.e. the digital twin). "
        "Note --n-rollouts rebuilds the environment per rollout, so each "
        "rollout adds its own startup prompts; prefer --n-rollouts 1.",
    )
    parser.add_argument(
        "--without-lanelet-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, disable lanelet respawn on agent reset (for lab/mock backends). This is useful for evaluating the policy in a more realistic setting where lanelet collisions do not cause a reset.",
    )
    parser.add_argument(
        "--fast-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, enable fast inference mode (e.g. skip reward computations during evaluation).",
    )
    parser.add_argument(
        "--lab-mode",
        choices=["direct", "path_tracking", "speed_curvature", "trajectory"],
        default="speed_curvature",
        help="Lab control mode: 'speed_curvature' (default) or 'trajectory'.",
    )
    parser.add_argument(
        "--lab-traj-horizon",
        type=int,
        default=3,
        help="Number of steps for policy-generated virtual trajectory (default 3; relevant for trajectory mode).",
    )
    return parser.parse_args()


def find_reward_json(model_dir: str) -> str:
    for name in os.listdir(model_dir):
        if name.endswith(".json") and name.startswith("reward"):
            return os.path.join(model_dir, name)
    raise FileNotFoundError(f"No reward*.json found in {model_dir}")


def main(args):
    vehicle_ids = [int(v) for v in args.vehicle_ids.split(",")]

    json_path = find_reward_json(args.model_dir)
    with open(json_path, "r") as f:
        parameters: Parameters = SaveData.from_dict(json.load(f)).parameters

    cli_dt = args.period_ms / 1000.0
    if not math.isclose(parameters.dt, cli_dt, rel_tol=1e-3, abs_tol=1e-5):
        raise ValueError(
            f"Loaded dt from model ({parameters.dt} s) does not match the CLI specified period ({args.period_ms} ms / {cli_dt} s)."
        )

    parameters.n_agents = len(vehicle_ids)
    parameters.is_zero_initial_speed = True  # Not important because the lab reset will override the initial state with the lab's actual state

    # Inference overrides (mirror main_testing.py), plus the lab backend.
    parameters.is_testing_mode = True
    parameters.is_load_model = True
    parameters.is_load_final_model = args.load_final_model
    parameters.is_continue_train = False
    parameters.is_load_out_td = False
    parameters.where_to_save = args.model_dir
    parameters.num_vmas_envs = 1
    parameters.device = "cpu"
    parameters.scenario_type = args.scenario_type
    parameters.max_steps = args.max_steps
    parameters.random_seed = args.seed
    parameters.is_save_simulation_video = False
    parameters.is_real_time_rendering = False
    ########################## CBF ################################
    parameters.is_using_centralized_cbf = True
    parameters.is_using_cbf_testing = False
    parameters.is_using_cbf_training = False
    parameters.is_apply_cbf_action = False
    parameters.is_solve_qp = False
    parameters.is_keep_nominal_when_safe = False
    parameters.is_store_rl_action_under_cbf = False
    ###############################################################
    parameters.rew_method = "sparse"  # Use sparse reward for evaluation, regardless of the pretrained model's reward function
    parameters.physics_backend = args.backend
    parameters.without_lanelet_reset = args.without_lanelet_reset
    parameters.is_fast_inference_mode = args.fast_inference
    parameters.lab_vehicle_ids = vehicle_ids
    parameters.lab_period_ms = args.period_ms
    parameters.lab_manual_reset = args.manual_reset
    parameters.lab_mode = args.lab_mode
    parameters.lab_traj_horizon = args.lab_traj_horizon
    if args.manual_reset:
        # The pretrained JSON may carry pose-imposing settings; manual mode
        # must derive everything from the measured poses. The scenario asserts
        # on these too, so this is a convenience rather than the safety net.
        parameters.predefined_ref_path_idx = None
        parameters.init_state = None
        parameters.is_challenging_initial_state_buffer = False

    print(
        f"[INFO] Model directory: {args.model_dir}\n"
        f"[INFO] Backend: {args.backend}; "
        f"Mode: {parameters.lab_mode} (horizon: {parameters.lab_traj_horizon}); "
        f"Scenario: {parameters.scenario_type}; "
        f"Agents: {parameters.n_agents} (vehicles: {vehicle_ids});\n"
        f"[INFO] Max steps: {parameters.max_steps}; "
        f"Random seed: {parameters.random_seed}; "
        f"dt: {parameters.dt} s ({args.period_ms} ms); "
        f"Manual reset: {parameters.lab_manual_reset}; "
        f"Without lanelet reset: {parameters.without_lanelet_reset}; "
        f"Fast inference: {parameters.is_fast_inference_mode}"
    )

    (
        env,
        decision_making_module,
        _,
        priority_module,
        cbf_controllers,
        parameters,
    ) = mappo_cavs(parameters=parameters)

    if hasattr(env.scenario, "set_policy"):
        env.scenario.set_policy(decision_making_module.policy)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.model_dir, f"out_td_lab_{timestamp}_seed_{parameters.random_seed}.td"
    )
    print(f"[INFO] Seed {parameters.random_seed}")
    try:
        t_0 = time.time()
        with torch.inference_mode(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ):
            out_td, _ = env.rollout(
                max_steps=parameters.max_steps - 1,
                policy=decision_making_module.policy,
                priority_module=priority_module,
                cbf_controllers=cbf_controllers,
                callback=None,  # no rendering: the lab is the environment
                auto_cast_to_device=True,
                break_when_any_done=False,
                is_save_simulation_video=False,
            )
        simulation_time = time.time() - t_0
        num_frames = out_td.numel()
        simulation_speed = num_frames / simulation_time if simulation_time > 0 else 0.0
        real_time_factor = (
            num_frames * parameters.dt / simulation_time if simulation_time > 0 else 0.0
        )
        print(f"[INFO] Lab rollout time: {simulation_time:.2f} s")
        print(f"[INFO] Simulation speed: {simulation_speed:.2f} FPS")
        print(f"[INFO] Real-time factor: {real_time_factor:.2f}x")

        torch.save(out_td, out_path)
        print(f"[INFO] Saved rollout to {out_path}")
    except (LabStopped, LabStateTimeout) as exc:
        print(f"[ERROR] Lab interrupted the rollout: {exc}")
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        env.scenario.bridge.close()


if __name__ == "__main__":
    args = parse_args()
    log_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "eval_logs.txt"
    rel_log_path = os.path.relpath(log_path)

    with log_path.open("a", encoding="utf-8") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                for rollout_idx in range(args.n_rollouts):
                    args.seed = (rollout_idx + 1) % (10 + 1)
                    print(
                        f"[INFO] Starting rollout {rollout_idx + 1}/{args.n_rollouts} seed {args.seed}"
                    )
                    main(args)
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                print(f"[INFO] Saved log file to {rel_log_path}")
