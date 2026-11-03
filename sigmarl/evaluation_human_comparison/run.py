# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Let a controller drive human-driven vehicles of rounD location 0 and store the trajectories.

Each episode starts the controlled vehicles at the recorded states of human drivers, gives them the
humans' routes and the points where the humans left the recording as goals, and replays all other road
users (see `episodes.py` and `sigmarl/log_replay.py`). Results are stored per episode as pickles under
`<out_dir>/<mode>/<controller>/`; existing results are skipped, so runs can be resumed.

Example (run from the repository root):

    SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_human_comparison.run \
        --controller cbf_qp_dec --recordings 5,11,17,23 --workers 6

`--mode` defaults to homogeneous. A controlled vehicle enters at its recorded state only if the joint state
including it is certified (`--entry_check certified`, default with a CBF-QP controller: its lane and pair
backup values are nonnegative, see `CertifiedEntryCheck`); otherwise its entry is delayed. `--entry_check
distance` restores the earlier distance gate (`LogReplay.spawn_gap_time`).

Controlled vehicles have the nominal vehicle parameters of the agent profile (`--vehicles nominal`, default) or
those of the recorded vehicles they replace (`--vehicles recorded`: recorded footprint, axle distances scaled
with the length, acceleration limits of the class; see `episodes.recorded_vehicle_params`). Recorded vehicles
need the fully decentralized BH-CBF filter (`--controller marl_filter --method ours --filter_mode dec`).
"""

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# marl_filter: a MARL policy with a safety filter of
# sigmarl/evaluation_tits26/before_stages_filter_benchmark.py (--method, --filter_mode),
# as in the T-ITS evaluation (Stage 3)
CONTROLLERS = ("marl", "marl_cbf", "marl_filter", "cbf_qp_dec", "cbf_qp_cen")
DEFAULT_DATA_DIR = os.environ.get(
    "ROUND_DATA_DIR",
    os.path.expanduser("~/projects/drone-dataset-tools/rounD_v1.0/data"),
)
DEFAULT_MODEL = "checkpoints/round_location0/vanilla_seed_1"
ENTRY_CHECKS = ("certified", "distance")
VEHICLES = ("nominal", "recorded")
# An agent-step with a joint backup value below this value [m] is outside the certified set
OUTSIDE_C_TOL = 1e-3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode",
        choices=("replay", "closed_loop", "homogeneous"),
        default="homogeneous",
        help="Episode type (default: homogeneous).",
    )
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model_file",
        default=None,
        help="marl_filter: policy weights to load instead of the final model of "
        "--model_path (e.g., a fine-tuning checkpoint).",
    )
    parser.add_argument(
        "--method",
        default="ours",
        help="marl_filter: safety filter (a method of before_stages_filter_benchmark.METHODS).",
    )
    parser.add_argument(
        "--filter_mode",
        choices=("dec", "cen"),
        default="dec",
        help="marl_filter: fully decentralized or centralized filter.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Output subfolder under <out_dir>/<mode> (default: the controller).",
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", default="outputs/human_comparison")
    parser.add_argument(
        "--n_per_recording",
        type=int,
        default=20,
        help="Replay: ego cars per recording. Closed loop: windows per recording. Homogeneous: unused.",
    )
    parser.add_argument(
        "--window", type=float, default=60.0, help="Closed loop: spawn window [s]."
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default=None,
        help="Comma-separated subset, e.g. '2,3'.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode names, e.g. 'rec05_h0887.0,rec11_h0067.9'.",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the episode sampling."
    )
    parser.add_argument(
        "--n_slots",
        type=int,
        default=None,
        help="Agents of the env (default: from episodes).",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--n_videos", type=int, default=0, help="Save videos of the first N episodes."
    )
    parser.add_argument(
        "--no_snapshots",
        action="store_true",
        help="Do not store the states of other vehicles.",
    )
    parser.add_argument(
        "--veh_veh_cbf_footprint",
        choices=("rectangle_sat", "circle"),
        default="rectangle_sat",
        help="Vehicle-vehicle CBF footprint of the CBF-QP controllers.",
    )
    parser.add_argument(
        "--no_shield_priority",
        action="store_true",
        help="CBF-QP controllers: resolve shield conflicts without priority (the centralized shield brakes "
        "all violating vehicles, the decentralized one prioritizes lower indices; default: arrival order).",
    )
    parser.add_argument(
        "--no_start_repair",
        action="store_true",
        help="Start at the recorded poses even if the simulated footprint crosses the route boundary "
        "(default: repair them, see start_repair.py).",
    )
    add_entry_check_argument(parser)
    add_vehicles_argument(parser)
    add_max_vehicle_length_argument(parser)
    return parser.parse_args(argv)


def add_vehicles_argument(parser):
    """Add --vehicles (shared with run_single.py)."""
    parser.add_argument(
        "--vehicles",
        choices=VEHICLES,
        default="nominal",
        help="Parameters of the controlled vehicles: 'nominal' (the agent profile) or 'recorded' (the "
        "recorded footprint of each human's vehicle, axle distances scaled with its length, and the "
        "acceleration limits of its class).",
    )


def add_max_vehicle_length_argument(parser):
    """Add --max_vehicle_length (shared with run_single.py)."""
    parser.add_argument(
        "--max_vehicle_length",
        type=float,
        default=None,
        help="Homogeneous: cars and vans longer than this [m] count as heavy vehicles, i.e., they end the "
        "homogeneous intervals like trucks and buses (default: no limit).",
    )


def add_entry_check_argument(parser):
    """Add --entry_check (shared with run_single.py)."""
    parser.add_argument(
        "--entry_check",
        choices=ENTRY_CHECKS,
        default=None,
        help="Entry of a controlled vehicle: 'certified' (the joint state including it must be certified, "
        "see CertifiedEntryCheck; default with a CBF-QP controller) or 'distance' (no other controlled "
        "vehicle within its length + 1 s x its speed; the earlier gate, default without a CBF-QP).",
    )


def configure_cbf(
    parameters,
    max_group_size,
    nom_controller_type,
    veh_veh_cbf_footprint,
    is_shield_priority=True,
):
    """CBF-QP settings of main_testing_clf_cbf_qp.py (the defaults documented in docs/clf_cbf_qp_tuning.md)."""
    parameters.rew_method = "sparse"
    parameters.is_use_sampled_lane_boundary_for_cbf = True
    parameters.is_using_centralized_cbf = True
    parameters.is_using_cbf_testing = True
    parameters.is_using_cbf_training = False
    parameters.is_solve_qp = True
    parameters.is_apply_cbf_action = True
    parameters.lane_distance_type = "softmin"  # Unused by rectangle_sat
    parameters.lane_cbf_footprint = "rectangle_sat"
    parameters.is_proxsuite_qp = True
    parameters.is_grouping_agents = max_group_size is not None
    parameters.max_group_size = max_group_size if max_group_size is not None else 2
    parameters.grouping_method = "interaction_risk"
    parameters.grouping_use_hysteresis = True
    if parameters.is_grouping_agents and parameters.max_group_size == 1:
        parameters.is_using_centralized_cbf = False
    parameters.nom_controller_type = nom_controller_type
    parameters.n_circles_approximate_vehicle = 3
    parameters.adaptive_lambda = True
    parameters.is_learn_cbf_lambda = False
    parameters.is_log_cbf_boundary_distance = False
    parameters.is_aggregate_veh_veh_safety = True
    parameters.is_aggregate_lane_veh_safety = False
    parameters.veh_veh_cbf_footprint = veh_veh_cbf_footprint
    # The backup-horizon rows need the rectangle_sat footprints
    parameters.is_backup_horizon_cbf = veh_veh_cbf_footprint == "rectangle_sat"
    parameters.is_exact_backup_shield = True
    parameters.is_shield_priority = is_shield_priority
    parameters.is_standstill_escape = True
    parameters.is_visu_shield_rollout = False
    parameters.is_alternating_action_lambda_training = False
    parameters.pretrained_motion_policy_path = ""
    parameters.pretrained_lambda_policy_path = ""
    parameters.is_load_pretrained_motion_policy = False
    parameters.is_load_pretrained_lambda_policy = False
    if nom_controller_type == "clf":
        parameters.is_train_motion_policy = False
        parameters.is_train_lambda_policy = False


def configure_filter(parameters, n_slots, method, filter_mode):
    """Filter settings of the T-ITS evaluation (stage2_finetune_run.apply_target_settings)
    on round_location0, final model of the checkpoint."""
    from sigmarl.evaluation_tits26.stage2_finetune_run import apply_target_settings

    apply_target_settings(
        parameters,
        argparse.Namespace(
            scenario_type="round_location0",
            n_agents=n_slots,
            seed=0,
            method=method,
            mode=filter_mode,
        ),
    )
    parameters.is_load_model = True
    parameters.is_load_final_model = True
    parameters.is_using_cbf_testing = True
    parameters.is_using_cbf_training = False
    parameters.is_apply_cbf_action = True
    parameters.is_store_rl_action_under_cbf = False


def build_parameters(
    controller,
    model_path,
    n_slots,
    max_steps,
    veh_veh_cbf_footprint="rectangle_sat",
    is_shield_priority=True,
    method="ours",
    filter_mode="dec",
):
    from sigmarl.helper_training import Parameters, SaveData

    if controller in ("marl", "marl_cbf", "marl_filter"):
        json_files = [
            f
            for f in os.listdir(model_path)
            if f.startswith("reward") and f.endswith(".json")
        ]
        with open(os.path.join(model_path, json_files[0])) as f:
            parameters = SaveData.from_dict(json.load(f)).parameters
        parameters.where_to_save = model_path
        parameters.is_load_model = True
        parameters.is_load_final_model = False
        parameters.rew_method = "sparse"
        parameters.is_using_cbf_testing = False
        parameters.is_using_cbf_training = False
        if controller == "marl_filter":
            configure_filter(parameters, n_slots, method, filter_mode)
        if controller == "marl_cbf":
            configure_cbf(
                parameters,
                max_group_size=1,
                nom_controller_type="rl",
                veh_veh_cbf_footprint=veh_veh_cbf_footprint,
                is_shield_priority=is_shield_priority,
            )
    else:
        parameters = Parameters()
        parameters.scenario_type = "round_location0"
        parameters.dt = 0.1
        parameters.is_load_model = False
        parameters.is_load_final_model = False
        parameters.where_to_save = "outputs/human_comparison/_unused"
        configure_cbf(
            parameters,
            max_group_size=1 if controller == "cbf_qp_dec" else None,
            nom_controller_type="clf",
            veh_veh_cbf_footprint=veh_veh_cbf_footprint,
            is_shield_priority=is_shield_priority,
        )
    parameters.n_agents = n_slots
    parameters.num_vmas_envs = 1
    parameters.max_steps = max_steps
    parameters.is_testing_mode = True
    parameters.is_continue_train = False
    parameters.is_joint_task = False
    parameters.is_save_eval_results = False
    parameters.is_load_out_td = False
    parameters.is_real_time_rendering = False
    parameters.is_save_simulation_video = False
    parameters.is_enable_debug_diagnostics = False
    parameters.is_visualize_short_term_path = False
    parameters.is_visualize_lane_boundary = False
    parameters.is_visualize_extra_info = False
    parameters.is_visualize_observed_agents = False
    parameters.is_using_prioritized_marl = False
    return parameters


class CertifiedEntryCheck:
    """Certified entry check of a log replay (`LogReplay.entry_check`).

    A vehicle placed at its recorded state may enter if the joint state including it is certified (in C):
    its own lane margins and its pair margins with every other vehicle on the map are nonnegative (up to
    the filter's numerical tolerance) at the current poses and along the braking backup until standstill
    (`BackupHorizonCBF.entry_values`, the margins, buffers, routes, and backup of the BH-CBF filter).
    The constraints among the other vehicles are not checked: the filter keeps them certified, and the new
    vehicle does not change them.

    Args:
        ctrl: a CBF-QP controller of the env (its vehicle parameters, routes, and lane barrier are used).
    """

    def __init__(self, ctrl):
        from sigmarl.backup_horizon_cbf import BackupHorizonCBF

        self.horizon = BackupHorizonCBF(ctrl)
        # Keep the check's counters out of the filter's statistics
        self.horizon.stats = defaultdict(int)
        self.tol = float(ctrl.lane_sat_tol)
        self.env_idx = ctrl.env_idx

    def __call__(self, scenario, slot: int, others, held) -> tuple:
        """(is certified, lane value, pair value) of the vehicle in `slot` (see `LogReplay.entry_check`)."""
        lane_value, pair_value = self.horizon.entry_values(
            self.env_idx, slot, others, held
        )
        return min(lane_value, pair_value) >= -self.tol, lane_value, pair_value


class CertifiedStartRepair:
    """Certified start-pose repair of a log replay (`LogReplay.entry_repair`).

    Searches the start-pose corrections of `StartPoseRepair.candidates` (smallest heading change, then
    smallest lateral shift, each with a lane barrier of at least its margin) for the first one whose own
    lane value along the braking backup (`BackupHorizonCBF.entry_values` without other vehicles) is
    nonnegative up to the filter's tolerance. Speed and steering stay as recorded.

    Args:
        check: the certified entry check of the env.
        start_repair: the start-pose repair of the episodes (its geometry and search grid).
        max_tries: largest number of candidates whose backup is rolled out.
    """

    def __init__(self, check: CertifiedEntryCheck, start_repair, max_tries: int = 3000):
        self.check = check
        self.start_repair = start_repair
        self.max_tries = max_tries

    def __call__(self, scenario, slot: int, vehicle, speed: float, steering: float):
        """(xy, yaw, shift, heading change) of the repaired start pose, or None (see
        `LogReplay.entry_repair`)."""
        track, k0 = vehicle.track, vehicle.k_start
        progress = float(track.match.s[k0])
        poses, shifts, dyaws = self.start_repair.candidates(
            vehicle.path_id,
            progress,
            track.xy[k0],
            float(track.heading[k0]),
            float(vehicle.route.yaw_at(progress)),
            vehicle.params,
        )
        for k in range(min(poses.shape[0], self.max_tries)):
            scenario.place_agent(
                0, slot, poses[k, :2], float(poses[k, 2]), speed, steering,
                path_id=vehicle.path_id,
            )  # fmt: skip
            lane_value, _ = self.check.horizon.entry_values(
                self.check.env_idx, slot, [], []
            )
            if lane_value >= -self.check.tol:
                return poses[k, :2].copy(), float(poses[k, 2]), shifts[k], dyaws[k]
        return None


class ControllerRunner:
    """One environment with a controller; runs log-replay episodes on it."""

    def __init__(
        self,
        controller,
        model_path,
        n_slots,
        max_steps,
        veh_veh_cbf_footprint="rectangle_sat",
        is_shield_priority=True,
        method="ours",
        filter_mode="dec",
        model_file=None,
    ):
        from sigmarl.mappo_cavs import mappo_cavs

        self.controller = controller
        parameters = build_parameters(
            controller,
            model_path,
            n_slots,
            max_steps,
            veh_veh_cbf_footprint,
            is_shield_priority,
            method,
            filter_mode,
        )
        (
            self.env,
            self.decision_making_module,
            _,
            _,
            self.cbf_controllers,
            self.parameters,
        ) = mappo_cavs(parameters=parameters)
        self.scenario = self.env.base_env.scenario_name
        self.policy = self.decision_making_module.policy
        if model_file is not None:
            self.policy.load_state_dict(torch.load(model_file, weights_only=True))
        # A CBF-QP controller of the env (fully decentralized: the first agent's), for the certified entry
        # check and the backup-value diagnostic
        self.first_cbf_controller = None
        if self.cbf_controllers is not None:
            first = self.cbf_controllers[0]
            self.first_cbf_controller = first[0] if isinstance(first, list) else first

    def entry_check(self, kind):
        """`LogReplay.entry_check` of an entry check kind (None: certified with a CBF-QP controller, else
        distance)."""
        if kind is None:
            kind = "distance" if self.first_cbf_controller is None else "certified"
        if kind == "distance":
            return None
        if self.first_cbf_controller is None:
            raise ValueError(
                "--entry_check certified needs a CBF-QP controller; use --entry_check distance."
            )
        return CertifiedEntryCheck(self.first_cbf_controller)

    def _control(self, td, controlled_slots):
        """Compute actions in place; returns (policy time, sequential CBF time, parallelized CBF time)."""
        if self.controller == "marl_filter":
            # The filter pipeline of the other stages (helper_training rollout)
            from sigmarl.helper_training import cbf_constrained_policy_main

            timing = {}
            time_rl, time_cbf, _, _ = cbf_constrained_policy_main(
                td, self.policy, self.cbf_controllers, timing=timing
            )
            return time_rl, time_cbf, timing["parallel"]
        t_start = time.perf_counter()
        self.policy(td)
        time_policy = time.perf_counter() - t_start
        if ("agents", "action") not in td.keys(True, True):
            # No motion policy (pure CLF-CBF-QP): the CBF-QP overwrites this placeholder
            obs = td.get(("agents", "observation"))
            td.set(
                ("agents", "action"), torch.zeros(*obs.shape[:-1], 2, dtype=obs.dtype)
            )
        if self.cbf_controllers is None:
            return time_policy, 0.0, 0.0

        controllers = self.cbf_controllers[0]
        t_start = time.perf_counter()
        if isinstance(controllers, list):
            # Fully decentralized: shared per-step cache, then one QP (and
            # shield round 1) per agent, then shield round 2 per agent
            controllers[0].prepare_fully_decentralized_step_cache(td, controllers)
            time_shared = time.perf_counter() - t_start
            per_agent = np.zeros(len(controllers))
            for j, ctrl in enumerate(controllers):
                t_agent = time.perf_counter()
                ctrl.update_qp(td)
                per_agent[j] = time.perf_counter() - t_agent
            round2 = np.zeros(len(controllers))
            if self.parameters.is_exact_backup_shield:
                infos = controllers[0].apply_fully_decentralized_exact_shield(
                    td, controllers
                )
                for j, info in infos.items():
                    round2[j] = info["time"]
            controllers[0].clear_fully_decentralized_step_cache(controllers)
            time_seq = time.perf_counter() - t_start
            time_par = time_shared
            if len(controlled_slots):
                time_par += (
                    per_agent[controlled_slots].max() + round2[controlled_slots].max()
                )
        else:
            controllers.update_qp(td)
            time_seq = time.perf_counter() - t_start
            time_par = time_seq
        return time_policy, time_seq, time_par

    def _render(self, log_replay, is_show_window=False):
        """Render a frame, optionally also in a window; parked agents are moved to the render origin meanwhile
        (VMAS fits the view to all agents, and parked agents are far away; they are not drawn)."""
        parked = torch.nonzero(log_replay.parked[0]).flatten().tolist()
        saved = {i: self.scenario.world.agents[i].state.pos.clone() for i in parked}
        origin = torch.tensor(self.scenario.render_origin, dtype=torch.float32)
        for i in parked:
            self.scenario.world.agents[i].state.pos[0] = origin
        frame = self.env.render(mode="rgb_array", visualize_when_rgb=is_show_window)
        for i, pos in saved.items():
            self.scenario.world.agents[i].state.pos = pos
        return frame

    def run_episode(self, log_replay, is_video=False, is_show_window=False):
        """Run one episode; returns the per-step timing and the video frames (if `is_video`).

        With `is_show_window`, each step is shown in a window, at most in real time.
        """
        from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

        from sigmarl.cbf_qp_proxsuite import qp_statistics

        self.scenario.log_replay = log_replay
        qp_before = (
            qp_statistics(self.cbf_controllers) if self.cbf_controllers else (0, {})
        )
        frames = []
        timing = {
            k: []
            for k in (
                "policy",
                "cbf_seq",
                "cbf_par",
                "n_controlled",
                "env_step",
                "backup_value_min",
                "n_outside_c",
            )
        }
        # Joint backup value of the controlled vehicles at each step (CBFQP.backup_values_all_envs, with our
        # margins; not timed), if the controller has the rectangle_sat lane barrier
        is_log_backup_value = (
            self.first_cbf_controller is not None
            and self.first_cbf_controller.lane_sat is not None
        )
        with torch.inference_mode(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ):
            td = self.env.reset()
            n_steps = 0
            while (
                not log_replay.is_finished and n_steps < self.parameters.max_steps - 2
            ):
                t_step = time.perf_counter()
                controlled_slots = [
                    v.slot for v in log_replay.controlled if v.status == "active"
                ]
                t_policy, t_seq, t_par = self._control(td, controlled_slots)
                if is_log_backup_value:
                    values = self.first_cbf_controller.backup_values_all_envs(
                        self.cbf_controllers
                    )[0, controlled_slots]
                    timing["backup_value_min"].append(values.min(initial=np.inf))
                    timing["n_outside_c"].append(int((values < -OUTSIDE_C_TOL).sum()))
                t_env = time.perf_counter()
                td = self.env.step(td)
                td = step_mdp(td)
                timing["env_step"].append(time.perf_counter() - t_env)
                timing["policy"].append(t_policy)
                timing["cbf_seq"].append(t_seq)
                timing["cbf_par"].append(t_par)
                timing["n_controlled"].append(len(controlled_slots))
                if is_video or is_show_window:
                    frame = self._render(log_replay, is_show_window=is_show_window)
                    if is_video:
                        frames.append(frame)
                if is_show_window:
                    time.sleep(
                        max(0.0, self.parameters.dt - (time.perf_counter() - t_step))
                    )
                n_steps += 1
        self.scenario.log_replay = None
        timing = {k: np.asarray(v) for k, v in timing.items()}
        if self.cbf_controllers:
            # QP solves and failures of this episode (failed QPs use the fallback control)
            n_solves, failures = qp_statistics(self.cbf_controllers)
            timing["qp_solves"] = np.asarray(n_solves - qp_before[0])
            timing["qp_failures"] = np.asarray(
                sum(failures.values()) - sum(qp_before[1].values())
            )
        return timing, frames


def _episode_result(spec, log_replay, timing, removed, vehicle_kind):
    """Stored result of one episode; `removed` are the keys of the vehicles removed for a bad start,
    `vehicle_kind` the parameters of the controlled vehicles (one of VEHICLES)."""
    vehicles = []
    for v in log_replay.controlled:
        vehicles.append(
            {
                "key": (v.track.recording_id, v.track.track_id),
                # Vehicle parameters (None: nominal)
                "params": None if v.params is None else dataclasses.asdict(v.params),
                "path_id": v.path_id,
                "s_goal": v.s_goal,
                "status": v.status,
                "t_spawn": v.t_spawn,
                "t_finish": v.t_finish,
                "start_repair": None
                if v.start_repair is None
                else dataclasses.asdict(v.start_repair),
                "t_begin": v.t_begin,
                "n_entry_rejections": v.n_entry_rejections,
                "entry_values": v.entry_values,
                "first_rejection_values": v.first_rejection_values,
                "log": {k: np.asarray(x) for k, x in v.log.items()},
            }
        )
    return {
        "spec": spec,
        "vehicles": vehicles,
        "snapshots": log_replay.snapshots,
        "timing": timing,
        "max_slots_used": log_replay.max_slots_used,
        "n_replay_steps_dropped": log_replay.n_replay_steps_dropped,
        "removed_bad_start": [tuple(k) for k in removed],
        "entry_check": "distance" if log_replay.entry_check is None else "certified",
        "vehicles_kind": vehicle_kind,
    }


def _worker(args, specs, n_slots, worker_idx):
    """Run a list of episodes in one process."""
    torch.set_num_threads(1)
    from sigmarl.helper_common import save_video
    from sigmarl.log_replay import LogReplay
    from sigmarl.parse_lanelet2 import ParseLanelet2
    from sigmarl.evaluation_human_comparison.episodes import (
        build_episode,
        load_location0,
    )
    from sigmarl.evaluation_human_comparison.start_repair import StartPoseRepair
    from sigmarl.round_tracks import RouteMatcher

    out_dir = os.path.join(args.out_dir, args.mode, args.label or args.controller)
    parser = ParseLanelet2("round_location0", "cpu")
    routes = RouteMatcher(parser.reference_paths).routes
    data = load_location0(
        args.data_dir, parser, os.path.join(args.out_dir, "matched_tracks.pkl")
    )
    start_repair = (
        None if args.no_start_repair else StartPoseRepair(parser.reference_paths)
    )
    dt = 0.1
    is_recorded_vehicles = args.vehicles == "recorded"
    max_t = (
        max(
            build_episode(spec, data, routes, start_repair, is_recorded_vehicles)[2]
            - spec.t0
            for spec in specs
        )
        if specs
        else 0.0
    )
    runner = ControllerRunner(
        args.controller,
        args.model_path,
        n_slots,
        int(max_t / dt) + 20,
        args.veh_veh_cbf_footprint,
        not args.no_shield_priority,
        args.method,
        args.filter_mode,
        args.model_file,
    )
    entry_check = runner.entry_check(args.entry_check)
    entry_repair = (
        None
        if entry_check is None or start_repair is None
        else CertifiedStartRepair(entry_check, start_repair)
    )
    for k, spec in enumerate(specs):
        path = os.path.join(out_dir, f"{spec.name}.pkl")
        if os.path.exists(path):
            continue
        controlled, replayed, _, removed = build_episode(
            spec, data, routes, start_repair, is_recorded_vehicles
        )
        log_replay = LogReplay(
            t0=spec.t0,
            dt=runner.parameters.dt,
            n_slots=n_slots,
            controlled=controlled,
            replayed=replayed,
            entry_check=entry_check,
            entry_repair=entry_repair,
            is_record_snapshots=not args.no_snapshots,
        )
        is_video = worker_idx == 0 and k < args.n_videos
        t_start = time.time()
        timing, frames = runner.run_episode(log_replay, is_video=is_video)
        result = _episode_result(spec, log_replay, timing, removed, args.vehicles)
        result["wall_time"] = time.time() - t_start
        with open(path + ".tmp", "wb") as f:
            pickle.dump(result, f)
        os.replace(path + ".tmp", path)
        if is_video and frames:
            save_video(
                os.path.join(out_dir, f"video_{spec.name}"),
                frames,
                fps=1 / runner.parameters.dt,
            )
        statuses = [v["status"] for v in result["vehicles"]]
        n_delayed = sum(v["n_entry_rejections"] > 0 for v in result["vehicles"])
        print(
            f"[worker {worker_idx}] {k + 1}/{len(specs)} {spec.name}: "
            f"{statuses.count('goal')}/{len(statuses)} goal, {n_delayed} entries delayed, "
            f"{result['wall_time']:.1f} s"
            + (
                f", min. backup value {timing['backup_value_min'].min(initial=np.inf):.3f} m"
                if len(timing["backup_value_min"])
                else ""
            )
            + (
                f", {int(timing['qp_failures'])}/{int(timing['qp_solves'])} QPs failed"
                if "qp_solves" in timing
                else ""
            ),
            flush=True,
        )


def main(argv=None):
    args = parse_args(argv)
    assert (
        os.environ.get("SIGMARL_AGENT_PROFILE") == "full_scale"
    ), "Set SIGMARL_AGENT_PROFILE=full_scale."
    from tqdm import tqdm

    from sigmarl.parse_lanelet2 import ParseLanelet2
    from sigmarl.evaluation_human_comparison.episodes import (
        closed_loop_episodes,
        homogeneous_episodes,
        load_location0,
        max_concurrent_slots,
        replay_episodes,
    )
    from sigmarl.round_tracks import RouteMatcher

    parser = ParseLanelet2("round_location0", "cpu")
    routes = RouteMatcher(parser.reference_paths).routes
    data = load_location0(
        args.data_dir, parser, os.path.join(args.out_dir, "matched_tracks.pkl")
    )
    if args.mode == "replay":
        specs = replay_episodes(data, args.n_per_recording, args.seed)
    elif args.mode == "closed_loop":
        specs = closed_loop_episodes(data, args.window, args.n_per_recording, args.seed)
    else:
        specs = homogeneous_episodes(data, max_vehicle_length=args.max_vehicle_length)
    if args.recordings is not None:
        # Filter after sampling so that the episodes do not depend on the subset
        keep = {int(r) for r in args.recordings.split(",")}
        specs = [s for s in specs if s.recording_id in keep]
    if args.episodes is not None:
        names = set(args.episodes.split(","))
        specs = [s for s in specs if s.name in names]
        assert len(specs) == len(names), "Unknown episode name."
    if args.max_episodes is not None:
        specs = specs[: args.max_episodes]
    n_slots = args.n_slots
    if n_slots is None:
        n_slots = max(max_concurrent_slots(s, data, routes, 0.1) for s in specs) + 2
    print(
        f"[INFO] {len(specs)} {args.mode} episodes, {n_slots} agents per environment."
    )

    out_dir = os.path.join(args.out_dir, args.mode, args.label or args.controller)
    os.makedirs(out_dir, exist_ok=True)
    todo = [
        s for s in specs if not os.path.exists(os.path.join(out_dir, f"{s.name}.pkl"))
    ]
    print(f"[INFO] {len(specs) - len(todo)} episodes already done, {len(todo)} to run.")
    if not todo:
        return

    shards = [todo[i :: args.workers] for i in range(args.workers)]
    if args.workers == 1:
        _worker(args, shards[0], n_slots, 0)
        return
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=_worker, args=(args, shard, n_slots, i))
        for i, shard in enumerate(shards)
        if shard
    ]
    for p in processes:
        p.start()
    # Progress bar over finished episode files
    with tqdm(total=len(todo), desc=f"{args.mode}/{args.controller}") as bar:
        done = 0
        while any(p.is_alive() for p in processes):
            time.sleep(5)
            now = sum(
                os.path.exists(os.path.join(out_dir, f"{s.name}.pkl")) for s in todo
            )
            bar.update(now - done)
            done = now
        now = sum(os.path.exists(os.path.join(out_dir, f"{s.name}.pkl")) for s in todo)
        bar.update(now - done)
    failed = [p.exitcode for p in processes if p.exitcode != 0]
    if failed:
        sys.exit(f"[ERROR] {len(failed)} worker(s) failed.")


if __name__ == "__main__":
    main()
