# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Repair rounds of the BH-CBF filter for the video demo (video.py, scene
``repair``).

The evaluation stores neither the QP constraints of the certified filter nor
its repair rounds. This script reruns a Stage 2 deployment with the BH-CBF
filter exactly as the evaluation ran it (as tails.py does) and records, for
every filter problem of env 0 (one vehicle, one step):

- the safety constraints e of the vehicle (lane sides L, R and the corner
  budgets of its neighbors), their right-hand sides, and the QP constraints
  (e, t) of the first QP, linearized at the nominal control;
- per round: the candidate, its prediction (the candidate, then braking until
  standstill), the violated (e, t) of the exact check, and the QP constraints
  added at the candidate;
- the braking predictions of the neighbors, whether the QP was solved, and
  whether the fallback switched the vehicle to its backup.

Prediction steps follow the paper: t = 1 is the next step (the code's tail
index plus one), so the QP constraints have t = 2, ..., N.

``capture`` writes <out>/<name>.pkl; ``list`` prints the filter problems with
repairs (candidates for a scene).

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.video_demo.repair capture \\
        --run outputs/tits26/stage1_filters_ttc_iter100/<label>/<map>/ours/dec/seed202 --steps 0 300 --name repair_round
    ... repair list --name repair_round
"""

import argparse
import glob
import os
import pickle
import runpy
import shlex
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = "outputs/tits26/video_demo/repair"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--run", required=True, help="Stage 2 run folder (ours/dec).")
    capture.add_argument("--steps", type=int, nargs=2, required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--min_rounds", type=int, default=2)
    for command in (capture, listing):
        command.add_argument("--name", required=True)
        command.add_argument("--out", default=OUT)
    return parser.parse_args()


class Recorder:
    """Wraps the build, check, and fallback of BackupHorizonCBF and the repair
    loop of CBFQP to record the filter problems of env 0."""

    def __init__(self, targets: np.ndarray, steps: range):
        self.targets, self.wanted = targets, set(steps)
        self.problems = []
        self.current = None

    def step_of(self, agent: int, position: np.ndarray):
        """Step s of the stored rollout whose state (after step s) the filter reads."""
        diff = np.abs(self.targets[:, agent] - position).max(axis=1)
        hits = np.nonzero(diff < 1e-4)[0]
        return int(hits[0]) if len(hits) else None

    def install(self):
        from sigmarl.backup_horizon_cbf import BackupHorizonCBF, run_steps
        from sigmarl.cbf_qp_proxsuite import CBFQP

        recorder = self
        build, check = BackupHorizonCBF.build_steps, BackupHorizonCBF.check_steps
        fallback, solve = (
            BackupHorizonCBF.fallback_steps,
            CBFQP._solve_qp_with_sat_repairs_steps,
        )

        def prediction(horizon, block, U):
            """Poses [T, 3] of the controlled agent (t = 1, ..., N + 1) and the
            exact values [E, T] under the control U [1, 2]."""
            poses_c, _ = run_steps(horizon._rollouts_steps(block, U, n_var=1))
            values = horizon._values(block, horizon._assemble(block, poses_c))
            return poses_c[:, 0, 0].copy(), values

        def build_and_record(self, rows, lowers, uppers, n_vars):
            pending_pairs, pending_lanes = list(self._pending_pairs), list(self._pending_lanes)
            block = yield from build(self, rows, lowers, uppers, n_vars)
            recorder.current = None
            if block is None or self.ctrl.env_idx != 0 or len(block["controlled"]) != 1:
                return block
            order = sorted(block["index"], key=block["index"].get)
            ego = order[block["controlled"][0]]
            step = recorder.step_of(ego, block["X"][block["controlled"][0], :2])
            if step is None or step not in recorder.wanted:
                return block
            # Safety constraints in entry order: budgets (pending cross pairs), then lane sides
            labels = [None] * len(block["entries"])
            if "budget" in block:
                others = [
                    pair["b"] if pair["slices"][0] is not None else pair["a"]
                    for pairs, _, _, _ in pending_pairs
                    for pair in pairs
                    if pair["share"] is not None
                ]
                for e, other in zip(block["budget"]["idx"], others):
                    labels[e] = ("pair", int(other))
            if "lane" in block:
                for e, side in zip(block["lane"]["idx"], block["lane"]["side"]):
                    labels[e] = ("lane", "LR"[int(side)])
            U = np.array([pending_lanes[0]["u_lin"]]) if pending_lanes else None
            poses, values = prediction(self, block, U)
            recorder.current = dict(
                step=step,
                agent=ego,
                state=block["X"][block["controlled"][0]].copy(),
                labels=labels,
                rhs=block["rhs"].copy(),
                u_nom=U[0].copy(),
                initial=sorted(block["linearized_at"]),
                prediction_nom=poses,
                values_nom=values,
                # Braking predictions [T, A, 3] (t = 1, ...) of the agents of the table
                braking=block["ref"].copy(),
                agents=order,
                rounds=[],
                solved=None,
                fallback=False,
            )
            block["_record"] = recorder.current
            recorder.problems.append(recorder.current)
            return block

        def check_and_record(self, block, x, rows, lowers, uppers, n_vars):
            record = block.get("_record")
            before = {key: len(value) for key, value in block["linearized_at"].items()}
            result = yield from check(self, block, x, rows, lowers, uppers, n_vars)
            if record is not None:
                U = self.controls_of(block, x)
                poses, values = prediction(self, block, U)
                record["rounds"].append(dict(
                    u=U[0].copy(), prediction=poses, values=values,
                    violated=[tuple(map(int, v)) for v in self.violated(block, values)],
                    added=sorted(key for key, value in block["linearized_at"].items()
                                 if len(value) > before.get(key, 0)),
                ))  # fmt: skip
            return result

        def fallback_and_record(self, block, x):
            switched = yield from fallback(self, block, x)
            record = block.get("_record")
            if record is not None:
                record["fallback"] = bool(switched)
                U = self.controls_of(block, x)
                record["u_fallback"] = U[0].copy()
                record["prediction_fallback"], _ = prediction(self, block, U)
            return switched

        def solve_and_record(ctrl, *args):
            result = yield from solve(ctrl, *args)
            if recorder.current is not None:
                recorder.current["solved"] = bool(result["solved"])
                recorder.current = None
            return result

        BackupHorizonCBF.build_steps = build_and_record
        BackupHorizonCBF.check_steps = check_and_record
        BackupHorizonCBF.fallback_steps = fallback_and_record
        CBFQP._solve_qp_with_sat_repairs_steps = solve_and_record

    def save(self, path: str):
        with open(path, "wb") as file:
            pickle.dump(self.problems, file)
        print(f"[INFO] Saved {path} ({len(self.problems)} filter problems)")


def capture(args):
    """Rerun a Stage 2 deployment as its command_testing.txt and record the repairs."""
    (path,) = glob.glob(os.path.join(args.run, "out_td_*.td"))
    td = torch.load(path, weights_only=False)
    targets = td["next", "agents", "info", "pos"][0].numpy()
    start, end = args.steps
    recorder = Recorder(targets, range(start, end + 1))
    recorder.install()
    with open(os.path.join(args.run, "command_testing.txt")) as file:
        tokens = shlex.split(file.read())
    argv = tokens[
        tokens.index(
            next(t for t in tokens if t.endswith("main_testing_cbf_filter.py"))
        )
        + 1 :
    ]
    replace = {
        "--max_steps": str(end + 3),
        "--output_dir": os.path.join(args.out, "_sim", args.name),
    }
    for flag, value in replace.items():
        argv[argv.index(flag) + 1] = value
    sys.argv = ["main_testing_cbf_filter.py"] + argv
    runpy.run_path(
        os.path.join(REPO_ROOT, "main_testing_cbf_filter.py"), run_name="__main__"
    )
    recorder.save(os.path.join(args.out, f"{args.name}.pkl"))


def summary(problem: dict) -> str:
    rounds = problem["rounds"]
    initial = set(problem["initial"])
    first = rounds[0]["violated"] if rounds else []
    omitted = sum(v not in initial for v in first)
    neighbors = [label[1] for label in problem["labels"] if label[0] == "pair"]
    return (
        f"step {problem['step']:4d}  agent {problem['agent']:2d}  solves {len(rounds)}  "
        f"QP constraints {len(initial):2d}  violated in round 1: {len(first)} "
        f"({omitted} omitted)  fallback {problem['fallback']}  neighbors {neighbors}"
    )


def list_problems(args):
    with open(os.path.join(args.out, f"{args.name}.pkl"), "rb") as file:
        problems = pickle.load(file)
    for problem in problems:
        if len(problem["rounds"]) >= args.min_rounds or problem["fallback"]:
            print(summary(problem))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"capture": capture, "list": list_problems}[args.command](args)


if __name__ == "__main__":
    main()
