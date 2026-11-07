# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Filter problems of BH-CBF for the explainer video (explainer.py).

The explainer shows single filter problems of the decentralized BH-CBF filter
in detail: the backup and candidate predictions, the margins along them, the
exact set of controls that satisfy the BH-CBF inequalities (the control
plane), and the corner budgets of a pair. None of this is stored by the
evaluation. This script reruns a Stage 2 deployment exactly as the evaluation
ran it (as tails.py and repair.py do) and records, for every filter problem of
env 0 at the given steps and of the given vehicles, everything that repair.py
records plus:

- the vehicle table of the problem (states, routes, backup predictions), the
  shares and decay rates of the pairs, the corner-budget data (separating
  normals, supports, budgets), and the lane facets;
- the exact values of all inequalities on a grid over the admissible box of
  the controls (a, omega), reduced to the smallest slack over the prediction
  steps per constraint;
- the predictions and exact values along the straight path from the nominal
  to the applied control in the control plane;
- the QP constraints of the first QP and of the repairs as half-planes
  (value and gradient at the linearization control).

Prediction steps follow the code: index 0 is t = 1 of the paper.

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.video_demo.explainer_capture \\
        --run outputs/tits26/stage1_filters_ttc_iter100/<label>/<map>/ours/dec/seed202 \\
        --steps 115 155 --agents 6 10 --name explainer_pair
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

from sigmarl.evaluation_tits26.video_demo.repair import REPO_ROOT, Recorder

OUT = "outputs/tits26/video_demo/explainer"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="Stage 2 run folder (ours/dec).")
    parser.add_argument("--steps", type=int, nargs=2, required=True)
    parser.add_argument("--agents", type=int, nargs="+", required=True)
    parser.add_argument("--grid", type=int, default=61, help="Grid points per control.")
    parser.add_argument("--path", type=int, default=41, help="Controls on the path.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default=OUT)
    return parser.parse_args()


def evaluate(horizon, block, U: np.ndarray, chunk: int = 400) -> tuple:
    """Predictions [G, T, 3] of the controlled vehicle under the controls U
    [G, 2] (then the backup) and the exact values [G, E, T] of all entries of
    the block (as BackupHorizonCBF._values evaluates them)."""
    from sigmarl.backup_horizon_cbf import take_params

    ctrl = horizon.ctrl
    c = int(block["controlled"][0])
    poses_all, values_all = [], []
    for start in range(0, U.shape[0], chunk):
        Uc = U[start : start + chunk]
        G = Uc.shape[0]
        poses = horizon.rollout(
            np.repeat(block["X"][c : c + 1], G, axis=0),
            Uc,
            [block["routes"][c]] * G,
            np.zeros(G, dtype=bool),
            block["n_steps"],
            take_params(block["params"], np.full(G, c)),
        )  # [T, G, 3]
        n_t = poses.shape[0]
        values = np.full((G,) + block["rhs"].shape, np.nan)
        if "lane" in block:
            lane = block["lane"]
            for s, e in enumerate(lane["idx"]):
                rows = np.full(n_t * G, s)
                v = ctrl.lane_sat.padded_values(
                    {key: value[rows] for key, value in lane["facets"].items()},
                    lane["valid"][rows],
                    lane["side"][rows],
                    poses.reshape(-1, 3),
                    None if lane["half"] is None else lane["half"][rows],
                )
                values[:, e] = v.reshape(n_t, G).T
        if "budget" in block:
            budget = block["budget"]
            half = horizon.half if block["half"] is None else block["half"][c]
            corners = horizon.support_points(poses, half)  # [T, G, 4, 2]
            for s, e in enumerate(budget["idx"]):
                support = np.einsum("td,tgcd->tgc", budget["witness"][:, s], corners).min(2)
                values[:, e] = (
                    support - budget["ref_support"][:, s, None] + budget["budget"][:, s, None]
                ).T
        if "pair" in block:
            raise NotImplementedError("Joint pairs (centralized filter) are not supported.")
        poses_all.append(poses.transpose(1, 0, 2))
        values_all.append(values)
    return np.concatenate(poses_all), np.concatenate(values_all)


def gradients(horizon, block, u: np.ndarray, keys) -> dict:
    """Value and forward-difference gradient of the entries (e, t) at the
    control u [2]: {(e, t): (value, gradient [2])}."""
    from sigmarl.backup_horizon_cbf import take_params

    c = int(block["controlled"][0])
    perturbed, steps = horizon.fd_controls(
        block["X"][c : c + 1], u[None], take_params(block["params"], np.array([c]))
    )
    U = np.concatenate([u[None], perturbed[0]])
    _, values = evaluate(horizon, block, U)
    out = {}
    for e, t in keys:
        grad = np.array(
            [(values[1 + col, e, t] - values[0, e, t]) / steps[0, col] for col in range(2)]
        )
        out[(int(e), int(t))] = (float(values[0, e, t]), grad)
    return out


class ExplainerRecorder(Recorder):
    """repair.Recorder plus the data of the explainer (see the module
    docstring) for the filter problems of the given vehicles."""

    def __init__(self, targets, steps, agents, grid: int, path: int):
        super().__init__(targets, steps)
        self.agents, self.grid, self.path = set(agents), grid, path
        self.meta = None
        # (horizon, block, record) of the filter problem being solved, per controller
        self.pending = {}

    def install(self):
        super().install()
        from sigmarl.backup_horizon_cbf import BackupHorizonCBF
        from sigmarl.cbf_qp_proxsuite import CBFQP

        recorder = self
        build, check = BackupHorizonCBF.build_steps, BackupHorizonCBF.check_steps
        solve = CBFQP._solve_qp_with_sat_repairs_steps

        def build_rich(self, rows, lowers, uppers, n_vars):
            pending_pairs = list(self._pending_pairs)
            block = yield from build(self, rows, lowers, uppers, n_vars)
            record = None if block is None else block.get("_record")
            if record is None or record["agent"] not in recorder.agents:
                if block is not None:
                    block.pop("_record", None)
                return block
            recorder.enrich(self, block, record, pending_pairs)
            recorder.pending[id(self.ctrl)] = (self, block, record)
            return block

        def check_rich(self, block, x, rows, lowers, uppers, n_vars):
            record = block.get("_record")
            result = yield from check(self, block, x, rows, lowers, uppers, n_vars)
            if record is not None and "grid_slack" in record:
                rnd = record["rounds"][-1]
                rnd["rows"] = gradients(self, block, rnd["u"], rnd["added"])
            return result

        def solve_rich(ctrl, *args):
            result = yield from solve(ctrl, *args)
            # Generators of several vehicles interleave: the problem built by this controller
            pending = recorder.pending.pop(id(ctrl), None)
            if pending is not None and result["solved"]:
                horizon, block, record = pending
                # The solution after the repairs and the fallback (in place)
                record["u_applied"] = horizon.controls_of(block, result["x"])[0].copy()
                poses, values = evaluate(horizon, block, record["u_applied"][None])
                record["prediction_applied"], record["values_applied"] = poses[0], values[0]
                record.update(recorder.path_sweep(horizon, block, record))
            return result

        BackupHorizonCBF.build_steps = build_rich
        BackupHorizonCBF.check_steps = check_rich
        CBFQP._solve_qp_with_sat_repairs_steps = solve_rich

    def enrich(self, horizon, block, record, pending_pairs):
        ctrl = horizon.ctrl
        c = int(block["controlled"][0])
        if self.meta is None:
            weight = ctrl.nom_weight @ ctrl.nom_unit_scale
            self.meta = dict(
                dt=float(ctrl.dt), a_min=float(ctrl.a_min), a_max=float(ctrl.a_max),
                length=float(ctrl.length), width=float(ctrl.width), half=horizon.half.copy(),
                lam_lane=float(horizon.decay_rate(float(ctrl.lambda_ttcbf_predefined))),
                pair_buffer=float(ctrl.pair_safety_buffer), tol=float(ctrl.lane_sat_tol),
                lin_margin=float(ctrl.lane_sat_linearization_margin), W=weight.T @ weight,
                l_r=float(ctrl.l_r), l_wb=float(ctrl.l_wb),
            )  # fmt: skip
        # Shares and decay rates of the budget entries (in the entry order, as repair.py labels them)
        shares = [
            (float(pair["share"]), float(horizon.decay_rate(float(pair["lam"]))))
            for pairs, _, _, _ in pending_pairs
            for pair in pairs
            if pair["share"] is not None
        ]
        record.update(
            X=block["X"].copy(),
            ego=c,
            n_steps=int(block["n_steps"]),
            routes=[route["points"].copy() for route in block["routes"]],
            shares=shares,
        )
        if "budget" in block:
            b = block["budget"]
            record["budget"] = dict(idx=b["idx"].copy(), witness=b["witness"].copy(),
                                    ref_support=b["ref_support"].copy(), budget=b["budget"].copy())  # fmt: skip
        if "lane" in block:
            lane = block["lane"]
            record["lane"] = dict(idx=lane["idx"].copy(), side=lane["side"].copy(),
                                  start=lane["facets"]["start"].copy(), end=lane["facets"]["end"].copy(),
                                  valid=lane["valid"].copy())  # fmt: skip
        # The evaluation must reproduce the filter's values
        poses, values = evaluate(horizon, block, record["u_nom"][None])
        finite = np.isfinite(record["rhs"])
        assert np.allclose(values[0][finite], record["values_nom"][finite], atol=1e-9), "evaluate"
        # Admissible box and the grid over it
        lower, upper = ctrl._shield_input_bounds(block["X"][c : c + 1], block["params"])
        record["bounds"] = (lower[0].copy(), upper[0].copy())
        a = np.linspace(lower[0, 0], upper[0, 0], self.grid)
        w = np.linspace(lower[0, 1], upper[0, 1], self.grid)
        U = np.stack(np.meshgrid(a, w, indexing="ij"), axis=-1).reshape(-1, 2)
        _, values = evaluate(horizon, block, U)
        slack = values[:, :, 1:] - record["rhs"][None, :, 1:]  # t = 2, ..., N
        record["grid_a"], record["grid_w"] = a, w
        record["grid_slack"] = slack.min(axis=2).reshape(self.grid, self.grid, -1).astype(np.float32)
        record["grid_t"] = (slack.argmin(axis=2) + 1).reshape(self.grid, self.grid, -1).astype(np.int16)
        record["backup_control"] = block["backup"][c].copy()
        # The candidate prediction of the backup control is the backup prediction
        poses, values = evaluate(horizon, block, record["backup_control"][None])
        record["prediction_backup"], record["values_backup"] = poses[0], values[0]
        record["initial_rows"] = gradients(horizon, block, record["u_nom"], record["initial"])

    def path_sweep(self, horizon, block, record) -> dict:
        """Predictions and values along the straight path from the nominal to
        the applied control."""
        s = np.linspace(0.0, 1.0, self.path)
        U = record["u_nom"][None] + s[:, None] * (record["u_applied"] - record["u_nom"])[None]
        poses, values = evaluate(horizon, block, U)
        return dict(path_s=s, path_u=U, path_poses=poses, path_values=values.astype(np.float32))

    def save(self, path: str):
        problems = [p for p in self.problems if "u_applied" in p]
        with open(path, "wb") as file:
            pickle.dump(dict(meta=self.meta, problems=problems), file)
        print(f"[INFO] Saved {path} ({len(problems)} filter problems)")


def capture(args):
    (path,) = glob.glob(os.path.join(args.run, "out_td_*.td"))
    td = torch.load(path, weights_only=False)
    targets = td["next", "agents", "info", "pos"][0].numpy()
    start, end = args.steps
    recorder = ExplainerRecorder(
        targets, range(start, end + 1), args.agents, args.grid, args.path
    )
    recorder.install()
    with open(os.path.join(args.run, "command_testing.txt")) as file:
        tokens = shlex.split(file.read())
    argv = tokens[
        tokens.index(next(t for t in tokens if t.endswith("main_testing_cbf_filter.py"))) + 1 :
    ]
    replace = {"--max_steps": str(end + 3), "--output_dir": os.path.join(args.out, "_sim", args.name)}
    for flag, value in replace.items():
        argv[argv.index(flag) + 1] = value
    sys.argv = ["main_testing_cbf_filter.py"] + argv
    runpy.run_path(os.path.join(REPO_ROOT, "main_testing_cbf_filter.py"), run_name="__main__")
    recorder.save(os.path.join(args.out, f"{args.name}.pkl"))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    capture(args)


if __name__ == "__main__":
    main()
