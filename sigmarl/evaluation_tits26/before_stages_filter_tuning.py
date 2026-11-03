# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tune every safety filter with the same budget, before Stages 1-3.

Phase 1 sweeps the CBF decay rate lambda of every method, mode, and block.
Phase 2 sweeps each method's own parameter (if any, see METHOD_KNOBS) at the
lambda selected in phase 1. Tuning uses its own seeds (TUNING_SEEDS), disjoint
from the evaluation seeds 1-5, and runs every setting on all tuning scenarios
(SCENARIOS); one setting is selected per method, mode, and block for all of
them. The selection rule (see select) is the same for all methods. A setting's
goal score is its goals relative to the method's best on each scenario,
averaged over the scenarios. Among the settings that keep at least
GOAL_FRACTION of the best goal score (a stalled fleet is trivially
collision-free), take those whose mean total collisions per run (over all
scenarios) are at most COLLISION_TOLERANCE above the lowest, and among them the
one with the highest goal score.
The result, <out_root>/tuned.json, is read by before_stages_filter_benchmark.py --tuned.

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m sigmarl.evaluation_tits26.before_stages_filter_tuning [--workers 6]
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from sigmarl.evaluation_tits26.before_stages_filter_benchmark import (
    BLOCKS,
    METHODS,
    MODES,
    REPO_ROOT,
    base_env,
    build_jobs,
    run_jobs,
)

LAMBDAS = (0.1, 0.2, 0.3, 0.5, 0.7)
TUNING_SEEDS = (101, 102, 103)
SCENARIOS = ("round_location0", "ind_location4")
# Collisions per run within which settings count as equally safe
COLLISION_TOLERANCE = 1.0
# Liveness: a setting must reach this fraction of the method's best goals
GOAL_FRACTION = 0.75
# Method -> (environment variable, values) of its own parameter (phase 2)
METHOD_KNOBS = {
    # B1 smoothing kappa at lab scale (divided by LENGTH_SCALE = 20: 2-50 /m)
    "b1_lse_sat": ("SIGMARL_B1_LSE_KAPPA", ("40", "100", "200", "400", "1000")),
    "b1_bh": ("SIGMARL_B1_LSE_KAPPA", ("40", "100", "200", "400", "1000")),
    # B4 radius: ego length (paper text) or ego width (Fig. 3 caption)
    "b4_collision_cone": ("SIGMARL_B4_RADIUS", ("length", "width")),
}
# Own parameters during phase 1: the published values (Wu et al.'s 5 /m; the
# radius of Goswami et al.'s text), not the tuned defaults
PHASE1_ENV = ("SIGMARL_B1_LSE_KAPPA=100", "SIGMARL_B4_RADIUS=length")


def tag_of(cbf_lambda: float, knob: str = "") -> str:
    return f"lam{cbf_lambda:g}" + (f"_k{knob}" if knob else "")


def make_args(out_root, methods, modes, blocks, cbf_lambda, tag, n_agents, scenario):
    """Arguments of before_stages_filter_benchmark.build_jobs for one setting on one scenario."""
    return SimpleNamespace(
        methods=methods,
        modes=modes,
        blocks=blocks,
        seeds=list(TUNING_SEEDS),
        max_steps=601,
        n_agents=n_agents,
        scenario_type=scenario,
        out_root=os.path.join(out_root, scenario),
        tag=tag,
        cbf_lambda=cbf_lambda,
        tuned=None,
    )


def setting_scores(out_root, block, method, mode, tag) -> dict | None:
    """Mean total collisions and goals of one setting over the tuning seeds,
    per scenario."""
    from sigmarl.evaluation_tits26.before_stages_filter_summary import run_metrics

    scores = {}
    for scenario in SCENARIOS:
        runs = [
            os.path.join(
                out_root, scenario, block, f"{method}_{tag}", mode, f"seed{seed}"
            )
            for seed in TUNING_SEEDS
        ]
        metrics = [run_metrics(run) for run in runs if os.path.isdir(run)]
        if len(metrics) < len(TUNING_SEEDS):
            return None
        scores[scenario] = dict(
            collisions=float(np.mean([m["total_col"] for m in metrics])),
            goals=float(np.mean([m["goals"] for m in metrics])),
        )
    return scores


def select(scores: dict) -> str:
    """Setting with the highest goal score among the live settings (at least
    GOAL_FRACTION of the best goal score) within COLLISION_TOLERANCE mean
    collisions per run of the safest live one; see the module docstring."""
    best = {
        sc: max(max(s[sc]["goals"] for s in scores.values()), 1e-9) for sc in SCENARIOS
    }
    goal_score = {
        t: float(np.mean([s[sc]["goals"] / best[sc] for sc in SCENARIOS]))
        for t, s in scores.items()
    }
    collisions = {
        t: float(np.mean([s[sc]["collisions"] for sc in SCENARIOS]))
        for t, s in scores.items()
    }
    best_score = max(goal_score.values())
    live = [t for t in scores if goal_score[t] >= GOAL_FRACTION * best_score]
    safest = min(collisions[t] for t in live)
    candidates = [t for t in live if collisions[t] <= safest + COLLISION_TOLERANCE]
    return max(candidates, key=lambda t: goal_score[t])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--methods", nargs="+", default=[m for m in METHODS if METHODS[m] is not None]
    )
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n_agents", type=int, default=14)
    parser.add_argument(
        "--out_root",
        default=os.path.join(
            REPO_ROOT, "outputs", "tits26", "before_stages_filter_tuning"
        ),
    )
    args = parser.parse_args()

    # Phase 1: lambda of every method
    jobs = []
    for cbf_lambda in LAMBDAS:
        for scenario in SCENARIOS:
            jobs += build_jobs(
                make_args(
                    args.out_root,
                    args.methods,
                    args.modes,
                    args.blocks,
                    cbf_lambda,
                    tag_of(cbf_lambda),
                    args.n_agents,
                    scenario,
                ),
                base_env(PHASE1_ENV),
            )
    run_jobs(jobs, args.workers)

    tuned, report = {}, {}
    for block in args.blocks:
        for method in args.methods:
            for mode in args.modes:
                scores = {
                    tag_of(lam): setting_scores(
                        args.out_root, block, method, mode, tag_of(lam)
                    )
                    for lam in LAMBDAS
                }
                scores = {tag: s for tag, s in scores.items() if s is not None}
                best = select(scores)
                key = f"{block}/{method}/{mode}"
                tuned[key] = dict(cbf_lambda=float(best[3:]))
                report[key] = dict(phase1=scores, lambda_selected=float(best[3:]))

    # Phase 2: the method's own parameter at its selected lambda
    jobs = []
    for method, (variable, values) in METHOD_KNOBS.items():
        if method not in args.methods:
            continue
        for block in args.blocks:
            for mode in args.modes:
                lam = tuned[f"{block}/{method}/{mode}"]["cbf_lambda"]
                for value in values:
                    for scenario in SCENARIOS:
                        jobs += build_jobs(
                            make_args(
                                args.out_root,
                                [method],
                                [mode],
                                [block],
                                lam,
                                tag_of(lam, value),
                                args.n_agents,
                                scenario,
                            ),
                            base_env([f"{variable}={value}"]),
                        )
    run_jobs(jobs, args.workers)
    for method, (variable, values) in METHOD_KNOBS.items():
        if method not in args.methods:
            continue
        for block in args.blocks:
            for mode in args.modes:
                key = f"{block}/{method}/{mode}"
                lam = tuned[key]["cbf_lambda"]
                scores = {
                    value: setting_scores(
                        args.out_root, block, method, mode, tag_of(lam, value)
                    )
                    for value in values
                }
                scores = {v: s for v, s in scores.items() if s is not None}
                best = select(scores)
                tuned[key]["env"] = {variable: best}
                report[key].update(phase2=scores, knob_selected={variable: best})

    with open(os.path.join(args.out_root, "tuned.json"), "w") as file:
        json.dump(tuned, file, indent=2)
    with open(os.path.join(args.out_root, "tuning_report.json"), "w") as file:
        json.dump(report, file, indent=2)
    for key, value in tuned.items():
        print(f"{key:45s} {value}")


if __name__ == "__main__":
    main()
