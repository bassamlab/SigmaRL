# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Numbers of the paper's evaluation tables (Stages 1-3), pooled as in the paper.

Prints one Markdown table per paper table, each value as mean ± standard error
over policies (Stage 1) or fine-tuning runs (Stage 2); the own and the unseen
map are pooled. Every policy is first averaged over its evaluation seeds.
Collisions are given per environment-minute ("col") and per km driven
("col/km"; a policy's or run's collisions over the distance of all its
vehicles in all its evaluation runs, distance = mean speed x vehicle time).

- Stage 1 rewards (tab:stage1-reward): all 30 pre-trained policies, unfiltered
  (collisions, throughput) and with BH-CBF dec (interventions), own and unseen
  map separately, both training maps pooled;
- Stage 1 filters (tab:stage1-filters): the TTC policies of iteration 100 with
  every filter, fully decentralized; parallelized time per map; centralized
  collisions and steps outside C as a check;
- stress test: the same policies on rounD with 24 vehicles;
- Stage 2 (tab:stage2): training collisions, the final checkpoint without and
  with the filter, and the source policies (iteration 0 of the BH-CBF runs);
- Stage 3 (tab:stage3): read from the analyze.py summaries.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.paper_tables_stage123
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from sigmarl.evaluation_tits26.settings import N_AGENTS, STAGE2_EXCLUDED_LANE_STEPS
from sigmarl.evaluation_tits26.stage1_tables import load_runs, policy_means

OUT = "outputs/tits26"
STAGE2_METHODS = [
    ("none", "Without filter, TTC reward"),
    ("none_bh", "Without filter, BH-CBF reward"),
    ("ours", "BH-CBF (ours)"),
    ("b6_backup_cbf", "B-Backup"),
    ("mps", "B-MPS"),
    ("t2_dual_distance", "B-Dual"),
]
FILTERS = [
    ("unfiltered", "Unfiltered"),
    ("ours", "BH-CBF (ours)"),
    ("ours_onestep", "A-Condition"),
    ("b1_lse_sat", "B-SAT"),
    ("b2_frenet_heading", "B-Frenet"),
    ("b3_wang", "B-Disk"),
    ("t2_dual_distance", "B-Dual"),
    ("bh_capsule", "A-Geometry"),
    ("bh_half_split", "A-Split"),
    ("b6_backup_cbf", "B-Backup"),
    ("mps", "B-MPS"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out_root", default=OUT)
    return parser.parse_args()


def fmt(values, digits: int) -> str:
    """Mean ± standard error of the finite values."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    se = values.std(ddof=1) / math.sqrt(len(values))
    return f"{values.mean():.{digits}f} ± {se:.{digits}f}"


def total_col(policy: dict) -> float:
    return policy["veh_col"] + policy["lane_col"]


def per_km(rows: list, keys: tuple, n_agents: int = None) -> dict:
    """{group key: [per-policy collisions per km]} for the Stage 1 rows
    grouped by `keys` and the policy, pooled over the policy's runs. Each
    run is one environment-minute of N_AGENTS (or n_agents) vehicles."""
    pooled = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        n = n_agents or N_AGENTS[row["deploy_map"]]
        key = tuple(row[k] for k in keys) + (row["checkpoint"],)
        pooled[key][0] += row["veh_col"] + row["lane_col"]
        pooled[key][1] += n * row["speed"] * 60.0 / 1000.0
    groups = defaultdict(list)
    for key, (collisions, km) in pooled.items():
        groups[key[:-1]].append(collisions / km)
    return groups


def table(title: str, header: list, rows: list):
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def stage1_rewards(root: str):
    runs = load_runs(os.path.join(root, "stage1_zero_shot", "runs.csv"))
    unfiltered = policy_means(
        [r for r in runs if r["method"] == "unfiltered"], ("reward", "relation")
    )
    filtered = policy_means(
        [r for r in runs if r["method"] == "ours" and r["mode"] == "dec"],
        ("reward", "relation"),
    )
    unfiltered_km = per_km(
        [r for r in runs if r["method"] == "unfiltered"], ("reward", "relation")
    )
    rows = []
    for reward, name in (("ttc", "TTC"), ("onestep", "One-Step CBF"), ("bh", "BH-CBF")):
        row = [name]
        for relation in ("own", "unseen"):
            free = unfiltered[(reward, relation)]
            row += [
                fmt([total_col(p) for p in free], 2),
                fmt(unfiltered_km[(reward, relation)], 2),
                fmt([p["veh_col"] for p in free], 2)
                + " / "
                + fmt([p["lane_col"] for p in free], 2),
                fmt([p["throughput"] for p in free], 3),
                fmt([p["intervention"] for p in filtered[(reward, relation)]], 1),
            ]
        rows.append(row)
    header = ["Reward"] + [
        f"{rel}: {m}"
        for rel in ("own", "unseen")
        for m in ("col", "col/km", "veh / lane", "thr", "int BH-CBF")
    ]
    table("Stage 1 rewards (all policies, both training maps pooled)", header, rows)


def stage1_filters(root: str, folder: str, title: str, n_agents: int = None):
    runs = load_runs(os.path.join(root, folder, "runs.csv"))
    rows = []
    for mode in ("dec", "cen"):
        groups = policy_means(
            [r for r in runs if r["mode"] == mode or r["method"] == "unfiltered"],
            ("method",),
        )
        groups_km = per_km(
            [r for r in runs if r["mode"] == mode or r["method"] == "unfiltered"],
            ("method",),
            n_agents,
        )
        by_map = policy_means(
            [r for r in runs if r["mode"] == mode], ("method", "deploy_map")
        )
        for method, name in FILTERS:
            if (method,) not in groups:
                continue
            policies = groups[(method,)]
            runs_outside = sum(
                1
                for r in runs
                if r["method"] == method
                and (r["mode"] == mode or method == "unfiltered")
                and r["outside_c"] > 0
            )
            n_runs = sum(
                1
                for r in runs
                if r["method"] == method
                and (r["mode"] == mode or method == "unfiltered")
            )
            rows.append(
                [
                    mode,
                    name,
                    fmt([total_col(p) for p in policies], 2),
                    fmt(groups_km[(method,)], 3),
                    fmt([p["veh_col"] for p in policies], 2)
                    + " / "
                    + fmt([p["lane_col"] for p in policies], 2),
                    fmt([p["outside_c"] for p in policies], 2),
                    f"{runs_outside}/{n_runs}",
                    f"{min(p['min_backup'] for p in policies):.2f}",
                    fmt([p["fallback"] for p in policies], 1),
                    fmt([p["intervention"] for p in policies], 1),
                    fmt([p["infeas"] for p in policies], 1),
                    fmt([p["throughput"] for p in policies], 3),
                    " / ".join(
                        fmt([p["t_par"] for p in by_map.get((method, m), [])], 1)
                        for m in ("round_location0", "ind_location4")
                    ),
                ]
            )
    header = [
        "mode", "filter", "col", "col/km", "veh / lane", "outside C [%]", "runs leaving C",
        "min backup", "fallback [1/1000]", "int [%]", "infeas [%]", "thr", "t_par rounD / inD",
    ]  # fmt: skip
    table(title, header, rows)


def jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def rate(entry: dict) -> float:
    """Colliding vehicles per environment-minute."""
    return entry["collision_events"] / (entry["num_envs"] * entry["duration_s"] / 60.0)


def rate_km(entry: dict, n_agents: int) -> float:
    """Colliding vehicles per km driven."""
    km = (
        entry["num_envs"]
        * entry["duration_s"]
        * n_agents
        * entry["mean_speed"]
        / 1000.0
    )
    return entry["collision_events"] / km


def stage2_km(row: dict, prefix: str) -> float:
    """Collisions per km of a runs.csv row (seed means of equal durations, so
    the ratio of the means pools the seeds)."""
    km_per_min = (
        N_AGENTS[row["target_map"]] * float(row[f"{prefix}_speed"]) * 60.0 / 1000.0
    )
    return float(row[f"{prefix}_col"]) / km_per_min


def stage2_train_km(base: str, run: str) -> float:
    """Km driven in the training rollouts of a run (agent time times the mean
    speed per iteration, as in paper_fig_stage12_collisions)."""
    with open(os.path.join(base, run, "train_stats.csv")) as file:
        return sum(
            float(r["agent_steps"]) * 0.1 * float(r["mean_speed"]) / 1000.0
            for r in csv.DictReader(file)
        )


def stage2(root: str):
    base = os.path.join(root, "stage2_finetune")
    with open(os.path.join(base, "runs.csv")) as file:
        runs = list(csv.DictReader(file))
    rows = []
    # Source policies: iteration 0 of the BH-CBF runs (without the filter and with BH-CBF)
    source_free, source_filter, source_km, source_filter_km = [], [], [], []
    for run in glob.glob(os.path.join(base, "*__iter100", "*", "ours", "seed1")):
        free = [
            e for e in jsonl(f"{run}/eval_nofilter_seed201.jsonl") if e["iter"] == 0
        ]
        filt = [e for e in jsonl(f"{run}/eval_filter_seed201.jsonl") if e["iter"] == 0]
        source_free += free
        source_filter += filt
        n_agents = N_AGENTS[run.split(os.sep)[-3]]
        source_km += [rate_km(e, n_agents) for e in free]
        source_filter_km += [rate_km(e, n_agents) for e in filt]
    rows.append(
        [
            "Without fine-tuning",
            "-",
            "-",
            "-",
            fmt([rate(e) for e in source_free], 2),
            fmt(source_km, 3),
            fmt([e["throughput_veh_s"] for e in source_free], 3),
            fmt([rate(e) for e in source_filter], 2),
            fmt(source_filter_km, 3),
            fmt([e["fallback_per_1000"] for e in source_filter], 2),
            fmt([100 * e["intervention_rate"] for e in source_filter], 1),
            fmt([e["throughput_veh_s"] for e in source_filter], 3),
            "-",
        ]
    )
    # Lane-boundary contact steps of the map artifact (settings.STAGE2_EXCLUDED_LANE_STEPS)
    excluded = defaultdict(int)
    for (run, _), steps in STAGE2_EXCLUDED_LANE_STEPS.items():
        excluded[run] += steps
    for row in runs:
        run = f"{row['source']}/{row['target_map']}/{row['method']}/seed{row['seed']}"
        if excluded[run]:
            for key in ("train_lane_col", "train_col"):
                row[key] = str(float(row[key]) - excluded[run])
    for method, name in STAGE2_METHODS:
        group = [r for r in runs if r["method"] == method]

        def col(key):
            return [
                float(r[key]) if r[key] not in ("", "nan") else math.nan for r in group
            ]

        leaving = sum(1 for r in group if float(r["train_min_backup"] or "nan") < -1e-3)
        rows.append(
            [
                name,
                fmt(col("train_col"), 1)
                + f" (veh {fmt(col('train_veh_col'), 1)}, lane {fmt(col('train_lane_col'), 1)})",
                fmt(
                    [
                        float(r["train_col"])
                        / stage2_train_km(
                            base,
                            f"{r['source']}/{r['target_map']}/{r['method']}/seed{r['seed']}",
                        )
                        for r in group
                    ],
                    3,
                ),
                f"{leaving}/{len(group)} runs, {fmt(col('train_outside_c'), 3)} %"
                if method not in ("none", "none_bh")
                else "-",
                fmt(col("nofilter_col"), 2),
                fmt([stage2_km(r, "nofilter") for r in group], 3),
                fmt(col("nofilter_throughput"), 3),
                fmt(col("filter_col"), 3),
                fmt([stage2_km(r, "filter") for r in group], 3),
                fmt(col("filter_fallback"), 2),
                fmt(col("filter_intervention"), 1),
                fmt(col("filter_throughput"), 3),
                fmt(col("wall_h"), 1),
            ]
        )
    header = [
        "condition", "train col (agent-steps)", "train col/km", "train leaving C", "no filter col",
        "no filter col/km", "no filter thr", "filter col", "filter col/km", "filter fallback", "filter int [%]", "filter thr", "time [h]",
    ]  # fmt: skip
    table(
        "Stage 2 (20 runs per row, own and unseen pooled; final checkpoint, 3 seeds)",
        header,
        rows,
    )


def main():
    args = parse_args()
    stage1_rewards(args.out_root)
    stage1_filters(
        args.out_root,
        "stage1_filters_ttc_iter100",
        "Stage 1 filters (TTC iteration-100 policies, own and unseen pooled)",
    )
    stage1_filters(
        args.out_root,
        "stage1_stress_round24",
        "Stress test (rounD, 24 vehicles)",
        n_agents=24,
    )
    stage2(args.out_root)


if __name__ == "__main__":
    main()
