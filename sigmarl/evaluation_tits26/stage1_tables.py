# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 1 of the T-ITS evaluation: the paper's tables of zero-shot deployment.

Reads <out_root>/runs.csv (stage1_zero_shot.py --summarize) and writes
<out_root>/tables.md and one CSV per table:

- table_reward.csv (tab:stage1-reward): every policy unfiltered (collisions,
  throughput) and with BH-CBF dec (intervention), per training map, reward,
  and deployment map;
- table_filters.csv (tab:stage1-filters): the source policies (--filter_reward,
  default the BH-CBF-reward policies) with every filter, per mode, own and
  unseen map;
- table_dec.csv (tab:stage1-dec): BH-CBF centralized, decentralized with the
  corner budgets, and with the half split;
- table_time.csv (tab:time): CBF time per step, centralized and decentralized
  (sequential wall-clock time of the implementation and parallelized
  estimate).

A run lasts 601 steps of 0.1 s, i.e., one environment-minute. Collisions are
counted as colliding vehicles per environment-minute: a colliding vehicle is
reset in the next step, and an inter-vehicle collision counts both vehicles.
Every value is first averaged over the evaluation seeds of a policy; the tables
give the mean and the standard error over policies (training seeds, and both
training maps where pooled). Each table is followed by the definitions of its
columns and by the conclusions in <out_root>/conclusions/<table>.md (written by
hand, kept when the tables are regenerated).

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.stage1_tables \\
        [--out_root outputs/tits26/stage1_zero_shot]
    ... --out_root outputs/tits26/stage1_filters_ttc_iter100 --filter_reward ttc \\
        --tables filters dec time
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

REWARDS = {"ttc": "TTC", "onestep": "One-Step CBF", "bh": "BH-CBF (ours)"}
MAP_NAMES = {"round_location0": "rounD", "ind_location4": "inD"}
# Filters in the order of the paper's tables
FILTERS = [
    ("unfiltered", "Unfiltered"),
    ("ours", "BH-CBF (ours)"),
    ("ours_onestep", "A-Condition"),
    ("b1_lse_sat", "B-SAT"),
    ("b2_frenet_heading", "B-Frenet"),
    ("b3_wang", "B-Disk"),
    ("t2_dual_distance", "B-Dual"),
    ("bh_capsule", "A-Geometry"),
    ("b6_backup_cbf", "B-Backup"),
    ("bh_half_split", "A-Split"),
    ("mps", "B-MPS"),
]
MODES = {"dec": "Fully decentralized", "cen": "Centralized"}
NUMERIC = [
    "veh_col",
    "lane_col",
    "throughput",
    "speed",
    "intervention",
    "infeas",
    "fallback",
    "outside_c",
    "min_backup",
    "t_seq",
    "t_par",
    "n_onsets",
    "n_onsets_after_infeasible",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out_root", default="outputs/tits26/stage1_zero_shot")
    parser.add_argument(
        "--filter_reward",
        choices=list(REWARDS),
        default="bh",
        help="Reward of the source policies of the filter, dec, and time tables.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=("reward", "filters", "dec", "time"),
        default=["reward", "filters", "dec", "time"],
    )
    return parser.parse_args()


def load_runs(path: str) -> list:
    """Rows of runs.csv with numbers and the policy's reward."""
    rows = []
    with open(path) as file:
        for row in csv.DictReader(file):
            for key in NUMERIC:
                row[key] = float(row[key]) if row[key] not in ("", "nan") else math.nan
            # Checkpoint label <training map>__stage1_<reward>_seed_<k>[__iter<N>]
            row["reward"] = row["checkpoint"].split("__stage1_")[1].split("_seed_")[0]
            rows.append(row)
    return rows


def policy_means(rows: list, keys: tuple) -> dict:
    """{group key: [per-policy dict of seed means]} for the rows grouped by
    `keys` and the policy (checkpoint)."""
    by_policy = defaultdict(list)
    for row in rows:
        by_policy[tuple(row[k] for k in keys) + (row["checkpoint"],)].append(row)
    groups = defaultdict(list)
    for key, runs in by_policy.items():
        means = {m: float(np.nanmean([r[m] for r in runs])) for m in NUMERIC}
        means["min_backup"] = float(np.nanmin([r["min_backup"] for r in runs]))
        means["onsets"] = sum(r["n_onsets"] for r in runs)
        means["onsets_after_infeasible"] = sum(
            r["n_onsets_after_infeasible"] for r in runs
        )
        means["runs"] = len(runs)
        groups[key[:-1]].append(means)
    return groups


def stat(policies: list, metric: str, digits: int) -> str:
    """Mean +- standard error over policies."""
    values = np.array([p[metric] for p in policies], dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "-"
    mean = values.mean()
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    se = values.std(ddof=1) / math.sqrt(len(values))
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def after_infeasible(policies: list) -> str:
    """Share of the collision onsets that follow a step with an infeasible QP."""
    onsets = sum(p["onsets"] for p in policies)
    if onsets == 0:
        return "-"
    return f"{100 * sum(p['onsets_after_infeasible'] for p in policies) / onsets:.0f}%"


# Definitions of the columns, printed below each table
NOTES = {
    "table_reward": [
        "Policies: training seeds (one policy each), each deployed with 3 evaluation seeds.",
        "Coll.: colliding vehicles per environment-minute without a filter (inter-vehicle / lane-boundary).",
        "Thr.: vehicles that reach their goal per second, without a filter. A colliding vehicle respawns "
        "at a random point of its route, so collisions do not lower the throughput directly.",
        "Speed: mean speed of all vehicles, without a filter.",
        "Int. BH-CBF dec: the same policy deployed with the fully decentralized BH-CBF filter; share of "
        "agent-steps at which the filter changes the policy's speed or steering target (lower: the policy "
        "already proposes actions that the filter accepts).",
    ],
    "table_filters": [
        "Policies: the source policies (5 training seeds on each map, see the top of the page), each "
        "with 3 evaluation seeds; own: deployed on its training map, unseen: on the other map.",
        "Coll.: colliding vehicles per environment-minute (inter-vehicle / lane-boundary).",
        "Thr.: vehicles that reach their goal per second (colliding vehicles respawn, see above).",
        "ΔThr. vs BH-CBF: paired difference to BH-CBF in the same mode (same policy, map, and seed); "
        "negative: less throughput than BH-CBF.",
        "Int.: share of agent-steps at which the filter changes the policy's speed or steering target.",
        "Infeas.: share of steps at which the linearized QP of at least one vehicle needs slack or fails. For "
        "BH-CBF and the backup-horizon filters, the control is then repaired or replaced by the braking "
        "backup, which is always feasible; for the one-step filters, the slack violates the condition.",
        "Coll. after infeas.: share of the collisions that follow a step with an infeasible QP within 1.5 s.",
        "Fallback: vehicles switched to their braking backup per 1000 agent-steps (backup-horizon filters).",
        "Steps outside C: share of steps at which the joint state is not certified, i.e., the backup value "
        "of at least one vehicle is negative (braking all vehicles from there violates a margin); evaluated "
        "with our margins for every filter.",
        "Min. backup value: smallest backup value over all runs, vehicles, and steps; >= 0 means that the "
        "state stayed certified throughout.",
    ],
    "table_dec": [
        "Policies, Coll., Steps outside C, Min. backup value, Thr., Int., Fallback: as in the filter table "
        "(both training maps, own and unseen map pooled).",
        "CBF seq.: sequential wall-clock time per step of the implementation (all vehicles one after the "
        "other); CBF par.: parallelized estimate for decentralized execution (shared preparation + the "
        "slowest vehicle's QP + the slowest vehicle's second round).",
    ],
    "table_time": [
        "CBF time per step with the source policies, measured with 10 rollouts running in parallel "
        "on the same machine (equal load for every filter). Dec., sequential: wall-clock loop over all "
        "vehicles; Dec., parallelized: estimate of the time with one processor per vehicle.",
    ],
}


def write(out_root: str, name: str, header: list, rows: list, lines: list, title: str):
    """Append a markdown table, its column notes, and its conclusions to
    `lines`, and write the table as CSV."""
    with open(os.path.join(out_root, f"{name}.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    lines += [f"## {title}", "", "| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    lines.append("")
    lines += [f"- {note}" for note in NOTES[name]]
    lines.append("")
    conclusions = os.path.join(out_root, "conclusions", f"{name}.md")
    if os.path.exists(conclusions):
        with open(conclusions) as file:
            lines += ["**Conclusions.**", "", file.read().strip(), ""]


def paired_difference(rows: list, method: str, mode: str, relation: str) -> list:
    """Per-policy means of the throughput difference to BH-CBF in `mode`,
    paired by policy, map, and evaluation seed (the unfiltered reference runs
    only centralized)."""
    method_mode = "cen" if method == "unfiltered" else mode
    reference = {
        (r["checkpoint"], r["deploy_map"], r["seed"]): r["throughput"]
        for r in rows
        if r["method"] == "ours" and r["mode"] == mode
    }
    by_policy = defaultdict(list)
    for r in rows:
        key = (r["checkpoint"], r["deploy_map"], r["seed"])
        if (
            r["method"] == method
            and r["mode"] == method_mode
            and r["relation"] == relation
            and key in reference
        ):
            by_policy[r["checkpoint"]].append(r["throughput"] - reference[key])
    return [{"diff": float(np.mean(v))} for v in by_policy.values()]


def main():
    args = parse_args()
    runs = load_runs(os.path.join(args.out_root, "runs.csv"))
    lines = [
        "# Stage 1: tables of zero-shot deployment",
        "",
        "Mean ± standard error over policies (training seeds); every policy is first "
        "averaged over its evaluation seeds. Collisions: colliding vehicles per "
        "environment-minute (inter-vehicle / lane-boundary); an inter-vehicle collision "
        "counts both vehicles.",
        "",
    ]
    filter_runs = [r for r in runs if r["reward"] == args.filter_reward]
    iterations = {
        r["checkpoint"].split("__iter")[1]
        for r in filter_runs
        if "__iter" in r["checkpoint"]
    }
    source = f"iteration {', '.join(sorted(iterations))}" if iterations else "final"
    if set(args.tables) - {"reward"}:
        lines += [
            f"Source policies of the filter, decentralization, and time tables: trained "
            f"with the {REWARDS[args.filter_reward]} reward, {source} checkpoint.",
            "",
        ]
    if "reward" in args.tables:
        reward_table(runs, args.out_root, lines)
    if "filters" in args.tables:
        filter_table(filter_runs, args.out_root, lines)
    if "dec" in args.tables:
        dec_table(filter_runs, args.out_root, lines)
    if "time" in args.tables:
        time_table(filter_runs, args.out_root, lines)

    text = "\n".join(lines)
    with open(os.path.join(args.out_root, "tables.md"), "w") as file:
        file.write(text)
    print(text)


def reward_table(runs: list, out_root: str, lines: list):
    """tab:stage1-reward: every policy unfiltered and with BH-CBF dec."""
    groups = policy_means(runs, ("train_map", "reward", "deploy_map", "method", "mode"))
    header = ["Trained on", "Reward", "Deployed on", "Policies", "Coll. (veh / lane)"]
    header += ["Thr. [veh/s]", "Speed [m/s]", "Int. BH-CBF dec [%]"]
    rows = []
    for train_map in MAP_NAMES:
        for reward, reward_name in REWARDS.items():
            for deploy_map in MAP_NAMES:
                free = groups.get((train_map, reward, deploy_map, "unfiltered", "cen"))
                ours = groups.get((train_map, reward, deploy_map, "ours", "dec"))
                if not free:
                    continue
                rows.append(
                    [
                        MAP_NAMES[train_map],
                        reward_name,
                        MAP_NAMES[deploy_map],
                        len(free),
                        f"{stat(free, 'veh_col', 2)} / {stat(free, 'lane_col', 2)}",
                        stat(free, "throughput", 3),
                        stat(free, "speed", 2),
                        stat(ours, "intervention", 1) if ours else "-",
                    ]
                )
    write(
        out_root,
        "table_reward",
        header,
        rows,
        lines,
        "Rewards (tab:stage1-reward)",
    )


def filter_table(bh_runs: list, out_root: str, lines: list):
    """tab:stage1-filters: the source policies with every filter, own and
    unseen map pooled over both training maps."""
    groups = policy_means(bh_runs, ("relation", "method", "mode"))
    header = ["Mode", "Filter", "Map", "Policies", "Coll. (veh / lane)", "Thr. [veh/s]"]
    header += [
        "ΔThr. vs BH-CBF [veh/s]",
        "Int. [%]",
        "Infeas. [%]",
        "Coll. after infeas.",
        "Fallback [1/1000]",
    ]
    header += ["Steps outside C [%]", "Min. backup value [m]"]
    rows = []
    for mode, mode_name in MODES.items():
        for method, name in FILTERS:
            for relation in ("own", "unseen"):
                # The unfiltered reference is mode-independent (run centralized)
                policies = groups.get(
                    (relation, method, "cen" if method == "unfiltered" else mode)
                )
                if not policies:
                    continue
                rows.append(
                    [
                        mode_name,
                        name,
                        relation,
                        len(policies),
                        f"{stat(policies, 'veh_col', 2)} / {stat(policies, 'lane_col', 2)}",
                        stat(policies, "throughput", 3),
                        "-"
                        if method == "ours"
                        else stat(
                            paired_difference(bh_runs, method, mode, relation),
                            "diff",
                            3,
                        ),
                        stat(policies, "intervention", 1),
                        stat(policies, "infeas", 1),
                        after_infeasible(policies),
                        stat(policies, "fallback", 2),
                        stat(policies, "outside_c", 2),
                        f"{min(p['min_backup'] for p in policies):.3f}",
                    ]
                )
    write(
        out_root,
        "table_filters",
        header,
        rows,
        lines,
        "Filters (tab:stage1-filters)",
    )


def dec_table(bh_runs: list, out_root: str, lines: list):
    """tab:stage1-dec: BH-CBF centralized, with the corner budgets, and with
    the half split."""
    groups = policy_means(bh_runs, ("method", "mode"))
    header = ["Mode", "Policies", "Coll. (veh / lane)", "Steps outside C [%]"]
    header += [
        "Min. backup value [m]",
        "Thr. [veh/s]",
        "Int. [%]",
        "Fallback [1/1000]",
        "CBF seq. [ms]",
        "CBF par. [ms]",
    ]
    rows = []
    for (method, mode), name in [
        (("ours", "cen"), "Centralized"),
        (("ours", "dec"), "Dec., corner budgets (ours)"),
        (("bh_half_split", "dec"), "Dec., half split"),
    ]:
        policies = groups.get((method, mode))
        if not policies:
            continue
        rows.append(
            [
                name,
                len(policies),
                f"{stat(policies, 'veh_col', 2)} / {stat(policies, 'lane_col', 2)}",
                stat(policies, "outside_c", 2),
                f"{min(p['min_backup'] for p in policies):.3f}",
                stat(policies, "throughput", 3),
                stat(policies, "intervention", 1),
                stat(policies, "fallback", 2),
                stat(policies, "t_seq", 1),
                stat(policies, "t_par", 1) if mode == "dec" else "-",
            ]
        )
    write(
        out_root,
        "table_dec",
        header,
        rows,
        lines,
        "Decentralization (tab:stage1-dec)",
    )


def time_table(bh_runs: list, out_root: str, lines: list):
    """tab:time: CBF time per step and map of the source policies."""
    groups = policy_means(bh_runs, ("deploy_map", "method", "mode"))
    header = ["Filter", "Map", "Centralized [ms]", "Dec., sequential [ms]"]
    header += ["Dec., parallelized (est.) [ms]"]
    rows = []
    for method, name in FILTERS[1:]:
        for deploy_map, map_name in MAP_NAMES.items():
            cen = groups.get((deploy_map, method, "cen"))
            dec = groups.get((deploy_map, method, "dec"))
            if not cen and not dec:
                continue
            rows.append(
                [
                    name,
                    map_name,
                    stat(cen, "t_seq", 1) if cen else "-",
                    stat(dec, "t_seq", 1) if dec else "-",
                    stat(dec, "t_par", 1) if dec else "-",
                ]
            )
    write(out_root, "table_time", header, rows, lines, "Computation time (tab:time)")


if __name__ == "__main__":
    main()
