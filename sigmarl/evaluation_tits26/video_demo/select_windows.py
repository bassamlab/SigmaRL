# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Candidate time windows for the scenes of the video demo (video.py).

Every window must be free of deadlocks: no vehicle flagged as deadlocked and
no vehicle in view slower than 0.3 m/s for --stall seconds. Collisions count
lane-boundary onsets plus vehicle-vehicle onsets / 2 (a pair counts once).

Four searches (subcommands), each printing the best candidates:

- ``rewards``: Stage 1 pre-trained policies (iteration 500,
  outputs/tits26/stage1_zero_shot) of the three safety rewards with the same
  map, training seed, and deployment seed, deployed on their own map without
  a filter over the first --steps steps (same start for all three): the
  BH-CBF reward without collisions and the TTC and one-step CBF rewards with
  at least one each; ranked by the fewer collisions of the two, then their sum;
- ``comparison``: Stage 2 runs (outputs/tits26/stage1_filters_ttc_iter100)
  of one policy and seed without a filter, with the one-step CBF filter, and
  with BH-CBF, over the first --steps steps (same start for all three): BH-CBF
  without collisions, the one-step filter with at least one, and no filter
  with at least as many as the one-step filter; ranked by the one-step
  filter's collisions, then those without a filter;
- ``backup``: windows of --steps steps of the Stage 2 BH-CBF runs on --map, ranked
  by the mean number of moving vehicles (faster than 2 m/s) in view;
- ``stage2``: the Stage 3 (fine-tuning) rollouts of rollouts.py (before and after
  fine-tuning, without a filter) per seed and window length from step 0, with
  their collisions and longest standstill in view.

Usage (from the repository root):
    SIGMARL_AGENT_PROFILE=full_scale python -m \\
        sigmarl.evaluation_tits26.video_demo.select_windows rewards --steps 300
    ... select_windows comparison --steps 250
    ... select_windows backup --map ind_location4 --steps 100
    ... select_windows stage2
"""

import argparse
import glob
import os

import numpy as np
import torch

STAGE1 = "outputs/tits26/stage1_filters_ttc_iter100"
REWARDS = "outputs/tits26/stage1_zero_shot"
STAGE2 = "outputs/tits26/video_demo/stage2/round_location0__stage1_ttc_seed_1__iter100__ind_location4__ours__seed1"
SEEDS = (201, 202, 203)
# Region of each map shown in the video [m]: x0, x1, y0, y1
VIEW = {"round_location0": (68, 132, 22, 86), "ind_location4": (70, 140, 36, 90)}
SLOW = 0.3  # Speed below which a vehicle stands still [m/s]
MOVING = 2.0  # Speed above which a vehicle moves [m/s]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rewards = sub.add_parser("rewards")
    rewards.add_argument("--steps", type=int, nargs="+", default=[250, 300, 350])
    comparison = sub.add_parser("comparison")
    comparison.add_argument("--steps", type=int, default=250)
    backup = sub.add_parser("backup")
    backup.add_argument("--map", choices=list(VIEW), default="ind_location4")
    backup.add_argument("--steps", type=int, default=100)
    stage2 = sub.add_parser("stage2")
    stage2.add_argument("--run", default=STAGE2)
    stage2.add_argument("--lengths", type=int, nargs="+", default=[250, 300, 350, 400])
    for command, stall in ((rewards, 3.0), (comparison, 3.0), (backup, 2.0), (stage2, 3.0)):
        command.add_argument(
            "--stall",
            type=float,
            default=stall,
            help="Standstill that counts as a deadlock [s].",
        )
        command.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def load(path: str) -> dict:
    """Positions [T, n, 2], speeds [T, n], and flags [T, n] after each step of
    a rollout (`path`: its out_td file, or the run folder that holds one)."""
    (path,) = (
        [path] if path.endswith(".td") else glob.glob(os.path.join(path, "out_td_*.td"))
    )
    info = torch.load(path, weights_only=False)["next", "agents", "info"]
    return dict(
        pos=info["pos"][0].numpy(),
        speed=info["vel"][0].norm(dim=-1).numpy(),
        veh=(info["is_collision_with_agents"][0, ..., 0] > 0).numpy(),
        lane=(info["is_collision_with_lanelets"][0, ..., 0] > 0).numpy(),
        deadlock=(info["is_deadlocked"][0, ..., 0] > 0).numpy(),
    )


def in_view(data: dict, a: int, b: int, view: tuple) -> np.ndarray:
    x0, x1, y0, y1 = view
    p = data["pos"][a : b + 1]
    return (p[..., 0] > x0) & (p[..., 0] < x1) & (p[..., 1] > y0) & (p[..., 1] < y1)


def longest_stall(data: dict, a: int, b: int, view: tuple) -> float:
    """Longest standstill of a vehicle in view in steps a..b [s]."""
    slow = (data["speed"][a : b + 1] < SLOW) & in_view(data, a, b, view)
    longest = 0
    for k in range(slow.shape[1]):
        run = 0
        for flag in slow[:, k]:
            run = run + 1 if flag else 0
            longest = max(longest, run)
    return longest * 0.1


def is_deadlocked(data: dict, a: int, b: int, view: tuple, stall: float) -> bool:
    return (
        bool(data["deadlock"][a : b + 1].any())
        or longest_stall(data, a, b, view) >= stall
    )


def collisions(data: dict, a: int, b: int) -> tuple:
    """(collisions, of which with the road boundary) in steps a..b."""

    def onsets(flags):
        prev = (
            flags[a - 1 : b]
            if a > 0
            else np.vstack([np.zeros_like(flags[:1]), flags[a:b]])
        )
        return int((flags[a : b + 1] & ~prev).sum())

    lane = onsets(data["lane"])
    return int(np.ceil(onsets(data["veh"]) / 2)) + lane, lane


def stage1_runs():
    """(policy label, map, seed) of the Stage 1 runs."""
    for label in sorted(os.listdir(STAGE1)):
        if label.endswith("__iter100"):
            for seed in SEEDS:
                yield label, label.split("__")[0], seed


def rewards(args):
    rows = []
    for scenario in VIEW:
        for train_seed in range(1, 6):
            for seed in SEEDS:
                runs = {
                    reward: load(f"{REWARDS}/{scenario}__stage1_{reward}_seed_{train_seed}"
                                 f"/{scenario}/unfiltered/cen/seed{seed}")
                    for reward in ("ttc", "onestep", "bh")
                }  # fmt: skip
                for steps in args.steps:
                    if any(is_deadlocked(d, 0, steps, VIEW[scenario], args.stall)
                           for d in runs.values()):  # fmt: skip
                        continue
                    count = {k: collisions(d, 0, steps)[0] for k, d in runs.items()}
                    if count["bh"] == 0 and min(count["ttc"], count["onestep"]) >= 1:
                        rows.append((min(count["ttc"], count["onestep"]), count["ttc"] + count["onestep"],
                                     count["ttc"], count["onestep"], scenario, train_seed, seed, steps))  # fmt: skip
    print("TTC | one-step | map | training seed | seed | steps")
    for *_, ttc, onestep, scenario, train_seed, seed, steps in sorted(rows, reverse=True)[: args.top]:
        print(f"{ttc} | {onestep} | {scenario} | {train_seed} | {seed} | 0-{steps}")


def comparison(args):
    rows = []
    for label, scenario, seed in stage1_runs():
        base = os.path.join(STAGE1, label, scenario)
        runs = {
            "none": load(f"{base}/unfiltered/cen/seed{seed}"),
            "one_step": load(f"{base}/ours_onestep/dec/seed{seed}"),
            "bh_cbf": load(f"{base}/ours/dec/seed{seed}"),
        }
        if any(
            is_deadlocked(d, 0, args.steps, VIEW[scenario], args.stall)
            for d in runs.values()
        ):
            continue
        count = {k: collisions(d, 0, args.steps) for k, d in runs.items()}
        if count["bh_cbf"][0] == 0 and 1 <= count["one_step"][0] <= count["none"][0]:
            rows.append(
                (
                    count["one_step"][0],
                    count["none"][0],
                    count["one_step"][1],
                    label,
                    seed,
                )
            )
    print("one-step collisions (lane) | no filter | policy | seed")
    for one_step, none, lane, label, seed in sorted(rows, reverse=True)[: args.top]:
        print(f"{one_step} ({lane}) | {none} | {label} | {seed}")


def backup(args):
    rows = []
    for label, scenario, seed in stage1_runs():
        if scenario != args.map:
            continue
        data = load(os.path.join(STAGE1, label, scenario, "ours", "dec", f"seed{seed}"))
        for a in range(0, len(data["pos"]) - args.steps, 10):
            b = a + args.steps
            if is_deadlocked(data, a, b, VIEW[scenario], args.stall):
                continue
            moving = (
                (
                    in_view(data, a, b, VIEW[scenario])
                    & (data["speed"][a : b + 1] > MOVING)
                )
                .sum(1)
                .mean()
            )
            rows.append((float(moving), label, seed, a, b))
    print("moving vehicles in view | policy | seed | steps")
    for moving, label, seed, a, b in sorted(rows, reverse=True)[: args.top]:
        print(f"{moving:.2f} | {label} | {seed} | {a}-{b}")


def stage2(args):
    view = VIEW["ind_location4"]
    for path in sorted(
        glob.glob(os.path.join(args.run, "out_td_iter0_nofilter_seed*.td"))
    ):
        seed = path.split("seed")[-1].removesuffix(".td")
        before, after = load(path), load(path.replace("iter0_", "iter100_"))
        for length in args.lengths:
            cells = []
            for data in (before, after):
                stall = longest_stall(data, 0, length, view)
                flag = (
                    "deadlock"
                    if is_deadlocked(data, 0, length, view, args.stall)
                    else "ok"
                )
                cells.append(
                    f"{collisions(data, 0, length)[0]} collisions, standstill {stall:.1f} s ({flag})"
                )
            print(
                f"seed {seed}, steps 0-{length}: before: {cells[0]} | after: {cells[1]}"
            )


def main():
    args = parse_args()
    {"rewards": rewards, "comparison": comparison, "backup": backup, "stage2": stage2}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
