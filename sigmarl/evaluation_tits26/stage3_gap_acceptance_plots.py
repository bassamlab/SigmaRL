# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Analysis and figures of the gap acceptance study (stage3_gap_acceptance.py) at the rounD roundabout:
recorded humans vs. the ten fine-tuned policies with the decentralized BH-CBF filter.

Acceptance curves are logistic fits of P(accept | x) over the lag and gap events of entering vehicles that
are not queued behind a vehicle of their own lane and whose circulating vehicle does not (nearly) stand; the
TTA and the arc distance are fitted in their logarithm. The critical gap is the x at P = 0.5. Confidence
intervals are cluster bootstraps over the windows (humans) or the windows of all policies (policies).

Writes to <out>/: fig_*.pdf/png and summary.json.

Usage (from the repository root):
    python -m sigmarl.evaluation_tits26.stage3_gap_acceptance_plots
"""

import argparse
import csv
import json
import math
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from sigmarl.evaluation_tits26.paper_fig_headline import RWTH_PALETTE

OUT = "outputs/tits26/stage3_gap_acceptance"
# (full, tint): RWTH blue for our policies as in the paper figures, the human reference in neutral dark gray
# (red marks collisions in the paper; green, purple, magenta, and black mark the baselines)
HUMAN = ("#404040", "#BFBFBF")
POLICY = RWTH_PALETTE[0]
FALSE = RWTH_PALETTE[3]  # Green
GROUPS = {"human": "Humans", "policy": "BH-CBF (ours)"}
N_BOOT = 500
SCENE = ("rec05_h0639.6", 433)  # Episode and entering vehicle of the drawn human lag decision (paper figure)
MIN_SPEED_J = 1.0  # [m/s] Decisions whose circulating vehicle averages less are excluded: it blocks the entry
LOG_FIT = {"tta", "arc"}  # Fitted in the logarithm (log-normal critical gap); the time margin can be negative
# Acceptance panels: x label, curve grid, bin edges of the data points, unit
PANELS = {
    "tta": ("Time to arrival of the circulating vehicle [s]", np.linspace(0.02, 8, 400),
            np.r_[0:3:0.25, 3:6:0.5, 6:9], "s"),
    "arc": ("Arc distance to the conflict point [m]", np.linspace(0.1, 70, 400), np.r_[0:30:2.5, 30:72:5], "m"),
    "margin": ("Time margin [s]", np.linspace(-3, 6, 181), np.r_[-3:-1:1, -1:3:0.25, 3:7:1], "s"),
}  # fmt: skip
ACC_COMFORT = 2.0  # [m/s^2] Acceleration of the entering vehicle in its time to the conflict point
BRAKE = 6.0  # [m/s^2] Braking of a car in the braking-distance reference
CAR_LENGTH = 4.6  # [m]
FALSE_FREE_TTA = 6.0  # [s] False-conflict events: no true conflict closer than this TTA
WAIT_THRESHOLD = 4.0  # [s] Stop attribution: conflicts within this TTA


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=OUT)
    return parser.parse_args()


def load(path) -> dict:
    """CSV as a dict of arrays (numeric where possible, booleans as bool)."""
    with open(path) as file:
        rows = list(csv.DictReader(file))
    table = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        if all(v in ("True", "False", "") for v in values):
            table[key] = np.array([v == "True" for v in values])
            continue
        try:
            table[key] = np.array([float(v) if v != "" else np.nan for v in values])
        except ValueError:
            table[key] = np.array(values)
    table["group"] = np.where(table["world"] == "human", "human", "policy")
    table["cluster"] = np.char.add(np.char.add(table["world"], "/"), table["episode"])
    return table


def select(table, mask) -> dict:
    return {k: v[mask] for k, v in table.items()}


def own_time(dist, speed, acc: float = ACC_COMFORT):
    """Time [s] of the entering vehicle to the conflict point, accelerating from its speed at `acc`."""
    return (-speed + np.sqrt(speed**2 + 2 * acc * dist)) / acc


def fit_logit(x, y, iterations: int = 50):
    """Logistic regression P(y | x) = 1 / (1 + exp(-(b0 + b1 x))) by Newton's method; returns (b0, b1)."""
    X = np.stack([np.ones_like(x), x], axis=1)
    beta = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = p * (1 - p)
        hessian = X.T @ (X * w[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hessian, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def bootstrap(x, y, clusters, grid, rng, n_boot: int = N_BOOT, log: bool = False):
    """Fit, critical value, and 95% bands of the curve and the critical value (cluster bootstrap); with
    `log`, the fit is in log(x) and only x > 0 is used."""
    if log:
        keep = x > 0
        x, y, clusters = np.log(x[keep]), y[keep], clusters[keep]
        grid = np.log(grid)
    back = np.exp if log else (lambda z: z)
    beta = fit_logit(x, y)
    groups = {}
    for idx, c in enumerate(clusters):
        groups.setdefault(c, []).append(idx)
    keys = list(groups)
    curves, criticals = [], []
    for _ in range(n_boot):
        pick = rng.integers(len(keys), size=len(keys))
        idx = np.concatenate([groups[keys[p]] for p in pick])
        b = fit_logit(x[idx], y[idx])
        curves.append(1 / (1 + np.exp(-(b[0] + b[1] * grid))))
        criticals.append(-b[0] / b[1])
    curves = np.array(curves)
    return dict(
        beta=beta.tolist(),
        critical=float(back(-beta[0] / beta[1])),
        critical_ci=back(np.percentile(criticals, [2.5, 97.5])).tolist(),
        curve=1 / (1 + np.exp(-(beta[0] + beta[1] * grid))),
        band=np.percentile(curves, [2.5, 97.5], axis=0),
        n=int(len(x)),
    )


def wilson(k, n, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, center - half, center + half


def binned(x, y, edges, min_n: int = 8):
    """Bin centers (mean x), proportions, and Wilson intervals of y over x bins."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_n:
            p, a, b = wilson(int(y[m].sum()), int(m.sum()))
            out.append((float(x[m].mean()), p, a, b, int(m.sum())))
    return np.array(out).reshape(-1, 5)


def style_axes(ax):
    ax.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved {name}")


def decisions(events):
    """Lag and gap events with a true conflict of entering vehicles that are not queued, without those whose
    circulating vehicle (nearly) stands: its TTA diverges although it blocks the entry."""
    return select(events, events["has_conflict"] & ~events["queued"] & (events["va_j"] >= MIN_SPEED_J))


def acceptance_panel(ax, d, key, rng, summary=None, legend: bool = True):
    """P(accept) over `key` (a column of `d`, see PANELS): pooled fits with 95% bands and critical values
    (dotted), binned data with Wilson intervals, and thin lines for the single policies. Paper style.
    Returns the critical values per group."""
    label, grid, edges, unit = PANELS[key]
    log = key in LOG_FIT
    criticals = {}
    for group, (color, tint) in (("human", HUMAN), ("policy", POLICY)):
        g = select(d, d["group"] == group)
        ok = np.isfinite(g[key]) & ((g[key] > 0) if log else True)
        x, y, c = g[key][ok], g["accept"][ok].astype(float), g["cluster"][ok]
        if group == "policy":
            for world in sorted(set(g["world"])):
                m = g["world"][ok] == world
                b = fit_logit(np.log(x[m]) if log else x[m], y[m])
                ax.plot(grid, 1 / (1 + np.exp(-(b[0] + b[1] * (np.log(grid) if log else grid)))), color=tint,
                        lw=0.4, alpha=0.8, zorder=1)  # fmt: skip
                if summary is not None:
                    summary["critical"].setdefault(key, {}).setdefault("per_policy", {})[world] = float(
                        np.exp(-b[0] / b[1]) if log else -b[0] / b[1])  # fmt: skip
        fit = bootstrap(x, y, c, grid, rng, log=log)
        criticals[group] = fit["critical"]
        if summary is not None:
            summary["critical"].setdefault(key, {})[group] = {k: fit[k] for k in ("critical", "critical_ci", "beta", "n")}
        critical = f"{fit['critical']:.2f}".replace("-", "\u2212")
        ax.fill_between(grid, *fit["band"], color=color, alpha=0.15, lw=0, zorder=2)
        ax.plot(grid, fit["curve"], color=color, lw=1.0, zorder=3, label=f"{GROUPS[group]}: {critical} {unit}")
        ax.vlines(fit["critical"], -0.02, 0.5, color=color, lw=0.5, ls=":", zorder=1)
        bins = binned(x, y, edges)
        ax.errorbar(bins[:, 0], bins[:, 1], yerr=np.clip([bins[:, 1] - bins[:, 2], bins[:, 3] - bins[:, 1]], 0, None),
                    fmt="o", ms=1.8, mew=0, color=color, elinewidth=0.5, capsize=0, zorder=4)  # fmt: skip
    ax.axhline(0.5, color="0.6", lw=0.4, ls=":", zorder=0)
    ax.set_xlim(min(0, grid[0]), grid[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticks([0.25, 0.75], minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_xlabel(label, labelpad=1)
    ax.grid(axis="y", which="both", color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        ax.legend(loc="lower right", frameon=False, fontsize=6.5, handlelength=1.8, borderaxespad=0.2,
                  labelspacing=0.25)  # fmt: skip
    return criticals


def acceptance_figure(events, out, rng, summary):
    """P(accept) over the TTA, the arc distance, and the time margin of the circulating vehicle."""
    d = decisions(events)
    d["margin"] = d["tta"] - own_time(d["dist_i"], d["v_i"])
    summary.setdefault("critical", {})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.4))
    for ax, key in zip(axes, PANELS):
        acceptance_panel(ax, d, key, rng, summary)
    axes[0].set_ylabel("Probability of entering first")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.17, top=0.97, wspace=0.18)
    save(fig, out, "fig_acceptance")


def lag_gap_figure(events, out, rng, summary):
    """Acceptance over the TTA separately for the first decision (lag) and the later ones (gaps), and per
    entry arm for the humans."""
    d = decisions(events)
    grid = np.linspace(0.02, 8, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for group, (color, tint) in (("human", HUMAN), ("policy", POLICY)):
        for kind, ls in (("lag", "-"), ("gap", "--")):
            g = select(d, (d["group"] == group) & (d["kind"] == kind))
            fit = bootstrap(g["tta"], g["accept"].astype(float), g["cluster"], grid, rng, log=True)
            summary.setdefault("critical_by_kind", {}).setdefault(group, {})[kind] = {
                k: fit[k] for k in ("critical", "critical_ci", "n")
            }
            axes[0].plot(grid, fit["curve"], color=color, ls=ls, lw=1.3,
                         label=f"{GROUPS[group]}, {kind}s: {fit['critical']:.2f} s (n={fit['n']})")  # fmt: skip
    # Entry arms by the merge angle of the route (humans)
    with open(os.path.join(out, "geometry.json")) as file:
        geometry = json.load(file)
    arms = {"W": (150, 200), "S": (-120, -80), "E": (-20, 30), "N": (60, 100)}

    def arm(route):
        angle = math.degrees(geometry[str(int(route))]["theta_merge"]) % 360
        for name, (lo, hi) in arms.items():
            if lo % 360 <= angle <= hi % 360 or (lo < 0 and (angle >= lo % 360 or angle <= hi)):
                return name
        return "?"

    h = select(d, d["group"] == "human")
    arm_of = np.array([arm(r) for r in h["route_i"]])
    for index, name in enumerate(arms):
        g = select(h, arm_of == name)
        fit = bootstrap(g["tta"], g["accept"].astype(float), g["cluster"], grid, rng, n_boot=200, log=True)
        summary.setdefault("critical_by_arm_human", {})[name] = {
            k: fit[k] for k in ("critical", "critical_ci", "n")
        }
        axes[1].plot(grid, fit["curve"], color=RWTH_PALETTE[index][0], lw=1.2,
                     label=f"{name} entry: {fit['critical']:.2f} s (n={fit['n']})")  # fmt: skip
    axes[0].set_ylabel("Probability of entering first")
    for ax in axes:
        ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
        ax.set_xlabel("TTA of the circulating vehicle [s]")
        ax.set_xlim(0, 8)
        style_axes(ax)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig_acceptance_lag_gap_arm")


def cost_figure(events, out, summary):
    """What an accepted entry costs the circulating vehicle: its speed loss until it reaches the conflict
    point (rejected decisions as the baseline), the TTA of the accepted entries, and their post-encroachment
    time (PET)."""
    d = decisions(events)
    edges = np.r_[0.5:3:0.5, 3:9:1]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for group, (color, tint) in (("human", HUMAN), ("policy", POLICY)):
        g = select(d, d["group"] == group)
        for accept, ls, marker in ((True, "-", "o"), (False, ":", "s")):
            m = g["accept"] == accept
            x = g["tta"][m]
            for ax, values in ((axes[0], g["speed_loss_j"][m]),):
                rows = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    b = (x >= lo) & (x < hi) & np.isfinite(values)
                    if b.sum() >= 8:
                        rows.append((x[b].mean(), *np.percentile(values[b], [25, 50, 75])))
                rows = np.array(rows)
                if not len(rows):
                    continue
                ax.plot(rows[:, 0], rows[:, 2], ls=ls, marker=marker, ms=3, color=color, lw=1.1,
                        label=f"{GROUPS[group]}, {'accepted' if accept else 'rejected'}")  # fmt: skip
                if accept:
                    ax.fill_between(rows[:, 0], rows[:, 1], rows[:, 3], color=tint, alpha=0.6, lw=0)
        a = select(g, g["accept"] & np.isfinite(g["t_arr_j"]))
        axes[1].hist(a["tta"], bins=np.arange(0, 8.01, 0.25), density=True, histtype="step", color=color,
                     lw=1.3, label=GROUPS[group])  # fmt: skip
        pet = a["t_arr_j"] - a["t_arr_i"]
        axes[2].plot(np.sort(pet), np.linspace(0, 1, len(pet)), color=color, lw=1.3, label=GROUPS[group])
        loss = a["speed_loss_j"]
        short = a["tta"] < 2.0
        summary.setdefault("cost", {})[group] = dict(
            accepted=int(len(a["tta"])),
            share_speed_loss_gt2=float(np.mean(loss > 2.0)),
            share_speed_loss_gt2_tta_lt2=float(np.mean(loss[short] > 2.0)) if short.any() else None,
            share_accepted_tta_lt2=float(np.mean(short)),
            median_pet=float(np.median(pet)),
            share_pet_lt1=float(np.mean(pet < 1.0)),
            median_delay_j_tta_lt2=float(np.median((a["t_arr_j"] - a["tta"])[short])) if short.any() else None,
        )
    axes[0].set_ylabel("Speed loss of the circulating vehicle [m/s]")
    axes[1].set_ylabel("Density of accepted entries [1/s]")
    for ax in axes[:2]:
        ax.set_xlabel("TTA of the circulating vehicle [s]")
        ax.set_xlim(0, 8)
    axes[2].set_xlabel("PET of accepted entries [s]")
    axes[2].set_ylabel("Cumulative share")
    axes[2].set_xlim(0, 8)
    for ax in axes:
        style_axes(ax)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "fig_cost_circulating")


def information_figure(events, waits, out, summary):
    """True vs. false conflicts: minimum speed of the entering vehicle until it merges, and what its
    waiting time is spent on."""
    lag = select(events, (events["kind"] == "lag") & ~events["queued"])
    true = lag["has_conflict"] & ~lag["f_closer"]
    false = (lag["f"] == lag["f"]) & np.isfinite(lag["tta_f"]) & (
        ~lag["has_conflict"] | (lag["f_closer"] & (lag["tta"] >= FALSE_FREE_TTA)))  # fmt: skip
    free = ~lag["has_conflict"] & ~np.isfinite(lag["tta_f"])
    edges = np.r_[0:3:0.5, 3:9:1]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    for group, (color, tint) in (("human", HUMAN), ("policy", POLICY)):
        is_group = lag["group"] == group
        for mask, x_key, ls, marker, name in ((true, "tta", "-", "o", "true conflict"),
                                              (false, "tta_f", "--", "^", "false conflict (exits before)")):  # fmt: skip
            m = mask & is_group
            x, v = lag[x_key][m], lag["min_v_i"][m]
            rows = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = (x >= lo) & (x < hi)
                if b.sum() >= 8:
                    rows.append((x[b].mean(), *np.percentile(v[b], [25, 50, 75])))
            rows = np.array(rows)
            ax.plot(rows[:, 0], rows[:, 2], ls=ls, marker=marker, ms=3, color=color, lw=1.1,
                    label=f"{GROUPS[group]}, {name}")  # fmt: skip
        v_free = lag["min_v_i"][free & is_group]
        ax.axhspan(*np.percentile(v_free, [25, 75]), color=tint, alpha=0.5, lw=0)
        ax.axhline(np.median(v_free), color=color, lw=0.8, ls=":")
        summary.setdefault("information", {})[group] = dict(
            n_true=int((true & is_group).sum()), n_false=int((false & is_group).sum()),
            n_free=int((free & is_group).sum()),
            median_min_speed_free=float(np.median(v_free)),
            stop_share_true_tta_1_4=float(np.mean(lag["stops_i"][true & is_group & (lag["tta"] >= 1) & (lag["tta"] < 4)])),
            stop_share_false_tta_1_4=float(np.mean(lag["stops_i"][false & is_group & (lag["tta_f"] >= 1) & (lag["tta_f"] < 4)])),
        )  # fmt: skip
    ax.set_xlabel("TTA of the nearest circulating vehicle [s]\n(false conflict: as if it did not exit)")
    ax.set_ylabel("Min. speed of the entering vehicle [m/s]")
    ax.set_xlim(0, 8)
    style_axes(ax)
    ax.legend(fontsize=6.3, loc="lower right")

    # Waiting time by cause (per world; policies averaged over the ten)
    ax = axes[1]
    causes = ["queued", "true conflict", "false conflict only", "none"]
    colors = ["0.65", HUMAN[0], FALSE[0], "0.85"]
    totals = {}
    for group in ("human", "policy"):
        w = select(waits, waits["group"] == group)
        true_w = ~w["queued"] & (w["tta_true"] <= WAIT_THRESHOLD)
        false_w = ~w["queued"] & ~true_w & (w["tta_false"] <= WAIT_THRESHOLD)
        none_w = ~w["queued"] & ~true_w & ~false_w
        n_worlds = len(set(w["world"]))
        totals[group] = [m.sum() * 0.1 / n_worlds for m in (w["queued"], true_w, false_w, none_w)]
    bottom = np.zeros(2)
    for index, cause in enumerate(causes):
        values = np.array([totals["human"][index], totals["policy"][index]])
        ax.bar([0, 1], values, bottom=bottom, color=colors[index], width=0.6, label=cause)
        bottom += values
    for x, total in enumerate(bottom):
        ax.text(x, total + 20, f"{total:.0f} s", ha="center", fontsize=7)
    summary["waiting_time_s"] = {g: dict(zip(causes, map(float, v))) for g, v in totals.items()}
    ax.set_xticks([0, 1], ["Humans", "Policies\n(mean of 10)"])
    ax.set_ylabel(f"Waiting time before the entry [s]")
    style_axes(ax)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "fig_information")


def bound_figure(events, out):
    """Accepted and rejected decisions over the speed and the arc distance of the circulating vehicle, with
    its braking distance as a physical reference."""
    d = decisions(events)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
    speed = np.linspace(0, 14, 100)
    for ax, group in zip(axes, ("human", "policy")):
        g = select(d, d["group"] == group)
        idx = rng.permutation(len(g["arc"]))[:1500]
        for accept, color, marker in ((False, "0.55", "x"), (True, (HUMAN if group == "human" else POLICY)[0], "o")):
            m = idx[g["accept"][idx] == accept]
            ax.scatter(g["v_j"][m], g["arc"][m], s=4, marker=marker, color=color, lw=0.5,
                       label="accepted" if accept else "rejected", alpha=0.7)  # fmt: skip
        ax.plot(speed, speed**2 / (2 * BRAKE), color="k", lw=1.0, label="braking distance (6 m/s²)")
        ax.plot(speed, speed**2 / (2 * BRAKE) + CAR_LENGTH, color="k", lw=1.0, ls="--",
                label="braking distance + car length")  # fmt: skip
        ax.set_title(GROUPS[group], fontsize=9)
        ax.set_xlabel("Speed of the circulating vehicle [m/s]")
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 60)
        style_axes(ax)
    axes[0].set_ylabel("Arc distance to the conflict point [m]")
    axes[1].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_speed_arc")


def trips_figure(trips, out, summary):
    """Times before (first position to merge point) and after (merge point to last position) the merge point of
    the same evaluated trips (humans vs. policies)."""
    human = {(e, i): (a, r) for w, e, i, a, r in zip(trips["world"], trips["episode"], trips["i"],
             trips["before_time"], trips["after_time"]) if w == "human"}  # fmt: skip
    diffs = {"before": [], "after": []}
    pol = {"before": [], "after": []}
    for w, e, i, a, r in zip(trips["world"], trips["episode"], trips["i"], trips["before_time"], trips["after_time"]):
        if w == "human" or (e, i) not in human:
            continue
        diffs["before"].append(a - human[(e, i)][0])
        diffs["after"].append(r - human[(e, i)][1])
        pol["before"].append(a)
        pol["after"].append(r)
    hum = {"before": [v[0] for v in human.values()], "after": [v[1] for v in human.values()]}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, phase, label in ((axes[0], "before", "Time before the merge point [s]"),
                             (axes[1], "after", "Time after the merge point [s]")):  # fmt: skip
        for values, (color, _), name in ((hum[phase], HUMAN, "Humans"), (pol[phase], POLICY, "Policies (BH-CBF)")):
            values = np.sort(values)
            ax.plot(values, np.linspace(0, 1, len(values)), color=color, lw=1.3,
                    label=f"{name}: median {np.median(values):.2f} s, mean {np.mean(values):.2f} s")  # fmt: skip
        ax.set_xlabel(label)
        ax.set_xlim(0, 15 if phase == "before" else 25)
        style_axes(ax)
        ax.legend(fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("Cumulative share of trips")
    fig.tight_layout()
    save(fig, out, "fig_trips")
    summary["trips"] = dict(n_human=len(human), n_pairs=len(diffs["before"]))
    for phase in ("before", "after"):
        for name, values in (("human", hum[phase]), ("policy", pol[phase]), ("diff", diffs[phase])):
            summary["trips"][f"{name}_{phase}_median"] = float(np.median(values))
            summary["trips"][f"{name}_{phase}_mean"] = float(np.mean(values))


def sensitivity_figure(out, rng, summary):
    """Critical values of humans and policies over the decision distance (TTA of the first decision, TTA of
    the later ones, time margin) and over the speed averaging window (TTA of all decisions)."""
    d_ref = [(5, "events_d5_w1.csv"), (10, "events_d10_w1.csv"), (11.3, "events.csv"), (15, "events_d15_w1.csv"),
             (20, "events_d20_w1.csv")]  # fmt: skip
    window = [(0.1, "events_d11.3_w0.1.csv"), (0.5, "events_d11.3_w0.5.csv"), (1.0, "events.csv"),
              (2.0, "events_d11.3_w2.csv")]  # fmt: skip
    panels = [
        ("lag_tta", d_ref, "lag", "tta", "Critical TTA, first decision [s]"),
        ("gap_tta", d_ref, "gap", "tta", "Critical TTA, later decisions [s]"),
        ("margin", d_ref, None, "margin", "Critical time margin [s]"),
        ("speed_window", window, None, "tta", "Critical TTA, all decisions [s]"),
    ]
    cache = {}
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.7))
    for ax, (name, files, kind, key, ylabel) in zip(axes, panels):
        for group, (color, _) in (("human", HUMAN), ("policy", POLICY)):
            rows = []
            for value, file in files:
                if file not in cache:
                    cache[file] = decisions(load(os.path.join(out, file)))
                    cache[file]["margin"] = cache[file]["tta"] - own_time(cache[file]["dist_i"], cache[file]["v_i"])
                d = cache[file]
                m = (d["group"] == group) & np.isfinite(d[key])
                if kind is not None:
                    m &= d["kind"] == kind
                g = select(d, m)
                fit = bootstrap(g[key], g["accept"].astype(float), g["cluster"], np.ones(1), rng, n_boot=200,
                                log=key in LOG_FIT)
                rows.append((value, fit["critical"], *fit["critical_ci"]))
                summary.setdefault("sensitivity", {}).setdefault(name, {}).setdefault(group, []).append(
                    dict(value=value, critical=fit["critical"], ci=fit["critical_ci"], n=fit["n"]))  # fmt: skip
            rows = np.array(rows)
            ax.errorbar(rows[:, 0], rows[:, 1], yerr=np.clip([rows[:, 1] - rows[:, 2], rows[:, 3] - rows[:, 1]], 0, None),
                        marker="o", ms=3, color=color, lw=1.1, capsize=2, label=GROUPS[group])  # fmt: skip
        ax.set_xlabel("Speed averaging window [s]" if name == "speed_window"
                      else "Decision distance before the merge point [m]")  # fmt: skip
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="0.6", lw=0.5)
        style_axes(ax)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_sensitivity")


def draw_definition_scene(ax, events, out, chevron_lw: float = 1.0, star_size: float = 12, arc_lw: float = 2.5):
    """Draw a recorded human lag decision without other vehicles in the gap: the road, the entering vehicle
    (blue), the circulating vehicle (green), each with a heading chevron, the estimated merge point on the ring
    centerline, the arc distance, and the geometry behind the merge point: the reference route of the entering
    vehicle, the outer ring-lane and the ring centerline circle, and the ray at theta* (used by
    paper_fig_stage4_gap_acceptance). Returns the poses [x, y, yaw] of both vehicles, the merge point, the
    ring angles of the arc, the event, and the route points of i."""
    from sigmarl.evaluation_human_comparison.metrics import MERGE_RADIUS, RING_CENTER
    from sigmarl.evaluation_tits26.paper_fig_headline import heading_chevron, rectangle
    from sigmarl.evaluation_tits26.paper_fig_stage1_footprints import draw_road
    from sigmarl.evaluation_tits26.stage3_gap_acceptance import RESULTS, RING_RADIUS
    from sigmarl.parse_lanelet2 import ParseLanelet2
    from sigmarl.round_tracks import RouteMatcher

    with open(os.path.join(out, "geometry.json")) as file:
        geometry = json.load(file)
    lag = select(events, (events["group"] == "human") & (events["kind"] == "lag") & events["has_conflict"]
                 & ~events["queued"] & ~events["f_closer"] & (events["arc"] > 25) & (events["arc"] < 45))  # fmt: skip
    pick = int(np.flatnonzero((lag["episode"] == SCENE[0]) & (lag["i"] == SCENE[1]))[0])
    event = {k: v[pick] for k, v in lag.items()}
    with open(os.path.join(RESULTS, "natural", "matched_tracks.pkl"), "rb") as file:
        data = pickle.load(file)
    tracks = {tr.track_id: tr for tr in data[int(event["episode"][3:5])].tracks}
    draw_road(ax, "round_location0")
    theta_star = geometry[str(int(event["route_i"]))]["theta_merge"]
    # Hidden geometry of the merge point: the reference route of i, the outer ring-lane circle that the route
    # crosses at the merge point, the ring centerline circle, and the ray from the ring center at theta*
    route = RouteMatcher(ParseLanelet2("round_location0", "cpu").reference_paths).routes[int(event["route_i"])]
    route_xy = route.xy_at(np.arange(0.0, route.length, 0.1))
    ax.plot(*route_xy.T, color=RWTH_PALETTE[0][0], lw=0.5, ls=(0, (3, 2)), zorder=2.6)
    circle = np.linspace(0, 2 * np.pi, 400)
    for radius, style in ((MERGE_RADIUS, dict(lw=0.4, ls=(0, (4, 1.5, 1, 1.5)))), (RING_RADIUS, dict(lw=0.7, ls=(0, (3, 2))))):
        ax.plot(RING_CENTER[0] + radius * np.cos(circle), RING_CENTER[1] + radius * np.sin(circle), color="k",
                zorder=2.5, **style)  # fmt: skip
    ax.plot(*np.c_[RING_CENTER, RING_CENTER + MERGE_RADIUS * np.array([math.cos(theta_star), math.sin(theta_star)])],
            color="k", lw=0.4, ls=(0, (2, 1.5)), zorder=2.5)  # fmt: skip
    ax.plot(*RING_CENTER, marker="o", ms=2.5, color="k", mew=0, zorder=2.5)
    poses = {}
    for name, color in (("i", RWTH_PALETTE[0][0]), ("j", RWTH_PALETTE[3][0])):
        track = tracks[int(event[name])]
        pose = np.array(track.state_at(event["t"])[:3])
        half = np.array([track.length, track.width]) / 2
        ax.add_patch(Polygon(rectangle(pose, half), fc=color, ec="none", zorder=5))
        ax.plot(*heading_chevron(pose, half).T, color="white", lw=chevron_lw, solid_joinstyle="miter",
                solid_capstyle="butt", zorder=5.5)  # fmt: skip
        poses[name] = pose
    theta_j = math.atan2(poses["j"][1] - RING_CENTER[1], poses["j"][0] - RING_CENTER[0])
    arc = np.linspace(theta_j, theta_j + (theta_star - theta_j) % (2 * np.pi), 80)
    ax.plot(RING_CENTER[0] + RING_RADIUS * np.cos(arc), RING_CENTER[1] + RING_RADIUS * np.sin(arc),
            color=RWTH_PALETTE[3][0], lw=arc_lw, alpha=0.5, solid_capstyle="butt", zorder=3)  # fmt: skip
    point = RING_CENTER + RING_RADIUS * np.array([math.cos(theta_star), math.sin(theta_star)])
    ax.plot(*point, marker="*", ms=star_size, color="k", mec="w", mew=0.4, zorder=6, ls="none")
    ax.set_aspect("equal")
    ax.axis("off")
    return dict(poses=poses, point=point, arc=arc, event=event, route=route_xy)


def definition_figure(events, out):
    """A recorded decision: the entering vehicle at its decision point, the nearest true and false
    conflict, the conflict point, and the arc distance along the ring."""
    from sigmarl.evaluation_human_comparison.metrics import MERGE_RADIUS, RING_CENTER
    from sigmarl.evaluation_tits26.paper_fig_headline import rectangle
    from sigmarl.evaluation_tits26.paper_fig_stage1_footprints import draw_road
    from sigmarl.evaluation_tits26.stage3_gap_acceptance import RESULTS, RING_RADIUS

    with open(os.path.join(out, "geometry.json")) as file:
        geometry = json.load(file)
    # A human lag decision with a false conflict closer than the true one, both at moderate distance
    lag = select(events, (events["group"] == "human") & (events["kind"] == "lag") & events["has_conflict"]
                 & events["f_closer"] & ~events["queued"] & (events["arc"] > 25) & (events["arc"] < 45)
                 & (events["arc_f"] > 10) & (events["arc_f"] < 22))  # fmt: skip
    pick = int(np.argmin(np.abs(lag["tta"] - 3.5)))
    episode, t, keys = lag["episode"][pick], lag["t"][pick], (lag["i"][pick], lag["j"][pick], lag["f"][pick])
    with open(os.path.join(RESULTS, "natural", "matched_tracks.pkl"), "rb") as file:
        data = pickle.load(file)
    rec_id = int(episode[3:5])
    tracks = {tr.track_id: tr for tr in data[rec_id].tracks}
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    draw_road(ax, "round_location0")
    theta_star = geometry[str(int(lag["route_i"][pick]))]["theta_merge"]
    colors = [RWTH_PALETTE[0][0], RWTH_PALETTE[5][0], FALSE[0]]
    names = ["entering vehicle $i$", "true conflict $j$", "false conflict (exits before)"]
    for key, color, name in zip(keys, colors, names):
        track = tracks[int(key)]
        x, y, yaw, _ = track.state_at(t)
        ax.add_patch(Polygon(rectangle(np.array([x, y, yaw]), np.array([track.length, track.width]) / 2),
                             fc=color, ec="k", lw=0.5, zorder=5, label=name))  # fmt: skip
        future = track.t[(track.t >= t) & (track.t <= t + 4)]
        path = np.array([track.state_at(s)[:2] for s in future])
        ax.plot(*path.T, color=color, lw=0.8, ls=":", zorder=4)
        if key == keys[1]:
            theta_j = math.atan2(y - RING_CENTER[1], x - RING_CENTER[0])
            arc = np.linspace(theta_j, theta_j + (theta_star - theta_j) % (2 * np.pi), 50)
            ax.plot(RING_CENTER[0] + RING_RADIUS * np.cos(arc), RING_CENTER[1] + RING_RADIUS * np.sin(arc),
                    color=color, lw=2.0, alpha=0.6, zorder=3, label=rf"arc distance = {RING_RADIUS} m $\times\,\Delta\theta$")  # fmt: skip
    point = RING_CENTER + MERGE_RADIUS * np.array([math.cos(theta_star), math.sin(theta_star)])
    ax.plot(*point, marker="*", ms=11, color="k", zorder=6, ls="none", label="conflict point (merge)")
    # Ring angles: theta = 0 along +x, theta* of the conflict point, and the angle from j to theta*
    def ray(theta, length, **kwargs):
        ax.plot(*np.c_[RING_CENTER, RING_CENTER + length * np.array([math.cos(theta), math.sin(theta)])],
                lw=0.6, zorder=3, **kwargs)  # fmt: skip

    def angle(start, stop, radius, text, color, offset=2.5):
        theta = np.linspace(start, stop, 50)
        ax.plot(RING_CENTER[0] + radius * np.cos(theta), RING_CENTER[1] + radius * np.sin(theta), color=color,
                lw=0.6, zorder=3)  # fmt: skip
        mid = 0.5 * (start + stop)
        ax.text(*(RING_CENTER + (radius + offset) * np.array([math.cos(mid), math.sin(mid)])), text, color=color,
                fontsize=8, ha="center", va="center", zorder=6)  # fmt: skip

    ax.plot(*RING_CENTER, marker="+", ms=5, color="k", zorder=6)
    ray(0.0, MERGE_RADIUS, color="0.4", ls="--")
    ray(theta_star, MERGE_RADIUS, color="k")
    angle(0.0, (theta_star + np.pi) % (2 * np.pi) - np.pi, 6.0, r"$\theta^*$", "k")
    ray(theta_j, RING_RADIUS, color=colors[1])
    angle(theta_j, theta_j + (theta_star - theta_j) % (2 * np.pi), 12.0, r"$\Delta\theta$", colors[1])
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(RING_CENTER[0] + MERGE_RADIUS * np.cos(circle), RING_CENTER[1] + MERGE_RADIUS * np.sin(circle),
            color="0.5", lw=0.5, ls="--", zorder=2)  # fmt: skip
    ax.set_xlim(RING_CENTER[0] - 40, RING_CENTER[0] + 40)
    ax.set_ylim(RING_CENTER[1] - 40, RING_CENTER[1] + 40)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title(f"{episode}, t = {t:.1f} s: TTA {lag['tta'][pick]:.1f} s, "
                 f"{'accepted' if lag['accept'][pick] else 'rejected'}", fontsize=8)  # fmt: skip
    fig.tight_layout()
    save(fig, out, "fig_definition")


def main():
    args = parse_args()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times"], "font.size": 8, "mathtext.fontset": "stix",
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })  # fmt: skip
    rng = np.random.default_rng(1)
    events = load(os.path.join(args.out, "events.csv"))
    waits = load(os.path.join(args.out, "waits.csv"))
    trips = load(os.path.join(args.out, "trips.csv"))
    summary = {}
    d = decisions(events)
    summary["counts"] = {
        group: dict(
            entering=int(np.sum((events["group"] == group) & (events["kind"] == "lag"))),
            decisions=int(np.sum(d["group"] == group)),
            lags=int(np.sum((d["group"] == group) & (d["kind"] == "lag"))),
            gaps=int(np.sum((d["group"] == group) & (d["kind"] == "gap"))),
            free_entries=int(np.sum((events["group"] == group) & (events["kind"] == "lag") & ~events["has_conflict"])),
            queued=int(np.sum((events["group"] == group) & events["queued"])),
        )
        for group in ("human", "policy")
    }  # fmt: skip
    definition_figure(events, args.out)
    acceptance_figure(events, args.out, rng, summary)
    lag_gap_figure(events, args.out, rng, summary)
    cost_figure(events, args.out, summary)
    information_figure(events, waits, args.out, summary)
    bound_figure(events, args.out)
    trips_figure(trips, args.out, summary)
    sensitivity_figure(args.out, rng, summary)
    with open(os.path.join(args.out, "summary.json"), "w") as file:
        json.dump(summary, file, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
