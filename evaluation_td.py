import os
import re
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import tensordict

# ---------------------------------------------------------------------
# TensorDict safe loading
# ---------------------------------------------------------------------
torch.serialization.add_safe_globals(
    [
        tensordict._reductions._make_td,
        tensordict._td.TensorDict,
    ]
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
FOLDER = "outputs/marl_cbf_2/"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLLISION_TYPES = ["agents", "lanelets", "total"]
COLLISION_XLABELS = ["Agent-Agent", "Agent-Lanelet", "Total"]

# ---------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------
TD_PATTERN = re.compile(
    r"out_td"
    r"_agents_(?P<agents>\d+)"
    r"_seed_(?P<seed>\d+)"
    r"_grouping_(?P<grouping>on|off)"
    r"_maxgroup_(?P<maxgroup>\d+)"
    r"_nom_(?P<nom>[^_]+)"
    r"_scenario_(?P<scenario>[^.]+)"
    r"\.td$"
)

TIME_PATTERN = re.compile(
    r"computation_t"
    r"_agents_(?P<agents>\d+)"
    r"_seed_(?P<seed>\d+)"
    r"_grouping_(?P<grouping>on|off)"
    r"_maxgroup_(?P<maxgroup>\d+)"
    r"_nom_(?P<nom>[^_]+)"
    r"_scenario_(?P<scenario>[^.]+)"
    r"\.json$"
)

# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def load_collision_rates(td_path: str) -> tuple[float, float, float, float]:
    """
    Returns:
        (agent-agent collision rate,
         agent-lanelet collision rate,
         total collision rate,
         average speed over all agents and steps)
    """
    out_td = torch.load(td_path, weights_only=True)

    is_collision_with_agents = out_td[
        "agents", "info", "is_collision_with_agents"
    ].bool()
    is_collision_with_lanelets = out_td[
        "agents", "info", "is_collision_with_lanelets"
    ].bool()

    is_collide = is_collision_with_agents | is_collision_with_lanelets
    num_steps = int(is_collision_with_agents.shape[1])

    aa = is_collision_with_agents.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
    al = is_collision_with_lanelets.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
    tot = is_collide.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps

    # ------------------------------
    # Average speed
    # ------------------------------
    speed = out_td["agents", "info", "vel"].norm(dim=-1)  # (1, num_steps, num_agents)
    avg_speed = speed.mean().item()

    return aa.item(), al.item(), tot.item(), avg_speed


def load_computation_time_per_step(json_path: str, grouping: str) -> list[float]:
    """
    - grouping == "off": json is list[float], one value per step
    - grouping == "on":  json is list[list[float]], each step has per-group times;
                         we average across groups per step -> list[float]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if grouping == "off":
        if not isinstance(obj, list) or (
            len(obj) > 0 and not isinstance(obj[0], (int, float))
        ):
            raise ValueError(f"Unexpected format for grouping=off in {json_path}")
        return [float(x) for x in obj]

    # grouping == "on"
    if not isinstance(obj, list):
        raise ValueError(f"Unexpected format for grouping=on in {json_path}")

    per_step_avg = []
    for t, step in enumerate(obj):
        if not isinstance(step, list) or len(step) == 0:
            # allow empty step, treat as NaN and drop later
            per_step_avg.append(float("nan"))
            continue
        per_step_avg.append(float(np.mean(step)))

    # drop NaNs (if any)
    return [x for x in per_step_avg if np.isfinite(x)]


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------
# collision_data[(agents, scenario, nom)][grouping][maxgroup][metric] -> list[float]
# metric ∈ {"agents", "lanelets", "total", "avg_speed"}
collision_data = defaultdict(
    lambda: {
        "off": defaultdict(lambda: defaultdict(list)),
        "on": defaultdict(lambda: defaultdict(list)),
    }
)


# time_data[(agents, scenario, nom)][grouping][maxgroup] -> list[float] pooled across seeds and steps
time_data = defaultdict(
    lambda: {
        "off": defaultdict(list),
        "on": defaultdict(list),
    }
)

# ---------------------------------------------------------------------
# Discover files
# ---------------------------------------------------------------------
td_paths = []
time_paths = []

for f in os.listdir(FOLDER):
    full = os.path.join(FOLDER, f)
    if f.endswith(".td"):
        td_paths.append(full)
    elif f.endswith(".json") and f.startswith("computation_t_"):
        time_paths.append(full)

td_paths.sort()
time_paths.sort()

# ---------------------------------------------------------------------
# Load collisions (.td)
# ---------------------------------------------------------------------
for path in td_paths:
    base = os.path.basename(path)
    m = TD_PATTERN.match(base)
    if m is None:
        raise ValueError(f"Filename does not match expected TD pattern: {base}")

    agents = m.group("agents")
    scenario = m.group("scenario")
    nom = m.group("nom")
    grouping = m.group("grouping")
    maxgroup = int(m.group("maxgroup"))

    aa, al, tot, avg_speed = load_collision_rates(path)
    key = (agents, scenario, nom)

    collision_data[key][grouping][maxgroup]["agents"].append(aa)
    collision_data[key][grouping][maxgroup]["lanelets"].append(al)
    collision_data[key][grouping][maxgroup]["total"].append(tot)
    collision_data[key][grouping][maxgroup]["avg_speed"].append(avg_speed)

# ---------------------------------------------------------------------
# Load computation time (.json)
# ---------------------------------------------------------------------
for path in time_paths:
    base = os.path.basename(path)
    m = TIME_PATTERN.match(base)
    if m is None:
        # ignore unknown json files
        continue

    agents = m.group("agents")
    scenario = m.group("scenario")
    nom = m.group("nom")
    grouping = m.group("grouping")
    maxgroup = int(m.group("maxgroup"))

    per_step = load_computation_time_per_step(path, grouping=grouping)

    key = (agents, scenario, nom)
    time_data[key][grouping][maxgroup].extend(per_step)

# ---------------------------------------------------------------------
# Plot: collision boxplots per (agents, scenario, nom)
# X-axis: collision types (3)
# Boxes: grouping off (single) + grouping on (one per maxgroup)
# ---------------------------------------------------------------------
for (agents, scenario, nom), d in collision_data.items():
    fig, ax = plt.subplots(figsize=(8, 4))

    x_base = np.arange(len(COLLISION_TYPES))
    width = 0.15

    legend_handles = []

    # Grouping OFF: merge across all maxgroup entries (should normally be one)
    off_data = []
    for ct in COLLISION_TYPES:
        vals = []
        for mg_dict in d["off"].values():
            vals.extend(mg_dict[ct])
        off_data.append(vals)

    pos_off = x_base - width
    bp_off = ax.boxplot(
        off_data,
        positions=pos_off,
        widths=width,
        patch_artist=True,
    )
    for box in bp_off["boxes"]:
        box.set_facecolor("tab:blue")
    legend_handles.append(bp_off["boxes"][0])

    # Grouping ON: one box per maxgroup
    # on_groups = sorted(d["on"].keys())
    # for i, g in enumerate(on_groups):
    #     on_data = [d["on"][g][ct] for ct in COLLISION_TYPES]
    #     pos_on = x_base + (i + 1) * width

    #     bp_on = ax.boxplot(
    #         on_data,
    #         positions=pos_on,
    #         widths=width,
    #         patch_artist=True,
    #     )
    #     for box in bp_on["boxes"]:
    #         box.set_facecolor("tab:orange")
    #     legend_handles.append(bp_on["boxes"][0])

    # Grouping ON: one box per maxgroup, color-coded by maxgroup
    on_groups = sorted(d["on"].keys())
    cmap = plt.get_cmap("tab10")

    for i, g in enumerate(on_groups):
        color = cmap(i % 10)
        on_data = [d["on"][g][ct] for ct in COLLISION_TYPES]
        pos_on = x_base + (i + 1) * width

        bp_on = ax.boxplot(
            on_data,
            positions=pos_on,
            widths=width,
            patch_artist=True,
        )

        for box in bp_on["boxes"]:
            box.set_facecolor(color)

        legend_handles.append(bp_on["boxes"][0])

    ax.set_xticks(x_base)
    ax.set_xticklabels(COLLISION_XLABELS)
    ax.set_ylabel("Collision Rate")
    ax.set_title(f"Agents={agents}, Scenario={scenario}, Nominal={nom}")
    ax.set_ylim((0, 0.3))
    ax.grid(True, axis="y")

    legend_labels = ["Grouping Off"] + [
        f"Grouping On (maxgroup={g})" for g in on_groups
    ]
    ax.legend(legend_handles, legend_labels, loc="best")

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR,
        f"collision_boxplot_agents_{agents}_scenario_{scenario}_nom_{nom}.pdf",
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

# ---------------------------------------------------------------------
# Plot: computation time boxplots per (agents, scenario, nom)
# X-axis: grouping configs
# Y-axis: computation time per step (averaged across groups if grouping=on)
# ---------------------------------------------------------------------
for (agents, scenario, nom), d in time_data.items():
    # If no timing data exists for this setup, skip
    has_any = any(len(v) > 0 for v in d["off"].values()) or any(
        len(v) > 0 for v in d["on"].values()
    )
    if not has_any:
        continue

    labels = []
    series = []

    # Grouping OFF: merge across maxgroup entries (should normally be one)
    off_vals = []
    for vals in d["off"].values():
        off_vals.extend(vals)
    if len(off_vals) > 0:
        labels.append("Grouping Off")
        series.append(off_vals)

    # Grouping ON: one box per maxgroup
    for g in sorted(d["on"].keys()):
        vals = d["on"][g]
        if len(vals) == 0:
            continue
        labels.append(f"Grouping On (maxgroup={g})")
        series.append(vals)

    if len(series) == 0:
        continue

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(series, patch_artist=True)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim((0, 0.1))
    ax.set_ylabel("Computation Time per Step [s]")
    ax.set_title(f"Agents={agents}, Scenario={scenario}, Nominal={nom}")
    ax.grid(True, axis="y")

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR,
        f"computation_time_boxplot_agents_{agents}_scenario_{scenario}_nom_{nom}.pdf",
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

# ---------------------------------------------------------------------
# Plot: average speed (same layout as collision plots)
# X-axis: single category "Average Speed"
# Y-axis: average speed
# ---------------------------------------------------------------------
for (agents, scenario, nom), d in collision_data.items():
    fig, ax = plt.subplots(figsize=(6, 4))

    x_base = np.array([0])
    width = 0.15
    legend_handles = []

    # Grouping OFF
    off_vals = []
    for mg_dict in d["off"].values():
        off_vals.extend(mg_dict["avg_speed"])

    bp_off = ax.boxplot(
        [off_vals],
        positions=x_base - width,
        widths=width,
        patch_artist=True,
    )
    for box in bp_off["boxes"]:
        box.set_facecolor("tab:blue")
    legend_handles.append(bp_off["boxes"][0])

    # Grouping ON (color-coded by maxgroup)
    on_groups = sorted(d["on"].keys())
    cmap = plt.get_cmap("tab10")

    for i, g in enumerate(on_groups):
        color = cmap(i % 10)
        vals = d["on"][g]["avg_speed"]

        bp_on = ax.boxplot(
            [vals],
            positions=x_base + (i + 1) * width,
            widths=width,
            patch_artist=True,
        )
        for box in bp_on["boxes"]:
            box.set_facecolor(color)

        legend_handles.append(bp_on["boxes"][0])

    ax.set_xticks(x_base)
    ax.set_xticklabels(["Average Speed"])
    ax.set_ylabel("Speed")
    ax.set_title(f"Agents={agents}, Scenario={scenario}, Nominal={nom}")
    ax.grid(True, axis="y")

    legend_labels = ["Grouping Off"] + [
        f"Grouping On (maxgroup={g})" for g in on_groups
    ]
    ax.legend(legend_handles, legend_labels, loc="best")
    ax.set_ylim((0, 0.8))

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR,
        f"avg_speed_boxplot_agents_{agents}_scenario_{scenario}_nom_{nom}.pdf",
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")
