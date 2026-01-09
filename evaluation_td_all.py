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
FOLDER = "outputs/marl_cbf_fixed_group_2_best_rl/"
OUTPUT_DIR = os.path.join(FOLDER, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLLISION_TYPES = ["agents", "lanelets", "total"]
COLLISION_XLABELS = ["Agent-Agent", "Agent-Lanelet", "Total"]

# ---------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------
TD_PATTERN = re.compile(
    r"out_td_"
    r"agents_(?P<agents>\d+)_"
    r"seed_(?P<seed>\d+)_"
    r"(?:(?P<grouping>grouping_on)_maxgroup_(?P<maxgroup>\d+)_|"
    r"(?P<grouping_off>grouping_off_)|)"
    r"nom_(?P<nom>[^_]+)_"
    r"(?:(?P<only>only)_)?"
    r"scenario_(?P<scenario>[^.]+)"
    r"\.td$"
)
TIME_PATTERN = re.compile(
    r"computation_t_"
    r"agents_(?P<agents>\d+)_"
    r"seed_(?P<seed>\d+)_"
    r"(?:(?P<grouping>grouping_on)_maxgroup_(?P<maxgroup>\d+)_|"
    r"(?P<grouping_off>grouping_off_)|)"
    r"nom_(?P<nom>[^_]+)_"
    r"(?:(?P<only>only)_)?"
    r"scenario_(?P<scenario>[^.]+)"
    r"\.json$"
)


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def load_collision_and_speed(td_path: str) -> tuple[float, float, float, float]:
    """
    Returns:
        agent-agent collision rate,
        agent-lanelet collision rate,
        total collision rate,
        average speed over all agents and steps
    """
    out_td = torch.load(td_path, weights_only=True)

    aa_flag = out_td["agents", "info", "is_collision_with_agents"].bool()
    al_flag = out_td["agents", "info", "is_collision_with_lanelets"].bool()

    collide = aa_flag | al_flag
    num_steps = int(aa_flag.shape[1])

    aa = aa_flag.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
    al = al_flag.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
    tot = collide.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps

    speed = out_td["agents", "info", "vel"].norm(dim=-1)
    avg_speed = speed.mean().item()

    return aa.item(), al.item(), tot.item(), avg_speed


def load_computation_time_per_step(json_path: str, grouping: str) -> list[float]:
    """
    grouping == "off":
        list[float] per step
    grouping == "on":
        list[list[float]] per step -> averaged across groups
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if grouping == "off":
        return [float(x) for x in obj]

    per_step = []
    for step in obj:
        if isinstance(step, list) and len(step) > 0:
            per_step.append(float(np.mean(step)))
    return per_step


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------
collision_data = defaultdict(
    lambda: {
        "off": defaultdict(lambda: defaultdict(list)),
        "on": defaultdict(lambda: defaultdict(list)),
    }
)

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

for fname in os.listdir(FOLDER):
    path = os.path.join(FOLDER, fname)
    if fname.endswith(".td"):
        td_paths.append(path)
    elif fname.startswith("computation_t_") and fname.endswith(".json"):
        time_paths.append(path)

td_paths.sort()
time_paths.sort()

# ---------------------------------------------------------------------
# Load TD files
# ---------------------------------------------------------------------
for path in td_paths:
    m = TD_PATTERN.match(os.path.basename(path))
    if m is None:
        raise ValueError(f"Invalid TD filename: {path}")

    agents = m.group("agents")
    scenario = m.group("scenario")
    nom = m.group("nom")
    if m.group("grouping") == "grouping_on":
        grouping = "on"
        maxgroup = int(m.group("maxgroup"))
    else:
        grouping = "off"
        maxgroup = float("inf")

    aa, al, tot, avg_speed = load_collision_and_speed(path)

    key = (agents, scenario, nom)
    collision_data[key][grouping][maxgroup]["agents"].append(aa)
    collision_data[key][grouping][maxgroup]["lanelets"].append(al)
    collision_data[key][grouping][maxgroup]["total"].append(tot)
    collision_data[key][grouping][maxgroup]["avg_speed"].append(avg_speed)

# ---------------------------------------------------------------------
# Load computation time files
# ---------------------------------------------------------------------
for path in time_paths:
    m = TIME_PATTERN.match(os.path.basename(path))
    if m is None:
        continue

    agents = m.group("agents")
    scenario = m.group("scenario")
    nom = m.group("nom")

    if m.group("grouping") == "grouping_on":
        grouping = "on"
        maxgroup = int(m.group("maxgroup"))
    else:
        grouping = "off"
        maxgroup = float("inf")

    per_step = load_computation_time_per_step(path, grouping)

    key = (agents, scenario, nom)
    time_data[key][grouping][maxgroup].extend(per_step)

# ---------------------------------------------------------------------
# Plot: collision boxplots
# ---------------------------------------------------------------------
for (agents, scenario, nom), d in collision_data.items():
    fig, ax = plt.subplots(figsize=(8, 4))

    x_base = np.arange(len(COLLISION_TYPES))
    width = 0.15
    on_groups = sorted(d["on"].keys())

    legend_handles = []

    # Grouping ON (finite maxgroup)
    cmap = plt.get_cmap("tab10")
    n_on = len(on_groups)
    n_boxes = n_on + 1  # +1 for maxgroup = inf

    # centered offsets, symmetric around zero
    offsets = (np.arange(n_boxes) - (n_boxes - 1) / 2.0) * width
    for i, g in enumerate(on_groups):
        color = cmap(i % 10)
        data_g = [d["on"][g][ct] for ct in COLLISION_TYPES]
        pos = x_base + offsets[i]

        bp = ax.boxplot(data_g, positions=pos, widths=width, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(color)

        legend_handles.append(bp["boxes"][0])

    # Grouping OFF treated as maxgroup = inf
    off_data = []
    for ct in COLLISION_TYPES:
        vals = []
        for mg_dict in d["off"].values():
            vals.extend(mg_dict[ct])
        off_data.append(vals)

    pos_off = x_base + offsets[-1]
    bp_off = ax.boxplot(off_data, positions=pos_off, widths=width, patch_artist=True)
    for box in bp_off["boxes"]:
        box.set_facecolor("black")

    legend_handles.append(bp_off["boxes"][0])

    ax.set_xticks(x_base)
    ax.set_xticklabels(COLLISION_XLABELS)
    ax.set_ylabel("Collision Rate")
    ax.set_ylim((0, 0.3))
    ax.set_title(f"Agents={agents}, Scenario={scenario}, Nominal={nom}")
    ax.grid(True, axis="y")

    legend_labels = [f"Grouping On (maxgroup={g})" for g in on_groups] + [
        "Grouping Off (maxgroup = inf)"
    ]
    ax.legend(legend_handles, legend_labels, loc="best")

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_collision_agents_{agents}_nom_{nom}.pdf",
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

# ---------------------------------------------------------------------
# Plot: computation time boxplots
# ---------------------------------------------------------------------
for (agents, scenario, nom), d in time_data.items():
    on_groups = sorted(d["on"].keys())

    series = []
    labels = []
    colors = []

    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(on_groups):
        if len(d["on"][g]) == 0:
            continue
        series.append(d["on"][g])
        labels.append(f"Grouping On (maxgroup={g})")
        colors.append(cmap(i % 10))

    off_vals = []
    for vals in d["off"].values():
        off_vals.extend(vals)

    if len(off_vals) > 0:
        series.append(off_vals)
        labels.append("Grouping Off (maxgroup = inf)")
        colors.append("black")

    if len(series) == 0:
        continue

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(series, patch_artist=True)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)

    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim((0, 0.05))
    ax.set_ylabel("Computation Time per Step [s]")
    ax.set_title(f"Agents={agents}, Scenario={scenario}, Nominal={nom}")
    ax.grid(True, axis="y")

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_computation_agents_{agents}_nom_{nom}.pdf",
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

# ---------------------------------------------------------------------
# Plot: average speed boxplots
# ---------------------------------------------------------------------
for (agents, scenario, nom), d in collision_data.items():
    fig, ax = plt.subplots(figsize=(6, 4))

    x_base = np.array([0])
    width = 0.15
    on_groups = sorted(d["on"].keys())

    n_on = len(on_groups)
    n_boxes = n_on + 1  # +1 for grouping off (maxgroup = inf)
    offsets = (np.arange(n_boxes) - (n_boxes - 1) / 2.0) * width

    legend_handles = []

    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(on_groups):
        vals = d["on"][g]["avg_speed"]
        pos = x_base + offsets[i]

        bp = ax.boxplot([vals], positions=pos, widths=width, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(cmap(i % 10))

        legend_handles.append(bp["boxes"][0])

    off_vals = []
    for mg_dict in d["off"].values():
        off_vals.extend(mg_dict["avg_speed"])

    pos_off = x_base + offsets[-1]
    bp_off = ax.boxplot([off_vals], positions=pos_off, widths=width, patch_artist=True)
    for box in bp_off["boxes"]:
        box.set_facecolor("black")

    legend_handles.append(bp_off["boxes"][0])

    ax.set_xticks(x_base)
    ax.set_xticklabels(["Average Speed"])
    ax.set_ylabel("Speed")
    ax.set_ylim((0, 0.8))
    ax.set_title(f"Agents={agents}, Scenario={scenario}, Nominal={nom}")
    ax.grid(True, axis="y")

    legend_labels = [f"Grouping On (maxgroup={g})" for g in on_groups] + [
        "Grouping Off (maxgroup = inf)"
    ]
    ax.legend(legend_handles, legend_labels, loc="best")

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_speed_agents_{agents}_nom_{nom}.pdf",
    )
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")
