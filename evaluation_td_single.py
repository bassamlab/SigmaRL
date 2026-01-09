import os
import re
import json
import numpy as np
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
# Input (change this only)
# ---------------------------------------------------------------------
td_file_path = (
    "outputs/marl_cbf_test/"
    "out_td_agents_8_seed_1_nom_rl_only_scenario_interchange_3.td"
)

# ---------------------------------------------------------------------
# Filename patterns (grouping optional)
# ---------------------------------------------------------------------
TD_PATTERN = re.compile(
    r"out_td"
    r"_agents_(?P<agents>\d+)"
    r"_seed_(?P<seed>\d+)"
    r"(?:_grouping_(?P<grouping>on|off)_maxgroup_(?P<maxgroup>\d+))?"
    r"_nom_(?P<nom>[^_]+(?:_[^_]+)*)"
    r"_scenario_(?P<scenario>[^.]+)"
    r"\.td$"
)

TIME_PATTERN = re.compile(
    r"computation_t"
    r"_agents_(?P<agents>\d+)"
    r"_seed_(?P<seed>\d+)"
    r"(?:_grouping_(?P<grouping>on|off)_maxgroup_(?P<maxgroup>\d+))?"
    r"_nom_(?P<nom>[^_]+(?:_[^_]+)*)"
    r"_scenario_(?P<scenario>[^.]+)"
    r"\.json$"
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_collision_and_speed(td_path: str):
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


def load_computation_time(json_path: str, grouping: str | None):
    if not os.path.isfile(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if grouping is None or grouping == "off":
        return float(np.mean(obj))

    per_step = []
    for step in obj:
        if isinstance(step, list) and len(step) > 0:
            per_step.append(float(np.mean(step)))

    return float(np.mean(per_step)) if per_step else None


# ---------------------------------------------------------------------
# Parse filename
# ---------------------------------------------------------------------
m = TD_PATTERN.match(os.path.basename(td_file_path))
if m is None:
    raise ValueError(f"Invalid TD filename: {td_file_path}")

agents = m.group("agents")
seed = m.group("seed")
grouping = m.group("grouping")  # None if not present
maxgroup = m.group("maxgroup")  # None if not present
nom = m.group("nom")
scenario = m.group("scenario")

# ---------------------------------------------------------------------
# Load TD metrics
# ---------------------------------------------------------------------
aa, al, tot, avg_speed = load_collision_and_speed(td_file_path)

# ---------------------------------------------------------------------
# Locate computation-time file (if exists)
# ---------------------------------------------------------------------
folder = os.path.dirname(td_file_path)
avg_time = None

for fname in os.listdir(folder):
    tm = TIME_PATTERN.match(fname)
    if tm is None:
        continue
    if (
        tm.group("agents") == agents
        and tm.group("seed") == seed
        and tm.group("nom") == nom
        and tm.group("scenario") == scenario
        and tm.group("grouping") == grouping
        and tm.group("maxgroup") == maxgroup
    ):
        avg_time = load_computation_time(os.path.join(folder, fname), grouping)
        break

# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------
print("=== Single Simulation Evaluation ===")
print(f"Agents           : {agents}")
print(f"Scenario         : {scenario}")
print(f"Nominal Control  : {nom}")
print(f"Seed             : {seed}")

if grouping is None:
    print("Grouping         : not used")
else:
    print(f"Grouping         : {grouping}")
    print(f"Max Group Size   : {maxgroup}")

print()
print(f"Average Speed                : {avg_speed:.4f}")
print(f"Agent-Agent Collision Rate   : {aa:.4f}")
print(f"Agent-Lanelet Collision Rate : {al:.4f}")
print(f"Total Collision Rate         : {tot:.4f}")

if avg_time is not None:
    print(f"Avg Computation Time / Step  : {avg_time:.6f} s")
else:
    print("Avg Computation Time / Step  : N/A")
