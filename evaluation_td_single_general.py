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
td_file_path = "outputs/cbf_informed_marl/baseline/out_td_grouping_off_seed_1.td"

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
# Load TD metrics
# ---------------------------------------------------------------------
aa, al, tot, avg_speed = load_collision_and_speed(td_file_path)


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------
print("=" * 80)
print(td_file_path)
print(f"Average Speed                : {avg_speed:.4f}")
print(f"Agent-Agent Collision Rate   : {aa:.4f}")
print(f"Agent-Lanelet Collision Rate : {al:.4f}")
print(f"Total Collision Rate         : {tot:.4f}")
