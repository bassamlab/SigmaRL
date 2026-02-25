import os
import json
import time

import torch

from sigmarl.helper_training import SaveData, Parameters, SaveData
from tensordict import TensorDict

from sigmarl.helper_common import is_latex_available, save_video
from sigmarl.mappo_cavs import mappo_cavs

from typing import Optional, Dict, List
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

import math
from matplotlib.ticker import FuncFormatter
import numpy as np


METHOD_STYLE = {
    "CBF (our)": {
        "color": "tab:blue",
        "ls": "-",
        "lw": 1.2,
        "marker": None,
        "ms": 0,
        "mew": 1.0,
        "markevery": None,
    },
    "Distance": {  # Distance V1
        "color": "tab:green",
        "ls": "-",
        "lw": 1.2,
        "marker": None,
        "ms": 4.5,
        "mew": 1.0,
        "markevery": 200,
    },
    "Distance+Sparse": {  # Distance V2
        "color": "tab:green",
        "ls": "--",
        "lw": 1.2,
        "marker": None,
        "ms": 4.5,
        "mew": 1.0,
        "markevery": 200,
    },
    "TTC": {  # TTC V1
        "color": "tab:orange",
        "ls": "-",
        "lw": 1.2,
        "marker": None,
        "ms": 5.0,
        "mew": 1.0,
        "markevery": 200,
    },
    "TTC+Sparse": {  # TTC V2
        "color": "tab:orange",
        "ls": "--",
        "lw": 1.2,
        "marker": None,
        "ms": 5.0,
        "mew": 1.0,
        "markevery": 200,
    },
    "Sparse": {
        "color": "#F0E442",
        "ls": "--",
        "lw": 1.4,
        "marker": None,
        "ms": 0,
        "mew": 1.0,
        "markevery": None,
    },
}


DEFAULT_STYLE = {"color": "0.25", "ls": "-", "lw": 1.2}  # fallback: dark gray

FRAME_PER_ITERATION = 128 * 32

plt.rcParams.update(
    {
        "font.size": 12,  # slightly smaller for IEEE single-column
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        "text.usetex": is_latex_available(),
        # PDF export quality
        "pdf.fonttype": 42,  # embed TrueType fonts (better in paper PDFs)
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
)


def get_name_suffix(scenario_type: str, n_agents: int, random_seed: int) -> str:
    return f"{scenario_type}_nagents{n_agents}_seed{random_seed}"


def _load_episode_reward_mean_list(model_dir: Path) -> Optional[List[float]]:
    """
    Load episode_reward_mean_list from reward*.json in a given training folder.
    Returns None if not found or malformed.
    """
    json_files = sorted(model_dir.glob("reward*.json"))
    if len(json_files) == 0:
        return None

    # Use the first match; adjust if you want a specific naming rule
    path = json_files[0]
    try:
        with open(path, "r") as f:
            data = json.load(f)
        lst = data.get("episode_reward_mean_list", None)
        if not isinstance(lst, list) or len(lst) == 0:
            return None
        return [float(x) for x in lst]
    except Exception:
        return None


def _aggregate_curves(
    curves: List[List[float]],
    smoothing: str = "ema",
    ema_alpha: float = 0.05,
    rolling_window: int = 25,
):
    """
    Align curves by truncating to the minimum length.
    Optionally smooth each curve first, then compute mean and std.

    smoothing: "none" | "ema" | "rolling"
    """
    if len(curves) == 0:
        return None, None, None

    min_len = min(len(c) for c in curves)
    if min_len <= 1:
        return None, None, None

    proc = []
    for c in curves:
        c = [float(v) for v in c[:min_len]]
        if smoothing == "ema":
            c = _smooth_curve_ema(c, alpha=ema_alpha)
        elif smoothing == "rolling":
            c = _smooth_curve_rolling_mean(c, window=rolling_window)
        elif smoothing == "none":
            pass
        else:
            raise ValueError(f"Unknown smoothing={smoothing}")
        proc.append(c)

    arr = torch.tensor(proc, dtype=torch.float32)  # [n_runs, T]
    mean = arr.mean(dim=0)
    std = arr.std(dim=0, unbiased=False) if arr.shape[0] > 1 else torch.zeros_like(mean)
    x = list(range(min_len))

    # If you want standard error of the mean instead of std, uncomment below:
    # sem = std / (arr.shape[0] ** 0.5)
    # return x, mean.tolist(), sem.tolist()

    return x, mean.tolist(), std.tolist()


def _wrap_method_label(s: str, is_line_break=True) -> str:
    """
    Insert line breaks to shorten long method names for x-axis tick labels.
    """
    if s == "CBF (our)":
        return "CBF\n(our)" if is_line_break else "CBF (our)"
    if s == "Distance+Sparse":
        return "Distance\nV2" if is_line_break else "Distance V2"
    if s == "TTC+Sparse":
        return "TTC\nV2" if is_line_break else "TTC V2"
    if s == "TTC":
        return "TTC\nV1" if is_line_break else "TTC V1"
    if s == "Distance":
        return "Distance\nV1" if is_line_break else "Distance V1"
    return s


def _reward_method_from_path(p: str) -> str:
    """
    Extract reward method from a path containing '/rew_method_cbf', etc.
    Returns 'unknown' if not found.
    """
    m = re.search(r"/rew_method_([^/]+)", p)
    return m.group(1) if m else "unknown"


def _reward_method_label(m: str) -> str:
    label_map = {
        "cbf": "CBF (our)",
        "distance_sparse": "Distance+Sparse",
        "ttc_sparse": "TTC+Sparse",
        "distance": "Distance",
        "sparse": "Sparse",
        "ttc": "TTC",
        "distance_old": "Distance (Old)",
    }
    return label_map.get(m, m)


def _method_sort_key(label: str) -> int:
    priority = {
        "CBF (our)": 0,
        "Distance": 1,
        "Distance+Sparse": 2,
        "TTC": 3,
        "TTC+Sparse": 4,
        "Sparse": 5,
    }
    return priority.get(label, 10**6)


def _seed_from_path(p: str) -> Optional[int]:
    """
    Extract seed index from a path containing '/seed3', etc.
    """
    m = re.search(r"/seed(\d+)", p)
    return int(m.group(1)) if m else None


def _path_sort_key(p: str):
    """
    Primary: reward method (custom order)
    Secondary: reward_progress (ascending, if present)
    Tertiary: seed (ascending)
    """
    method = _reward_method_from_path(p)
    rp = _reward_progress_from_path(p)
    seed = _seed_from_path(p)

    method_priority = {
        "cbf": 0,
        "distance": 1,
        "distance_old": 2,
        "sparse": 3,
        "ttc": 4,
    }
    method_key = method_priority.get(method, 10**6)

    rp_key = rp if rp is not None else float("inf")
    seed_key = seed if seed is not None else 10**9
    return (method_key, rp_key, seed_key)


def _reward_progress_from_path(p: str) -> Optional[float]:
    """
    Extract reward_progress value from a path containing 'reward_progress2.5', etc.
    Returns None if not found.
    """
    m = re.search(r"reward_progress([0-9]+(?:\.[0-9]+)?)", p)
    return float(m.group(1)) if m else None


def _exp_label_from_path(p: str) -> str:
    """
    New experiment label: reward method only, extracted from '/rew_method_*'.
    """
    m = _reward_method_from_path(p)
    return _reward_method_label(m)


def _smooth_curve_ema(y: List[float], alpha: float) -> List[float]:
    """
    Exponential moving average smoothing.
    alpha in (0, 1]. Smaller alpha => more smoothing.
    """
    if len(y) == 0:
        return y
    out = [float(y[0])]
    for t in range(1, len(y)):
        out.append(alpha * float(y[t]) + (1.0 - alpha) * out[-1])
    return out


def _smooth_curve_rolling_mean(y: List[float], window: int) -> List[float]:
    """
    Rolling mean with a causal window (uses past values only).
    window >= 1.
    """
    if window <= 1 or len(y) == 0:
        return [float(v) for v in y]
    out = []
    s = 0.0
    q = []
    for v in y:
        v = float(v)
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def trim_td(out_td: TensorDict, keys_to_keep=None) -> TensorDict:
    """
    Trim a TensorDict to keep only selected fields.

    Args:
        out_td: Original TensorDict.

    Returns:
        A new TensorDict containing only the specified keys.
    """
    if keys_to_keep is None:
        keys_to_keep = [
            ("agents", "info", "pos"),
            ("agents", "info", "rot"),
            ("agents", "info", "vel"),
            ("agents", "info", "ref"),
            ("agents", "info", "ref_lanelet_ids"),
            ("agents", "info", "is_collision_with_agents"),
            ("agents", "info", "is_collision_with_lanelets"),
            ("agents", "info", "is_reach_goal"),
            ("agents", "info", "rew_reach_goal"),
            ("agents", "info", "rew_collide_other_agents"),
            ("agents", "info", "rew_collide_lane"),
            ("agents", "info", "applied_action_vel"),
            ("agents", "info", "applied_action_steer"),
            ("agents", "info", "nominal_action_vel"),
            ("agents", "info", "nominal_action_steer"),
        ]

    trimmed = TensorDict(
        {},
        batch_size=out_td.batch_size,
        device=out_td.device,
    )

    for key in keys_to_keep:
        trimmed.set(key, out_td.get(key))

    return trimmed


def run_evaluations():
    # Each element of path_list is a folder that contains one or more .pth files
    model_folders = [p for p in path_list if os.path.isdir(p)]

    # Optional: progress accounting (approx)
    total_models = len(model_folders)

    total_evaluations = total_models * len(scenario_specs) * len(random_seed_list)
    current_evaluation = 0

    current_evaluation = 0
    for train_path in model_folders:
        # Find config json (you used reward*.json). Keep your logic.
        path_to_json_file: Optional[str] = None
        for file in os.listdir(train_path):
            if file.endswith(".json") and file.startswith("reward"):
                path_to_json_file = os.path.join(train_path, file)
                break
        if path_to_json_file is None:
            print(f"[WARNING] No config JSON found in {train_path}, skipping.")
            continue

        with open(path_to_json_file, "r") as file:
            data = json.load(file)

        # Enumerate all .pth models under this folder
        pth_files = sorted([f for f in os.listdir(train_path) if f.endswith(".pth")])
        if len(pth_files) == 0:
            print(f"[WARNING] No .pth files found in {train_path}, skipping.")
            continue

        for scenario_type, n_agents in scenario_specs:
            for random_seed in random_seed_list:
                current_evaluation += 1
                print("=" * 80)
                print(
                    f"[INFO] Evaluation {current_evaluation}/{total_evaluations}: {train_path}"
                )
                print("=" * 80)

                # Make a fresh copy of parameters per run (avoid state carry-over)
                parameters: Parameters = SaveData.from_dict(data).parameters

                parameters.is_testing_mode = True
                parameters.rew_method = "sparse"  # Reward method: {"distance", "cbf_constraint", "cbf_qp", "ttc", "sparse"}
                parameters.is_real_time_rendering = False
                parameters.is_save_eval_results = True
                parameters.is_load_model = True

                parameters.is_using_cbf_training = False
                parameters.is_using_cbf_testing = True
                parameters.is_solve_qp = True
                parameters.is_apply_cbf_action = "do_not_apply" not in train_path

                parameters.is_load_final_model = "final" in train_path
                parameters.is_load_out_td = False
                parameters.max_steps = 600
                parameters.num_vmas_envs = 1

                parameters.scenario_type = scenario_type
                parameters.n_agents = n_agents

                parameters.is_save_simulation_video = (
                    True if random_seed == random_seed_list[0] else False
                )  # Only save video for the first seed to save disk space
                parameters.is_visualize_short_term_path = True
                parameters.is_visualize_lane_boundary = False
                parameters.is_visualize_extra_info = True
                parameters.random_seed = random_seed
                parameters.is_continue_train = False
                parameters.lane_width = 0.25

                # Save into the same folder that holds this .pth
                parameters.where_to_save = train_path

                name_suffix = get_name_suffix(scenario_type, n_agents, random_seed)
                out_td_filename = f"out_td_{name_suffix}.td"
                video_basename = f"video_{name_suffix}"
                task_performance_filename = f"task_performance_{name_suffix}.json"
                task_performance_path = os.path.join(
                    train_path, task_performance_filename
                )
                out_td_path = os.path.join(train_path, out_td_filename)

                if os.path.exists(out_td_path) and os.path.exists(
                    task_performance_path
                ):
                    print(f"[INFO] Skipping existing output: {train_path}")
                    continue

                (
                    env,
                    decision_making_module,
                    optimization_module,
                    priority_module,
                    cbf_controllers,
                    parameters,
                ) = mappo_cavs(parameters=parameters)

                t_0 = time.time()
                out_td, frame_list = env.rollout(
                    max_steps=parameters.max_steps - 1,
                    policy=decision_making_module.policy,
                    priority_module=priority_module,
                    cbf_controllers=cbf_controllers,
                    callback=lambda env, _: env.render(
                        mode="rgb_array", visualize_when_rgb=False
                    )
                    if parameters.is_save_simulation_video
                    else None,  # Mode can be "human" or "rgb_array". Use "rgb_array" for headless evaluation.
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    is_save_simulation_video=parameters.is_save_simulation_video,
                )
                print(f"[INFO] Simulation time: {time.time() - t_0:.2f} seconds")

                task_performance_filename = f"task_performance_{name_suffix}.json"
                env.scenario_name.num_task_tries  # example value: tensor([11], dtype=torch.int32)
                env.scenario_name.task_success_times  # example value: tensor([3], dtype=torch.int32)

                # Pack into a dictionary
                task_performance_data = {
                    "num_task_tries": int(env.scenario_name.num_task_tries.item()),
                    "task_success_times": int(
                        env.scenario_name.task_success_times.item()
                    ),
                }

                # Save to JSON
                with open(task_performance_path, "w") as f:
                    json.dump(task_performance_data, f, indent=4)

                print(f"[INFO] Saved task performance to {task_performance_path}")

                if parameters.is_save_eval_results:
                    is_trim_td = True
                    if is_trim_td:
                        out_td = trim_td(out_td)

                    torch.save(out_td, out_td_path)
                    print(f"[INFO] Simulation outputs saved to {out_td_path}")

                    if parameters.is_save_simulation_video:
                        save_video(
                            os.path.join(train_path, video_basename),
                            frame_list,
                            fps=1 / parameters.dt,
                            fmt="mp4",
                            quality="low",
                        )
                        print(
                            f"[INFO] Video saved to {os.path.join(train_path, video_basename)}.mp4"
                        )


def _load_out_td(out_td_path: Path):
    return torch.load(out_td_path, weights_only=False)


def compute_collision_rate_percent(out_td) -> float:
    """
    Return collision rate in percent.

    out_td[("agents","info","is_collision_with_agents")] has shape:
      [B, T, N, 1]   (B=batch_size, T=n_steps, N=n_agents)
    Same for is_collision_with_lanelets.

    Default definition:
      percentage of time steps where at least one agent collides
      (with another agent or with boundaries), averaged over B.
    """
    key_agents = ("agents", "info", "is_collision_with_agents")
    key_lane = ("agents", "info", "is_collision_with_lanelets")

    coll_agents = out_td.get(key_agents).bool()  # [B,T,N,1]
    coll_lane = out_td.get(key_lane).bool()  # [B,T,N,1]
    coll = coll_agents | coll_lane  # [B,T,N,1]

    if COLLISION_MODE == "per_timestep_any_agent":
        # any collision among agents at each time step
        coll_bt = coll.any(dim=(2, 3))  # [B,T]
        return coll_bt.float().mean().item() * 100.0

    if COLLISION_MODE == "per_agent_timestep":
        # mean over all agent-time pairs (and B)
        return coll.float().mean().item() * 100.0  # Usually less than the above method

    raise ValueError(f"Unknown COLLISION_MODE={COLLISION_MODE}")


def compute_average_speed(out_td) -> float:
    """
    Return average speed [m/s] across all agents and all time steps (and envs).
    Uses key:
      ("agents","info","vel") -> [..., 2]
    """
    key_vel = ("agents", "info", "vel")
    vel = out_td.get(key_vel).float()

    # speed = ||v||_2 on the last dim
    speed = torch.linalg.norm(vel, dim=-1)

    # Average over all entries (time * env * agent)
    return speed.mean().item()


def compute_total_reward(out_td) -> float:
    """
    Return the total reward as a Python float.

    Reward keys:
      ("agents","info","rew_reach_goal")
      ("agents","info","rew_collide_other_agents")
      ("agents","info","rew_collide_lane")

    Assumes each tensor is broadcast-compatible (typically [B, T, N, 1]).
    Computes: mean over all entries (B * T * N * ...).
    """
    key_reach = ("agents", "info", "rew_reach_goal")
    key_coll_agents = ("agents", "info", "rew_collide_other_agents")
    key_coll_lane = ("agents", "info", "rew_collide_lane")

    r_reach = out_td.get(key_reach).float()
    r_coll_agents = out_td.get(key_coll_agents).float()
    r_coll_lane = out_td.get(key_coll_lane).float()

    r_total = r_reach + r_coll_agents + r_coll_lane
    return r_total.sum().item()


def compute_cbf_activation_percentage(out_td) -> float:
    """
    Compute the mean absolute difference between nominal and applied actions,
    averaged over all batches, time steps, agents, and action dimensions (vel, steer).

    Expected shapes (for each key):
        [B, T, N, 1]
    """
    key_applied_vel = ("agents", "info", "applied_action_vel")
    key_applied_steer = ("agents", "info", "applied_action_steer")
    key_nominal_vel = ("agents", "info", "nominal_action_vel")
    key_nominal_steer = ("agents", "info", "nominal_action_steer")

    applied_vel = out_td.get(key_applied_vel).float()
    applied_steer = out_td.get(key_applied_steer).float()
    nominal_vel = out_td.get(key_nominal_vel).float()
    nominal_steer = out_td.get(key_nominal_steer).float()

    # Differences: [B, T, N, 1]
    d_vel = applied_vel - nominal_vel
    d_steer = applied_steer - nominal_steer

    # Mean absolute difference over all dims (B, T, N, 1) and over both channels (vel+steer)
    mean_abs_diff = 0.5 * (d_vel.abs().mean() + d_steer.abs().mean())

    applied_vel_abs_mean = applied_vel.abs().mean()
    applied_steer_abs_mean = applied_steer.abs().mean()
    mean_abs_diff_normalized = 0.5 * (
        d_vel.abs().mean() / (applied_vel_abs_mean + 1e-9)
        + d_steer.abs().mean() / (applied_steer_abs_mean + 1e-9)
    )

    # Activation count: any channel differs (with tolerance)
    eps = 1e-9
    activated = (d_vel.abs() > eps) | (d_steer.abs() > eps)  # [B, T, N, 1]
    n_activations = int(activated.sum().item())
    total_elements = activated.numel()

    return float(100 * mean_abs_diff_normalized.item()), float(
        100 * n_activations / total_elements
    )


def _load_task_performance(task_perf_path: Path) -> Optional[Dict[str, int]]:
    """
    Load task performance JSON:
      {"num_task_tries": int, "task_success_times": int}
    Returns None if not found or malformed.
    """
    if not task_perf_path.exists():
        return None
    try:
        with open(task_perf_path, "r") as f:
            d = json.load(f)
        a = d.get("num_task_tries", None)
        b = d.get("task_success_times", None)
        if a is None or b is None:
            return None
        return {"num_task_tries": int(a), "task_success_times": int(b)}
    except Exception:
        return None


# -----------------------------
# Plotting
# -----------------------------
def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", linewidth=0.6, alpha=0.40)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.15)

    # Start from a clean tick state
    ax.minorticks_off()

    # Tick direction "in" for both major and minor
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6)


def save_boxplot_pdf(
    data_by_method: Dict[str, List[float]],
    method_order: List[str],
    ylabel: str,
    out_pdf: Path,
    unit: Optional[str] = "",
    ylim: Optional[tuple] = None,
    show_fliers: bool = False,  # default cleaner for paper
    loc_text: str = "center",  # "center" | "right"
    type_text: str = "average",  # "average" | "median"
    is_show_x_ticks: bool = True,
    is_show_x_label: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 2.8), constrained_layout=True)

    data = [data_by_method.get(m, []) for m in method_order]

    bp = ax.boxplot(
        data,
        showfliers=show_fliers,
        widths=0.55,
        patch_artist=True,
        showmeans=False,
        medianprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )
    iqr_eps = 1e-9  # or something like 1e-6 if your values are floats with noise

    for i, m in enumerate(method_order):
        style = METHOD_STYLE.get(m, DEFAULT_STYLE)

        # boxes
        bp["boxes"][i].set_facecolor(style["color"])
        bp["boxes"][i].set_edgecolor("black")
        bp["boxes"][i].set_linewidth(1.0)

        # whiskers and caps (2 each per box)
        bp["whiskers"][2 * i].set_color("black")
        bp["whiskers"][2 * i + 1].set_color("black")
        bp["caps"][2 * i].set_color("black")
        bp["caps"][2 * i + 1].set_color("black")

        # median color depends on IQR collapse
        vals = data[i]  # same order as method_order
        if vals is None or len(vals) == 0:
            is_collapsed = False
        else:
            arr = np.asarray(vals, dtype=float)
            q1, q3 = np.percentile(arr, [25, 75])
            is_collapsed = abs(q3 - q1) < iqr_eps

        bp["medians"][i].set_color(style["color"] if is_collapsed else "white")
        bp["medians"][i].set_linewidth(0.5)

    if ylim is not None:
        ax.set_ylim(ylim)
    y0, y1 = ax.get_ylim()
    yr = max(y1 - y0, 1e-12)

    pad_y = 0.02 * yr  # keep text inside axes when clamped

    # box width must match bp widths=0.55
    box_width = 0.55
    half_box = 0.5 * box_width
    gap_x = 0.04  # small gap from box edge when loc_text="right"

    if loc_text not in {"center", "right"}:
        raise ValueError(f"Unknown loc_text={loc_text}, must be 'center' or 'right'")
    if type_text not in {"average", "median"}:
        raise ValueError(
            f"Unknown type_text={type_text}, must be 'average' or 'median'"
        )

    for i, vals in enumerate(data, start=1):  # box centers are 1..K
        if vals is None or len(vals) == 0:
            continue

        v = torch.tensor([float(x) for x in vals], dtype=torch.float32)
        stat_val = (
            float(v.mean().item())
            if type_text == "average"
            else float(v.median().item())
        )

        # y position: at the statistic; clamp inside visible range
        y_text = stat_val
        va = "center"
        if y_text > y1:
            y_text = y1 - pad_y
            va = "top"
        elif y_text < y0:
            y_text = y0 + pad_y
            va = "bottom"

        # x position + alignment
        if loc_text == "right":
            x_text = i + half_box + gap_x
            ha = "left"
        else:  # "center"
            x_text = i
            ha = "center"
            # tightly above the statistic (unless clamped)
            if y0 < stat_val < y1:
                y_text = min(stat_val + pad_y, y1 - pad_y)
                va = "bottom"

        # format with optional unit
        if unit is None or unit == "":
            text_str = f"{stat_val:.2f}"
        else:
            if plt.rcParams.get("text.usetex", False) and unit == "%":
                text_str = f"{stat_val:.2f}\\%"
            else:
                text_str = f"{stat_val:.2f}{unit}"

        ax.text(
            x_text,
            y_text,
            text_str,
            color="tab:red",
            ha=ha,
            va=va,
            clip_on=True,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0.15),
        )

    ax.set_ylabel(ylabel)

    # X label on/off
    if is_show_x_label:
        ax.set_xlabel("Reward Methods")
    else:
        ax.set_xlabel("")

    if is_show_x_ticks:
        ax.set_xticks(list(range(1, len(method_order) + 1)))

        if plt.rcParams.get("text.usetex", False):
            # LaTeX path: make labels italic via LaTeX, and replace '\n' with '\\' using \shortstack
            def _tex_italic_multiline(s: str) -> str:
                parts = _wrap_method_label(s).split("\n")
                # italicize each line explicitly
                parts = [rf"\textit{{{p}}}" for p in parts]
                return r"\shortstack{" + r"\\".join(parts) + "}"

            ax.set_xticklabels([_tex_italic_multiline(m) for m in method_order])
        else:
            # Matplotlib text path: normal newline + fontstyle works
            ax.set_xticklabels([_wrap_method_label(m) for m in method_order])
            for t in ax.get_xticklabels():
                t.set_fontstyle("italic")

        ax.tick_params(axis="x", rotation=0)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    _style_axis(ax)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_reward_curve_pdf(
    curves_by_method: Dict[str, List[List[float]]],
    method_order: List[str],
    out_pdf: Path,
    ylabel: str = "Episode Reward",
    frames_per_iteration: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)

    for method in method_order:
        curves = curves_by_method.get(method, [])
        x, mean, std = _aggregate_curves(
            curves,
            smoothing="ema",
            ema_alpha=0.03,
        )
        if x is None:
            continue

        x_frames = [frames_per_iteration * xi for xi in x]

        x = x_frames

        style = METHOD_STYLE.get(method, DEFAULT_STYLE)
        method_legend = _wrap_method_label(method, is_line_break=False)
        line = ax.plot(
            x,
            mean,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            marker=style.get("marker", None),
            markersize=style.get("ms", 0),
            markeredgewidth=style.get("mew", 1.0),
            markevery=style.get("markevery", None),
            label=rf"\textit{{{method_legend}}}",
        )[0]

        # Use STD (run-to-run spread) for the shaded area
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.2, linewidth=0.0)

    def _nice_step(x: float) -> float:
        """Return a 'nice' step size close to x using {1,2,5}x10^k."""
        if x <= 0:
            return 1.0
        k = math.floor(math.log10(x))
        base = 10**k
        m = x / base
        if m <= 1:
            return 1 * base
        if m <= 2:
            return 2 * base
        if m <= 5:
            return 5 * base
        return 10 * base

    # Nice x ticks in frames
    if ax.lines:
        # Find xmax in frames from plotted data
        # Use axis limit (not data max) to define ticks
        xmax_frames = ax.get_xlim()[1]  # this is 4e6

        raw_step = xmax_frames / 5.0
        step_frames = _nice_step(raw_step)

        n_steps = int(math.floor(xmax_frames / step_frames))
        xticks = [k * step_frames for k in range(0, n_steps + 1)]  # no +2
        ax.set_xticks(xticks)

        # Display ticks in millions with clean numbers
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e6:g}"))
        ax.set_xlabel(r"Frames ($\times10^6$)")

    _style_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_xlim((0.0, 4.0e6))
    ax.set_ylim((-0.5, 1.0))
    ax.set_yticks(np.arange(-0.4, 1.1, 0.2))

    handles, labels = ax.get_legend_handles_labels()

    order = [0, 3, 1, 4, 2]

    handles = [handles[i] for i in order if i < len(handles)]
    labels = [labels[i] for i in order if i < len(labels)]

    leg = ax.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        frameon=True,
        framealpha=0.4,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    for t in leg.get_texts():
        t.set_fontstyle("italic")

    # ax.legend(frameon=True, framealpha=0.4, loc="upper left")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_figures(fig_dir: Path) -> None:
    # Derive unique reward methods
    # exp_labels = sorted({_exp_label_from_path(p) for p in path_list})
    exp_labels = sorted(
        {_exp_label_from_path(p) for p in path_list}, key=_method_sort_key
    )
    method_order = exp_labels

    method_order = exp_labels

    coll_rates: Dict[str, Dict[str, List[float]]] = {}
    avg_speeds: Dict[str, Dict[str, List[float]]] = {}
    avg_rews: Dict[str, Dict[str, List[float]]] = {}
    train_reward_curves: Dict[str, List[List[float]]] = {m: [] for m in method_order}
    cbf_activation_degree: Dict[str, Dict[str, List[float]]] = {}
    cbf_activation_rates: Dict[str, Dict[str, List[float]]] = {}
    task_num_tries: Dict[str, Dict[str, List[float]]] = {}
    task_success_times: Dict[str, Dict[str, List[float]]] = {}
    task_success_rate: Dict[str, Dict[str, List[float]]] = {}

    # Training reward curves (independent of scenario)
    for model_dir in path_list:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            continue

        method_label = _exp_label_from_path(str(model_dir))

        curve = _load_episode_reward_mean_list(model_dir)
        if curve is None:
            continue

        train_reward_curves[method_label].append(curve)

    for scenario_type, n_agents in scenario_specs:
        coll_rates[scenario_type] = {m: [] for m in method_order}
        avg_speeds[scenario_type] = {m: [] for m in method_order}
        avg_rews[scenario_type] = {m: [] for m in method_order}
        cbf_activation_degree[scenario_type] = {m: [] for m in method_order}
        cbf_activation_rates[scenario_type] = {m: [] for m in method_order}
        task_num_tries[scenario_type] = {m: [] for m in method_order}
        task_success_times[scenario_type] = {m: [] for m in method_order}
        task_success_rate[scenario_type] = {m: [] for m in method_order}

        # Each entry in path_list is a model folder (often .../h*/seed*)
        for model_dir in path_list:
            method_label = _exp_label_from_path(str(model_dir))

            model_dir = Path(model_dir)
            if not model_dir.is_dir():
                continue

            # Collect all out_td files for all models in this folder:
            # you used: out_td_{model_tag}_{scenario...}.td
            for eval_seed in random_seed_list:
                name_suffix = get_name_suffix(scenario_type, n_agents, eval_seed)

                # Match any model_tag prefix
                pattern = f"out_td_{name_suffix}.td"
                matches = list(model_dir.glob(pattern))
                if len(matches) == 0:
                    continue

                for out_td_path in matches:
                    try:
                        out_td = _load_out_td(out_td_path)
                        cr = compute_collision_rate_percent(out_td)
                        vbar = compute_average_speed(out_td)
                        rew = compute_total_reward(out_td)
                        cbf_act_deg, cbf_act_times = compute_cbf_activation_percentage(
                            out_td
                        )
                        task_perf_path = (
                            model_dir / f"task_performance_{name_suffix}.json"
                        )
                        tp = _load_task_performance(task_perf_path)

                        coll_rates[scenario_type][method_label].append(cr)
                        avg_speeds[scenario_type][method_label].append(vbar)
                        avg_rews[scenario_type][method_label].append(rew)
                        cbf_activation_degree[scenario_type][method_label].append(
                            cbf_act_deg
                        )
                        cbf_activation_rates[scenario_type][method_label].append(
                            cbf_act_times
                        )

                        task_num_tries[scenario_type][method_label].append(
                            float(tp["num_task_tries"])
                        )
                        task_success_times[scenario_type][method_label].append(
                            float(tp["task_success_times"])
                        )
                        task_success_rate[scenario_type][method_label].append(
                            float(100 * tp["task_success_times"] / tp["num_task_tries"])
                        )

                    except Exception as e:
                        print(f"[WARNING] Failed on {out_td_path}: {e}")

        # Per-scenario plots
        out_coll_pdf = fig_dir / "fig_collision_rate.pdf"
        out_speed_pdf = fig_dir / "fig_speed.pdf"
        out_rew_pdf = fig_dir / "fig_total_reward.pdf"
        out_train_rew_pdf = fig_dir / "fig_training_reward_curve.pdf"
        out_cbf_act_degree_pdf = fig_dir / "fig_cbf_activation_degree.pdf"
        out_cbf_act_rate_pdf = fig_dir / "fig_cbf_activation_rate.pdf"
        out_task_tries_pdf = fig_dir / "fig_task_attempts_total.pdf"
        out_success_task_tries_pdf = fig_dir / "fig_task_attempts_success.pdf"
        out_task_success_pdf = fig_dir / "fig_task_success_rate.pdf"

        log_path = fig_dir / f"{scenario_type}_summary.txt"

        # Use CBF (our) as reference method
        ref_method = "CBF (our)" if ("CBF (our)" in method_order) else None

        ref_stats = None
        if ref_method is not None:
            cr_ref = coll_rates[scenario_type][ref_method]
            v_ref = avg_speeds[scenario_type][ref_method]
            rew_ref = avg_rews[scenario_type][ref_method]
            deg_ref = cbf_activation_degree[scenario_type][ref_method]
            t_ref = cbf_activation_rates[scenario_type][ref_method]
            task_num_tries_ref = task_num_tries[scenario_type][ref_method]
            task_success_times_ref = task_success_times[scenario_type][ref_method]
            task_success_rate_ref = task_success_rate[scenario_type][ref_method]

            if len(cr_ref) > 0:
                ref_stats = {
                    "cr": sum(cr_ref) / len(cr_ref),
                    "v": sum(v_ref) / len(v_ref),
                    "rew": sum(rew_ref) / len(rew_ref),
                    "deg": sum(deg_ref) / len(deg_ref),
                    "t": sum(t_ref) / len(t_ref),
                    "task_num_tries": sum(task_num_tries_ref) / len(task_num_tries_ref),
                    "task_success_times": sum(task_success_times_ref)
                    / len(task_success_times_ref),
                    "task_success_rate": sum(task_success_rate_ref)
                    / len(task_success_rate_ref),
                }

        with open(log_path, "w", encoding="utf-8") as f:
            header = f"--- Scenario: {scenario_type} ---"
            print(header)
            f.write(header + "\n")

            if ref_stats is None:
                warn = '[WARNING] Reference method "distance" not found or has no data; improvements will not be logged.'
                print(warn)
                f.write(warn + "\n")

            for method in method_order:
                cr_list = coll_rates[scenario_type][method]
                vbar_list = avg_speeds[scenario_type][method]
                rew_list = avg_rews[scenario_type][method]
                cbf_act_deg_list = cbf_activation_degree[scenario_type][method]
                cbf_act_time_list = cbf_activation_rates[scenario_type][method]
                task_num_tries_ref_list = task_num_tries[scenario_type][method]
                task_success_times_ref_list = task_success_times[scenario_type][method]
                task_success_rate_ref_list = task_success_rate[scenario_type][method]

                if len(cr_list) == 0:
                    line = f"[INFO] Method: {method}: No data"
                    print(line)
                    f.write(line + "\n")
                    continue

                cr_avg = sum(cr_list) / len(cr_list)
                vbar_avg = sum(vbar_list) / len(vbar_list)
                rew_avg = sum(rew_list) / len(rew_list)
                cbf_act_deg_avg = sum(cbf_act_deg_list) / len(cbf_act_deg_list)
                cbf_act_time_avg = sum(cbf_act_time_list) / len(cbf_act_time_list)
                task_num_tries_ref_avg = sum(task_num_tries_ref_list) / len(
                    task_num_tries_ref_list
                )
                task_success_times_ref_avg = sum(task_success_times_ref_list) / len(
                    task_success_times_ref_list
                )
                task_success_rate_ref_avg = sum(task_success_rate_ref_list) / len(
                    task_success_rate_ref_list
                )

                line = (
                    f"[INFO] Method: {method}: "
                    # f"Collision Rate: {cr_avg:.3f} %, "
                    # f"Avg Speed: {vbar_avg:.3f} m/s, "
                    f"Avg Total Reward: {rew_avg:.3f}, "
                    f"CBF Activation Degree: {cbf_act_deg_avg:.3f} %, "
                    # f"CBF Activation Rate: {cbf_act_time_avg:.3f} %, "
                    # f"Task Num Tries: {task_num_tries_ref_avg:.3f}, "
                    f"Task Success Times: {task_success_times_ref_avg:.3f}, "
                    f"Task Success Rate: {task_success_rate_ref_avg:.3f}."
                )
                print(line)
                f.write(line + "\n")

                # Log percentage improvement/degradation of CBF (our) vs each other method
                if ref_stats is not None and method != ref_method:

                    def _pct_cbf_better_higher_is_better(
                        cbf: float, other: float
                    ) -> float:
                        # positive if CBF higher than other
                        if abs(other) < 1e-12:
                            return float("nan")
                        return 100.0 * (cbf - other) / abs(other)

                    def _pct_cbf_better_lower_is_better(
                        cbf: float, other: float
                    ) -> float:
                        # positive if CBF lower than other
                        if abs(other) < 1e-12:
                            return float("nan")
                        return 100.0 * (other - cbf) / abs(other)

                    # CBF values (reference)
                    cbf_cr = ref_stats["cr"]
                    cbf_v = ref_stats["v"]
                    cbf_rew = ref_stats["rew"]
                    cbf_deg = ref_stats["deg"]
                    cbf_t = ref_stats["t"]
                    cbf_task_num_tries = ref_stats["task_num_tries"]
                    cbf_task_success_times = ref_stats["task_success_times"]
                    cbf_task_success_rate = ref_stats["task_success_rate"]

                    # Percentage "CBF vs method" (positive = CBF better)
                    p_cr = _pct_cbf_better_lower_is_better(cbf_cr, cr_avg)
                    p_v = _pct_cbf_better_higher_is_better(cbf_v, vbar_avg)
                    p_rew = _pct_cbf_better_higher_is_better(cbf_rew, rew_avg)
                    p_deg = _pct_cbf_better_lower_is_better(cbf_deg, cbf_act_deg_avg)
                    p_t = _pct_cbf_better_lower_is_better(cbf_t, cbf_act_time_avg)
                    p_task_num_tries = _pct_cbf_better_lower_is_better(
                        cbf_task_num_tries, task_num_tries_ref_avg
                    )
                    p_task_success_times = _pct_cbf_better_lower_is_better(
                        cbf_task_success_times, task_success_times_ref_avg
                    )
                    p_task_success_rate = _pct_cbf_better_lower_is_better(
                        cbf_task_success_rate, task_success_rate_ref_avg
                    )

                    ref_name = _wrap_method_label(ref_method, is_line_break=False)
                    method_name = _wrap_method_label(method, is_line_break=False)

                    line_imp = (
                        f"    {ref_name} vs {method_name}: "
                        # f"Collision Rate {p_cr:+.1f} %, "
                        # f"Avg Speed {p_v:+.1f} %, "
                        f"Avg Total Reward {p_rew:+.1f} %, "
                        f"CBF Activation Degree {p_deg:+.1f} %, "
                        # f"CBF Activation Rate {p_t:+.1f} %, "
                        # f"Task Attempts {p_task_num_tries:+.1f} %, "
                        f"Task Success Times {p_task_success_times:+.1f} %, "
                        f"Task Success Rate {p_task_success_rate:+.1f} %"
                    )

                    print(line_imp)
                    f.write(line_imp + "\n")

        save_boxplot_pdf(
            data_by_method=coll_rates[scenario_type],
            method_order=method_order,
            ylabel=r"Collision Rate [$\%$]",
            out_pdf=out_coll_pdf,
            unit="%",
            ylim=(0, 1),
        )

        save_boxplot_pdf(
            data_by_method=avg_speeds[scenario_type],
            method_order=method_order,
            ylabel=r"Speed [$\mathrm{m/s}$]",
            out_pdf=out_speed_pdf,
            unit=" m/s",
            ylim=(0, 0.8),
        )

        save_boxplot_pdf(
            data_by_method=avg_rews[scenario_type],
            method_order=method_order,
            ylabel=r"Total Reward",
            out_pdf=out_rew_pdf,
            unit="",
            ylim=(0, 10),
        )

        save_reward_curve_pdf(
            curves_by_method=train_reward_curves,
            method_order=method_order,
            out_pdf=out_train_rew_pdf,
            ylabel="Episode Reward",
            frames_per_iteration=FRAME_PER_ITERATION,
        )

        save_boxplot_pdf(
            data_by_method=cbf_activation_degree[scenario_type],
            method_order=method_order,
            ylabel=r"CBF Act. Degree [$\%$]",
            out_pdf=out_cbf_act_degree_pdf,
            unit="%",
            ylim=(0, 20),
        )

        save_boxplot_pdf(
            data_by_method=cbf_activation_rates[scenario_type],
            method_order=method_order,
            ylabel=r"CBF Act. Ratio [$\%$]",
            out_pdf=out_cbf_act_rate_pdf,
            unit="%",
            ylim=(0, 20),
        )

        save_boxplot_pdf(
            data_by_method=task_num_tries[scenario_type],
            method_order=method_order,
            ylabel="\#Task Attempts",
            out_pdf=out_task_tries_pdf,
            unit="",
            ylim=(0, 50),
        )

        save_boxplot_pdf(
            data_by_method=task_success_times[scenario_type],
            method_order=method_order,
            ylabel="\#Successful Task Attempts",
            out_pdf=out_success_task_tries_pdf,
            unit="",
            ylim=(0, 50),
        )

        save_boxplot_pdf(
            data_by_method=task_success_rate[scenario_type],
            method_order=method_order,
            ylabel=r"Task Success Rate [$\%$]",
            out_pdf=out_task_success_pdf,
            unit="%",
            ylim=(0, 110),
        )

        print(f"[INFO] Saved: {out_coll_pdf}")
        print(f"[INFO] Saved: {out_speed_pdf}")
        print(f"[INFO] Saved: {out_rew_pdf}")
        print(f"[INFO] Saved: {out_train_rew_pdf}")
        print(f"[INFO] Saved: {out_cbf_act_degree_pdf}")
        print(f"[INFO] Saved: {out_cbf_act_rate_pdf}")
        print(f"[INFO] Saved: {out_task_tries_pdf}")
        print(f"[INFO] Saved: {out_task_success_pdf}")


if __name__ == "__main__":
    scenario_specs = [
        ("cpm_mixed", 4),
        # ("cpm_entire", 10),
        # ("interchange_1", 8),
        # ("interchange_3", 10),
        # ("intersection_4", 8),
        # ("intersection_5", 10),
        # ("intersection_6", 10),
        # ("on_ramp_1", 6),
        # ("roundabout_2", 10),
    ]
    if len(sys.argv) == 2:
        scenario_specs = [scenario_specs[int(sys.argv[1])]]
    elif len(sys.argv) > 2:
        scenario_specs = scenario_specs[int(sys.argv[1]) : int(sys.argv[2])]

    print(f"[INFO] Evaluating scenarios: {scenario_specs}")

    n_agents_list = [n_agents for _, n_agents in scenario_specs]

    print(n_agents_list)

    policy_parent_folder = "checkpoints/itsc26_new/cpm_mixed_do_not_apply_cbf_action/"
    # policy_parent_folder = "checkpoints/itsc26_new/cpm_mixed_apply_cbf_action/"
    # policy_parent_folder = "checkpoints/itsc26_new/cpm_mixed_apply_cbf_action_final_model/"
    # policy_parent_folder = "checkpoints/itsc26_new/cpm_mixed_do_not_apply_cbf_action_final_model/"

    filtered_path_list = []

    for root, dirs, files in os.walk(policy_parent_folder):
        if any(f.endswith(".pth") for f in files):
            if "reward_progress10.0" in root:
                filtered_path_list.append(root)

    path_list = filtered_path_list

    # path_list must contains one of the method in rew_method_list
    rew_method_list = ["cbf/", "distance/", "distance_sparse/", "ttc/", "ttc_sparse/"]
    path_list = [
        path for path in path_list if any(method in path for method in rew_method_list)
    ]

    random_seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # -----------------------------
    # Metric extraction from out_td
    # -----------------------------
    # Two reasonable collision definitions. Default matches your description.
    # - "per_timestep_any_agent": mean over time of [any collision among agents at that time]
    # - "per_agent_timestep": mean over all agent-time pairs (and envs)
    COLLISION_MODE = "per_agent_timestep"  # change if needed, one of {"per_timestep_any_agent", "per_agent_timestep"}
    run_evaluations()

    fig_dir = Path(policy_parent_folder) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_figures(fig_dir)
