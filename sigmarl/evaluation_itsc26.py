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
import sys

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.family": "serif",
        "text.usetex": is_latex_available(),
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


def _seed_from_path(p: str) -> Optional[int]:
    """
    Extract seed index from a path containing '/seed3', etc.
    """
    m = re.search(r"/seed(\d+)", p)
    return int(m.group(1)) if m else None


def _h_value_from_key(h_key: str) -> float:
    """
    'hNone' -> -inf (so baseline comes first within the same reward_progress)
    'h0.10' -> 0.10
    """
    if h_key == "hNone":
        return float("-inf")
    m = re.match(r"h([0-9]+(?:\.[0-9]+)?)", h_key)
    return float(m.group(1)) if m else float("inf")


def _path_sort_key(p: str):
    """
    Primary: reward_progress (ascending)
    Secondary: baseline first, then increasing h
    Tertiary: seed (ascending)
    """
    rp = _reward_progress_from_path(p)
    h_key = _method_key_from_path(p)
    h_val = _h_value_from_key(h_key)
    seed = _seed_from_path(p)

    # Put missing values at the end (should not happen for your paths)
    rp_key = rp if rp is not None else float("inf")
    seed_key = seed if seed is not None else 10**9

    # h_val already puts baseline first (-inf)
    return (rp_key, h_val, seed_key)


def _method_key_from_path(p: str) -> str:
    """
    Extract 'h0.04', 'hNone', etc. from any path containing that segment.
    Falls back to 'unknown' if not found.
    """
    m = re.search(r"(/h[^/]+)", p)
    return m.group(1)[1:] if m else "unknown"


def _method_label_from_key(h_key: str) -> str:
    """
    Map method key to a label for figures.
    'hNone' -> 'baseline'
    'h0.04' -> 'h=0.04'
    """
    if h_key == "hNone":
        return "baseline"
    if h_key.startswith("h"):
        return f"h={h_key[1:]}"
    return h_key


def _reward_progress_from_path(p: str) -> Optional[float]:
    """
    Extract reward_progress value from a path containing 'reward_progress2.5', etc.
    Returns None if not found.
    """
    m = re.search(r"reward_progress([0-9]+(?:\.[0-9]+)?)", p)
    return float(m.group(1)) if m else None


def _exp_label_from_path(p: str) -> str:
    """
    Build legend label that includes both h and reward_progress.
    Examples:
      hNone + reward_progress2.5 -> "baseline, rp=2.5"
      h0.10 + reward_progress5.0 -> "h=0.10, rp=5.0"
    """
    h_key = _method_key_from_path(p)
    h_label = _method_label_from_key(h_key)
    rp = _reward_progress_from_path(p)
    if rp is None:
        return h_label
    return f"{h_label}, rp={rp:g}"


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
                parameters.rew_method = "sparse"  # Reward method: {"default", "cbf_constraint", "cbf_qp", "ttc", "sparse"}
                parameters.is_real_time_rendering = False
                parameters.is_save_eval_results = True
                parameters.is_load_model = True

                parameters.is_using_cbf_training = False
                parameters.is_using_cbf_testing = True
                parameters.is_solve_qp = True
                parameters.is_apply_cbf_action = False

                parameters.is_load_final_model = "final" in train_path
                parameters.is_load_out_td = False
                parameters.max_steps = 600
                parameters.num_vmas_envs = 1

                parameters.scenario_type = scenario_type
                parameters.n_agents = n_agents

                parameters.is_save_simulation_video = True
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
                out_td_path = os.path.join(train_path, out_td_filename)

                if os.path.exists(out_td_path):
                    print(f"[INFO] Skipping existing output: {out_td_path}")
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


# -----------------------------
# Plotting
# -----------------------------
def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)


def save_boxplot_pdf(
    data_by_method: Dict[str, List[float]],
    method_order: List[str],
    ylabel: str,
    out_pdf: Path,
    title: Optional[str] = None,
    ylim: Optional[tuple] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 2.6), constrained_layout=True)

    data = [data_by_method.get(m, []) for m in method_order]

    bp = ax.boxplot(
        data,
        tick_labels=method_order,
        showfliers=True,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=False,  # mean as a point marker
        meanprops=dict(marker="o", markersize=4, markeredgewidth=0.8),
        medianprops=dict(linewidth=1.4),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    # bp = ax.boxplot(
    #     data,
    #     tick_labels=method_order,
    #     showfliers=True,
    #     widths=0.6,
    #     patch_artist=True,
    #     showmeans=True,
    #     meanline=False,
    #     meanprops=dict(marker="o", markersize=4, markeredgewidth=0.8),
    #     medianprops=dict(linewidth=0),  # hide median
    # )

    if ylim is not None:
        ax.set_ylim(ylim)

    # Use matplotlib default color cycle for boxes (no explicit color spec)
    # but set a light alpha for better print readability.
    for b in bp["boxes"]:
        b.set_alpha(0.75)

    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.tick_params(axis="x", rotation=20)
    _style_axis(ax)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_reward_curve_pdf(
    curves_by_method: Dict[str, List[List[float]]],
    method_order: List[str],
    out_pdf: Path,
    title: Optional[str] = None,
    ylabel: str = "Episode Reward",
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 6.6), constrained_layout=True)

    for method in method_order:
        curves = curves_by_method.get(method, [])
        x, mean, std = _aggregate_curves(
            curves,
            smoothing="ema",  # "none" | "ema" | "rolling"
            ema_alpha=0.02,  # try 0.02–0.08; the smaller the smoother
        )
        if x is None:
            continue

        # One method: plot mean line and +/- std band
        linestyle = "--" if method.startswith("baseline") else "-"
        line = ax.plot(
            x,
            mean,
            linewidth=1.0,
            linestyle=linestyle,
            label=method,
        )[0]

        # line = ax.plot(x, mean, linewidth=1.0, label=method)[0]
        c = line.get_color()
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0.0)

    ax.set_xlabel("Episode Index")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    _style_axis(ax)
    ax.legend(frameon=False, ncol=1)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# def plot_figures(fig_dir: Path) -> None:
#     # Derive unique experiment labels: (h, reward_progress)
#     exp_labels = sorted({_exp_label_from_path(p) for p in path_list})

#     # Put baseline first if present; keep remaining order stable
#     if any(lbl.startswith("baseline") for lbl in exp_labels):
#         exp_labels = (
#             [lbl for lbl in exp_labels if lbl.startswith("baseline")]
#             + [lbl for lbl in exp_labels if not lbl.startswith("baseline")]
#         )

#     # Optional: sort non-baseline by numeric h then rp (more readable)
#     def _sort_key(lbl: str):
#         # lbl example: "h=0.10, rp=2.5" or "baseline, rp=2.5"
#         if lbl.startswith("baseline"):
#             return (-1.0, -1.0)
#         mh = re.search(r"h=([0-9]+(?:\.[0-9]+)?)", lbl)
#         mr = re.search(r"rp=([0-9]+(?:\.[0-9]+)?)", lbl)
#         h = float(mh.group(1)) if mh else 1e9
#         rp = float(mr.group(1)) if mr else 1e9
#         return (h, rp)

#     baseline_part = [lbl for lbl in exp_labels if lbl.startswith("baseline")]
#     others_part = sorted([lbl for lbl in exp_labels if not lbl.startswith("baseline")], key=_sort_key)
#     method_order = baseline_part + others_part
def plot_figures(fig_dir: Path) -> None:
    # Derive unique experiment labels: (h, reward_progress)
    exp_labels = list({_exp_label_from_path(p) for p in path_list})

    def _label_sort_key(lbl: str):
        # lbl example: "h=0.10, rp=2.5" or "baseline, rp=2.5"
        mr = re.search(r"rp=([0-9]+(?:\.[0-9]+)?)", lbl)
        rp = float(mr.group(1)) if mr else float("inf")

        if lbl.startswith("baseline"):
            # baseline first within each rp
            return (rp, 0, float("-inf"))

        mh = re.search(r"h=([0-9]+(?:\.[0-9]+)?)", lbl)
        h = float(mh.group(1)) if mh else float("inf")
        return (rp, 1, h)

    method_order = sorted(exp_labels, key=_label_sort_key)

    coll_rates: Dict[str, Dict[str, List[float]]] = {}
    avg_speeds: Dict[str, Dict[str, List[float]]] = {}
    avg_rews: Dict[str, Dict[str, List[float]]] = {}
    train_reward_curves: Dict[str, List[List[float]]] = {m: [] for m in method_order}
    cbf_activation_degree: Dict[str, Dict[str, List[float]]] = {}
    cbf_activation_times: Dict[str, Dict[str, List[float]]] = {}

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
        cbf_activation_times[scenario_type] = {m: [] for m in method_order}

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

                        coll_rates[scenario_type][method_label].append(cr)
                        avg_speeds[scenario_type][method_label].append(vbar)
                        avg_rews[scenario_type][method_label].append(rew)
                        cbf_activation_degree[scenario_type][method_label].append(
                            cbf_act_deg
                        )
                        cbf_activation_times[scenario_type][method_label].append(
                            cbf_act_times
                        )
                    except Exception as e:
                        print(f"[WARNING] Failed on {out_td_path}: {e}")

        # Per-scenario plots
        scenario_tag = scenario_type
        out_coll_pdf = fig_dir / f"{scenario_tag}_collision_rate.pdf"
        out_speed_pdf = fig_dir / f"{scenario_tag}_avg_speed.pdf"
        out_rew_pdf = fig_dir / f"{scenario_tag}_avg_total_reward.pdf"
        out_train_rew_pdf = fig_dir / "training_reward_curve.pdf"
        out_cbf_act_degree_pdf = fig_dir / f"{scenario_tag}_cbf_activation_degree.pdf"
        out_cbf_act_time_pdf = fig_dir / f"{scenario_tag}_cbf_activation_time.pdf"

        title_coll = f"{scenario_type}"
        title_speed = f"{scenario_type}"

        log_path = fig_dir / f"{scenario_type}_summary.txt"

        # 1) Find baseline averages for this scenario (same rp if multiple exist).
        baseline_method = None
        for m in method_order:
            if m.startswith("baseline"):
                baseline_method = m
                break

        baseline_stats = None
        if baseline_method is not None:
            cr_list_b = coll_rates[scenario_type][baseline_method]
            vbar_list_b = avg_speeds[scenario_type][baseline_method]
            rew_list_b = avg_rews[scenario_type][baseline_method]
            deg_list_b = cbf_activation_degree[scenario_type][baseline_method]
            time_list_b = cbf_activation_times[scenario_type][baseline_method]

            if len(cr_list_b) > 0:
                baseline_stats = {
                    "cr": sum(cr_list_b) / len(cr_list_b),
                    "v": sum(vbar_list_b) / len(vbar_list_b),
                    "rew": sum(rew_list_b) / len(rew_list_b),
                    "deg": sum(deg_list_b) / len(deg_list_b),
                    "t": sum(time_list_b) / len(time_list_b),
                }

        with open(log_path, "w", encoding="utf-8") as f:
            header = f"--- Scenario: {scenario_type} ---"
            print(header)
            f.write(header + "\n")

            if baseline_stats is None:
                warn = "[WARNING] Baseline not found or has no data; improvements will not be logged."
                print(warn)
                f.write(warn + "\n")

            for method in method_order:
                cr_list = coll_rates[scenario_type][method]
                vbar_list = avg_speeds[scenario_type][method]
                rew_list = avg_rews[scenario_type][method]
                cbf_act_deg_list = cbf_activation_degree[scenario_type][method]
                cbf_act_time_list = cbf_activation_times[scenario_type][method]

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

                line = (
                    f"[INFO] Method: {method}: Collision Rate: {cr_avg:.3f} %, "
                    f"Avg Speed: {vbar_avg:.3f} m/s, Avg Total Reward: {rew_avg:.3f}, "
                    f"CBF Activation Degree: {cbf_act_deg_avg:.3f} %, CBF Activation Time: {cbf_act_time_avg:.3f} %"
                )
                print(line)
                f.write(line + "\n")

                # 2) Log improvements vs baseline (if available and method is not baseline)
                if baseline_stats is not None and not method.startswith("baseline"):
                    d_cr = baseline_stats["cr"] - cr_avg  # positive = fewer collisions
                    d_v = vbar_avg - baseline_stats["v"]  # positive = faster
                    d_rew = rew_avg - baseline_stats["rew"]  # positive = higher reward
                    d_deg = (
                        baseline_stats["deg"] - cbf_act_deg_avg
                    )  # positive = fewer activations
                    d_t = (
                        baseline_stats["t"] - cbf_act_time_avg
                    )  # positive = fewer activations

                    # Relative improvements (%), guarded for divide-by-zero
                    def rel(delta, base):
                        return (
                            100.0 * delta / base if abs(base) > 1e-12 else float("nan")
                        )

                    line_imp = (
                        f"    Improvement vs {baseline_method}: "
                        f"ΔCollision Rate={d_cr:+.3f} % (rel {rel(d_cr, baseline_stats['cr']):+.2f} %), "
                        f"ΔAvg Speed={d_v:+.3f} m/s (rel {rel(d_v, baseline_stats['v']):+.2f} %), "
                        f"ΔAvg Total Reward={d_rew:+.3f} (rel {rel(d_rew, baseline_stats['rew']):+.2f} %), "
                        f"ΔCBF Activation Degree={d_deg:+.3f} % (rel {rel(d_deg, baseline_stats['deg']):+.2f} %), "
                        f"ΔCBF Activation Time={d_t:+.3f} % (rel {rel(d_t, baseline_stats['t']):+.2f} %)"
                    )
                    print(line_imp)
                    f.write(line_imp + "\n")

        save_boxplot_pdf(
            data_by_method=coll_rates[scenario_type],
            method_order=method_order,
            ylabel=r"Collision Rate [$\%$]",
            out_pdf=out_coll_pdf,
            title=title_coll,
            ylim=(0, 2),
        )

        save_boxplot_pdf(
            data_by_method=avg_speeds[scenario_type],
            method_order=method_order,
            ylabel=r"Avg. Speed [$\mathrm{m/s}$]",
            out_pdf=out_speed_pdf,
            title=title_speed,
            ylim=(0, 0.8),
        )

        save_boxplot_pdf(
            data_by_method=avg_rews[scenario_type],
            method_order=method_order,
            ylabel=r"Avg. Total Reward",
            out_pdf=out_rew_pdf,
            title=title_speed,
            ylim=(0, 25),
        )

        save_reward_curve_pdf(
            curves_by_method=train_reward_curves,
            method_order=method_order,
            out_pdf=out_train_rew_pdf,
            title="Training Reward",
            ylabel="Episode Reward",
        )

        save_boxplot_pdf(
            data_by_method=cbf_activation_degree[scenario_type],
            method_order=method_order,
            ylabel=r"CBF Activation Rel. Degree [$\%$]",
            out_pdf=out_cbf_act_degree_pdf,
            # title="CBF Activation (Rel. Mean Abs. Diff.)",
            ylim=(0, 20),
        )

        save_boxplot_pdf(
            data_by_method=cbf_activation_times[scenario_type],
            method_order=method_order,
            ylabel=r"CBF Activation Ratio [$\%$]",
            out_pdf=out_cbf_act_time_pdf,
            ylim=(0, 20),
        )

        print(f"[INFO] Saved: {out_coll_pdf}")
        print(f"[INFO] Saved: {out_speed_pdf}")
        print(f"[INFO] Saved: {out_rew_pdf}")
        print(f"[INFO] Saved: {out_train_rew_pdf}")
        print(f"[INFO] Saved: {out_cbf_act_degree_pdf}")
        print(f"[INFO] Saved: {out_cbf_act_time_pdf}")


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

    policy_parent_folder = "checkpoints/itsc26/cpm_mixed_apply_cbf_action/"

    filtered_path_list = []

    for root, dirs, files in os.walk(policy_parent_folder):
        if any(f.endswith(".pth") for f in files):
            if "reward_progress" in root and "reward_progress10.0" in root:
                # if "reward_progress" in root and "hNone" in root:
                # if "reward_progress" in root and "h0.10" in root:
                filtered_path_list.append(root)

    # Optional: sort for stable order
    filtered_path_list = sorted(filtered_path_list, key=_path_sort_key)

    print(f"Found {len(filtered_path_list)} folders containing .pth files.")

    path_list = filtered_path_list

    # path_list = [
    #     'checkpoints/itsc26/cpm_mixed/h0.12/seed1',
    #     'checkpoints/itsc26/cpm_mixed/h0.12/seed2',
    #     'checkpoints/itsc26/cpm_mixed/h0.12/seed3',
    #     'checkpoints/itsc26/cpm_mixed/h0.12/seed4',
    #     'checkpoints/itsc26/cpm_mixed/h0.12/seed5',
    # ]

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
