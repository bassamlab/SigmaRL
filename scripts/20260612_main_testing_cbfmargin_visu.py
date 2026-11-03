#!/usr/bin/env python3
"""Visualize CBF-margin evaluation TensorDicts as 2D parameter heatmaps."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sigmarl.helper_training import save_evaluation_summary


DEFAULT_BASE_DIR = Path(
    "checkpoints/20260617-cbfmargin/" "is_using_centralized_cbfFalse"
)
TD_SEED_PATTERN = re.compile(r"_seed(?P<seed>\d+)\.td$")


def get_tensor(out_td, key):
    try:
        return out_td.get(key)
    except Exception:
        return None


def squeeze_scalar_field(tensor):
    if tensor is None:
        return None
    tensor = torch.as_tensor(tensor).detach().cpu()
    while tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def parse_prefixed_float(path: Path, prefix: str) -> Optional[float]:
    for part in path.parts:
        if part.startswith(prefix):
            value = part[len(prefix) :]
            try:
                return float(value)
            except ValueError:
                return None
    return None


def parse_eval_seed(path: Path) -> Optional[int]:
    match = TD_SEED_PATTERN.search(path.name)
    if match is None:
        return None
    return int(match.group("seed"))


def total_reward(out_td) -> float:
    reward_total = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "rew_total"))
    )
    if reward_total is not None:
        return float(reward_total.float().sum().item())

    reward_parts = []
    for key in (
        ("agents", "info", "rew_reach_goal"),
        ("agents", "info", "rew_collide_other_agents"),
        ("agents", "info", "rew_collide_lane"),
        ("agents", "info", "rew_deadlock"),
    ):
        tensor = squeeze_scalar_field(get_tensor(out_td, key))
        if tensor is not None:
            reward_parts.append(tensor.float())

    if not reward_parts:
        return math.nan
    return float(torch.stack(reward_parts, dim=0).sum(dim=0).sum().item())


def event_rate_percent(out_td, key) -> float:
    events = squeeze_scalar_field(get_tensor(out_td, key))
    if events is None:
        return math.nan
    return float(events.bool().float().mean().item() * 100.0)


def collision_rate_percent(out_td) -> float:
    collision_agents = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "is_collision_with_agents"))
    )
    collision_lanelets = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "is_collision_with_lanelets"))
    )

    if collision_agents is None and collision_lanelets is None:
        return math.nan
    if collision_agents is None:
        collision = collision_lanelets.bool()
    elif collision_lanelets is None:
        collision = collision_agents.bool()
    else:
        collision = torch.logical_or(collision_agents.bool(), collision_lanelets.bool())

    return float(collision.float().mean().item() * 100.0)


def parameter_independent_reward(out_td, td_path: Path) -> float:
    name_suffix = td_path.stem.removeprefix("out_td_")
    summary = save_evaluation_summary(
        out_td=out_td,
        output_dir=td_path.parent,
        name_suffix=name_suffix,
    )
    return float(summary["parameter_independent_reward_per_agent_per_minute"])


def load_metrics(td_path: Path) -> dict[str, float]:
    out_td = torch.load(td_path, weights_only=False, map_location="cpu")
    return {
        "total_reward": total_reward(out_td),
        "parameter_independent_reward": parameter_independent_reward(out_td, td_path),
        "collision_rate": collision_rate_percent(out_td),
        "deadlock_rate": event_rate_percent(
            out_td, ("agents", "info", "is_deadlocked")
        ),
        "goal_reaching_rate": event_rate_percent(
            out_td, ("agents", "info", "is_reach_goal")
        ),
    }


def discover_records(args: argparse.Namespace) -> list[dict[str, float | int | Path]]:
    pattern = f"**/out_td_{args.scenario_type}_agents{args.n_agents}_seed*.td"
    records = []
    for td_path in sorted(args.base_dir.glob(pattern)):
        x_value = parse_prefixed_float(
            td_path.relative_to(args.base_dir), args.x_prefix
        )
        y_value = parse_prefixed_float(
            td_path.relative_to(args.base_dir), args.y_prefix
        )
        eval_seed = parse_eval_seed(td_path)
        if x_value is None or y_value is None or eval_seed is None:
            print(f"[WARNING] Skipping unparseable TD path: {td_path}")
            continue
        metrics = load_metrics(td_path)
        records.append(
            {
                "td_path": td_path,
                "x_value": x_value,
                "y_value": y_value,
                "eval_seed": eval_seed,
                **metrics,
            }
        )
    return records


def aggregate_records(records: list[dict[str, float | int | Path]]) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (float(record["x_value"]), float(record["y_value"]))
        for metric in (
            "total_reward",
            "parameter_independent_reward",
            "collision_rate",
            "deadlock_rate",
            "goal_reaching_rate",
        ):
            grouped[key][metric].append(float(record[metric]))

    aggregated = {}
    for key, metric_values in grouped.items():
        aggregated[key] = {
            metric: finite_mean(values) for metric, values in metric_values.items()
        }
        aggregated[key]["n_seeds"] = len(metric_values["total_reward"])
    return aggregated


def save_summary_csv(
    *,
    base_dir: Path,
    aggregated: dict,
    x_values: list[float],
    y_values: list[float],
    x_name: str,
    y_name: str,
) -> Path:
    csv_path = base_dir / "20260612_cbfmargin_metrics_summary.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                x_name,
                y_name,
                "n_seeds",
                "total_reward",
                "parameter_independent_reward",
                "collision_rate",
                "deadlock_rate",
                "goal_reaching_rate",
            ],
        )
        writer.writeheader()
        for y_value in y_values:
            for x_value in x_values:
                values = aggregated.get((x_value, y_value), {})
                writer.writerow(
                    {
                        x_name: x_value,
                        y_name: y_value,
                        "n_seeds": values.get("n_seeds", 0),
                        "total_reward": values.get("total_reward", math.nan),
                        "parameter_independent_reward": values.get(
                            "parameter_independent_reward", math.nan
                        ),
                        "collision_rate": values.get("collision_rate", math.nan),
                        "deadlock_rate": values.get("deadlock_rate", math.nan),
                        "goal_reaching_rate": values.get(
                            "goal_reaching_rate", math.nan
                        ),
                    }
                )
    return csv_path


def make_grid(
    *,
    aggregated: dict,
    metric: str,
    x_values: list[float],
    y_values: list[float],
) -> list[list[float]]:
    grid = []
    for y_value in y_values:
        row = []
        for x_value in x_values:
            row.append(aggregated.get((x_value, y_value), {}).get(metric, math.nan))
        grid.append(row)
    return grid


def plot_heatmap(
    *,
    grid: list[list[float]],
    x_values: list[float],
    y_values: list[float],
    title: str,
    colorbar_label: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    cmap: str,
) -> None:
    plt.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    image = ax.imshow(
        torch.tensor(grid).numpy(), origin="lower", aspect="auto", cmap=cmap
    )

    ax.set_xticks(range(len(x_values)))
    ax.set_yticks(range(len(y_values)))
    ax.set_xticklabels([f"{value:g}" for value in x_values])
    ax.set_yticklabels([f"{value:g}" for value in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for y_idx, row in enumerate(grid):
        for x_idx, value in enumerate(row):
            if not math.isfinite(value):
                label = "nan"
            elif abs(value) >= 100:
                label = f"{value:.0f}"
            elif abs(value) >= 10:
                label = f"{value:.1f}"
            else:
                label = f"{value:.2f}"
            ax.text(x_idx, y_idx, label, ha="center", va="center", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CBF-margin rollout TDs and save parameter heatmaps."
    )
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--x_prefix", type=str, default="reward_progress")
    parser.add_argument("--y_prefix", type=str, default="h_nom")
    parser.add_argument("--x_label", type=str, default="reward_progress")
    parser.add_argument("--y_label", type=str, default="h_nom")
    parser.add_argument("--scenario_type", type=str, default="cpm_mixed")
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--cmap", type=str, default="viridis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_dir.is_dir():
        raise FileNotFoundError(f"Base directory does not exist: {args.base_dir}")

    records = discover_records(args)
    if not records:
        raise FileNotFoundError(
            f"No out_td files found under {args.base_dir}. "
            "Run the evaluation jobs before visualization."
        )

    aggregated = aggregate_records(records)
    x_values = sorted({float(record["x_value"]) for record in records})
    y_values = sorted({float(record["y_value"]) for record in records})

    csv_path = save_summary_csv(
        base_dir=args.base_dir,
        aggregated=aggregated,
        x_values=x_values,
        y_values=y_values,
        x_name=args.x_prefix,
        y_name=args.y_prefix,
    )
    print(f"[INFO] Wrote metric summary: {csv_path}")

    figures = [
        (
            "total_reward",
            "Total Reward",
            "Total reward [-]",
            "20260612_cbfmargin_total_reward_heatmap.pdf",
        ),
        (
            "parameter_independent_reward",
            "Parameter-Independent Reward",
            "Parameter-independent reward [reward/(agent*min)]",
            "20260612_cbfmargin_parameter_independent_reward_heatmap.pdf",
        ),
        (
            "collision_rate",
            "Collision Rate",
            "Collision rate [%]",
            "20260612_cbfmargin_collision_rate_heatmap.pdf",
        ),
        (
            "deadlock_rate",
            "Deadlock Rate",
            "Deadlock rate [%]",
            "20260612_cbfmargin_deadlock_rate_heatmap.pdf",
        ),
        (
            "goal_reaching_rate",
            "Goal-Reaching Rate",
            "Goal-reaching rate [%]",
            "20260612_cbfmargin_goal_reaching_rate_heatmap.pdf",
        ),
    ]

    for metric, title, colorbar_label, filename in figures:
        output_path = args.base_dir / filename
        grid = make_grid(
            aggregated=aggregated,
            metric=metric,
            x_values=x_values,
            y_values=y_values,
        )
        plot_heatmap(
            grid=grid,
            x_values=x_values,
            y_values=y_values,
            title=title,
            colorbar_label=colorbar_label,
            x_label=args.x_label,
            y_label=args.y_label,
            output_path=output_path,
            cmap=args.cmap,
        )
        print(f"[INFO] Wrote heatmap: {output_path}")


if __name__ == "__main__":
    main()
