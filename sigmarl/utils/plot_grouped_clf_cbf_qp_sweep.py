import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


DEFAULT_OUTPUT_DIR = Path("checkpoints/20260503-grouped_clf_cbf_qp_comparison_v1")
DEFAULT_EXPECTED_SEEDS = [1, 2, 3, 4, 5]
TD_PATTERN = re.compile(
    r"^out_td_"
    r"(?P<scenario>.+?)_agents(?P<n_agents>\d+)_?"
    r"max_group_size(?P<group_size>None|\d+)"
    r"(?:_(?P<grouping_method>.+?))?"
    r"_(?P<solver>proxsuite|cvxpy)_seed(?P<seed>\d+)\.td$"
)


METRICS = [
    (
        "avg_reward",
        "Average Episode Reward Per Agent",
        "Reward [-]",
    ),
    (
        "avg_collision_count",
        "Average Collision Count Per Agent",
        "Count [-]",
    ),
    (
        "avg_goal_count",
        "Average Goal-Reaching Count Per Agent",
        "Count [-]",
    ),
    (
        "avg_deadlock_count",
        "Average Deadlock Count Per Agent",
        "Count [-]",
    ),
    (
        "avg_cbf_time_ms",
        "CBF Computation Time",
        "Time [ms]",
    ),
    (
        "avg_speed",
        "Average Speed",
        "Speed [m/s]",
    ),
]

FIGURE_GROUPS = [
    ("reward_collision", ["avg_reward", "avg_collision_count"]),
    ("goal_deadlock", ["avg_goal_count", "avg_deadlock_count"]),
    ("cbf_time_speed", ["avg_cbf_time_ms", "avg_speed"]),
]


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


def finite_mean(tensor, scale: float = 1.0) -> float:
    if tensor is None:
        return math.nan
    tensor = torch.as_tensor(tensor).detach().cpu().float() * scale
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return math.nan
    return float(finite.mean().item())


def event_count_per_agent(out_td, key) -> float:
    events = squeeze_scalar_field(get_tensor(out_td, key))
    if events is None or events.ndim < 3:
        return math.nan
    return float(events.bool().float().sum(dim=1).mean().item())


def collision_count_per_agent(out_td) -> float:
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
        collision = collision_agents.bool() | collision_lanelets.bool()

    if collision.ndim < 3:
        return math.nan
    return float(collision.float().sum(dim=1).mean().item())


def average_reward(out_td) -> float:
    reward_total = squeeze_scalar_field(
        get_tensor(out_td, ("agents", "info", "rew_total"))
    )
    if reward_total is None:
        reward_parts = [
            squeeze_scalar_field(get_tensor(out_td, key))
            for key in [
                ("agents", "info", "rew_reach_goal"),
                ("agents", "info", "rew_collide_other_agents"),
                ("agents", "info", "rew_collide_lane"),
                ("agents", "info", "rew_deadlock"),
            ]
        ]
        reward_parts = [item.float() for item in reward_parts if item is not None]
        if not reward_parts:
            return math.nan
        reward_total = torch.stack(reward_parts, dim=0).sum(dim=0)

    if reward_total.ndim < 3:
        return math.nan

    episode_reward_per_agent = reward_total.float().sum(dim=1)
    return float(episode_reward_per_agent.mean().item())


def average_speed(out_td) -> float:
    speed = squeeze_scalar_field(get_tensor(out_td, ("agents", "info", "speed")))
    if speed is not None:
        return finite_mean(speed.abs())

    vel = get_tensor(out_td, ("agents", "info", "vel"))
    if vel is None:
        return math.nan
    vel = torch.as_tensor(vel).detach().cpu().float()
    if vel.ndim < 4 or vel.shape[-1] < 2:
        return math.nan
    return finite_mean(torch.linalg.norm(vel[..., :2], dim=-1))


def average_cbf_time_ms(out_td) -> float:
    time_cbf = get_tensor(out_td, "time_cbf")
    if time_cbf is None:
        time_cbf = get_tensor(out_td, ("collector", "time_cbf"))
    return finite_mean(time_cbf, scale=1e3)


def load_metrics(td_path: Path) -> dict[str, float]:
    out_td = torch.load(td_path, weights_only=False, map_location="cpu")
    return {
        "avg_reward": average_reward(out_td),
        "avg_collision_count": collision_count_per_agent(out_td),
        "avg_goal_count": event_count_per_agent(
            out_td, ("agents", "info", "is_reach_goal")
        ),
        "avg_deadlock_count": event_count_per_agent(
            out_td, ("agents", "info", "is_deadlocked")
        ),
        "avg_cbf_time_ms": average_cbf_time_ms(out_td),
        "avg_speed": average_speed(out_td),
    }


def parse_td_filename(td_path: Path):
    match = TD_PATTERN.match(td_path.name)
    if match is None:
        return None

    n_agents = int(match.group("n_agents"))
    group_size_raw = match.group("group_size")
    group_size = n_agents if group_size_raw == "None" else int(group_size_raw)
    grouping_method = match.group("grouping_method")

    return {
        "scenario_type": match.group("scenario"),
        "n_agents": n_agents,
        "group_size": group_size,
        "grouping_method": grouping_method,
        "solver": match.group("solver"),
        "seed": int(match.group("seed")),
    }


def discover_records(args: argparse.Namespace) -> list[dict]:
    records = []
    for td_path in sorted(Path(args.output_dir).glob("out_td_*.td")):
        parsed = parse_td_filename(td_path)
        if parsed is None:
            continue
        if args.solver != "all" and parsed["solver"] != args.solver:
            continue
        if (
            args.scenario_type != "all"
            and parsed["scenario_type"] != args.scenario_type
        ):
            continue
        if args.grouping_method and parsed["grouping_method"] not in {
            args.grouping_method,
            None,
        }:
            continue

        metrics = load_metrics(td_path)
        records.append(
            {
                **parsed,
                **metrics,
                "td_path": str(td_path),
            }
        )

    return records


def aggregate_records(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["scenario_type"],
            record["n_agents"],
            record["group_size"],
        )
        grouped[key].append(record)

    rows = []
    for (scenario_type, n_agents, group_size), group_records in sorted(grouped.items()):
        row = {
            "scenario_type": scenario_type,
            "n_agents": n_agents,
            "group_size": group_size,
            "n_runs": len(group_records),
        }
        for metric_key, _, _ in METRICS:
            values = [
                record[metric_key]
                for record in group_records
                if math.isfinite(record[metric_key])
            ]
            if values:
                row[f"{metric_key}_mean"] = sum(values) / len(values)
                if len(values) > 1:
                    mean = row[f"{metric_key}_mean"]
                    variance = sum((value - mean) ** 2 for value in values) / (
                        len(values) - 1
                    )
                    row[f"{metric_key}_std"] = math.sqrt(variance)
                else:
                    row[f"{metric_key}_std"] = 0.0
            else:
                row[f"{metric_key}_mean"] = math.nan
                row[f"{metric_key}_std"] = math.nan
        rows.append(row)
    return rows


def warn_missing_seed_coverage(records: list[dict], expected_seeds: list[int]):
    if not expected_seeds:
        return

    expected_seed_set = set(expected_seeds)
    grouped = defaultdict(set)
    for record in records:
        key = (
            record["scenario_type"],
            record["n_agents"],
            record["group_size"],
        )
        grouped[key].add(record["seed"])

    for (scenario_type, n_agents, group_size), seeds in sorted(grouped.items()):
        missing_seeds = sorted(expected_seed_set - seeds)
        if missing_seeds:
            print(
                "[WARNING] "
                f"{scenario_type} agents={n_agents} group={group_size}: "
                f"missing seeds {missing_seeds}"
            )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_group(
    records: list[dict],
    scenario_type: str,
    n_agents: int,
    metric_keys: list[str],
    figure_name: str,
    fig_dir: Path,
):
    metric_lookup = {key: (title, ylabel) for key, title, ylabel in METRICS}
    scenario_records = [
        record
        for record in records
        if record["scenario_type"] == scenario_type and record["n_agents"] == n_agents
    ]
    if not scenario_records:
        return None

    group_sizes = sorted({record["group_size"] for record in scenario_records})
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5.2 * len(metric_keys), 4))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, metric_key in zip(axes, metric_keys):
        title, ylabel = metric_lookup[metric_key]
        box_data = []
        x_labels = []

        for group_size in group_sizes:
            group_records = [
                record
                for record in scenario_records
                if record["group_size"] == group_size
                and math.isfinite(record[metric_key])
            ]
            group_records = sorted(group_records, key=lambda record: record["seed"])
            values = [record[metric_key] for record in group_records]

            if not values:
                continue

            box_data.append(values)
            x_labels.append(str(group_size))

        if not box_data:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="grey",
            )
            ax.set_title(title)
            ax.set_xlabel("Maximum Group Size")
            ax.set_ylabel(ylabel)
            continue

        positions = list(range(1, len(box_data) + 1))
        box = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "tab:red",
                "markersize": 4.5,
            },
            medianprops={"color": "black", "linewidth": 1.4},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "tab:blue",
                "markersize": 4,
            },
        )
        for patch in box["boxes"]:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.22)

        for position, values in zip(positions, box_data):
            jitter = torch.linspace(-0.11, 0.11, steps=len(values)).tolist()
            ax.scatter(
                [position + offset for offset in jitter],
                values,
                s=16,
                color="tab:blue",
                alpha=0.75,
                zorder=3,
            )

        ax.set_title(title)
        ax.set_xlabel("Maximum Group Size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        f"{scenario_type} | {n_agents} agents | boxplots across seeds",
        fontsize=12,
    )
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"fig_{scenario_type}_agents{n_agents}_{figure_name}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_all(records: list[dict], fig_dir: Path) -> list[Path]:
    figures = []
    scenario_agent_pairs = sorted(
        {(record["scenario_type"], record["n_agents"]) for record in records}
    )
    for scenario_type, n_agents in scenario_agent_pairs:
        for figure_name, metric_keys in FIGURE_GROUPS:
            fig_path = plot_metric_group(
                records=records,
                scenario_type=scenario_type,
                n_agents=n_agents,
                metric_keys=metric_keys,
                figure_name=figure_name,
                fig_dir=fig_dir,
            )
            if fig_path is not None:
                figures.append(fig_path)
    return figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize grouped CLF-CBF-QP sweep results."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing out_td_*.td files.",
    )
    parser.add_argument(
        "--fig_dir",
        type=Path,
        default=None,
        help="Directory where summary figures and CSV files are saved.",
    )
    parser.add_argument(
        "--scenario_type",
        choices=["all", "cpm_mixed", "cpm_entire"],
        default="all",
        help="Scenario subset to plot.",
    )
    parser.add_argument(
        "--solver",
        choices=["all", "proxsuite", "cvxpy"],
        default="proxsuite",
        help="Solver subset to include.",
    )
    parser.add_argument(
        "--grouping_method",
        choices=["interaction_risk", "k_nearest"],
        default="interaction_risk",
        help="Grouping method subset to include.",
    )
    parser.add_argument(
        "--expected_seeds",
        type=lambda value: [
            int(item.strip()) for item in value.split(",") if item.strip()
        ],
        default=DEFAULT_EXPECTED_SEEDS,
        help="Comma-separated seeds expected in each boxplot. Default: 1,2,3,4,5.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else output_dir / "summary_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args)
    if not records:
        print(f"[WARNING] No matching TD files found in {output_dir}.")
        return

    raw_csv_path = fig_dir / "grouped_clf_cbf_qp_raw_metrics.csv"
    write_csv(raw_csv_path, records)

    aggregated_rows = aggregate_records(records)
    aggregate_csv_path = fig_dir / "grouped_clf_cbf_qp_aggregated_metrics.csv"
    write_csv(aggregate_csv_path, aggregated_rows)

    warn_missing_seed_coverage(records, expected_seeds=args.expected_seeds)
    figures = plot_all(records, fig_dir=fig_dir)

    print(f"[INFO] Loaded {len(records)} TD files.")
    print(f"[INFO] Saved raw metrics to {raw_csv_path}.")
    print(f"[INFO] Saved aggregated metrics to {aggregate_csv_path}.")
    for fig_path in figures:
        print(f"[INFO] Saved figure to {fig_path}.")


if __name__ == "__main__":
    main()
