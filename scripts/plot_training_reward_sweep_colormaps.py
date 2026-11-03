#!/usr/bin/env python3
"""Plot best training rewards from checkpoint sweep JSON files."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "marl_for_cavs_matplotlib"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASE_DIR = Path(
    "outputs/20260617-cbfmargin_sparse/" "is_using_centralized_cbfFalse"
)
PARAMETER_PART_PATTERN = re.compile(
    r"^(?P<name>.*?)(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


@dataclass(frozen=True)
class Record:
    """Reward metrics extracted from one reward data JSON file."""

    parameters: tuple[tuple[str, float], ...]
    seed: str
    episode_best: float
    param_independent_best: float
    param_independent_at_episode_best: float
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read reward*_data.json files from a training sweep folder and "
            "save best-reward colormaps."
        )
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Checkpoint sweep root. Default: {DEFAULT_BASE_DIR}",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name. Default: viridis",
    )
    return parser.parse_args()


def parse_parameter_part(part: str) -> tuple[str, float] | None:
    match = PARAMETER_PART_PATTERN.match(part)
    if match is None:
        return None

    name = match.group("name")
    if not name:
        return None

    try:
        value = float(match.group("value"))
    except ValueError:
        return None
    return name, value


def extract_path_parameters(
    data_path: Path, base_dir: Path
) -> tuple[tuple[tuple[str, float], ...], str]:
    relative_parent = data_path.relative_to(base_dir).parent
    parameters: list[tuple[str, float]] = []
    seed = "no_seed"

    for part in relative_parent.parts:
        if part.startswith("seed"):
            seed = part
            continue

        parsed = parse_parameter_part(part)
        if parsed is not None:
            parameters.append(parsed)

    return tuple(parameters), seed


def require_numeric_list(data: dict, key: str, path: Path) -> list[float]:
    values = data.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a non-empty list at key {key!r}")

    numeric_values: list[float] = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{path} has a non-numeric value at {key}[{index}]: {value!r}"
            )
        numeric_values.append(float(value))
    return numeric_values


def finite_argmax(values: list[float], key: str, path: Path) -> int:
    best_index: int | None = None
    best_value = -math.inf

    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        if best_index is None or value > best_value:
            best_index = index
            best_value = value

    if best_index is None:
        raise ValueError(f"{path} contains no finite values in {key!r}")
    return best_index


def load_record(path: Path, base_dir: Path) -> Record:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    episode_rewards = require_numeric_list(data, "episode_reward_mean_list", path)
    parameter_independent_rewards = require_numeric_list(
        data, "parameter_independent_reward_mean_list", path
    )

    episode_best_index = finite_argmax(
        episode_rewards, "episode_reward_mean_list", path
    )
    parameter_independent_best_index = finite_argmax(
        parameter_independent_rewards,
        "parameter_independent_reward_mean_list",
        path,
    )

    if episode_best_index >= len(parameter_independent_rewards):
        raise ValueError(
            f"{path} has no parameter-independent reward at episode-best index "
            f"{episode_best_index}"
        )

    parameters, seed = extract_path_parameters(path, base_dir)
    return Record(
        parameters=parameters,
        seed=seed,
        episode_best=episode_rewards[episode_best_index],
        param_independent_best=(
            parameter_independent_rewards[parameter_independent_best_index]
        ),
        param_independent_at_episode_best=(
            parameter_independent_rewards[episode_best_index]
        ),
        path=path,
    )


def discover_records(base_dir: Path) -> list[Record]:
    json_paths = sorted(base_dir.rglob("reward*_data.json"))
    if not json_paths:
        raise FileNotFoundError(f"No reward*_data.json files found under {base_dir}")
    return [load_record(path, base_dir) for path in json_paths]


def detect_swept_parameters(records: Iterable[Record]) -> list[str]:
    values_by_name: dict[str, set[float]] = defaultdict(set)
    order: list[str] = []

    for record in records:
        for name, value in record.parameters:
            if name not in values_by_name:
                order.append(name)
            values_by_name[name].add(value)

    return [name for name in order if len(values_by_name[name]) > 1]


def validate_record_parameters(
    records: Iterable[Record], swept_names: list[str]
) -> None:
    expected = set(swept_names)
    for record in records:
        names = {name for name, _ in record.parameters if name in expected}
        missing = expected - names
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"{record.path} is missing swept parameter(s): {missing_text}"
            )


def mean_finite(values: Iterable[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return math.nan
    return float(sum(finite_values) / len(finite_values))


def parameter_key(record: Record, swept_names: list[str]) -> tuple[float, ...]:
    values_by_name = dict(record.parameters)
    return tuple(values_by_name[name] for name in swept_names)


def aggregate_records(
    records: Iterable[Record], swept_names: list[str]
) -> dict[tuple[float, ...], dict[str, float]]:
    grouped: dict[tuple[float, ...], dict[str, list[Record]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for record in records:
        grouped[parameter_key(record, swept_names)][record.seed].append(record)

    aggregated: dict[tuple[float, ...], dict[str, float]] = {}
    for parameters, records_by_seed in grouped.items():
        seed_metric_means: dict[str, list[float]] = defaultdict(list)

        for seed_records in records_by_seed.values():
            seed_metric_means["episode_best"].append(
                mean_finite(record.episode_best for record in seed_records)
            )
            seed_metric_means["param_independent_best"].append(
                mean_finite(record.param_independent_best for record in seed_records)
            )
            seed_metric_means["param_independent_at_episode_best"].append(
                mean_finite(
                    record.param_independent_at_episode_best for record in seed_records
                )
            )

        aggregated[parameters] = {
            metric_name: mean_finite(metric_values)
            for metric_name, metric_values in seed_metric_means.items()
        }

    return aggregated


def sorted_unique(values: Iterable[float]) -> list[float]:
    return sorted(set(values))


def build_grid(
    *,
    aggregated: dict[tuple[float, ...], dict[str, float]],
    swept_values: list[list[float]],
    metric_name: str,
) -> np.ndarray:
    if len(swept_values) == 1:
        x_values = swept_values[0]
        return np.array(
            [
                [
                    aggregated.get((x_value,), {}).get(metric_name, math.nan)
                    for x_value in x_values
                ]
            ]
        )

    x_values, y_values = swept_values
    return np.array(
        [
            [
                aggregated.get((x_value, y_value), {}).get(metric_name, math.nan)
                for x_value in x_values
            ]
            for y_value in y_values
        ]
    )


def format_cell(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_colormap(
    *,
    grid: np.ndarray,
    swept_names: list[str],
    swept_values: list[list[float]],
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str,
) -> None:
    matplotlib.rcParams["text.usetex"] = False

    fig_height = 2.8 if len(swept_names) == 1 else 5.2
    fig, ax = plt.subplots(figsize=(6.8, fig_height), constrained_layout=True)
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    x_values = swept_values[0]
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{value:g}" for value in x_values])
    ax.set_xlabel(swept_names[0])

    if len(swept_names) == 1:
        ax.set_yticks([0])
        ax.set_yticklabels([""])
        ax.set_ylabel("")
    else:
        y_values = swept_values[1]
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([f"{value:g}" for value in y_values])
        ax.set_ylabel(swept_names[1])

    ax.set_title(title)

    for y_index in range(grid.shape[0]):
        for x_index in range(grid.shape[1]):
            value = float(grid[y_index, x_index])
            ax.text(
                x_index,
                y_index,
                format_cell(value),
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_colormaps(
    *,
    base_dir: Path,
    aggregated: dict[tuple[float, ...], dict[str, float]],
    swept_names: list[str],
    cmap: str,
) -> list[Path]:
    swept_values = [
        sorted_unique(parameters[index] for parameters in aggregated)
        for index in range(len(swept_names))
    ]

    figure_specs = [
        (
            "episode_best",
            "Best Episode Reward",
            "Best episode_reward_mean_list value",
            "episode_reward_best_colormap.png",
        ),
        (
            "param_independent_best",
            "Best Parameter-Independent Reward",
            "Best parameter_independent_reward_mean_list value",
            "parameter_independent_reward_best_colormap.png",
        ),
        (
            "param_independent_at_episode_best",
            "Parameter-Independent Reward at Best Episode Reward",
            "Parameter-independent reward at episode-best index",
            "parameter_independent_reward_at_episode_best_colormap.png",
        ),
    ]

    output_paths: list[Path] = []
    for metric_name, title, colorbar_label, filename in figure_specs:
        output_path = base_dir / filename
        grid = build_grid(
            aggregated=aggregated,
            swept_values=swept_values,
            metric_name=metric_name,
        )
        plot_colormap(
            grid=grid,
            swept_names=swept_names,
            swept_values=swept_values,
            title=title,
            colorbar_label=colorbar_label,
            output_path=output_path,
            cmap=cmap,
        )
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    records = discover_records(base_dir)
    swept_names = detect_swept_parameters(records)
    if not swept_names:
        raise ValueError(f"No swept parameters detected under {base_dir}")
    if len(swept_names) > 2:
        names_text = ", ".join(swept_names)
        raise NotImplementedError(
            f"Only one or two swept parameters are supported; detected {names_text}"
        )

    validate_record_parameters(records, swept_names)
    aggregated = aggregate_records(records, swept_names)
    output_paths = save_colormaps(
        base_dir=base_dir,
        aggregated=aggregated,
        swept_names=swept_names,
        cmap=args.cmap,
    )

    print(
        f"Detected {len(swept_names)} swept parameter(s): " f"{', '.join(swept_names)}"
    )
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
