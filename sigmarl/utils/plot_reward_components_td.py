#!/usr/bin/env python3
"""Plot reward signal components from a rollout TensorDict (.td file)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize reward components in one figure."
    )
    parser.add_argument(
        "td_path",
        nargs="?",
        default="outputs/20260616-cbfmargin/is_using_centralized_cbfFalse/reward_progress0.1/h_nom0.1/seed1/out_td_grouping_on_seed_1.td",
        help="Path to a rollout TensorDict (.td) file.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "agents.info", "next.agents.info"),
        default="auto",
        help="Where to read reward components from.",
    )
    parser.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="Environment index in batch dimension.",
    )
    parser.add_argument(
        "--components",
        default=None,
        help=(
            "Optional comma-separated reward component list. Default uses all keys"
            " starting with 'rew_'."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Simulation timestep in seconds for the x-axis. Defaults to mean(time_rl) if available, else 1.0.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to file stem + '_reward_components.png'.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    return parser.parse_args()


def _resolve_source(td: torch.Tensor) -> str:
    if "agents" in td and "info" in td["agents"]:
        return "agents.info"
    if "next" in td and "agents" in td["next"] and "info" in td["next"]["agents"]:
        return "next.agents.info"
    raise KeyError("Could not find reward info in 'agents.info' or 'next.agents.info'.")


def _resolve_reward_keys(info_td: torch.Tensor, components: str | None) -> List[str]:
    if components:
        keys = [k.strip() for k in components.split(",") if k.strip()]
        if not keys:
            raise ValueError("No valid component names provided.")
        return keys

    reward_keys = [
        k for k in info_td.keys() if isinstance(k, str) and k.startswith("rew_")
    ]
    if not reward_keys:
        raise ValueError("No keys starting with 'rew_' found under info TensorDict.")
    return sorted(reward_keys)


def _to_series(
    tensor: torch.Tensor,
    env_index: int,
    vehicle_indices: Sequence[int] | None,
    reduce_across_vehicles: bool,
) -> np.ndarray | None:
    arr = torch.as_tensor(tensor, dtype=torch.float32).cpu()
    if arr.ndim < 3:
        return None
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim != 3:
        return None
    if env_index >= arr.shape[0]:
        raise IndexError(
            f"env_index {env_index} out of bounds for batch size {arr.shape[0]}."
        )
    arr = arr[env_index]  # [T, N]

    if vehicle_indices is not None:
        arr = arr[:, vehicle_indices]
    if arr.ndim != 2:
        return None

    if arr.shape[1] == 0:
        return None

    if not reduce_across_vehicles:
        return arr[:, 0].numpy()

    if arr.shape[1] == 1:
        return arr[:, 0].numpy()
    return arr.mean(dim=1).numpy()


def _load_reward_tensors(
    td: torch.Tensor, source: str, components: str | None
) -> tuple[str, Dict[str, torch.Tensor]]:
    resolved = source
    if source == "auto":
        resolved = _resolve_source(td)

    if resolved == "agents.info":
        info_td = td["agents"]["info"]
    elif resolved == "next.agents.info":
        info_td = td["next"]["agents"]["info"]
    else:
        raise ValueError(f"Unsupported source '{resolved}'.")

    reward_keys = _resolve_reward_keys(info_td, components)
    reward_tensors = {}
    for key in reward_keys:
        if key in info_td:
            reward_tensors[key] = info_td[key]
    if not reward_tensors:
        raise KeyError(f"No requested reward components found in {resolved}.")
    return resolved, reward_tensors


def _plot_reward_components(
    reward_series: Dict[str, np.ndarray],
    output_path: Path,
    td_path: Path,
    source: str,
    dt: float,
    aggregation_label: str = "mean across vehicles",
) -> None:
    n = len(reward_series)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, max(2.5, 2.1 * n_rows)),
        squeeze=False,
    )
    axes = axes.ravel()

    max_steps = max(len(series) for series in reward_series.values())
    x_max = max_steps * dt
    for idx, (name, series) in enumerate(sorted(reward_series.items())):
        time_axis = np.arange(len(series)) * dt
        ax = axes[idx]
        ax.plot(time_axis, series, linewidth=1.0)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("reward")
        ax.grid(alpha=0.35, linestyle="--")
        ax.set_xlim(0.0, x_max)

    for j in range(len(reward_series), len(axes)):
        axes[j].axis("off")

    for ax in axes:
        ax.set_xlabel("time [s]")

    fig.suptitle(
        f"Reward components from {td_path.name} ({source}, {aggregation_label})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def _plot_all_reward_components(
    reward_series: Dict[str, np.ndarray],
    output_path: Path,
    dt: float,
) -> None:
    import matplotlib.pyplot as plt

    plotted = {}
    zero_components = []
    dashed_components = []
    for name, series in sorted(reward_series.items()):
        series_np = np.asarray(series)
        if np.all(series_np == 0):
            zero_components.append(name)
            continue
        finite = np.isfinite(series_np)
        if finite.any():
            zero_ratio = float(np.mean(series_np[finite] == 0))
            if zero_ratio > 0.9:
                dashed_components.append(name)
        plotted[name] = series_np

    if not plotted:
        print("[WARNING] _plot_all_reward_components: no non-zero components to plot.")
        return

    max_steps = max(len(series) for series in reward_series.values())
    x_max = max_steps * dt

    fig, ax = plt.subplots(figsize=(12, 7))
    for name, series in plotted.items():
        time_axis = np.arange(len(series)) * dt
        if name == "rew_total":
            line_style = ":"
            line_zorder = 10
        else:
            line_style = "--" if name in dashed_components else "-"
            line_zorder = 2
        ax.plot(
            time_axis,
            series,
            linewidth=2.4,
            linestyle=line_style,
            zorder=line_zorder,
            label=name,
        )

    ax.set_title("All reward components (same axis)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reward")
    ax.set_xlim(0.0, x_max)
    ax.grid(alpha=0.35, linestyle="--")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=360, bbox_inches="tight")


def _add_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def plot_reward_components_td(
    td_path: str | Path,
    source: str = "auto",
    env_index: int = 0,
    components: str | list[str] | None = None,
    dt: float = 0.1,
    output: str | Path | None = None,
    show: bool = False,
) -> List[Path]:
    """Plot reward components from a rollout TensorDict.

    Args:
        td_path: Path to a rollout TensorDict (.td) file.
        source: Where to read reward components from.
            One of {"auto", "agents.info", "next.agents.info"}.
        env_index: Environment index in batch dimension.
        components: Optional comma-separated string or list of reward components.
        dt: Simulation timestep in seconds for the x-axis.
        output: Optional output image path or directory.
        show: Whether to display the figure window.

    Returns:
        List of saved figure paths.
    """

    path = Path(td_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing td file: {path}")

    td = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(components, (list, tuple)):
        components_arg = ",".join(
            str(name).strip() for name in components if str(name).strip()
        )
    else:
        components_arg = components

    source_resolved, reward_tensors = _load_reward_tensors(td, source, components_arg)

    # Determine vehicle count from first tensor after validating tensor rank.
    first = next(iter(reward_tensors.values()))
    first_tensor = torch.as_tensor(first)
    if first_tensor.ndim < 3:
        raise ValueError(
            "Unexpected reward tensor rank; expected at least 3 dimensions."
        )
    if env_index >= first_tensor.shape[0]:
        raise IndexError(
            f"env_index {env_index} exceeds batch size {first_tensor.shape[0]} in TensorDict."
        )
    first_env = first_tensor[env_index]
    if first_env.ndim >= 2 and first_env.shape[-1] == 1:
        first_env = first_env.squeeze(-1)
    if first_env.ndim < 2:
        raise ValueError("Unexpected reward tensor rank after env selection.")
    n_agents = int(first_env.shape[1])
    vehicle_indices = list(range(n_agents))

    series_by_component: Dict[str, np.ndarray] = {}
    for name, tensor in reward_tensors.items():
        series = _to_series(
            tensor,
            env_index=env_index,
            vehicle_indices=vehicle_indices,
            reduce_across_vehicles=True,
        )
        if series is not None and series.size > 0:
            series_by_component[name] = series

    if not series_by_component:
        raise RuntimeError("No reward series could be extracted.")

    if output is None:
        output_path = path.with_name(f"{path.stem}_reward_components.png")
    else:
        output_path = Path(output).expanduser()
        if output_path.is_dir():
            output_path = output_path / f"{path.stem}_reward_components.png"
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    all_output_path = _add_suffix(output_path, "_all")

    saved_files: List[Path] = []
    _plot_reward_components(
        reward_series=series_by_component,
        output_path=output_path,
        td_path=path,
        source=source_resolved,
        dt=dt,
    )
    print(f"[INFO] saved: {output_path}")
    saved_files.append(output_path)

    _plot_all_reward_components(
        reward_series=series_by_component,
        output_path=all_output_path,
        dt=dt,
    )
    print(f"[INFO] saved: {all_output_path}")
    saved_files.append(all_output_path)

    # Per-agent figures.
    for agent_idx in range(n_agents):
        series_by_component_per_agent: Dict[str, np.ndarray] = {}
        for name, tensor in reward_tensors.items():
            series = _to_series(
                tensor,
                env_index=env_index,
                vehicle_indices=[agent_idx],
                reduce_across_vehicles=False,
            )
            if series is not None and series.size > 0:
                series_by_component_per_agent[name] = series

        if not series_by_component_per_agent:
            print(
                f"[WARNING] Could not extract per-agent reward series for agent {agent_idx}, skipping figures."
            )
            continue

        agent_output_path = _add_suffix(output_path, f"_agent{agent_idx}")
        agent_all_output_path = _add_suffix(output_path, f"_agent{agent_idx}_all")
        _plot_reward_components(
            reward_series=series_by_component_per_agent,
            output_path=agent_output_path,
            td_path=path,
            source=source_resolved,
            dt=dt,
            aggregation_label="single agent",
        )
        print(f"[INFO] saved: {agent_output_path}")
        saved_files.append(agent_output_path)

        _plot_all_reward_components(
            reward_series=series_by_component_per_agent,
            output_path=agent_all_output_path,
            dt=dt,
        )
        print(f"[INFO] saved: {agent_all_output_path}")
        saved_files.append(agent_all_output_path)

    if show:
        import matplotlib.pyplot as plt

        plt.show()

    return saved_files


def main() -> None:
    args = _parse_args()
    plot_reward_components_td(
        td_path=args.td_path,
        source=args.source,
        env_index=args.env_index,
        components=args.components,
        dt=args.dt,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
