#!/usr/bin/env python3
"""Plot per-agent speed distributions from a rollout TensorDict.

For joint-task rollouts, samples recorded after an agent reaches its exit are
marked by ``agents/info/is_joint_task_on_hold`` and are excluded from the
speed statistics. Non-joint-task rollouts have no held samples, so their
statistics are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_TD_PATH = Path(
    "path/to/folder/that/contains/many/.td-files"
    # "path/to/single/.td-file.td"
)
OUTPUT_DIR_NAME = "fig_speed_distribution"
SPEED_KEY = ("agents", "info", "speed")
VELOCITY_KEY = ("agents", "info", "vel")
HOLD_KEY = ("agents", "info", "is_joint_task_on_hold")


def _get_field(td: Any, key: tuple[str, ...]) -> Any | None:
    """Return a nested TensorDict/dict field, or ``None`` when it is absent."""

    try:
        value = td.get(key)
    except (AttributeError, KeyError, TypeError):
        value = None
    if value is not None:
        return value

    current = td
    for part in key:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError):
            return None
    return current


def _squeeze_trailing_singletons(value: Any) -> torch.Tensor:
    """Convert a field to a detached CPU tensor and remove trailing size-one axes."""

    tensor = torch.as_tensor(value).detach().cpu().float()
    while tensor.ndim > 0 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def _as_speed_tensor(td: Any) -> torch.Tensor:
    """Extract speed with a fallback for rollout files that only retain velocity.

    The returned tensor has shape ``[num_envs, num_steps, num_agents]``.
    A scalar speed field is preferred when it exists. Otherwise, speed is the
    Euclidean norm of the first two components of ``agents/info/vel``.
    """

    speed_value = _get_field(td, SPEED_KEY)
    if speed_value is None:
        speed_value = _get_field(td, ("next", *SPEED_KEY))

    if speed_value is not None:
        speed = _squeeze_trailing_singletons(speed_value)
        if speed.ndim == 2:
            speed = speed.unsqueeze(-1)
        if speed.ndim != 3:
            raise ValueError(
                "Expected scalar speed data with shape [env, timestep, agent] "
                f"after squeezing, got {tuple(speed.shape)}."
            )
        return speed.abs()

    velocity_value = _get_field(td, VELOCITY_KEY)
    if velocity_value is None:
        velocity_value = _get_field(td, ("next", *VELOCITY_KEY))
    if velocity_value is None:
        raise KeyError(
            "Could not find a speed field or velocity field at "
            "'agents/info/speed' or 'agents/info/vel'."
        )

    velocity = _squeeze_trailing_singletons(velocity_value)
    if velocity.ndim == 3 and velocity.shape[-1] >= 2:
        # Permit a one-agent rollout represented as [env, timestep, 2].
        velocity = velocity.unsqueeze(-2)
    if velocity.ndim != 4 or velocity.shape[-1] < 2:
        raise ValueError(
            "Expected velocity data with shape [env, timestep, agent, 2+] "
            f"after squeezing, got {tuple(velocity.shape)}."
        )
    return torch.linalg.norm(velocity[..., :2], dim=-1)


def _as_hold_mask(td: Any, expected_shape: tuple[int, ...]) -> torch.Tensor | None:
    """Extract the per-agent joint-task hold mask when it is available."""

    hold_value = _get_field(td, HOLD_KEY)
    if hold_value is None:
        hold_value = _get_field(td, ("next", *HOLD_KEY))
    if hold_value is None:
        return None

    hold_mask = _squeeze_trailing_singletons(hold_value).bool()
    if hold_mask.ndim == 2:
        # Permit a one-agent mask represented as [env, timestep].
        hold_mask = hold_mask.unsqueeze(-1)
    if tuple(hold_mask.shape) != expected_shape:
        raise ValueError(
            "Expected the joint-task hold mask to have shape "
            f"{expected_shape}, got {tuple(hold_mask.shape)}."
        )
    return hold_mask


def _column_statistics(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-sample means and population standard deviations by agent."""

    if samples.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional sample array, got {samples.shape}."
        )

    finite = np.isfinite(samples)
    counts = finite.sum(axis=0)
    if np.any(counts == 0):
        missing_agents = np.flatnonzero(counts == 0).tolist()
        raise ValueError(
            f"No finite speed samples found for agent IDs {missing_agents}."
        )

    clean_samples = np.where(finite, samples, 0.0)
    means = np.zeros(samples.shape[1], dtype=np.float64)
    np.divide(
        clean_samples.sum(axis=0),
        counts,
        out=means,
        where=counts > 0,
    )

    centered = np.where(finite, samples - means, 0.0)
    variances = np.zeros(samples.shape[1], dtype=np.float64)
    np.divide(
        np.square(centered).sum(axis=0),
        counts,
        out=variances,
        where=counts > 0,
    )
    return means, np.sqrt(np.maximum(variances, 0.0))


def _environment_means(speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-environment, per-agent means and finite-sample counts."""

    if speed.ndim != 3:
        raise ValueError(
            f"Expected speed data shaped [env, timestep, agent], got {speed.shape}."
        )

    finite = np.isfinite(speed)
    counts = finite.sum(axis=1)
    clean_speed = np.where(finite, speed, 0.0)
    sums = clean_speed.sum(axis=1, dtype=np.float64)
    means = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means, counts


def compute_speed_report(speed: np.ndarray) -> dict[str, Any]:
    """Compute pooled and environment-balanced speed statistics by agent ID.

    The plotted bars use pooled timestep samples. The method-level metrics in
    the returned report instead use one mean per environment, so an agent is
    not weighted more heavily merely because it remains active for more
    timesteps. Agent IDs are treated as categorical labels; no ordering among
    IDs is assumed.

    Args:
        speed: Speed data shaped ``[num_envs, num_steps, num_agents]``. NaN
            values are treated as excluded samples, as used for held agents.

    Returns:
        A dictionary containing per-agent statistics and categorical ID
        heterogeneity metrics.
    """

    speed = np.asarray(speed, dtype=np.float64)
    if speed.ndim != 3:
        raise ValueError(
            f"Expected speed data shaped [env, timestep, agent], got {speed.shape}."
        )

    num_envs, num_steps, num_agents = speed.shape
    environment_means, environment_counts = _environment_means(speed)
    flattened = speed.reshape(num_envs * num_steps, num_agents)
    per_agent: list[dict[str, Any]] = []

    for agent_id in range(num_agents):
        pooled_values = flattened[:, agent_id]
        pooled_values = pooled_values[np.isfinite(pooled_values)]
        environment_values = environment_means[:, agent_id]
        environment_values = environment_values[np.isfinite(environment_values)]

        environment_mean = float(np.mean(environment_values))
        if environment_values.size > 1:
            environment_std = float(np.std(environment_values, ddof=1))
            standard_error = environment_std / np.sqrt(environment_values.size)
            ci95 = (
                environment_mean - 1.96 * standard_error,
                environment_mean + 1.96 * standard_error,
            )
        else:
            environment_std = 0.0
            ci95 = (environment_mean, environment_mean)

        per_agent.append(
            {
                "agent_id": agent_id,
                "active_samples": int(pooled_values.size),
                "active_environments": int(environment_values.size),
                "pooled_mean_mps": float(np.mean(pooled_values)),
                "pooled_median_mps": float(np.median(pooled_values)),
                "pooled_std_mps": float(np.std(pooled_values)),
                "pooled_q25_mps": float(np.percentile(pooled_values, 25)),
                "pooled_q75_mps": float(np.percentile(pooled_values, 75)),
                "environment_mean_mps": environment_mean,
                "environment_std_mps": environment_std,
                "environment_ci95_mps": ci95,
            }
        )

    complete_environment_mask = np.all(np.isfinite(environment_means), axis=1)
    paired_environment_means = environment_means[complete_environment_mask]
    heterogeneity: dict[str, Any] = {
        "complete_environments": int(paired_environment_means.shape[0]),
        "mean_pairwise_gap_mps": None,
        "relative_mean_pairwise_gap": None,
        "mean_environment_range_mps": None,
        "relative_mean_environment_range": None,
        "environment_balanced_range_mps": None,
        "relative_environment_balanced_range": None,
        "fastest_agent_id": None,
        "slowest_agent_id": None,
        "largest_mean_gap_pair": None,
        "partial_eta_squared_id": None,
    }

    if paired_environment_means.shape[0] > 0:
        environment_balanced_means = np.mean(paired_environment_means, axis=0)
        overall_mean = float(np.mean(environment_balanced_means))
        pairwise_mask = np.triu(np.ones((num_agents, num_agents), dtype=bool), k=1)
        pairwise_gaps_by_environment = np.abs(
            paired_environment_means[:, :, None] - paired_environment_means[:, None, :]
        )
        pairwise_gaps = pairwise_gaps_by_environment[:, pairwise_mask]
        mean_pairwise_gap = float(np.mean(pairwise_gaps))
        environment_ranges = np.ptp(paired_environment_means, axis=1)
        mean_environment_range = float(np.mean(environment_ranges))
        balanced_range = float(np.ptp(environment_balanced_means))

        largest_pair_indices = np.unravel_index(
            np.argmax(
                np.abs(
                    environment_balanced_means[:, None]
                    - environment_balanced_means[None, :]
                )
            ),
            (num_agents, num_agents),
        )

        grand_mean = float(np.mean(paired_environment_means))
        environment_effect = num_agents * np.sum(
            (np.mean(paired_environment_means, axis=1) - grand_mean) ** 2
        )
        id_effect = paired_environment_means.shape[0] * np.sum(
            (environment_balanced_means - grand_mean) ** 2
        )
        total_effect = np.sum((paired_environment_means - grand_mean) ** 2)
        residual_effect = max(float(total_effect - environment_effect - id_effect), 0.0)
        eta_denominator = id_effect + residual_effect

        heterogeneity.update(
            {
                "mean_pairwise_gap_mps": mean_pairwise_gap,
                "relative_mean_pairwise_gap": mean_pairwise_gap / overall_mean,
                "mean_environment_range_mps": mean_environment_range,
                "relative_mean_environment_range": mean_environment_range
                / overall_mean,
                "environment_balanced_range_mps": balanced_range,
                "relative_environment_balanced_range": balanced_range / overall_mean,
                "fastest_agent_id": int(np.argmax(environment_balanced_means)),
                "slowest_agent_id": int(np.argmin(environment_balanced_means)),
                "largest_mean_gap_pair": (
                    int(largest_pair_indices[0]),
                    int(largest_pair_indices[1]),
                    float(
                        abs(
                            environment_balanced_means[largest_pair_indices[0]]
                            - environment_balanced_means[largest_pair_indices[1]]
                        )
                    ),
                ),
                "partial_eta_squared_id": (
                    float(id_effect / eta_denominator) if eta_denominator > 0 else 0.0
                ),
            }
        )

    return {
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_agents": num_agents,
        "active_samples_total": int(np.isfinite(speed).sum()),
        "active_sample_counts": environment_counts.sum(axis=0).astype(int).tolist(),
        "agents": per_agent,
        "heterogeneity": heterogeneity,
    }


def _format_report_value(value: float | int | None, decimals: int = 4) -> str:
    """Format a report value without exposing ``None`` as a Python literal."""

    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def _print_speed_report(
    td_path: Path,
    report: dict[str, Any],
    hold_label: str,
) -> None:
    """Print a compact per-ID speed report for one rollout."""

    print(f"[SPEED REPORT] {td_path}")
    print(
        "  data: "
        f"{report['num_envs']} environments × {report['num_steps']} timesteps × "
        f"{report['num_agents']} agents; {hold_label}"
    )
    print(
        "  per-ID statistics "
        "(environment-balanced mean/95% CI; pooled distribution statistics):"
    )
    print(
        "    ID  active_envs  active_samples  env_mean  env_CI95          "
        "pooled_mean  pooled_median  pooled_std  pooled_q25  pooled_q75"
    )
    for agent in report["agents"]:
        ci_low, ci_high = agent["environment_ci95_mps"]
        print(
            f"    {agent['agent_id']:>2}  "
            f"{agent['active_environments']:>11}  "
            f"{agent['active_samples']:>14}  "
            f"{agent['environment_mean_mps']:>8.4f}  "
            f"[{ci_low:.4f}, {ci_high:.4f}]  "
            f"{agent['pooled_mean_mps']:>11.4f}  "
            f"{agent['pooled_median_mps']:>13.4f}  "
            f"{agent['pooled_std_mps']:>10.4f}  "
            f"{agent['pooled_q25_mps']:>10.4f}  "
            f"{agent['pooled_q75_mps']:>10.4f}"
        )

    heterogeneity = report["heterogeneity"]
    print(
        "  categorical ID heterogeneity "
        f"(paired across {heterogeneity['complete_environments']} complete environments):"
    )
    print(
        "    mean within-environment pairwise gap: "
        f"{_format_report_value(heterogeneity['mean_pairwise_gap_mps'])} m/s "
        f"({_format_report_value(heterogeneity['relative_mean_pairwise_gap'] * 100 if heterogeneity['relative_mean_pairwise_gap'] is not None else None, 2)}%)"
    )
    print(
        "    mean within-environment range: "
        f"{_format_report_value(heterogeneity['mean_environment_range_mps'])} m/s "
        f"({_format_report_value(heterogeneity['relative_mean_environment_range'] * 100 if heterogeneity['relative_mean_environment_range'] is not None else None, 2)}%)"
    )
    print(
        "    range of environment-balanced ID means: "
        f"{_format_report_value(heterogeneity['environment_balanced_range_mps'])} m/s "
        f"({_format_report_value(heterogeneity['relative_environment_balanced_range'] * 100 if heterogeneity['relative_environment_balanced_range'] is not None else None, 2)}%)"
    )
    print(
        "    fastest / slowest ID: "
        f"{heterogeneity['fastest_agent_id']} / {heterogeneity['slowest_agent_id']}"
    )
    largest_pair = heterogeneity["largest_mean_gap_pair"]
    if largest_pair is None:
        largest_pair_text = "n/a"
    else:
        largest_pair_text = (
            f"IDs {largest_pair[0]} and {largest_pair[1]} "
            f"({_format_report_value(largest_pair[2])} m/s)"
        )
    print(f"    largest environment-balanced ID gap: {largest_pair_text}")
    print(
        "    repeated-measures ID effect (partial eta²): "
        f"{_format_report_value(heterogeneity['partial_eta_squared_id'], 3)}"
    )


def _plot_bar_distribution(
    samples: np.ndarray,
    title: str,
    sample_label: str,
    output_path: Path,
) -> None:
    """Save one agent-ID bar plot from samples shaped ``[samples, agents]``."""

    means, standard_deviations = _column_statistics(samples)
    agent_ids = np.arange(samples.shape[1])

    figure_width = max(8.0, 1.2 * len(agent_ids))
    fig, ax = plt.subplots(figsize=(figure_width, 6.0))
    ax.bar(
        agent_ids,
        means,
        yerr=standard_deviations,
        capsize=4,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.set_xlabel("Agent ID")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title(title)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(agent_ids)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.text(
        0.99,
        0.98,
        sample_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", dpi=300)
    plt.close(fig)


def _find_td_files(path: Path) -> list[Path]:
    """Return the input file or all ``.td`` files below an input directory."""

    if path.is_file():
        return [path]
    if path.is_dir():
        td_files = sorted(
            candidate for candidate in path.rglob("*.td") if candidate.is_file()
        )
        if td_files:
            return td_files
        raise FileNotFoundError(f"No .td files found under directory: {path}")
    raise FileNotFoundError(f"Missing td file or directory: {path}")


def plot_speed_distribution(
    td_path: str | Path,
    output_dir: str | Path | None = None,
    print_report: bool = True,
) -> list[Path]:
    """Plot speed bars and optionally print an ID-conditioned speed report.

    Args:
        td_path: Path to a rollout TensorDict saved by ``main_testing.py``.
        output_dir: Optional output directory. By default, create
            ``fig_speed_distribution`` beside ``td_path``.
        print_report: Whether to print per-ID statistics and categorical
            ID-heterogeneity metrics.

    Returns:
        Paths to the saved per-environment figures followed by the aggregate
        figure.
    """

    path = Path(td_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing td file: {path}")

    out_td = torch.load(path, map_location="cpu", weights_only=False)
    speed_tensor = _as_speed_tensor(out_td)
    hold_mask = _as_hold_mask(out_td, tuple(speed_tensor.shape))
    if hold_mask is None:
        hold_label = "hold mask unavailable"
    else:
        held_count = int(hold_mask.sum().item())
        if held_count:
            speed_tensor = speed_tensor.masked_fill(hold_mask, float("nan"))
        hold_label = "on-hold samples excluded"

    speed = speed_tensor.numpy()
    if speed.shape[0] == 0 or speed.shape[1] == 0 or speed.shape[2] == 0:
        raise ValueError(f"Speed data has an empty dimension: {speed.shape}")

    if print_report:
        _print_speed_report(path, compute_speed_report(speed), hold_label)

    destination = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else path.parent / OUTPUT_DIR_NAME
    )
    destination.mkdir(parents=True, exist_ok=True)

    num_envs, num_steps, num_agents = speed.shape
    saved_paths: list[Path] = []
    for env_index in range(num_envs):
        output_path = destination / f"speed_distribution_env_{env_index}.jpeg"
        if hold_mask is None:
            sample_label = f"{num_steps} timestep samples per bar"
        else:
            active_counts = np.isfinite(speed[env_index]).sum(axis=0)
            sample_label = (
                f"{num_steps} saved timesteps; {hold_label} "
                f"({active_counts.min()}–{active_counts.max()} active)"
            )
        _plot_bar_distribution(
            samples=speed[env_index],
            title=f"Speed distribution — environment {env_index}",
            sample_label=sample_label,
            output_path=output_path,
        )
        saved_paths.append(output_path)

    all_envs_output_path = destination / "speed_distribution_all_envs.jpeg"
    all_envs_samples = speed.reshape(num_envs * num_steps, num_agents)
    if hold_mask is None:
        all_envs_sample_label = f"{num_envs * num_steps} env/timestep samples per bar"
    else:
        active_counts = np.isfinite(all_envs_samples).sum(axis=0)
        all_envs_sample_label = (
            f"{num_envs * num_steps} saved env/timesteps; {hold_label} "
            f"({active_counts.min()}–{active_counts.max()} active)"
        )
    _plot_bar_distribution(
        samples=all_envs_samples,
        title="Speed distribution — all environments",
        sample_label=all_envs_sample_label,
        output_path=all_envs_output_path,
    )
    print(f"[INFO] saved: {all_envs_output_path}")
    saved_paths.append(all_envs_output_path)
    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-agent speed distributions from a rollout .td file."
    )
    parser.add_argument(
        "td_path",
        nargs="?",
        default=str(DEFAULT_TD_PATH),
        help="Path to a rollout .td file or a directory to search recursively.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Defaults to fig_speed_distribution "
            "beside the input .td file."
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip the printed per-ID speed statistics and heterogeneity report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.td_path).expanduser()
    td_files = _find_td_files(input_path)
    for td_file in td_files:
        plot_speed_distribution(
            td_file,
            output_dir=args.output_dir,
            print_report=not args.no_report,
        )


if __name__ == "__main__":
    main()
