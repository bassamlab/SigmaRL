#!/usr/bin/env python3
"""Interactive desktop inspector for rollout TensorDict (.td) files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from tensordict import TensorDictBase


@dataclass(frozen=True)
class FieldSummary:
    """Metadata for one tensor leaf inside a TensorDict."""

    path: str
    key_path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    numel: int
    parent_batch_shape: tuple[int, ...]
    finite_count: int | None = None
    nan_count: int | None = None
    inf_count: int | None = None
    true_count: int | None = None
    false_count: int | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None


@dataclass(frozen=True)
class SeriesLine:
    """One plottable line extracted from a tensor field."""

    label: str
    y: np.ndarray


@dataclass(frozen=True)
class ExtractedSeries:
    """Time-series lines ready to draw on a Matplotlib axes."""

    field_path: str
    title: str
    x_label: str
    y_label: str
    x: np.ndarray
    lines: tuple[SeriesLine, ...]


def _normalize_key_path(field_path: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(field_path, str):
        key_path = tuple(part for part in field_path.split("/") if part)
    else:
        key_path = tuple(str(part) for part in field_path)
    if not key_path:
        raise ValueError("field path must not be empty")
    return key_path


def _join_key_path(key_path: Sequence[str]) -> str:
    return "/".join(str(part) for part in key_path)


def load_tensordict(path: str | Path) -> TensorDictBase:
    """Load a trusted local TensorDict file from disk."""

    td_path = Path(path).expanduser()
    if not td_path.exists():
        raise FileNotFoundError(f"missing TensorDict file: {td_path}")
    if not td_path.is_file():
        raise ValueError(f"expected a file path, got: {td_path}")
    if td_path.suffix != ".td":
        raise ValueError(f"expected a .td file, got: {td_path}")

    td = torch.load(td_path, map_location="cpu", weights_only=False)
    if not isinstance(td, TensorDictBase):
        raise TypeError(f"loaded object is not a TensorDict: {type(td)!r}")
    return td


def get_field(
    td: TensorDictBase, field_path: str | Sequence[str]
) -> tuple[TensorDictBase, torch.Tensor]:
    """Return the parent TensorDict and tensor leaf for a slash-separated path."""

    key_path = _normalize_key_path(field_path)
    parent: TensorDictBase = td
    current = td
    for idx, key in enumerate(key_path):
        if not isinstance(current, TensorDictBase):
            prefix = _join_key_path(key_path[:idx])
            raise KeyError(f"{prefix} is not a TensorDict")
        if key not in current.keys():
            prefix = _join_key_path(key_path[:idx])
            raise KeyError(f"missing key {key!r} under {prefix or '<root>'}")
        value = current[key]
        if idx == len(key_path) - 1:
            if not torch.is_tensor(value):
                raise TypeError(f"{_join_key_path(key_path)} is not a tensor field")
            return current, value
        parent = current
        current = value
    raise KeyError(_join_key_path(key_path))


def _tensor_stats(path: str, key_path: tuple[str, ...], parent, tensor) -> FieldSummary:
    value = tensor.detach().cpu()
    parent_batch_shape = (
        tuple(int(dim) for dim in parent.batch_size)
        if isinstance(parent, TensorDictBase)
        else ()
    )

    summary = {
        "path": path,
        "key_path": key_path,
        "shape": tuple(int(dim) for dim in value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "parent_batch_shape": parent_batch_shape,
    }

    if value.numel() == 0:
        return FieldSummary(**summary)

    if value.dtype == torch.bool:
        true_count = int(value.sum().item())
        return FieldSummary(
            **summary,
            true_count=true_count,
            false_count=int(value.numel() - true_count),
        )

    if torch.is_floating_point(value):
        finite_mask = torch.isfinite(value)
        finite_values = value[finite_mask]
        stats = {
            "finite_count": int(finite_mask.sum().item()),
            "nan_count": int(torch.isnan(value).sum().item()),
            "inf_count": int(torch.isinf(value).sum().item()),
        }
        if finite_values.numel() > 0:
            stats.update(
                {
                    "min_value": float(finite_values.min().item()),
                    "max_value": float(finite_values.max().item()),
                    "mean_value": float(finite_values.mean().item()),
                }
            )
        return FieldSummary(**summary, **stats)

    if value.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        numeric = value.to(torch.float64)
        return FieldSummary(
            **summary,
            min_value=float(numeric.min().item()),
            max_value=float(numeric.max().item()),
            mean_value=float(numeric.mean().item()),
        )

    return FieldSummary(**summary)


def flatten_tensordict(td: TensorDictBase) -> list[FieldSummary]:
    """Return summaries for tensor leaves in depth-first path order."""

    fields: list[FieldSummary] = []

    def walk(current: TensorDictBase, prefix: tuple[str, ...]) -> None:
        for key in current.keys():
            value = current[key]
            key_path = prefix + (str(key),)
            if isinstance(value, TensorDictBase):
                walk(value, key_path)
            elif torch.is_tensor(value):
                path = _join_key_path(key_path)
                fields.append(_tensor_stats(path, key_path, current, value))

    walk(td, ())
    return sorted(fields, key=lambda item: item.path)


def format_summary(summary: FieldSummary) -> str:
    """Format a field summary for display in the inspector UI."""

    lines = [
        f"path: {summary.path}",
        f"shape: {summary.shape}",
        f"dtype: {summary.dtype}",
        f"numel: {summary.numel}",
        f"parent batch shape: {summary.parent_batch_shape}",
    ]
    if summary.true_count is not None:
        lines.extend(
            [
                f"true count: {summary.true_count}",
                f"false count: {summary.false_count}",
            ]
        )
    if summary.finite_count is not None:
        lines.extend(
            [
                f"finite count: {summary.finite_count}",
                f"NaN count: {summary.nan_count}",
                f"inf count: {summary.inf_count}",
            ]
        )
    if summary.min_value is not None:
        lines.extend(
            [
                f"min: {summary.min_value:.8g}",
                f"max: {summary.max_value:.8g}",
                f"mean: {summary.mean_value:.8g}",
            ]
        )
    return "\n".join(lines)


def _as_plot_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        return tensor.to(torch.float32)
    if torch.is_floating_point(tensor):
        return tensor.to(torch.float32)
    if tensor.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return tensor.to(torch.float32)
    raise TypeError(f"tensor dtype {tensor.dtype} is not plottable as a time series")


def _parse_optional_index(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = value.strip()
    if value == "":
        return None
    return int(value)


def _validate_index(name: str, value: int, size: int) -> None:
    if value < 0 or value >= size:
        raise IndexError(f"{name} index {value} is out of bounds for size {size}")


def _series_label(
    file_label: str | None,
    field_path: str,
    suffix: str | None = None,
) -> str:
    label = f"{file_label}: {field_path}" if file_label else field_path
    if suffix:
        label = f"{label} [{suffix}]"
    return label


def _is_agent_field(
    root_batch_shape: tuple[int, ...],
    parent_batch_shape: tuple[int, ...],
    tensor_shape: tuple[int, ...],
) -> bool:
    return (
        len(root_batch_shape) >= 2
        and len(parent_batch_shape) >= 3
        and len(tensor_shape) >= 3
        and parent_batch_shape[:2] == root_batch_shape[:2]
        and tensor_shape[:3] == parent_batch_shape[:3]
    )


def get_env_count(td: TensorDictBase) -> int:
    """Return rollout environment count, defaulting to one for non-rollout data."""

    if len(td.batch_size) < 2:
        return 1
    return int(td.batch_size[0])


def get_agent_count_for_field(
    td: TensorDictBase, field_path: str | Sequence[str]
) -> int | None:
    """Return the agent count for an agent-indexed tensor field."""

    parent, tensor = get_field(td, field_path)
    root_batch_shape = tuple(int(dim) for dim in td.batch_size)
    parent_batch_shape = tuple(int(dim) for dim in parent.batch_size)
    tensor_shape = tuple(int(dim) for dim in tensor.shape)
    if not _is_agent_field(root_batch_shape, parent_batch_shape, tensor_shape):
        return None
    return int(parent_batch_shape[2])


def extract_time_series(
    td: TensorDictBase,
    field_path: str | Sequence[str],
    *,
    env_index: int = 0,
    agent_mode: str = "all",
    agent_index: int = 0,
    component_index: int | str | None = None,
    max_auto_columns: int = 8,
    x_scale: float = 1.0,
    file_label: str | None = None,
) -> ExtractedSeries:
    """Extract plottable time-series lines from a rollout TensorDict field."""

    if len(td.batch_size) < 2:
        raise ValueError(
            "rollout TensorDict batch size must contain at least env and time dimensions"
        )

    key_path = _normalize_key_path(field_path)
    path_label = _join_key_path(key_path)
    parent, tensor = get_field(td, key_path)

    root_batch_shape = tuple(int(dim) for dim in td.batch_size)
    tensor_shape = tuple(int(dim) for dim in tensor.shape)
    if len(tensor_shape) < 2 or tensor_shape[:2] != root_batch_shape[:2]:
        raise ValueError(
            f"{path_label} shape {tensor_shape} does not start with rollout batch "
            f"{root_batch_shape[:2]}"
        )

    env_count, time_count = root_batch_shape[:2]
    _validate_index("env", env_index, env_count)
    if x_scale <= 0:
        raise ValueError("x_scale must be positive")

    value = _as_plot_tensor(tensor.detach().cpu())[env_index]
    x = np.arange(time_count, dtype=np.float64) * float(x_scale)
    x_label = "time" if float(x_scale) != 1.0 else "step"

    parent_batch_shape = tuple(int(dim) for dim in parent.batch_size)
    component = _parse_optional_index(component_index)
    per_agent = _is_agent_field(root_batch_shape, parent_batch_shape, tensor_shape)
    lines: list[SeriesLine] = []

    if per_agent:
        if value.ndim < 2:
            raise ValueError(f"{path_label} does not contain an agent dimension")

        agent_count = int(value.shape[1])
        trailing_shape = tuple(int(dim) for dim in value.shape[2:])
        if value.ndim == 2 or int(np.prod(trailing_shape, dtype=np.int64)) == 1:
            data = value.reshape(time_count, agent_count)
            component_suffix = None
        else:
            flat = value.reshape(time_count, agent_count, -1)
            component = 0 if component is None else component
            _validate_index("component", component, int(flat.shape[-1]))
            data = flat[:, :, component]
            component_suffix = f"component {component}"

        if agent_mode == "all":
            for idx_agent in range(agent_count):
                suffix_parts = [f"agent {idx_agent}"]
                if component_suffix:
                    suffix_parts.append(component_suffix)
                lines.append(
                    SeriesLine(
                        label=_series_label(
                            file_label, path_label, ", ".join(suffix_parts)
                        ),
                        y=data[:, idx_agent].numpy(),
                    )
                )
        elif agent_mode == "mean":
            suffix = "mean agents"
            if component_suffix:
                suffix = f"{suffix}, {component_suffix}"
            lines.append(
                SeriesLine(
                    label=_series_label(file_label, path_label, suffix),
                    y=data.mean(dim=1).numpy(),
                )
            )
        elif agent_mode == "single":
            _validate_index("agent", agent_index, agent_count)
            suffix = f"agent {agent_index}"
            if component_suffix:
                suffix = f"{suffix}, {component_suffix}"
            lines.append(
                SeriesLine(
                    label=_series_label(file_label, path_label, suffix),
                    y=data[:, agent_index].numpy(),
                )
            )
        else:
            raise ValueError(
                f"unsupported agent_mode {agent_mode!r}; expected all, mean, or single"
            )
    else:
        if value.ndim == 1:
            lines.append(
                SeriesLine(
                    label=_series_label(file_label, path_label),
                    y=value.numpy(),
                )
            )
        else:
            flat = value.reshape(time_count, -1)
            column_count = int(flat.shape[1])
            if column_count == 1:
                lines.append(
                    SeriesLine(
                        label=_series_label(file_label, path_label),
                        y=flat[:, 0].numpy(),
                    )
                )
            elif component is None and column_count <= max_auto_columns:
                for idx_col in range(column_count):
                    lines.append(
                        SeriesLine(
                            label=_series_label(
                                file_label, path_label, f"component {idx_col}"
                            ),
                            y=flat[:, idx_col].numpy(),
                        )
                    )
            else:
                if component is None:
                    raise ValueError(
                        f"{path_label} has {column_count} flattened components; "
                        "set a component index before plotting"
                    )
                _validate_index("component", component, column_count)
                lines.append(
                    SeriesLine(
                        label=_series_label(
                            file_label, path_label, f"component {component}"
                        ),
                        y=flat[:, component].numpy(),
                    )
                )

    if not lines:
        raise ValueError(f"no plottable time-series lines extracted from {path_label}")

    return ExtractedSeries(
        field_path=path_label,
        title=path_label,
        x_label=x_label,
        y_label="value",
        x=x,
        lines=tuple(lines),
    )


def infer_time_step(td: TensorDictBase, env_index: int = 0) -> float:
    """Infer a time step from common rollout fields, falling back to 1.0."""

    for candidate in (
        "next/agents/info/step_time",
        "agents/info/step_time",
    ):
        try:
            series = extract_time_series(
                td,
                candidate,
                env_index=env_index,
                agent_mode="mean",
                component_index=None,
            )
        except Exception:
            continue
        values = np.concatenate([line.y for line in series.lines])
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0]
        if positive.size:
            return float(np.mean(positive))
    return 1.0


class PlotWindow:
    """One native Tk top-level window containing an embedded Matplotlib plot."""

    def __init__(
        self,
        master,
        window_id: int,
        on_close: Callable[["PlotWindow"], None] | None = None,
    ) -> None:
        import tkinter as tk

        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        self.window_id = window_id
        self.title = f"Plot {window_id}"
        self._on_close = on_close
        self.top = tk.Toplevel(master)
        self.top.title(self.title)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.figure = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True, alpha=0.35, linestyle="--")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.top)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._line_count = 0

    def close(self) -> None:
        if self._on_close is not None:
            self._on_close(self)
        self.top.destroy()

    def add_series(self, series: ExtractedSeries) -> None:
        for line in series.lines:
            self.ax.plot(series.x, line.y, linewidth=1.1, label=line.label)
            self._line_count += 1
        self.ax.set_xlabel(series.x_label)
        self.ax.set_ylabel(series.y_label)
        self.ax.set_title(self.title)
        if self._line_count:
            self.ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def clear(self) -> None:
        self.ax.clear()
        self.ax.grid(True, alpha=0.35, linestyle="--")
        self._line_count = 0
        self.canvas.draw_idle()


class TensorDictInspectorApp:
    """Tkinter application for interactive TensorDict inspection."""

    def __init__(self, initial_path: str | Path | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("TensorDict Inspector")
        self.root.geometry("1200x760")

        self.td: TensorDictBase | None = None
        self.td_path: Path | None = None
        self.fields: list[FieldSummary] = []
        self.filtered_fields: list[FieldSummary] = []
        self.plot_windows: dict[str, PlotWindow] = {}
        self.env_count = 1
        self.current_agent_count: int | None = None
        self.next_window_id = 1

        self.path_var = tk.StringVar(value=str(initial_path) if initial_path else "")
        self.search_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Open a .td file to begin.")
        self.env_var = tk.IntVar(value=0)
        self.agent_mode_var = tk.StringVar(value="all")
        self.agent_index_var = tk.IntVar(value=0)
        self.component_var = tk.StringVar(value="")
        self.x_scale_var = tk.StringVar(value="1.0")
        self.selected_window_var = tk.StringVar(value="")

        self._build_ui()
        self.search_var.trace_add("write", lambda *_: self._refresh_field_list())
        self.agent_mode_var.trace_add(
            "write", lambda *_: self._refresh_agent_controls()
        )

        if initial_path:
            self.load_file(initial_path)
        else:
            self.root.after(100, self.open_file_dialog)

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        file_frame = ttk.Frame(self.root, padding=8)
        file_frame.grid(row=0, column=0, sticky="ew")
        file_frame.columnconfigure(1, weight=1)
        ttk.Label(file_frame, text="File").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.path_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(file_frame, text="Open...", command=self.open_file_dialog).grid(
            row=0, column=2, padx=(0, 6)
        )
        ttk.Button(file_frame, text="Reload", command=self.reload_file).grid(
            row=0, column=3
        )

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = ttk.Frame(main_pane)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        ttk.Label(left, text="Search").grid(row=0, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.search_var).grid(
            row=1, column=0, sticky="ew", pady=(0, 6)
        )

        columns = ("shape", "dtype")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings")
        self.tree.heading("#0", text="Path")
        self.tree.heading("shape", text="Shape")
        self.tree.heading("dtype", text="Dtype")
        self.tree.column("#0", width=430, stretch=True)
        self.tree.column("shape", width=160, stretch=False)
        self.tree.column("dtype", width=110, stretch=False)
        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=2, column=0, sticky="nsew")
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", lambda _: self._on_field_select())
        main_pane.add(left, weight=3)

        right = ttk.Frame(main_pane)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.metadata = tk.Text(right, height=10, wrap="none")
        self.metadata.grid(row=0, column=0, sticky="nsew")
        meta_scroll = ttk.Scrollbar(
            right, orient=tk.VERTICAL, command=self.metadata.yview
        )
        self.metadata.configure(yscrollcommand=meta_scroll.set)
        meta_scroll.grid(row=0, column=1, sticky="ns")
        main_pane.add(right, weight=2)

        controls = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(13, weight=1)

        ttk.Label(controls, text="Env").grid(row=0, column=0, sticky="w")
        self.env_spinbox = ttk.Spinbox(
            controls,
            from_=0,
            to=0,
            textvariable=self.env_var,
            width=6,
        )
        self.env_spinbox.grid(row=0, column=1, padx=(3, 12))

        ttk.Label(controls, text="Agent mode").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.agent_mode_var,
            values=("all", "mean", "single"),
            state="readonly",
            width=8,
        ).grid(row=0, column=3, padx=(3, 12))

        self.agent_index_label = ttk.Label(controls, text="Agent index")
        self.agent_index_label.grid(row=0, column=4, sticky="w")
        self.agent_spinbox = ttk.Spinbox(
            controls,
            from_=0,
            to=0,
            textvariable=self.agent_index_var,
            width=6,
            state="disabled",
        )
        self.agent_spinbox.grid(row=0, column=5, padx=(3, 12))

        ttk.Label(controls, text="Component").grid(row=0, column=6, sticky="w")
        ttk.Entry(controls, textvariable=self.component_var, width=8).grid(
            row=0, column=7, padx=(3, 12)
        )

        ttk.Label(controls, text="X scale").grid(row=0, column=8, sticky="w")
        ttk.Entry(controls, textvariable=self.x_scale_var, width=8).grid(
            row=0, column=9, padx=(3, 12)
        )

        ttk.Label(controls, text="Window").grid(row=0, column=10, sticky="w")
        self.window_combo = ttk.Combobox(
            controls,
            textvariable=self.selected_window_var,
            state="readonly",
            width=14,
        )
        self.window_combo.grid(row=0, column=11, padx=(3, 12))

        ttk.Button(controls, text="New Plot Window", command=self.new_plot_window).grid(
            row=0, column=12, padx=(0, 6)
        )
        ttk.Button(
            controls,
            text="Add To Selected Window",
            command=self.add_to_selected_window,
        ).grid(row=0, column=13, sticky="w", padx=(0, 6))
        ttk.Button(
            controls,
            text="Clear Selected Window",
            command=self.clear_selected_window,
        ).grid(row=0, column=14, sticky="w")

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 6))

    def run(self) -> None:
        self.root.mainloop()

    def open_file_dialog(self) -> None:
        from tkinter import filedialog

        initial_dir = str(Path(self.path_var.get()).expanduser().parent)
        if not Path(initial_dir).exists():
            initial_dir = str(Path.cwd())
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Open TensorDict file",
            initialdir=initial_dir,
            filetypes=(("TensorDict files", "*.td"), ("All files", "*")),
        )
        if selected:
            self.path_var.set(selected)
            self.load_file(selected)

    def reload_file(self) -> None:
        path = self.path_var.get().strip()
        if not path:
            self._show_error("No file path is set.")
            return
        self.load_file(path)

    def load_file(self, path: str | Path) -> None:
        try:
            td_path = Path(path).expanduser()
            td = load_tensordict(td_path)
            fields = flatten_tensordict(td)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.td = td
        self.td_path = td_path
        self.path_var.set(str(td_path))
        self.fields = fields
        self.env_count = get_env_count(td)
        self.env_var.set(0)
        self.env_spinbox.configure(to=max(0, self.env_count - 1))
        if self.env_count > 1:
            selected_env = self._prompt_for_env(self.env_count)
            if selected_env is not None:
                self.env_var.set(selected_env)

        self.current_agent_count = None
        self.agent_index_var.set(0)
        self._refresh_agent_controls()
        try:
            self.x_scale_var.set(f"{infer_time_step(td, int(self.env_var.get())):.8g}")
        except Exception:
            self.x_scale_var.set("1.0")
        self._refresh_field_list()
        self.status_var.set(
            f"Loaded {td_path.name}: batch_size={tuple(td.batch_size)}, "
            f"env={int(self.env_var.get())}, {len(fields)} tensor fields."
        )

    def _prompt_for_env(self, env_count: int) -> int | None:
        tk = self.tk
        ttk = self.ttk

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Environment")
        dialog.transient(self.root)
        dialog.resizable(False, False)

        selected_var = tk.IntVar(value=0)
        result: dict[str, int | None] = {"value": None}

        ttk.Label(
            dialog,
            text=f"This TensorDict contains {env_count} environments.",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 4))
        ttk.Label(dialog, text="Environment index").grid(
            row=1, column=0, sticky="w", padx=12, pady=4
        )
        ttk.Spinbox(
            dialog,
            from_=0,
            to=env_count - 1,
            textvariable=selected_var,
            width=8,
        ).grid(row=1, column=1, sticky="ew", padx=12, pady=4)

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="e", padx=12, pady=12)

        def confirm() -> None:
            value = int(selected_var.get())
            if 0 <= value < env_count:
                result["value"] = value
                dialog.destroy()
                return
            self._show_error(
                f"Environment index must be between 0 and {env_count - 1}."
            )

        def cancel() -> None:
            dialog.destroy()

        ttk.Button(button_frame, text="OK", command=confirm).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(button_frame, text="Cancel", command=cancel).grid(row=0, column=1)

        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.grab_set()
        dialog.wait_window()
        return result["value"]

    def _refresh_field_list(self) -> None:
        query = self.search_var.get().strip().lower()
        self.filtered_fields = [
            field for field in self.fields if not query or query in field.path.lower()
        ]
        self.tree.delete(*self.tree.get_children())
        for field in self.filtered_fields:
            self.tree.insert(
                "",
                "end",
                iid=field.path,
                text=field.path,
                values=(field.shape, field.dtype),
            )

    def _selected_field_path(self) -> str | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return str(selection[0])

    def _on_field_select(self) -> None:
        path = self._selected_field_path()
        if path is None:
            return
        summary = next((field for field in self.fields if field.path == path), None)
        if summary is None:
            return
        self._update_agent_count_for_field(path)
        self.metadata.configure(state="normal")
        self.metadata.delete("1.0", "end")
        self.metadata.insert("1.0", format_summary(summary))
        self.metadata.configure(state="disabled")
        self.status_var.set(f"Selected {path}.")

    def _update_agent_count_for_field(self, path: str) -> None:
        if self.td is None:
            self.current_agent_count = None
            self._refresh_agent_controls()
            return
        try:
            self.current_agent_count = get_agent_count_for_field(self.td, path)
        except Exception:
            self.current_agent_count = None

        if self.current_agent_count is not None:
            self.agent_spinbox.configure(to=max(0, self.current_agent_count - 1))
            if int(self.agent_index_var.get()) >= self.current_agent_count:
                self.agent_index_var.set(0)
        self._refresh_agent_controls()

    def _refresh_agent_controls(self) -> None:
        is_single_agent = self.agent_mode_var.get() == "single"
        has_agent_field = self.current_agent_count is not None
        state = "normal" if is_single_agent and has_agent_field else "disabled"
        self.agent_spinbox.configure(state=state)
        self.agent_index_label.configure(state=state)

    def _extract_selected_series(self) -> ExtractedSeries:
        if self.td is None:
            raise RuntimeError("No TensorDict file is loaded.")
        path = self._selected_field_path()
        if path is None:
            raise RuntimeError("Select a tensor field first.")
        file_label = self.td_path.stem if self.td_path else None
        return extract_time_series(
            self.td,
            path,
            env_index=int(self.env_var.get()),
            agent_mode=self.agent_mode_var.get(),
            agent_index=int(self.agent_index_var.get()),
            component_index=self.component_var.get(),
            x_scale=float(self.x_scale_var.get()),
            file_label=file_label,
        )

    def _register_window(self, window: PlotWindow) -> None:
        self.plot_windows[window.title] = window
        self.window_combo.configure(values=tuple(self.plot_windows.keys()))
        self.selected_window_var.set(window.title)

    def _unregister_window(self, window: PlotWindow) -> None:
        self.plot_windows.pop(window.title, None)
        names = tuple(self.plot_windows.keys())
        self.window_combo.configure(values=names)
        if self.selected_window_var.get() == window.title:
            self.selected_window_var.set(names[-1] if names else "")

    def _get_selected_window(self) -> PlotWindow | None:
        name = self.selected_window_var.get()
        return self.plot_windows.get(name)

    def new_plot_window(self) -> None:
        window = PlotWindow(
            self.root, self.next_window_id, on_close=self._unregister_window
        )
        self.next_window_id += 1
        self._register_window(window)
        try:
            series = self._extract_selected_series()
        except RuntimeError:
            self.status_var.set(f"Opened {window.title}.")
            return
        except Exception as exc:
            self._show_error(str(exc))
            return
        window.add_series(series)
        self.status_var.set(f"Added {series.field_path} to {window.title}.")

    def add_to_selected_window(self) -> None:
        window = self._get_selected_window()
        if window is None:
            window = PlotWindow(
                self.root, self.next_window_id, on_close=self._unregister_window
            )
            self.next_window_id += 1
            self._register_window(window)
        try:
            series = self._extract_selected_series()
            window.add_series(series)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.status_var.set(f"Added {series.field_path} to {window.title}.")

    def clear_selected_window(self) -> None:
        window = self._get_selected_window()
        if window is None:
            self._show_error("No plot window is selected.")
            return
        window.clear()
        self.status_var.set(f"Cleared {window.title}.")

    def _show_error(self, message: str) -> None:
        from tkinter import messagebox

        self.status_var.set(f"Error: {message}")
        messagebox.showerror("TensorDict Inspector", message, parent=self.root)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively inspect rollout TensorDict (.td) files."
    )
    parser.add_argument(
        "td_path",
        nargs="?",
        default=None,
        help="Optional path to a rollout TensorDict (.td) file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    app = TensorDictInspectorApp(initial_path=args.td_path)
    app.run()


if __name__ == "__main__":
    main()
