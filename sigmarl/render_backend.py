# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Select the VMAS render backend and provide a Matplotlib drop-in viewer.

Scenarios keep building `vmas.simulator.rendering` geoms (`extra_render`,
`Entity.render`). Only the final rasterization changes: `pyglet` draws them
with OpenGL in a window, `matplotlib` replays them off-screen with Agg, which
also works without a display (e.g., over SSH on macOS).

Select the backend with the `SIGMARL_RENDERER` environment variable:
`auto` (default; pyglet if a screen is available, otherwise Matplotlib),
`pyglet`, or `matplotlib`.
"""

import math
import os
from types import SimpleNamespace

import numpy as np
import torch

RENDERER_ENV_VAR = "SIGMARL_RENDERER"
RENDERERS = ("auto", "pyglet", "matplotlib")

_DPI = 100

# Video frame size defaults (see `compute_viewer_size`). The scale targets a
# vehicle length in pixels, so lab (1:18) and full-scale maps look alike.
RENDER_VEHICLE_PX = 48  # Target vehicle length [px]
RENDER_MAX_LONG_SIDE = 3840  # 4K UHD width
RENDER_MAX_PIXELS = 3840 * 2160  # 4K UHD area
RENDER_MIN_LONG_SIDE = 720
RENDER_MIN_SHORT_SIDE = 360
RENDER_SIZE_MULTIPLE = 16  # H.264 macroblock size
RENDER_TEXT_REFERENCE_SIZE = (1920, 1080)  # Display that text is sized for
_PYGLET_FONT_DPI = 96  # pyglet rasterizes font sizes (pt) at 96 DPI
_GL_INITIAL_LINE_WIDTH = 2.0  # Set by `rendering.Viewer.__init__`
_GL_INITIAL_COLOR = (1.0, 1.0, 1.0, 1.0)
_RC_PARAMS = {
    "text.usetex": False,
    "font.monospace": ["Courier New", "Courier", "DejaVu Sans Mono"],
    "lines.scale_dashes": True,
    "lines.solid_joinstyle": "round",
}

_renderer = None


def resolve_renderer() -> str:
    """Return the active renderer (`pyglet` or `matplotlib`), resolved once per process."""
    global _renderer
    if _renderer is None:
        choice = os.getenv(RENDERER_ENV_VAR, "auto").lower()
        if choice not in RENDERERS:
            raise ValueError(
                f"{RENDERER_ENV_VAR}={choice!r} is invalid; use one of {RENDERERS}."
            )
        if choice == "auto":
            choice = "pyglet" if _has_pyglet_screen() else "matplotlib"
            if choice == "matplotlib":
                print("[INFO] No display found; rendering with Matplotlib.")
        _renderer = choice
    return _renderer


def _has_pyglet_screen() -> bool:
    """Return whether pyglet can open a window (a screen or an EGL device exists)."""
    try:
        import pyglet

        return len(pyglet.canvas.get_display().get_screens()) > 0
    except Exception:  # No X server, no Cocoa session, no EGL device, ...
        return False


def import_vmas_rendering():
    """Import `vmas.simulator.rendering` so it works with the active renderer.

    Must run before anything else imports the module. For Matplotlib it disables
    pyglet's hidden GL shadow window, which cannot be created without a display,
    and replaces `TextLine` (a pyglet label that needs a GL context) with a
    context-free version holding the same `label` fields.
    """
    is_matplotlib = resolve_renderer() == "matplotlib"
    if is_matplotlib:
        import pyglet

        pyglet.options["shadow_window"] = False

    from vmas.simulator import rendering

    if is_matplotlib and not issubclass(rendering.TextLine, _TextLineMixin):
        rendering.TextLine = type(
            "TextLine", (_TextLineMixin, rendering.Geom), {"__module__": __name__}
        )
    return rendering


def install_render_backend(vmas_env):
    """Give a VMAS `Environment` a Matplotlib viewer if that renderer is active.

    VMAS creates its pyglet viewer only while `env.viewer is None`, so a preset
    viewer leaves the rest of `env.render` (camera bounds, `extra_render`,
    entity geoms) unchanged.
    """
    if resolve_renderer() != "matplotlib":
        return
    rendering = import_vmas_rendering()
    vmas_env.viewer = MatplotlibViewer(vmas_env.scenario)
    # Mirror `Environment._init_rendering` for agent communication messages
    vmas_env.text_lines = []
    if vmas_env.world.dim_c > 0:
        for agent in vmas_env.world.agents:
            if not agent.silent:
                text_line = rendering.TextLine(y=len(vmas_env.text_lines) * 40)
                vmas_env.viewer.geoms.append(text_line)
                vmas_env.text_lines.append(text_line)


class _TextLineMixin:
    """Pyglet-free `rendering.TextLine` exposing the label fields scenarios use."""

    def __init__(
        self, text: str = "", font_size: int = 15, x: float = 0.0, y: float = 0.0
    ):
        super().__init__()
        self.label = SimpleNamespace(
            text=text, font_size=font_size, color=(0, 0, 0, 255), x=x, y=y
        )

    def render1(self):
        raise RuntimeError("TextLine is drawn by MatplotlibViewer, not by OpenGL.")

    def set_text(self, text, font_size=None):
        self.label.text = text
        if font_size is not None:
            self.label.font_size = font_size


def compute_viewer_size(
    content_width: float,
    content_height: float,
    px_per_unit: float,
    *,
    max_long_side: int = RENDER_MAX_LONG_SIDE,
    max_pixels: int = RENDER_MAX_PIXELS,
    min_long_side: int = RENDER_MIN_LONG_SIDE,
    min_short_side: int = RENDER_MIN_SHORT_SIDE,
    multiple: int = RENDER_SIZE_MULTIPLE,
) -> tuple:
    """Size a video frame for content of `content_width` x `content_height` world units.

    The content is drawn at `px_per_unit`, reduced if the frame would exceed
    `max_long_side` or `max_pixels`, and enlarged if its long side would stay
    below `min_long_side`. A short side below `min_short_side` is padded (more
    world area is shown; the scale is kept), and both sides are rounded up to
    a multiple of `multiple` (codec friendly). Because of the rounding, the
    frame can exceed `max_pixels` by less than one `multiple` per side.

    Returns:
        (width_px, height_px, px_per_unit): the frame size and the scale used.
    """
    if not (content_width > 0 and content_height > 0 and px_per_unit > 0):
        raise ValueError(
            "Content size and px_per_unit must be positive; got "
            f"{content_width} x {content_height} at {px_per_unit} px/unit."
        )
    if not 0 < min_short_side <= min_long_side <= max_long_side:
        raise ValueError(
            "Expected 0 < min_short_side <= min_long_side <= max_long_side; got "
            f"{min_short_side}, {min_long_side}, {max_long_side}."
        )
    if min_long_side**2 > max_pixels:
        raise ValueError(f"min_long_side**2 must not exceed max_pixels ({max_pixels}).")
    long_side = max(content_width, content_height)
    scale = min(
        px_per_unit,
        max_long_side / long_side,
        math.sqrt(max_pixels / (content_width * content_height)),
    )
    scale = max(scale, min_long_side / long_side)

    def frame_side(content_side):
        side = max(content_side * scale, min_short_side)
        return int(math.ceil(side / multiple - 1e-9) * multiple)

    return frame_side(content_width), frame_side(content_height), scale


def compute_text_scale(
    width: int, height: int, reference_size: tuple = RENDER_TEXT_REFERENCE_SIZE
) -> float:
    """Return the factor for pixel-sized text (font sizes, offsets) in a frame.

    A frame larger than `reference_size` is shrunk to fit that display when
    viewed, so text enlarged by this factor keeps its apparent size. Frames that
    already fit get factor 1.
    """
    if not (width > 0 and height > 0):
        raise ValueError(f"Frame size must be positive; got {width} x {height}.")
    ref_width, ref_height = reference_size
    return max(1.0, width / ref_width, height / ref_height)


def _validate_viewer_size(viewer_size) -> tuple:
    """Return `viewer_size` as positive integer pixels (width, height)."""
    width, height = viewer_size
    if any(not float(size).is_integer() or int(size) < 1 for size in (width, height)):
        raise ValueError(
            f"Viewer size must be positive integer pixels; got {viewer_size}."
        )
    return int(width), int(height)


def _inches_for_pixels(pixels: int, dpi: float) -> float:
    """Return the figure size in inches that Agg rasterizes to exactly `pixels`.

    Matplotlib truncates `inches * dpi` to whole pixels, and `pixels / dpi * dpi`
    can round to just below `pixels` (e.g., 57 / 100 * 100 = 56.99...), which
    would drop one pixel. Step the size up by single ulps until it is not below.
    """
    inches = pixels / dpi
    while inches * dpi < pixels:
        inches = math.nextafter(inches, math.inf)
    return inches


class MatplotlibViewer:
    """Drop-in replacement of `vmas.simulator.rendering.Viewer` drawing with Agg.

    Geoms are replayed with the same semantics as their OpenGL `render`
    methods, including the GL state they leak (current color and line width),
    so frames match the pyglet viewer without per-scenario drawing code.
    The canvas takes `scenario.viewer_size` at the first render, so callers can
    still resize it after the env is built.
    """

    def __init__(self, scenario):
        self._scenario = scenario
        self.geoms = []
        self.onetime_geoms = []
        self.bounds = None
        self._figure = None
        self._artists = []
        self._pending_polygons = []
        self._gl_color = _GL_INITIAL_COLOR
        self._gl_line_width = _GL_INITIAL_LINE_WIDTH
        self._gl_line_style = None

    def close(self):
        self._figure = None

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        self.bounds = torch.tensor([left, right, bottom, top], device=left.device)

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def add_onetime_list(self, geoms):
        self.onetime_geoms.extend(geoms)

    def render(self, return_rgb_array=False):
        """Draw all geoms and return an (H, W, 3) uint8 frame if requested."""
        import matplotlib

        # Other modules change global rcParams (e.g., `text.usetex`) for plots
        with matplotlib.rc_context(_RC_PARAMS):
            return self._render(return_rgb_array)

    def _render(self, return_rgb_array):
        if self._figure is None:
            self._init_figure()
        self._rendering = import_vmas_rendering()

        for artist in self._artists:
            artist.remove()
        self._artists = []

        left, right, bottom, top = self.bounds.tolist()
        self._ax.set_xlim(left, right)
        self._ax.set_ylim(bottom, top)

        text_lines = []
        for geom in [*self.geoms, *self.onetime_geoms]:
            if isinstance(geom, self._rendering.TextLine):
                text_lines.append(geom)
            else:
                self._render_geom(geom, [])
        self._flush_polygons()
        for text_line in text_lines:
            self._render_text(text_line)
        self.onetime_geoms = []

        self._canvas.draw()
        if not return_rgb_array:
            return None
        return np.asarray(self._canvas.buffer_rgba())[:, :, :3].copy()

    def _init_figure(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.width, self.height = _validate_viewer_size(self._scenario.viewer_size)
        self._figure = Figure(
            figsize=(
                _inches_for_pixels(self.width, _DPI),
                _inches_for_pixels(self.height, _DPI),
            ),
            dpi=_DPI,
            facecolor="white",
        )
        self._canvas = FigureCanvasAgg(self._figure)
        if self._canvas.get_width_height() != (self.width, self.height):
            raise RuntimeError(
                f"Matplotlib canvas is {self._canvas.get_width_height()} pixels; "
                f"expected viewer size {(self.width, self.height)}."
            )
        self._ax = self._figure.add_axes((0, 0, 1, 1))
        self._ax.set_axis_off()

    def _render_geom(self, geom, outer_transforms):
        """Replay `Geom.render`: enable attrs in reverse, draw, disable."""
        rendering = self._rendering
        transforms = [
            attr for attr in geom.attrs if isinstance(attr, rendering.Transform)
        ] + outer_transforms
        for attr in reversed(geom.attrs):
            if isinstance(attr, rendering.Color):
                self._gl_color = tuple(attr.vec4)
            elif isinstance(attr, rendering.LineWidth):
                self._gl_line_width = attr.stroke
            elif isinstance(attr, rendering.LineStyle):
                self._gl_line_style = attr.style
            elif not isinstance(attr, rendering.Transform):
                raise TypeError(f"Unsupported render attribute {type(attr).__name__}.")

        self._render1(geom, transforms)

        if any(isinstance(attr, rendering.LineStyle) for attr in geom.attrs):
            self._gl_line_style = None

    def _render1(self, geom, transforms):
        """Replay `render1` of each VMAS geom type."""
        rendering = self._rendering
        if isinstance(geom, rendering.Compound):
            for child in geom.gs:
                self._render_geom(child, transforms)
        elif isinstance(geom, rendering.FilledPolygon):
            points = _transform_points(geom.v, transforms)
            self._add_polygon(points)
            if geom.draw_border:
                self._gl_color = tuple(channel * 0.5 for channel in self._gl_color)
                self._add_line(points, close=True)
        elif isinstance(geom, rendering.PolyLine):
            self._add_line(_transform_points(geom.v, transforms), close=geom.close)
        elif isinstance(geom, rendering.Line):
            self._add_line(_transform_points([geom.start, geom.end], transforms))
        elif isinstance(geom, rendering.Point):
            self._add_point(_transform_points([(0.0, 0.0)], transforms))
        elif isinstance(geom, rendering.Grid):
            half = geom.length / 2
            for point in np.arange(-half, half, geom.spacing):
                for line in (
                    ((point, -half), (point, half)),
                    ((-half, point), (half, point)),
                ):
                    self._add_line(_transform_points(line, transforms))
        else:
            raise TypeError(
                f"MatplotlibViewer cannot draw {type(geom).__name__}; "
                f"use {RENDERER_ENV_VAR}=pyglet."
            )

    def _render_text(self, text_line):
        """Draw a label in pixel coordinates, anchored at its bottom left."""
        label = text_line.label
        transforms = [
            attr
            for attr in text_line.attrs
            if isinstance(attr, self._rendering.Transform)
        ]
        ((x, y),) = _transform_points([(label.x, label.y)], transforms)
        artist = self._figure.text(
            x / self.width,
            y / self.height,
            label.text,
            fontsize=label.font_size * _PYGLET_FONT_DPI / _DPI,
            family="monospace",
            color=_rgba(np.asarray(label.color) / 255),
            ha="left",
            va="bottom",
            parse_math=False,
            zorder=len(self._artists),
        )
        self._artists.append(artist)

    def _add_polygon(self, points):
        # Batched with the following polygons; one artist per polygon is slow
        self._pending_polygons.append((points, _rgba(self._gl_color)))

    def _flush_polygons(self):
        """Draw the pending polygons as one collection, keeping their order."""
        if not self._pending_polygons:
            return
        from matplotlib.collections import PolyCollection

        polygons, face_colors = zip(*self._pending_polygons)
        self._pending_polygons = []
        artist = PolyCollection(
            polygons,
            closed=True,
            facecolors=face_colors,
            # A thin edge in the face color hides antialiasing seams
            # between adjacent polygons
            edgecolors="face",
            linewidths=_pixels_to_points(0.5),
            zorder=len(self._artists),
        )
        self._ax.add_collection(artist, autolim=False)
        self._artists.append(artist)

    def _add_line(self, points, close=False):
        from matplotlib.lines import Line2D

        self._flush_polygons()

        if close:
            points = np.vstack([points, points[:1]])
        line_width = _pixels_to_points(self._gl_line_width)
        artist = Line2D(
            points[:, 0],
            points[:, 1],
            color=_rgba(self._gl_color),
            linewidth=line_width,
            linestyle=_stipple_to_linestyle(self._gl_line_style, line_width),
            solid_capstyle="butt",
            zorder=len(self._artists),
        )
        self._ax.add_line(artist)
        self._artists.append(artist)

    def _add_point(self, points):
        from matplotlib.lines import Line2D

        self._flush_polygons()

        artist = Line2D(
            points[:, 0],
            points[:, 1],
            linestyle="none",
            marker="s",
            markersize=_pixels_to_points(1.0),
            markeredgewidth=0,
            color=_rgba(self._gl_color),
            zorder=len(self._artists),
        )
        self._ax.add_line(artist)
        self._artists.append(artist)


def _to_numpy(value):
    """Convert (nested sequences of) tensors and numbers to a float array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(float)
    if isinstance(value, (list, tuple)):
        return np.array([_to_numpy(item) for item in value], dtype=float)
    return np.asarray(value, dtype=float)


def _transform_points(vertices, transforms):
    """Apply VMAS `Transform`s (scale, rotate, translate) innermost first."""
    points = _to_numpy(vertices).reshape(-1, 2)
    for transform in transforms:
        cos, sin = math.cos(transform.rotation), math.sin(transform.rotation)
        scaled = points * np.asarray(transform.scale)
        points = np.stack(
            [
                cos * scaled[:, 0] - sin * scaled[:, 1],
                sin * scaled[:, 0] + cos * scaled[:, 1],
            ],
            axis=1,
        ) + np.asarray(transform.translation)
    return points


def _rgba(color):
    return tuple(float(np.clip(channel, 0.0, 1.0)) for channel in color)


def _pixels_to_points(pixels):
    return pixels * 72 / _DPI


def _stipple_to_linestyle(pattern, line_width):
    """Convert a 16-bit OpenGL stipple pattern (1 bit = 1 px) to a Matplotlib dash style."""
    if pattern is None or pattern & 0xFFFF == 0xFFFF:
        return "solid"
    if pattern & 0xFFFF == 0:
        return "none"
    bits = [(pattern >> i) & 1 for i in range(16)]
    first_on = bits.index(1)
    bits = bits[first_on:] + bits[:first_on]
    runs = []
    for bit in bits:
        if runs and runs[-1][0] == bit:
            runs[-1][1] += 1
        else:
            runs.append([bit, 1])
    if len(runs) % 2:  # A trailing "on" run wraps around into the first one
        runs[0][1] += runs.pop()[1]
    # Matplotlib scales dash lengths by the line width
    return (0, [_pixels_to_points(length) / line_width for _, length in runs])
