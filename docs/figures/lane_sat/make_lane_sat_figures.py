# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the figures of the rectangle_sat lane barrier section.

Figure 1 shows the whole safety-filter pipeline (CBF-QP with the backup-horizon
rows of BH-TTCBF; sigmarl/cbf_qp_proxsuite.py, sigmarl/backup_horizon_cbf.py).

The figures illustrate the exact lane-boundary margins of BH-CBF (lane_cbf_footprint =
rectangle_sat). Geometry follows sigmarl/lane_sat_barrier.py (LaneSatBarrier
.facets, .atom_tables) and dynamics sigmarl/cbf_qp_proxsuite.py
(_lane_sat_predict).

Usage:
    python docs/figures/lane_sat/make_lane_sat_figures.py
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Vehicle profile real_dyn_2
HALF_LENGTH, HALF_WIDTH = 0.11, 0.0535
L_R, L_WB = 0.075, 0.15
DT = 0.1
LAMBDA = 0.2

COLORS = dict(
    car="#2459c4",
    car_fill="#dbe5f7",
    edge="#27313c",
    lane="#eef1f4",
    bad="#c3372b",
    good="#1d8456",
    violet="#7547cc",
    amber="#a86d12",
    muted="#6b7684",
)

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

# Local corner offsets c1..c4: front-left, front-right, rear-right, rear-left
CORNERS_LOCAL = np.array(
    [
        [HALF_LENGTH, HALF_WIDTH],
        [HALF_LENGTH, -HALF_WIDTH],
        [-HALF_LENGTH, -HALF_WIDTH],
        [-HALF_LENGTH, HALF_WIDTH],
    ]
)


def rotation(psi: float) -> np.ndarray:
    """2D rotation matrix."""
    return np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])


def corners(r, psi: float) -> np.ndarray:
    """World corners c1..c4 of the vehicle rectangle."""
    return np.asarray(r) + CORNERS_LOCAL @ rotation(psi).T


def e_lon(psi: float) -> np.ndarray:
    return np.array([math.cos(psi), math.sin(psi)])


def e_lat(psi: float) -> np.ndarray:
    return np.array([-math.sin(psi), math.cos(psi)])


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def save(fig, name: str):
    """Save a figure as PNG (for the markdown) and PDF (for the paper)."""
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=220)
    plt.close(fig)
    print(f"saved {name}")


def arrow(ax, start, vec, color, lw=1.3, style="-|>", ms=9, zorder=5):
    start = np.asarray(start, dtype=float)
    ax.annotate(
        "",
        xy=start + np.asarray(vec),
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=ms,
        ),
        zorder=zorder,
    )


def draw_car(ax, r, psi, fill=True, color=None, lw=1.4, ls="-", alpha=1.0, z=3):
    color = color or COLORS["car"]
    ax.add_patch(
        Polygon(
            corners(r, psi),
            closed=True,
            facecolor=COLORS["car_fill"] if fill else "none",
            edgecolor=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=z,
        )
    )


def lane_normal(a, b, sigma: int) -> np.ndarray:
    """Lane-pointing unit normal of segment a->b (sigma=+1: left boundary)."""
    t = unit(np.asarray(b) - np.asarray(a))
    return np.array([t[1], -t[0]]) if sigma == 1 else np.array([-t[1], t[0]])


def atoms(a, b, sigma, r, psi, eps):
    """Atom values of one segment per direction, as in LaneSatBarrier.atom_tables."""
    a, b, r = (np.asarray(v, dtype=float) for v in (a, b, r))
    n = lane_normal(a, b, sigma)
    cs = corners(r, psi)
    ends = np.stack([a, b])
    lon = (ends - r) @ e_lon(psi)
    lat = sigma * ((ends - r) @ e_lat(psi))
    return {
        "normal": cs @ n - n @ a - eps,
        "front": lon - HALF_LENGTH - eps,
        "rear": -lon - HALF_LENGTH - eps,
        "side": lat - HALF_WIDTH - eps,
    }


def euler_step(state, u):
    """One explicit-Euler step of the kinematic bicycle model."""
    x, y, psi, v, delta = state
    beta = math.atan(L_R / L_WB * math.tan(delta))
    return np.array(
        [
            x + DT * v * math.cos(psi + beta),
            y + DT * v * math.sin(psi + beta),
            psi + DT * v * math.sin(beta) / L_R,
            v + DT * u[0],
            delta + DT * u[1],
        ]
    )


def point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - a - t * ab))


# ---------------------------------------------------------------------------
# Figure 1: control pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    """Safety-filter pipeline of one control step (update_centralized_cbf_qp)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, body, color, fc="white"):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                fc=fc,
                ec=color,
                lw=1.2,
            )
        )
        ax.text(
            x,
            y + h / 2 - 0.035,
            title,
            ha="center",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x,
            y + h / 2 - 0.11,
            body,
            ha="center",
            va="top",
            fontsize=7.0,
            color=COLORS["edge"],
            linespacing=1.25,
        )

    def link(p, q, text="", color=COLORS["muted"], rad=0.0, off=(0, 0.025), **kw):
        ax.annotate(
            "",
            xy=q,
            xytext=p,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.1,
                shrinkA=0,
                shrinkB=0,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )
        if text:
            ax.text(
                (p[0] + q[0]) / 2 + off[0],
                (p[1] + q[1]) / 2 + off[1],
                text,
                ha="center",
                va="bottom",
                fontsize=7.4,
                color=color,
                **kw,
            )

    y, h = 0.68, 0.5
    xs = [0.1, 0.5, 0.88]
    ws = [0.18, 0.4, 0.18]
    box(
        xs[0],
        y,
        ws[0],
        h,
        "Nominal policy",
        "MAPPO policy\n$\\pi(o_k)$ or CLF;\naction converted\nto $u^{\\mathrm{nom}}=(a,\\omega)$",
        COLORS["edge"],
    )
    box(
        xs[1],
        y,
        ws[1],
        h,
        "CBF-QP with BH-TTCBF",
        "$\\min\\,\\|u-u^{\\mathrm{nom}}\\|^2$ s.t. input box and\n"
        "tail rows (6), (8): exact SAT barriers along the\n"
        "backup rollout to standstill, $t=1..N$:\n"
        "$m_e(\\Psi_t(X_k,U)) \\geq (1-\\lambda)\\,B_e(X_k)$;\n"
        "exact check and repair (Fig. 9)",
        COLORS["car"],
        fc="#f3f6fc",
    )
    box(
        xs[2],
        y,
        ws[2],
        h,
        "Simulator",
        "explicit Euler\nbicycle model;\ncollision checker\n(lanes, vehicles)",
        COLORS["edge"],
    )
    y_main = y - 0.1
    for i, text in enumerate(("$u^{\\mathrm{nom}}$", "$U_k = u^\\star$")):
        link(
            (xs[i] + ws[i] / 2 + 0.006, y_main),
            (xs[i + 1] - ws[i + 1] / 2 - 0.006, y_main),
            text,
            off=(0, 0.012),
        )

    # Fallback of the QP
    ax.text(
        xs[1],
        y + h / 2 + 0.03,
        "tail rows violated after repairs, or QP unsolved $\\to$ backup control"
        " of the vehicles involved",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=COLORS["bad"],
    )

    # Feedback of the next state
    yb = 0.2
    ax.plot(
        [xs[2], xs[2], xs[0]],
        [y - h / 2 - 0.006, yb, yb],
        color=COLORS["muted"],
        lw=1.1,
    )
    link((xs[0], yb), (xs[0], y - h / 2 - 0.006))
    ax.text(
        0.5,
        yb - 0.02,
        "next joint state $X_{k+1}$, observations $o_{k+1}$ "
        "(decentralized: one broadcast of states and routes)",
        ha="center",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )
    ax.plot([xs[1]], [yb], marker="o", ms=3, color=COLORS["muted"])
    link((xs[1], yb), (xs[1], y - h / 2 - 0.006))

    # Guarantees
    ax.text(
        xs[1] + 0.012,
        0.315,
        "Thms. 2, 3: recursive feasibility,\nlane + vehicle safety (centralized,\nfully decentralized)",
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["car"],
        style="italic",
    )
    ax.set_ylim(0.1, 1)
    save(fig, "fig_lane_sat_pipeline")


# ---------------------------------------------------------------------------
# Figure 2: motivation
# ---------------------------------------------------------------------------
def fig_motivation():
    fig = plt.figure(figsize=(7.2, 4.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.28)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    # (a) Scene: corner passes a vertex that turns toward the lane
    theta = math.radians(11.5)
    q0, q1 = np.array([-0.45, 0.0]), np.array([0.0, 0.0])
    q2 = 0.45 * np.array([math.cos(theta), math.sin(theta)])
    clearance, v, horizon = 0.02, 0.6, 2 * DT
    x_front = -0.03
    ax = ax_a
    ax.add_patch(
        Polygon([q0, q1, q2, [0.45, 0.2], [-0.45, 0.2]], color=COLORS["lane"], lw=0)
    )
    ax.plot(*np.stack([q0, q1, q2]).T, color=COLORS["edge"], lw=2)
    ax.plot([0, 0.42], [0, 0], color=COLORS["violet"], lw=1.2, ls="--")
    ax.plot(0, 0, "o", color=COLORS["edge"], ms=4)
    for x_f, ls, label in (
        (x_front, "-", "$t$"),
        (x_front + v * horizon, "--", "$t+T$"),
    ):
        rect = np.array(
            [
                [x_f, clearance],
                [x_f, clearance + 2 * HALF_WIDTH],
                [x_f - 2 * HALF_LENGTH, clearance + 2 * HALF_WIDTH],
                [x_f - 2 * HALF_LENGTH, clearance],
            ]
        )
        ax.add_patch(
            Polygon(
                rect,
                facecolor=COLORS["car_fill"] if ls == "-" else "none",
                edgecolor=COLORS["car"],
                lw=1.3,
                ls=ls,
            )
        )
        ax.plot(
            x_f,
            clearance,
            "o",
            color=COLORS["car"],
            ms=4,
            mfc="white" if ls == "--" else COLORS["car"],
        )
        ax.text(
            x_f - HALF_LENGTH,
            clearance + HALF_WIDTH,
            label,
            ha="center",
            va="center",
            color=COLORS["car"],
            fontsize=10,
        )
    arrow(ax, [-0.36, 0.0], [0, 0.045], COLORS["edge"])
    ax.text(-0.355, 0.05, "$n_j$", fontsize=10, va="bottom")
    n2 = np.array([-math.sin(theta), math.cos(theta)])
    arrow(ax, q2 * 0.75, 0.045 * n2, COLORS["edge"])
    ax.text(*(q2 * 0.75 + 0.05 * n2), "$n_{j+1}$", fontsize=10, ha="right")
    ax.text(-0.22, -0.018, "$F_j$", fontsize=10, ha="center", va="top")
    ax.text(*(q2 * 0.55 + np.array([0.0, -0.02])), "$F_{j+1}$", fontsize=10, va="top")
    t_arc = np.linspace(0, theta, 20)
    ax.plot(0.3 * np.cos(t_arc), 0.3 * np.sin(t_arc), color=COLORS["muted"], lw=0.8)
    ax.text(0.31, 0.03, r"$\Delta\theta$", fontsize=10, va="center")
    ax.text(
        0.42,
        -0.012,
        "edge assumed by the\nTaylor prediction",
        color=COLORS["violet"],
        ha="right",
        va="top",
        fontsize=8,
    )
    arrow(
        ax,
        [x_front - 0.2, clearance + 2 * HALF_WIDTH + 0.012],
        [v * horizon, 0],
        COLORS["car"],
    )
    ax.text(
        x_front - 0.2 + v * horizon / 2,
        clearance + 2 * HALF_WIDTH + 0.017,
        r"$vT$",
        color=COLORS["car"],
        ha="center",
        fontsize=10,
    )
    ax.set_xlim(-0.46, 0.46)
    ax.set_ylim(-0.035, 0.165)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "(a) A corner of the vehicle passes vertex $q_{j+1}$ within the horizon $T=2\\Delta t$",
        loc="left",
    )

    # (b) Clearance of the corner over the horizon
    ts = np.linspace(0, horizon, 200)
    true = []
    for t in ts:
        p = np.array([x_front + v * t, clearance])
        d = min(point_segment_distance(p, q0, q1), point_segment_distance(p, q1, q2))
        inside = p[0] > 0 and np.dot(p - q1, n2) < 0
        true.append(-d if inside else d)
    true = np.array(true) * 1000
    ax = ax_b
    ax.plot(
        ts,
        np.full_like(ts, clearance * 1000),
        color=COLORS["violet"],
        ls="--",
        lw=1.6,
        label="Taylor prediction (uses $n_j$)",
    )
    ax.plot(ts, true, color=COLORS["edge"], lw=1.8, label="true clearance")
    ax.fill_between(
        [horizon - 0.006, horizon],
        true[-1],
        clearance * 1000,
        color=COLORS["bad"],
        alpha=0.25,
        lw=0,
    )
    ax.annotate(
        r"error $\approx vT\sin\Delta\theta$",
        xy=(horizon - 0.006, (true[-1] + clearance * 1000) / 2),
        xytext=(0.075, 6),
        color=COLORS["bad"],
        arrowprops=dict(arrowstyle="-", color=COLORS["bad"], lw=0.8),
    )
    ax.axhline(0, color=COLORS["bad"], lw=0.8, ls=":")
    ax.set_xticks([0, DT, 2 * DT], ["$t$", r"$t+\Delta t$", r"$t+2\Delta t$"])
    ax.set_ylabel("clearance of the corner [mm]")
    ax.set_ylim(-2, 24)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title(
        r"(b) Prediction error ($v=0.6$ m/s, $\Delta\theta=11.5^\circ$)", loc="left"
    )

    # (c) Blended vs softmin distance across a vertex
    theta_c, c0, width = math.radians(15), 0.03, 0.03
    xs = np.linspace(-0.06, 0.06, 400)
    p0, p1 = np.array([-1.0, 0.0]), np.array([0.0, 0.0])
    p2 = np.array([math.cos(theta_c), math.sin(theta_c)])
    d1 = np.full_like(xs, c0)
    d2 = c0 * math.cos(theta_c) - xs * math.sin(theta_c)
    true_c = np.array(
        [
            min(
                point_segment_distance(np.array([x, c0]), p0, p1),
                point_segment_distance(np.array([x, c0]), p1, p2),
            )
            for x in xs
        ]
    )
    s = np.clip((xs + width) / (2 * width), 0, 1)
    w = s * s * (3 - 2 * s)
    blend = (1 - w) * d1 + w * d2
    beta = 4 / (width * math.sin(theta_c))
    m = np.minimum(d1, d2)
    soft = m - np.log(np.exp(-beta * (d1 - m)) + np.exp(-beta * (d2 - m))) / beta
    ax = ax_c
    ax.fill_between(
        xs * 1000,
        true_c * 1000,
        blend * 1000,
        where=blend > true_c + 1e-7,
        color=COLORS["bad"],
        alpha=0.25,
        lw=0,
        label="overestimate",
    )
    ax.plot(
        xs * 1000,
        true_c * 1000,
        color=COLORS["edge"],
        lw=1.8,
        label=r"true: $\min(d_j, d_{j+1})$",
    )
    ax.plot(
        xs * 1000,
        blend * 1000,
        color=COLORS["amber"],
        lw=1.6,
        ls="--",
        label=r"blend: $(1-w)d_j + w\,d_{j+1}$",
    )
    ax.plot(xs * 1000, soft * 1000, color=COLORS["violet"], lw=1.4, label="softmin")
    ax.axvline(0, color=COLORS["muted"], lw=0.6, ls=":")
    ax.set_xlabel("position relative to the vertex [mm]")
    ax.set_ylabel("clearance [mm]")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax.set_title(
        r"(c) Blending normals at a vertex ($\Delta\theta=15^\circ$)", loc="left"
    )
    save(fig, "fig_lane_sat_motivation")


# ---------------------------------------------------------------------------
# Figure 3: notation
# ---------------------------------------------------------------------------
def fig_notation():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    q = np.array([[-0.34, 0.10], [-0.13, 0.118], [0.12, 0.112], [0.33, 0.075]])
    eps_vis = 0.012
    r = np.array([-0.02, -0.045])
    psi = math.radians(8)
    ax.add_patch(
        Polygon(
            np.vstack([q, [[0.33, -0.12], [-0.34, -0.12]]]), color=COLORS["lane"], lw=0
        )
    )
    ax.plot(*q.T, color=COLORS["edge"], lw=2, zorder=4)
    ax.plot(*q.T, "o", color=COLORS["edge"], ms=4, zorder=5)
    for i, label in enumerate(["$q_{j-1}$", "$q_j$", "$q_{j+1}$", "$q_{j+2}$"]):
        ax.text(q[i, 0], q[i, 1] + 0.014, label, ha="center", va="bottom", fontsize=10)
    a, b = q[1], q[2]
    t = unit(b - a)
    n = lane_normal(a, b, 1)
    ax.add_patch(
        Polygon(
            [a, b, b + eps_vis * n, a + eps_vis * n],
            facecolor=COLORS["amber"],
            alpha=0.25,
            lw=0,
        )
    )
    ax.annotate(
        r"buffer $\varepsilon_j$",
        xy=a + 0.8 * (b - a) + 0.5 * eps_vis * n,
        xytext=(0.2, 0.065),
        color=COLORS["amber"],
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=COLORS["amber"], lw=0.7),
    )
    mid = 0.5 * (a + b)
    ax.text(*(mid + np.array([-0.075, 0.012])), "$F_j$", fontsize=11, ha="center")
    arrow(ax, mid, 0.07 * t, COLORS["edge"])
    ax.text(*(mid + 0.075 * t + np.array([0.003, 0.004])), "$t_j$", fontsize=10)
    arrow(ax, mid + eps_vis * n, 0.035 * n, COLORS["edge"])
    ax.text(
        *(mid + (eps_vis + 0.03) * n + np.array([0.008, 0])),
        "$n_j$",
        fontsize=10,
        va="center",
    )
    ax.text(
        0.35,
        -0.15,
        "left boundary ($\\sigma=+1$),\nlane on its right",
        ha="right",
        fontsize=8,
        color=COLORS["muted"],
    )

    draw_car(ax, r, psi)
    cs = corners(r, psi)
    offsets = [(0.006, 0.006), (0.006, -0.012), (-0.026, -0.006), (-0.014, 0.006)]
    for i, (c, off) in enumerate(zip(cs, offsets)):
        ax.plot(*c, "o", color=COLORS["car"], ms=3.5, zorder=6)
        ax.text(
            c[0] + off[0],
            c[1] + off[1],
            f"$c_{i + 1}$",
            color=COLORS["car"],
            fontsize=10,
        )
    ax.plot(*r, "o", color=COLORS["car"], ms=4, zorder=6)
    ax.text(
        r[0] - 0.012, r[1] - 0.012, "$r$", color=COLORS["car"], fontsize=10, ha="right"
    )
    arrow(ax, r, 0.075 * e_lon(psi), COLORS["car"], lw=1.4)
    ax.text(
        *(r + 0.078 * e_lon(psi) + np.array([-0.01, 0.006])),
        r"$e_{\mathrm{lon}}$",
        color=COLORS["car"],
        fontsize=10,
        ha="right",
    )
    arrow(ax, r, 0.042 * e_lat(psi), COLORS["car"], lw=1.4)
    ax.text(
        *(r + 0.045 * e_lat(psi) + np.array([0.006, -0.004])),
        r"$e_{\mathrm{lat}}$",
        color=COLORS["car"],
        fontsize=10,
    )
    ax.plot(
        [r[0], r[0] + 0.2],
        [r[1], r[1]],
        color=COLORS["muted"],
        lw=0.7,
        ls="--",
        zorder=6,
    )
    angles = np.linspace(0, psi, 20)
    ax.plot(
        r[0] + 0.17 * np.cos(angles),
        r[1] + 0.17 * np.sin(angles),
        color=COLORS["muted"],
        lw=0.9,
        zorder=6,
    )
    ax.text(r[0] + 0.175, r[1] + 0.008, r"$\psi$", fontsize=10, color=COLORS["muted"])
    # Half-length and half-width dimension lines
    rear_mid = r - HALF_LENGTH * e_lon(psi)
    ax.plot(*np.stack([rear_mid, r]).T, color=COLORS["muted"], lw=0.9, zorder=6)
    ax.text(
        *(0.5 * (rear_mid + r) + np.array([0, 0.006])),
        "$a_v$",
        ha="center",
        fontsize=10,
        color=COLORS["muted"],
    )
    side_mid = r - HALF_WIDTH * e_lat(psi) - 0.05 * e_lon(psi)
    ax.plot(
        *np.stack([r - 0.05 * e_lon(psi), side_mid]).T,
        color=COLORS["muted"],
        lw=0.9,
        zorder=6,
    )
    ax.text(
        *(0.5 * (r - 0.05 * e_lon(psi) + side_mid) + np.array([-0.012, 0])),
        "$b_v$",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["muted"],
    )
    ax.text(
        -0.35,
        -0.15,
        r"pose $\pi=(p_x,p_y,\psi)$, position $r=(p_x,p_y)$, rectangle $\mathcal{R}(\pi)$",
        fontsize=9,
    )
    ax.set_xlim(-0.36, 0.36)
    ax.set_ylim(-0.16, 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "fig_lane_sat_notation")


# ---------------------------------------------------------------------------
# Figure 4: the four separating directions of one segment
# ---------------------------------------------------------------------------
DIR_SEGMENT = (np.array([0.03, 0.085]), np.array([0.20, 0.115]))
DIR_POSE = (np.array([0.0, 0.0]), 0.0)
DIR_EPS = 0.005
DIRECTION_LABEL_OFFSETS = {
    "normal": (-70, 10),
    "front": (-150, 5),
    "rear": (-150, 5),
    "side": (25, -12),
}
DIRECTION_SYMBOLS = {
    "front": r"$e_{\mathrm{lon}}$",
    "rear": r"$-e_{\mathrm{lon}}$",
    "side": r"$\sigma e_{\mathrm{lat}}$",
}


def fig_directions():
    a, b = DIR_SEGMENT
    r, psi = DIR_POSE
    sigma, eps = 1, DIR_EPS
    values = atoms(a, b, sigma, r, psi, eps)
    n = lane_normal(a, b, sigma)
    t = unit(b - a)
    cs = corners(r, psi)
    mins = {k: float(v.min()) for k, v in values.items()}
    best = max(mins, key=mins.get)

    specs = [
        ("normal", r"$s=n_j$", "4 corners $c_m$"),
        ("front", r"$s=e_{\mathrm{lon}}$", "endpoints $q_j, q_{j+1}$"),
        ("rear", r"$s=-e_{\mathrm{lon}}$", "endpoints $q_j, q_{j+1}$"),
        ("side", r"$s=\sigma e_{\mathrm{lat}}$", "endpoints $q_j, q_{j+1}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for ax, (key, title, pts_label), letter in zip(axes.ravel(), specs, "abcd"):
        ax.add_patch(
            Polygon([a, b, [0.32, 0.17], [-0.36, 0.17]], color="#f6f2ea", lw=0)
        )
        ax.plot(*np.stack([a, b]).T, color=COLORS["edge"], lw=2.2, zorder=4)
        ax.text(
            *(0.5 * (a + b) + np.array([0.0, 0.012])), "$F_j$", ha="center", fontsize=10
        )
        draw_car(ax, r, psi, alpha=0.9)
        vals = values[key]
        if key == "normal":
            direction = n
            ref_point = a + eps * n
            ref_dir = t
            points = cs
            feet = [c - g * n for c, g in zip(cs, vals)]
            arrow(ax, 0.5 * (a + b), 0.04 * n, COLORS["edge"])
            ax.text(
                *(0.5 * (a + b) + 0.045 * n + np.array([0.005, 0])),
                "$n_j$",
                fontsize=10,
            )
        else:
            direction = {
                "front": e_lon(psi),
                "rear": -e_lon(psi),
                "side": sigma * e_lat(psi),
            }[key]
            half = HALF_WIDTH if key == "side" else HALF_LENGTH
            ref_point = r + (half + eps) * direction
            ref_dir = np.array([-direction[1], direction[0]])
            points = np.stack([a, b])
            feet = [p - g * direction for p, g in zip(points, vals)]
            arrow(ax, r, 0.06 * direction, COLORS["car"])
            ax.text(
                *(r + 0.065 * direction + np.array([0.004, 0.004])),
                DIRECTION_SYMBOLS[key],
                color=COLORS["car"],
                fontsize=10,
            )
        line = np.stack([ref_point - 0.5 * ref_dir, ref_point + 0.5 * ref_dir])
        ax.plot(*line.T, color=COLORS["muted"], lw=0.8, ls="--", zorder=2)
        i_min = int(np.argmin(vals))
        for i, (p, f, g) in enumerate(zip(points, feet, vals)):
            color = COLORS["good"] if g >= 0 else COLORS["bad"]
            ax.plot(
                *np.stack([f, p]).T,
                color=color,
                lw=2.2 if i == i_min else 1.0,
                zorder=6,
            )
            ax.plot(*p, "o", color=color, ms=3.5, zorder=7)
        p_min, f_min = points[i_min], feet[i_min]
        label_pos = 0.5 * (p_min + f_min)
        ax.annotate(
            f"$g_{{j,s,{i_min + 1}}}={vals[i_min] * 1000:.0f}$ mm (min)",
            xy=label_pos,
            xytext=DIRECTION_LABEL_OFFSETS[key],
            textcoords="offset points",
            fontsize=8.5,
            color=COLORS["good"] if vals[i_min] >= 0 else COLORS["bad"],
            arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.6),
            zorder=8,
        )
        verdict = "separates" if mins[key] >= 0 else "does not separate"
        star = "  (best)" if key == best else ""
        ax.set_title(
            f"({letter}) {title}, check points: {pts_label}\n"
            + rf"$\mathrm{{min}}_i\, g_{{j,s,i}}={mins[key] * 1000:.0f}$ mm, {verdict}{star}",
            loc="left",
            fontsize=8.8,
        )
        ax.set_xlim(-0.36, 0.32)
        ax.set_ylim(-0.12, 0.17)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(
        rf"$h_j=\max_s\ \min_i\ g_{{j,s,i}} = {mins[best] * 1000:.0f}$ mm"
        + r"$\quad$(dashed: reference line at distance $\varepsilon_j$ from the vehicle or segment; green $\geq 0$, red $<0$)",
        y=0.995,
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig_lane_sat_directions")
    return values, mins, best


# ---------------------------------------------------------------------------
# Figure 5: projection lemma and the min-max-min structure
# ---------------------------------------------------------------------------
def fig_structure(dir_mins, best):
    fig = plt.figure(figsize=(7.2, 3.3))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.25], wspace=0.05)
    ax = fig.add_subplot(grid[0, 0])

    # (a) Projection gap <= Euclidean distance
    r, psi = np.array([0.17, 0.12]), 0.35
    cs = corners(r, psi)
    sa, sb = np.array([0.40, 0.27]), np.array([0.52, 0.01])
    draw_car(ax, r, psi)
    ax.plot(*np.stack([sa, sb]).T, color=COLORS["edge"], lw=2.4)
    ax.text(sb[0] + 0.01, sb[1], "$F_j$", fontsize=10)
    ax.text(
        *r, r"$\mathcal{R}$", color=COLORS["car"], fontsize=11, ha="center", va="center"
    )
    d = unit([1.0, -0.06])
    origin = np.array([0.0, -0.1])

    def at(s):
        return origin + s * d

    ax.plot(*np.stack([at(-0.02), at(0.64)]).T, color=COLORS["muted"], lw=0.9)
    arrow(ax, at(0.56), 0.07 * d, COLORS["muted"])
    ax.text(
        *(at(0.63) + np.array([0, -0.03])), "$d$", fontsize=10, color=COLORS["muted"]
    )
    proj_car = (cs - origin) @ d
    proj_seg = (np.stack([sa, sb]) - origin) @ d
    for lo, hi, color in (
        (proj_car.min(), proj_car.max(), COLORS["car"]),
        (proj_seg.min(), proj_seg.max(), COLORS["edge"]),
    ):
        ax.plot(*np.stack([at(lo), at(hi)]).T, color=color, lw=5, solid_capstyle="butt")
    c_ext = cs[np.argmax(proj_car)]
    s_ext = [sa, sb][int(np.argmin(proj_seg))]
    for p, s in ((c_ext, proj_car.max()), (s_ext, proj_seg.min())):
        ax.plot(*np.stack([p, at(s)]).T, color=COLORS["muted"], lw=0.7, ls=":")
    g0, g1 = at(proj_car.max()), at(proj_seg.min())
    off = np.array([0, -0.035])
    ax.annotate(
        "",
        xy=g1 + off,
        xytext=g0 + off,
        arrowprops=dict(arrowstyle="<->", color=COLORS["car"], lw=1),
    )
    ax.text(
        *(0.5 * (g0 + g1) + off + np.array([0, -0.04])),
        r"$\mathrm{min}_i\, g_{j,s,i}+\varepsilon_j$",
        ha="center",
        color=COLORS["car"],
        fontsize=9,
    )
    best_d, best_pair = None, None
    for c in cs:
        ab = sb - sa
        tt = np.clip(np.dot(c - sa, ab) / np.dot(ab, ab), 0, 1)
        foot = sa + tt * ab
        dist = np.linalg.norm(c - foot)
        if best_d is None or dist < best_d:
            best_d, best_pair = dist, (c, foot)
    ax.plot(*np.stack(best_pair).T, color=COLORS["edge"], lw=1.2, ls="--")
    ax.text(
        *(0.5 * (best_pair[0] + best_pair[1]) + np.array([-0.01, 0.025])),
        r"$\mathrm{dist}(\mathcal{R},F_j)$",
        ha="right",
        fontsize=9,
    )
    ax.text(0.0, 0.33, r"$d^\top(w-z)\leq\|w-z\|$ for unit $d$", fontsize=9)
    ax.set_xlim(-0.03, 0.66)
    ax.set_ylim(-0.24, 0.36)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Any projection gap is a lower bound", loc="left")

    # (b) Tree of the min-max-min composition
    ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) Composition of the barrier", loc="left")

    def box(
        x, y, text, color=COLORS["edge"], fc="white", w=0.2, h=0.11, fs=8.5, bold=False
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                fc=fc,
                ec=color,
                lw=1.6 if bold else 0.9,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color)

    def link(p, q, color=COLORS["muted"]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.8, zorder=0)

    top = (0.615, 0.9)
    box(*top, r"$h=\mathrm{min}_{\mathrm{sides},\,j}\ h_j$", w=0.3)
    mids = [(0.36, 0.66), (0.615, 0.66), (0.87, 0.66)]
    for (x, y), label in zip(
        mids, [r"$h_{j-1}$", r"$h_j=\mathrm{max}_s(\cdot)$", r"$h_{j+1}$"]
    ):
        link((top[0], top[1] - 0.055), (x, y + 0.055))
        box(x, y, label, w=0.15 if x != 0.615 else 0.2)
    keys = ["normal", "front", "rear", "side"]
    names = [
        r"$n_j$",
        r"$e_{\mathrm{lon}}$",
        r"$-e_{\mathrm{lon}}$",
        r"$\sigma e_{\mathrm{lat}}$",
    ]
    xs = [0.36, 0.53, 0.70, 0.87]
    for x, key, name in zip(xs, keys, names):
        link((0.615, 0.605), (x, 0.46))
        is_best = key == best
        color = (
            COLORS["car"]
            if is_best
            else (COLORS["good"] if dir_mins[key] >= 0 else COLORS["bad"])
        )
        box(
            x,
            0.4,
            f"{name}\n{dir_mins[key] * 1000:.0f} mm",
            color=color,
            w=0.15,
            h=0.12,
            bold=is_best,
        )
    leaves = ["$c_1..c_4$", "$q_j,q_{j+1}$", "$q_j,q_{j+1}$", "$q_j,q_{j+1}$"]
    for x, leaf in zip(xs, leaves):
        link((x, 0.34), (x, 0.205))
        box(
            x, 0.15, r"$\mathrm{min}_i\, g_{j,s,i}$" + "\n" + leaf, w=0.15, h=0.12, fs=8
        )
    for y, text in (
        (0.78, "min: every segment\nmust be cleared"),
        (0.53, "max: one separating\ndirection suffices"),
        (0.275, "min: every check\npoint must clear"),
        (0.08, "atoms: smooth in $\\pi$"),
    ):
        ax.text(0.0, y, text, fontsize=7.8, color=COLORS["muted"], va="center")
    save(fig, "fig_lane_sat_structure")


# ---------------------------------------------------------------------------
# Figure 6: safe set of positions for a fixed heading
# ---------------------------------------------------------------------------
def barrier_and_distance_grid(points, sigma, psi, eps, xs, ys):
    """Vectorized barrier h and exact distance over a grid of positions."""
    X, Y = np.meshgrid(xs, ys)
    R = np.stack([X, Y], axis=-1)
    rot = rotation(psi)
    cs = R[..., None, :] + (CORNERS_LOCAL @ rot.T)[None, None]
    el, et = e_lon(psi), e_lat(psi)
    h = np.full(X.shape, np.inf)
    dist = np.full(X.shape, np.inf)
    for a, b in zip(points[:-1], points[1:]):
        n = lane_normal(a, b, sigma)
        ends = np.stack([a, b])
        rel = ends[None, None] - R[..., None, :]
        lon = rel @ el
        lat = sigma * (rel @ et)
        normal = (cs @ n - n @ a - eps).min(-1)
        front = (lon - HALF_LENGTH - eps).min(-1)
        rear = (-lon - HALF_LENGTH - eps).min(-1)
        side = (lat - HALF_WIDTH - eps).min(-1)
        h = np.minimum(h, np.maximum.reduce([normal, front, rear, side]))
        # Exact distance: full SAT decides intersection, then vertex-edge pairs
        gaps = []
        for axis in (n, el, et):
            pc = cs @ axis
            ps = ends @ axis
            gaps.append(ps.min() - pc.max(-1))
            gaps.append(pc.min(-1) - ps.max())
        separated = np.maximum.reduce(gaps) > 0
        ab = b - a
        tt = np.clip(((cs - a) @ ab) / (ab @ ab), 0, 1)
        d_corner = np.linalg.norm(cs - (a + tt[..., None] * ab), axis=-1).min(-1)
        lon_l = -(rel @ el)
        lat_l = -(rel @ et)
        d_end = np.hypot(
            np.maximum(np.abs(lon_l) - HALF_LENGTH, 0),
            np.maximum(np.abs(lat_l) - HALF_WIDTH, 0),
        ).min(-1)
        dist = np.minimum(dist, np.where(separated, np.minimum(d_corner, d_end), 0.0))
    return X, Y, h, dist


def fig_safe_set():
    radius = 0.5
    angles = np.arange(-math.pi / 2 - 1.0, -math.pi / 2 + 1.0001, 0.2)
    points = np.stack(
        [radius * np.cos(angles), radius + radius * np.sin(angles)], axis=1
    )
    sigma, eps = -1, 0.002  # right boundary, lane to its left (inside the bend)
    xs = np.linspace(-0.3, 0.3, 601)
    ys = np.linspace(0.0, 0.2, 301)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, psi_deg, letter in zip(axes, (0, 20), "ab"):
        psi = math.radians(psi_deg)
        X, Y, h, dist = barrier_and_distance_grid(points, sigma, psi, eps, xs, ys)
        ax.contourf(X, Y, h, levels=[0, 10], colors=[COLORS["car_fill"]])
        ax.contour(X, Y, h, levels=[0], colors=[COLORS["car"]], linewidths=1.6)
        ax.contour(
            X,
            Y,
            np.where(h > -0.03, dist - eps, -1.0),
            levels=[0],
            colors=[COLORS["edge"]],
            linewidths=0.9,
            linestyles="--",
        )
        ax.plot(*points.T, color=COLORS["edge"], lw=2)
        ax.plot(*points.T, "o", color=COLORS["edge"], ms=3)
        # Two vehicles on the zero level set
        for x_probe in (-0.12, 0.07):
            col = np.argmin(np.abs(xs - x_probe))
            row = np.nonzero(h[:, col] >= 0)[0][0]
            r = np.array([xs[col], ys[row]])
            draw_car(ax, r, psi, fill=False, lw=0.8)
            ax.plot(*r, "o", color=COLORS["car"], ms=3)
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.03, 0.2)
        ax.set_aspect("equal")
        ax.set_xlabel("$p_x$ [m]")
        ax.set_ylabel("$p_y$ [m]")
        ax.set_title(rf"({letter}) heading $\psi={psi_deg}^\circ$", loc="left")
    handles = [
        matplotlib.patches.Patch(
            color=COLORS["car_fill"], label=r"$\{r: h(r,\psi)\geq 0\}$"
        ),
        matplotlib.lines.Line2D([], [], color=COLORS["car"], lw=1.6, label="$h=0$"),
        matplotlib.lines.Line2D(
            [],
            [],
            color=COLORS["edge"],
            lw=0.9,
            ls="--",
            label=r"$\mathrm{dist}(\mathcal{R},\mathrm{polyline})=\varepsilon$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout()
    save(fig, "fig_lane_sat_safe_set")


# ---------------------------------------------------------------------------
# Figure 7: exact two-step prediction
# ---------------------------------------------------------------------------
def fig_prediction():
    state = np.array([0.0, 0.0, 0.0, 0.6, 0.25])
    u_lo = np.array([-2.5, -1.75])
    u_hi = np.array([2.5, min(1.75, (math.radians(24) - state[4]) / DT)])
    x1 = euler_step(state, np.zeros(2))  # pose of x1 does not depend on u
    fig = plt.figure(figsize=(7.2, 4.3))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 0.55], width_ratios=[1.5, 1.0], hspace=0.35
    )
    ax = fig.add_subplot(grid[0, 0])
    draw_car(ax, state[:2], state[2], fill=False, color=COLORS["muted"], lw=1.0)
    draw_car(ax, x1[:2], x1[2], fill=True, color=COLORS["edge"], lw=1.3)
    ax.text(
        state[0] - 0.1,
        state[1] - 0.075,
        "$x_k$",
        ha="center",
        color=COLORS["muted"],
        fontsize=10,
    )
    ax.text(
        x1[0] - 0.05,
        x1[1] + 0.075,
        r"$y_k=\Phi_1(x_k)$",
        ha="center",
        color=COLORS["edge"],
        fontsize=10,
    )
    cloud = np.array(
        [
            euler_step(euler_step(state, np.array([acc, omega])), np.zeros(2))[:2]
            for acc in np.linspace(u_lo[0], u_hi[0], 9)
            for omega in np.linspace(u_lo[1], u_hi[1], 9)
        ]
    )
    ax.scatter(*cloud.T, s=2, color=COLORS["car"], zorder=6)
    x2_nom = euler_step(euler_step(state, np.zeros(2)), np.zeros(2))
    draw_car(
        ax, x2_nom[:2], x2_nom[2], fill=False, color=COLORS["car"], lw=1.4, ls="--"
    )
    ax.text(
        x2_nom[0] + 0.03,
        -0.125,
        r"$z_k(u_k)=\Phi_2(x_k,u_k)$ (dashed: $u_k=0$;"
        + "\n"
        + r"dots: centers for all $u_k\in\mathcal{U}$)",
        color=COLORS["car"],
        ha="center",
        fontsize=10,
    )
    ax.plot(
        [state[0], x1[0], x2_nom[0]],
        [state[1], x1[1], x2_nom[1]],
        "o-",
        color=COLORS["muted"],
        ms=3,
        lw=0.8,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.14, 0.34)
    ax.set_ylim(-0.14, 0.12)
    ax.axis("off")
    ax.set_title(
        r"(a) Poses at $k$, $k+1$, $k+2$ ($v=0.6$ m/s, $\delta=0.25$ rad)", loc="left"
    )

    ax = fig.add_subplot(grid[0, 1])
    a_grid = np.linspace(u_lo[0], u_hi[0], 11)
    w_grid = np.linspace(u_lo[1], u_hi[1], 11)
    pts, col = [], []
    for acc in a_grid:
        for omega in w_grid:
            x2 = euler_step(euler_step(state, np.array([acc, omega])), np.zeros(2))
            pts.append(x2[:2])
            col.append(omega)
    pts = (np.array(pts) - x2_nom[:2]) * 1000
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=col, cmap="coolwarm", s=8)
    ax.plot(0, 0, "k+", ms=8)
    ax.set_xlabel("$\\Delta p_x$ of $z_k$ [mm]")
    ax.set_ylabel("$\\Delta p_y$ of $z_k$ [mm]")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(r"steering rate $\omega$ [rad/s]")
    ax.set_title(r"(b) $\{z_k(u): u\in\mathcal{U}\}$, relative to $u=0$", loc="left")
    ax.text(
        0.03,
        0.92,
        r"$\pm\Delta t^2 a_{\max}=\pm25$ mm",
        transform=ax.transAxes,
        fontsize=8,
    )

    ax = fig.add_subplot(grid[1, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        r"(c) Information flow of one explicit Euler step per tick", loc="left"
    )
    xs = [0.08, 0.45, 0.82]
    labels = [
        "step $k$\n$x_k$ measured",
        "step $k+1$\npose $y_k$ fixed by $(v_k,\\delta_k)$",
        "step $k+2$\npose $z_k$ depends on $u_k$",
    ]
    for x, label in zip(xs, labels):
        ax.plot(x, 0.55, "o", color=COLORS["edge"], ms=6)
        ax.text(x, 0.12, label, ha="center", fontsize=8.5)
    ax.annotate(
        "",
        xy=(xs[1] - 0.02, 0.55),
        xytext=(xs[0] + 0.02, 0.55),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"]),
    )
    ax.annotate(
        "",
        xy=(xs[2] - 0.02, 0.55),
        xytext=(xs[1] + 0.02, 0.55),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["car"], lw=1.6),
    )
    ax.text(
        0.265,
        0.68,
        r"$\Phi_1$: no $u_k$",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    ax.text(
        0.635,
        0.68,
        r"$v_{k+1}=v_k+\Delta t\,a_k,\ \delta_{k+1}=\delta_k+\Delta t\,\omega_k$",
        ha="center",
        fontsize=9,
        color=COLORS["car"],
    )
    ax.text(
        0.99,
        0.95,
        r"control weight in $z_k$: $\Delta t^2$ (Euler) vs. $T^2/2=2\Delta t^2$ (Taylor, $T=2\Delta t$)",
        ha="right",
        fontsize=8.5,
        color=COLORS["bad"],
    )
    save(fig, "fig_lane_sat_prediction")


# ---------------------------------------------------------------------------
# Figure 8: discrete-time condition and induction
# ---------------------------------------------------------------------------
def fig_condition():
    g = np.array([30, 26, 22, 19, 16.5, 14, 12.5, 11, 10.5, 9.5, 9.0])
    k = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for i in range(len(g) - 1):
        bound = (1 - LAMBDA) * g[i]
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (k[i + 1] - 0.28, -3),
                0.56,
                bound + 3,
                color=COLORS["bad"],
                alpha=0.13,
                lw=0,
            )
        )
        ax.plot(
            [k[i + 1] - 0.28, k[i + 1] + 0.28],
            [bound, bound],
            color=COLORS["bad"],
            lw=1.2,
        )
    ax.plot(
        k,
        g,
        "o-",
        color=COLORS["car"],
        lw=1.4,
        ms=5,
        label=r"boundary barrier $h^b(y_k)$",
    )
    ax.plot(
        k,
        g[0] * (1 - LAMBDA) ** k,
        color=COLORS["muted"],
        ls="--",
        lw=1,
        label=r"worst case $(1-\lambda)^k h^b(y_0)>0$",
    )
    ax.axhline(0, color=COLORS["edge"], lw=0.8)
    ax.plot(
        [],
        [],
        color=COLORS["bad"],
        lw=1.2,
        label=r"bound $(1-\lambda)\,h^b(y_{k-1})$; shaded: forbidden",
    )
    ax.set_xlabel("step $k$")
    ax.set_ylabel("barrier value [mm]")
    ax.set_ylim(-3, 33)
    ax.set_xticks(k)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.annotate(
        r"$h^b(z_k)\geq(1-\lambda)\,h^b(y_k)$ and $z_k=y_{k+1}$",
        xy=(3, (1 - LAMBDA) * g[2]),
        xytext=(3.6, 4.0),
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.7),
    )
    ax.set_title(
        r"Decay implied by (1) with $\lambda=0.2$: a nonnegative barrier stays nonnegative",
        loc="left",
    )
    save(fig, "fig_lane_sat_condition")


# ---------------------------------------------------------------------------
# Figure 9: certification loop
# ---------------------------------------------------------------------------
def fig_certification():
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, color=COLORS["edge"], fc="white"):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                fc=fc,
                ec=color,
                lw=1.1,
            )
        )
        ax.text(
            x, y, text, ha="center", va="center", fontsize=8.3, color=COLORS["edge"]
        )

    def link(p, q, text="", color=COLORS["muted"], rad=0.0, text_off=(0, 0.04)):
        ax.annotate(
            "",
            xy=q,
            xytext=p,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.1, connectionstyle=f"arc3,rad={rad}"
            ),
        )
        if text:
            ax.text(
                (p[0] + q[0]) / 2 + text_off[0],
                (p[1] + q[1]) / 2 + text_off[1],
                text,
                ha="center",
                fontsize=8,
                color=color,
            )

    box(
        0.1,
        0.62,
        0.18,
        0.42,
        "1. Select directions\nat $u^{\\mathrm{sel}}$; linearize (1)\nat clipped nominal;\nrows for atoms with\nslack $<1$ cm",
    )
    box(0.33, 0.62, 0.16, 0.42, "2. Solve QP\nrows tightened\nby $\\mu=0.2$ mm")
    box(
        0.56,
        0.62,
        0.18,
        0.42,
        "3. Exact check:\nevaluate all atoms\nwith $\\Phi_2(x_k,u^\\star)$",
    )
    box(
        0.86,
        0.8,
        0.2,
        0.26,
        "certified: pass $u^\\star$\nto the backup shield",
        color=COLORS["good"],
    )
    box(
        0.86,
        0.33,
        0.2,
        0.3,
        "4. Repair: add rows\nlinearized at $u^\\star$\nfor violated atoms",
        color=COLORS["bad"],
    )
    link((0.19, 0.62), (0.25, 0.62))
    link((0.41, 0.62), (0.47, 0.62))
    link((0.65, 0.7), (0.76, 0.8), "all hold", COLORS["good"], text_off=(-0.02, 0.04))
    link((0.65, 0.54), (0.76, 0.4), "", COLORS["bad"])
    ax.text(
        0.73,
        0.37,
        r"violation $>\tau$",
        fontsize=8,
        color=COLORS["bad"],
        ha="right",
        va="top",
    )
    link((0.8, 0.18), (0.33, 0.4), "", COLORS["bad"], rad=-0.25)
    ax.text(
        0.5,
        0.27,
        "re-solve (at most 3 times)",
        fontsize=8,
        color=COLORS["bad"],
        ha="center",
    )
    ax.text(
        0.99,
        0.03,
        "still violated after 3 repairs (or lane slack needed): step flagged uncertified;"
        " the backup shield still decides",
        ha="right",
        fontsize=8,
        color=COLORS["muted"],
    )
    save(fig, "fig_lane_sat_certification")


# ---------------------------------------------------------------------------
# Figure 10: resampling allowance
# ---------------------------------------------------------------------------
def fig_resampling():
    theta, l1, l2 = math.radians(55), 0.1, 0.07
    q0, q1 = np.array([-l1, 0.0]), np.array([0.0, 0.0])
    q2 = l2 * np.array([math.cos(theta), math.sin(theta)])

    def gap(f):
        s1 = q0 + f * (q1 - q0)
        s2 = q1 + f * (q2 - q1)
        chord = s2 - s1
        normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
        return abs(np.dot(q1 - s1, normal)), s1, s2

    fs = np.linspace(0.01, 0.99, 300)
    gaps = np.array([gap(f)[0] for f in fs])
    f_star = fs[np.argmax(gaps)]
    bound = 2 * math.sin(theta / 2) * l1 * l2 / (math.sqrt(l1) + math.sqrt(l2)) ** 2

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1])
    )
    ax = axes[0]
    ax.add_patch(Polygon([q0, q1, q2, [-0.1, q2[1]]], color=COLORS["lane"], lw=0))
    ax.plot(*np.stack([q0, q1, q2]).T, color=COLORS["edge"], lw=2)
    for p, label, off in (
        (q0, "$q_{j-1}$", (0, -0.012)),
        (q1, "$q_j$", (0.004, -0.012)),
        (q2, "$q_{j+1}$", (0.004, 0.0)),
    ):
        ax.plot(*p, "o", color=COLORS["edge"], ms=4)
        ax.text(p[0] + off[0], p[1] + off[1], label, fontsize=10)
    g, s1, s2 = gap(f_star)
    ax.plot(*np.stack([s1, s2]).T, color=COLORS["bad"], lw=1.6)
    for p in (s1, s2):
        ax.plot(*p, "s", color=COLORS["bad"], ms=4)
    chord = unit(s2 - s1)
    foot = s1 + np.dot(q1 - s1, chord) * chord
    ax.annotate(
        "",
        xy=foot,
        xytext=q1,
        arrowprops=dict(arrowstyle="<->", color=COLORS["bad"], lw=1),
    )
    ax.annotate(
        r"$\gamma$",
        xy=0.5 * (foot + q1),
        xytext=(0.006, -0.03),
        color=COLORS["bad"],
        fontsize=11,
        arrowprops=dict(arrowstyle="-", color=COLORS["bad"], lw=0.6),
    )
    ax.text(
        *(0.5 * (q0 + q1) + np.array([0, -0.012])), "$L_1$", ha="center", fontsize=10
    )
    ax.text(
        *(0.75 * (q1 + q2) - 0.75 * q1 + np.array([0.006, -0.004])),
        "$L_2$",
        fontsize=10,
    )
    arc = np.linspace(0, theta, 20)
    ax.plot(0.03 * np.cos(arc), 0.03 * np.sin(arc), color=COLORS["muted"], lw=0.8)
    ax.plot([0, 0.04], [0, 0], color=COLORS["muted"], lw=0.6, ls=":")
    ax.text(0.036, 0.008, r"$\Delta\theta$", fontsize=10)
    ax.text(-0.095, 0.045, "lane", color=COLORS["muted"], fontsize=9)

    ax.text(
        -0.1,
        -0.05,
        "map polyline (black); checker chord between samples\nat the same fraction $f$ of each segment (red)",
        fontsize=8,
        color=COLORS["muted"],
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.105, 0.065)
    ax.set_ylim(-0.06, 0.065)
    ax.axis("off")
    ax.set_title("(a) The checker's chord cuts into the lane", loc="left")

    ax = axes[1]
    ax.plot(fs, gaps * 1000, color=COLORS["bad"], lw=1.6, label=r"$\gamma(f)$")
    ax.axhline(
        bound * 1000,
        color=COLORS["edge"],
        ls="--",
        lw=1,
        label=r"$2\sin\frac{\Delta\theta}{2}\,\frac{L_1L_2}{(\sqrt{L_1}+\sqrt{L_2})^2}$",
    )
    ax.set_xlabel("sampling fraction $f$")
    ax.set_ylabel(r"$\gamma$ [mm]")
    ax.set_ylim(0, bound * 1000 * 1.35)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title(
        rf"(b) $\Delta\theta={math.degrees(theta):.0f}^\circ$, $L_1=10$ cm, $L_2=7$ cm",
        loc="left",
    )
    save(fig, "fig_lane_sat_resampling")


# ---------------------------------------------------------------------------
# Figure 11: local window
# ---------------------------------------------------------------------------
def fig_window():
    radius, half_lane = 0.9, 0.12
    s = np.arange(0, 2.21, 0.1)
    ang = s / radius - 0.9
    center = np.stack([radius * np.sin(ang), radius - radius * np.cos(ang)], axis=1)
    normal = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    left = center + half_lane * normal
    right = center - half_lane * normal
    i0 = 9
    window = np.arange(i0 - 5, i0 + 7)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.add_patch(Polygon(np.vstack([left, right[::-1]]), color=COLORS["lane"], lw=0))
    ax.plot(*center.T, color=COLORS["muted"], lw=0.7, ls="--")
    for boundary in (left, right):
        ax.plot(*boundary.T, color=COLORS["edge"], lw=1.2)
        ax.plot(
            *boundary[window[0] : window[-1] + 1].T,
            color=COLORS["amber"],
            lw=3,
            alpha=0.8,
        )
        ax.plot(*boundary.T, "o", ms=3, mfc="white", mec=COLORS["edge"])
        ax.plot(*boundary[window].T, "o", ms=3.5, color=COLORS["amber"])
    psi = ang[i0] + 0.0
    r = center[i0] + 0.03 * (center[i0 + 1] - center[i0]) / 0.1
    draw_car(ax, r, psi)
    ax.plot(*center[i0], "x", color=COLORS["edge"], ms=6)
    ax.annotate(
        "projected route progress",
        xy=center[i0],
        xytext=(center[i0][0] - 0.45, center[i0][1] - 0.22),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="-", lw=0.6, color=COLORS["muted"]),
    )
    travel_end = center[i0 + 2]
    ax.annotate(
        "",
        xy=travel_end,
        xytext=center[i0],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["car"], lw=1.6),
    )
    ax.text(
        *(travel_end + np.array([0.02, -0.17])),
        r"two-step travel $\leq 2\Delta t\,v_{\max}=0.2$ m",
        color=COLORS["car"],
        fontsize=9,
    )
    ax.text(
        *(left[window[0]] + np.array([0.06, 0.17])),
        "5 vertices behind",
        fontsize=8.5,
        color=COLORS["amber"],
        ha="center",
    )
    ax.text(
        *(left[window[-1]] + np.array([-0.06, 0.04])),
        "6 vertices ahead ($\\approx0.6$ m)",
        fontsize=8.5,
        color=COLORS["amber"],
        ha="right",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Local window of boundary segments (highlighted) used for the barrier of one vehicle",
        loc="left",
    )
    save(fig, "fig_lane_sat_window")


# ---------------------------------------------------------------------------
# Figure 12: benchmark results
# ---------------------------------------------------------------------------
def fig_results():
    scenarios = [
        "cpm_mixed\n8 veh., 10 seeds",
        "cpm_entire\n12 veh., 5 seeds",
        "cpm_mixed\n3 veh., 3 seeds",
    ]
    collisions = dict(rect=[40, 10, 11], sat=[0, 0, 0])
    time_ms = dict(rect=[73.6, 129, 15], sat=[27.7, 80, 9])
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, data, ylabel, title in (
        (
            axes[0],
            collisions,
            "lane collisions (checker)",
            "(a) Lane collisions in 150-step episodes",
        ),
        (axes[1], time_ms, "CBF time per step [ms]", "(b) CBF computation time"),
    ):
        b1 = ax.bar(
            x - 0.18,
            data["rect"],
            0.34,
            color=COLORS["muted"],
            label="rectangle slices (softmin)",
        )
        b2 = ax.bar(
            x + 0.18, data["sat"], 0.34, color=COLORS["car"], label="rectangle_sat"
        )
        for bars in (b1, b2):
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{bar.get_height():g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x, scenarios, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_lane_sat_results")


def check_barrier_lower_bound():
    """Sanity check: the figure geometry reproduces Lemma 1."""
    radius = 0.5
    angles = np.arange(-math.pi / 2 - 0.6, -math.pi / 2 + 0.6001, 0.2)
    points = np.stack(
        [radius * np.cos(angles), radius + radius * np.sin(angles)], axis=1
    )
    xs = np.linspace(-0.3, 0.3, 121)
    ys = np.linspace(0.0, 0.2, 41)
    for psi in (0.0, 0.3, -0.4):
        _, _, h, dist = barrier_and_distance_grid(points, -1, psi, 0.0, xs, ys)
        assert np.all(h <= dist + 1e-9), "barrier exceeds the distance"
        assert np.all(h[dist == 0] < 0), "barrier nonnegative on an intersection"


if __name__ == "__main__":
    check_barrier_lower_bound()
    figures = [
        fig_pipeline,
        fig_motivation,
        fig_notation,
        None,
        fig_safe_set,
        fig_prediction,
        fig_condition,
        fig_certification,
        fig_resampling,
        fig_window,
        fig_results,
    ]
    for idx, make in enumerate(figures):
        print(f"[{idx + 1}/{len(figures)}]", end=" ")
        if make is None:
            _, dir_mins, best = fig_directions()
            fig_structure(dir_mins, best)
        else:
            make()
