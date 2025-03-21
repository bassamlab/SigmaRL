# Copyright (c) 2025, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import multiprocessing as mp
import pickle
import time


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "font.family": "serif",
        "text.usetex": True,
    }
)


class HOCBF:
    def __init__(
        self,
        relative_degree,
        num_steps,
        dt,
        is_virtual_control,
        lambda_1,
        lambda_2,
        lambda_3,
        approach,
    ):
        self.relative_degree = relative_degree
        self.num_steps = num_steps
        self.dt = dt
        self.sim_duration = (self.num_steps - 1) * self.dt
        self.is_virtual_control = is_virtual_control
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.approach = approach

        # Initial states
        self.p_x0 = -10.0  # m
        self.p_y0 = 0.0  # m
        self.v_x0 = 10.0  # m/s
        self.v_y0 = 0.0  # m/s
        self.a_x0 = 0.0  # m/s^2
        self.a_y0 = 0.0  # m/s^2
        self.j_x0 = 0.0  # m/s^3
        self.j_y0 = 0.0  # m/s^3
        self.x_obs = 0.0  # m
        self.y_obs = -2.2  # m

        self.v_x_target = self.v_x0  # Target speed in x direction
        self.v_y_target = self.v_y0  # Target speed in y direction
        self.p_y_target = 0.0  # Target position in y direction

        self.v_x_min, self.v_x_max = 0, 20
        self.v_y_min, self.v_y_max = -5, 5
        self.a_x_min, self.a_x_max = -1000, 1000
        self.a_y_min, self.a_y_max = -1000, 1000
        self.j_x_min, self.j_x_max = -10, 10
        self.j_y_min, self.j_y_max = -10, 10

        self.ra = 1.0  # agent radius
        self.ro = 2.0  # obstacle radius
        self.radii_sqr = (self.ra + self.ro) ** 2

        self.u_x_nominal = 5.0  # Nominal control input
        self.u_y_nominal = 0.0  # Nominal control input

        self.check_initial_conditions()

        if self.approach.lower() == "taylor":
            assert (
                lambda_1 > 0 and lambda_1 <= 1
            ), "lambda_1 must be in (0, 1] for Taylor-based approach"

    def check_initial_conditions(self):
        h_0 = (
            (self.p_x0 - self.x_obs) ** 2
            + (self.p_y0 - self.y_obs) ** 2
            - self.radii_sqr
        )
        d_h_0 = (
            2 * (self.p_x0 - self.x_obs) * self.v_x0
            + 2 * (self.p_y0 - self.y_obs) * self.v_y0
        )
        dd_h_0 = 2 * (self.v_x0**2 + (self.p_x0 - self.x_obs) * self.a_x0) + 2 * (
            self.v_y0**2 + (self.p_y0 - self.y_obs) * self.a_y0
        )

        if self.approach.lower() == "hocbf":
            # Make sure the initial condition of psi-0 and psi-1 satisfies the HOCBF
            if self.relative_degree == 1:
                psi_0_init = h_0
                assert (
                    psi_0_init >= 0
                ), f"Initial condition does not satisfy: psi_0 > 0, got {psi_0_init}"
            elif self.relative_degree == 2:
                psi_0_init = h_0
                d_psi_0_init = d_h_0
                psi_1_init = d_psi_0_init + self.lambda_1 * psi_0_init
                assert (
                    psi_0_init >= 0
                ), f"Initial condition does not satisfy: psi_0 > 0, got {psi_0_init}"
                assert (
                    psi_1_init >= 0
                ), f"Initial condition does not satisfy: psi_1 > 0, got {psi_1_init}"
            elif self.relative_degree == 3:
                psi_0_init = h_0
                d_psi_0_init = d_h_0
                psi_1_init = d_psi_0_init + self.lambda_1 * psi_0_init
                d_psi_1_init = dd_h_0 + self.lambda_1 * d_psi_0_init
                psi_2_init = d_psi_1_init + self.lambda_2 * psi_1_init
                assert (
                    psi_0_init >= 0
                ), f"Initial condition does not satisfy: psi_0 > 0, got {psi_0_init}"
                assert (
                    psi_1_init >= 0
                ), f"Initial condition does not satisfy: psi_1 > 0, got {psi_1_init}"
                assert (
                    psi_2_init >= 0
                ), f"Initial condition does not satisfy: psi_2 > 0, got {psi_2_init}"
        elif self.approach.lower() == "taylor":
            # In Taylor-based approach, only one class K function is required
            psi_0_init = h_0
            assert (
                psi_0_init >= 0
            ), f"Initial condition does not satisfy: psi_0_init > 0, got {psi_0_init}"

    def run_simulation(
        self,
    ):
        if self.approach.lower() == "hocbf" or self.relative_degree != 1:
            self.is_virtual_control = False  # Virtual control is only supported if approach is "taylor" and is only implemented for relative degree 1

        # Initial conditions
        p_x_cur = self.p_x0
        p_y_cur = self.p_y0
        v_x_cur = self.v_x0
        v_y_cur = self.v_y0
        a_x_cur = self.a_x0
        a_y_cur = self.a_y0
        j_x_cur = self.j_x0
        j_y_cur = self.j_y0

        # Storage for time, states, and control
        self.time = np.zeros(self.num_steps)
        self.p_x_vals = np.zeros(self.num_steps)
        self.p_y_vals = np.zeros(self.num_steps)
        self.v_x_vals = np.zeros(self.num_steps)
        self.v_y_vals = np.zeros(self.num_steps)
        self.a_x_vals = np.zeros(self.num_steps)
        self.a_y_vals = np.zeros(self.num_steps)
        self.j_x_vals = np.zeros(self.num_steps)
        self.j_y_vals = np.zeros(self.num_steps)
        self.h_vals = np.zeros(self.num_steps)
        self.dh_vals = np.zeros(self.num_steps)
        self.ddh_vals = np.zeros(self.num_steps)
        self.dddh_vals = np.zeros(self.num_steps)
        self.is_cbf_cond_active = np.zeros(self.num_steps, dtype=bool)
        self.qp_solve_duration = np.zeros(self.num_steps)

        for k in range(self.num_steps):
            self.time[k] = k * self.dt
            # Define QP variables
            u_x_var = cp.Variable()
            u_y_var = cp.Variable()

            # Taylor approximation of the CBF
            if self.relative_degree == 1:  # Speed as control input
                if self.is_virtual_control:
                    # Virtual control input is only implemented for case 1
                    # Use virtual acceleration as control input
                    h = (
                        (p_x_cur - self.x_obs) ** 2
                        + (p_y_cur - self.y_obs) ** 2
                        - self.radii_sqr
                    )
                    d_h = (
                        2 * (p_x_cur - self.x_obs) * v_x_cur
                        + 2 * (p_y_cur - self.y_obs) * v_y_cur
                    )
                    dd_h = 2 * (v_x_cur**2 + (p_x_cur - self.x_obs) * u_x_var) + 2 * (
                        v_y_cur**2 + (p_y_cur - self.y_obs) * u_y_var
                    )
                    ddd_h = None
                    # CBF condition
                    if self.approach.lower() == "hocbf":
                        raise ValueError(
                            "Virtual control is not supported for HOCBF approach."
                        )
                    elif self.approach.lower() == "taylor":
                        # Second order Taylor approximation
                        cbf_cond = (
                            self.lambda_1 * h
                            + d_h * self.dt
                            + 1 / 2 * dd_h * self.dt**2
                        )
                    # Predict the next speed
                    v_x_next_predict = (
                        v_x_cur + u_x_var * self.dt
                    )  # Virtual control input is acceleration
                    v_y_next_predict = (
                        v_y_cur + u_y_var * self.dt
                    )  # Virtual control input is acceleration
                    p_y_next_predict = (
                        p_y_cur + v_y_cur * self.dt + 1 / 2 * u_y_var * self.dt**2
                    )
                else:
                    h = (
                        (p_x_cur - self.x_obs) ** 2
                        + (p_y_cur - self.y_obs) ** 2
                        - self.radii_sqr
                    )
                    d_h = (
                        2 * (p_x_cur - self.x_obs) * u_x_var
                        + 2 * (p_y_cur - self.y_obs) * u_y_var
                    )
                    dd_h = None
                    ddd_h = None
                    # CBF condition
                    if self.approach.lower() == "hocbf":
                        cbf_cond = d_h + self.lambda_1 * h
                    elif self.approach.lower() == "taylor":
                        # First order Taylor approximation
                        cbf_cond = self.lambda_1 * h + d_h * self.dt
                    # Predict the next speed
                    v_x_next_predict = u_x_var  # Control input is speed
                    v_y_next_predict = u_y_var  # Control input is speed
                    p_y_next_predict = p_y_cur + u_y_var * self.dt

                constraints = [
                    cbf_cond >= 0,
                    # u_var >= v_min,
                    # u_var <= v_max
                ]

            elif self.relative_degree == 2:  # Acceleration as control input
                h = (
                    (p_x_cur - self.x_obs) ** 2
                    + (p_y_cur - self.y_obs) ** 2
                    - self.radii_sqr
                )
                d_h = (
                    2 * (p_x_cur - self.x_obs) * v_x_cur
                    + 2 * (p_y_cur - self.y_obs) * v_y_cur
                )
                dd_h = 2 * (v_x_cur**2 + (p_x_cur - self.x_obs) * u_x_var) + 2 * (
                    v_y_cur**2 + (p_y_cur - self.y_obs) * u_y_var
                )
                ddd_h = None
                # CBF condition
                if self.approach.lower() == "hocbf":
                    cbf_cond = (
                        dd_h
                        + (self.lambda_1 + self.lambda_2) * d_h
                        + self.lambda_1 * self.lambda_2 * h
                    )
                elif self.approach.lower() == "taylor":
                    # Second order Taylor approximation
                    cbf_cond = (
                        self.lambda_1 * h + d_h * self.dt + 1 / 2 * dd_h * self.dt**2
                    )
                # Predict the next speed
                v_x_next_predict = (
                    v_x_cur + u_x_var * self.dt
                )  # Virtual control input is acceleration
                v_y_next_predict = (
                    v_y_cur + u_y_var * self.dt
                )  # Virtual control input is acceleration
                p_y_next_predict = (
                    p_y_cur + v_y_cur * self.dt + 1 / 2 * u_y_var * self.dt**2
                )

                constraints = [
                    cbf_cond >= 0,
                    # v_next >= v_min,
                    # v_next <= v_max,
                    u_x_var >= self.a_x_min,
                    u_x_var <= self.a_x_max,
                    u_y_var >= self.a_y_min,
                    u_y_var <= self.a_y_max,
                ]

            elif self.relative_degree == 3:  # Jerk as control input
                # CBF and possibly derivatives of the CBF
                h = (
                    (p_x_cur - self.x_obs) ** 2
                    + (p_y_cur - self.y_obs) ** 2
                    - self.radii_sqr
                )
                d_h = (
                    2 * (p_x_cur - self.x_obs) * v_x_cur
                    + 2 * (p_y_cur - self.y_obs) * v_y_cur
                )
                dd_h = 2 * (v_x_cur**2 + (p_x_cur - self.x_obs) * a_x_cur) + 2 * (
                    v_y_cur**2 + (p_y_cur - self.y_obs) * a_y_cur
                )
                ddd_h = 2 * (
                    3 * v_x_cur * a_x_cur + (p_x_cur - self.x_obs) * u_x_var
                ) + 2 * (3 * v_y_cur * a_y_cur + (p_y_cur - self.y_obs) * u_y_var)
                # CBF condition
                if self.approach.lower() == "hocbf":
                    cbf_cond = (
                        ddd_h
                        + (self.lambda_1 + self.lambda_2 + self.lambda_3) * dd_h
                        + (
                            self.lambda_1 * self.lambda_2
                            + self.lambda_1 * self.lambda_3
                            + self.lambda_2 * self.lambda_3
                        )
                        * d_h
                        + self.lambda_1 * self.lambda_2 * self.lambda_3 * h
                    )
                elif self.approach.lower() == "taylor":
                    # Third order Taylor approximation
                    cbf_cond = (
                        self.lambda_1 * h
                        + d_h * self.dt
                        + 1 / 2 * dd_h * self.dt**2
                        + 1 / 6 * ddd_h * self.dt**3
                    )
                # Predict the next speed
                v_x_next_predict = (
                    v_x_cur + a_x_cur * self.dt + 1 / 2 * u_x_var * self.dt**2
                )  # Virtual control input is acceleration
                v_y_next_predict = (
                    v_y_cur + a_y_cur * self.dt + 1 / 2 * u_y_var * self.dt**2
                )  # Virtual control input is acceleration
                p_y_next_predict = (
                    p_y_cur
                    + v_y_cur * self.dt
                    + 1 / 2 * a_y_cur * self.dt**2
                    + 1 / 4 * u_y_var * self.dt**3
                )

                constraints = [
                    cbf_cond >= 0,
                    # v_next >= v_min,
                    # v_next <= v_max,
                    # a_next >= a_min,
                    # a_next <= a_max,
                    # u_x_var >= self.j_min,
                    # u_y_var <= self.j_max,
                ]
            else:
                raise ValueError("Invalid case number")

            if p_y_cur <= 0.0:
                # Encourage moving forawrd
                penalty_v_x = 1
                penalty_v_y = 1
                penalty_p_y = 1000
            else:
                # Encourage staying close to target y position
                penalty_v_x = 1
                penalty_v_y = 1
                penalty_p_y = 1000

            cost = (
                penalty_v_x * cp.square(v_x_next_predict - self.v_x_target)
                + penalty_v_y * cp.square(v_y_next_predict - self.v_y_target)
                + penalty_p_y * cp.square(p_y_next_predict - self.p_y_target)
            )

            # Form and solve QP
            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                t_start = time.time()
                prob.solve(solver=cp.OSQP)
                t_end = time.time()
                # self.qp_solve_duration[k] = prob.solver_stats.solve_time  # This is much shorter than using timer
                self.qp_solve_duration[k] = t_end - t_start

                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    # If solver fails, revert to nominal control
                    print(
                        f"[case {self.relative_degree} own] Warning at step {k}: QP solver status {prob.status}. Using nominal control."
                    )
                    u_x_star = self.u_x_nominal
                    u_y_star = self.u_y_nominal
                    self.is_cbf_cond_active[k] = False
                else:
                    u_x_star = u_x_var.value
                    u_y_star = u_y_var.value
                    self.is_cbf_cond_active[k] = np.isclose(
                        cbf_cond.value, 0.0, atol=1e-6
                    )

            except:
                # If any error occurs, revert to nominal control
                print(
                    f"[case {self.relative_degree} own] Warning at step {k}: QP solve failed. Using nominal control."
                )
                u_x_star = self.u_x_nominal
                u_y_star = self.u_y_nominal
                self.is_cbf_cond_active[k] = False

            # Evaluate the control input
            if self.relative_degree == 1:
                if self.is_virtual_control:
                    dd_h = 2 * (
                        v_x_cur**2 + (p_x_cur - self.x_obs) * u_x_star
                    ) + 2 * (v_y_cur**2 + (p_y_cur - self.y_obs) * u_y_star)
                else:
                    d_h = (
                        2 * (p_x_cur - self.x_obs) * u_x_star
                        + 2 * (p_y_cur - self.y_obs) * u_y_star
                    )
            elif self.relative_degree == 2:
                dd_h = 2 * (v_x_cur**2 + (p_x_cur - self.x_obs) * u_x_star) + 2 * (
                    v_y_cur**2 + (p_y_cur - self.y_obs) * u_y_star
                )
            elif self.relative_degree == 3:
                ddd_h = 2 * (
                    3 * v_x_cur * a_x_cur + (p_x_cur - self.x_obs) * u_x_star
                ) + 2 * (3 * v_y_cur * a_y_cur + (p_y_cur - self.y_obs) * u_y_star)

            # Store data
            self.p_x_vals[k] = p_x_cur
            self.p_y_vals[k] = p_y_cur
            self.v_x_vals[k] = v_x_cur
            self.v_y_vals[k] = v_y_cur
            self.a_x_vals[k] = a_x_cur
            self.a_y_vals[k] = a_y_cur
            self.j_x_vals[k] = j_x_cur
            self.j_y_vals[k] = j_y_cur
            self.h_vals[k] = h
            self.dh_vals[k] = d_h
            self.ddh_vals[k] = dd_h
            self.dddh_vals[k] = ddd_h

            # Apply control input and update states
            if self.relative_degree == 1:
                if self.is_virtual_control:
                    # Compute next states
                    j_x_next = None
                    j_y_next = None
                    a_x_next = u_x_star
                    a_y_next = u_y_star
                    v_x_next = v_x_cur + a_x_next * self.dt
                    v_y_next = v_y_cur + a_y_next * self.dt
                    p_x_next = p_x_cur + (v_x_cur + v_x_next) / 2 * self.dt
                    p_y_next = p_y_cur + (v_y_cur + v_y_next) / 2 * self.dt
                    # Update states
                    p_x_cur = p_x_next
                    p_y_cur = p_y_next
                    v_x_cur = v_x_next
                    v_y_cur = v_y_next
                    a_x_cur = a_x_next
                    a_y_cur = a_y_next
                    j_x_cur = j_x_next
                    j_y_cur = j_y_next
                else:
                    # Compute next states
                    j_x_next = None
                    j_y_next = None
                    a_x_next = None
                    a_y_next = None
                    v_x_next = u_x_star
                    v_y_next = u_y_star
                    p_x_next = p_x_cur + v_x_next * self.dt
                    p_y_next = p_y_cur + v_y_next * self.dt
                    # Update states
                    p_x_cur = p_x_next
                    p_y_cur = p_y_next
                    v_x_cur = v_x_next
                    v_y_cur = v_y_next
                    a_x_cur = a_x_next
                    a_y_cur = a_y_next
                    j_x_cur = j_x_next
                    j_y_cur = j_y_next
            elif self.relative_degree == 2:
                # Compute next states
                j_x_next = None
                j_y_next = None
                a_x_next = u_x_star
                a_y_next = u_y_star
                v_x_next = v_x_cur + a_x_next * self.dt
                v_y_next = v_y_cur + a_y_next * self.dt
                p_x_next = p_x_cur + (v_x_cur + v_x_next) / 2 * self.dt
                p_y_next = p_y_cur + (v_y_cur + v_y_next) / 2 * self.dt
                # Update states
                p_x_cur = p_x_next
                p_y_cur = p_y_next
                v_x_cur = v_x_next
                v_y_cur = v_y_next
                a_x_cur = a_x_next
                a_y_cur = a_y_next
                j_x_cur = j_x_next
                j_y_cur = j_y_next
            elif self.relative_degree == 3:
                # Compute next states
                j_x_next = u_x_star
                j_y_next = u_y_star
                a_x_next = a_x_cur + self.dt * u_x_star
                a_y_next = a_y_cur + self.dt * u_y_star
                v_x_next = v_x_cur + (a_x_cur + a_x_next) / 2 * self.dt
                v_y_next = v_y_cur + (a_y_cur + a_y_next) / 2 * self.dt
                p_x_next = p_x_cur + (v_x_cur + v_x_next) / 2 * self.dt
                p_y_next = p_y_cur + (v_y_cur + v_y_next) / 2 * self.dt
                # Update states
                p_x_cur = p_x_next
                p_y_cur = p_y_next
                v_x_cur = v_x_next
                v_y_cur = v_y_next
                a_x_cur = a_x_next
                a_y_cur = a_y_next
                j_x_cur = j_x_next
                j_y_cur = j_y_next
            else:
                raise ValueError(f"Invalid case number: {self.relative_degree}")

    def plot_footprint(
        self, footprint_interval=10, is_visu_moving_direction=True, label=""
    ):
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        # Visualize obstacle as a circle: center at (self.x_obs, self.y_obs), radius ro
        circle = plt.Circle(
            (self.x_obs, self.y_obs),
            self.ro,
            color="red",
            alpha=0.5,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(circle)
        ax.text(
            self.x_obs,
            self.y_obs,
            "Obstacle",
            ha="center",
            va="center",
            fontsize=16,
            color="black",
            weight="bold",
        )
        plt.plot(
            self.p_x_vals,
            self.p_y_vals,
            linestyle="-",
            color="green",
            linewidth=3,
            label=label,
        )
        # Select a subset of time steps for footprint visualization
        footprint_interval = 10
        for i in range(0, len(self.p_x_vals), footprint_interval):
            footprint = plt.Circle(
                (self.p_x_vals[i], self.p_y_vals[i]), self.ra, color="green", alpha=0.2
            )
            ax.add_patch(footprint)

        if is_visu_moving_direction:
            # Indicate movement direction with an arrow
            arrow_x, arrow_y = self.p_x_vals[20], self.p_y_vals[20]  # Midpoint
            ax.annotate(
                "",
                xy=(arrow_x + 1.0, arrow_y),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=2),
            )

        ax.legend(loc="lower right", frameon=False)

        # Remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove outer box
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_aspect("equal", adjustable="datalim")
        # ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fig_name = f"fig_example_class_k_hocbf"
        plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
        print(f"Figure saved to {fig_name}.jpeg")
        plt.show()

        return ax

    def plot_h(self, is_visu_lb=True):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        if self.relative_degree == 1:
            label_h = rf"$h(t) (r={{{self.relative_degree}}}, \lambda_1 = {{{self.lambda_1}}})$"
            label_h_lb = rf"$h_{{lb}}(t) (r={{{self.relative_degree}}}, \lambda_1 = {{{self.lambda_1}}})$"
        elif self.relative_degree == 2:
            label_h = rf"$h(t) (r={{{self.relative_degree}}}, \lambda_1 = {{{self.lambda_1}}}, \lambda_2={{{self.lambda_2}}})$"
            label_h_lb = rf"$h_{{lb}}(t) (r={{{self.relative_degree}}}, \lambda_1 = {{{self.lambda_1}}}, \lambda_2={{{self.lambda_2}}})$"
        elif self.relative_degree == 3:
            label_h = rf"$h(t) (r={{{self.relative_degree}}}, \lambda_1 = {{{self.lambda_1}}}, \lambda_2={{{self.lambda_2}}}, \lambda_3={{{self.lambda_3}}})$"
            label_h_lb = rf"$h_{{lb}}(t) (r={{{self.relative_degree}}}, \lambda_1 = {{{self.lambda_1}}}, \lambda_2={{{self.lambda_2}}}, \lambda_3={{{self.lambda_3}}})$"

        plt.plot(
            self.time,
            self.h_vals,
            linestyle="-",
            color=colors[self.relative_degree - 1],
            linewidth=3,
            label=label_h,
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        # Plot lower bound
        if is_visu_lb:
            t_idx_where = np.where(self.v_y_vals != self.v_y_target)[0]
            if len(t_idx_where) >= 1:
                t0_idx = max(t_idx_where[0] - 1, 0)
            else:
                t0_idx = 0  # Use 0 s as default time
                print(
                    "[INFO]: Use the initial point as the starting point for plotting lower bound."
                )

            t_lb, h_lb = self.compute_lower_bound(t0_idx)

            plt.plot(
                t_lb,
                h_lb,
                linestyle=":",
                color=colors[self.relative_degree - 1],
                linewidth=3,
                label=label_h_lb,
                marker="o",
                markersize=4,
                alpha=0.5,
            )
            # plt.axvline(x=t_lb[0], color="black", linestyle="--", linewidth=1)

        plt.legend(loc="lower right", frameon=False)
        plt.xlabel(r"Time (s)")
        plt.ylabel(r"CBF $h$($t$) (m$^2$)")
        plt.tight_layout()
        fig_name = f"fig_h"
        plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
        print(f"Figure saved to {fig_name}.jpeg")
        plt.show()

        return fig, ax

    def compute_lower_bound(self, t0_idx=0):
        t0 = self.time[t0_idx]
        if self.approach.lower() == "hocbf":
            t_lb = np.linspace(t0, self.sim_duration, 100)
            if self.relative_degree == 1:
                A = np.array([1])
                b = np.array([self.h_vals[t0_idx]])
                C1 = b / A
                h_lb = np.exp(-(t_lb - t0) * self.lambda_1) * C1
                # print(f"A={A}, b={b}, C1={C1}, lambda_1={self.lambda_1}")
            elif self.relative_degree == 2:
                A = np.array(
                    [
                        [1, 1],
                        [-self.lambda_1, -self.lambda_2],
                    ]
                )
                b = np.array([self.h_vals[t0_idx], self.dh_vals[t0_idx]])

                C = np.linalg.inv(A) @ b
                C1, C2 = C[0], C[1]
                h_lb = (
                    np.exp(-(t_lb - t0) * self.lambda_1) * C1
                    + np.exp(-(t_lb - t0) * self.lambda_2) * C2
                )
                # print(
                #     f"A={A}, b={b}, C1={C1}, C2={C2}, lambda_1={self.lambda_1}, lambda_2={self.lambda_2}"
                # )
            elif self.relative_degree == 3:
                A = np.array(
                    [
                        [1, 1, 1],
                        [-self.lambda_1, -self.lambda_2, -self.lambda_3],
                        [self.lambda_1**2, self.lambda_2**2, self.lambda_3**2],
                    ]
                )
                b = np.array(
                    [self.h_vals[t0_idx], self.dh_vals[t0_idx], self.ddh_vals[t0_idx]]
                )
                C = np.linalg.inv(A) @ b
                C1, C2, C3 = C[0], C[1], C[2]
                h_lb = (
                    np.exp(-(t_lb - t0) * self.lambda_1) * C1
                    + np.exp(-(t_lb - t0) * self.lambda_2) * C2
                    + np.exp(-(t_lb - t0) * self.lambda_3) * C3
                )
                # print(
                #     f"A={A}, b={b}, C1={C1}, C2={C2}, C3={C3}, lambda_1={self.lambda_1}, lambda_2={self.lambda_2}, lambda_3={self.lambda_3}"
                # )
        else:
            k_lb = np.arange(0, self.num_steps - t0_idx, 1)
            t_lb = k_lb * self.dt + t0
            C1 = self.h_vals[t0_idx]
            h_lb = (1 - self.lambda_1) ** k_lb * C1

        return t_lb, h_lb


# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
def get_colormap(num_colors):
    colormap = plt.colormaps.get_cmap(
        "viridis"
    )  # viridis, plasma, magma, inferno, cividis
    num_colors_ = max(num_colors, 3)  # Minimum of 3 colors
    colormap = [
        colormap(i / (num_colors_ - 1)) for i in range(num_colors_)
    ]  # Generate colors

    return colormap


def enumerate_parameters(
    DT_list,
    relative_degree_list,
    approach_list,
    lambda_1_list,
    lambda_2_list,
    lambda_3_list,
):
    idx = 0
    for DT in DT_list:
        for relative_degree in relative_degree_list:
            for approach in approach_list:
                for lambda_1 in lambda_1_list:
                    for lambda_2 in lambda_2_list:
                        for lambda_3 in lambda_3_list:
                            yield idx, DT, relative_degree, approach, lambda_1, lambda_2, lambda_3
                            idx += 1


def run_single_experiment(para_tuple):
    """Runs a single experiment with given parameters."""
    (
        idx,
        DT,
        relative_degree,
        approach,
        lambda_1,
        lambda_2,
        lambda_3,
        SIM_DURATION,
        total_experiments,
    ) = para_tuple
    print(
        f"Running experiment {idx+1}/{total_experiments} with parameters: DT={DT}, relative_degree={relative_degree}, approach={approach}, lambda_1={lambda_1}, lambda_2={lambda_2}, lambda_3={lambda_3}"
    )

    NUM_STEPS = int(SIM_DURATION / DT) + 1
    print(f"SIM_DURATION={SIM_DURATION}")

    hocbf = HOCBF(
        relative_degree=relative_degree,
        num_steps=NUM_STEPS,
        dt=DT,
        is_virtual_control=False,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        approach=approach,
    )

    hocbf.run_simulation()

    t_idx_where = np.where(
        (hocbf.v_y_vals != hocbf.v_y_target) | (hocbf.v_x_vals != hocbf.v_x_target)
    )[0]
    t0_idx = max(t_idx_where[0] - 1, 0) if len(t_idx_where) >= 1 else 0
    t_h_lb, h_lb = hocbf.compute_lower_bound(t0_idx=t0_idx)

    max_evasion = (
        max(hocbf.p_y_vals - hocbf.y_obs - hocbf.ro - hocbf.ra)
        if hocbf.p_x_vals[-1] >= hocbf.x_obs
        else None
    )

    avg_x_speed = np.mean(hocbf.v_x_vals)
    avg_y_speed = np.mean(hocbf.v_y_vals)
    avg_speed = np.sqrt(avg_x_speed**2 + avg_y_speed**2)
    x_movement = hocbf.p_x_vals[-1] - hocbf.p_x_vals[0]

    return {
        "idx": idx,
        "t_h_lb": t_h_lb,
        "h_lb": h_lb,
        "time": hocbf.time,
        "h_vals": hocbf.h_vals,
        "p_x_vals": hocbf.p_x_vals,
        "p_y_vals": hocbf.p_y_vals,
        "v_y_vals": hocbf.v_y_vals,
        "max_evasion": max_evasion,
        "avg_x_speed": avg_x_speed,
        "avg_y_speed": avg_y_speed,
        "avg_speed": avg_speed,
        "x_movement": x_movement,
        "qp_solve_duration": hocbf.qp_solve_duration,
    }


def run_experiment_multi_parameters(
    DT_list,
    relative_degree_list,
    approach_list,
    lambda_1_list,
    lambda_2_list,
    lambda_3_list,
    SIM_DURATION=3.0,
    num_workers=None,
    data_name="experiment_data.pkl",
    is_parallel_workers=True,
):
    """Runs all experiments in parallel."""
    total_experiments = (
        len(DT_list)
        * len(relative_degree_list)
        * len(approach_list)
        * len(lambda_1_list)
        * len(lambda_2_list)
        * len(lambda_3_list)
    )

    param_list = list(
        enumerate_parameters(
            DT_list,
            relative_degree_list,
            approach_list,
            lambda_1_list,
            lambda_2_list,
            lambda_3_list,
        )
    )

    if is_parallel_workers:
        # Determine number of CPU cores to use
        num_workers = num_workers or mp.cpu_count()
        num_workers = min(num_workers, len(param_list))

        print(
            f"Running {total_experiments} experiments using {num_workers} parallel workers..."
        )

        param_list = [p + (SIM_DURATION,) + (total_experiments,) for p in param_list]

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(run_single_experiment, param_list)
    else:
        print(f"Running {total_experiments} experiments sequentially...")
        results = []
        for i, params in enumerate(param_list):
            print(params + (SIM_DURATION,))
            results.append(
                run_single_experiment(params + (SIM_DURATION,) + (total_experiments,))
            )

    data = {
        "max_evasion_list": [res["max_evasion"] for res in results],
        "x_movement_list": [res["x_movement"] for res in results],
        "avg_x_speed_list": [res["avg_x_speed"] for res in results],
        "avg_speed_list": [res["avg_speed"] for res in results],
        "p_x_list": [res["p_x_vals"] for res in results],
        "p_y_list": [res["p_y_vals"] for res in results],
        "t_h_lb_list": [res["t_h_lb"] for res in results],
        "h_lb_list": [res["h_lb"] for res in results],
        "time_list": [res["time"] for res in results],
        "h_list": [res["h_vals"] for res in results],
        "qp_solve_duration": [res["qp_solve_duration"] for res in results],
    }

    print(
        f"Average QP solve duration: {np.mean(data['qp_solve_duration']):.6f} seconds"
    )
    with open(data_name, "wb") as f:
        pickle.dump(data, f)

    print("Experiments completed and results saved.")


def plot_heatmap(data_name, fig_name, lambda_1_list, lambda_2_list):
    # Load data from the pickle file
    with open(data_name, "rb") as f:
        loaded_data = pickle.load(f)
    data = loaded_data["avg_x_speed_list"]

    # Convert data to a NumPy array, replacing None values with NaN
    data_array = np.array([np.nan if x is None else x for x in data])

    # Check if lambda_2_list is missing, None, or empty
    if lambda_2_list[0] is None:
        # --- 1D Case: Only lambda_1 is used ---
        fig, ax = plt.subplots(
            figsize=(8, 2.0)
        )  # Wide and short figure for a clean colorbar

        # Create the horizontal colorbar-like heatmap
        img = ax.imshow(
            data_array[np.newaxis, :],  # Add a new dimension to make it a row
            origin="lower",
            aspect="auto",
            extent=[min(lambda_1_list), max(lambda_1_list), 0, 1],
            cmap="viridis",
            interpolation="bilinear",
        )

        # Remove default y-axis since it's not needed
        ax.set_yticks([])
        num_ticks = 5

        # Format x-axis to show lambda_1 values (below the colorbar)
        ax.set_xticks(
            np.linspace(min(lambda_1_list), max(lambda_1_list), num=num_ticks)
        )
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_xlabel(r"$\lambda_1$")

        # Add a second x-axis on top for speed values
        ax2 = ax.twiny()

        ax2.set_xscale("log")  # Use log scale for better spacing
        ax2.set_xlim(ax.get_xlim())  # Ensure both axes share limits

        # Select a few key points for speed values

        selected_indices = np.linspace(0, len(lambda_1_list) - 1, num_ticks, dtype=int)

        # Ensure speed values correspond to selected lambda values
        ax2.set_xticks(np.array(lambda_1_list)[selected_indices])
        ax2.set_xticklabels(
            [f"{data_array[i]:.1f}" for i in selected_indices]
        )  # Match lambda ticks

        ax2.set_xlabel(r"Mean $x$-speed (m/s)", labelpad=8)
    else:
        # --- 2D Case: Both lambda_1 and lambda_2 are used ---
        # Reshape into a 2D matrix
        data_matrix = data_array.reshape((len(lambda_1_list), len(lambda_2_list))).T

        # Mask NaN values for proper visualization
        masked_data = np.ma.masked_invalid(data_matrix)

        fig, ax = plt.subplots(figsize=(8, 3))

        # Heatmap visualization
        img = ax.imshow(
            masked_data,
            origin="lower",
            aspect="auto",
            extent=[
                min(lambda_1_list),
                max(lambda_1_list),
                min(lambda_2_list),
                max(lambda_2_list),
            ],
            cmap="viridis",
            interpolation="bilinear",
        )

        num_ticks = 5
        # Add colorbar
        cbar = plt.colorbar(img)
        cbar.set_label(r"Mean $x$-speed (m/s)")
        cbar.set_ticks(
            np.linspace(np.nanmin(data_matrix), np.nanmax(data_matrix), num=num_ticks)
        )
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

        ax.set_xticks(
            np.linspace(min(lambda_1_list), max(lambda_1_list), num=num_ticks)
        )
        ax.set_yticks(
            np.linspace(min(lambda_2_list), max(lambda_2_list), num=num_ticks)
        )

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

        # Set axis labels
        ax.set_xlabel(r"$\lambda_1$")
        ax.set_ylabel(r"$\lambda_2$")

    plt.tight_layout()
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")

    # Show the plot
    plt.show()


def plot_footprint(
    DT_list,
    relative_degree_list,
    approach_list,
    lambda_1_list,
    lambda_2_list,
    lambda_3_list,
    data_name,
    is_visu_moving_direction=True,
    is_visu_legend=False,
    num_curves=10,
    is_example=False,
):
    # Load data from the pickle file
    with open(data_name, "rb") as f:
        loaded_data = pickle.load(f)

    avg_x_speed_list = loaded_data["avg_x_speed_list"]
    p_x_list = loaded_data["p_x_list"]
    p_y_list = loaded_data["p_y_list"]
    t_h_lb_list = loaded_data["t_h_lb_list"]
    h_lb_list = loaded_data["h_lb_list"]
    time_list = loaded_data["time_list"]
    h_list = loaded_data["h_list"]

    num_curves = min(num_curves, len(avg_x_speed_list))

    avg_x_speed_array = np.array(avg_x_speed_list)
    sorted_indices = np.argsort(avg_x_speed_array)
    # Select evenly spaced indices from the sorted list
    selected_indices = np.linspace(0, len(sorted_indices) - 1, num_curves, dtype=int)
    selected_indices = sorted_indices[
        selected_indices
    ]  # Get corresponding original indices

    if is_example:
        colormap = plt.colormaps.get_cmap(
            "viridis"
        )  # viridis, plasma, magma, inferno, cividis
        color_list = [
            colormap(i / (len(h_list) + 1 - 1)) for i in range(len(h_list) + 1)
        ]  # Generate colors
    else:
        colormap = plt.colormaps.get_cmap("viridis")
        norm = colors.Normalize(
            vmin=np.nanmin(avg_x_speed_list), vmax=np.nanmax(avg_x_speed_list)
        )  # Normalize speed values

    fig, ax = plt.subplots(figsize=(7, 3))
    plt.xlabel(r"$x$ (m)")
    plt.ylabel(r"$y$ (m)")
    ax.set_aspect("equal", adjustable="datalim")

    for idx in selected_indices:
        para_tuple = list(
            enumerate_parameters(
                DT_list,
                relative_degree_list,
                approach_list,
                lambda_1_list,
                lambda_2_list,
                lambda_3_list,
            )
        )[idx]
        _, DT, relative_degree, approach, lambda_1, lambda_2, lambda_3 = para_tuple

        if "hocbf" not in locals():
            hocbf = HOCBF(
                relative_degree=relative_degree,
                num_steps=0,
                dt=DT,
                is_virtual_control=False,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                approach=approach,
            )
            circle = plt.Circle(
                (hocbf.x_obs, hocbf.y_obs),
                hocbf.ro,
                facecolor="red",
                alpha=0.3,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(circle)
            ax.text(
                hocbf.x_obs,
                hocbf.y_obs,
                "Obstacle",
                ha="center",
                va="center",
                fontsize=16,
                color="black",
                weight="bold",
            )

        p_x_vals = p_x_list[idx]
        p_y_vals = p_y_list[idx]

        if is_example:
            color = color_list[idx]
        else:
            avg_speed = avg_x_speed_list[idx]
            color = colormap(norm(avg_speed))

        if is_example:
            labels = ["Conservative", "Moderate", "Aggressive"]
            label_fp = labels[idx]
            is_visu_legend = True
        else:
            if approach.lower() == "hocbf":
                if relative_degree == 1:
                    label_fp = rf"HOCBF $(\lambda_1 = {{{lambda_1:.2f}}})$"
                elif relative_degree == 2:
                    label_fp = rf"HOCBF $(\lambda_1 = {{{lambda_1:.2f}}}, \lambda_2={{{lambda_2:.2f}}})$"
                elif relative_degree == 3:
                    label_fp = rf"HOCBF $(\lambda_1 = {{{lambda_1:.2f}}}, \lambda_2={{{lambda_2:.2f}}}, \lambda_3={{{lambda_3:.2f}}})$"
            else:
                label_fp = rf"Our $(\lambda_1 = {{{lambda_1:.2f}}})$"

        plt.plot(
            p_x_vals,
            p_y_vals,
            linestyle="-",
            color=color,
            linewidth=3,
            label=label_fp,
        )
        # Select a subset of time steps for footprint visualization
        footprint_interval = 10
        for i in range(0, len(p_x_vals), footprint_interval):
            footprint = plt.Circle(
                (p_x_vals[i], p_y_vals[i]), hocbf.ra, color=color, alpha=0.2
            )
            ax.add_patch(footprint)

    if is_visu_moving_direction:
        # Indicate movement direction with an arrow
        arrow_x, arrow_y = p_x_vals[0], p_y_vals[0]  # Midpoint
        ax.annotate(
            "",
            xy=(arrow_x + 1.5, arrow_y),
            xytext=(arrow_x, arrow_y),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=2),
        )

    if is_visu_legend:
        ax.legend(loc="lower right", frameon=False)

    if is_example:
        # Remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove outer box
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Remove x and y labels
        ax.set_xlabel("")
        ax.set_ylabel("")

    if is_example:
        fig_name = "fig_example_class_k"
    else:
        fig_name = f"fig_footprint_{approach.lower()}"

    plt.tight_layout()
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")

    plt.show()


def plot_h(
    DT_list,
    relative_degree_list,
    approach_list,
    lambda_1_list,
    lambda_2_list,
    lambda_3_list,
    data_name,
    is_visu_legend=True,
    is_visu_lb=True,
    num_curves=3,
):

    # Load data from the pickle file
    with open(data_name, "rb") as f:
        loaded_data = pickle.load(f)

    avg_x_speed_list = loaded_data["avg_x_speed_list"]
    p_x_list = loaded_data["p_x_list"]
    p_y_list = loaded_data["p_y_list"]
    t_h_lb_list = loaded_data["t_h_lb_list"]
    h_lb_list = loaded_data["h_lb_list"]
    time_list = loaded_data["time_list"]
    h_list = loaded_data["h_list"]

    avg_x_speed_array = np.array(avg_x_speed_list)

    speed_min = np.nanmin(avg_x_speed_array)
    speed_max = np.nanmax(avg_x_speed_array)

    # Compute target speeds uniformly spaced in the speed space
    if num_curves > 2:
        target_speeds = np.linspace(speed_min, speed_max, num_curves)
    else:
        target_speeds = [speed_min, speed_min + (speed_max - speed_min) / 1.1]

    # For each target speed, find the index of the closest actual speed
    selected_indices = []
    used_indices = set()  # Optional: to avoid selecting the same index twice

    for target_speed in target_speeds:
        # Compute the absolute distance between target speed and all speeds
        distances = np.abs(avg_x_speed_array - target_speed)

        # Optionally mask already selected indices to avoid duplicates
        if used_indices:
            distances[list(used_indices)] = np.inf

        # Find the closest index
        closest_idx = np.argmin(distances)
        selected_indices.append(closest_idx)
        used_indices.add(closest_idx)

    colormap = plt.colormaps.get_cmap("viridis")  # Use same colormap as heatmap
    norm = colors.Normalize(
        vmin=np.nanmin(avg_x_speed_list), vmax=np.nanmax(avg_x_speed_list)
    )  # Normalize speed values

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    plt.xlabel(r"Time $t$ (s)")
    plt.ylabel(r"CBF $h, h_{{\mathrm{lb}}}$ (m$^2$)")
    plt.grid(True, linestyle="--", alpha=0.4)

    for idx in selected_indices:
        para_tuple = list(
            enumerate_parameters(
                DT_list,
                relative_degree_list,
                approach_list,
                lambda_1_list,
                lambda_2_list,
                lambda_3_list,
            )
        )[idx]
        _, DT, relative_degree, approach, lambda_1, lambda_2, lambda_3 = para_tuple

        avg_speed = avg_x_speed_list[idx]
        color = colormap(norm(avg_speed))

        if approach.lower() == "hocbf":
            if relative_degree == 1:
                label_h = rf"$h (\lambda_1 = {{{lambda_1:.1f}}})$"
                label_h_lb = rf"$h_{{\mathrm{{lb}}}} (\lambda_1 = {{{lambda_1:.1f}}})$"
            elif relative_degree == 2:
                label_h = rf"$h (\lambda_1 = {{{lambda_1:.1f}}}, \lambda_2={{{lambda_2:.1f}}})$"
                label_h_lb = rf"$h_{{\mathrm{{lb}}}} (\lambda_1 = {{{lambda_1:.1f}}}, \lambda_2={{{lambda_2:.1f}}})$"
            elif relative_degree == 3:
                label_h = rf"$h (\lambda_1 = {{{lambda_1:.1f}}}, \lambda_2={{{lambda_2:.1f}}}, \lambda_3={{{lambda_3:.1f}}})$"
                label_h_lb = rf"$h_{{\mathrm{{lb}}}} (\lambda_1 = {{{lambda_1:.1f}}}, \lambda_2={{{lambda_2:.1f}}}, \lambda_3={{{lambda_3:.1f}}})$"
        else:
            label_h = rf"$h (\lambda_1 = {{{lambda_1:.2f}}})$"
            label_h_lb = rf"$h_{{\mathrm{{lb}}}} (\lambda_1 = {{{lambda_1:.2f}}})$"

        plt.plot(
            time_list[idx],
            h_list[idx],
            linestyle="-",
            color=color,
            linewidth=3,
            label=label_h,
        )

        # Plot lower bound
        if is_visu_lb:
            plt.plot(
                t_h_lb_list[idx],
                h_lb_list[idx],
                linestyle=":",
                color=color,
                linewidth=3,
                label=label_h_lb,
                marker="o",
                markersize=7,
                markevery=2 if approach.lower() == "hocbf" else 4,
                alpha=0.4,
            )
    if is_visu_legend:
        plt.legend(
            loc="best",
            frameon=False,
            ncol=2,
            fontsize=14 if approach.lower() == "hocbf" else 16,
        )

    plt.xlim(time_list[idx][0], time_list[idx][-1])
    plt.ylim(-5, 100)
    plt.xticks(np.arange(0, time_list[idx][-1] + 0.1, 0.5))

    plt.tight_layout()
    fig_name = f"fig_h_{approach.lower()}"
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")

    plt.show()


if __name__ == "__main__":
    print("!!!!!!!!!!!!")
    SIM_DURATION = 2.0
    DT_list = [0.01]
    relative_degree_list = [2]
    lambda_3_list = [None]

    experiment = input(
        f"Select an experiment by pressing either 0 (Example: Three class K functions), 1 (Experiment 1: HOCBF), or 2 (Experiment 2: Taylor): "
    )
    if experiment == "0":
        data_name = "experiment_class_k.pkl"
        approach_list = ["taylor"]
        lambda_1_list = [0.01, 0.03, 0.5]
        lambda_2_list = [None]
        is_example = True
    elif experiment == "1":
        approach_list = ["hocbf"]
        data_name = "experiment_1_hocbf.pkl"
        lambda_1_list = np.logspace(0.322, 1, 10)  # About [2.1, 10]
        lambda_2_list = np.logspace(-0.3, 1, 10)  # About [0.5, 10]
        is_example = False
    elif experiment == "2":
        approach_list = ["taylor"]
        data_name = "experiment_2_taylor.pkl"
        lambda_1_list = np.logspace(-2, -0.3, 20)  # About [0.01, 0.5]
        lambda_2_list = [None]
        is_example = False
    else:
        print("Invalid input. Please press either 0, 1, or 2.")

    is_parallel_workers = (
        input("Do you want to use parallel workers? (y/n): ").lower() == "y"
    )

    run_experiment_multi_parameters(
        DT_list=DT_list,
        relative_degree_list=relative_degree_list,
        approach_list=approach_list,
        lambda_1_list=lambda_1_list,
        lambda_2_list=lambda_2_list,
        lambda_3_list=lambda_3_list,
        SIM_DURATION=SIM_DURATION,
        data_name=data_name,
        is_parallel_workers=is_parallel_workers,  # Set to True to use parallel workers
    )

    plot_heatmap(
        data_name=data_name,
        fig_name=f"fig_heatmap_{approach_list[0].lower()}",
        lambda_1_list=lambda_1_list,
        lambda_2_list=lambda_2_list,
    )
    plot_footprint(
        DT_list,
        relative_degree_list,
        approach_list,
        lambda_1_list,
        lambda_2_list,
        lambda_3_list,
        data_name,
        is_visu_legend=False,
        is_example=is_example,
    )
    plot_h(
        DT_list,
        relative_degree_list,
        approach_list,
        lambda_1_list,
        lambda_2_list,
        lambda_3_list,
        data_name,
        is_visu_legend=True,
        num_curves=2,
    )
