import numpy as np
import cvxpy as cp
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# plt.rcParams.update({'font.size': 11})

plt.rcParams.update(
    {
        "font.size": 14,  # Increase overall font size
        "axes.labelsize": 14,  # Axis labels
        "axes.titlesize": 14,  # Titles (if any)
        "xtick.labelsize": 14,  # X-axis tick labels
        "ytick.labelsize": 14,  # Y-axis tick labels
        "legend.fontsize": 14,  # Legend font size
        "font.family": "serif",  # Use a LaTeX-like serif font
        "text.usetex": True,  # Use LaTeX for text rendering
    }
)

our_1_without_virtual_c = "tab:gray"
our_1_without_virtual_m = "*"
our_1_without_virtual_l = ":"
our_1_without_virtual_lw = 4

our_1_c = "tab:blue"
our_1_m = "1"
our_1_l = ":"
our_1_lw = 4

our_2_c = "tab:green"
our_2_m = "2"
our_2_l = ":"
our_2_lw = 4

our_3_c = "tab:orange"
our_3_m = "3"
our_3_l = ":"
our_3_lw = 4

sota_1_c = "tab:blue"
sota_1_m = "1"
sota_1_l = "--"
sota_1_l_lam5 = ":"
sota_1_lw = 3

sota_2_c = "tab:green"
sota_2_m = "2"
sota_2_l = "--"
sota_2_l_lam5 = ":"
sota_2_lw = 3

sota_3_c = "tab:orange"
sota_3_m = "3"
sota_3_l = "--"
sota_3_l_lam5 = ":"
sota_3_lw = 3

grid_ls = "--"
grid_lw = 0.6
grid_alpha = 0.5

markersize = 8

COLORS = [
    "tab:green",
    "tab:blue",
    "tab:orange",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:olive",
    "tab:cyan",
    "tab:olive",
    "tab:cyan",
    "tab:olive",
    "tab:cyan",
    "tab:olive",
    "tab:cyan",
    "tab:olive",
    "tab:cyan",
]


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
        # self.y_obs = -3.001  # m

        self.v_x_target = self.v_x0  # Target speed in x direction
        self.v_y_target = self.v_y0  # Target speed in y direction
        self.p_y_target = 0.0  # Target position in y direction

        self.v_min, self.v_max = -100, 100
        self.a_min, self.a_max = -100, 100
        self.j_min, self.j_max = -200, 200

        self.ra = 1.0  # agent radius
        self.ro = 2.0  # obstacle radius
        self.radii_sqr = (self.ra + self.ro) ** 2

        self.u_x_nominal = 5.0  # Nominal control input
        self.u_y_nominal = 0.0  # Nominal control input

        # self.cmap = truncate_colormap(cm.Blues_r, 0.0, 0.8)
        # self.c_norm = colors.Normalize(vmin=min(dt_values), vmax=max(dt_values))

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
        """
        Run the simulation for the single-integrator case (speed control).

        System dynamics:
            x_{k+1} = x_k + dt * u_k
        where u_k (speed) is in the range [-5, 5] m/s.

        The state here is just x_k. The y-position is assumed constant.

        Returns:
            time: array of time points
            x_vals: array of x positions over time
            u_vals: array of control (speed) values over time
        """

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

        print(f"[INFO] Relative degree {self.relative_degree}")

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
                    # u_var >= a_min,
                    # u_var <= a_max
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
                penalty_p_y = 1
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
                prob.solve(solver=cp.OSQP)
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    # If solver fails, revert to nominal control
                    print(
                        f"[case {self.relative_degree} own] Warning at step {k}: QP solver status {prob.status}. Using nominal control."
                    )
                    u_x_star = self.u_x_nominal
                    u_y_star = self.u_y_nominal
                else:
                    u_x_star = u_x_var.value
                    u_y_star = u_y_var.value
            except:
                # If any error occurs, revert to nominal control
                print(
                    f"[case {self.relative_degree} own] Warning at step {k}: QP solve failed. Using nominal control."
                )
                u_x_star = self.u_x_nominal
                u_y_star = self.u_y_nominal

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

            # if self.relative_degree == 3:
            #     # print(f"v_x_vals: {self.v_x_vals}")
            #     # print(f"v_y_vals: {self.v_y_vals}")
            #     print(f"Cost: {cost.value}")
            #     print(f"CBF Condition: {cbf_cond.value}")
            #     print("eeee")

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
            fontsize=14,
            color="red",
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
        else:
            k_lb = np.arange(0, self.num_steps - t0_idx, 1)
            t_lb = k_lb * self.dt + t0
            C1 = self.h_vals[t0_idx]
            h_lb = (1 - self.lambda_1) ** k_lb * C1

        return t_lb, h_lb


def truncate_colormap(cmap, min_val=0.2, max_val=1.0, n=256):
    """Create a truncated version of a colormap."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{min_val},{max_val})",
        cmap(np.linspace(min_val, max_val, n)),
    )
    return new_cmap


def conservative_aggressive_class_k():

    p_x_list = []
    p_y_list = []
    v_y_list = []
    t_h_list = []
    h_lb_list = []
    t_h_lb_list = []
    h_list = []
    is_visu_lb = True
    is_visu_moving_direction = True
    lambda_1_list = []
    lambda_2_list = []
    lambda_3_list = []

    DT = 0.01
    lambda_1s = [0.025, 0.5]

    SIM_DURATION = 2

    # approach = "hocbf"  # One of "hocbf" (for the standard HOCBF) and "taylor" (for our Taylor-based approach)
    approach = "taylor"  # One of "hocbf" (for the standard HOCBF) and "taylor" (for our Taylor-based approach)
    relative_degree = 2

    # --------------------------
    # Run simulation to collect data
    # --------------------------
    for idx, lambda_1 in enumerate(lambda_1s):
        NUM_STEPS = int(SIM_DURATION / DT) + 1
        lambda_2 = None
        lambda_3 = None

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

        # Some constants
        v_y_target = hocbf.v_y_target
        v_x_targert = hocbf.v_x_target
        x_obs = hocbf.x_obs
        y_obs = hocbf.y_obs
        ro = hocbf.ro
        ra = hocbf.ra
        sim_duration = hocbf.sim_duration

        hocbf.run_simulation()

        t_idx_where = np.where(
            (hocbf.v_y_vals != v_y_target) | (hocbf.v_x_vals != v_x_targert)
        )[0]

        if len(t_idx_where) >= 1:
            t0_idx = max(t_idx_where[0] - 1, 0)
        else:
            t0_idx = 0  # Use 0 s as default time
            print(
                "[INFO]: Use the initial point as the starting point for plotting lower bound."
            )

        t_h_lb, h_lb = hocbf.compute_lower_bound(t0_idx=t0_idx)

        t_h_lb_list.append(t_h_lb)
        h_lb_list.append(h_lb)

        t_h_list.append(hocbf.time)
        h_list.append(hocbf.h_vals)
        p_x_list.append(hocbf.p_x_vals)
        p_y_list.append(hocbf.p_y_vals)
        v_y_list.append(hocbf.v_y_vals)

        lambda_1_list.append(hocbf.lambda_1)
        lambda_2_list.append(hocbf.lambda_2)
        lambda_3_list.append(hocbf.lambda_3)

    colormap = plt.colormaps.get_cmap(
        "viridis"
    )  # viridis, plasma, magma, inferno, cividis
    colormap = [
        colormap(i / (len(h_list) + 1 - 1)) for i in range(len(h_list) + 1)
    ]  # Generate colors

    # ----------------------------
    # Plot footprint
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    # plt.xlabel(r"$x$ (m)")
    # plt.ylabel(r"$y$ (m)")
    ax.set_aspect("equal", adjustable="datalim")

    circle = plt.Circle(
        (x_obs, y_obs), ro, color="red", alpha=0.5, edgecolor="black", linewidth=2
    )
    ax.add_patch(circle)
    ax.text(
        x_obs,
        y_obs,
        "Obstacle",
        ha="center",
        va="center",
        fontsize=14,
        color="red",
        weight="bold",
    )

    for idx, _ in enumerate(lambda_1s):
        lambda_1 = lambda_1_list[idx]
        lambda_2 = lambda_2_list[idx]
        lambda_3 = lambda_3_list[idx]

        p_x_vals = p_x_list[idx]
        p_y_vals = p_y_list[idx]

        if idx == 0:
            label_fp = r"Conservative class $\mathcal{K}$ function"
        else:
            label_fp = r"Aggressive class $\mathcal{K}$ function"

        plt.plot(
            p_x_vals,
            p_y_vals,
            linestyle="-",
            color=colormap[idx],
            linewidth=3,
            label=label_fp,
        )
        # Select a subset of time steps for footprint visualization
        footprint_interval = 10
        for i in range(0, len(p_x_vals), footprint_interval):
            footprint = plt.Circle(
                (p_x_vals[i], p_y_vals[i]), ra, color=colormap[idx], alpha=0.2
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

    ax.legend(loc="lower right", frameon=False)

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove outer box
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_name = f"fig_example_class_k"
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")

    plt.show()

    # -------------------------------
    # Plot h
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    plt.xlabel(r"Time $t$ (s)")
    plt.ylabel(r"CBF $h, h_{{\mathrm{lb}}}$ (m$^2$)")
    plt.grid(True, linestyle="--", alpha=0.4)
    for idx, _ in enumerate(lambda_1s):
        lambda_1 = lambda_1_list[idx]
        lambda_2 = lambda_2_list[idx]
        lambda_3 = lambda_3_list[idx]

        h_vals = h_list[idx]
        t_h = t_h_list[idx]
        v_y_vals = v_y_list[idx]
        t_h_lb = t_h_lb_list[idx]
        h_lb = h_lb_list[idx]

        if idx == 0:
            label_h = r"Conservative class $\mathcal{K}$ function"
        else:
            label_h = r"Aggressive class $\mathcal{K}$ function"
        label_h_lb = r"Corresponding lower bound $h_{{\mathrm{lb}}}$"

        plt.plot(
            t_h,
            h_vals,
            linestyle="-",
            color=colormap[idx],
            linewidth=3,
            label=label_h,
        )
        # plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        # Plot lower bound
        if is_visu_lb:
            t_h_lb = t_h_lb_list[idx]
            h_lb = h_lb_list[idx]
            plt.plot(
                t_h_lb,
                h_lb,
                linestyle=":",
                color=colormap[idx],
                linewidth=3,
                label=label_h_lb,
                marker="o",
                markersize=6,
                markevery=2,
                alpha=0.4,
            )

    plt.legend(loc="best", frameon=False)
    plt.xlim(0, sim_duration)
    plt.ylim(-0.5, 80)
    plt.xticks(np.arange(0, sim_duration + 0.1, 0.5))
    plt.tight_layout()
    fig_name = f"fig_h_{hocbf.approach}"
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")
    plt.show()


def run_experiment_1():

    p_x_list = []
    p_y_list = []
    v_y_list = []
    t_h_list = []
    h_lb_list = []
    t_h_lb_list = []
    h_list = []
    is_visu_lb = True
    is_visu_moving_direction = True
    lambda_1_list = []
    lambda_2_list = []
    lambda_3_list = []

    DT = 0.01

    SIM_DURATION = 2.5
    approaches = ["hocbf", "taylor"]

    relative_degree = 2

    # --------------------------
    # Run simulation to collect data
    # --------------------------
    for idx, approach in enumerate(approaches):
        print(approach)
        NUM_STEPS = int(SIM_DURATION / DT) + 1
        if approach.lower() == "hocbf":
            if relative_degree == 1:
                lambda_1 = 3.0
                lambda_2 = None
                lambda_3 = None
            elif relative_degree == 2:
                lambda_1 = 3.0
                lambda_2 = 5.0
                lambda_3 = None
            elif relative_degree == 3:
                lambda_1 = 5.0
                lambda_2 = 6.0
                lambda_3 = 8.0
        else:
            lambda_1 = 0.1
            lambda_2 = None
            lambda_3 = None

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

        # Some constants
        v_y_target = hocbf.v_y_target
        v_x_targert = hocbf.v_x_target
        x_obs = hocbf.x_obs
        y_obs = hocbf.y_obs
        ro = hocbf.ro
        ra = hocbf.ra
        sim_duration = hocbf.sim_duration

        hocbf.run_simulation()

        t_idx_where = np.where(
            (hocbf.v_y_vals != v_y_target) | (hocbf.v_x_vals != v_x_targert)
        )[0]

        if len(t_idx_where) >= 1:
            t0_idx = max(t_idx_where[0] - 1, 0)
        else:
            t0_idx = 0  # Use 0 s as default time
            print(
                "[INFO]: Use the initial point as the starting point for plotting lower bound."
            )

        t_h_lb, h_lb = hocbf.compute_lower_bound(t0_idx=t0_idx)

        t_h_lb_list.append(t_h_lb)
        h_lb_list.append(h_lb)

        t_h_list.append(hocbf.time)
        h_list.append(hocbf.h_vals)
        p_x_list.append(hocbf.p_x_vals)
        p_y_list.append(hocbf.p_y_vals)
        v_y_list.append(hocbf.v_y_vals)

        lambda_1_list.append(hocbf.lambda_1)
        lambda_2_list.append(hocbf.lambda_2)
        lambda_3_list.append(hocbf.lambda_3)

    colormap = plt.colormaps.get_cmap(
        "viridis"
    )  # viridis, plasma, magma, inferno, cividis
    colormap = [
        colormap(i / (len(h_list) + 1 - 1)) for i in range(len(h_list) + 1)
    ]  # Generate colors

    # ----------------------------
    # Plot footprint
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    plt.xlabel(r"$x$ (m)")
    plt.ylabel(r"$y$ (m)")
    ax.set_aspect("equal", adjustable="datalim")

    circle = plt.Circle(
        (x_obs, y_obs), ro, color="red", alpha=0.4, edgecolor="black", linewidth=2
    )
    ax.add_patch(circle)
    ax.text(
        x_obs,
        y_obs,
        "Obstacle",
        ha="center",
        va="center",
        fontsize=14,
        color="red",
        weight="bold",
    )

    for idx, approach in enumerate(approaches):
        lambda_1 = lambda_1_list[idx]
        lambda_2 = lambda_2_list[idx]
        lambda_3 = lambda_3_list[idx]

        p_x_vals = p_x_list[idx]
        p_y_vals = p_y_list[idx]

        if approach.lower() == "hocbf":
            if relative_degree == 1:
                label_fp = rf"HOCBF $(\lambda_1 = {{{lambda_1}}})$"
            elif relative_degree == 2:
                label_fp = (
                    rf"HOCBF $(\lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}})$"
                )
            elif relative_degree == 3:
                label_fp = rf"HOCBF $(\lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}}, \lambda_3={{{lambda_3}}})$"
        else:
            label_fp = rf"Our $(\lambda_1 = {{{lambda_1}}})$"

        plt.plot(
            p_x_vals,
            p_y_vals,
            linestyle="-",
            color=colormap[idx],
            linewidth=3,
            label=label_fp,
        )
        # Select a subset of time steps for footprint visualization
        footprint_interval = 10
        for i in range(0, len(p_x_vals), footprint_interval):
            footprint = plt.Circle(
                (p_x_vals[i], p_y_vals[i]), ra, color=colormap[idx], alpha=0.2
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

    ax.legend(loc="lower right", frameon=False)

    # Remove x and y ticks
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Remove outer box
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)

    # ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_name = f"fig_footprints_two_approaches"
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")

    plt.show()

    # -------------------------------
    # Plot h
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    plt.xlabel(r"Time $t$ (s)")
    plt.ylabel(r"CBF $h, h_{{\mathrm{lb}}}$ (m$^2$)")
    plt.grid(True, linestyle="--", alpha=0.4)
    for idx, approach in enumerate(approaches):
        lambda_1 = lambda_1_list[idx]
        lambda_2 = lambda_2_list[idx]
        lambda_3 = lambda_3_list[idx]

        h_vals = h_list[idx]
        t_h = t_h_list[idx]
        v_y_vals = v_y_list[idx]
        t_h_lb = t_h_lb_list[idx]
        h_lb = h_lb_list[idx]

        if approach.lower() == "hocbf":
            if relative_degree == 1:
                label_h = rf"HOCBF $(\lambda_1 = {{{lambda_1}}})$"
            elif relative_degree == 2:
                label_h = (
                    rf"HOCBF $(\lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}})$"
                )
            elif relative_degree == 3:
                label_h = rf"HOCBF $(\lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}}, \lambda_3={{{lambda_3}}})$"
        else:
            label_h = rf"Our $(\lambda_1 = {{{lambda_1}}})$"
        label_h_lb = r"Corresponding lower bound $h_{{\mathrm{lb}}}$"

        plt.plot(
            t_h,
            h_vals,
            linestyle="-",
            color=colormap[idx],
            linewidth=3,
            label=label_h,
        )
        # plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        # Plot lower bound
        if is_visu_lb:
            t_h_lb = t_h_lb_list[idx]
            h_lb = h_lb_list[idx]
            plt.plot(
                t_h_lb,
                h_lb,
                linestyle=":",
                color=colormap[idx],
                linewidth=3,
                label=label_h_lb,
                marker="o",
                markersize=6,
                markevery=2,
                alpha=0.4,
            )

    plt.legend(loc="best", frameon=False)
    plt.xlim(0, sim_duration)
    plt.ylim(-5, 100)
    plt.xticks(np.arange(0, sim_duration + 0.1, 0.5))
    plt.tight_layout()
    fig_name = f"fig_h_two_approaches"
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")
    plt.show()


def run_experiment_2():

    p_x_list = []
    p_y_list = []
    v_y_list = []
    t_h_list = []
    h_lb_list = []
    t_h_lb_list = []
    h_list = []
    is_visu_lb = True
    is_visu_moving_direction = True
    lambda_1_list = []
    lambda_2_list = []
    lambda_3_list = []

    DTs = np.arange(0.01, 0.08, 0.01)

    SIM_DURATION = 4

    # approach = "hocbf"  # One of "hocbf" (for the standard HOCBF) and "taylor" (for our Taylor-based approach)
    approach = "taylor"  # One of "hocbf" (for the standard HOCBF) and "taylor" (for our Taylor-based approach)
    relative_degree = 2

    # --------------------------
    # Run simulation to collect data
    # --------------------------
    for idx, DT in enumerate(DTs):
        NUM_STEPS = int(SIM_DURATION / DT) + 1
        if approach.lower() == "hocbf":
            if relative_degree == 1:
                lambda_1 = 3.0
                lambda_2 = None
                lambda_3 = None
            elif relative_degree == 2:
                lambda_1 = 3.0
                lambda_2 = 5.0
                lambda_3 = None
            elif relative_degree == 3:
                lambda_1 = 5.0
                lambda_2 = 6.0
                lambda_3 = 8.0
        else:
            lambda_1 = 0.15
            lambda_2 = None
            lambda_3 = None

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

        # Some constants
        v_y_target = hocbf.v_y_target
        v_x_targert = hocbf.v_x_target
        x_obs = hocbf.x_obs
        y_obs = hocbf.y_obs
        ro = hocbf.ro
        ra = hocbf.ra
        sim_duration = hocbf.sim_duration

        hocbf.run_simulation()

        t_idx_where = np.where(
            (hocbf.v_y_vals != v_y_target) | (hocbf.v_x_vals != v_x_targert)
        )[0]

        if len(t_idx_where) >= 1:
            t0_idx = max(t_idx_where[0] - 1, 0)
        else:
            t0_idx = 0  # Use 0 s as default time
            print(
                "[INFO]: Use the initial point as the starting point for plotting lower bound."
            )

        t_h_lb, h_lb = hocbf.compute_lower_bound(t0_idx=t0_idx)

        t_h_lb_list.append(t_h_lb)
        h_lb_list.append(h_lb)

        t_h_list.append(hocbf.time)
        h_list.append(hocbf.h_vals)
        p_x_list.append(hocbf.p_x_vals)
        p_y_list.append(hocbf.p_y_vals)
        v_y_list.append(hocbf.v_y_vals)

        lambda_1_list.append(hocbf.lambda_1)
        lambda_2_list.append(hocbf.lambda_2)
        lambda_3_list.append(hocbf.lambda_3)

    colormap = plt.colormaps.get_cmap(
        "viridis"
    )  # viridis, plasma, magma, inferno, cividis
    colormap = [
        colormap(i / (len(h_list) - 1)) for i in range(len(h_list))
    ]  # Generate colors

    # ----------------------------
    # Plot footprint
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    plt.xlabel(r"$x$ (m)")
    plt.ylabel(r"$y$ (m)")
    ax.set_aspect("equal", adjustable="datalim")

    circle = plt.Circle(
        (x_obs, y_obs), ro, color="red", alpha=0.5, edgecolor="black", linewidth=2
    )
    ax.add_patch(circle)
    ax.text(
        x_obs,
        y_obs,
        "Obstacle",
        ha="center",
        va="center",
        fontsize=12,
        color="red",
        weight="bold",
    )

    for idx, DT in enumerate(DTs):
        lambda_1 = lambda_1_list[idx]
        lambda_2 = lambda_2_list[idx]
        lambda_3 = lambda_3_list[idx]

        p_x_vals = p_x_list[idx]
        p_y_vals = p_y_list[idx]

        if approach.lower() == "hocbf":
            if relative_degree == 1:
                label_fp = rf"Footprint $(r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}})$"
            elif relative_degree == 2:
                label_fp = rf"Footprint $(r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}})$"
            elif relative_degree == 3:
                label_fp = rf"Footprint $(r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}}, \lambda_3={{{lambda_3}}})$"
        else:
            label_fp = rf"$\Delta t={{{DT:.2f}}}$ s"

        plt.plot(
            p_x_vals,
            p_y_vals,
            linestyle="-",
            color=colormap[idx],
            linewidth=3,
            label=label_fp,
        )
        # Select a subset of time steps for footprint visualization
        footprint_interval = 10
        for i in range(0, len(p_x_vals), footprint_interval):
            footprint = plt.Circle(
                (p_x_vals[i], p_y_vals[i]), ra, color=colormap[idx], alpha=0.2
            )
            ax.add_patch(footprint)

        if is_visu_moving_direction:
            # Indicate movement direction with an arrow
            arrow_x, arrow_y = p_x_vals[0], p_y_vals[0]  # Midpoint
            ax.annotate(
                "",
                xy=(arrow_x + 2, arrow_y),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=2),
            )

    ax.legend(loc="lower right", frameon=False, ncol=2)

    # Remove x and y ticks
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Remove outer box
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)

    # ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_name = f"fig_footprints_{hocbf.approach}"
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")

    plt.show()

    # -------------------------------
    # Plot h
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    plt.xlabel(r"Time $t$ (s)")
    plt.ylabel(r"CBF $h, h_{{\mathrm{lb}}}$ (m$^2$)")
    plt.grid(True, linestyle="--", alpha=0.4)
    for idx, DT in enumerate(DTs):
        lambda_1 = lambda_1_list[idx]
        lambda_2 = lambda_2_list[idx]
        lambda_3 = lambda_3_list[idx]

        h_vals = h_list[idx]
        t_h = t_h_list[idx]
        v_y_vals = v_y_list[idx]
        t_h_lb = t_h_lb_list[idx]
        h_lb = h_lb_list[idx]

        if approach.lower() == "hocbf":
            if relative_degree == 1:
                label_h = rf"$h (r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}})$"
                label_h_lb = (
                    rf"$h_{{lb}} (r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}})$"
                )
            elif relative_degree == 2:
                label_h = rf"$h (r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}})$"
                label_h_lb = rf"$h_{{lb}} (r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}})$"
            elif relative_degree == 3:
                label_h = rf"$h (r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}}, \lambda_3={{{lambda_3}}})$"
                label_h_lb = rf"$h_{{lb}} (r={{{relative_degree}}}, \lambda_1 = {{{lambda_1}}}, \lambda_2={{{lambda_2}}}, \lambda_3={{{lambda_3}}})$"
        else:
            label_h = rf"$\Delta t={{{DT:.2f}}}$ s"
            label_h_lb = rf"$\Delta t={{{DT:.2f}}}$ s"

        plt.plot(
            t_h,
            h_vals,
            linestyle="-",
            color=colormap[idx],
            linewidth=3,
            label=label_h,
        )
        # plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        # Plot lower bound
        if is_visu_lb:
            t_h_lb = t_h_lb_list[idx]
            h_lb = h_lb_list[idx]
            plt.plot(
                t_h_lb,
                h_lb,
                linestyle=":",
                color=colormap[idx],
                linewidth=3,
                # label=label_h_lb,
                marker="o",
                markersize=6,
                markevery=2,
                alpha=0.4,
            )

    plt.legend(loc="best", frameon=False, ncol=2)
    plt.xlim(0, sim_duration)
    plt.ylim(-0.5, 30)
    plt.xticks(np.arange(0, sim_duration + 0.1, 0.5))
    plt.tight_layout()
    fig_name = f"fig_h_{hocbf.approach}"
    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_name}.jpeg")
    plt.show()


if __name__ == "__main__":
    conservative_aggressive_class_k()
    run_experiment_1()
    run_experiment_2()
