import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
alpha = 0.4

v_target = 10.0  # Target speed

x0 = -10.0  # m
v0 = v_target  # m/s
a0 = 0.0  # m/s
y_const = 3.1  # m

v_min, v_max = -100, 100
a_min, a_max = -100, 100
j_min, j_max = -100, 100

ra = 1.0  # agent radius
ro = 2.0  # obstacle radius

u_nominal = 5.0  # Nominal control input


def run_simulation_own(
    case_num=0, num_steps=50, dt=0.2, virtual_control_input=False, lambda_class_k=0.5
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
    # Initial conditions
    x_current = x0
    v_current = v0
    a_current = a0

    # CBF parameters
    cbf_safe_dist_sqr = (ra + ro) ** 2

    # Storage for time, states, and control
    time = np.zeros(num_steps)
    x_vals = np.zeros(num_steps)
    v_vals = np.zeros(num_steps)
    a_vals = np.zeros(num_steps)
    j_vals = np.zeros(num_steps)
    h_vals = np.zeros(num_steps)

    for k in range(num_steps):
        time[k] = k * dt
        # Define QP variable
        u_var = cp.Variable()

        # Taylor approximation of the CBF
        if case_num == 1:  # Speed as control input
            if virtual_control_input:
                # Virtual control input is only implemented for case 1
                # Use virtual acceleration as control input
                h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
                d_h = 2 * x_current * v_current
                dd_h = 2 * (v_current**2 + x_current * u_var)
                # Second order Taylor approximation
                cbf_cond_taylor = lambda_class_k * h + d_h * dt + 1 / 2 * dd_h * dt**2
                # Store data
                x_vals[k] = x_current
                h_vals[k] = h

                v_next = v_current + u_var * dt  # Virtual control input is acceleration

                cost = cp.square(v_next - v_target)
            else:
                h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
                d_h = 2 * x_current * u_var
                # First order Taylor approximation
                cbf_cond_taylor = lambda_class_k * h + d_h * dt
                # Store data
                x_vals[k] = x_current
                h_vals[k] = h

                v_next = u_var

                cost = cp.square(v_next - v_target)

            constraints = [
                cbf_cond_taylor >= 0,
                # u_var >= v_min,
                # u_var <= v_max
            ]

            # # Since the solution space in one dimension is an interval, we can find the bounds by solving two LPs.
            # # Lower bound: minimize u_var subject to constraints
            # prob_lower = cp.Problem(cp.Minimize(u_var), constraints)
            # prob_lower.solve()
            # u_lower = u_var.value

            # # Upper bound: maximize u_var subject to constraints
            # # Note: maximizing u_var is equivalent to minimizing -u_var
            # prob_upper = cp.Problem(cp.Minimize(-u_var), constraints)
            # prob_upper.solve()
            # u_upper = u_var.value

            # print("Feasible region for u_var is [{}, {}]".format(u_lower, u_upper))
            # percentage = (u_upper - u_lower) / (v_max - v_min) * 100
            # print("Percentage of feasible region: {:.2f}%".format(percentage))

        elif case_num == 2:  # Acceleration as control input
            h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
            d_h = 2 * x_current * v_current
            dd_h = 2 * (v_current**2 + x_current * u_var)
            # Second order Taylor approximation
            cbf_cond_taylor = lambda_class_k * h + d_h * dt + 1 / 2 * dd_h * dt**2
            # Store data
            x_vals[k] = x_current
            v_vals[k] = v_current
            h_vals[k] = h

            v_next = v_current + dt * u_var

            cost = cp.square(v_next - v_target)

            constraints = [
                cbf_cond_taylor >= 0,
                # v_next >= v_min,
                # v_next <= v_max,
                # u_var >= a_min,
                # u_var <= a_max
            ]

        elif case_num == 3:  # Jerk as control input
            # CBF and possibly derivatives of the CBF
            h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
            d_h = 2 * x_current * v_current
            dd_h = 2 * (v_current**2 + x_current * a_current)
            ddd_h = 2 * (3 * v_current * a_current + x_current * u_var)
            # Third order Taylor approximation
            cbf_cond_taylor = (
                lambda_class_k * h
                + d_h * dt
                + 1 / 2 * dd_h * dt**2
                + 1 / 6 * ddd_h * dt**3
            )

            # Store data
            x_vals[k] = x_current
            v_vals[k] = v_current
            a_vals[k] = a_current
            h_vals[k] = h

            v_next = v_current + dt * a_current + 0.5 * (dt**2) * u_var
            a_next = a_current + dt * u_var

            cost = cp.square(v_next - v_target)

            constraints = [
                cbf_cond_taylor >= 0,
                # v_next >= v_min,
                # v_next <= v_max,
                # a_next >= a_min,
                # a_next <= a_max,
                # u_var >= j_min,
                # u_var <= j_max
            ]

        # Form and solve QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                # If solver fails, revert to nominal control
                print(
                    f"[case {case_num} own] Warning at step {k}: QP solver status {prob.status}. Using nominal control."
                )
                u_star = u_nominal
            else:
                u_star = u_var.value
        except:
            # If any error occurs, revert to nominal control
            print(
                f"[case {case_num} own] Warning at step {k}: QP solve failed. Using nominal control."
            )
            u_star = u_nominal

        # Apply control input
        if case_num == 1:
            if virtual_control_input:
                # x_current = x_current + dt * v_current + 0.5 * dt**2 * u_star
                # v_current = v_current + dt * u_star
                v_next = v_current + dt * u_star  # Virtual state
                x_current = x_current + dt * (v_current + v_next) / 2
                v_current = v_next
                # Store data
                a_vals[k] = u_star  # Virtual control input
                u_star = v_next  # Actual control input
                v_vals[k] = u_star
            else:
                x_current = x_current + dt * u_star
                # Store data
                v_vals[k] = u_star
        elif case_num == 2:
            x_current = x_current + dt * v_current + 0.5 * dt**2 * u_star
            v_current = v_current + dt * u_star
            # Store data
            a_vals[k] = u_star
        elif case_num == 3:
            x_current = (
                x_current
                + dt * v_current
                + 0.5 * (dt**2) * a_current
                + 0.25 * (dt**3) * u_star
            )
            v_current = v_current + dt * a_current + 0.5 * (dt**2) * u_star
            a_current = a_current + dt * u_star
            # Store data
            j_vals[k] = u_star

    return time, x_vals, v_vals, a_vals, j_vals, h_vals


def run_simulation_case1(num_steps=50, dt=0.2, alpha_1=1.0):
    """
    Runs the simulation for the single-integrator case (speed control).

    System dynamics:
        x_{k+1} = x_k + dt * u_k
    where u_k (speed) is in the range [-5, 5] m/s.

    The state here is just x_k. The y-position is assumed constant.

    Returns:
        time: array of time points
        x_vals: array of x positions over time
        u_vals: array of control (speed) values over time
    """
    # Initial conditions
    x_current = x0

    # CBF parameters
    cbf_safe_dist_sqr = (ra + ro) ** 2

    # Storage for time, states, and control
    time = np.zeros(num_steps)
    x_vals = np.zeros(num_steps)
    u_vals = np.zeros(num_steps)
    h_vals = np.zeros(num_steps)

    for k in range(num_steps):
        time[k] = k * dt
        # Define QP variable
        u_k = cp.Variable()

        # CBF condition: d_h * dt + alpha_case_1 * dt >= 0
        d_h = 2 * x_current * u_k
        h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
        cbf_cond = d_h * dt + alpha_1 * h

        # Store data
        x_vals[k] = x_current
        h_vals[k] = h

        v_next = u_k
        cost = cp.square(v_next - v_target)

        constraints = [
            cbf_cond >= 0,
            # u_k >= v_min,
            # u_k <= v_max
        ]

        # Form and solve QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                # If solver fails, revert to nominal control
                print(
                    f"[case 1] Warning at step {k}: QP solver status {prob.status}. Using nominal control."
                )
                u_star = u_nominal
            else:
                u_star = u_k.value
        except:
            # If any error occurs, revert to nominal control
            print(
                f"[case 1] Warning at step {k}: QP solve failed. Using nominal control."
            )
            u_star = u_nominal

        # Apply control input
        x_current = x_current + dt * u_star

        # Store data
        u_vals[k] = u_star

    return time, x_vals, u_vals, h_vals


def run_simulation_case2(num_steps=50, dt=0.2, alpha_1=1.0, alpha_2=1.0):
    """
    Runs the simulation for the double-integrator case (acceleration control).

    System dynamics:
        x_{k+1} = x_k + dt * v_k + 0.5 * dt^2 * a_k
        v_{k+1} = v_k + dt * a_k

    Control input a_k (acceleration) is in the range [-5, 5] m/s^2.
    Speed v_k is also enforced in the range [-5, 5] m/s.

    Returns:
        time: array of time points
        x_vals: array of x positions
        v_vals: array of speed values
        a_vals: array of control (acceleration) values
    """
    # Initial conditions
    x_current = x0
    v_current = v0

    # CBF parameters
    cbf_safe_dist_sqr = (ra + ro) ** 2  # 9.0

    # Storage
    time = np.zeros(num_steps)
    x_vals = np.zeros(num_steps)
    v_vals = np.zeros(num_steps)
    a_vals = np.zeros(num_steps)
    h_vals = np.zeros(num_steps)

    for k in range(num_steps):
        time[k] = k * dt
        # Define QP variable
        a_k = cp.Variable()

        # CBF condition (second order)
        d_h = 2 * x_current * v_current
        dd_h = 2 * (v_current**2 + x_current * a_k)
        h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
        cbf_cond = (
            dd_h * dt**2 + (alpha_1 + alpha_2) * d_h * dt + alpha_1 * alpha_2 * h
        )

        # Store data
        x_vals[k] = x_current
        v_vals[k] = v_current
        h_vals[k] = h

        # Next state if we apply a_k
        v_next = v_current + dt * a_k

        # Cost function: keep a_k close to nominal
        cost = cp.square(v_next - v_target)

        # CBF constraint: cbf_cond >= 0
        # Speed constraint: v_next in [v_min, v_max]
        # Acceleration constraint: a_k in [a_min, a_max]
        constraints = [
            cbf_cond >= 0,
            # v_next >= v_min,
            # v_next <= v_max,
            # a_k >= a_min,
            # a_k <= a_max
        ]

        # Form and solve QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(
                    f"[case 2] Warning at step {k}: QP solver status {prob.status}. Using nominal control."
                )
                a_star = u_nominal
            else:
                a_star = a_k.value
        except:
            print(
                f"[case 2] Warning at step {k}: QP solve failed. Using nominal control."
            )
            a_star = u_nominal

        # Apply control input
        x_current = x_current + dt * v_current + 0.5 * dt**2 * a_star
        v_current = v_current + dt * a_star

        # Store data
        a_vals[k] = a_star

    return time, x_vals, v_vals, a_vals, h_vals


def run_simulation_case3(num_steps=50, dt=0.2, alpha_1=1.0, alpha_2=1.0, alpha_3=1.0):
    """
    Runs the simulation for the triple-integrator case (jerk control).

    System dynamics:
        x_{k+1} = x_k + dt * v_k + 0.5 * dt^2 * a_k + (dt^3 / 6) * j_k
        v_{k+1} = v_k + dt * a_k + 0.5 * dt^2 * j_k
        a_{k+1} = a_k + dt * j_k

    Control input j_k (jerk) is in the range [-5, 5] m/s^3.
    We also enforce:
        - speed v_k in [-5, 5] m/s
        - acceleration a_k in [-5, 5] m/s^2

    Returns:
        time: array of time points
        x_vals: array of x positions
        v_vals: array of speed values
        a_vals: array of acceleration values
        j_vals: array of jerk control values
    """
    # Initial conditions
    x_current = x0
    v_current = v0
    a_current = a0

    # CBF parameters
    cbf_safe_dist_sqr = (ra + ro) ** 2  # 9.0

    # Storage
    time = np.zeros(num_steps)
    x_vals = np.zeros(num_steps)
    v_vals = np.zeros(num_steps)
    a_vals = np.zeros(num_steps)
    j_vals = np.zeros(num_steps)
    h_vals = np.zeros(num_steps)

    for k in range(num_steps):
        time[k] = k * dt
        # Define QP variable
        j_k = cp.Variable()

        # CBF condition (third order)
        h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
        d_h = 2 * x_current * v_current
        dd_h = 2 * (v_current**2 + x_current * a_current)
        ddd_h = 2 * (3 * v_current * a_current + x_current * j_k)
        cbf_cond = (
            ddd_h * dt**3
            + (alpha_1 + alpha_2 + alpha_3) * dd_h * dt**2
            + (alpha_1 * alpha_2 + alpha_1 * alpha_3 + alpha_2 * alpha_3) * d_h * dt
            + alpha_1 * alpha_2 * alpha_3 * h
        )

        # Next states if we apply j_k
        v_next = v_current + dt * a_current + 0.5 * (dt**2) * j_k
        a_next = a_current + dt * j_k

        # Cost function: keep j_k close to nominal
        cost = cp.square(v_next - v_target)

        # Store data
        x_vals[k] = x_current
        v_vals[k] = v_current
        a_vals[k] = a_current
        h_vals[k] = h

        # CBF constraint: cbf_cond >= 0
        # Speed constraint: v_next in [v_min, v_max]
        # Acceleration constraint: a_next in [a_min, a_max]
        # Jerk constraint: j_k in [j_min, j_max]
        constraints = [
            cbf_cond >= 0,
            # v_next >= v_min,
            # v_next <= v_max,
            # a_next >= a_min,
            # a_next <= a_max,
            # j_k >= j_min,
            # j_k <= j_max
        ]

        # Form and solve QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(
                    f"[case 3] Warning at step {k}: QP solver status {prob.status}. Using nominal control."
                )
                j_star = u_nominal
            else:
                j_star = j_k.value
        except:
            print(
                f"[case 3] Warning at step {k}: QP solve failed. Using nominal control."
            )
            j_star = u_nominal

        # Apply control input
        x_current = (
            x_current
            + dt * v_current
            + 0.5 * (dt**2) * a_current
            + 0.25 * (dt**3) * j_star
        )
        v_current = v_current + dt * a_current + 0.5 * (dt**2) * j_star
        a_current = a_current + dt * j_star

        # Store data
        j_vals[k] = j_star

    return time, x_vals, v_vals, a_vals, j_vals, h_vals


# Function to create an inset zoom-in
def add_zoom(
    ax, x_range, y_range, loc="upper right", inset_size_x="30%", inset_size_y="30%"
):
    axins = inset_axes(ax, width=inset_size_x, height=inset_size_y, loc=loc)

    # Re-plot all curves from the main axis into the inset
    for idx, line in enumerate(ax.lines):
        (new_line,) = axins.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            label=line.get_label(),
            linewidth=line.get_linewidth(),
        )

        if idx == 1:
            new_line.set_dashes([1, 2])
        elif idx == 2:
            new_line.set_dashes([1, 4])
        elif idx == 3:
            new_line.set_dashes([1, 5])

    # Set zoom-in range
    axins.set_xlim(x_range)
    axins.set_ylim(y_range)

    # Enable grid
    axins.grid(True, linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)

    # Only show the min and max sticks for x and y
    # axins.set_xticks([x_range[0], x_range[1]])
    # axins.set_yticks([y_rangee[0], y_range[1]])

    # Remove tick labels to avoid clutter
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    # Draw rectangle in the main plot
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black")


def run(SIM_DURATION=1.0, DT=0.05, lambda_class_k=1.0, is_plot=False):
    NUM_STEPS = int(SIM_DURATION / DT) + 1

    # Run all three cases using our approach
    (
        time_our_1_without_virtual,
        x_vals_our_1_without_virtual,
        v_vals_our_1_without_virtual,
        a_vals_our_1_without_virtual,
        j_vals_our_1_without_virtual,
        h_vals_our_1_without_virtual,
    ) = run_simulation_own(
        case_num=1,
        num_steps=NUM_STEPS,
        dt=DT,
        virtual_control_input=False,
        lambda_class_k=lambda_class_k,
    )
    (
        time_our_1,
        x_vals_our_1,
        v_vals_our_1,
        a_vals_our_1,
        j_vals_our_1,
        h_vals_our_1,
    ) = run_simulation_own(
        case_num=1,
        num_steps=NUM_STEPS,
        dt=DT,
        virtual_control_input=True,
        lambda_class_k=lambda_class_k,
    )
    (
        time_our_2,
        x_vals_our_2,
        v_vals_our_2,
        a_vals_our_2,
        j_vals_our_2,
        h_vals_our_2,
    ) = run_simulation_own(
        case_num=2,
        num_steps=NUM_STEPS,
        dt=DT,
        virtual_control_input=False,
        lambda_class_k=lambda_class_k,
    )
    (
        time_our_3,
        x_vals_our_3,
        v_vals_our_3,
        a_vals_our_3,
        j_vals_our_3,
        h_vals_our_3,
    ) = run_simulation_own(
        case_num=3,
        num_steps=NUM_STEPS,
        dt=DT,
        virtual_control_input=False,
        lambda_class_k=lambda_class_k,
    )

    # Run all three cases using STOA approach with alpha = 1.0
    time1_alp1, x_vals1_alp1, v_vals1_alp1, h_vals1_alp1 = run_simulation_case1(
        num_steps=NUM_STEPS, dt=DT, alpha_1=1.0
    )
    (
        time2_alp1,
        x_vals2_alp1,
        v_vals2_alp1,
        a_vals2_alp1,
        h_vals2_alp1,
    ) = run_simulation_case2(num_steps=NUM_STEPS, dt=DT, alpha_1=1.0, alpha_2=1.0)
    (
        time3_alp1,
        x_vals3_alp1,
        v_vals3_alp1,
        a_vals3_alp1,
        j_vals3_alp1,
        h_vals3_alp1,
    ) = run_simulation_case3(
        num_steps=NUM_STEPS, dt=DT, alpha_1=1.0, alpha_2=1.0, alpha_3=1.0
    )

    # Run all three cases using STOA approach with alpha = 0.5
    time1_alp5, x_vals1_alp5, v_vals1_alp5, h_vals1_alp5 = run_simulation_case1(
        num_steps=NUM_STEPS, dt=DT, alpha_1=0.5
    )
    (
        time2_alp5,
        x_vals2_alp5,
        v_vals2_alp5,
        a_vals2_alp5,
        h_vals2_alp5,
    ) = run_simulation_case2(num_steps=NUM_STEPS, dt=DT, alpha_1=1.0, alpha_2=0.5)
    (
        time3_alp5,
        x_vals3_alp5,
        v_vals3_alp5,
        a_vals3_alp5,
        j_vals3_alp5,
        h_vals3_alp5,
    ) = run_simulation_case3(
        num_steps=NUM_STEPS, dt=DT, alpha_1=1.0, alpha_2=1.0, alpha_3=0.5
    )

    if is_plot:
        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        lines = []

        # 1) Position for all three cases
        # l_p_our_1_without_virtual = axs[0].plot(
        #     time_our_1_without_virtual,
        #     x_vals_our_1_without_virtual,
        #     label=r"$r=1$ (our, w/o virtual control)",
        #     color=our_1_without_virtual_c,
        #     linestyle=our_1_without_virtual_l,
        #     marker=our_1_without_virtual_m,
        #     linewidth=our_1_without_virtual_lw,
        #     markersize=markersize,
        #     zorder=3,
        #     alpha=0.7,
        # )
        # l_p_our_1 = axs[0].plot(
        #     time_our_1,
        #     x_vals_our_1,
        #     label=r"$r=1$ (our, with virtual control)",
        #     color=our_1_c,
        #     linestyle=our_1_l,
        #     marker=our_1_m,
        #     linewidth=our_1_lw,
        #     markersize=markersize,
        # )
        # l_p_our_2 = axs[0].plot(
        #     time_our_2,
        #     x_vals_our_2,
        #     label=r"$r=2$ (our, w/o virtual control)",
        #     color=our_2_c,
        #     linestyle=our_2_l,
        #     marker=our_2_m,
        #     linewidth=our_2_lw,
        #     markersize=markersize,
        # )
        # l_p_our_3 = axs[0].plot(
        #     time_our_3,
        #     x_vals_our_3,
        #     label=r"$r=3$ (our, w/o virtual control)",
        #     color=our_3_c,
        #     linestyle=our_3_l,
        #     marker=our_3_m,
        #     linewidth=our_3_lw,
        #     markersize=markersize,
        # )
        # axs[0].plot(
        #     time1_alp1,
        #     x_vals1_alp1,
        #     label=r"$r=1$ (HOCBF)",
        #     color=sota_1_c,
        #     linestyle=sota_1_l,
        #     marker=sota_1_m,
        #     linewidth=sota_1_lw,
        #     markersize=markersize,
        # )
        # axs[0].plot(
        #     time2_alp1,
        #     x_vals2_alp1,
        #     label=r"$r=2$ (HOCBF)",
        #     color=sota_2_c,
        #     linestyle=sota_2_l,
        #     marker=sota_2_m,
        #     linewidth=sota_2_lw,
        #     markersize=markersize,
        # )
        # axs[0].plot(
        #     time3_alp1,
        #     x_vals3_alp1,
        #     label=r"$r=3$ (HOCBF)",
        #     color=sota_3_c,
        #     linestyle=sota_3_l,
        #     marker=sota_3_m,
        #     linewidth=sota_3_lw,
        #     markersize=markersize,
        # )
        # axs[0].plot(
        #     time1_alp5,
        #     x_vals1_alp5,
        #     label=r"$r=1$ (HOCBF, $\lambda=0.5$)",
        #     color=sota_1_c,
        #     linestyle=sota_1_l_lam5,
        #     marker=sota_1_m,
        #     alpha=alpha,
        #     linewidth=linewidth,
        #     markersize=markersize,
        # )
        # axs[0].plot(
        #     time2_alp5,
        #     x_vals2_alp5,
        #     label=r"$r=2$ (HOCBF, $\lambda=0.5$)",
        #     color=sota_2_c,
        #     linestyle=sota_2_l_lam5,
        #     marker=sota_2_m,
        #     alpha=alpha,
        #     linewidth=linewidth,
        #     markersize=markersize,
        # )
        # axs[0].plot(
        #     time3_alp5,
        #     x_vals3_alp5,
        #     label=r"$r=3$ (HOCBF, $\lambda=0.5$)",
        #     color=sota_3_c,
        #     linestyle=sota_3_l_lam5,
        #     marker=sota_3_m,
        #     alpha=alpha,
        #     linewidth=linewidth,
        #     markersize=markersize,
        # )

        # # l_p_our_1_without_virtual[0].set_dashes([1, 2])
        # l_p_our_1[0].set_dashes([1, 2])
        # l_p_our_2[0].set_dashes([1, 4])
        # l_p_our_3[0].set_dashes([1, 5])

        # axs[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        # axs[0].set_xticks(
        #     np.arange(0, (NUM_STEPS + 1) * DT, 0.2)
        # )  # + 1 to include the last point
        # axs[0].set_xlim(0, SIM_DURATION)
        # # axs[0].set_xlabel("Time (s)")
        # # Set x axis title
        # # axs[0].set_title("(a)")
        # axs[0].set_ylim([x0 - 1, x0 + SIM_DURATION * v_target + 1])
        # axs[0].set_ylabel(r"Position $x$ (m)")
        # axs[0].legend(loc="best")
        # axs[0].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # axs[0].grid(True)

        # 2) Speed for all three cases
        axs[0].plot(
            time_our_1_without_virtual,
            v_vals_our_1_without_virtual,
            label=r"$r=1$ (our, w/o virtual control)",
            color=our_1_without_virtual_c,
            linestyle=our_1_without_virtual_l,
            marker=our_1_without_virtual_m,
            linewidth=our_1_without_virtual_lw,
            markersize=markersize,
            zorder=3,
            alpha=0.7,
        )
        l_v_our_1 = axs[0].plot(
            time_our_1,
            v_vals_our_1,
            label=r"$r=1$ (our, with virtual control)",
            color=our_1_c,
            linestyle=our_1_l,
            marker=our_1_m,
            linewidth=our_1_lw,
            markersize=markersize,
        )
        l_v_our_2 = axs[0].plot(
            time_our_2,
            v_vals_our_2,
            label=r"$r=2$ (our, w/o virtual control)",
            color=our_2_c,
            linestyle=our_2_l,
            marker=our_2_m,
            linewidth=our_2_lw,
            markersize=markersize,
        )
        l_v_our_3 = axs[0].plot(
            time_our_3,
            v_vals_our_3,
            label=r"$r=3$ (our, w/o virtual control)",
            color=our_3_c,
            linestyle=our_3_l,
            marker=our_3_m,
            linewidth=our_3_lw,
            markersize=markersize,
        )
        axs[0].plot(
            time1_alp1,
            v_vals1_alp1,
            label=r"$r=1$ (HOCBF)",
            color=sota_1_c,
            linestyle=sota_1_l,
            marker=sota_1_m,
            linewidth=sota_1_lw,
            markersize=markersize,
        )
        axs[0].plot(
            time2_alp1,
            v_vals2_alp1,
            label=r"$r=2$ (HOCBF)",
            color=sota_2_c,
            linestyle=sota_2_l,
            marker=sota_2_m,
            linewidth=sota_2_lw,
            markersize=markersize,
        )
        axs[0].plot(
            time3_alp1,
            v_vals3_alp1,
            label=r"$r=3$ (HOCBF)",
            color=sota_3_c,
            linestyle=sota_3_l,
            marker=sota_3_m,
            linewidth=sota_3_lw,
            markersize=markersize,
        )
        # axs[0].plot(
        #     time1_alp5,
        #     v_vals1_alp5,
        #     label=r"$r=1$ (HOCBF, $\lambda=0.5$)",
        #     color=sota_1_c,
        #     linestyle=sota_1_l_lam5,
        #     marker=sota_1_m,
        #     alpha=alpha,
        #     linewidth=linewidth,
        #     markersize=markersize,
        # )
        # axs[0].plot(
        #     time2_alp5,
        #     v_vals2_alp5,
        #     label=r"$r=2$ (HOCBF, $\lambda=0.5$)",
        #     color=sota_2_c,
        #     linestyle=sota_2_l_lam5,
        #     marker=sota_2_m,
        #     alpha=alpha,
        #     linewidth=linewidth,
        #     markersize=markersize,
        # )
        # axs[0].plot(
        #     time3_alp5,
        #     v_vals3_alp5,
        #     label=r"$r=3$ (HOCBF, $\lambda=0.5$)",
        #     color=sota_3_c,
        #     linestyle=sota_3_l_lam5,
        #     marker=sota_3_m,
        #     alpha=alpha,
        #     linewidth=linewidth,
        #     markersize=markersize,
        # )
        # axs[0].set_xticks(np.arange(0, NUM_STEPS * DT, 0.2))
        # axs[0].set_xlabel("Time (s)")
        axs[0].grid(True, linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
        axs[0].set_ylabel(r"(a) Speed $v$ (m/s)")
        axs[0].set_ylim([0, 11])
        axs[0].grid(True)
        axs[0].legend(loc="best")

        l_v_our_1[0].set_dashes([1, 1])
        l_v_our_2[0].set_dashes([1, 2])
        l_v_our_3[0].set_dashes([1, 3])

        # 5) CBF values for all cases
        axs[1].plot(
            time_our_1_without_virtual,
            h_vals_our_1_without_virtual,
            # label=r"$r=1$ (our, no virtual control)",
            color=our_1_without_virtual_c,
            linestyle=our_1_without_virtual_l,
            marker=our_1_without_virtual_m,
            linewidth=our_1_without_virtual_lw,
            markersize=markersize,
            zorder=3,
            alpha=0.7,
        )
        l_h_our_1 = axs[1].plot(
            time_our_1,
            h_vals_our_1,
            # label=r"$r=1$ (our)",
            color=our_1_c,
            linestyle=our_1_l,
            marker=our_1_m,
            linewidth=our_1_lw,
            markersize=markersize,
        )
        l_h_our_2 = axs[1].plot(
            time_our_2,
            h_vals_our_2,
            # label=r"$r=2$ (our)",
            color=our_2_c,
            linestyle=our_2_l,
            marker=our_2_m,
            linewidth=our_2_lw,
            markersize=markersize,
        )
        l_h_our_3 = axs[1].plot(
            time_our_3,
            h_vals_our_3,
            # label=r"$r=3$ (our)",
            color=our_3_c,
            linestyle=our_3_l,
            marker=our_3_m,
            linewidth=our_3_lw,
            markersize=markersize,
        )
        axs[1].plot(
            time1_alp1,
            h_vals1_alp1,
            # label=r"$r=1$ (HOCBF)",
            color=sota_1_c,
            linestyle=sota_1_l,
            marker=sota_1_m,
            linewidth=sota_1_lw,
            markersize=markersize,
        )
        axs[1].plot(
            time2_alp1,
            h_vals2_alp1,
            # label=r"$r=2$ (HOCBF)",
            color=sota_2_c,
            linestyle=sota_2_l,
            marker=sota_2_m,
            linewidth=sota_2_lw,
            markersize=markersize,
        )
        axs[1].plot(
            time3_alp1,
            h_vals3_alp1,
            # label=r"$r=3$ (HOCBF)",
            color=sota_3_c,
            linestyle=sota_3_l,
            marker=sota_3_m,
            linewidth=sota_3_lw,
            markersize=markersize,
        )

        # For r=2:
        r = 2
        # Find the first point where v_vals2_alp1 is smaller than 10
        t_idx_where = np.where(v_vals2_alp1 < 10)[0]
        if len(t_idx_where) >= 1:
            t_idx = t_idx_where[0] - 1
            assert t_idx >= 0
        else:
            t_idx = np.where(time2_alp1 >= 0.2)[0][0]  # Use 0.2 s as default time

        t_start = time2_alp1[t_idx]
        h_start = h_vals2_alp1[t_idx]
        # Plot a exponentially decreasing function starting from point (t_start, h_start), i.e., y = h_start * ((r-1)/r)^(t - t_start)
        t_range = np.linspace(t_start, SIM_DURATION, 100)
        h_range = h_start * ((r - 1) / r) ** ((t_range - t_start) / DT)
        axs[1].plot(
            t_range,
            h_range,
            color="black",
            linestyle=":",
            linewidth=2.5,
            label=rf"$h_{{{t_start:.1f}}} \cdot (\frac{{1}}{2})^{{t-{t_start:.1f}}}$ (for $r=2$)",
            alpha=1.0,
            zorder=3,
        )

        l_h_our_1[0].set_dashes([1, 1])
        l_h_our_2[0].set_dashes([1, 2])
        l_h_our_3[0].set_dashes([1, 3])

        axs[1].set_xlabel(r"Time $t$ (s)")
        axs[1].set_ylabel(r"(b) CBF $h$ (m$^2$)")
        axs[1].grid(True)
        axs[1].legend(loc="lower left", frameon=False, fontsize=12)
        axs[1].set_xlim(0, SIM_DURATION)
        axs[1].set_xticks(np.arange(0, SIM_DURATION + 0.1, 0.2))

        zoom_x_range = (0.75, 1.1)
        zoom_y_range = (0, 6)
        add_zoom(
            axs[1],
            zoom_x_range,
            zoom_y_range,
            loc="upper center",
            inset_size_x="60%",
            inset_size_y="60%",
        )

        # Create a shared legend
        handles, labels = axs[
            0
        ].get_legend_handles_labels()  # Collect labels from the first subplot
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.47, 0.78),
            ncol=2,
            frameon=False,
        )

        # Remove individual legends
        for ax in axs[0:1]:
            ax.legend().remove()

        axs[1].grid(True, linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)

        plt.tight_layout()
        fig_name = f"cbf_ho_{DT}"
        plt.savefig(f"{fig_name}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
        print(f"Figure saved to {fig_name}.jpeg")

        # plt.show()

    data = np.array(
        [
            [
                time_our_1_without_virtual,
                x_vals_our_1_without_virtual,
                v_vals_our_1_without_virtual,
                h_vals_our_1_without_virtual,
            ],
            [time_our_1, x_vals_our_1, v_vals_our_1, h_vals_our_1],
            [time_our_2, x_vals_our_2, v_vals_our_2, h_vals_our_2],
            [time_our_3, x_vals_our_3, v_vals_our_3, h_vals_our_3],
            [time1_alp1, x_vals1_alp1, v_vals1_alp1, h_vals1_alp1],
            [time2_alp1, x_vals2_alp1, v_vals2_alp1, h_vals2_alp1],
            [time3_alp1, x_vals3_alp1, v_vals3_alp1, h_vals3_alp1],
        ]
    )

    return data


if __name__ == "__main__":
    # Number of steps for the simulation
    data = run(SIM_DURATION=2.0, DT=0.1, lambda_class_k=1.0, is_plot=True)
