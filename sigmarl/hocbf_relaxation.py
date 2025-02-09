import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})

our_1_no_virtual_c = "tab:gray"
our_1_no_virtual_m = "*"
our_1_no_virtual_l = "-."
our_1_c = "tab:blue"
our_1_m = 2
our_1_l = "-"
our_2_c = "tab:green"
our_2_m = 3
our_2_l = "-"
our_3_c = "tab:orange"
our_3_m = "_"
our_3_l = "-"

sota_1_c = "tab:blue"
sota_1_m = "2"
sota_1_l = "--"
sota_2_c = "tab:green"
sota_2_m = "3"
sota_2_l = "--"
sota_3_c = "tab:orange"
sota_3_m = "_"
sota_3_l = "--"

v_target = 10.0  # Target speed

x0 = -5.0  # m 
v0 = v_target  # m/s
a0 = 0.0  # m/s
y_const = 3.005  # m

v_min, v_max = -100, 100
a_min, a_max = -100, 100
j_min, j_max = -100, 100

ra = 1.0  # agent radius
ro = 2.0  # obstacle radius

u_nominal = 5.0  # Nominal control input

def run_simulation_own(case_num=0, num_steps=50, dt=0.2, virtual_control_input=False):
    """
    Runs the simulation for the single-integrator case (speed control).
    
    System dynamics:
        x_{k+1} = x_k + dt * u_k
    where u_k (speed) is in the range [-5, 5] cm/s.
    
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
    cbf_safe_dist_sqr = (ra + ro)**2
    
    # Storage for time, states, and control
    time = np.arange(0, num_steps * dt, dt)
    x_vals = np.zeros(num_steps)
    v_vals = np.zeros(num_steps)
    a_vals = np.zeros(num_steps)
    j_vals = np.zeros(num_steps)
    h_vals = np.zeros(num_steps)
    
    for k in range(num_steps):        
        # Define QP variable
        u_var = cp.Variable()
        
        # Taylor approximation of the CBF
        if case_num == 1:
            if virtual_control_input:
                # Virtual control input is only implemented for case 1
                # Use virtual acceleration as control input
                h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
                d_h = 2 * x_current * v_current
                dd_h = 2 * (v_current**2 + x_current * u_var)
                # Second order Taylor approximation
                cbf_cond_taylor = h + d_h * dt + 1 / 2 * dd_h * dt**2
                # Store data
                x_vals[k] = x_current
                h_vals[k] = h
                
                v_next = v_current + u_var * dt  # Virtual control input is acceleration
                
                cost = cp.square(v_next - v_target)
            else:
                h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
                d_h = 2 * x_current * u_var
                # First order Taylor approximation
                cbf_cond_taylor = h + d_h * dt
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
            
        elif case_num == 2:
            h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
            d_h = 2 * x_current * v_current
            dd_h = 2 * (v_current**2 + x_current * u_var)
            # Second order Taylor approximation
            cbf_cond_taylor = h + d_h * dt + 1 / 2 * dd_h * dt**2
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
            
        elif case_num == 3:
            # CBF and possibly derivatives of the CBF
            h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
            d_h = 2 * x_current * v_current
            dd_h = 2 * (v_current**2 + x_current * a_current)
            ddd_h = 2 * (3 * v_current * a_current + x_current * u_var)
            # Third order Taylor approximation
            cbf_cond_taylor = h + d_h * dt + 1 / 2 * dd_h * dt**2 + 1 / 6 * ddd_h * dt**3
                
            # Store data
            x_vals[k] = x_current
            v_vals[k] = v_current
            a_vals[k] = a_current
            h_vals[k] = h

            v_next = (v_current 
                    + dt * a_current 
                    + 0.5 * (dt**2) * u_var)
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
                print(f"[case {case_num} own] Warning at step {k}: QP solver status {prob.status}. Using nominal control.")
                u_star = u_nominal
            else:
                u_star = u_var.value
        except:
            # If any error occurs, revert to nominal control
            print(f"[case {case_num} own] Warning at step {k}: QP solve failed. Using nominal control.")
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
            x_current = (x_current 
                        + dt * v_current 
                        + 0.5 * (dt**2) * a_current 
                        + 0.25 * (dt**3) * u_star)
            v_current = (v_current 
                        + dt * a_current 
                        + 0.5 * (dt**2) * u_star)
            a_current = (a_current 
                        + dt * u_star)
            # Store data
            j_vals[k] = u_star
    
    return time, x_vals, v_vals, a_vals, j_vals, h_vals



def run_simulation_case1(num_steps=50, dt=0.2, alpha=1.0):
    """
    Runs the simulation for the single-integrator case (speed control).
    
    System dynamics:
        x_{k+1} = x_k + dt * u_k
    where u_k (speed) is in the range [-5, 5] cm/s.
    
    The state here is just x_k. The y-position is assumed constant.
    
    Returns:
        time: array of time points
        x_vals: array of x positions over time
        u_vals: array of control (speed) values over time
    """
    # Initial conditions
    x_current = x0
    
    # CBF parameters
    cbf_safe_dist_sqr = (ra + ro)**2
    
    # Storage for time, states, and control
    time = np.arange(0, num_steps * dt, dt)
    x_vals = np.zeros(num_steps)
    u_vals = np.zeros(num_steps)
    h_vals = np.zeros(num_steps)
    
    for k in range(num_steps):        
        # Define QP variable
        u_k = cp.Variable()
        
        # CBF condition: d_h * dt + alpha_case_1 * dt >= 0
        d_h = 2 * x_current * u_k
        h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
        cbf_cond = d_h * dt + alpha * h
        
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
                print(f"[case 1] Warning at step {k}: QP solver status {prob.status}. Using nominal control.")
                u_star = u_nominal
            else:
                u_star = u_k.value
        except:
            # If any error occurs, revert to nominal control
            print(f"[case 1] Warning at step {k}: QP solve failed. Using nominal control.")
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
    
    Control input a_k (acceleration) is in the range [-5, 5] cm/s^2.
    Speed v_k is also enforced in the range [-5, 5] cm/s.
    
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
    cbf_safe_dist_sqr = (ra + ro)**2  # 9.0
    
    # Storage
    time = np.arange(0, num_steps * dt, dt)
    x_vals = np.zeros(num_steps)
    v_vals = np.zeros(num_steps)
    a_vals = np.zeros(num_steps)
    h_vals = np.zeros(num_steps)
    
    for k in range(num_steps):       
        # Define QP variable
        a_k = cp.Variable()
        
        # CBF condition (second order)
        d_h = 2 * x_current * v_current
        dd_h = 2 * (v_current**2 + x_current * a_k)
        h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
        cbf_cond = dd_h * dt**2 + (alpha_1 + alpha_2) * d_h * dt + alpha_1 * alpha_2 * h
        
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
                print(f"[case 2] Warning at step {k}: QP solver status {prob.status}. Using nominal control.")
                a_star = u_nominal
            else:
                a_star = a_k.value
        except:
            print(f"[case 2] Warning at step {k}: QP solve failed. Using nominal control.")
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
    
    Control input j_k (jerk) is in the range [-5, 5] cm/s^3.
    We also enforce:
        - speed v_k in [-5, 5] cm/s
        - acceleration a_k in [-5, 5] cm/s^2
    
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
    cbf_safe_dist_sqr = (ra + ro)**2  # 9.0
    
    # Storage
    time = np.arange(0, num_steps * dt, dt)
    x_vals = np.zeros(num_steps)
    v_vals = np.zeros(num_steps)
    a_vals = np.zeros(num_steps)
    j_vals = np.zeros(num_steps)
    h_vals = np.zeros(num_steps)
    
    for k in range(num_steps):       
        # Define QP variable
        j_k = cp.Variable()
        
        # CBF condition (third order)
        h = (x_current**2 + y_const**2) - cbf_safe_dist_sqr
        d_h = 2 * x_current * v_current
        dd_h = 2 * (v_current**2 + x_current * a_current)
        ddd_h = 2 * (3 * v_current * a_current + x_current * j_k)
        cbf_cond = ddd_h * dt**3 + (alpha_1 + alpha_2 + alpha_3) * dd_h * dt**2 + (alpha_1 * alpha_2 + alpha_1 * alpha_3 + alpha_2 * alpha_3) * d_h * dt + alpha_1 * alpha_2 * alpha_3 * h
        
        # Next states if we apply j_k
        v_next = (v_current 
                  + dt * a_current 
                  + 0.5 * (dt**2) * j_k)
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
                print(f"[case 3] Warning at step {k}: QP solver status {prob.status}. Using nominal control.")
                j_star = u_nominal
            else:
                j_star = j_k.value
        except:
            print(f"[case 3] Warning at step {k}: QP solve failed. Using nominal control.")
            j_star = u_nominal
        
        print(f"j_star: {j_star}")
        # Apply control input
        x_current = (x_current 
                     + dt * v_current 
                     + 0.5 * (dt**2) * a_current 
                     + 0.25 * (dt**3) * j_star)
        v_current = (v_current 
                     + dt * a_current 
                     + 0.5 * (dt**2) * j_star)
        a_current = (a_current 
                     + dt * j_star)
        
        # Store data
        j_vals[k] = j_star
    
    return time, x_vals, v_vals, a_vals, j_vals, h_vals


if __name__ == "__main__":
    # Number of steps for the simulation
    SIM_DURATION = 1.0
    DT = 0.05
    NUM_STEPS = int(SIM_DURATION / DT)
    
    # Run all three cases using our approach
    time_our_1_no_virtual, x_vals_our_1_no_virtual, v_vals_our_1_no_virtual, a_vals_our_1_no_virtual, j_vals_our_1_no_virtual, h_vals_our_1_no_virtual = run_simulation_own(case_num=1, num_steps=NUM_STEPS, dt=DT, virtual_control_input=False)
    time_our_1, x_vals_our_1, v_vals_our_1, a_vals_our_1, j_vals_our_1, h_vals_our_1 = run_simulation_own(case_num=1, num_steps=NUM_STEPS, dt=DT, virtual_control_input=True)
    time_our_2, x_vals_our_2, v_vals_our_2, a_vals_our_2, j_vals_our_2, h_vals_our_2 = run_simulation_own(case_num=2, num_steps=NUM_STEPS, dt=DT, virtual_control_input=False)
    time_our_3, x_vals_our_3, v_vals_our_3, a_vals_our_3, j_vals_our_3, h_vals_our_3 = run_simulation_own(case_num=3, num_steps=NUM_STEPS, dt=DT, virtual_control_input=False)
    
    # Run all three cases using STOA approach
    time1, x_vals1, u_vals1, h_vals1 = run_simulation_case1(num_steps=NUM_STEPS, dt=DT, alpha=1.0)
    time2, x_vals2, v_vals2, a_vals2, h_vals2 = run_simulation_case2(num_steps=NUM_STEPS, dt=DT, alpha_1=1.0, alpha_2=1.0)
    time3, x_vals3, v_vals3, a_vals3, j_vals3, h_vals3 = run_simulation_case3(num_steps=NUM_STEPS, dt=DT, alpha_1=1.0, alpha_2=1.0, alpha_3=1.0)
    
    # Compute safety distance for all cases 
    safety_dist_our_1 = np.sqrt(x_vals_our_1**2 + y_const**2) - ra - ro
    safety_dist_our_2 = np.sqrt(x_vals_our_2**2 + y_const**2) - ra - ro
    safety_dist_our_3 = np.sqrt(x_vals_our_3**2 + y_const**2) - ra - ro
    safety_dist1 = np.sqrt(x_vals1**2 + y_const**2) - ra - ro
    safety_dist2 = np.sqrt(x_vals2**2 + y_const**2) - ra - ro
    safety_dist3 = np.sqrt(x_vals3**2 + y_const**2) - ra - ro
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    
    # 1) Position for all three cases
    axs[0].plot(time_our_1_no_virtual, x_vals_our_1_no_virtual, label="Order 1 (our, no virtual control)", color=our_1_no_virtual_c, linestyle=our_1_no_virtual_l, marker=our_1_no_virtual_m)
    axs[0].plot(time_our_1, x_vals_our_1, label="Order 1 (our)", color=our_1_c, linestyle=our_1_l, marker=our_1_m)
    axs[0].plot(time_our_2, x_vals_our_2, label="Order 2 (our)", color=our_2_c, linestyle=our_2_l, marker=our_2_m)
    axs[0].plot(time_our_3, x_vals_our_3, label="Order 3 (our)", color=our_3_c, linestyle=our_3_l, marker=our_3_m)
    axs[0].plot(time1, x_vals1, label="Order 1 (HOCBF)", color=sota_1_c, linestyle=sota_1_l, marker=sota_1_m)
    axs[0].plot(time2, x_vals2, label="Order 2 (HOCBF)", color=sota_2_c, linestyle=sota_2_l, marker=sota_2_m)
    axs[0].plot(time3, x_vals3, label="Order 3 (HOCBF)", color=sota_3_c, linestyle=sota_3_l, marker=sota_3_m)
    axs[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axs[0].set_xticks(np.arange(0, NUM_STEPS * DT, 0.2))
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position (cm)")
    axs[0].legend(loc="best")
    axs[0].grid(True)
    
    # 2) Speed for all three cases
    axs[1].plot(time_our_1_no_virtual, v_vals_our_1_no_virtual, label="Order 1 (our, no virtual control)", color=our_1_no_virtual_c, linestyle=our_1_no_virtual_l, marker=our_1_no_virtual_m)
    axs[1].plot(time_our_1, v_vals_our_1, label="Order 1 (our)", color=our_1_c, linestyle=our_1_l, marker=our_1_m)
    axs[1].plot(time_our_2, v_vals_our_2, label="Order 2 (our)", color=our_2_c, linestyle=our_2_l, marker=our_2_m)
    axs[1].plot(time_our_3, v_vals_our_3, label="Order 3 (our)", color=our_3_c, linestyle=our_3_l, marker=our_3_m)
    axs[1].plot(time1, u_vals1, label="Order 1 (HOCBF)", color=sota_1_c, linestyle=sota_1_l, marker=sota_1_m)
    axs[1].plot(time2, v_vals2, label="Order 2 (HOCBF)", color=sota_2_c, linestyle=sota_2_l, marker=sota_2_m)
    axs[1].plot(time3, v_vals3, label="Order 3 (HOCBF)", color=sota_3_c, linestyle=sota_3_l, marker=sota_3_m)
    axs[1].set_xticks(np.arange(0, NUM_STEPS * DT, 0.2))
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Speed (cm/s)")
    axs[1].grid(True)
    axs[1].legend(loc="best")
    
    # # 3) Acceleration for cases 2 and 3
    # axs[2].plot(time_our_2, a_vals_our_2, label="Order 2 (our)", color=c_case_2, linestyle=line_style_own, marker=marker_style_own)
    # axs[2].plot(time_our_3, a_vals_our_3, label="Order 3 (our)", color=c_case_3, linestyle=line_style_own, marker=marker_style_own)
    # axs[2].plot(time2, a_vals2, label="Order 2 (HOCBF)", color=c_case_2, linestyle=line_style_stoa, marker=marker_style_stoa)
    # axs[2].plot(time3, a_vals3, label="Order 3 (HOCBF)", color=c_case_3, linestyle=line_style_stoa, marker=marker_style_stoa)
    # axs[2].set_xticks(np.arange(0, NUM_STEPS * DT, 0.2))
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Acceleration (cm/s^2)")
    # axs[2].grid(True)
    # axs[2].legend(loc="best")
    
    # # 4) Jerk for case 3
    # axs[3].plot(time_our_3, j_vals_our_3, label="Order 3 (our)", color=c_case_3, linestyle=line_style_own, marker=marker_style_own)
    # axs[3].plot(time3, j_vals3, label="Order 3 (HOCBF)", color=c_case_3, linestyle=line_style_stoa, marker=marker_style_stoa)
    # axs[3].set_ylabel("Jerk (cm/s^3)")
    # axs[3].set_xlabel("Time (s)")
    # axs[3].grid(True)
    # axs[3].legend(loc="best")
    
    # 5) CBF values for all cases
    axs[2].plot(time_our_1_no_virtual, h_vals_our_1_no_virtual, label="Order 1 (our, no virtual control)", color=our_1_no_virtual_c, linestyle=our_1_no_virtual_l, marker=our_1_no_virtual_m)
    axs[2].plot(time_our_1, h_vals_our_1, label="Order 1 (our)", color=our_1_c, linestyle=our_1_l, marker=our_1_m)
    axs[2].plot(time_our_2, h_vals_our_2, label="Order 2 (our)", color=our_2_c, linestyle=our_2_l, marker=our_2_m)
    axs[2].plot(time_our_3, h_vals_our_3, label="Order 3 (our)", color=our_3_c, linestyle=our_3_l, marker=our_3_m)
    axs[2].plot(time1, h_vals1, label="Order 1 (HOCBF)", color=sota_1_c, linestyle=sota_1_l, marker=sota_1_m)
    axs[2].plot(time2, h_vals2, label="Order 2 (HOCBF)", color=sota_2_c, linestyle=sota_2_l, marker=sota_2_m)
    axs[2].plot(time3, h_vals3, label="Order 3 (HOCBF)", color=sota_3_c, linestyle=sota_3_l, marker=sota_3_m)
    print(f"h_vals_our_1: {h_vals_our_1}")
    print(f"h_vals_our_2: {h_vals_our_2}")
    print(f"h_vals_our_3: {h_vals_our_3}")
    print(f"h_vals1: {h_vals1}")
    print(f"h_vals2: {h_vals2}")
    print(f"h_vals3: {h_vals3}")
    axs[2].set_xticks(np.arange(0, NUM_STEPS * DT, 0.2))
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("CBF value")
    axs[2].set_ylim(0, 10)
    axs[2].grid(True)
    axs[2].legend(loc="best")
    
    plt.tight_layout()
    fig_name = "cbf_ho.png"
    plt.savefig("cbf_ho.png", bbox_inches="tight", dpi=600)
    print(f"Figure saved to {fig_name}")
    plt.show()
