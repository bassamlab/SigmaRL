import numpy as np
from sigmarl.hocbf_taylor import run
import matplotlib.pyplot as plt

# Number of steps for the simulation
SIM_DURATION = 2.0
lambda_class_k = 1.0

dt_interval = 0.02
dt_min = 0.02
dt_max = 0.20
dt_list = np.arange(dt_min, dt_max + dt_interval, dt_interval)

avg_v_our_1_without_virtual = []
avg_v_our_1 = []
avg_v_our_2 = []
avg_v_our_3 = []
avg_v_hocbf_1 = []
avg_v_hocbf_2 = []
avg_v_hocbf_3 = []

test = []

for DT in dt_list:
    if DT == 0.10:
        is_plot = True
    else:
        is_plot = False
    data = run(SIM_DURATION, DT, lambda_class_k, is_plot)

    avg_v_our_1_without_virtual.append(np.mean(data[0][2]))
    avg_v_our_1.append(np.mean(data[1][2]))
    avg_v_our_2.append(np.mean(data[2][2]))
    avg_v_our_3.append(np.mean(data[3][2]))
    avg_v_hocbf_1.append(np.mean(data[4][2]))
    avg_v_hocbf_2.append(np.mean(data[5][2]))
    avg_v_hocbf_3.append(np.mean(data[6][2]))

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
        "lines.linewidth": 3,
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

fig, axs = plt.subplots(1, 1, figsize=(8, 4))

plt.plot(
    dt_list,
    avg_v_our_1_without_virtual,
    label=r"$r=1$ (our, w/o virtual control)",
    color=our_1_without_virtual_c,
    marker=our_1_without_virtual_m,
    linestyle=our_1_without_virtual_l,
    zorder=3,
    alpha=0.7,
    linewidth=our_1_without_virtual_lw,
    markersize=markersize,
)
l_p_our_1 = plt.plot(
    dt_list,
    avg_v_our_1,
    label=r"$r=1$ (our, with virtual control)",
    color=our_1_c,
    marker=our_1_m,
    linestyle=our_1_l,
    linewidth=our_1_lw,
    markersize=markersize,
)
l_p_our_2 = plt.plot(
    dt_list,
    avg_v_our_2,
    label=r"$r=2$ (our, w/o virtual control)",
    color=our_2_c,
    marker=our_2_m,
    linestyle=our_2_l,
    linewidth=our_2_lw,
    markersize=markersize,
)
l_p_our_3 = plt.plot(
    dt_list,
    avg_v_our_3,
    label=r"$r=3$ (our, w/o virtual control)",
    color=our_3_c,
    marker=our_3_m,
    linestyle=our_3_l,
    linewidth=our_3_lw,
    markersize=markersize,
)
plt.plot(
    dt_list,
    avg_v_hocbf_1,
    label=r"$r=1$ (HOCBF)",
    color=sota_1_c,
    marker=sota_1_m,
    linestyle=sota_1_l,
    linewidth=sota_1_lw,
    markersize=markersize,
)
plt.plot(
    dt_list,
    avg_v_hocbf_2,
    label=r"$r=2$ (HOCBF)",
    color=sota_2_c,
    marker=sota_2_m,
    linestyle=sota_2_l,
    linewidth=sota_2_lw,
    markersize=markersize,
)
plt.plot(
    dt_list,
    avg_v_hocbf_3,
    label=r"$r=3$ (HOCBF)",
    color=sota_3_c,
    marker=sota_3_m,
    linestyle=sota_3_l,
    linewidth=sota_3_lw,
    markersize=markersize,
)

l_p_our_1[0].set_dashes([1, 2])
l_p_our_2[0].set_dashes([1, 4])
l_p_our_3[0].set_dashes([1, 5])


plt.xlabel(r"Sampling period $\Delta t$ (s)")
plt.ylabel(r"Average speed $v$ (m/s)")

# set x limist to fit the dt_list
plt.xlim(dt_list[0], dt_list[-1])
plt.ylim(0, 11)
# xsticks
plt.xticks(dt_list)
# yticks
plt.yticks(np.arange(0, 11, 2))
# grid
plt.grid(True, which="both", linestyle=grid_ls, linewidth=grid_lw, alpha=grid_alpha)
plt.legend(loc="lower left", frameon=False)

# Save to file
fig_name = "fig_eva_dt"
plt.savefig(f"{fig_name}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
print(f"Figure saved to {fig_name}.jpeg")

plt.show()
