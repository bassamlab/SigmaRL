# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from sigmarl.helper_training import is_latex_available
from sigmarl.map_manager import MapManager

import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt

from sigmarl.helper_scenario import compute_pseudo_tangent_vector
from sigmarl.pseudo_distance import PseudoDistance

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "font.family": "serif",
        "text.usetex": is_latex_available(),
    }
)

scenario_type = "pseudo_distance_example"

map = MapManager(scenario_type=scenario_type, device="cpu", lane_width=0.3)


left_pts = map.parser.lanelets_all[0]["center_line"]
right_pts = map.parser.lanelets_all[1]["center_line"]

left_boundary_pseudo_vector = compute_pseudo_tangent_vector(left_pts)
right_boundary_pseudo_vector = compute_pseudo_tangent_vector(right_pts)


# Overall bounding box
min_x, min_y = (
    map.parser.bounds["min_x"],
    map.parser.bounds["min_y"],
)
max_x, max_y = (
    map.parser.bounds["max_x"],
    map.parser.bounds["max_y"],
)

grid_resolution = 0.05

# Sample a uniform grid over the bounding box
xs = np.arange(min_x, max_x, grid_resolution)
ys = np.arange(min_y + 1, max_y - 5, grid_resolution)
X, Y = np.meshgrid(xs, ys)
grid_points = np.vstack([X.ravel(), Y.ravel()]).T

# Prepare arrays to hold distances for all grid points
shape = X.shape
left_map = np.full(shape, np.nan)
right_map = np.full(shape, np.nan)
final_map = np.full(shape, np.nan)


verts = np.vstack([left_pts, right_pts.flip(dims=[0])])

path = Path(verts)

# Mask grid points inside this lanelet
inside = path.contains_points(grid_points)
pts_inside = grid_points[inside]


pseudo_distance = PseudoDistance(scenario_type, map)

pos = torch.from_numpy(pts_inside).float().unsqueeze(1)


left_distance, _ = pseudo_distance.get_pseudo_distance(
    left_boundary_pseudo_vector, left_pts, pos
)
right_distance, _ = pseudo_distance.get_pseudo_distance(
    right_boundary_pseudo_vector, right_pts, pos
)

left_list = left_distance.detach().numpy()
right_list = right_distance.detach().numpy()


left_vals = np.array(left_list)
right_vals = np.array(right_list)

# Compute final distance as the minimum of left and right
final_vals = np.minimum(left_vals, right_vals)

# Fill into the full maps (flattened indices)
left_map.ravel()[inside] = left_vals
right_map.ravel()[inside] = right_vals
final_map.ravel()[inside] = final_vals

# Remove outliers from final_map
valid_data = final_map[~np.isnan(final_map)]
# Compute IQR statistics
q1 = np.percentile(valid_data, 25)
q3 = np.percentile(valid_data, 75)
iqr = q3 - q1
# Define outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
# Build mask of outliers (preserving nans)
outlier_mask = (final_map < lower_bound) | (final_map > upper_bound)
if np.any(outlier_mask):
    print(f"Outliers detected: {np.sum(outlier_mask)}")

# Set outliers to np.nan
final_map[outlier_mask] = np.nan


# Plot
fig, ax = plt.subplots(figsize=(3, 7), constrained_layout=True)
ax.set_aspect("equal")

ax.plot(
    left_pts[:, 0],
    left_pts[:, 1],
    color="black",
    linestyle="-",
    linewidth=2,
    label="Left Boundary",
    marker="o",
    markersize=3,
    zorder=10,
)
ax.plot(
    right_pts[:, 0],
    right_pts[:, 1],
    color="black",
    linestyle="-",
    linewidth=2,
    label="Right Boundary",
    marker="o",
    markersize=3,
    zorder=10,
)


# Set all spines (outer box lines) to gray
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color("gray")

# Set tick marks and labels to gray
ax.tick_params(axis="both", colors="gray")  # both ticks and tick labels
ax.xaxis.label.set_color("gray")
ax.yaxis.label.set_color("gray")
ax.set_xticks(np.arange(0, 40, 10))
ax.set_yticks(np.arange(0, 80, 20))

# Remove the outer box
# for spine in ax.spines.values():
#     spine.set_visible(False)
ax.tick_params(axis="both", direction="in")
ax.set_xlim((0, 30))
ax.set_ylim((0, 70))
ax.set_xlabel(r"$x$ [m]")
ax.set_ylabel(r"$y$ [m]")

ax.grid(False)


vmax = np.nanmax(final_map)
# mesh = ax.pcolormesh(xs, ys, data, shading='auto', vmin=0, vmax=vmax, cmap='Blues')
data_masked = np.ma.masked_invalid(final_map)
mesh = ax.pcolormesh(
    xs, ys, data_masked, shading="auto", vmin=0, vmax=vmax, cmap="Blues"
)
# Add colorbar
cbar = fig.colorbar(mesh, ax=ax, label=r"$d_{\mathrm{pseudo}}$ [m]", shrink=0.4)

tick_vals = np.linspace(0, vmax, 4)
cbar.set_ticks(tick_vals)
cbar.ax.set_yticklabels([f"{v:.2f}" for v in tick_vals])

mesh.set_rasterized(True)

# ax.autoscale()

fig_name = f"fig_pseudo_distance_example.pdf"
fig.savefig(fig_name, dpi=450)
plt.close(fig)

print(f"Fig has been saved at {fig_name}")
