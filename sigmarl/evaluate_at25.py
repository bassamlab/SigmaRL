import torch
from matplotlib import pyplot as plt
from matplotlib import animation
from sigmarl.helper_scenario import get_distances_between_agents, interX, get_rectangle_vertices, get_perpendicular_distances
from sigmarl.parse_xml import ParseXML

def get_agent_agent_collision_timesteps(
    a2a_distances: torch.Tensor,
    n_1: int = 3,
    n_2: int = 5
) -> dict:
    """
    Detect and group agent-agent collisions using hysteresis.
    
    Args:
        a2a_distances (Tensor): [num_steps, num_agents, num_agents]
        n_1 (int): Min consecutive steps of collision to count as one event
        n_2 (int): Min consecutive steps without collision to end event

    Returns:
        dict: {agent_index: [list of collision time steps]}
    """
    num_steps, num_agents, _ = a2a_distances.shape
    collisions = {i: [] for i in range(num_agents)}

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            distance_ij = a2a_distances[:, i, j]
            neg_count = 0
            pos_count = 0
            in_collision = False

            for t, d in enumerate(distance_ij):
                if d < 0:
                    neg_count += 1
                    pos_count = 0
                    if not in_collision and neg_count >= n_1:
                        collisions[i].append(t)
                        collisions[j].append(t)
                        in_collision = True
                else:
                    pos_count += 1
                    neg_count = 0
                    if in_collision and pos_count >= n_2:
                        in_collision = False

    return collisions

def get_boundary_collision_timesteps_interX(
    vertices: torch.Tensor,
    boundaries: list[tuple[torch.Tensor, torch.Tensor]],  # [(left, right), ...] for each agent
    n_1: int = 3,
    n_2: int = 5
) -> dict:
    """
    Detect boundary collisions per agent using interX only (no distance check).

    Args:
        vertices (Tensor): [num_steps, num_agents, 5, 2]
        boundaries (list): [(left, right)] for each agent
        n_1 (int): Min steps to register a collision
        n_2 (int): Min steps to clear a collision

    Returns:
        dict: {agent_index: [collision start timesteps]}
    """
    num_steps, num_agents, _, _ = vertices.shape
    collision_timesteps = {i: [] for i in range(num_agents)}

    for i in range(num_agents):
        agent_vertices = vertices[:, i]  # [num_steps, 5, 2]
        left = boundaries[i][0].unsqueeze(0).repeat(num_steps, 1, 1)
        right = boundaries[i][1].unsqueeze(0).repeat(num_steps, 1, 1)

        collision_left = interX(agent_vertices, left)  # [num_steps]
        collision_right = interX(agent_vertices, right)
        collision = collision_left | collision_right

        in_collision = False
        pos_count = 0
        neg_count = 0

        for t, c in enumerate(collision):
            if c:
                neg_count += 1
                pos_count = 0
                if not in_collision and neg_count >= n_1:
                    collision_timesteps[i].append(t)
                    in_collision = True
            else:
                pos_count += 1
                neg_count = 0
                if in_collision and pos_count >= n_2:
                    in_collision = False

    return collision_timesteps

def get_boundary_collision_timesteps_distance(
    a2ref_distances: torch.Tensor,
    threshold: float = 0.3,
    n_1: int = 3,
    n_2: int = 5
) -> dict:
    """
    Detect boundary violations based only on distance from the reference path.

    Args:
        a2ref_distances (Tensor): [num_steps, num_agents]
        threshold (float): Distance above which violation is detected
        n_1 (int): Min steps above threshold to count as collision
        n_2 (int): Min steps below threshold to end collision

    Returns:
        dict: {agent_index: [collision start timesteps]}
    """
    num_steps, num_agents = a2ref_distances.shape
    collision_timesteps = {i: [] for i in range(num_agents)}

    for i in range(num_agents):
        distances = a2ref_distances[:, i]
        above = distances > threshold

        in_violation = False
        pos_count = 0
        neg_count = 0

        for t, is_violating in enumerate(above):
            if is_violating:
                neg_count += 1
                pos_count = 0
                if not in_violation and neg_count >= n_1:
                    collision_timesteps[i].append(t)
                    in_violation = True
            else:
                pos_count += 1
                neg_count = 0
                if in_violation and pos_count >= n_2:
                    in_violation = False

    return collision_timesteps

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.family": "serif",
        "text.usetex": False,  # Enable Latex for publication
    }
)

# from sigmarl.helper_training
# td = torch.load("../outputs/sigmarl-eval.td", weights_only=False)  # Load TensorDict data 
td = torch.load("outputs/at_v2/seed1/out_td.td", weights_only=False)  # Load TensorDict data 

max_steps_to_use = 3000  # Set to None to use all steps

# Get full size
num_steps_full = td["pos"].shape[0]

# Determine how many steps to use
if max_steps_to_use is not None:
    num_steps = min(max_steps_to_use, num_steps_full)
else:
    num_steps = num_steps_full

# print(td)

w = 0.107  # width of the vehicle
l = 0.220  # length of the vehicle

pos = td["pos"][:num_steps] # [num_steps, num_agents, 2]
rot = td["rot"][:num_steps] # [num_steps, num_agents]
vel = td["vel"][:num_steps] # [num_steps, num_agents, 2]
speed = vel.norm(dim=-1) # [num_steps, num_agents]

print(pos[1])
print(rot[1])

is_collision_with_agents = td["is_collision_with_agents"][:num_steps] # [num_steps, num_agents]
is_collision_with_lanelets = td["is_collision_with_lanelets"][:num_steps] # [num_steps, num_agents]
pos_predicted_list = td["pos_predicted_list"][:num_steps] # [num_steps, num_agents, num_predicted_steps, 2]
vel_predicted_list = td["vel_predicted_list"][:num_steps] # [num_steps, num_agents, num_predicted_steps, 2]


num_steps, num_agents, _ = pos.shape

# Initialize
is_closed_shape = True
vertices = torch.zeros(num_steps, num_agents, 5 if is_closed_shape else 4, 2)  # [num_steps, num_agents, 4/5, 2]
a2ref_distances = torch.zeros(num_steps, num_agents)  # [num_steps, num_agents], distance between agents and reference point


# Map-related data
ref_path_ids = [5, 6, 2, 2, 1]  # This is hard coded!!
map = ParseXML(scenario_type="CPM_entire")

ref_paths = []
left_boundaries = []
right_boundaries = []
for r_id in ref_path_ids:
    ref_p = map.reference_paths[r_id]
    ref_paths.append(ref_p)
    left_boundaries.append(ref_p["left_boundary"])
    right_boundaries.append(ref_p["right_boundary"])
    # left_boundaries.append(ref_p["left_boundary_shared"])  # Two parallel lanes share the same boundary
    # right_boundaries.append(ref_p["right_boundary_shared"])



for i_a in range(num_agents):
    vertices[:, i_a] = get_rectangle_vertices(pos[:, i_a, :], rot[:, i_a].unsqueeze(-1), w, l, True)
    a2ref_distances[:, i_a], _ = get_perpendicular_distances(pos[:, i_a], ref_paths[i_a]["center_line"])

# Max deviation from the reference path
print(f"Max deviation from the reference path of each agent: {a2ref_distances.max(dim=0)}")
print(f"Mean deviation from the reference path of each agent: {a2ref_distances.mean(dim=0)}")
print(f"Average speed of each agent: {speed.mean(dim=0)}")

a2a_distances = get_distances_between_agents(vertices, distance_type="mtv", is_set_diagonal=False)  # [num_steps, num_agents, num_agents] Minimum translation vector-based distance, negative value means collision. However, due to localization error, the distance can be negative even if there is no collision.
a2a_distances.diagonal(dim1=-2, dim2=-1).fill_(10)  # Set diagonal to a large value to avoid "self-collision"
a2a_distances_step_min = a2a_distances.amin(dim=(-1, -2))  # [num_steps], minimum distance between agents of each step
smallest_distances, indices = torch.topk(a2a_distances_step_min, k=20, largest=False)  # TODO: Manurally count the number of collisions. Note that one collision can lead to multiple times of small distances, depending on how long the agents resolved the collision.
print(f"Smallest distances: {smallest_distances}")

a2a_collision_steps = get_agent_agent_collision_timesteps(a2a_distances, n_1=3, n_2=5)

boundary_collision_steps = get_boundary_collision_timesteps_interX(
    vertices,
    list(zip(left_boundaries, right_boundaries)),  # creates [(left, right), ...]
    n_1=3,
    n_2=5
)
#boundary_collision_steps = get_boundary_collision_timesteps_distance(
#    a2ref_distances,
#    threshold=0.03,  # e.g., 3 cm threshold for lane departure
#    n_1=3,
#    n_2=5
#)

print("\n--- Agent-Agent Collisions ---")
total_a2a = sum(len(steps) for steps in a2a_collision_steps.values())
print(f"Total agent-agent collisions: {total_a2a}")
for agent, steps in a2a_collision_steps.items():
    print(f"Agent {agent}: involved in {len(steps)} agent-agent collisions")

print("\n--- Lane Boundary Collisions ---")
total_boundary = sum(len(steps) for steps in boundary_collision_steps.values())
print(f"Total boundary collisions: {total_boundary}")
for agent, steps in boundary_collision_steps.items():
    print(f"Agent {agent}: {len(steps)} boundary collisions")


# Visualization
plt.subplots(figsize=(10, 6), constrained_layout=True)

# Visualize the map
for lanelet in map.lanelets_all:
    # Extract coordinates for left, right, and center lines
    left_bound = lanelet["left_boundary"]
    right_bound = lanelet["right_boundary"]
    center_line = lanelet["center_line"]

    # Extract line markings
    left_line_marking = lanelet["left_line_marking"]
    right_line_marking = lanelet["right_line_marking"]
    center_line_marking = lanelet["center_line_marking"]
    
    # Plot left boundary
    plt.plot(
        left_bound[:, 0],
        left_bound[:, 1],
        linestyle="--" if left_line_marking == "dashed" else "-",
        color="grey",
        linewidth=0.5,
        zorder=10,  # Set zorder to ensure it appears above the map
    )
    # Plot right boundary
    plt.plot(
        right_bound[:, 0],
        right_bound[:, 1],
        linestyle="--" if right_line_marking == "dashed" else "-",
        color="grey",
        linewidth=0.5,
        zorder=10,
    )


plt.scatter(pos[:, 0, 0], pos[:, 0, 1], label="Footprint: vehicle 1", color="tab:blue")  # ref_id = 2
plt.scatter(pos[:, 1, 0], pos[:, 1, 1], label="Footprint: vehicle 2", color="tab:orange")  # ref_id = 5
plt.scatter(pos[:, 2, 0], pos[:, 2, 1], label="Footprint: vehicle 3", color="tab:green")  # ref_id = 1
plt.scatter(pos[:, 3, 0], pos[:, 3, 1], label="Footprint: vehicle 4", color="tab:pink")  # ref_id = 1
plt.scatter(pos[:, 4, 0], pos[:, 4, 1], label="Footprint: vehicle 5", color="tab:red")  # ref_id = 1

# plt.plot(left_boundaries[0][:, 0], left_boundaries[0][:, 1], label="Road boundaries", color="black", linestyle="--", linewidth=2)
# plt.plot(right_boundaries[0][:, 0], right_boundaries[0][:, 1], color="black", linestyle="--", linewidth=2)
# plt.plot(left_boundaries[1][:, 0], left_boundaries[1][:, 1], color="black", linestyle="--", linewidth=2)
# plt.plot(right_boundaries[1][:, 0], right_boundaries[1][:, 1], color="black", linestyle="--", linewidth=2)
# plt.plot(left_boundaries[2][:, 0], left_boundaries[2][:, 1], color="black", linestyle="--", linewidth=2)
# plt.plot(right_boundaries[2][:, 0], right_boundaries[2][:, 1], color="black", linestyle="--", linewidth=2)

plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.xlim(0, 4.5)
plt.ylim(0, 4.0)
plt.legend()
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')  # set equal aspect ratio

plt.gca().xaxis.grid(False)
plt.gca().yaxis.grid(False)


# Save 
plt.tight_layout()  # Set the layout to be tight to minimize white space
plt.savefig("fig_footprint.pdf", format="pdf", bbox_inches="tight")
print(f"A fig is saved at {'fig_footprint.pdf'}")

plt.show()

# === Extract initial position and orientation ===
start_pos = pos[0]       # [num_agents, 2]
start_rot = rot[0]       # [num_agents]

# === Compute vehicle rectangles at t = 0 ===
start_vertices = get_rectangle_vertices(
    start_pos, start_rot.unsqueeze(-1), w, l, is_closed_shape
)  # [num_agents, 5, 2]

# === Plot ===
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Map: lane boundaries
# Visualize the map
for lanelet in map.lanelets_all:
    # Extract coordinates for left, right, and center lines
    left_bound = lanelet["left_boundary"]
    right_bound = lanelet["right_boundary"]
    center_line = lanelet["center_line"]

    # Extract line markings
    left_line_marking = lanelet["left_line_marking"]
    right_line_marking = lanelet["right_line_marking"]
    center_line_marking = lanelet["center_line_marking"]
    
    # Plot left boundary
    plt.plot(
        left_bound[:, 0],
        left_bound[:, 1],
        linestyle="--" if left_line_marking == "dashed" else "-",
        color="grey",
        linewidth=0.5,
        zorder=10,  # Set zorder to ensure it appears above the map
    )
    # Plot right boundary
    plt.plot(
        right_bound[:, 0],
        right_bound[:, 1],
        linestyle="--" if right_line_marking == "dashed" else "-",
        color="grey",
        linewidth=0.5,
        zorder=10,
    )

# 2. Reference centerlines
for i, ref_path in enumerate(ref_paths):
    center = ref_path["center_line"]
    ax.plot(center[:, 0], center[:, 1], label=f"Agent {i+1}", linewidth=1.0)

# 3. Starting rectangles
for i in range(num_agents):
    poly = plt.Polygon(start_vertices[i], closed=True, edgecolor='black', facecolor='C'+str(i), alpha=0.7)
    ax.add_patch(poly)
    ax.text(start_pos[i, 0], start_pos[i, 1], f"{i}", fontsize=12, ha='center', va='center', color='white')

# === Final plot formatting ===
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("fig_experiment-setup.pdf", format="pdf", bbox_inches="tight")
print(f"A fig is saved at {'fig_experiment-setup.pdf'}")

plt.show()


# === Setup figure ===
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:red']
agent_patches = []

# === Initialize agent footprint polygons ===
for i in range(num_agents):
    polygon = plt.Polygon(vertices[0, i].cpu().numpy(), closed=True, color=colors[i], alpha=0.7, zorder=2)
    agent_patches.append(polygon)
    ax.add_patch(polygon)

# === Plot static map elements ===
# Visualize the map
for lanelet in map.lanelets_all:
    # Extract coordinates for left, right, and center lines
    left_bound = lanelet["left_boundary"]
    right_bound = lanelet["right_boundary"]
    center_line = lanelet["center_line"]

    # Extract line markings
    left_line_marking = lanelet["left_line_marking"]
    right_line_marking = lanelet["right_line_marking"]
    center_line_marking = lanelet["center_line_marking"]
    
    # Plot left boundary
    plt.plot(
        left_bound[:, 0],
        left_bound[:, 1],
        linestyle="--" if left_line_marking == "dashed" else "-",
        color="grey",
        linewidth=0.5,
        zorder=10,  # Set zorder to ensure it appears above the map
    )
    # Plot right boundary
    plt.plot(
        right_bound[:, 0],
        right_bound[:, 1],
        linestyle="--" if right_line_marking == "dashed" else "-",
        color="grey",
        linewidth=0.5,
        zorder=10,
    )

ax.set_xlim(0, 4.5)
ax.set_ylim(0, 4.0)
ax.set_aspect("equal")
ax.set_title("Agent Motion with Collision Highlighting")

# === Prepare animation update function ===
def update(t):
    ax.set_title(f"Time Step: {t}", fontsize=12)

    for i in range(num_agents):
        polygon = agent_patches[i]
        polygon.set_xy(vertices[t, i].cpu().numpy())

        # Reset colors
        polygon.set_facecolor(colors[i])
        polygon.set_edgecolor('black')
        polygon.set_linewidth(1)

        # Highlight if collision
        if t in a2a_collision_steps[i]:
            polygon.set_edgecolor('red')
            polygon.set_linewidth(2.5)

        if t in boundary_collision_steps[i]:
            polygon.set_facecolor('red')
            polygon.set_alpha(0.8)

    return agent_patches

# === Animate ===
ani = animation.FuncAnimation(
    fig, update, frames=num_steps, interval=100, blit=False
)

# === Save animation (optional) ===
ani.save("collision_animation.mp4", writer="ffmpeg", fps=10)
print("Animation saved to 'collision_animation.mp4'")

# === Show animation ===
plt.show()