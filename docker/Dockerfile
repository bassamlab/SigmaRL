# syntax=docker/dockerfile:1

FROM ros:jazzy

LABEL maintainer="georgehany064@gmail.com"
LABEL version="0.1"
LABEL description="Docker image to run SigmaRL control inside the CPM Lab"

# -----------------------------------------------------------------------------
# System and ROS dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    nano \
    iputils-ping \
    netcat-openbsd \
    git \
    openssh-client \
    python3.12-venv \
    ffmpeg \
    ros-jazzy-desktop-full \
    ros-jazzy-rmw-zenoh-cpp \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# SSH host verification for RWTH GitLab
# -----------------------------------------------------------------------------
RUN mkdir -p -m 0700 /root/.ssh && \
    ssh-keyscan git.rwth-aachen.de >> /root/.ssh/known_hosts

# -----------------------------------------------------------------------------
# CPM Lab ROS 2 workspace
# The SSH key is forwarded only during the build and is not stored in the image.
# -----------------------------------------------------------------------------
WORKDIR /workspace

RUN --mount=type=ssh \
    git clone \
        --branch dev \
        --single-branch \
        git@git.rwth-aachen.de:CPM/Project/Lab/ros2/workspaces/ros2-workspace-lab.git && \
    cd ros2-workspace-lab && \
    git submodule update --init src/common-interfaces && \
    cd src/common-interfaces && git fetch origin gh-path-mode && git switch gh-path-mode

WORKDIR /workspace/ros2-workspace-lab

# Default middleware. It can still be overridden at docker run time.
ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp

# Build the ROS package needed by SigmaRL.
# Sourcing is done in the same RUN layer in which colcon is executed.
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --packages-select cpm_lab_lab_msgs

# Runtime shell initialization is intentionally left to run_sigmarl_control.bash.
WORKDIR /workspace
CMD ["/bin/bash"]
