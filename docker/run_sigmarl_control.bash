#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

WORKSPACE_ON_HOST="${SCRIPT_DIR}/../"
WORKSPACE_IN_CONTAINER="/workspace/sigmarl/"
ROS2_WORKSPACE_IN_CONTAINER="/workspace/ros2-workspace-lab/"

IMAGE="sigmarl-dev"
CONTAINER="sigmarl-controller"

XSOCK="/tmp/.X11-unix"

ZENOH_CONFIG_ON_HOST="${SCRIPT_DIR}/zenoh_session_config.json5"
ZENOH_CONFIG_IN_CONTAINER="/etc/zenoh/session.json5"

echo "Executing sigmarl dev terminal with Docker image ${IMAGE}"

# Optional GUI / X11 forwarding if available
GUI_ARGS=()
if [ -n "${DISPLAY:-}" ] && [ -d "${XSOCK}" ]; then
    GUI_ARGS+=(-e DISPLAY="$DISPLAY" -v "${XSOCK}:${XSOCK}")
    XAUTH_FILE="${XAUTHORITY:-$HOME/.Xauthority}"
    if [ -f "$XAUTH_FILE" ]; then
        GUI_ARGS+=(-v "${XAUTH_FILE}:/root/.Xauthority")
    fi
fi

# Optional SSH agent forwarding if socket is available
SSH_ARGS=()
if [ -n "${SSH_AUTH_SOCK:-}" ] && [ -S "${SSH_AUTH_SOCK}" ]; then
    SSH_ARGS+=(-v "${SSH_AUTH_SOCK}:/ssh-agent" -e SSH_AUTH_SOCK=/ssh-agent)
fi

trap "docker rm -f ${CONTAINER} >/dev/null 2>&1 || true" EXIT

docker run -it \
    --network host \
    --privileged \
    "${GUI_ARGS[@]}" \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
    -e ZENOH_SESSION_CONFIG_URI="${ZENOH_CONFIG_IN_CONTAINER}" \
    -v "${WORKSPACE_ON_HOST}:${WORKSPACE_IN_CONTAINER}" \
    -v "${ZENOH_CONFIG_ON_HOST}:${ZENOH_CONFIG_IN_CONTAINER}:ro" \
    "${SSH_ARGS[@]}" \
    --shm-size=1g \
    --name "${CONTAINER}" \
    "${IMAGE}" \
    /bin/bash -c "
        cd ${ROS2_WORKSPACE_IN_CONTAINER}

        source install/setup.bash

        cd ${WORKSPACE_IN_CONTAINER}
        source .venv/bin/activate

        exec bash
    "
