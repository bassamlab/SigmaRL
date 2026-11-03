#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

WORKSPACE_ON_HOST="${SCRIPT_DIR}/../"
WORKSPACE_IN_CONTAINER="/workspace/sigmarl"
IMAGE="sigmarl-dev"

# -----------------------------------------------------------------------------
# 1. Make sure Docker can access an SSH key during image build
# -----------------------------------------------------------------------------
echo "Checking SSH agent..."

if [ -z "${SSH_AUTH_SOCK:-}" ]; then
    echo "No SSH agent found. Starting one..."
    eval "$(ssh-agent -s)"
fi

if ! ssh-add -l >/dev/null 2>&1; then
    echo "No SSH key is loaded in the SSH agent."
    echo
    echo "Available private-key candidates on this host:"

    find "$HOME/.ssh" \
        -maxdepth 1 \
        -type f \
        ! -name "*.pub" \
        ! -name "known_hosts*" \
        ! -name "config" \
        2>/dev/null || true

    echo
    echo "Load the appropriate key first, for example:"
    echo "  ssh-add ~/.ssh/id_ed25519"
    exit 1
fi

 echo "SSH keys available to Docker:"
ssh-add -l

# -----------------------------------------------------------------------------
# 2. Build the Docker image
# -----------------------------------------------------------------------------
echo
echo "Building Docker image ${IMAGE}..."

docker build \
    --ssh default \
    -t "${IMAGE}" \
    "${SCRIPT_DIR}"

# -----------------------------------------------------------------------------
# 3. Create/update the SigmaRL Python environment on the host-mounted workspace
# -----------------------------------------------------------------------------
echo
echo "Creating SigmaRL virtual environment..."

# all commands are ran inside the volume-mounted workspace, so that the virtual environment is created on the host
docker run --rm \
    -v "${WORKSPACE_ON_HOST}:${WORKSPACE_IN_CONTAINER}" \
    -w "${WORKSPACE_IN_CONTAINER}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        rm -rf .venv
        python3 -m venv .venv
        source .venv/bin/activate
        python3 -m pip install --upgrade pip
        pip install -e .
    '

echo
echo "SigmaRL development image and virtual environment are ready."
