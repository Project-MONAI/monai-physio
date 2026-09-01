#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly NGC_IMAGE="nvcr.io/0569033758414229/physiomotion:v0.3-cu126"
readonly LOCAL_IMAGE="monai-physio:tutorials"
readonly INSTALL_DIR="${HOME}/monai-physio"

require_parameter() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "${name} must be supplied as a required Brev launch parameter" >&2
        exit 1
    fi
}

require_parameter NGC_API_KEY
require_parameter HF_TOKEN

if [[ "${NGC_API_KEY}" =~ [[:space:]] ]]; then
    echo "NGC_API_KEY must not contain whitespace" >&2
    exit 1
fi

command -v docker >/dev/null 2>&1 || {
    echo "Docker is required" >&2
    exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is required" >&2
    exit 1
}
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "The tutorial image requires an x86_64 Brev instance" >&2
    exit 1
fi

use_sudo=false
if docker info >/dev/null 2>&1; then
    :
elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    use_sudo=true
else
    echo "The setup user cannot access Docker or passwordless sudo" >&2
    exit 1
fi

docker_cmd() {
    if [[ "${use_sudo}" == "true" ]]; then
        sudo docker "$@"
    else
        docker "$@"
    fi
}

registry_config_dir="$(mktemp -d)"
registry_docker_cmd() {
    docker_cmd --config "${registry_config_dir}" "$@"
}

logged_in=false
cleanup() {
    if [[ "${logged_in}" == "true" ]]; then
        registry_docker_cmd logout nvcr.io >/dev/null 2>&1 || true
    fi
    unset NGC_API_KEY
    rm -rf "${registry_config_dir}"
}
trap cleanup EXIT

echo "Authenticating to the private NGC registry..."
printf '%s' "${NGC_API_KEY}" |
    registry_docker_cmd login nvcr.io --username '$oauthtoken' --password-stdin \
        >/dev/null
logged_in=true

echo "Pulling ${NGC_IMAGE}..."
registry_docker_cmd pull "${NGC_IMAGE}"
docker_cmd tag "${NGC_IMAGE}" "${LOCAL_IMAGE}"

echo "Checking GPU access and core Python dependencies..."
docker_cmd run --rm --gpus all --entrypoint python "${LOCAL_IMAGE}" -c '
import torch
import monai_physio

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the tutorial container")
print(f"GPU ready: {torch.cuda.get_device_name(0)}")
'

image_id="$(docker_cmd image inspect "${LOCAL_IMAGE}" --format '{{.Id}}')"
cleanup
logged_in=false
trap - EXIT

echo "Preparing the persistent workspace at ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
docker_cmd run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "${INSTALL_DIR}:/deployment" \
    --entrypoint bash \
    "${LOCAL_IMAGE}" \
    -c 'cp -a /workspace/monai-physio/. /deployment/'

echo "Downloading the private lung workshop bundles..."
MONAI_PHYSIO_DOCKER_USE_SUDO="${use_sudo}" \
MONAI_PHYSIO_IMAGE="${LOCAL_IMAGE}" \
    "${INSTALL_DIR}/docker/download-lung-bundles.sh"
unset HF_TOKEN

cat <<EOF

MONAI Physio is ready.

Private image: ${NGC_IMAGE}
Local image:   ${LOCAL_IMAGE}
Image ID:      ${image_id}
Workspace:     ${INSTALL_DIR}

The course and offline-segmentation bundles have been downloaded and verified.
Run:

  cd ${INSTALL_DIR}
  ./docker/check-host.sh
  ./docker/tutorial-shell.sh

The NGC and Hugging Face credentials were used only during setup and have been
removed from the setup environment and Docker client configuration.
EOF
