#!/usr/bin/env bash
# Download the private lung workshop bundles through the tutorial image.

set -Eeuo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cache_dir="$repo_root/.cache/monai-physio"
container_root="/workspace/monai-physio"
image="${MONAI_PHYSIO_IMAGE:-monai-physio:tutorials}"
bundle_repo="${LUNG_BUNDLE_REPO:-maximilianofir/physioMotionWorkshop}"
bundle_revision="${LUNG_BUNDLE_REVISION:-a6127dd1d2e27c5b59ed3a81c5b4e7490b4bd1bf}"
docker_command=(docker)
if [[ "${MONAI_PHYSIO_DOCKER_USE_SUDO:-false}" == "true" ]]; then
    docker_command=(sudo --preserve-env=HF_TOKEN docker)
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required to download the private workshop bundles" >&2
    exit 1
fi

mkdir -p "$cache_dir/home" "$repo_root/tutorials/network_weights" \
    "$repo_root/tutorials/output"

"${docker_command[@]}" run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "$repo_root:$container_root" \
    --volume "$cache_dir:/cache" \
    --workdir "$container_root" \
    --env HOME=/cache/home \
    --env HF_HOME=/cache/huggingface \
    --env HF_TOKEN \
    --env LOGNAME=monai-physio \
    --env PYTHONPATH="$container_root/src" \
    --env USER=monai-physio \
    "$image" \
    python utils/lung_bundle.py \
        --repository-root "$container_root" \
        --repo-id "$bundle_repo" \
        --revision "$bundle_revision" \
        --profile course \
        --profile offline-segmentation

chest_ct_file="$repo_root/data/Chest-CT/Chest-CT.mha"
if [[ -s "$chest_ct_file" ]]; then
    echo "Reusing the public Chest-CT input at $chest_ct_file"
else
    echo "Downloading the public Chest-CT input for Tutorial 7..."
    "${docker_command[@]}" run --rm \
        --user "$(id -u):$(id -g)" \
        --volume "$repo_root:$container_root" \
        --volume "$cache_dir:/cache" \
        --workdir "$container_root" \
        --env HOME=/cache/home \
        --env XDG_CACHE_HOME=/cache/xdg \
        "$image" \
        monai-physio-download-data Chest-CT --directory data/Chest-CT
fi
