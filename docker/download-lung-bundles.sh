#!/usr/bin/env bash
# Download the private lung workshop bundles through the tutorial image.

set -Eeuo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cache_dir="$repo_root/.cache/monai-physio"
container_root="/workspace/monai-physio"
image="${MONAI_PHYSIO_IMAGE:-monai-physio:tutorials}"
bundle_repo="${LUNG_BUNDLE_REPO:-maximilianofir/physioMotionWorkshop}"
bundle_revision="${LUNG_BUNDLE_REVISION:-459c538385d36eb2ebb7a92bb0086494ee2ebdcf}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required to download the private workshop bundles" >&2
    exit 1
fi

mkdir -p "$cache_dir/home" "$repo_root/tutorials/network_weights" \
    "$repo_root/tutorials/output"

docker run --rm \
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
    python utils/lung_bundle.py download \
        --repository-root "$container_root" \
        --repo-id "$bundle_repo" \
        --revision "$bundle_revision" \
        --profile course \
        --profile offline-segmentation
