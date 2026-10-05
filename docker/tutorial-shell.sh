#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cache_dir="$repo_root/.cache/monai-physio"
container_root="/workspace/monai-physio"

mkdir -p "$cache_dir/home" "$repo_root/tutorials/network_weights" \
    "$repo_root/tutorials/output"

if [[ ! -t 0 || ! -t 1 ]]; then
    echo "docker/tutorial-shell.sh requires an interactive terminal." >&2
    exit 1
fi

echo "Opening the MONAI Physio tutorial shell in $container_root"
echo "Run a lesson with: python tutorials/<tutorial_script>.py"

exec docker run --rm -it --gpus all --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    --volume "$repo_root:$container_root" \
    --volume "$cache_dir:/cache" \
    --workdir "$container_root" \
    --env HOME=/cache/home \
    --env HF_HOME=/cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env LOGNAME=monai-physio \
    --env PYTHONPATH="$container_root/src" \
    --env TOTALSEG_HOME_DIR=/cache/totalsegmentator \
    --env USER=monai-physio \
    --env MPLCONFIGDIR=/tmp/matplotlib \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env 'PS1=monai-physio:\w$ ' \
    monai-physio:tutorials bash --noprofile --norc
