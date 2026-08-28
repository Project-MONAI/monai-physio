#!/usr/bin/env bash
set -uo pipefail

fail=0
check() {
    if "$@" >/dev/null 2>&1; then
        printf "OK   %s\n" "$*"
    else
        printf "FAIL %s\n" "$*"
        fail=1
    fi
}

check command -v docker
check test "$(uname -m)" = "x86_64"
check docker info
check command -v nvidia-smi
check nvidia-smi
check bash -c 'docker info --format "{{json .Runtimes}}" | grep -q '"'"'"nvidia"'"'"''

available_gb=$(
    df -Pk "$(dirname "$0")/.." \
        | awk 'NR == 2 {print int($4 / 1024 / 1024)}'
)
if (( available_gb < 20 )); then
    printf "FAIL at least 20 GB free disk space (%s GB available)\n" "$available_gb"
    fail=1
else
    printf "OK   free disk space (%s GB available)\n" "$available_gb"
fi

check docker run --rm --gpus all \
    nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi -L

exit "$fail"
