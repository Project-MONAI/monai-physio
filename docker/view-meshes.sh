#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--port PORT] [--fps FPS] INPUT_FILE [INPUT_FILE ...]" >&2
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

repo_root=$(cd "$(dirname "$0")/.." && pwd)
port=8080
fps=""
input_paths=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            port=$2
            shift 2
            ;;
        --fps)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            fps=$2
            shift 2
            ;;
        --)
            shift
            input_paths+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            input_paths+=("$1")
            shift
            ;;
    esac
done

if [[ ${#input_paths[@]} -eq 0 ]]; then
    usage
    exit 2
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
    echo "PORT must be an integer between 1 and 65535" >&2
    exit 2
fi

cache_dir="$repo_root/.cache/monai-physio"
container_root="/workspace/monai-physio"
container_source="$container_root/source"
container_inputs=()
volume_args=()
source_args=()
if [[ -f "$repo_root/src/monai_physio/cli/view_meshes.py" ]]; then
    source_args+=(--volume "$repo_root/src:$container_source/src:ro")
    source_args+=(--env "PYTHONPATH=$container_source/src")
fi
for input_path in "${input_paths[@]}"; do
    if [[ ! -f "$input_path" ]]; then
        echo "Viewer input file not found: $input_path" >&2
        exit 2
    fi
    resolved_file=$(realpath "$input_path")
    input_dir=$(dirname "$resolved_file")
    input_index=${#container_inputs[@]}
    container_dir="$container_root/preview/$input_index"
    container_inputs+=("$container_dir/$(basename "$resolved_file")")
    volume_args+=(--volume "$input_dir:$container_dir:ro")
done
mkdir -p "$cache_dir/home"

echo "Viewer: http://127.0.0.1:$port"
viewer_args=(
    "${container_inputs[@]}"
    --host 0.0.0.0
    --port "$port"
    --no-browser
)
if [[ -n "$fps" ]]; then
    viewer_args+=(--fps "$fps")
    echo "Playback override: $fps FPS"
fi
exec docker run --rm --gpus all --shm-size=2g \
    --user "$(id -u):$(id -g)" \
    --publish "127.0.0.1:$port:$port" \
    "${volume_args[@]}" \
    "${source_args[@]}" \
    --volume "$cache_dir:/cache" \
    --env HOME=/cache/home \
    --env LOGNAME=monai-physio \
    --env USER=monai-physio \
    monai-physio:tutorials \
    python -m monai_physio.cli.view_meshes "${viewer_args[@]}"
