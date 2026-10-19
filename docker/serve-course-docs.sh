#!/usr/bin/env bash
# Start the persistent static server used by the Brev course documentation.

set -Eeuo pipefail

docs_root="${MONAI_PHYSIO_COURSE_DOCS_DIR:-${HOME}/monai-physio-course-docs}"
port="${MONAI_PHYSIO_COURSE_DOCS_PORT:-8000}"
image="${MONAI_PHYSIO_IMAGE:-monai-physio:tutorials}"
container_name="monai-physio-course-docs"
docker_command=(docker)

if [[ "${MONAI_PHYSIO_DOCKER_USE_SUDO:-false}" == "true" ]]; then
    docker_command=(sudo docker)
fi

if [[ ! "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
    echo "MONAI_PHYSIO_COURSE_DOCS_PORT must be between 1 and 65535" >&2
    exit 2
fi

mkdir -p "${docs_root}/releases/waiting"
if [[ -e "${docs_root}/current" && ! -L "${docs_root}/current" ]]; then
    echo "${docs_root}/current must be a symbolic link" >&2
    exit 1
fi
if [[ ! -e "${docs_root}/current" ]]; then
    cat >"${docs_root}/releases/waiting/index.html" <<'EOF'
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>MONAI Physio Course</title></head>
<body><h1>MONAI Physio Course</h1><p>The course is not published yet.</p></body>
</html>
EOF
    ln -sfn "releases/waiting" "${docs_root}/current"
fi

if "${docker_command[@]}" container inspect "${container_name}" \
    >/dev/null 2>&1; then
    "${docker_command[@]}" container rm --force "${container_name}" >/dev/null
fi

"${docker_command[@]}" run --detach \
    --name "${container_name}" \
    --restart unless-stopped \
    --publish "0.0.0.0:${port}:${port}" \
    --volume "${docs_root}:/course-docs:ro" \
    --entrypoint python \
    "${image}" \
    -m http.server "${port}" \
    --bind 0.0.0.0 \
    --directory /course-docs/current \
    >/dev/null

echo "Course documentation server: http://127.0.0.1:${port}/"
