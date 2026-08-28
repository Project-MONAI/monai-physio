ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        ninja-build \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml README.md LICENSE MANIFEST.in ./

RUN uv venv --python /usr/bin/python3 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache setuptools \
    && uv pip install --python /opt/venv/bin/python --no-cache \
        --index https://download.pytorch.org/whl/cu126 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    && uv pip install --python /opt/venv/bin/python --no-cache \
        --no-build-isolation-package torch-scatter \
        --extra cuda12 --extra physicsnemo --extra viewer \
        --requirement pyproject.toml

COPY src ./src
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps .


FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        g++ \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV HF_HOME=/cache/huggingface \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/cache/matplotlib \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYVISTA_OFF_SCREEN=true \
    TORCH_HOME=/cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    TOTALSEG_HOME_DIR=/cache/totalsegmentator \
    TRITON_CACHE_DIR=/cache/triton \
    VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow \
    XDG_CACHE_HOME=/cache/xdg

WORKDIR /workspace/monai-physio
COPY --chown=1000:1000 tutorials ./tutorials
COPY --chown=1000:1000 docker ./docker
COPY --chown=1000:1000 \
    utils/create_motion_comparison_usd.py \
    utils/lung_bundle.py \
    ./utils/

RUN chmod 755 docker/*.sh \
    && chmod 644 utils/*.py \
    && chmod -R a+rX tutorials \
    && mkdir -p data network_weights tests/baselines tutorials/network_weights \
        tutorials/output /cache \
    && chmod 1777 tests/baselines \
    && chown -R 1000:1000 /workspace/monai-physio /cache

USER 1000:1000
CMD ["bash"]
