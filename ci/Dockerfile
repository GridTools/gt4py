ARG BASE_IMAGE=docker.io/ubuntu:24.04
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG WORKDIR_PATH=/workspace
ENV WORKDIR=${WORKDIR_PATH}
WORKDIR ${WORKDIR}

ARG DEBIAN_FRONTEND=noninteractive
ARG EXTRA_APTGET=""
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    build-essential \
    tar \
    wget \
    curl \
    ca-certificates \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    libffi-dev \
    liblzma-dev \
    libreadline-dev \
    git \
    ${EXTRA_APTGET} && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ARG UV_VERSION=0.6.12
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN UV_UNMANAGED_INSTALL="/opt/uv" sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/opt/uv:$PATH"

# Set up the environment vars with paths for uv
ARG CACHE_DIR=${WORKDIR_PATH}/cache/
ENV UV_CACHE_DIR="${CACHE_DIR}/uv"
ENV UV_PROJECT_ENVIRONMENT=${WORKDIR_PATH}/venv
ENV UV_PYTHON_INSTALL_DIR="/opt/uv-python"

# Create and activate the python venv
ARG PY_VERSION=3.11
ENV VIRTUAL_ENV=${UV_PROJECT_ENVIRONMENT}
RUN uv venv -p ${PY_VERSION} --link-mode copy ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Install python packages using uv just to fill the uv cache,
# since nox will create other test session venvs anyway
ARG EXTRA_UV_ENV_VARS="" 
ARG EXTRA_UV_PIP_ARGS=""
ARG EXTRA_UV_SYNC_ARGS=""
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml <<EOF
set -eu
if [ ! -z "${EXTRA_UV_ENV_VARS}" ]; then
    eval "export ${EXTRA_UV_ENV_VARS}"
fi
uv pip install --compile-bytecode setuptools wheel pip
uv sync --compile-bytecode --frozen --no-install-project \
    --group dev \
    --extra standard \
    --extra jax \
    --extra testing \
    ${EXTRA_UV_SYNC_ARGS}
if [ ! -z "${EXTRA_UV_PIP_ARGS}" ]; then
    uv pip install --compile-bytecode ${EXTRA_UV_PIP_ARGS}
fi
EOF
