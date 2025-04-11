ARG ROCM_VERSION=6.2.4
ARG UBUNTU_VERSION=24.04
FROM docker.io/rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
    strace \
    build-essential \
    tar \
    wget \
    curl \
    ca-certificates \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    $( [ "${UBUNTU_VERSION}" = "20.04" ] && echo "python-openssl" || echo "python3-openssl" ) \
    libreadline-dev \
    git \
    rustc \
    htop && \
    rm -rf /var/lib/apt/lists/*

ENV ROCM_HOME=/opt/rocm

# Install uv
ARG UV_VERSION=0.6.12
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install basic python environment
ARG PY_VERSION
ARG COMPILE_BYTECODE=0
ENV UV_COMPILE_BYTECODE=${COMPILE_BYTECODE}
ENV UV_LINK_MODE=copy

RUN uv venv -p ${PY_VERSION} /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install setuptools wheel pip jax[rocm]

# Install dependencies

# CuPy from source
ARG CUPY_VERSION=13.4.1
ENV CUPY_INSTALL_USE_HIP=1
RUN uv pip install cupy==${CUPY_VERSION}

