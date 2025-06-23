ARG BASE_IMAGE=ubuntu:24.04
FROM $BASE_IMAGE

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
    libreadline-dev \
    git \
    rustc \
    htop && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ARG UV_VERSION=0.6.12
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Create main python venv for development only,
# since nox creates a different venv per session
ARG PY_VERSION=3.10
ARG COMPILE_BYTECODE=1
ENV UV_COMPILE_BYTECODE=${COMPILE_BYTECODE}
ENV UV_LINK_MODE=copy

ENV VIRTUAL_ENV=/gt4py.venv
RUN uv venv -p ${PY_VERSION} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Install core dependencies to fill the uv cache
RUN uv pip install setuptools wheel pip
ARG EXTRA_VENV_PACKAGES="clang-format jaxlib numpy pytest scipy"
RUN uv pip install ${EXTRA_VENV_PACKAGES} 
