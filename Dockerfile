FROM nvidia/cuda:10.1-devel
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends \
        strace \
        build-essential \
        cuda-toolkit-10.1 \
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
        python-openssl \
        libreadline-dev \
        git \
        htop && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz && \
    echo c66e88d5786f2ca4dbebb14e06b566fb642a1a6947ad8cc9091f9f445134143f boost_1_72_0.tar.gz > boost_hash.txt && \
    sha256sum -c boost_hash.txt && \
    tar xzf boost_1_72_0.tar.gz && \
    mv boost_1_72_0/boost /usr/local/include/ && \
    rm boost_1_72_0.tar.gz boost_hash.txt

ENV BOOST_ROOT /usr/local/
ENV CUDA_HOME /usr/local/cuda

ARG PYVERSION

RUN curl https://pyenv.run | bash

ENV PYENV_ROOT /root/.pyenv
ENV PATH="/root/.pyenv/bin:${PATH}"

RUN pyenv update && \
    pyenv install ${PYVERSION} && \
    echo 'eval "$(pyenv init -)"' >> /root/.bashrc && \
    eval "$(pyenv init -)" && \
    pyenv global ${PYVERSION}

ENV PATH="/root/.pyenv/shims:${PATH}"


RUN pip install --upgrade pip setuptools wheel tox cupy-cuda101

