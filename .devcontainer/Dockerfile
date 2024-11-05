FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive && apt-get install -y libboost-dev \
    && apt-get clean && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
