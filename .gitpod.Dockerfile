FROM gitpod/workspace-python-3.12
USER root
RUN apt-get update \
    && apt-get install -y libboost-dev \
    && apt-get clean && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
USER gitpod
