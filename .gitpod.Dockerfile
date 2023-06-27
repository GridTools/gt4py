FROM gitpod/workspace-python
USER root
RUN apt-get update \
    && apt-get install -y libboost-dev \
    && apt-get clean && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
USER gitpod
RUN pyenv install 3.10.2
RUN pyenv global 3.10.2
