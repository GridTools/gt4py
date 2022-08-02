FROM gitpod/workspace-python

USER root
RUN apt-get update \
    && apt-get install -y libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

USER gitpod

ENV PYTHONUSERBASE=/workspace/.pip-modules
ENV PATH=$PYTHONUSERBASE/bin:$PATH
ENV PIP_USER=yes
RUN pyenv install 3.10.2
RUN pyenv global 3.10.2

COPY .gitpod/aliases.txt .
RUN cat aliases.txt >> $HOME/.bashrc
