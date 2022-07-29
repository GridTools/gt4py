FROM gitpod/workspace-python

USER root
RUN apt-get update \
    && apt-get install -y libboost-all-dev \
    && apt-get clean && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

USER gitpod
ENV PYTHONUSERBASE=/workspace/.pip-modules
ENV PATH=$PYTHONUSERBASE/bin:$PATH
ENV PIP_USER=yes

COPY .gitpod/aliases.txt .
RUN cat aliases.txt >> $HOME/.bashrc
