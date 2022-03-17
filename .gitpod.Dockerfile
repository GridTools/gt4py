FROM gitpod/workspace-python
ENV PYTHONUSERBASE=/workspace/.pip-modules
ENV PATH=$PYTHONUSERBASE/bin:$PATH
ENV PIP_USER=yes

COPY .gitpod/aliases.txt .
RUN cat aliases.txt >> $HOME/.bashrc
