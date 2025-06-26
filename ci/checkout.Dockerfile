ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /workspace/gt4py
RUN cd /gt4py && git remote add gh https://github.com/gridtools/gt4py && git fetch gh
