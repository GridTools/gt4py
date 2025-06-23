ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /gt4py.src
RUN cd /gt4py.src && git remote add gh https://github.com/gridtools/gt4py && git fetch gh
