#!/bin/bash

source $(dirname "$BASH_SOURCE")/dom.sh

module swap PrgEnv-cray PrgEnv-gnu
module switch gcc/10.3.0

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)
export CUDAHOSTCXX="$CXX"

export GTCMAKE_CMAKE_CXX_FLAGS='-march=haswell'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export CTEST_PARALLEL_LEVEL=1
