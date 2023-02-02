#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

module swap PrgEnv-cray PrgEnv-gnu
module load cdt-cuda

#if [ "$build_type" != "debug" ]; then
#  module load HPX/1.5.0-CrayGNU-20.11-cuda
#fi

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)
export CUDAHOSTCXX="$CXX"

export GTCMAKE_CMAKE_CXX_FLAGS='-march=haswell'
export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export CTEST_PARALLEL_LEVEL=1
