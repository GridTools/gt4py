#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint.sh

export CXX=$(which CC)
export CC=$(which cc)
export FC=$(which ftn)
export CUDAHOSTCXX="$CXX"

export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'

export CTEST_PARALLEL_LEVEL=1
