#!/bin/bash

source $(dirname "$BASH_SOURCE")/tsa.sh

module use /apps/common/UES/sandbox/kraushm/tsa-nvhpc/easybuild/modules/all
module load openmpi

export CXX=$(which g++)
export CC=$(which gcc)
export FC=$(which pgfortran)
export CUDAHOSTCXX="$CXX"

export GTCMAKE_CMAKE_CXX_FLAGS_RELEASE='-Ofast -DNDEBUG'
export GTCMAKE_GT_REQUIRE_OpenMP="ON"
export GTCMAKE_GT_REQUIRE_GPU="ON"

export CTEST_PARALLEL_LEVEL=1
