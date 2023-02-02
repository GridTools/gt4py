#!/bin/bash

source $(dirname "$BASH_SOURCE")/daint_cray.sh

# only supported configuration on daint for NVCC-CUDA mode
module switch cudatoolkit/11.2.0_3.39-2.1__gf93aa1c
module switch cce/10.0.2

export GTCMAKE_GT_CLANG_CUDA_MODE=NVCC-CUDA

export CTEST_PARALLEL_LEVEL=1

