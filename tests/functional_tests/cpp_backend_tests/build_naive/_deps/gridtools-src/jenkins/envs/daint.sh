#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

# the module command on Daint does not properly return an error code, so we
# override the function to catch failing module invocations
eval "$(declare -f module | sed 's/^module () *$/original_\0/')"
function module() {
    local tmpout=$(mktemp)
    original_module $* 2> "$tmpout"
    local output=$(cat "$tmpout" && rm "$tmpout")
    if [[ "$output" =~ "ERROR" ]]; then
        >&2 echo "'module $*' exited with error: $output"
        return 1
    fi
}

module load daint-gpu
module load cudatoolkit

export EASYBUILD_PREFIX=/apps/daint/SSL/gridtools/jenkins/easybuild
module use $EASYBUILD_PREFIX/modules/all
module load CMake

export BOOST_ROOT=/apps/daint/SSL/gridtools/jenkins/boost_1_77_0
export CUDATOOLKIT_HOME=$CUDA_PATH
export CUDA_ARCH=sm_60

export GTRUN_BUILD_COMMAND='srun -C gpu -p cscsci --time=00:20:00 make -j 24'
export GTRUN_SBATCH_ACCOUNT='d75'
export GTRUN_SBATCH_PARTITION='cscsci'
export GTRUN_SBATCH_NODES=1
export GTRUN_SBATCH_NTASKS_PER_CORE=2
export GTRUN_SBATCH_NTASKS_PER_NODE=1
export GTRUN_SBATCH_CPUS_PER_TASK=24
export GTRUN_SBATCH_CONSTRAINT='gpu'
export GTRUN_SBATCH_CPU_FREQ='high'
export GTRUNMPI_SBATCH_PARTITION='normal'
export GTRUNMPI_SBATCH_NODES=4

export CUDA_AUTO_BOOST=0
export GCLOCK=1328
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_G2G_PIPELINE=30
export OMP_NUM_THREADS=24
export OMP_PLACES='{0}:24'
