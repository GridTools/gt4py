#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

export EASYBUILD_PREFIX=/apps/tsa/SSL/gridtools/jenkins/easybuild
module use $EASYBUILD_PREFIX/modules/all
module load cmake

module load slurm

export BOOST_ROOT=/apps/tsa/SSL/gridtools/jenkins/boost_1_77_0
export CUDA_ARCH=sm_70

export GTRUN_BUILD_COMMAND='make -j 8'
export GTRUN_SBATCH_PARTITION='debug'
export GTRUN_SBATCH_NODES=1
export GTRUN_SBATCH_NTASKS_PER_CORE=1
export GTRUN_SBATCH_NTASKS_PER_NODE=1
export GTRUN_SBATCH_GRES='gpu:1'
export GTRUNMPI_SBATCH_PARTITION='debug'
export GTRUNMPI_SBATCH_NODES=1
export GTRUNMPI_SBATCH_NTASKS_PER_NODE=4
export GTRUNMPI_SBATCH_GRES='gpu:4'

export CUDA_AUTO_BOOST=0
export GCLOCK=1530
export OMP_NUM_THREADS=16

export UCX_MEMTYPE_CACHE=n
export UCX_TLS=rc_x,ud_x,mm,shm,cuda_copy,cuda_ipc,cma

