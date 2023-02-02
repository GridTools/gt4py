#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

module load daint-gpu
module load cudatoolkit/11.2.0_3.39-2.1__gf93aa1c
module load CMake

export BOOST_ROOT=/apps/daint/SSL/gridtools/jenkins/boost_1_77_0
export CUDATOOLKIT_HOME=$CUDA_PATH
export CUDA_ARCH=sm_60

export GTRUN_BUILD_COMMAND='srun --account d75 -C gpu --time=00:20:00 make -j 24'
export GTRUN_SBATCH_ACCOUNT='d75'
export GTRUN_SBATCH_PARTITION='normal'
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
