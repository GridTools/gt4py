#!/bin/bash

source $(dirname "$BASH_SOURCE")/base.sh

source /users/fthaler/public/jenkins/spack/share/spack/setup-env.sh

spack load boost
spack load cmake

export GTRUN_BUILD_COMMAND='make -j 8'
export GTRUN_SBATCH_NODES=1
