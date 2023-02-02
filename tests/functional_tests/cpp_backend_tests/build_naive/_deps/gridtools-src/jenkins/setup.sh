#!/bin/bash

echo "Running on host $(hostname)"

# remove -cn from label (for daint)
label=${label%%-*}

envfile=./jenkins/envs/${label}_$env.sh

export GT_PERFORMANCE_HISTORY_PATH_PREFIX=/apps/${label}/SSL/gridtools/jenkins/gridtools-performance-history-gt2

# use the machines python virtualenv with required modules installed
if [[ $label = ault ]]; then
    venv_dir=/users/fthaler/public/jenkins/gridtools-venv
elif [[ $label = dom ]]; then
    venv_dir=/apps/daint/SSL/gridtools/jenkins/venv
else
    venv_dir=/apps/$label/SSL/gridtools/jenkins/venv
fi
source $venv_dir/bin/activate

if [[ $label != "tsa" ]]; then
    export SLURM_ACCOUNT=d75
    export SBATCH_ACCOUNT=d75
fi

if [[ ! -v build_examples ]] && [[ $env =~ ^(hip)$ ]]; then
    build_examples=false
fi

logdir=/var/tmp/gridtools_${USER}/ # log to subfolder to workaround https://webrt.cscs.ch/Ticket/Display.html?id=38406
mkdir -p $logdir
chmod +rwx $logdir
# possibly delete old log files and create new log file
find $logdir -maxdepth 1 -mtime +5 -name 'gridtools-jenkins-*.log' -execdir rm -f {} + 2>/dev/null
logfile=$(mktemp -p $logdir gridtools-jenkins-XXXXX.log)
chmod +rw $logfile

# create directory for temporaries
if [[ $label == "tave" ]]; then
    # use /dev/shm on Tave due to small /tmp size
    tmpdir=$(mktemp -d /dev/shm/gridtools-tmp-XXXXXXXXXX)
else
    # use a subdirectory of /tmp on other systems to avoid memory problems
    tmpdir=$(mktemp -d /tmp/gridtools-tmp-XXXXXXXXXX)
fi
mkdir -p $tmpdir
export TMPDIR=$tmpdir

# register cleanup function
cleanup() {
    # clean possible temporary leftovers
    rm -rf $tmpdir
}

trap cleanup EXIT
