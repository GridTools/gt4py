#!/bin/bash

source $(dirname "$0")/setup.sh

# build binaries for performance tests
./pyutils/driver.py -v -l $logfile build -b release -o build -e $envfile -t perftests || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=./build/pyutils/perftest/results/${label}_$env
  mkdir -p $resultdir
  result=$resultdir/$domain.json

  # run performance tests
  ./build/pyutils/driver.py -v -l $logfile perftest run -s $domain $domain 80 -o $result || { echo 'Running failed'; rm -rf $tmpdir; exit 1; }

  # create directory for reports
  mkdir -p reports
  # find references for same configuration
  reference=./pyutils/perftest/references/${label}_$env/$domain.json
  if [[ -f $reference ]]; then
    # plot comparison of current result with references
    ./build/pyutils/driver.py -v -l $logfile perftest plot compare -i $reference $result -o reports/reference-comparison-$domain || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
  else
    echo "WARNING: no reference found for config ${label}_$env, domain size $domain" | tee $logfile
  fi
  # plot comparison between backends
  ./build/pyutils/driver.py -v -l $logfile perftest plot compare-backends -i $result -o reports/backends-comparison-$domain || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
done
