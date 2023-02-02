#!/bin/bash

source $(dirname "$0")/setup.sh

# build binaries for performance tests
./pyutils/driver.py -v -l $logfile build -b release -o build -e $envfile -t perftests || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }

for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=$GT_PERFORMANCE_HISTORY_PATH_PREFIX/$env/$domain
  mkdir -p $resultdir

  # name result file by date/time
  result="$resultdir/$(date +%F-%H-%M-%S).json"

  # run performance tests
  ./build/pyutils/driver.py -v -l $logfile perftest run -s $domain $domain 80 -o $result || { echo 'Running failed'; rm -rf $tmpdir; exit 1; }

  # create directory for reports
  mkdir reports
  # find previous results for history plot
  results=$(find $resultdir -name '*.json')
  if [[ -n "$results" ]]; then
    # plot history
    ./build/pyutils/driver.py -v -l $logfile perftest plot history -i $results -o reports/full-history-$domain || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
    ./build/pyutils/driver.py -v -l $logfile perftest plot history -i $results -o reports/last-history-$domain --limit=10 || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
  fi

  # plot backend comparison
  ./build/pyutils/driver.py -v -l $logfile perftest plot compare-backends -i $result -o reports/backend-comparison-$domain || { echo 'Plotting failed'; rm -rf $tmpdir; exit 1; }
done
