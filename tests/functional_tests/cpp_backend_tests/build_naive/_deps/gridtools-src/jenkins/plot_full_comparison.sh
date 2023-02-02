#!/bin/bash

source $(dirname "$0")/setup.sh

# create directory for reports
mkdir reports
for domain in 128 256; do
    for label in daint; do
        resultdirs=$(ls -d $GT_PERFORMANCE_HISTORY_PATH_PREFIX/*/$domain)
        results=''
        for dir in $resultdirs; do
            results="$results $(ls -t $dir/*.json 2> /dev/null | head -n 1)"
        done

        if [[ -n "$results" ]]; then
            ./pyutils/driver.py -v perftest plot compare-backends -i $results -o reports/report-$label-$domain
        fi
    done
done
