#!/bin/bash

function submit() {
    local commit_status=${1}
    local commit_sha=${2}

    curl --verbose \
        --url "https://api.github.com/repos/gridtools/gt4py/statuses/${commit_sha}" \
        --header 'Content-Type: application/json' \
        --header "authorization: Bearer ${GITHUB_TOKEN}" \
        --data "{ \"state\": \"${commit_status}\", \"target_url\": \"${CI_PIPELINE_URL}\", \"description\": \"All Gitlab pipelines\", \"context\": \"ci/gitlab/full-pipeline\" }" | \
        jq -e 'error(.message) // .'
    exit $?
}

commit_status=${1}

# always submit for the current commit
submit "${commit_status}" "$CI_COMMIT_SHA"

# For Bors: get the latest commit before the merge to set the status.
if [[ $CI_COMMIT_REF_NAME =~ ^(trying|staging)$ ]]; then
    parent_sha=`git rev-parse --verify -q "$CI_COMMIT_SHA"^2`
    if [[ $? -eq 0 ]]; then
        submit "${commit_status}" "${parent_sha}"
    fi
fi

exit 0

