#!/bin/bash
set -x
set -e

function submit() {
    local commit_status=${1}
    local commit_sha=${2}

    curl --verbose \
        --url "https://api.github.com/repos/gridtools/gt4py/statuses/${commit_sha}" \
        --header 'Content-Type: application/json' \
        --header "authorization: Bearer ${GITHUB_TOKEN}" \
        --data "{ \"state\": \"${commit_status}\", \"target_url\": \"${CI_PIPELINE_URL}\", \"description\": \"All Gitlab pipelines\", \"context\": \"ci/gitlab/full-pipeline\" }" | \
        jq 'error(.message) // .' # exit code 0 (success) only if returned json object does not contain a "message" key

    return $?  # propagate exit code of jq
}

commit_status=${1}

# always submit for the current commit
submit "${commit_status}" "$CI_COMMIT_SHA"

# For Bors: get the latest commit before the merge to set the status.
if [[ $CI_COMMIT_REF_NAME =~ ^(trying|staging)$ ]]; then
    parent_sha=`git rev-parse --verify -q "$CI_COMMIT_SHA"^2`
    submit "${commit_status}" "${parent_sha}"
fi
