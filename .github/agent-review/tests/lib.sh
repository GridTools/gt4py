# Minimal assertion helpers for the agent-review shell tests.
# shellcheck shell=bash

TESTS_RUN=0
TESTS_FAILED=0
# shellcheck disable=SC2034 # used by the test scripts
BIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../bin" && pwd)

fail() {
  echo "  FAIL: $*"
  TESTS_FAILED=$((TESTS_FAILED + 1))
}

# Runs one test function in a fresh temporary directory, in a subshell.
run_test() {
  local name=$1 tmp rc=0
  tmp=$(mktemp -d)
  TESTS_RUN=$((TESTS_RUN + 1))
  echo "- ${name}"
  (cd "${tmp}" && TMP="${tmp}" "${name}") || rc=$?
  [ "${rc}" -eq 0 ] || fail "${name} (exit ${rc})"
  rm -rf "${tmp}"
}

assert() { # message command...
  local msg=$1
  shift
  "$@" || {
    echo "    assertion failed: ${msg}"
    return 1
  }
}

summary() {
  echo "${TESTS_RUN} tests, ${TESTS_FAILED} failed."
  [ "${TESTS_FAILED}" -eq 0 ]
}
