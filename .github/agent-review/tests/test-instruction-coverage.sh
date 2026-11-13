#!/usr/bin/env bash
# Guards invariant 5: every file 'prompt.md.tmpl' tells the agent to FOLLOW as
# rules or conventions must also be reset to its base version before the run,
# i.e. be covered by a pattern in 'instruction-files.txt'. Otherwise a PR can
# edit that file and have the agent obey the PR's version (finding F1, Phase 2:
# 'CODING_GUIDELINES.md' was named by the prompt but not reset).
#
# Offline, no network. Run: bash tests/test-instruction-coverage.sh

set -uo pipefail
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"

AR_DIR=$(cd "$(dirname "$0")/.." && pwd)
PROMPT="${AR_DIR}/prompt.md.tmpl"
LIST="${AR_DIR}/instruction-files.txt"

# The instruction files the prompt names in its "Review rules" section. Keep
# this in sync with prompt.md.tmpl; the asserts below check it two ways.
REQUIRED=(
  AGENTS.md
  CODING_GUIDELINES.md
  .github/copilot-instructions.md
  src/gt4py/next/AGENTS.md
)

# True if path $1 is reset by some pattern in instruction-files.txt, matching
# run-review's logic: a plain entry 'E' becomes ':(glob)**/E' (E at any depth,
# by trailing path segment); an entry 'D/' becomes ':(glob)**/D/**' (anything
# under a directory named D at any depth).
covered() {
  local path=$1 entry
  while IFS= read -r entry; do
    entry=${entry#"${entry%%[![:space:]]*}"} # ltrim
    entry=${entry%"${entry##*[![:space:]]}"} # rtrim
    case "${entry}" in '' | '#'*) continue ;; esac
    if [[ "${entry}" == */ ]]; then
      local dir=${entry%/}
      case "/${path}/" in */"${dir}"/*) return 0 ;; esac
    else
      [ "${path}" = "${entry}" ] && return 0
      case "${path}" in */"${entry}") return 0 ;; esac
    fi
  done <"${LIST}"
  return 1
}

test_prompt_named_files_are_reset() {
  local f rc=0
  for f in "${REQUIRED[@]}"; do
    assert "'${f}' named in prompt.md.tmpl" grep -qF "${f}" "${PROMPT}" || rc=1
    assert "'${f}' reset by instruction-files.txt" covered "${f}" || rc=1
  done
  return "${rc}"
}

# A plain source file is reviewed as data, not reset. This checks 'covered'
# does not match everything (which would make the test above vacuous).
test_plain_repo_file_is_not_falsely_required() {
  ! covered src/gt4py/next/common.py
}

run_test test_prompt_named_files_are_reset
run_test test_plain_repo_file_is_not_falsely_required
summary
