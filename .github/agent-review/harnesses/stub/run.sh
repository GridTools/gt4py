#!/usr/bin/env bash
# 'stub' harness: makes NO model calls and writes a fixed review.
#
# It exists to test the pipeline (prepare -> AWF -> collect -> publish)
# without any secret. The review deliberately contains hostile content that
# the 'publish' job must neutralise: @mentions (also as character
# references), HTML comments imitating the
# comment marker, and optional padding (STUB_PADDING_KIB) to exceed the size
# limits.
#
# Environment (the adapter contract, see harnesses/README.md): PROMPT_FILE,
# REPO_DIR, OUT_DIR, MODEL, HARNESS_PREFIX, PR_NUMBER, CONTEXT_DIR, and here
# also STUB_PADDING_KIB.

set -euo pipefail

: "${PROMPT_FILE:?}" "${REPO_DIR:?}" "${OUT_DIR:?}" "${CONTEXT_DIR:?}" "${PR_NUMBER:?}"

files=$(wc -l <"${CONTEXT_DIR}/changed-files.txt")
diff_bytes=$(wc -c <"${CONTEXT_DIR}/diff.patch")
prompt_bytes=$(wc -c <"${PROMPT_FILE}")
head_sha=$(git -c core.hooksPath=/dev/null -C "${REPO_DIR}" rev-parse HEAD)

# Names only, never values: confirms in the log that no credential-like
# variable other than AWF's placeholders reaches the sandbox.
echo "stub: credential-like variables in the sandbox environment:"
env | cut -d= -f1 | grep -E 'KEY|TOKEN|SECRET|PASSWORD' | sort | sed 's/^/stub:   /' || true

{
  echo "### Summary"
  echo
  echo "Stub review of PR #${PR_NUMBER} at \`${head_sha}\`: ${files} changed files," \
    "a ${diff_bytes}-byte diff and a ${prompt_bytes}-byte prompt. No model was called."
  echo
  echo "### Findings"
  echo
  echo "1. **Mention neutralisation**: these mentions must not notify anyone: @egparedes," \
    "&#64;egparedes, &#x40;egparedes, &commat;egparedes."
  echo "2. **Marker spoofing**: the next lines must not hide text or imitate the comment marker."
  echo "   <!-- agent-review:stub -->"
  echo "   <!<!---- nested comment start -->"
  echo "   hidden text? -->"
  echo
  if [ "${STUB_PADDING_KIB:-0}" -gt 0 ]; then
    echo "### Padding"
    echo
    # 64 characters per line, 16 lines per KiB.
    for _ in $(seq $((STUB_PADDING_KIB * 16))); do
      echo "padding padding padding padding padding padding padding padding"
    done
  fi
} >"${OUT_DIR}/review.md"
