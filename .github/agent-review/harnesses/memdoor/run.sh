#!/usr/bin/env bash
# memdoor adapter: a headless review, run INSIDE the sandbox by run-review.
#
# Environment: the adapter contract (harnesses/README.md); the wire is the
# OpenAI chat API on AWF's OpenAI listener. Exit codes: 0 review written,
# 2 no final answer, other: failure.
#
# memdoor has no turn cap, only a detector for an identical call repeated
# five times: AWF's apiProxy.maxTurns and run-review's timeout bound a run.

set -uo pipefail
# shellcheck source=SCRIPTDIR/../common.sh
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
adapter_init memdoor
: "${AWF_API_PROXY_OPENAI_URL:?}"

bin="${HARNESS_PREFIX}/bin/memdoor"
tag=$(head -n 1 "${HARNESS_PREFIX}/VERSION")
check_version "memdoor ${tag}?( *)" "${bin}" --version || exit 1

# A fresh stamp skips the daily version check against memdoor.ai.
mkdir -p "${HOME}/.memdoor"
printf '{"checked":"%s","remote":""}' "$(date -Iseconds)" >"${HOME}/.memdoor/update-check.json"
chmod 600 "${HOME}/.memdoor/update-check.json"

# "Company AI gateway" mode contacts nothing but the gateway; 'chat' selects
# '<base>/chat/completions' (the default is the Responses API), so the base
# carries '/v1'. memdoor also probes 'GET /v1/models'. The placeholder token
# is replaced by AWF's sidecar. Vendor keys would select another mode.
unset OPENAI_API_KEY OPENAI_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN \
  OPENROUTER_API_KEY GEMINI_API_KEY
export AI_GATEWAY_BASE_URL="${AWF_API_PROXY_OPENAI_URL%/}/v1" AI_GATEWAY_TOKEN=awf-placeholder \
  AI_GATEWAY_API=chat MEMDOOR_MODEL="${MODEL}" MEMDOOR_HOST=localhost MEMDOOR_TELEMETRY_ENABLED=0

# memdoor runs agents through a local gateway daemon; 'run' would otherwise
# start one on a fixed port that outlives it.
port=$(shuf -i 20000-29999 -n 1)
gateway="http://127.0.0.1:${port}"
"${bin}" gateway --host localhost --port "${port}" >"${OUT_DIR}/gateway.log" 2>&1 &
gateway_pid=$!
trap 'kill "${gateway_pid}" 2>/dev/null; sleep 0.3; kill -9 "${gateway_pid}" 2>/dev/null' EXIT
for _ in $(seq 1 120); do
  (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null && break
  sleep 0.25
done

# A scratch admin account (random password, never leaves this HOME) is
# needed to add guards. Guards refuse a tool call and tell the model why:
# no edits, web access, subagents, scheduling or memory; no 'skill' (it loads
# the repository's .agents/skills); no 'ask_user_question' (unanswered, it
# ends the run with exit 3).
setup() {
  "${bin}" --gateway "${gateway}" setup --non-interactive --admin-email ci@localhost \
    --admin-password "$(head -c 18 /dev/urandom | base64 | tr -d '/+=')x1" || return 1
  local tool
  for tool in apply_patch write_file edit_file web_fetch web_search sessions_spawn cron workflow memory \
    skill ask_user_question; do
    "${bin}" --gateway "${gateway}" guards add --tool "${tool}" --pattern '.*' \
      --message "Not available in this read-only review." || return 1
  done
}
if ! setup >"${OUT_DIR}/setup.log" 2>&1; then
  echo "setup failed" | diag
  tail -n 20 "${OUT_DIR}/setup.log" "${OUT_DIR}/gateway.log" | diag
  exit 1
fi

cd "${REPO_DIR}" || exit 1
"${bin}" run --gateway "${gateway}" --dir "${REPO_DIR}" --json \
  <"${PROMPT_FILE}" >"${OUT_DIR}/harness.jsonl" 2>"${OUT_DIR}/harness.stderr"
rc=$?

# 0: checked, 2: "unverified" (normal for a review), 3: an unanswered
# question (guarded, tolerated: the answer check decides); 1: failure.
case "${rc}" in
  0 | 2 | 3)
    # The answer is the text after the last tool call, without the trailers
    # memdoor appends as separate text events: "Recap: ..." (what ran) and
    # "(note: this turn used no tools ...)".
    jq -rs 'def trim: if length > 0 and (.[-1] | test("^\\s*(Recap: |\\(note: this turn used no tools)"))
        then .[:-1] | trim else . end;
      (map(.event) | rindex("tool_done")) as $i | (if $i == null then . else .[$i + 1:] end)
      | map(select(.event == "text").text) | trim | join("")' \
      "${OUT_DIR}/harness.jsonl" >"${OUT_DIR}/review.md" || rc=1
    ;;
esac
# The stall detector ends a run with exit 0 and a synthetic "Stopped: ..."
# text, which is not a review.
case "${rc}" in
  0 | 2 | 3)
    if has_text "${OUT_DIR}/review.md" &&
      ! grep -q '^Stopped: the same request kept repeating' "${OUT_DIR}/review.md"; then
      exit 0
    fi
    rc=0
    ;;
esac
rm -f "${OUT_DIR}/review.md"
diag_failure "${rc}" "${OUT_DIR}/harness.jsonl" event
[ "${rc}" -eq 0 ] && exit 2
exit "${rc}"
