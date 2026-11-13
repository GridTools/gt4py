#!/usr/bin/env bash
# Pi adapter: a headless review, run INSIDE the sandbox by run-review.
#
# Environment: the adapter contract (harnesses/README.md). The wire is read
# from the 'wire' file next to this script, which is also what selects the
# AWF listener that holds the key. Exit codes: 0 review written, 2 no final
# answer, other: failure.
#
# Pi has no turn cap and no stall detection, and exits 0 when every request
# failed: AWF's apiProxy.maxTurns and run-review's timeout bound a run.

set -uo pipefail
here=$(dirname "${BASH_SOURCE[0]}")
# shellcheck source=SCRIPTDIR/../common.sh
. "${here}/../common.sh"
adapter_init pi

# Model metadata Pi needs for context management.
readonly CONTEXT_WINDOW=128000 MAX_TOKENS=16384

export PI_CODING_AGENT_DIR="${SCRATCH}/agent" PI_OFFLINE=1 PI_TELEMETRY=0
mkdir -p "${PI_CODING_AGENT_DIR}"

wire=$(head -n 1 "${here}/wire")
case "${wire}" in
  anthropic-messages)
    # Not used: the workflow supports only 'openai-chat'. Pi's Anthropic API
    # would go to the OpenAI listener, which forwards '/v1/messages' (Pi
    # appends '/v1/messages?beta=true' to the origin).
    api=anthropic-messages
    base=${AWF_API_PROXY_OPENAI_URL:?}
    ;;
  openai-chat)
    # The OpenAI SDK appends '/chat/completions', so the base carries '/v1'.
    api=openai-completions
    base=${AWF_API_PROXY_OPENAI_URL:?}/v1
    ;;
  *) echo "unknown wire '${wire}'" | diag; exit 1 ;;
esac
jq -n --arg id "${MODEL}" --arg base "${base%/}" --arg api "${api}" \
  --argjson ctx "${CONTEXT_WINDOW}" --argjson max "${MAX_TOKENS}" '
  {providers: {"awf-proxy": {baseUrl: $base, api: $api, apiKey: "awf-placeholder",
    models: [{id: $id, name: $id, reasoning: false, input: ["text"], contextWindow: $ctx, maxTokens: $max}]}}}' \
  >"${PI_CODING_AGENT_DIR}/models.json"

cd "${REPO_DIR}" || exit 1
# --no-approve ignores the repository's .pi/ (extensions there are executed,
# MCP servers spawned) and .agents/skills; --no-context-files also ignores
# AGENTS.md and CLAUDE.md, so the prompt names the rules to read. The prompt
# comes on stdin: with stdin open and no prompt, Pi waits forever.
"${HARNESS_PREFIX}/node/bin/node" "${HARNESS_PREFIX}/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js" \
  --print --mode json --no-session --no-approve --no-context-files --tools read,grep,find,ls,bash \
  --model "awf-proxy/${MODEL}" <"${PROMPT_FILE}" >"${OUT_DIR}/harness.jsonl" 2>"${OUT_DIR}/harness.stderr"
rc=$?

# The answer is the text of the last turn, if that turn ended normally
# ('stop'); failed requests end it with 'error'.
if [ "${rc}" -eq 0 ]; then
  jq -rs '[.[] | select(.type == "turn_end") | .message] | last | select(.stopReason == "stop")
    | [.content[] | select(.type == "text").text] | join("")' \
    "${OUT_DIR}/harness.jsonl" >"${OUT_DIR}/review.md" || rc=1
fi
if [ "${rc}" -eq 0 ] && has_text "${OUT_DIR}/review.md"; then
  exit 0
fi
rm -f "${OUT_DIR}/review.md"
diag_failure "${rc}" "${OUT_DIR}/harness.jsonl" type
# Pi reports failed requests only in its events.
jq -rs '[.[] | select(.type == "turn_end") | .message | select(.stopReason != "toolUse")
  | "stopReason \(.stopReason): \(.errorMessage // "")"] | .[-3:][]' "${OUT_DIR}/harness.jsonl" 2>/dev/null | diag
[ "${rc}" -eq 0 ] && exit 2
exit "${rc}"
