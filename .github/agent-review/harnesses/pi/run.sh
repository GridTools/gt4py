#!/usr/bin/env bash
# Pi adapter: a headless review, run INSIDE the sandbox by run-review.
#
# Environment: the adapter contract (harnesses/README.md); the wire is the
# OpenAI chat API on AWF's OpenAI listener, through Pi's 'openai-completions'
# API (Pi's built-in 'openai' provider would use the Responses API, which
# CSCS does not serve). Exit codes: 0 review written, 2 no final answer,
# other: failure.
#
# Pi has no turn cap and no stall detection, and exits 0 when every request
# failed: AWF's apiProxy.maxTurns and run-review's timeout bound a run.

set -uo pipefail
# shellcheck source=SCRIPTDIR/../common.sh
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
adapter_init pi
: "${AWF_API_PROXY_OPENAI_URL:?}"

# Model metadata Pi needs for context management.
readonly CONTEXT_WINDOW=128000 MAX_TOKENS=16384

export PI_CODING_AGENT_DIR="${SCRATCH}/agent" PI_OFFLINE=1 PI_TELEMETRY=0
mkdir -p "${PI_CODING_AGENT_DIR}"
pi_cli="${HARNESS_PREFIX}/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
check_version "$(head -n 1 "${HARNESS_PREFIX}/VERSION")" "${HARNESS_PREFIX}/node/bin/node" "${pi_cli}" --version || exit 1

# Pi appends '/chat/completions' to the base URL, so it carries '/v1'; the
# placeholder key is sent as a Bearer token and replaced by AWF's sidecar.
jq -n --arg id "${MODEL}" --arg base "${AWF_API_PROXY_OPENAI_URL%/}/v1" \
  --argjson ctx "${CONTEXT_WINDOW}" --argjson max "${MAX_TOKENS}" '
  {providers: {"awf-proxy": {baseUrl: $base, api: "openai-completions", apiKey: "awf-placeholder",
    models: [{id: $id, name: $id, reasoning: false, input: ["text"], contextWindow: $ctx, maxTokens: $max}]}}}' \
  >"${PI_CODING_AGENT_DIR}/models.json"

cd "${REPO_DIR}" || exit 1
# --no-approve ignores the repository's .pi/ (extensions there are executed,
# MCP servers spawned) and .agents/skills; --no-context-files also ignores
# AGENTS.md and CLAUDE.md, so the prompt names the rules to read. The prompt
# comes on stdin: with stdin open and no prompt, Pi waits forever.
"${HARNESS_PREFIX}/node/bin/node" "${pi_cli}" \
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
