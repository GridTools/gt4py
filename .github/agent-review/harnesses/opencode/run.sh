#!/usr/bin/env bash
# opencode adapter: a headless review, run INSIDE the sandbox by run-review.
#
# Environment: the adapter contract (harnesses/README.md); the wire is the
# OpenAI chat API on AWF's OpenAI listener. Exit codes: 0 review written,
# 2 no final answer, other: failure.

set -uo pipefail
# shellcheck source=SCRIPTDIR/../common.sh
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
adapter_init opencode
: "${AWF_API_PROXY_OPENAI_URL:?}"

# Model steps per run. Below AWF's apiProxy.maxTurns, minus the session-title
# request opencode makes first, so opencode's own cap ends the run cleanly:
# at the last step tools are disabled and the model has to answer.
readonly MAX_STEPS=35

bin="${HARNESS_PREFIX}/node_modules/@opencode/cli-linux-x64-baseline/bin/opencode"

# Ignore the repository's opencode.json(c) and .opencode/ (plugins there are
# executed, MCP servers spawned) and its AGENTS.md. A nested AGENTS.md is
# still attached to 'read' results; run-review resets those.
export OPENCODE_DISABLE_PROJECT_CONFIG=1
# No model catalogue fetch (models.opencode.ai), auto-update or file watcher.
export OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_FILEWATCHER=1
check_version "opencode v$(head -n 1 "${HARNESS_PREFIX}/VERSION")" "${bin}" --version || exit 1

# Only the AWF provider may be used. opencode appends '/chat/completions' to
# the base URL, so it carries '/v1'; the placeholder key is sent as a Bearer
# token and replaced by AWF's sidecar. Tools that edit, reach the web, ask
# questions or spawn subagents are denied; 'shell' stays for git and grep.
OPENCODE_CONFIG_CONTENT=$(jq -n --arg model "${MODEL}" --arg base "${AWF_API_PROXY_OPENAI_URL%/}/v1" \
  --argjson steps "${MAX_STEPS}" '{
    model: ("awf-proxy/" + $model), update: "disable", share: "disabled",
    experimental: {policies: [
      {action: "provider.use", resource: "*", effect: "deny"},
      {action: "provider.use", resource: "awf-proxy", effect: "allow"}]},
    permissions: (["edit", "webfetch", "websearch", "question", "subagent", "execute", "opencode_*"]
      | map({action: ., resource: "*", effect: "deny"})),
    agents: {build: {steps: $steps}},
    providers: {"awf-proxy": {name: "AWF proxy", package: "@opencode/ai/providers/openai-compatible",
      settings: {baseURL: $base, apiKey: "awf-placeholder"}, models: {($model): {name: $model}}}}}')
export OPENCODE_CONFIG_CONTENT

cd "${REPO_DIR}" || exit 1
"${bin}" run --standalone --format json --agent build --model "awf-proxy/${MODEL}" \
  <"${PROMPT_FILE}" >"${OUT_DIR}/harness.jsonl" 2>"${OUT_DIR}/harness.stderr"
rc=$?

# The answer is the text after the last tool call. A run that hits the step
# cap exits 0 without such text; an 'error' event means a failed request.
if [ "${rc}" -eq 0 ]; then
  jq -rs '[.[] | select(.type == "text" or .type == "tool_use")]
    | ((map(.type) | rindex("tool_use")) as $i | if $i == null then . else .[$i + 1:] end)
    | map(select(.type == "text").part.text) | join("")' \
    "${OUT_DIR}/harness.jsonl" >"${OUT_DIR}/review.md" || rc=1
fi
if [ "${rc}" -eq 0 ] && ! jq -es 'any(.[]; .type == "error")' "${OUT_DIR}/harness.jsonl" >/dev/null &&
  has_text "${OUT_DIR}/review.md"; then
  exit 0
fi
rm -f "${OUT_DIR}/review.md"
diag_failure "${rc}" "${OUT_DIR}/harness.jsonl" type
[ "${rc}" -eq 0 ] && exit 2
exit "${rc}"
