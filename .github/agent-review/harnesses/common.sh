# Helpers shared by the harness adapters' run.sh; sourced, not executed.
# shellcheck shell=bash
#
# Runs INSIDE the sandbox. Everything a harness prints may be steered by the
# pull request, so diagnostics go through 'diag', which prefixes every line:
# a line starting with '::' would otherwise be a GitHub workflow command in
# the step log.

# Checks the contract's environment and creates a scratch HOME and XDG
# directories under /tmp, so no configuration is read from (or written to) the
# runner's home. Sets SCRATCH.
adapter_init() { # name
  : "${PROMPT_FILE:?}" "${REPO_DIR:?}" "${OUT_DIR:?}" "${MODEL:?}" "${HARNESS_PREFIX:?}"
  ADAPTER_NAME=$1
  SCRATCH=$(mktemp -d "/tmp/agent-review-${ADAPTER_NAME}.XXXXXX")
  export HOME="${SCRATCH}/home" XDG_CONFIG_HOME="${SCRATCH}/config" XDG_DATA_HOME="${SCRATCH}/data" \
    XDG_CACHE_HOME="${SCRATCH}/cache" XDG_STATE_HOME="${SCRATCH}/state" NO_COLOR=1
  mkdir -p "${HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${XDG_CACHE_HOME}" "${XDG_STATE_HOME}" "${OUT_DIR}"
  # Loopback (memdoor's local gateway) and the AWF listeners must never go
  # through an HTTP proxy, should the sandbox set one.
  local hosts="127.0.0.1,localhost,::1" url host
  for url in "${AWF_API_PROXY_OPENAI_URL:-}" "${AWF_API_PROXY_ANTHROPIC_URL:-}"; do
    host=${url#*://}
    host=${host%%[:/]*}
    [ -z "${host}" ] || hosts="${hosts},${host}"
  done
  export NO_PROXY="${hosts}${NO_PROXY:+,${NO_PROXY}}"
  export no_proxy="${NO_PROXY}"
}

# Prints stdin to stderr with every line prefixed and clipped, at most
# DIAG_MAX_LINES lines.
diag() {
  head -n "${DIAG_MAX_LINES:-40}" | cut -c1-400 | sed "s/^/${ADAPTER_NAME:-harness}| /" >&2
}

# A bounded summary of a run that produced no review: the harness exit code,
# the JSONL event types, and the tail of the harness's stderr.
diag_failure() { # exit-code jsonl-file type-key
  local rc=$1 jsonl=$2 key=$3
  echo "no review (harness exit ${rc}); event counts and the tail of stderr follow" | diag
  if [ -s "${jsonl}" ]; then
    jq -rRs --arg k "${key}" '
      [split("\n")[] | select(length > 0) | (try fromjson catch {($k): "<not json>"})]
      | (group_by(.[$k] | tostring) | map("\(.[0][$k]): \(length)") | .[]),
        (map(select(type == "object" and .error != null)) | last // empty
          | "last error: \(.error | if type == "string" then . else tojson end)")' "${jsonl}" 2>&1 | diag
  else
    echo "no JSON output" | diag
  fi
  [ ! -s "${OUT_DIR}/harness.stderr" ] || tail -n 20 "${OUT_DIR}/harness.stderr" | diag
}

# True if the file holds a non-whitespace character.
has_text() {
  [ -f "$1" ] && grep -q '[^[:space:]]' "$1"
}
