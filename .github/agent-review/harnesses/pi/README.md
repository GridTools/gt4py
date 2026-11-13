# Pi adapter

[Pi](https://github.com/earendil-works/pi) (MIT), npm package
`@earendil-works/pi-coding-agent` (`@mariozechner/pi-coding-agent` is its old,
stale name). Wire: OpenAI chat, through AWF's OpenAI listener, with Pi's
`openai-completions` API (Pi's built-in `openai` provider would use the
Responses API, which CSCS does not serve).

## Version and upgrades

Pinned to 1.5.2 through `package.json` and `package-lock.json` (npm integrity
hashes for all 123 packages). Pi needs Node.js >= 22.19.0, so the adapter
ships its own: `node.pin` holds the version and the SHA-256 of the official
`linux-x64` tarball, from `SHASUMS256.txt`. `npm ci --ignore-scripts` skips
the install scripts of two dependencies (`@google/genai`, `protobufjs`); Pi
works without them.

To upgrade, set the version in `package.json` and regenerate the lockfile with
`npm install --package-lock-only --ignore-scripts`.

## Configuration and isolation

- `models.json` in a scratch `PI_CODING_AGENT_DIR` defines one provider
  (`awf-proxy`) with a placeholder key and the base URL
  `$AWF_API_PROXY_OPENAI_URL/v1` (Pi appends `/chat/completions`).
- Pi's Anthropic API would also work through the OpenAI listener (which
  forwards `/v1/messages`; Pi appends `/v1/messages?beta=true` to the
  origin), but is not used: one API for all harnesses is simpler, and the
  chat route is the one verified against CSCS with Pi.
- `--no-approve`: the repository's `.pi/` (settings, extensions, which would be
  **executed**, MCP servers, which would be **spawned**, system prompts,
  skills) and `.agents/skills` are ignored. `--no-context-files`: `AGENTS.md`
  and `CLAUDE.md` are ignored too, so the prompt names the files to read.
- Tools: `read`, `grep`, `find`, `ls`, `bash` (no `edit`, `write`). `bash`
  cannot be restricted; the sandbox is the boundary.
- The prompt is passed on stdin: with stdin open and no prompt, Pi waits
  forever.
- **No turn cap and no stall detection**: AWF's `apiProxy.maxTurns` and the
  harness timeout bound a run.

## Outbound calls

None seen in `--print` mode; `PI_OFFLINE=1` and `PI_TELEMETRY=0` are set
anyway (model catalogue refresh, install telemetry).

## Answer and exit codes

`--mode json` writes NDJSON. The review is the text of the last `turn_end`
event, if its `stopReason` is `stop`. Pi exits 0 even when every request
failed (`stopReason` `error`); `run.sh` then exits 2 and prints the errors.
