# opencode adapter

[opencode](https://github.com/anomalyco/opencode) (MIT), the v2 line published
as `@opencode/cli` on npm (`opencode-ai` is the older v1 line, a different
product). Wire: OpenAI chat, through AWF's OpenAI listener.

## Version and upgrades

Pinned to 2.2.7 through `package.json` and `package-lock.json` (npm integrity
hash). The adapter installs the platform package
`@opencode/cli-linux-x64-baseline` directly: the `@opencode/cli` wrapper only
hard-links that same binary in a postinstall script, and its lockfile pulls
the binaries of every platform (992 MB instead of 247 MB). `baseline` also
runs on CPUs without AVX2. Linux x64 only.

To upgrade, set the version in `package.json`, then regenerate the lockfile
with `npm install --package-lock-only --ignore-scripts` and check that the
release still has the flags and environment variables used in `run.sh`.

## Configuration and isolation

- The configuration comes from `OPENCODE_CONFIG_CONTENT` only: one provider
  (`awf-proxy`, base URL `$AWF_API_PROXY_OPENAI_URL/v1`, placeholder key), a
  policy denying every other provider, `share` and `update` disabled.
- `OPENCODE_DISABLE_PROJECT_CONFIG=1`: the repository's `opencode.json(c)`,
  `.opencode/` and `AGENTS.md` are ignored. Without it, `.opencode/plugins/*.js`
  are **executed** and MCP servers in `opencode.json` are **spawned**.
- Even with it, a nested `AGENTS.md` is attached to the result of a `read` in
  its directory, which is why `run-review` resets instruction files at any
  depth.
- Denied tools: `edit`, `webfetch`, `websearch`, `question`, `subagent`,
  `execute`, `opencode_*`. Left: `glob`, `grep`, `read`, `shell`, `skill`.
  `shell` is not restricted (an allowlist is possible but would block
  harmless read-only commands); the sandbox is the boundary.
- `agents.build.steps` caps a run at 35 model steps, below AWF's
  `apiProxy.maxTurns`. opencode makes one more request first, for the session
  title. At the last step tools are disabled, so the model has to answer.

## Outbound calls

`models.opencode.ai` (model catalogue), suppressed with
`OPENCODE_DISABLE_MODELS_FETCH=1`. With it, opencode connects only to the
listener and probes local model servers on loopback (Ollama, LM Studio,
vLLM ports), which is harmless.

## Answer and exit codes

`opencode run --format json` writes JSONL. The review is the text after the
last `tool_use` event. opencode exits 0 when the step cap ends a run without
an answer (then `run.sh` exits 2) and 1 when the endpoint is unreachable.
