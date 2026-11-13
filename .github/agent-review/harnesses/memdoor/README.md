# memdoor adapter

> [!WARNING]
> Experimental, not recommended at the current caps: in a quality evaluation on real PRs, memdoor
> with `zai-org/GLM-5.3-Flash` timed out at the 20-minute harness limit without a review (3 of 3).

[memdoor](https://github.com/guregodevo/memdoor) (Apache-2.0), a single static
Go binary. Wire: OpenAI chat, through AWF's OpenAI listener.

## Version and upgrades

Pinned in `release.pin` to a release tag and the SHA-256 of
`memdoor-linux-amd64`. `install.sh` downloads the binary from the GitHub
release and checks the hash. The release's `SHA256SUMS` sits next to the
binary, so the hash in this repository is the integrity check; take it from
`SHA256SUMS` when bumping, and compare it with a second download. There is no
GitHub attestation. Releases appear several times a day; bump deliberately,
not to every release.

## Configuration and isolation

- "Company AI gateway" mode: `AI_GATEWAY_BASE_URL=$AWF_API_PROXY_OPENAI_URL/v1`,
  `AI_GATEWAY_API=chat` (the default is the Responses API, which CSCS does not
  serve), a placeholder `AI_GATEWAY_TOKEN`, and `MEMDOOR_MODEL`. Vendor key
  variables are unset, since they would select another mode.
- memdoor runs agents through a local gateway daemon: `run.sh` starts one on a
  random loopback port, with a scratch admin account, and stops it at the end.
- memdoor has **no switch to ignore the repository's `AGENTS.md`**. It loads
  the root one only (not nested ones, not `CLAUDE.md` or
  `.github/copilot-instructions.md`); `run-review` resets it to the base
  version. Its `skill` tool would load `.agents/skills/*/SKILL.md` from the
  repository on demand.
- Guards refuse `apply_patch`, `write_file`, `edit_file`, `web_fetch`,
  `web_search`, `sessions_spawn`, `cron`, `workflow`, `memory`, `skill` and
  `ask_user_question` (an unanswered question ends a run with exit 3); the
  model is told the tool is unavailable. `bash` stays; the sandbox is the
  boundary.
- **No turn cap**, only a detector that stops after the same call is repeated
  five times. AWF's `apiProxy.maxTurns` stops a well-behaved run; the bound
  is AWF's request rate limits and the harness timeout (see `../../awf.yaml`).

## Outbound calls

A daily version check against `memdoor.ai`, suppressed by pre-seeding
`$HOME/.memdoor/update-check.json`. Telemetry is opt-in and stays off. memdoor
also sends `GET /v1/models` to the listener three times per run.

## Answer and exit codes

`memdoor run --json` writes NDJSON. The review is the text after the last
`tool_done` event, without memdoor's trailing "Recap: ..." and "(note: this
turn used no tools ...)" text events. Exit 2 ("unverified") is normal for a
review; the stall detector ends a run with exit 0 and a "Stopped: ..." text,
which `run.sh` turns into exit 2.
