# Agent Review

The `Agent Review` workflow ([`.github/workflows/agent-review.yml`](../../../.github/workflows/agent-review.yml)) posts an advisory review of a pull request written by a coding agent (a _harness_ such as opencode, memdoor or Pi) that can explore the repository with tools. The agent runs inside the [AWF](https://github.com/github/gh-aw-firewall) sandbox and uses the [CSCS inference service](https://docs.cscs.ch/services/inference/) through AWF's API proxy, so neither the inference key nor a write-capable token is ever readable by the agent or by code from the PR. It is **never a required status check**: if the agent fails or times out, the run stays green with a warning annotation. It runs independently of [PR Agent](pr-agent.md), and uses no comment commands, so it does not interfere with PR Agent's.

The design and the alternatives considered are in [ADR 0001](../ADRs/ci/0001-sandboxed-agent-pr-review.md).

> [!WARNING]
> Pilot. The opencode, memdoor and Pi adapters have been run on a fork only, and the AWF facts marked `TODO(phase0)` in [`awf.yaml`](../../../.github/agent-review/awf.yaml) and the workflow are still being verified.

## Usage

- A review runs automatically when a non-draft PR is opened, reopened or marked ready, if its **author** has write access.
- Anyone with write access can request a review of any PR, including a fork or draft PR, by adding the **`agent-review` label**. The **labeller's** permission is checked, not the author's. To review again, remove and re-add the label. Pushes to the PR do not trigger a review.
- **Run workflow** on the Actions tab (`workflow_dispatch`) reviews a given PR number, optionally with another harness or as a dry run that only shows the comment in the run summary.
- Write access comes from the same teams and accounts as for PR Agent (see [pr-agent.md](pr-agent.md#usage)). `triage` is not enough: the permission API reports it as `read`.

Each harness keeps one comment per PR, updated in place by later runs. The comment names the harness, its version and the model.

## How it works

| Job         | Holds                                             | Does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `authorize` | read token                                        | Checks permissions (fails closed on API errors), pins the PR head and base sha, resolves the harness and model. Every other job requires its `allowed` output.                                                                                                                                                                                                                                                                                                                                                                                                          |
| `prepare`   | read token, **no secrets**                        | Runs the harness's `install.sh` from the workflow's own commit and uploads the installed harness as an artifact. Package install hooks run here, where there is nothing to steal.                                                                                                                                                                                                                                                                                                                                                                                       |
| `review`    | read token; inference key in **one step's** env   | Copies [`.github/agent-review/`](../../../.github/agent-review/) from the workflow's own commit to `/opt/agent-review` (root-owned, read-only in the sandbox), writes the PR metadata as raw bytes, and runs `awf … -- run-review`. Inside the sandbox, [`run-review`](../../../.github/agent-review/bin/run-review) fetches the PR with git, resets agent instruction files to the base version, renders the prompt and runs the harness. Afterwards [`collect-result`](../../../.github/agent-review/bin/collect-result) copies only `review.md` and `meta.json` out. |
| `publish`   | write token (`issues`, `pull-requests`), no model | On a fresh VM, [`publish-comment`](../../../.github/agent-review/bin/publish-comment) validates the review again and upserts the comment.                                                                                                                                                                                                                                                                                                                                                                                                                               |

The review rules are in [`prompt.md.tmpl`](../../../.github/agent-review/prompt.md.tmpl), which points the agent at `.github/copilot-instructions.md`, `AGENTS.md`, `CODING_GUIDELINES.md` and `src/gt4py/next/AGENTS.md`. Extend those files rather than the prompt. The PR title and body are included as delimited untrusted data.

## Security invariants

The workflow uses `pull_request_target` because most GT4Py PRs come from forks, which get no repository secrets under `pull_request`. That trigger runs with secrets next to untrusted code, so all PR content is handled inside AWF only. Keep these when changing it:

1. **The real key is only in AWF's API-proxy sidecar** and the environment of the `awf` step; the agent sees placeholders. Never inline it into a `run:` script or pass it to another step.
2. **No PR-derived byte is executed or parsed on the host.** No `actions/checkout`, no `uv sync`, no git on the PR tree outside the sandbox. `pr.json` is written as bytes and parsed inside the sandbox.
3. **The host is tainted once `awf` returns.** AWF mounts the workspace, `/tmp` and parts of the home directory read-write. Steps after it use no token, run no git, use no cache action, and run only `/usr/bin` tools with a fixed `PATH`. The only data leaving `review` is at most 64 KiB per file of regular, single-link files named `review.md` and `meta.json`, copied without following symlinks.
4. **Write tokens exist only in `publish`**, which runs no model and no PR code, and treats the review as hostile: it must be valid UTF-8, HTML comment delimiters are broken up (so the review can neither hide text nor imitate the comment marker), `@mentions` are neutralised with a zero-width joiner (and character references such as `&#64;` or `&commat;`, which GitHub also turns into mentions, are escaped), and the length is capped. The marker is matched only on the first line of the bot's own comments, never with `--edit-last`, because PR Agent posts as the same bot.
5. **Harness, prompt, AWF config and instruction files come from the workflow's own commit** (`github.workflow_sha`, fetched as a tarball), never from the PR, so a PR cannot change the policy it is reviewed by. Files matching [`instruction-files.txt`](../../../.github/agent-review/instruction-files.txt), at any depth, are reset to the base version (or removed) before the harness starts; the PR's versions are kept as data for the review.
6. **The `review` job starts no service on the host.** The sandbox can connect to any host port bound to `0.0.0.0` through AWF's network gateway (`172.30.0.1`); on a stock runner that is only `sshd`. No `services:` containers, no daemons started in a step.
7. **AWF's logs are read only from `$RUNNER_TEMP`** (`--proxy-logs-dir`, `--audit-dir`). Without `--proxy-logs-dir`, AWF moves them to `/tmp` after the run, where the sandbox can plant fake ones.
8. **Everything is pinned**: actions by commit SHA, the AWF version, AWF images by digest (`image-tag` output of the setup action), and each harness by exact version with an integrity check.

## What remains risky

- **Prompt injection can steer the review's content**: a false approval, misleading findings, links. Mitigations: the comment is labelled advisory, never a status; mentions are neutralised and the size is capped. Cross-references such as `owner/repo#123` are not neutralised and can create backlinks in other repositories.
- **Budget burn** up to the caps below and the key's own budget, by anyone with write access or a PR author with write access.
- **A sandbox escape** (kernel or Docker bug) would expose the key on the host. Mitigations: a dedicated, low-budget key that is quick to rotate; gVisor (`--container-runtime gvisor`) is planned once the harnesses are known to work with it.
- `@` and `&#` are neutralised everywhere, including code blocks, so a Python decorator copied from a review contains an invisible character, and numeric character references show literally.

## Configuration

The key is the `CSCS_INFERENCE_API_KEY` repository secret, the same name as PR Agent's. On upstream it should be a **dedicated key** for this workflow, with its own token budget, reset period and model restriction, created in the [inference UI](https://ui.inference.cscs.ch/login): it is the hard spending cap. If the secret is missing, the review is skipped with a notice (the `stub` harness needs no key).

| Variable                    | Default                            | Purpose                                                                             |
| --------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------- |
| `AGENT_REVIEW_HARNESS`      | **required**                       | `opencode`, `memdoor`, `pi` or `stub`; the `workflow_dispatch` input overrides it   |
| `AGENT_REVIEW_MODEL`        | **required** (except for `stub`)   | CSCS model id, as listed by `GET /v1/models`; also the only model AWF lets through  |
| `AGENT_REVIEW_API_BASE`     | `https://api.inference.cscs.ch/v1` | OpenAI-compatible endpoint                                                          |
| `AGENT_REVIEW_UPLOAD_AUDIT` | unset                              | `true` uploads AWF's audit and proxy logs (requests, tokens, egress) as an artifact |

Fixed limits, changed by a reviewed PR: 80 model calls (`apiProxy.maxTurns`; opencode also stops itself after 35 steps, memdoor and Pi have no cap of their own), 30 requests per minute, 20 minutes for the harness, 25 for AWF's `agentTimeout`, 30 for the step and 45 for the job. AWF's AI-credit cap is not used, since it prices CSCS models with an estimate.

The `stub` harness makes no model call and posts a fixed review containing hostile content; it is for testing the pipeline only.

## Harness adapters

A harness is a directory `.github/agent-review/harnesses/<name>/`; the name must also be added to the `workflow_dispatch` choice and to the `case` in the `Authorize` step. The contract is in [`harnesses/README.md`](../../../.github/agent-review/harnesses/README.md); in short:

- `install.sh <prefix>` runs in `prepare` and installs an exact version, integrity-checked, into `<prefix>`, writing the version to `<prefix>/VERSION`. The prefix is unpacked to `/opt/agent-review/harness/<name>` and must run from there without network access.
- `run.sh` runs inside the sandbox, from the PR checkout, with `PROMPT_FILE`, `REPO_DIR`, `OUT_DIR`, `CONTEXT_DIR`, `MODEL`, `HARNESS_PREFIX` and `PR_NUMBER` set, plus AWF's `AWF_API_PROXY_OPENAI_URL` / `AWF_API_PROXY_ANTHROPIC_URL`. It keeps its configuration in a scratch home under `/tmp`, disables updates, telemetry and sharing, and writes `$OUT_DIR/review.md`. Exit codes: 0 review written, 2 no review, anything else a failure; it must return 2, not 0, when the harness exits 0 without a real answer.
- `wire` names the API the harness speaks: `openai-chat`, `anthropic-messages` or `none`. It selects the variable the key is passed in, and so AWF's listener.
- File names the harness reads as instructions or configuration, at any depth, go into `instruction-files.txt`.

The [adapters README](../../../.github/agent-review/harnesses/README.md) compares the three harnesses (version, wire, how the repository's own agent configuration is disabled, turn cap); each adapter's `README.md` explains its pins, upgrades and isolation. In short: all three use the OpenAI chat API (AWF v0.29.12 must get only `OPENAI_API_KEY`: with `ANTHROPIC_API_KEY` its entrypoint fails when the home directory is read-only); opencode and Pi have switches that ignore the repository's agent configuration (which can otherwise execute code from the PR), memdoor has none and relies on the reset of `AGENTS.md` and on guards against its `skill` tool.

## Testing changes

`pull_request_target` and `workflow_dispatch` read the workflow from the default branch, so a change cannot be tested from a PR. To test on a fork:

1. Open a throwaway PR inside the fork to review and comment on.
2. On a branch named `ci/agent-review-<n>`, re-enable the two blocks marked `COMMENTED OUT` in the workflow (the `push` trigger and the `TEST_*` parameters, with the throwaway PR's number) and push it. Every push runs the pipeline with the `stub` harness unless `TEST_HARNESS` says otherwise; other harnesses need `TEST_MODEL` (or the `AGENT_REVIEW_MODEL` variable) and the `CSCS_INFERENCE_API_KEY` secret in the fork. Never merge these blocks enabled.

The scripts have offline tests: `bash .github/agent-review/tests/test-collect-result.sh` and `test-publish-comment.sh`. `test-run-review.sh` runs the driver with the `stub` harness against a real PR, without AWF, and needs network access. To try an adapter without a key, run `install.sh <prefix>`, then `run-review` with `AWF_API_PROXY_OPENAI_URL` and `AWF_API_PROXY_ANTHROPIC_URL` pointing at a local fake OpenAI/Anthropic server.
