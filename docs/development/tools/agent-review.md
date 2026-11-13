# Agent Review

The `Agent Review` workflow ([`.github/workflows/agent-review.yml`](../../../.github/workflows/agent-review.yml)) posts an advisory review of a pull request written by a coding agent (a _harness_ such as opencode or Pi) that can explore the repository with tools. The agent runs inside the [AWF](https://github.com/github/gh-aw-firewall) sandbox, without network access, and uses the [CSCS inference service](https://docs.cscs.ch/services/inference/) through AWF's API proxy, so that neither the inference key nor a write-capable token is readable by the agent or by code from the PR. It is **never a required status check**. It runs independently of [PR Agent](pr-agent.md) and uses no comment commands, so it does not interfere with PR Agent's.

The GT4Py workflow is a small caller of the reusable workflow in [GridTools/ci-workflows](https://github.com/GridTools/ci-workflows), which holds the pipeline (its four jobs, the harness adapters, the AWF configuration and the prompt), its security invariants, its remaining risks and its tests; see its [documentation](https://github.com/GridTools/ci-workflows/blob/main/docs/agent-review.md). Changes to the pipeline are made and reviewed there. The decision record is [ADR 0001](../ADRs/ci/0001-sandboxed-agent-pr-review.md).

## Usage

- Anyone with write access can request a review of any PR not opened by a bot, including a fork or draft PR, by adding the **`agent-review` label**. The **labeller's** permission is checked, not the author's. To review again, remove and re-add the label. Pushes to the PR do not trigger a review.
- **Run workflow** on the Actions tab (`workflow_dispatch`) reviews a given PR number, optionally with another harness or as a dry run that only shows the comment in the run summary.
- **Not enabled yet**: an automatic review when a non-draft PR is opened, reopened or marked ready by an author with write access. The called workflow handles it, but the trigger types are commented out until the label and dispatch paths have been observed. When enabled, note that a `triage` user can close and reopen a writer's PR to start one run per cycle.
- Write access comes from the same teams and accounts as for PR Agent (see [pr-agent.md](pr-agent.md#usage)). `triage` is not enough: the permission API reports it as `read`.

Each harness keeps one comment per PR, updated in place by later runs. The comment names the harness, its version, the model and the reviewed commit; pushes after that commit are not reviewed.

## What GT4Py configures

In the workflow file:

- The **rule files** the agent follows, in this order: `.github/copilot-instructions.md`, `AGENTS.md`, `CODING_GUIDELINES.md`, `src/gt4py/next/AGENTS.md`. Each is reset to the PR's base version before the agent runs, so a PR cannot change the rules it is reviewed by. To change what the agent checks, extend these files rather than the prompt.
- The **prompt addendum** [`.github/agent-review/prompt-addendum.md`](../../../.github/agent-review/prompt-addendum.md): GT4Py-specific notes added to the prompt, read from the workflow's commit (`main`), never from the PR. Every file it names in backticks must be one of the rule files, or the review fails.
- The **pinned version** of the reusable workflow (see [Updating the pinned version](#updating-the-pinned-version)).
- The **permission ceiling** for every called job, `contents: read` and `pull-requests: write`, and the explicit mapping of the key (never `secrets: inherit`).

Repository variables (**Settings → Secrets and variables → Actions → Variables**), changeable without a PR:

| Variable                    | Meaning                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `AGENT_REVIEW_HARNESS`      | `opencode` (default and recommended) or `pi`; `stub` for tests. A dispatch can choose another one.                              |
| `AGENT_REVIEW_MODEL`        | Model id on the CSCS inference service, e.g. `zai-org/GLM-5.3-Flash`. No default: without it, a review with a model fails.      |
| `AGENT_REVIEW_API_BASE`     | Optional OpenAI-compatible endpoint; default `https://api.inference.cscs.ch/v1`.                                                |
| `AGENT_REVIEW_UPLOAD_AUDIT` | Optional, `true` to upload AWF's audit and proxy logs as an artifact; in this public repository any signed-in user can read it. |

## Required repository settings

Without these, the workflow does not run, or runs without a key.

1. **The `agent-review` environment, restricted to the default branch** (**Settings → Environments → agent-review → Deployment branches and tags → Selected branches**: `main` only), holding the secret **`AGENT_REVIEW_INFERENCE_API_KEY`**. Only the called `review` job uses it, and its branch rule is evaluated against this repository's run, so the label and dispatches from `main` get the key while branch workflows and dispatches of other refs are denied. Never create a repository- or organization-level secret of that name.
2. **A dedicated CSCS key** in that secret, never PR Agent's `CSCS_INFERENCE_API_KEY`: created in the [inference UI](https://ui.inference.cscs.ch/login) for this workflow alone, with its own token budget and reset period, and restricted to the configured model. Its budget is the only hard spending cap.
3. **An Actions event policy** (**Settings → Actions → Policies**) scoped by workflow path to `.github/workflows/agent-review.yml`, this repository's caller file (not the path in GridTools/ci-workflows), allowing **`pull_request_target` and `workflow_dispatch`**. GitHub blocks `pull_request_target` by default in public repositories from **2 December 2026**, and a policy's event list is an exhaustive allowlist. PR Agent's policy (see [pr-agent.md](pr-agent.md#event-policy)) covers only `.github/workflows/pr-agent.yml`.
4. The repository variables above and the label `agent-review`.

**Deployment state in GridTools/gt4py (2026-11-13)**: items 1 and 2 are in place, and so are the variables (`AGENT_REVIEW_HARNESS=opencode`, `AGENT_REVIEW_MODEL=zai-org/GLM-5.3-Flash`). Still missing: the **`agent-review` label**, and **an event policy for this workflow** (item 3): the only one, "Allow PR Agent events" (policy 13604), is scoped to `.github/workflows/pr-agent.yml`.

## Updating the pinned version

The `uses:` line pins a full commit SHA of GridTools/ci-workflows, with the release tag in a comment (`# vX.Y.Z`); never a branch or a tag name. To update it:

1. Read the release notes. A new permission, a new required input, a raised cap or a harness or AWF change is marked as breaking there; a new permission must also be added to the calling job's `permissions:`, or every run fails at startup.
2. Replace the SHA (the commit the tag points to) and the comment, also in the commented-out `test` job.
3. After merging, check it with **Run workflow** as below.

## Testing changes

The workflow file and the addendum are read from `main`, so a change takes effect once merged. Then:

1. **Run workflow** with harness `stub` and dry run: the run summary shows the stub's comment, and the `authorize` job's notices name the pinned commit of GridTools/ci-workflows.
2. The same with the default harness, dry run (this uses the key).
3. The label on a PR.

Before merging, the blocks marked `COMMENTED OUT` in the workflow can be enabled on a branch of this repository with a PR to `main`, as for PR Agent; never merge them enabled. With the current pin this checks only that the run starts (the pin resolves and the permission ceiling is accepted): the called workflow accepts the `pull_request` event only in its own self-test, so `authorize` fails, and its `review` job would be denied the `main`-only environment on a branch ref anyway.

Changes to the pipeline itself are tested in GridTools/ci-workflows.
