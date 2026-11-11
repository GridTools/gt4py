# PR Agent

The `PR Agent` workflow runs [PR Agent](https://github.com/The-PR-Agent/pr-agent) to post an advisory review on pull requests. It is **never a required status check** and cannot gate a merge: the inference step is `continue-on-error`. Review policy lives in [`.pr_agent.toml`](../../../.pr_agent.toml); the endpoint, model and token limits are set in the workflow from [repository variables](#configuration-variables).

Only `/review` runs automatically. Users with write access can also comment `/review`, `/improve`, `/ask "<question>"` or `/describe` on a PR. `/describe` stays manual because it rewrites the PR body, which here is the filled-in PR template and becomes the squash-commit message.

The review rules come from `AGENTS.md`, `src/gt4py/next/AGENTS.md` and [`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md), shared with GitHub Copilot's reviewer. Extend those files rather than adding guidance to `.pr_agent.toml`.

## Who can trigger it

The `Authorize` step requires the actor's _effective_ repository permission to be `write` or higher. The actor is the PR author for the automatic review and the commenter for a command. PRs from anyone else are reviewed when someone with write access comments `/review`.

Write access comes from the `gt4py-core` and `gt4py-collaborators` teams, but not only from them:

- Organization owners have `admin` on every repository.
- Direct collaborators, such as the CSCS-CI machine account (see [cscs-ci.md](cscs-ci.md)). Its account type is `User`, so the workflow's bot filters do not exclude it.
- Every organization member, if the organization's base permission were ever raised to `write`.

Things to know before changing the workflow:

- **This check is the only gate.** `pull_request_target` workflows [always run, regardless of fork-approval settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories), so without it anyone could spend the CSCS allocation by opening PRs.
- **`author_association` is not a substitute.** It reports private organization members (most of them here) as `CONTRIBUTOR` or `NONE`, and `MEMBER` or `COLLABORATOR` do not imply write access.
- **There is no `maintain` case** because the API's `permission` field reports `maintain` as `write` and `triage` as `read`.
- **The check fails closed.** A denial is a `notice`; a failed API call is an error and fails the job. If the bot goes quiet for everyone, read the `Authorize` log first.
- **Runs are never cancelled.** A run joins its concurrency group when queued, before the job `if` is evaluated, so with `cancel-in-progress` any comment (e.g. `cscs-ci run ...`) would kill an in-flight review. Comment runs are keyed on the comment id so a later comment cannot evict a queued command. PR-event runs share one group per PR, where a newer pending review replaces an older one.
- **Drafts are skipped until marked ready.** PR Agent has no draft check of its own, so they would otherwise be reviewed twice.

## Why `pull_request_target`

Most GT4Py PRs come from forks, which get no repository secrets under `pull_request`. `pull_request_target` provides them, and its usual hazard (running untrusted code with secrets in scope) does not apply as long as two invariants hold:

- The job never checks out or executes PR code. PR Agent reads the diff through the GitHub API.
- The job never gets `contents: write`, which would turn a prompt injection in a diff into a way to push.

The remaining write scopes still let model output, derived from untrusted PR text, set labels and rewrite the PR title and body. That is why review labels are disabled and `/describe` is manual; keep them that way. The label settings are in the workflow environment because `.pr_agent.toml` is fetched over the GitHub API, and when the fetch fails PR Agent silently falls back to defaults that enable label writes.

`issues: write` is nearly redundant: comments and labels accept either write scope, and only the reaction acknowledging a command needs this one.

The workflow and `.pr_agent.toml` are both read from the default branch, so a change to either takes effect only once merged. To test one in a PR from an upstream branch, re-enable the blocks marked `COMMENTED OUT` in the workflow, which add a `pull_request` trigger and make `PR_AGENT_CONFIG_BRANCH` read the policy from the PR branch. Set that variable only to a literal branch name, since deriving it from PR input such as `github.head_ref` would let a PR supply its own review policy. Disable the blocks again before merging: once the branch is gone, PR Agent silently falls back to its built-in defaults. A run that loaded the policy logs `Applying repo settings`.

## Inference backend and cost

Inference runs on the [CSCS inference service](https://docs.cscs.ch/services/inference/) through its OpenAI-compatible API, with the `CSCS_INFERENCE_API_KEY` repository secret. Usage is billed against the CSCS project's compute allocation. Create the key in the [inference UI](https://ui.inference.cscs.ch/login) with a token budget, a reset period and a model restriction, so a runaway loop cannot drain the allocation, and store it with the CSCS-CI credentials in the internal CSCS credentials store.

The service does not record prompts or responses, and processing stays in Switzerland.

## Configuration variables

These are [repository variables](https://github.com/GridTools/gt4py/settings/variables/actions), so they can be changed without a PR. Fork PRs cannot override them.

| Variable                     | Default                            | Purpose                                                     |
| ---------------------------- | ---------------------------------- | ----------------------------------------------------------- |
| `PR_AGENT_MODEL`             | **required**                       | CSCS model id, as listed by `GET /v1/models`                |
| `PR_AGENT_FALLBACK_MODELS`   | **required**                       | Comma-separated ids, tried in order; `none` for no fallback |
| `PR_AGENT_TOKEN_BUDGET`      | `131072`                           | Context window PR Agent assumes                             |
| `PR_AGENT_MAX_OUTPUT_TOKENS` | `32768`                            | Tokens reserved for the answer                              |
| `PR_AGENT_AI_TIMEOUT`        | `720`                              | Seconds one model may take before the next is tried         |
| `PR_AGENT_API_BASE`          | `https://api.inference.cscs.ch/v1` | Inference endpoint                                          |

- **Model ids are stored without a provider prefix.** The workflow prepends `openai/`, LiteLLM's selector for the OpenAI-compatible route.
- **The models have no default**, so PR Agent never calls a model nobody chose. CSCS retires models on short notice, so review them periodically against `GET /v1/models`, the [model table](https://docs.cscs.ch/services/inference/api/) and the [service status](https://inference.status.cscs.ch).
- **The token limits apply to every model in the chain**, so they must fit the smallest one. The output reserve is sent as `max_tokens`, and a model with a lower output limit rejects the request. Check a new fallback before adding it, since it is only reached when the models before it have failed.
- **The output reserve must be generous.** The models worth using are reasoning models, whose reasoning tokens count against it. Too small a reserve yields an empty response, so the bot posts nothing instead of failing cleanly.
- **The budget is not a cap on the diff.** The diff gets the budget minus the reserve minus the prompt, which is substantial. Both values must be positive, because PR Agent reads zero or negative as "no limit" and the answer then truncates.
- **A more capable model is not a better primary if it is slow.** Every attempt that times out is billed and discarded before a fallback answers; the review names the model that produced it, and marks it as a fallback.
- **The step's time limit is derived**: one `PR_AGENT_AI_TIMEOUT` per model plus two minutes. `Resolve models` fails if that does not fit under the job's `timeout-minutes`.
- **A failed automatic review posts nothing.** The run stays green with a warning annotation; the cause is in the `Run PR Agent` log. A manual `/review` reports the reason.
- **The PR Agent version is pinned by digest in the workflow**, since GitHub does not allow expressions in `uses`. Upgrading is thus a reviewed change, as befits the image that receives the inference key.
