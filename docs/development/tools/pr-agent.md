# PR Agent

The `PR Agent` workflow runs [PR Agent](https://github.com/The-PR-Agent/pr-agent) (MIT-licensed, community-owned since Qodo donated it) to post an advisory review on new pull requests. It is **never a required status check**, and the step that calls the inference service is `continue-on-error`, so it cannot gate a merge. Review policy lives in [`.pr_agent.toml`](../../../.pr_agent.toml). The endpoint, model and token limits are set in the workflow instead, from the repository variables listed below. Only the endpoint and key are host-only in PR Agent, which refuses them from repository config or a comment argument; the model and token settings are placed there by choice, so they sit beside the endpoint they belong to.

Maintainers can invoke tools on demand by commenting on a PR, e.g. `/review`, `/improve`, `/ask "<question>"` or `/describe`. Only `/review` runs automatically. `/describe` is not automatic because it rewrites the PR body, which here is the squash-commit message body and the filled-in pull request template. It leaves the title alone at this pin, since `pr_description.generate_ai_title` is false. It and `/generate_labels` write PR metadata from model output, so both are better left behind a human request.

## Who can trigger it

Both paths that spend inference credit go through **one** authorization check: the `Authorize` step resolves the actor's *effective* repository permission and requires `write`, `maintain` or `admin`. The actor is the PR author for automatic review and the commenter for a command.

In practice that resolves to whoever currently holds `write` or better, which is granted from three places. Two of them are teams:

- **`gt4py-core`**, which has `admin` on this repository.
- **`gt4py-collaborators`**, which has `push`.

The third is anything granted outside those teams, and it matters because managing the two teams does not by itself determine who can spend inference credit:

- **Organization owners** hold `admin` on every repository in the organization, so they can trigger regardless of team membership — including any owner added later.
- **Direct repository collaborators** can be granted permission on this repository alone, bypassing both teams. One such grant exists for the machine account behind CSCS-CI (see [cscs-ci.md](cscs-ci.md)). A machine account of this kind has GitHub account type `User`, not `Bot`, so the workflow's sender and author bot filters do **not** exclude it: a pull request it authored, or a `/`-prefixed comment from it, would spend inference credit.

Organization members with none of those grants resolve to `read` and cannot trigger anything. They can still open pull requests and comment as usual — only the bot is gated, and their PRs are reviewed when someone with write access comments `/review`. The organization's base repository permission is effectively `read`; were it ever raised to `write`, every member would silently gain trigger rights with nothing in this repository to signal it.

This check is what makes the automatic path safe. GitHub's fork-PR approval setting does not apply, because "[since the base branch is considered trusted, workflows triggered by these events will always run, regardless of approval settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories)". Without it, any stranger could spend the project's CSCS allocation by opening pull requests. An outside contributor's PR is reviewed when a maintainer asks with `/review`.

The check uses effective permission rather than `author_association`, for two reasons:

- It does not identify who it appears to. Webhook payloads report `MEMBER` only for *public* organization members; private members appear as `CONTRIBUTOR` or `NONE`. Organization membership here is mostly private, so an association gate would skip nearly every core developer, and silently.
- It is not an authorization decision even when it is accurate. An organization whose base permission is read still yields `MEMBER`, and `COLLABORATOR` includes outside collaborators with read-only or triage access. Such a principal could otherwise spend credit and, through `/describe`, have the bot rewrite a PR title and body they are not allowed to edit themselves.

The check **fails closed**: an API error denies rather than opening the bot up. It distinguishes the two cases, because a silent deny is how a broken gate would hide — "asked, and denied" is a `notice`, while "could not ask" is an `::error::` and a non-zero step. GitHub documents the collaborator-permission endpoint as accepting an installation access token with `Metadata: read`, which `GITHUB_TOKEN` always carries, so a 403 is not expected, but this has not been exercised live. If the bot goes quiet on both paths at once, read the `Authorize` step's log first.

No in-flight run is ever cancelled. Automatic reviews are serialized per pull request, and a pending one is replaced by a newer one; comment commands run independently of them and of each other. A concurrency group is claimed when a run is queued, *before* the job condition is evaluated, so every comment event joins a group even when its own job is then skipped. `cancel-in-progress` therefore stays off, or any comment — including the `cscs-ci run ...` ones that routinely land seconds after a PR opens — would kill that PR's in-flight review with no trace. GitHub also evicts an existing *pending* run when a newer one joins the same group, so comment-triggered runs are keyed on the comment id: a queued `/improve` cannot be evicted by a later unrelated comment. PR-event runs share one group per pull request, where a newer automatic review replacing a pending one saves an inference call. The default queue holds a single pending run, so two PR events still cost two reviews; a third evicts the second.

Draft PRs are skipped until marked ready. The Action's automatic path has no draft check of its own, so without the workflow's `draft == false` condition a draft PR would be reviewed once on `opened` and again on `ready_for_review`.

PR Agent reads `AGENTS.md`, `src/gt4py/next/AGENTS.md` and [`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md) through its `repo_context_files` setting, so both reviewers work from the same rules and `.pr_agent.toml` restates none of them. Extend those files rather than adding guidance to `.pr_agent.toml`.

## Why `pull_request_target`

Most GT4Py pull requests come from forks, and GitHub gives fork PRs no access to repository secrets. A workflow triggered on `pull_request` could therefore never read the inference API key, and would fail on the majority of PRs. The workflow uses `pull_request_target` instead, which runs with secrets available.

The usual `pull_request_target` hazard — running untrusted code with secrets in scope — does not apply, because PR Agent reads the diff through the GitHub API and the job has no `actions/checkout` step. Two invariants keep that true, and both must be preserved:

- The job must never check out or execute pull request code.
- The job must never be granted `contents: write`. The model reads an untrusted diff, so a write permission would turn a prompt injection into a source-write primitive.

Withholding `contents: write` does not make the model's output inert. `issues: write` and `pull-requests: write` are needed to publish a review at all, and they also permit setting labels and rewriting a PR title and body — all from model output derived from untrusted PR text. Review labels are disabled and `/describe` is not run automatically; keep them that way. The label settings live in the workflow environment rather than in `.pr_agent.toml`, because that file is fetched over the GitHub API and a failure there is only logged before PR Agent falls back to built-in defaults, which enable label writes. An environment override cannot fail to load and takes precedence.

Note that both `pull_request_target` and `.pr_agent.toml` are read from the base (default) branch, so a change to either has no effect until merged to `main`. The workflow carries a commented-out `pull_request` trigger to exercise a workflow change on the PR that makes it; as with `daily-ci.yml`, it only works for PRs from branches in the upstream repository. It does **not** help with `.pr_agent.toml`, which PR Agent still reads from the default branch.

## Inference backend and cost

Inference runs on the [CSCS inference service](https://docs.cscs.ch/services/inference/) through its OpenAI-compatible API, using the `CSCS_INFERENCE_API_KEY` repository secret. Access is granted per CSCS project and billed as token-based resource consumption converted from node hours, so this spend competes with the project's compute allocation rather than being a separate subscription. The key should be created in the [inference UI](https://ui.inference.cscs.ch/login) with a token budget, a reset period, and a model restriction, so a runaway loop fails closed instead of draining the allocation. Store it alongside the CSCS-CI credentials in the internal CSCS credentials store.

Running inference at CSCS also keeps PR content on infrastructure CSCS controls: the service does not record prompts or model responses, and processing stays in Switzerland.

## Configuration variables

The model, its token limits and the endpoint are [repository variables](https://github.com/GridTools/gt4py/settings/variables/actions), not hardcoded values, so they can be retuned from the repository settings without opening a PR. The two model variables are required; the rest default in the workflow and only need setting to override.

| Variable                     | Default                            | Purpose                                                     |
| ---------------------------- | ---------------------------------- | ----------------------------------------------------------- |
| `PR_AGENT_MODEL`             | **none — required**                | One CSCS model id, exactly as `GET /v1/models` lists it     |
| `PR_AGENT_FALLBACK_MODELS`   | **none — required**                | Comma-separated ids, tried in order; `none` for no fallback |
| `PR_AGENT_TOKEN_BUDGET`      | `65536`                            | Input cap per request                                       |
| `PR_AGENT_MAX_OUTPUT_TOKENS` | `16384`                            | Headroom reserved for the answer                            |
| `PR_AGENT_API_BASE`          | `https://api.inference.cscs.ch/v1` | Inference endpoint                                          |

Model ids are stored bare. The workflow prepends `openai/`, which is LiteLLM's selector for the OpenAI-compatible route and not part of a model name: LiteLLM strips it and sends the remainder, so CSCS receives exactly the id it published. Without a prefix the call fails with `LLM Provider NOT provided`, and the prefix belongs with `PR_AGENT_API_BASE` rather than inside every id.

Neither model variable has a default, so PR Agent never calls a model nobody chose. The `Resolve models` step checks both before the container starts and fails with an `::error::` naming the unset one.

The two token limits are **policy, not model properties**. For a model absent from PR Agent's own `MAX_TOKENS` table — as the CSCS models are — `get_max_tokens` returns `min(max_model_tokens, custom_model_max_tokens)` and applies it to whichever model serves the request, so a single pair governs the primary and every fallback. Choose values that fit the smallest model in use:

| Model                                                | Context window | Max output |
| ---------------------------------------------------- | -------------- | ---------- |
| `zai-org/GLM-5.3`                                    | 1,048,576      | 32,768     |
| `zai-org/GLM-5.3-Flash`                              | 1,048,576      | 32,768     |
| `Qwen/Qwen3.8-27B-FP8`                               | 262,144        | —          |
| `Qwen/Qwen3.8-Flash-Next-FP8`                        | 262,144        | —          |
| `google/gemma-4-31B-it`                              | 262,144        | —          |
| `swiss-ai/Apertus-v1.5-70B` (and `-8B`, `-thinking`) | 262,144        | —          |

Check the current catalogue and limits against `GET /v1/models` and the [CSCS model table](https://docs.cscs.ch/services/inference/api/), since the catalogue changes on short notice. The defaults above sit far inside every row.

`config.custom_model_max_tokens` is set to the same value as the budget. It only has to be a positive number at least as large, because the `min()` above makes any larger value equivalent — so a model's true context window never has to be configured, only respected when choosing the budget.

The diff budget is `PR_AGENT_TOKEN_BUDGET` minus `PR_AGENT_MAX_OUTPUT_TOKENS` minus the rendered prompt, and the step refuses a budget that does not exceed the reserve. If it went negative, every file would be dropped and the review would fail with `No PR diff fits the /review request` — visible only in the run log, since the inference step is `continue-on-error`.

The PR Agent version is **not** a variable. GitHub does not allow expressions in a step's `uses`, so it is pinned in the workflow, by digest with the release version in a comment. Upgrading is therefore a reviewed change, which is appropriate for the image that receives the inference key. It is the only version this workflow pins; it uses no other actions.

Note that fork PRs read variables from the base branch, as with everything else in this workflow, so a fork cannot substitute its own model or endpoint.

Two consequences of this backend are worth knowing before changing the model:

- **The models worth using here are reasoning models** (the non-thinking Apertus variants are not), and reasoning tokens are drawn from the same output budget as the answer. `PR_AGENT_MAX_OUTPUT_TOKENS` must therefore be generous; left unset the provider's own limit applies, and a review can come back truncated or empty, which presents as the bot posting nothing rather than as a clean error.
- **CSCS rotates its model catalogue on short notice**, including retiring models that its own documentation examples still reference. `PR_AGENT_MODEL` needs periodic review against `GET /v1/models`. Service status is at [inference.status.cscs.ch](https://inference.status.cscs.ch).
