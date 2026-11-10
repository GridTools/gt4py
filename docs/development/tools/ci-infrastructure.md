# CI infrastructure

Any test job that runs on CI is encoded in automation tools like **nox** and **pre-commit** and can be run locally instead.

## GitHub Workflows

The following workflows are currently active.

### Code Quality

A `Code Quality` workflow runs `pre-commit` to check code quality requirements through tools like **mypy** or **ruff**.

### Tests

The `Test code components (CPU)`, `Test Package Root` and `Test examples in documentation` workflows run the correspondent `nox` sessions testing different parts of GT4Py. `Test code components (CPU)` is a parameterized workflow (see more details in [_Code component test sessions_](#code-component-test-sessions). `Test Package Root` tests the root package code and `Test examples in documentation` tests the code samples in the documentation. In all cases only tests are run that do not require the presence of a GPU.

#### Code component test sessions

The idea is to run test sessions only when needed. This means that a set of tests are only run when the associated sources or the sources of a dependency change. For example, `eve` tests will not be run when only GT4Py cartesian sources are changed. The criteria to trigger tests sessions is encoded in the custom [`nox-sessions-config`](../../../nox-sessions-config.yml) configuration file, which follows a similar structure of the GitHub Actions `paths` and `paths-ignore` filters. The [`/scripts/python/nox_sessions.py`](../../../scripts/python/nox_sessions.py) script reads this file to decide which nox sessions are required for the existing changes from a base commit.

### Daily CI

There is an extra CI workflow on GitHub scheduled to run daily and testing `main` with different sets of requirements: the `highest` and the `lowest-direct` `uv` resolution strategies. Failures are accessible in [GitHub web interface](https://github.com/GridTools/gt4py/actions/workflows/daily-ci.yml) and as the 'Daily CI' badge in the main [README.md](../../../README.md) file. Additionally, in case of failure a message _might_ be posted in the [#ci-notifications](https://app.slack.com/client/T0A5HP547/C0E145U65) channel of the GridTols slack, but those notifications do not work reliably.

### AI-assisted PR review (PR Agent)

The `PR Agent` workflow runs [PR Agent](https://github.com/The-PR-Agent/pr-agent) (MIT-licensed, community-owned since Qodo donated it) to post an advisory review on new pull requests. It is **never a required status check** and is marked `continue-on-error`, so it cannot gate a merge. Review policy lives in [`.pr_agent.toml`](../../../.pr_agent.toml). The endpoint, model and token limits are set in the workflow rather than there, because PR Agent treats those as host-only settings that a pull request cannot override from its own branch; their values come from the repository variables listed below.

Maintainers can invoke tools on demand by commenting on a PR, e.g. `/review`, `/improve`, `/ask "<question>"` or `/describe`. Only `/review` runs automatically. `/describe` is not automatic for two reasons: a GT4Py PR title becomes the squash-commit message and must follow Conventional Commits, so an overwritten description would conflict with the pull request template; and it is the one command that rewrites PR metadata from model output, which is better left behind a human request.

#### Who can trigger it

Both paths that spend inference credit go through **one** authorization check: the `Authorize` step resolves the actor's *effective* repository permission and requires `write`, `maintain` or `admin`. The actor is the PR author for automatic review and the commenter for a command.

In practice that resolves to whoever currently holds `write` or better, which is granted from three places. Two of them are teams:

- **`gt4py-core`**, which has `admin` on this repository.
- **`gt4py-collaborators`**, which has `push`.

The third is anything granted outside those teams, and it matters because managing the two teams does not by itself determine who can spend inference credit:

- **Organization owners** hold `admin` on every repository in the organization, so they can trigger regardless of team membership — including any owner added later.
- **Direct repository collaborators** can be granted permission on this repository alone, bypassing both teams. One such grant currently exists, at `maintain`, held by the machine account used for CSCS-CI notification tokens (see [cscs-ci.md](cscs-ci.md)). Its GitHub account type is `User`, not `Bot`, so the workflow's `github.event.sender.type != 'Bot'` filter does **not** exclude it: a pull request authored by that account, or a `/`-prefixed comment from it, would spend inference credit.

Organization members with none of those grants resolve to `read` and cannot trigger anything. They can still open pull requests and comment as usual — only the bot is gated, and their PRs are reviewed when someone with write access comments `/review`. The organization's base repository permission is effectively `read`; were it ever raised to `write`, every member would silently gain trigger rights with nothing in this repository to signal it.

This check is what makes the automatic path safe. GitHub's fork-PR approval setting does not apply, because "[since the base branch is considered trusted, workflows triggered by these events will always run, regardless of approval settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories)". Without it, any stranger could spend the project's CSCS allocation by opening pull requests. An outside contributor's PR is reviewed when a maintainer asks with `/review`.

The check uses effective permission rather than `author_association`, for two reasons:

- It does not identify who it appears to. Webhook payloads report `MEMBER` only for *public* organization members; private members appear as `CONTRIBUTOR` or `NONE`. GridTools has exactly one public member, so an association gate would skip essentially every core developer — and with `continue-on-error` on a skipped job, silently.
- It is not an authorization decision even when it is accurate. An organization whose base permission is read still yields `MEMBER`, and `COLLABORATOR` includes outside collaborators with read-only or triage access. Such a principal could otherwise spend credit and, through `/describe`, have the bot rewrite a PR title and body they are not allowed to edit themselves.

The check **fails closed**: an API error denies rather than opening the bot up. It distinguishes the two cases, because a silent deny is how a broken gate would hide — "asked, and denied" is a `notice`, while "could not ask" is an `::error::` and a non-zero step. GitHub documents the collaborator-permission endpoint as accepting an installation access token with `Metadata: read`, which `GITHUB_TOKEN` always carries, so a 403 is not expected, but this has not been exercised live. If the bot goes quiet on both paths at once, read the `Authorize` step's log first.

No in-flight run is ever cancelled. Automatic reviews are serialized per pull request, and a pending one is replaced by a newer one; comment commands run independently of them and of each other. A concurrency group is claimed when a run is queued, *before* the job condition is evaluated, so every comment event joins a group even when its own job is then skipped. `cancel-in-progress` therefore stays off, or any comment — including the `cscs-ci run ...` ones that routinely land seconds after a PR opens — would kill that PR's in-flight review with no trace. GitHub also evicts an existing *pending* run when a newer one joins the same group, so comment-triggered runs are keyed on the comment id: a queued `/improve` cannot be evicted by a later unrelated comment. PR-event runs share one group per pull request, where a newer automatic review replacing a pending one saves an inference call. The default queue holds a single pending run, so two PR events still cost two reviews; a third evicts the second.

Draft PRs are skipped until marked ready. The Action's automatic path has no draft check of its own, so without the workflow's `draft == false` condition a draft PR would be reviewed once on `opened` and again on `ready_for_review`.

Unlike GitHub Copilot code review, which reads only `.github/` files and therefore needs the review rules restated in [`copilot-instructions.md`](../../../.github/copilot-instructions.md), PR Agent reads `AGENTS.md` directly through its `repo_context_files` setting. Prefer extending `AGENTS.md` over duplicating guidance into `.pr_agent.toml`.

#### Why `pull_request_target`

Most GT4Py pull requests come from forks, and GitHub gives fork PRs no access to repository secrets. A workflow triggered on `pull_request` could therefore never read the inference API key, and would fail on the majority of PRs. The workflow uses `pull_request_target` instead, which runs with secrets available.

The usual `pull_request_target` hazard — running untrusted code with secrets in scope — does not apply, because PR Agent reads the diff through the GitHub API and the job has no `actions/checkout` step. Two invariants keep that true, and both must be preserved:

- The job must never check out or execute pull request code.
- The job must never be granted `contents: write`. The model reads an untrusted diff, so a write permission would turn a prompt injection into a source-write primitive.

Withholding `contents: write` does not make the model's output inert. `issues: write` and `pull-requests: write` are needed to publish a review at all, and they also permit setting labels and rewriting a PR title and body — all from model output derived from untrusted PR text. Review labels are disabled and `/describe` is not run automatically; keep them that way. The two label settings are disabled in both `.pr_agent.toml` and the workflow environment, because `.pr_agent.toml` is fetched over the GitHub API and a failure there is only logged before PR Agent falls back to built-in defaults, which enable label writes. An environment override cannot fail to load and takes precedence, so it holds either way.

Note that both `pull_request_target` and `.pr_agent.toml` are read from the base (default) branch. A change to either has no effect until it is merged to `main`, which also means such a change cannot be exercised by the pull request that introduces it. The workflow carries a commented-out `pull_request` trigger for that case; as with `daily-ci.yml`, it only works for PRs from branches in the upstream repository.

#### Inference backend and cost

Inference runs on the [CSCS inference service](https://docs.cscs.ch/services/inference/) through its OpenAI-compatible API, using the `CSCS_INFERENCE_API_KEY` repository secret. Access is granted per CSCS project and billed as token-based resource consumption converted from node hours, so this spend competes with the project's compute allocation rather than being a separate subscription. The key should be created in the [inference UI](https://ui.inference.cscs.ch/login) with a token budget, a reset period, and a model restriction, so a runaway loop fails closed instead of draining the allocation. Store it alongside the CSCS-CI credentials in the internal CSCS credentials store.

Running inference at CSCS also keeps PR content on infrastructure CSCS controls: the service does not record prompts or model responses, and processing stays in Switzerland.

#### Configuration variables

The model, its token limits and the endpoint are [repository variables](https://github.com/GridTools/gt4py/settings/variables/actions), not hardcoded values, so they can be retuned from the repository settings without opening a PR. Each has a default in the workflow that applies when the variable is unset, so the workflow works as committed and a variable only needs to be set to override it.

| Variable                        | Default                            | Purpose                                                                                              |
| ------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `PR_AGENT_MODEL`                | **none — required**                | Reviewer model                                                                                       |
| `PR_AGENT_FALLBACK_MODELS`      | **none — required**                | JSON array, used when the primary model errors or returns a truncated response. `[]` for no fallback |
| `PR_AGENT_MODEL_CONTEXT_WINDOW` | `131072`                           | Context window `PR_AGENT_MODEL` is capable of                                                        |
| `PR_AGENT_TOKEN_BUDGET`         | `65536`                            | Context window we actually allow; the operative cap                                                  |
| `PR_AGENT_MAX_OUTPUT_TOKENS`    | `16384`                            | Headroom reserved for the answer                                                                     |
| `PR_AGENT_API_BASE`             | `https://api.inference.cscs.ch/v1` | Inference endpoint                                                                                   |

The two model variables have no workflow default on purpose: every model PR Agent may call is chosen explicitly in repository settings, so no inference is ever billed against a model nobody picked. The workflow checks both before starting the container and fails with an `::error::` naming the unset variable, rather than letting an empty model string fail inside it.

Both take an `openai/` prefix in front of the model id CSCS serves — `openai/zai-org/GLM-5.3`, not `zai-org/GLM-5.3`. The prefix is LiteLLM's selector for the OpenAI-compatible route and is not part of the model name: LiteLLM strips it and sends the remainder, so CSCS receives exactly the id that `GET /v1/models` lists. Without it there is no provider to route to and the call fails with `LLM Provider NOT provided`. PR Agent's documentation gives the same form, `model = "openai/<provider-model-id>"`.

All of these travel with `PR_AGENT_MODEL`: they are properties of the selected model, and changing the model without updating them is the way this gets misconfigured. The failure is quiet.

The three token variables in particular **interact, and must not be tuned one at a time**. PR Agent computes:

```text
context_window = min(PR_AGENT_MODEL_CONTEXT_WINDOW, PR_AGENT_TOKEN_BUDGET)
diff_budget    = context_window - PR_AGENT_MAX_OUTPUT_TOKENS - rendered_prompt_tokens
```

The diff packer evaluates that budget *unclamped*. An output reserve at or above the token budget therefore makes it negative and every file is skipped as "too large"; PR Agent logs `Empty diff for PR`, raises `No PR diff fits the /review request`, exhausts the fallback chain and fails. Since the job is `continue-on-error`, the PR shows nothing and the only trace is the run log — so keep a wide margin between `PR_AGENT_TOKEN_BUDGET` and `PR_AGENT_MAX_OUTPUT_TOKENS`. With the defaults above the diff budget starts at 49152 tokens, far more than a typical GT4Py PR needs.

`PR_AGENT_MODEL_CONTEXT_WINDOW` cannot be dropped: the model CSCS serves is absent from PR Agent's model table, and `PR_AGENT_TOKEN_BUDGET` "only caps a resolved limit; it cannot supply a missing model limit". Values of `PR_AGENT_MAX_OUTPUT_TOKENS` below 4096 are raised to 4096, with a warning in the run log.

The PR Agent version is **not** a variable. GitHub does not allow expressions in a step's `uses`, so it is pinned in the workflow, by digest with the release version in a comment. Upgrading is therefore a reviewed change, which is appropriate for the image that receives the inference key. It is the only version this workflow pins; it uses no other actions.

Note that fork PRs read variables from the base branch, as with everything else in this workflow, so a fork cannot substitute its own model or endpoint.

Two consequences of this backend are worth knowing before changing the model pin:

- **Every model CSCS serves is a reasoning model**, and reasoning tokens are drawn from the same output budget as the answer. `PR_AGENT_MAX_OUTPUT_TOKENS` must therefore be set explicitly; left unset the provider's own limit applies, and a review can come back truncated or empty, which presents as the bot posting nothing rather than as a clean error. Mind its interaction with `PR_AGENT_TOKEN_BUDGET` above.
- **CSCS rotates its model catalogue on short notice**, including retiring models that its own documentation examples still reference. `PR_AGENT_MODEL` needs periodic review against `GET /v1/models`. Service status is at [inference.status.cscs.ch](https://inference.status.cscs.ch).

## CSCS-CI

CI pipelines for all tests can be triggered via CSCS-CI. These automatically run from a Gitlab mirror for whitelisted users only, and have to be explicitly run by a whitelisted user via the comment "cscs-ci run default" on PRs from other users. There is currently no finegrained control over which subpackage tests are run. Neither can a subset be started manually from the comments nor can tests be skipped based on which files have been changed. Both are achievable (the latter with considerable effort), however given the current duration of the pipeline it does not seem worth doing so.

Since all tests routinely run here, this might be a better match for reintroducing test coverage in the future than GitHub workflows.

Additional information on how to change this process, such as adding whitelisted users, regenerating tokens etc can be found in [cscs-ci.md](cscs-ci.md)

## Tested operating systems

The testing workflows use a matrix strategy to run the automated tests on multiple operating systems. There is, however, a baseline 2000 hour monthly limit for the total time CI runs take. Since MacOS builds consume node hours 10 times as fast, this leaves an effective 200 hours for MacOS CI runs. MacOS is therefore only part of the Daily CI matrix, while the per-pull-request workflows run on Ubuntu only.

## Future improvements

- Reenable code coverage workflows (potentially on CSCS-CI).
- Split code quality: it might be better to run code quality tools separate for each project in the monorepo.
- Split documentation: once there is proper HTML documentation generated for the projects, it might make sense to have that run as one job per project.
- Template for tests: It would probably make sense to reuse some of the workflow descriptions for the tests.
