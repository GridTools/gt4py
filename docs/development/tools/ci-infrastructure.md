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

Maintainers can invoke tools on demand by commenting on a PR, e.g. `/review`, `/improve`, `/ask "<question>"` or `/describe`. Only `/review` runs automatically. `/describe` is deliberately not automatic for two reasons: a GT4Py PR title becomes the squash-commit message and must follow Conventional Commits, so an overwritten description would conflict with the pull request template; and it is the one command that rewrites PR metadata from model output, which is better left behind a human request.

#### Who can trigger it

Every path that spends inference credit requires a trusted author, in two layers:

- **Automatic review** requires the PR author's `author_association` to be `OWNER`, `MEMBER` or `COLLABORATOR`. This is not merely belt-and-braces: GitHub's fork-PR approval settings do not apply here, because "[since the base branch is considered trusted, workflows triggered by these events will always run, regardless of approval settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories)". Without the check, any stranger could spend the project's CSCS allocation by opening pull requests. GT4Py's core developers are all `MEMBER` even when pushing from a personal fork, so automatic review still covers essentially all real traffic; an outside contributor's PR gets reviewed when a maintainer asks with `/review`.
- **Comment commands** additionally resolve the commenter's *effective* repository permission and require `write`, `maintain` or `admin`. `author_association` alone is not an authorization decision: `MEMBER` only proves organization membership, and an organization whose base permission is read still yields `MEMBER`; `COLLABORATOR` includes outside collaborators with read-only or triage access. Those principals could otherwise spend credit and, through `/describe`, have the bot rewrite a PR title and body they are not allowed to edit themselves. The check fails closed, so an API error disables comment commands but leaves automatic review working.

Unlike GitHub Copilot code review, which reads only `.github/` files and therefore needs the review rules restated in [`copilot-instructions.md`](../../../.github/copilot-instructions.md), PR Agent reads `AGENTS.md` directly through its `repo_context_files` setting. Prefer extending `AGENTS.md` over duplicating guidance into `.pr_agent.toml`.

#### Why `pull_request_target`

Most GT4Py pull requests come from forks, and GitHub gives fork PRs no access to repository secrets. A workflow triggered on `pull_request` could therefore never read the inference API key, and would fail on the majority of PRs. The workflow uses `pull_request_target` instead, which runs with secrets available.

The usual `pull_request_target` hazard — running untrusted code with secrets in scope — does not apply, because PR Agent reads the diff through the GitHub API and the job has no `actions/checkout` step. Two invariants keep that true, and both must be preserved:

- The job must never check out or execute pull request code.
- The job must never be granted `contents: write`. The model reads an untrusted diff, so a write permission would turn a prompt injection into a source-write primitive.

Withholding `contents: write` does not make the model's output inert, though, and it is worth being precise about that. `issues: write` and `pull-requests: write` are needed to publish a review at all, and they also permit setting labels and rewriting a PR title and body — all from model output derived from untrusted PR text. Those two paths are closed deliberately rather than by accident: review labels are disabled in `.pr_agent.toml`, and `/describe` is not run automatically. Keep them that way.

Note that both `pull_request_target` and `.pr_agent.toml` are read from the base (default) branch. A change to either has no effect until it is merged to `main`, which also means such a change cannot be exercised by the pull request that introduces it. The workflow carries a commented-out `pull_request` trigger for that case; as with `daily-ci.yml`, it only works for PRs from branches in the upstream repository.

#### Inference backend and cost

Inference runs on the [CSCS inference service](https://docs.cscs.ch/services/inference/) through its OpenAI-compatible API, using the `CSCS_INFERENCE_API_KEY` repository secret. Access is granted per CSCS project and billed as token-based resource consumption converted from node hours, so this spend competes with the project's compute allocation rather than being a separate subscription. The key should be created in the [inference UI](https://ui.inference.cscs.ch/login) with a token budget, a reset period, and a model restriction, so a runaway loop fails closed instead of draining the allocation. Store it alongside the CSCS-CI credentials in the internal CSCS credentials store.

Running inference at CSCS also keeps PR content on infrastructure CSCS controls: the service does not record prompts or model responses, and processing stays in Switzerland.

#### Configuration variables

The model, its token limits and the endpoint are [repository variables](https://github.com/GridTools/gt4py/settings/variables/actions), not hardcoded values, so they can be retuned from the repository settings without opening a PR. Each has a default in the workflow that applies when the variable is unset, so the workflow works as committed and a variable only needs to be set to override it.

| Variable                     | Default                            | Purpose                                                                        |
| ---------------------------- | ---------------------------------- | ------------------------------------------------------------------------------ |
| `PR_AGENT_MODEL`             | `openai/zai-org/GLM-5.3`           | Reviewer model                                                                 |
| `PR_AGENT_FALLBACK_MODELS`   | `["openai/Qwen/Qwen3.8-27B-FP8"]`  | JSON array, used when the primary model errors or returns a truncated response |
| `PR_AGENT_MODEL_MAX_TOKENS`  | `131072`                           | Context window of `PR_AGENT_MODEL`                                             |
| `PR_AGENT_MAX_OUTPUT_TOKENS` | `32768`                            | Output budget of `PR_AGENT_MODEL`                                              |
| `PR_AGENT_API_BASE`          | `https://api.inference.cscs.ch/v1` | Inference endpoint                                                             |

The two token limits travel with `PR_AGENT_MODEL`: they are properties of the selected model, and changing the model without updating them is the way this gets misconfigured. The failure is quiet — see the reasoning-token note below.

The PR Agent version is **not** a variable. GitHub does not allow expressions in a step's `uses`, so it is pinned in the workflow, by digest with the release version in a comment. Upgrading is therefore a reviewed change, which is appropriate for the image that receives the inference key. It is the only version this workflow pins; it uses no other actions.

Note that fork PRs read variables from the base branch, as with everything else in this workflow, so a fork cannot substitute its own model or endpoint.

Two consequences of this backend are worth knowing before changing the model pin:

- **Every model CSCS serves is a reasoning model**, and reasoning tokens are drawn from the same output budget as the answer. `config.max_output_tokens` must therefore be set explicitly; left at its default the provider's own limit applies, and a review can come back truncated or empty, which presents as the bot posting nothing rather than as a clean error.
- **CSCS rotates its model catalogue on short notice**, including retiring models that its own documentation examples still reference. `PR_AGENT_MODEL` needs periodic review against `GET /v1/models`; this is a maintenance cost that a commercial API pin would not carry. Service status is at [inference.status.cscs.ch](https://inference.status.cscs.ch).

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
