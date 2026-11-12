# PR Agent

The `PR Agent` workflow ([`.github/workflows/pr-agent.yml`](../../../.github/workflows/pr-agent.yml)) runs [PR Agent](https://github.com/The-PR-Agent/pr-agent) on the [CSCS inference service](https://docs.cscs.ch/services/inference/) to post an advisory review on pull requests. It is **never a required status check**: if the review step fails or times out, the run stays green with a warning annotation, and the cause is in the `Run PR Agent` log. A failed manual `/review` also reports the reason on the PR.

## Usage

- `/review` runs automatically when a non-draft PR is opened, reopened or marked ready, if its author has write access. PRs titled `Releasing v…` (releases) or `[Auto] …` are skipped, and comment commands on them are ignored too: the run stays green and posts nothing.
- Users with write access can comment `/review`, `/improve`, `/ask "<question>"` or `/describe` on any PR, including one from a fork. `/describe` is manual because it rewrites the PR body, which here is the filled-in PR template and becomes the squash-commit message.
- Write access comes from the `gt4py-core` and `gt4py-collaborators` teams, from organization ownership, and from direct collaborators such as the CSCS-CI machine account (see [cscs-ci.md](cscs-ci.md)). That account is of type `User`, so the workflow's bot filters do not exclude it.

The review policy is in [`.pr_agent.toml`](../../../.pr_agent.toml). The review rules are in [`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md), shared with GitHub Copilot's reviewer, followed by `AGENTS.md` and `src/gt4py/next/AGENTS.md`. Extend those files rather than adding guidance to `.pr_agent.toml`.

## Security invariants

The workflow uses `pull_request_target` because most GT4Py PRs come from forks, which get no repository secrets under `pull_request`. Keep these when changing it:

- **Never check out or run PR code** (no `actions/checkout`), and never grant `contents: write`. PR Agent reads the diff through the GitHub API.
- **The `Authorize` step is the only gate.** It requires write access from the PR author (automatic review) or the commenter (commands), and fails closed. `pull_request_target` ignores the fork-approval setting, so without it anyone could spend the CSCS allocation.
- **Review labels stay disabled and `/describe` manual**, since their output is derived from untrusted PR text. Those settings are in the workflow environment, not `.pr_agent.toml`, so a failed fetch of the policy file cannot re-enable them.
- **The image is pinned by digest**, since it receives the inference key. Upgrading is a reviewed change.

The workflow and `.pr_agent.toml` are read from `main`, so changes take effect only once merged. To test changes from a branch in this repository, re-enable the blocks marked `COMMENTED OUT` in the workflow (with `PR_AGENT_CONFIG_BRANCH` set to the literal branch name, never one derived from PR input) and disable them again before merging. The files in `repo_context_files` are always read from `main`, so changes to the review rules cannot be tested this way. A run that applied a policy file logs `Applying repo settings (sections: […])`; the bare `Applying repo settings` appears in every run.

## Configuration variables

The key is the `CSCS_INFERENCE_API_KEY` repository secret; its usage is billed against the CSCS project's compute allocation. Create it in the [inference UI](https://ui.inference.cscs.ch/login) with a token budget, a reset period and a model restriction, and store it with the CSCS-CI credentials in the internal CSCS credentials store. The rest are [repository variables](https://github.com/GridTools/gt4py/settings/variables/actions), changeable without a PR:

| Variable                     | Default                            | Purpose                                                     |
| ---------------------------- | ---------------------------------- | ----------------------------------------------------------- |
| `PR_AGENT_MODEL`             | **required**                       | CSCS model id, as listed by `GET /v1/models`                |
| `PR_AGENT_FALLBACK_MODELS`   | **required**                       | Comma-separated ids, tried in order; `none` for no fallback |
| `PR_AGENT_TOKEN_BUDGET`      | `262144`                           | Context window PR Agent assumes                             |
| `PR_AGENT_MAX_OUTPUT_TOKENS` | `100000`                           | Tokens reserved for the answer, reasoning included          |
| `PR_AGENT_AI_TIMEOUT`        | `720`                              | Seconds one model may take before the next one is tried     |
| `PR_AGENT_API_BASE`          | `https://api.inference.cscs.ch/v1` | Inference endpoint                                          |

- Model ids are stored without the `openai/` prefix, which the workflow adds. CSCS retires models on short notice, so check them against the [model table](https://docs.cscs.ch/services/inference/api/).
- The token limits apply to every model in the chain, so they must fit the smallest one: the output reserve is sent as `max_tokens`, which a model with a lower output limit rejects. Check a fallback before adding it, since it is only reached when the models before it fail. Too small an output reserve makes reasoning models return empty responses.
- The `Run PR Agent` step limit is derived as one `PR_AGENT_AI_TIMEOUT` per model plus two minutes; `Resolve models` rejects a chain that does not fit in the job's `timeout-minutes` with two minutes to spare.
- The review names the model that produced it, so a fallback answer is visible.
