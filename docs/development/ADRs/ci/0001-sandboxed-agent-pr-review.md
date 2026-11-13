# Sandboxed agent PR review

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-11-13
- **Updated**: 2026-11-13

Why-statement: In the context of advisory AI review of pull requests, including PRs from forks, facing the need to let a tool-using coding agent explore untrusted PR content next to an inference credential, we decided to run the agent inside the AWF sandbox from a `pull_request_target` workflow split into four jobs, with the harness as a pluggable adapter, to achieve reviews that can read the whole repository while no secret is readable by the agent or by PR code. We considered running the agent directly on the runner, a `pull_request`-only workflow, and an external proxy with OIDC token exchange, and accept that a sandbox escape would expose a low-budget key, and that prompt injection can still steer the review's text.

## Context

[PR Agent](../../tools/pr-agent.md) reviews a PR from its diff, fetched through the API, with a single model call; it never sees the rest of the repository. A coding agent (opencode, memdoor, Pi, ...) can read related code, run `git log` and follow call sites, which is what a useful review of GT4Py needs. Such an agent executes tools on attacker-controlled content, so a prompt injection in a PR could make it read or send anything it can reach.

- Most GT4Py PRs come from forks, which get no secrets under `pull_request`; the inference key therefore requires `pull_request_target`, which runs with secrets.
- The CSCS inference key is billed against the project's allocation; a write-capable token could alter the repository.
- Agent harnesses are third-party code with install hooks and frequent releases, and which one works best with the CSCS models is not yet known.
- The agent reads instruction files (`AGENTS.md`, harness configuration) from the repository, which a PR can change.
- `.github/workflows/pr-agent.yml` must not be disturbed.

## Decision

We run the agent inside [AWF](https://github.com/github/gh-aw-firewall) (agentic workflow firewall): a container sandbox with a domain allowlist (only `github.com`) and an API-proxy sidecar that holds the real key and gives the agent placeholders. The workflow ([`agent-review.yml`](../../../../.github/workflows/agent-review.yml)) has four jobs, so that each secret lives where no untrusted code runs:

1. `authorize` checks that the PR author (automatic runs) or the labeller (`agent-review` label) has write access, fails closed, and pins the head sha.
2. `prepare` installs the pinned harness without any secret, so install hooks or a compromised release have nothing to steal.
3. `review` passes the key only to the `awf` step. The host never handles PR content: the PR is fetched by git inside the sandbox, the sandbox can write only to the review workspace, and the host is still treated as tainted after `awf` returns. Only two small regular files leave the job.
4. `publish` posts the comment from a fresh VM with a write token, no model and no PR code, and sanitises the review (mentions, HTML comments, size).

Everything the agent is instructed by (prompt, AWF configuration, harness, instruction files) comes from the workflow's own commit, never from the PR. The harness is an adapter with a small contract (`install.sh`, `run.sh`, `wire`) and is selected by a repository variable, one per run. Details and the invariants to keep are in [agent-review.md](../../tools/agent-review.md).

We chose AWF because it is the only option that provides credential isolation, egress control and per-run spend caps together, maintained and used by GitHub's own agentic workflows, without infrastructure of our own.

## Consequences

- A review can use the whole repository, not just the diff, and fork PRs are covered.
- Harnesses can be compared and replaced without changing the workflow.
- The design depends on AWF's isolation; a kernel or Docker escape exposes the key on the host. The key is dedicated, low-budget and quick to rotate; gVisor is a later hardening step.
- Prompt injection can still steer review content (false approvals, misleading findings). The review stays advisory, never a status check.
- Changes to the workflow can only be tested from a fork or a branch with a temporary `push` trigger, since `pull_request_target` reads the workflow from the default branch.
- AWF, its images and each harness are pinned and must be upgraded by reviewed PRs.

## Alternatives considered

### Agent directly on the runner

- Good, because it is simple and every harness supports it.
- Bad, because the agent would share the runner with the key and the token; one prompt injection exfiltrates them.

### `pull_request` only, no secrets for forks

- Good, because untrusted code never runs next to a secret.
- Bad, because most PRs come from forks and would get no review.

### External proxy with OIDC token exchange

- Good, because no long-lived key is ever on the runner, so even a sandbox escape exposes nothing durable.
- Bad, because it needs a service of our own to run and maintain; out of scope for a pilot.

### Extending PR Agent

- Good, because it already runs here.
- Bad, because PR Agent is not a tool-using agent; it cannot explore the repository.

## References

- [agent-review.md](../../tools/agent-review.md): usage, invariants, configuration, adapter contract.
- [pr-agent.md](../../tools/pr-agent.md): the existing diff-based reviewer.
- [AWF documentation](https://github.com/github/gh-aw-firewall)
