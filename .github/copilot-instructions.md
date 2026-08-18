<!--
Restates the review-relevant subset of AGENTS.md on purpose: Copilot code
review reads only .github/ files (from the PR base branch, ~4k chars each)
and never AGENTS.md, so a redirect here would propagate nothing. Keep short.
-->

# Code review rules

## Always flag

- New or changed code paths — branches, early returns, error/exception paths,
  new public functions — whose behaviour isn't covered by a test in the
  matching `tests/<subpackage>_tests/` tree. Tests are the spec here.
- Unhandled edge cases: empty or zero-dimensional fields, mismatched or
  non-promotable dimensions, off-by-one in domain / offset arithmetic,
  integer-vs-float dtype assumptions, and `None` / optional inputs.
- Missing error handling on I/O, filesystem, or subprocess calls (compiler /
  build invocation), where failure would surface as an opaque crash.
- A new runtime dependency (`pyproject.toml` `dependencies =`) without a
  matching ADR under `docs/development/ADRs/`.
- Secrets, tokens, hostnames, or per-developer absolute paths committed in
  source.

## Don't comment on

- Formatting or import order — `uv run ruff format` / `uv run ruff check` own
  that, and pre-commit + CI enforce it.
- Missing license headers — pre-commit inserts them automatically.
- Missing docstrings on private or internal helpers.
- Anything `tach`, `ruff`, or `mypy` already enforces deterministically; CI
  will catch it without a comment.

## Review style

- One comment per distinct issue; lead with the risk, then the fix.
- Cite evidence as `path/to/file.py:LINE`.
- When a PR changes architecture or adds an architecture-level feature,
  question the design and propose refactorings where warranted. For a
  localized fix, prefer the smallest change over a refactor.
- If a finding is a question rather than a defect, phrase it as one.
