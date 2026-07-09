---
name: bump-version
description: Open a PR in GitHub that bumps the GT4Py version in three files and adds a CHANGELOG entry.
---

# bump-version

Bump the GT4Py version and open a PR. Makes exactly three edits with the same
new version `X.Y.Z`:

1. `CHANGELOG.md` — add a new release block at the top.
2. `pyproject.toml` — bump `tool.versioningit.default-version`.
3. `src/gt4py/__about__.py` — bump `on_build_version` to match.

## Inputs

Confirm before starting:

- **Version** to bump to, e.g. `1.1.12`. The project uses a CalVer-style
  sequence (`1.1.N`); the natural next value is `previous + 1`. Read the
  latest release from PyPI, not the local tree, to keep the bump idempotent.
  If a `releasing_v*` branch or an open "Releasing v\*" PR already exists,
  use the version from that in-flight release instead of PyPI. Confirm with
  the user if the bump is not a simple `+1`.
- **Date** in ISO form `YYYY-MM-DD`. Default to today.
- **Changelog entries** grouped under `General`, `Cartesian`, and `Next`.
  Do not invent entries. Each bullet should read like user-facing release
  notes: describe the impact or benefit and avoid implementation details,
  internal class names, or backend-specific mechanics.

## Categorizing changes

Use the PR title scope as the primary signal; fall back to the files touched.

| Scope                                                                                                                                          | Section         |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `feat[next]`, `fix[next]`, `perf[next]`, `refactor[next]`, `test[next]`, `docs[next]`, `build[next]`, `feat[next-dace]`, `fix[next-dace]`, ... | `### Next`      |
| `feat[cartesian]`, `fix[cartesian]`, `perf[cartesian]`, `refactor[cartesian]`, `test[cartesian]`, `build[cartesian]`, `ci[cartesian]`, ...     | `### Cartesian` |
| `fix[eve]`, `feat[eve]`, `refactor[eve]`, `perf[eve]`, `ci:`, `build:`, `chore:`                                                               | `### General`   |

File-location tie-breakers:

- `src/gt4py/cartesian/` → `### Cartesian`
- `src/gt4py/next/` → `### Next`
- `src/gt4py/eve/` → `### General`
- `src/gt4py/storage/` → `### General`

### Section contents

- `### General` — repo-wide changes: CI plumbing, `eve`, Python/NumPy support,
  top-level dependency floors, dev-infra, coding-agents harness.
- `### Cartesian` — user-visible `gt4py.cartesian` changes: stencils,
  backends, frontend syntax, code generation, caching, cartesian dependency
  bumps, and user-visible build/CI changes for cartesian.
- `### Next` — **always** exactly:
  ```
  ### Next

  See commit history.
  ```
  Review next PRs only to ensure none belong in `General` or `Cartesian`;
  do not write bullets for them.

Exclude entirely when purely internal with no user-visible effect:

- Pure refactoring with no behaviour change.
- Pure test code additions/fixes.
- Pure docs/chore with no user-facing impact.

When a PR mixes categories, file it under its primary user-visible effect.

## Step 1 — `CHANGELOG.md`

Insert a new block immediately after the two-line file header and before the
previous release:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### General

- <bullet, if any>

### Cartesian

- <bullet, one per change>
- <another change>

### Next

See commit history.
```

Rules:

- Headings appear in this order: `General`, `Cartesian`, `Next`.
- `General` is optional; omit it if empty.
- `Next` is always the fixed placeholder above.
- Use present-tense imperative-ish phrasing for bullets.
- Keep bullets user-focused, not implementation-focused.
- Blank lines separate the header, each section, and the previous release.

## Step 2 — `pyproject.toml`

Under `[tool.versioningit]`, set only the version number; keep the suffix:

```toml
[tool.versioningit]
default-version = "X.Y.Z+unknown.version.details"
```

## Step 3 — `src/gt4py/__about__.py`

Set the matching value:

```python
on_build_version: Final = "X.Y.Z+unknown.version.details"
```

## Step 4 — verify

Confirm all three occurrences use the same `X.Y.Z`:

- `CHANGELOG.md`: `## [X.Y.Z] - ...`
- `pyproject.toml`: `default-version = "X.Y.Z+unknown.version.details"`
- `src/gt4py/__about__.py`: `on_build_version: Final = "X.Y.Z+unknown.version.details"`

## Step 5 — branch, commit, push, PR

- **Branch**: `releasing_vX.Y.Z`
- **Commit / PR title**: `Releasing vX.Y.Z` (note the `v` prefix here only)
- **PR body**: `Bump version to X.Y.Z and update CHANGELOG.` (optional)
- **Target**: `main`

If `releasing_vX.Y.Z` already exists, commit on that branch and push; do not
open a duplicate PR.

Stage the three changed files, then run:

```
uv run pre-commit run
```

Do not run the full `-a` suite or the test suites for a pure metadata bump.
Do not create a git tag unless explicitly asked. Do not merge the PR yourself.
