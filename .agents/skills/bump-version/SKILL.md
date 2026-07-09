---
name: bump-version
description: Open a PR in GitHub that bumps the GT4Py version in three files and adds a CHANGELOG entry.
---

# bump-version

Bump the GT4Py version and open a PR with the change. This automates
the manual version-bump edits that have been done by hand for past
versions.

Every version bump makes exactly the same three edits, all carrying the
same new version number `X.Y.Z`:

1. **`CHANGELOG.md`** — add a new changelog block at the top.
2. **`pyproject.toml`** — bump the `versioningit` default version.
3. **`src/gt4py/__about__.py`** — bump the `on_build_version` fallback
   (the file's own comment says it MUST stay in sync with
   `pyproject.toml`).

## Inputs you need first

Ask the user (or confirm from context) before starting:

- **Version** to bump to, e.g. `1.1.12`. The project uses a CalVer-style
  sequence (`1.1.N`); the natural next value is `previous + 1`. Confirm
  the bump with the user if it is anything other than a simple `+1`.
  The **current version must always be read from the latest release on
  PyPI**, not from the local working tree or branch. Before determining
  the target version, query PyPI for the latest `gt4py` release. This
  keeps the bump idempotent: running the skill twice locally will compute
  the same target version both times.
- **Date** in ISO form `YYYY-MM-DD` (this becomes the
  `## [X.Y.Z] - YYYY-MM-DD` line). Default to today; confirm if the user
  wants a different date.
- **Changelog entries** for the new version. Collect these from the
  user or from the commit history since the last bump. Group them under
  the section headings below. Each bullet should read like a release note for
  GT4Py users: describe the impact or benefit and avoid implementation details,
  internal class names, or backend-specific mechanics.

If you cannot obtain the version or the changelog content, stop and ask
— do not invent changelog entries.

## Categorizing changes (General / Cartesian / Next)

For each merged PR since the last version bump, decide which section —
`General`, `Cartesian`, or `Next` — it belongs to. Two signals, in
order of reliability:

1. **PR title scope** (Conventional Commits prefix). This is the
   primary signal.
   - `feat[next]`, `fix[next]`, `perf[next]`, `refactor[next]`,
     `test[next]`, `docs[next]`, `build[next]`, `feat[next-dace]`,
     `fix[next-dace]` … → **Next**.
   - `feat[cartesian]`, `fix[cartesian]`, `perf[cartesian]`,
     `refactor[cartesian]`, `test[cartesian]`, `build[cartesian]`,
     `ci[cartesian]` … → **Cartesian**.
   - `fix[eve]`, `feat[eve]`, `refactor[eve]`, `perf[eve]` → **General**.
   - `ci:`, `build:` (no scope), `chore:` → **General**.
2. **Code location** (the files the PR touches), as a tie-breaker when
   the title is missing or ambiguous:
   - `src/gt4py/cartesian/` → **Cartesian**.
   - `src/gt4py/next/` → **Next**.
   - `src/gt4py/eve/` → **General** (the `eve` IR framework is shared;
     its changes are never filed under `Cartesian` or `Next`).

### What goes into `### Cartesian`

User-visible changes to `gt4py.cartesian` — stencils, backends (numpy,
dace, the C++/GPU backends), the cartesian frontend, and the build
configuration of those backends. Real examples from past versions:

- A new backend or backend variant (e.g. a DaCe GPU `IJK`-loop layout).
- Backend behaviour changes (memory layout/alignment, loop nesting,
  compiler flags, OpenMP toggles, DaCe version bumps for cartesian).
- Frontend changes that affect stencil authoring (interval parsing,
  field indexation, region syntax).
- Fixes to code generation, caching, or array lifetime in cartesian
  backends.
- Dependency bumps that affect the cartesian backends specifically
  (e.g. `pybind11`).

`build[cartesian]` and `ci[cartesian]` PRs go here too when they change
user-visible build/compilation behaviour (compiler defaults, supported
CUDA/ROCm versions, default flags). Pure CI plumbing that doesn't
affect how users build cartesian (e.g. "restore OpenMP for macos on
daily CI") stays in `### General`.

### What goes into `### General`

Repo-wide changes that don't belong to a single subpackage:

- **CI**: changes to GitHub Actions, workflow files, daily CI job
  selection/timing, runner config (e.g. "update `actions/checkout`",
  "move daily job into the European night", Kubernetes memory limits).
- **`eve` package**: any `feat[eve]` / `fix[eve]` / `perf[eve]` /
  `refactor[eve]` change, because `eve` is the shared IR framework used
  by both `cartesian` and `next` (e.g. "use identity check for None in
  `infer_type`", "`lru_cache` should not call `__eq__`", performance
  optimizations in `eve.visitors`).
- **Repo-wide infrastructure**: versioning, Python version support,
  top-level dependency floors, the coding-agents harness, dev-scripts
  infrastructure (e.g. "Drop support for NumPy 1.x", "Add support for
  Python 3.14", "Add minimal harness for coding agents").

When in doubt between General and a subpackage, ask: *does this change
apply to the whole repo or to the `eve` shared layer?* If yes → General.
Does it apply only to how `cartesian` builds or runs? → Cartesian.

### What goes into `### Next`

Everything touching `gt4py.next` — and the `### Next` section in the
changelog is **always** the fixed placeholder `See commit history.`,
regardless of how many next PRs landed. The `next` package changes too
quickly and is not summarized per version; interested readers are sent
to the commit history.

So you must still **review** the `next` PRs (to confirm they are not
mis-scoped and that none of them actually belongs in `### General` or
`### Cartesian`), but you do **not** write bullets for them.

### Exclude from the changelog entirely

Do not list a PR in any section when it is purely internal and has no
user-visible effect:

- **Pure refactoring** with no behaviour change
  (`refactor[...]` PRs that only rename, reorganize, or simplify code,
  e.g. "drop cartesian subscript syntax", "descriptive loop names in
  dace backend", "replace `apply_to_primitive_constituents` with
  `tree_map_type`"). If a `refactor` does change user-visible
  behaviour, file the behaviour under the right section instead.
- **Pure test code** (`test[...]` PRs that only add, fix, move, or
  reorganize tests, e.g. "Fix `fvm_nabla_setup` for daily CI",
  "simplify a few style outliers", "copy over `as_offset` tests").
  This includes test-tree hygiene, fixture relocations, and daily-CI
  test selection fixes.
- **Pure docs/chore** with no user-facing impact
  (e.g. internal doc rewrites, `chore` housekeeping), unless the docs
  are user-facing and worth advertising.

When a PR mixes categories (e.g. a `fix[cartesian]` that also tweaks a
CI workflow), file it under the section matching its *primary,
user-visible* effect — usually the subpackage scope — and ignore the
incidental part.

## Step 1 — `CHANGELOG.md`

The file begins with a two-line header:

```
# GT4Py Changelog

Notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
```

Insert a new changelog block **immediately after that header line and
its blank line**, before the previous release block. The block format is:

```
## [X.Y.Z] - YYYY-MM-DD

### General

- <bullet, if any General changes>

### Cartesian

- <bullet, one per change>
- <another change>

### Next

See commit history.
```

The `### General` and `### Cartesian` bullets are filled from the
rules in **Categorizing changes** above (review every merged PR since
the last tag; assign each to `General`, `Cartesian`, or `Next`, or
exclude it entirely if it's pure refactoring / pure test code).

Rules, observed in every past version:

- The three section headings appear **in this order**: `### General`,
  `### Cartesian`, `### Next`.
- `### General` is optional. Omit it entirely when there are no
  General changes; do not leave an empty heading.
- `### Cartesian` lists the user-visible changes as one bullet per
  line, present-tense imperative-ish phrasing
  (e.g. "Add ...", "Change ...", "Fix ...").
- `### Next` is **always** exactly:
  ```
  ### Next

  See commit history.
  ```
  Never expand it. The `next` package is in active development and its
  changes are not summarized in the changelog.
- A blank line separates the header line from the new block, a blank
  line separates each heading section, and a blank line separates the
  new block from the previous one.

## Step 2 — `pyproject.toml`

Under the `[tool.versioningit]` table (search for `# -- versioningit --`
near line 483), set:

```toml
[tool.versioningit]
default-version = "X.Y.Z+unknown.version.details"
```

Only the version number changes; the `+unknown.version.details` suffix
stays exactly as-is.

## Step 3 — `src/gt4py/__about__.py`

Near line 39, set:

```python
on_build_version: Final = "X.Y.Z+unknown.version.details"
```

Again only the version number changes; the suffix is identical to the
one in `pyproject.toml`. The two values **must match** — the docstring
above the line explicitly requires this, and the two are always
updated in lockstep.

## Step 4 — verify the three edits agree

Before opening the PR, confirm all three occurrences of the version are
identical:

- `CHANGELOG.md`: `## [X.Y.Z] - ...`
- `pyproject.toml`: `default-version = "X.Y.Z+unknown.version.details"`
- `src/gt4py/__about__.py`: `on_build_version: Final = "X.Y.Z+unknown.version.details"`

A mismatch between `pyproject.toml` and `__about__.py` is the single
most likely mistake; check it explicitly.

## Step 5 — branch, commit, push, open PR

This project uses **squash-merge**, and version-bump PRs do **not**
follow the Conventional Commits convention used elsewhere. So:

- **Branch name**: `releasing_vX.Y.Z` — note the `v` prefix and the
  underscore (not a hyphen).
- **Commit message / PR title**: `Releasing vX.Y.Z` — note the `v`
  prefix in the title, even though the version string inside the files
  has **no** `v` prefix.
- **PR body**: a short body like `Bump version to X.Y.Z and update CHANGELOG.` is fine but not required.
- **Target**: `main` (per AGENTS.md, all PRs target `main`).

If a PR or local branch already exists for `releasing_vX.Y.Z`, do not
open a duplicate. Instead, add a new commit on the existing branch with
the three file edits and push it; the existing PR will pick up the new
commit automatically.

After staging the three edits, run pre-commit on them before pushing:

```
uv run pre-commit run
```

This is the same formatting/QA pass every other Claude edit gets
(license header, ruff format, trailing whitespace, etc.). The bump
touches three small files, so this is fast and there is no reason to
skip it. Do **not** run the full `uv run pre-commit run -a` or the
test suites — they are unnecessary for pure version metadata (see
"Things this skill deliberately does NOT do").

Use `gh` to create the PR. Do **not** force-push to `main` and do **not**
merge the PR yourself unless explicitly asked — version-bump PRs in
this project have been merged by maintainers.

## Things this skill deliberately does NOT do

- It does not create or push a git **tag** object. This skill only
  bumps the version files and opens the PR; tagging happens separately
  by maintainers after merge. Do not create tags unless the user
  explicitly asks.
- It does not run `uv sync` or the test suites (`nox`, `pytest`).
  The edits are pure version metadata and don't touch code under test,
  so a full environment sync and the per-subpackage test sessions
  (required for real code changes per AGENTS.md) are unnecessary here.
  Pre-commit is **not** optional, though — run it as described in
  Step 5.
- It does not change the copyright year in `__about__.py`
  (`Copyright (c) 2014-2024 ETH Zurich`) — that is not part of the
  version bump.
