# GT4Py — Agent Instructions

> README for AI coding agents. Closest `AGENTS.md` to the file being edited
> wins (only this root one exists today).

GT4Py is the GridTools toolchain for performance-portable weather and climate
DSLs. It exposes two embedded Python DSLs — `gt4py.cartesian` (stencil
programming on regular grids) and `gt4py.next` (iterator IR for unstructured /
mixed grids) — built on the `gt4py.eve` IR framework, with C++, CUDA, ROCm,
DaCe, and pure-Python backends.

## Stack

- Language: **Python 3.10–3.14** (see `.python-versions`).
- Environment / dependencies: **`uv`** (lockfile is `uv.lock`).
- Test runner: **`nox`** (sessions in `noxfile.py`).
- QA: **`pre-commit`** (config in `.pre-commit-config.yaml`) — runs `ruff`
  (lint + format), `mypy`, `tach` (subpackage import enforcement), license
  header insertion, markdown/TOML/YAML formatters, and `uv lock` validation.
- Build backend: `setuptools` with native extensions (Cython / pybind11 /
  C++); see `pyproject.toml`.

## Commands (prefer these over guessing)

- `uv sync` — create / refresh the dev environment.
- `uv run pre-commit run` — full QA on staged files (fast).
- `uv run pre-commit run -a` — full QA on the whole tree (slow; mypy + tach
  dominate).
- `uv run nox --list` — see all test sessions.
- `uv run nox -s "test_<subpackage>-<py>(...)"` — run one suite, e.g.
  `test_cartesian`, `test_next`, `test_eve`, `test_storage`, `test_package`,
  `test_examples`, `test_typing_exports`.
- `uv run pytest <path>` — targeted unit runs during development.

If a command above is wrong for your environment, fix `pyproject.toml`,
`noxfile.py`, or `.pre-commit-config.yaml` — not this file.

## Where things live (capabilities, not paths)

- Contributor workflow (forking, PRs, merge rules):
  [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Code style, error messages, docstrings, module structure, test layout:
  [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md) — **authoritative**. There
  is no separate `docs/style.md`.
- Architecture Decision Records:
  [`docs/development/ADRs/`](docs/development/ADRs/), organized per
  subsystem. Add new ADRs there, not in a flat `docs/adr/`.
- Dev-environment setup and CI infrastructure:
  [`docs/development/`](docs/development/) (onboarding, CI tooling, CSCS
  HPC integration).
- User-facing docs: [`docs/user/cartesian/`](docs/user/cartesian/) and
  [`docs/user/next/`](docs/user/next/).
- Subpackage dependency DAG (enforced in CI): [`tach.toml`](tach.toml).
- Project metadata, dependency groups, ruff / mypy / coverage / pytest
  config: [`pyproject.toml`](pyproject.toml).
- License header to prepend to new source files: [`HEADER.txt`](HEADER.txt)
  (pre-commit inserts it automatically).

## Do

- Run `uv run pre-commit run` on staged files before claiming a task done.
- When touching `gt4py.cartesian`, `gt4py.next`, `gt4py.eve`, `gt4py.storage`,
  or `gt4py._core`, run the matching `nox -s test_<subpackage>` session
  before opening the PR.
- For a non-trivial architectural choice (new dependency, IR change,
  backend interface, public-API shape), add an ADR under the appropriate
  `docs/development/ADRs/<subsystem>/` directory. ADRs are append-only;
  supersede with a new file.
- Respect `tach.toml` — don't introduce subpackage imports that violate the
  declared DAG. If you genuinely need a new edge, edit `tach.toml`
  explicitly and explain why in the PR.
- For first-time contributors, add them to [`AUTHORS.md`](AUTHORS.md).

## Don't

- Don't edit anything under a `generated/` directory — it's overwritten by
  the codegen pipeline.
- Don't add a runtime dependency (`pyproject.toml` `dependencies =`)
  without an ADR if it affects architecture or supported platforms.
- Don't run destructive Git on shared branches: `push --force`,
  `reset --hard origin/*`, history rewrites.
- Don't put secrets, hostnames, or per-developer paths in this file — they
  belong in a git-ignored file (e.g. `AGENTS.local.md` or
  `CLAUDE.local.md`).
- Don't auto-expand this file. The instruction budget is finite (~200 lines);
  push deeper guidance into `CODING_GUIDELINES.md`, an ADR, or
  `docs/development/`.

## Conventions

- **Commits**: gt4py uses **squash-merge**, so the convention applies to the
  **PR title** (which becomes the commit landing on `main`).
  [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
  is encouraged (see CONTRIBUTING.md §6). The type list in active use:
  `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `style`,
  `test` (historically also `feature`). Scopes follow the subpackage where
  applicable: `feat[next]`, `fix[cartesian]`, `refactor[next-dace]`, etc.
- **Branch names**: free-form. PRs should target `main`.
- **PR description**: state intent and main changes; this text doubles as
  the squash-commit message.
- **Tests are the spec.** If you change behaviour, change a test first or
  alongside, in the matching `tests/<subpackage>_tests/` tree.

## Local overrides

Per-machine notes that shouldn't be committed go in `CLAUDE.local.md` or
`AGENTS.local.md` (gitignored). Do not promote them into this file.
