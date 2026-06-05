# GT4Py — Agent Instructions

GT4Py is a Python library for performance-portable weather and climate
computations using the GridTools framework. It exposes two embedded
Python DSLs: `gt4py.cartesian` (stencil programming on structured Cartesian
grids) and `gt4py.next` (supports both structured and unstructured meshes).
The toolchain infrastructure is defined in the internal `gt4py.eve` IR framework,
and the implementation relies on GridTools, DaCe, or pure-Python backends
dispatching operation to standard Python array libraries.

Editing under `src/gt4py/next/` or `tests/next_tests/`? Those carry extra
conventions in [`src/gt4py/next/AGENTS.md`](src/gt4py/next/AGENTS.md).

## Stack

- Language: **Python 3.10–3.14** (see `.python-versions`).
- Environment / dependencies: **`uv`** (lockfile is `uv.lock`) — always go
  through `uv`, never bare `pip` / `python`.
- Test runner: **`nox`** (sessions in `noxfile.py`).
- QA: **`pre-commit`** (config in `.pre-commit-config.yaml`) — runs `ruff`
  (lint + format), `mypy`, `tach` (import-boundary enforcement), plus file
  formatters and license-header insertion.
- Build backend: `setuptools.build_meta`; `cython` is the only compiled
  build-time extension. The C++/CUDA/DaCe code that backends emit is
  JIT-compiled at runtime, not at install (see `pyproject.toml`).

## Commands (prefer these over guessing)

- `uv sync` — create / refresh the dev environment.
- `uv run pre-commit run` — full QA on staged files (fast).
- `uv run pre-commit run -a` — full QA on the whole tree (slow; mypy + tach
  dominate).
- `uv run mypy src/` — type check (mypy runs on `src/` only).
- `uv run tach check` — enforce the subpackage import DAG.
- `uv run nox --list` — see all test sessions.
- `uv run nox -s "test_<subpackage>-<py>(...)"` — run one suite (names from
  `nox --list`), e.g. `test_next`.
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
- `gt4py.next`-specific conventions and test framework:
  [`src/gt4py/next/AGENTS.md`](src/gt4py/next/AGENTS.md).
- Dev-environment setup and CI infrastructure:
  [`docs/development/`](docs/development/).
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

- Don't add a runtime dependency (`pyproject.toml` `dependencies =`)
  without an ADR if it affects architecture or supported platforms.
- Don't run destructive Git on shared branches: `push --force`,
  `reset --hard origin/*`, history rewrites.
- Don't put secrets, hostnames, or per-developer paths in this file — they
  belong in a git-ignored file (e.g. `AGENTS.local.md` or
  `CLAUDE.local.md`).
- Don't auto-expand this file. The instruction budget is finite (~200 lines);
  push deeper guidance into `CODING_GUIDELINES.md`, an ADR, a nested
  `AGENTS.md`, or `docs/development/`.

## Conventions

- **Commits**: gt4py uses **squash-merge**, so the convention applies to the
  **PR title** (which becomes the commit landing on `main`).
  [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
  is encouraged (see CONTRIBUTING.md §6). Types in active use: `build`, `ci`,
  `docs`, `feat`, `fix`, `perf`, `refactor`, `style`, `test`. Scopes follow
  the subpackage: `feat[next]`, `fix[cartesian]`, `refactor[next-dace]`, etc.
- **Branch names**: free-form. PRs should target `main`.
- **Tests are the spec.** If you change behaviour, change a test first or
  alongside, in the matching `tests/<subpackage>_tests/` tree.

## Local overrides

Per-machine notes that shouldn't be committed go in `CLAUDE.local.md` or
`AGENTS.local.md` (gitignored). Do not promote them into this file.
