# AGENTS.md

Guidance for AI coding agents (Claude Code, OpenCode, and any tool that reads
`AGENTS.md`) working in the GT4Py repository. Keep this file short — it is loaded into
context every session. It is the **single source of truth**; the human-facing docs it
links to hold the full detail, so prefer linking over duplicating.

## Project overview

GT4Py (GridTools for Python) generates high-performance stencil-kernel implementations from
high-level Python. There are two subpackages with different maturity:

- `gt4py.cartesian` — **stable**, Cartesian-grid DSL.
- `gt4py.next` — **experimental**, the new (not backward-compatible) version.

Shared internals: `gt4py.eve` (visitor/IR toolkit), `gt4py._core`, `gt4py.storage`.
See [`README.md`](README.md) for the full description.

## Environment setup

The project is managed with [`uv`](https://docs.astral.sh/uv/) — **always** go through
`uv`, never bare `pip`/`pytest`/`python`.

```bash
uv sync --extra cartesian   # dev env for gt4py.cartesian
uv sync --extra next        # dev env for gt4py.next
# The `dev` dependency group (lint/test/typing/docs/...) is installed by default.
```

See [`README.md`](README.md) (Development Instructions) for details.

## Verify your work (default: fast, CPU-only loop)

Run these after making changes; fix and re-run until clean.

```bash
pre-commit run                       # ruff check+format, mypy, tach on staged files
pre-commit run -a                    # ...across the whole repo
uv run ruff check --fix <paths>      # targeted lint
uv run ruff format <paths>           # targeted format
uv run mypy src/                     # type check — src/ ONLY (matches the pre-commit hook)
uv run tach check                    # module-dependency boundaries
uv run pytest tests/<sub>_tests/... -x -q     # targeted tests (-n auto for parallel)
```

Subpackage test roots: `tests/cartesian_tests/`, `tests/eve_tests/`, `tests/next_tests/`,
`tests/storage_tests/`, `tests/package_tests/`.

## Full test matrix (opt-in / thorough)

CI runs [`nox`](https://nox.thea.codes/). It is heavy (Python 3.10–3.14 × backends) and some
sessions need a GPU (`requires_gpu`) that is unavailable here — those will skip or fail, which
is expected. Use it only when a change needs matrix-level confidence:

```bash
./noxfile.py --list
./noxfile.py -s "test_next-3.10(internal, cpu, nomesh)"
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (Testing) for the full workflow.

## Code style

- Line length **100**; Google-style docstrings (Sphinx/napoleon); `ruff` + `mypy` enforced.
- Respect the module-dependency layering in [`tach.toml`](tach.toml):
  `eve → _core → storage → {cartesian, next}` (a layer may not import a higher one).
- The BSD-3-Clause header is inserted automatically by pre-commit — don't add it by hand.
- The **full** rules (SOLID, naming, error messages, test layout) live in
  [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md). Read it before non-trivial changes.

## Commits & pull requests

- Conventional-commit style: `<type>[<scope>]: <summary>`.
  - types: `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `style`, `test`
  - scopes: `cartesian`, `eve`, `next`, `storage`
- PRs must include tests; new functionality must be documented; significant design changes
  need an ADR under [`docs/development/ADRs/`](docs/development/ADRs/).
- See [`.github/pull_request_template.md`](.github/pull_request_template.md) and
  [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Gotchas

- `mypy` runs on `src/` only (tests/docs/examples are excluded).
- Editing `pyproject.toml`/`uv.lock` triggers the `uv-lock` pre-commit hook — let it run.
- `markdown`/`toml`/`yaml` are auto-formatted by pre-commit; expect it to reformat new files.

## Custom commands

Shared slash commands are provided for both tools: `/qa`, `/test-changed`, `/typecheck`
(in `.claude/commands/` for Claude Code and `.opencode/command/` for OpenCode).

> Future extension (intentionally not set up yet, to stay lightweight): a SessionStart
> bootstrap hook and nested per-subpackage `AGENTS.md` files.
