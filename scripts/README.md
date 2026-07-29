# GT4Py dev-scripts toolbox

Standardized conventions for GT4Py's Python, Bash, and other development
scripts. Every script is a standalone executable; `./scripts/run` is an
optional unified entry point that auto-discovers them as sub-commands.

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — runs Python scripts in isolated
  environments via a `uv run` shebang. Required for everything here.
- **[bats-core](https://github.com/bats-core/bats-core)** — only needed to run
  the Bash test suite.

## Layout

```
scripts/
├── run                    # Unified entry point (uv shebang)
├── test                   # Unified test runner (pytest + bats)
├── python/                # Python package with importable modules
│   ├── helpers/           # Shared utilities and common definitions
│   └── <name>.py          # Individual scripts, each with a `cli` Typer app
├── sh/                    # Bash scripts
│   ├── _lib.sh            # Sourced shared functions (not a sub-command)
│   └── <name>.sh          # Individual executable scripts
├── tests/                 # Test suites
│   ├── python/            # pytest tests (run from `scripts/python`)
│   └── sh/                # bats tests
└── data/                  # Supporting data files for scripts
```

## Conventions

**Naming**

- A leading underscore (`_`) marks shared infrastructure or a private utility
  (e.g. `helpers/`, `sh/_lib.sh`); these are **not** discovered as
  sub-commands.
- Every other Python module and `*.sh` script is auto-discovered. Underscores
  in shell script names become hyphens in the command name.

**Python scripts**

- Start with the uv shebang for automatic dependency isolation:
  `#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3`
- Expose a global `cli = typer.Typer(...)` app.
- Call it under `if __name__ == "__main__": cli()` so the module also runs
  standalone.
- Keep dependencies in the `scripts` group of the root `pyproject.toml`.
- Shared logic goes in `python/helpers/` (e.g. `helpers.common` exposes
  `REPO_ROOT`, `SCRIPTS_DIR`, `PY_SCRIPTS_DIR`, `PYTHON_VERSIONS`).

**Bash scripts**

- Set strict mode and source `_lib.sh` for shared helpers (`_lib.sh` does not
  set shell options itself):
  ```bash
  set -euo pipefail
  source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
  ```
- Add `# [help] ...` comment lines to customize the help text shown by
  `./scripts/run`.

## Usage

```bash
./scripts/run --help               # list all sub-commands
./scripts/run <sub-command> --help # help for one sub-command

./scripts/test python              # run python script tests (pytest)
./scripts/test python -- -k common # forward args to pytest after `--`
./scripts/test sh                  # run bash script tests (bats)
```
