#!/usr/bin/env python3
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Bootstrap icon4py into the *currently activated* venv (the gt4py CI venv).

Edits icon4py's `pyproject.toml` so that `[tool.uv.sources]` points
`gt4py` (and optionally `dace`) at local-path editable installs, regenerates
the lockfile, and runs `uv sync --active` to install icon4py + its other
dependencies into `$VIRTUAL_ENV`.

This is what makes the editable gt4py / dace branches survive everything
downstream — including the icon4py noxfile's own `uv sync` call when our
dace_deterministic_codegen harness runs `nox --no-venv`.

Usage (run from anywhere):

    python ci/dace_deterministic_codegen/bootstrap_icon4py.py \\
        --icon4py /path/to/icon4py \\
        --gt4py   /path/to/gt4py-dace_toolchain_deterministic \\
        --dace    /path/to/dace                  # optional

Idempotent: re-running it is safe; the [tool.uv.sources] entries are
overwritten in place.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found]

try:
    import tomli_w
except ModuleNotFoundError:
    print(
        "error: this script needs `tomli_w`. install with:\n"
        "    uv pip install tomli_w     # or pip install tomli_w",
        file=sys.stderr,
    )
    sys.exit(2)


def _is_python_project(path: Path) -> bool:
    """A directory is installable by uv if it has any of these markers."""
    return any((path / m).is_file() for m in ("pyproject.toml", "setup.py", "setup.cfg"))


def patch_sources(pyproject: Path, overrides: dict[str, Path]) -> None:
    """Set `[tool.uv.sources][<pkg>] = {path = "...", editable = true}` for
    every (pkg, path) in overrides. Other entries are preserved."""
    with pyproject.open("rb") as f:
        doc = tomllib.load(f)

    sources = doc.setdefault("tool", {}).setdefault("uv", {}).setdefault("sources", {})
    for pkg, path in overrides.items():
        sources[pkg] = {"path": str(path), "editable": True}

    # Make a backup once. Idempotent: don't overwrite an existing backup,
    # which would clobber the pristine original after a re-run.
    backup = pyproject.with_suffix(pyproject.suffix + ".dace_deterministic_codegen.bak")
    if not backup.exists():
        shutil.copy2(pyproject, backup)

    with pyproject.open("wb") as f:
        tomli_w.dump(doc, f)
    print(f"patched {pyproject}  (backup at {backup.name})")


def run(cmd: list[str], cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}    (cwd={cwd})")
    rc = subprocess.run(cmd, cwd=str(cwd)).returncode
    if rc != 0:
        sys.exit(rc)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--icon4py",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Path to icon4py checkout. Accepts BOTH absolute and relative "
            "paths. Relative paths are resolved against the current working "
            "directory."
        ),
    )
    p.add_argument(
        "--gt4py",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Path to gt4py checkout to install editable. Accepts BOTH "
            "absolute and relative paths (resolved against cwd)."
        ),
    )
    p.add_argument(
        "--dace",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to dace checkout (absolute or relative). If "
            "omitted, dace resolves through icon4py's existing source pin."
        ),
    )
    p.add_argument(
        "--no-lock", action="store_true", help="Skip `uv lock`. Useful if you already locked."
    )
    p.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip `uv sync`. Useful for CI steps that sync later.",
    )
    args = p.parse_args()

    # Resolve every path NOW. The script can be run from any cwd.
    icon4py = args.icon4py.expanduser().resolve()
    gt4py = args.gt4py.expanduser().resolve()
    dace = args.dace.expanduser().resolve() if args.dace else None

    pyproject = icon4py / "pyproject.toml"
    if not pyproject.is_file():
        print(f"error: no pyproject.toml at {pyproject}", file=sys.stderr)
        return 2
    if not _is_python_project(gt4py):
        print(
            f"error: --gt4py path is not a python project (no pyproject.toml, "
            f"setup.py, or setup.cfg): {gt4py}",
            file=sys.stderr,
        )
        return 2
    if dace and not _is_python_project(dace):
        print(
            f"error: --dace path is not a python project (no pyproject.toml, "
            f"setup.py, or setup.cfg): {dace}",
            file=sys.stderr,
        )
        return 2

    # Loud warning if no venv is active — the whole point of this script
    # is to install INTO the gt4py CI venv. Without VIRTUAL_ENV set, uv
    # would create a new .venv and we'd get nowhere.
    if not os.environ.get("VIRTUAL_ENV"):
        print(
            "warning: VIRTUAL_ENV is not set. This script is meant to install "
            "icon4py into the *currently activated* venv (typically your "
            "gt4py CI venv). Activate it first, then re-run.",
            file=sys.stderr,
        )

    overrides: dict[str, Path] = {"gt4py": gt4py}
    if dace:
        overrides["dace"] = dace
    patch_sources(pyproject, overrides)

    if not args.no_lock:
        # Regenerate uv.lock so it matches the new [tool.uv.sources].
        run(["uv", "lock"], cwd=icon4py)
    if not args.no_sync:
        # --active = use $VIRTUAL_ENV (the gt4py venv) instead of ./venv/.
        run(["uv", "sync", "--active"], cwd=icon4py)

    print()
    print("done. quick sanity check:")
    print('    python -c "import gt4py.next; print(gt4py.next.__file__)"')
    print(f"        # should print a path inside {gt4py}")
    if dace:
        print('    python -c "import dace; print(dace.__file__)"')
        print(f"        # should print a path inside {dace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
