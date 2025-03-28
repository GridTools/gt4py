#! /usr/bin/env -S uv run -q --script
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "typer>=0.12.3",
# ]
# [tool.uv]
# exclude-newer = "2025-01-31T00:00:00Z"
# ///


"""Script for running recurrent development tasks."""

from __future__ import annotations

import pathlib
import subprocess
from typing import Final

import typer

ROOT_DIR: Final = pathlib.Path(__file__).parent


# -- Helpers --
def gather_versions() -> dict[str, str]:
    with subprocess.Popen(
        [*"uv export --frozen --no-hashes --project".split(), ROOT_DIR], stdout=subprocess.PIPE
    ) as proc:
        return dict(
            line.split("==")
            for line in proc.stdout.read().decode().splitlines()
            if not any(line.startswith(c) for c in ["-", "#"])
        )


# -- CLI --
app = typer.Typer(no_args_is_help=True)


@app.command()
def sync_precommit() -> None:
    """Sync versions of tools used in pre-commit hooks with the project versions."""
    versions = gather_versions()
    # Update ruff version in pre-commit config
    subprocess.run(
        f"""uvx -q --from 'yamlpath' yaml-set --mustexist --change='repos[.repo%https://github.com/astral-sh/ruff-pre-commit].rev' --value='v{versions["ruff"]}' .pre-commit-config.yaml""",
        cwd=ROOT_DIR,
        shell=True,
        check=True,
    )

    # Update tach version in pre-commit config
    subprocess.run(
        f"""uvx -q --from 'yamlpath' yaml-set --mustexist --change='repos[.repo%https://github.com/gauge-sh/tach-pre-commit].rev' --value='v{versions["tach"]}' .pre-commit-config.yaml""",
        cwd=ROOT_DIR,
        shell=True,
        check=True,
    )

    # Format yaml files
    subprocess.run(
        f"uv run --project {ROOT_DIR} pre-commit run pretty-format-yaml --all-files", shell=True
    )


@app.command()
def update_precommit() -> None:
    """Update and sync pre-commit hooks with the latest compatible versions."""
    subprocess.run(f"uv run --project {ROOT_DIR} pre-commit autoupdate", shell=True)
    sync_precommit()


@app.command()
def update_versions() -> None:
    """Update all project dependencies to their latest compatible versions."""
    subprocess.run("uv lock --upgrade", cwd=ROOT_DIR, shell=True, check=True)


@app.command()
def update_all() -> None:
    """Update all project dependencies and pre-commit hooks."""
    update_versions()
    update_precommit()


if __name__ == "__main__":
    app()
