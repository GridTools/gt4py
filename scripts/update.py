#! /usr/bin/env -S uv run -q --frozen --isolated --group scripts --script
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Update project dependencies and pre-commit hooks."""

from __future__ import annotations

import pathlib
import subprocess
from typing import Final

import typer


REPO_ROOT: Final = pathlib.Path(__file__).parent


app = typer.Typer(no_args_is_help=True, name="update", help=__doc__)


@app.command()
def dependencies() -> None:
    """Update project dependencies to their latest compatible versions."""
    subprocess.run("uv lock --upgrade", cwd=REPO_ROOT, shell=True, check=True)


@app.command("pre-commit")
def precommit() -> None:
    """Update versions of pre-commit hooks."""

    subprocess.run(
        f"uv run --quiet --locked --project {REPO_ROOT} pre-commit autoupdate", shell=True
    )


@app.command()
def all() -> None:
    """Update both project dependencies and pre-commit hooks."""
    dependencies()
    precommit()


if __name__ == "__main__":
    app()
