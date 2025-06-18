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

import subprocess

import typer

from . import _common


app = typer.Typer(no_args_is_help=True)


@app.command()
def dependencies() -> None:
    """Update project dependencies to their latest compatible versions."""
    subprocess.run("uv lock --upgrade", cwd=_common.ROOT_DIR, shell=True, check=True)


@app.command("pre-commit")
def precommit() -> None:
    """Update versions of pre-commit hooks."""

    subprocess.run(
        f"uv run --quiet --locked --project {_common.ROOT_DIR} pre-commit autoupdate", shell=True
    )


@app.command()
def all() -> None:
    """Update both project dependencies and pre-commit hooks."""
    dependencies()
    precommit()


if __name__ == "__main__":
    app()
