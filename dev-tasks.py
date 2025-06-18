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


import importlib
import importlib.util
import pathlib
import sys
import types
from typing import Final

import typer

TASKS_PACKAGE_NAME: Final = "dev-tasks"

app = typer.Typer(no_args_is_help=True)


def import_tasks_package(tasks_pkg_path: pathlib.Path) -> None:
    """Dynamically import the tasks subcommands defined from the package folder."""
    spec = importlib.util.spec_from_file_location("tasks", tasks_pkg_path / "__init__.py")
    assert spec is not None, f"Could not find tasks package at {tasks_pkg_path}"
    assert spec.loader is not None, f"Could not find tasks package at {tasks_pkg_path}"
    tasks_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks_module)
    sys.modules[TASKS_PACKAGE_NAME] = tasks_module


def add_commands(app: typer.Typer) -> typer.Typer:
    """Add all subcommands from the tasks package to the main app."""
    for name, value in vars(sys.modules[TASKS_PACKAGE_NAME]).items():
        if (
            isinstance(value, types.ModuleType)
            and not name.startswith("_")
            and hasattr(value, "app")
            and isinstance(value.app, typer.Typer)
        ):
            app.add_typer(value.app, name=name)

    return app


if __name__ == "__main__":
    tasks_pkg_path = pathlib.Path(__file__).parent.resolve().absolute() / "tasks"
    import_tasks_package(tasks_pkg_path)
    add_commands(app)
    app()
