#! /usr/bin/env -S uv run -q -p 3.11 --frozen --isolated --group scripts --script
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""CLI tool to run recurrent development tasks. Subcommands are defined in the `scripts` folder."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

try:
    import typer
except ImportError:
    import sys

    print(
        "ERROR: Missing required package!!\n\n"
        "Make sure 'uv' is installed in your system and run directly this script "
        "as an executable file, to let 'uv' create a temporary venv with all the "
        "required dependencies.",
        file=sys.stderr,
    )
    sys.exit(127)


app = typer.Typer(no_args_is_help=True, name="dev-scripts", help=__doc__)


def main() -> None:
    tasks_dir = pathlib.Path(__file__).parent.resolve().absolute() / "scripts"
    for path in tasks_dir.glob("*.py"):
        if not path.name.startswith("_"):
            spec = importlib.util.spec_from_file_location(path.stem, path.resolve())
            assert spec is not None, f"Could not load module at {path.resolve()}"
            task_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None, f"Could not load module at {path.resolve()}"
            spec.loader.exec_module(task_module)

            for key, value in vars(task_module).items():
                if key == "app" and isinstance(value, typer.Typer):
                    app.add_typer(value, name=value.info.name or task_module.__name__)

    app()


if __name__ == "__main__":
    main()
