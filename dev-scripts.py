#! /usr/bin/env -S uv run -q --frozen --isolated --group scripts --script
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

import importlib
import importlib.util
import pathlib

import typer


app = typer.Typer(no_args_is_help=True)

def main() -> None:
    """Run a typer app with all scripts in the `scripts` folder as subcommands."""    
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
                    app.add_typer(value, name=task_module.__name__)

    app()


if __name__ == "__main__":
    main()
