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
# requires-python = ">=3.12"
# dependencies = [
#   'pyyaml>=6.0.1',
#   'typer>=0.12.3'
# ]
# ///

"""CLI tool to run recurrent development tasks. Subcommands are defined in the `scripts` folder."""

from __future__ import annotations

import pathlib
import sys


try:
    import typer

    import scripts

    assert len(scripts.__path__) == 1 and scripts.__path__[0] == str(
        pathlib.Path(__file__).parent.resolve().absolute() / "scripts"
    ), (
        "The 'scripts' package path does not match the expected path. "
        "Please check the structure of the repository."
    )

    cli = typer.Typer(no_args_is_help=True, name="dev-scripts", help=__doc__)

    def main() -> None:
        """Main entry point for the dev-scripts CLI."""

        for name, sub_cli in scripts.typer_clis.items():
            cli.add_typer(sub_cli, name=sub_cli.info.name or name)

        cli()

    if __name__ == "__main__":
        main()


except ImportError as e:
    import sys

    print(
        f"ERROR: '{e.name}' package cannot be imported!!\n"
        "Make sure 'uv' is installed in your system and run directly this script "
        "as an executable file, to let 'uv' create a temporary venv with all the "
        "required dependencies.\n",
        file=sys.stderr,
    )
    sys.exit(127)
