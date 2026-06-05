#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""CLI for the development scripts in the 'ROOT/scripts' folder."""

from __future__ import annotations

import importlib
import pathlib
import pkgutil
from typing import Final

import typer


SCRIPTS_DIR: Final = pathlib.Path(__file__).parent.resolve().absolute()
assert str(SCRIPTS_DIR.name) == __package__

cli = typer.Typer(no_args_is_help=True, name=__package__, help=__doc__)

# Assemble the CLI from all public commands in the 'scripts' package
for _, name, _ in pkgutil.walk_packages([str(SCRIPTS_DIR)]):
    if not name.startswith("_"):
        module = importlib.import_module(f"{__package__}.{name}")
        if hasattr(module, "cli") and isinstance(module.cli, typer.Typer):
            if len(module.cli.registered_commands) > 1:
                cli.add_typer(module.cli, name=module.cli.info.name or name)
            else:
                cli.add_typer(module.cli)

cli()
