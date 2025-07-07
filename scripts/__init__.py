#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""
Package for development scripts.

Dependencies for all the modules in this package should be declared
in the inlined metadata section of the `/scripts-cli.py` script.
"""

from __future__ import annotations

import importlib
import pathlib
import pkgutil

import typer


__all__: list[str] = []

typer_clis: dict[str, typer.Typer] = {}

for _, _name, _ in pkgutil.walk_packages([str(pathlib.Path(__file__).parent.resolve().absolute())]):
    if not _name.startswith("_"):
        __all__.append(_name)
        _module = importlib.import_module(f"{__name__}.{_name}")
        if hasattr(_module, "cli") and isinstance(_module.cli, typer.Typer):
            typer_clis[_name] = _module.cli

del _name, _module, _
