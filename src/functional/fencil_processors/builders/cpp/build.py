# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build system functionality."""


from __future__ import annotations

import importlib
import json
import pathlib
from typing import Callable, Optional

from functional.fencil_processors import pipeline
from functional.fencil_processors.builders import cache, importer
from functional.fencil_processors.source_modules import source_modules


def python_module_suffix():
    return importlib.machinery.EXTENSION_SUFFIXES[0][1:]


def jit_path_from_jit_module(
    jit_module: source_modules.JITSourceModule, cache_strategy: cache.Strategy
) -> pathlib.Path:
    # TODO: the output of this should depend also on at least the bindings module
    return cache.get_cache_folder(jit_module.source_module, cache_strategy)


def data_is_in_jit_path(jit_path: pathlib.Path) -> bool:
    return (jit_path / "gt4py.json").exists()


def data_from_jit_path(jit_path: pathlib.Path) -> dict:
    data_file = jit_path / "gt4py.json"
    if not data_file.exists():
        return {"status": "unknown"}
    return json.loads(data_file.read_text())


def data_to_jit_path(data: dict, jit_path: pathlib.Path):
    (jit_path / "gt4py.json").write_text(json.dumps(data))


def compiled_fencil_from_jit_path(jit_path: pathlib.Path) -> Optional[Callable]:
    data = data_from_jit_path(jit_path)
    if data["status"] != "built":  # @todo turn into enum or such
        return None
    return importer.import_from_path(jit_path / data["extension"])


def jit_module_to_compiled_fencil(
    jit_module: source_modules.JITSourceModule,
    jit_builder_generator: pipeline.JITBuilderGenerator,
    cache_strategy: cache.Strategy,
) -> Callable:
    jit_dir = jit_path_from_jit_module(jit_module, cache_strategy)
    compiled_module = compiled_fencil_from_jit_path(jit_dir)
    if compiled_module:
        return getattr(compiled_module, jit_module.source_module.entry_point.name)
    jit_builder = jit_builder_generator(jit_module, cache_strategy)
    jit_builder.build()
    compiled_module = compiled_fencil_from_jit_path(jit_dir)
    if not compiled_module:
        raise AssertionError(
            "Build completed but no compiled python extension was found"
        )  # @todo: make safer, improve error msg
    return getattr(compiled_module, jit_module.source_module.entry_point.name)
