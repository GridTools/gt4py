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
import pathlib
from typing import Callable, Optional

from functional.fencil_processors import pipeline
from functional.fencil_processors.builders import build_data, cache, importer
from functional.fencil_processors.source_modules import source_modules


def python_module_suffix():
    return importlib.machinery.EXTENSION_SUFFIXES[0][1:]


def otf_path_from_otf_module(
    otf_module: source_modules.JITSourceModule, cache_strategy: cache.Strategy
) -> pathlib.Path:
    # TODO: the output of this should depend also on at least the bindings module
    return cache.get_cache_folder(otf_module.source_module, cache_strategy)


def compiled_fencil_from_otf_path(otf_path: pathlib.Path) -> Optional[Callable]:
    data = build_data.read_data(otf_path)
    if not data or data.status < build_data.OTFBuildStatus.COMPILED:
        return None
    return getattr(importer.import_from_path(otf_path / data.module), data.entry_point_name)


def otf_module_to_compiled_fencil(
    inp: source_modules.JITSourceModule,
    otf_builder_generator: pipeline.JITBuilderGenerator,
    cache_strategy: cache.Strategy,
) -> Callable:
    # @todo: move to language-agnostic module
    otf_module = inp
    otf_dir = otf_path_from_otf_module(otf_module, cache_strategy)
    compiled_fencil = compiled_fencil_from_otf_path(otf_dir)
    if compiled_fencil:
        return compiled_fencil
    otf_builder = otf_builder_generator(otf_module, cache_strategy)
    otf_builder.build()
    compiled_fencil = compiled_fencil_from_otf_path(otf_dir)
    if not compiled_fencil:
        raise AssertionError(
            "Build completed but no compiled python extension was found"
        )  # @todo: make safer, improve error msg
    return compiled_fencil
