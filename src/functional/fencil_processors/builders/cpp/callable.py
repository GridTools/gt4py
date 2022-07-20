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


from collections.abc import Callable

from functional.fencil_processors.builders.cache import Strategy as CacheStrategy, get_cache_folder
from functional.fencil_processors.builders.importer import import_from_path

from ...source_modules import source_modules as defs
from . import bindings, build


# TODO (ricoh): split into pipeline steps
def create_callable(
    source_module: defs.SourceModule, *, cache_strategy=CacheStrategy.SESSION
) -> Callable:
    """Build the source module and return its entry point as a Python function object."""
    cache_folder = get_cache_folder(source_module, cache_strategy)
    module_file = build.CMakeProject.get_binary_path(cache_folder, source_module.entry_point.name)
    try:
        return getattr(import_from_path(module_file), source_module.entry_point.name)
    except ModuleNotFoundError:
        pass

    src_header_file = source_module.entry_point.name + ".cpp.inc"
    bindings_file = source_module.entry_point.name + "_bindings.cpp"
    bindings_module = bindings.create_bindings(source_module.entry_point, src_header_file)

    deps = [*source_module.library_deps, *bindings_module.library_deps]
    sources = {
        src_header_file: source_module.source_code,
        bindings_file: bindings_module.source_code,
    }
    project = build.CMakeProject(
        name=source_module.entry_point.name, dependencies=deps, sources=sources
    )

    project.write(cache_folder)
    project.configure()
    project.build()

    return getattr(import_from_path(module_file), source_module.entry_point.name)
