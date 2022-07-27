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
"""Caching for compiled backend artifacts."""


import enum
import hashlib
import pathlib
import tempfile

from functional.fencil_processors import source_modules


class Strategy(enum.Enum):
    SESSION = 1
    PERSISTENT = 2


_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)
_persistent_cache_dir_path = pathlib.Path(tempfile.gettempdir()) / "gt4py_cache"


def _serialize_param(
    parameter: source_modules.ScalarParameter | source_modules.BufferParameter,
) -> str:
    if isinstance(parameter, source_modules.ScalarParameter):
        return f"{parameter.name}: {str(parameter.scalar_type)}"
    elif isinstance(parameter, source_modules.BufferParameter):
        return f"{parameter.name}: {str(parameter.scalar_type)}<{', '.join(parameter.dimensions)}>"
    raise ValueError("Invalid parameter type. This is a bug.")


def _serialize_library_dependency(dependency: source_modules.LibraryDependency) -> str:
    return f"{dependency.name}/{dependency.version}"


def _serialize_module(module: source_modules.SourceModule) -> str:
    parameters = [_serialize_param(param) for param in module.entry_point.parameters]
    dependencies = [_serialize_library_dependency(dep) for dep in module.library_deps]
    return f"""\
    language: {module.language}
    name: {module.entry_point.name}
    params: {', '.join(parameters)}
    deps: {', '.join(dependencies)}
    src: {module.source_code}\
    """


def _cache_folder_name(module: source_modules.SourceModule) -> str:
    serialized = _serialize_module(module)
    fingerprint = hashlib.sha256(serialized.encode(encoding="utf-8"))
    fingerprint_hex_str = fingerprint.hexdigest()
    return module.entry_point.name + "_" + fingerprint_hex_str


def get_cache_folder(module: source_modules.SourceModule, strategy: Strategy) -> pathlib.Path:
    """
    Construct the path to where the build system project artifact of a source module should be cached.

    The returned path points to an existing folder in all cases.
    """
    folder_name = _cache_folder_name(module)

    match strategy:
        case Strategy.SESSION:
            base_path = _session_cache_dir_path
        case Strategy.PERSISTENT:
            base_path = _persistent_cache_dir_path
        case _:
            raise ValueError("Unsupported caching strategy.")

    base_path.mkdir(exist_ok=True)

    complete_path = base_path / folder_name
    complete_path.mkdir(exist_ok=True)

    return complete_path
