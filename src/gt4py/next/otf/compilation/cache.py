# GT4Py - GridTools Framework
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

from gt4py.next.otf import stages
from gt4py.next.otf.binding import interface


class Strategy(enum.Enum):
    SESSION = 1
    PERSISTENT = 2


_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)
_persistent_cache_dir_path = pathlib.Path(tempfile.gettempdir()) / "gt4py_cache"


def _serialize_param(
    parameter: interface.ScalarParameter
    | interface.BufferParameter
    | interface.ConnectivityParameter,
) -> str:
    if isinstance(parameter, interface.ScalarParameter):
        return f"{parameter.name}: {str(parameter.scalar_type)}"
    elif isinstance(parameter, interface.BufferParameter):
        return f"{parameter.name}: {str(parameter.scalar_type)}<{', '.join(parameter.dimensions)}>"
    elif isinstance(parameter, interface.ConnectivityParameter):
        return f"{parameter.name}: {parameter.offset_tag}"
    raise ValueError("Invalid parameter type. This is a bug.")


def _serialize_library_dependency(dependency: interface.LibraryDependency) -> str:
    return f"{dependency.name}/{dependency.version}"


def _serialize_source(source: stages.ProgramSource) -> str:
    parameters = [_serialize_param(param) for param in source.entry_point.parameters]
    dependencies = [_serialize_library_dependency(dep) for dep in source.library_deps]
    return f"""\
    language: {source.language}
    name: {source.entry_point.name}
    params: {', '.join(parameters)}
    deps: {', '.join(dependencies)}
    src: {source.source_code}\
    """


def _cache_folder_name(source: stages.ProgramSource) -> str:
    serialized = _serialize_source(source)
    fingerprint = hashlib.sha256(serialized.encode(encoding="utf-8"))
    fingerprint_hex_str = fingerprint.hexdigest()
    return source.entry_point.name + "_" + fingerprint_hex_str


def get_cache_folder(
    compilable_source: stages.CompilableSource, strategy: Strategy
) -> pathlib.Path:
    """
    Construct the path to where the build system project artifact of a compilable source should be cached.

    The returned path points to an existing folder in all cases.
    """
    # TODO(ricoh): make dependent on binding source too or add alternative that depends on bindings
    folder_name = _cache_folder_name(compilable_source.program_source)

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
