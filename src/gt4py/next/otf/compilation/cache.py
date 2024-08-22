# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Caching for compiled backend artifacts."""

import hashlib
import pathlib
import tempfile

from gt4py.next import config
from gt4py.next.otf import stages
from gt4py.next.otf.binding import interface


_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)


def _serialize_param(parameter: interface.Parameter) -> str:
    return f"{parameter.name}: {parameter.type_!s}"


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
    compilable_source: stages.CompilableSource, lifetime: config.BuildCacheLifetime
) -> pathlib.Path:
    """
    Construct the path to where the build system project artifact of a compilable source should be cached.

    The returned path points to an existing folder in all cases.
    """
    # TODO(ricoh): make dependent on binding source too or add alternative that depends on bindings
    folder_name = _cache_folder_name(compilable_source.program_source)

    match lifetime:
        case config.BuildCacheLifetime.SESSION:
            base_path = _session_cache_dir_path
        case config.BuildCacheLifetime.PERSISTENT:
            base_path = config.BUILD_CACHE_DIR
        case _:
            raise ValueError("Unsupported caching lifetime.")

    base_path.mkdir(exist_ok=True)

    complete_path = base_path / folder_name
    complete_path.mkdir(exist_ok=True)

    return complete_path
