# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Caching for compiled backend artifacts."""

import pathlib
import tempfile
from typing import Any

from gt4py.next import config, utils
from gt4py.next.otf import stages


_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)


def _cache_folder_name(source: stages.ProgramSource, *ctx: Any) -> str:
    fingerprint_hex_str = utils.stable_fingerprinter((source, *ctx))
    return source.entry_point.name + "_" + fingerprint_hex_str


def get_cache_base_path(lifetime: config.BuildCacheLifetime) -> pathlib.Path:
    """Return the base directory for cached artifacts with the given lifetime."""
    match lifetime:
        case config.BuildCacheLifetime.SESSION:
            return _session_cache_dir_path
        case config.BuildCacheLifetime.PERSISTENT:
            return config.BUILD_CACHE_DIR
        case _:
            raise ValueError("Unsupported caching lifetime.")


def get_cache_folder(
    compilable_source: stages.CompilableProject,
    lifetime: config.BuildCacheLifetime,
    *ctx: Any,
) -> pathlib.Path:
    """
    Construct the path to where the build system project artifact of a compilable source should be cached.

    The returned path points to an existing folder in all cases.
    """
    # TODO(ricoh): make dependent on binding source too or add alternative that depends on bindings
    folder_name = _cache_folder_name(compilable_source.program_source, ctx)

    base_path = get_cache_base_path(lifetime)
    base_path.mkdir(exist_ok=True)

    complete_path = base_path / folder_name
    complete_path.mkdir(exist_ok=True)

    # Resolve symlinks to workaround an issue on MacOS where the default tmp directory is a symlink
    # which might sometimes get resolved in a way we don't control.
    return complete_path.resolve()
