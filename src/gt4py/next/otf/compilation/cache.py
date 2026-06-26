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

from gt4py.next import config, fingerprinting
from gt4py.next.otf import stages


_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)


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
    ext_source: stages.ExtensionSource,
    lifetime: config.BuildCacheLifetime,
    build_context_id: str = "",
) -> pathlib.Path:
    """
    Construct the path to where the build system project artifact of an extension source should be cached.

    An optional ``build_context_id`` can be provided to distinguish between different contexts
    that may produce different artifacts for the same extension source.
    The returned path points to an existing folder in all cases.
    """
    fingerprinter = fingerprinting.strict_fingerprinter
    slug = ext_source.program_source.entry_point.name
    if ext_source.binding_source:
        slug = f"{slug}_bound"
    folder_name = f"{slug}_{fingerprinter(ext_source)}_{build_context_id}"

    base_path = get_cache_base_path(lifetime)
    base_path.mkdir(exist_ok=True)

    complete_path = base_path / folder_name
    complete_path.mkdir(exist_ok=True)

    # Resolve symlinks to workaround an issue on MacOS where the default tmp directory is a symlink
    # which might sometimes get resolved in a way we don't control.
    return complete_path.resolve()
