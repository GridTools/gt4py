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
from typing import Final

from gt4py.next import config, fingerprinting
from gt4py.next.otf import stages


#: Regex describing the folder names produced by `get_cache_folder` (use
#: `re.fullmatch`): `{name}_{fingerprint}_{version_id}`, plus a trailing
#: `_{build_context_id}` when a fingerprint-style (16-hex) build context id was
#: given. Where `get_cache_folder` assembles a folder name from these parts, the
#: pattern's capture groups take an existing folder name apart again — most
#: importantly recovering the program `name`. External tools use it to recognize
#: cached program folders — e.g. `scripts/python/dace_determinism.py` reads this
#: pattern from the installed gt4py at runtime. When changing the naming scheme
#: in `get_cache_folder`, update this pattern with it; the round-trip test in
#: `test_cache.py` fails otherwise.
CACHE_FOLDER_NAME_PATTERN: Final[str] = (
    r"(?P<name>.+)_(?P<fingerprint>[0-9a-f]{16})_(?P<version_id>.+?)"
    r"(?:_(?P<build_context_id>[0-9a-f]{16}))?"
)

#: Suffix appended by `get_cache_folder` to the program name when the cached
#: artifact includes bindings. It is part of the pattern's `name` group, so
#: recovering the plain program name means stripping this suffix.
BINDINGS_NAME_SUFFIX: Final[str] = "_pyext"

#: Directory under the cache base holding the translation caches, one
#: sub-directory per backend. These hold the output of the translation step: an
#: optimized SDFG or generated source per translated program.
TRANSLATION_CACHE_DIR_NAME: Final[str] = "translation_cache"

#: Directory under the cache base holding the build caches, one folder per
#: compiled program variant, named by `CACHE_FOLDER_NAME_PATTERN`. These hold
#: build artifacts and the compiled library, so unlike a translation cache they
#: are shared by all backends.
BUILD_CACHE_DIR_NAME: Final[str] = "build_cache"

#: Backends that persist the output of their translation step, i.e. those whose
#: workflow factory enables the `cached_translation` trait.
TRANSLATION_CACHE_BACKENDS: Final[tuple[str, ...]] = ("dace", "gtfn")

_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)


def get_translation_cache_folder(cache_base: pathlib.Path, backend: str) -> pathlib.Path:
    """Return the folder under `cache_base` where `backend` caches its translations.

    Unlike `get_cache_folder`, this only computes a path. The folder is created by
    the cache that writes into it, so that asking where a cache would be does not
    create it — which tools that only inspect a cache rely on.
    """
    return cache_base / TRANSLATION_CACHE_DIR_NAME / backend


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

    The folder name is salted with ``config.BUILD_CACHE_VERSION_ID`` (defaulting to the gt4py
    version, overridable via the ``GT4PY_BUILD_CACHE_VERSION_ID`` env var) so that a change to
    the build-cache version forces incompatibility with previously cached builds, even when the
    extension source fingerprint is unchanged.

    An optional ``build_context_id`` can be provided to distinguish between different contexts
    that may produce different artifacts for the same extension source.
    The returned path points to an existing folder in all cases.

    The folder name layout is documented by ``CACHE_FOLDER_NAME_PATTERN``; keep the
    two in sync when changing the naming scheme here.
    """
    fingerprinter = fingerprinting.strict_fingerprinter
    slug = ext_source.program_source.entry_point.name
    if ext_source.binding_source:
        slug = f"{slug}{BINDINGS_NAME_SUFFIX}"
    folder_name = f"{slug}_{fingerprinter(ext_source)}_{config.BUILD_CACHE_VERSION_ID}"
    if build_context_id:
        folder_name = f"{folder_name}_{build_context_id}"

    build_cache_path = get_cache_base_path(lifetime) / BUILD_CACHE_DIR_NAME
    build_cache_path.mkdir(parents=True, exist_ok=True)

    complete_path = build_cache_path / folder_name
    complete_path.mkdir(exist_ok=True)

    # Resolve symlinks to workaround an issue on MacOS where the default tmp directory is a symlink
    # which might sometimes get resolved in a way we don't control.
    return complete_path.resolve()
