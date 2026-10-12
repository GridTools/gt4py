# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Caching for compiled backend artifacts."""

import functools
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import warnings
from typing import Final, TypeVar

from gt4py._core import filecache, locking
from gt4py.next import config, fingerprinting
from gt4py.next.otf import artifacts, workflow


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
#: sub-directory per backend. Unlike a build cache folder, which holds the
#: artifacts of one compiled program variant, these hold the output of the
#: translation step: an optimized SDFG or generated source per translated
#: program.
TRANSLATION_CACHE_DIR_NAME: Final[str] = "translation_cache"

#: Backends that persist the output of their translation step, i.e. those whose
#: builders wrap it with `persistent_translation_cache`.
TRANSLATION_CACHE_BACKENDS: Final[tuple[str, ...]] = ("dace", "gtfn")

StartT = TypeVar("StartT")
EndT = TypeVar("EndT")

#: File in a persistent cache base directory recording the toolchain that filled the cache.
TOOLCHAIN_FILE_NAME: Final[str] = "toolchain.json"

#: Python distributions whose version shapes the compiled artifacts but not the cache key.
_TOOLCHAIN_DISTRIBUTIONS: Final[tuple[str, ...]] = ("dace", "gridtools-cpp", "nanobind")


class BuildCacheToolchainError(RuntimeError):
    """The toolchain differs from the one that filled the persistent build cache."""


def _compiler_version(env_var: str, default: str) -> str | None:
    executable = shutil.which(os.environ.get(env_var) or default)
    if executable is None:
        return None
    try:
        output = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # nvcc prints a banner first; its version is on the "release" line
    return next((line for line in lines if "release" in line), lines[0] if lines else None)


def current_toolchain() -> dict[str, str]:
    """Versions of what turns generated code into binaries but is not part of the cache key."""
    toolchain = {"machine": platform.machine(), "python": platform.python_version()}
    for distribution in _TOOLCHAIN_DISTRIBUTIONS:
        try:
            toolchain[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    for key, env_var, default in (("cxx", "CXX", "c++"), ("cuda_cxx", "CUDACXX", "nvcc")):
        if (version := _compiler_version(env_var, default)) is not None:
            toolchain[key] = version
    if archs := os.environ.get("CUDAARCHS", "").strip():
        toolchain["cuda_archs"] = archs
    return toolchain


@functools.cache
def check_build_cache_toolchain(base_path: pathlib.Path) -> None:
    """
    Record the toolchain in a persistent cache on first use, and compare it on every later use.

    Only entries present in both the record and the current toolchain are compared; entries the
    record lacks (e.g. a CUDA compiler first seen by a GPU build) are added to it. On a mismatch,
    ``config.BUILD_CACHE_TOOLCHAIN_CHECK`` decides between raising and warning.

    Raises:
        BuildCacheToolchainError: if an entry differs and the check is set to ``ERROR``.
    """
    mode = config.BUILD_CACHE_TOOLCHAIN_CHECK
    if mode == config.BuildCacheToolchainCheck.OFF:
        return

    current = current_toolchain()
    record_path = base_path / TOOLCHAIN_FILE_NAME
    base_path.mkdir(parents=True, exist_ok=True)
    with locking.lock(base_path, suffix=".toolchain.lock"):
        recorded = json.loads(record_path.read_text()) if record_path.exists() else {}
        changed = {
            key: (recorded[key], value)
            for key, value in current.items()
            if key in recorded and recorded[key] != value
        }
        if not changed and any(key not in recorded for key in current):
            record_path.write_text(json.dumps({**current, **recorded}, indent=2) + "\n")

    if not changed:
        return
    differences = "\n".join(
        f"  {key}: {old!r} (cache) -> {new!r} (now)" for key, (old, new) in changed.items()
    )
    message = (
        f"The toolchain differs from the one that filled the GT4Py build cache '{base_path}':\n"
        f"{differences}\n"
        "Its compiled programs may be stale. Delete the cache, point GT4PY_BUILD_CACHE_DIR to "
        "another directory, or set GT4PY_BUILD_CACHE_TOOLCHAIN_CHECK=warn if the change does "
        "not require a rebuild."
    )
    if mode == config.BuildCacheToolchainCheck.ERROR:
        raise BuildCacheToolchainError(message)
    warnings.warn(message, UserWarning, stacklevel=2)


_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)


def get_translation_cache_folder(cache_base: pathlib.Path, backend: str) -> pathlib.Path:
    """Return the folder under `cache_base` where `backend` caches its translations.

    Unlike `get_cache_folder`, this only computes a path. The folder is created by
    the cache that writes into it, so that asking where a cache would be does not
    create it — which tools that only inspect a cache rely on.
    """
    return cache_base / TRANSLATION_CACHE_DIR_NAME / backend


def persistent_translation_cache(
    step: workflow.Workflow[StartT, EndT], backend: str, lifetime: config.BuildCacheLifetime
) -> workflow.CachedStep[StartT, EndT, str]:
    """
    Wrap a translation step in the persistent translation cache of `backend`.

    Args:
        step: The translation step to cache.
        backend: Name of the backend family, which selects the cache folder.
        lifetime: Build-cache lifetime, which selects the cache base path.

    Returns:
        The step, cached in the translation cache folder of `backend`.
    """
    return workflow.CachedStep[StartT, EndT, str].persistent(
        step,
        input_fingerprinter=fingerprinting.strict_fingerprinter,
        cache=filecache.FileCache(
            get_translation_cache_folder(get_cache_base_path(lifetime), backend)
        ),
    )


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
    ext_source: artifacts.ExtensionSource,
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

    base_path = get_cache_base_path(lifetime)
    base_path.mkdir(exist_ok=True)
    if lifetime == config.BuildCacheLifetime.PERSISTENT:
        check_build_cache_toolchain(base_path)

    complete_path = base_path / folder_name
    complete_path.mkdir(exist_ok=True)

    # Resolve symlinks to workaround an issue on MacOS where the default tmp directory is a symlink
    # which might sometimes get resolved in a way we don't control.
    return complete_path.resolve()
