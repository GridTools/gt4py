# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from typing import ContextManager, Protocol

import filelock

from gt4py.eve import utils as eve_utils


class _LockConstructor(Protocol):
    def __call__(self, path: str, /) -> ContextManager: ...

    @property
    def __name__(self) -> str: ...


# Changing the lock is private API, but could be made public in the future
_lock_constructor: _LockConstructor = filelock.FileLock


def lock(directory: pathlib.Path | str, suffix: str = ".lock") -> ContextManager:
    """Create a lock for the given path."""
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory, got: {directory}")

    # Identifier of the lock implementation to avoid conflicts
    # when switching between different implementations.
    identifier = f"{eve_utils.a10n(_lock_constructor.__module__)}_{eve_utils.a10n(_lock_constructor.__name__)}"
    path = directory / f"{identifier}{suffix}"
    return _lock_constructor(str(path))
