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


class _LockConstructor(Protocol):
    def __call__(self, path: str, /) -> ContextManager: ...


# Changing the lock is private API, but could be made public in the future
# The second element (str) is an identifier of the lock implementation to avoid conflicts
# when switching between different lock implementations.
_lock: tuple[_LockConstructor, str] = (filelock.FileLock, "filelock")


def lock(directory: pathlib.Path | str, suffix: str = ".lock") -> ContextManager:
    """Create a lock for the given path."""
    constructor, identifier = _lock

    path = pathlib.Path(directory) / f"{identifier}{suffix}"
    return constructor(str(path))
