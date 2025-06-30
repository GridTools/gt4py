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


_default_lock: _LockConstructor = filelock.FileLock


def lock(path: pathlib.Path | str) -> ContextManager:
    """Create a lock for the given path."""
    return _default_lock(str(path))
