# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import pathlib
import pickle
from typing import Any, Hashable

from gt4py._core import file_utils, locking
from gt4py.eve import utils as eve_utils


def _file_identity(stat: os.stat_result) -> tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class FileCache:
    """
    Dictionary-like object for persistently caching objects on disk.
    """

    path: pathlib.Path

    def __init__(self, path: str | os.PathLike):
        self.path = pathlib.Path(path).resolve()

    def _get_path(self, key: Hashable) -> pathlib.Path:
        """Return the path where an item with `key` is stored."""
        key = eve_utils.content_hash(key)
        path = self.path / f"{key}.pkl"
        return path

    def __getitem__(self, key: Hashable) -> Any:
        # Lock-free: entries are only ever replaced atomically, so a reader sees
        # either a complete entry or none.
        path = self._get_path(key)
        identity = None
        try:
            with open(path, "rb") as f:
                identity = _file_identity(os.fstat(f.fileno()))
                return pickle.load(f)
        except FileNotFoundError as e:
            raise KeyError(key) from e
        except (EOFError, pickle.UnpicklingError, OSError, AttributeError, ImportError) as e:
            # An interrupted write can leave a truncated/corrupt entry. Treat it as a
            # miss and drop it so the caller recomputes it, unless a writer has
            # replaced it in the meantime.
            if identity is not None:
                self._evict_if_unchanged(path, identity)
            raise KeyError(key) from e

    def _evict_if_unchanged(self, path: pathlib.Path, identity: tuple[int, ...]) -> None:
        with locking.lock(path):
            try:
                current = _file_identity(path.stat())
            except FileNotFoundError:
                return
            if current == identity:
                path.unlink(missing_ok=True)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with locking.lock(path := self._get_path(key)):
            file_utils.atomic_write_bytes(path, pickle.dumps(value, protocol=5))

    def __delitem__(self, key: Hashable) -> None:
        if key not in self:
            raise KeyError(key)
        with locking.lock(path := self._get_path(key)):
            path.unlink()

    def __contains__(self, key: Hashable) -> bool:
        return self._get_path(key).exists()
