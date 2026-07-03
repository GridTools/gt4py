# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for crash-consistent on-disk caches."""

from __future__ import annotations

import os
import pathlib
import uuid


def atomic_write_bytes(target: str | os.PathLike, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically.

    The bytes are written to a temporary file in ``target``'s directory and then
    renamed into place with ``os.replace``. A reader (or the next process after a
    crash) therefore sees either the previous contents or the complete new
    contents, never a half-written file. The temporary file is a sibling of
    ``target`` so the rename stays on a single filesystem, and is created with a
    plain ``open`` so the result gets the usual umask-derived permissions
    (``tempfile.mkstemp`` would pin it to ``0600``).

    This protects against process kill / Ctrl-C / OOM during the write. It does
    *not* fsync, so a power loss may still lose the entry (callers tolerate this
    by treating an unreadable entry as a cache miss).
    """
    target = pathlib.Path(target)
    tmp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(target: str | os.PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``target`` atomically (see `atomic_write_bytes`)."""
    atomic_write_bytes(target, text.encode(encoding))
