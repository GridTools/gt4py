# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest

from gt4py._core import file_utils


def test_atomic_write_bytes(tmp_path):
    target = tmp_path / "data.bin"

    file_utils.atomic_write_bytes(target, b"payload")

    assert target.read_bytes() == b"payload"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_text_overwrites(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("old")

    file_utils.atomic_write_text(target, "new")

    assert target.read_text() == "new"
    assert list(tmp_path.glob("*.tmp")) == []


def test_interrupted_publish_leaves_target_untouched(tmp_path, monkeypatch):
    target = tmp_path / "data.bin"
    target.write_bytes(b"old")

    def failing_replace(src, dst):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(file_utils.os, "replace", failing_replace)
    with pytest.raises(RuntimeError, match="interrupted"):
        file_utils.atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_published_file_has_umask_derived_permissions(tmp_path):
    atomic_target = tmp_path / "atomic.bin"
    plain_target = tmp_path / "plain.bin"

    file_utils.atomic_write_bytes(atomic_target, b"x")
    with open(plain_target, "wb") as f:
        f.write(b"x")

    assert (atomic_target.stat().st_mode & 0o777) == (plain_target.stat().st_mode & 0o777)
