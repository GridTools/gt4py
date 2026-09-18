# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import tempfile
import shutil

from gt4py._core import filecache


@pytest.fixture
def temp_cache():
    """Fixture to create a temporary FileCache directory and clean it up."""
    tmpdir = tempfile.mkdtemp()
    cache = filecache.FileCache(tmpdir)
    yield cache
    shutil.rmtree(tmpdir)


def test_set_and_get_item(temp_cache):
    key = "foo"
    value = {"a": 1, "b": 2}
    temp_cache[key] = value

    assert key in temp_cache
    assert temp_cache[key] == value


def test_delete_item(temp_cache):
    key = "bar"
    temp_cache[key] = 123
    assert key in temp_cache

    del temp_cache[key]
    assert key not in temp_cache

    with pytest.raises(KeyError):
        _ = temp_cache[key]


def test_keyerror_on_missing(temp_cache):
    with pytest.raises(KeyError):
        _ = temp_cache["does_not_exist"]


# --- Crash consistency ---
# An interrupted write (Ctrl-C / SIGKILL / OOM during ``pickle.dump``) can leave a
# truncated entry on disk. ``__contains__`` still reports a HIT, but reading must
# behave like a cache miss (raise ``KeyError``) instead of propagating an
# unpickling error, and the unusable entry must be evicted so the cache recovers.


def test_truncated_entry_is_treated_as_missing(temp_cache):
    key = "interrupted"
    temp_cache[key] = {"a": 1}

    # Simulate a write killed mid-``pickle.dump``: truncate the payload.
    path = temp_cache._get_path(key)
    path.write_bytes(path.read_bytes()[:4])

    with pytest.raises(KeyError):
        _ = temp_cache[key]


def test_corrupt_entry_self_heals(temp_cache):
    key = "interrupted"
    temp_cache[key] = {"a": 1}

    # A syntactically-invalid pickle (valid proto header, garbage body).
    temp_cache._get_path(key).write_bytes(b"\x80\x05garbage-not-a-pickle")

    with pytest.raises(KeyError):
        _ = temp_cache[key]

    # The unusable entry is evicted, so the cache is writable/usable again.
    assert key not in temp_cache
    temp_cache[key] = {"a": 2}
    assert temp_cache[key] == {"a": 2}
