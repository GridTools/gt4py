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
