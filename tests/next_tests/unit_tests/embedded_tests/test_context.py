# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next.embedded.context as ctx
from gt4py import eve


def test_update_with_both_parameters():
    initial_column_range = "INITIAL_COLUMN_RANGE"
    initial_offset_provider = "INITIAL_OFFSET_PROVIDER"

    assert ctx.closure_column_range.get() is eve.NOTHING
    assert ctx.offset_provider.get() is eve.NOTHING

    with ctx.update(
        closure_column_range=initial_column_range, offset_provider=initial_offset_provider
    ):
        assert ctx.closure_column_range.get() == initial_column_range
        assert ctx.offset_provider.get() == initial_offset_provider

        test_column_range = "TEST_COLUMN_RANGE"
        test_offset_provider = "TEST_OFFSET_PROVIDER"

        with ctx.update(
            closure_column_range=test_column_range, offset_provider=test_offset_provider
        ):
            assert ctx.closure_column_range.get() == test_column_range
            assert ctx.offset_provider.get() == test_offset_provider

        assert ctx.closure_column_range.get() == initial_column_range
        assert ctx.offset_provider.get() == initial_offset_provider

    assert ctx.closure_column_range.get() is eve.NOTHING
    assert ctx.offset_provider.get() is eve.NOTHING


def test_update_with_no_parameters():
    assert ctx.closure_column_range.get() is eve.NOTHING
    assert ctx.offset_provider.get() is eve.NOTHING

    initial_column_range = "INITIAL_COLUMN_RANGE"
    initial_offset_provider = "INITIAL_OFFSET_PROVIDER"

    with ctx.update(
        closure_column_range=initial_column_range, offset_provider=initial_offset_provider
    ):
        assert ctx.closure_column_range.get() == initial_column_range
        assert ctx.offset_provider.get() == initial_offset_provider

        with ctx.update():
            assert ctx.closure_column_range.get() == initial_column_range
            assert ctx.offset_provider.get() == initial_offset_provider

        assert ctx.closure_column_range.get() == initial_column_range
        assert ctx.offset_provider.get() == initial_offset_provider

    assert ctx.closure_column_range.get() is eve.NOTHING
    assert ctx.offset_provider.get() is eve.NOTHING


def test_within_valid_context():
    assert not ctx.within_valid_context()

    test_column_range = "TEST_COLUMN_RANGE"
    with ctx.update(closure_column_range=test_column_range):
        assert not ctx.within_valid_context()

    test_offset_provider = "TEST_OFFSET_PROVIDER"
    with ctx.update(offset_provider=test_offset_provider):
        assert ctx.within_valid_context()

    assert not ctx.within_valid_context()
