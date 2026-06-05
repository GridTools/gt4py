# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import gt4py.next.embedded.context as ctx
import gt4py.next.common as common
from gt4py.next.errors import exceptions


def test_getters():
    DEFAULT = object()
    assert ctx.get_closure_column_range(DEFAULT) is DEFAULT
    assert ctx.get_offset_provider(DEFAULT) is DEFAULT

    with pytest.raises(exceptions.EmbeddedExecutionError, match="closure execution context."):
        assert ctx.get_closure_column_range()

    with pytest.raises(exceptions.EmbeddedExecutionError, match="closure execution context."):
        assert ctx.get_offset_provider()


def test_update_with_both_parameters():
    DEFAULT = object()
    assert ctx.get_closure_column_range(DEFAULT) is DEFAULT
    assert ctx.get_offset_provider(DEFAULT) is DEFAULT

    initial_column_range = common.NamedRange(common.Dimension("IDim"), common.UnitRange(0, 4))
    initial_offset_provider = {}

    with ctx.update(
        closure_column_range=initial_column_range, offset_provider=initial_offset_provider
    ):
        assert ctx.get_closure_column_range() is initial_column_range
        assert ctx.get_offset_provider() is initial_offset_provider

        test_column_range = common.NamedRange(common.Dimension("NewDim"), common.UnitRange(-1, 1))
        test_offset_provider = {"I": "NewDim"}

        with ctx.update(
            closure_column_range=test_column_range, offset_provider=test_offset_provider
        ):
            assert ctx.get_closure_column_range() is test_column_range
            assert ctx.get_offset_provider() is test_offset_provider

        assert ctx.get_closure_column_range() is initial_column_range
        assert ctx.get_offset_provider() is initial_offset_provider

    assert ctx.get_closure_column_range(DEFAULT) is DEFAULT
    assert ctx.get_offset_provider(DEFAULT) is DEFAULT


def test_update_with_no_parameters():
    DEFAULT = object()
    assert ctx.get_closure_column_range(DEFAULT) is DEFAULT
    assert ctx.get_offset_provider(DEFAULT) is DEFAULT

    initial_column_range = common.NamedRange(common.Dimension("IDim"), common.UnitRange(0, 4))
    initial_offset_provider = {}

    with ctx.update(
        closure_column_range=initial_column_range, offset_provider=initial_offset_provider
    ):
        assert ctx.get_closure_column_range() is initial_column_range
        assert ctx.get_offset_provider() is initial_offset_provider

        with ctx.update():
            assert ctx.get_closure_column_range() is initial_column_range
            assert ctx.get_offset_provider() is initial_offset_provider

        assert ctx.get_closure_column_range() is initial_column_range
        assert ctx.get_offset_provider() is initial_offset_provider

    assert ctx.get_closure_column_range(DEFAULT) is DEFAULT
    assert ctx.get_offset_provider(DEFAULT) is DEFAULT


def test_update_with_exception():
    DEFAULT = object()
    assert ctx.get_closure_column_range(DEFAULT) is DEFAULT
    assert ctx.get_offset_provider(DEFAULT) is DEFAULT

    initial_column_range = common.NamedRange(common.Dimension("IDim"), common.UnitRange(0, 4))
    initial_offset_provider = {}

    with pytest.raises(RuntimeError, match="Outer exception"):
        with ctx.update(
            closure_column_range=initial_column_range, offset_provider=initial_offset_provider
        ):
            assert ctx.get_closure_column_range() is initial_column_range
            assert ctx.get_offset_provider() is initial_offset_provider

            test_column_range = common.NamedRange(
                common.Dimension("NewDim"), common.UnitRange(-1, 1)
            )
            test_offset_provider = {"I": "NewDim"}

            with pytest.raises(RuntimeError, match="Inner exception"):
                with ctx.update(
                    closure_column_range=test_column_range, offset_provider=test_offset_provider
                ):
                    assert ctx.get_closure_column_range() is test_column_range
                    assert ctx.get_offset_provider() is test_offset_provider

                    raise RuntimeError("Inner exception")

            assert ctx.get_closure_column_range() is initial_column_range
            assert ctx.get_offset_provider() is initial_offset_provider

            raise RuntimeError("Outer exception")

    assert ctx.get_closure_column_range(DEFAULT) is DEFAULT
    assert ctx.get_offset_provider(DEFAULT) is DEFAULT


def test_within_valid_context():
    assert not ctx.within_valid_context()

    test_column_range = "TEST_COLUMN_RANGE"
    with ctx.update(closure_column_range=test_column_range):
        assert not ctx.within_valid_context()

    test_offset_provider = "TEST_OFFSET_PROVIDER"
    with ctx.update(offset_provider=test_offset_provider):
        assert ctx.within_valid_context()

    assert not ctx.within_valid_context()
