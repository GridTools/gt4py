# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.embedded import context as embedded_context
from gt4py.next.iterator import embedded, runtime
from gt4py.next.iterator.builtins import (
    as_fieldop,
    deref,
    if_,
    make_const_list,
    map_,
    neighbors,
    plus,
)


E = gtx.Dimension("E")
V = gtx.Dimension("V")
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
E2V = gtx.FieldOffset("E2V", source=V, target=(E, E2VDim))


# 0 --0-- 1 --1-- 2
e2v_arr = np.array([[0, 1], [1, 2]])
e2v_conn = gtx.NeighborTableOffsetProvider(
    table=e2v_arr,
    origin_axis=E,
    neighbor_axis=V,
    max_neighbors=2,
    has_skip_values=False,
)


def test_write_neighbors():
    def testee(inp):
        domain = runtime.UnstructuredDomain({E: range(2)})
        return as_fieldop(lambda it: neighbors(E2V, it), domain)(inp)

    inp = gtx.as_field([V], np.arange(3))
    with embedded_context.new_context(offset_provider={"E2V": e2v_conn}) as ctx:
        result = ctx.run(testee, inp)

    ref = e2v_arr
    np.testing.assert_array_equal(result.asnumpy(), ref)


def test_write_const_list():
    def testee():
        domain = runtime.UnstructuredDomain({E: range(2)})
        return as_fieldop(lambda: make_const_list(42.0), domain)()

    with embedded_context.new_context(offset_provider={}) as ctx:
        result = ctx.run(testee)

    ref = np.asarray([[42.0], [42.0]])

    assert result.domain.dims[0] == E
    assert result.domain.dims[1] == embedded._CONST_DIM  # this is implementation detail
    assert result.shape[1] == 1  # this is implementation detail
    np.testing.assert_array_equal(result.asnumpy(), ref)


def test_write_map_neighbors_and_const_list():
    def testee(inp):
        domain = runtime.UnstructuredDomain({E: range(2)})
        return as_fieldop(lambda x, y: map_(plus)(deref(x), deref(y)), domain)(
            as_fieldop(lambda it: neighbors(E2V, it), domain)(inp),
            as_fieldop(lambda: make_const_list(42.0), domain)(),
        )

    inp = gtx.as_field([V], np.arange(3))
    with embedded_context.new_context(offset_provider={"E2V": e2v_conn}) as ctx:
        result = ctx.run(testee, inp)

    ref = e2v_arr + 42.0
    np.testing.assert_array_equal(result.asnumpy(), ref)


def test_write_map_conditional_neighbors_and_const_list():
    def testee(inp, mask):
        domain = runtime.UnstructuredDomain({E: range(2)})
        return as_fieldop(lambda m, x, y: map_(if_)(deref(m), deref(x), deref(y)), domain)(
            as_fieldop(lambda it: make_const_list(deref(it)), domain)(mask),
            as_fieldop(lambda it: neighbors(E2V, it), domain)(inp),
            as_fieldop(lambda it: make_const_list(deref(it)), domain)(42.0),
        )

    inp = gtx.as_field([V], np.arange(3))
    mask_field = gtx.as_field([E], np.array([True, False]))
    with embedded_context.new_context(offset_provider={"E2V": e2v_conn}) as ctx:
        result = ctx.run(testee, inp, mask_field)

    ref = np.empty_like(e2v_arr, dtype=float)
    ref[0, :] = e2v_arr[0, :]
    ref[1, :] = 42.0
    np.testing.assert_array_equal(result.asnumpy(), ref)


def test_write_non_mapped_conditional_neighbors_and_const_list():
    """
    This test-case demonstrates a non-supported pattern:
    Current ITIR requires the `if_` to be `map_`ed, see `test_write_map_conditional_neighbors_and_const_list`.
    We keep it here for documenting corner cases of the `itir.List` implementation for future discussions.
    """

    pytest.skip("Unsupported.")

    def testee(inp, mask):
        domain = runtime.UnstructuredDomain({E: range(2)})
        return as_fieldop(lambda m, x, y: if_(deref(m), deref(x), deref(y)), domain)(
            mask,
            as_fieldop(lambda it: make_const_list(deref(it)), domain)(42.0),
            as_fieldop(lambda it: neighbors(E2V, it), domain)(inp),
        )

    inp = gtx.as_field([V], np.arange(3))
    mask_field = gtx.as_field([E], np.array([True, False]))
    with embedded_context.new_context(offset_provider={"E2V": e2v_conn}) as ctx:
        result = ctx.run(testee, inp, mask_field)

    ref = np.empty_like(e2v_arr, dtype=float)
    ref[0, :] = e2v_arr[0, :]
    ref[1, :] = 42.0
    np.testing.assert_array_equal(result.asnumpy(), ref)


def test_write_map_const_list_and_const_list():
    def testee():
        domain = runtime.UnstructuredDomain({E: range(2)})
        return as_fieldop(lambda x, y: map_(plus)(deref(x), deref(y)), domain)(
            as_fieldop(lambda: make_const_list(1.0), domain)(),
            as_fieldop(lambda: make_const_list(42.0), domain)(),
        )

    with embedded_context.new_context(offset_provider={}) as ctx:
        result = ctx.run(testee)

    ref = np.asarray([[43.0], [43.0]])

    assert result.domain.dims[0] == E
    assert result.domain.dims[1] == embedded._CONST_DIM  # this is implementation detail
    assert result.shape[1] == 1  # this is implementation detail
    np.testing.assert_array_equal(result.asnumpy(), ref)
