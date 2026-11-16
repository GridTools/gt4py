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
from gt4py.next import common, float64, max_over, min_over, neighbor_sum, where
from gt4py.next.ffront.experimental import concat_where

from next_tests.unit_tests.conftest import asarray


Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))

NV = 5
# every edge keeps at least one neighbor, so max_over and min_over are defined everywhere
_TABLE = np.asarray([[0, 1], [1, -9], [2, 3], [-9, 4], [4, 0], [-9, 2]], dtype=np.int32)


@gtx.field_operator
def gather(inp: gtx.Field[[Vertex], float64]) -> gtx.Field[[Edge, E2VDim], float64]:
    return inp(E2V)


@gtx.field_operator
def reductions(
    inp: gtx.Field[[Vertex], float64],
) -> tuple[gtx.Field[[Edge], float64], gtx.Field[[Edge], float64], gtx.Field[[Edge], float64]]:
    return (
        neighbor_sum(inp(E2V), axis=E2VDim),
        max_over(inp(E2V), axis=E2VDim),
        min_over(inp(E2V), axis=E2VDim),
    )


@gtx.field_operator
def masked_and_concatenated(inp: gtx.Field[[Vertex], float64]) -> gtx.Field[[Edge], float64]:
    positive_sum = neighbor_sum(where(inp(E2V) > 2.0, inp(E2V), 0.0), axis=E2VDim)
    return concat_where(Edge < 3, positive_sum, max_over(inp(E2V), axis=E2VDim))


def _case(xp, skip_value):
    table = np.where(_TABLE == -9, 0 if skip_value is None else skip_value, _TABLE)
    values = np.asarray([1.0, 4.0, 2.0, 8.0, 3.0])
    conn = common._connectivity(
        asarray(xp, table, dtype=np.int32),
        codomain=Vertex,
        domain=common.domain({Edge: len(table), E2VDim: 2}),
        skip_value=skip_value,
    )
    inp = common._field(asarray(xp, values, dtype=np.float64), domain=common.domain({Vertex: NV}))
    is_neighbor = table != skip_value if skip_value is not None else np.ones(table.shape, bool)
    neighbors = [values[row[valid]] for row, valid in zip(table, is_neighbor)]
    return conn, inp, table, is_neighbor, neighbors


def _out(xp, shape, dims):
    return common._field(
        asarray(xp, np.zeros(shape), dtype=np.float64),
        domain=common.domain(dict(zip(dims, shape))),
    )


SKIP_VALUES = [-2, -1, None]


@pytest.mark.parametrize("skip_value", SKIP_VALUES)
def test_gather_to_sparse_field(nd_array_implementation, skip_value):
    xp = nd_array_implementation
    conn, inp, table, is_neighbor, _ = _case(xp, skip_value)
    out = _out(xp, table.shape, (Edge, E2VDim))

    gather(inp, out=out, offset_provider={"E2V": conn})

    expected = inp.asnumpy()[np.where(is_neighbor, table, 0)]
    assert np.array_equal(out.asnumpy()[is_neighbor], expected[is_neighbor])


@pytest.mark.parametrize("skip_value", SKIP_VALUES)
def test_neighbor_reductions(nd_array_implementation, skip_value):
    xp = nd_array_implementation
    conn, inp, table, _, neighbors = _case(xp, skip_value)
    outs = tuple(_out(xp, (len(table),), (Edge,)) for _ in range(3))

    reductions(inp, out=outs, offset_provider={"E2V": conn})

    for out, np_reduction in zip(outs, (np.sum, np.max, np.min)):
        assert np.array_equal(out.asnumpy(), [np_reduction(n) for n in neighbors])


@pytest.mark.parametrize("skip_value", SKIP_VALUES)
def test_where_and_concat_where_over_neighbors(nd_array_implementation, skip_value):
    xp = nd_array_implementation
    conn, inp, table, _, neighbors = _case(xp, skip_value)
    out = _out(xp, (len(table),), (Edge,))

    masked_and_concatenated(inp, out=out, offset_provider={"E2V": conn})

    expected = [np.sum(n[n > 2.0]) if e < 3 else np.max(n) for e, n in enumerate(neighbors)]
    assert np.array_equal(out.asnumpy(), expected)
