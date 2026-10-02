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
from gt4py.next import common, neighbor_sum
from gt4py.next.embedded import nd_array_field


Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2E2CDim = gtx.Dimension("C2E2C", kind=gtx.DimensionKind.LOCAL)
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
C2E2C = gtx.FieldOffset("C2E2C", source=Cell, target=(Cell, C2E2CDim))

# edges 0, 1 are boundary edges (one skip neighbour); cells 0, 1 each own one of them
E2C_TABLE = np.asarray(
    [[0, -1], [1, -1], [0, 2], [0, 3], [1, 3], [1, 2], [2, 3], [2, 3]], dtype=np.int32
)
C2E_TABLE = np.asarray([[0, 2, 3], [1, 4, 5], [2, 5, 6], [3, 4, 7]], dtype=np.int32)
C2E2C_TABLE = np.asarray([[2, 3, 3], [3, 2, 2], [0, 1, 3], [0, 1, 2]], dtype=np.int32)
THETA = np.asarray([1.0, 3.0, 7.0, 15.0])
GEOFAC = np.arange(12, dtype=float).reshape(4, 3) / 10.0 + 1.0


@gtx.field_operator
def _nabla(
    theta: gtx.Field[gtx.Dims[Cell], float], geofac: gtx.Field[gtx.Dims[Cell, C2EDim], float]
) -> gtx.Field[gtx.Dims[Cell], float]:
    # `z` is not defined on the boundary edges, which only cells 0, 1 read
    z = theta(E2C[1]) - theta(E2C[0])
    return neighbor_sum(z(C2E) * geofac, axis=C2EDim)


@gtx.field_operator
def _nabla_c2e2c(
    theta: gtx.Field[gtx.Dims[Cell], float], geofac: gtx.Field[gtx.Dims[Cell, C2EDim], float]
) -> gtx.Field[gtx.Dims[Cell], float]:
    return neighbor_sum(_nabla(theta, geofac)(C2E2C), axis=C2E2CDim)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def nabla(
    theta: gtx.Field[gtx.Dims[Cell], float],
    geofac: gtx.Field[gtx.Dims[Cell, C2EDim], float],
    out: gtx.Field[gtx.Dims[Cell], float],
    lo: gtx.int32,
    hi: gtx.int32,
):
    _nabla(theta, geofac, out=out, domain={Cell: (lo, hi)})


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def nabla_c2e2c(
    theta: gtx.Field[gtx.Dims[Cell], float],
    geofac: gtx.Field[gtx.Dims[Cell, C2EDim], float],
    out: gtx.Field[gtx.Dims[Cell], float],
    lo: gtx.int32,
    hi: gtx.int32,
):
    _nabla_c2e2c(theta, geofac, out=out, domain={Cell: (lo, hi)})


@pytest.fixture(params=[True, False], ids=["fill", "no_fill"])
def fill_holes(request, monkeypatch):
    monkeypatch.setattr(nd_array_field, "_FILL_GATHER_HOLES", request.param)
    return request.param


def _run(program, lo, hi):
    offset_provider = {
        "E2C": gtx.as_connectivity([Edge, E2CDim], Cell, data=E2C_TABLE, skip_value=-1),
        "C2E": gtx.as_connectivity([Cell, C2EDim], Edge, data=C2E_TABLE),
        "C2E2C": gtx.as_connectivity([Cell, C2E2CDim], Cell, data=C2E2C_TABLE),
    }
    out = gtx.as_field([Cell], np.full(4, -1.0))
    program(
        gtx.as_field([Cell], THETA),
        gtx.as_field([Cell, C2EDim], GEOFAC),
        out,
        lo,
        hi,
        offset_provider=offset_provider,
    )
    return out.asnumpy()


def _nabla_reference(cells):
    return np.asarray(
        [
            sum(
                GEOFAC[c, j]
                * (THETA[E2C_TABLE[C2E_TABLE[c, j], 1]] - THETA[E2C_TABLE[C2E_TABLE[c, j], 0]])
                for j in range(3)
            )
            for c in cells
        ]
    )


def test_holes_not_read(fill_holes):
    result = _run(nabla, 2, 4)

    assert np.array_equal(result[:2], [-1.0, -1.0])
    assert np.allclose(result[2:], _nabla_reference([2, 3]))


# With the fill off, the values at points that read a hole are unspecified.


def test_holes_read(fill_holes):
    result = _run(nabla, 0, 4)

    assert np.allclose(result[2:], _nabla_reference([2, 3]))
    if fill_holes:
        assert np.all(np.isnan(result[:2]))


def test_holes_read_through_second_shift(fill_holes):
    result = _run(nabla_c2e2c, 2, 4)

    assert np.array_equal(result[:2], [-1.0, -1.0])
    if fill_holes:
        assert np.all(np.isnan(result[2:]))


@gtx.field_operator
def _sum_c2e(z: gtx.Field[gtx.Dims[Edge], float]) -> gtx.Field[gtx.Dims[Cell], float]:
    return neighbor_sum(z(C2E), axis=C2EDim)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sum_c2e(z: gtx.Field[gtx.Dims[Edge], float], out: gtx.Field[gtx.Dims[Cell], float]):
    _sum_c2e(z, out=out)


@gtx.field_operator
def _sum_c2e2c(y: gtx.Field[gtx.Dims[Cell], float]) -> gtx.Field[gtx.Dims[Cell], float]:
    return neighbor_sum(y(C2E2C), axis=C2E2CDim)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sum_c2e2c(y: gtx.Field[gtx.Dims[Cell], float], out: gtx.Field[gtx.Dims[Cell], float]):
    _sum_c2e2c(y, out=out, domain={Cell: (2, 4)})


def test_partial_local_dimension_raises():
    # neighbours 0, 1 of every cell are outside the edge field, neighbour 2 inside
    z = gtx.as_field(common.domain({Edge: (4, 8)}), np.asarray([1.0, 10.0, 100.0, 1000.0]))
    table = np.asarray([[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]], dtype=np.int32)
    out = gtx.as_field([Cell], np.full(4, -1.0))

    with pytest.raises(ValueError, match="partial local dimension"):
        sum_c2e(
            z,
            out,
            offset_provider={"C2E": gtx.as_connectivity([Cell, C2EDim], Edge, data=table)},
        )


def test_partial_local_dimension_same_codomain_raises():
    y = gtx.as_field(common.domain({Cell: (2, 4)}), np.asarray([10.0, 100.0]))
    table = np.asarray([[0, 1, 2], [0, 1, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int32)
    out = gtx.as_field([Cell], np.full(4, -1.0))

    with pytest.raises(ValueError, match="partial local dimension"):
        sum_c2e2c(
            y,
            out,
            offset_provider={"C2E2C": gtx.as_connectivity([Cell, C2E2CDim], Cell, data=table)},
        )
