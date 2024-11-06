# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

import numpy as np
import pytest

pytest.importorskip("always_skip")
pytest.importorskip("atlas4py")

from gt4py import next as gtx
from gt4py.next import allocators, neighbor_sum
from gt4py.next.iterator import atlas_utils

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)
from next_tests.integration_tests.multi_feature_tests.fvm_nabla_setup import (
    assert_close,
    nabla_setup,
)


Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)

V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))


@gtx.field_operator
def compute_zavgS(
    pp: gtx.Field[[Vertex], float], S_M: gtx.Field[[Edge], float]
) -> gtx.Field[[Edge], float]:
    zavg = 0.5 * (pp(E2V[0]) + pp(E2V[1]))
    return S_M * zavg


@gtx.field_operator
def compute_pnabla(
    pp: gtx.Field[[Vertex], float],
    S_M: gtx.Field[[Edge], float],
    sign: gtx.Field[[Vertex, V2EDim], float],
    vol: gtx.Field[[Vertex], float],
) -> gtx.Field[[Vertex], float]:
    zavgS = compute_zavgS(pp, S_M)
    pnabla_M = neighbor_sum(zavgS(V2E) * sign, axis=V2EDim)
    return pnabla_M / vol


@gtx.field_operator
def pnabla(
    pp: gtx.Field[[Vertex], float],
    S_M: Tuple[gtx.Field[[Edge], float], gtx.Field[[Edge], float]],
    sign: gtx.Field[[Vertex, V2EDim], float],
    vol: gtx.Field[[Vertex], float],
) -> Tuple[gtx.Field[[Vertex], float], gtx.Field[[Vertex], float]]:
    return compute_pnabla(pp, S_M[0], sign, vol), compute_pnabla(pp, S_M[1], sign, vol)


def test_ffront_compute_zavgS(exec_alloc_descriptor):
    executor, allocator = exec_alloc_descriptor.executor, exec_alloc_descriptor.allocator

    setup = nabla_setup()

    pp = gtx.as_field([Vertex], setup.input_field, allocator=allocator)
    S_M = tuple(map(gtx.as_field.partial([Edge], allocator=allocator), setup.S_fields))

    zavgS = gtx.zeros({Edge: setup.edges_size}, allocator=allocator)

    e2v = gtx.NeighborTableOffsetProvider(
        atlas_utils.AtlasTable(setup.edges2node_connectivity).asnumpy(), Edge, Vertex, 2, False
    )

    compute_zavgS.with_backend(exec_alloc_descriptor)(
        pp, S_M[0], out=zavgS, offset_provider={"E2V": e2v}
    )

    assert_close(-199755464.25741270, np.min(zavgS.asnumpy()))
    assert_close(388241977.58389181, np.max(zavgS.asnumpy()))


def test_ffront_nabla(exec_alloc_descriptor):
    executor, allocator = exec_alloc_descriptor.executor, exec_alloc_descriptor.allocator

    setup = nabla_setup()

    sign = gtx.as_field([Vertex, V2EDim], setup.sign_field, allocator=allocator)
    pp = gtx.as_field([Vertex], setup.input_field, allocator=allocator)
    S_M = tuple(map(gtx.as_field.partial([Edge], allocator=allocator), setup.S_fields))
    vol = gtx.as_field([Vertex], setup.vol_field, allocator=allocator)

    pnabla_MXX = gtx.zeros({Vertex: setup.nodes_size}, allocator=allocator)
    pnabla_MYY = gtx.zeros({Vertex: setup.nodes_size}, allocator=allocator)

    e2v = gtx.NeighborTableOffsetProvider(
        atlas_utils.AtlasTable(setup.edges2node_connectivity).asnumpy(), Edge, Vertex, 2, False
    )
    v2e = gtx.NeighborTableOffsetProvider(
        atlas_utils.AtlasTable(setup.nodes2edge_connectivity).asnumpy(), Vertex, Edge, 7
    )

    pnabla.with_backend(exec_alloc_descriptor)(
        pp, S_M, sign, vol, out=(pnabla_MXX, pnabla_MYY), offset_provider={"E2V": e2v, "V2E": v2e}
    )

    # TODO this check is not sensitive enough, need to implement a proper numpy reference!
    assert_close(-3.5455427772566003e-003, np.min(pnabla_MXX.asnumpy()))
    assert_close(3.5455427772565435e-003, np.max(pnabla_MXX.asnumpy()))
    assert_close(-3.3540113705465301e-003, np.min(pnabla_MYY.asnumpy()))
    assert_close(3.3540113705465301e-003, np.max(pnabla_MYY.asnumpy()))
