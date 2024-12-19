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

pytest.importorskip("atlas4py")

from gt4py import next as gtx
from gt4py.next import allocators, neighbor_sum
from gt4py.next.iterator import atlas_utils

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)
from next_tests.integration_tests.multi_feature_tests.fvm_nabla_setup import (
    E2V,
    V2E,
    E2VDim,
    Edge,
    V2EDim,
    Vertex,
    assert_close,
    nabla_setup,
)


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
    _, allocator = exec_alloc_descriptor.executor, exec_alloc_descriptor.allocator

    setup = nabla_setup(allocator=allocator)

    zavgS = gtx.zeros({Edge: setup.edges_size}, allocator=allocator)

    compute_zavgS.with_backend(
        None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor
    )(
        setup.input_field,
        setup.S_fields[0],
        out=zavgS,
        offset_provider={"E2V": setup.edges2node_connectivity},
    )

    assert_close(-199755464.25741270, np.min(zavgS.asnumpy()))
    assert_close(388241977.58389181, np.max(zavgS.asnumpy()))


def test_ffront_nabla(exec_alloc_descriptor):
    _, allocator = exec_alloc_descriptor.executor, exec_alloc_descriptor.allocator

    setup = nabla_setup(allocator=allocator)

    pnabla_MXX = gtx.zeros({Vertex: setup.nodes_size}, allocator=allocator)
    pnabla_MYY = gtx.zeros({Vertex: setup.nodes_size}, allocator=allocator)

    pnabla.with_backend(None if exec_alloc_descriptor.executor is None else exec_alloc_descriptor)(
        setup.input_field,
        setup.S_fields,
        setup.sign_field,
        setup.vol_field,
        out=(pnabla_MXX, pnabla_MYY),
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    # TODO this check is not sensitive enough, need to implement a proper numpy reference!
    assert_close(-3.5455427772566003e-003, np.min(pnabla_MXX.asnumpy()))
    assert_close(3.5455427772565435e-003, np.max(pnabla_MXX.asnumpy()))
    assert_close(-3.3540113705465301e-003, np.min(pnabla_MYY.asnumpy()))
    assert_close(3.3540113705465301e-003, np.max(pnabla_MYY.asnumpy()))
