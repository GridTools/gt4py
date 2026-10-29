# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""An offset provider keyed by a (local) dimension class: `compile` and call agree on the key."""

import numpy as np
import pytest

import gt4py.next as gtx
from next_tests.toy_connectivity import Edge, V2EDim, Vertex


V2E = gtx.FieldOffset(V2EDim.tag, source=Edge, target=(Vertex, V2EDim))


@gtx.field_operator
def _first_neighbor(
    e: gtx.Field[gtx.Dims[Edge], gtx.float64],
) -> gtx.Field[gtx.Dims[Vertex], gtx.float64]:
    return e(V2E[0])


@gtx.program
def first_neighbor(
    e: gtx.Field[gtx.Dims[Edge], gtx.float64], out: gtx.Field[gtx.Dims[Vertex], gtx.float64]
):
    _first_neighbor(e, out=out)


@pytest.fixture
def data():
    e = gtx.as_field([Edge], np.arange(3.0))
    out = gtx.as_field([Vertex], np.zeros(2))
    v2e = gtx.as_connectivity(
        domain={Vertex: 2, V2EDim: 1}, codomain=Edge, data=np.array([[2], [0]], np.int32)
    )
    return e, out, v2e


def test_compile_then_call_with_class_keyed_provider(data):
    e, out, v2e = data
    program = first_neighbor.with_backend(gtx.gtfn_cpu).compile(offset_provider={V2EDim: v2e})
    # a cache miss would raise: `enable_jit=False`
    program(e, out, offset_provider={V2EDim: v2e}, enable_jit=False)
    assert np.allclose(out.asnumpy(), [2.0, 0.0])
    # ...and the string-keyed spelling of the same provider is the same variant
    program(e, out, offset_provider={V2EDim.tag: v2e}, enable_jit=False)


def test_connectivities_option_with_class_keys(data):
    e, out, v2e = data
    program = (
        first_neighbor.with_backend(gtx.gtfn_cpu)
        .with_compilation_options(connectivities={V2EDim: v2e})
        .compile()
    )
    program(e, out, offset_provider={V2EDim: v2e}, enable_jit=False)
    assert np.allclose(out.asnumpy(), [2.0, 0.0])
