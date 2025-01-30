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
from gt4py.next.iterator.builtins import deref, named_range, shift, unstructured_domain, as_fieldop
from gt4py.next.iterator.runtime import set_at, fendef, fundef, offset

from next_tests.unit_tests.conftest import program_processor, run_processor
from gt4py.next.iterator.embedded import StridedConnectivityField


LocA = gtx.Dimension("LocA")
LocAB = gtx.Dimension("LocAB")
LocB = gtx.Dimension("LocB")  # unused

LocA2LocAB = offset("O")
LocA2LocAB_offset_provider = StridedConnectivityField(
    domain_dims=(LocA, gtx.Dimension("Dummy", kind=gtx.DimensionKind.LOCAL)),
    codomain_dim=LocAB,
    max_neighbors=2,
)


@fundef
def foo(inp):
    return deref(shift(LocA2LocAB, 0)(inp)) + deref(shift(LocA2LocAB, 1)(inp))


@fendef(offset_provider={"O": LocA2LocAB_offset_provider})
def fencil(size, out, inp):
    domain = unstructured_domain(named_range(LocA, 0, size))
    set_at(as_fieldop(foo, domain)(inp), domain, out)


@pytest.mark.uses_strided_neighbor_offset
def test_strided_offset_provider(program_processor):
    program_processor, validate = program_processor

    LocA_size = 2
    max_neighbors = LocA2LocAB_offset_provider.__gt_type__().max_neighbors
    LocAB_size = LocA_size * max_neighbors

    rng = np.random.default_rng()
    inp = gtx.as_field([LocAB], rng.normal(size=(LocAB_size,)))
    out = gtx.as_field([LocA], np.zeros((LocA_size,)))
    ref = np.sum(inp.asnumpy().reshape(LocA_size, max_neighbors), axis=-1)

    run_processor(fencil, program_processor, LocA_size, out, inp)

    if validate:
        assert np.allclose(out.asnumpy(), ref)
