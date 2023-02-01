# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from gt4py.next.common import Dimension
from gt4py.next.iterator.builtins import deref, lift, named_range, shift, unstructured_domain
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider, np_as_located_field
from gt4py.next.iterator.runtime import closure, fendef, fundef, offset

from .conftest import run_processor


LocA = Dimension("LocA")
LocAB = Dimension("LocAB")
LocB = Dimension("LocB")  # unused

LocA2LocAB = offset("O")
LocA2LocAB_offset_provider = StridedNeighborOffsetProvider(
    origin_axis=LocA, neighbor_axis=LocAB, max_neighbors=2, has_skip_values=False
)


@fundef
def foo(inp):
    return deref(shift(LocA2LocAB, 0)(inp)) + deref(shift(LocA2LocAB, 1)(inp))


@fendef(offset_provider={"O": LocA2LocAB_offset_provider})
def fencil(size, out, inp):
    closure(
        unstructured_domain(named_range(LocA, 0, size)),
        foo,
        out,
        [inp],
    )


def test_strided_offset_provider(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    LocA_size = 2
    max_neighbors = LocA2LocAB_offset_provider.max_neighbors
    LocAB_size = LocA_size * max_neighbors

    rng = np.random.default_rng()
    inp = np_as_located_field(LocAB)(
        rng.normal(
            size=(LocAB_size,),
        )
    )
    out = np_as_located_field(LocA)(np.zeros((LocA_size,)))
    ref = np.sum(np.asarray(inp).reshape(LocA_size, max_neighbors), axis=-1)

    run_processor(fencil, program_processor, LocA_size, out, inp)

    if validate:
        assert np.allclose(out, ref)
