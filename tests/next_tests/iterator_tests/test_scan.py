# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.iterator.builtins import cartesian_domain, deref, lift, named_range, scan, shift
from gt4py.next.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from gt4py.next.iterator.runtime import fundef, offset
from gt4py.next.program_processors.codegens.gtfn import gtfn_backend
from gt4py.next.program_processors.runners import gtfn_cpu, roundtrip

from .conftest import run_processor


def test_scan_in_stencil(program_processor, lift_mode):
    program_processor, validate = program_processor

    isize = 1
    ksize = 3
    IDim = Dimension("I")
    KDim = Dimension("K")
    Koff = offset("Koff")
    inp = np_as_located_field(IDim, KDim)(np.ones((isize, ksize)))
    out = np_as_located_field(IDim, KDim)(np.zeros((isize, ksize)))

    reference = np.zeros((isize, ksize - 1))
    reference[:, 0] = inp[:, 0] + inp[:, 1]
    for k in range(1, ksize - 1):
        reference[:, k] = reference[:, k - 1] + inp[:, k] + inp[:, k + 1]

    @fundef
    def sum(state, k, kp):
        return state + deref(k) + deref(kp)

    @fundef
    def shifted(inp):
        return deref(shift(Koff, 1)(inp))

    @fundef
    def wrapped(inp):
        return scan(sum, True, 0.0)(inp, lift(shifted)(inp))

    run_processor(
        wrapped[cartesian_domain(named_range(IDim, 0, isize), named_range(KDim, 0, ksize - 1))],
        program_processor,
        inp,
        out=out,
        lift_mode=lift_mode,
        offset_provider={"Koff": KDim},
        column_axis=KDim,
    )

    if validate:
        assert np.allclose(out[:, :-1], reference)
