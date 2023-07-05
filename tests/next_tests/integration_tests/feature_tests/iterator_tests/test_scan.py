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

import gt4py.next as gtx
from gt4py.next.iterator.builtins import cartesian_domain, deref, named_range, scan, shift, if_, eq
from gt4py.next.iterator.runtime import fundef, offset

from next_tests.integration_tests.cases import IDim, KDim
from next_tests.unit_tests.conftest import lift_mode, program_processor, run_processor


def test_scan_in_stencil(program_processor, lift_mode):
    program_processor, validate = program_processor

    isize = 1
    ksize = 3
    Koff = offset("Koff")
    inp = gtx.np_as_located_field(IDim, KDim)(
        np.copy(np.broadcast_to(np.arange(0, ksize), (isize, ksize)))
    )
    out = gtx.np_as_located_field(IDim, KDim)(np.zeros((isize, ksize)))

    reference = np.zeros((isize, ksize - 1))
    reference[:, 0] = inp[:, 0] + inp[:, 1]
    for k in range(1, ksize - 1):
        reference[:, k] = reference[:, k - 1] + inp[:, k] + inp[:, k + 1]

    @fundef
    def sum(state, k, kp):
        return state + deref(k) + deref(kp)

    @fundef
    def wrapped(inp):
        return scan(sum, True, 0.0)(inp, shift(Koff, 1)(inp))

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


def test_scan_vertical_boundary_condition(program_processor, lift_mode):
    program_processor, validate = program_processor

    isize = 1
    ksize = 5
    Koff = offset("Koff")
    inp = gtx.np_as_located_field(IDim, KDim)(
        np.copy(np.broadcast_to(np.arange(0, ksize, dtype=np.float64), (isize, ksize))))
    kindices = gtx.np_as_located_field(KDim)(np.arange(0, ksize, dtype=np.int32))
    out = gtx.np_as_located_field(IDim, KDim)(np.zeros((isize, ksize)))

    reference = np.zeros((isize, ksize))
    reference[:, 0] = inp[:, 0]
    for k in range(1, ksize - 1):
        reference[:, k] = reference[:, k - 1] + (inp[:, k - 1] + inp[:, k + 1])/2
    reference[:, -1] = reference[:, -2] + inp[:, -1]

    @fundef
    def scan_pass(state, inp, kindex, ksize):
        value_lowest_level = state+deref(inp)
        value_interior_levels = state+(deref(shift(Koff, -1)(inp))+deref(shift(Koff, 1)(inp)))/2.
        value_upper_level = state+deref(inp)
        return if_(
            eq(deref(kindex), 0),
            value_lowest_level,
            if_(eq(deref(kindex), deref(ksize) - 1),
                value_upper_level,
                value_interior_levels))

    @fundef
    def wrapped(inp, k_index, ksize):
        return scan(scan_pass, True, 0.0)(inp, k_index, ksize)

    run_processor(
        wrapped[cartesian_domain(named_range(IDim, 0, isize), named_range(KDim, 0, ksize))],
        program_processor,
        inp,
        kindices,
        np.int32(ksize),
        out=out,
        lift_mode=lift_mode,
        offset_provider={"Koff": KDim},
        column_axis=KDim,
    )

    if validate:
        assert np.allclose(out[:, :], reference)