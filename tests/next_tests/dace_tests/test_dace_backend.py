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

from gt4py.next.common import Dimension, DimensionKind, Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import FieldOffset
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider, np_as_located_field
from gt4py.next.program_processors.runners.dace_iterator import run_dace_iterator


IDim = Dimension("IDim")
I2IDim = Dimension("I2IDim", kind=DimensionKind.LOCAL)
IOff = FieldOffset("IOff", IDim, (IDim,))
I2IOff = FieldOffset("I2IOff", IDim, (IDim, I2IDim))

i2i_neighbor_data = np.array(
    [
        [2, 1],
        [0, 2],
        [1, 0],
    ],
    dtype=np.int64,
)
i2i_neighbor_table = NeighborTableOffsetProvider(i2i_neighbor_data, IDim, IDim, 2)

ADim = Dimension("ADim", kind=DimensionKind.HORIZONTAL)
BDim = Dimension("BDim", kind=DimensionKind.HORIZONTAL)
A2BDim = Dimension("A2BDim", kind=DimensionKind.LOCAL)
A2BOff = FieldOffset("A2BOff", BDim, (ADim, A2BDim))
KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)

a2b_neighbor_data = np.array(
    [
        [2, 1, 0],
        [0, 2, 1],
    ],
    dtype=np.int64,
)
a2b_neighbor_table = NeighborTableOffsetProvider(a2b_neighbor_data, ADim, BDim, 3)


def test_pointwise():
    @field_operator(backend=run_dace_iterator)
    def add(a: Field[[IDim], float], b: Field[[IDim], float]) -> Field[[IDim], float]:
        return a + b + a - a

    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    b = np_as_located_field(IDim)(np.array([5, 7, 0], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0, 0], dtype=float))
    add(a, b, out=r, offset_provider={})

    assert np.allclose(np.asarray(a) + np.asarray(b), np.asarray(r))


def test_shift_left():
    @field_operator
    def shift_left(a: Field[[IDim], float]) -> Field[[IDim], float]:
        return a(IOff[1])

    @program(backend=run_dace_iterator)
    def shift_program(a: Field[[IDim], float], out: Field[[IDim], float]):
        shift_left(a, out=out, domain={IDim: (0, 2)})

    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0], dtype=float))

    shift_program(a, out=r, offset_provider={"IOff": IDim})

    assert np.allclose(np.asarray(a)[1::], np.asarray(r))


def test_indirect_addressing():
    @field_operator
    def indirect_addressing(a: Field[[IDim], float]) -> Field[[IDim], float]:
        return a(I2IOff[1]) - a(I2IOff[0])

    @program(backend=run_dace_iterator)
    def indirect_addressing_program(z: Field[[IDim], float], out: Field[[IDim], float]):
        indirect_addressing(z, out=out)

    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0, 0], dtype=float))
    e = np_as_located_field(IDim)(np.array([-1, 2, -1], dtype=float))

    indirect_addressing_program(a, out=r, offset_provider={"I2IOff": i2i_neighbor_table})

    assert np.allclose(np.asarray(e), np.asarray(r))


def test_dimensions():
    @field_operator
    def fieldop(b: Field[[BDim, KDim], float]) -> Field[[ADim, KDim], float]:
        return b(A2BOff[0]) - b(A2BOff[1]) + b(A2BOff[2])

    @program(backend=run_dace_iterator)
    def prog(z: Field[[BDim, KDim], float], out: Field[[ADim, KDim], float]):
        fieldop(z, out=out)

    b = np_as_located_field(BDim, KDim)(np.array([[1, 5], [2, 4], [3, 6]], dtype=float))
    r = np_as_located_field(ADim, KDim)(np.array([[0, 0], [0, 0]], dtype=float))
    e = np_as_located_field(ADim, KDim)(np.array([[2, 7], [0, 3]], dtype=float))

    prog(b, out=r, offset_provider={"A2BOff": a2b_neighbor_table})

    assert np.allclose(np.asarray(e), np.asarray(r))
