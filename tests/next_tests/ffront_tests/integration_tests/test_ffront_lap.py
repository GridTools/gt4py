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

from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Dimension, FieldOffset
from gt4py.next.iterator.embedded import np_as_located_field


IDim = Dimension("IDim")
JDim = Dimension("JDim")

Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = FieldOffset("Joff", source=JDim, target=(JDim,))


@field_operator
def lap(in_field: Field[[IDim, JDim], "float"]) -> Field[[IDim, JDim], "float"]:
    return (
        -4.0 * in_field
        + in_field(Ioff[1])
        + in_field(Joff[1])
        + in_field(Ioff[-1])
        + in_field(Joff[-1])
    )


@field_operator
def laplap(in_field: Field[[IDim, JDim], "float"]) -> Field[[IDim, JDim], "float"]:
    return lap(lap(in_field))


@program
def lap_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lap(in_field, out=out_field[1:-1, 1:-1])


@program
def laplap_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    laplap(in_field, out=out_field[2:-2, 2:-2])


def lap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]


def test_ffront_lap():
    shape = (20, 20)
    as_ij = np_as_located_field(IDim, JDim)
    input = as_ij(np.fromfunction(lambda x, y: x**2 + y**2, shape))

    result_lap = as_ij(np.zeros_like(input))
    lap_program(input, result_lap, offset_provider={"Ioff": IDim, "Joff": JDim})
    assert np.allclose(np.asarray(result_lap)[1:-1, 1:-1], lap_ref(np.asarray(input)))

    result_laplap = as_ij(np.zeros_like(input))
    laplap_program(input, result_laplap, offset_provider={"Ioff": IDim, "Joff": JDim})
    assert np.allclose(np.asarray(result_laplap)[2:-2, 2:-2], lap_ref(lap_ref(np.asarray(input))))
