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

from gt4py.next.common import Dimension
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.iterator.runtime import closure, fendef, fundef, offset
from gt4py.next.program_processors.runners import double_roundtrip, roundtrip


I = offset("I")
J = offset("J")
I_loc = Dimension("I_loc")
J_loc = Dimension("J_loc")


@fundef
def foo(inp):
    return deref(shift(J, 1)(inp))


@fendef(offset_provider={"I": I_loc, "J": J_loc})
def fencil(output, input):
    closure(
        cartesian_domain(named_range(I_loc, 0, 1), named_range(J_loc, 0, 1)),
        foo,
        output,
        [input],
    )


@fendef(offset_provider={"I": J_loc, "J": I_loc})
def fencil_swapped(output, input):
    closure(
        cartesian_domain(named_range(I_loc, 0, 1), named_range(J_loc, 0, 1)),
        foo,
        output,
        [input],
    )


def test_cartesian_offset_provider():
    inp = np_as_located_field(I_loc, J_loc)(np.asarray([[0, 42], [1, 43]]))
    out = np_as_located_field(I_loc, J_loc)(np.asarray([[-1]]))

    fencil(out, inp)
    assert out[0][0] == 42

    fencil_swapped(out, inp)
    assert out[0][0] == 1

    fencil(out, inp, backend=roundtrip.executor)
    assert out[0][0] == 42

    fencil(out, inp, backend=double_roundtrip.executor)
    assert out[0][0] == 42


@fundef
def delay_complete_shift(inp):
    return deref(shift(I, J, 1, 1)(inp))


@fendef(offset_provider={"I": J_loc, "J": I_loc})
def delay_complete_shift_fencil(output, input):
    closure(
        cartesian_domain(named_range(I_loc, 0, 1), named_range(J_loc, 0, 1)),
        delay_complete_shift,
        output,
        [input],
    )


def test_delay_complete_shift():
    inp = np_as_located_field(I_loc, J_loc)(np.asarray([[0, 42], [1, 43]]))

    out = np_as_located_field(I_loc, J_loc)(np.asarray([[-1]]))
    delay_complete_shift_fencil(out, inp)
    assert out[0, 0] == 43

    out = np_as_located_field(I_loc, J_loc)(np.asarray([[-1]]))
    delay_complete_shift_fencil(out, inp, backend=roundtrip.executor)
    assert out[0, 0] == 43
