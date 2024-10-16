# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import set_at, fendef, fundef, offset
from gt4py.next.program_processors.runners import double_roundtrip, roundtrip


I = offset("I")
J = offset("J")
I_loc = gtx.Dimension("I_loc")
J_loc = gtx.Dimension("J_loc")


@fundef
def foo(inp):
    return deref(shift(J, 1)(inp))


@fendef(offset_provider={"I": I_loc, "J": J_loc})
def fencil(output, input):
    domain = cartesian_domain(named_range(I_loc, 0, 1), named_range(J_loc, 0, 1))
    set_at(as_fieldop(foo, domain)(input), domain, output)


@fendef(offset_provider={"I": J_loc, "J": I_loc})
def fencil_swapped(output, input):
    domain = cartesian_domain(named_range(I_loc, 0, 1), named_range(J_loc, 0, 1))
    set_at(as_fieldop(foo, domain)(input), domain, output)


def test_cartesian_offset_provider():
    inp = gtx.as_field([I_loc, J_loc], np.asarray([[0, 42], [1, 43]]))
    out = gtx.as_field([I_loc, J_loc], np.asarray([[-1]]))

    fencil(out, inp)
    assert out[0][0] == 42

    fencil_swapped(out, inp)
    assert out[0][0] == 1

    fencil(out, inp, backend=roundtrip.default)
    assert out[0][0] == 42

    fencil(out, inp, backend=double_roundtrip.backend)
    assert out[0][0] == 42
