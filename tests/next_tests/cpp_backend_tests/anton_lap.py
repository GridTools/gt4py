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

import sys

from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import CartesianAxis, closure, fundef, offset
from gt4py.next.iterator.tracing import trace
from gt4py.next.program_processors.codegens.gtfn.gtfn_backend import generate


@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    return lambda inp: ldif(d)(shift(d, 1)(inp))


@fundef
def dif2(d):
    return lambda inp: ldif(d)(lift(rdif(d))(inp))


i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


def lap_fencil(i_size, j_size, k_size, i_off, j_off, k_off, out, inp):
    closure(
        cartesian_domain(
            named_range(IDim, i_off, i_size + i_off),
            named_range(JDim, j_off, j_size + j_off),
            named_range(KDim, k_off, k_size + k_off),
        ),
        lap,
        out,
        [inp],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = trace(lap_fencil, [None] * 8)
    generated_code = generate(prog, offset_provider={"i": IDim, "j": JDim})

    with open(output_file, "w+") as output:
        output.write(generated_code)
