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
from gt4py.next.iterator.runtime import CartesianAxis, closure, fundef
from gt4py.next.iterator.tracing import trace
from gt4py.next.iterator.transforms import LiftMode
from gt4py.next.program_processors.codegens.gtfn.gtfn_backend import generate


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fundef
def tridiag_forward(state, a, b, c, d):
    return make_tuple(
        deref(c) / (deref(b) - deref(a) * tuple_get(0, state)),
        (deref(d) - deref(a) * tuple_get(1, state)) / (deref(b) - deref(a) * tuple_get(0, state)),
    )


@fundef
def tridiag_backward(x_kp1, cpdp):
    cpdpv = deref(cpdp)
    cp = tuple_get(0, cpdpv)
    dp = tuple_get(1, cpdpv)
    return dp - cp * x_kp1


@fundef
def solve_tridiag(a, b, c, d):
    cpdp = lift(scan(tridiag_forward, True, make_tuple(0.0, 0.0)))(a, b, c, d)
    return scan(tridiag_backward, False, 0.0)(cpdp)


def tridiagonal_solve_fencil(isize, jsize, ksize, a, b, c, d, x):
    closure(
        cartesian_domain(
            named_range(IDim, 0, isize), named_range(JDim, 0, jsize), named_range(KDim, 0, ksize)
        ),
        solve_tridiag,
        x,
        [a, b, c, d],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = trace(tridiagonal_solve_fencil, [None] * 8)
    offset_provider = {"I": CartesianAxis("IDim"), "J": CartesianAxis("JDim")}
    generated_code = generate(
        prog,
        offset_provider=offset_provider,
        lift_mode=LiftMode.SIMPLE_HEURISTIC,
        column_axis=KDim,
    )

    with open(output_file, "w+") as output:
        output.write(generated_code)
