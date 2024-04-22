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

import sys

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import closure, fundef
from gt4py.next.iterator.tracing import trace_fencil_definition
from gt4py.next.iterator.transforms import LiftMode
from gt4py.next.program_processors.runners.gtfn import run_gtfn
from gt4py.next.type_system import type_specifications as ts


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
KDim = gtx.Dimension("KDim")


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

    ijk_field_type = ts.FieldType(
        dims=[IDim, JDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    )
    prog = trace_fencil_definition(tridiagonal_solve_fencil, [ijk_field_type] * 8)
    offset_provider = {"I": gtx.Dimension("IDim"), "J": gtx.Dimension("JDim")}
    generated_code = run_gtfn.executor.otf_workflow.translation.generate_stencil_source(
        prog,
        offset_provider=offset_provider,
        runtime_lift_mode=LiftMode.SIMPLE_HEURISTIC,
        column_axis=KDim,
    )

    with open(output_file, "w+") as output:
        output.write(generated_code)
