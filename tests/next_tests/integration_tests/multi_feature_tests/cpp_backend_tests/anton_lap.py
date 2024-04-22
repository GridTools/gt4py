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
from gt4py.next.iterator.runtime import closure, fundef, offset
from gt4py.next.iterator.tracing import trace_fencil_definition
from gt4py.next.program_processors.runners.gtfn import run_gtfn
from gt4py.next.type_system import type_specifications as ts

i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return -4.0 * deref(inp) + (
        deref(shift(i, 1)(inp))
        + deref(shift(i, -1)(inp))
        + deref(shift(j, 1)(inp))
        + deref(shift(j, -1)(inp))
    )


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
KDim = gtx.Dimension("KDim")


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

    ijk_field_type = ts.FieldType(
        dims=[IDim, JDim, KDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    )
    prog = trace_fencil_definition(lap_fencil, [ijk_field_type] * 8)
    generated_code = run_gtfn.executor.otf_workflow.translation.generate_stencil_source(
        prog, offset_provider={"i": IDim, "j": JDim}, column_axis=None
    )

    with open(output_file, "w+") as output:
        output.write(generated_code)
