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
from gt4py.next.program_processors.codegens.gtfn.gtfn_backend import generate


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
KDim = gtx.Dimension("KDim")


@fundef
def copy_stencil(inp):
    return deref(inp)


def copy_fencil(isize, jsize, ksize, inp, out):
    closure(
        cartesian_domain(
            named_range(IDim, 0, isize), named_range(JDim, 0, jsize), named_range(KDim, 0, ksize)
        ),
        copy_stencil,
        out,
        [inp],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = trace_fencil_definition(copy_fencil, [None] * 5, use_arg_types=False)
    generated_code = generate(prog, offset_provider={})

    with open(output_file, "w+") as output:
        output.write(generated_code)
