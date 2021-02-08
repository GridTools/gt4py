# GridTools Compiler Toolchain (GTC) - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GTC project and the GridTools framework.
# GTC is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
# FVM nabla stencil
#

import os
import sys

from gtc_unstructured.frontend.frontend import GTScriptCompilationTask
from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Edge,
    Field,
    Local,
    Mesh,
    Vertex,
    computation,
    edges,
    location,
    vertices,
)
from gtc_unstructured.irs.common import DataType
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


dtype = DataType.FLOAT64


def nabla(
    mesh: Mesh,
    S_MXX: Field[Edge, dtype],
    S_MYY: Field[Edge, dtype],
    pp: Field[Vertex, dtype],
    pnabla_MXX: Field[Vertex, dtype],
    pnabla_MYY: Field[Vertex, dtype],
    vol: Field[Vertex, dtype],
    sign: Field[Vertex, Local[Edge], dtype],
):
    with computation(FORWARD):
        with location(Edge) as e:
            zavg = 0.5 * sum(pp[v] for v in vertices(e))
            zavgS_MXX = S_MXX * zavg
            zavgS_MYY = S_MYY * zavg
        with location(Vertex) as v:
            pnabla_MXX = sum(zavgS_MXX[e] * sign[v, e] for e in edges(v))
            pnabla_MYY = sum(zavgS_MYY[e] * sign[v, e] for e in edges(v))
            pnabla_MXX = pnabla_MXX / vol
            pnabla_MYY = pnabla_MYY / vol


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator

    generated_code = GTScriptCompilationTask(nabla).generate(
        debug=False, code_generator=code_generator
    )

    # print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_fvm_nabla_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)


if __name__ == "__main__":
    main()
