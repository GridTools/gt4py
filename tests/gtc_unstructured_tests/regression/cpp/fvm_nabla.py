# -*- coding: utf-8 -*-
#
# FVM nabla stencil
#

import os
import sys
import types

from gtc_unstructured.frontend.frontend import GTScriptCompilationTask
from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Edge,
    Field,
    SparseField,
    Vertex,
    computation,
    Connectivity,
    location,
)
from gtc_unstructured.irs.common import DataType
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator

E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 2, False],))
V2E = types.new_class("V2E", (Connectivity[Vertex, Edge, 7, True],))
dtype = DataType.FLOAT64


def nabla(
    v2e: V2E,
    e2v: E2V,
    S_MXX: Field[Edge, dtype],
    S_MYY: Field[Edge, dtype],
    pp: Field[Vertex, dtype],
    pnabla_MXX: Field[Vertex, dtype],
    pnabla_MYY: Field[Vertex, dtype],
    vol: Field[Vertex, dtype],
    sign: SparseField[V2E, dtype],
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            zavg = 0.5 * sum(pp[v] for v in e2v[e])
            zavgS_MXX = S_MXX * zavg
            zavgS_MYY = S_MYY * zavg
        with location(Vertex) as v:
            pnabla_MXX = sum(zavgS_MXX[e] * sign[v, e] for e in v2e[v])
            pnabla_MYY = sum(zavgS_MYY[e] * sign[v, e] for e in v2e[v])
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
