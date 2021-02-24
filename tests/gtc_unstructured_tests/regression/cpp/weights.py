# -*- coding: utf-8 -*-
#
# Weighted reduction.

import os
import sys
import types

from gtc_unstructured.frontend.frontend import GTScriptCompilationTask
from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Connectivity,
    Edge,
    Field,
    LocalField,
    Vertex,
    computation,
    interval,
    location,
)
from gtc_unstructured.irs.common import DataType
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 2, False],))

dtype = DataType.FLOAT64


def sten(e2v: E2V, in_field: Field[Vertex, dtype], out_field: Field[Edge, dtype]):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            # TODO: ints don't work right now
            weights = LocalField[E2V, dtype]([-1.0, 1.0])
            out_field = sum(in_field[v] * weights[e, v] for v in e2v[e])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator

    generated_code = GTScriptCompilationTask(sten).generate(
        debug=True, code_generator=code_generator
    )

    print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_weights_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)


if __name__ == "__main__":
    main()
