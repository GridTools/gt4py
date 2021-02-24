# -*- coding: utf-8 -*-
#
# Simple vertex to edge reduction.
#
# ```python
# for e in edges(mesh):
#     out = sum(in[v] for v in vertices(e))
# ```

import os
import sys
import types

from gtc_unstructured.frontend.frontend import GTScriptCompilationTask
from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Connectivity,
    Edge,
    Field,
    Vertex,
    computation,
    location,
)
from gtc_unstructured.irs.common import DataType
from gtc_unstructured.irs.icon_bindings_codegen import IconBindingsCodegen
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 2, False],))
dtype = DataType.FLOAT64


def sten(e2v: E2V, field_in: Field[Vertex, dtype], field_out: Field[Edge, dtype]):
    with computation(FORWARD), location(Edge) as e:
        field_out[e] = sum(field_in[v] for v in e2v[e])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
        extension = ".cc"
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator
        extension = ".cu"

    compilation_task = GTScriptCompilationTask(sten)
    generated_code = compilation_task.generate(debug=True, code_generator=code_generator)

    print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_vertex2edge_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)

    icon_bindings = IconBindingsCodegen().apply(compilation_task.gtir, generated_code)
    print(icon_bindings)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_icon_vertex2edge" + extension
    )
    with open(output_file, "w+") as output:
        output.write(icon_bindings)


if __name__ == "__main__":
    main()
