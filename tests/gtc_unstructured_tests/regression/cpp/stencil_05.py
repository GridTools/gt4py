# -*- coding: utf-8 -*-
#
# Cell to cell reduction.
# Note that the reduction refers to a LocationRef from outside!
#
# ```python
# for c1 in cells(mesh):
#     field1 = sum(f[c1] * f[c2] for c2 in cells(c1))
# ```

import os
import sys
import types

from gtc_unstructured.frontend.built_in_types import LocalField
from gtc_unstructured.frontend.frontend import GTScriptCompilationTask
from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Connectivity,
    Edge,
    Field,
    K,
    SparseField,
    Vertex,
    computation,
    location,
)
from gtc_unstructured.irs.common import DataType
from gtc_unstructured.irs.icon_bindings_codegen import IconBindingsCodegen
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 4, True],))
dtype = DataType.FLOAT64


def nh_diff_05(
    e2v: E2V,
    z_nabla4_e2: Field[[Edge, K], dtype],
    u_vert: Field[[Vertex, K], dtype],
    v_vert: Field[[Vertex, K], dtype],
    primal_normal_vert_v1: SparseField[E2V, dtype],
    primal_normal_vert_v2: SparseField[E2V, dtype],
    z_nabla2_e: Field[[Edge, K], dtype],
    inv_vert_vert_length: Field[Edge, dtype],
    inv_primal_edge_length: Field[Edge, dtype],
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            weights_0 = LocalField[E2V, dtype]([1.0, 1.0, 0.0, 0.0])
            weights_1 = LocalField[E2V, dtype]([0.0, 0.0, 1.0, 1.0])
            nabv_tang = sum(
                weights_0[e, v]
                * (
                    u_vert[v] * primal_normal_vert_v1[e, v]
                    + v_vert[v] * primal_normal_vert_v2[e, v]
                )
                for v in e2v[e]
            )
            nabv_norm = sum(
                weights_1[e, v]
                * (
                    u_vert[v] * primal_normal_vert_v1[e, v]
                    + v_vert[v] * primal_normal_vert_v2[e, v]
                )
                for v in e2v[e]
            )
            z_nabla4_e2 = 4.0 * (
                (nabv_norm[e] - 2.0 * z_nabla2_e[e])
                * inv_vert_vert_length[e]
                * inv_vert_vert_length[e]
                + (nabv_tang[e] - 2.0 * z_nabla2_e[e])
                * inv_primal_edge_length[e]
                * inv_primal_edge_length[e]
            )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
        extension = ".cc"
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator
        extension = ".cu"

    compilation_task = GTScriptCompilationTask(nh_diff_05)
    generated_code = compilation_task.generate(debug=True, code_generator=code_generator)

    print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_stencil_05_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)

    icon_bindings = IconBindingsCodegen().apply(compilation_task.gtir, stencil_code=generated_code)
    print(icon_bindings)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_icon_stencil_05" + extension
    )
    with open(output_file, "w+") as output:
        output.write(icon_bindings)


if __name__ == "__main__":
    main()
