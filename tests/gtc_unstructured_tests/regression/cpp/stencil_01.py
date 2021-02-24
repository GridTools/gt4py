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


def nh_diff_01(
    e2v: E2V,
    diff_multfac_smag: Field[K, dtype],
    tangent_orientation: Field[Edge, dtype],
    inv_primal_edge_length: Field[Edge, dtype],
    inv_vert_vert_length: Field[Edge, dtype],
    u_vert: Field[[Vertex, K], dtype],
    v_vert: Field[[Vertex, K], dtype],
    primal_normal_x: SparseField[E2V, dtype],
    primal_normal_y: SparseField[E2V, dtype],
    dual_normal_x: SparseField[E2V, dtype],
    dual_normal_y: SparseField[E2V, dtype],
    vn: Field[[Edge, K], dtype],
    kh_smag: Field[[Edge, K], dtype],
    kh_smag_ec: Field[[Edge, K], dtype],
    nabla2: Field[[Edge, K], dtype],
    smag_limit: Field[K, dtype],
    smag_offset: Field[K, dtype],
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            weights_tang = LocalField[E2V, dtype]([-1.0, 1.0, 0.0, 0.0])
            weights_norm = LocalField[E2V, dtype]([0.0, 0.0, -1.0, 1.0])

            weights_close = LocalField[E2V, dtype]([1.0, 1.0, 0.0, 0.0])
            weights_far = LocalField[E2V, dtype]([0.0, 0.0, 1.0, 1.0])

            # fill sparse dimension vn vert using the loop concept
            vn_vert = (
                u_vert[v] * primal_normal_x[e, v] + v_vert[v] * primal_normal_y[e, v]
                for v in e2v[e]
            )
            # dvt_tang for smagorinsky
            dvt_tang = sum(
                weights_tang[e, v]
                * ((u_vert[v] * dual_normal_x[e, v]) + (v_vert[v] * dual_normal_y[e, v]))
                for v in e2v[e]
            )
            dvt_tang = dvt_tang * tangent_orientation
            # dvt_norm for smagorinsky
            dvt_norm = sum(
                weights_norm[e, v]
                * (u_vert[v] * dual_normal_x[e, v] + v_vert[v] * dual_normal_y[e, v])
                for v in e2v[e]
            )
            # compute smagorinsky
            kh_smag_1 = sum(weights_tang[e, v] * vn_vert[e, v] for v in e2v[e])
            kh_smag_1 = (kh_smag_1 * tangent_orientation * inv_primal_edge_length) + (
                dvt_norm * inv_vert_vert_length
            )
            kh_smag_1 = kh_smag_1 * kh_smag_1
            kh_smag_2 = sum(weights_norm[e, v] * vn_vert[e, v] for v in e2v[e])
            kh_smag_2 = (kh_smag_2 * inv_vert_vert_length) - (dvt_tang * inv_primal_edge_length)
            kh_smag_2 = kh_smag_2 * kh_smag_2
            kh_smag = diff_multfac_smag * sqrt(kh_smag_2 + kh_smag_1)
            # compute nabla2 using the diamond reduction
            nabla2 = (sum(weights_close[e, v] * vn_vert[e, v] for v in e2v[e]) - 2.0 * vn) * (
                inv_primal_edge_length * inv_primal_edge_length
            )
            nabla2 = nabla2 + (
                sum(weights_far[e, v] * vn_vert[e, v] for v in e2v[e]) - 2.0 * vn
            ) * (inv_vert_vert_length * inv_vert_vert_length)
            nabla2 = 4.0 * nabla2
            kh_smag_ec = kh_smag
            kh_smag = max(0.0, kh_smag - smag_offset)
            kh_smag = min(kh_smag, smag_limit)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
        extension = ".cc"
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator
        extension = ".cu"

    compilation_task = GTScriptCompilationTask(nh_diff_01)
    generated_code = compilation_task.generate(debug=True, code_generator=code_generator)

    print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_stencil_01_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)

    icon_bindings = IconBindingsCodegen().apply(compilation_task.gtir, stencil_code=generated_code)
    print(icon_bindings)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_icon_stencil_01" + extension
    )
    with open(output_file, "w+") as output:
        output.write(icon_bindings)


if __name__ == "__main__":
    main()
