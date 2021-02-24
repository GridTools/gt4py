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

from gtc_unstructured.frontend.frontend import GTScriptCompilationTask
from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Cell,
    Connectivity,
    Field,
    K,
    SparseField,
    computation,
    location,
)
from gtc_unstructured.irs.common import DataType
from gtc_unstructured.irs.icon_bindings_codegen import IconBindingsCodegen
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


C2C = types.new_class("C2C", (Connectivity[Cell, Cell, 4, False],))

dtype = DataType.FLOAT64


def sten(
    c2c: C2C,
    field_in: Field[[Cell, K], dtype],
    field_out: Field[[Cell, K], dtype],
    field_sparse: SparseField[[C2C, K], dtype],
    field_k: Field[K, dtype],
):
    with computation(FORWARD), location(Cell) as c1:
        field_out[c1] = 2.0 * field_k + sum(
            field_in[c1] + field_in[c2] + field_sparse[c1, c2] for c2 in c2c[c1]
        )


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
        os.path.dirname(os.path.realpath(__file__)) + "/generated_cell2cell_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)

    icon_bindings = IconBindingsCodegen().apply(compilation_task.gtir, generated_code)
    print(icon_bindings)
    output_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/generated_icon_sparse_diamond" + extension
    )
    with open(output_file, "w+") as output:
        output.write(icon_bindings)


if __name__ == "__main__":
    main()
