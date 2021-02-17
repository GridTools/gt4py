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
    Edge,
    Vertex,
    Field,
    SparseField,
    computation,
    Connectivity,
    location,
)
from gtc_unstructured.irs.common import DataType
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator

E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 2, False],))

dtype = DataType.FLOAT64


def sten(e2v: E2V, in_field: SparseField[E2V, dtype], out_field: SparseField[E2V, dtype]):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            # TODO: maybe support slicing for lhs: out_sparse_field[e,:]
            out_field = (in_field[e,v] for v in e2v[e])
            # TODO: Fix silently generates invalid code
            # out_sparse_field = in_sparse_field


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
        os.path.dirname(os.path.realpath(__file__)) + "/generated_sparse_assign_" + mode + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)


if __name__ == "__main__":
    main()
