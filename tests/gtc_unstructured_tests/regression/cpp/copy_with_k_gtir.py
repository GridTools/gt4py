# -*- coding: utf-8 -*-
#
# Simple vertex to edge reduction.
#
# ```python
# for e in edges(mesh):
#     out = sum(in[v] for v in vertices(e))
# ```

import os
import pathlib
import sys

from devtools import debug

from gtc_unstructured.irs.common import DataType, LocationType, LoopOrder
from gtc_unstructured.irs.gtir import (
    AssignStmt,
    Computation,
    Connectivity,
    ConnectivityRef,
    Dimensions,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    LocationComprehension,
    LocationRef,
    NeighborReduce,
    PrimaryLocation,
    ReduceOperator,
    Stencil,
    UField,
    VerticalDimension,
    VerticalLoop,
)
from gtc_unstructured.irs.gtir_to_nir import GtirToNir
from gtc_unstructured.irs.nir_passes.merge_horizontal_loops import find_and_merge_horizontal_loops
from gtc_unstructured.irs.nir_to_usid import NirToUsid
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


dtype = DataType.FLOAT64


def main():
    params = [
        UField(
            name="in_field",
            vtype=dtype,
            dimensions=Dimensions(
                horizontal=HorizontalDimension(primary=LocationType.Vertex),
                vertical=VerticalDimension(),
            ),
        ),
        UField(
            name="out_field",
            vtype=dtype,
            dimensions=Dimensions(
                horizontal=HorizontalDimension(primary=LocationType.Edge),
                vertical=VerticalDimension(),
            ),
        ),
    ]

    hloop = HorizontalLoop(
        stmt=AssignStmt(
            left=FieldAccess(
                name="out_field",
                subscript=[LocationRef(name="edge")],
                location_type=LocationType.Edge,
            ),
            right=FieldAccess(
                name="in_field",
                subscript=[LocationRef(name="edge")],
                location_type=LocationType.Edge,
            ),
        ),
        location=PrimaryLocation(name="edge", location_type=LocationType.Edge),
    )

    stencils = [
        Stencil(
            vertical_loops=[VerticalLoop(horizontal_loops=[hloop], loop_order=LoopOrder.FORWARD)]
        )
    ]

    comp = Computation(name="sten", connectivities=[], params=params, stencils=stencils)

    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    nir_comp = GtirToNir().visit(comp)
    nir_comp = find_and_merge_horizontal_loops(nir_comp)
    usid_comp = NirToUsid().visit(nir_comp)

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator

    generated_code = code_generator.apply(usid_comp)
    print(generated_code)
    output_file = (
        os.path.dirname(os.path.realpath(__file__))
        + "/generated_"
        + pathlib.Path(__file__).stem
        + "_"
        + mode
        + ".hpp"
    )
    with open(output_file, "w+") as output:
        output.write(generated_code)


if __name__ == "__main__":
    main()
