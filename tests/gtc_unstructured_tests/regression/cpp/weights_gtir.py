# -*- coding: utf-8 -*-
#
# Simple vertex to edge reduction.
#
# ```python
# for e in edges(mesh):
#     weights: SparseField[E2V, double] = LocalField[E2V]([-1,+1])
#     out = sum(in[v]*weights[v] for v in vertices(e))
# ```

import os
import pathlib
import sys

from devtools import debug

from gtc_unstructured.irs.common import BinaryOperator, DataType, LocationType, LoopOrder
from gtc_unstructured.irs.gtir import (
    AssignStmt,
    BinaryOp,
    Computation,
    Connectivity,
    ConnectivityRef,
    Dimensions,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    Literal,
    LocationComprehension,
    LocationRef,
    NeighborAssignStmt,
    NeighborReduce,
    NeighborVectorAccess,
    PrimaryLocation,
    ReduceOperator,
    Stencil,
    TemporarySparseField,
    UField,
    VerticalLoop,
)
from gtc_unstructured.irs.gtir_to_nir import GtirToNir
from gtc_unstructured.irs.nir_passes.merge_horizontal_loops import find_and_merge_horizontal_loops
from gtc_unstructured.irs.nir_to_usid import NirToUsid
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


dtype = DataType.FLOAT64


def main():
    conn = Connectivity(
        name="e2v",
        primary=LocationType.Edge,
        secondary=LocationType.Vertex,
        max_neighbors=2,
        has_skip_values=False,
    )

    params = [
        UField(
            name="in_field",
            vtype=dtype,
            dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
        ),
        UField(
            name="out_field",
            vtype=dtype,
            dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
        ),
    ]

    declarations = [
        TemporarySparseField(
            name="weights",
            connectivity="e2v",
            vtype=dtype,
            dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
        )
    ]

    hloop0 = HorizontalLoop(
        stmt=NeighborAssignStmt(
            left=FieldAccess(
                name="weights",
                subscript=[LocationRef(name="edge")],
                location_type=LocationType.Edge,  # wrong
            ),
            right=NeighborVectorAccess(
                exprs=[
                    Literal(value="-1", vtype=dtype, location_type=LocationType.Vertex),
                    Literal(value="1", vtype=dtype, location_type=LocationType.Vertex),
                ],
                location_ref=LocationRef(name="v_neighs_of_e"),
                location_type=LocationType.Edge,  # wrong
            ),
            neighbors=LocationComprehension(name="other_v_of_e", of=ConnectivityRef(name="e2v")),
            location_type=LocationType.Edge,
        ),
        location=PrimaryLocation(name="edge", location_type=LocationType.Edge),
    )

    hloop1 = HorizontalLoop(
        stmt=AssignStmt(
            left=FieldAccess(
                name="out_field",
                subscript=[LocationRef(name="edge")],
                location_type=LocationType.Edge,
            ),
            right=NeighborReduce(
                op=ReduceOperator.ADD,
                operand=BinaryOp(
                    op=BinaryOperator.MUL,
                    left=FieldAccess(
                        name="in_field",
                        subscript=[LocationRef(name="v_neighs_of_e")],
                        location_type=LocationType.Vertex,
                    ),
                    right=FieldAccess(
                        name="weights",
                        subscript=[LocationRef(name="edge"), LocationRef(name="v_neighs_of_e")],
                        location_type=LocationType.Vertex,
                    ),
                ),
                neighbors=LocationComprehension(
                    name="v_neighs_of_e", of=ConnectivityRef(name="e2v")
                ),
                location_type=LocationType.Edge,
            ),
        ),
        location=PrimaryLocation(name="edge", location_type=LocationType.Edge),
    )

    stencils = [
        Stencil(
            vertical_loops=[
                VerticalLoop(horizontal_loops=[hloop0, hloop1], loop_order=LoopOrder.FORWARD)
            ]
        )
    ]

    comp = Computation(
        name="sten",
        connectivities=[conn],
        params=params,
        declarations=declarations,
        stencils=stencils,
    )

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
