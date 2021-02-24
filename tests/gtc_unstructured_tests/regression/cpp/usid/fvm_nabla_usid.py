# -*- coding: utf-8 -*-
# Eve toolchain

import os
import sys

from devtools import debug

import eve  # noqa: F401
from gtc_unstructured.irs import common
from gtc_unstructured.irs.common import LocationType
from gtc_unstructured.irs.gtir_to_nir import GtirToNir
from gtc_unstructured.irs.nir_passes.merge_horizontal_loops import find_and_merge_horizontal_loops
from gtc_unstructured.irs.nir_to_usid import NirToUsid
from gtc_unstructured.irs.usid import (
    AssignStmt,
    BinaryOp,
    Computation,
    Connectivity,
    FieldAccess,
    Kernel,
    KernelCall,
    Literal,
    NeighborLoop,
    PtrRef,
    SidComposite,
    SidCompositeEntry,
    SidCompositeSparseEntry,
    SparseField,
    Temporary,
    UField,
    VerticalDimension,
)
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


connectivities = [
    Connectivity(name="v2e", max_neighbors=7, has_skip_values=True),
    Connectivity(name="e2v", max_neighbors=2, has_skip_values=False),
]

pp_acc = FieldAccess(name="pp_tag", sid="sec", location_type=LocationType.Vertex)
zavg_tmp_acc = FieldAccess(name="zavg_tmp_tag", sid="primary_edge", location_type=LocationType.Edge)
zavg_tmp_acc_on_v = FieldAccess(name="zavg_tmp_tag", sid="prim", location_type=LocationType.Vertex)
zavgS_MXX_acc = FieldAccess(
    name="zavgS_MXX_tag", sid="primary_edge", location_type=LocationType.Edge
)
zavgS_MYY_acc = FieldAccess(
    name="zavgS_MYY_tag", sid="primary_edge", location_type=LocationType.Edge
)
S_MXX_acc = FieldAccess(name="S_MXX_tag", sid="primary_edge", location_type=LocationType.Edge)
S_MYY_acc = FieldAccess(name="S_MYY_tag", sid="primary_edge", location_type=LocationType.Edge)

edge1_assign0 = AssignStmt(
    left=zavg_tmp_acc,
    right=Literal(value="0.0", location_type=LocationType.Edge, vtype=common.DataType.FLOAT64),
)

edge1_assign1 = AssignStmt(
    left=zavg_tmp_acc,
    right=BinaryOp(
        left=Literal(value="0.5", location_type=LocationType.Edge, vtype=common.DataType.FLOAT64),
        right=zavg_tmp_acc,
        op=common.BinaryOperator.MUL,
    ),
)
edge1_assign2 = AssignStmt(
    left=zavgS_MXX_acc,
    right=BinaryOp(left=S_MXX_acc, right=zavg_tmp_acc, op=common.BinaryOperator.MUL),
)
edge1_assign3 = AssignStmt(
    left=zavgS_MYY_acc,
    right=BinaryOp(left=S_MYY_acc, right=zavg_tmp_acc, op=common.BinaryOperator.MUL),
)

vertex_on_edge_loop = NeighborLoop(
    primary_sid="primary_edge",
    secondary_sid="vertices",
    connectivity="e2v",
    primary=PtrRef(name="prim"),
    secondary=PtrRef(name="sec"),
    location_type=LocationType.Edge,
    body=[
        AssignStmt(
            location_type=LocationType.Vertex,
            left=zavg_tmp_acc_on_v,
            right=BinaryOp(
                left=zavg_tmp_acc_on_v,
                right=pp_acc,
                op=common.BinaryOperator.ADD,
                location_type=LocationType.Vertex,
            ),
        )
    ],
)

nabla_edge_1 = Kernel(
    name="nabla_edge_1",
    primary_location=common.LocationType.Edge,
    primary_composite=SidComposite(
        name="primary_edge",
        entries=[
            SidCompositeEntry(ref="e2v"),
            SidCompositeEntry(ref="zavg_tmp"),
            SidCompositeEntry(ref="S_MXX"),
            SidCompositeEntry(ref="S_MYY"),
            SidCompositeEntry(ref="zavgS_MXX"),
            SidCompositeEntry(ref="zavgS_MYY"),
        ],
    ),
    secondary_composites=[SidComposite(name="vertices", entries=[SidCompositeEntry(ref="pp")])],
    body=[edge1_assign0, vertex_on_edge_loop, edge1_assign1, edge1_assign2, edge1_assign3],
)


zavgS_MXX_acc = FieldAccess(name="zavgS_MXX_tag", sid="n", location_type=LocationType.Edge)
zavgS_MYY_acc = FieldAccess(name="zavgS_MYY_tag", sid="n", location_type=LocationType.Edge)

pnabla_MXX_acc = FieldAccess(name="pnabla_MXX_tag", location_type=LocationType.Edge, sid="p")
pnabla_MXX_acc_vertex = FieldAccess(
    name="pnabla_MXX_tag", location_type=LocationType.Vertex, sid="vertex_prim"
)
pnabla_MYY_acc = FieldAccess(name="pnabla_MYY_tag", location_type=LocationType.Edge, sid="p")
pnabla_MYY_acc_vertex = FieldAccess(
    name="pnabla_MYY_tag", location_type=LocationType.Vertex, sid="vertex_prim"
)
sign_acc = FieldAccess(name="sign_tag", location_type=LocationType.Edge, sid="p")


edge_on_vertex_loop_x = NeighborLoop(
    primary_sid="vertex_prim",
    secondary_sid="edge_neighbors",
    primary=PtrRef(name="p"),
    secondary=PtrRef(name="n"),
    # body_location_type=LocationType.Edge,
    location_type=LocationType.Vertex,
    connectivity="v2e",
    body=[
        AssignStmt(
            location_type=LocationType.Edge,
            left=pnabla_MXX_acc,
            right=BinaryOp(
                left=pnabla_MXX_acc,
                right=BinaryOp(
                    left=zavgS_MXX_acc,
                    right=sign_acc,
                    op=common.BinaryOperator.MUL,
                    location_type=LocationType.Edge,
                ),
                op=common.BinaryOperator.ADD,
                location_type=LocationType.Edge,
            ),
        )
    ],
)

edge_on_vertex_loop_y = NeighborLoop(
    primary_sid="vertex_prim",
    secondary_sid="edge_neighbors",
    primary=PtrRef(name="p"),
    secondary=PtrRef(name="n"),
    # body_location_type=LocationType.Edge,
    location_type=LocationType.Vertex,
    connectivity="v2e",
    body=[
        AssignStmt(
            location_type=LocationType.Edge,
            left=pnabla_MYY_acc,
            right=BinaryOp(
                left=pnabla_MYY_acc,
                right=BinaryOp(
                    left=zavgS_MYY_acc,
                    right=sign_acc,
                    op=common.BinaryOperator.MUL,
                    location_type=LocationType.Edge,
                ),
                op=common.BinaryOperator.ADD,
                location_type=LocationType.Edge,
            ),
        )
    ],
)

vertex_2_init_to_zero_x = AssignStmt(
    left=pnabla_MXX_acc_vertex,
    right=Literal(value="0.0", location_type=LocationType.Vertex, vtype=common.DataType.FLOAT64),
    location_type=LocationType.Vertex,
)

vertex_2_init_to_zero_y = AssignStmt(
    left=pnabla_MYY_acc_vertex,
    right=Literal(value="0.0", location_type=LocationType.Vertex, vtype=common.DataType.FLOAT64),
    location_type=LocationType.Vertex,
)


nabla_vertex_2 = Kernel(
    name="nabla_vertex_2",
    primary_location=common.LocationType.Vertex,
    primary_composite=SidComposite(
        name="vertex_prim",
        entries=[
            SidCompositeEntry(ref="v2e"),  # TODO?
            # SidCompositeNeighborTableEntry(connectivity="v2e"),
            SidCompositeEntry(ref="pnabla_MXX"),
            SidCompositeEntry(ref="pnabla_MYY"),
            SidCompositeSparseEntry(ref="sign", connectivity="v2e"),
        ],
    ),
    secondary_composites=[
        SidComposite(
            name="edge_neighbors",
            entries=[
                SidCompositeEntry(ref="zavgS_MXX"),
                SidCompositeEntry(ref="zavgS_MYY"),
            ],
        )
    ],
    body=[
        vertex_2_init_to_zero_x,
        edge_on_vertex_loop_x,
        vertex_2_init_to_zero_y,
        edge_on_vertex_loop_y,
    ],
)

pnabla_MXX_acc = FieldAccess(
    name="pnabla_MXX_tag", sid="primary", location_type=LocationType.Vertex
)
pnabla_MYY_acc = FieldAccess(
    name="pnabla_MYY_tag", sid="primary", location_type=LocationType.Vertex
)
vol_acc = FieldAccess(name="vol_tag", sid="primary", location_type=LocationType.Vertex)
div = BinaryOp(left=pnabla_MXX_acc, right=vol_acc, op=common.BinaryOperator.DIV)
div2 = BinaryOp(left=pnabla_MYY_acc, right=vol_acc, op=common.BinaryOperator.DIV)
assign = AssignStmt(left=pnabla_MXX_acc, right=div)
assign2 = AssignStmt(left=pnabla_MYY_acc, right=div2)

nabla_vertex_4 = Kernel(
    name="nabla_vertex_4",
    primary_location=common.LocationType.Vertex,
    primary_composite=SidComposite(
        name="primary",
        entries=[
            SidCompositeEntry(ref="pnabla_MXX"),
            SidCompositeEntry(ref="pnabla_MYY"),
            SidCompositeEntry(ref="vol"),
        ],
    ),
    secondary_composites=[],
    body=[assign, assign2],
)

S_MXX = UField(
    name="S_MXX", dimensions=[LocationType.Edge, VerticalDimension()], vtype=common.DataType.FLOAT64
)
S_MYY = UField(
    name="S_MYY", dimensions=[LocationType.Edge, VerticalDimension()], vtype=common.DataType.FLOAT64
)
pp = UField(
    name="pp", dimensions=[LocationType.Vertex, VerticalDimension()], vtype=common.DataType.FLOAT64
)
pnabla_MXX = UField(
    name="pnabla_MXX",
    dimensions=[LocationType.Vertex, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)
pnabla_MYY = UField(
    name="pnabla_MYY",
    dimensions=[LocationType.Vertex, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)
vol = UField(
    name="vol", dimensions=[LocationType.Vertex, VerticalDimension()], vtype=common.DataType.FLOAT64
)
sign = SparseField(
    name="sign",
    dimensions=[
        LocationType.Vertex,
        VerticalDimension(),
    ],
    connectivity="v2e",
    vtype=common.DataType.FLOAT64,
)


zavgS_MXX = Temporary(
    name="zavgS_MXX",
    dimensions=[LocationType.Edge, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)
zavgS_MYY = Temporary(
    name="zavgS_MYY",
    dimensions=[LocationType.Edge, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)

zavg_tmp = Temporary(
    name="zavg_tmp",
    dimensions=[LocationType.Edge, VerticalDimension()],
    vtype=common.DataType.FLOAT64,
)

comp = Computation(
    name="nabla",
    connectivities=connectivities,
    parameters=[S_MXX, S_MYY, pp, pnabla_MXX, pnabla_MYY, vol, sign],
    temporaries=[zavgS_MXX, zavgS_MYY, zavg_tmp],
    kernels=[nabla_edge_1, nabla_vertex_2, nabla_vertex_4],
    ctrlflow_ast=[
        KernelCall(name="nabla_edge_1"),
        KernelCall(name="nabla_vertex_2"),
        KernelCall(name="nabla_vertex_4"),
    ],
)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unaive"

    if mode == "unaive":
        code_generator = UsidNaiveCodeGenerator
    else:  # 'ugpu':
        code_generator = UsidGpuCodeGenerator

    generated_code = code_generator.apply(comp)

    output_file = os.path.dirname(os.path.realpath(__file__)) + "/nabla_" + mode + ".hpp"
    with open(output_file, "w+") as output:
        output.write(generated_code)


if __name__ == "__main__":
    main()
