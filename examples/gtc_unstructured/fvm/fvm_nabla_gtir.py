# -*- coding: utf-8 -*-
# Eve toolchain

import os

from devtools import debug

import eve  # noqa: F401
from gtc_unstructured.irs.common import BinaryOperator, DataType, LocationType, LoopOrder
from gtc_unstructured.irs.gtir import (
    AssignStmt,
    BinaryOp,
    Computation,
    Dimensions,
    Domain,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    Literal,
    LocationComprehension,
    LocationRef,
    NeighborChain,
    NeighborReduce,
    ReduceOperator,
    Stencil,
    TemporaryField,
    UField,
    VerticalLoop,
)
from gtc_unstructured.irs.gtir_to_nir import GtirToNir
from gtc_unstructured.irs.nir_passes.merge_horizontal_loops import find_and_merge_horizontal_loops
from gtc_unstructured.irs.nir_to_usid import NirToUsid
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator, UsidNaiveCodeGenerator


vertical_loops = []
fields = []

S_MXX = UField(
    name="S_MXX",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
)
fields.append(S_MXX)
S_MYY = UField(
    name="S_MYY",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
)
fields.append(S_MYY)

zavgS_MXX = UField(
    name="zavgS_MXX",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
)
fields.append(zavgS_MXX)
zavgS_MYY = UField(
    name="zavgS_MYY",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
)
fields.append(zavgS_MYY)

pp = UField(
    name="pp",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
)
fields.append(pp)

pnabla_MXX = UField(
    name="pnabla_MXX",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
)
fields.append(pnabla_MXX)
pnabla_MYY = UField(
    name="pnabla_MYY",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
)
fields.append(pnabla_MYY)

vol = UField(
    name="vol",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
)
fields.append(vol)

sign = UField(
    name="sign",
    vtype=DataType.FLOAT64,
    dimensions=Dimensions(
        horizontal=HorizontalDimension(
            primary=LocationType.Vertex,
            secondary=NeighborChain(elements=[LocationType.Vertex, LocationType.Edge]),
        )
    ),
)
fields.append(sign)

# ===========================

zavg_red = NeighborReduce(
    op=ReduceOperator.ADD,
    operand=FieldAccess(
        name="pp", location_type=LocationType.Vertex, subscript=[LocationRef(name="v_of_e")]
    ),
    neighbors=LocationComprehension(
        name="v_of_e",
        chain=NeighborChain(elements=[LocationType.Edge, LocationType.Vertex]),
        of=LocationRef(name="e"),
    ),
    location_type=LocationType.Edge,
)
zavg_mul = BinaryOp(
    left=Literal(value="0.5", vtype=DataType.FLOAT64, location_type=LocationType.Edge),
    op=BinaryOperator.MUL,
    right=zavg_red,
)
zavg_assign = AssignStmt(
    left=FieldAccess(
        name="zavg_tmp", location_type=LocationType.Edge, subscript=[LocationRef(name="e")]
    ),
    right=zavg_mul,
)

assign_zavgS_MXX = AssignStmt(
    left=FieldAccess(
        name="zavgS_MXX", location_type=LocationType.Edge, subscript=[LocationRef(name="e")]
    ),
    right=BinaryOp(
        left=FieldAccess(
            name="zavg_tmp", location_type=LocationType.Edge, subscript=[LocationRef(name="e")]
        ),
        op=BinaryOperator.MUL,
        right=FieldAccess(
            name="S_MXX", location_type=LocationType.Edge, subscript=[LocationRef(name="e")]
        ),
    ),
)

assign_zavgS_MYY = AssignStmt(
    left=FieldAccess(
        name="zavgS_MYY", location_type=LocationType.Edge, subscript=[LocationRef(name="e")]
    ),
    right=BinaryOp(
        left=FieldAccess(
            name="zavg_tmp", location_type=LocationType.Edge, subscript=[LocationRef(name="e")]
        ),
        op=BinaryOperator.MUL,
        right=FieldAccess(
            name="S_MYY", location_type=LocationType.Edge, subscript=[LocationRef(name="e")]
        ),
    ),
)

vertical_loops.append(
    VerticalLoop(
        loop_order=LoopOrder.FORWARD,
        horizontal_loops=[
            HorizontalLoop(
                location=LocationComprehension(
                    name="e", chain=NeighborChain(elements=[LocationType.Edge]), of=Domain()
                ),
                stmt=zavg_assign,
            ),
            HorizontalLoop(
                location=LocationComprehension(
                    name="e", chain=NeighborChain(elements=[LocationType.Edge]), of=Domain()
                ),
                stmt=assign_zavgS_MXX,
            ),
            HorizontalLoop(
                location=LocationComprehension(
                    name="e", chain=NeighborChain(elements=[LocationType.Edge]), of=Domain()
                ),
                stmt=assign_zavgS_MYY,
            ),
        ],
    )
)

# ===========================


assign_pnabla_MXX = AssignStmt(
    left=FieldAccess(
        name="pnabla_MXX", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
    ),
    right=NeighborReduce(
        operand=BinaryOp(
            left=FieldAccess(
                name="zavgS_MXX",
                location_type=LocationType.Edge,
                subscript=[LocationRef(name="e_of_v")],
            ),
            op=BinaryOperator.MUL,
            right=FieldAccess(
                name="sign",
                location_type=LocationType.Edge,
                subscript=[LocationRef(name="v"), LocationRef(name="e_of_v")],
            ),
        ),
        op=ReduceOperator.ADD,
        location_type=LocationType.Vertex,
        neighbors=LocationComprehension(
            name="e_of_v",
            chain=NeighborChain(elements=[LocationType.Vertex, LocationType.Edge]),
            of=LocationRef(name="v"),
        ),
    ),
    location_type=LocationType.Vertex,
)
assign_pnabla_MYY = AssignStmt(
    left=FieldAccess(
        name="pnabla_MYY", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
    ),
    right=NeighborReduce(
        operand=BinaryOp(
            left=FieldAccess(
                name="zavgS_MYY",
                location_type=LocationType.Edge,
                subscript=[LocationRef(name="e_of_v")],
            ),
            op=BinaryOperator.MUL,
            right=FieldAccess(
                name="sign",
                location_type=LocationType.Edge,
                subscript=[LocationRef(name="e_of_v"), LocationRef(name="v")],
            ),
        ),
        op=ReduceOperator.ADD,
        location_type=LocationType.Vertex,
        neighbors=LocationComprehension(
            name="e_of_v",
            chain=NeighborChain(elements=[LocationType.Vertex, LocationType.Edge]),
            of=LocationRef(name="v"),
        ),
    ),
    location_type=LocationType.Vertex,
)


vertical_loops.append(
    VerticalLoop(
        loop_order=LoopOrder.FORWARD,
        horizontal_loops=[
            HorizontalLoop(
                location=LocationComprehension(
                    name="v", chain=NeighborChain(elements=[LocationType.Vertex]), of=Domain()
                ),
                stmt=assign_pnabla_MXX,
            ),
            HorizontalLoop(
                location=LocationComprehension(
                    name="v", chain=NeighborChain(elements=[LocationType.Vertex]), of=Domain()
                ),
                stmt=assign_pnabla_MYY,
            ),
        ],
    )
)

# # ===========================
# # TODO pole correction for pnabla_MYY
# # ===========================

assign_pnabla_MXX_vol = AssignStmt(
    left=FieldAccess(
        name="pnabla_MXX", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
    ),
    right=BinaryOp(
        left=FieldAccess(
            name="pnabla_MXX", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
        ),
        op=BinaryOperator.DIV,
        right=FieldAccess(
            name="vol", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
        ),
    ),
)
assign_pnabla_MYY_vol = AssignStmt(
    left=FieldAccess(
        name="pnabla_MYY", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
    ),
    right=BinaryOp(
        left=FieldAccess(
            name="pnabla_MYY", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
        ),
        op=BinaryOperator.DIV,
        right=FieldAccess(
            name="vol", location_type=LocationType.Vertex, subscript=[LocationRef(name="v")]
        ),
    ),
)

vertical_loops.append(
    VerticalLoop(
        loop_order=LoopOrder.FORWARD,
        horizontal_loops=[
            HorizontalLoop(
                location=LocationComprehension(
                    name="v", chain=NeighborChain(elements=[LocationType.Vertex]), of=Domain()
                ),
                stmt=assign_pnabla_MXX_vol,
            ),
            HorizontalLoop(
                location=LocationComprehension(
                    name="v", chain=NeighborChain(elements=[LocationType.Vertex]), of=Domain()
                ),
                stmt=assign_pnabla_MYY_vol,
            ),
        ],
    )
)

nabla_stencil = Stencil(vertical_loops=vertical_loops)

comp = Computation(
    name="nabla",
    params=fields,
    stencils=[nabla_stencil],
    declarations=[
        TemporaryField(
            name="zavg_tmp",
            vtype=DataType.FLOAT64,
            dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
        )
    ],
)

nir_comp = GtirToNir().visit(comp)
nir_comp = find_and_merge_horizontal_loops(nir_comp)
usid_comp = NirToUsid().visit(nir_comp)
debug(usid_comp)

generated_code = UsidGpuCodeGenerator.apply(usid_comp)
print(generated_code)

output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_fvm_nabla_ugpu.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)

generated_code = UsidNaiveCodeGenerator.apply(usid_comp)
output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_fvm_nabla_unaive.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)
