# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, List, Mapping, Union

from devtools import debug

import eve
from gtc_unstructured.irs import common, gtir, nir


class GtirToNir(eve.NodeTranslator):
    @dataclass
    class HorizontalLoopContext:
        declarations: List[nir.LocalVar] = field(default_factory=list)
        statements: List[nir.Stmt] = field(default_factory=list)

        def add_declaration(self, decl: nir.LocalVar) -> "GtirToNir.HorizontalLoopContext":
            self.declarations.append(decl)
            return self

        def add_statement(self, stmt: nir.Stmt) -> "GtirToNir.HorizontalLoopContext":
            self.statements.append(stmt)
            return self

    REDUCE_OP_INIT_VAL: ClassVar[
        Mapping[gtir.ReduceOperator, common.BuiltInLiteral]
    ] = MappingProxyType(
        {
            gtir.ReduceOperator.ADD: common.BuiltInLiteral.ZERO,
            gtir.ReduceOperator.MUL: common.BuiltInLiteral.ONE,
            gtir.ReduceOperator.MIN: common.BuiltInLiteral.MIN_VALUE,
            gtir.ReduceOperator.MAX: common.BuiltInLiteral.MAX_VALUE,
        }
    )

    REDUCE_OP_TO_BINOP: ClassVar[
        Mapping[gtir.ReduceOperator, Union[common.BinaryOperator, common.BuiltInLiteral]]
    ] = MappingProxyType(
        {
            gtir.ReduceOperator.ADD: common.BinaryOperator.ADD,
            gtir.ReduceOperator.MUL: common.BinaryOperator.MUL,
            gtir.ReduceOperator.MIN: common.BuiltInLiteral.MIN_VALUE,
            gtir.ReduceOperator.MAX: common.BuiltInLiteral.MAX_VALUE,
        }
    )

    def visit_HorizontalDimension(self, node: gtir.HorizontalDimension, **kwargs):
        return nir.HorizontalDimension(primary=node.primary)

    def visit_VerticalDimension(self, node: gtir.VerticalDimension, **kwargs):
        return nir.VerticalDimension()

    def visit_Dimensions(self, node: gtir.Dimensions, **kwargs):
        return nir.Dimensions(
            horizontal=self.visit(node.horizontal) if node.horizontal else None,
            vertical=self.visit(node.vertical) if node.vertical else None,
        )

    def visit_UField(self, node: gtir.UField, **kwargs):
        return nir.UField(name=node.name, vtype=node.vtype, dimensions=self.visit(node.dimensions))

    def visit_TemporaryField(self, node: gtir.UField, **kwargs):
        return nir.TemporaryField(
            name=node.name, vtype=node.vtype, dimensions=self.visit(node.dimensions)
        )

    # TODO test
    # TODO super ugly
    @staticmethod
    def order_location_refs(location_refs: List[gtir.LocationRef], symtable):
        # TODO deal with k
        primary = None
        secondary = None

        decls = [symtable[ref.name] for ref in location_refs]
        debug(decls)

        for decl in decls:
            if not primary:
                primary = decl.name
            elif isinstance(decl, gtir.PrimaryLocation):
                secondary = primary
                primary = decl.name

        assert primary
        return primary, secondary

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs):
        primary, secondary = self.order_location_refs(node.subscript, kwargs["symtable"])
        return nir.FieldAccess(
            name=node.name,
            location_type=node.location_type,
            primary=primary,
            secondary=secondary,
        )

    def visit_NeighborVectorAccess(
        self,
        node: gtir.NeighborVectorAccess,
        *,
        symtable,
        loop_var,
        hloop_ctx: "HorizontalLoopContext",
        **kwargs,
    ):
        location_ref_deref = symtable[node.location_ref.name]
        name = node.id_
        hloop_ctx.add_declaration(
            nir.LocalFieldVar(
                name=name,
                init=self.visit(node.exprs),
                connectivity=location_ref_deref.of.name,
                location_type=node.location_type,
            )
        )
        return nir.FieldAccess(name=name, primary=loop_var, location_type=node.location_type)

    def visit_NeighborAssignStmt(
        self,
        node: gtir.NeighborAssignStmt,
        *,
        symtable,
        hloop_ctx: "HorizontalLoopContext",
        **kwargs,
    ):
        symtable = {**symtable, **node.symtable_}
        name = node.neighbors.name
        hloop_ctx.add_statement(
            nir.NeighborLoop(
                name=name,
                connectivity=node.neighbors.of.name,
                body=nir.BlockStmt(
                    declarations=[],
                    statements=[
                        nir.AssignStmt(
                            left=nir.FieldAccess(
                                name=node.left.name,
                                primary=node.left.subscript[0].name,
                                secondary=node.neighbors.name,
                                location_type=node.location_type,
                            ),
                            right=self.visit(
                                node.right,
                                loop_var=name,
                                symtable=symtable,
                                hloop_ctx=hloop_ctx,
                                **kwargs,
                            ),
                            location_type=node.location_type,
                        )
                    ],
                    location_type=node.location_type,
                ),
                location_type=node.location_type,
            )
        )

    def visit_NeighborReduce(
        self, node: gtir.NeighborReduce, *, hloop_ctx: "HorizontalLoopContext", **kwargs
    ):
        connectivity_deref: gtir.Connectivity = kwargs["symtable"][node.neighbors.of.name]

        reduce_var_name = "local" + str(node.id_)
        hloop_ctx.add_declaration(
            nir.LocalVar(
                name=reduce_var_name,
                vtype=common.DataType.FLOAT64,  # TODO
                location_type=node.location_type,
            )
        )
        hloop_ctx.add_statement(
            nir.AssignStmt(
                left=nir.VarAccess(name=reduce_var_name, location_type=node.location_type),
                right=nir.Literal(
                    value=self.REDUCE_OP_INIT_VAL[node.op],
                    location_type=node.location_type,
                    vtype=common.DataType.FLOAT64,  # TODO
                ),
                location_type=node.location_type,
            ),
        )
        body_location = connectivity_deref.secondary
        op = self.REDUCE_OP_TO_BINOP[node.op]
        if op == common.BuiltInLiteral.MIN_VALUE or op == common.BuiltInLiteral.MAX_VALUE:
            right = nir.NativeFuncCall(
                func=nir.NativeFunction.MAX
                if op == common.BuiltInLiteral.MAX_VALUE
                else nir.NativeFunction.MIN,
                args=[
                    nir.VarAccess(name=reduce_var_name, location_type=body_location),
                    self.visit(node.operand, in_neighbor_loop=True, **kwargs),
                ],
                location_type=body_location,
            )
        else:
            right = nir.BinaryOp(
                left=nir.VarAccess(name=reduce_var_name, location_type=body_location),
                op=op,
                right=self.visit(node.operand, in_neighbor_loop=True, **kwargs),
                location_type=body_location,
            )
        body = nir.BlockStmt(
            declarations=[],
            statements=[
                nir.AssignStmt(
                    left=nir.VarAccess(name=reduce_var_name, location_type=body_location),
                    right=right,
                    location_type=body_location,
                )
            ],
            location_type=body_location,
        )
        hloop_ctx.add_statement(
            nir.NeighborLoop(
                name=node.neighbors.name,
                connectivity=connectivity_deref.name,
                body=body,
                location_type=node.location_type,
            )
        )
        return nir.VarAccess(name=reduce_var_name, location_type=node.location_type)  # TODO

    def visit_Literal(self, node: gtir.Literal, **kwargs):
        return nir.Literal(value=node.value, vtype=node.vtype, location_type=node.location_type)

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        return nir.BinaryOp(
            left=self.visit(node.left, **kwargs),
            op=node.op,
            right=self.visit(node.right, **kwargs),
            location_type=node.location_type,
        )

    def visit_AssignStmt(
        self, node: gtir.AssignStmt, *, hloop_ctx: "HorizontalLoopContext", **kwargs
    ):
        hloop_ctx.add_statement(
            nir.AssignStmt(
                left=self.visit(node.left, **kwargs),
                right=self.visit(node.right, hloop_ctx=hloop_ctx, **kwargs),
                location_type=node.location_type,
            )
        )

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs):
        debug(node)
        nir_node = nir.NativeFuncCall(
            func=nir.NativeFunction(node.func.value),
            args=self.visit(node.args, **kwargs),
            location_type=node.location_type,
        )
        debug(nir_node)
        return nir_node

    def visit_HorizontalLoop(self, node: gtir.HorizontalLoop, **kwargs):
        hloop_ctx = self.HorizontalLoopContext()
        self.visit(
            node.stmt,
            hloop_ctx=hloop_ctx,
            location_comprehensions={node.location.name: node.location},
            **kwargs,
        )
        return nir.HorizontalLoop(
            stmt=nir.BlockStmt(
                declarations=hloop_ctx.declarations,
                statements=hloop_ctx.statements,
                location_type=node.location.location_type,
            ),
            iteration_space=nir.IterationSpace(
                name=node.location.name, location_type=node.location.location_type
            ),
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        return nir.VerticalLoop(
            horizontal_loops=self.visit(node.horizontal_loops, **kwargs),
            loop_order=node.loop_order,
        )

    def visit_Connectivity(self, node: gtir.Connectivity, **kwargs):
        return nir.Connectivity(
            name=node.name,
            primary=node.primary,
            secondary=node.secondary,
            max_neighbors=node.max_neighbors,
            has_skip_values=node.has_skip_values,
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return nir.Stencil(
            vertical_loops=[self.visit(loop, **kwargs) for loop in node.vertical_loops],
        )
        # TODO

    def visit_Computation(self, node: gtir.Computation, **kwargs):
        return nir.Computation(
            name=node.name,
            connectivities=self.visit(node.connectivities),
            params=self.visit(node.params),
            declarations=self.visit(node.declarations) if node.declarations else [],
            stencils=self.visit(node.stencils, symtable=node.symtable_),
        )
