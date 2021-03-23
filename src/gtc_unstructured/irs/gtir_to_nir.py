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


import copy
from types import MappingProxyType
from typing import ClassVar, Dict, List, Mapping

import eve
from gtc_unstructured.irs import common, gtir, nir


class GtirToNir(eve.NodeTranslator):
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
        Mapping[gtir.ReduceOperator, common.BinaryOperator]
    ] = MappingProxyType(
        {
            gtir.ReduceOperator.ADD: common.BinaryOperator.ADD,
            gtir.ReduceOperator.MUL: common.BinaryOperator.MUL,
            # TODO
            # gtir.ReduceOperator.MIN: nir.BuiltInLiteral.MIN_VALUE,
            # gtir.ReduceOperator.MAX: nir.BuiltInLiteral.MAX_VALUE,
        }
    )

    def visit_NeighborChain(self, node: gtir.NeighborChain, **kwargs):
        return nir.NeighborChain(elements=node.elements)

    def visit_HorizontalDimension(self, node: gtir.HorizontalDimension, **kwargs):
        return nir.HorizontalDimension(
            primary=node.primary, secondary=self.visit(node.secondary) if node.secondary else None
        )

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
    # TODO discuss if this actually works: can we uniquely identify which ref in the field references which dimension or do we need other techniques (e.g. refer to primary and secondary dimension by name)
    @staticmethod
    def order_location_refs(
        location_refs: List[gtir.LocationRef],
        location_comprehensions: Dict[str, gtir.LocationComprehension],
    ):
        """Compute a dict with primary, secondary and vertical (TODO)."""
        result = {}

        decls = [location_comprehensions[ref.name] for ref in location_refs]

        # If there is a secondary dimension (sparse), then one of the LocationComprehensions references the other.
        for decl in decls:
            if not isinstance(decl.of, gtir.Domain) and decl.of.name in [
                ref.name for ref in location_refs
            ]:
                assert "secondary" not in result
                result["secondary"] = decl.name
            else:
                assert "primary" not in result
                result["primary"] = decl.name

        return result

    def visit_FieldAccess(self, node: gtir.FieldAccess, *, location_comprehensions, **kwargs):
        ordered_location_refs = self.order_location_refs(node.subscript, location_comprehensions)
        primary_chain = location_comprehensions[ordered_location_refs["primary"]].chain
        secondary_chain = (
            location_comprehensions[ordered_location_refs["secondary"]].chain
            if "secondary" in ordered_location_refs
            else None
        )

        return nir.FieldAccess(
            name=node.name,
            location_type=node.location_type,
            primary=primary_chain,
            secondary=secondary_chain,
        )

    def visit_NeighborReduce(self, node: gtir.NeighborReduce, *, last_block, **kwargs):
        loc_comprehension = copy.deepcopy(kwargs["location_comprehensions"])
        assert node.neighbors.name not in loc_comprehension
        loc_comprehension[node.neighbors.name] = node.neighbors
        kwargs["location_comprehensions"] = loc_comprehension

        body_location = node.neighbors.chain.elements[-1]
        reduce_var_name = "local" + str(node.id_)
        last_block.declarations.append(
            nir.LocalVar(
                name=reduce_var_name,
                vtype=common.DataType.FLOAT64,  # TODO
                location_type=node.location_type,
            )
        )
        last_block.statements.append(
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
        body = nir.BlockStmt(
            declarations=[],
            statements=[
                nir.AssignStmt(
                    left=nir.VarAccess(name=reduce_var_name, location_type=body_location),
                    right=nir.BinaryOp(
                        left=nir.VarAccess(name=reduce_var_name, location_type=body_location),
                        op=self.REDUCE_OP_TO_BINOP[node.op],
                        right=self.visit(node.operand, in_neighbor_loop=True, **kwargs),
                        location_type=body_location,
                    ),
                    location_type=body_location,
                )
            ],
            location_type=body_location,
        )
        last_block.statements.append(
            nir.NeighborLoop(
                neighbors=self.visit(node.neighbors.chain),
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

    def visit_AssignStmt(self, node: gtir.AssignStmt, **kwargs):
        return nir.AssignStmt(
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            location_type=node.location_type,
        )

    def visit_HorizontalLoop(self, node: gtir.HorizontalLoop, **kwargs):
        block = nir.BlockStmt(declarations=[], statements=[], location_type=node.stmt.location_type)
        stmt = self.visit(
            node.stmt, last_block=block, location_comprehensions={node.location.name: node.location}
        )
        block.statements.append(stmt)
        return nir.HorizontalLoop(
            stmt=block,
            location_type=node.location.chain.elements[0],
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        return nir.VerticalLoop(
            horizontal_loops=[self.visit(h) for h in node.horizontal_loops],
            loop_order=node.loop_order,
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return nir.Stencil(
            vertical_loops=[self.visit(loop) for loop in node.vertical_loops],
        )
        # TODO

    def visit_Computation(self, node: gtir.Stencil, **kwargs):
        return nir.Computation(
            name=node.name,
            params=[self.visit(p) for p in node.params],
            declarations=[self.visit(decl) for decl in node.declarations]
            if node.declarations
            else [],
            stencils=[self.visit(s) for s in node.stencils],
        )
