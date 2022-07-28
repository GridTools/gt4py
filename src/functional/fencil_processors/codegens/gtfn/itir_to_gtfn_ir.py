# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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


from typing import Any, Iterable, Type

import eve
from eve.concepts import SymbolName
from functional import common
from functional.fencil_processors.codegens.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    CartesianDomain,
    Expr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    Lambda,
    Literal,
    Node,
    OffsetLiteral,
    SidComposite,
    StencilExecution,
    Sym,
    SymRef,
    TaggedValues,
    TernaryExpr,
    UnaryExpr,
    UnstructuredDomain,
)
from functional.iterator import ir as itir


class GTFN_lowering(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    _binary_op_map = {
        "plus": "+",
        "minus": "-",
        "multiplies": "*",
        "divides": "/",
        "eq": "==",
        "less": "<",
        "greater": ">",
        "and_": "&&",
        "or_": "||",
    }
    _unary_op_map = {"not_": "!"}

    def visit_Sym(self, node: itir.Sym, **kwargs: Any) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(self, node: itir.SymRef, **kwargs: Any) -> SymRef:
        return SymRef(id=node.id)

    def visit_Lambda(self, node: itir.Lambda, **kwargs: Any) -> Lambda:
        return Lambda(params=self.visit(node.params), expr=self.visit(node.expr))

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type=node.type)

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs: Any) -> OffsetLiteral:
        if node.value in self.offset_provider:
            if isinstance(
                self.offset_provider[node.value], common.Dimension
            ):  # replace offset tag by dimension tag
                return OffsetLiteral(value=self.offset_provider[node.value].value)
        return OffsetLiteral(value=node.value)

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type="axis_literal")

    @staticmethod
    def _is_sparse_deref_shift(node: itir.FunCall) -> bool:
        return (
            node.fun == itir.SymRef(id="deref")
            and isinstance(node.args[0], itir.FunCall)
            and isinstance(node.args[0].fun, itir.FunCall)
            and node.args[0].fun.fun == itir.SymRef(id="shift")
            and bool(len(node.args[0].fun.args) % 2)
        )

    def _sparse_deref_shift_to_tuple_get(self, node: itir.FunCall) -> Expr:
        # deref(shift(i)(sparse)) -> tuple_get(i, deref(sparse))
        # TODO: remove once ‘real’ sparse field handling is available
        assert isinstance(node.args[0], itir.FunCall)
        assert isinstance(node.args[0].fun, itir.FunCall)
        offsets = node.args[0].fun.args
        deref_arg = node.args[0].args[0]
        if len(offsets) > 1:
            deref_arg = itir.FunCall(
                fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=offsets[:-1]),
                args=[deref_arg],
            )
        derefed = itir.FunCall(fun=itir.SymRef(id="deref"), args=[deref_arg])
        sparse_access = itir.FunCall(fun=itir.SymRef(id="tuple_get"), args=[offsets[-1], derefed])
        return self.visit(sparse_access)

    def _make_domain(self, node: itir.FunCall):
        tags = []
        sizes = []
        offsets = []
        for named_range in node.args:
            if not (
                isinstance(named_range, itir.FunCall)
                and named_range.fun == itir.SymRef(id="named_range")
            ):
                raise ValueError("Arguments to `domain` need to be calls to `named_range`.")
            tags.append(self.visit(named_range.args[0]))
            sizes.append(
                BinaryExpr(
                    op="-", lhs=self.visit(named_range.args[2]), rhs=self.visit(named_range.args[1])
                )
            )
            offsets.append(self.visit(named_range.args[1]))
        return TaggedValues(tags=tags, values=sizes), TaggedValues(tags=tags, values=offsets)

    @staticmethod
    def _collect_offset_or_axis_node(
        node_type: Type, tree: eve.Node | Iterable[eve.Node]
    ) -> set[str]:
        if not isinstance(tree, Iterable):
            tree = [tree]
        result = set()
        for n in tree:
            result.update(
                n.pre_walk_values()
                .if_isinstance(node_type)
                .getattr("value")
                .if_isinstance(str)
                .to_set()
            )
        return result

    def visit_FunCall(self, node: itir.FunCall, **kwargs: Any) -> Node:
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id in self._unary_op_map:
                assert len(node.args) == 1
                return UnaryExpr(op=self._unary_op_map[node.fun.id], expr=self.visit(node.args[0]))
            elif node.fun.id in self._binary_op_map:
                assert len(node.args) == 2
                return BinaryExpr(
                    op=self._binary_op_map[node.fun.id],
                    lhs=self.visit(node.args[0]),
                    rhs=self.visit(node.args[1]),
                )
            elif node.fun.id == "if_":
                assert len(node.args) == 3
                return TernaryExpr(
                    cond=self.visit(node.args[0]),
                    true_expr=self.visit(node.args[1]),
                    false_expr=self.visit(node.args[2]),
                )
            elif self._is_sparse_deref_shift(node):
                return self._sparse_deref_shift_to_tuple_get(node)
            elif node.fun.id == "cartesian_domain":
                sizes, domain_offsets = self._make_domain(node)
                return CartesianDomain(tagged_sizes=sizes, tagged_offsets=domain_offsets)
            elif node.fun.id == "unstructured_domain":
                sizes, domain_offsets = self._make_domain(node)
                assert "stencil" in kwargs
                shift_offsets = self._collect_offset_or_axis_node(
                    itir.OffsetLiteral, kwargs["stencil"]
                )
                connectivities = []
                for o in shift_offsets:
                    if o in self.offset_provider and isinstance(
                        self.offset_provider[o], common.Connectivity
                    ):
                        connectivities.append(SymRef(id=o))
                return UnstructuredDomain(
                    tagged_sizes=sizes,
                    tagged_offsets=domain_offsets,
                    connectivities=connectivities,
                )
        elif isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="shift"):
            assert len(node.args) == 1
            return FunCall(
                fun=self.visit(node.fun.fun), args=self.visit(node.args) + self.visit(node.fun.args)
            )
        elif isinstance(node.fun, itir.FunCall) and node.fun == itir.SymRef(id="shift"):
            raise ValueError("unapplied shift call not supported: {node}")
        return FunCall(fun=self.visit(node.fun), args=self.visit(node.args))

    def visit_FunctionDefinition(
        self, node: itir.FunctionDefinition, **kwargs: Any
    ) -> FunctionDefinition:
        return FunctionDefinition(
            id=node.id, params=self.visit(node.params), expr=self.visit(node.expr)
        )

    def _visit_output_argument(self, node: itir.SymRef | itir.FunCall):
        if isinstance(node, itir.SymRef):
            return self.visit(node)
        elif isinstance(node, itir.FunCall) and node.fun == itir.SymRef(id="make_tuple"):
            return SidComposite(values=[self._visit_output_argument(v) for v in node.args])
        raise ValueError("Expected `SymRef` or `make_tuple` in output argument.")

    def visit_StencilClosure(self, node: itir.StencilClosure, **kwargs: Any) -> StencilExecution:
        assert isinstance(node.stencil, itir.SymRef)
        backend = Backend(
            domain=self.visit(node.domain, stencil=kwargs["symtable"][node.stencil.id], **kwargs)
        )

        return StencilExecution(
            stencil=self.visit(node.stencil),
            output=self._visit_output_argument(node.output),
            inputs=self.visit(node.inputs),
            backend=backend,
        )

    def visit_FencilDefinition(
        self, node: itir.FencilDefinition, *, grid_type: str, **kwargs: Any
    ) -> FencilDefinition:
        grid_type = getattr(common.GridType, grid_type.upper())
        self.offset_provider = kwargs["offset_provider"]
        executions = self.visit(node.closures, grid_type=grid_type, **kwargs)
        function_definitions = self.visit(node.function_definitions)
        axes = self._collect_offset_or_axis_node(itir.AxisLiteral, node)
        offsets = self._collect_offset_or_axis_node(
            OffsetLiteral, executions + function_definitions
        )  # collect offsets from gtfn nodes as some might have been dropped
        offset_declarations = list(map(lambda x: Sym(id=x), axes | offsets))
        return FencilDefinition(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=executions,
            grid_type=grid_type,
            offset_declarations=offset_declarations,
            function_definitions=function_definitions,
        )
