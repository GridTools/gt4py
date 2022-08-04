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
    OffsetDefinition,
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
        "not_eq": "!=",
        "less": "<",
        "less_equal": "<=",
        "greater": ">",
        "greater_equal": ">=",
        "and_": "&&",
        "or_": "||",
    }
    _unary_op_map = {"not_": "!"}

    def __init__(self):
        self.offset_definitions: dict[str, common.Dimension | common.Connectivity] = {}

    def visit_Sym(self, node: itir.Sym, **kwargs: Any) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(self, node: itir.SymRef, **kwargs: Any) -> SymRef:
        return SymRef(id=node.id)

    def visit_Lambda(self, node: itir.Lambda, **kwargs: Any) -> Lambda:
        return Lambda(params=self.visit(node.params), expr=self.visit(node.expr, **kwargs))

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type=node.type)

    _vertical_dimension = "unstructured::dim::vertical"
    _horizontal_dimension = "unstructured::dim::horizontal"

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs: Any) -> OffsetLiteral:
        if isinstance(node.value, int):
            return OffsetLiteral(value=node.value)

        if node.value not in self.offset_provider:
            raise ValueError(f"Missing offset_provider entry for {node.value}")
        offset_name = node.value
        if isinstance(self.offset_provider[node.value], common.Dimension):
            dim = self.offset_provider[node.value]
            if self.grid_type == common.GridType.CARTESIAN:
                # rename offset to dimension
                self.offset_definitions[dim.value] = OffsetDefinition(name=Sym(id=dim.value))
                return OffsetLiteral(value=dim.value)
            else:
                assert self.grid_type == common.GridType.UNSTRUCTURED
                if not dim.kind == common.DimensionKind.VERTICAL:
                    raise ValueError(
                        "Mapping an offset to a horizontal dimension in unstructured is not allowed."
                    )
                self.offset_definitions[offset_name] = OffsetDefinition(
                    name=Sym(id=offset_name), alias=self._vertical_dimension
                )
                return OffsetLiteral(value=offset_name)
        self.offset_definitions[offset_name] = OffsetDefinition(name=Sym(id=offset_name))
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
                return UnaryExpr(
                    op=self._unary_op_map[node.fun.id], expr=self.visit(node.args[0], **kwargs)
                )
            elif node.fun.id in self._binary_op_map:
                assert len(node.args) == 2
                return BinaryExpr(
                    op=self._binary_op_map[node.fun.id],
                    lhs=self.visit(node.args[0], **kwargs),
                    rhs=self.visit(node.args[1], **kwargs),
                )
            elif node.fun.id == "if_":
                assert len(node.args) == 3
                return TernaryExpr(
                    cond=self.visit(node.args[0], **kwargs),
                    true_expr=self.visit(node.args[1], **kwargs),
                    false_expr=self.visit(node.args[2], **kwargs),
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
                fun=self.visit(node.fun.fun, **kwargs),
                args=self.visit(node.args, **kwargs) + self.visit(node.fun.args, **kwargs),
            )
        elif isinstance(node.fun, itir.FunCall) and node.fun == itir.SymRef(id="shift"):
            raise ValueError("unapplied shift call not supported: {node}")
        return FunCall(fun=self.visit(node.fun, **kwargs), args=self.visit(node.args, **kwargs))

    def visit_FunctionDefinition(
        self, node: itir.FunctionDefinition, **kwargs: Any
    ) -> FunctionDefinition:
        return FunctionDefinition(
            id=node.id, params=self.visit(node.params), expr=self.visit(node.expr, **kwargs)
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
            stencil=self.visit(node.stencil, **kwargs),
            output=self._visit_output_argument(node.output),
            inputs=self.visit(node.inputs),
            backend=backend,
        )

    @staticmethod
    def _get_gridtype(closures: list[itir.StencilClosure]) -> common.GridType:
        def _extract_grid_type(domain_arg: itir.Expr) -> common.GridType:
            assert isinstance(domain_arg, itir.FunCall)
            if domain_arg.fun == itir.SymRef(id="cartesian_domain"):
                return common.GridType.CARTESIAN
            else:
                assert domain_arg.fun == itir.SymRef(id="unstructured_domain")
                return common.GridType.UNSTRUCTURED

        first = _extract_grid_type(closures[0].domain)
        assert all(first == _extract_grid_type(c.domain) for c in closures)
        return first

    def _collect_dimensions_from_domain(self, closures: list[itir.StencilClosure]) -> None:
        for c in closures:
            if self.grid_type == common.GridType.CARTESIAN:
                if isinstance(c.domain, itir.FunCall) and c.domain.fun == itir.SymRef(
                    id="cartesian_domain"
                ):
                    for nr in c.domain.args:
                        assert isinstance(nr, itir.FunCall) and nr.fun == itir.SymRef(
                            id="named_range"
                        )
                        dim_name = nr.args[0].value
                        self.offset_definitions[dim_name] = OffsetDefinition(name=Sym(id=dim_name))
            else:
                if isinstance(c.domain, itir.FunCall) and c.domain.fun == itir.SymRef(
                    id="unstructured_domain"
                ):
                    if len(c.domain.args) > 0:
                        horizontal_name = c.domain.args[0].args[0].value
                        self.offset_definitions[horizontal_name] = OffsetDefinition(
                            name=Sym(id=horizontal_name), alias=self._horizontal_dimension
                        )
                    if len(c.domain.args) > 1:
                        vertical_name = c.domain.args[1].args[0].value
                        self.offset_definitions[vertical_name] = OffsetDefinition(
                            name=Sym(id=vertical_name), alias=self._vertical_dimension
                        )
                    if len(c.domain.args) > 2:
                        raise RuntimeError(
                            "unstructured_domain must not have more than 2 arguments."
                        )

    def visit_FencilDefinition(
        self, node: itir.FencilDefinition, **kwargs: Any
    ) -> FencilDefinition:
        self.grid_type = self._get_gridtype(node.closures)
        self.offset_provider = kwargs["offset_provider"]
        executions = self.visit(node.closures, **kwargs)
        function_definitions = self.visit(node.function_definitions)
        self._collect_dimensions_from_domain(node.closures)
        return FencilDefinition(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=executions,
            grid_type=self.grid_type,
            offset_definitions=list(self.offset_definitions.values()),
            function_definitions=function_definitions,
        )
