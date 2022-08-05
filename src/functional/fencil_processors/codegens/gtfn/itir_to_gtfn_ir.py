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
    TagDefinition,
    TaggedValues,
    TernaryExpr,
    UnaryExpr,
    UnstructuredDomain,
)
from functional.iterator import ir as itir


_vertical_dimension = "unstructured::dim::vertical"
_horizontal_dimension = "unstructured::dim::horizontal"


def _get_domains(closures: Iterable[itir.StencilClosure]) -> Iterable[itir.FunCall]:
    domains = [c.domain for c in closures]
    assert all(isinstance(d, itir.FunCall) for d in domains)
    return domains


def _extract_grid_type(domain: itir.FunCall) -> common.GridType:
    if domain.fun == itir.SymRef(id="cartesian_domain"):
        return common.GridType.CARTESIAN
    else:
        assert domain.fun == itir.SymRef(id="unstructured_domain")
        return common.GridType.UNSTRUCTURED


def _get_gridtype(closures: list[itir.StencilClosure]) -> common.GridType:
    domains = _get_domains(closures)
    grid_types = set(_extract_grid_type(d) for d in domains)
    if len(grid_types) != 1:
        raise ValueError(
            f"Found StencilClosures with more than one GridType: {grid_types}. This is currently not supported."
        )
    return grid_types.pop()


def _name_from_named_range(named_range_call: itir.FunCall) -> str:
    assert isinstance(named_range_call, itir.FunCall) and named_range_call.fun == itir.SymRef(
        id="named_range"
    )
    return named_range_call.args[0].value


def _collect_dimensions_from_domain(
    closures: Iterable[itir.StencilClosure],
) -> dict[str, TagDefinition]:
    domains = _get_domains(closures)
    offset_definitions = {}
    for domain in domains:
        if domain.fun == itir.SymRef(id="cartesian_domain"):
            for nr in domain.args:
                dim_name = _name_from_named_range(nr)
                offset_definitions[dim_name] = TagDefinition(name=Sym(id=dim_name))
        elif domain.fun == itir.SymRef(id="unstructured_domain"):
            if len(domain.args) > 2:
                raise ValueError("unstructured_domain must not have more than 2 arguments.")
            if len(domain.args) > 0:
                horizontal_range = domain.args[0]
                horizontal_name = _name_from_named_range(horizontal_range)
                offset_definitions[horizontal_name] = TagDefinition(
                    name=Sym(id=horizontal_name), alias=_horizontal_dimension
                )
            if len(domain.args) > 1:
                vertical_range = domain.args[1]
                vertical_name = _name_from_named_range(vertical_range)
                offset_definitions[vertical_name] = TagDefinition(
                    name=Sym(id=vertical_name), alias=_vertical_dimension
                )
        else:
            raise AssertionError(
                "Expected either a call to `cartesian_domain` or to `unstructured_domain`."
            )
    return offset_definitions


def _collect_offset_definitions(
    node: itir.Node,
    grid_type: common.GridType,
    offset_provider: dict[str, common.Dimension | common.Connectivity],
):
    offset_tags: Iterable[itir.OffsetLiteral] = (
        node.walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .filter(lambda offset_literal: isinstance(offset_literal.value, str))
    )
    offset_definitions = {}

    for o in offset_tags:
        if o.value not in offset_provider:
            raise ValueError(f"Missing offset_provider entry for {o.value}")

        offset_name = o.value
        if isinstance(offset_provider[o.value], common.Dimension):
            dim = offset_provider[o.value]
            if grid_type == common.GridType.CARTESIAN:
                # create alias from offset to dimension
                offset_definitions[dim.value] = TagDefinition(name=Sym(id=dim.value))
                offset_definitions[offset_name] = TagDefinition(
                    name=Sym(id=offset_name), alias=SymRef(id=dim.value)
                )
            else:
                assert grid_type == common.GridType.UNSTRUCTURED
                if not dim.kind == common.DimensionKind.VERTICAL:
                    raise ValueError(
                        "Mapping an offset to a horizontal dimension in unstructured is not allowed."
                    )
                # create alias from vertical offset to vertical dimension
                offset_definitions[offset_name] = TagDefinition(
                    name=Sym(id=offset_name), alias=_vertical_dimension
                )
        else:
            assert isinstance(offset_provider[o.value], common.Connectivity)
            offset_definitions[offset_name] = TagDefinition(name=Sym(id=offset_name))
    return offset_definitions


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

    def visit_Sym(self, node: itir.Sym, **kwargs: Any) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(self, node: itir.SymRef, **kwargs: Any) -> SymRef:
        return SymRef(id=node.id)

    def visit_Lambda(self, node: itir.Lambda, **kwargs: Any) -> Lambda:
        return Lambda(params=self.visit(node.params), expr=self.visit(node.expr, **kwargs))

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type=node.type)

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs: Any) -> OffsetLiteral:
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

    def visit_FencilDefinition(
        self, node: itir.FencilDefinition, **kwargs: Any
    ) -> FencilDefinition:
        self.grid_type = _get_gridtype(node.closures)
        self.offset_provider = kwargs["offset_provider"]

        executions = self.visit(node.closures, **kwargs)
        function_definitions = self.visit(node.function_definitions)
        offset_definitions = {
            **_collect_dimensions_from_domain(node.closures),
            **_collect_offset_definitions(node, self.grid_type, self.offset_provider),
        }
        return FencilDefinition(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=executions,
            grid_type=self.grid_type,
            offset_definitions=list(offset_definitions.values()),
            function_definitions=function_definitions,
        )
