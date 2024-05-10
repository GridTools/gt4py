# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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


import itertools
from dataclasses import dataclass
from typing import Optional

import dace

from gt4py import eve
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_python_codegen import (
    MATH_BUILTINS_MAPPING,
)
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type, unique_name
from gt4py.next.type_system import type_specifications as ts


@dataclass(frozen=True)
class SymbolExpr:
    """Any symbolic expression that is constant in the context of current SDFG."""

    value: dace.symbolic.SymbolicType
    dtype: dace.typeclass


@dataclass(frozen=True)
class TaskletExpr:
    """Result of the computation provided by a tasklet node."""

    node: dace.nodes.Tasklet
    connector: str
    dtype: dace.typeclass


@dataclass(frozen=True)
class ValueExpr:
    """Data provided by a scalar access node."""

    value: dace.nodes.AccessNode
    dtype: dace.typeclass


@dataclass(frozen=True)
class IteratorExpr:
    """Iterator to access the field provided by an array access node."""

    field: dace.nodes.AccessNode
    dimensions: list[str]
    offset: list[int]
    indices: dict[str, SymbolExpr | TaskletExpr | ValueExpr]
    dtype: dace.typeclass


class GTIRToTasklet(eve.NodeVisitor):
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""

    sdfg: dace.SDFG
    state: dace.SDFGState
    input_connections: list[tuple[dace.nodes.AccessNode, tuple[dace.nodes.Tasklet, str, list[int]]]]
    offset_provider: dict[str, Connectivity | Dimension]
    symbol_map: dict[str, IteratorExpr | SymbolExpr | ValueExpr]

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        offset_provider: dict[str, Connectivity | Dimension],
    ):
        self.sdfg = sdfg
        self.state = state
        self.input_connections = []
        self.offset_provider = offset_provider
        self.symbol_map = {}

    def _visit_deref(self, node: itir.FunCall) -> TaskletExpr:
        assert len(node.args) == 1
        it = self.visit(node.args[0])

        if isinstance(it, SymbolExpr):
            cast_sym = str(it.dtype)
            cast_fmt = MATH_BUILTINS_MAPPING[cast_sym]
            deref_node = self.state.add_tasklet(
                "deref_symbol", {}, {"val"}, code=f"val = {cast_fmt.format(it.value)}"
            )
            deref_expr = TaskletExpr(deref_node, "val", it.dtype)

        elif isinstance(it, ValueExpr):
            deref_node = self.state.add_tasklet(
                "deref_scalar", {"scalar"}, {"val"}, code="val = scalar"
            )
            deref_expr = TaskletExpr(deref_node, "val", it.dtype)

            # add new termination point for the data access
            self.input_connections.append((it.value, (deref_node, "scalar", [])))

        else:
            if all(isinstance(index, SymbolExpr) for index in it.indices.values()):
                # use direct field access through memlet subset
                deref_node = self.state.add_tasklet(
                    "deref_field", {"field"}, {"val"}, code="val = field"
                )
            else:
                index_connectors = [
                    f"i_{dim}"
                    for dim, index in it.indices.items()
                    if isinstance(index, TaskletExpr | ValueError)
                ]
                sorted_indices = [it.indices[dim] for dim in it.dimensions]
                index_internals = ",".join(
                    index.value if isinstance(index, SymbolExpr) else f"i_{dim}"
                    for dim, index in zip(it.dimensions, sorted_indices)
                )
                deref_node = self.state.add_tasklet(
                    "deref_field_indirecton",
                    set("field", *index_connectors),
                    {"val"},
                    code=f"val = field[{index_internals}]",
                )
                for dim, index_expr in it.indices.items():
                    deref_connector = f"i_{dim}"
                    if isinstance(index_expr, TaskletExpr):
                        self.state.add_edge(
                            index_expr.node,
                            index_expr.connector,
                            deref_node,
                            deref_connector,
                            dace.Memlet(data=index_expr.node.data, subset="0"),
                        )
                    elif isinstance(index_expr, ValueExpr):
                        self.state.add_edge(
                            index_expr.value,
                            None,
                            deref_node,
                            deref_connector,
                            dace.Memlet(data=index_expr.value.data, subset="0"),
                        )
                    else:
                        assert isinstance(index_expr, SymbolExpr)

            deref_expr = TaskletExpr(deref_node, "val", it.dtype)

            # add new termination point for this field parameter
            self.input_connections.append((it.field, (deref_node, "field", it.offset)))

        return deref_expr

    def _split_shift_args(
        self, args: list[itir.Expr]
    ) -> tuple[list[itir.Expr], Optional[list[itir.Expr]]]:
        pairs = [args[i : i + 2] for i in range(0, len(args), 2)]
        assert len(pairs) >= 1
        assert all(len(pair) == 2 for pair in pairs)
        return pairs[-1], list(itertools.chain(*pairs[0:-1])) if len(pairs) > 1 else None

    def _make_shift_for_rest(self, rest: list[itir.Expr], iterator: itir.Expr) -> itir.FunCall:
        return itir.FunCall(
            fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=rest),
            args=[iterator],
        )

    def _visit_shift(self, node: itir.FunCall) -> IteratorExpr:
        shift_node = node.fun
        assert isinstance(shift_node, itir.FunCall)

        # the iterator to be shifted is the argument to the function node
        head, tail = self._split_shift_args(shift_node.args)
        if tail:
            it = self.visit(self._make_shift_for_rest(tail, node.args[0]))
        else:
            it = self.visit(node.args[0])
        assert isinstance(it, IteratorExpr)

        # first argument of the shift node is the offset provider
        assert isinstance(head[0], itir.OffsetLiteral)
        offset = head[0].value
        assert isinstance(offset, str)
        offset_provider = self.offset_provider[offset]
        # second argument should be the offset value
        if isinstance(head[1], itir.OffsetLiteral):
            offset_value = head[1].value
            assert isinstance(offset_value, int)
        else:
            raise NotImplementedError("Dynamic offset not supported.")

        if isinstance(offset_provider, Dimension):
            # cartesian offset along one dimension
            dim_index = it.dimensions.index(offset_provider.value)
            new_offset = [
                prev_offset + offset_value if i == dim_index else prev_offset
                for i, prev_offset in enumerate(it.offset)
            ]
            shifted_it = IteratorExpr(it.field, it.dimensions, new_offset, it.indices, it.dtype)
        else:
            # shift in unstructured domain by means of a neighbor table
            raise NotImplementedError

        return shifted_it

    def visit_FunCall(self, node: itir.FunCall) -> IteratorExpr | TaskletExpr:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

        elif cpm.is_call_to(node.fun, "shift"):
            return self._visit_shift(node)

        else:
            assert isinstance(node.fun, itir.SymRef)

        node_internals = []
        node_connections = {}
        for i, arg in enumerate(node.args):
            arg_expr = self.visit(arg)
            if isinstance(arg_expr, SymbolExpr):
                # use the argument value without adding any connector
                node_internals.append(arg_expr.value)
            else:
                assert isinstance(arg_expr, TaskletExpr)
                # the argument value is the result of a tasklet node
                connector = f"__inp_{i}"
                node_connections[connector] = arg_expr
                node_internals.append(connector)

        # TODO: use type inference to determine the result type
        node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)

        # create a tasklet node implementing the builtin function
        builtin_name = str(node.fun.id)
        if builtin_name in MATH_BUILTINS_MAPPING:
            fmt = MATH_BUILTINS_MAPPING[builtin_name]
            code = fmt.format(*node_internals)
        else:
            raise NotImplementedError(f"'{builtin_name}' not implemented.")

        out_connector = "result"
        tasklet_node = self.state.add_tasklet(
            unique_name("tasklet"),
            node_connections.keys(),
            {out_connector},
            "{} = {}".format(out_connector, code),
        )

        for connector, arg_expr in node_connections.items():
            self.state.add_edge(
                arg_expr.node, arg_expr.connector, tasklet_node, connector, dace.Memlet()
            )

        dtype = as_dace_type(node_type)
        return TaskletExpr(tasklet_node, "result", dtype)

    def visit_Lambda(
        self, node: itir.Lambda, args: list[IteratorExpr | SymbolExpr | ValueExpr]
    ) -> tuple[
        list[tuple[dace.nodes.AccessNode, tuple[dace.nodes.Tasklet, str, list[int]]]], TaskletExpr
    ]:
        for p, arg in zip(node.params, args, strict=True):
            self.symbol_map[str(p.id)] = arg
        output_expr = self.visit(node.expr)
        assert isinstance(output_expr, TaskletExpr)
        return self.input_connections, output_expr

    def visit_Literal(self, node: itir.Literal) -> SymbolExpr:
        dtype = as_dace_type(node.type)
        return SymbolExpr(node.value, dtype)

    def visit_SymRef(self, node: itir.SymRef) -> IteratorExpr | SymbolExpr | ValueExpr:
        param = str(node.id)
        assert param in self.symbol_map
        return self.symbol_map[param]
