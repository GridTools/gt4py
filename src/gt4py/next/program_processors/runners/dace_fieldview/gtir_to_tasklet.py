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


from dataclasses import dataclass
from typing import TypeAlias

import dace
import dace.subsets as sbs

from gt4py import eve
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_python_codegen import (
    MATH_BUILTINS_MAPPING,
)
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type, unique_name


@dataclass(frozen=True)
class MemletExpr:
    """Scalar or array data access thorugh a memlet."""

    source: dace.nodes.AccessNode
    subset: sbs.Indices | sbs.Range


@dataclass(frozen=True)
class SymbolExpr:
    """Any symbolic expression that is constant in the context of current SDFG."""

    value: dace.symbolic.SymExpr
    dtype: dace.typeclass


@dataclass(frozen=True)
class TaskletExpr:
    """Result of the computation provided by a tasklet node."""

    node: dace.nodes.Tasklet
    connector: str


@dataclass(frozen=True)
class IteratorExpr:
    """Iterator for field access to be consumed by `deref` or `shift` builtin functions."""

    field: dace.nodes.AccessNode
    dimensions: list[str]
    offset: list[dace.symbolic.SymExpr]
    indices: dict[str, MemletExpr | SymbolExpr | TaskletExpr]


InputConnection: TypeAlias = tuple[
    dace.nodes.AccessNode,
    sbs.Range,
    dace.nodes.Tasklet,
    str,
]


class GTIRToTasklet(eve.NodeVisitor):
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""

    sdfg: dace.SDFG
    state: dace.SDFGState
    input_connections: list[InputConnection]
    offset_provider: dict[str, Connectivity | Dimension]
    symbol_map: dict[str, SymbolExpr | IteratorExpr | MemletExpr]

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

    def _visit_deref(self, node: itir.FunCall) -> MemletExpr | TaskletExpr:
        assert len(node.args) == 1
        it = self.visit(node.args[0])

        if isinstance(it, SymbolExpr):
            cast_sym = str(it.dtype)
            cast_fmt = MATH_BUILTINS_MAPPING[cast_sym]
            deref_node = self.state.add_tasklet(
                "deref_symbol", {}, {"val"}, code=f"val = {cast_fmt.format(it.value)}"
            )
            return TaskletExpr(deref_node, "val")

        elif isinstance(it, IteratorExpr):
            if all(isinstance(index, SymbolExpr) for index in it.indices.values()):
                # use direct field access through memlet subset
                data_index = sbs.Indices(
                    [
                        it.indices[dim].value + off  # type: ignore[union-attr]
                        for dim, off in zip(it.dimensions, it.offset, strict=True)
                    ]
                )
                return MemletExpr(it.field, data_index)

            else:
                raise NotImplementedError

        else:
            assert isinstance(it, MemletExpr)
            return it

    def _visit_shift(self, node: itir.FunCall) -> IteratorExpr:
        raise NotImplementedError

    def visit_FunCall(self, node: itir.FunCall) -> IteratorExpr | TaskletExpr | MemletExpr:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

        elif cpm.is_call_to(node.fun, "shift"):
            return self._visit_shift(node)

        else:
            assert isinstance(node.fun, itir.SymRef)

        node_internals = []
        node_connections: dict[str, MemletExpr | TaskletExpr] = {}
        for i, arg in enumerate(node.args):
            arg_expr = self.visit(arg)
            if isinstance(arg_expr, MemletExpr | TaskletExpr):
                # the argument value is the result of a tasklet node or direct field access
                connector = f"__inp_{i}"
                node_connections[connector] = arg_expr
                node_internals.append(connector)
            else:
                assert isinstance(arg_expr, SymbolExpr)
                # use the argument value without adding any connector
                node_internals.append(arg_expr.value)

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
            if isinstance(arg_expr, TaskletExpr):
                self.state.add_edge(
                    arg_expr.node, arg_expr.connector, tasklet_node, connector, dace.Memlet()
                )
            else:
                self.input_connections.append(
                    (arg_expr.source, arg_expr.subset, tasklet_node, connector)
                )

        return TaskletExpr(tasklet_node, "result")

    def visit_Lambda(
        self, node: itir.Lambda, args: list[SymbolExpr | IteratorExpr | MemletExpr]
    ) -> tuple[
        list[InputConnection],
        TaskletExpr,
    ]:
        for p, arg in zip(node.params, args, strict=True):
            self.symbol_map[str(p.id)] = arg
        output_expr = self.visit(node.expr)
        if isinstance(output_expr, TaskletExpr):
            return self.input_connections, output_expr

        # special case where the field operator is simply copying data from source to destination node
        assert isinstance(output_expr, MemletExpr)
        tasklet_node = self.state.add_tasklet("copy", {"__inp"}, {"__out"}, "__out = __inp")
        self.input_connections.append(
            (output_expr.source, output_expr.subset, tasklet_node, "__inp")
        )
        return self.input_connections, TaskletExpr(tasklet_node, "__out")

    def visit_Literal(self, node: itir.Literal) -> SymbolExpr:
        dtype = as_dace_type(node.type)
        return SymbolExpr(node.value, dtype)

    def visit_SymRef(self, node: itir.SymRef) -> SymbolExpr | IteratorExpr | MemletExpr:
        param = str(node.id)
        assert param in self.symbol_map
        return self.symbol_map[param]
