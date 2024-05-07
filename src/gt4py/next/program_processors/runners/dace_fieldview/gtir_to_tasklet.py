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

import dace

from gt4py import eve
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_python_codegen import (
    MATH_BUILTINS_MAPPING,
)
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type, unique_name


@dataclass(frozen=True)
class LiteralExpr:
    """Any symbolic expression that can be evaluated at compile time."""

    value: dace.symbolic.SymbolicType


@dataclass(frozen=True)
class SymbolExpr:
    """The data access to a scalar or field through a symbolic reference."""

    data: str


@dataclass(frozen=True)
class ValueExpr:
    """The result of a computation provided by a tasklet node."""

    node: dace.nodes.Tasklet
    connector: str


class GTIRToTasklet(eve.NodeVisitor):
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""

    sdfg: dace.SDFG
    state: dace.SDFGState
    input_connections: list[tuple[ValueExpr, str]]
    offset_provider: dict[str, Connectivity | Dimension]

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

    def _visit_shift(self, node: itir.FunCall) -> str:
        assert len(node.args) == 2
        raise NotImplementedError

    def visit_FunCall(self, node: itir.FunCall) -> ValueExpr | SymbolExpr:
        inp_tasklets = {}
        inp_symbols = set()
        inp_connectors = set()
        node_internals = []
        for i, arg in enumerate(node.args):
            arg_expr = self.visit(arg)
            if isinstance(arg_expr, LiteralExpr):
                # use the value without adding any connector
                node_internals.append(arg_expr.value)
            else:
                if isinstance(arg_expr, ValueExpr):
                    # the value is the result of a tasklet node
                    connector = f"__inp_{i}"
                    inp_tasklets[connector] = arg_expr
                else:
                    # the value is the result of a tasklet node
                    assert isinstance(arg_expr, SymbolExpr)
                    connector = f"__inp_{arg_expr.data}"
                    inp_symbols.add((connector, arg_expr.data))
                inp_connectors.add(connector)
                node_internals.append(connector)

        if cpm.is_call_to(node, "deref"):
            assert len(inp_tasklets) == 0
            assert len(inp_symbols) == 1
            _, data = inp_symbols.pop()
            return SymbolExpr(data)

        elif cpm.is_call_to(node.fun, "shift"):
            code = self._visit_shift(node.fun)

        elif isinstance(node.fun, itir.SymRef):
            # create a tasklet node implementing the builtin function
            builtin_name = str(node.fun.id)
            if builtin_name in MATH_BUILTINS_MAPPING:
                fmt = MATH_BUILTINS_MAPPING[builtin_name]
                code = fmt.format(*node_internals)
            else:
                raise NotImplementedError(f"'{builtin_name}' not implemented.")

            out_connector = "__out"
            tasklet_node = self.state.add_tasklet(
                unique_name("tasklet"),
                inp_connectors,
                {out_connector},
                "{} = {}".format(out_connector, code),
            )

        else:
            raise NotImplementedError(f"Unexpected 'FunCall' node ({node}).")

        for input_conn, inp_expr in inp_tasklets.items():
            self.state.add_edge(
                inp_expr.node, inp_expr.connector, tasklet_node, input_conn, dace.Memlet()
            )
        self.input_connections.extend(
            (ValueExpr(tasklet_node, connector), data) for connector, data in inp_symbols
        )
        return ValueExpr(tasklet_node, out_connector)

    def visit_Lambda(self, node: itir.Lambda) -> tuple[list[tuple[ValueExpr, str]], ValueExpr]:
        assert len(self.input_connections) == 0
        output_expr = self.visit(node.expr)
        assert isinstance(output_expr, ValueExpr)
        return self.input_connections, output_expr

    def visit_Literal(self, node: itir.Literal) -> LiteralExpr:
        cast_sym = str(as_dace_type(node.type))
        cast_fmt = MATH_BUILTINS_MAPPING[cast_sym]
        typed_value = cast_fmt.format(node.value)
        return LiteralExpr(typed_value)

    def visit_SymRef(self, node: itir.SymRef) -> SymbolExpr:
        """
        Symbol references are mapped to tasklet connectors that access some kind of data.
        """
        sym_name = str(node.id)
        return SymbolExpr(sym_name)
