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
from typing import Callable, Sequence

import dace

import gt4py.eve as eve
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_builtin_field_operator import (
    GtirBuiltinAsFieldOp as AsFieldOp,
)
from gt4py.next.program_processors.runners.dace_fieldview.gtir_builtin_select import (
    GtirBuiltinSelect as Select,
)
from gt4py.next.program_processors.runners.dace_fieldview.gtir_builtin_symbol_ref import (
    GtirBuiltinSymbolRef as SymbolRef,
)
from gt4py.next.program_processors.runners.dace_fieldview.gtir_tasklet_codegen import (
    GtirTaskletCodegen,
)
from gt4py.next.type_system import type_specifications as ts


@dataclass(frozen=True)
class GtirDataflowBuilder(eve.NodeVisitor):
    """Translates a GTIR `ir.Stmt` node to a dataflow graph."""

    _sdfg: dace.SDFG
    _data_types: dict[str, ts.FieldType | ts.ScalarType]

    def visit_domain(self, node: itir.Expr) -> Sequence[tuple[Dimension, str, str]]:
        domain = []
        assert cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))
        for named_range in node.args:
            assert cpm.is_call_to(named_range, "named_range")
            assert len(named_range.args) == 3
            axis = named_range.args[0]
            assert isinstance(axis, itir.AxisLiteral)
            dim = Dimension(axis.value)
            bounds = [self.visit_symbolic(arg) for arg in named_range.args[1:3]]
            domain.append((dim, bounds[0], bounds[1]))

        return domain

    def visit_expression(
        self, node: itir.Expr, head_state: dace.SDFGState
    ) -> list[dace.nodes.AccessNode]:
        expr_builder = self.visit(node, state=head_state)
        assert callable(expr_builder)
        results = expr_builder()

        expressions_nodes = []
        for node, _ in results:
            assert isinstance(node, dace.nodes.AccessNode)
            expressions_nodes.append(node)

        return expressions_nodes

    def visit_symbolic(self, node: itir.Expr) -> str:
        state = self._sdfg.start_state
        codegen = GtirTaskletCodegen(self._sdfg, state)
        return codegen.visit(node)

    def visit_FunCall(self, node: itir.FunCall, state: dace.SDFGState) -> Callable:
        if cpm.is_call_to(node.fun, "as_fieldop"):
            fun_node = node.fun
            assert len(fun_node.args) == 2
            # expect stencil (represented as a lambda function) as first argument
            assert isinstance(fun_node.args[0], itir.Lambda)
            # the domain of the field operator is passed as second argument
            assert isinstance(fun_node.args[1], itir.FunCall)
            field_domain = self.visit_domain(fun_node.args[1])

            stencil_args = [self.visit(arg, state=state) for arg in node.args]

            # add local storage to compute the field operator over the given domain
            # TODO: use type inference to determine the result type
            node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)

            return AsFieldOp(
                sdfg=self._sdfg,
                state=state,
                stencil=fun_node.args[0],
                domain=field_domain,
                args=stencil_args,
                field_dtype=node_type,
            )

        elif cpm.is_call_to(node.fun, "select"):
            fun_node = node.fun
            assert len(fun_node.args) == 3

            # expect condition as first argument
            cond = self.visit_symbolic(fun_node.args[0])

            # use join state to terminate the dataflow on a single exit node
            select_state = self._sdfg.add_state_before(state, state.label + "_select")
            self._sdfg.remove_edge(self._sdfg.out_edges(select_state)[0])

            # expect true branch as second argument
            true_state = self._sdfg.add_state(state.label + "_true_branch")
            self._sdfg.add_edge(select_state, true_state, dace.InterstateEdge(condition=cond))
            self._sdfg.add_edge(true_state, state, dace.InterstateEdge())
            true_br_callable = self.visit(fun_node.args[1], state=true_state)

            # and false branch as third argument
            false_state = self._sdfg.add_state(state.label + "_false_branch")
            self._sdfg.add_edge(
                select_state, false_state, dace.InterstateEdge(condition=f"not {cond}")
            )
            self._sdfg.add_edge(false_state, state, dace.InterstateEdge())
            false_br_callable = self.visit(fun_node.args[2], state=false_state)

            return Select(
                sdfg=self._sdfg,
                state=state,
                true_br_builder=true_br_callable,
                false_br_builder=false_br_callable,
            )

        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    def visit_SymRef(self, node: itir.SymRef, state: dace.SDFGState) -> Callable:
        arg_name = str(node.id)
        assert arg_name in self._data_types
        arg_type = self._data_types[arg_name]
        return SymbolRef(self._sdfg, state, arg_name, arg_type)
