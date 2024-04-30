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
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type
from gt4py.next.type_system import type_specifications as ts


def unique_name(prefix: str) -> str:
    unique_id = getattr(unique_name, "_unique_id", 0)  # static variable
    setattr(unique_name, "_unique_id", unique_id + 1)  # noqa: B010 [set-attr-with-constant]

    return f"{prefix}_{unique_id}"


class GtirDataflowBuilder(eve.NodeVisitor):
    """Translates a GTIR `ir.Stmt` node to a dataflow graph."""

    _sdfg: dace.SDFG
    _data_types: dict[str, ts.FieldType | ts.ScalarType]

    def __init__(
        self,
        sdfg: dace.SDFG,
        data_types: dict[str, ts.FieldType | ts.ScalarType],
    ):
        self._sdfg = sdfg
        self._data_types = data_types

    def _add_local_storage(
        self, type_: ts.DataType, shape: list[str]
    ) -> tuple[str, dace.data.Data]:
        name = unique_name("var")
        if isinstance(type_, ts.FieldType):
            dtype = as_dace_type(type_.dtype)
            assert len(type_.dims) == len(shape)
            # TODO: for now we let DaCe decide the array strides, evaluate if symblic strides should be used
            name, data = self._sdfg.add_array(
                name, shape, dtype, find_new_name=True, transient=True
            )
        else:
            assert isinstance(type_, ts.ScalarType)
            assert len(shape) == 0
            dtype = as_dace_type(type_)
            name, data = self._sdfg.add_scalar(name, dtype, find_new_name=True, transient=True)
        return name, data

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
    ) -> tuple[list[dace.nodes.AccessNode], dace.SDFGState]:
        expr_builder = self.visit(node, state=head_state)
        assert callable(expr_builder)
        results, head_state = expr_builder()

        expressions_nodes = []
        for node, _ in results:
            assert isinstance(node, dace.nodes.AccessNode)
            expressions_nodes.append(node)

        return expressions_nodes, head_state

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
            join_state = self._sdfg.add_state(state.label + "_join")

            # expect true branch as second argument
            true_state = self._sdfg.add_state(state.label + "_true_branch")
            self._sdfg.add_edge(state, true_state, dace.InterstateEdge(condition=cond))
            self._sdfg.add_edge(true_state, join_state, dace.InterstateEdge())
            true_br_callable = self.visit(fun_node.args[1], state=true_state)

            # and false branch as third argument
            false_state = self._sdfg.add_state(state.label + "_false_branch")
            self._sdfg.add_edge(state, false_state, dace.InterstateEdge(condition=f"not {cond}"))
            self._sdfg.add_edge(false_state, join_state, dace.InterstateEdge())
            false_br_callable = self.visit(fun_node.args[2], state=false_state)

            return Select(
                sdfg=self._sdfg,
                state=join_state,
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
