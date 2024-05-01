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
from typing import Any, Callable, final

import dace

import gt4py.eve as eve
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_tasklet_codegen import (
    GtirTaskletCodegen,
)
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type, unique_name
from gt4py.next.type_system import type_specifications as ts


@dataclass(frozen=True)
class GtirDataflowBuilder(eve.NodeVisitor):
    """Translates a GTIR `ir.Stmt` node to a dataflow graph."""

    _sdfg: dace.SDFG
    _state: dace.SDFGState
    _data_types: dict[str, ts.FieldType | ts.ScalarType]

    @final
    def __call__(
        self,
    ) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        """Creates the dataflow representing the given GTIR builtin.

        Returns a list of SDFG nodes and the associated GT4Py data types:
        tuple(node, data_type)

        The GT4Py data type is useful in the case of fields, because it provides
        information on the field domain (e.g. order of dimensions, types of dimensions).
        """
        return self._build()

    @final
    def _add_local_storage(
        self, data_type: ts.FieldType | ts.ScalarType, shape: list[str]
    ) -> dace.nodes.AccessNode:
        name = unique_name("var")
        if isinstance(data_type, ts.FieldType):
            assert len(data_type.dims) == len(shape)
            dtype = as_dace_type(data_type.dtype)
            name, _ = self._sdfg.add_array(name, shape, dtype, find_new_name=True, transient=True)
        else:
            assert len(shape) == 0
            dtype = as_dace_type(data_type)
            name, _ = self._sdfg.add_scalar(name, dtype, find_new_name=True, transient=True)
        return self._state.add_access(name)

    def _build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        raise NotImplementedError

    def _visit_node(self, node: itir.FunCall) -> None:
        raise NotImplementedError

    def fork(self, state: dace.SDFGState) -> "GtirDataflowBuilder":
        return GtirDataflowBuilder(self._sdfg, state, self._data_types)

    def visit_domain(self, node: itir.Expr) -> list[tuple[Dimension, str, str]]:
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
        self, node: itir.Expr
    ) -> tuple[dace.SDFGState, list[dace.nodes.AccessNode]]:
        expr_builder = self.visit(node)
        assert callable(expr_builder)
        results = expr_builder()

        expressions_nodes = []
        for node, _ in results:
            assert isinstance(node, dace.nodes.AccessNode)
            expressions_nodes.append(node)

        # sanity check: each statement should result in a single exit state, i.e. only internal branches
        sink_states = self._sdfg.sink_nodes()
        assert len(sink_states) == 1
        head_state = sink_states[0]

        return head_state, expressions_nodes

    def visit_symbolic(self, node: itir.Expr) -> str:
        return GtirTaskletCodegen().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> Callable:
        from gt4py.next.program_processors.runners.dace_fieldview import gtir_builtins

        arg_builders: list[Callable] = []
        for arg in node.args:
            arg_builder = self.visit(arg)
            assert callable(arg_builder)
            arg_builders.append(arg_builder)

        if cpm.is_call_to(node.fun, "as_fieldop"):
            return gtir_builtins.AsFieldOp(
                self._sdfg,
                self._state,
                self._data_types,
                node,
                arg_builders,
            )

        elif cpm.is_call_to(node.fun, "select"):
            assert len(arg_builders) == 0
            return gtir_builtins.Select(
                self._sdfg,
                self._state,
                self._data_types,
                node,
            )

        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    @final
    def visit_Lambda(self, node: itir.Lambda) -> Any:
        # This visitor class should never encounter `itir.Lambda` expressions
        # because a lambda represents a stencil, which translates from iterator to value.
        # In fieldview, lambdas should only be arguments to field operators (`as_field_op`).
        raise RuntimeError("Unexpected 'itir.Lambda' node encountered by 'GtirTaskletCodegen'.")

    def visit_SymRef(self, node: itir.SymRef) -> Callable:
        from gt4py.next.program_processors.runners.dace_fieldview import gtir_builtins

        return gtir_builtins.SymbolRef(self._sdfg, self._state, self._data_types, node)
