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
from gt4py.next.type_system import type_specifications as ts


@dataclass(frozen=True)
class GtirDataflowBuilder(eve.NodeVisitor):
    """Translates a GTIR `ir.Stmt` node to a dataflow graph."""

    sdfg: dace.SDFG
    data_types: dict[str, ts.FieldType | ts.ScalarType]

    def visit_domain(self, node: itir.Expr) -> list[tuple[Dimension, str, str]]:
        """
        Specialized visit method for domain expressions.

        Returns a list of dimensions and the corresponding range.
        """
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
        """
        Specialized visit method for fieldview expressions.

        This method represents the entry point to visit 'Stmt' expressions.
        As such, it must preserve the property of single exit state in the SDFG.

        TODO: do we need to return the GT4Py `FieldType`/`ScalarType`?
        """
        expr_builder = self.visit(node, head_state=head_state)
        assert callable(expr_builder)
        results = expr_builder()

        expressions_nodes = []
        for node, _ in results:
            assert isinstance(node, dace.nodes.AccessNode)
            expressions_nodes.append(node)

        # sanity check: each statement should preserve the property of single exit state (aka head state),
        # i.e. eventually only introduce internal branches, and keep the same head state
        sink_states = self.sdfg.sink_nodes()
        assert len(sink_states) == 1
        assert sink_states[0] == head_state

        return expressions_nodes

    def visit_symbolic(self, node: itir.Expr) -> str:
        """
        Specialized visit method for pure stencil expressions.

        Returns a string represnting the Python code to be used as tasklet body.
        TODO: should we return a list of code strings in case of tuple returns,
        one for each output value?
        """
        return GtirTaskletCodegen().visit(node)

    def visit_FunCall(self, node: itir.FunCall, head_state: dace.SDFGState) -> Callable:
        from gt4py.next.program_processors.runners.dace_fieldview import gtir_builtins

        arg_builders: list[Callable] = []
        for arg in node.args:
            arg_builder = self.visit(arg, head_state=head_state)
            assert callable(arg_builder)
            arg_builders.append(arg_builder)

        if cpm.is_call_to(node.fun, "as_fieldop"):
            return gtir_builtins.AsFieldOp(self, head_state, node, arg_builders)

        elif cpm.is_call_to(node.fun, "select"):
            assert len(arg_builders) == 0
            return gtir_builtins.Select(self, head_state, node)

        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    @final
    def visit_Lambda(self, node: itir.Lambda) -> Any:
        """
        This visitor class should never encounter `itir.Lambda` expressions
        because a lambda represents a stencil, which translates from iterator to values.
        In fieldview, lambdas should only be arguments to field operators (`as_field_op`).
        """
        raise RuntimeError("Unexpected 'itir.Lambda' node encountered by 'GtirTaskletCodegen'.")

    def visit_SymRef(self, node: itir.SymRef, head_state: dace.SDFGState) -> Callable:
        from gt4py.next.program_processors.runners.dace_fieldview import gtir_builtins

        return gtir_builtins.SymbolRef(self, head_state, node)
