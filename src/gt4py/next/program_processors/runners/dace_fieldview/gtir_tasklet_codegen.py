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

import dataclasses
from typing import Optional, final

import dace

from gt4py.eve import codegen
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts

from .gtir_dataflow_context import GtirDataflowContext as DataflowContext


@dataclasses.dataclass(frozen=True)
class GtirTaskletSubgraph:
    """Defines a tasklet subgraph representing a stencil expression.

    The tasklet subgraph will be used by the consumer to build a fieldview expression.
    For example, it could be used in a map scope to build a fieldview expression;
    or it could become the body of a scan expression.
    """

    # generic DaCe node, most often this will be a tasklet node but it could also be a nested SDFG
    node: dace.nodes.Node

    # for each input/output connections, specify the field type or None if scalar
    input_connections: list[tuple[str, Optional[ts.FieldType]]]
    output_connections: list[tuple[str, Optional[ts.FieldType]]]


class GtirTaskletCodegen(codegen.TemplatedGenerator):
    """Base class to translate GTIR to Python code to be used as tasklet body."""

    _ctx: DataflowContext
    # list of input/output connectors and expected field type (None if scalar)
    _input_connections: list[tuple[str, Optional[ts.FieldType]]]
    _output_connections: list[tuple[str, Optional[ts.FieldType]]]

    def __init__(self, ctx: DataflowContext):
        self._ctx = ctx
        self._input_connections = []
        self._output_connections = []

    @staticmethod
    def can_handle(lambda_node: itir.Lambda) -> bool:
        raise NotImplementedError

    @final
    def build_stencil(self, node: itir.Lambda) -> GtirTaskletSubgraph:
        tlet_expr = self.visit(node.expr)

        params = [str(p.id) for p in node.params]
        assert len(self._input_connections) == len(params)

        outvar = "__out"
        tlet_code = f"{outvar} = {tlet_expr}"
        results = [outvar]
        self._output_connections.append((outvar, None))

        tlet_node: dace.tasklet = self._ctx.state.add_tasklet(
            f"{self._ctx.tasklet_name()}_lambda", set(params), set(results), tlet_code
        )

        subgraph = GtirTaskletSubgraph(tlet_node, self._input_connections, self._output_connections)

        return subgraph

    @final
    def visit_Lambda(self, node: itir.Lambda) -> GtirTaskletSubgraph:
        # This visitor class should never encounter `itir.Lambda` expressionsß
        raise RuntimeError("Unexpected 'itir.Lambda' node encountered by 'GtirTaskletCodegen'.")
