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


from typing import Optional, Sequence, Tuple

import dace

import gt4py.eve as eve
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.type_system import type_specifications as ts

from .gtir_fieldview_context import GtirFieldviewContext as FieldviewContext
from .gtir_tasklet_arithmetic import GtirTaskletArithmetic
from .gtir_tasklet_codegen import (
    GtirTaskletCodegen as TaskletCodegen,
    GtirTaskletSubgraph as TaskletSubgraph,
)


REGISTERED_TASKGENS: list[type[TaskletCodegen]] = [
    GtirTaskletArithmetic,
]


def _make_fieldop(
    ctx: FieldviewContext,
    tasklet_subgraph: TaskletSubgraph,
    domain: Sequence[Tuple[str, str, str]],
) -> None:
    # create ordered list of input nodes
    input_arrays = [(name, ctx.sdfg.arrays[name]) for name in ctx.input_nodes]
    assert len(tasklet_subgraph.input_connections) == len(input_arrays)

    # TODO: use type inference to determine the result type
    type_ = ts.ScalarKind.FLOAT64
    output_arrays: list[Tuple[str, dace.data.Array]] = []
    for _, field_type in tasklet_subgraph.output_connections:
        if field_type is None:
            field_dims = [Dimension(d) for d, _, _ in domain]
            field_type = ts.FieldType(field_dims, ts.ScalarType(type_))
        field_shape = [f"{ub} - {lb}" for d, lb, ub in domain if Dimension(d) in field_type.dims]
        output_name, output_array = ctx.add_local_storage(ctx.var_name(), field_type, field_shape)
        output_arrays.append((output_name, output_array))

    map_ranges = {f"i_{d}": f"{lb}:{ub}" for d, lb, ub in domain}
    me, mx = ctx.state.add_map("fieldop", map_ranges)

    for (connector, field_type), (dname, _) in zip(
        tasklet_subgraph.input_connections, input_arrays
    ):
        src_node = ctx.node_mapping[dname]
        if field_type is None:
            subset = ",".join([f"i_{d.value}" for d in ctx.field_types[dname].dims])
        else:
            raise NotImplementedError("Array subset on tasklet connector not supported.")
        ctx.state.add_memlet_path(
            src_node,
            me,
            tasklet_subgraph.node,
            dst_conn=connector,
            memlet=dace.Memlet(data=dname, subset=subset),
        )

    for (connector, field_type), (dname, _) in zip(
        tasklet_subgraph.output_connections, output_arrays
    ):
        dst_node = ctx.add_output_node(dname)
        if field_type is None:
            subset = ",".join([f"i_{d}" for d, _, _ in domain])
        else:
            raise NotImplementedError("Array subset on tasklet connector not supported.")
        ctx.state.add_memlet_path(
            tasklet_subgraph.node,
            mx,
            dst_node,
            src_conn=connector,
            memlet=dace.Memlet(data=dname, subset=subset),
        )


def _make_fieldop_domain(
    ctx: FieldviewContext, node: itir.FunCall
) -> Sequence[Tuple[str, str, str]]:
    assert cpm.is_call_to(node, ["cartesian_domain", "unstructured_domain"])

    domain = []
    translator = TaskletCodegen(ctx)

    for named_range in node.args:
        assert cpm.is_call_to(named_range, "named_range")
        assert len(named_range.args) == 3
        dimension = named_range.args[0]
        assert isinstance(dimension, itir.AxisLiteral)
        lower_bound = named_range.args[1]
        upper_bound = named_range.args[2]
        lb = translator.visit(lower_bound)
        ub = translator.visit(upper_bound)
        domain.append((dimension.value, lb, ub))

    return domain


class GtirFieldviewBuilder(eve.NodeVisitor):
    """Translates GTIR fieldview operator to some kind of map scope in DaCe SDFG."""

    _ctx: FieldviewContext

    def __init__(
        self, sdfg: dace.SDFG, state: dace.SDFGState, field_types: dict[str, ts.FieldType]
    ):
        self._ctx = FieldviewContext(sdfg, state, field_types.copy())

    def _get_tasklet_codegen(self, lambda_node: itir.Lambda) -> Optional[TaskletCodegen]:
        for taskgen in REGISTERED_TASKGENS:
            if taskgen.can_handle(lambda_node):
                return taskgen(self._ctx)
        return None

    def visit_FunCall(self, node: itir.FunCall) -> None:
        parent_ctx = self._ctx
        child_ctx = parent_ctx.clone()
        self._ctx = child_ctx

        if cpm.is_call_to(node.fun, "as_fieldop"):
            fun_node = node.fun
            assert len(fun_node.args) == 2
            # expect stencil (represented as a lambda function) as first argument
            assert isinstance(fun_node.args[0], itir.Lambda)
            taskgen = self._get_tasklet_codegen(fun_node.args[0])
            if not taskgen:
                raise NotImplementedError(f"Unsupported 'as_fieldop' node ({node}).")
            # the domain of the field operator is passed as second argument
            assert isinstance(fun_node.args[1], itir.FunCall)
            domain = _make_fieldop_domain(self._ctx, fun_node.args[1])

            self.visit(node.args)
            tasklet_subgraph = taskgen.visit(fun_node.args[0])
            _make_fieldop(self._ctx, tasklet_subgraph, domain)
        else:
            raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

        assert self._ctx == child_ctx
        parent_ctx.input_nodes.extend(child_ctx.output_nodes)
        self._ctx = parent_ctx

    def visit_SymRef(self, node: itir.SymRef) -> None:
        dname = str(node.id)
        self._ctx.add_input_node(dname)

    def write_to(self, target_expr: itir.Expr, domain_expr: itir.Expr) -> None:
        assert len(self._ctx.output_nodes) == 0

        # TODO: add support for tuple return
        assert len(self._ctx.input_nodes) == 1
        assert isinstance(target_expr, itir.SymRef)
        self._ctx.add_output_node(target_expr.id)

        assert isinstance(domain_expr, itir.FunCall)
        domain = _make_fieldop_domain(self._ctx, domain_expr)
        write_subset = ",".join(f"{lb}:{ub}" for _, lb, ub in domain)

        for tasklet_node, target_node in zip(self._ctx.input_nodes, self._ctx.output_nodes):
            target_array = self._ctx.sdfg.arrays[target_node]
            target_array.transient = False

            self._ctx.state.add_nedge(
                self._ctx.node_mapping[tasklet_node],
                self._ctx.node_mapping[target_node],
                dace.Memlet(data=target_node, subset=write_subset),
            )
