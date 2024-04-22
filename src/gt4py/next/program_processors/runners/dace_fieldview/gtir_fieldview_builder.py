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


from typing import Any, Callable, List, Optional, Sequence, Tuple

import dace

import gt4py.eve as eve
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type
from gt4py.next.type_system import type_specifications as ts

from .fieldview_dataflow import FieldviewRegion
from .gtir_tasklet_codegen import GtirTaskletCodegen as TaskletCodegen


class GtirFieldviewBuilder(eve.NodeVisitor):
    """Translates GTIR to Python code to be used as tasklet body.

    TODO: this class needs to be revisited in next commit.
    """

    _ctx: FieldviewRegion
    _field_types: dict[str, ts.FieldType]

    def __init__(
        self, sdfg: dace.SDFG, state: dace.SDFGState, field_types: dict[str, ts.FieldType]
    ):
        self._ctx = FieldviewRegion(sdfg, state)
        self._field_types = field_types.copy()

    @staticmethod
    def create_ctx(func: Callable) -> Callable:
        def newf(
            self: "GtirFieldviewBuilder", *args: Any, **kwargs: Optional[Any]
        ) -> FieldviewRegion:
            prev_ctx = self._ctx
            new_ctx = prev_ctx.clone()
            self._ctx = new_ctx

            child_ctx = func(self, *args, **kwargs)

            assert self._ctx == new_ctx
            self._ctx = prev_ctx

            return child_ctx

        return newf

    def visit_FunCall(self, node: itir.FunCall) -> None:
        if isinstance(node.fun, itir.FunCall) and isinstance(node.fun.fun, itir.SymRef):
            if node.fun.fun.id == "as_fieldop":
                child_ctx = self._make_fieldop(node.fun, node.args)
                assert child_ctx.state == self._ctx.state
                self._ctx.input_nodes.extend(child_ctx.output_nodes)
            else:
                raise NotImplementedError(f"Unexpected 'FunCall' with function {node.fun.fun.id}.")
        else:
            raise NotImplementedError(f"Unexpected 'FunCall' with type {type(node.fun)}.")

    def visit_Lambda(self, node: itir.Lambda) -> None:
        params = [str(p.id) for p in node.params]
        results = []

        tlet_code_lines = []
        expr_list = TaskletCodegen.apply(node.expr)

        for i, expr in enumerate(expr_list):
            outvar = f"__out_{i}"
            tlet_code_lines.append(outvar + " = " + expr)
            results.append(outvar)
        tlet_code = "\n".join(tlet_code_lines)

        tlet_node: dace.tasklet = self._ctx.state.add_tasklet(
            f"{self._ctx.state.label}_lambda", set(params), set(results), tlet_code
        )

        # TODO: distinguish between external and local connections (now assume all external)
        for inpvar in params:
            self._ctx.input_connections.append((tlet_node, inpvar))
        for outvar in results:
            self._ctx.output_connections.append((tlet_node, outvar))

    def visit_SymRef(self, node: itir.SymRef) -> None:
        dname = str(node.id)
        self._ctx.add_input_node(dname)

    def write_to(self, node: itir.Expr) -> None:
        result_nodes = self._ctx.input_nodes.copy()
        self._ctx = self._ctx.clone()
        self.visit(node)
        # the target expression should only produce a set of access nodes (no tasklets, no output nodes)
        assert len(self._ctx.output_nodes) == 0
        output_nodes = self._ctx.input_nodes

        assert len(result_nodes) == len(output_nodes)
        for tasklet_node, target_node in zip(result_nodes, output_nodes):
            target_array = self._ctx.sdfg.arrays[target_node]
            target_array.transient = False

            # TODO: visit statement domain to define the memlet subset
            self._ctx.state.add_nedge(
                self._ctx.node_mapping[tasklet_node],
                self._ctx.node_mapping[target_node],
                dace.Memlet.from_array(target_node, target_array),
            )

    def _visit_domain(self, node: itir.FunCall) -> Sequence[Tuple[str, str, str]]:
        assert isinstance(node.fun, itir.SymRef)
        assert node.fun.id == "cartesian_domain" or node.fun.id == "unstructured_domain"

        domain = []
        translator = TaskletCodegen()

        for named_range in node.args:
            assert isinstance(named_range, itir.FunCall)
            assert isinstance(named_range.fun, itir.SymRef)
            assert len(named_range.args) == 3
            dimension = named_range.args[0]
            assert isinstance(dimension, itir.AxisLiteral)
            lower_bound = named_range.args[1]
            upper_bound = named_range.args[2]
            lb = translator.visit(lower_bound)
            ub = translator.visit(upper_bound)
            domain.append((dimension.value, lb, ub))

        return domain

    @create_ctx
    def _make_fieldop(self, fun_node: itir.FunCall, fun_args: List[itir.Expr]) -> FieldviewRegion:
        ctx = self._ctx

        self.visit(fun_args)

        # create ordered list of input nodes
        input_arrays = [(name, ctx.sdfg.arrays[name]) for name in ctx.input_nodes]

        assert len(fun_node.args) == 2
        # expect stencil (represented as a lambda function) as first argument
        self.visit(fun_node.args[0])
        # the domain of the field operator is passed as second argument
        assert isinstance(fun_node.args[1], itir.FunCall)
        domain = self._visit_domain(fun_node.args[1])
        map_ranges = {f"i_{d}": f"{lb}:{ub}" for d, lb, ub in domain}
        me, mx = ctx.state.add_map("fieldop", map_ranges)

        # TODO: use type inference to determine the result type
        type_ = ts.ScalarKind.FLOAT64
        dtype = as_dace_type(type_)
        shape = [f"{ub} - {lb}" for _, lb, ub in domain]
        output_name, output_array = ctx.sdfg.add_array(
            ctx.var_name(), shape, dtype, transient=True, find_new_name=True
        )
        output_arrays = [(output_name, output_array)]
        self._field_types[output_name] = ts.FieldType(
            dims=[Dimension(d) for d, _, _ in domain], dtype=ts.ScalarType(type_)
        )

        for (node, connector), (dname, _) in zip(self._ctx.input_connections, input_arrays):
            src_node = self._ctx.node_mapping[dname]
            subset = ",".join([f"i_{d.value}" for d in self._field_types[dname].dims])
            self._ctx.state.add_memlet_path(
                src_node,
                me,
                node,
                dst_conn=connector,
                memlet=dace.Memlet(data=dname, subset=subset),
            )

        for (node, connector), (dname, _) in zip(self._ctx.output_connections, output_arrays):
            dst_node = ctx.add_output_node(dname)
            subset = ",".join([f"i_{d}" for d, _, _ in domain])
            self._ctx.state.add_memlet_path(
                node,
                mx,
                dst_node,
                src_conn=connector,
                memlet=dace.Memlet(data=dname, subset=subset),
            )

        return ctx
