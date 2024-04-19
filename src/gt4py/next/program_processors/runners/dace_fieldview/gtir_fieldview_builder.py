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


from typing import Any, Callable, List, Optional

import dace

import gt4py.eve as eve
from gt4py.next.iterator import ir as itir

from .fieldview_dataflow import FieldviewRegion
from .gtir_tasklet_codegen import GtirTaskletCodegen as TaskletCodegen


class GtirFieldviewBuilder(eve.NodeVisitor):
    """Translates GTIR to Python code to be used as tasklet body.

    TODO: this class needs to be revisited in next commit.
    """

    _ctx: FieldviewRegion

    def __init__(self, sdfg: dace.SDFG, state: dace.SDFGState):
        self._ctx = FieldviewRegion(sdfg, state)

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

    @create_ctx
    def _make_fieldop(self, fun_node: itir.FunCall, fun_args: List[itir.Expr]) -> FieldviewRegion:
        ctx = self._ctx

        self.visit(fun_args)

        # create ordered list of input nodes
        input_arrays = [(name, ctx.sdfg.arrays[name]) for name in ctx.input_nodes]

        # TODO: define shape based on domain and dtype based on type inference
        shape = [10]
        dtype = dace.float64
        output_name, output_array = ctx.sdfg.add_array(
            ctx.var_name(), shape, dtype, transient=True, find_new_name=True
        )
        output_arrays = [(output_name, output_array)]

        assert len(fun_node.args) == 1
        self.visit(fun_node.args[0])

        # TODO: define map range based on domain
        map_ranges = dict(i="0:10")
        me, mx = ctx.state.add_map("fieldop", map_ranges)

        for (node, connector), (dname, _) in zip(self._ctx.input_connections, input_arrays):
            # TODO: define memlet subset based on domain
            src_node = self._ctx.node_mapping[dname]
            self._ctx.state.add_memlet_path(
                src_node, me, node, dst_conn=connector, memlet=dace.Memlet(data=dname, subset="i")
            )

        for (node, connector), (dname, _) in zip(self._ctx.output_connections, output_arrays):
            # TODO: define memlet subset based on domain
            dst_node = ctx.add_output_node(dname)
            self._ctx.state.add_memlet_path(
                node, mx, dst_node, src_conn=connector, memlet=dace.Memlet(data=dname, subset="i")
            )

        return ctx
