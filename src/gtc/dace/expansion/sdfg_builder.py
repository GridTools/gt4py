# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
from dataclasses import dataclass
from typing import Any, ChainMap, Dict, List, Optional, Set, Tuple

import dace
import dace.data
import dace.library
import dace.subsets

import eve
from gtc import daceir as dcir
from gtc.dace.expansion.tasklet_codegen import TaskletCodegen
from gtc.dace.expansion.utils import get_dace_debuginfo
from gtc.dace.symbol_utils import data_type_to_dace_typeclass
from gtc.dace.utils import make_dace_subset


class StencilComputationSDFGBuilder(eve.VisitorWithSymbolTableTrait):
    @dataclass
    class NodeContext:
        input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]
        output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]

    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        state: dace.SDFGState
        state_stack: List[dace.SDFGState] = dataclasses.field(default_factory=list)

        def add_state(self):
            new_state = self.sdfg.add_state()
            for edge in self.sdfg.out_edges(self.state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(new_state, edge.dst, edge.data)
            self.sdfg.add_edge(self.state, new_state, dace.InterstateEdge())
            self.state = new_state
            return self

        def add_loop(self, index_range: dcir.Range):
            loop_state = self.sdfg.add_state()
            after_state = self.sdfg.add_state()
            for edge in self.sdfg.out_edges(self.state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(
                    after_state,
                    edge.dst,
                    edge.data,
                )

            assert isinstance(index_range.interval, dcir.DomainInterval)
            if index_range.stride < 0:
                initialize_expr = f"{index_range.interval.end} - 1"
                end_expr = f"{index_range.interval.start} - 1"
            else:
                initialize_expr = str(index_range.interval.start)
                end_expr = str(index_range.interval.end)
            comparison_op = "<" if index_range.stride > 0 else ">"
            condition_expr = f"{index_range.var} {comparison_op} {end_expr}"
            _, _, after_state = self.sdfg.add_loop(
                before_state=self.state,
                loop_state=loop_state,
                after_state=after_state,
                loop_var=index_range.var,
                initialize_expr=initialize_expr,
                condition_expr=condition_expr,
                increment_expr=f"{index_range.var}+({index_range.stride})",
            )
            if index_range.var not in self.sdfg.symbols:
                self.sdfg.add_symbol(index_range.var, stype=dace.int32)

            self.state_stack.append(after_state)
            self.state = loop_state
            return self

        def pop_loop(self):
            self.state = self.state_stack[-1]
            del self.state_stack[-1]

    def visit_Memlet(
        self,
        node: dcir.Memlet,
        *,
        scope_node: dcir.ComputationNode,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        connector_prefix="",
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
    ) -> None:
        field_decl = symtable[node.field]
        assert isinstance(field_decl, dcir.FieldDecl)
        memlet = dace.Memlet(
            node.field,
            subset=make_dace_subset(field_decl.access_info, node.access_info, field_decl.data_dims),
            dynamic=field_decl.is_dynamic,
        )
        if node.is_read:
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[memlet.data],
                scope_node,
                connector_prefix + node.connector,
                memlet,
            )
        if node.is_write:
            sdfg_ctx.state.add_edge(
                scope_node,
                connector_prefix + node.connector,
                *node_ctx.output_node_and_conns[memlet.data],
                memlet,
            )

    @classmethod
    def _add_empty_edges(
        cls,
        entry_node: dace.nodes.Node,
        exit_node: dace.nodes.Node,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ) -> None:
        if not sdfg_ctx.state.in_degree(entry_node) and None in node_ctx.input_node_and_conns:
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[None], entry_node, None, dace.Memlet()
            )
        if not sdfg_ctx.state.out_degree(exit_node) and None in node_ctx.output_node_and_conns:
            sdfg_ctx.state.add_edge(
                exit_node, None, *node_ctx.output_node_and_conns[None], dace.Memlet()
            )

    def visit_Tasklet(
        self,
        node: dcir.Tasklet,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        **kwargs,
    ) -> None:
        code = TaskletCodegen.apply_codegen(
            node,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            sdfg_ctx=sdfg_ctx,
            symtable=symtable,
        )

        tasklet = sdfg_ctx.state.add_tasklet(
            name=f"{sdfg_ctx.sdfg.label}_Tasklet",
            code=code,
            inputs=set(memlet.connector for memlet in node.read_memlets),
            outputs=set(memlet.connector for memlet in node.write_memlets),
            debuginfo=get_dace_debuginfo(node),
        )

        self.visit(
            node.read_memlets,
            scope_node=tasklet,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            symtable=symtable,
            **kwargs,
        )
        self.visit(
            node.write_memlets,
            scope_node=tasklet,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            symtable=symtable,
            **kwargs,
        )
        StencilComputationSDFGBuilder._add_empty_edges(
            tasklet, tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
        )

    def visit_Range(self, node: dcir.Range, **kwargs) -> Dict[str, str]:
        start, end = node.interval.to_dace_symbolic()
        return {node.var: str(dace.subsets.Range([(start, end - 1, node.stride)]))}

    def visit_DomainMap(
        self,
        node: dcir.DomainMap,
        *,
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ) -> None:
        ndranges = {
            k: v
            for index_range in node.index_ranges
            for k, v in self.visit(index_range, **kwargs).items()
        }
        name = sdfg_ctx.sdfg.label + "".join(ndranges.keys()) + "_map"
        map_entry, map_exit = sdfg_ctx.state.add_map(
            name=name,
            ndrange=ndranges,
            schedule=node.schedule.to_dace_schedule(),
            debuginfo=get_dace_debuginfo(node),
        )

        for scope_node in node.computations:
            input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = {}
            output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = {}
            for field in set(memlet.field for memlet in scope_node.read_memlets):
                map_entry.add_in_connector("IN_" + field)
                map_entry.add_out_connector("OUT_" + field)
                input_node_and_conns[field] = (map_entry, "OUT_" + field)
            for field in set(memlet.field for memlet in scope_node.write_memlets):
                map_exit.add_in_connector("IN_" + field)
                map_exit.add_out_connector("OUT_" + field)
                output_node_and_conns[field] = (map_exit, "IN_" + field)
            if not input_node_and_conns:
                input_node_and_conns[None] = (map_entry, None)
            if not output_node_and_conns:
                output_node_and_conns[None] = (map_exit, None)
            inner_node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=input_node_and_conns,
                output_node_and_conns=output_node_and_conns,
            )
            self.visit(scope_node, sdfg_ctx=sdfg_ctx, node_ctx=inner_node_ctx, **kwargs)

        self.visit(
            node.read_memlets,
            scope_node=map_entry,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            connector_prefix="IN_",
            **kwargs,
        )
        self.visit(
            node.write_memlets,
            scope_node=map_exit,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            connector_prefix="OUT_",
            **kwargs,
        )
        StencilComputationSDFGBuilder._add_empty_edges(
            map_entry, map_exit, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
        )

    def visit_DomainLoop(
        self,
        node: dcir.DomainLoop,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ) -> None:
        sdfg_ctx = sdfg_ctx.add_loop(node.index_range)
        self.visit(node.loop_states, sdfg_ctx=sdfg_ctx, **kwargs)
        sdfg_ctx.pop_loop()

    def visit_ComputationState(
        self,
        node: dcir.ComputationState,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ) -> None:

        sdfg_ctx.add_state()
        read_acc_and_conn: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = {}
        write_acc_and_conn: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = {}
        for computation in node.computations:
            assert isinstance(computation, dcir.ComputationNode)
            for memlet in computation.read_memlets:
                if memlet.field not in read_acc_and_conn:
                    read_acc_and_conn[memlet.field] = (
                        sdfg_ctx.state.add_access(memlet.field, debuginfo=dace.DebugInfo(0)),
                        None,
                    )
            for memlet in computation.write_memlets:
                if memlet.field not in write_acc_and_conn:
                    write_acc_and_conn[memlet.field] = (
                        sdfg_ctx.state.add_access(memlet.field, debuginfo=dace.DebugInfo(0)),
                        None,
                    )
            node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=read_acc_and_conn, output_node_and_conns=write_acc_and_conn
            )
            self.visit(computation, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, **kwargs)

    def visit_FieldDecl(
        self,
        node: dcir.FieldDecl,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        non_transients: Set[eve.SymbolRef],
        **kwargs,
    ) -> None:
        assert len(node.strides) == len(node.shape)
        sdfg_ctx.sdfg.add_array(
            node.name,
            shape=node.shape,
            strides=[dace.symbolic.pystr_to_symbolic(s) for s in node.strides],
            dtype=data_type_to_dace_typeclass(node.dtype),
            storage=node.storage.to_dace_storage(),
            transient=node.name not in non_transients,
            debuginfo=dace.DebugInfo(0),
        )

    def visit_SymbolDecl(
        self,
        node: dcir.SymbolDecl,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ) -> None:
        if node.name not in sdfg_ctx.sdfg.symbols:
            sdfg_ctx.sdfg.add_symbol(node.name, stype=data_type_to_dace_typeclass(node.dtype))

    def visit_NestedSDFG(
        self,
        node: dcir.NestedSDFG,
        *,
        sdfg_ctx: Optional["StencilComputationSDFGBuilder.SDFGContext"] = None,
        node_ctx: Optional["StencilComputationSDFGBuilder.NodeContext"] = None,
        symtable: ChainMap[eve.SymbolRef, Any],
        **kwargs,
    ) -> dace.nodes.NestedSDFG:

        sdfg = dace.SDFG(node.label)
        inner_sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg, state=sdfg.add_state(is_start_state=True)
        )
        self.visit(
            node.field_decls,
            sdfg_ctx=inner_sdfg_ctx,
            non_transients={memlet.connector for memlet in node.read_memlets + node.write_memlets},
            **kwargs,
        )
        self.visit(node.symbol_decls, sdfg_ctx=inner_sdfg_ctx, **kwargs)
        symbol_mapping = {decl.name: decl.to_dace_symbol() for decl in node.symbol_decls}

        for computation_state in node.states:
            self.visit(computation_state, sdfg_ctx=inner_sdfg_ctx, symtable=symtable, **kwargs)

        if sdfg_ctx is not None and node_ctx is not None:
            nsdfg = sdfg_ctx.state.add_nested_sdfg(
                sdfg=sdfg,
                parent=None,
                inputs=node.input_connectors,
                outputs=node.output_connectors,
                symbol_mapping=symbol_mapping,
                debuginfo=dace.DebugInfo(0),
            )
            self.visit(
                node.read_memlets,
                scope_node=nsdfg,
                sdfg_ctx=sdfg_ctx,
                node_ctx=node_ctx,
                symtable=symtable.parents,
                **kwargs,
            )
            self.visit(
                node.write_memlets,
                scope_node=nsdfg,
                sdfg_ctx=sdfg_ctx,
                node_ctx=node_ctx,
                symtable=symtable.parents,
                **kwargs,
            )
            StencilComputationSDFGBuilder._add_empty_edges(
                nsdfg, nsdfg, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
            )
        else:
            nsdfg = dace.nodes.NestedSDFG(
                label=sdfg.label,
                sdfg=sdfg,
                inputs={memlet.connector for memlet in node.read_memlets},
                outputs={memlet.connector for memlet in node.write_memlets},
                symbol_mapping=symbol_mapping,
            )

        return nsdfg
