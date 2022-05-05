import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import dace
import dace.data
import dace.library
import dace.subsets
import numpy as np

import gtc.common as common
from eve import NodeVisitor
from gtc import daceir as dcir
from gtc.dace.expansion.tasklet_codegen import TaskletCodegen
from gtc.dace.expansion.utils import get_dace_debuginfo
from gtc.dace.utils import get_axis_bound_str, make_subset_str


class StencilComputationSDFGBuilder(NodeVisitor):
    @dataclass
    class NodeContext:
        input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]
        output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]

    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        state: dace.SDFGState
        field_decls: Dict[str, dcir.FieldDecl] = dataclasses.field(default_factory=dict)
        state_stack: List[dace.SDFGState] = dataclasses.field(default_factory=list)

        def add_state(self):
            old_state = self.state
            state = self.sdfg.add_state()
            for edge in self.sdfg.out_edges(old_state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(
                    state,
                    edge.dst,
                    edge.data,
                )
            self.sdfg.add_edge(old_state, state, dace.InterstateEdge())
            self.state = state
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
            comparison_op = "<" if index_range.stride > 0 else ">"
            condition_expr = f"{index_range.var} {comparison_op} {index_range.end}"
            _, _, after_state = self.sdfg.add_loop(
                before_state=self.state,
                loop_state=loop_state,
                after_state=after_state,
                loop_var=index_range.var,
                initialize_expr=str(index_range.start),
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
    ):
        field_decl = sdfg_ctx.field_decls[node.field]
        memlet = dace.Memlet.simple(
            node.field,
            subset_str=make_subset_str(
                field_decl.access_info, node.access_info, field_decl.data_dims
            ),
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
    ):

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
    ):
        code = TaskletCodegen.apply(
            node,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            sdfg_ctx=sdfg_ctx,
        )

        tasklet = sdfg_ctx.state.add_tasklet(
            name=f"{sdfg_ctx.sdfg.label}_Tasklet",
            code=code,
            inputs=set(memlet.connector for memlet in node.read_memlets),
            outputs=set(memlet.connector for memlet in node.write_memlets),
            debuginfo=get_dace_debuginfo(node),
        )

        self.visit(node.read_memlets, scope_node=tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)
        self.visit(node.write_memlets, scope_node=tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)
        StencilComputationSDFGBuilder._add_empty_edges(
            tasklet, tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
        )

    def visit_Range(self, node: dcir.Range, **kwargs):
        if isinstance(node.start, dcir.AxisBound):
            start = get_axis_bound_str(node.start, node.start.axis.domain_symbol())
        else:
            start = str(node.start)
        if isinstance(node.end, dcir.AxisBound):
            end = get_axis_bound_str(node.end, node.end.axis.domain_symbol())
        else:
            end = str(node.end)
        return {node.var: f"{start}:{end}:{node.stride}"}

    def visit_DomainMap(
        self,
        node: dcir.DomainMap,
        *,
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
    ):

        ndranges = {
            k: v for index_range in node.index_ranges for k, v in self.visit(index_range).items()
        }
        name = sdfg_ctx.sdfg.label + "".join(ndranges.keys()) + "_map"
        map_entry, map_exit = sdfg_ctx.state.add_map(
            name=name,
            ndrange=ndranges,
            schedule=node.schedule.to_dace_schedule(),
            debuginfo=get_dace_debuginfo(node),
        )

        for scope_node in node.computations:
            input_node_and_conns: Dict[
                Optional[str], Tuple[dace.nodes.Node, Optional[str]]
            ] = dict()
            output_node_and_conns: Dict[
                Optional[str], Tuple[dace.nodes.Node, Optional[str]]
            ] = dict()
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
            self.visit(scope_node, sdfg_ctx=sdfg_ctx, node_ctx=inner_node_ctx)

        self.visit(
            node.read_memlets,
            scope_node=map_entry,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            connector_prefix="IN_",
        )
        self.visit(
            node.write_memlets,
            scope_node=map_exit,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            connector_prefix="OUT_",
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
    ):
        sdfg_ctx = sdfg_ctx.add_loop(node.index_range)
        self.visit(node.loop_states, sdfg_ctx=sdfg_ctx, **kwargs)
        sdfg_ctx.pop_loop()

    def visit_ComputationState(
        self,
        node: dcir.ComputationState,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ):

        sdfg_ctx = sdfg_ctx.add_state()
        read_acc_and_conn: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = dict()
        write_acc_and_conn: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = dict()
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
                input_node_and_conns=read_acc_and_conn,
                output_node_and_conns=write_acc_and_conn,
            )
            self.visit(computation, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, **kwargs)

    def visit_StateMachine(
        self,
        node: dcir.StateMachine,
        *,
        sdfg_ctx: Optional["StencilComputationSDFGBuilder.SDFGContext"] = None,
        node_ctx: Optional["StencilComputationSDFGBuilder.NodeContext"] = None,
    ):

        sdfg = dace.SDFG(node.label)
        state = sdfg.add_state()
        symbol_mapping = {}
        for axis in dcir.Axis.dims_3d():
            sdfg.add_symbol(axis.domain_symbol(), stype=dace.int32)
            symbol_mapping[axis.domain_symbol()] = dace.symbol(
                axis.domain_symbol(), dtype=dace.int32
            )
        if sdfg_ctx is not None and node_ctx is not None:
            nsdfg = sdfg_ctx.state.add_nested_sdfg(
                sdfg=sdfg,
                parent=None,
                inputs=node.input_connectors,
                outputs=node.output_connectors,
                symbol_mapping=symbol_mapping,
                debuginfo=dace.DebugInfo(0),
            )
            self.visit(node.read_memlets, scope_node=nsdfg, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)
            self.visit(node.write_memlets, scope_node=nsdfg, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)
            StencilComputationSDFGBuilder._add_empty_edges(
                nsdfg, nsdfg, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
            )
        else:
            nsdfg = dace.nodes.NestedSDFG(
                label=sdfg.label,
                sdfg=sdfg,
                inputs=set(memlet.connector for memlet in node.read_memlets),
                outputs=set(memlet.connector for memlet in node.write_memlets),
                symbol_mapping=symbol_mapping,
            )

        inner_sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg,
            state=state,
            field_decls=node.field_decls,
        )

        for name, decl in node.field_decls.items():
            non_transients = set(
                memlet.connector for memlet in node.read_memlets + node.write_memlets
            )
            assert len(decl.shape) == len(decl.strides)
            inner_sdfg_ctx.sdfg.add_array(
                name,
                shape=decl.shape,
                strides=[dace.symbolic.pystr_to_symbolic(s) for s in decl.strides],
                dtype=np.dtype(common.data_type_to_typestr(decl.dtype)).type,
                storage=decl.storage.to_dace_storage(),
                transient=name not in non_transients,
                debuginfo=dace.DebugInfo(0),
            )
        for symbol, dtype in node.symbols.items():
            if symbol not in inner_sdfg_ctx.sdfg.symbols:
                inner_sdfg_ctx.sdfg.add_symbol(
                    symbol,
                    stype=dace.typeclass(np.dtype(common.data_type_to_typestr(dtype)).name),
                )
            nsdfg.symbol_mapping[symbol] = dace.symbol(
                symbol,
                dtype=dace.typeclass(np.dtype(common.data_type_to_typestr(dtype)).name),
            )

        for computation_state in node.states:
            self.visit(computation_state, sdfg_ctx=inner_sdfg_ctx)
        for sym in nsdfg.sdfg.free_symbols:
            if sym not in nsdfg.sdfg.symbols:
                nsdfg.sdfg.add_symbol(sym, stype=dace.int32)
            nsdfg.symbol_mapping.setdefault(str(sym), dace.symbol(sym, dtype=dace.int32))

        return nsdfg
