# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple

import dace
import numpy as np
from dace import SDFG, InterstateEdge, SDFGState
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.nodes import AccessNode
from dace.subsets import Subset

import gtc.common as common
import gtc.oir as oir
from eve import field
from eve.visitors import NodeVisitor
from gtc.dace.nodes import VerticalLoopLibraryNode
from gtc.passes.oir_optimizations.utils import AccessCollector


class OirToSDFGVisitor(NodeVisitor):
    class ForwardContext:
        sdfg: SDFG
        state: SDFGState

        recent_write_accesses: Dict[str, oir.IntervalMapping]
        recent_read_accesses: Dict[str, oir.IntervalMapping]
        accesses: Dict[str, List[AccessNode]]
        delete_candidates: List[MultiConnectorEdge]

        def __init__(self, stencil: oir.Stencil):
            self.sdfg = SDFG(stencil.name)
            self.state = self.sdfg.add_state(stencil.name + "_state", is_start_state=True)
            self.recent_write_accesses = dict()
            self.recent_read_accesses = dict()
            self.accesses = dict()
            self.delete_candidates = list()

    class BackwardContext:
        sdfg: SDFG
        state: SDFGState
        next_write_accesses: Dict[str, oir.IntervalMapping]
        delete_candidates: List[MultiConnectorEdge]

        def __init__(self, forward_context: "OirToSDFGVisitor.ForwardContext"):
            self.sdfg = forward_context.sdfg
            self.state = forward_context.state
            self.next_write_accesses = dict()
            self.delete_candidates = forward_context.delete_candidates

    def _add_read_edges_forward(self, ln, access_collection, context):
        vl = ln._oir_node
        # for reads: add edge from last access node with an overlaping write to LN
        for name in access_collection.read_fields():
            most_recent_write = None
            for offset in access_collection.read_offsets()[name]:
                interval = vl.interval.shift(offset[2])
                if (
                    name in context.recent_write_accesses
                    and len(context.recent_write_accesses[name][interval]) > 0
                ):
                    if most_recent_write is None:
                        most_recent_write = max(context.recent_write_accesses[name][interval])
                    most_recent_write = max(
                        most_recent_write, *context.recent_write_accesses[name][interval]
                    )

                if name not in context.accesses:
                    access_node = context.state.add_read(name)
                    context.accesses[name] = [access_node]
                    most_recent_write = 0
                elif name not in context.recent_write_accesses:
                    # if the field has only been read so far, all reads are from the original state
                    assert len(context.accesses[name]) == 1
                    access_node = context.accesses[name][0]
                    most_recent_write = 0
                else:
                    if most_recent_write is not None:
                        access_node = context.accesses[name][most_recent_write]
                    else:
                        # the node was written but in different interval. we can read the original state.
                        for i, acc in enumerate(context.accesses[name]):
                            if (
                                acc.access != dace.AccessType.WriteOnly
                                and context.state.in_degree(acc) == 0
                            ):
                                access_node = acc
                                most_recent_write = i
                                break
                        else:
                            # no read-only node was found, so we have to create the access node
                            # for the original state.
                            access_node = context.state.add_read(name)
                            context.accesses[name].append(access_node)
                            most_recent_write = len(context.accesses[name]) - 1

            if access_node.access == dace.AccessType.WriteOnly:
                access_node.access = dace.AccessType.ReadWrite
            ln.add_in_connector("IN_" + name)
            # TODO fix memlet
            context.state.add_edge(access_node, None, ln, "IN_" + name, dace.Memlet())
            if name not in context.recent_read_accesses:
                context.recent_read_accesses[name] = oir.IntervalMapping()
            context.recent_read_accesses[name][interval] = most_recent_write

    def _add_vertical_loop_forward(self, vl, context: "OirToSDFGVisitor.ForwardContext"):
        ln = VerticalLoopLibraryNode(oir_node=vl)
        context.state.add_node(ln)
        access_collection = AccessCollector.apply(vl)
        self._add_read_edges_forward(ln, access_collection, context)

        # for writes: add empty edge from last overlaping write to current LN (check if needed later)
        for name in access_collection.write_fields():
            if name in context.recent_write_accesses:
                writes = context.recent_write_accesses[name][vl.interval]
                if len(writes) > 0:
                    most_recent_write = context.accesses[name][max(writes)]
                    edge = context.state.add_edge(most_recent_write, None, ln, None, dace.Memlet())
                    context.delete_candidates.append(edge)

        # for writes: add edge to existing last access if None of its reads or writes overlap with the write and
        #   otherwise, add new access node
        for name in access_collection.write_fields():

            recent_read_node = None
            recent_write_node = None
            candidate_node = None
            if name in context.accesses:
                candidate_node = context.accesses[name][-1]
                candidate_node_idx = len(context.accesses[name]) - 1
                if name in context.recent_write_accesses:
                    writes = context.recent_write_accesses[name][vl.interval]
                    if len(writes) > 0:
                        recent_write_node = context.accesses[name][max(writes)]

                if name in context.recent_read_accesses:
                    reads = context.recent_read_accesses[name][vl.interval]
                    if len(reads) > 0:
                        recent_read_node = context.accesses[name][max(reads)]

            if (
                candidate_node is None
                or candidate_node is recent_read_node
                or candidate_node is recent_write_node
            ):
                candidate_node = context.state.add_write(name)
                if name not in context.accesses:
                    context.accesses[name] = []
                context.accesses[name].append(candidate_node)
                candidate_node_idx = len(context.accesses[name]) - 1

            ln.add_out_connector("OUT_" + name)
            # TODO fix memlet
            context.state.add_edge(ln, "OUT_" + name, candidate_node, None, dace.Memlet())
            if name not in context.recent_write_accesses:
                context.recent_write_accesses[name] = oir.IntervalMapping()
            context.recent_write_accesses[name][vl.interval] = candidate_node_idx
            if candidate_node.access == dace.AccessType.ReadOnly:
                candidate_node.access = dace.AccessType.ReadWrite

    def visit_Decl(self, node, *, sdfg):
        dtype = dace.dtypes.typeclass(np.dtype(common.data_type_to_typestr(node.dtype)).name)
        if isinstance(node, oir.Temporary):
            sdfg.add_transient(name=node.name, dtype=dtype, shape=())
        elif isinstance(node, oir.FieldDecl):
            sdfg.add_array(name=node.name, dtype=dtype, shape=())
        else:
            assert isinstance(node, oir.ScalarDecl)
            sdfg.add_scalar(name=node.name, dtype=dtype)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> SDFG:

        context = self.ForwardContext(node)
        # add arrays
        self.generic_visit(node, sdfg=context.sdfg)

        for vl in node.vertical_loops:
            self._add_vertical_loop_forward(vl, context)
        # for vl in reversed(node.vertical_loops):
        #     self._add_vertical_loop_backward()

        import networkx as nx

        for edge in context.delete_candidates:
            assert nx.has_path(context.state.nx, edge.src, edge.dst)
            context.state.remove_edge(edge)
            if not nx.has_path(context.state.nx, edge.src, edge.dst):
                context.state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

        context.sdfg.validate()
        return context.sdfg
