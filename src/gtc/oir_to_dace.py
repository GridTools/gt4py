# -*- coding: utf-8 -*-
from typing import Any, Dict, List

import dace
import numpy as np
from dace import SDFG, SDFGState
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.nodes import AccessNode

import gtc.common as common
import gtc.oir as oir
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

    def _access_space_to_subset(self, access_space, access_k_interval, origin):
        i_subset = "{start}:I{end:+d}".format(
            start=origin[0] + access_space.i_interval.start.offset,
            end=origin[0] + access_space.i_interval.end.offset,
        )
        j_subset = "{start}:J{end:+d}".format(
            start=origin[1] + access_space.j_interval.start.offset,
            end=origin[1] + access_space.j_interval.end.offset,
        )
        k_subset = (
            "K{:+d}:" if access_k_interval.start.level == common.LevelMarker.END else "{:d}:"
        ) + ("K{:+d}" if access_k_interval.end.level == common.LevelMarker.END else "{:d}")
        k_subset = k_subset.format(access_k_interval.start.offset, access_k_interval.end.offset)
        return f"{i_subset},{j_subset},{k_subset}"

    def _access_space_to_shape(self, access_space: oir.CartesianIterationSpace):
        i_shape = f"I+{access_space.i_interval.end.offset-access_space.i_interval.start.offset}"
        j_shape = f"J+{access_space.j_interval.end.offset-access_space.j_interval.start.offset}"
        return (i_shape, j_shape, "K")

    def _get_input_subset(self, node: oir.VerticalLoop, name, origin):
        access_space = oir.CartesianIterationSpace.domain()
        access_k_interval = node.sections[0].interval

        for section in node.sections:
            interval = section.interval
            for he in section.horizontal_executions:
                iteration_space = he.iteration_space
                he_access_space = oir.CartesianIterationSpace.domain()
                assert he.iteration_space is not None
                access_collection = AccessCollector.apply(he)
                if name in access_collection.read_fields():
                    for offset in access_collection.read_offsets()[name]:
                        he_access_space = he_access_space | oir.CartesianIterationSpace.from_offset(
                            common.CartesianOffset(i=offset[0], j=offset[1], k=offset[2])
                        )
                        k_interval = interval.shift(offset[2])
                        access_k_interval = oir.Interval(
                            start=min(access_k_interval.start, k_interval.start),
                            end=max(access_k_interval.end, k_interval.end),
                        )
                    access_space = access_space | (iteration_space.compose(he_access_space))

        return self._access_space_to_subset(access_space, access_k_interval, origin)

    def _get_output_subset(self, node, name, origin):
        access_space = oir.CartesianIterationSpace.domain()
        access_k_interval = node.sections[0].interval

        for section in node.sections:
            interval = section.interval
            for he in section.horizontal_executions:
                iteration_space = he.iteration_space
                he_access_space = oir.CartesianIterationSpace.domain()
                assert he.iteration_space is not None
                access_collection = AccessCollector.apply(he)
                if name in access_collection.write_fields():
                    access_k_interval = oir.Interval(
                        start=min(access_k_interval.start, interval.start),
                        end=max(access_k_interval.end, interval.end),
                    )
                    access_space = access_space | (iteration_space.compose(he_access_space))

        return self._access_space_to_subset(access_space, access_k_interval, origin)

    def _add_write_edges_forward(self, ln, access_collections, context, origins):
        write_fields = set(
            name for collection in access_collections.values() for name in collection.write_fields()
        )
        vl = ln.oir_node
        for name in write_fields:

            recent_read_node = None
            recent_write_node = None
            candidate_node = None
            if name in context.accesses:
                candidate_node = context.accesses[name][-1]
                candidate_node_idx = len(context.accesses[name]) - 1
                if name in context.recent_write_accesses:
                    writes = list(
                        w
                        for section in vl.sections
                        for w in context.recent_write_accesses[name][section.interval]
                    )
                    if len(writes) > 0:
                        recent_write_node = context.accesses[name][max(writes)]

                if name in context.recent_read_accesses:
                    reads = list(
                        r
                        for section in vl.sections
                        for r in context.recent_read_accesses[name][section.interval]
                    )
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

            subset = self._get_output_subset(vl, name, origins[name])
            context.state.add_edge(
                ln,
                "OUT_" + name,
                candidate_node,
                None,
                dace.Memlet.simple(data=name, subset_str=subset),
            )
            if name not in context.recent_write_accesses:
                context.recent_write_accesses[name] = oir.IntervalMapping()
            for section in vl.sections:
                if name in access_collections[section.id_].write_fields():
                    context.recent_write_accesses[name][section.interval] = candidate_node_idx
            if candidate_node.access == dace.AccessType.ReadOnly:
                candidate_node.access = dace.AccessType.ReadWrite

    def _add_read_edges_forward(self, ln, access_collections, context, origins):
        vl = ln.oir_node
        reads = set(
            name for collection in access_collections.values() for name in collection.read_fields()
        )
        # for reads: add edge from last access node with an overlaping write to LN
        for name in reads:
            most_recent_write = None
            for section in vl.sections:
                access_collection = access_collections[section.id_]
                for offset in access_collection.read_offsets()[name]:
                    interval = section.interval.shift(offset[2])
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

            subset = self._get_input_subset(vl, name, origins[name])
            context.state.add_edge(
                access_node,
                None,
                ln,
                "IN_" + name,
                dace.Memlet.simple(data=name, subset_str=subset),
            )
            if name not in context.recent_read_accesses:
                context.recent_read_accesses[name] = oir.IntervalMapping()
            context.recent_read_accesses[name][interval] = most_recent_write

    def _add_vertical_loop_forward(self, vl, context: "OirToSDFGVisitor.ForwardContext", origins):
        ln = VerticalLoopLibraryNode(oir_node=vl)
        context.state.add_node(ln)
        access_collections = {ls.id_: AccessCollector.apply(ls) for ls in vl.sections}
        self._add_read_edges_forward(ln, access_collections, context, origins)
        # for writes: add empty edge from last overlaping write to current LN (check if needed later)
        write_fields = set(
            name for collection in access_collections.values() for name in collection.write_fields()
        )
        for name in write_fields:
            if name in context.recent_write_accesses:
                for section in vl.sections:
                    writes = context.recent_write_accesses[name][section.interval]
                    if len(writes) > 0:
                        most_recent_write = context.accesses[name][max(writes)]
                        edge = context.state.add_edge(
                            most_recent_write, None, ln, None, dace.Memlet()
                        )
                        context.delete_candidates.append(edge)

        # for writes: add edge to existing last access if None of its reads or writes overlap with the write and
        #   otherwise, add new access node
        self._add_write_edges_forward(ln, access_collections, context, origins)

    def visit_Decl(
        self,
        node: oir.Decl,
        *,
        sdfg: SDFG,
        boundaries: Dict[str, oir.CartesianIterationSpace],
        **kwargs: Any,
    ):
        dtype = dace.dtypes.typeclass(np.dtype(common.data_type_to_typestr(node.dtype)).name)

        if isinstance(node, oir.ScalarDecl):
            sdfg.add_scalar(name=node.name, dtype=dtype)
        else:
            shape = self._access_space_to_shape(boundaries[node.name])
            if isinstance(node, oir.Temporary):
                sdfg.add_transient(name=node.name, dtype=dtype, shape=shape)
            elif isinstance(node, oir.FieldDecl):
                sdfg.add_array(name=node.name, dtype=dtype, shape=shape)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> SDFG:
        from gtc.gtir_to_oir import oir_field_boundary_computation

        boundaries = oir_field_boundary_computation(node)

        context = self.ForwardContext(node)
        # add arrays
        self.generic_visit(node, sdfg=context.sdfg, context=context, boundaries=boundaries)

        from gtc.gtir_to_oir import oir_field_boundary_computation

        boundaries = oir_field_boundary_computation(node)
        origins = {
            k: (-bound.i_interval.start.offset, -bound.j_interval.start.offset, 0)
            for k, bound in boundaries.items()
        }

        for vl in node.vertical_loops:
            self._add_vertical_loop_forward(vl, context, origins)
        # context = self.BackwardContext(context) # noqa: E800
        # for vl in reversed(node.vertical_loops):
        #     self._add_vertical_loop_backward(vl, context)  # noqa: E800

        import networkx as nx

        for edge in context.delete_candidates:
            assert nx.has_path(context.state.nx, edge.src, edge.dst)
            context.state.remove_edge(edge)
            if not nx.has_path(context.state.nx, edge.src, edge.dst):
                context.state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

        context.sdfg.validate()
        return context.sdfg
