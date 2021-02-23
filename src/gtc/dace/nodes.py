# -*- coding: utf-8 -*-
from abc import ABC, abstractclassmethod, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Union

import dace.properties
import networkx as nx
from dace import SDFG, library

from gtc.common import AxisBound, LoopOrder
from gtc.oir import (
    CacheDesc,
    CartesianIterationSpace,
    HorizontalExecution,
    Interval,
    IntervalMapping,
    Stencil,
    VerticalLoop,
    VerticalLoopSection,
)
from gtc.passes.oir_optimizations.utils import AccessCollector


class OirSDFGBuilder(ABC):
    def __init__(self, name, origins, dtypes):
        self._mode = "FORWARD"
        self._sdfg = SDFG(name)
        self._state = self._sdfg.add_state(name + "_state")
        self._origins = origins

        self._recent_write_acc = dict()
        self._next_read_acc = dict()
        self._next_write_acc = dict()

        self._access_collection_cache = dict()
        self._source_nodes = dict()
        self._sink_nodes = dict()
        self._delete_candidates = list()

    @abstractmethod
    def _get_read_subset(self, node):
        pass

    @abstractmethod
    def _get_write_subset(self, node):
        pass

    def _get_access_collection(self, node):
        if node.id_ not in self._access_collection_cache:
            self._access_collection_cache[node.id_] = AccessCollector.apply(node)
        return self._access_collection_cache[node.id_]

    def _get_recent_writes(self, name, interval):
        return []

    def _get_next_writes(self, name, interval):
        return []

    def _get_next_reads(self, name, interval):
        return []

    def _set_read(self, name, interval, node):
        pass

    def _set_write_backward(self, name, interval, node):
        assert self._mode == "BACKWARD"
        candidates = self._get_next_reads(name, interval)
        if len(candidates) == 0:
            res = self._get_sink_node(name)
        else:
            res = candidates[0]
            for cand in candidates:
                if nx.has_path(self._state.nx, cand, res):
                    res = cand

        self._next_write_acc[name][interval] = res
        return res

    def _set_write_forward(self, name, interval):
        assert self._mode == "FORWARD"
        res = ...
        self._recent_write_acc[name][interval] = res

    def _add_read_edges(self, node, collections):
        recent_accesses = dict()
        for interval, access_collection in collections:

            for name in access_collection.read_fields():
                for offset in access_collection.read_offsets[name]:
                    read_interval = interval.shift(offset[2])
                    for candidate_access in self._get_recent_writes(name, read_interval):
                        if name not in recent_accesses is None or nx.has_path(
                            self._state.nx, recent_accesses[name], candidate_access
                        ):
                            # candidate_access is downstream from recent_access, therefore candidate is more recent
                            recent_accesses[name] = candidate_access
        for name, recent_access in recent_accesses.items():
            subset_str = self._get_read_subset(node, name)
            self._state.add_edge(
                recent_access,
                None,
                node,
                "IN_" + name,
                dace.memlet.Memlet.simple(data=name, subset_str=subset_str),
            )

    def _add_write_edges(self, node, sections):
        access_collection = self._get_access_collection(node)

        # write edges ()
        for name in access_collection.write_fields():
            access_node = self._set_write(name, interval, node)
            subset_str = self._get_write_subset(node, name)
            self._state.add_edge(
                node,
                "OUT_" + name,
                access_node,
                None,
                dace.memlet.Memlet.simple(data=name, subset_str=subset_str),
            )

    def add_write_after_read_edges(self, node):
        pass
        # access_collection = self._get_access_collection(node)
        # # read-write dependency edges (LN to LN)
        # for name in access_collection.write_fields():
        #     next_access = self._get_next_writes(name, interval)
        #     self._delete_candidates.append(
        #         self._state.add_edge(node, None, next_access, None, dace.memlet.Memlet())
        #     )

    def add_node(self, node):
        self._state.add_node(node)

    # def add_oir_node(self, node: 'Union[HorizontalExecutionLibraryNode, VerticalLoopLibraryNode]'):
    #
    #     if isinstance(node, HorizontalExecutionLibraryNode):
    #         nodes = [(Interval(start=AxisBound.start(), end=AxisBound.end()), node)]
    #     else:
    #         assert isinstance(node, VerticalLoop)
    #         nodes = [(n.interval, n) for n in node.sections]
    #     if self._mode == "FORWARD":
    #         for interval, n in nodes:
    #             self._state.add_node(n)
    #             self._edges_oir_node_forward(n, interval)
    #     else:
    #         assert self._mode == "BACKWARD"
    #         for interval, n in reversed(nodes):
    #             self._edges_oir_node_backward(n, interval)

    def get_sdfg(self):
        self._mode = "FINAL"
        for edge in self._delete_candidates:
            self._state.remove_edge(edge)
            if not nx.has_path(self._state.nx, edge.src, edge.dst):
                self._state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
        return self._sdfg

    @abstractmethod
    def add_read_edges(self, node):
        pass

    @abstractmethod
    def add_write_edges(self, node):
        pass

    @classmethod
    def build(cls, name, nodes, origins, dtypes):
        builder = cls(name, origins, dtypes)

        for n in nodes:
            builder.add_node(n)
        for n in nodes:
            builder.add_read_edges(n)
        for n in reversed(nodes):
            builder.add_write_edges(n)
        # for n in nodes:
        #     builder.add_write_after_write_edges(n)
        # for n in reversed(nodes):
        #     builder.add_write_after_read_edges(n)
        return builder.get_sdfg()


class VerticalLoopOirSDFGBuilder(OirSDFGBuilder):

    def _get_collection_from_sections(self, sections):
        res = []
        for interval, sdfg in sections:
            collector = AccessCollector()
            collection = AccessCollector.Result([])
            for node in sdfg.states()[0].nodes():
                if isinstance(node, HorizontalExecutionLibraryNode)
                    collector.visit(node.oir_node, accesses=collection._ordered_accesses)
            res.append((interval, collection))
        return res

    def add_read_edges(self, node):
        collections = self._get_collection_from_sections(node.sections)
        return self._add_read_edges(node, collections)

    def add_write_edges(self, node):
        collections = self._get_collection_from_sections(node.sections)
        return self._add_write_edges(node, collections)

class HorizontalExecutionOirSDFGBuilder(OirSDFGBuilder):

    def add_read_edges(self, node):
        interval = Interval(start=AxisBound.start(), end=AxisBound.end())
        return self._add_read_edges(node, [(interval, self._get_access_collection(node))])

    def add_write_edges(self, node):
        interval = Interval(start=AxisBound.start(), end=AxisBound.end())
        return self._add_write_edges(node, [(interval, self._get_access_collection(node))])



def get_vertical_loop_section_sdfg(section: "VerticalLoopSection") -> SDFG:
    sdfg = SDFG(section.id_)
    state = sdfg.add_state("start_state", is_start_state=True)
    he_iter = iter(section.horizontal_executions)
    last_node = HorizontalExecutionLibraryNode(oir_node=next(he_iter))
    state.add_node(last_node)
    for he in he_iter:
        new_node = HorizontalExecutionLibraryNode(oir_node=he)
        state.add_node(new_node)
        state.add_edge(last_node, None, new_node, None, dace.memlet.Memlet())
        last_node = new_node
    return sdfg


@library.expansion
class NoLibraryNodeImplementation(dace.library.ExpandTransformation):
    environments: List = []


@library.node
class VerticalLoopLibraryNode(dace.nodes.LibraryNode):
    implementations = {"none": NoLibraryNodeImplementation}
    default_implementation = "none"

    loop_order = dace.properties.Property(dtype=LoopOrder, default=None, allow_none=True)
    sections = dace.properties.ListProperty(
        element_type=Tuple[Interval, dace.SDFG], default=[], allow_none=False
    )
    caches = dace.properties.ListProperty(
        element_type=List[CacheDesc], default=[], allow_none=False
    )

    def __init__(self, name="unnamed_vloop", oir_node: VerticalLoop = None, *args, **kwargs):
        if oir_node is not None:
            name = oir_node.id_

        self.loop_order = oir_node.loop_order
        self.sections = [
            (section.interval, get_vertical_loop_section_sdfg(section))
            for section in oir_node.sections
        ]
        self.caches = oir_node.caches

        super().__init__(name=name, *args, **kwargs)


@library.node
class HorizontalExecutionLibraryNode(dace.nodes.LibraryNode):
    implementations = {"none": NoLibraryNodeImplementation}
    default_implementation = "none"

    oir_node = dace.properties.DataclassProperty(
        dtype=HorizontalExecution, default=None, allow_none=True
    )

    def __init__(self, name="unnamed_vloop", oir_node: HorizontalExecution = None, *args, **kwargs):
        if oir_node is not None:
            name = oir_node.id_
            self.oir_node = oir_node

        super().__init__(name=name, *args, **kwargs)
