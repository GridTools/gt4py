# -*- coding: utf-8 -*-
from abc import ABC, abstractclassmethod, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import dace.properties
import networkx as nx
from dace import SDFG, library

from gtc.common import AxisBound, LoopOrder, data_type_to_typestr
from gtc.oir import (
    ScalarDecl, Temporary,
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


class BaseOirSDFGBuilder(ABC):
    def __init__(self, name, stencil, extents):
        self._stencil = stencil
        self._sdfg = SDFG(name)
        self._state = self._sdfg.add_state(name + "_state")
        self._extents = extents

        self._dtypes = {decl.name: decl.dtype for decl in stencil.params + stencil.declarations}

        self._recent_write_acc = dict()
        self._recent_read_acc = dict()

        self._access_nodes = dict()
        self._access_collection_cache = dict()
        self._source_nodes = dict()
        self._sink_nodes = dict()
        self._delete_candidates = list()

    def _are_nodes_ordered(self, name, node1, node2):
        assert name in self._access_nodes
        assert node1.data == name
        assert node2.data == name
        return self._access_nodes[name].index(node1) < self._access_nodes[name].index(node2)

    def _get_source(self, name):
        if name not in self._source_nodes:
            self._source_nodes[name] = self._state.add_read(name)
            if name not in self._access_nodes:
                self._access_nodes[name] = []
            self._access_nodes[name].insert(0, self._source_nodes[name])
        return self._source_nodes[name]

    def _get_new_sink(self, name):
        res = self._state.add_access(name)
        if name not in self._access_nodes:
            self._access_nodes[name] = []
        self._access_nodes[name].append(res)
        if len(self._access_nodes[name]) == 1:
            self._source_nodes[name] = res
        return res

    def _get_current_sink(self, name):
        if name in self._access_nodes:
            return self._access_nodes[name][-1]
        return None

    def _get_access_collection(
        self, node: "Union[HorizontalExecutionLibraryNode, VerticalLoopLibraryNode, SDFG]"
    ) -> AccessCollector.Result:
        if isinstance(node, SDFG):
            res = AccessCollector.Result([])
            for node in node.states()[0].nodes():
                if isinstance(node, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode)):
                    collection = self._get_access_collection(node)
                    res._ordered_accesses.extend(collection._ordered_accesses)
            return res
        elif isinstance(node, HorizontalExecutionLibraryNode):
            if node.oir_node.id_ not in self._access_collection_cache:
                self._access_collection_cache[node.oir_node.id_] = AccessCollector.apply(
                    node.oir_node
                )
            return self._access_collection_cache[node.oir_node.id_]
        else:
            assert isinstance(node, VerticalLoopLibraryNode)
            res = AccessCollector.Result([])
            for _, sdfg in node.sections:
                collection = self._get_access_collection(sdfg)
                res._ordered_accesses.extend(collection._ordered_accesses)
            return res

    def _get_recent_writes(self, name, interval):
        if name not in self._recent_write_acc:
            self._recent_write_acc[name] = IntervalMapping()
        return self._recent_write_acc[name][interval]


    def _set_read(self, name, interval, node):
        if name not in self._recent_read_acc:
            self._recent_read_acc[name] = IntervalMapping()
        self._recent_read_acc[name][interval] = node

    def _set_write(self, name, interval, node):
        if name not in self._recent_write_acc:
            self._recent_write_acc[name] = IntervalMapping()
        self._recent_write_acc[name][interval] = node

    def _reset_writes(self):
        self._recent_write_acc = dict()

    def _add_read_edges(self, node, collections: List[Tuple[Interval, AccessCollector.Result]]):
        read_accesses = dict()
        for interval, access_collection in collections:

            for name in access_collection.read_fields():
                for offset in access_collection.read_offsets()[name]:
                    read_interval = interval.shift(offset[2])
                    for candidate_access in self._get_recent_writes(name, read_interval):
                        if name not in read_accesses or self._are_nodes_ordered(
                            name, read_accesses[name], candidate_access
                        ):
                            # candidate_access is downstream from recent_access, therefore candidate is more recent
                            read_accesses[name] = candidate_access

        for interval, access_collection in collections:
            for name in access_collection.read_fields():
                for offset in access_collection.read_offsets()[name]:
                    read_interval = interval.shift(offset[2])
                    if name not in read_accesses:
                        read_accesses[name] = self._get_source(name)
                    self._set_read(name, read_interval, read_accesses[name])

        for name, recent_access in read_accesses.items():
            node.add_in_connector("IN_" + name)
            self._state.add_edge(
                recent_access,
                None,
                node,
                "IN_" + name,
                dace.Memlet(),
            )

    def _add_write_edges(self, node, collections: List[Tuple[Interval, AccessCollector.Result]]):
        write_accesses = dict()
        for interval, access_collection in collections:
            for name in access_collection.write_fields():
                access_node = self._get_current_sink(name)
                if (
                    name not in write_accesses
                    or access_node in self._get_recent_reads(name, interval)
                    or access_node in self._get_recent_writes(self, name, interval)
                ):
                    write_accesses[name] = self._get_new_sink(name)
                else:
                    write_accesses[name] = access_node
                self._set_write(name, interval, write_accesses[name])

        for name, access_node in write_accesses.items():
            node.add_out_connector("OUT_" + name)
            self._state.add_edge(
                node,
                "OUT_" + name,
                access_node,
                None,
                dace.Memlet(),
            )

    def _add_write_after_write_edges(
        self, node, collections: List[Tuple[Interval, AccessCollector.Result]]
    ):
        for interval, collection in collections:
            for name in collection.write_fields():
                for src in self._get_recent_writes(name, interval):
                    edge = self._state.add_edge(src, None, node, None, dace.Memlet())
                    self._delete_candidates.append(edge)

    def _add_write_after_read_edges(
        self, node, collections: List[Tuple[Interval, AccessCollector.Result]]
    ):
        for interval, collection in collections:
            for name in collection.read_fields():
                for offset in collection.read_offsets()[name]:
                    read_interval = interval.shift(offset[2])
                    for dst in self._get_recent_writes(name, read_interval):
                        self._state.add_edge(node, None, dst, None, dace.Memlet())

        for interval, collection in collections:
            for name in collection.write_fields():
                self._set_write(name, interval, node)

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

    def _get_sdfg(self):
        for edge in self._delete_candidates:
            self._state.remove_edge(edge)
            if not nx.has_path(self._state.nx, edge.src, edge.dst):
                self._state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

        self.add_subsets()
        self.add_arrays()
        self._sdfg.save(self._sdfg.name + ".sdfg")
        self._sdfg.validate()
        return self._sdfg

    @abstractmethod
    def add_subsets(self):
        pass

    @abstractmethod
    def add_arrays(self):
        pass

    @abstractmethod
    def add_read_edges(self, node):
        pass

    @abstractmethod
    def add_write_edges(self, node):
        pass

    @abstractmethod
    def add_write_after_read_edges(self, node):
        pass

    @abstractmethod
    def add_write_after_write_edges(self, node):
        pass

    @classmethod
    def build(cls, name, stencil: Stencil, nodes):
        from gtc.gtir_to_oir import oir_field_boundary_computation

        extents = dict()
        for n, access_space in oir_field_boundary_computation(stencil).items():
            i_extent = (access_space.i_interval.start.offset, access_space.i_interval.end.offset)
            j_extent = (access_space.j_interval.start.offset, access_space.j_interval.end.offset)
            extents[n] = (i_extent, j_extent)
        builder = cls(name, stencil, extents)
        for n in nodes:
            builder.add_write_after_write_edges(n)
            builder.add_read_edges(n)
            builder.add_write_edges(n)
        builder._reset_writes()
        for n in reversed(nodes):
            builder.add_write_after_read_edges(n)
        return builder._get_sdfg()


class VerticalLoopSectionOirSDFGBuilder(BaseOirSDFGBuilder):
    def add_arrays(self):
        for decl in self._stencil.params+self._stencil.declarations:
            name = decl.name
            dtype = dace.dtypes.typeclass(np.dtype(data_type_to_typestr(self._dtypes[name])).name)
            if isinstance(decl, ScalarDecl):
                self._sdfg.add_scalar(name, dtype=dtype)
            else:
                assert name in self._dtypes
                di = self._extents[name][0][1] - self._extents[name][0][0]
                dj = self._extents[name][1][1] - self._extents[name][1][0]
                self._sdfg.add_array(name, dtype=dtype, shape=(f"I{di:+d}", f"J{dj:+d}", "K"), transient=isinstance(decl, Temporary))

    def add_subsets(self):
        pass

    def add_read_edges(self, node):
        interval = Interval(start=AxisBound.start(), end=AxisBound.end())
        return self._add_read_edges(node, [(interval, self._get_access_collection(node))])

    def add_write_edges(self, node):
        interval = Interval(start=AxisBound.start(), end=AxisBound.end())
        return self._add_write_edges(node, [(interval, self._get_access_collection(node))])

    def add_write_after_write_edges(self, node):
        interval = Interval(start=AxisBound.start(), end=AxisBound.end())
        return self._add_write_after_write_edges(
            node, [(interval, self._get_access_collection(node))]
        )

    def add_write_after_read_edges(self, node):
        interval = Interval(start=AxisBound.start(), end=AxisBound.end())
        return self._add_write_after_read_edges(
            node, [(interval, self._get_access_collection(node))]
        )

    @classmethod
    def build(cls, name, stencil: Stencil, node: VerticalLoopSection):
        nodes = [HorizontalExecutionLibraryNode(oir_node=he) for he in node.horizontal_executions]
        return super().build(name, stencil, nodes)


class OirSDFGBuilder(BaseOirSDFGBuilder):

    def add_arrays(self):
        for decl in self._stencil.params+self._stencil.declarations:
            name = decl.name
            dtype = dace.dtypes.typeclass(np.dtype(data_type_to_typestr(self._dtypes[name])).name)
            if isinstance(decl, ScalarDecl):
                self._sdfg.add_scalar(name, dtype=dtype)
            else:
                assert name in self._dtypes
                di = self._extents[name][0][1] - self._extents[name][0][0]
                dj = self._extents[name][1][1] - self._extents[name][1][0]
                self._sdfg.add_array(name, dtype=dtype, shape=(f"I{di:+d}", f"J{dj:+d}", "K"), transient=isinstance(decl, Temporary))

    def add_subsets(self):
        pass

    def _get_collection_from_sections(self, sections):
        res = []
        for interval, sdfg in sections:
            collection = self._get_access_collection(sdfg)
            res.append((interval, collection))
        return res

    def add_read_edges(self, node):
        collections = self._get_collection_from_sections(node.sections)
        return self._add_read_edges(node, collections)

    def add_write_edges(self, node):
        collections = self._get_collection_from_sections(node.sections)
        return self._add_write_edges(node, collections)

    def add_write_after_write_edges(self, node):
        collections = self._get_collection_from_sections(node.sections)
        return self._add_write_after_write_edges(node, collections)

    def add_write_after_read_edges(self, node):
        collections = self._get_collection_from_sections(node.sections)
        return self._add_write_after_read_edges(node, collections)

    @classmethod
    def build(cls, name, stencil: Stencil):
        nodes = [
            VerticalLoopLibraryNode(stencil=stencil, oir_node=vl) for vl in stencil.vertical_loops
        ]
        return super().build(name, stencil, nodes)


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

    def __init__(
        self,
        name="unnamed_vloop",
        stencil: Stencil = None,
        oir_node: VerticalLoop = None,
        *args,
        **kwargs,
    ):
        if oir_node is not None:
            name = oir_node.id_

        if stencil is not None:
            self.loop_order = oir_node.loop_order
            self.sections = [
                (
                    section.interval,
                    VerticalLoopSectionOirSDFGBuilder.build(section.id_, stencil, section),
                )
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
