# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union

import dace
import dace.properties
import networkx as nx
import numpy as np
from dace import SDFG

from eve.concepts import TreeNode
from eve.utils import XIterator, xiter
from eve.visitors import NodeVisitor
from gtc import oir
from gtc.common import AxisBound, CartesianOffset, LevelMarker, data_type_to_typestr
from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode
from gtc.oir import (
    CartesianIterationSpace,
    Interval,
    IntervalMapping,
    ScalarDecl,
    Stencil,
    Temporary,
    VerticalLoopSection,
)


@dataclass(frozen=True)
class Access:
    field: str
    offset: Tuple[int, int, int]
    is_write: bool

    @property
    def is_read(self) -> bool:
        return not self.is_write


class BaseOirSDFGBuilder(ABC):
    class AccessCollector(NodeVisitor):
        """Collects all field accesses and corresponding offsets."""

        def visit_FieldAccess(
            self,
            node: oir.FieldAccess,
            *,
            accesses: List[Access],
            is_write: bool,
            **kwargs: Any,
        ) -> None:
            accesses.append(
                Access(
                    field=node.name,
                    offset=(node.offset.i, node.offset.j, node.offset.k),
                    is_write=is_write,
                )
            )

        def visit_AssignStmt(
            self,
            node: oir.AssignStmt,
            **kwargs: Any,
        ) -> None:
            self.visit(node.right, is_write=False, **kwargs)
            self.visit(node.left, is_write=True, **kwargs)

        def visit_HorizontalExecution(
            self,
            node: oir.HorizontalExecution,
            **kwargs: Any,
        ) -> None:
            if node.mask is not None:
                self.visit(node.mask, is_write=False, **kwargs)
            for stmt in node.body:
                self.visit(stmt, **kwargs)

        @dataclass
        class Result:
            _ordered_accesses: List["Access"]

            @staticmethod
            def _offset_dict(accesses: XIterator) -> Dict[str, Set[Tuple[int, int, int]]]:
                return accesses.reduceby(
                    lambda acc, x: acc | {x.offset}, "field", init=set(), as_dict=True
                )

            def offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
                """Get a dictonary, mapping all accessed fields' names to sets of offset tuples."""
                return self._offset_dict(xiter(self._ordered_accesses))

            def read_offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
                """Get a dictonary, mapping read fields' names to sets of offset tuples."""
                return self._offset_dict(xiter(self._ordered_accesses).filter(lambda x: x.is_read))

            def write_offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
                """Get a dictonary, mapping written fields' names to sets of offset tuples."""
                return self._offset_dict(xiter(self._ordered_accesses).filter(lambda x: x.is_write))

            def fields(self) -> Set[str]:
                """Get a set of all accessed fields' names."""
                return {acc.field for acc in self._ordered_accesses}

            def read_fields(self) -> Set[str]:
                """Get a set of all read fields' names."""
                return {acc.field for acc in self._ordered_accesses if acc.is_read}

            def write_fields(self) -> Set[str]:
                """Get a set of all written fields' names."""
                return {acc.field for acc in self._ordered_accesses if acc.is_write}

            def ordered_accesses(self) -> List[Access]:
                """Get a list of ordered accesses."""
                return self._ordered_accesses

        @classmethod
        def apply(cls, node: TreeNode, **kwargs: Any) -> "Result":
            result = cls.Result([])
            cls().visit(node, accesses=result._ordered_accesses, **kwargs)
            return result

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

    def _access_space_to_subset(self, name, access_space):
        extent = self._extents[name]
        origin = (-extent[0][0], -extent[1][0])
        i_subset = "{start}:I{end:+d}".format(
            start=origin[0] + access_space.i_interval.start.offset,
            end=origin[0] + access_space.i_interval.end.offset,
        )
        j_subset = "{start}:J{end:+d}".format(
            start=origin[1] + access_space.j_interval.start.offset,
            end=origin[1] + access_space.j_interval.end.offset,
        )
        return f"{i_subset},{j_subset}"

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
        return res

    def _get_current_sink(self, name):
        if name in self._access_nodes:
            return self._access_nodes[name][-1]
        return None

    def _get_access_collection(
        self, node: "Union[HorizontalExecutionLibraryNode, VerticalLoopLibraryNode, SDFG]"
    ) -> "BaseOirSDFGBuilder.AccessCollector.Result":
        if isinstance(node, SDFG):
            res = BaseOirSDFGBuilder.AccessCollector.Result([])
            for node in node.states()[0].nodes():
                if isinstance(node, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode)):
                    collection = self._get_access_collection(node)
                    res._ordered_accesses.extend(collection._ordered_accesses)
            return res
        elif isinstance(node, HorizontalExecutionLibraryNode):
            if node.oir_node.id_ not in self._access_collection_cache:
                self._access_collection_cache[
                    node.oir_node.id_
                ] = BaseOirSDFGBuilder.AccessCollector.apply(node.oir_node)
            return self._access_collection_cache[node.oir_node.id_]
        else:
            assert isinstance(node, VerticalLoopLibraryNode)
            res = BaseOirSDFGBuilder.AccessCollector.Result([])
            for _, sdfg in node.sections:
                collection = self._get_access_collection(sdfg)
                res._ordered_accesses.extend(collection._ordered_accesses)
            return res

    def _get_recent_reads(self, name, interval):
        if name not in self._recent_read_acc:
            self._recent_read_acc[name] = IntervalMapping()
        return self._recent_read_acc[name][interval]

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
        read_accesses: Dict[str, dace.nodes.AccessNode] = dict()
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
                    access_node is None
                    or access_node in self._get_recent_reads(name, interval)
                    or access_node in self._get_recent_writes(name, interval)
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
                        edge = self._state.add_edge(node, None, dst, None, dace.Memlet())
                        self._delete_candidates.append(edge)

        for interval, collection in collections:
            for name in collection.write_fields():
                self._set_write(name, interval, node)

    def add_node(self, node):
        self._state.add_node(node)

    def _get_sdfg(self):
        for edge in self._delete_candidates:
            assert edge.src_conn is None
            assert edge.dst_conn is None
            self._state.remove_edge(edge)
            if not nx.has_path(self._state.nx, edge.src, edge.dst):
                self._state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

        self.add_subsets()
        self.add_arrays()
        self._sdfg.validate()
        for acc in self._sink_nodes.values():
            acc.access = dace.AccessType.WriteOnly
        return self._sdfg

    @abstractmethod
    def get_k_size(self, name):
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

    def add_arrays(self):
        for decl in self._stencil.params + self._stencil.declarations:
            name = decl.name
            dtype = dace.dtypes.typeclass(np.dtype(data_type_to_typestr(self._dtypes[name])).name)
            if isinstance(decl, ScalarDecl):
                self._sdfg.add_scalar(name, dtype=dtype)
            else:
                if name not in self._get_access_collection(self._sdfg).offsets():
                    continue
                assert name in self._dtypes
                di = self._extents[name][0][1] - self._extents[name][0][0]
                dj = self._extents[name][1][1] - self._extents[name][1][0]
                self._sdfg.add_array(
                    name,
                    dtype=dtype,
                    shape=(f"I{di:+d}", f"J{dj:+d}", self.get_k_size(name)),
                    transient=isinstance(decl, Temporary),
                )

    @classmethod
    def _build(cls, name, stencil: Stencil, nodes):
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

    @abstractmethod
    def get_k_subsets(self, node):
        pass

    def add_subsets(self):
        for node in self._state.nodes():
            if isinstance(node, dace.nodes.LibraryNode):
                access_spaces_input, access_spaces_output = self.get_access_spaces(node)
                k_subset_strs_input, k_subset_strs_output = self.get_k_subsets(node)
                for edge in self._state.in_edges(node) + self._state.out_edges(node):
                    if edge.dst_conn is not None:
                        name = edge.src.data
                        access_space = access_spaces_input[name]
                        subset_str_k = k_subset_strs_input[name]
                    elif edge.src_conn is not None:
                        name = edge.dst.data
                        access_space = access_spaces_output[name]
                        subset_str_k = k_subset_strs_output[name]
                    else:
                        continue
                    subset_str_ij = self._access_space_to_subset(name, access_space)
                    edge.data = dace.Memlet.simple(
                        data=name, subset_str=subset_str_ij + "," + subset_str_k
                    )


class VerticalLoopSectionOirSDFGBuilder(BaseOirSDFGBuilder):
    def get_k_size(self, name):
        collection = self._get_access_collection(self._sdfg)
        min_k = min(o[2] for o in collection.offsets()[name])
        max_k = max(o[2] for o in collection.offsets()[name])

        return f"{max_k-min_k+1}"

    def get_k_subsets(self, node):
        assert isinstance(node, HorizontalExecutionLibraryNode)
        collection = self._get_access_collection(node)
        write_subsets = dict()
        read_subsets = dict()
        k_origins = dict()
        for name, offsets in collection.offsets().items():
            k_origins[name] = -min(o[2] for o in offsets)
        for name, offsets in collection.read_offsets().items():
            read_subsets[name] = "{}:{}".format(
                k_origins[name] + min(o[2] for o in offsets),
                k_origins[name] + max(o[2] for o in offsets) + 1,
            )
        for name in collection.write_fields():
            write_subsets[name] = "{}:{}".format(
                k_origins[name],
                k_origins[name] + 1,
            )
        return read_subsets, write_subsets

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
        return super()._build(name, stencil, nodes)

    def get_access_spaces(self, node):
        assert isinstance(node, HorizontalExecutionLibraryNode)
        input_spaces = dict()
        output_spaces = dict()

        iteration_space = node.oir_node.iteration_space
        assert iteration_space is not None
        collection = self._get_access_collection(node)
        for name in collection.read_fields():
            access_space = CartesianIterationSpace.domain()
            for offset in collection.read_offsets()[name]:
                access_space = access_space | CartesianIterationSpace.from_offset(
                    CartesianOffset(i=offset[0], j=offset[1], k=offset[2])
                )
            input_spaces[name] = access_space.compose(iteration_space)
        for name in collection.write_fields():
            access_space = CartesianIterationSpace.domain()
            for offset in collection.write_offsets()[name]:
                access_space = access_space | CartesianIterationSpace.from_offset(
                    CartesianOffset(i=offset[0], j=offset[1], k=offset[2])
                )
            output_spaces[name] = access_space.compose(iteration_space)
        return input_spaces, output_spaces


class OirSDFGBuilder(BaseOirSDFGBuilder):
    def get_k_size(self, name):
        return "K"

    def get_k_subsets(self, node):
        assert isinstance(node, VerticalLoopLibraryNode)

        write_intervals = dict()
        read_intervals = dict()
        for interval, sdfg in node.sections:
            collection = self._get_access_collection(sdfg)
            for name, offsets in collection.read_offsets().items():
                for offset in offsets:
                    read_interval = interval.shift(offset[2])
                    read_intervals.setdefault(name, read_interval)
                    read_intervals[name] = Interval(
                        start=min(read_intervals[name].start, read_interval.start),
                        end=max(read_intervals[name].end, read_interval.end),
                    )

            for name in collection.write_fields():
                write_intervals.setdefault(name, interval)
                write_intervals[name] = Interval(
                    start=min(write_intervals[name].start, interval.start),
                    end=max(write_intervals[name].end, interval.end),
                )
        write_subsets = dict()
        for name, interval in write_intervals.items():
            write_subsets[name] = "{}{:+d}:{}{:+d}".format(
                "K" if interval.start.level == LevelMarker.END else "",
                interval.start.offset,
                "K" if interval.end.level == LevelMarker.END else "",
                interval.end.offset,
            )
        read_subsets = dict()
        for name, interval in read_intervals.items():
            read_subsets[name] = "{}{:+d}:{}{:+d}".format(
                "K" if interval.start.level == LevelMarker.END else "",
                interval.start.offset,
                "K" if interval.end.level == LevelMarker.END else "",
                interval.end.offset,
            )
        return read_subsets, write_subsets

    def get_access_spaces(self, node):
        assert isinstance(node, VerticalLoopLibraryNode)
        input_spaces = dict()
        output_spaces = dict()
        for _, sdfg in node.sections:
            for n in sdfg.states()[0].nodes():

                if not isinstance(n, HorizontalExecutionLibraryNode):
                    continue
                iteration_space = n.oir_node.iteration_space
                assert iteration_space is not None
                collection = self._get_access_collection(n)
                for name in collection.read_fields():
                    access_space = CartesianIterationSpace.domain()
                    for offset in collection.read_offsets()[name]:
                        access_space = access_space | CartesianIterationSpace.from_offset(
                            CartesianOffset(i=offset[0], j=offset[1], k=offset[2])
                        )
                    if name not in input_spaces:
                        input_spaces[name] = access_space.compose(iteration_space)
                    else:
                        input_spaces[name] = input_spaces[name] | access_space.compose(
                            iteration_space
                        )
                for name in collection.write_fields():
                    access_space = CartesianIterationSpace.domain()
                    for offset in collection.write_offsets()[name]:
                        access_space = access_space | CartesianIterationSpace.from_offset(
                            CartesianOffset(i=offset[0], j=offset[1], k=offset[2])
                        )
                    if name not in output_spaces:
                        output_spaces[name] = access_space.compose(iteration_space)
                    else:
                        output_spaces[name] = output_spaces[name] | access_space.compose(
                            iteration_space
                        )
        return input_spaces, output_spaces

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
        return super()._build(name, stencil, nodes)
