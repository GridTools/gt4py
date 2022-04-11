# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import dace
import dace.properties
import dace.subsets
import networkx as nx
import numpy as np
from dace import SDFG
from dace.sdfg.graph import MultiConnectorEdge

import eve
import gtc.oir as oir
from gt4py.definitions import Extent
from gtc.common import LevelMarker, VariableKOffset, data_type_to_typestr
from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode
from gtc.dace.utils import IntervalMapping, nodes_extent_calculation
from gtc.oir import FieldDecl, Interval, ScalarDecl, Stencil, Temporary
from gtc.passes.oir_optimizations.utils import AccessCollector


def _offset_origin(interval: oir.Interval, origin: Optional[oir.AxisBound]) -> oir.Interval:
    if origin is None:
        return interval
    if origin.level != LevelMarker.START:
        return interval
    return interval.shifted(-origin.offset)


class BaseOirSDFGBuilder(ABC):
    has_transients = True

    def __init__(self, name, stencil: Stencil, nodes):
        self._stencil = stencil
        self._sdfg = SDFG(name)
        self._state = self._sdfg.add_state(name + "_state")
        self._extents = nodes_extent_calculation(nodes)

        self._dtypes = {decl.name: decl.dtype for decl in stencil.declarations + stencil.params}
        self._axes = {
            decl.name: decl.dimensions
            for decl in stencil.declarations + stencil.params
            if isinstance(decl, FieldDecl)
        }

        self._recent_write_acc: Dict[str, dace.nodes.AccessNode] = dict()
        self._recent_read_acc: Dict[str, dace.nodes.AccessNode] = dict()

        self._access_nodes: Dict[str, dace.nodes.AccessNode] = dict()
        self._access_collection_cache: Dict[int, AccessCollector.GeneralAccessCollection] = dict()
        self._source_nodes: Dict[str, dace.nodes.AccessNode] = dict()
        self._delete_candidates: List[MultiConnectorEdge] = list()

        def generate_access_nodes(node):
            if isinstance(node, VerticalLoopLibraryNode):
                for _, s in node.sections:
                    yield from generate_access_nodes(s)
            elif isinstance(node, dace.SDFG):
                for n, _ in node.all_nodes_recursive():
                    if isinstance(n, dace.nodes.LibraryNode):
                        yield from generate_access_nodes(n)
            elif isinstance(node, HorizontalExecutionLibraryNode):
                yield from [
                    acc.name
                    for acc in node.oir_node.iter_tree().if_isinstance(oir.FieldAccess)
                    if isinstance(acc.offset, VariableKOffset)
                ]
            else:
                for n in node:
                    yield from generate_access_nodes(n)

        self._dynamic_k_fields = set(generate_access_nodes(nodes))

    def _field_extent_to_subset(self, name, field_extent):
        origin = (-self._extents[name][0][0], -self._extents[name][1][0])
        subsets = []
        if self._axes[name][0]:
            subsets.append(
                "{start}:__I{end:+d}".format(
                    start=origin[0] + field_extent[0][0], end=origin[0] + field_extent[0][1]
                )
            )
        if self._axes[name][1]:
            subsets.append(
                "{start}:__J{end:+d}".format(
                    start=origin[1] + field_extent[1][0], end=origin[1] + field_extent[1][1]
                )
            )
        return subsets

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
    ) -> AccessCollector.GeneralAccessCollection:
        if isinstance(node, SDFG):
            res = AccessCollector.GeneralAccessCollection([])
            for n, _ in node.all_nodes_recursive():
                if isinstance(n, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode)):
                    collection = self._get_access_collection(n)
                    res._ordered_accesses.extend(collection._ordered_accesses)
            return res
        elif isinstance(node, HorizontalExecutionLibraryNode):
            if id(node.oir_node) not in self._access_collection_cache:
                self._access_collection_cache[id(node.oir_node)] = AccessCollector.apply(
                    node.oir_node
                )
            return self._access_collection_cache[id(node.oir_node)]
        else:
            assert isinstance(node, VerticalLoopLibraryNode)
            res = AccessCollector.GeneralAccessCollection([])
            for _, sdfg in node.sections:
                collection = self._get_access_collection(sdfg)
                res._ordered_accesses.extend(collection._ordered_accesses)
            return res

    def _get_recent_reads(self, name, interval):
        if name not in self._recent_read_acc:
            self._recent_read_acc[name] = IntervalMapping()
        if not self._axes[name][2]:
            interval = Interval.full()
        return self._recent_read_acc[name][interval]

    def _get_recent_writes(self, name, interval):
        if name not in self._recent_write_acc:
            self._recent_write_acc[name] = IntervalMapping()
        if not self._axes[name][2]:
            interval = Interval.full()
        return self._recent_write_acc[name][interval]

    def _set_read(self, name, interval, node):
        if name not in self._recent_read_acc:
            self._recent_read_acc[name] = IntervalMapping()
        if not self._axes[name][2]:
            interval = Interval.full()
        self._recent_read_acc[name][interval] = node

    def _set_write(self, name, interval, node):
        if name not in self._recent_write_acc:
            self._recent_write_acc[name] = IntervalMapping()
        if not self._axes[name][2]:
            interval = Interval.full()
        self._recent_write_acc[name][interval] = node

    def _reset_writes(self):
        self._recent_write_acc = dict()

    def _add_read_edges(
        self, node, collections: List[Tuple[Interval, AccessCollector.GeneralAccessCollection]]
    ):
        read_accesses: Dict[str, dace.nodes.AccessNode] = dict()
        for interval, access_collection in collections:

            for name in access_collection.read_fields():
                for offset in access_collection.read_offsets()[name]:
                    read_interval = (
                        interval.shifted(offset[2])
                        if offset[2] is not None
                        else oir.UnboundedInterval.full()
                    )
                    for candidate_access in self._get_recent_writes(name, read_interval):
                        if name not in read_accesses or self._are_nodes_ordered(
                            name, read_accesses[name], candidate_access
                        ):
                            # candidate_access is downstream from recent_access, therefore candidate is more recent
                            read_accesses[name] = candidate_access

        for interval, access_collection in collections:
            for name in access_collection.read_fields():
                for offset in access_collection.read_offsets()[name]:
                    read_interval = (
                        interval.shifted(offset[2])
                        if offset[2] is not None
                        else oir.UnboundedInterval.full()
                    )
                    if name not in read_accesses:
                        read_accesses[name] = self._get_source(name)
                    self._set_read(name, read_interval, read_accesses[name])

        for name, recent_access in read_accesses.items():
            node.add_in_connector("IN_" + name)
            self._state.add_edge(recent_access, None, node, "IN_" + name, dace.Memlet())

    def _add_write_edges(
        self, node, collections: List[Tuple[Interval, AccessCollector.GeneralAccessCollection]]
    ):
        write_accesses = dict()
        for interval, access_collection in collections:
            for name in access_collection.write_fields():
                access_node = self._get_current_sink(name)
                if access_node is None or (
                    (name not in write_accesses)
                    and (
                        access_node in self._get_recent_reads(name, interval)
                        or access_node in self._get_recent_writes(name, interval)
                        or nx.has_path(self._state.nx, access_node, node)
                    )
                ):
                    write_accesses[name] = self._get_new_sink(name)
                else:
                    write_accesses[name] = access_node
                self._set_write(name, interval, write_accesses[name])

        for name, access_node in write_accesses.items():
            node.add_out_connector("OUT_" + name)
            self._state.add_edge(node, "OUT_" + name, access_node, None, dace.Memlet())

    def _add_write_after_write_edges(
        self, node, collections: List[Tuple[Interval, AccessCollector.GeneralAccessCollection]]
    ):
        for interval, collection in collections:
            for name in collection.write_fields():
                for src in self._get_recent_writes(name, interval):
                    edge = self._state.add_edge(src, None, node, None, dace.Memlet())
                    self._delete_candidates.append(edge)

    def _add_write_after_read_edges(
        self, node, collections: List[Tuple[Interval, AccessCollector.GeneralAccessCollection]]
    ):
        for interval, collection in collections:
            for name in collection.read_fields():
                for offset in collection.read_offsets()[name]:
                    read_interval = (
                        interval.shifted(offset[2])
                        if offset[2] is not None
                        else oir.UnboundedInterval()
                    )
                    for dst in self._get_recent_writes(name, read_interval):
                        edge = self._state.add_edge(node, None, dst, None, dace.Memlet())
                        self._delete_candidates.append(edge)

        for interval, collection in collections:
            for name in collection.write_fields():
                self._set_write(name, interval, node)

    def add_node(self, node):
        self._state.add_node(node)

    def finalize(self):
        for edge in self._delete_candidates:
            assert edge.src_conn is None
            assert edge.dst_conn is None
            self._state.remove_edge(edge)
            if not nx.has_path(self._state.nx, edge.src, edge.dst):
                self._state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

        self.add_subsets()
        self.add_arrays()

    def _get_sdfg(self):
        self.finalize()
        return self._sdfg

    def add_arrays(self):
        shapes = self.get_shapes()
        for decl in self._stencil.params + self._stencil.declarations:
            name = decl.name
            dtype = dace.dtypes.typeclass(np.dtype(data_type_to_typestr(self._dtypes[name])).name)
            if isinstance(decl, ScalarDecl):
                self._sdfg.add_symbol(name, stype=dtype)
            else:
                if name not in self._get_access_collection(self._sdfg).offsets():
                    continue
                assert name in self._dtypes
                strides = tuple(
                    dace.symbolic.pystr_to_symbolic(f"__{name}_{var}_stride")
                    for is_axis, var in zip(self._axes[name], "IJK")
                    if is_axis
                ) + tuple(
                    dace.symbolic.pystr_to_symbolic(f"__{name}_d{dim}_stride")
                    for dim, _ in enumerate(decl.data_dims)
                )
                self._sdfg.add_array(
                    name,
                    dtype=dtype,
                    shape=shapes[name],
                    strides=strides,
                    transient=isinstance(decl, Temporary) and self.has_transients,
                    lifetime=dace.AllocationLifetime.Persistent,
                )

    def add_subsets(self):
        decls = {decl.name: decl for decl in self._stencil.params + self._stencil.declarations}
        for node in self._state.nodes():
            if isinstance(node, dace.nodes.LibraryNode):
                input_extents, output_extents = self.get_field_extents(node)
                k_subset_strs_input, k_subset_strs_output = self.get_k_subsets(node)
                for edge in self._state.in_edges(node) + self._state.out_edges(node):
                    if edge.dst_conn is not None:
                        name = edge.src.data
                        access_extent = input_extents[name]
                        subset_str_k = k_subset_strs_input.get(name, None)
                        dynamic = (
                            isinstance(node, HorizontalExecutionLibraryNode)
                            and len(node.oir_node.iter_tree().if_isinstance(oir.MaskStmt).to_list())
                            > 0
                        )
                        dynamic = dynamic or (
                            isinstance(node, HorizontalExecutionLibraryNode)
                            and any(
                                isinstance(acc.offset, VariableKOffset)
                                for acc in node.oir_node.iter_tree().if_isinstance(oir.FieldAccess)
                                if acc.name == name
                            )
                        )
                    elif edge.src_conn is not None:
                        name = edge.dst.data
                        access_extent = output_extents[name]
                        subset_str_k = k_subset_strs_output.get(name, None)
                        dynamic = False
                    else:
                        continue
                    subset_strs = self._field_extent_to_subset(name, access_extent)
                    if subset_str_k is not None:
                        if name in self._dynamic_k_fields and isinstance(
                            node, HorizontalExecutionLibraryNode
                        ):
                            subset_strs_k = subset_str_k.split(":")
                            subset_strs.append(f"k+({subset_strs_k[0]}):k+({subset_strs_k[1]})")
                        else:
                            subset_strs.append(subset_str_k)
                    elif self.get_k_size(name) is not None:
                        subset_strs.append(f"0:{self.get_k_size(name)}")
                    for dim in decls[name].data_dims:
                        subset_strs.append(f"0:{dim}")
                    edge.data = dace.Memlet.simple(
                        data=name, subset_str=",".join(subset_strs), dynamic=dynamic
                    )

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

    @abstractmethod
    def get_k_subsets(self, node):
        pass

    @abstractmethod
    def get_field_extents(self, node):
        pass

    @abstractmethod
    def get_shapes(self):
        pass

    @classmethod
    def build(cls, name, stencil: Stencil, nodes: List[dace.nodes.LibraryNode]):
        builder = cls(name, stencil, nodes)
        for n in nodes:
            builder.add_node(n)
            builder.add_write_after_write_edges(n)
            builder.add_read_edges(n)
            builder.add_write_edges(n)
        builder._reset_writes()
        for n in reversed(nodes):
            builder.add_write_after_read_edges(n)
        res = builder._get_sdfg()
        res.validate()
        return res


class VerticalLoopSectionOirSDFGBuilder(BaseOirSDFGBuilder):
    has_transients = False

    def get_shapes(self):
        import dace.subsets

        subsets = dict()
        for edge in self._state.edges():
            if edge.data.data is not None:
                if "k" in edge.data.subset.free_symbols:
                    continue
                if edge.data.data not in subsets:
                    subsets[edge.data.data] = edge.data.subset
                subsets[edge.data.data] = dace.subsets.union(
                    subsets[edge.data.data], edge.data.subset
                )
        bounding_boxes = {name: subset.bounding_box_size() for name, subset in subsets.items()}
        origins = {name: [r[0] for r in subset.ranges] for name, subset in subsets.items()}
        return {
            name: tuple(b + s for b, s in zip(bounding_boxes[name], origins[name]))
            for name in bounding_boxes.keys()
        }

    def get_k_size(self, name):
        if not self._axes[name][2]:
            return None
        collection = self._get_access_collection(self._sdfg)
        min_k = min(o[2] if o[2] is not None else 0 for o in collection.offsets()[name])
        max_k = max(o[2] if o[2] is not None else 0 for o in collection.offsets()[name])
        if any(o[2] is None for o in collection.offsets()[name]):
            return f"({str(max_k - min_k)}) + __K"
        else:
            return str(max_k - min_k + 1)

    def get_k_subsets(self, node):
        assert isinstance(node, HorizontalExecutionLibraryNode)
        collection = self._get_access_collection(node)
        write_subsets = dict()
        read_subsets = dict()
        k_origins = dict()
        sdfg_collection = self._get_access_collection(self._sdfg)
        for name, offsets in sdfg_collection.offsets().items():
            k_origins[name] = (
                -min(o[2] for o in offsets) if all(o[2] is not None for o in offsets) else None
            )
        for name, offsets in collection.read_offsets().items():
            if self._axes[name][2]:
                if all(o[2] is not None for o in offsets):
                    read_subsets[name] = "{origin}{min_off:+d}:{origin}{max_off:+d}".format(
                        origin=k_origins[name],
                        min_off=min(o[2] for o in offsets),
                        max_off=max(o[2] for o in offsets) + 1,
                    )
                else:
                    read_subsets[name] = None
        for name in collection.write_fields():
            if self._axes[name][2]:
                if k_origins[name] is not None:
                    write_subsets[name] = "{origin}:{origin}+1".format(origin=k_origins[name])
                else:
                    write_subsets[name] = None
        return read_subsets, write_subsets

    def add_read_edges(self, node):
        interval = Interval.full()
        return self._add_read_edges(node, [(interval, self._get_access_collection(node))])

    def add_write_edges(self, node):
        interval = Interval.full()
        return self._add_write_edges(node, [(interval, self._get_access_collection(node))])

    def add_write_after_write_edges(self, node):
        interval = Interval.full()
        return self._add_write_after_write_edges(
            node, [(interval, self._get_access_collection(node))]
        )

    def add_write_after_read_edges(self, node):
        interval = Interval.full()
        return self._add_write_after_read_edges(
            node, [(interval, self._get_access_collection(node))]
        )

    def get_field_extents(self, node):
        assert isinstance(node, HorizontalExecutionLibraryNode)
        input_extents = dict()
        output_extents = dict()

        block_extent: Extent = node.extent
        assert block_extent is not None
        collection = self._get_access_collection(node)

        for acc in collection.read_accesses():
            extent = acc.to_extent(block_extent) | Extent.zeros(2)
            input_extents.setdefault(acc.field, extent)
            input_extents[acc.field] |= extent
        for acc in collection.write_accesses():
            extent = acc.to_extent(block_extent) | Extent.zeros(2)
            output_extents.setdefault(acc.field, extent)
            output_extents[acc.field] |= extent

        return input_extents, output_extents


class StencilOirSDFGBuilder(BaseOirSDFGBuilder):
    def get_shapes(self):
        shapes = dict()
        for decl in self._stencil.params + self._stencil.declarations:
            name = decl.name
            if name not in self._axes:
                continue
            shape = []
            if self._axes[name][0]:
                di = self._extents[name].frame_size[0]
                shape.append(f"__I{di:+d}")
            if self._axes[name][1]:
                dj = self._extents[name].frame_size[1]
                shape.append(f"__J{dj:+d}")
            if self._axes[name][2]:
                shape.append(self.get_k_size(name))
            for dim in decl.data_dims:
                shape.append(str(dim))
            shapes[name] = shape
        return shapes

    def get_k_size(self, name):

        if not self._axes[name][2]:
            return None

        axis_idx = sum(self._axes[name][:2])

        subset = None
        for edge in self._state.edges():
            if edge.data.data is not None and edge.data.data == name:
                if subset is None:
                    subset = edge.data.subset
                subset = dace.subsets.union(subset, edge.data.subset)
        subset: dace.subsets.Range
        k_size = subset.bounding_box_size()[axis_idx] + subset.ranges[axis_idx][0]

        k_sym = dace.symbol("__K")
        k_size_symbolic = dace.symbolic.pystr_to_symbolic(k_size)
        # this is the right way to check conditions with sympy (therefore the noqa)
        if (
            k_sym in k_size_symbolic.free_symbols
            and (k_size_symbolic >= k_sym)
            == True  # noqa: E712  # comparison to True should be 'if cond is True:' or 'if cond:'
        ):
            return k_size
        else:
            return "__K"

    def get_k_subsets(self, node):
        assert isinstance(node, VerticalLoopLibraryNode)

        write_intervals = dict()
        read_intervals = dict()
        k_origins = dict()
        dynamic_read_intervals = dict()
        for interval, sdfg in node.sections:
            collection = self._get_access_collection(sdfg)
            for name, offsets in collection.read_offsets().items():
                if self._axes[name][2]:
                    for offset in offsets:
                        k_offset = 0 if offset[2] is None else offset[2]
                        read_interval = interval.shifted(k_offset)
                        if offset[2] is None:
                            dynamic_read_intervals.setdefault(name, interval)
                            dynamic_read_intervals[name] = Interval(
                                start=min(dynamic_read_intervals[name].start, interval.start),
                                end=max(dynamic_read_intervals[name].end, interval.end),
                            )
                        read_intervals.setdefault(name, read_interval)
                        start = (
                            min(read_intervals[name].start, read_interval.start)
                            if read_intervals[name].start is not None
                            and read_interval.start is not None
                            else None
                        )
                        end = (
                            max(read_intervals[name].end, read_interval.end)
                            if read_intervals[name].end is not None
                            and read_interval.end is not None
                            else None
                        )
                        read_intervals[name] = oir.UnboundedInterval(
                            start=start,
                            end=end,
                        )
                        k_origins.setdefault(name, read_interval.start)
                        if read_interval.start is not None:
                            if k_origins[name] is None:
                                k_origins[name] = read_interval.start
                            else:
                                k_origins[name] = min(k_origins[name], read_interval.start)

            for name in collection.write_fields():
                if self._axes[name][2]:
                    write_intervals.setdefault(name, interval)
                    write_intervals[name] = Interval(
                        start=min(write_intervals[name].start, interval.start),
                        end=max(write_intervals[name].end, interval.end),
                    )
                    k_origins[name] = min(k_origins.get(name, interval.start), interval.start)
        write_subsets = dict()
        for name, interval in write_intervals.items():
            res_interval = _offset_origin(interval, k_origins[name])
            write_subsets[name] = "{}{:+d}:{}{:+d}".format(
                "__K" if res_interval.start.level == LevelMarker.END else "",
                res_interval.start.offset,
                "__K" if res_interval.end.level == LevelMarker.END else "",
                res_interval.end.offset,
            )
        read_subsets = dict()
        for name, interval in read_intervals.items():
            res_interval = _offset_origin(interval, k_origins[name])
            if name in dynamic_read_intervals:
                dyn_interval = dynamic_read_intervals[name]
                res_interval = Interval(
                    start=min(res_interval.start, dyn_interval.start),
                    end=max(res_interval.end, dyn_interval.end),
                )
            if res_interval.start is None:
                interval_start_str = "0"
            else:
                interval_start_str = "{}{:+d}".format(
                    "__K" if res_interval.start.level == LevelMarker.END else "",
                    res_interval.start.offset,
                )
            if res_interval.end is None:
                interval_end_str = "__K"
            else:
                interval_end_str = "{}{:+d}".format(
                    "__K" if res_interval.end.level == LevelMarker.END else "",
                    res_interval.end.offset,
                )

            read_subsets[name] = f"{interval_start_str}:{interval_end_str}"
        return read_subsets, write_subsets

    def get_field_extents(self, node):
        assert isinstance(node, VerticalLoopLibraryNode)
        input_extents = dict()
        output_extents = dict()
        for _, sdfg in node.sections:
            for n in sdfg.states()[0].nodes():

                if not isinstance(n, HorizontalExecutionLibraryNode):
                    continue
                block_extent = n.extent
                assert block_extent is not None
                collection = self._get_access_collection(n)
                for acc in collection.read_accesses():
                    extent = acc.to_extent(block_extent) | Extent.zeros(2)
                    input_extents.setdefault(acc.field, extent)
                    input_extents[acc.field] |= extent
                for acc in collection.write_accesses():
                    extent = acc.to_extent(block_extent) | Extent.zeros(2)
                    output_extents.setdefault(acc.field, extent)
                    output_extents[acc.field] |= extent
        return input_extents, output_extents

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


class OirSDFGBuilder(eve.NodeVisitor):
    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, *, block_extents, **kwargs):
        return HorizontalExecutionLibraryNode(
            name=f"HorizontalExecution_{id(node)}",
            oir_node=node,
            extent=block_extents[id(node)],
        )

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection, **kwargs):
        library_nodes = [self.visit(he, **kwargs) for he in node.horizontal_executions]
        sdfg = VerticalLoopSectionOirSDFGBuilder.build(
            f"VerticalLoopSection_{id(node)}", kwargs["stencil"], library_nodes
        )
        return node.interval, sdfg

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs):
        sections = [self.visit(section, **kwargs) for section in node.sections]
        return VerticalLoopLibraryNode(
            name=f"VerticalLoop_{id(node)}",
            loop_order=node.loop_order,
            sections=sections,
            caches=node.caches,
            oir_node=node,
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs):
        from gtc.passes.oir_optimizations.utils import compute_horizontal_block_extents

        block_extents = compute_horizontal_block_extents(node)
        library_nodes = [
            self.visit(vl, stencil=node, block_extents=block_extents, **kwargs)
            for vl in node.vertical_loops
        ]
        return StencilOirSDFGBuilder.build(node.name, node, library_nodes)
