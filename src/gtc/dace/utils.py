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

import re
from typing import TYPE_CHECKING, Any, Collection, Dict, Iterator, List, Union

import dace
import dace.data
import networkx as nx
import numpy as np
from dace import SDFG, InterstateEdge

import eve
import gtc.oir as oir
from gt4py.definitions import Extent
from gtc.common import DataType, typestr_to_data_type
from gtc.passes.oir_optimizations.utils import AccessCollector


if TYPE_CHECKING:
    from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode
    from gtc.oir import VerticalLoopSection


def internal_symbols(sdfg: dace.SDFG):
    res = ["__I", "__J", "__K"]
    for name, array in sdfg.arrays.items():
        if isinstance(array, dace.data.Array):
            dimensions = array_dimensions(array)
            n_data_dims = len(array.shape) - sum(dimensions)
            res.extend(
                [f"__{name}_{var}_stride" for idx, var in enumerate("IJK") if dimensions[idx]]
            )
            res.extend([f"__{name}_d{dim}_stride" for dim in range(n_data_dims)])
    return res


def validate_oir_sdfg(sdfg: dace.SDFG):

    from gtc.dace.nodes import VerticalLoopLibraryNode

    sdfg.validate()
    is_correct_node_types = all(
        isinstance(n, (dace.SDFGState, dace.nodes.AccessNode, VerticalLoopLibraryNode))
        for n, _ in sdfg.all_nodes_recursive()
    )
    is_correct_data_and_dtype = all(
        isinstance(array, dace.data.Array)
        and typestr_to_data_type(dace_dtype_to_typestr(array.dtype)) != DataType.INVALID
        for array in sdfg.arrays.values()
    )
    if not is_correct_node_types or not is_correct_data_and_dtype:
        raise ValueError("Not a valid OIR-level SDFG")


def dace_dtype_to_typestr(dtype: Any):
    if not isinstance(dtype, dace.typeclass):
        dtype = dace.typeclass(dtype)
    return dtype.as_numpy_dtype().str


def array_dimensions(array: dace.data.Array):
    dims = [
        any(
            re.match(f"__.*_{k}_stride", str(sym))
            for st in array.strides
            for sym in st.free_symbols
        )
        for k in "IJK"
    ]
    return dims


def replace_strides(arrays, get_layout_map):
    symbol_mapping = {}
    for array in arrays:
        dims = array_dimensions(array)
        ndata_dims = len(array.shape) - sum(dims)
        layout = get_layout_map(dims + [True] * ndata_dims)
        if array.transient:
            stride = 1
            for idx in reversed(np.argsort(layout)):
                symbol = array.strides[idx]
                size = array.shape[idx]
                symbol_mapping[str(symbol)] = stride
                stride *= size
    return symbol_mapping


def get_tasklet_symbol(name, offset, is_target):
    if is_target:
        return f"__{name}"

    acc_name = name + "__"
    offset_strs = []
    for var, o in zip("ijk", offset):
        if o is not None and o != 0:
            offset_strs.append(var + ("m" if o < 0 else "p") + f"{abs(o):d}")
    suffix = "_".join(offset_strs)
    if suffix != "":
        acc_name += suffix
    return acc_name


def get_axis_bound_str(axis_bound, var_name):
    from gtc.common import LevelMarker

    if axis_bound is None:
        return ""
    elif axis_bound.level == LevelMarker.END:
        return f"{var_name}{axis_bound.offset:+d}"
    else:
        return f"{axis_bound.offset}"


def get_interval_range_str(interval, var_name):
    return "{}:{}".format(
        get_axis_bound_str(interval.start, var_name), get_axis_bound_str(interval.end, var_name)
    )


def get_axis_bound_diff_str(axis_bound1, axis_bound2, var_name: str):

    if axis_bound1 >= axis_bound2:
        tmp = axis_bound2
        axis_bound2 = axis_bound1
        axis_bound1 = tmp
        sign = "-"
    else:
        sign = ""

    if axis_bound1.level != axis_bound2.level:
        var = var_name
    else:
        var = ""
    return f"{sign}{var}{axis_bound2.offset-axis_bound1.offset:+d}"


def get_interval_length_str(interval, var_name):

    return "({})-({})".format(
        get_axis_bound_str(interval.end, var_name), get_axis_bound_str(interval.start, var_name)
    )


def get_vertical_loop_section_sdfg(section: "VerticalLoopSection") -> SDFG:
    from gtc.dace.nodes import HorizontalExecutionLibraryNode

    sdfg = SDFG("VerticalLoopSection_" + str(id(section)))
    old_state = sdfg.add_state("start_state", is_start_state=True)
    for he in section.horizontal_executions:
        new_state = sdfg.add_state("HorizontalExecution_" + str(id(he)) + "_state")
        sdfg.add_edge(old_state, new_state, InterstateEdge())
        new_state.add_node(HorizontalExecutionLibraryNode(oir_node=he))

        old_state = new_state
    return sdfg


class OIRFieldRenamer(eve.NodeTranslator):
    def visit_FieldAccess(self, node: oir.FieldAccess):
        if node.name not in self._field_table:
            return node
        return oir.FieldAccess(
            name=self._field_table[node.name],
            offset=node.offset,
            dtype=node.dtype,
            data_index=node.data_index,
        )

    def visit_ScalarAccess(self, node: oir.ScalarAccess):
        if node.name not in self._field_table:
            return node
        return oir.ScalarAccess(name=self._field_table[node.name], dtype=node.dtype)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution):
        assert all(
            decl.name not in self._field_table or self._field_table[decl.name] == decl.name
            for decl in node.declarations
        )
        return node

    def __init__(self, field_table):
        self._field_table = field_table


def get_node_name_mapping(state: dace.SDFGState, node: dace.nodes.LibraryNode):

    name_mapping = dict()
    for edge in state.in_edges(node):
        if edge.dst_conn is not None:
            assert edge.dst_conn.startswith("IN_")
            internal_name = edge.dst_conn[len("IN_") :]
            outer_name = edge.data.data
            if internal_name not in name_mapping:
                name_mapping[internal_name] = outer_name
            else:
                msg = (
                    f"input and output of field '{internal_name}' to node'{node.name}' refer to "
                    + "different arrays"
                )
                assert name_mapping[internal_name] == outer_name, msg
    for edge in state.out_edges(node):
        if edge.src_conn is not None:
            assert edge.src_conn.startswith("OUT_")
            internal_name = edge.src_conn[len("OUT_") :]
            outer_name = edge.data.data
            if internal_name not in name_mapping:
                name_mapping[internal_name] = outer_name
            else:
                msg = (
                    f"input and output of field '{internal_name}' to node'{node.name}' refer to"
                    + "different arrays"
                )
                assert name_mapping[internal_name] == outer_name, msg
    return name_mapping


def get_access_collection(
    node: Union[dace.SDFG, "HorizontalExecutionLibraryNode", "VerticalLoopLibraryNode"],
):
    from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode

    if isinstance(node, dace.SDFG):
        res = AccessCollector.CartesianAccessCollection([])
        for n, _ in node.all_nodes_recursive():
            if isinstance(n, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode)):
                collection = get_access_collection(n)
                res._ordered_accesses.extend(collection._ordered_accesses)
        return res
    elif isinstance(node, HorizontalExecutionLibraryNode):
        return AccessCollector.apply(node.oir_node)
    else:
        assert isinstance(node, VerticalLoopLibraryNode)
        res = AccessCollector.CartesianAccessCollection([])
        for _, sdfg in node.sections:
            collection = get_access_collection(sdfg)
            res._ordered_accesses.extend(collection._ordered_accesses)
        return res


def nodes_extent_calculation(
    nodes: Collection[Union["VerticalLoopLibraryNode", "HorizontalExecutionLibraryNode"]]
) -> Dict[str, Extent]:
    field_extents: Dict[str, Extent] = dict()
    inner_nodes = []
    from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode

    for node in nodes:
        if isinstance(node, VerticalLoopLibraryNode):
            for _, section_sdfg in node.sections:
                for he in (
                    ln
                    for ln, _ in section_sdfg.all_nodes_recursive()
                    if isinstance(ln, dace.nodes.LibraryNode)
                ):
                    inner_nodes.append(he)
        else:
            assert isinstance(node, HorizontalExecutionLibraryNode)
            inner_nodes.append(node)
    for node in inner_nodes:
        access_collection = AccessCollector.apply(node.oir_node)
        block_extent = node.extent
        if block_extent is not None:
            for acc in access_collection.ordered_accesses():
                offset_extent = acc.to_extent(block_extent) | Extent.zeros(2)
                field_extents.setdefault(acc.field, offset_extent)
                field_extents[acc.field] |= offset_extent

    return field_extents


def iter_vertical_loop_section_sub_sdfgs(graph: SDFG) -> Iterator[SDFG]:
    from gtc.dace.nodes import VerticalLoopLibraryNode

    for node, _ in graph.all_nodes_recursive():
        if isinstance(node, VerticalLoopLibraryNode):
            yield from (subgraph for _, subgraph in node.sections)


class IntervalMapping:
    def __init__(self) -> None:
        self.interval_starts: List[oir.AxisBound] = list()
        self.interval_ends: List[oir.AxisBound] = list()
        self.values: List[Any] = list()

    def _setitem_subset_of_existing(self, i: int, key: oir.Interval, value: Any) -> None:
        start = self.interval_starts[i]
        end = self.interval_ends[i]
        if self.values[i] is not value:
            idx = i
            if key.start != start:
                self.interval_ends[i] = key.start
                self.interval_starts.insert(i + 1, key.start)
                self.interval_ends.insert(i + 1, key.end)
                self.values.insert(i + 1, value)
                idx = i + 1
            if key.end != end:
                self.interval_starts.insert(idx + 1, key.end)
                self.interval_ends.insert(idx + 1, end)
                self.values.insert(idx + 1, self.values[i])
                self.interval_ends[idx] = key.end
                self.values[idx] = value

    def _setitem_partial_overlap(self, i: int, key: oir.Interval, value: Any) -> None:
        start = self.interval_starts[i]
        if key.start < start:
            if self.values[i] is value:
                self.interval_starts[i] = key.start
            else:
                self.interval_starts[i] = key.end
                self.interval_starts.insert(i, key.start)
                self.interval_ends.insert(i, key.end)
                self.values.insert(i, value)
        else:  # key.end > end
            if self.values[i] is value:
                self.interval_ends[i] = key.end
                nextidx = i + 1
            else:
                self.interval_ends[i] = key.start
                self.interval_starts.insert(i + 1, key.start)
                self.interval_ends.insert(i + 1, key.end)
                self.values.insert(i + 1, value)
                nextidx = i + 2
            if nextidx < len(self.interval_starts) and (
                key.intersects(
                    oir.Interval(
                        start=self.interval_starts[nextidx], end=self.interval_ends[nextidx]
                    )
                )
                or self.interval_starts[nextidx] == key.end
            ):
                if self.values[nextidx] is value:
                    self.interval_ends[nextidx - 1] = self.interval_ends[nextidx]
                    del self.interval_starts[nextidx]
                    del self.interval_ends[nextidx]
                    del self.values[nextidx]
                else:
                    self.interval_starts[nextidx] = key.end

    def __setitem__(self, key: oir.Interval, value: Any) -> None:
        if not isinstance(key, oir.Interval):
            raise TypeError("Only OIR intervals supported for method add of IntervalSet.")
        key = oir.UnboundedInterval(start=key.start, end=key.end)
        delete = list()
        for i, (start, end) in enumerate(zip(self.interval_starts, self.interval_ends)):
            if key.covers(oir.UnboundedInterval(start=start, end=end)):
                delete.append(i)

        for i in reversed(delete):  # so indices keep validity while deleting
            del self.interval_starts[i]
            del self.interval_ends[i]
            del self.values[i]

        if len(self.interval_starts) == 0:
            self.interval_starts.append(key.start)
            self.interval_ends.append(key.end)
            self.values.append(value)
            return

        for i, (start, end) in enumerate(zip(self.interval_starts, self.interval_ends)):
            if oir.UnboundedInterval(start=start, end=end).covers(key):
                self._setitem_subset_of_existing(i, key, value)
                return

        for i, (start, end) in enumerate(zip(self.interval_starts, self.interval_ends)):
            if (
                key.intersects(oir.UnboundedInterval(start=start, end=end))
                or start == key.end
                or end == key.start
            ):
                self._setitem_partial_overlap(i, key, value)
                return

        for i, start in enumerate(self.interval_starts):
            if start > key.start:
                self.interval_starts.insert(i, key.start)
                self.interval_ends.insert(i, key.end)
                self.values.insert(i, value)
                return
        self.interval_starts.append(key.start)
        self.interval_ends.append(key.end)
        self.values.append(value)
        return

    def __getitem__(self, key: oir.Interval) -> List[Any]:
        if not isinstance(key, oir.Interval):
            raise TypeError("Only OIR intervals supported for keys of IntervalMapping.")

        res = []
        key = oir.UnboundedInterval(start=key.start, end=key.end)
        for start, end, value in zip(self.interval_starts, self.interval_ends, self.values):
            if key.intersects(oir.UnboundedInterval(start=start, end=end)):
                res.append(value)
        return res


def equal_vl_node(n1: "VerticalLoopLibraryNode", n2: "VerticalLoopLibraryNode"):
    from gtc.dace.nodes import VerticalLoopLibraryNode

    if not (
        isinstance(n2, VerticalLoopLibraryNode)
        and n1.loop_order == n2.loop_order
        and n1.caches == n2.caches
        and len(n1.sections) == len(n2.sections)
    ):
        return False
    for (interval1, he_sdfg1), (interval2, he_sdfg2) in zip(n1.sections, n2.sections):
        if not interval1 == interval2 and is_sdfg_equal(he_sdfg1, he_sdfg2):
            return False
    return True


def equal_he_node(n1: "HorizontalExecutionLibraryNode", n2: "HorizontalExecutionLibraryNode"):
    from gtc.dace.nodes import HorizontalExecutionLibraryNode

    return isinstance(n2, HorizontalExecutionLibraryNode) and n1.as_oir() == n2.as_oir()


def edge_match(edge1, edge2):
    edge1 = next(iter(edge1.values()))
    edge2 = next(iter(edge2.values()))
    if edge1["src_conn"] is not None:
        if not (edge2["src_conn"] is not None and edge1["src_conn"] == edge2["src_conn"]):
            return False
    else:
        if edge2["src_conn"] is not None:
            return False
    if not (edge1["data"] == edge2["data"] and edge1["data"].data == edge2["data"].data):
        return False
    return True


def node_match(n1, n2):
    from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode

    n1 = n1["node"]
    n2 = n2["node"]
    if not isinstance(
        n1, (dace.nodes.AccessNode, VerticalLoopLibraryNode, HorizontalExecutionLibraryNode)
    ):
        raise TypeError
    if isinstance(n1, dace.nodes.AccessNode):
        if not (isinstance(n2, dace.nodes.AccessNode) and n1.data == n2.data):
            return False
    elif isinstance(n1, VerticalLoopLibraryNode):
        if not equal_vl_node(n1, n2):
            return False
    elif isinstance(n1, HorizontalExecutionLibraryNode):
        if not equal_he_node(n1, n2):
            return False
    return True


def is_sdfg_equal(sdfg1: dace.SDFG, sdfg2: dace.SDFG):

    if not (len(sdfg1.states()) == 1 and len(sdfg2.states()) == 1):
        return False
    state1 = sdfg1.states()[0]
    state2 = sdfg2.states()[0]

    # SDFGState.nx does not contain any node info in the networkx node attrs (but does for edges),
    # so we add it here manually.
    nx.set_node_attributes(state1.nx, {n: n for n in state1.nx.nodes}, "node")
    nx.set_node_attributes(state2.nx, {n: n for n in state2.nx.nodes}, "node")

    if not nx.is_isomorphic(state1.nx, state2.nx, edge_match=edge_match, node_match=node_match):
        return False

    for name in sdfg1.arrays.keys():
        if not (
            isinstance(sdfg1.arrays[name], type(sdfg2.arrays[name]))
            and isinstance(sdfg2.arrays[name], type(sdfg1.arrays[name]))
            and sdfg1.arrays[name].dtype == sdfg2.arrays[name].dtype
            and sdfg1.arrays[name].transient == sdfg2.arrays[name].transient
            and sdfg1.arrays[name].shape == sdfg2.arrays[name].shape
        ):
            return False
    return True
