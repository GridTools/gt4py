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
from typing import TYPE_CHECKING, Any, Collection, Dict, Iterator, List, Tuple, Union

import dace
import dace.data
import networkx as nx
import numpy as np
from dace import SDFG, InterstateEdge
from pydantic import validator

import eve
import gtc.oir as oir
from eve.iterators import TraversalOrder, iter_tree
from gtc import common
from gtc.common import CartesianOffset, DataType, ExprKind, LevelMarker, typestr_to_data_type
from gtc.passes.oir_optimizations.utils import AccessCollector, GenericAccess


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


class CartesianIterationSpace(oir.LocNode):
    i_interval: oir.Interval
    j_interval: oir.Interval

    @validator("i_interval", "j_interval")
    def minimum_domain(cls, v: oir.Interval) -> oir.Interval:
        if (
            v.start.level != LevelMarker.START
            or v.start.offset > 0
            or v.end.level != LevelMarker.END
            or v.end.offset < 0
        ):
            raise ValueError("iteration space must include the whole domain")
        return v

    @staticmethod
    def domain() -> "CartesianIterationSpace":
        return CartesianIterationSpace(
            i_interval=oir.Interval(start=oir.AxisBound.start(), end=oir.AxisBound.end()),
            j_interval=oir.Interval(start=oir.AxisBound.start(), end=oir.AxisBound.end()),
        )

    @staticmethod
    def from_offset(offset: CartesianOffset) -> "CartesianIterationSpace":

        return CartesianIterationSpace(
            i_interval=oir.Interval(
                start=oir.AxisBound.from_start(min(0, offset.i)),
                end=oir.AxisBound.from_end(max(0, offset.i)),
            ),
            j_interval=oir.Interval(
                start=oir.AxisBound.from_start(min(0, offset.j)),
                end=oir.AxisBound.from_end(max(0, offset.j)),
            ),
        )

    def compose(self, other: "CartesianIterationSpace") -> "CartesianIterationSpace":
        i_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                self.i_interval.start.offset + other.i_interval.start.offset,
            ),
            end=oir.AxisBound.from_end(
                self.i_interval.end.offset + other.i_interval.end.offset,
            ),
        )
        j_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                self.j_interval.start.offset + other.j_interval.start.offset,
            ),
            end=oir.AxisBound.from_end(
                self.j_interval.end.offset + other.j_interval.end.offset,
            ),
        )
        return CartesianIterationSpace(i_interval=i_interval, j_interval=j_interval)

    def __or__(self, other: "CartesianIterationSpace") -> "CartesianIterationSpace":
        i_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                min(
                    self.i_interval.start.offset,
                    other.i_interval.start.offset,
                )
            ),
            end=oir.AxisBound.from_end(
                max(
                    self.i_interval.end.offset,
                    other.i_interval.end.offset,
                )
            ),
        )
        j_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                min(
                    self.j_interval.start.offset,
                    other.j_interval.start.offset,
                )
            ),
            end=oir.AxisBound.from_end(
                max(
                    self.j_interval.end.offset,
                    other.j_interval.end.offset,
                )
            ),
        )
        return CartesianIterationSpace(i_interval=i_interval, j_interval=j_interval)

    def __and__(self, other: "CartesianIterationSpace") -> "CartesianIterationSpace":
        i_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                max(
                    self.i_interval.start.offset,
                    other.i_interval.start.offset,
                )
            ),
            end=oir.AxisBound.from_end(
                min(
                    self.i_interval.end.offset,
                    other.i_interval.end.offset,
                )
            ),
        )
        j_interval = oir.Interval(
            start=oir.AxisBound.from_start(
                max(
                    self.j_interval.start.offset,
                    other.j_interval.start.offset,
                )
            ),
            end=oir.AxisBound.from_end(
                min(
                    self.j_interval.end.offset,
                    other.j_interval.end.offset,
                )
            ),
        )
        return CartesianIterationSpace(i_interval=i_interval, j_interval=j_interval)


class CartesianIJIndexSpace(tuple):
    def __new__(cls, *args, **kwargs):
        return super(CartesianIJIndexSpace, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        msg = "CartesianIJIndexSpace must be a pair of pairs of integers."
        if not len(self) == 2:
            raise ValueError(msg)
        if not all(len(v) == 2 for v in self):
            raise ValueError(msg)
        if not all(isinstance(v[0], int) and isinstance(v[1], int) for v in self):
            raise ValueError(msg)

    @staticmethod
    def from_offset(offset: Union[Tuple[int, ...], CartesianOffset]) -> "CartesianIJIndexSpace":
        if isinstance(offset, CartesianOffset):
            res = ((offset.i, offset.i), (offset.j, offset.j))
        else:
            res = ((offset[0], offset[0]), (offset[1], offset[1]))

        return CartesianIJIndexSpace(
            (
                (
                    min(res[0][0], 0),
                    max(res[0][1], 0),
                ),
                (
                    min(res[1][0], 0),
                    max(res[1][1], 0),
                ),
            )
        )

    @staticmethod
    def from_access(access: GenericAccess):
        if access.region is None:
            return CartesianIJIndexSpace.from_offset(access.offset)

        res = []

        for interval, off in zip((access.region.i, access.region.j), access.offset[:2]):
            dim_tuple = [0, 0]
            if interval.start is not None:
                if interval.start.level == common.LevelMarker.START:
                    dim_tuple[0] = min(0, off + interval.start.offset)
                else:
                    dim_tuple[0] = 0
            else:
                dim_tuple[0] = min(0, off)
            if interval.end is not None:
                if interval.end.level == common.LevelMarker.END:
                    dim_tuple[1] = max(0, off + interval.end.offset)
                else:
                    dim_tuple[1] = 0
            else:
                dim_tuple[1] = min(0, off)
            res.append(tuple(dim_tuple))

        return CartesianIJIndexSpace(res)

    @staticmethod
    def from_iteration_space(iteration_space: CartesianIterationSpace) -> "CartesianIJIndexSpace":
        return CartesianIJIndexSpace(
            (
                (iteration_space.i_interval.start.offset, iteration_space.i_interval.end.offset),
                (iteration_space.j_interval.start.offset, iteration_space.j_interval.end.offset),
            )
        )

    def compose(
        self, other: Union["CartesianIterationSpace", "CartesianIJIndexSpace"]
    ) -> "CartesianIJIndexSpace":
        if isinstance(other, CartesianIterationSpace):
            other = CartesianIJIndexSpace.from_iteration_space(other)
        return CartesianIJIndexSpace(
            (
                (self[0][0] + other[0][0], (self[0][1] + other[0][1])),
                (self[1][0] + other[1][0], (self[1][1] + other[1][1])),
            )
        )

    def __or__(
        self, other: Union["CartesianIterationSpace", "CartesianIJIndexSpace"]
    ) -> "CartesianIJIndexSpace":
        if isinstance(other, CartesianIterationSpace):
            other = CartesianIJIndexSpace.from_iteration_space(other)
        return CartesianIJIndexSpace(
            (
                (min(self[0][0], other[0][0]), max(self[0][1], other[0][1])),
                (min(self[1][0], other[1][0]), max(self[1][1], other[1][1])),
            )
        )


def iteration_to_access_space(iteration_space: CartesianIJIndexSpace, access: GenericAccess):
    if access.region is None:
        return CartesianIJIndexSpace.from_access(access).compose(iteration_space)

    res = []

    for region_interval, index_interval, off in zip(
        (access.region.i, access.region.j),
        iteration_space,
        access.offset[:2],
    ):
        dim_tuple = list(index_interval)

        if region_interval.start is not None:
            if region_interval.start.level == common.LevelMarker.START:
                dim_tuple[0] = min(
                    0, max(index_interval[0] + off, region_interval.start.offset + off)
                )
        else:
            dim_tuple[0] += min(0, off)

        if region_interval.end is not None:
            if region_interval.end.level == common.LevelMarker.END:

                dim_tuple[1] = max(
                    0, min(index_interval[1] + off, region_interval.end.offset + off)
                )
        else:
            dim_tuple[1] += min(0, off)

        res.append(tuple(dim_tuple))

    return CartesianIJIndexSpace(res)


def oir_iteration_space_computation(stencil: oir.Stencil) -> Dict[int, CartesianIterationSpace]:
    iteration_spaces = dict()

    offsets: Dict[str, List[CartesianOffset]] = dict()
    outputs = set()
    access_spaces: Dict[str, CartesianIterationSpace] = dict()
    # reversed pre_order traversal is post-order but the last nodes per node come first.
    # this is not possible with visitors unless a (reverseed) visit is implemented for every node
    for node in reversed(list(iter_tree(stencil, traversal_order=TraversalOrder.PRE_ORDER))):
        if isinstance(node, oir.FieldAccess):
            if node.name not in offsets:
                offsets[node.name] = list()
            offsets[node.name].append(node.offset)
        elif isinstance(node, oir.AssignStmt):
            if node.left.kind == ExprKind.FIELD:
                outputs.add(node.left.name)
        elif isinstance(node, oir.HorizontalExecution):
            iteration_spaces[id(node)] = CartesianIterationSpace.domain()
            for name in outputs:
                access_space = access_spaces.get(name, CartesianIterationSpace.domain())
                iteration_spaces[id(node)] = iteration_spaces[id(node)] | access_space
            for name in offsets:
                access_space = CartesianIterationSpace.domain()
                for offset in offsets[name]:
                    access_space = access_space | CartesianIterationSpace.from_offset(offset)
                accumulated_extent = iteration_spaces[id(node)].compose(access_space)
                access_spaces[name] = (
                    access_spaces.get(name, CartesianIterationSpace.domain()) | accumulated_extent
                )

            offsets = dict()
            outputs = set()

    return iteration_spaces


def oir_field_boundary_computation(stencil: oir.Stencil) -> Dict[str, CartesianIterationSpace]:
    offsets: Dict[str, List[CartesianOffset]] = dict()
    access_spaces: Dict[str, CartesianIterationSpace] = dict()
    iteration_spaces = oir_iteration_space_computation(stencil)
    for node in iter_tree(stencil, traversal_order=TraversalOrder.POST_ORDER):
        if isinstance(node, oir.FieldAccess):
            if node.name not in offsets:
                offsets[node.name] = list()
            offsets[node.name].append(node.offset)
        elif isinstance(node, oir.HorizontalExecution):
            if iteration_spaces.get(id(node), None) is not None:
                for name in offsets:
                    access_space = CartesianIterationSpace.domain()
                    for offset in offsets[name]:
                        access_space = access_space | CartesianIterationSpace.from_offset(offset)
                    access_spaces[name] = access_spaces.get(
                        name, CartesianIterationSpace.domain()
                    ) | (iteration_spaces[id(node)].compose(access_space))

            offsets = dict()

    return access_spaces


def get_access_collection(
    node: Union[dace.SDFG, "HorizontalExecutionLibraryNode", "VerticalLoopLibraryNode"],
    compensate_regions: bool = False,
):
    from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode

    if isinstance(node, dace.SDFG):
        res = AccessCollector.CartesianAccessCollection([])
        for node, _ in node.all_nodes_recursive():
            if isinstance(node, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode)):
                collection = get_access_collection(node)
                res._ordered_accesses.extend(collection._ordered_accesses)
        return res
    elif isinstance(node, HorizontalExecutionLibraryNode):
        return AccessCollector.apply(node.oir_node, compensate_regions=compensate_regions)
    else:
        assert isinstance(node, VerticalLoopLibraryNode)
        res = AccessCollector.CartesianAccessCollection([])
        for _, sdfg in node.sections:
            collection = get_access_collection(sdfg)
            res._ordered_accesses.extend(collection._ordered_accesses)
        return res


def nodes_extent_calculation(
    nodes: Collection[Union["VerticalLoopLibraryNode", "HorizontalExecutionLibraryNode"]]
) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
    access_spaces: Dict[str, Tuple[Tuple[int, int], ...]] = dict()
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
        iteration_space = node.iteration_space
        if iteration_space is not None:
            for acc in access_collection.ordered_accesses():
                if acc.region is None:

                    access_extent: List[Tuple[int, int]] = [
                        (
                            min(0, iteration_space.i_interval.start.offset + acc.offset[0]),
                            max(0, iteration_space.i_interval.end.offset + acc.offset[0]),
                        ),
                        (
                            min(0, iteration_space.j_interval.start.offset + acc.offset[1]),
                            max(0, iteration_space.j_interval.end.offset + acc.offset[1]),
                        ),
                    ]
                else:
                    access_extent = []
                    for dim, region_interval, iteration_interval in zip(
                        (0, 1),
                        (acc.region.i, acc.region.j),
                        (iteration_space.i_interval, iteration_space.j_interval),
                    ):
                        ext = [0, 0]

                        ext[0] = iteration_interval.start.offset
                        if region_interval.start is not None:
                            if region_interval.start.level == common.LevelMarker.START:
                                ext[0] = max(
                                    ext[0] + acc.offset[dim],
                                    region_interval.start.offset + acc.offset[dim],
                                )
                        else:
                            ext[0] += acc.offset[dim]
                        ext[0] = min(0, ext[0])

                        ext[1] = iteration_interval.end.offset
                        if region_interval.end is not None:
                            if region_interval.end.level == common.LevelMarker.END:
                                ext[1] = min(
                                    ext[1] + acc.offset[dim],
                                    region_interval.end.offset + acc.offset[dim],
                                )
                        else:
                            ext[1] += acc.offset[dim]
                        ext[1] = max(0, ext[1])

                        access_extent.append((ext[0], ext[1]))

                if acc.field not in access_spaces:
                    access_spaces[acc.field] = tuple(access_extent)
                access_spaces[acc.field] = tuple(
                    (min(asp[0], ext[0]), max(asp[1], ext[1]))
                    for asp, ext in zip(access_spaces[acc.field], access_extent)
                )

    return {
        name: ((-asp[0][0], asp[0][1]), (-asp[1][0], asp[1][1]))
        for name, asp in access_spaces.items()
    }


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

    try:
        assert isinstance(n2, VerticalLoopLibraryNode)
        assert n1.loop_order == n2.loop_order
        assert n1.caches == n2.caches
        assert len(n1.sections) == len(n2.sections)
        for (interval1, he_sdfg1), (interval2, he_sdfg2) in zip(n1.sections, n2.sections):
            assert interval1 == interval2
            assert_sdfg_equal(he_sdfg1, he_sdfg2)
    except AssertionError:
        return False
    return True


def equal_he_node(n1: "HorizontalExecutionLibraryNode", n2: "HorizontalExecutionLibraryNode"):
    from gtc.dace.nodes import HorizontalExecutionLibraryNode

    try:
        assert isinstance(n2, HorizontalExecutionLibraryNode)
        assert n1.as_oir() == n2.as_oir()
    except AssertionError:
        return False
    return True


def assert_sdfg_equal(sdfg1: dace.SDFG, sdfg2: dace.SDFG):
    from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode

    def edge_match(edge1, edge2):
        edge1 = next(iter(edge1.values()))
        edge2 = next(iter(edge2.values()))
        try:
            if edge1["src_conn"] is not None:
                assert edge2["src_conn"] is not None
                assert edge1["src_conn"] == edge2["src_conn"]
            else:
                assert edge2["src_conn"] is None
            assert edge1["data"] == edge2["data"]
            assert edge1["data"].data == edge2["data"].data
        except AssertionError:
            return False
        return True

    def node_match(n1, n2):
        n1 = n1["node"]
        n2 = n2["node"]
        try:
            if not isinstance(
                n1, (dace.nodes.AccessNode, VerticalLoopLibraryNode, HorizontalExecutionLibraryNode)
            ):
                raise TypeError
            if isinstance(n1, dace.nodes.AccessNode):
                assert isinstance(n2, dace.nodes.AccessNode)
                assert n1.data == n2.data
            elif isinstance(n1, VerticalLoopLibraryNode):
                assert equal_vl_node(n1, n2)
            elif isinstance(n1, HorizontalExecutionLibraryNode):
                assert equal_he_node(n1, n2)
        except AssertionError:
            return False
        return True

    assert len(sdfg1.states()) == 1
    assert len(sdfg2.states()) == 1
    state1 = sdfg1.states()[0]
    state2 = sdfg2.states()[0]

    # SDFGState.nx does not contain any node info in the networkx node attrs (but does for edges),
    # so we add it here manually.
    nx.set_node_attributes(state1.nx, {n: n for n in state1.nx.nodes}, "node")
    nx.set_node_attributes(state2.nx, {n: n for n in state2.nx.nodes}, "node")

    assert nx.is_isomorphic(state1.nx, state2.nx, edge_match=edge_match, node_match=node_match)

    for name in sdfg1.arrays.keys():
        assert isinstance(sdfg1.arrays[name], type(sdfg2.arrays[name]))
        assert isinstance(sdfg2.arrays[name], type(sdfg1.arrays[name]))
        assert sdfg1.arrays[name].dtype == sdfg2.arrays[name].dtype
        assert sdfg1.arrays[name].transient == sdfg2.arrays[name].transient
        assert sdfg1.arrays[name].shape == sdfg2.arrays[name].shape
