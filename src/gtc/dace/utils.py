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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Collection, Dict, Iterator, List, Optional, Tuple, Union

import dace
import dace.data
import networkx as nx
import numpy as np
from dace import SDFG, InterstateEdge
from pydantic import validator

import eve
import gtc.oir as oir
from eve import NodeVisitor
from eve.iterators import TraversalOrder, iter_tree
from gt4py.definitions import Extent
from gtc import common
from gtc.common import (
    CartesianOffset,
    DataType,
    ExprKind,
    LevelMarker,
    data_type_to_typestr,
    typestr_to_data_type,
)
from gtc.passes.oir_optimizations.utils import AccessCollector, GenericAccess


if TYPE_CHECKING:
    from gtc import daceir as dcir
if TYPE_CHECKING:
    from .nodes import StencilComputation


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
            for sym in dace.symbolic.pystr_to_symbolic(st).free_symbols
        )
        or any(
            re.match(f"__{k}", str(sym))
            for sh in array.shape
            for sym in dace.symbolic.pystr_to_symbolic(sh).free_symbols
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
                if symbol.is_symbol:
                    symbol_mapping[str(symbol)] = dace.symbolic.pystr_to_symbolic(stride)
                stride *= array.shape[idx]
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
        get_axis_bound_str(interval.start, var_name),
        get_axis_bound_str(interval.end, var_name),
    )


def get_axis_bound_diff_str(axis_bound1, axis_bound2, var_name: str):

    if axis_bound1 <= axis_bound2:
        axis_bound1, axis_bound2 = axis_bound2, axis_bound1
        sign = "-"
    else:
        sign = ""

    if axis_bound1.level != axis_bound2.level:
        var = var_name
    else:
        var = ""
    return f"{sign}({var}{axis_bound1.offset-axis_bound2.offset:+d})"


def get_interval_length_str(interval, var_name):

    return "({})-({})".format(
        get_axis_bound_str(interval.end, var_name),
        get_axis_bound_str(interval.start, var_name),
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
            assert edge.dst_conn.startswith("__in_")
            internal_name = edge.dst_conn[len("__in_") :]
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
            assert edge.src_conn.startswith("__out_")
            internal_name = edge.src_conn[len("__out_") :]
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
    def from_iteration_space(
        iteration_space: CartesianIterationSpace,
    ) -> "CartesianIJIndexSpace":
        return CartesianIJIndexSpace(
            (
                (
                    iteration_space.i_interval.start.offset,
                    iteration_space.i_interval.end.offset,
                ),
                (
                    iteration_space.j_interval.start.offset,
                    iteration_space.j_interval.end.offset,
                ),
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


def oir_iteration_space_computation(
    stencil: oir.Stencil,
) -> Dict[int, CartesianIterationSpace]:
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


def oir_field_boundary_computation(
    stencil: oir.Stencil,
) -> Dict[str, CartesianIterationSpace]:
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
            # for name, offsets in access_collection.offsets().items():
            for acc in access_collection.ordered_accesses():
                if acc.region is None:

                    access_extent = [
                        (
                            min(
                                0,
                                iteration_space.i_interval.start.offset + acc.offset[0],
                            ),
                            max(0, iteration_space.i_interval.end.offset + acc.offset[0]),
                        ),
                        (
                            min(
                                0,
                                iteration_space.j_interval.start.offset + acc.offset[1],
                            ),
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

                        ext_tuple = (int(ext[0]), int(ext[1]))
                        access_extent.append(ext_tuple)

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
                        start=self.interval_starts[nextidx],
                        end=self.interval_ends[nextidx],
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


def assert_sdfg_equal(sdfg1: dace.SDFG, sdfg2: dace.SDFG) -> bool:
    from gtc.dace.nodes import (
        HorizontalExecutionLibraryNode,
        OIRLibraryNode,
        VerticalLoopLibraryNode,
    )

    def edge_match(edge1, edge2) -> bool:
        edge1 = next(iter(edge1.values()))
        edge2 = next(iter(edge2.values()))
        if edge1["src_conn"] is not None:
            if edge2["src_conn"] is None or edge1["src_conn"] != edge2["src_conn"]:
                return False
        elif edge2["src_conn"] is not None:
            return False

        if edge1["data"] != edge2["data"] or edge1["data"].data != edge2["data"].data:
            return False

        return True

    def node_match(n1, n2) -> bool:
        n1 = n1["node"]
        n2 = n2["node"]
        if not isinstance(
            n1,
            (
                dace.nodes.AccessNode,
                VerticalLoopLibraryNode,
                HorizontalExecutionLibraryNode,
            ),
        ):
            raise TypeError

        if isinstance(n1, dace.nodes.AccessNode):
            if (
                not isinstance(n2, dace.nodes.AccessNode)
                or n1.access != n2.access
                or n1.data != n2.data
            ):
                return False
        elif isinstance(n1, OIRLibraryNode):
            if n1 != n2:
                return False

        return True

    if len(sdfg1.states()) != 1 or len(sdfg2.states()) != 1:
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
        if (
            not isinstance(sdfg1.arrays[name], type(sdfg2.arrays[name]))
            or not isinstance(sdfg2.arrays[name], type(sdfg1.arrays[name]))
            or sdfg1.arrays[name].dtype != sdfg2.arrays[name].dtype
            or sdfg1.arrays[name].transient != sdfg2.arrays[name].transient
            or sdfg1.arrays[name].shape != sdfg2.arrays[name].shape
        ):
            return False

    return True


def axes_list_from_flags(flags):
    from gtc import daceir as dcir

    return [ax for f, ax in zip(flags, dcir.Axis.dims_3d()) if f]


def data_type_to_dace_typeclass(data_type):
    dtype = np.dtype(data_type_to_typestr(data_type))
    return dace.dtypes.typeclass(dtype.type)


def compute_horizontal_block_extents(node: oir.Stencil) -> Dict[int, Extent]:
    iteration_spaces = oir_iteration_space_computation(node)
    res = {}
    for he_id, it in iteration_spaces.items():
        assert it.i_interval.start.level == common.LevelMarker.START
        assert it.i_interval.end.level == common.LevelMarker.END
        assert it.j_interval.start.level == common.LevelMarker.START
        assert it.j_interval.end.level == common.LevelMarker.END
        res[he_id] = Extent(
            (it.i_interval.start.offset, it.i_interval.end.offset),
            (it.j_interval.start.offset, it.j_interval.end.offset),
            (0, 0),
        )
    return res


class AccessInfoCollector(NodeVisitor):
    def __init__(self, collect_read: bool, collect_write: bool, include_full_domain: bool = False):
        self.collect_read: bool = collect_read
        self.collect_write: bool = collect_write
        self.include_full_domain: bool = include_full_domain

    @dataclass
    class Context:
        axes: Dict[str, List["dcir.Axis"]]
        access_infos: Dict[str, "dcir.FieldAccessInfo"] = field(default_factory=dict)

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, block_extents, ctx, **kwargs: Any
    ) -> Dict[str, "dcir.FieldAccessInfo"]:
        for section in reversed(node.sections):
            self.visit(section, block_extents=block_extents, ctx=ctx, **kwargs)
        return ctx.access_infos

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        block_extents,
        ctx,
        grid_subset=None,
        **kwargs: Any,
    ) -> Dict[str, "dcir.FieldAccessInfo"]:
        from gtc import daceir as dcir

        inner_ctx = self.Context(
            axes=ctx.axes,
        )

        if grid_subset is None:
            grid_subset = dcir.GridSubset.from_interval(node.interval, dcir.Axis.K)
        elif dcir.Axis.K not in grid_subset.intervals:
            intervals = dict(dcir.GridSubset.from_interval(node.interval, dcir.Axis.K).intervals)
            intervals.update(grid_subset.intervals)
            grid_subset = dcir.GridSubset(intervals=intervals)
        self.visit(
            node.horizontal_executions,
            block_extents=block_extents,
            ctx=inner_ctx,
            grid_subset=grid_subset,
            k_interval=node.interval,
            **kwargs,
        )
        inner_infos = inner_ctx.access_infos

        k_grid = dcir.GridSubset.from_interval(grid_subset.intervals[dcir.Axis.K], dcir.Axis.K)
        inner_infos = {name: info.apply_iteration(k_grid) for name, info in inner_infos.items()}

        ctx.access_infos.update(
            {
                name: info.union(ctx.access_infos.get(name, info))
                for name, info in inner_infos.items()
            }
        )

        return ctx.access_infos

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        block_extents,
        ctx: Context,
        k_interval,
        grid_subset=None,
        **kwargs,
    ) -> Dict[str, "dcir.FieldAccessInfo"]:
        from gtc import daceir as dcir

        horizontal_extent = block_extents(node)

        inner_ctx = self.Context(
            axes=ctx.axes,
        )
        inner_infos = inner_ctx.access_infos
        ij_grid = dcir.GridSubset.from_gt4py_extent(horizontal_extent)
        he_grid = ij_grid.set_interval(dcir.Axis.K, k_interval)
        self.visit(
            node.body,
            horizontal_extent=horizontal_extent,
            ctx=inner_ctx,
            he_grid=he_grid,
            grid_subset=grid_subset,
            **kwargs,
        )

        if grid_subset is not None:
            for axis in ij_grid.axes():
                if axis in grid_subset.intervals:
                    ij_grid = ij_grid.set_interval(axis, grid_subset.intervals[axis])

        inner_infos = {name: info.apply_iteration(ij_grid) for name, info in inner_infos.items()}

        ctx.access_infos.update(
            {
                name: info.union(ctx.access_infos.get(name, info))
                for name, info in inner_infos.items()
            }
        )

        return ctx.access_infos

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs):
        self.visit(node.right, is_write=False, **kwargs)
        self.visit(node.left, is_write=True, **kwargs)

    def visit_MaskStmt(self, node: oir.MaskStmt, *, is_conditional=False, **kwargs):
        regions = node.mask.iter_tree().if_isinstance(oir.HorizontalMask).to_list()

        self.visit(node.mask, is_conditional=is_conditional, **kwargs)
        self.visit(node.body, is_conditional=True, regions=regions, **kwargs)

    def visit_While(self, node: oir.While, *, is_conditional=False, **kwargs):
        self.generic_visit(node, is_conditional=True, **kwargs)

    @staticmethod
    def _global_grid_subset(
        regions: List[oir.HorizontalMask],
        he_grid: "dcir.GridSubset",
        offset: List[Optional[int]],
    ):
        from gtc import daceir as dcir

        res: Dict[
            dcir.Axis,
            Union[dcir.DomainInterval, dcir.TileInterval, dcir.IndexWithExtent],
        ] = dict()
        if regions is not None:
            for mask in regions:
                for axis, oir_interval in zip(dcir.Axis.horizontal_axes(), mask.intervals):
                    start = (
                        oir_interval.start
                        if oir_interval.start is not None
                        else he_grid.intervals[axis].start
                    )
                    end = (
                        oir_interval.end
                        if oir_interval.end is not None
                        else he_grid.intervals[axis].end
                    )
                    dcir_interval = dcir.DomainInterval(
                        start=dcir.AxisBound.from_common(axis, start),
                        end=dcir.AxisBound.from_common(axis, end),
                    )
                    res[axis] = dcir.DomainInterval.union(
                        dcir_interval, res.get(axis, dcir_interval)
                    )
        if dcir.Axis.K in he_grid.intervals:
            off = offset[dcir.Axis.K.to_idx()] or 0
            res[dcir.Axis.K] = he_grid.intervals[dcir.Axis.K].shifted(off)
        for axis in dcir.Axis.horizontal_axes():
            iteration_interval = he_grid.intervals[axis]
            mask_interval = res.get(axis, iteration_interval)
            res[axis] = dcir.DomainInterval.intersection(
                axis, iteration_interval, mask_interval
            ).shifted(offset[axis.to_idx()])
        return dcir.GridSubset(intervals=res)

    def _make_access_info(
        self,
        offset_node: Union[CartesianOffset, oir.VariableKOffset],
        axes,
        is_conditional,
        regions,
        he_grid,
        grid_subset,
    ):
        from gtc import daceir as dcir

        offset = list(offset_node.to_tuple())
        if isinstance(offset_node, oir.VariableKOffset):
            variable_offset_axes = [dcir.Axis.K]
        else:
            variable_offset_axes = []

        k_interval = grid_subset.intervals[dcir.Axis.K]
        global_subset = self._global_grid_subset(regions, he_grid, offset)
        intervals = dict()
        for axis in axes:
            if axis in variable_offset_axes:
                intervals[axis] = dcir.IndexWithExtent(
                    axis=axis, value=axis.iteration_symbol(), extent=[0, 0]
                )
            else:
                intervals[axis] = dcir.IndexWithExtent(
                    axis=axis,
                    value=axis.iteration_symbol(),
                    extent=[offset[axis.to_idx()], offset[axis.to_idx()]],
                )
        grid_subset = dcir.GridSubset(intervals=intervals)
        return dcir.FieldAccessInfo(
            grid_subset=grid_subset,
            global_grid_subset=global_subset,
            dynamic_access=len(variable_offset_axes) > 0 or is_conditional or bool(regions),
            variable_offset_axes=variable_offset_axes,
        )

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        is_write: bool = False,
        ctx: "AccessInfoCollector.Context",
        is_conditional=False,
        regions=None,
        he_grid,
        grid_subset,
        **kwargs,
    ):
        self.visit(
            node.offset,
            is_conditional=is_conditional,
            ctx=ctx,
            is_write=False,
            regions=regions,
            he_grid=he_grid,
            grid_subset=grid_subset,
            **kwargs,
        )

        if (not self.collect_read and (not is_write)) or (not self.collect_write and is_write):
            return

        access_info = self._make_access_info(
            node.offset,
            axes=ctx.axes[node.name],
            is_conditional=is_conditional,
            regions=regions,
            he_grid=he_grid,
            grid_subset=grid_subset,
        )
        ctx.access_infos[node.name] = access_info.union(
            ctx.access_infos.get(node.name, access_info)
        )


def compute_dcir_access_infos(
    oir_node,
    *,
    oir_decls=None,
    block_extents=None,
    collect_read=True,
    collect_write=True,
    include_full_domain=False,
    **kwargs,
) -> Dict[str, "dcir.FieldAccessInfo"]:
    from gtc import daceir as dcir

    if block_extents is None:
        assert isinstance(oir_node, oir.Stencil)
        block_extents = compute_horizontal_block_extents(oir_node)

    axes = {
        name: axes_list_from_flags(decl.dimensions)
        for name, decl in oir_decls.items()
        if isinstance(decl, oir.FieldDecl)
    }
    ctx = AccessInfoCollector.Context(axes=axes, access_infos=dict())
    AccessInfoCollector(collect_read=collect_read, collect_write=collect_write).visit(
        oir_node, block_extents=block_extents, ctx=ctx, **kwargs
    )
    if include_full_domain:
        res = dict()
        for name, access_info in ctx.access_infos.items():
            res[name] = access_info.union(
                dcir.FieldAccessInfo(
                    grid_subset=dcir.GridSubset.full_domain(axes=access_info.axes()),
                    global_grid_subset=access_info.global_grid_subset,
                )
            )
    else:
        res = ctx.access_infos

    return res


def make_subset_str(
    context_info: "dcir.FieldAccessInfo", access_info: "dcir.FieldAccessInfo", data_dims
):
    res_strs = []
    clamped_access_info = access_info
    clamped_context_info = context_info
    for axis in access_info.axes():
        if axis in access_info.variable_offset_axes:
            clamped_access_info = clamped_access_info.clamp_full_axis(axis)
            clamped_context_info = clamped_context_info.clamp_full_axis(axis)

    for axis in clamped_access_info.axes():
        context_strs = clamped_context_info.grid_subset.intervals[axis].idx_range
        subset_strs = clamped_access_info.grid_subset.intervals[axis].idx_range
        res_strs.append(
            f"({subset_strs[0]})-({context_strs[0]}):({subset_strs[1]})-({context_strs[0]})"
        )
    res_strs.extend(f"0:{dim}" for dim in data_dims)
    return ",".join(res_strs)


def get_access_info_from_stencil_computation_sdfg(sdfg):
    from gtc.dace.nodes import StencilComputation

    nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, StencilComputation)]
    # decls = {
    #     name: decl
    #     for node in nodes
    #     for name, decl in node.declarations,
    #     if isinstance(decl, oir.FieldDecl)
    # }
    # block_extents = dict()
    #
    # for node in nodes:
    #     for i, section in enumerate(node.oir_node.sections):
    #         for j, he in enumerate(section.horizontal_executions):
    #             block_extents[id(he)] = node.extents[j * len(node.oir_node.sections) + i]
    #
    #
    # block_extents = lambda he: block_extents[id(he)]
    res_access_infos = dict()
    for node in nodes:
        access_infos = compute_dcir_access_infos(
            node.oir_node,
            oir_decls=node.declarations,
            block_extents=node.get_extents,
            collect_read=True,
            collect_write=True,
            include_full_domain=True,
        )
        for k, v in access_infos.items():
            res_access_infos.setdefault(k, v)
            res_access_infos[k] = res_access_infos[k].union(v)
    return res_access_infos


class DaceStrMaker:
    def __init__(self, stencil: oir.Stencil):
        self.decls = {
            decl.name: decl
            for decl in stencil.params + stencil.declarations
            if isinstance(decl, oir.FieldDecl)
        }
        block_extents = compute_horizontal_block_extents(stencil)
        self.block_extents = lambda he: block_extents[id(he)]

        self.access_infos = compute_dcir_access_infos(
            stencil,
            oir_decls=self.decls,
            block_extents=self.block_extents,
            collect_read=True,
            collect_write=True,
            include_full_domain=True,
        )
        self.access_collection = AccessCollector.apply(stencil)

    def make_shape(self, field):
        from gtc import daceir as dcir

        if field not in self.access_infos:
            return [
                axis.domain_symbol()
                for axis in dcir.Axis.dims_3d()
                if self.decls[field].dimensions[axis.to_idx()]
            ] + [d for d in self.decls[field].data_dims]
        return self.access_infos[field].shape + self.decls[field].data_dims

    def make_input_subset_str(self, node, field):
        local_access_info = compute_dcir_access_infos(
            node,
            collect_read=True,
            collect_write=False,
            block_extents=self.block_extents,
            oir_decls=self.decls,
        )[field]
        for axis in local_access_info.variable_offset_axes:
            local_access_info = local_access_info.clamp_full_axis(axis)

        return self._make_subset_str(local_access_info, field)

    def make_output_subset_str(self, node, field):
        local_access_info = compute_dcir_access_infos(
            node,
            collect_read=False,
            collect_write=True,
            block_extents=self.block_extents,
            oir_decls=self.decls,
        )[field]
        for axis in local_access_info.variable_offset_axes:
            local_access_info = local_access_info.clamp_full_axis(axis)

        return self._make_subset_str(local_access_info, field)

    def _make_subset_str(self, local_access_info, field):
        global_access_info = self.access_infos[field]
        return make_subset_str(global_access_info, local_access_info, self.decls[field].data_dims)


def untile_access_info_dict(access_infos: Dict[str, "dcir.FieldAccessInfo"], axes):

    res_infos = dict()
    for name, access_info in access_infos.items():
        res_infos[name] = access_info.untile(axes)
    return res_infos


def union_node_grid_subsets(nodes: List[eve.Node]):
    grid_subset = None

    for node in collect_toplevel_iteration_nodes(nodes):
        if grid_subset is None:
            grid_subset = node.grid_subset
        grid_subset = grid_subset.union(node.grid_subset)

    return grid_subset


def union_node_access_infos(nodes: List[eve.Node]):
    from gtc import daceir as dcir

    read_accesses: Dict[str, dcir.FieldAccessInfo] = dict()
    write_accesses: Dict[str, dcir.FieldAccessInfo] = dict()
    for node in collect_toplevel_computation_nodes(nodes):
        read_accesses.update(
            {
                name: access_info.union(read_accesses.get(name, access_info))
                for name, access_info in node.read_accesses.items()
            }
        )
        write_accesses.update(
            {
                name: access_info.union(write_accesses.get(name, access_info))
                for name, access_info in node.write_accesses.items()
            }
        )

    return (
        read_accesses,
        write_accesses,
        union_access_info_dicts(read_accesses, write_accesses),
    )


def union_access_info_dicts(
    first_infos: Dict[str, "dcir.FieldAccessInfo"],
    second_infos: Dict[str, "dcir.FieldAccessInfo"],
):
    res = dict(first_infos)
    for key, access_info in second_infos.items():
        res[key] = access_info.union(first_infos.get(key, access_info))
    return res


def flatten_list(list_or_node: Union[List[Any], eve.Node]):
    list_or_node = [list_or_node]
    while not all(isinstance(ref, eve.Node) for ref in list_or_node):
        list_or_node = [r for li in list_or_node for r in li]
    return list_or_node


def collect_toplevel_computation_nodes(
    list_or_node: Union[List[Any], eve.Node]
) -> List["dcir.ComputationNode"]:
    from gtc import daceir as dcir

    class ComputationNodeCollector(eve.NodeVisitor):
        def visit_ComputationNode(self, node: dcir.ComputationNode, *, collection: List):
            collection.append(node)

    collection: List[dcir.ComputationNode] = []
    ComputationNodeCollector().visit(list_or_node, collection=collection)
    return collection


def collect_toplevel_iteration_nodes(
    list_or_node: Union[List[Any], eve.Node]
) -> List["dcir.IterationNode"]:
    from gtc import daceir as dcir

    class IterationNodeCollector(eve.NodeVisitor):
        def visit_IterationNode(self, node: dcir.IterationNode, *, collection: List):
            collection.append(node)

    collection: List[dcir.IterationNode] = []
    IterationNodeCollector().visit(list_or_node, collection=collection)
    return collection


class HorizontalIntervalRemover(eve.NodeMutator):
    def visit_HorizontalMask(self, node: common.HorizontalMask, *, axis: "dcir.Axis"):
        mask_attrs = dict(i=node.i, j=node.j)
        mask_attrs[axis.lower()] = self.visit(getattr(node, axis.lower()))
        return common.HorizontalMask(**mask_attrs)

    def visit_HorizontalInterval(self, node: common.HorizontalInterval):
        return common.HorizontalInterval(start=None, end=None)


class HorizontalMaskRemover(eve.NodeMutator):
    def visit_Tasklet(self, node: "dcir.Tasklet"):
        from gtc import daceir as dcir

        res_body = []
        for stmt in node.stmts:
            newstmt = self.visit(stmt)
            if isinstance(newstmt, list):
                res_body.extend(newstmt)
            else:
                res_body.append(newstmt)
        return dcir.Tasklet(
            stmts=res_body,
            name_map=node.name_map,
            read_accesses=node.read_accesses,
            write_accesses=node.write_accesses,
        )

    def visit_MaskStmt(self, node: oir.MaskStmt):
        if isinstance(node.mask, common.HorizontalMask):
            if (
                node.mask.i.start is None
                and node.mask.j.start is None
                and node.mask.i.end is None
                and node.mask.j.end is None
            ):
                return self.generic_visit(node.body)
        return self.generic_visit(node)


def remove_horizontal_region(node, axis):
    intervals_removed = HorizontalIntervalRemover().visit(node, axis=axis)
    return HorizontalMaskRemover().visit(intervals_removed)


def mask_includes_inner_domain(mask: oir.HorizontalMask):
    for interval in mask.intervals:
        if interval.start is None and interval.end is None:
            return True
        elif interval.start is None and interval.end.level == common.LevelMarker.END:
            return True
        elif interval.end is None and interval.start.level == common.LevelMarker.START:
            return True
        elif (
            interval.start is not None
            and interval.end is not None
            and interval.start.level != interval.end.level
        ):
            return True
    return False


class HorizontalExecutionSplitter(eve.NodeTranslator):
    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, *, extents, library_node):
        if any(node.iter_tree().if_isinstance(oir.LocalScalar)):
            extents.append(library_node.get_extents(node))
            return node

        last_stmts = []
        res_he_stmts = [last_stmts]
        for stmt in node.body:
            if last_stmts and (
                (
                    isinstance(stmt, oir.MaskStmt)
                    and isinstance(stmt.mask, common.HorizontalMask)
                    and not mask_includes_inner_domain(stmt.mask)
                )
                or (
                    (
                        isinstance(last_stmts[0], oir.MaskStmt)
                        and isinstance(last_stmts[0].mask, common.HorizontalMask)
                        and not mask_includes_inner_domain(last_stmts[0].mask)
                    )
                )
            ):
                last_stmts = [stmt]
                res_he_stmts.append(last_stmts)
            else:
                last_stmts.append(stmt)

        res_hes = []
        for stmts in res_he_stmts:
            accessed_scalars = set(
                acc.name for acc in iter_tree(stmts).if_isinstance(oir.ScalarAccess)
            )
            declarations = [decl for decl in node.declarations if decl.name in accessed_scalars]
            res_he = oir.HorizontalExecution(declarations=declarations, body=stmts)
            res_hes.append(res_he)
            extents.append(library_node.get_extents(node))
        return res_hes

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection, **kwargs):
        res_hes = []
        for he in node.horizontal_executions:
            new_he = self.visit(he, **kwargs)
            if isinstance(new_he, list):
                res_hes.extend(new_he)
            else:
                res_hes.append(new_he)
        return oir.VerticalLoopSection(interval=node.interval, horizontal_executions=res_hes)


def split_horizontal_exeuctions_regions(node: "StencilComputation"):

    extents = []

    node.oir_node = HorizontalExecutionSplitter().visit(
        node.oir_node, library_node=node, extents=extents
    )
    ctr = 0
    for i, section in enumerate(node.oir_node.sections):
        for j, he in enumerate(section.horizontal_executions):
            node.extents[j * len(node.oir_node.sections) + i] = extents[ctr]
            ctr += 1
