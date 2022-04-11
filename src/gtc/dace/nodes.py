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

import base64
import copy
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import combinations, permutations, product
from typing import Dict, List, Optional, Set, Tuple, Union

import dace.data
import dace.dtypes
import dace.properties
import dace.subsets
import networkx as nx
import numpy as np
from dace import library

from gt4py.definitions import Extent
from gtc import common
from gtc import daceir as dcir
from gtc import oir
from gtc.common import DataType, LoopOrder, VariableKOffset, typestr_to_data_type
from gtc.dace.utils import (
    CartesianIterationSpace,
    OIRFieldRenamer,
    dace_dtype_to_typestr,
    get_node_name_mapping,
    mask_includes_inner_domain,
)
from gtc.oir import (
    CacheDesc,
    Decl,
    FieldDecl,
    HorizontalExecution,
    Interval,
    VerticalLoop,
    VerticalLoopSection,
)


class OIRLibraryNode(ABC, dace.nodes.LibraryNode):
    @abstractmethod
    def as_oir(self):
        raise NotImplementedError("Implement in child class.")

    def to_json(self, parent):
        protocol = pickle.DEFAULT_PROTOCOL
        pbytes = pickle.dumps(self, protocol=protocol)

        jsonobj = super().to_json(parent)
        jsonobj["classpath"] = dace.nodes.full_class_path(self)
        jsonobj["attributes"]["protocol"] = protocol
        jsonobj["attributes"]["pickle"] = base64.b64encode(pbytes).decode("utf-8")

        return jsonobj

    @classmethod
    def from_json(cls, json_obj, context=None):
        if "attributes" not in json_obj:
            b64string = json_obj["pickle"]
        else:
            b64string = json_obj["attributes"]["pickle"]
        byte_repr = base64.b64decode(b64string)
        return pickle.loads(byte_repr)


@library.node
class VerticalLoopLibraryNode(OIRLibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
    default_implementation = "naive"

    loop_order = dace.properties.Property(dtype=LoopOrder, default=None, allow_none=True)
    sections = dace.properties.ListProperty(
        element_type=Tuple[Interval, dace.SDFG], default=[], allow_none=False
    )
    caches = dace.properties.ListProperty(element_type=CacheDesc, default=[], allow_none=False)
    default_storage_type = dace.properties.EnumProperty(
        dtype=dace.StorageType, default=dace.StorageType.Default
    )
    ijcache_storage_type = dace.properties.EnumProperty(
        dtype=dace.StorageType, default=dace.StorageType.Default
    )
    kcache_storage_type = dace.properties.EnumProperty(
        dtype=dace.StorageType, default=dace.StorageType.Default
    )
    tiling_map_schedule = dace.properties.EnumProperty(
        dtype=dace.ScheduleType, default=dace.ScheduleType.Default
    )
    map_schedule = dace.properties.EnumProperty(
        dtype=dace.ScheduleType, default=dace.ScheduleType.Default
    )
    tile_sizes = dace.properties.ListProperty(element_type=int, default=None, allow_none=True)

    _dace_library_name = "oir.VerticalLoop"

    def __init__(
        self,
        name="unnamed_vloop",
        loop_order: LoopOrder = None,
        sections: List[Tuple[Interval, dace.SDFG]] = None,
        caches: List[CacheDesc] = None,
        *args,
        **kwargs,
    ):

        if loop_order is not None:
            self.loop_order = loop_order
            self.sections = sections
            self.caches = caches

        super().__init__(name=name, *args, **kwargs)

    def validate(self, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState, *args, **kwargs):

        get_node_name_mapping(parent_state, self)

        for _, sdfg in self.sections:
            sdfg.validate()
            is_correct_node_types = all(
                isinstance(
                    n,
                    (
                        dace.SDFGState,
                        dace.nodes.AccessNode,
                        HorizontalExecutionLibraryNode,
                    ),
                )
                for n, _ in sdfg.all_nodes_recursive()
            )
            is_correct_data_and_dtype = all(
                isinstance(array, dace.data.Array)
                and typestr_to_data_type(dace_dtype_to_typestr(array.dtype)) != DataType.INVALID
                for array in sdfg.arrays.values()
            )
            if not is_correct_node_types or not is_correct_data_and_dtype:
                raise ValueError("Tried to convert incompatible SDFG to OIR.")

        super().validate(parent_sdfg, parent_state, *args, **kwargs)

    def as_oir(self):

        sections = []
        for interval, sdfg in self.sections:
            horizontal_executions = []
            for state in sdfg.topological_sort(sdfg.start_state):

                for node in (
                    n
                    for n in nx.topological_sort(state.nx)
                    if isinstance(n, HorizontalExecutionLibraryNode)
                ):
                    horizontal_executions.append(
                        OIRFieldRenamer(get_node_name_mapping(state, node)).visit(node.as_oir())
                    )
            sections.append(
                VerticalLoopSection(interval=interval, horizontal_executions=horizontal_executions)
            )

        return VerticalLoop(
            sections=sections,
            loop_order=self.loop_order,
            caches=self.caches,
        )


@dataclass
class PreliminaryHorizontalExecution:
    body: List[oir.Stmt]
    declarations: List[oir.LocalScalar]


@library.node
class HorizontalExecutionLibraryNode(OIRLibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
    default_implementation = "naive"

    _oir_node: Union[HorizontalExecution, PreliminaryHorizontalExecution] = None
    iteration_space = dace.properties.Property(
        dtype=CartesianIterationSpace, default=None, allow_none=True
    )

    map_schedule = dace.properties.EnumProperty(
        dtype=dace.ScheduleType, default=dace.ScheduleType.Default
    )
    index_symbols = dace.properties.ListProperty(element_type=str, default=["i", "j", "0"])
    global_domain_symbols = dace.properties.ListProperty(element_type=str, default=["__I", "__J"])

    _dace_library_name = "oir.HorizontalExecution"

    def __init__(
        self,
        name="unnamed_vloop",
        oir_node: HorizontalExecution = None,
        iteration_space: CartesianIterationSpace = None,
        *args,
        **kwargs,
    ):
        if oir_node is not None:
            self._oir_node = oir_node
            self.iteration_space = iteration_space

        super().__init__(name=name, *args, **kwargs)

    @property
    def oir_node(self):
        return self._oir_node

    def commit_horizontal_execution(self):
        self._oir_node = HorizontalExecution(
            body=self._oir_node.body, declarations=self._oir_node.declarations
        )

    def as_oir(self):
        self.commit_horizontal_execution()
        return self.oir_node

    def validate(self, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState, *args, **kwargs):
        get_node_name_mapping(parent_state, self)

    @property
    def free_symbols(self):
        res = super().free_symbols
        if len(self.oir_node.iter_tree().if_isinstance(VariableKOffset).to_list()) > 0:
            res.add("k")
        return res


def get_expansion_order_axis(item):
    if is_domain_map(item) or (is_domain_loop(item) and not item.startswith("Cached")):
        return dcir.Axis(item[0])
    elif is_tiling(item):
        return dcir.Axis(item[-1])
    elif item.startswith("Cached"):
        return dcir.Axis(item[len("Cached")])
    else:
        return dcir.Axis(item)


def is_domain_map(item):
    return any(f"{axis}Map" == item for axis in dcir.Axis.dims_3d())


def is_domain_loop(item):
    return any(item.endswith(f"{axis}Loop") for axis in dcir.Axis.dims_3d())


def is_tiling(item):
    return any(f"Tile{axis}" == item for axis in dcir.Axis.dims_3d())


def get_expansion_order_index(expansion_order, axis):
    for idx, item in enumerate(expansion_order):
        if isinstance(item, Iteration) and item.axis == axis:
            return idx
        elif isinstance(item, Map):
            for it in item.iterations:
                if it.kind == "contiguous" and it.axis == axis:
                    return idx


def _is_expansion_order_implemented(expansion_specification):

    # TODO: Could have single IJ map with predicates in K, e.g. also for tiling??
    seen_sections = False
    for item in expansion_specification:
        if isinstance(item, Sections):
            break
        if isinstance(item, Iteration) and item.axis == dcir.Axis.K:
            return False
        if isinstance(item, Map) and any(it.axis == dcir.Axis.K for it in item.iterations):
            return False
    cached_loops = [
        item
        for item in expansion_specification
        if isinstance(item, Loop) and len(item.localcache_fields) > 0
    ]
    if len(cached_loops) > 1:
        return False

    not_outermost_dims = set()
    for item in expansion_specification:
        iterations = []
        if isinstance(item, Loop):
            if item.localcache_fields:
                if not all(
                    axis in not_outermost_dims or axis == item.axis for axis in dcir.Axis.dims_3d()
                ):
                    return False
            iterations = [item]
        elif isinstance(item, Iteration):
            iterations = [item]
        elif isinstance(item, Map):
            iterations = item.iterations
        for it in iterations:
            not_outermost_dims.add(it.axis)

    is_outermost_loop_in_sdfg = True
    for item in expansion_specification:
        if isinstance(item, Loop):
            if not is_outermost_loop_in_sdfg and item.localcache_fields:
                return False
            is_outermost_loop_in_sdfg = False
        else:
            is_outermost_loop_in_sdfg = True

    return True


def _order_as_spec(computation_node, expansion_order):
    expansion_order = list(computation_node._sanitize_expansion_item(eo) for eo in expansion_order)
    expansion_specification = []
    for item in expansion_order:
        if isinstance(item, ExpansionItem):
            expansion_specification.append(item)
        elif isinstance(item, Iteration):
            if isinstance(item, Iteration) and item.axis == "K" and item.kind == "contiguous":
                if computation_node.oir_node is not None:
                    if computation_node.oir_node.loop_order == common.LoopOrder.PARALLEL:
                        expansion_specification.append(Map(iterations=[item]))
                    else:
                        expansion_specification.append(
                            Loop(axis=item.axis, kind=item.kind, stride=item.stride)
                        )
                else:
                    expansion_specification.append(item)
        elif is_tiling(item):
            axis = get_expansion_order_axis(item)
            expansion_specification.append(
                Map(
                    iterations=[
                        Iteration(
                            axis=axis,
                            kind="tiling",
                            stride=None,
                        )
                    ]
                )
            )
        elif is_domain_map(item):
            axis = get_expansion_order_axis(item)
            expansion_specification.append(
                Map(
                    iterations=[
                        Iteration(
                            axis=axis,
                            kind="contiguous",
                            stride=1,
                        )
                    ]
                )
            )
        elif is_domain_loop(item):
            axis = get_expansion_order_axis(item)
            if item.startswith("Cached"):
                localcache_fields = set(computation_node.field_decls.keys())
                for acc in computation_node.oir_node.iter_tree().if_isinstance(oir.FieldAccess):
                    if isinstance(acc.offset, oir.VariableKOffset):
                        if acc.name in localcache_fields:
                            localcache_fields.remove(acc.name)

                for mask_stmt in computation_node.oir_node.iter_tree().if_isinstance(oir.MaskStmt):
                    if mask_stmt.mask.iter_tree().if_isinstance(oir.HorizontalMask).to_list():
                        for stmt in mask_stmt.body:
                            for acc in (
                                stmt.iter_tree()
                                .if_isinstance(oir.AssignStmt)
                                .getattr("left")
                                .if_isinstance(oir.FieldAccess)
                            ):
                                if acc.name in localcache_fields:
                                    localcache_fields.remove(acc.name)
            else:
                localcache_fields = set()
            expansion_specification.append(
                Loop(
                    axis=axis,
                    stride=-1
                    if computation_node.oir_node.loop_order == common.LoopOrder.BACKWARD
                    else 1,
                    localcache_fields=localcache_fields,
                )
            )

        elif item == "Sections":
            expansion_specification.append(Sections())
        else:
            assert item == "Stages", item
            expansion_specification.append(Stages())

    return expansion_specification


def _populate_strides(self, expansion_specification):
    if expansion_specification is None:
        return
    assert all(isinstance(es, (ExpansionItem, Iteration)) for es in expansion_specification)

    iterations = []
    for es in expansion_specification:
        if isinstance(es, Map):
            iterations.extend(es.iterations)
        elif isinstance(es, Loop):
            iterations.append(es)

    for it in iterations:
        if isinstance(it, Loop):
            if it.stride is None:
                if self.oir_node.loop_order == common.LoopOrder.BACKWARD:
                    it.stride = -1
                else:
                    it.stride = 1
        else:
            if it.stride is None:
                if it.kind == "tiling":
                    from gt4py.definitions import Extent

                    if hasattr(self, "_tile_sizes"):
                        if self.extents is not None and it.axis.to_idx() < 2:
                            extent = Extent.zeros(3)
                            for he_extent in self.extents.values():
                                extent = extent.union(he_extent)
                            extent = extent[it.axis.to_idx()]
                        else:
                            extent = (0, 0)
                        it.stride = self.tile_sizes.get(it.axis, 8) + extent[0] - extent[1]
                else:
                    it.stride = 1


def _populate_storages(self, expansion_specification):
    if expansion_specification is None:
        return
    assert all(isinstance(es, (ExpansionItem, Iteration)) for es in expansion_specification)
    innermost_axes = set(dcir.Axis.dims_3d())
    tiled_axes = set()
    for item in expansion_specification:
        if isinstance(item, Map):
            for it in item.iterations:
                if it.kind == "tiling":
                    tiled_axes.add(it.axis)
    for es in reversed(expansion_specification):
        if isinstance(es, Loop) and es.localcache_fields:
            if hasattr(self, "_device"):
                if es.storage is None:
                    if len(innermost_axes) == 3:
                        es.storage = dace.StorageType.Register
                    elif (
                        self.device == dace.DeviceType.GPU and len(innermost_axes | tiled_axes) == 3
                    ):
                        es.storage = dace.StorageType.GPU_Shared
                    else:
                        if self.device == dace.DeviceType.GPU:
                            es.storage = dace.StorageType.GPU_Global
                        else:
                            es.storage = dace.StorageType.CPU_Heap
            innermost_axes.remove(es.axis)
        elif isinstance(es, Map):
            for it in es.iterations:
                if it.axis in innermost_axes:
                    innermost_axes.remove(it.axis)
                if it.kind == "tiling":
                    tiled_axes.remove(it.axis)


def _populate_schedules(self, expansion_specification):
    if expansion_specification is None:
        return
    assert all(isinstance(es, (ExpansionItem, Iteration)) for es in expansion_specification)
    is_outermost = True
    is_inside = False

    if hasattr(self, "_device"):
        if self.device == dace.DeviceType.GPU:
            # On GPU if any dimension is tiled and has a contiguous map in the same axis further in
            # pick those two maps as Device/ThreadBlock maps. If not, Make just device map with
            # default blocksizes
            tiled = False
            for i, item in enumerate(expansion_specification):
                if isinstance(item, Map):
                    for it in item.iterations:
                        if not tiled and it.kind == "tiling":
                            for inner_item in expansion_specification[i + 1 :]:
                                if isinstance(inner_item, Map) and any(
                                    inner_it.kind == "contiguous" and inner_it.axis == it.axis
                                    for inner_it in inner_item.iterations
                                ):
                                    item.schedule = dace.ScheduleType.GPU_Device
                                    inner_item.schedule = dace.ScheduleType.GPU_ThreadBlock
                                    tiled = True
                                    break
            if not tiled:
                assert any(
                    isinstance(item, Map) for item in expansion_specification
                ), "needs at least one map to avoid dereferencing on CPU"
                for es in expansion_specification:
                    if isinstance(es, Map):
                        if es.schedule is None:
                            if is_outermost:
                                es.schedule = dace.ScheduleType.GPU_Device
                                is_outermost = False
                            else:
                                es.schedule = dace.ScheduleType.Default

        else:
            for es in expansion_specification:
                if isinstance(es, Map):
                    if es.schedule is None:
                        if is_outermost:
                            es.schedule = dace.ScheduleType.CPU_Multicore
                            is_outermost = False
                        else:
                            es.schedule = dace.ScheduleType.Default


def _collapse_maps(self, expansion_specification):
    if hasattr(self, "_device"):
        res_items = []
        if self.device == dace.DeviceType.GPU:
            tiled_axes = set()
            for item in expansion_specification:
                if isinstance(item, Map):
                    if (
                        not res_items
                        or not isinstance(res_items[-1], Map)
                        or any(
                            it.axis in set(outer_it.axis for outer_it in res_items[-1].iterations)
                            for it in item.iterations
                        )
                    ):
                        res_items.append(item)
                    elif item.schedule == res_items[-1].schedule:
                        res_items[-1].iterations.extend(item.iterations)
                    elif item.schedule is None or item.schedule == dace.ScheduleType.Default:
                        if res_items[-1].schedule == dace.ScheduleType.GPU_Device:
                            res_items[-1].iterations.extend(item.iterations)
                        elif res_items[-1].schedule == dace.ScheduleType.GPU_ThreadBlock:
                            if all(it.axis in tiled_axes for it in item.iterations):
                                res_items[-1].iterations.extend(item.iterations)
                            else:
                                res_items.append(item)
                        elif res_items[-1].iterations:
                            res_items.append(item)
                        else:
                            res_items[-1] = item
                    elif (
                        res_items[-1].schedule is None
                        or res_items[-1].schedule == dace.ScheduleType.Default
                    ):
                        if item.schedule == dace.ScheduleType.GPU_Device:
                            res_items[-1].iterations.extend(item.iterations)
                            res_items[-1].schedule = dace.ScheduleType.GPU_Device
                        elif item.schedule == dace.ScheduleType.GPU_ThreadBlock:
                            if all(it.axis in tiled_axes for it in res_items[-1].iterations):
                                res_items[-1].iterations.extend(item.iterations)
                                res_items[-1].schedule = dace.ScheduleType.GPU_ThreadBlock
                            else:
                                res_items.append(item)
                        elif res_items[-1].iterations:
                            res_items.append(item)
                        else:
                            res_items[-1] = item

                    elif res_items[-1].iterations:
                        res_items.append(item)
                    else:
                        res_items[-1] = item
                    tiled_axes = tiled_axes.union(
                        (it.axis for it in item.iterations if it.kind == "tiling")
                    )
                else:
                    res_items.append(item)
        else:
            for item in expansion_specification:
                if isinstance(item, Map):
                    if (
                        not res_items
                        or not isinstance(res_items[-1], Map)
                        or any(
                            it.axis in set(outer_it.axis for outer_it in res_items[-1].iterations)
                            for it in item.iterations
                        )
                    ):
                        res_items.append(item)
                    elif item.schedule == res_items[-1].schedule:
                        res_items[-1].iterations.extend(item.iterations)
                    elif item.schedule is None or item.schedule == dace.ScheduleType.Default:
                        if res_items[-1].schedule == dace.ScheduleType.CPU_Multicore:
                            res_items[-1].iterations.extend(item.iterations)
                        else:
                            res_items.append(item)
                    elif (
                        res_items[-1].schedule is None
                        or res_items[-1].schedule == dace.ScheduleType.Default
                    ):
                        if item.schedule == dace.ScheduleType.CPU_Multicore:
                            res_items[-1].iterations.extend(item.iterations)
                            res_items[-1].schedule = dace.ScheduleType.CPU_Multicore
                        else:
                            res_items.append(item)
                    else:
                        res_items.append(item)
                else:
                    res_items.append(item)

        for item in expansion_specification:
            if isinstance(item, Map) and (
                item.schedule is None or item.schedule == dace.ScheduleType.Default
            ):
                item.schedule = dace.ScheduleType.Sequential
        expansion_specification.clear()
        expansion_specification.extend(res_items)


@dataclass
class ExpansionItem:
    pass


def make_expansion_order(
    node: "StencilComputation", expansion_order: List[str]
) -> List[ExpansionItem]:
    if expansion_order is None:
        return None
    expansion_order = copy.deepcopy(expansion_order)
    expansion_specification = _order_as_spec(node, expansion_order)

    if not _is_expansion_order_implemented(expansion_specification):
        raise ValueError("Provided StencilComputation.expansion_order is not supported.")
    if node.oir_node is not None:
        if not node.is_valid_expansion_order(expansion_specification):
            raise ValueError("Provided StencilComputation.expansion_order is invalid.")

    _populate_strides(node, expansion_specification)
    _populate_schedules(node, expansion_specification)
    _collapse_maps(node, expansion_specification)
    _populate_storages(node, expansion_specification)
    return expansion_specification


def set_expansion_order(self, expansion_order):
    res = make_expansion_order(self, expansion_order)
    self._expansion_specification = res


@dataclass
class Iteration:
    axis: dcir.Axis
    kind: str  # tiling, contiguous
    # if stride is not specified, it is chosen based on backend (tiling) and loop order (K)
    stride: Optional[int] = None


@dataclass
class Map(ExpansionItem):
    iterations: List[Iteration]
    schedule: Optional[dace.ScheduleType] = None


@dataclass
class Loop(Iteration, ExpansionItem):
    kind: str = "contiguous"
    localcache_fields: Set[str] = dataclass_field(default_factory=set)
    storage: dace.StorageType = None


@dataclass
class Stages(ExpansionItem):
    pass


@dataclass
class Sections(ExpansionItem):
    pass


@library.node
class StencilComputation(library.LibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
    default_implementation = "default"

    oir_node = dace.properties.DataclassProperty(dtype=VerticalLoop, default=None, allow_none=True)

    declarations = dace.properties.DictProperty(
        key_type=str, value_type=Decl, default=None, allow_none=True
    )
    extents = dace.properties.DictProperty(
        key_type=int, value_type=Extent, default=None, allow_none=True
    )

    expansion_specification = dace.properties.ListProperty(
        element_type=ExpansionItem,
        default=[
            Map(
                iterations=[
                    Iteration(axis=dcir.Axis.I, kind="tiling"),
                    Iteration(axis=dcir.Axis.J, kind="tiling"),
                ],
            ),
            Sections(),
            Iteration(axis=dcir.Axis.K, kind="contiguous"),  # Expands to either Loop or Map
            Stages(),
            Map(
                iterations=[
                    Iteration(axis=dcir.Axis.I, kind="contiguous"),
                    Iteration(axis=dcir.Axis.J, kind="contiguous"),
                ]
            ),
        ],
        allow_none=False,
        setter=set_expansion_order,
    )
    tile_sizes = dace.properties.DictProperty(
        key_type=str,
        value_type=int,
        default={dcir.Axis.I: 64, dcir.Axis.J: 8, dcir.Axis.K: 8},
    )
    device = dace.properties.EnumProperty(
        dtype=dace.DeviceType, default=dace.DeviceType.CPU, allow_none=True
    )

    symbol_mapping = dace.properties.DictProperty(
        key_type=str, value_type=object, default=None, allow_none=True
    )
    _dace_library_name = "StencilComputation"

    def __init__(
        self,
        name="unnamed_vloop",
        oir_node: VerticalLoop = None,
        extents: Dict[int, Extent] = None,
        declarations: Dict[str, Decl] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)

        if oir_node is not None:

            extents_dict = dict()
            for i, section in enumerate(oir_node.sections):
                for j, he in enumerate(section.horizontal_executions):
                    extents_dict[j * len(oir_node.sections) + i] = extents[id(he)]

            self.oir_node = oir_node
            self.extents = extents_dict
            self.declarations = declarations
            self.symbol_mapping = {
                decl.name: dace.symbol(
                    decl.name,
                    dtype=dace.typeclass(np.dtype(common.data_type_to_typestr(decl.dtype)).type),
                )
                for decl in declarations.values()
                if isinstance(decl, oir.ScalarDecl)
            }
            self.symbol_mapping.update(
                {
                    axis.domain_symbol(): dace.symbol(axis.domain_symbol(), dtype=dace.int32)
                    for axis in dcir.Axis.horizontal_axes()
                }
            )
            if any(
                interval.start.level == common.LevelMarker.END
                or interval.end.level == common.LevelMarker.END
                for interval in oir_node.iter_tree()
                .if_isinstance(VerticalLoopSection)
                .getattr("interval")
            ) or any(
                decl.dimensions[dcir.Axis.K.to_idx()]
                for decl in self.declarations.values()
                if isinstance(decl, oir.FieldDecl)
            ):
                self.symbol_mapping[dcir.Axis.K.domain_symbol()] = dace.symbol(
                    dcir.Axis.K.domain_symbol(), dtype=dace.int32
                )

            if oir_node.loc is not None:

                self.debuginfo = dace.dtypes.DebugInfo(
                    oir_node.loc.line,
                    oir_node.loc.column,
                    oir_node.loc.line,
                    oir_node.loc.column,
                    oir_node.loc.source,
                )
            assert self.oir_node is not None
            set_expansion_order(self, self._expansion_specification)

    def get_extents(self, he):
        for i, section in enumerate(self.oir_node.sections):
            for j, cand_he in enumerate(section.horizontal_executions):
                if he is cand_he:
                    return self.extents[j * len(self.oir_node.sections) + i]

    @property
    def field_decls(self) -> Dict[str, FieldDecl]:
        return {
            name: decl for name, decl in self.declarations.items() if isinstance(decl, FieldDecl)
        }

    def to_json(self, parent):
        protocol = pickle.DEFAULT_PROTOCOL
        pbytes = pickle.dumps(self, protocol=protocol)

        jsonobj = super().to_json(parent)
        jsonobj["classpath"] = dace.nodes.full_class_path(self)
        jsonobj["attributes"]["protocol"] = protocol
        jsonobj["attributes"]["pickle"] = base64.b64encode(pbytes).decode("utf-8")

        return jsonobj

    @classmethod
    def from_json(cls, json_obj, context=None):
        if "attributes" not in json_obj:
            b64string = json_obj["pickle"]
        else:
            b64string = json_obj["attributes"]["pickle"]
        byte_repr = base64.b64decode(b64string)
        return pickle.loads(byte_repr)

    def _sanitize_expansion_item(self, eo):
        if any(eo == axis for axis in dcir.Axis.dims_horizontal()):
            return f"{eo}Map"
        if eo == dcir.Axis.K:
            if self.oir_node.loop_order == common.LoopOrder.PARALLEL:
                return f"{eo}Map"
            else:
                return f"Cached{eo}Loop"
        return eo

    def is_expansion_order_valid(self, expansion_order, *, k_inside_dims, k_inside_stages):
        expansion_specification = [self._sanitize_expansion_item(eo) for eo in expansion_order]
        # K can't be Map if not parallel
        if self.oir_node.loop_order != common.LoopOrder.PARALLEL and any(
            (isinstance(item, Map) and any(it.axis == dcir.Axis.K for it in item.iterations))
            for item in expansion_specification
        ):
            return False
        if not any(
            isinstance(item, Map) or is_tiling(item) or is_domain_map(item)
            for item in expansion_specification
        ):
            return False

        sections_idx = next(
            idx for idx, item in enumerate(expansion_specification) if isinstance(item, Sections)
        )
        stages_idx = next(
            idx for idx, item in enumerate(expansion_specification) if isinstance(item, Stages)
        )
        if stages_idx < sections_idx:
            return False

        for axis in dcir.Axis.dims_horizontal():
            if (
                get_expansion_order_index(expansion_specification, axis)
                < get_expansion_order_index(expansion_specification, dcir.Axis.K)
                and axis not in k_inside_dims
            ):
                return False
        if (
            stages_idx < get_expansion_order_index(expansion_specification, dcir.Axis.K)
            and not k_inside_stages
        ):
            return False

        # enforce [tiling]...map per axis: no repetition of inner of maps, no tiling without
        # subsequent iteration map, no tiling inside contiguous, all dims present
        tiled_axes = set()
        contiguous_axes = set()
        for item in expansion_specification:
            if isinstance(item, (Map, Loop, Iteration)):
                if isinstance(item, Map):
                    iterations = item.iterations
                else:
                    iterations = [item]
                for it in iterations:
                    if it.kind == "tiling":
                        if it.axis in tiled_axes or it.axis in contiguous_axes:
                            return False
                        tiled_axes.add(it.axis)
                    else:
                        if it.axis in contiguous_axes:
                            return False
                        contiguous_axes.add(it.axis)
        if not all(axis in contiguous_axes for axis in dcir.Axis.dims_3d()):
            return False

        # if there are multiple horizontal executions in any section, IJ iteration must go inside
        # TODO: do mergeability checks on a per-axis basis.
        for item in expansion_specification:
            if isinstance(item, Stages):
                break
            if isinstance(item, (Map, Loop, Iteration)):
                if isinstance(item, Map):
                    iterations = item.iterations
                else:
                    iterations = [item]
                for it in iterations:
                    if it.axis in dcir.Axis.horizontal_axes() and it.kind == "contiguous":
                        if any(
                            len(section.horizontal_executions) > 1
                            for section in self.oir_node.sections
                        ):
                            return False

        # if there are horizontal executions with different iteration ranges in axis across sections,
        # axis-contiguous must be inside sections
        # TODO enable this with predicates?
        # TODO less conservative: allow larger domains if all outputs in larger are temporaries?
        for item in expansion_specification:
            if isinstance(item, Sections):
                break
            if isinstance(item, Map):
                iterations = item.iterations
            else:
                iterations = [item]
            for it in iterations:
                if it.axis in dcir.Axis.horizontal_axes() and it.kind == "contiguous":
                    xiter = iter(self.oir_node.iter_tree().if_isinstance(oir.HorizontalExecution))
                    extent = self.get_extents(next(xiter))
                    for he in xiter:
                        if self.get_extents(he)[it.axis.to_idx()] != extent[it.axis.to_idx()]:
                            return False

        res = _is_expansion_order_implemented(expansion_specification)
        return res

    def k_inside_dims(self):
        # Putting K inside of i or j is valid if
        # * K parallel
        # * All reads with k-offset to values modified in same HorizontalExecution are not
        #   to fields that are also accessed horizontally (in I or J, respectively)
        #   (else, race condition in other column)

        if self.oir_node.loop_order == common.LoopOrder.PARALLEL:
            return {dcir.Axis.I, dcir.Axis.J}

        res = {dcir.Axis.I, dcir.Axis.J}
        for section in self.oir_node.sections:
            for he in section.horizontal_executions:
                i_offset_fields = set(
                    (
                        acc.name
                        for acc in he.iter_tree().if_isinstance(oir.FieldAccess)
                        if acc.offset.i != 0
                    )
                )
                j_offset_fields = set(
                    (
                        acc.name
                        for acc in he.iter_tree().if_isinstance(oir.FieldAccess)
                        if acc.offset.j != 0
                    )
                )
                k_offset_fields = set(
                    (
                        acc.name
                        for acc in he.iter_tree().if_isinstance(oir.FieldAccess)
                        if isinstance(acc.offset, oir.VariableKOffset) or acc.offset.k != 0
                    )
                )
                modified_fields = (
                    he.iter_tree()
                    .if_isinstance(oir.AssignStmt)
                    .getattr("left")
                    .if_isinstance(oir.FieldAccess)
                    .getattr("name")
                    .to_set()
                )
                for name in modified_fields:
                    if name in k_offset_fields and name in i_offset_fields:
                        res.remove(dcir.Axis.I)
                    if name in k_offset_fields and name in j_offset_fields:
                        res.remove(dcir.Axis.J)
        return res

    def k_inside_stages(self):
        # Putting K inside of stages is valid if
        # * K parallel
        # * not "ahead" in order of iteration to fields that are modified in previous
        #   HorizontalExecutions (else, reading updated values that should be old)

        if self.oir_node.loop_order == common.LoopOrder.PARALLEL:
            return True

        for section in self.oir_node.sections:
            modified_fields = set()
            for he in section.horizontal_executions:
                if modified_fields:
                    ahead_acc = list()
                    for acc in he.iter_tree().if_isinstance(oir.FieldAccess):
                        if (
                            isinstance(acc.offset, oir.VariableKOffset)
                            or (
                                self.oir_node.loop_order == common.LoopOrder.FORWARD
                                and acc.offset.k > 0
                            )
                            or (
                                self.oir_node.loop_order == common.LoopOrder.BACKWARD
                                and acc.offset.k < 0
                            )
                        ):
                            ahead_acc.append(acc)
                    if any(acc.name in modified_fields for acc in ahead_acc):
                        return False

                modified_fields.update(
                    he.iter_tree()
                    .if_isinstance(oir.AssignStmt)
                    .getattr("left")
                    .if_isinstance(oir.FieldAccess)
                    .getattr("name")
                    .to_set()
                )

        return True

    def is_valid_expansion_order(self, expansion_order: List[str]) -> bool:
        expansion_order = list(self._sanitize_expansion_item(eo) for eo in expansion_order)
        return self.is_expansion_order_valid(
            expansion_order,
            k_inside_dims=self.k_inside_dims(),
            k_inside_stages=self.k_inside_stages(),
        )

    def valid_expansion_orders(self):
        optionals = {"TileI", "TileJ", "TileK"}
        required = {
            ("KMap", "KLoop", "CachedKLoop"),  # tuple represents "one of",
            ("JMap", "JLoop", "CachedJLoop"),
            ("IMap", "ILoop", "CachedILoop"),
        }
        prepends = []
        appends = []
        if len(self.oir_node.sections) > 1:
            required.add("Sections")
        else:
            prepends.append("Sections")
        if any(len(section.horizontal_executions) > 1 for section in self.oir_node.sections):
            required.add("Stages")
        else:
            appends.append("Stages")

        def expansion_subsets():
            for k in range(len(optionals) + 1):
                for subset in combinations(optionals, k):
                    subset = {s if isinstance(s, tuple) else (s,) for s in set(subset) | required}
                    yield from product(*subset)

        for expansion_subset in expansion_subsets():
            for expansion_order in permutations(expansion_subset):
                expansion_order = prepends + list(expansion_order) + appends
                expansion_specification = _order_as_spec(self, expansion_order)
                if self.is_expansion_order_valid(
                    expansion_specification,
                    k_inside_dims=self.k_inside_dims(),
                    k_inside_stages=self.k_inside_stages(),
                ):
                    yield expansion_order

    @property
    def free_symbols(self) -> Set[str]:
        result: Set[str] = set()
        for v in self.symbol_mapping.values():
            result.update(map(str, v.free_symbols))
        return result

    def has_splittable_regions(self):
        for he in self.oir_node.iter_tree().if_isinstance(oir.HorizontalExecution):
            if not he.declarations and any(
                isinstance(stmt, oir.MaskStmt)
                and isinstance(stmt.mask, common.HorizontalMask)
                and not mask_includes_inner_domain(stmt.mask)
                for stmt in he.body
            ):
                return True
        return False

    #
    # def __eq__(self, other):
    #     super(StencilComputation, self).__eq__(other)
    #
    # def __hash__(self):#TODO
    #     properties = (self.__dict__.get(k, v.default) for k, v in StencilComputation.__dict__.items() if isinstance(v, dace.properties.Property))
    #     property_hashes = []
    #     for property in properties:
    #         try:
    #             property_hashes.append(hash(property))
    #         except:
    #             try:
    #                 property_hashes.append(hash(property.json()))
    #             except AttributeError:
    #                 property_hashes.append(hash(str(property)))
    #     return hash((*property_hashes, StencilComputation))
