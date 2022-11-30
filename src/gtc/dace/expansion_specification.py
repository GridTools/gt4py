# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Union

import dace

from gtc import common
from gtc import daceir as dcir
from gtc import oir
from gtc.definitions import Extent


if TYPE_CHECKING:
    from .nodes import StencilComputation

_EXPANSION_VALIDITY_CHECKS: List[Callable] = []


def _register_validity_check(x):
    _EXPANSION_VALIDITY_CHECKS.append(x)
    return x


@dataclass
class ExpansionItem:
    pass


@dataclass
class Iteration:
    axis: dcir.Axis
    kind: str  # tiling, contiguous
    # if stride is not specified, it is chosen based on backend (tiling) and loop order (K)
    stride: Optional[int] = None

    @property
    def iterations(self) -> List["Iteration"]:
        return [self]


@dataclass
class Map(ExpansionItem):
    iterations: List[Iteration]
    schedule: Optional[dace.ScheduleType] = None


@dataclass
class Loop(Iteration, ExpansionItem):
    kind: str = "contiguous"
    storage: dace.StorageType = None

    @property
    def iterations(self) -> List[Iteration]:
        return [self]


@dataclass
class Stages(ExpansionItem):
    pass


@dataclass
class Sections(ExpansionItem):
    pass


def _get_axis_from_pattern(item, fmt):
    for axis in dcir.Axis.dims_3d():
        if fmt.format(axis=axis) == item:
            return axis
    return ""


def _is_domain_loop(item):
    return _get_axis_from_pattern(item, fmt="{axis}Loop")


def _is_domain_map(item):
    return _get_axis_from_pattern(item, fmt="{axis}Map")


def _is_tiling(item):
    return _get_axis_from_pattern(item, fmt="Tile{axis}")


def get_expansion_order_axis(item):
    if axis := (
        _is_domain_map(item)
        or _is_domain_loop(item)
        or _is_tiling(item)
        or _get_axis_from_pattern(item, fmt="{axis}")
    ):
        return dcir.Axis(axis)
    raise ValueError(f"Can't get axis for item '{item}'.")


def get_expansion_order_index(expansion_order, axis):
    for idx, item in enumerate(expansion_order):
        if isinstance(item, Iteration) and item.axis == axis:
            return idx
        elif isinstance(item, Map):
            for it in item.iterations:
                if it.kind == "contiguous" and it.axis == axis:
                    return idx


def _is_expansion_order_implemented(expansion_specification):

    for item in expansion_specification:
        if isinstance(item, Sections):
            break
        if isinstance(item, Iteration) and item.axis == dcir.Axis.K:
            return False
        if isinstance(item, Map) and any(it.axis == dcir.Axis.K for it in item.iterations):
            return False

    return True


def _choose_loop_or_map(node, eo):
    if any(eo == axis for axis in dcir.Axis.dims_horizontal()):
        return f"{eo}Map"
    if eo == dcir.Axis.K:
        if node.oir_node.loop_order == common.LoopOrder.PARALLEL:
            return f"{eo}Map"
        else:
            return f"{eo}Loop"
    return eo


def _order_as_spec(computation_node, expansion_order):
    expansion_order = list(_choose_loop_or_map(computation_node, eo) for eo in expansion_order)
    expansion_specification = []
    for item in expansion_order:
        if isinstance(item, ExpansionItem):
            expansion_specification.append(item)
        elif axis := _is_tiling(item):
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
        elif axis := _is_domain_map(item):
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
        elif axis := _is_domain_loop(item):
            expansion_specification.append(
                Loop(
                    axis=axis,
                    stride=-1
                    if computation_node.oir_node.loop_order == common.LoopOrder.BACKWARD
                    else 1,
                )
            )
        elif item == "Sections":
            expansion_specification.append(Sections())
        else:
            assert item == "Stages", item
            expansion_specification.append(Stages())

    return expansion_specification


def _populate_strides(node, expansion_specification):
    """Fill in `stride` attribute of `Iteration` and `Loop` dataclasses.

    For loops, stride is set to either -1 or 1, based on iteration order.
    For tiling maps, the stride is chosen such that the resulting tile size
    is that of the tile_size attribute.
    Other maps get stride 1.
    """
    assert all(isinstance(es, ExpansionItem) for es in expansion_specification)

    iterations = [it for item in expansion_specification for it in getattr(item, "iterations", [])]

    for it in iterations:
        if isinstance(it, Loop):
            if it.stride is None:
                if node.oir_node.loop_order == common.LoopOrder.BACKWARD:
                    it.stride = -1
                else:
                    it.stride = 1
        else:
            if it.stride is None:
                if it.kind == "tiling":
                    if node.extents is not None and it.axis.to_idx() < 2:
                        extent = Extent.zeros(2)
                        for he_extent in node.extents.values():
                            extent = extent.union(he_extent)
                        extent = extent[it.axis.to_idx()]
                    else:
                        extent = (0, 0)
                    it.stride = node.tile_strides.get(it.axis, 8)
                else:
                    it.stride = 1


def _populate_storages(self, expansion_specification):
    assert all(isinstance(es, ExpansionItem) for es in expansion_specification)
    innermost_axes = set(dcir.Axis.dims_3d())
    tiled_axes = set()
    for item in expansion_specification:
        if isinstance(item, Map):
            for it in item.iterations:
                if it.kind == "tiling":
                    tiled_axes.add(it.axis)
    for es in reversed(expansion_specification):
        if isinstance(es, Map):
            for it in es.iterations:
                if it.axis in innermost_axes:
                    innermost_axes.remove(it.axis)
                if it.kind == "tiling":
                    tiled_axes.remove(it.axis)


def _populate_cpu_schedules(self, expansion_specification):
    is_outermost = True
    for es in expansion_specification:
        if isinstance(es, Map):
            if es.schedule is None:
                if is_outermost:
                    es.schedule = dace.ScheduleType.CPU_Multicore
                    is_outermost = False
                else:
                    es.schedule = dace.ScheduleType.Default


def _populate_gpu_schedules(self, expansion_specification):
    # On GPU if any dimension is tiled and has a contiguous map in the same axis further in
    # pick those two maps as Device/ThreadBlock maps. If not, Make just device map with
    # default blocksizes
    is_outermost = True
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


def _populate_schedules(self, expansion_specification):
    assert all(isinstance(es, ExpansionItem) for es in expansion_specification)
    assert hasattr(self, "_device")
    if self.device == dace.DeviceType.GPU:
        _populate_gpu_schedules(self, expansion_specification)
    else:
        _populate_cpu_schedules(self, expansion_specification)


def _collapse_maps_gpu(self, expansion_specification):
    def _union_map_items(last_item, next_item):
        if last_item.schedule == next_item.schedule:
            return (
                Map(
                    iterations=last_item.iterations + next_item.iterations,
                    schedule=last_item.schedule,
                ),
            )

        if next_item.schedule is None or next_item.schedule == dace.ScheduleType.Default:
            specified_item = last_item
        else:
            specified_item = next_item

        if specified_item.schedule is not None and not specified_item == dace.ScheduleType.Default:
            return (
                Map(
                    iterations=last_item.iterations + next_item.iterations,
                    schedule=specified_item.schedule,
                ),
            )

        # one is default and the other None
        return (
            Map(
                iterations=last_item.iterations + next_item.iterations,
                schedule=dace.ScheduleType.Default,
            ),
        )

    res_items = []
    for item in expansion_specification:
        if isinstance(item, Map):
            if not res_items or not isinstance(res_items[-1], Map):
                res_items.append(item)
            else:
                res_items[-1:] = _union_map_items(last_item=res_items[-1], next_item=item)
        else:
            res_items.append(item)
    for item in res_items:
        if isinstance(item, Map) and (
            item.schedule is None or item.schedule == dace.ScheduleType.Default
        ):
            item.schedule = dace.ScheduleType.Sequential
    return res_items


def _collapse_maps_cpu(self, expansion_specification):
    res_items = []
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
    return res_items


def _collapse_maps(self, expansion_specification):
    assert hasattr(self, "_device")
    if self.device == dace.DeviceType.GPU:
        res_items = _collapse_maps_gpu(self, expansion_specification)
    else:
        res_items = _collapse_maps_cpu(self, expansion_specification)
    expansion_specification.clear()
    expansion_specification.extend(res_items)


def make_expansion_order(
    node: "StencilComputation", expansion_order: Union[List[str], List[ExpansionItem]]
) -> List[ExpansionItem]:
    if expansion_order is None:
        return None
    expansion_order = copy.deepcopy(expansion_order)
    expansion_specification = _order_as_spec(node, expansion_order)

    if not _is_expansion_order_implemented(expansion_specification):
        raise ValueError("Provided StencilComputation.expansion_order is not supported.")
    if node.oir_node is not None:
        if not is_expansion_order_valid(node, expansion_specification):
            raise ValueError("Provided StencilComputation.expansion_order is invalid.")

    _populate_strides(node, expansion_specification)
    _populate_schedules(node, expansion_specification)
    _collapse_maps(node, expansion_specification)
    _populate_storages(node, expansion_specification)
    return expansion_specification


def _k_inside_dims(node: "StencilComputation"):
    # Putting K inside of i or j is valid if
    # * K parallel or
    # * All reads with k-offset to values modified in same HorizontalExecution are not
    #   to fields that are also accessed horizontally (in I or J, respectively)
    #   (else, race condition in other column)

    if node.oir_node.loop_order == common.LoopOrder.PARALLEL:
        return {dcir.Axis.I, dcir.Axis.J}

    res = {dcir.Axis.I, dcir.Axis.J}
    for section in node.oir_node.sections:
        for he in section.horizontal_executions:
            i_offset_fields = set(
                (
                    acc.name
                    for acc in he.walk_values().if_isinstance(oir.FieldAccess)
                    if acc.offset.to_dict()["i"] != 0
                )
            )
            j_offset_fields = set(
                (
                    acc.name
                    for acc in he.walk_values().if_isinstance(oir.FieldAccess)
                    if acc.offset.to_dict()["j"] != 0
                )
            )
            k_offset_fields = set(
                (
                    acc.name
                    for acc in he.walk_values().if_isinstance(oir.FieldAccess)
                    if isinstance(acc.offset, oir.VariableKOffset) or acc.offset.to_dict()["k"] != 0
                )
            )
            modified_fields: Set[str] = (
                he.walk_values()
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


def _k_inside_stages(node: "StencilComputation"):
    # Putting K inside of stages is valid if
    # * K parallel
    # * not "ahead" in order of iteration to fields that are modified in previous
    #   HorizontalExecutions (else, reading updated values that should be old)

    if node.oir_node.loop_order == common.LoopOrder.PARALLEL:
        return True

    for section in node.oir_node.sections:
        modified_fields: Set[str] = set()
        for he in section.horizontal_executions:
            if modified_fields:
                ahead_acc = list()
                for acc in he.walk_values().if_isinstance(oir.FieldAccess):
                    if (
                        isinstance(acc.offset, oir.VariableKOffset)
                        or (
                            node.oir_node.loop_order == common.LoopOrder.FORWARD
                            and acc.offset.k > 0
                        )
                        or (
                            node.oir_node.loop_order == common.LoopOrder.BACKWARD
                            and acc.offset.k < 0
                        )
                    ):
                        ahead_acc.append(acc)
                if any(acc.name in modified_fields for acc in ahead_acc):
                    return False

            modified_fields.update(
                he.walk_values()
                .if_isinstance(oir.AssignStmt)
                .getattr("left")
                .if_isinstance(oir.FieldAccess)
                .getattr("name")
                .to_set()
            )

    return True


@_register_validity_check
def _sequential_as_loops(
    node: "StencilComputation", expansion_specification: List[ExpansionItem]
) -> bool:
    # K can't be Map if not parallel
    if node.oir_node.loop_order != common.LoopOrder.PARALLEL and any(
        (isinstance(item, Map) and any(it.axis == dcir.Axis.K for it in item.iterations))
        for item in expansion_specification
    ):
        return False
    return True


@_register_validity_check
def _stages_inside_sections(expansion_specification: List[ExpansionItem], **kwargs) -> bool:
    # Oir defines that HorizontalExecutions have to be applied per VerticalLoopSection. A meaningful inversion of this
    # is not possible.
    sections_idx = next(
        idx for idx, item in enumerate(expansion_specification) if isinstance(item, Sections)
    )
    stages_idx = next(
        idx for idx, item in enumerate(expansion_specification) if isinstance(item, Stages)
    )
    if stages_idx < sections_idx:
        return False
    return True


@_register_validity_check
def _k_inside_ij_valid(
    node: "StencilComputation", expansion_specification: List[ExpansionItem]
) -> bool:
    # OIR defines that horizontal maps go inside vertical K loop (i.e. all grid points are updated in a
    # HorizontalExecution before the computation of the next one is executed.).  Under certain conditions the semantics
    # remain unchanged even if a single horizontal map is executing all contained HorizontalExecution nodes.
    # Note: Opposed to e.g. Fusions in OIR, this can here be done on a per-dimension basis. See `_k_inside_dims` for
    # details.
    for axis in dcir.Axis.dims_horizontal():
        if get_expansion_order_index(expansion_specification, axis) < get_expansion_order_index(
            expansion_specification, dcir.Axis.K
        ) and axis not in _k_inside_dims(node):
            return False
    return True


@_register_validity_check
def _k_inside_stages_valid(
    node: "StencilComputation", expansion_specification: List[ExpansionItem]
) -> bool:
    # OIR defines that all horizontal executions of a VerticalLoopSection are run per level. Under certain conditions
    # the semantics remain unchanged even if the k loop is run per horizontal execution. See `_k_inside_stages` for
    # details
    stages_idx = next(
        idx for idx, item in enumerate(expansion_specification) if isinstance(item, Stages)
    )
    if stages_idx < get_expansion_order_index(
        expansion_specification, dcir.Axis.K
    ) and not _k_inside_stages(node):
        return False
    return True


@_register_validity_check
def _ij_outside_sections_valid(
    node: "StencilComputation", expansion_specification: List[ExpansionItem]
) -> bool:
    # If there are multiple horizontal executions in any section, IJ iteration must go inside sections.
    # TODO: do mergeability checks on a per-axis basis.
    for item in expansion_specification:
        if isinstance(item, Sections):
            break
        if isinstance(item, (Map, Loop, Iteration)):
            for it in item.iterations:
                if it.axis in dcir.Axis.dims_horizontal() and it.kind == "contiguous":
                    if any(
                        len(section.horizontal_executions) > 1 for section in node.oir_node.sections
                    ):
                        return False

    # if there are horizontal executions with different iteration ranges in an axis across sections,
    # that iteration must be per section
    # TODO less conservative: allow different domains if all outputs smaller than bounding box are temporaries
    # TODO implement/allow this with predicates implicit regions / predicates
    for item in expansion_specification:
        if isinstance(item, Sections):
            break
        for it in getattr(item, "iterations", []):
            if it.axis in dcir.Axis.dims_horizontal() and it.kind == "contiguous":
                xiter = iter(node.oir_node.walk_values().if_isinstance(oir.HorizontalExecution))
                extent = node.get_extents(next(xiter))
                for he in xiter:
                    if node.get_extents(he)[it.axis.to_idx()] != extent[it.axis.to_idx()]:
                        return False
    return True


@_register_validity_check
def _iterates_domain(expansion_specification: List[ExpansionItem], **kwargs) -> bool:
    # There must be exactly one iteration per dimension, except for tiled dimensions, where a Tiling has to go outside
    # and the corresponding contiguous iteration inside.
    tiled_axes = set()
    contiguous_axes = set()
    for item in expansion_specification:
        if isinstance(item, (Map, Loop, Iteration)):
            for it in item.iterations:
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
    return True


def is_expansion_order_valid(node: "StencilComputation", expansion_order) -> bool:
    """Check if a given expansion specification valid.

    That is, it is semantically valid for the StencilComputation node that is to be configured and currently
    implemented.
    """
    expansion_specification = list(_choose_loop_or_map(node, eo) for eo in expansion_order)

    for check in _EXPANSION_VALIDITY_CHECKS:
        if not check(node=node, expansion_specification=expansion_specification):
            return False

    return _is_expansion_order_implemented(expansion_specification)
