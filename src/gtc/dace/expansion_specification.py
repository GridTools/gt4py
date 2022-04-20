import copy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import combinations, permutations, product
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import dace

from gtc import common
from gtc import daceir as dcir
from gtc import oir


if TYPE_CHECKING:
    from .nodes import StencilComputation


@dataclass
class ExpansionItem:
    pass


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
    expansion_order = list(_sanitize_expansion_item(computation_node, eo) for eo in expansion_order)
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
                    if mask_stmt.mask.iter_tree().if_isinstance(common.HorizontalMask).to_list():
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
                            extent = Extent.zeros(2)
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
        if not is_expansion_order_valid(node, expansion_specification):
            raise ValueError("Provided StencilComputation.expansion_order is invalid.")

    _populate_strides(node, expansion_specification)
    _populate_schedules(node, expansion_specification)
    _collapse_maps(node, expansion_specification)
    _populate_storages(node, expansion_specification)
    return expansion_specification


def is_expansion_order_valid(
    node: "StencilComputation", expansion_order, *, k_inside_dims=None, k_inside_stages=None
):
    expansion_specification = list(_sanitize_expansion_item(node, eo) for eo in expansion_order)
    if k_inside_dims is None:
        k_inside_dims = _k_inside_dims(node)
    if k_inside_stages is None:
        k_inside_stages = _k_inside_stages(node)

    # K can't be Map if not parallel
    if node.oir_node.loop_order != common.LoopOrder.PARALLEL and any(
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
                        len(section.horizontal_executions) > 1 for section in node.oir_node.sections
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
                xiter = iter(node.oir_node.iter_tree().if_isinstance(oir.HorizontalExecution))
                extent = node.get_extents(next(xiter))
                for he in xiter:
                    if node.get_extents(he)[it.axis.to_idx()] != extent[it.axis.to_idx()]:
                        return False

    res = _is_expansion_order_implemented(expansion_specification)
    return res


def _k_inside_dims(node: "StencilComputation"):
    # Putting K inside of i or j is valid if
    # * K parallel
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
                    for acc in he.iter_tree().if_isinstance(oir.FieldAccess)
                    if acc.offset.to_dict()["i"] != 0
                )
            )
            j_offset_fields = set(
                (
                    acc.name
                    for acc in he.iter_tree().if_isinstance(oir.FieldAccess)
                    if acc.offset.to_dict()["j"] != 0
                )
            )
            k_offset_fields = set(
                (
                    acc.name
                    for acc in he.iter_tree().if_isinstance(oir.FieldAccess)
                    if isinstance(acc.offset, oir.VariableKOffset) or acc.offset.to_dict()["k"] != 0
                )
            )
            modified_fields: Set[str] = (
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
                for acc in he.iter_tree().if_isinstance(oir.FieldAccess):
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
                he.iter_tree()
                .if_isinstance(oir.AssignStmt)
                .getattr("left")
                .if_isinstance(oir.FieldAccess)
                .getattr("name")
                .to_set()
            )

    return True


def valid_expansion_orders(node: "StencilComputation"):
    optionals = {"TileI", "TileJ", "TileK"}
    required: Set[Union[Tuple[str, ...], str]] = {
        ("KMap", "KLoop", "CachedKLoop"),  # tuple represents "one of",
        ("JMap", "JLoop", "CachedJLoop"),
        ("IMap", "ILoop", "CachedILoop"),
    }
    prepends: List[str] = []
    appends: List[str] = []
    if len(node.oir_node.sections) > 1:
        required.add("Sections")
    else:
        prepends.append("Sections")
    if any(len(section.horizontal_executions) > 1 for section in node.oir_node.sections):
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
            expansion_specification = _order_as_spec(node, expansion_order)
            if is_expansion_order_valid(
                node,
                expansion_specification,
                k_inside_dims=_k_inside_dims(node),
                k_inside_stages=_k_inside_stages(node),
            ):
                yield expansion_order


def _sanitize_expansion_item(node, eo):
    if any(eo == axis for axis in dcir.Axis.dims_horizontal()):
        return f"{eo}Map"
    if eo == dcir.Axis.K:
        if node.oir_node.loop_order == common.LoopOrder.PARALLEL:
            return f"{eo}Map"
        else:
            return f"Cached{eo}Loop"
    return eo
