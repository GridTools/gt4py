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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dace
import dace.data
import numpy as np

import eve
import gtc.oir as oir
from eve import NodeVisitor
from gtc import common
from gtc.common import CartesianOffset, data_type_to_typestr
from gtc.passes.oir_optimizations.utils import AccessCollector, compute_horizontal_block_extents


if TYPE_CHECKING:
    from gtc import daceir as dcir


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


def axes_list_from_flags(flags):
    from gtc import daceir as dcir

    return [ax for f, ax in zip(flags, dcir.Axis.dims_3d()) if f]


def data_type_to_dace_typeclass(data_type):
    dtype = np.dtype(data_type_to_typestr(data_type))
    return dace.dtypes.typeclass(dtype.type)


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
        regions = node.mask.iter_tree().if_isinstance(common.HorizontalMask).to_list()

        self.visit(node.mask, is_conditional=is_conditional, **kwargs)
        self.visit(node.body, is_conditional=True, regions=regions, **kwargs)

    def visit_While(self, node: oir.While, *, is_conditional=False, **kwargs):
        self.generic_visit(node, is_conditional=True, **kwargs)

    @staticmethod
    def _global_grid_subset(
        regions: List[common.HorizontalMask],
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

        offset = list(offset_node.to_dict()[k] for k in "ijk")
        if isinstance(offset_node, oir.VariableKOffset):
            variable_offset_axes = [dcir.Axis.K]
        else:
            variable_offset_axes = []

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


def mask_includes_inner_domain(mask: common.HorizontalMask):
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


def layout_maker_factory(base_layout):
    def layout_maker(mask):
        ranks = []
        for m, l in zip(mask, base_layout):
            if m:
                ranks.append(l)
        if len(mask) > 3:
            if base_layout[2] == 2:
                ranks.extend(3 + c for c in range(len(mask) - 3))
            else:
                ranks.extend(-c for c in range(len(mask) - 3))

        res_layout = [0] * len(ranks)
        for i, idx in enumerate(np.argsort(ranks)):
            res_layout[idx] = i
        return tuple(res_layout)

    return layout_maker
