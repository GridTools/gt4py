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

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import dace
import dace.data
import numpy as np

import eve
from gtc import common
from gtc import daceir as dcir
from gtc import oir
from gtc.common import CartesianOffset
from gtc.passes.oir_optimizations.utils import compute_horizontal_block_extents


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
        axes = [ax for ax, m in zip("IJK", dims) if m] + [str(i) for i in range(ndata_dims)]
        layout = get_layout_map(axes)
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
    if offset is not None:
        offset_strs = []
        for axis in dcir.Axis.dims_3d():
            off = offset.to_dict()[axis.lower()]
            if off is not None and off != 0:
                offset_strs.append(axis.lower() + ("m" if off < 0 else "p") + f"{abs(off):d}")
        suffix = "_".join(offset_strs)
        if suffix != "":
            acc_name += suffix
    return acc_name


def axes_list_from_flags(flags):
    return [ax for f, ax in zip(flags, dcir.Axis.dims_3d()) if f]


class AccessInfoCollector(eve.NodeVisitor):
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
        inner_ctx = self.Context(axes=ctx.axes)

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
        horizontal_extent = block_extents(node)

        inner_ctx = self.Context(axes=ctx.axes)
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

    def visit_HorizontalRestriction(
        self, node: oir.HorizontalRestriction, *, is_conditional=False, **kwargs
    ):
        self.visit(node.mask, is_conditional=is_conditional, **kwargs)
        self.visit(node.body, is_conditional=True, region=node.mask, **kwargs)

    def visit_MaskStmt(self, node: oir.MaskStmt, *, is_conditional=False, **kwargs):

        self.visit(node.mask, is_conditional=is_conditional, **kwargs)
        self.visit(node.body, is_conditional=True, **kwargs)

    def visit_While(self, node: oir.While, *, is_conditional=False, **kwargs):
        self.generic_visit(node, is_conditional=True, **kwargs)

    @staticmethod
    def _global_grid_subset(
        region: common.HorizontalMask, he_grid: "dcir.GridSubset", offset: List[Optional[int]]
    ):
        res: Dict[
            dcir.Axis, Union[dcir.DomainInterval, dcir.TileInterval, dcir.IndexWithExtent]
        ] = {}
        if region is not None:
            for axis, oir_interval in zip(dcir.Axis.dims_horizontal(), region.intervals):
                he_grid_interval = he_grid.intervals[axis]
                assert isinstance(he_grid_interval, dcir.DomainInterval)
                start = (
                    oir_interval.start if oir_interval.start is not None else he_grid_interval.start
                )
                end = oir_interval.end if oir_interval.end is not None else he_grid_interval.end
                dcir_interval = dcir.DomainInterval(
                    start=dcir.AxisBound.from_common(axis, start),
                    end=dcir.AxisBound.from_common(axis, end),
                )
                res[axis] = dcir.DomainInterval.union(dcir_interval, res.get(axis, dcir_interval))
        if dcir.Axis.K in he_grid.intervals:
            off = offset[dcir.Axis.K.to_idx()] or 0
            he_grid_k_interval = he_grid.intervals[dcir.Axis.K]
            assert not isinstance(he_grid_k_interval, dcir.TileInterval)
            res[dcir.Axis.K] = he_grid_k_interval.shifted(off)
        for axis in dcir.Axis.dims_horizontal():
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
        region,
        he_grid,
        grid_subset,
    ) -> "dcir.FieldAccessInfo":
        offset = [offset_node.to_dict()[k] for k in "ijk"]
        if isinstance(offset_node, oir.VariableKOffset):
            variable_offset_axes = [dcir.Axis.K]
        else:
            variable_offset_axes = []

        global_subset = self._global_grid_subset(region, he_grid, offset)
        intervals = {}
        for axis in axes:
            if axis in variable_offset_axes:
                intervals[axis] = dcir.IndexWithExtent(
                    axis=axis, value=axis.iteration_symbol(), extent=(0, 0)
                )
            else:
                intervals[axis] = dcir.IndexWithExtent(
                    axis=axis,
                    value=axis.iteration_symbol(),
                    extent=(offset[axis.to_idx()], offset[axis.to_idx()]),
                )
        grid_subset = dcir.GridSubset(intervals=intervals)
        return dcir.FieldAccessInfo(
            grid_subset=grid_subset,
            global_grid_subset=global_subset,
            dynamic_access=len(variable_offset_axes) > 0 or is_conditional or region is not None,
            variable_offset_axes=variable_offset_axes,
        )

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        he_grid,
        grid_subset,
        is_write: bool = False,
        is_conditional: bool = False,
        region=None,
        ctx: "AccessInfoCollector.Context",
        **kwargs,
    ):
        self.visit(
            node.offset,
            is_conditional=is_conditional,
            ctx=ctx,
            is_write=False,
            region=region,
            he_grid=he_grid,
            grid_subset=grid_subset,
            **kwargs,
        )

        if (is_write and not self.collect_write) or (not is_write and not self.collect_read):
            return

        access_info = self._make_access_info(
            node.offset,
            axes=ctx.axes[node.name],
            is_conditional=is_conditional,
            region=region,
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
) -> dace.properties.DictProperty:
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


def make_dace_subset(
    context_info: "dcir.FieldAccessInfo",
    access_info: "dcir.FieldAccessInfo",
    data_dims: Tuple[int, ...],
) -> dace.subsets.Range:
    clamped_access_info = access_info
    clamped_context_info = context_info
    for axis in access_info.axes():
        if axis in access_info.variable_offset_axes:
            clamped_access_info = clamped_access_info.clamp_full_axis(axis)
        if axis in clamped_context_info.variable_offset_axes:
            clamped_context_info = clamped_context_info.clamp_full_axis(axis)
    res_ranges = []

    for axis in clamped_access_info.axes():
        context_start, _ = clamped_context_info.grid_subset.intervals[axis].to_dace_symbolic()
        subset_start, subset_end = clamped_access_info.grid_subset.intervals[
            axis
        ].to_dace_symbolic()
        res_ranges.append((subset_start - context_start, subset_end - context_start - 1, 1))
    res_ranges.extend((0, dim - 1, 1) for dim in data_dims)
    return dace.subsets.Range(res_ranges)


def untile_memlets(
    memlets: Sequence["dcir.Memlet"], axes: Sequence["dcir.Axis"]
) -> List["dcir.Memlet"]:
    res_memlets: List["dcir.Memlet"] = []
    for memlet in memlets:
        res_memlets.append(
            dcir.Memlet(
                field=memlet.field,
                access_info=memlet.access_info.untile(axes),
                connector=memlet.connector,
                is_read=memlet.is_read,
                is_write=memlet.is_write,
            )
        )
    return res_memlets


def union_node_grid_subsets(nodes: List[eve.Node]):
    grid_subset = None

    for node in collect_toplevel_iteration_nodes(nodes):
        if grid_subset is None:
            grid_subset = node.grid_subset
        grid_subset = grid_subset.union(node.grid_subset)

    return grid_subset


def _union_memlets(*memlets: "dcir.Memlet") -> List["dcir.Memlet"]:
    res: Dict[str, dcir.Memlet] = {}
    for memlet in memlets:
        res[memlet.field] = memlet.union(res.get(memlet.field, memlet))
    return list(res.values())


def union_inout_memlets(nodes: List[eve.Node]):
    read_memlets: List[dcir.Memlet] = []
    write_memlets: List[dcir.Memlet] = []
    for node in collect_toplevel_computation_nodes(nodes):
        read_memlets = _union_memlets(*read_memlets, *node.read_memlets)
        write_memlets = _union_memlets(*write_memlets, *node.write_memlets)

    return (read_memlets, write_memlets, _union_memlets(*read_memlets, *write_memlets))


def flatten_list(list_or_node: Union[List[Any], eve.Node]):
    list_or_node = [list_or_node]
    while not all(isinstance(ref, eve.Node) for ref in list_or_node):
        list_or_node = [r for li in list_or_node for r in li]
    return list_or_node


def collect_toplevel_computation_nodes(
    list_or_node: Union[List[Any], eve.Node]
) -> List["dcir.ComputationNode"]:
    class ComputationNodeCollector(eve.NodeVisitor):
        def visit_ComputationNode(self, node: dcir.ComputationNode, *, collection: List):
            collection.append(node)

    collection: List[dcir.ComputationNode] = []
    ComputationNodeCollector().visit(list_or_node, collection=collection)
    return collection


def collect_toplevel_iteration_nodes(
    list_or_node: Union[List[Any], eve.Node]
) -> List["dcir.IterationNode"]:
    class IterationNodeCollector(eve.NodeVisitor):
        def visit_IterationNode(self, node: dcir.IterationNode, *, collection: List):
            collection.append(node)

    collection: List[dcir.IterationNode] = []
    IterationNodeCollector().visit(list_or_node, collection=collection)
    return collection


def layout_maker_factory(base_layout: Tuple[int, ...]) -> Callable[[List[bool]], Tuple[int, ...]]:
    def layout_maker(dimensions: List[bool]) -> Tuple[int, ...]:
        mask = [dim in dimensions for dim in "IJK"]
        mask += [True] * (len(dimensions) - sum(mask))
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
