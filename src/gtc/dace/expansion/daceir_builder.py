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

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import dace
import dace.data
import dace.library
import dace.subsets

import eve
import eve.utils
import gtc.common as common
import gtc.oir as oir
from eve import NodeTranslator, SymbolRef
from eve.iterators import iter_tree
from gtc import daceir as dcir
from gtc.dace.expansion_specification import Loop, Map, Sections, Stages
from gtc.dace.nodes import StencilComputation
from gtc.dace.utils import (
    compute_dcir_access_infos,
    flatten_list,
    get_tasklet_symbol,
    union_inout_memlets,
    union_node_grid_subsets,
    untile_memlets,
)
from gtc.definitions import Extent

from .utils import remove_horizontal_region


def _access_iter(node: oir.HorizontalExecution, get_outputs: bool):
    if get_outputs:
        xiter = eve.utils.xiter(
            itertools.chain(
                *node.iter_tree().if_isinstance(oir.AssignStmt).getattr("left").map(iter_tree)
            )
        )
    else:
        xiter = eve.utils.xiter(
            itertools.chain(
                *node.iter_tree().if_isinstance(oir.AssignStmt).getattr("right").map(iter_tree),
                *node.iter_tree().if_isinstance(oir.While).getattr("cond").map(iter_tree),
                *node.iter_tree().if_isinstance(oir.MaskStmt).getattr("mask").map(iter_tree),
            )
        )

    yield from (
        xiter.if_isinstance(oir.FieldAccess)
        .map(
            lambda acc: (
                acc.name,
                acc.offset,
                get_tasklet_symbol(acc.name, acc.offset, is_target=get_outputs),
            )
        )
        .unique(key=lambda x: x[2])
    )


def _get_tasklet_inout_memlets(node: oir.HorizontalExecution, *, get_outputs, global_ctx, **kwargs):

    access_infos = compute_dcir_access_infos(
        node,
        block_extents=global_ctx.library_node.get_extents,
        oir_decls=global_ctx.library_node.declarations,
        collect_read=not get_outputs,
        collect_write=get_outputs,
        **kwargs,
    )

    res = list()
    for name, offset, tasklet_symbol in _access_iter(node, get_outputs=get_outputs):
        access_info = access_infos[name]
        if not access_info.variable_offset_axes:
            offset_dict = offset.to_dict()
            for axis in access_info.axes():
                access_info = access_info.restricted_to_index(
                    axis, extent=(offset_dict[axis.lower()], offset_dict[axis.lower()])
                )

        memlet = dcir.Memlet(
            field=name,
            connector=tasklet_symbol,
            access_info=access_info,
            is_read=not get_outputs,
            is_write=get_outputs,
        )
        res.append(memlet)
    return res


def _all_stmts_same_region(scope_nodes, axis: dcir.Axis, interval):
    return (
        axis in dcir.Axis.dims_horizontal()
        and isinstance(interval, dcir.DomainInterval)
        and all(
            isinstance(stmt, oir.MaskStmt) and isinstance(stmt.mask, common.HorizontalMask)
            for tasklet in iter_tree(scope_nodes).if_isinstance(dcir.Tasklet)
            for stmt in tasklet.stmts
        )
        and len(
            set(
                (
                    None
                    if mask.intervals[axis.to_idx()].start is None
                    else mask.intervals[axis.to_idx()].start.level,
                    None
                    if mask.intervals[axis.to_idx()].start is None
                    else mask.intervals[axis.to_idx()].start.offset,
                    None
                    if mask.intervals[axis.to_idx()].end is None
                    else mask.intervals[axis.to_idx()].end.level,
                    None
                    if mask.intervals[axis.to_idx()].end is None
                    else mask.intervals[axis.to_idx()].end.offset,
                )
                for mask in iter_tree(scope_nodes).if_isinstance(common.HorizontalMask)
            )
        )
        == 1
    )


class DaCeIRBuilder(NodeTranslator):
    @dataclass
    class GlobalContext:
        library_node: StencilComputation
        arrays: Dict[str, dace.data.Data]

        def get_dcir_decls(
            self,
            access_infos: Dict[SymbolRef, dcir.FieldAccessInfo],
            symbol_collector: "DaCeIRBuilder.SymbolCollector",
        ) -> List[dcir.FieldDecl]:
            return [
                self._get_dcir_decl(field, access_info, symbol_collector=symbol_collector)
                for field, access_info in access_infos.items()
            ]

        def _get_dcir_decl(
            self,
            field: SymbolRef,
            access_info: dcir.FieldAccessInfo,
            symbol_collector: "DaCeIRBuilder.SymbolCollector",
        ) -> dcir.FieldDecl:
            oir_decl: oir.Decl = self.library_node.declarations[field]
            assert isinstance(oir_decl, oir.FieldDecl)
            dace_array = self.arrays[field]
            for s in dace_array.strides:
                for sym in dace.symbolic.symlist(s).values():
                    symbol_collector.add_symbol(str(sym))
            for sym in access_info.grid_subset.free_symbols:
                symbol_collector.add_symbol(sym)

            return dcir.FieldDecl(
                name=field,
                dtype=oir_decl.dtype,
                strides=[str(s) for s in dace_array.strides],
                data_dims=oir_decl.data_dims,
                access_info=access_info,
                storage=dcir.StorageType.from_dace_storage(dace.StorageType.Default),
            )

    @dataclass
    class IterationContext:
        grid_subset: dcir.GridSubset
        parent: Optional["DaCeIRBuilder.IterationContext"]

        @classmethod
        def init(cls, *args, **kwargs):
            res = cls(*args, parent=None, **kwargs)
            return res

        def push_axes_extents(self, axes_extents) -> "DaCeIRBuilder.IterationContext":
            res = self.grid_subset
            for axis, extent in axes_extents.items():
                if isinstance(res.intervals[axis], dcir.DomainInterval):
                    res__interval = dcir.DomainInterval(
                        start=dcir.AxisBound(
                            level=common.LevelMarker.START, offset=extent[0], axis=axis
                        ),
                        end=dcir.AxisBound(
                            level=common.LevelMarker.END, offset=extent[1], axis=axis
                        ),
                    )
                    res = res.set_interval(axis, res__interval)
                elif isinstance(res.intervals[axis], dcir.TileInterval):
                    tile_interval = dcir.TileInterval(
                        axis=axis,
                        start_offset=extent[0],
                        end_offset=extent[1],
                        tile_size=res.intervals[axis].tile_size,
                        domain_limit=res.intervals[axis].domain_limit,
                    )
                    res = res.set_interval(axis, tile_interval)
                # if is IndexWithExtent, do nothing.
            return DaCeIRBuilder.IterationContext(grid_subset=res, parent=self)

        def push_interval(
            self, axis: dcir.Axis, interval: Union[dcir.DomainInterval, oir.Interval]
        ) -> "DaCeIRBuilder.IterationContext":
            return DaCeIRBuilder.IterationContext(
                grid_subset=self.grid_subset.set_interval(axis, interval),
                parent=self,
            )

        def push_expansion_item(self, item: Union[Map, Loop]) -> "DaCeIRBuilder.IterationContext":

            if not isinstance(item, (Map, Loop)):
                raise ValueError

            if isinstance(item, Map):
                iterations = item.iterations
            else:
                iterations = [item]

            grid_subset = self.grid_subset
            for it in iterations:
                axis = it.axis
                if it.kind == "tiling":
                    grid_subset = grid_subset.tile(tile_sizes={axis: it.stride})
                else:
                    grid_subset = grid_subset.restricted_to_index(axis)
            return DaCeIRBuilder.IterationContext(grid_subset=grid_subset, parent=self)

        def push_expansion_items(
            self, items: Iterable[Union[Map, Loop]]
        ) -> "DaCeIRBuilder.IterationContext":
            res = self
            for item in items:
                res = res.push_expansion_item(item)
            return res

        def pop(self) -> "DaCeIRBuilder.IterationContext":
            assert self.parent is not None
            return self.parent

    @dataclass
    class SymbolCollector:
        symbol_decls: Dict[SymbolRef, dcir.SymbolDecl] = dataclasses.field(default_factory=dict)

        def add_symbol(self, name: SymbolRef, dtype: common.DataType = common.DataType.INT32):
            if name not in self.symbol_decls:
                self.symbol_decls[name] = dcir.SymbolDecl(name=name, dtype=dtype)
            else:
                assert self.symbol_decls[name].dtype == dtype

        def remove_symbol(self, name: SymbolRef):
            if name in self.symbol_decls:
                del self.symbol_decls[name]

    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> dcir.Literal:
        return dcir.Literal(value=node.value, dtype=node.dtype)

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> dcir.UnaryOp:
        return dcir.UnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs), dtype=node.dtype)

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> dcir.BinaryOp:
        return dcir.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            dtype=node.dtype,
        )

    def visit_HorizontalRestriction(
        self, node: oir.HorizontalRestriction, **kwargs: Any
    ) -> dcir.HorizontalRestriction:
        return dcir.HorizontalRestriction(mask=node.mask, body=self.visit(node.body, **kwargs))

    def visit_VariableKOffset(self, node: oir.VariableKOffset, **kwargs):
        return dcir.VariableKOffset(k=self.visit(node.k, **kwargs))

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        is_target: bool,
        targets: Set[SymbolRef],
        var_offset_fields: Set[SymbolRef],
        **kwargs: Any,
    ) -> Union[dcir.IndexAccess, dcir.ScalarAccess]:
        if node.name in var_offset_fields:
            res = dcir.IndexAccess(
                name=node.name + "__",
                offset=self.visit(
                    node.offset,
                    is_target=False,
                    targets=targets,
                    var_offset_fields=var_offset_fields,
                    **kwargs,
                ),
                data_index=node.data_index,
                dtype=node.dtype,
            )
        else:
            is_target = is_target or (
                node.name in targets and node.offset == common.CartesianOffset.zero()
            )
            name = get_tasklet_symbol(node.name, node.offset, is_target=is_target)
            if node.data_index:
                res = dcir.IndexAccess(
                    name=name, offset=None, data_index=node.data_index, dtype=node.dtype
                )
            else:
                res = dcir.ScalarAccess(name=name, dtype=node.dtype)
        if is_target:
            targets.add(node.name)
        return res

    def visit_ScalarAccess(
        self,
        node: oir.ScalarAccess,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
        symbol_collector: "DaCeIRBuilder.SymbolCollector",
        **kwargs: Any,
    ) -> dcir.ScalarAccess:
        if node.name in global_ctx.library_node.declarations:
            symbol_collector.add_symbol(node.name, dtype=node.dtype)
        return dcir.ScalarAccess(name=node.name, dtype=node.dtype)

    def visit_AssignStmt(self, node: oir.AssignStmt, *, targets, **kwargs: Any) -> dcir.AssignStmt:
        # the visiting order matters here, since targets must not contain the target symbols from the left visit
        right = self.visit(node.right, is_target=False, targets=targets, **kwargs)
        left = self.visit(node.left, is_target=True, targets=targets, **kwargs)
        return dcir.AssignStmt(left=left, right=right)

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> dcir.MaskStmt:
        return dcir.MaskStmt(
            mask=self.visit(node.mask, is_target=False, **kwargs),
            body=self.visit(node.body, **kwargs),
        )

    def visit_While(self, node: oir.While, **kwargs: Any) -> dcir.While:
        return dcir.While(
            cond=self.visit(node.cond, is_target=False, **kwargs),
            body=self.visit(node.body, **kwargs),
        )

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> dcir.Cast:
        return dcir.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> dcir.NativeFuncCall:
        return dcir.NativeFuncCall(
            func=node.func, args=self.visit(node.args, **kwargs), dtype=node.dtype
        )

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> dcir.TernaryOp:
        return dcir.TernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
            dtype=node.dtype,
        )

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        symbol_collector: "DaCeIRBuilder.SymbolCollector",
        loop_order,
        k_interval,
        **kwargs,
    ):
        # skip type checking due to https://github.com/python/mypy/issues/5485
        extent = global_ctx.library_node.get_extents(node)  # type: ignore
        decls = [self.visit(decl, **kwargs) for decl in node.declarations]
        targets: Set[str] = set()
        stmts = [
            self.visit(
                stmt,
                targets=targets,
                global_ctx=global_ctx,
                symbol_collector=symbol_collector,
                **kwargs,
            )
            for stmt in node.body
        ]

        stages_idx = next(
            idx
            for idx, item in enumerate(global_ctx.library_node.expansion_specification)
            if isinstance(item, Stages)
        )
        expansion_items = global_ctx.library_node.expansion_specification[stages_idx + 1 :]

        iteration_ctx = iteration_ctx.push_axes_extents(
            {k: v for k, v in zip(dcir.Axis.dims_horizontal(), extent)}
        )
        iteration_ctx = iteration_ctx.push_expansion_items(expansion_items)

        assert iteration_ctx.grid_subset == dcir.GridSubset.single_gridpoint()

        read_memlets = _get_tasklet_inout_memlets(
            node,
            get_outputs=False,
            global_ctx=global_ctx,
            grid_subset=iteration_ctx.grid_subset,
            k_interval=k_interval,
        )

        write_memlets = _get_tasklet_inout_memlets(
            node,
            get_outputs=True,
            global_ctx=global_ctx,
            grid_subset=iteration_ctx.grid_subset,
            k_interval=k_interval,
        )

        dcir_node = dcir.Tasklet(
            decls=decls,
            stmts=stmts,
            read_memlets=read_memlets,
            write_memlets=write_memlets,
        )

        for item in reversed(expansion_items):
            iteration_ctx = iteration_ctx.pop()
            dcir_node = self._process_iteration_item(
                [dcir_node],
                item,
                global_ctx=global_ctx,
                iteration_ctx=iteration_ctx,
                symbol_collector=symbol_collector,
                **kwargs,
            )
        # pop stages context (pushed with push_grid_subset)
        iteration_ctx.pop()
        return dcir_node

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        loop_order,
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        global_ctx: "DaCeIRBuilder.GlobalContext",
        symbol_collector: "DaCeIRBuilder.SymbolCollector",
        **kwargs,
    ):
        sections_idx, stages_idx = [
            idx
            for idx, item in enumerate(global_ctx.library_node.expansion_specification)
            if isinstance(item, (Sections, Stages))
        ]
        expansion_items = global_ctx.library_node.expansion_specification[
            sections_idx + 1 : stages_idx
        ]

        iteration_ctx = iteration_ctx.push_interval(
            dcir.Axis.K, node.interval
        ).push_expansion_items(expansion_items)

        dcir_nodes = self.generic_visit(
            node.horizontal_executions,
            iteration_ctx=iteration_ctx,
            global_ctx=global_ctx,
            symbol_collector=symbol_collector,
            loop_order=loop_order,
            k_interval=node.interval,
            **kwargs,
        )

        # if multiple horizontal executions, enforce their order by means of a state machine
        if len(dcir_nodes) > 1:
            dcir_nodes = [
                self.to_state([node], grid_subset=node.grid_subset)
                for node in flatten_list(dcir_nodes)
            ]

        for item in reversed(expansion_items):
            iteration_ctx = iteration_ctx.pop()
            dcir_nodes = self._process_iteration_item(
                scope=dcir_nodes,
                item=item,
                iteration_ctx=iteration_ctx,
                global_ctx=global_ctx,
                symbol_collector=symbol_collector,
            )
        # pop off interval
        iteration_ctx.pop()
        return dcir_nodes

    def to_dataflow(
        self,
        nodes,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
        symbol_collector: "DaCeIRBuilder.SymbolCollector",
    ):

        nodes = flatten_list(nodes)
        if all(isinstance(n, (dcir.NestedSDFG, dcir.DomainMap, dcir.Tasklet)) for n in nodes):
            return nodes
        elif not all(isinstance(n, (dcir.ComputationState, dcir.DomainLoop)) for n in nodes):
            raise ValueError("Can't mix dataflow and state nodes on same level.")

        read_memlets, write_memlets, field_memlets = union_inout_memlets(nodes)

        field_decls = global_ctx.get_dcir_decls(
            {memlet.field: memlet.access_info for memlet in field_memlets},
            symbol_collector=symbol_collector,
        )
        read_fields = {memlet.field for memlet in read_memlets}
        write_fields = {memlet.field for memlet in write_memlets}
        read_memlets = [
            memlet.remove_write() for memlet in field_memlets if memlet.field in read_fields
        ]
        write_memlets = [
            memlet.remove_read() for memlet in field_memlets if memlet.field in write_fields
        ]
        return [
            dcir.NestedSDFG(
                label=global_ctx.library_node.label,
                field_decls=field_decls,
                # NestedSDFG must have same shape on input and output, matching corresponding
                # nsdfg.sdfg's array shape
                read_memlets=read_memlets,
                write_memlets=write_memlets,
                states=nodes,
                symbol_decls=list(symbol_collector.symbol_decls.values()),
            )
        ]

    def to_state(self, nodes, *, grid_subset: dcir.GridSubset):

        nodes = flatten_list(nodes)
        if all(isinstance(n, (dcir.ComputationState, dcir.DomainLoop)) for n in nodes):
            return nodes
        elif all(isinstance(n, (dcir.NestedSDFG, dcir.DomainMap, dcir.Tasklet)) for n in nodes):
            return [dcir.ComputationState(computations=nodes, grid_subset=grid_subset)]
        else:
            raise ValueError("Can't mix dataflow and state nodes on same level.")

    def _process_map_item(
        self,
        scope_nodes,
        item: Map,
        *,
        global_ctx,
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        symbol_collector: "DaCeIRBuilder.SymbolCollector",
        **kwargs,
    ):

        grid_subset = iteration_ctx.grid_subset
        read_memlets, write_memlets, _ = union_inout_memlets(list(scope_nodes))
        scope_nodes = self.to_dataflow(
            scope_nodes, global_ctx=global_ctx, symbol_collector=symbol_collector
        )

        ranges = []
        for iteration in item.iterations:
            axis = iteration.axis
            interval = iteration_ctx.grid_subset.intervals[axis]
            grid_subset = grid_subset.set_interval(axis, interval)
            if iteration.kind == "tiling":
                read_memlets = untile_memlets(read_memlets, axes=[axis])
                write_memlets = untile_memlets(write_memlets, axes=[axis])
                if not axis == dcir.Axis.K:
                    interval = dcir.DomainInterval(
                        start=dcir.AxisBound.from_common(axis, oir.AxisBound.start()),
                        end=dcir.AxisBound.from_common(axis, oir.AxisBound.end()),
                    )
                ranges.append(
                    dcir.Range(
                        var=axis.tile_symbol(),
                        interval=interval,
                        stride=iteration.stride,
                    )
                )
            else:
                if _all_stmts_same_region(scope_nodes, axis, interval):
                    horizontal_mask_interval = next(
                        iter(
                            (
                                mask.intervals[axis.to_idx()]
                                for mask in iter_tree(scope_nodes).if_isinstance(
                                    common.HorizontalMask
                                )
                            )
                        )
                    )
                    interval = dcir.DomainInterval.intersection(
                        axis, horizontal_mask_interval, interval
                    )
                    scope_nodes = remove_horizontal_region(scope_nodes, axis)
                assert iteration.kind == "contiguous"
                read_memlets = [
                    dcir.Memlet(
                        field=memlet.field,
                        connector=memlet.connector,
                        access_info=memlet.access_info.apply_iteration(
                            dcir.GridSubset.from_interval(interval, axis)
                        ),
                        is_read=True,
                        is_write=False,
                    )
                    for memlet in read_memlets
                ]

                write_memlets = [
                    dcir.Memlet(
                        field=memlet.field,
                        connector=memlet.connector,
                        access_info=memlet.access_info.apply_iteration(
                            dcir.GridSubset.from_interval(interval, axis)
                        ),
                        is_read=False,
                        is_write=True,
                    )
                    for memlet in write_memlets
                ]
                ranges.append(dcir.Range.from_axis_and_interval(axis, interval))

        return [
            dcir.DomainMap(
                computations=scope_nodes,
                index_ranges=ranges,
                schedule=dcir.MapSchedule.from_dace_schedule(item.schedule),
                read_memlets=read_memlets,
                write_memlets=write_memlets,
                grid_subset=grid_subset,
            )
        ]

    def _process_loop_item(
        self,
        scope_nodes,
        item: Loop,
        *,
        global_ctx,
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        symbol_collector: "DaCeIRBuilder.SymbolCollector",
        **kwargs,
    ):

        grid_subset = union_node_grid_subsets(list(scope_nodes))
        read_memlets, write_memlets, _ = union_inout_memlets(list(scope_nodes))
        scope_nodes = self.to_state(scope_nodes, grid_subset=grid_subset)

        axis = item.axis
        interval = iteration_ctx.grid_subset.intervals[axis]
        grid_subset = grid_subset.set_interval(axis, interval)
        if item.kind == "tiling":
            raise NotImplementedError("Tiling as a state machine not implemented.")

        assert item.kind == "contiguous"
        read_memlets = [
            dcir.Memlet(
                field=memlet.field,
                connector=memlet.connector,
                access_info=memlet.access_info.apply_iteration(
                    dcir.GridSubset.from_interval(interval, axis)
                ),
                is_read=True,
                is_write=False,
            )
            for memlet in read_memlets
        ]

        write_memlets = [
            dcir.Memlet(
                field=memlet.field,
                connector=memlet.connector,
                access_info=memlet.access_info.apply_iteration(
                    dcir.GridSubset.from_interval(interval, axis)
                ),
                is_read=False,
                is_write=True,
            )
            for memlet in write_memlets
        ]

        assert not isinstance(interval, dcir.IndexWithExtent)
        index_range = dcir.Range.from_axis_and_interval(axis, interval, stride=item.stride)
        for sym in index_range.free_symbols:
            symbol_collector.add_symbol(sym, common.DataType.INT32)
        symbol_collector.remove_symbol(index_range.var)
        return [
            dcir.DomainLoop(
                axis=axis,
                loop_states=scope_nodes,
                index_range=index_range,
                read_memlets=read_memlets,
                write_memlets=write_memlets,
                grid_subset=grid_subset,
            )
        ]

    def _process_iteration_item(self, scope, item, **kwargs):
        if isinstance(item, Map):
            return self._process_map_item(scope, item, **kwargs)
        elif isinstance(item, Loop):
            return self._process_loop_item(scope, item, **kwargs)
        else:
            raise ValueError("Invalid expansion specification set.")

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
        **kwargs,
    ):
        start, end = (node.sections[0].interval.start, node.sections[0].interval.end)

        overall_interval = dcir.DomainInterval(
            start=dcir.AxisBound(axis=dcir.Axis.K, level=start.level, offset=start.offset),
            end=dcir.AxisBound(axis=dcir.Axis.K, level=end.level, offset=end.offset),
        )
        overall_extent = Extent.zeros(2)
        for he in node.iter_tree().if_isinstance(oir.HorizontalExecution):
            overall_extent = overall_extent.union(global_ctx.library_node.get_extents(he))

        iteration_ctx = DaCeIRBuilder.IterationContext.init(
            grid_subset=dcir.GridSubset.from_gt4py_extent(overall_extent).set_interval(
                axis=dcir.Axis.K, interval=overall_interval
            )
        )

        var_offset_fields = {
            acc.name
            for acc in node.iter_tree().if_isinstance(oir.FieldAccess)
            if isinstance(acc.offset, oir.VariableKOffset)
        }
        sections_idx = next(
            idx
            for idx, item in enumerate(global_ctx.library_node.expansion_specification)
            if isinstance(item, Sections)
        )
        expansion_items = global_ctx.library_node.expansion_specification[:sections_idx]
        iteration_ctx = iteration_ctx.push_expansion_items(expansion_items)

        symbol_collector = DaCeIRBuilder.SymbolCollector()
        sections = flatten_list(
            self.generic_visit(
                node.sections,
                loop_order=node.loop_order,
                global_ctx=global_ctx,
                iteration_ctx=iteration_ctx,
                symbol_collector=symbol_collector,
                var_offset_fields=var_offset_fields,
                **kwargs,
            )
        )
        if node.loop_order != common.LoopOrder.PARALLEL:
            sections = [self.to_state(s, grid_subset=iteration_ctx.grid_subset) for s in sections]
        computations = sections
        for item in reversed(expansion_items):
            iteration_ctx = iteration_ctx.pop()
            computations = self._process_iteration_item(
                scope=computations,
                item=item,
                iteration_ctx=iteration_ctx,
                global_ctx=global_ctx,
                symbol_collector=symbol_collector,
            )

        read_memlets, write_memlets, field_memlets = union_inout_memlets(computations)

        field_decls = global_ctx.get_dcir_decls(
            {memlet.field: memlet.access_info for memlet in field_memlets},
            symbol_collector=symbol_collector,
        )

        read_fields = set(memlet.field for memlet in read_memlets)
        write_fields = set(memlet.field for memlet in write_memlets)
        res = dcir.NestedSDFG(
            label=global_ctx.library_node.label,
            states=self.to_state(computations, grid_subset=iteration_ctx.grid_subset),
            field_decls=field_decls,
            read_memlets=[memlet for memlet in field_memlets if memlet.field in read_fields],
            write_memlets=[memlet for memlet in field_memlets if memlet.field in write_fields],
            symbol_decls=list(symbol_collector.symbol_decls.values()),
        )

        return res
