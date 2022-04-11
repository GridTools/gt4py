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
import copy
import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dace
import dace.data
import dace.library
import dace.subsets
import numpy as np
import sympy

import eve
import gtc.common as common
import gtc.oir as oir
from eve import NodeTranslator, NodeVisitor, codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.iterators import iter_tree
from gt4py import definitions as gt_def
from gt4py.definitions import Extent
from gtc import daceir as dcir
from gtc.dace.nodes import (
    Loop,
    Map,
    Sections,
    Stages,
    StencilComputation,
    get_expansion_order_axis,
    get_expansion_order_index,
    is_domain_loop,
    is_domain_map,
)
from gtc.dace.utils import get_axis_bound_str, get_tasklet_symbol
from gtc.passes.oir_optimizations.utils import AccessCollector

from .utils import (
    compute_dcir_access_infos,
    make_subset_str,
    remove_horizontal_region,
    split_horizontal_exeuctions_regions,
)


def make_access_subset_dict(
    extent: gt_def.Extent, interval: oir.Interval, axes: List[dcir.Axis]
) -> Dict[dcir.Axis, Union[dcir.IndexWithExtent, dcir.DomainInterval]]:
    from gtc import daceir as dcir

    i_interval = dcir.DomainInterval(
        start=dcir.AxisBound(level=common.LevelMarker.START, offset=extent[0][0], axis=dcir.Axis.I),
        end=dcir.AxisBound(level=common.LevelMarker.END, offset=extent[0][1], axis=dcir.Axis.I),
    )
    j_interval = dcir.DomainInterval(
        start=dcir.AxisBound(level=common.LevelMarker.START, offset=extent[1][0], axis=dcir.Axis.J),
        end=dcir.AxisBound(level=common.LevelMarker.END, offset=extent[1][1], axis=dcir.Axis.J),
    )
    k_interval: Union[dcir.IndexWithExtent, dcir.DomainInterval]
    if isinstance(interval, dcir.IndexWithExtent):
        k_interval = interval
    else:
        k_interval = dcir.DomainInterval(
            start=dcir.AxisBound(
                level=interval.start.level,
                offset=interval.start.offset,
                axis=dcir.Axis.K,
            ),
            end=dcir.AxisBound(
                level=interval.end.level, offset=interval.end.offset, axis=dcir.Axis.K
            ),
        )
    res = {dcir.Axis.I: i_interval, dcir.Axis.J: j_interval, dcir.Axis.K: k_interval}
    return {axis: res[axis] for axis in axes}


def make_read_accesses(node: common.Node, *, global_ctx, **kwargs):
    return compute_dcir_access_infos(
        node,
        block_extents=global_ctx.block_extents,
        oir_decls=global_ctx.library_node.declarations,
        collect_read=True,
        collect_write=False,
        **kwargs,
    )


def make_write_accesses(node: common.Node, *, global_ctx, **kwargs):
    return compute_dcir_access_infos(
        node,
        block_extents=global_ctx.block_extents,
        oir_decls=global_ctx.library_node.declarations,
        collect_read=False,
        collect_write=True,
        **kwargs,
    )


class TaskletCodegen(codegen.TemplatedGenerator):

    ScalarAccess = as_fmt("{name}")

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        is_target,
        targets,
        read_accesses: Dict[str, dcir.FieldAccessInfo],
        write_accesses: Dict[str, dcir.FieldAccessInfo],
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ):
        field_decl = sdfg_ctx.field_decls[node.name]
        is_offset_nil = (
            isinstance(node.offset, common.CartesianOffset)
            and self.visit(node.offset, is_dynamic_offset=False, field_decl=field_decl) == ""
        )

        if is_target or (node.name in targets and is_offset_nil):
            access_info = write_accesses[node.name]
            is_dynamic_offset = len(access_info.variable_offset_axes) > 0
        else:
            access_info = read_accesses[node.name]
            is_dynamic_offset = len(access_info.variable_offset_axes) > 0

        if is_target or (node.name in targets and (is_offset_nil or is_dynamic_offset)):
            targets.add(node.name)
            name = "__" + node.name
        elif is_dynamic_offset:
            name = node.name + "__"
        else:
            name = (
                node.name
                + "__"
                + self.visit(
                    node.offset,
                    is_dynamic_offset=False,
                    is_target=is_target,
                    targets=targets,
                    field_decl=field_decl,
                    access_info=access_info,
                    sdfg_ctx=sdfg_ctx,
                    read_accesses=read_accesses,
                    write_accesses=write_accesses,
                )
            )
        if node.data_index or is_dynamic_offset:
            offset_str = "["
            if is_dynamic_offset:
                offset_str += self.visit(
                    node.offset,
                    is_dynamic_offset=True,
                    is_target=is_target,
                    targets=targets,
                    field_decl=field_decl,
                    access_info=access_info,
                    sdfg_ctx=sdfg_ctx,
                    read_accesses=read_accesses,
                    write_accesses=write_accesses,
                )
            if node.data_index:
                offset_str += ",".join(self.visit(node.data_index))
            offset_str += "]"
        else:
            offset_str = ""
        return name + offset_str

    def visit_CartesianOffset(
        self,
        node: common.CartesianOffset,
        *,
        is_dynamic_offset,
        field_decl,
        access_info=None,
        **kwargs,
    ):
        if is_dynamic_offset:
            return self.visit_VariableKOffset(
                node,
                is_dynamic_offset=is_dynamic_offset,
                field_decl=field_decl,
                access_info=access_info,
                **kwargs,
            )
        else:
            res = []
            if node.i != 0:
                res.append(f'i{"m" if node.i<0 else "p"}{abs(node.i):d}')
            if node.j != 0:
                res.append(f'j{"m" if node.j<0 else "p"}{abs(node.j):d}')
            if node.k != 0:
                res.append(f'k{"m" if node.k<0 else "p"}{abs(node.k):d}')
            return "_".join(res)

    def visit_VariableKOffset(
        self,
        node: Union[oir.VariableKOffset, common.CartesianOffset],
        *,
        is_dynamic_offset,
        field_decl: dcir.FieldDecl,
        **kwargs,
    ):
        assert is_dynamic_offset
        offset_strs = []
        for axis in field_decl.axes():
            if axis == dcir.Axis.K:
                raw_idx = axis.iteration_symbol() + f"+({self.visit(node.k, **kwargs)})"
                index_str = f"max(0,min({field_decl.access_info.grid_subset.intervals[axis].size},{raw_idx}))"
                offset_strs.append(index_str)
            else:
                offset_strs.append(axis.iteration_symbol())
        res: dace.subsets.Range = StencilComputationSDFGBuilder._add_origin(
            field_decl.access_info, ",".join(offset_strs), add_for_variable=True
        )
        res.ranges = [res.ranges[list(field_decl.axes()).index(dcir.Axis.K)]]
        return str(res)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs):
        right = self.visit(node.right, is_target=False, **kwargs)
        left = self.visit(node.left, is_target=True, **kwargs)
        return f"{left} = {right}"

    BinaryOp = as_fmt("({left} {op} {right})")

    UnaryOp = as_fmt("({op}{expr})")

    TernaryOp = as_fmt("({true_expr} if {cond} else {false_expr})")

    def visit_BuiltInLiteral(self, builtin: common.BuiltInLiteral, **kwargs: Any) -> str:
        if builtin == common.BuiltInLiteral.TRUE:
            return "True"
        elif builtin == common.BuiltInLiteral.FALSE:
            return "False"
        raise NotImplementedError("Not implemented BuiltInLiteral encountered.")

    Literal = as_fmt("{value}")

    Cast = as_fmt("{dtype}({expr})")

    def visit_NativeFunction(self, func: common.NativeFunction, **kwargs: Any) -> str:
        try:
            return {
                common.NativeFunction.ABS: "abs",
                common.NativeFunction.MIN: "min",
                common.NativeFunction.MAX: "max",
                common.NativeFunction.MOD: "fmod",
                common.NativeFunction.SIN: "dace.math.sin",
                common.NativeFunction.COS: "dace.math.cos",
                common.NativeFunction.TAN: "dace.math.tan",
                common.NativeFunction.ARCSIN: "asin",
                common.NativeFunction.ARCCOS: "acos",
                common.NativeFunction.ARCTAN: "atan",
                common.NativeFunction.SINH: "dace.math.sinh",
                common.NativeFunction.COSH: "dace.math.cosh",
                common.NativeFunction.TANH: "dace.math.tanh",
                common.NativeFunction.ARCSINH: "asinh",
                common.NativeFunction.ARCCOSH: "acosh",
                common.NativeFunction.ARCTANH: "atanh",
                common.NativeFunction.SQRT: "dace.math.sqrt",
                common.NativeFunction.POW: "dace.math.pow",
                common.NativeFunction.EXP: "dace.math.exp",
                common.NativeFunction.LOG: "dace.math.log",
                common.NativeFunction.GAMMA: "tgamma",
                common.NativeFunction.CBRT: "cbrt",
                common.NativeFunction.ISFINITE: "isfinite",
                common.NativeFunction.ISINF: "isinf",
                common.NativeFunction.ISNAN: "isnan",
                common.NativeFunction.FLOOR: "dace.math.ifloor",
                common.NativeFunction.CEIL: "ceil",
                common.NativeFunction.TRUNC: "trunc",
            }[func]
        except KeyError as error:
            raise NotImplementedError("Not implemented NativeFunction encountered.") from error

    def visit_NativeFuncCall(self, call: common.NativeFuncCall, **kwargs: Any) -> str:
        if call.func == common.NativeFunction.POW:
            return f"{self.visit(call.args[0], **kwargs)} ** {self.visit(call.args[1], **kwargs)}"
        return f"{self.visit(call.func, **kwargs)}({','.join([self.visit(a, **kwargs) for a in call.args])})"

    def visit_DataType(self, dtype: common.DataType, **kwargs: Any) -> str:
        if dtype == common.DataType.BOOL:
            return "dace.bool_"
        elif dtype == common.DataType.INT8:
            return "dace.int8"
        elif dtype == common.DataType.INT16:
            return "dace.int16"
        elif dtype == common.DataType.INT32:
            return "dace.int32"
        elif dtype == common.DataType.INT64:
            return "dace.int64"
        elif dtype == common.DataType.FLOAT32:
            return "dace.float32"
        elif dtype == common.DataType.FLOAT64:
            return "dace.float64"
        raise NotImplementedError("Not implemented DataType encountered.")

    def visit_UnaryOperator(self, op: common.UnaryOperator, **kwargs: Any) -> str:
        if op == common.UnaryOperator.NOT:
            return " not "
        elif op == common.UnaryOperator.NEG:
            return "-"
        elif op == common.UnaryOperator.POS:
            return "+"
        raise NotImplementedError("Not implemented UnaryOperator encountered.")

    Arg = as_fmt("{name}")

    Param = as_fmt("{name}")

    LocalScalar = as_fmt("{name}: {dtype}")

    def visit_Tasklet(self, node: dcir.Tasklet, **kwargs):
        return "\n".join(self.visit(node.stmts, targets=set(), **kwargs))

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs):
        mask_str = ""
        indent = ""
        if node.mask is not None:
            mask_str = f"if {self.visit(node.mask, is_target=False, **kwargs)}:"
            indent = "    "
        body_code = self.visit(node.body, **kwargs)
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    def visit_While(self, node: oir.While, **kwargs: Any):
        body = self.visit(node.body, **kwargs)
        body = [line for block in body for line in block.split("\n")]
        cond = self.visit(node.cond, is_target=False, **kwargs)
        init = "num_iter = 0"
        max_iter = 1000
        cond += f" and (num_iter < {max_iter})"
        body.append("num_iter += 1")
        indent = " " * 4
        delim = f"\n{indent}"
        code_as_str = f"{init}\nwhile {cond}:\n{indent}{delim.join(body)}"
        return code_as_str

    def visit_HorizontalMask(self, node: oir.HorizontalMask, **kwargs):
        clauses: List[str] = []
        imin = get_axis_bound_str(node.i.start, dcir.Axis.I.domain_symbol())
        if imin:
            clauses.append(f"{dcir.Axis.I.iteration_symbol()} >= {imin}")
        imax = get_axis_bound_str(node.i.end, dcir.Axis.I.domain_symbol())
        if imax:
            clauses.append(f"{dcir.Axis.I.iteration_symbol()} < {imax}")
        jmin = get_axis_bound_str(node.j.start, dcir.Axis.J.domain_symbol())
        if jmin:
            clauses.append(f"{dcir.Axis.J.iteration_symbol()} >= {jmin}")
        jmax = get_axis_bound_str(node.j.end, dcir.Axis.J.domain_symbol())
        if jmax:
            clauses.append(f"{dcir.Axis.J.iteration_symbol()} < {jmax}")
        return " and ".join(clauses)

    class RemoveCastInIndexVisitor(eve.NodeTranslator):
        def visit_FieldAccess(self, node: oir.FieldAccess):
            if node.data_index:
                return self.generic_visit(node, in_idx=True)
            else:
                return self.generic_visit(node)

        def visit_Cast(self, node: oir.Cast, in_idx=False):
            if in_idx:
                return node.expr
            else:
                return node

        def visit_Literal(self, node: oir.Cast, in_idx=False):
            if in_idx:
                return node
            else:
                return oir.Cast(dtype=node.dtype, expr=node)

    @classmethod
    def apply(cls, node: oir.HorizontalExecution, **kwargs: Any) -> str:
        preprocessed_node = cls.RemoveCastInIndexVisitor().visit(node)
        if not isinstance(node, oir.HorizontalExecution):
            raise ValueError("apply() requires oir.HorizontalExecution node")
        generated_code = super().apply(preprocessed_node)
        formatted_code = codegen.format_source("python", generated_code)
        return formatted_code


class DaCeIRBuilder(NodeTranslator):
    @dataclass
    class GlobalContext:
        library_node: StencilComputation
        block_extents: Callable[[oir.HorizontalExecution], Extent]
        arrays: Dict[str, dace.data.Data]

        def get_dcir_decls(self, access_infos):
            return {
                field: self.get_dcir_decl(field, access_info)
                for field, access_info in access_infos.items()
            }

        def get_dcir_decl(self, field, access_info):
            oir_decl: oir.FieldDecl = self.library_node.declarations[field]
            dace_array = self.arrays[field]
            return dcir.FieldDecl(
                name=field,
                dtype=oir_decl.dtype,
                strides=[str(s) for s in dace_array.strides],
                data_dims=oir_decl.data_dims,
                access_info=access_info,
                storage=dcir.StorageType.from_dace_storage(dace.StorageType.Default),
            )

    @dataclass
    class SymbolCollector:
        symbols: Dict[str, common.DataType]

    @dataclass
    class IterationContext:
        grid_subset: dcir.GridSubset
        _context_stack = list()

        @classmethod
        def init(cls, *args, **kwargs):
            assert len(cls._context_stack) == 0
            res = cls(*args, **kwargs)
            cls._context_stack.append(res)
            return res

        @classmethod
        def push_axes_extents(cls, axes_extents):
            self = cls._context_stack[-1]
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
            res = DaCeIRBuilder.IterationContext(
                grid_subset=res,
            )

            cls._context_stack.append(res)
            return res

        @classmethod
        def push_interval(cls, axis: dcir.Axis, interval: Union[dcir.DomainInterval, oir.Interval]):
            self = cls._context_stack[-1]
            res = DaCeIRBuilder.IterationContext(
                grid_subset=self.grid_subset.set_interval(axis, interval),
            )

            cls._context_stack.append(res)
            return res

        @classmethod
        def push_expansion_item(cls, item):
            self = cls._context_stack[-1]

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
            res = DaCeIRBuilder.IterationContext(grid_subset=grid_subset)

            cls._context_stack.append(res)
            return res

        @classmethod
        def push_expansion_items(cls, items):
            res = cls._context_stack[-1]
            for item in items:
                res = cls.push_expansion_item(item)
            return res

        @classmethod
        def pop(cls):
            del cls._context_stack[-1]
            return cls._context_stack[-1]

        @classmethod
        def clear(cls):
            while cls._context_stack:
                del cls._context_stack[-1]

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        expansion_specification,
        loop_order,
        k_interval,
        **kwargs,
    ):
        extent = global_ctx.block_extents(node)
        decls = [self.visit(decl) for decl in node.declarations]
        stmts = [self.visit(stmt) for stmt in node.body]

        stages_idx = next(
            idx for idx, item in enumerate(expansion_specification) if isinstance(item, Stages)
        )
        expansion_items = expansion_specification[stages_idx + 1 :]

        iteration_ctx = iteration_ctx.push_axes_extents(
            {k: v for k, v in zip(dcir.Axis.horizontal_axes(), extent)}
        )
        iteration_ctx = iteration_ctx.push_expansion_items(expansion_items)

        assert iteration_ctx.grid_subset == dcir.GridSubset.single_gridpoint()

        tasklet_read_accesses = make_read_accesses(
            node,
            global_ctx=global_ctx,
            grid_subset=iteration_ctx.grid_subset,
            k_interval=k_interval,
        )

        tasklet_write_accesses = make_write_accesses(
            node,
            global_ctx=global_ctx,
            grid_subset=iteration_ctx.grid_subset,
            k_interval=k_interval,
        )

        dcir_node = dcir.Tasklet(
            stmts=decls + stmts,
            read_accesses=tasklet_read_accesses,
            write_accesses=tasklet_write_accesses,
            name_map={
                k: k
                for k in set().union(tasklet_read_accesses.keys(), tasklet_write_accesses.keys())
            },
        )

        for item in reversed(expansion_items):
            iteration_ctx = iteration_ctx.pop()
            dcir_node = self._process_iteration_item(
                [dcir_node],
                item,
                global_ctx=global_ctx,
                iteration_ctx=iteration_ctx,
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
        expansion_specification: List[str],
        **kwargs,
    ):

        # from .utils import flatten_list

        sections_idx, stages_idx = [
            idx
            for idx, item in enumerate(expansion_specification)
            if isinstance(item, (Sections, Stages))
        ]
        expansion_items = expansion_specification[sections_idx + 1 : stages_idx]

        iteration_ctx = iteration_ctx.push_interval(
            dcir.Axis.K, node.interval
        ).push_expansion_items(expansion_items)

        dcir_nodes = self.generic_visit(
            node.horizontal_executions,
            iteration_ctx=iteration_ctx,
            global_ctx=global_ctx,
            expansion_specification=expansion_specification,
            loop_order=loop_order,
            k_interval=node.interval,
            **kwargs,
        )
        from .utils import flatten_list

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
            )
        # pop off interval
        iteration_ctx.pop()
        return dcir_nodes

    def to_dataflow(
        self,
        nodes,
        *,
        global_ctx: "DaCeIRBuilder.GlobalContext",
    ):
        from .utils import flatten_list

        nodes = flatten_list(nodes)
        if all(isinstance(n, (dcir.StateMachine, dcir.DomainMap, dcir.Tasklet)) for n in nodes):
            return nodes
        elif not all(isinstance(n, (dcir.ComputationState, dcir.DomainLoop)) for n in nodes):
            raise ValueError("Can't mix dataflow and state nodes on same level.")

        from .utils import union_node_access_infos

        read_accesses, write_accesses, field_accesses = union_node_access_infos(nodes)

        declared_symbols = set(
            n.name
            for node in nodes
            for n in node.iter_tree().if_isinstance(oir.ScalarDecl, oir.LocalScalar)
        )
        symbols = dict()

        for name in field_accesses.keys():
            for s in global_ctx.arrays[name].strides:
                for sym in dace.symbolic.symlist(s).values():
                    symbols[str(sym)] = common.DataType.INT32
        for node in nodes:
            for acc in node.iter_tree().if_isinstance(oir.ScalarAccess):
                if acc.name not in declared_symbols:
                    declared_symbols.add(acc.name)
                    symbols[acc.name] = acc.dtype

        for axis in dcir.Axis.dims_3d():
            if axis.domain_symbol() not in declared_symbols:
                declared_symbols.add(axis.domain_symbol())
                symbols[axis.domain_symbol()] = common.DataType.INT32

        return [
            dcir.StateMachine(
                label=global_ctx.library_node.label,
                field_decls=global_ctx.get_dcir_decls(field_accesses),
                symbols=symbols,
                # NestedSDFG must have same shape on input and output, matching corresponding
                # nsdfg.sdfg's array shape
                read_accesses={key: field_accesses[key] for key in read_accesses.keys()},
                write_accesses={key: field_accesses[key] for key in write_accesses.keys()},
                states=nodes,
                name_map={key: key for key in field_accesses.keys()},
            )
        ]

    def to_state(self, nodes, *, grid_subset: dcir.GridSubset):
        from .utils import flatten_list

        nodes = flatten_list(nodes)
        if all(isinstance(n, (dcir.ComputationState, dcir.DomainLoop)) for n in nodes):
            return nodes
        elif all(isinstance(n, (dcir.StateMachine, dcir.DomainMap, dcir.Tasklet)) for n in nodes):
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
        **kwargs,
    ):
        from .utils import union_node_access_infos, union_node_grid_subsets, untile_access_info_dict

        # grid_subset = union_node_grid_subsets(list(scope_nodes))
        grid_subset = iteration_ctx.grid_subset
        read_accesses, write_accesses, _ = union_node_access_infos(list(scope_nodes))
        scope_nodes = self.to_dataflow(scope_nodes, global_ctx=global_ctx)

        ranges = []
        for iteration in item.iterations:
            axis = iteration.axis
            interval = iteration_ctx.grid_subset.intervals[axis]
            grid_subset = grid_subset.set_interval(axis, interval)
            if iteration.kind == "tiling":
                read_accesses = untile_access_info_dict(read_accesses, axes=[axis])
                write_accesses = untile_access_info_dict(write_accesses, axes=[axis])
                if axis == dcir.Axis.K:
                    start, end = interval.start, interval.end
                else:
                    start, end = (
                        dcir.AxisBound.from_common(axis, oir.AxisBound.start()),
                        dcir.AxisBound.from_common(axis, oir.AxisBound.end()),
                    )
                ranges.append(
                    dcir.Range(
                        var=axis.tile_symbol(),
                        start=start,
                        end=end,
                        stride=iteration.stride,
                    )
                )
            else:

                if (
                    axis in dcir.Axis.horizontal_axes()
                    and isinstance(interval, dcir.DomainInterval)
                    and all(
                        isinstance(stmt, oir.MaskStmt)
                        and isinstance(stmt.mask, common.HorizontalMask)
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
                ):
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
                read_accesses = {
                    key: access_info.apply_iteration(dcir.GridSubset.from_interval(interval, axis))
                    for key, access_info in read_accesses.items()
                }
                write_accesses = {
                    key: access_info.apply_iteration(dcir.GridSubset.from_interval(interval, axis))
                    for key, access_info in write_accesses.items()
                }
                ranges.append(dcir.Range.from_axis_and_interval(axis, interval))

        return [
            dcir.DomainMap(
                computations=scope_nodes,
                index_ranges=ranges,
                schedule=dcir.MapSchedule.from_dace_schedule(item.schedule),
                read_accesses=read_accesses,
                write_accesses=write_accesses,
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
        **kwargs,
    ):
        from .utils import union_node_access_infos, union_node_grid_subsets, untile_access_info_dict

        grid_subset = union_node_grid_subsets(list(scope_nodes))
        read_accesses, write_accesses, _ = union_node_access_infos(list(scope_nodes))
        scope_nodes = self.to_state(scope_nodes, grid_subset=grid_subset)

        ranges = []
        axis = item.axis
        interval = iteration_ctx.grid_subset.intervals[axis]
        grid_subset = grid_subset.set_interval(axis, interval)
        if item.kind == "tiling":
            raise NotImplementedError("Tiling as a state machine not implemented.")
        else:
            assert item.kind == "contiguous"
            read_accesses = {
                key: access_info
                if key.startswith("__local_")
                else access_info.apply_iteration(dcir.GridSubset.from_interval(interval, axis))
                for key, access_info in read_accesses.items()
            }
            write_accesses = {
                key: access_info
                if key.startswith("__local_")
                else access_info.apply_iteration(dcir.GridSubset.from_interval(interval, axis))
                for key, access_info in write_accesses.items()
            }

            if isinstance(interval, oir.Interval):
                start, end = (
                    dcir.AxisBound.from_common(axis, interval.start),
                    dcir.AxisBound.from_common(axis, interval.end),
                )
            else:
                start, end = interval.idx_range
            if item.stride < 0:
                start, end = f"({end}{item.stride:+1})", f"({start}{item.stride:+1})"

            index_range = dcir.Range(
                var=axis.iteration_symbol(), start=start, end=end, stride=item.stride
            )

        ranges.append(dcir.Range.from_axis_and_interval(axis, interval))

        return [
            dcir.DomainLoop(
                axis=axis,
                loop_states=scope_nodes,
                index_range=index_range,
                read_accesses=read_accesses,
                write_accesses=write_accesses,
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
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        expansion_specification,
        **kwargs,
    ):
        from .utils import flatten_list, union_node_access_infos

        sections_idx = next(
            idx for idx, item in enumerate(expansion_specification) if isinstance(item, Sections)
        )
        expansion_items = expansion_specification[:sections_idx]
        iteration_ctx = iteration_ctx.push_expansion_items(expansion_items)

        sections = flatten_list(
            self.generic_visit(
                node.sections,
                loop_order=node.loop_order,
                global_ctx=global_ctx,
                iteration_ctx=iteration_ctx,
                expansion_specification=expansion_specification,
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
            )

        read_accesses, write_accesses, field_accesses = union_node_access_infos(computations)

        declared_symbols = set(
            n.name for n in node.iter_tree().if_isinstance(oir.ScalarDecl, oir.LocalScalar)
        )
        symbols = dict()
        for acc in node.iter_tree().if_isinstance(oir.ScalarAccess):
            if acc.name not in declared_symbols:
                declared_symbols.add(acc.name)
                symbols[acc.name] = acc.dtype
        for axis in dcir.Axis.dims_3d():
            if axis.domain_symbol() not in declared_symbols:
                declared_symbols.add(axis.domain_symbol())
                symbols[axis.domain_symbol()] = common.DataType.INT32
        for name in global_ctx.get_dcir_decls(field_accesses).keys():
            for s in global_ctx.arrays[name].strides:
                for sym in dace.symbolic.symlist(s).values():
                    symbols[str(sym)] = common.DataType.INT32

        return dcir.StateMachine(
            label=global_ctx.library_node.label,
            states=self.to_state(computations, grid_subset=iteration_ctx.grid_subset),
            field_decls=global_ctx.get_dcir_decls(field_accesses),
            read_accesses={key: field_accesses[key] for key in read_accesses.keys()},
            write_accesses={key: field_accesses[key] for key in write_accesses.keys()},
            symbols=symbols,
            name_map={key: key for key in field_accesses.keys()},
        )


class StencilComputationSDFGBuilder(NodeVisitor):
    @dataclass
    class NodeContext:
        input_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]
        output_node_and_conns: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]]

    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        state: dace.SDFGState
        field_decls: Dict[str, dcir.FieldDecl] = dataclasses.field(default_factory=dict)
        state_stack: List[dace.SDFGState] = dataclasses.field(default_factory=list)

        def add_state(self):
            old_state = self.state
            state = self.sdfg.add_state()
            for edge in self.sdfg.out_edges(old_state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(
                    state,
                    edge.dst,
                    edge.data,
                )
            self.sdfg.add_edge(old_state, state, dace.InterstateEdge())
            self.state = state
            return self

        def add_loop(self, index_range: dcir.Range):

            loop_state = self.sdfg.add_state()
            after_state = self.sdfg.add_state()
            for edge in self.sdfg.out_edges(self.state):
                self.sdfg.remove_edge(edge)
                self.sdfg.add_edge(
                    after_state,
                    edge.dst,
                    edge.data,
                )
            comparison_op = "<" if index_range.stride > 0 else ">"
            condition_expr = f"{index_range.var} {comparison_op} {index_range.end}"
            _, _, after_state = self.sdfg.add_loop(
                before_state=self.state,
                loop_state=loop_state,
                after_state=after_state,
                loop_var=index_range.var,
                initialize_expr=str(index_range.start),
                condition_expr=condition_expr,
                increment_expr=f"{index_range.var}+({index_range.stride})",
            )
            if index_range.var not in self.sdfg.symbols:
                self.sdfg.add_symbol(index_range.var, stype=dace.int32)
            self.state_stack.append(after_state)
            self.state = loop_state
            return self

        def pop_loop(self):
            self.state = self.state_stack[-1]
            del self.state_stack[-1]

    @staticmethod
    def _interval_or_idx_range(
        node: Union[dcir.DomainInterval, dcir.IndexWithExtent]
    ) -> Tuple[str, str]:
        if isinstance(node, dcir.DomainInterval):
            return str(node.start), str(node.end)
        else:
            return (
                f"{node.value}{node.extent[0]:+d}",
                f"{node.value}{node.extent[1]+1:+d}",
            )

    @staticmethod
    def _get_memlets(
        node: dcir.ComputationNode,
        *,
        with_prefix: bool = True,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ):
        if with_prefix:
            prefix = "IN_"
        else:
            prefix = ""
        in_memlets = dict()
        for field, access_info in node.read_accesses.items():
            field_decl = sdfg_ctx.field_decls[field]
            in_memlets[prefix + field] = dace.Memlet.simple(
                field,
                subset_str=make_subset_str(
                    field_decl.access_info, access_info, field_decl.data_dims
                ),
                dynamic=access_info.is_dynamic,
            )

        if with_prefix:
            prefix = "OUT_"
        else:
            prefix = ""
        out_memlets = dict()
        for field, access_info in node.write_accesses.items():
            field_decl = sdfg_ctx.field_decls[field]
            out_memlets[prefix + field] = dace.Memlet.simple(
                field,
                subset_str=make_subset_str(
                    field_decl.access_info, access_info, field_decl.data_dims
                ),
                dynamic=access_info.is_dynamic,
            )

        return in_memlets, out_memlets

    @staticmethod
    def _add_edges(
        entry_node: dace.nodes.Node,
        exit_node: dace.nodes.Node,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        in_memlets: Dict[str, dace.Memlet],
        out_memlets: Dict[str, dace.Memlet],
    ):
        for conn, memlet in in_memlets.items():
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[memlet.data], entry_node, conn, memlet
            )
        if not in_memlets and None in node_ctx.input_node_and_conns:
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[None], entry_node, None, dace.Memlet()
            )
        for conn, memlet in out_memlets.items():
            sdfg_ctx.state.add_edge(
                exit_node, conn, *node_ctx.output_node_and_conns[memlet.data], memlet
            )
        if not out_memlets and None in node_ctx.output_node_and_conns:
            sdfg_ctx.state.add_edge(
                exit_node, None, *node_ctx.output_node_and_conns[None], dace.Memlet()
            )

    @staticmethod
    def _add_origin(
        access_info: dcir.FieldAccessInfo,
        subset: Union[dace.subsets.Range, str],
        add_for_variable=False,
    ):
        if isinstance(subset, str):
            subset = dace.subsets.Range.from_string(subset)
        origin_strs = []
        for axis in access_info.axes():
            if axis in access_info.variable_offset_axes and not add_for_variable:
                origin_strs.append(str(0))
            elif add_for_variable:
                clamped_interval = access_info.clamp_full_axis(axis).grid_subset.intervals[axis]
                origin_strs.append(
                    f"-({get_axis_bound_str(clamped_interval.start, axis.domain_symbol())})"
                )
            else:
                interval = access_info.grid_subset.intervals[axis]
                if isinstance(interval, dcir.DomainInterval):
                    origin_strs.append(
                        f"-({get_axis_bound_str(interval.start, axis.domain_symbol())})"
                    )
                elif isinstance(interval, dcir.TileInterval):
                    origin_strs.append(
                        f"-({interval.axis.tile_symbol()}{interval.start_offset:+d})"
                    )
                else:
                    assert isinstance(interval, dcir.IndexWithExtent)
                    origin_strs.append(f"-({interval.value}{interval.extent[0]:+d})")

        sym = dace.symbolic.pystr_to_symbolic
        res_ranges = [
            (sym(f"({rng[0]})+({orig})"), sym(f"({rng[1]})+({orig})"), rng[2])
            for rng, orig in zip(subset.ranges, origin_strs)
        ]
        return dace.subsets.Range(res_ranges)

    def visit_Tasklet(
        self,
        node: dcir.Tasklet,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ):
        code = TaskletCodegen().visit(
            node,
            read_accesses={node.name_map[k]: v for k, v in node.read_accesses.items()},
            write_accesses={node.name_map[k]: v for k, v in node.write_accesses.items()},
            sdfg_ctx=sdfg_ctx,
        )
        access_collection = AccessCollector.apply(node)
        in_memlets = dict()
        for array_name, access_info in node.read_accesses.items():
            field = node.name_map[array_name]
            field_decl = sdfg_ctx.field_decls[array_name]
            for offset in access_collection.read_offsets()[field]:
                conn_name = get_tasklet_symbol(field, offset, is_target=False)
                subset_strs = []
                for axis in access_info.axes():
                    if axis in access_info.variable_offset_axes:
                        full_size = (
                            field_decl.access_info.clamp_full_axis(axis)
                            .grid_subset.intervals[axis]
                            .size
                        )
                        subset_strs.append(f"0:{full_size}")
                    else:
                        subset_strs.append(axis.iteration_symbol() + f"{offset[axis.to_idx()]:+d}")
                subset_str = ",".join(subset_strs)
                subset_str = str(
                    StencilComputationSDFGBuilder._add_origin(
                        access_info=field_decl.access_info, subset=subset_str
                    )
                )
                if sdfg_ctx.field_decls[field].data_dims:
                    subset_str += "," + ",".join(
                        f"0:{dim}" for dim in sdfg_ctx.field_decls[field].data_dims
                    )
                in_memlets[conn_name] = dace.Memlet.simple(
                    array_name,
                    subset_str=subset_str,
                    dynamic=access_info.is_dynamic,
                )

        out_memlets = dict()
        for array_name, access_info in node.write_accesses.items():
            field = node.name_map[array_name]
            field_decl = sdfg_ctx.field_decls[array_name]
            conn_name = get_tasklet_symbol(field, (0, 0, 0), is_target=True)
            subset_strs = []
            for axis in access_info.axes():
                if axis in access_info.variable_offset_axes:
                    full_size = (
                        field_decl.access_info.clamp_full_axis(axis)
                        .grid_subset.intervals[axis]
                        .tile_size
                    )
                    subset_strs.append(f"0:{full_size}")
                else:
                    subset_strs.append(axis.iteration_symbol())
            subset_str = ",".join(subset_strs)
            subset_str = str(
                StencilComputationSDFGBuilder._add_origin(
                    access_info=field_decl.access_info, subset=subset_str
                )
            )
            if sdfg_ctx.field_decls[field].data_dims:
                subset_str += "," + ",".join(
                    f"0:{dim}" for dim in sdfg_ctx.field_decls[field].data_dims
                )
            out_memlets[conn_name] = dace.Memlet.simple(
                array_name,
                subset_str=subset_str,
                dynamic=access_info.is_dynamic,
            )

        tasklet = dace.nodes.Tasklet(
            label=f"{sdfg_ctx.sdfg.label}_Tasklet",
            code=code,
            inputs={inp for inp in in_memlets.keys()},
            outputs={outp for outp in out_memlets.keys()},
        )

        sdfg_ctx.state.add_node(tasklet)

        StencilComputationSDFGBuilder._add_edges(
            tasklet,
            tasklet,
            sdfg_ctx=sdfg_ctx,
            in_memlets=in_memlets,
            out_memlets=out_memlets,
            node_ctx=node_ctx,
        )

    def visit_Range(self, node: dcir.Range, **kwargs):
        return node.to_ndrange()

    def visit_DomainMap(
        self,
        node: dcir.DomainMap,
        *,
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
    ):
        ndranges = [self.visit(index_range) for index_range in node.index_ranges]
        ndranges = {k: v for ndrange in ndranges for k, v in ndrange.items()}
        name = sdfg_ctx.sdfg.label + "".join(ndranges.keys()) + "_map"
        map_entry, map_exit = sdfg_ctx.state.add_map(
            name=name,
            ndrange=ndranges,
            schedule=node.schedule.to_dace_schedule(),
        )

        for scope_node in node.computations:
            input_node_and_conns: Dict[
                Optional[str], Tuple[dace.nodes.Node, Optional[str]]
            ] = dict()
            output_node_and_conns: Dict[
                Optional[str], Tuple[dace.nodes.Node, Optional[str]]
            ] = dict()
            for field in scope_node.read_accesses.keys():
                map_entry.add_in_connector("IN_" + field)
                map_entry.add_out_connector("OUT_" + field)
                input_node_and_conns[field] = (map_entry, "OUT_" + field)
            for field in scope_node.write_accesses.keys():
                map_exit.add_in_connector("IN_" + field)
                map_exit.add_out_connector("OUT_" + field)
                output_node_and_conns[field] = (map_exit, "IN_" + field)
            if not input_node_and_conns:
                input_node_and_conns[None] = (map_entry, None)
            if not output_node_and_conns:
                output_node_and_conns[None] = (map_exit, None)
            inner_node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=input_node_and_conns,
                output_node_and_conns=output_node_and_conns,
            )
            self.visit(scope_node, sdfg_ctx=sdfg_ctx, node_ctx=inner_node_ctx)

        in_memlets, out_memlets = self._get_memlets(node, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)

        self._add_edges(
            map_entry,
            map_exit,
            node_ctx=node_ctx,
            sdfg_ctx=sdfg_ctx,
            in_memlets=in_memlets,
            out_memlets=out_memlets,
        )

    def visit_DomainLoop(
        self,
        node: dcir.DomainLoop,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ):
        sdfg_ctx = sdfg_ctx.add_loop(node.index_range)
        self.visit(node.loop_states, sdfg_ctx=sdfg_ctx, **kwargs)
        sdfg_ctx.pop_loop()

    def visit_ComputationState(
        self,
        node: dcir.ComputationState,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ):

        sdfg_ctx = sdfg_ctx.add_state()
        read_acc_and_conn: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = dict()
        write_acc_and_conn: Dict[Optional[str], Tuple[dace.nodes.Node, Optional[str]]] = dict()
        for computation in node.computations:
            assert isinstance(computation, dcir.ComputationNode)
            for field in computation.read_accesses.keys():
                if field not in read_acc_and_conn:
                    read_acc_and_conn[field] = (sdfg_ctx.state.add_access(field), None)
            for field in computation.write_accesses.keys():
                if field not in write_acc_and_conn:
                    write_acc_and_conn[field] = (sdfg_ctx.state.add_access(field), None)
            node_ctx = StencilComputationSDFGBuilder.NodeContext(
                input_node_and_conns=read_acc_and_conn,
                output_node_and_conns=write_acc_and_conn,
            )
            self.visit(computation, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx, **kwargs)

    def visit_StateMachine(
        self,
        node: dcir.StateMachine,
        *,
        sdfg_ctx: Optional["StencilComputationSDFGBuilder.SDFGContext"] = None,
        node_ctx: Optional["StencilComputationSDFGBuilder.NodeContext"] = None,
    ):

        sdfg = dace.SDFG(node.label)
        state = sdfg.add_state()
        symbol_mapping = {}
        for axis in [dcir.Axis.I, dcir.Axis.J, dcir.Axis.K]:
            sdfg.add_symbol(axis.domain_symbol(), stype=dace.int32)
            symbol_mapping[axis.domain_symbol()] = dace.symbol(
                axis.domain_symbol(), dtype=dace.int32
            )
        if sdfg_ctx is not None and node_ctx is not None:
            nsdfg = sdfg_ctx.state.add_nested_sdfg(
                sdfg=sdfg,
                parent=None,
                inputs=set(node.name_map[k] for k in node.read_accesses.keys()),
                outputs=set(node.name_map[k] for k in node.write_accesses.keys()),
                symbol_mapping=symbol_mapping,
            )
            in_memlets, out_memlets = StencilComputationSDFGBuilder._get_memlets(
                node, with_prefix=False, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
            )
            StencilComputationSDFGBuilder._add_edges(
                nsdfg,
                nsdfg,
                sdfg_ctx=sdfg_ctx,
                node_ctx=node_ctx,
                in_memlets={node.name_map[k]: v for k, v in in_memlets.items()},
                out_memlets={node.name_map[k]: v for k, v in out_memlets.items()},
            )
        else:
            nsdfg = dace.nodes.NestedSDFG(
                label=sdfg.label,
                sdfg=sdfg,
                inputs=set(node.read_accesses.keys()),
                outputs=set(node.write_accesses.keys()),
                symbol_mapping=symbol_mapping,
            )

        inner_sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg,
            state=state,
            field_decls=node.field_decls,
        )

        for name, decl in node.field_decls.items():
            inner_sdfg_ctx.sdfg.add_array(
                name,
                shape=decl.shape
                if not name.startswith("__local_")
                else decl.access_info.overapproximated_shape,
                strides=[dace.symbolic.pystr_to_symbolic(s) for s in decl.strides],
                dtype=np.dtype(common.data_type_to_typestr(decl.dtype)).type,
                storage=decl.storage.to_dace_storage(),
                transient=(name not in set(node.name_map.values())),
            )
        for symbol, dtype in node.symbols.items():
            if symbol not in inner_sdfg_ctx.sdfg.symbols:
                inner_sdfg_ctx.sdfg.add_symbol(
                    symbol,
                    stype=dace.typeclass(np.dtype(common.data_type_to_typestr(dtype)).name),
                )
            nsdfg.symbol_mapping[symbol] = dace.symbol(
                symbol,
                dtype=dace.typeclass(np.dtype(common.data_type_to_typestr(dtype)).name),
            )

        for computation_state in node.states:
            self.visit(computation_state, sdfg_ctx=inner_sdfg_ctx)
        for sym in nsdfg.sdfg.free_symbols:
            if sym not in nsdfg.sdfg.symbols:
                nsdfg.sdfg.add_symbol(sym, stype=dace.int32)
            nsdfg.symbol_mapping.setdefault(str(sym), dace.symbol(sym, dtype=dace.int32))

        return nsdfg

    def visit_CopyState(
        self,
        node: dcir.CopyState,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        **kwargs,
    ):
        sdfg_ctx.add_state()
        read_nodes = dict()
        write_nodes = dict()
        intermediate_nodes = dict()

        for src_name, dst_name in node.name_map.items():
            if src_name not in node.read_accesses:
                continue
            src_decl = sdfg_ctx.field_decls[src_name]
            src_subset = make_subset_str(
                src_decl.access_info, node.read_accesses[src_name], src_decl.data_dims
            )
            dst_decl = sdfg_ctx.field_decls[dst_name]
            dst_subset = make_subset_str(
                dst_decl.access_info, node.write_accesses[dst_name], dst_decl.data_dims
            )

            if src_name not in read_nodes:
                read_nodes[src_name] = sdfg_ctx.state.add_access(src_name)
            if dst_name not in write_nodes:
                write_nodes[dst_name] = sdfg_ctx.state.add_access(dst_name)
            if src_name == dst_name:
                tmp_name = sdfg_ctx.sdfg.temp_data_name()
                intermediate_access = sdfg_ctx.state.add_access(tmp_name)
                stride = 1
                strides = []
                for s in reversed(node.read_accesses[src_name].overapproximated_shape):
                    strides = [stride, *strides]
                    stride = f"({stride}) * ({s})"

                field_decl = sdfg_ctx.field_decls[src_name]
                sdfg_ctx.sdfg.add_array(
                    name=tmp_name,
                    shape=node.read_accesses[src_name].overapproximated_shape,
                    strides=[dace.symbolic.pystr_to_symbolic(s) for s in strides],
                    dtype=np.dtype(common.data_type_to_typestr(field_decl.dtype)).type,
                    transient=True,
                )
                tmp_subset = make_subset_str(
                    node.read_accesses[src_name],
                    node.read_accesses[src_name],
                    src_decl.data_dims,
                )
                sdfg_ctx.state.add_edge(
                    read_nodes[src_name],
                    None,
                    intermediate_access,
                    None,
                    dace.Memlet(data=src_name, subset=src_subset, other_subset=tmp_subset),
                )
                sdfg_ctx.state.add_edge(
                    intermediate_access,
                    None,
                    write_nodes[dst_name],
                    None,
                    dace.Memlet(data=tmp_name, subset=tmp_subset, other_subset=dst_subset),
                )
            else:
                sdfg_ctx.state.add_edge(
                    read_nodes[src_name],
                    None,
                    write_nodes[dst_name],
                    None,
                    dace.Memlet(data=src_name, subset=src_subset, other_subset=dst_subset),
                )


@dace.library.register_expansion(StencilComputation, "default")
class StencilComputationExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def _solve_for_domain(field_decls: Dict[str, dcir.FieldDecl], outer_subsets):
        equations = []
        symbols = set()

        # Collect equations and symbols from arguments and shapes
        for field, decl in field_decls.items():
            inner_shape = [dace.symbolic.pystr_to_symbolic(s) for s in decl.shape]
            outer_shape = [
                dace.symbolic.pystr_to_symbolic(s) for s in outer_subsets[field].bounding_box_size()
            ]

            for inner_dim, outer_dim in zip(inner_shape, outer_shape):
                repldict = {}
                for sym in dace.symbolic.symlist(inner_dim).values():
                    newsym = dace.symbolic.symbol("__SOLVE_" + str(sym))
                    symbols.add(newsym)
                    repldict[sym] = newsym

                # Replace symbols with __SOLVE_ symbols so as to allow
                # the same symbol in the called SDFG
                if repldict:
                    inner_dim = inner_dim.subs(repldict)

                equations.append(inner_dim - outer_dim)
        if len(symbols) == 0:
            return {}

        # Solve for all at once
        results = sympy.solve(equations, *symbols, dict=True)
        result = results[0]
        result = {str(k)[len("__SOLVE_") :]: v for k, v in result.items()}
        return result

    @staticmethod
    def expansion(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:
        split_horizontal_exeuctions_regions(node)
        start, end = (
            node.oir_node.sections[0].interval.start,
            node.oir_node.sections[0].interval.end,
        )
        for section in node.oir_node.sections:
            start = min(start, section.interval.start)
            end = max(end, section.interval.end)

        overall_interval = dcir.DomainInterval(
            start=dcir.AxisBound(axis=dcir.Axis.K, level=start.level, offset=start.offset),
            end=dcir.AxisBound(axis=dcir.Axis.K, level=end.level, offset=end.offset),
        )
        overall_extent = Extent.zeros(ndims=3)
        for he in node.oir_node.iter_tree().if_isinstance(oir.HorizontalExecution):
            overall_extent = overall_extent.union(node.get_extents(he))

        parent_arrays = dict()
        for edge in parent_state.in_edges(node):
            if edge.dst_conn is not None:
                parent_arrays[edge.dst_conn[len("__in_") :]] = parent_sdfg.arrays[edge.data.data]
        for edge in parent_state.out_edges(node):
            if edge.src_conn is not None:
                parent_arrays[edge.src_conn[len("__out_") :]] = parent_sdfg.arrays[edge.data.data]

        daceir_builder_global_ctx = DaCeIRBuilder.GlobalContext(
            library_node=node, block_extents=node.get_extents, arrays=parent_arrays
        )

        iteration_ctx = DaCeIRBuilder.IterationContext.init(
            grid_subset=dcir.GridSubset.from_gt4py_extent(overall_extent).set_interval(
                axis=dcir.Axis.K, interval=overall_interval
            )
        )
        try:
            daceir: dcir.StateMachine = DaCeIRBuilder().visit(
                node.oir_node,
                global_ctx=daceir_builder_global_ctx,
                iteration_ctx=iteration_ctx,
                expansion_specification=list(node.expansion_specification),
            )
        finally:
            iteration_ctx.clear()

        from .daceir_passes import MakeLocalCaches

        cached_loops = [
            item
            for item in node.expansion_specification
            if isinstance(item, Loop) and len(item.localcache_fields) > 0
        ]
        if len(cached_loops) > 0:
            localcache_infos = dict()
            for item in cached_loops:
                localcache_infos[item.axis] = SimpleNamespace(
                    fields=item.localcache_fields,
                    storage=item.storage,
                )
            daceir = MakeLocalCaches().visit(daceir, localcache_infos=localcache_infos)

        nsdfg = StencilComputationSDFGBuilder().visit(daceir)

        for in_edge in parent_state.in_edges(node):
            assert in_edge.dst_conn.startswith("__in_")
            in_edge.dst_conn = in_edge.dst_conn[len("__in_") :]
        for out_edge in parent_state.out_edges(node):
            assert out_edge.src_conn.startswith("__out_")
            out_edge.src_conn = out_edge.src_conn[len("__out_") :]

        subsets = dict()
        for edge in parent_state.in_edges(node):
            subsets[edge.dst_conn] = edge.data.subset
        for edge in parent_state.out_edges(node):
            subsets[edge.src_conn] = dace.subsets.union(
                edge.data.subset, subsets.get(edge.src_conn, edge.data.subset)
            )
        for edge in parent_state.in_edges(node):
            edge.data.subset = copy.deepcopy(subsets[edge.dst_conn])
        for edge in parent_state.out_edges(node):
            edge.data.subset = copy.deepcopy(subsets[edge.src_conn])
        symbol_mapping = StencilComputationExpansion._solve_for_domain(
            {
                name: decl
                for name, decl in daceir.field_decls.items()
                if name in daceir.read_accesses or name in daceir.write_accesses
            },
            subsets,
        )
        if "__K" in nsdfg.sdfg.free_symbols and "__K" not in symbol_mapping:
            symbol_mapping["__K"] = 0
        nsdfg.symbol_mapping.update({**symbol_mapping, **node.symbol_mapping})

        return nsdfg
