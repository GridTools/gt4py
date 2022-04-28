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
import itertools
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Set, Tuple, Union

import dace
import dace.data
import dace.library
import dace.subsets
import numpy as np
import sympy

import eve
import eve.utils
import gtc.common as common
import gtc.oir as oir
from eve import NodeTranslator, NodeVisitor, SymbolRef, codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.iterators import iter_tree
from gt4py import definitions as gt_def
from gt4py.definitions import Extent
from gtc import daceir as dcir
from gtc.dace.expansion_specification import Loop, Map, Sections, Stages
from gtc.dace.nodes import StencilComputation
from gtc.dace.utils import get_axis_bound_str, get_tasklet_symbol

from .utils import compute_dcir_access_infos, make_subset_str


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


def get_tasklet_inout_memlets(node: oir.HorizontalExecution, *, get_outputs, global_ctx, **kwargs):

    access_infos = compute_dcir_access_infos(
        node,
        block_extents=global_ctx.block_extents,
        oir_decls=global_ctx.library_node.declarations,
        collect_read=not get_outputs,
        collect_write=get_outputs,
        **kwargs,
    )

    def make_access_iter():
        if get_outputs:
            return eve.utils.xiter(
                itertools.chain(
                    *node.iter_tree().if_isinstance(oir.AssignStmt).getattr("left").map(iter_tree)
                )
            )
        else:
            return eve.utils.xiter(
                itertools.chain(
                    *node.iter_tree().if_isinstance(oir.AssignStmt).getattr("right").map(iter_tree),
                    *node.iter_tree().if_isinstance(oir.While).getattr("cond").map(iter_tree),
                    *node.iter_tree().if_isinstance(oir.MaskStmt).getattr("mask").map(iter_tree),
                )
            )

    res = list()
    for name, info in access_infos.items():
        if info.variable_offset_axes:
            tasklet_symbol = get_tasklet_symbol(name, None, is_target=get_outputs)
            res.append(
                dcir.Memlet(
                    field=name,
                    connector=tasklet_symbol,
                    access_info=info,
                    is_read=not get_outputs,
                    is_write=get_outputs,
                )
            )
        else:
            for offset, tasklet_symbol in (
                make_access_iter()
                .if_isinstance(oir.FieldAccess)
                .filter(lambda x: x.name == name)
                .getattr("offset")
                .map(lambda off: (off, get_tasklet_symbol(name, off, is_target=get_outputs)))
                .unique(key=lambda x: x[1])
            ):
                offset_dict = offset.to_dict()
                intervals = {
                    axis: dcir.IndexWithExtent.from_axis(
                        axis, extent=(offset_dict[axis.lower()], offset_dict[axis.lower()])
                    )
                    for axis in info.axes()
                }
                res.append(
                    dcir.Memlet(
                        field=name,
                        connector=tasklet_symbol,
                        access_info=dcir.FieldAccessInfo(
                            global_grid_subset=info.global_grid_subset,
                            grid_subset=dcir.GridSubset(intervals=intervals),
                            dynamic_access=info.dynamic_access,
                        ),
                        is_read=not get_outputs,
                        is_write=get_outputs,
                    )
                )
    return res


class TaskletCodegen(codegen.TemplatedGenerator):

    ScalarAccess = as_fmt("{name}")

    def _visit_offset(
        self,
        node: Union[oir.VariableKOffset, common.CartesianOffset],
        *,
        access_info: dcir.FieldAccessInfo,
        decl: dcir.FieldDecl,
        **kwargs,
    ):
        int_sizes: List[Optional[int]] = []
        for i, axis in enumerate(access_info.axes()):
            memlet_shape = access_info.shape
            if (
                str(memlet_shape[i]).isnumeric()
                and axis not in decl.access_info.variable_offset_axes
            ):
                int_sizes.append(int(memlet_shape[i]))
            else:
                int_sizes.append(None)
        str_offset = [
            self.visit(off, **kwargs) for off in (node.to_dict()["i"], node.to_dict()["j"], node.k)
        ]
        str_offset = [
            f"{axis.iteration_symbol()} + {str_offset[axis.to_idx()]}"
            for i, axis in enumerate(access_info.axes())
        ]

        res: dace.subsets.Range = StencilComputationSDFGBuilder._add_origin(
            decl.access_info, ",".join(str_offset), add_for_variable=True
        )
        return str(dace.subsets.Range([r for i, r in enumerate(res.ranges) if int_sizes[i] != 1]))

    def visit_CartesianOffset(self, node: common.CartesianOffset, **kwargs):
        return self._visit_offset(node, **kwargs)

    def visit_VariableKOffset(self, node: common.CartesianOffset, **kwargs):
        return self._visit_offset(node, **kwargs)

    def visit_IndexAccess(self, node: dcir.IndexAccess, *, is_target, sdfg_ctx, **kwargs):

        memlets = kwargs["write_memlets" if is_target else "read_memlets"]
        memlet = next(mem for mem in memlets if mem.connector == node.name)

        index_strs = []
        if node.offset is not None:
            index_strs.append(
                self.visit(
                    node.offset,
                    decl=sdfg_ctx.field_decls[memlet.field],
                    access_info=memlet.access_info,
                    **kwargs,
                )
            )
        index_strs.extend(self.visit(idx, sdfg_ctx=sdfg_ctx, **kwargs) for idx in node.data_index)
        return f"{node.name}[{','.join(index_strs)}]"

    def visit_AssignStmt(self, node: dcir.AssignStmt, **kwargs):
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
        # TODO: Unroll integer POW
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
        return "\n".join(self.visit(node.stmts, **kwargs))

    def _visit_conditional(self, node: Union[oir.MaskStmt, oir.HorizontalRestriction], **kwargs):
        mask_str = ""
        indent = ""
        if node.mask is not None:
            mask_str = f"if {self.visit(node.mask, **kwargs)}:"
            indent = " " * 4
        body_code = [
            line for block in self.visit(node.body, **kwargs) for line in block.split("\n")
        ]
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs):
        return self._visit_conditional(node, **kwargs)

    def visit_HorizontalRestriction(self, node: oir.HorizontalRestriction, **kwargs):
        return self._visit_conditional(node, **kwargs)

    def visit_While(self, node: oir.While, **kwargs):
        cond = self.visit(node.cond, is_target=False, **kwargs)
        while_str = f"while {cond}:"
        indent = " " * 4
        body_code = [
            line for block in self.visit(node.body, **kwargs) for line in block.split("\n")
        ]
        body_code = [indent + b for b in body_code]
        return "\n".join([while_str] + body_code)

    def visit_HorizontalMask(self, node: common.HorizontalMask, **kwargs):
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

    @classmethod
    def apply(cls, node: oir.HorizontalExecution, **kwargs: Any) -> str:
        if not isinstance(node, oir.HorizontalExecution):
            raise ValueError("apply() requires oir.HorizontalExecution node")
        generated_code = super().apply(node)
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
        _context_stack: ClassVar[List["DaCeIRBuilder.IterationContext"]] = list()

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

    def visit_ScalarAccess(self, node: oir.ScalarAccess, **kwargs: Any) -> dcir.ScalarAccess:
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
        expansion_specification,
        loop_order,
        k_interval,
        **kwargs,
    ):
        # skip type checking due to https://github.com/python/mypy/issues/5485
        extent = global_ctx.block_extents(node)  # type: ignore
        decls = [self.visit(decl, **kwargs) for decl in node.declarations]
        targets: Set[str] = set()
        stmts = [self.visit(stmt, targets=targets, **kwargs) for stmt in node.body]

        stages_idx = next(
            idx for idx, item in enumerate(expansion_specification) if isinstance(item, Stages)
        )
        expansion_items = expansion_specification[stages_idx + 1 :]

        iteration_ctx = iteration_ctx.push_axes_extents(
            {k: v for k, v in zip(dcir.Axis.horizontal_axes(), extent)}
        )
        iteration_ctx = iteration_ctx.push_expansion_items(expansion_items)

        assert iteration_ctx.grid_subset == dcir.GridSubset.single_gridpoint()

        read_memlets = get_tasklet_inout_memlets(
            node,
            get_outputs=False,
            global_ctx=global_ctx,
            grid_subset=iteration_ctx.grid_subset,
            k_interval=k_interval,
        )

        write_memlets = get_tasklet_inout_memlets(
            node,
            get_outputs=True,
            global_ctx=global_ctx,
            grid_subset=iteration_ctx.grid_subset,
            k_interval=k_interval,
        )

        dcir_node = dcir.Tasklet(
            stmts=decls + stmts,
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

        from .utils import union_inout_memlets

        read_memlets, write_memlets, field_memlets = union_inout_memlets(nodes)

        declared_symbols = set(
            n.name
            for node in nodes
            for n in node.iter_tree().if_isinstance(oir.ScalarDecl, oir.LocalScalar)
        )
        symbols = dict()

        for memlet in field_memlets:
            for s in global_ctx.arrays[memlet.field].strides:
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

        for acc in iter_tree(nodes).if_isinstance(dcir.ScalarAccess):
            if (
                acc.name in global_ctx.library_node.declarations
                and acc.name not in declared_symbols
            ):
                declared_symbols.add(acc.name)
                symbols[acc.name] = global_ctx.library_node.declarations[acc.name].dtype
        field_decls = global_ctx.get_dcir_decls(
            {memlet.field: memlet.access_info for memlet in field_memlets}
        )
        read_fields = set(memlet.field for memlet in read_memlets)
        write_fields = set(memlet.field for memlet in write_memlets)
        read_memlets = [
            memlet.remove_write() for memlet in field_memlets if memlet.field in read_fields
        ]
        write_memlets = [
            memlet.remove_read() for memlet in field_memlets if memlet.field in write_fields
        ]
        return [
            dcir.StateMachine(
                label=global_ctx.library_node.label,
                field_decls=field_decls,
                symbols=symbols,
                # NestedSDFG must have same shape on input and output, matching corresponding
                # nsdfg.sdfg's array shape
                read_memlets=read_memlets,
                write_memlets=write_memlets,
                states=nodes,
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
        from .utils import union_inout_memlets, untile_memlets

        grid_subset = iteration_ctx.grid_subset
        read_memlets, write_memlets, _ = union_inout_memlets(list(scope_nodes))
        scope_nodes = self.to_dataflow(scope_nodes, global_ctx=global_ctx)

        ranges = []
        for iteration in item.iterations:
            axis = iteration.axis
            interval = iteration_ctx.grid_subset.intervals[axis]
            grid_subset = grid_subset.set_interval(axis, interval)
            if iteration.kind == "tiling":
                read_memlets = untile_memlets(read_memlets, axes=[axis])
                write_memlets = untile_memlets(write_memlets, axes=[axis])
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
        **kwargs,
    ):
        from .utils import union_inout_memlets, union_node_grid_subsets

        grid_subset = union_node_grid_subsets(list(scope_nodes))
        read_memlets, write_memlets, _ = union_inout_memlets(list(scope_nodes))
        scope_nodes = self.to_state(scope_nodes, grid_subset=grid_subset)

        ranges = []
        axis = item.axis
        interval = iteration_ctx.grid_subset.intervals[axis]
        grid_subset = grid_subset.set_interval(axis, interval)
        if item.kind == "tiling":
            raise NotImplementedError("Tiling as a state machine not implemented.")
        else:
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
        iteration_ctx: "DaCeIRBuilder.IterationContext",
        expansion_specification,
        **kwargs,
    ):
        from .utils import flatten_list, union_inout_memlets

        var_offset_fields = set(
            acc.name
            for acc in node.iter_tree().if_isinstance(oir.FieldAccess)
            if isinstance(acc.offset, oir.VariableKOffset)
        )
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
            )

        read_memlets, write_memlets, field_memlets = union_inout_memlets(computations)

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
        field_decls = global_ctx.get_dcir_decls(
            {memlet.field: memlet.access_info for memlet in field_memlets}
        )
        for name in field_decls.keys():
            for s in global_ctx.arrays[name].strides:
                for sym in dace.symbolic.symlist(s).values():
                    symbols[str(sym)] = common.DataType.INT32
        read_fields = set(memlet.field for memlet in read_memlets)
        write_fields = set(memlet.field for memlet in write_memlets)
        return dcir.StateMachine(
            label=global_ctx.library_node.label,
            states=self.to_state(computations, grid_subset=iteration_ctx.grid_subset),
            field_decls=field_decls,
            read_memlets=[memlet for memlet in field_memlets if memlet.field in read_fields],
            write_memlets=[memlet for memlet in field_memlets if memlet.field in write_fields],
            symbols=symbols,
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
        res_ranges = []
        for i, axis in enumerate(access_info.axes()):
            rng = subset.ranges[i]
            orig = origin_strs[axis.to_idx()]
            res_ranges.append((sym(f"({rng[0]})+({orig})"), sym(f"({rng[1]})+({orig})"), rng[2]))
        return dace.subsets.Range(res_ranges)

    def visit_Memlet(
        self,
        node: dcir.Memlet,
        *,
        scope_node: dcir.ComputationNode,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
        connector_prefix="",
    ):
        field_decl = sdfg_ctx.field_decls[node.field]
        memlet = dace.Memlet.simple(
            node.field,
            subset_str=make_subset_str(
                field_decl.access_info, node.access_info, field_decl.data_dims
            ),
            dynamic=field_decl.is_dynamic,
        )
        if node.is_read:
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[memlet.data],
                scope_node,
                connector_prefix + node.connector,
                memlet,
            )
        if node.is_write:
            sdfg_ctx.state.add_edge(
                scope_node,
                connector_prefix + node.connector,
                *node_ctx.output_node_and_conns[memlet.data],
                memlet,
            )

    @classmethod
    def _add_empty_edges(
        cls,
        entry_node: dace.nodes.Node,
        exit_node: dace.nodes.Node,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ):

        if not sdfg_ctx.state.in_degree(entry_node) and None in node_ctx.input_node_and_conns:
            sdfg_ctx.state.add_edge(
                *node_ctx.input_node_and_conns[None], entry_node, None, dace.Memlet()
            )
        if not sdfg_ctx.state.out_degree(exit_node) and None in node_ctx.output_node_and_conns:
            sdfg_ctx.state.add_edge(
                exit_node, None, *node_ctx.output_node_and_conns[None], dace.Memlet()
            )

    def visit_Tasklet(
        self,
        node: dcir.Tasklet,
        *,
        sdfg_ctx: "StencilComputationSDFGBuilder.SDFGContext",
        node_ctx: "StencilComputationSDFGBuilder.NodeContext",
    ):
        code = TaskletCodegen().visit(
            node,
            read_memlets=node.read_memlets,
            write_memlets=node.write_memlets,
            sdfg_ctx=sdfg_ctx,
        )

        tasklet = dace.nodes.Tasklet(
            label=f"{sdfg_ctx.sdfg.label}_Tasklet",
            code=code,
            inputs=set(memlet.connector for memlet in node.read_memlets),
            outputs=set(memlet.connector for memlet in node.write_memlets),
        )
        sdfg_ctx.state.add_node(tasklet)

        self.visit(node.read_memlets, scope_node=tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)
        self.visit(node.write_memlets, scope_node=tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)
        StencilComputationSDFGBuilder._add_empty_edges(
            tasklet, tasklet, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
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

        ndranges = {
            k: v for index_range in node.index_ranges for k, v in self.visit(index_range).items()
        }
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
            for field in set(memlet.field for memlet in scope_node.read_memlets):
                map_entry.add_in_connector("IN_" + field)
                map_entry.add_out_connector("OUT_" + field)
                input_node_and_conns[field] = (map_entry, "OUT_" + field)
            for field in set(memlet.field for memlet in scope_node.write_memlets):
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

        self.visit(
            node.read_memlets,
            scope_node=map_entry,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            connector_prefix="IN_",
        )
        self.visit(
            node.write_memlets,
            scope_node=map_exit,
            sdfg_ctx=sdfg_ctx,
            node_ctx=node_ctx,
            connector_prefix="OUT_",
        )
        StencilComputationSDFGBuilder._add_empty_edges(
            map_entry, map_exit, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
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
            for memlet in computation.read_memlets:
                if memlet.field not in read_acc_and_conn:
                    read_acc_and_conn[memlet.field] = (
                        sdfg_ctx.state.add_access(memlet.field),
                        None,
                    )
            for memlet in computation.write_memlets:
                if memlet.field not in write_acc_and_conn:
                    write_acc_and_conn[memlet.field] = (
                        sdfg_ctx.state.add_access(memlet.field),
                        None,
                    )
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
        for axis in dcir.Axis.dims_3d():
            sdfg.add_symbol(axis.domain_symbol(), stype=dace.int32)
            symbol_mapping[axis.domain_symbol()] = dace.symbol(
                axis.domain_symbol(), dtype=dace.int32
            )
        if sdfg_ctx is not None and node_ctx is not None:
            nsdfg = sdfg_ctx.state.add_nested_sdfg(
                sdfg=sdfg,
                parent=None,
                inputs=node.input_connectors,
                outputs=node.output_connectors,
                symbol_mapping=symbol_mapping,
            )
            self.visit(node.read_memlets, scope_node=nsdfg, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)
            self.visit(node.write_memlets, scope_node=nsdfg, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx)
            StencilComputationSDFGBuilder._add_empty_edges(
                nsdfg, nsdfg, sdfg_ctx=sdfg_ctx, node_ctx=node_ctx
            )
        else:
            nsdfg = dace.nodes.NestedSDFG(
                label=sdfg.label,
                sdfg=sdfg,
                inputs=set(memlet.connector for memlet in node.read_memlets),
                outputs=set(memlet.connector for memlet in node.write_memlets),
                symbol_mapping=symbol_mapping,
            )

        inner_sdfg_ctx = StencilComputationSDFGBuilder.SDFGContext(
            sdfg=sdfg,
            state=state,
            field_decls=node.field_decls,
        )

        for name, decl in node.field_decls.items():
            non_transients = set(
                memlet.connector for memlet in node.read_memlets + node.write_memlets
            )
            assert len(decl.shape) == len(decl.strides)
            inner_sdfg_ctx.sdfg.add_array(
                name,
                shape=decl.shape,
                strides=[dace.symbolic.pystr_to_symbolic(s) for s in decl.strides],
                dtype=np.dtype(common.data_type_to_typestr(decl.dtype)).type,
                storage=decl.storage.to_dace_storage(),
                transient=name not in non_transients,
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
    def fix_context(nsdfg, node: "StencilComputation", parent_state, daceir):

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
                if name
                in set(memlet.field for memlet in daceir.read_memlets + daceir.write_memlets)
            },
            subsets,
        )
        if "__K" in nsdfg.sdfg.free_symbols and "__K" not in symbol_mapping:
            symbol_mapping["__K"] = 0
        nsdfg.symbol_mapping.update({**symbol_mapping, **node.symbol_mapping})

    @staticmethod
    def expansion(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:
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
        overall_extent = Extent.zeros(2)
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

        nsdfg = StencilComputationSDFGBuilder().visit(daceir)
        StencilComputationExpansion.fix_context(nsdfg, node, parent_state, daceir)

        return nsdfg
