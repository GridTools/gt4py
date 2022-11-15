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
from typing import Any, ChainMap, List, Optional, Union

import dace
import dace.data
import dace.library
import dace.subsets

import eve
import gtc.common as common
from eve.codegen import FormatTemplate as as_fmt
from gtc import daceir as dcir
from gtc.dace.symbol_utils import get_axis_bound_str
from gtc.dace.utils import make_dace_subset


class TaskletCodegen(eve.codegen.TemplatedGenerator, eve.VisitorWithSymbolTableTrait):
    ScalarAccess = as_fmt("{name}")

    def _visit_offset(
        self,
        node: Union[dcir.VariableKOffset, common.CartesianOffset],
        *,
        access_info: dcir.FieldAccessInfo,
        decl: dcir.FieldDecl,
        **kwargs,
    ) -> str:
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
        sym_offsets = [
            dace.symbolic.pystr_to_symbolic(self.visit(off, **kwargs))
            for off in (node.to_dict()["i"], node.to_dict()["j"], node.k)
        ]
        for axis in access_info.variable_offset_axes:
            access_info = access_info.restricted_to_index(axis)
        context_info = copy.deepcopy(access_info)
        context_info.variable_offset_axes = []
        ranges = make_dace_subset(
            access_info, context_info, data_dims=()  # data_index added in visit_IndexAccess
        )
        ranges.offset(sym_offsets, negative=False)
        res = dace.subsets.Range([r for i, r in enumerate(ranges.ranges) if int_sizes[i] != 1])
        return str(res)

    def visit_CartesianOffset(self, node: common.CartesianOffset, **kwargs):
        return self._visit_offset(node, **kwargs)

    def visit_VariableKOffset(self, node: common.CartesianOffset, **kwargs):
        return self._visit_offset(node, **kwargs)

    def visit_IndexAccess(
        self,
        node: dcir.IndexAccess,
        *,
        is_target,
        sdfg_ctx,
        symtable: ChainMap[eve.SymbolRef, dcir.Decl],
        **kwargs,
    ):

        if is_target:
            memlets = kwargs["write_memlets"]
        else:
            # if this node is not a target, it will still use the symbol of the write memlet if the
            # field was previously written in the same memlet.
            memlets = kwargs["read_memlets"] + kwargs["write_memlets"]
        memlet = next(mem for mem in memlets if mem.connector == node.name)

        index_strs = []
        if node.offset is not None:
            index_strs.append(
                self.visit(
                    node.offset,
                    decl=symtable[memlet.field],
                    access_info=memlet.access_info,
                    symtable=symtable,
                    in_idx=True,
                    **kwargs,
                )
            )
        index_strs.extend(
            self.visit(idx, sdfg_ctx=sdfg_ctx, symtable=symtable, in_idx=True, **kwargs)
            for idx in node.data_index
        )
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

    def visit_Literal(self, literal: dcir.Literal, *, in_idx=False, **kwargs):
        value = self.visit(literal.value, in_idx=in_idx, **kwargs)
        if in_idx:
            return str(value)
        else:
            return "{dtype}({value})".format(
                dtype=self.visit(literal.dtype, in_idx=in_idx, **kwargs), value=value
            )

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

    LocalScalarDecl = as_fmt("{name}: {dtype}")

    def visit_Tasklet(self, node: dcir.Tasklet, **kwargs):
        return "\n".join(self.visit(node.decls, **kwargs) + self.visit(node.stmts, **kwargs))

    def _visit_conditional(
        self,
        cond: Optional[Union[dcir.Expr, common.HorizontalMask]],
        body: List[dcir.Stmt],
        keyword,
        **kwargs,
    ):
        mask_str = ""
        indent = ""
        if cond is not None and (cond_str := self.visit(cond, is_target=False, **kwargs)):
            mask_str = f"{keyword} {cond_str}:"
            indent = " " * 4
        body_code = [line for block in self.visit(body, **kwargs) for line in block.split("\n")]
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    def visit_MaskStmt(self, node: dcir.MaskStmt, **kwargs):
        return self._visit_conditional(cond=node.mask, body=node.body, keyword="if", **kwargs)

    def visit_HorizontalRestriction(self, node: dcir.HorizontalRestriction, **kwargs):
        return self._visit_conditional(cond=node.mask, body=node.body, keyword="if", **kwargs)

    def visit_While(self, node: dcir.While, **kwargs):
        return self._visit_conditional(cond=node.cond, body=node.body, keyword="while", **kwargs)

    def visit_HorizontalMask(self, node: common.HorizontalMask, **kwargs):
        clauses: List[str] = []

        for axis, interval in zip(dcir.Axis.dims_horizontal(), node.intervals):
            it_sym, dom_sym = axis.iteration_symbol(), axis.domain_symbol()

            min_val = get_axis_bound_str(interval.start, dom_sym)
            max_val = get_axis_bound_str(interval.end, dom_sym)
            if (
                min_val
                and max_val
                and interval.start is not None
                and interval.end is not None
                and interval.start.level == interval.end.level
                and interval.start.offset + 1 == interval.end.offset
            ):
                clauses.append(f"{it_sym} == {min_val}")
            else:
                if min_val:
                    clauses.append(f"{it_sym} >= {min_val}")
                if max_val:
                    clauses.append(f"{it_sym} < {max_val}")

        return " and ".join(clauses)

    @classmethod
    def apply_codegen(cls, node: dcir.Tasklet, **kwargs: Any) -> str:
        # NOTE This is not named 'apply' b/c the base class has a method with
        # that name and a different type signature.
        if not isinstance(node, dcir.Tasklet):
            raise ValueError("apply() requires dcir.Tasklet node")
        return super().apply(node, **kwargs)
