# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace
from typing import Any, Dict, List, Union, cast

from gt4py.ir import IRNodeVisitor
from gt4py.ir.nodes import (
    ArgumentInfo,
    Assign,
    AxisBound,
    AxisInterval,
    BinaryOperator,
    BinOpExpr,
    BlockStmt,
    Builtin,
    BuiltinLiteral,
    Cast,
    ComputationBlock,
    FieldDecl,
    FieldRef,
    If,
    IterationOrder,
    LevelMarker,
    NativeFuncCall,
    NativeFunction,
    ScalarLiteral,
    StencilDefinition,
    TernaryOpExpr,
    UnaryOperator,
    UnaryOpExpr,
    VarDecl,
    VarRef,
)
from gtc import common, gtir
from gtc.common import ExprKind


class DefIRToGTIR(IRNodeVisitor):

    GT4PY_ITERATIONORDER_TO_GTIR_LOOPORDER = {
        IterationOrder.BACKWARD: common.LoopOrder.BACKWARD,
        IterationOrder.PARALLEL: common.LoopOrder.PARALLEL,
        IterationOrder.FORWARD: common.LoopOrder.FORWARD,
    }

    GT4PY_LEVELMARKER_TO_GTIR_LEVELMARKER = {
        LevelMarker.START: common.LevelMarker.START,
        LevelMarker.END: common.LevelMarker.END,
    }

    GT4PY_OP_TO_GTIR_OP = {
        # arithmetic
        BinaryOperator.ADD: common.ArithmeticOperator.ADD,
        BinaryOperator.SUB: common.ArithmeticOperator.SUB,
        BinaryOperator.MUL: common.ArithmeticOperator.MUL,
        BinaryOperator.DIV: common.ArithmeticOperator.DIV,
        # logical
        BinaryOperator.AND: common.LogicalOperator.AND,
        BinaryOperator.OR: common.LogicalOperator.OR,
        # comparison
        BinaryOperator.EQ: common.ComparisonOperator.EQ,
        BinaryOperator.NE: common.ComparisonOperator.NE,
        BinaryOperator.LT: common.ComparisonOperator.LT,
        BinaryOperator.LE: common.ComparisonOperator.LE,
        BinaryOperator.GT: common.ComparisonOperator.GT,
        BinaryOperator.GE: common.ComparisonOperator.GE,
    }

    GT4PY_UNARYOP_TO_GTIR = {
        UnaryOperator.POS: common.UnaryOperator.POS,
        UnaryOperator.NEG: common.UnaryOperator.NEG,
        UnaryOperator.NOT: common.UnaryOperator.NOT,
    }

    GT4PY_NATIVE_FUNC_TO_GTIR = {
        NativeFunction.ABS: common.NativeFunction.ABS,
        NativeFunction.MIN: common.NativeFunction.MIN,
        NativeFunction.MAX: common.NativeFunction.MAX,
        NativeFunction.MOD: common.NativeFunction.MOD,
        NativeFunction.SIN: common.NativeFunction.SIN,
        NativeFunction.COS: common.NativeFunction.COS,
        NativeFunction.TAN: common.NativeFunction.TAN,
        NativeFunction.ARCSIN: common.NativeFunction.ARCSIN,
        NativeFunction.ARCCOS: common.NativeFunction.ARCCOS,
        NativeFunction.ARCTAN: common.NativeFunction.ARCTAN,
        NativeFunction.SQRT: common.NativeFunction.SQRT,
        NativeFunction.EXP: common.NativeFunction.EXP,
        NativeFunction.LOG: common.NativeFunction.LOG,
        NativeFunction.ISFINITE: common.NativeFunction.ISFINITE,
        NativeFunction.ISINF: common.NativeFunction.ISINF,
        NativeFunction.ISNAN: common.NativeFunction.ISNAN,
        NativeFunction.FLOOR: common.NativeFunction.FLOOR,
        NativeFunction.CEIL: common.NativeFunction.CEIL,
        NativeFunction.TRUNC: common.NativeFunction.TRUNC,
    }

    GT4PY_BUILTIN_TO_GTIR = {
        Builtin.TRUE: common.BuiltInLiteral.TRUE,
        Builtin.FALSE: common.BuiltInLiteral.FALSE,
    }

    @classmethod
    def apply(cls, root, **kwargs):
        return cls().visit(root)

    def __init__(self):
        self._field_params = {}
        self._scalar_params = {}

    def visit_StencilDefinition(self, node: StencilDefinition) -> gtir.Stencil:
        self._field_params = {f.name: self.visit(f) for f in node.api_fields}
        self._scalar_params = {p.name: self.visit(p) for p in node.parameters}
        vertical_loops = [self.visit(c) for c in node.computations if c.body.stmts]
        return gtir.Stencil(
            name=node.name.split(".")[
                -1
            ],  # TODO probably definition IR should not contain '.' in the name
            params=[
                self.visit(f, all_params={**self._field_params, **self._scalar_params})
                for f in node.api_signature
            ],
            vertical_loops=vertical_loops,
        )

    def visit_ArgumentInfo(
        self, node: ArgumentInfo, all_params: Dict[str, Union[gtir.Decl]]
    ) -> Union[gtir.Decl]:
        return all_params[node.name]

    def visit_ComputationBlock(self, node: ComputationBlock) -> gtir.VerticalLoop:
        stmts = []
        temporaries = []
        for s in node.body.stmts:
            # FieldDecl or VarDecls in the body are temporaries
            if isinstance(s, FieldDecl) or isinstance(s, VarDecl):
                dtype = common.DataType(int(s.data_type.value))
                if dtype == common.DataType.DEFAULT:
                    # TODO this will be a frontend choice later
                    # in non-GTC parts, this is set in the backend
                    dtype = cast(
                        common.DataType, common.DataType.FLOAT64
                    )  # see https://github.com/GridTools/gtc/issues/100
                temporaries.append(
                    gtir.FieldDecl(
                        name=s.name,
                        dtype=dtype,
                        dimensions=(True, True, True),
                    )
                )
            else:
                stmts.append(self.visit(s))
        start, end = self.visit(node.interval)
        interval = gtir.Interval(start=start, end=end)
        return gtir.VerticalLoop(
            interval=interval,
            loop_order=self.GT4PY_ITERATIONORDER_TO_GTIR_LOOPORDER[node.iteration_order],
            body=stmts,
            temporaries=temporaries,
        )

    def visit_BlockStmt(self, node: BlockStmt, **kwargs: Any) -> List[gtir.Stmt]:
        return [self.visit(s, **kwargs) for s in node.stmts]

    def visit_Assign(self, node: Assign, **kwargs: Any) -> gtir.ParAssignStmt:
        assert isinstance(node.target, FieldRef) or isinstance(node.target, VarRef)
        return gtir.ParAssignStmt(
            left=self.visit(node.target, **kwargs), right=self.visit(node.value, **kwargs)
        )

    def visit_ScalarLiteral(self, node: ScalarLiteral) -> gtir.Literal:
        return gtir.Literal(value=str(node.value), dtype=common.DataType(node.data_type.value))

    def visit_UnaryOpExpr(self, node: UnaryOpExpr, **kwargs: Any) -> gtir.UnaryOp:
        return gtir.UnaryOp(
            op=self.GT4PY_UNARYOP_TO_GTIR[node.op], expr=self.visit(node.arg, **kwargs)
        )

    def visit_BinOpExpr(
        self, node: BinOpExpr, **kwargs: Any
    ) -> Union[gtir.BinaryOp, gtir.NativeFuncCall]:
        if node.op in (BinaryOperator.POW, BinaryOperator.MOD):
            return gtir.NativeFuncCall(
                func=common.NativeFunction[node.op.name],
                args=[self.visit(node.lhs, **kwargs), self.visit(node.rhs, **kwargs)],
            )
        return gtir.BinaryOp(
            left=self.visit(node.lhs, **kwargs),
            right=self.visit(node.rhs, **kwargs),
            op=self.GT4PY_OP_TO_GTIR_OP[node.op],
        )

    def visit_TernaryOpExpr(self, node: TernaryOpExpr, **kwargs: Any) -> gtir.TernaryOp:
        return gtir.TernaryOp(
            cond=self.visit(node.condition, **kwargs),
            true_expr=self.visit(node.then_expr, **kwargs),
            false_expr=self.visit(node.else_expr, **kwargs),
        )

    def visit_BuiltinLiteral(self, node: BuiltinLiteral, **kwargs: Any) -> gtir.Literal:  # type: ignore[return]
        # currently deals only with boolean literals
        if node.value in self.GT4PY_BUILTIN_TO_GTIR.keys():
            return gtir.Literal(
                value=self.GT4PY_BUILTIN_TO_GTIR[node.value], dtype=common.DataType.BOOL
            )
        raise NotImplementedError(f"BuiltIn.{node.value} not implemented in lowering")

    def visit_Cast(self, node: Cast, **kwargs: Any) -> gtir.Cast:
        return gtir.Cast(
            dtype=common.DataType(node.data_type.value), expr=self.visit(node.expr, **kwargs)
        )

    def visit_NativeFuncCall(self, node: NativeFuncCall, **kwargs: Any) -> gtir.NativeFuncCall:
        return gtir.NativeFuncCall(
            func=self.GT4PY_NATIVE_FUNC_TO_GTIR[node.func],
            args=[self.visit(arg, **kwargs) for arg in node.args],
        )

    def visit_FieldRef(self, node: FieldRef, **kwargs: Any) -> gtir.FieldAccess:
        if field_decl := self._field_params.get(node.name, None):
            dtype = field_decl.dtype
        else:
            dtype = None
        return gtir.FieldAccess(
            name=node.name,
            offset=self._transform_offset(node.offset, **kwargs),
            data_index=node.data_index,
            dtype=dtype,
        )

    def visit_If(self, node: If, **kwargs: Any) -> Union[gtir.FieldIfStmt, gtir.ScalarIfStmt]:
        cond = self.visit(node.condition)
        if cond.kind == ExprKind.FIELD:
            return gtir.FieldIfStmt(
                cond=cond,
                true_branch=gtir.BlockStmt(body=self.visit(node.main_body, **kwargs)),
                false_branch=gtir.BlockStmt(body=self.visit(node.else_body, **kwargs))
                if node.else_body
                else None,
            )
        else:
            return gtir.ScalarIfStmt(
                cond=cond,
                true_branch=gtir.BlockStmt(body=self.visit(node.main_body, **kwargs)),
                false_branch=gtir.BlockStmt(body=self.visit(node.else_body, **kwargs))
                if node.else_body
                else None,
            )

    def visit_VarRef(
        self, node: VarRef, **kwargs: Any
    ) -> Union[gtir.ScalarAccess, gtir.FieldAccess]:
        # TODO(havogt) seems wrong, but check the DefinitionIR for
        # test_code_generation.py::test_generation_cpu[native_functions,
        # there we have a FieldAccess on a VarDecl
        # Probably the frontend needs to be fixed.
        if node.name in self._scalar_params:
            return gtir.ScalarAccess(name=node.name)
        else:
            return gtir.FieldAccess(name=node.name, offset=gtir.CartesianOffset.zero())

    def visit_AxisInterval(self, node: AxisInterval):
        return self.visit(node.start), self.visit(node.end)

    def visit_AxisBound(self, node: AxisBound):
        # TODO(havogt) add support VarRef
        return gtir.AxisBound(
            level=self.GT4PY_LEVELMARKER_TO_GTIR_LEVELMARKER[node.level], offset=node.offset
        )

    def visit_FieldDecl(self, node: FieldDecl):
        dimension_names = ["I", "J", "K"]
        dimensions = [dim in node.axes for dim in dimension_names]
        # datatype conversion works via same ID
        return gtir.FieldDecl(
            name=node.name,
            dtype=common.DataType(int(node.data_type.value)),
            dimensions=dimensions,
            data_dims=node.data_dims,
        )

    def visit_VarDecl(self, node: VarDecl):
        # datatype conversion works via same ID
        return gtir.ScalarDecl(name=node.name, dtype=common.DataType(int(node.data_type.value)))

    def _transform_offset(self, offset: Dict[str, int], **kwargs: Any) -> gtir.CartesianOffset:
        i = offset["I"] if "I" in offset else 0
        j = offset["J"] if "J" in offset else 0
        k = offset["K"] if "K" in offset else 0
        if isinstance(k, int):
            return gtir.CartesianOffset(i=i, j=j, k=k)
        return gtir.VariableOffset(i=i, j=j, k=self.visit(k, **kwargs))
