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

import copy
import functools
import itertools
import numbers
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from gt4py.frontend.node_util import IRNodeMapper, IRNodeVisitor, location_to_source_location
from gt4py.frontend.nodes import (
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
    DataType,
    Empty,
    Expr,
    FieldDecl,
    FieldRef,
    HorizontalIf,
    If,
    IterationOrder,
    LevelMarker,
    NativeFuncCall,
    NativeFunction,
    Node,
    ScalarLiteral,
    StencilDefinition,
    TernaryOpExpr,
    UnaryOperator,
    UnaryOpExpr,
    VarDecl,
    VarRef,
    While,
)
from gtc import common, gtir
from gtc.common import ExprKind


def _convert_dtype(data_type) -> common.DataType:
    dtype = common.DataType(int(data_type))
    if dtype == common.DataType.DEFAULT:
        # TODO: this will be a frontend choice later
        # in non-GTC parts, this is set in the backend
        dtype = cast(
            common.DataType, common.DataType.FLOAT64
        )  # see https://github.com/GridTools/gtc/issues/100
    return dtype


def _make_literal(v: numbers.Number) -> gtir.Literal:
    value: Union[BuiltinLiteral, str]
    if isinstance(v, bool):
        dtype = common.DataType.BOOL
        value = common.BuiltInLiteral.TRUE if v else common.BuiltInLiteral.FALSE
    else:
        if isinstance(v, int):
            # note: could be extended to support 32 bit integers by checking the values magnitude
            dtype = common.DataType.INT64
        elif isinstance(v, float):
            dtype = common.DataType.FLOAT64
        else:
            print(
                f"Warning: Only INT64 and FLOAT64 literals are supported currently. Implicitly upcasting `{v}` to FLOAT64"
            )
            dtype = common.DataType.FLOAT64
        value = str(v)
    return gtir.Literal(dtype=dtype, value=value)


class UnrollVectorAssignments(IRNodeMapper):
    @classmethod
    def apply(cls, root, **kwargs):
        return cls().visit(root, **kwargs)

    def _is_vector_assignment(self, stmt: Node, fields_decls: Dict[str, FieldDecl]):
        if not isinstance(stmt, Assign):
            return False

        # does the referenced field has data dimensions and the access is not element wise
        return fields_decls[stmt.target.name].data_dims and not stmt.target.data_index

    def visit_StencilDefinition(
        self, node: StencilDefinition, *, fields_decls: Dict[str, FieldDecl], **kwargs
    ) -> StencilDefinition:
        node = copy.deepcopy(node)

        for c in node.computations:
            if c.body.stmts:
                new_stmts = []
                for stmt in c.body.stmts:
                    new_stmt = self.visit(stmt, fields_decls=fields_decls, **kwargs)
                    if isinstance(new_stmt, list):
                        new_stmts.extend(new_stmt)
                    else:
                        new_stmts.append(stmt)  # take stmt as is

                c.body.stmts = new_stmts

        return node

    # computes dimensions of nested lists
    def _nested_list_dim(self, a: List) -> List[int]:
        if not isinstance(a, list):
            return []
        return [len(a)] + self._nested_list_dim(a[0])

    def visit_Assign(
        self, node: Assign, *, fields_decls: Dict[str, FieldDecl], **kwargs
    ) -> Union[gtir.ParAssignStmt, List[gtir.ParAssignStmt]]:
        if self._is_vector_assignment(node, fields_decls):
            assert isinstance(node.target, FieldRef) or isinstance(node.target, VarRef)
            target_dims = fields_decls[node.target.name].data_dims

            unrolled_rhs = UnrollVectorExpressions.apply(
                node.value, expected_dim=target_dims, fields_decls=fields_decls
            )

            value_dims = self._nested_list_dim(unrolled_rhs)
            if target_dims != value_dims:
                raise Exception(
                    f"Assignment dimension mismatch: '{node.target.name}' has dim = {target_dims}; rhs has dim {value_dims}."
                )

            assign_list = []
            for index in itertools.product(*(range(dim) for dim in target_dims)):
                tmp_node = copy.deepcopy(node)
                value_node = functools.reduce(lambda arr, el: arr[el], index, unrolled_rhs)
                data_type = DataType.INT32
                data_index = [
                    ScalarLiteral(value=index_el, data_type=data_type) for index_el in index
                ]

                tmp_node.target.data_index = data_index
                tmp_node.value = value_node

                assign_list.append(tmp_node)

            return assign_list
        return node


class UnrollVectorExpressions(IRNodeMapper):
    @classmethod
    def apply(cls, root, *, expected_dim: Tuple[int, ...], fields_decls: Dict[str, FieldDecl]):
        result = cls().visit(root, fields_decls=fields_decls)
        # if the expression is just a scalar broadcast to the expected dimensions
        if not isinstance(result, list):
            result = functools.reduce(
                lambda val, len: [val for _ in range(len)], reversed(expected_dim), result
            )
        return result

    def visit_FieldRef(self, node: FieldRef, *, fields_decls: Dict[str, FieldDecl], **kwargs):
        name = node.name
        if fields_decls[name].data_dims:
            field_list = []
            # vector
            if len(fields_decls[name].data_dims) == 1:
                dims = fields_decls[name].data_dims[0]
                for index in range(dims):
                    data_type = DataType.INT32
                    data_index = [ScalarLiteral(value=index, data_type=data_type)]
                    element_ref = FieldRef(
                        name=node.name,
                        offset=node.offset,
                        data_index=data_index,
                        loc=node.loc,
                    )
                    field_list.append(element_ref)
            # matrix
            elif len(fields_decls[name].data_dims) == 2:
                rows, cols = fields_decls[name].data_dims
                for row in range(rows):
                    row_list = []
                    for col in range(cols):
                        data_type = DataType.INT32
                        data_index = [
                            ScalarLiteral(value=row, data_type=data_type),
                            ScalarLiteral(value=col, data_type=data_type),
                        ]
                        element_ref = FieldRef(
                            name=node.name,
                            offset=node.offset,
                            **kwargs,
                            data_index=data_index,
                            loc=node.loc,
                        )
                        row_list.append(element_ref)
                    field_list.append(row_list)
            else:
                raise Exception(
                    "Higher dimensional fields only supported for vectors and matrices."
                )
            return field_list

        return node

    def visit_UnaryOpExpr(self, node: UnaryOpExpr, *, fields_decls: Dict[str, FieldDecl], **kwargs):
        if node.op == UnaryOperator.TRANSPOSED:
            node = self.visit(node.arg, fields_decls=fields_decls, **kwargs)
            assert isinstance(node, list) and all(
                isinstance(row, list) and len(row) == len(node[0]) for row in node
            )
            # transpose list
            node = [list(x) for x in zip(*node)]
            return node

        return self.generic_visit(node, **kwargs)

    def visit_BinOpExpr(self, node: BinOpExpr, *, fields_decls: Dict[str, FieldDecl], **kwargs):
        lhs = self.visit(node.lhs, fields_decls=fields_decls, **kwargs)
        rhs = self.visit(node.rhs, fields_decls=fields_decls, **kwargs)
        result: Union[List[BinOpExpr], BinOpExpr] = []

        if node.op == BinaryOperator.MATMULT:
            for j in range(len(lhs)):
                acc = BinOpExpr(op=BinaryOperator.MUL, lhs=lhs[j][0], rhs=rhs[0], loc=node.loc)
                for i in range(1, len(lhs[0])):
                    mul = BinOpExpr(op=BinaryOperator.MUL, lhs=lhs[j][i], rhs=rhs[i], loc=node.loc)
                    acc = BinOpExpr(op=BinaryOperator.ADD, lhs=acc, rhs=mul, loc=node.loc)

                result.append(acc)
        else:
            # vector and vector
            if isinstance(lhs, list) and isinstance(rhs, list):
                assert len(lhs) == len(rhs)
                for lhs_el, rhs_el in zip(lhs, rhs):
                    result.append(BinOpExpr(op=node.op, lhs=lhs_el, rhs=rhs_el, loc=node.loc))
            # scalar and vector
            elif isinstance(lhs, Expr) and isinstance(rhs, list):
                for rhs_el in rhs:
                    result.append(BinOpExpr(op=node.op, lhs=lhs, rhs=rhs_el, loc=node.loc))
            elif isinstance(lhs, list) and isinstance(rhs, Expr):
                for lhs_el in lhs:
                    result.append(BinOpExpr(op=node.op, lhs=lhs_el, rhs=rhs, loc=node.loc))
            # scalar and scalar fallback
            else:
                result = self.generic_visit(node, **kwargs)

        return result


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
        NativeFunction.SINH: common.NativeFunction.SINH,
        NativeFunction.COSH: common.NativeFunction.COSH,
        NativeFunction.TANH: common.NativeFunction.TANH,
        NativeFunction.ARCSINH: common.NativeFunction.ARCSINH,
        NativeFunction.ARCCOSH: common.NativeFunction.ARCCOSH,
        NativeFunction.ARCTANH: common.NativeFunction.ARCTANH,
        NativeFunction.SQRT: common.NativeFunction.SQRT,
        NativeFunction.EXP: common.NativeFunction.EXP,
        NativeFunction.LOG: common.NativeFunction.LOG,
        NativeFunction.GAMMA: common.NativeFunction.GAMMA,
        NativeFunction.CBRT: common.NativeFunction.CBRT,
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

    def visit_StencilDefinition(self, node: StencilDefinition) -> gtir.Stencil:
        field_params = {f.name: self.visit(f) for f in node.api_fields}
        scalar_params = {p.name: self.visit(p) for p in node.parameters}
        vertical_loops = [self.visit(c) for c in node.computations if c.body.stmts]
        if node.externals is not None:
            externals = {
                name: _make_literal(value)
                for name, value in node.externals.items()
                if isinstance(value, numbers.Number)
            }
        else:
            externals = {}
        return gtir.Stencil(
            name=node.name,
            api_signature=[
                gtir.Argument(
                    name=f.name,
                    is_keyword=f.is_keyword,
                    default=str(f.default) if not isinstance(f.default, type(Empty)) else "",
                )
                for f in node.api_signature
            ],
            params=[
                self.visit(f, all_params={**field_params, **scalar_params})
                for f in node.api_signature
            ],
            vertical_loops=vertical_loops,
            externals=externals,
            sources=node.sources or {},
            docstring=node.docstring,
            loc=location_to_source_location(node.loc),
        )

    def visit_ArgumentInfo(self, node: ArgumentInfo, all_params: Dict[str, gtir.Decl]) -> gtir.Decl:
        return all_params[node.name]

    def visit_ComputationBlock(self, node: ComputationBlock) -> gtir.VerticalLoop:
        stmts = []
        temporaries = []
        for s in node.body.stmts:
            decl_or_stmt = self.visit(s)
            if isinstance(decl_or_stmt, gtir.Decl):
                temporaries.append(decl_or_stmt)
            else:
                stmts.append(decl_or_stmt)
        start, end = self.visit(node.interval)
        interval = gtir.Interval(
            start=start,
            end=end,
            loc=location_to_source_location(node.interval.loc),
        )
        return gtir.VerticalLoop(
            interval=interval,
            loop_order=self.GT4PY_ITERATIONORDER_TO_GTIR_LOOPORDER[node.iteration_order],
            body=stmts,
            temporaries=temporaries,
            loc=location_to_source_location(node.loc),
        )

    def visit_BlockStmt(self, node: BlockStmt) -> List[gtir.Stmt]:
        return [self.visit(s) for s in node.stmts]

    def visit_Assign(self, node: Assign) -> gtir.ParAssignStmt:
        assert isinstance(node.target, FieldRef) or isinstance(node.target, VarRef)
        return gtir.ParAssignStmt(
            left=self.visit(node.target),
            right=self.visit(node.value),
            loc=location_to_source_location(node.loc),
        )

    def visit_ScalarLiteral(self, node: ScalarLiteral) -> gtir.Literal:
        return gtir.Literal(value=str(node.value), dtype=_convert_dtype(node.data_type.value))

    def visit_UnaryOpExpr(self, node: UnaryOpExpr) -> gtir.UnaryOp:
        return gtir.UnaryOp(
            op=self.GT4PY_UNARYOP_TO_GTIR[node.op],
            expr=self.visit(node.arg),
            loc=location_to_source_location(node.loc),
        )

    def visit_BinOpExpr(self, node: BinOpExpr) -> Union[gtir.BinaryOp, gtir.NativeFuncCall]:
        if node.op in (BinaryOperator.POW, BinaryOperator.MOD):
            return gtir.NativeFuncCall(
                func=common.NativeFunction[node.op.name],
                args=[self.visit(node.lhs), self.visit(node.rhs)],
                loc=location_to_source_location(node.loc),
            )
        return gtir.BinaryOp(
            left=self.visit(node.lhs),
            right=self.visit(node.rhs),
            op=self.GT4PY_OP_TO_GTIR_OP[node.op],
            loc=location_to_source_location(node.loc),
        )

    def visit_TernaryOpExpr(self, node: TernaryOpExpr) -> gtir.TernaryOp:
        return gtir.TernaryOp(
            cond=self.visit(node.condition),
            true_expr=self.visit(node.then_expr),
            false_expr=self.visit(node.else_expr),
            loc=location_to_source_location(node.loc),
        )

    def visit_BuiltinLiteral(self, node: BuiltinLiteral) -> gtir.Literal:  # type: ignore[return]
        # currently deals only with boolean literals
        if node.value in self.GT4PY_BUILTIN_TO_GTIR.keys():
            return gtir.Literal(
                value=self.GT4PY_BUILTIN_TO_GTIR[node.value], dtype=common.DataType.BOOL
            )
        raise NotImplementedError(f"BuiltIn.{node.value} not implemented in lowering")

    def visit_Cast(self, node: Cast) -> gtir.Cast:
        return gtir.Cast(
            dtype=common.DataType(node.data_type.value),
            expr=self.visit(node.expr),
            loc=location_to_source_location(node.loc),
        )

    def visit_NativeFuncCall(self, node: NativeFuncCall) -> gtir.NativeFuncCall:
        return gtir.NativeFuncCall(
            func=self.GT4PY_NATIVE_FUNC_TO_GTIR[node.func],
            args=[self.visit(arg) for arg in node.args],
            loc=location_to_source_location(node.loc),
        )

    def visit_FieldRef(self, node: FieldRef) -> gtir.FieldAccess:
        return gtir.FieldAccess(
            name=node.name,
            offset=self.transform_offset(node.offset),
            data_index=[self.visit(index) for index in node.data_index],
            loc=location_to_source_location(node.loc),
        )

    def visit_If(self, node: If) -> Union[gtir.FieldIfStmt, gtir.ScalarIfStmt]:
        cond = self.visit(node.condition)
        if cond.kind == ExprKind.FIELD:
            return gtir.FieldIfStmt(
                cond=cond,
                true_branch=gtir.BlockStmt(body=self.visit(node.main_body)),
                false_branch=gtir.BlockStmt(body=self.visit(node.else_body))
                if node.else_body
                else None,
                loc=location_to_source_location(node.loc),
            )
        else:
            return gtir.ScalarIfStmt(
                cond=cond,
                true_branch=gtir.BlockStmt(body=self.visit(node.main_body)),
                false_branch=gtir.BlockStmt(body=self.visit(node.else_body))
                if node.else_body
                else None,
                loc=location_to_source_location(node.loc),
            )

    def visit_HorizontalIf(self, node: HorizontalIf) -> gtir.FieldIfStmt:
        def make_bound_or_level(bound: AxisBound, level) -> Optional[common.AxisBound]:
            if (level == LevelMarker.START and bound.offset <= -10000) or (
                level == LevelMarker.END and bound.offset >= 10000
            ):
                return None
            else:
                return common.AxisBound(
                    level=self.GT4PY_LEVELMARKER_TO_GTIR_LEVELMARKER[bound.level],
                    offset=bound.offset,
                )

        axes = {
            axis.lower(): common.HorizontalInterval(
                start=make_bound_or_level(node.intervals[axis].start, LevelMarker.START),
                end=make_bound_or_level(node.intervals[axis].end, LevelMarker.END),
            )
            for axis in ("I", "J")
        }

        return gtir.HorizontalRestriction(
            mask=common.HorizontalMask(**axes),
            body=self.visit(node.body),
        )

    def visit_While(self, node: While) -> gtir.While:
        return gtir.While(
            cond=self.visit(node.condition),
            body=self.visit(node.body),
            loc=location_to_source_location(node.loc),
        )

    def visit_VarRef(self, node: VarRef, **kwargs):
        return gtir.ScalarAccess(name=node.name, loc=location_to_source_location(node.loc))

    def visit_AxisInterval(self, node: AxisInterval) -> Tuple[gtir.AxisBound, gtir.AxisBound]:
        return self.visit(node.start), self.visit(node.end)

    def visit_AxisBound(self, node: AxisBound) -> gtir.AxisBound:
        # TODO(havogt) add support VarRef
        return gtir.AxisBound(
            level=self.GT4PY_LEVELMARKER_TO_GTIR_LEVELMARKER[node.level], offset=node.offset
        )

    def visit_FieldDecl(self, node: FieldDecl) -> gtir.FieldDecl:
        dimension_names = ["I", "J", "K"]
        dimensions = tuple(dim in node.axes for dim in dimension_names)
        # datatype conversion works via same ID
        return gtir.FieldDecl(
            name=node.name,
            dtype=_convert_dtype(node.data_type.value),
            dimensions=dimensions,
            data_dims=tuple(node.data_dims),
            loc=location_to_source_location(node.loc),
        )

    def visit_VarDecl(self, node: VarDecl) -> gtir.ScalarDecl:
        # datatype conversion works via same ID
        return gtir.ScalarDecl(
            name=node.name,
            dtype=_convert_dtype(node.data_type.value),
            loc=location_to_source_location(node.loc),
        )

    def transform_offset(
        self, offset: Dict[str, Union[int, Expr]], **kwargs: Any
    ) -> Union[common.CartesianOffset, gtir.VariableKOffset]:
        k_val = offset.get("K", 0)
        if isinstance(k_val, numbers.Integral):
            return common.CartesianOffset(i=offset.get("I", 0), j=offset.get("J", 0), k=k_val)
        elif isinstance(k_val, Expr):
            return gtir.VariableKOffset(k=self.visit(k_val, **kwargs))
        else:
            raise TypeError("Unrecognized vertical offset type")
