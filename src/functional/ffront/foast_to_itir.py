# GT4Py Project - GridTools Framework
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

import enum
import itertools
from dataclasses import dataclass, field
from typing import Optional, cast

from eve import NodeTranslator
from functional.ffront import common_types as ct
from functional.ffront import field_operator_ast as foast
from functional.ffront import itir_makers as im
from functional.ffront.fbuiltins import FUN_BUILTIN_NAMES, float32, float64, int32, int64
from functional.ffront.type_info import TypeInfo
from functional.iterator import ir as itir


class ItOrValue(enum.Enum):
    ITERATOR = "iterator"
    VALUE = "value"


class FieldOperatorLowering(NodeTranslator):
    """
    Lower FieldOperator AST (FOAST) to Iterator IR (ITIR).

    Examples
    --------
    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>> from functional.common import Field
    >>>
    >>> float64 = float
    >>>
    >>> def fieldop(inp: Field[..., "float64"]):
    ...    return inp
    >>>
    >>> parsed = FieldOperatorParser.apply_to_function(fieldop)
    >>> lowered = FieldOperatorLowering.apply(parsed)
    >>> type(lowered)
    <class 'functional.iterator.ir.FunctionDefinition'>
    >>> lowered.id
    'fieldop'
    >>> lowered.params
    [Sym(id='inp')]
    """

    class lifted_lambda:
        def __init__(self, *params):
            self.params = params

        def __call__(self, expr):
            return im.lift_(im.lambda__(*self.params)(expr))(*self.params)

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> itir.Expr:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> itir.FunctionDefinition:
        symtable = node.symtable_
        params = self.visit(node.params, symtable=symtable)
        return itir.FunctionDefinition(
            id=node.id,
            params=params,
            expr=self._visit_body(node.body, params=params, symtable=symtable),
        )

    def _visit_body(
        self, body: list[foast.Stmt], params: Optional[list[itir.Sym]] = None, **kwargs
    ) -> itir.FunCall:
        *assigns, return_stmt = body
        current_expr = self.visit(return_stmt, **kwargs)

        for assign in reversed(assigns):
            current_expr = im.let(*self._visit_assign(cast(foast.Assign, assign), **kwargs))(
                current_expr
            )

        return im.deref_(current_expr)

    def _visit_assign(self, node: foast.Assign, **kwargs) -> tuple[itir.Sym, itir.Expr]:
        sym = self.visit(node.target, **kwargs)
        to_value = TypeInfo(node.value.type).is_scalar
        expr = self.visit(node.value, to_value=to_value, **kwargs)
        return sym, expr

    def visit_Return(self, node: foast.Return, **kwargs) -> itir.Expr:
        return self.visit(node.value, **kwargs)

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> itir.Sym:
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, *, to_value: bool = False, **kwargs) -> itir.SymRef:
        typeinfo = TypeInfo(node.type)
        if typeinfo.is_scalar and not to_value:
            return im.lift_(im.lambda__()(im.ref(node.id)))()
        elif typeinfo.is_field_type and to_value:
            return im.deref_(node.id)
        return im.ref(node.id)

    def _lift_lambda(self, node):
        def is_field(expr: foast.Expr) -> bool:
            return TypeInfo(expr.type).is_field_type

        param_names = list(
            node.iter_tree().if_isinstance(foast.Name).filter(is_field).getattr("id").unique()
        )
        return self.lifted_lambda(*param_names)

    def visit_Subscript(
        self, node: foast.Subscript, *, to_value: bool = False, **kwargs
    ) -> itir.FunCall:
        typeinfo = TypeInfo(TypeInfo(node.value.type).element_types[node.index])
        result = im.tuple_get_(node.index, self.visit(node.value, **kwargs))
        if typeinfo.is_scalar and not to_value:
            return im.lift_(im.lambda__()(result))()
        elif typeinfo.is_field_type and to_value:
            return im.deref_(result)
        return result

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        return im.make_tuple_(*self.visit(node.elts, **kwargs))

    def visit_UnaryOp(
        self, node: foast.UnaryOp, *, to_value: bool = False, **kwargs
    ) -> itir.FunCall:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        zero_arg = [itir.IntLiteral(value=0)] if node.op is not foast.UnaryOperator.NOT else []
        value = im.call_(node.op.value)(
            *[*zero_arg, self.visit(node.operand, to_value=True, **kwargs)]
        )
        if to_value:
            return value
        return self._lift_lambda(node)(value)

    def visit_BinOp(self, node: foast.BinOp, *, to_value: bool = False, **kwargs) -> itir.FunCall:
        value = im.call_(node.op.value)(
            self.visit(node.left, to_value=True, **kwargs),
            self.visit(node.right, to_value=True, **kwargs),
        )
        if to_value:
            return value
        return self._lift_lambda(node)(value)

    def visit_Compare(
        self, node: foast.Compare, *, to_value: bool = False, **kwargs
    ) -> itir.FunCall:
        value = im.call_(node.op.value)(
            im.deref_(self.visit(node.left, **kwargs)),
            im.deref_(self.visit(node.right, **kwargs)),
        )
        if to_value:
            return value
        return self._lift_lambda(node)(value)

    def _visit_shift(self, node: foast.Call, *, to_value: bool = True, **kwargs) -> itir.FunCall:
        result = None
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                result = im.shift_(offset_name, offset_index)(self.visit(node.func, **kwargs))
            case foast.Name(id=offset_name):
                result = im.shift_(offset_name)(self.visit(node.func, **kwargs))
            case _:
                raise FieldOperatorLoweringError("Unexpected shift arguments!")
        if to_value:
            return im.deref_(result)
        return result

    def _visit_reduce(self, node: foast.Call, *, to_value=True, **kwargs) -> itir.FunCall:
        lowering = InsideReductionLowering()
        expr = lowering.visit(node.args[0], **kwargs)
        params = list(lowering.lambda_params.items())
        result = im.call_("reduce")(
            im.lambda__("accum", *(param[0] for param in params))(im.plus_("accum", expr)),
            0,
        )
        if to_value:
            return result(*(param[1] for param in params))
        return im.lift_(result)(*(param[1] for param in params))

    def visit_Call(self, node: foast.Call, *, to_value: bool = False, **kwargs) -> itir.FunCall:
        if TypeInfo(node.func.type).is_field_type:
            return self._visit_shift(node, to_value=to_value, **kwargs)
        elif node.func.id in FUN_BUILTIN_NAMES:
            visitor = getattr(self, f"_visit_{node.func.id}")
            return visitor(node, to_value=to_value, **kwargs)
        result = im.call_(self.visit(node.func, **kwargs))(*self.visit(node.args, **kwargs))
        if to_value:
            return result
        return self._lift_lambda(node)(result)

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._visit_reduce(node, **kwargs)

    def _visit_float32_(self, node: foast.Call, **kwargs) -> itir.IntLiteral | itir.FloatLiteral:
        return self.visit(node.args[0], **kwargs)

    def _visit_float64_(self, node: foast.Call, **kwargs) -> itir.IntLiteral | itir.FloatLiteral:
        return self.visit(node.args[0], **kwargs)

    def _visit_int32_(self, node: foast.Call, **kwargs) -> itir.IntLiteral | itir.FloatLiteral:
        return self.visit(node.args[0], **kwargs)

    def _visit_int64_(self, node: foast.Call, **kwargs) -> itir.IntLiteral | itir.FloatLiteral:
        return self.visit(node.args[0], **kwargs)

    def visit_Constant(
        self, node: foast.Constant, *, to_value: bool = False, **kwargs
    ) -> itir.IntLiteral | itir.FloatLiteral | itir.BoolLiteral:
        result = None
        match node.dtype:
            case ct.ScalarType(kind=ct.ScalarKind.FLOAT32) | "float32":
                result = im.float_(float32(node.value))
            case ct.ScalarType(kind=ct.ScalarKind.FLOAT64) | "float64":
                result = im.float_(float64(node.value))
            case ct.ScalarType(kind=ct.ScalarKind.INT32) | "int32":
                result = im.int_(int32(node.value))
            case ct.ScalarType(kind=ct.ScalarKind.INT64) | "int64":
                result = im.int_(int64(node.value))
            case ct.ScalarType(kind=ct.ScalarKind.BOOL) | "bool":
                value = False if node.value == "False" else node.value
                result = im.bool_(bool(value))
        if not result:
            raise FieldOperatorLoweringError(f"Unsupported scalar type: {node.dtype}")
        if to_value:
            return result
        return im.lift_(im.lambda__()(result))()


@dataclass
class InsideReductionLowering(FieldOperatorLowering):
    """Variant of the lowering with special rules for inside reductions."""

    lambda_params: dict[str, itir.Expr] = field(default_factory=lambda: {})
    __counter: itertools.count = field(default_factory=lambda: itertools.count())

    def visit_Name(self, node: foast.Name, *, to_value: bool = False, **kwargs) -> itir.SymRef:
        uid = f"{node.id}__{self._sequential_id()}"
        self.lambda_params[uid] = super().visit_Name(node, **kwargs)
        return im.ref(uid)

    def visit_BinOp(self, node: foast.BinOp, *, to_value: bool = False, **kwargs) -> itir.FunCall:
        return im.call_(node.op.value)(
            self.visit(node.left, **kwargs), self.visit(node.right, **kwargs)
        )

    def visit_Compare(
        self, node: foast.Compare, *, to_value: bool = False, **kwargs
    ) -> itir.FunCall:
        return im.call_(node.op.value)(
            self.visit(node.left, **kwargs), self.visit(node.right, **kwargs)
        )

    def visit_UnaryOp(
        self, node: foast.UnaryOp, *, to_value: bool = False, **kwargs
    ) -> itir.FunCall:
        zero_arg = [itir.IntLiteral(value=0)] if node.op is not foast.UnaryOperator.NOT else []
        return im.call_(node.op.value)(*[*zero_arg, self.visit(node.operand, **kwargs)])

    def _visit_shift(self, node: foast.Call, *, to_value: bool = False, **kwargs) -> itir.SymRef:  # type: ignore[override]
        uid = f"{node.func.id}__{self._sequential_id()}"
        self.lambda_params[uid] = FieldOperatorLowering.apply(node)
        return im.ref(uid)

    def _sequential_id(self):
        return next(self.__counter)


class FieldOperatorLoweringError(Exception):
    ...
