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

import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional, cast

from eve import NodeTranslator
from functional.ffront import common_types as ct
from functional.ffront import field_operator_ast as foast
from functional.ffront import itir_makers as im
from functional.ffront.fbuiltins import FUN_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from functional.ffront.type_info import TypeInfo
from functional.iterator import ir as itir


def to_value(node: foast.LocatedNode) -> Callable[[itir.Expr], itir.Expr]:
    if TypeInfo(node.type).is_field_type:
        return im.deref_
    return lambda x: x


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
        expr = self.visit(node.value, **kwargs)
        return sym, expr

    def visit_Return(self, node: foast.Return, **kwargs) -> itir.Expr:
        return self.visit(node.value, **kwargs)

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> itir.Sym:
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.SymRef:
        return im.ref(node.id)

    def _lift_lambda(self, node):
        def is_field(expr: foast.Expr) -> bool:
            return TypeInfo(expr.type).is_field_type

        param_names = list(
            node.iter_tree().if_isinstance(foast.Name).filter(is_field).getattr("id").unique()
        )
        return self.lifted_lambda(*param_names)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        return im.tuple_get_(node.index, self.visit(node.value, **kwargs))

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        return im.make_tuple_(
            *(self.visit(element, **kwargs) for i, element in enumerate(node.elts))
        )

    def _lift_if_field(self, node: foast.LocatedNode) -> Callable[[itir.Expr], itir.Expr]:
        if TypeInfo(node.type).is_scalar:
            return lambda x: x
        return self._lift_lambda(node)

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        zero_arg = (
            [itir.NumberLiteral(value="0", type="int")]
            if node.op is not foast.UnaryOperator.NOT
            else []
        )
        result = im.call_(node.op.value)(
            *[*zero_arg, to_value(node.operand)(self.visit(node.operand, **kwargs))]
        )
        return self._lift_if_field(node)(result)

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        result = im.call_(node.op.value)(
            to_value(node.left)(self.visit(node.left, **kwargs)),
            to_value(node.right)(self.visit(node.right, **kwargs)),
        )
        return self._lift_if_field(node)(result)

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        result = im.call_(node.op.value)(
            to_value(node.left)(self.visit(node.left, **kwargs)),
            to_value(node.left)(self.visit(node.right, **kwargs)),
        )
        return self._lift_if_field(node)(result)

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.FunCall:
        result = None
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                result = im.shift_(offset_name, offset_index)(self.visit(node.func, **kwargs))
            case foast.Name(id=offset_name):
                result = im.shift_(offset_name)(self.visit(node.func, **kwargs))
            case _:
                raise FieldOperatorLoweringError("Unexpected shift arguments!")
        return result

    def _visit_reduce(self, node: foast.Call, **kwargs) -> itir.FunCall:
        lowering = InsideReductionLowering()
        expr = lowering.visit(node.args[0], **kwargs)
        params = list(lowering.lambda_params.items())
        result = im.call_("reduce")(
            im.lambda__("accum", *(param[0] for param in params))(im.plus_("accum", expr)),
            0,
        )
        return im.lift_(result)(*(param[1] for param in params))

    def visit_Call(self, node: foast.Call, **kwargs) -> itir.FunCall:
        if TypeInfo(node.func.type).is_field_type:
            return self._visit_shift(node, **kwargs)
        elif node.func.id in FUN_BUILTIN_NAMES:
            visitor = getattr(self, f"_visit_{node.func.id}")
            return visitor(node, **kwargs)
        elif node.func.id in TYPE_BUILTIN_NAMES:
            return self._visit_type_constr(node, **kwargs)
        result = im.call_(self.visit(node.func, **kwargs))(*self.visit(node.args, **kwargs))
        return self._lift_if_field(node)(result)

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._visit_reduce(node, **kwargs)

    def _visit_type_constr(self, node: foast.Call, **kwargs) -> itir.NumberLiteral:
        if isinstance(node.args[0], foast.Constant):
            return im.number_(node.func.id, str(node.args[0].value))
        return self.visit(node.args[0], **kwargs)

    def visit_Constant(
        self, node: foast.Constant, **kwargs
    ) -> itir.NumberLiteral | itir.BoolLiteral:
        result = None
        match node.dtype:
            case ct.ScalarType(kind=ct.ScalarKind.FLOAT32) | "float32":
                result = im.number_("float32", str(node.value))
            case ct.ScalarType(kind=ct.ScalarKind.FLOAT64) | "float64":
                result = im.number_("float64", str(node.value))
            case ct.ScalarType(kind=ct.ScalarKind.INT32) | "int32":
                result = im.number_("int32", str(node.value))
            case ct.ScalarType(kind=ct.ScalarKind.INT64) | "int64":
                result = im.number_("int64", str(node.value))
            case ct.ScalarType(kind=ct.ScalarKind.BOOL) | "bool":
                value = False if node.value == "False" else node.value
                result = im.bool_(bool(value))
        if not result:
            raise FieldOperatorLoweringError(f"Unsupported scalar type: {node.dtype}")
        return result


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
        zero_arg = (
            [itir.NumberLiteral(value="0", type="int")]
            if node.op is not foast.UnaryOperator.NOT
            else []
        )
        return im.call_(node.op.value)(*[*zero_arg, self.visit(node.operand, **kwargs)])

    def _visit_shift(self, node: foast.Call, *, to_value: bool = False, **kwargs) -> itir.SymRef:  # type: ignore[override]
        uid = f"{node.func.id}__{self._sequential_id()}"
        self.lambda_params[uid] = FieldOperatorLowering.apply(node)
        return im.ref(uid)

    def _sequential_id(self):
        return next(self.__counter)


class FieldOperatorLoweringError(Exception):
    ...
