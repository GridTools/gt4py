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
from typing import Callable, Optional, cast

from eve import NodeTranslator
from functional.ffront import (
    common_types as ct,
    fbuiltins,
    field_operator_ast as foast,
    itir_makers as im,
    type_info,
)
from functional.ffront.fbuiltins import FUN_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from functional.iterator import ir as itir


class TypeKind(enum.Enum):
    FIELD = 0
    SCALAR = 1
    UNKNOWN = 2


def resulting_type_kind(symbol_type: ct.SymbolType) -> TypeKind:
    """
    Determine whether the resulting type kind of ``symbol_type`` is scalar, field or unknown.

    Useful for determining whether an expression of ``symbol_type`` should be treated as a
    value or iterator expression during lowering.

    Examples:
    ---------
    >>> resulting_type_kind(ct.DeferredSymbolType(constraint=None)).name
    'UNKNOWN'

    >>> resulting_type_kind(ct.ScalarType(kind=ct.ScalarKind.BOOL)).name
    'SCALAR'

    >>> resulting_type_kind(ct.TupleType(types=[ct.ScalarType(kind=ct.ScalarKind.FLOAT64)])).name
    'SCALAR'
    """
    match symbol_type:
        case ct.DeferredSymbolType(constraint=None):
            return TypeKind.UNKNOWN
        case ct.TupleType(types=subtypes):
            if subtypes:
                return resulting_type_kind(subtypes[0])

    match type_info.type_class(symbol_type):
        case ct.FieldType:
            return TypeKind.FIELD
        case ct.ScalarType:
            return TypeKind.SCALAR

    return TypeKind.UNKNOWN


def can_be_value_or_iterator(symbol_type: ct.SymbolType):
    return resulting_type_kind(symbol_type) is not TypeKind.UNKNOWN


def to_value(node: foast.LocatedNode) -> Callable[[itir.Expr], itir.Expr]:
    """
    Either ``deref_`` or noop callable depending on the input node.

    Input node must have a scalar or field type.
    If the lowered input node will represent an iterator expression, return ``deref_``.
    Otherwise return a noop callable.

    Examples:
    ---------
    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>> from functional.ffront.fbuiltins import float64
    >>> from functional.common import Field
    >>> def foo(a: Field[..., "float64"]):
    ...    b = 5
    ...    return a, b

    >>> parsed = FieldOperatorParser.apply_to_function(foo)
    >>> field_a, scalar_b = parsed.body[-1].value.elts
    >>> to_value(field_a)(im.ref("a"))
    FunCall(fun=SymRef(id='deref'), args=[SymRef(id='a')])
    >>> to_value(scalar_b)(im.ref("a"))
    SymRef(id='a')
    """
    assert can_be_value_or_iterator(node.type)
    if resulting_type_kind(node.type) is TypeKind.FIELD:
        return im.deref_
    return lambda x: x


class FieldOperatorLowering(NodeTranslator):
    """
    Lower FieldOperator AST (FOAST) to Iterator IR (ITIR).

    Examples
    --------
    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>> from functional.ffront.fbuiltins import float64
    >>> from functional.common import Field
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

        return to_value(return_stmt.value)(current_expr)

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
            return type_info.type_class(expr.type) is ct.FieldType

        param_names = list(
            node.iter_tree().if_isinstance(foast.Name).filter(is_field).getattr("id").unique()
        )
        return self.lifted_lambda(*param_names)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        return im.tuple_get_(node.index, self.visit(node.value, **kwargs))

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        return im.make_tuple_(*self.visit(node.elts, **kwargs))

    def _lift_if_field(self, node: foast.LocatedNode) -> Callable[[itir.Expr], itir.Expr]:
        assert can_be_value_or_iterator(node.type)
        if resulting_type_kind(node.type) is TypeKind.SCALAR:
            return lambda x: x
        return self._lift_lambda(node)

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        if node.op is foast.UnaryOperator.NOT:
            return self._lift_if_field(node)(
                im.call_(node.op.value)(to_value(node.operand)(self.visit(node.operand, **kwargs)))
            )
        return self._lift_if_field(node)(
            im.call_(node.op.value)(
                im.literal_("0", "int"),
                to_value(node.operand)(self.visit(node.operand, **kwargs)),
            )
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        return self._lift_if_field(node)(
            im.call_(node.op.value)(
                to_value(node.left)(self.visit(node.left, **kwargs)),
                to_value(node.right)(self.visit(node.right, **kwargs)),
            )
        )

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        return self._lift_if_field(node)(
            im.call_(node.op.value)(
                to_value(node.left)(self.visit(node.left, **kwargs)),
                to_value(node.left)(self.visit(node.right, **kwargs)),
            )
        )

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.FunCall:
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                return im.shift_(offset_name, offset_index)(self.visit(node.func, **kwargs))
            case foast.Name(id=offset_name):
                return im.shift_(offset_name)(self.visit(node.func, **kwargs))
        raise FieldOperatorLoweringError("Unexpected shift arguments!")

    def _visit_reduce(self, node: foast.Call, **kwargs) -> itir.FunCall:
        lowering = InsideReductionLowering()
        expr = lowering.visit(node.args[0], **kwargs)
        params = list(lowering.lambda_params.items())
        return im.lift_(
            im.call_("reduce")(
                im.lambda__("accum", *(param[0] for param in params))(im.plus_("accum", expr)),
                0,
            )
        )(*(param[1] for param in params))

    def visit_Call(self, node: foast.Call, **kwargs) -> itir.FunCall:
        if type_info.type_class(node.func.type) is ct.FieldType:
            return self._visit_shift(node, **kwargs)
        elif node.func.id in FUN_BUILTIN_NAMES:
            visitor = getattr(self, f"_visit_{node.func.id}")
            return visitor(node, **kwargs)
        elif node.func.id in TYPE_BUILTIN_NAMES:
            return self._visit_type_constr(node, **kwargs)
        return self._lift_if_field(node)(
            im.call_(self.visit(node.func, **kwargs))(*self.visit(node.args, **kwargs))
        )

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> itir.FunCall:
        # just lower broadcasted field as iterator IR does not care about broadcasting
        broadcasted_field = node.args[0]
        return self.visit(broadcasted_field, **kwargs)

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._visit_reduce(node, **kwargs)

    def _visit_type_constr(self, node: foast.Call, **kwargs) -> itir.Literal:
        if isinstance(node.args[0], foast.Constant):
            target_type = fbuiltins.BUILTINS[node.type.kind.name.lower()]
            source_type = {**fbuiltins.BUILTINS, "string": str}[
                node.args[0].dtype.kind.name.lower()
            ]
            if target_type is bool and source_type is not bool:
                return im.literal_(str(bool(source_type(node.args[0].value))), node.func.id)
            return im.literal_(node.args[0].value, node.type.kind.name.lower())
        raise FieldOperatorLoweringError(f"Encountered a type cast, which is not supported: {node}")

    def visit_Constant(self, node: foast.Constant, **kwargs) -> itir.Literal:
        if isinstance(node.type, ct.ScalarType) and not node.type.shape:
            typename = node.type.kind.name.lower()
            return im.literal_(str(node.value), typename)
        raise FieldOperatorLoweringError(f"Unsupported scalar type: {node.dtype}")


@dataclass
class InsideReductionLowering(FieldOperatorLowering):
    """Variant of the lowering with special rules for inside reductions."""

    lambda_params: dict[str, itir.Expr] = field(default_factory=lambda: {})
    __counter: itertools.count = field(default_factory=lambda: itertools.count())

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.SymRef:
        uid = f"{node.id}__{self._sequential_id()}"
        self.lambda_params[uid] = super().visit_Name(node, **kwargs)
        return im.ref(uid)

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        return im.call_(node.op.value)(
            self.visit(node.left, **kwargs), self.visit(node.right, **kwargs)
        )

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        return im.call_(node.op.value)(
            self.visit(node.left, **kwargs), self.visit(node.right, **kwargs)
        )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        if node.op is foast.UnaryOperator.NOT:
            return im.call_(node.op.value)(self.visit(node.operand, **kwargs))

        return im.call_(node.op.value)(im.literal_("0", "int"), self.visit(node.operand, **kwargs))

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.SymRef:  # type: ignore[override]
        uid = f"{node.func.id}__{self._sequential_id()}"
        self.lambda_params[uid] = FieldOperatorLowering.apply(node)
        return im.ref(uid)

    def _sequential_id(self):
        return next(self.__counter)


class FieldOperatorLoweringError(Exception):
    ...
