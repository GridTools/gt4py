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

import numpy as np

from eve import NodeTranslator
from functional.common import DimensionKind
from functional.ffront import (
    common_types as ct,
    fbuiltins,
    field_operator_ast as foast,
    itir_makers as im,
    type_info,
)
from functional.ffront.fbuiltins import FUN_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from functional.iterator import ir as itir


def is_local_kind(symbol_type: ct.FieldType) -> bool:
    assert isinstance(symbol_type, ct.FieldType)
    if symbol_type.dims == ...:
        return False
    return any(dim.kind == DimensionKind.LOCAL for dim in symbol_type.dims)


class IteratorTypeKind(enum.Enum):
    VALUE = 0
    ITERATOR = 1
    ENCAPSULATED_ITERATOR = 2


def iterator_type_kind(
    symbol_type: ct.ScalarType | ct.FieldType | ct.TupleType,
) -> IteratorTypeKind:
    """
    Return the kind of type on iterator level an foast expr of given type corresponds to.

    This function is used both to decide on how to lower an foast expression
    of the given type and how to handle such expressions in other expressions.

    - VALUE: The expression is a value, e.g. a scalar.
    - ITERATOR: The expression can be derefed returning a value or composite
        object of values (e.g. tuple).
    - ENCAPSULATED_ITERATOR: The expression is a composite object (e.g. tuple)
        that contains at least one iterator.

    +------------------------------------+------------------------+
    | Expr                               | Iterator Type Kind     |
    +====================================+========================+
    | 1                                  | VALUE                  |
    | regular_field                      | ITERATOR               |
    | local_field                        | ITERATOR               |
    | (1, 1)                             | VALUE                  |
    | (1, regular_field)                 | ITERATOR               |
    | (1, local_field)                   | ENCAPSULATED_ITERATOR  |
    | (regular_field, local_field)       | ENCAPSULATED_ITERATOR  |
    | (1, (1, regular_field))            | ITERATOR               |
    | (1, (1, local_field))              | ENCAPSULATED_ITERATOR  |
    | (1, (local_field, regular_field))  | ENCAPSULATED_ITERATOR  |
    +------------------------------------+------------------------+
    """
    if isinstance(symbol_type, ct.FieldType):
        return IteratorTypeKind.ITERATOR
    elif any(type_info.primitive_constituents(symbol_type).if_isinstance(ct.FieldType)):
        # if we encounter any field type that is defined on a local dimension
        #  the resulting type on iterator ir level is not an iterator, but contains
        #  one, e.g. a tuple of iterators.
        if any(
            type_info.primitive_constituents(symbol_type)
            .if_isinstance(ct.FieldType)
            .filter(is_local_kind)
        ):
            return IteratorTypeKind.ENCAPSULATED_ITERATOR
        # otherwise we get an iterator, e.g. an iterator of values or tuples
        return IteratorTypeKind.ITERATOR
    return IteratorTypeKind.VALUE


def is_expr_with_iterator_type_kind(it_type_kind: IteratorTypeKind) -> Callable[[foast.Expr], bool]:
    def predicate(node: foast.Expr):
        return iterator_type_kind(node.type) is it_type_kind

    return predicate


def to_value(node: foast.LocatedNode) -> Callable[[itir.Expr], itir.Expr]:
    """
    Either ``deref_`` or noop callable depending on the input node.

    Input node must have a scalar, non-local field, or tuple of non-local fields
    type. If the lowered input node will represent an iterator expression,
    return ``deref_``. Otherwise return a noop callable.

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
    FunCall(fun=SymRef(id=SymbolRef('deref')), args=[SymRef(id=SymbolRef('a'))])
    >>> to_value(scalar_b)(im.ref("a"))
    SymRef(id=SymbolRef('a'))
    """
    if iterator_type_kind(node.type) is IteratorTypeKind.ITERATOR:
        # just to ensure we don't accidentally deref a local field
        assert not (isinstance(node.type, ct.FieldType) and is_local_kind(node.type))
        return im.deref_
    elif iterator_type_kind(node.type) is IteratorTypeKind.VALUE:
        return lambda x: x

    raise AssertionError(f"Type {node.type} can not be turned into a value.")


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
    SymbolName('fieldop')
    >>> lowered.params
    [Sym(id=SymbolName('inp'))]
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
        symtable = node.annex.symtable
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

    def _lift_lambda(self, node: foast.LocatedNode):
        if any(
            node.pre_walk_values()
            .if_isinstance(foast.Name)
            .filter(is_expr_with_iterator_type_kind(IteratorTypeKind.ENCAPSULATED_ITERATOR))
        ):
            raise NotImplementedError(
                "Using composite types (e.g. tuples) containing local fields not supported."
            )
        param_names = (
            node.pre_walk_values()
            .if_isinstance(foast.Name)
            .filter(is_expr_with_iterator_type_kind(IteratorTypeKind.ITERATOR))
            .getattr("id")
            .unique()
            .to_list()
        )
        return self.lifted_lambda(*param_names)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        value = self.visit(node.value, **kwargs)
        if iterator_type_kind(node.value.type) is IteratorTypeKind.ITERATOR:
            return self._lift_lambda(node)(im.tuple_get_(node.index, im.deref_(value)))
        elif iterator_type_kind(node.value.type) in (
            IteratorTypeKind.VALUE,
            IteratorTypeKind.ENCAPSULATED_ITERATOR,
        ):
            return im.tuple_get_(node.index, value)
        raise AssertionError("Unexpected `IteratorTypeKind`.")

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        # it is important to use `node` here instead of `el` to decide if we
        #  want to have a value. As soon as we have one local field in the
        #  expression we choose a tuple of iterators layout (which other
        #  parts of the lowering rely on).
        if iterator_type_kind(node.type) is IteratorTypeKind.ITERATOR:
            elts = tuple(to_value(el)(self.visit(el, **kwargs)) for el in node.elts)
            return self._lift_lambda(node)(im.make_tuple_(*elts))
        elif iterator_type_kind(node.type) in (
            IteratorTypeKind.VALUE,
            IteratorTypeKind.ENCAPSULATED_ITERATOR,
        ):
            elts = tuple(self.visit(el, **kwargs) for el in node.elts)
            return im.make_tuple_(*elts)
        raise AssertionError("Unexpected `IteratorTypeKind`.")

    def _lift_if_field(self, node: foast.LocatedNode) -> Callable[[itir.Expr], itir.Expr]:
        if iterator_type_kind(node.type) is IteratorTypeKind.VALUE:
            return lambda x: x
        elif iterator_type_kind(node.type) is IteratorTypeKind.ITERATOR:
            return self._lift_lambda(node)
        raise AssertionError("Unexpected `IteratorTypeKind`.")

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
                to_value(node.right)(self.visit(node.right, **kwargs)),
            )
        )

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.FunCall:
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                return im.shift_(offset_name, offset_index)(self.visit(node.func, **kwargs))
            case foast.Name(id=offset_name):
                return im.shift_(offset_name)(self.visit(node.func, **kwargs))
        raise FieldOperatorLoweringError("Unexpected shift arguments!")

    def _make_reduction_expr(
        self,
        node: foast.Call,
        op: Callable[[itir.Expr], itir.Expr],
        init_expr: int | itir.Literal,
        **kwargs,
    ):
        lowering = InsideReductionLowering()
        expr = lowering.visit(node.args[0], **kwargs)
        params = list(lowering.lambda_params.items())
        return im.lift_(
            im.call_("reduce")(
                im.lambda__("acc", *(param[0] for param in params))(op(expr)),
                init_expr,
            )
        )(*(param[1] for param in params))

    def _visit_reduce(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._make_reduction_expr(node, lambda expr: im.plus_("acc", expr), 0, **kwargs)

    def _visit_max_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): replace greater_ with max_ builtin as soon as itir supports it
        init_expr = itir.Literal(value=str(np.finfo(np.float64).min), type="float64")
        return self._make_reduction_expr(
            node,
            lambda expr: im.call_("if_")(im.greater_("acc", expr), "acc", expr),
            init_expr,
            **kwargs,
        )

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

    def _visit_where(self, node: foast.Call, **kwargs) -> itir.FunCall:
        mask, left, right = (to_value(arg)(self.visit(arg, **kwargs)) for arg in node.args)
        # since the if_ builtin expects a value for the condition we need to
        #  use a lifted-lambda here such that the mask is also shifted on a
        #  subsequent shift.
        return self._lift_lambda(node)(im.call_("if_")(mask, left, right))

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> itir.FunCall:
        broadcasted_field = node.args[0]

        # just lower broadcasted field and ignore second argument as iterator
        #  IR does not care about broadcasting
        lowered_arg = self.visit(broadcasted_field, **kwargs)

        # if the argument is a scalar though convert it into an iterator.
        #  This is an artefact originating from the relation between the type
        #  deduction and the lowering. When a scalar is broadcasted the resulting
        #  type is a field. As such the lowering expects an iterator and tries
        #  to deref it.
        if isinstance(broadcasted_field.type, ct.ScalarType):
            assert (
                len(
                    node.pre_walk_values()
                    .if_isinstance(foast.Name)
                    .filter(lambda expr: isinstance(expr.type, ct.FieldType))
                    .to_list()
                )
                == 0
            )
            lowered_arg = im.lift_(im.lambda__()(lowered_arg))()

        return lowered_arg

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
        if iterator_type_kind(node.type) is IteratorTypeKind.ENCAPSULATED_ITERATOR:
            raise NotImplementedError(
                "Using composite types (e.g. tuples) containing local fields not supported."
            )
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
