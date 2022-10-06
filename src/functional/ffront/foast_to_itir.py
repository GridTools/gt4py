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
from typing import Any, Optional

import numpy as np

from eve import NodeTranslator
from functional.ffront import (
    common_types as ct,
    fbuiltins,
    field_operator_ast as foast,
    itir_makers as im,
    type_info,
)
from functional.ffront.fbuiltins import FUN_BUILTIN_NAMES, MATH_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from functional.iterator import ir as itir


class ITIRTypeKind(enum.Enum):
    VALUE = 0
    ITERATOR = 1


def iterator_type_kind(
    symbol_type: ct.ScalarType | ct.FieldType | ct.TupleType,
) -> ITIRTypeKind:
    """
    Return the corresponding type kind (on iterator level) to a FOAST expression of the given symbol type.

    This function is used both to decide on how to lower an foast expression
    of the given type and how to handle such expressions in other expressions.

    - VALUE: The lowered expression is a value, e.g. a scalar or tuple thereof.
    - ITERATOR: The lowered expression is an iterator that can be dereferenced,
        returning a value or composite object of values (e.g. tuple).

    +------------------------------------+------------------------+
    | FOAST Expr                         | Iterator Type Kind     |
    +====================================+========================+
    | 1                                  | VALUE                  |
    | field                              | ITERATOR               |
    | (1, 1)                             | VALUE                  |
    | (1, field)                         | ITERATOR               |
    +------------------------------------+------------------------+
    """
    if any(type_info.primitive_constituents(symbol_type).if_isinstance(ct.FieldType)):
        return ITIRTypeKind.ITERATOR
    return ITIRTypeKind.VALUE


def to_iterator(node: foast.LocatedNode):
    if iterator_type_kind(node.type) is ITIRTypeKind.VALUE:
        return lambda x: im.lift_(im.lambda__()(x))()
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
    SymbolName('fieldop')
    >>> lowered.params
    [Sym(id=SymbolName('inp'))]
    """

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> itir.Expr:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> itir.FunctionDefinition:
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)

        # value arguments, e.g. scalars and tuples thereof, are passed as
        #  iterators (see visit_Call for corresponding promotion to iterators).
        #  deref them here such that they are in the "format" expected by the
        #  rest of the lowering inside the body. See ADR-0002 for more details.
        new_body = func_definition.expr
        for i, param in enumerate(func_definition.params):
            if isinstance(node.definition.params[i].type, ct.ScalarType):
                new_body = im.let(param.id, im.deref_(param.id))(new_body)

        return itir.FunctionDefinition(
            id=func_definition.id,
            params=func_definition.params,
            expr=im.deref_(new_body),
        )

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs
    ) -> itir.FunctionDefinition:
        params = self.visit(node.params)
        return itir.FunctionDefinition(
            id=node.id,
            params=params,
            expr=self._visit_body(node.body, params=params),
        )

    def visit_ScanOperator(self, node: foast.ScanOperator, **kwargs) -> itir.FunctionDefinition:
        # note: we don't need the axis here as this is handled by the program
        #  decorator

        forward = self.visit(node.forward, **kwargs)
        init = self.visit(node.init, **kwargs)

        # lower definition function
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)
        new_body = func_definition.expr
        for param in func_definition.params[1:]:
            new_body = im.let(param.id, im.deref_(param.id))(new_body)
        definition = itir.Lambda(params=func_definition.params, expr=new_body)
        body = im.call_(im.call_("scan")(definition, forward, init))(
            *(param.id for param in definition.params[1:])
        )

        return itir.FunctionDefinition(
            id=node.id,
            params=definition.params[1:],
            expr=body,
        )

    def _visit_body(
        self, body: list[foast.Stmt], params: Optional[list[itir.Sym]] = None, **kwargs
    ) -> itir.FunCall:
        *assigns, return_stmt = body
        current_expr = self.visit(return_stmt, **kwargs)

        for assign in reversed(assigns):
            target = self.visit(assign.target, **kwargs)
            expr = self.visit(assign.value, **kwargs)
            current_expr = im.let(target, expr)(current_expr)

        return current_expr

    def visit_Return(self, node: foast.Return, **kwargs) -> itir.Expr:
        return self.visit(node.value, **kwargs)

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> itir.Sym:
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.SymRef:
        return im.ref(node.id)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        return self._map(lambda val: im.tuple_get_(node.index, val), node.value)

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        return self._map(lambda *elts: im.make_tuple_(*elts), *node.elts)

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        if node.op is foast.UnaryOperator.NOT:
            return self._map(node.op.value, node.value)

        return self._map(node.op.value, im.literal_("0", "int"), node.operand)

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> itir.FunCall:
        op = lambda it1, it2: im.call_("if_")(self.visit(node.condition, **kwargs), it1, it2)
        return self._map(op, node.true_expr, node.false_expr)

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.FunCall:
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                return im.lift_(
                    im.lambda__("it")(
                        im.deref_(im.shift_(offset_name, offset_index)("it"))))(
                    self.visit(node.func, **kwargs))
            case foast.Name(id=offset_name):
                return im.shift_(offset_name)(self.visit(node.func, **kwargs))
        raise FieldOperatorLoweringError("Unexpected shift arguments!")

    def visit_Call(self, node: foast.Call, **kwargs) -> itir.FunCall:
        if type_info.type_class(node.func.type) is ct.FieldType:
            return self._visit_shift(node, **kwargs)
        elif node.func.id in MATH_BUILTIN_NAMES:
            return self._visit_math_built_in(node, **kwargs)
        elif node.func.id in FUN_BUILTIN_NAMES:
            visitor = getattr(self, f"_visit_{node.func.id}")
            return visitor(node, **kwargs)
        elif node.func.id in TYPE_BUILTIN_NAMES:
            return self._visit_type_constr(node, **kwargs)
        elif isinstance(node.func.type, (ct.FieldOperatorType, ct.ScanOperatorType)):
            # operators are lowered into stencils and only accept iterator
            #  arguments. As such transform all value arguments, e.g. scalars
            #  and tuples thereof, into iterators. See ADR-0002 for more
            #  details.
            lowered_func = self.visit(node.func, **kwargs)
            lowered_args = [to_iterator(arg)(self.visit(arg, **kwargs)) for arg in node.args]
            args = [f"__arg{i}" for i in range(len(lowered_args))]
            return im.lift_(im.lambda__(*args)(im.call_(lowered_func)(*args)))(*lowered_args)

        raise AssertionError(
            f"Call to object of type {type(node.func.type).__name__} not understood."
        )

    def _visit_where(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._map(im.call_("if_"), *node.args)

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return to_iterator(node.args[0])(self.visit(node.args[0], **kwargs))

    def _visit_math_built_in(self, node: foast.Call, **kwargs) -> itir.Expr:
        return self._map(self.visit(node.func, **kwargs), *node.args)

    def _make_reduction_expr(
        self,
        node: foast.Call,
        op: str,
        init_expr: int | itir.Literal,
        **kwargs,
    ):
        it = self.visit(node.args[0], **kwargs)
        val = im.call_(im.call_("reduce")(op, init_expr))("it")
        return im.lift_(im.lambda__("it")(val))(it)

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._make_reduction_expr(node, "plus", 0, **kwargs)
    def _visit_max_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        init_expr = itir.Literal(value=str(np.finfo(np.float64).min), type="float64")
        return self._make_reduction_expr(node, "maximum", init_expr, **kwargs)

    def _visit_min_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        init_expr = itir.Literal(value=str(np.finfo(np.float64).max), type="float64")
        return self._make_reduction_expr(node, "minimum", init_expr, **kwargs)

    def _visit_type_constr(self, node: foast.Call, **kwargs) -> itir.Literal:
        if isinstance(node.args[0], foast.Constant):
            target_type = fbuiltins.BUILTINS[node.type.kind.name.lower()]
            source_type = {**fbuiltins.BUILTINS, "string": str}[node.args[0].type.kind.name.lower()]
            if target_type is bool and source_type is not bool:
                return im.literal_(str(bool(source_type(node.args[0].value))), node.func.id)
            return im.literal_(str(node.args[0].value), node.type.kind.name.lower())
        raise FieldOperatorLoweringError(f"Encountered a type cast, which is not supported: {node}")

    def _make_literal(self, val: Any, type_: ct.ScalarType) -> itir.Literal:
        typename = type_.kind.name.lower()
        return im.literal_(str(val), typename)

    def visit_Constant(self, node: foast.Constant, **kwargs) -> itir.Literal:
        # TODO: check constant is supported in iterator ir
        if isinstance(node.type, ct.ScalarType) and not node.type.shape:
            return self._make_literal(node.value, node.type)
        elif isinstance(node.type, ct.TupleType):
            assert all(isinstance(type_, ct.ScalarType) for type_ in node.type.types)
            return im.make_tuple_(
                *(self._make_literal(val, type_) for val, type_ in zip(node.value, node.type.types))
            )
        raise FieldOperatorLoweringError(f"Unsupported scalar type: {node.type}")

    def _map(self, op, *args, **kwargs):
        if isinstance(op, str):
            op = im.call_(op)

        lowered_args = [self.visit(arg, **kwargs) for arg in args]
        if any(iterator_type_kind(arg.type) is ITIRTypeKind.ITERATOR for arg in args):
            lowered_args = [to_iterator(arg)(larg) for arg, larg in zip(args, lowered_args)]
            return im.map_(op, *lowered_args)
        elif all(iterator_type_kind(arg.type) is ITIRTypeKind.VALUE for arg in args):
            return op(*lowered_args)
        raise AssertionError()


class FieldOperatorLoweringError(Exception):
    ...
