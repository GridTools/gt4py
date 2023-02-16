# GT4Py - GridTools Framework
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

import enum
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from gt4py.eve import NodeTranslator
from gt4py.next.common import DimensionKind
from gt4py.next.ffront import (
    dialect_ast_enums,
    fbuiltins,
    field_operator_ast as foast,
    itir_makers as im,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.fbuiltins import FUN_BUILTIN_NAMES, MATH_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_info, type_specifications as ts


class ITIRTypeKind(enum.Enum):
    VALUE = 0
    ITERATOR = 1


def is_local_kind(symbol_type: ts.FieldType) -> bool:
    return any(dim.kind == DimensionKind.LOCAL for dim in symbol_type.dims)


def iterator_type_kind(
    symbol_type: ts.TypeSpec,
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
    assert not isinstance(symbol_type, ts.DeferredType)
    if any(type_info.primitive_constituents(symbol_type).if_isinstance(ts.FieldType)):
        return ITIRTypeKind.ITERATOR
    return ITIRTypeKind.VALUE


def to_iterator(node: foast.Symbol | foast.Expr):
    if iterator_type_kind(node.type) is ITIRTypeKind.VALUE:
        return lambda x: im.lift_(im.lambda__()(x))()
    return lambda x: x


class FieldOperatorLowering(NodeTranslator):
    """
    Lower FieldOperator AST (FOAST) to Iterator IR (ITIR).

    Examples
    --------
    >>> from gt4py.next.ffront.func_to_foast import FieldOperatorParser
    >>> from gt4py.next.ffront.fbuiltins import float64
    >>> from gt4py.next.common import Field, Dimension
    >>>
    >>> IDim = Dimension("IDim")
    >>> def fieldop(inp: Field[[IDim], "float64"]):
    ...    return inp
    >>>
    >>> parsed = FieldOperatorParser.apply_to_function(fieldop)
    >>> lowered = FieldOperatorLowering.apply(parsed)
    >>> type(lowered)
    <class 'gt4py.next.iterator.ir.FunctionDefinition'>
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
            if iterator_type_kind(node.definition.params[i].type) == ITIRTypeKind.VALUE:
                new_body = im.let(param.id, im.deref_(param.id))(new_body)

        assert isinstance(node.type, ts_ffront.FieldOperatorType)
        if iterator_type_kind(node.type.definition.returns) == ITIRTypeKind.ITERATOR:
            new_body = im.deref_(new_body)

        return itir.FunctionDefinition(
            id=func_definition.id,
            params=func_definition.params,
            expr=new_body,
        )

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs
    ) -> itir.FunctionDefinition:
        params = self.visit(node.params)
        return itir.FunctionDefinition(
            id=node.id,
            params=params,
            expr=self.visit_BlockStmt(node.body, inner_expr=None),
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

    def visit_Stmt(self, node: foast.Stmt, **kwargs):
        raise AssertionError("Statements must always be visited in the context of a function.")

    def visit_Return(
        self, node: foast.Return, *, inner_expr: itir.Expr | None, **kwargs
    ) -> itir.Expr:
        return self.visit(node.value, **kwargs)

    def visit_BlockStmt(
        self, node: foast.BlockStmt, *, inner_expr: itir.Expr | None, **kwargs
    ) -> itir.Expr:
        for stmt in reversed(node.stmts):
            inner_expr = self.visit(stmt, inner_expr=inner_expr, **kwargs)
        assert inner_expr
        return inner_expr

    def visit_Assign(
        self, node: foast.Assign, *, inner_expr: itir.Expr | None, **kwargs
    ) -> itir.Expr:
        return im.let(self.visit(node.target, **kwargs), self.visit(node.value, **kwargs))(
            inner_expr
        )

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> itir.Sym:
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.SymRef:
        return im.ref(node.id)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        if iterator_type_kind(node.type) is ITIRTypeKind.ITERATOR:
            return im.map_(
                lambda tuple_: im.tuple_get_(node.index, tuple_), self.visit(node.value, **kwargs)
            )
        return im.tuple_get_(node.index, self.visit(node.value, **kwargs))

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        if iterator_type_kind(node.type) is ITIRTypeKind.ITERATOR:
            return im.map_(
                lambda *elts: im.make_tuple_(*elts),
                *[to_iterator(el)(self.visit(el, **kwargs)) for el in node.elts],
            )
        return im.make_tuple_(*self.visit(node.elts, **kwargs))

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        if node.op in [dialect_ast_enums.UnaryOperator.NOT, dialect_ast_enums.UnaryOperator.INVERT]:
            # TODO: invert only for bool right now
            return self._map("not_", node.operand)

        return self._map(node.op.value, im.literal_("0", "int"), node.operand)

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> itir.FunCall:
        def op(it1, it2):
            return im.call_("if_")(self.visit(node.condition, **kwargs), it1, it2)

        return self._map(op, node.true_expr, node.false_expr)

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.FunCall:
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                return im.lift_(
                    im.lambda__("it")(im.deref_(im.shift_(offset_name, offset_index)("it")))
                )(self.visit(node.func, **kwargs))
            case foast.Name(id=offset_name):
                return im.call_(
                    im.call_("translate_shift")(
                        im.ensure_offset(str(offset_name + "Dim")),
                        im.ensure_offset(str(offset_name)),
                    )
                )(self.visit(node.func, **kwargs))
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
        ftype = node.type
        assert isinstance(ftype, ts.FieldType)
        init_expr = itir.Literal(value="0", type=str(ftype.dtype))
        return self._make_reduction_expr(
            node, lambda expr: im.plus_("acc", expr), init_expr, **kwargs
        )

    def _visit_max_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): replace greater_ with max_ builtin as soon as itir supports it
        init_expr = itir.Literal(value=str(np.finfo(np.float64).min), type="float64")
        return self._make_reduction_expr(
            node,
            lambda expr: im.call_("if_")(im.greater_("acc", expr), "acc", expr),
            init_expr,
            **kwargs,
        )

    def _visit_min_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        init_expr = itir.Literal(value=str(np.finfo(np.float64).max), type="float64")
        return self._make_reduction_expr(
            node,
            lambda expr: im.call_("if_")(im.less_("acc", expr), "acc", expr),
            init_expr,
            **kwargs,
        )

    def visit_Call(self, node: foast.Call, **kwargs) -> itir.FunCall | itir.Literal:
        if type_info.type_class(node.func.type) is ts.FieldType:
            return self._visit_shift(node, **kwargs)
        elif node.func.id in MATH_BUILTIN_NAMES:
            return self._visit_math_built_in(node, **kwargs)
        elif node.func.id in FUN_BUILTIN_NAMES:
            visitor = getattr(self, f"_visit_{node.func.id}")
            return visitor(node, **kwargs)
        elif node.func.id in TYPE_BUILTIN_NAMES:
            return self._visit_type_constr(node, **kwargs)
        elif isinstance(
            node.func.type,
            (
                ts_ffront.FieldOperatorType,
                ts_ffront.ScanOperatorType,
            ),
        ):
            # operators are lowered into stencils and only accept iterator
            #  arguments. As such transform all value arguments, e.g. scalars
            #  and tuples thereof, into iterators. See ADR-0002 for more
            #  details.
            lowered_func = self.visit(node.func, **kwargs)
            lowered_args = [to_iterator(arg)(self.visit(arg, **kwargs)) for arg in node.args]
            if iterator_type_kind(node.type) == ITIRTypeKind.ITERATOR:
                args = [f"__arg{i}" for i in range(len(lowered_args))]
                return im.lift_(im.lambda__(*args)(im.call_(lowered_func)(*args)))(*lowered_args)
            else:
                return im.call_(lowered_func)(*lowered_args)
        elif isinstance(node.func.type, ts.FunctionType):
            return im.call_(self.visit(node.func, **kwargs))(*self.visit(node.args, **kwargs))

        raise AssertionError(
            f"Call to object of type {type(node.func.type).__name__} not understood."
        )

    def _visit_astype(self, node: foast.Call, **kwargs) -> itir.FunCall:
        assert len(node.args) == 2 and isinstance(node.args[1], foast.Name)
        obj, dtype = node.args[0], node.args[1].id
        return self._map(lambda val: im.call_("cast_")(val, dtype), obj)

    def _visit_where(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._map(im.call_("if_"), *node.args)

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return to_iterator(node.args[0])(self.visit(node.args[0], **kwargs))

    def _visit_math_built_in(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._map(self.visit(node.func, **kwargs), *node.args)

    def _make_reduction_expr(
        self,
        node: foast.Call,
        op: str,
        init_expr: int | itir.Literal,
        **kwargs,
    ):
        it = self.visit(node.args[0], **kwargs)
        assert isinstance(node.kwargs["axis"].type, ts.DimensionType)
        reduction_axis = im.ensure_offset(str(node.kwargs["axis"].type.dim.value) + "Dim")
        val = im.call_(im.call_("reduce")(op, init_expr, reduction_axis))("it")
        return im.call_(im.call_("ignore_shift")(reduction_axis))(
            im.lift_(im.lambda__("it")(val))(it)
        )

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._make_reduction_expr(node, "plus", 0, **kwargs)

    def _visit_max_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        assert isinstance(node.type, ts.FieldType)
        np_type = getattr(np, node.type.dtype.kind.name.lower())
        min_value: SupportsInt | SupportsFloat
        if type_info.is_integral(node.type):
            min_value = np.iinfo(np.int32).min  # not sure why int64 min is converted into an int128
        elif type_info.is_floating_point(node.type):
            min_value = np.finfo(np_type).min
        else:
            raise AssertionError(
                "`max_over` is only defined for integral or floating point types."
                " This error should have been catched in type deduction aready."
            )
        init_expr = self._make_literal(min_value, node.type.dtype)
        return self._make_reduction_expr(node, "maximum", init_expr, **kwargs)

    def _visit_min_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        assert isinstance(node.type, ts.FieldType)
        np_type = getattr(np, node.type.dtype.kind.name.lower())
        max_value: SupportsInt | SupportsFloat
        if type_info.is_integral(node.type):
            max_value = np.iinfo(np.int32).max
        elif type_info.is_floating_point(node.type):
            max_value = np.finfo(np_type).max
        else:
            raise AssertionError(
                "`min_over` is only defined for integral or floating point types."
                " This error should have been catched in type deduction aready."
            )
        init_expr = self._make_literal(max_value, node.type.dtype)
        return self._make_reduction_expr(node, "minimum", init_expr, **kwargs)

    def _visit_type_constr(self, node: foast.Call, **kwargs) -> itir.Literal:
        if isinstance(node.args[0], foast.Constant):
            node_kind = self.visit(node.type).kind.name.lower()
            target_type = fbuiltins.BUILTINS[node_kind]
            source_type = {**fbuiltins.BUILTINS, "string": str}[node.args[0].type.__str__().lower()]
            if target_type is bool and source_type is not bool:
                return im.literal_(str(bool(source_type(node.args[0].value))), node.func.id)
            return im.literal_(str(node.args[0].value), node_kind)
        raise FieldOperatorLoweringError(f"Encountered a type cast, which is not supported: {node}")

    def _make_literal(self, val: Any, type_: ts.TypeSpec) -> itir.Literal:
        # TODO(tehrengruber): check constant of this type is supported in iterator ir
        if isinstance(type_, ts.TupleType):
            return im.make_tuple_(
                *(self._make_literal(val, type_) for val, type_ in zip(val, type_.types))
            )
        elif isinstance(type_, ts.ScalarType):
            typename = type_.kind.name.lower()
            return im.literal_(str(val), typename)
        raise ValueError(f"Unsupported literal type {type_}.")

    def visit_Constant(self, node: foast.Constant, **kwargs) -> itir.Literal:
        return self._make_literal(node.value, node.type)

    def _map(self, op, *args, **kwargs):
        if isinstance(op, (str, itir.SymRef)):
            op = im.call_(op)

        def is_local_type_kind(type_):
            return any(
                isinstance(t, ts.FieldType) and is_local_kind(t)
                for t in type_info.primitive_constituents(type_)
            )

        lowered_args = [self.visit(arg, **kwargs) for arg in args]
        if any(iterator_type_kind(arg.type) is ITIRTypeKind.ITERATOR for arg in args):
            lowered_args = [to_iterator(arg)(larg) for arg, larg in zip(args, lowered_args)]

            if any(is_local_type_kind(arg.type) for arg in args):
                # TODO: multiple different nb dims?
                def local_dim(field_type):
                    for dim in field_type.dims:
                        if dim.kind == DimensionKind.LOCAL:
                            return dim
                    raise AssertionError()

                nb_dims = [local_dim(arg.type) for arg in args if is_local_type_kind(arg.type)]
                assert all(nb_dim == nb_dims[0] for nb_dim in nb_dims)
                nb_dim = im.ensure_offset(str(nb_dims[0].value) + "Dim")
                for i, arg in enumerate(args):
                    if not is_local_type_kind(arg.type):
                        lowered_args[i] = im.call_(im.call_("ignore_shift")(nb_dim))(
                            lowered_args[i]
                        )

                result = im.map_(op, *lowered_args)
                return result
            else:
                return im.map_(op, *lowered_args)
        elif all(iterator_type_kind(arg.type) is ITIRTypeKind.VALUE for arg in args):
            return op(*lowered_args)
        raise AssertionError()


class FieldOperatorLoweringError(Exception):
    ...
