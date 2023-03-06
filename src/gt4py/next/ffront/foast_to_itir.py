# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from typing import Any, SupportsFloat, SupportsInt

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


def is_local_kind(symbol_type: ts.FieldType) -> bool:
    return any(dim.kind == DimensionKind.LOCAL for dim in symbol_type.dims)


def is_local_type_kind(type_):
    return any(
        isinstance(t, ts.FieldType) and is_local_kind(t)
        for t in type_info.primitive_constituents(type_)
    )


def promote_to_list(node: foast.Symbol | foast.Expr):
    if not hasattr(node, "type") or not is_local_type_kind(  # TODO hack for cast and unary op
        node.type
    ):
        return lambda x: im.as_lifted_lambda("make_const_list", x)
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

        # The pass works by making every expression a lifted stencil (i.e. takes iterators, returns iterator),
        # except FieldOperators are lowered to stencils (i.e. takes iterators, returns value) and lifted in calls.
        # TODO for consistency we could lift here, then deref in the program call.
        new_body = im.deref_(func_definition.expr)

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

        # TODO a lowering hack, they are valid itir expressions, but in a place where no iterators are allowed..
        # Could we fixed by running inline lift locally here and we would get a pure value expression.
        # TODO there are more places like this, e.g. init of reduce
        forward = im.deref_(self.visit(node.forward, **kwargs))
        init = im.deref_(self.visit(node.init, **kwargs))

        # lower definition function
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)
        new_body = func_definition.expr

        # promote carry to iterator
        new_body = im.let(
            func_definition.params[0].id,
            im.lift_(im.lambda__()(func_definition.params[0].id))(),
        )(im.deref_(new_body))
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
        return im.as_lifted_lambda(
            lambda tuple_: im.tuple_get_(node.index, tuple_), self.visit(node.value, **kwargs)
        )

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        return im.as_lifted_lambda(
            lambda *elts: im.make_tuple_(*elts),
            *[self.visit(el, **kwargs) for el in node.elts],
        )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        if node.op in [dialect_ast_enums.UnaryOperator.NOT, dialect_ast_enums.UnaryOperator.INVERT]:
            # TODO: invert only for bool right now
            return self._map("not_", node.operand)

        return self._map(node.op.value, im.as_lifted_capture(im.literal_("0", "int")), node.operand)

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> itir.FunCall:
        return self._map("if_", node.condition, node.true_expr, node.false_expr)

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.FunCall:
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                return im.shift_(offset_name, offset_index)(self.visit(node.func, **kwargs))
            case foast.Name(id=offset_name):
                return im.lift_(
                    im.lambda__("it")(im.neighbors_(im.ensure_offset(str(offset_name)), "it"))
                )(self.visit(node.func, **kwargs))

        raise FieldOperatorLoweringError("Unexpected shift arguments!")

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
            lowered_args = [self.visit(arg, **kwargs) for arg in node.args]
            args = [f"__arg{i}" for i in range(len(lowered_args))]
            return im.lift_(im.lambda__(*args)(im.call_(lowered_func)(*args)))(*lowered_args)
        elif isinstance(node.func.type, ts.FunctionType):
            return im.call_(self.visit(node.func, **kwargs))(*self.visit(node.args, **kwargs))

        raise AssertionError(
            f"Call to object of type {type(node.func.type).__name__} not understood."
        )

    def _visit_astype(self, node: foast.Call, **kwargs) -> itir.FunCall:
        assert len(node.args) == 2 and isinstance(node.args[1], foast.Name)
        obj, dtype = node.args[0], node.args[1].id

        # TODO check that we test astype that results in a itir.map_ operation
        return self._map("cast_", obj, im.as_lifted_capture(str(dtype)))

    def _visit_where(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._map("if_", *node.args)

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self.visit(node.args[0], **kwargs)

    def _visit_math_built_in(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._map(self.visit(node.func, **kwargs), *node.args)

    def _make_reduction_expr(
        self,
        node: foast.Call,
        op: Any,  # TODO fix
        init_expr: int | itir.Literal,
        **kwargs,
    ):
        # TODO(havogt): deal with nested reductions of the form neighbor_sum(neighbor_sum(field(off1)(off2)))
        it = self.visit(node.args[0], **kwargs)
        assert isinstance(node.kwargs["axis"].type, ts.DimensionType)
        val = im.call_(im.call_("reduce")(op, im.deref_(init_expr)))(im.deref_("it"))
        return im.lift_(im.lambda__("it")(val))(it)

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> itir.FunCall:
        assert isinstance(node.type, ts.FieldType)
        return self._make_reduction_expr(
            node, "plus", self._make_literal(0, node.type.dtype), **kwargs
        )

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
                return im.as_lifted_capture(  # TODO seems this codepath is not tested
                    im.literal_(str(bool(source_type(node.args[0].value))), node.func.id)
                )
            return im.as_lifted_capture(im.literal_(str(node.args[0].value), node_kind))
        raise FieldOperatorLoweringError(f"Encountered a type cast, which is not supported: {node}")

    def _make_literal(self, val: Any, type_: ts.TypeSpec) -> itir.Literal:
        # TODO(tehrengruber): check constant of this type is supported in iterator ir
        if isinstance(type_, ts.TupleType):
            return im.as_lifted_capture(
                im.make_tuple_(
                    *(
                        im.deref_(self._make_literal(val, type_))
                        for val, type_ in zip(val, type_.types)
                    )
                )
            )
        elif isinstance(type_, ts.ScalarType):
            typename = type_.kind.name.lower()
            return im.as_lifted_capture(im.literal_(str(val), typename))
        raise ValueError(f"Unsupported literal type {type_}.")

    def visit_Constant(self, node: foast.Constant, **kwargs) -> itir.Literal:
        return self._make_literal(node.value, node.type)

    def _map(self, op, *args, **kwargs):
        def _get_type(arg):  # TODO ugly hack for cast
            if hasattr(arg, "type"):
                return arg.type
            else:
                return None

        lowered_args = [self.visit(arg, **kwargs) for arg in args]
        if any(is_local_type_kind(_get_type(arg)) for arg in args):
            lowered_args = [promote_to_list(arg)(larg) for arg, larg in zip(args, lowered_args)]
            op = im.call_("map_")(op)

        return im.as_lifted_lambda(im.call_(op), *lowered_args)

        raise AssertionError()


class FieldOperatorLoweringError(Exception):
    ...
