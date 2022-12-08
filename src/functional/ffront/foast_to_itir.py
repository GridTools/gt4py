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
from typing import Any, Callable

import numpy as np

from eve import NodeTranslator
from eve.utils import UIDGenerator
from functional.common import DimensionKind
from functional.ffront import (
    common_types as ct,
    fbuiltins,
    field_operator_ast as foast,
    itir_makers as im,
    type_info,
)
from functional.ffront.fbuiltins import FUN_BUILTIN_NAMES, MATH_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from functional.iterator import ir as itir


def is_local_kind(symbol_type: ct.FieldType) -> bool:
    assert isinstance(symbol_type, ct.FieldType)
    if symbol_type.dims == ...:
        return False
    return any(dim.kind == DimensionKind.LOCAL for dim in symbol_type.dims)


class ITIRTypeKind(enum.Enum):
    VALUE = 0
    ITERATOR = 1
    ENCAPSULATED_ITERATOR = 2


def iterator_type_kind(symbol_type: ct.SymbolType) -> ITIRTypeKind:
    """
    Return the corresponding type kind (on iterator level) to a FOAST expression of the given symbol type.

    This function is used both to decide on how to lower an foast expression
    of the given type and how to handle such expressions in other expressions.

    - VALUE: The lowered expression is a value, e.g. a scalar.
    - ITERATOR: The lowered expression is an iterator that can be dereferenced,
        returning a value or composite object of values (e.g. tuple).
    - ENCAPSULATED_ITERATOR: The lowered expression is a composite object
        (e.g. tuple) that contains at least one iterator.

    +------------------------------------+------------------------+
    | FOAST Expr                         | Iterator Type Kind     |
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
    assert not isinstance(symbol_type, ct.DeferredSymbolType)
    if isinstance(symbol_type, ct.FieldType):
        return ITIRTypeKind.ITERATOR
    elif any(type_info.primitive_constituents(symbol_type).if_isinstance(ct.FieldType)):
        # if we encounter any field type that is defined on a local dimension
        #  the resulting type on iterator ir level is not an iterator, but contains
        #  one, e.g. a tuple of iterators.
        if any(
            type_info.primitive_constituents(symbol_type)
            .if_isinstance(ct.FieldType)
            .filter(is_local_kind)
        ):
            return ITIRTypeKind.ENCAPSULATED_ITERATOR
        # otherwise we get an iterator, e.g. an iterator of values or tuples
        return ITIRTypeKind.ITERATOR
    return ITIRTypeKind.VALUE


def is_expr_with_iterator_type_kind(it_type_kind: ITIRTypeKind) -> Callable[[foast.Expr], bool]:
    def predicate(node: foast.Expr):
        return iterator_type_kind(node.type) is it_type_kind

    return predicate


class StmtReturnKind(enum.IntEnum):
    UNCONDIOTIONAL_RETURN = 0
    CONDITIONAL_RETURN = 1
    NO_RETURN = 2


def deduce_return_kind(node: foast.Stmt) -> StmtReturnKind:
    """
    Deduce if a statement returns and if so, whether it does unconditionally.

    Example with ``StmtReturnKind.UNCONDIOTIONAL_RETURN``
    -----------------------------------------
    .. code-block:: python

        if cond:
          return 1
        else:
          return 2

    Example with ``StmtReturnKind.NO_RETURN``
    -----------------------------------------
    .. code-block:: python

        if cond:
          result = 1
        else:
          result = 2

    Example with ``StmtReturnKind.CONDITIONAL_RETURN``
    -----------------------------------------
    .. code-block:: python

        if cond:
          return 1
        else:
          result = 2
    """
    if isinstance(node, foast.IfStmt):
        return_kinds = (deduce_return_kind(node.true_branch), deduce_return_kind(node.false_branch))
        if all(return_kind is StmtReturnKind.UNCONDIOTIONAL_RETURN for return_kind in return_kinds):
            return StmtReturnKind.UNCONDIOTIONAL_RETURN
        elif any(
            return_kind in (StmtReturnKind.UNCONDIOTIONAL_RETURN, StmtReturnKind.CONDITIONAL_RETURN)
            for return_kind in return_kinds
        ):
            return StmtReturnKind.CONDITIONAL_RETURN
        assert all(return_kind is StmtReturnKind.NO_RETURN for return_kind in return_kinds)
        return StmtReturnKind.NO_RETURN
    elif isinstance(node, foast.Return):
        return StmtReturnKind.UNCONDIOTIONAL_RETURN
    elif isinstance(node, foast.BlockStmt):
        for stmt in node.stmts:
            return_kind = deduce_return_kind(stmt)
            if return_kind != StmtReturnKind.NO_RETURN:
                return return_kind
        return StmtReturnKind.NO_RETURN
    elif isinstance(node, (foast.Assign, foast.TupleTargetAssign)):
        return StmtReturnKind.NO_RETURN
    else:
        raise AssertionError(f"Statements of type `{type(node).__name__} not understood.`")


def to_value(node_or_type: foast.Expr | ct.DataType) -> Callable[[itir.Expr], itir.Expr]:
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
    type_ = node_or_type.type if isinstance(node_or_type, foast.LocatedNode) else node_or_type
    if iterator_type_kind(type_) is ITIRTypeKind.ITERATOR:
        # just to ensure we don't accidentally deref a local field
        assert not (isinstance(type_, ct.FieldType) and is_local_kind(type_))
        return im.deref_
    elif iterator_type_kind(type_) is ITIRTypeKind.VALUE:
        return lambda x: x

    raise AssertionError(f"Type {type_} can not be turned into a value.")


@dataclass
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

    uid_generator: UIDGenerator

    class lifted_lambda:
        def __init__(self, *params):
            self.params = params

        def __call__(self, expr):
            return im.lift_(im.lambda__(*self.params)(expr))(*self.params)

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> itir.Expr:
        return cls(uid_generator=UIDGenerator()).visit(node)

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

        assert isinstance(node.type, ct.FieldOperatorType)
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
        symtable = node.annex.symtable
        params = self.visit(node.params, symtable=symtable)
        return itir.FunctionDefinition(
            id=node.id,
            params=params,
            expr=self.visit_BlockStmt(node.body, inner_expr=None, symtable=symtable),
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
            *(itir.SymRef(id=param.id) for param in definition.params[1:])
        )

        return itir.FunctionDefinition(
            id=node.id,
            params=definition.params[1:],
            expr=body,
        )

    def visit_Stmt(self, node: foast.Stmt, **kwargs):
        raise AssertionError("Statements must always be visited in the context of a function.")

    def visit_Return(self, node: foast.Return, *, inner_expr: itir.Expr | None, **kwargs) -> itir.Expr:
        return self.visit(node.value, **kwargs)

    def visit_BlockStmt(
        self, node: foast.BlockStmt, *, inner_expr: itir.Expr | None, **kwargs
    ) -> itir.Expr:
        for stmt in reversed(node.stmts):
            inner_expr = self.visit(stmt, inner_expr=inner_expr, **kwargs)
        assert inner_expr
        return inner_expr

    def visit_Assign(self, node: foast.Assign, *, inner_expr: itir.Expr | None, **kwargs) -> itir.Expr:
        return im.let(self.visit(node.target, **kwargs), self.visit(node.value, **kwargs))(
            inner_expr
        )

    def visit_IfStmt(self, node: foast.IfStmt, *, inner_expr: itir.Expr | None, **kwargs) -> itir.Expr:
        cond = self.visit(node.condition, **kwargs)

        return_kind = deduce_return_kind(node)

        common_symbols: dict[str, foast.Symbol] = node.annex.propagated_symbols

        if return_kind is StmtReturnKind.NO_RETURN:
            # pack the common symbols into a tuple
            common_symrefs = im.make_tuple_(*(im.ref(sym) for sym in common_symbols.keys()))

            # apply both branches and extract the common symbols through the prepared tuple
            true_branch = self.visit(node.true_branch, inner_expr=common_symrefs, **kwargs)
            false_branch = self.visit(node.false_branch, inner_expr=common_symrefs, **kwargs)

            # unpack the common symbols' tuple for `inner_expr`
            for i, sym in enumerate(common_symbols.keys()):
                inner_expr = im.let(sym, im.tuple_get_(i, im.ref("__if_stmt_result")))(inner_expr)

            # here we assume neither branch returns
            return im.let("__if_stmt_result", im.if_(cond, true_branch, false_branch))(inner_expr)
        elif return_kind is StmtReturnKind.CONDITIONAL_RETURN:
            common_syms = tuple(im.sym(sym) for sym in common_symbols.keys())
            common_symrefs = tuple(im.ref(sym) for sym in common_symbols.keys())

            # wrap the inner expression in a lambda function. note that this increases the
            # operation count if both branches are evaluated.
            inner_expr_name = self.uid_generator.sequential_id(prefix="__inner_expr")
            inner_expr_evaluator = im.lambda__(*common_syms)(inner_expr)
            inner_expr = im.call_(inner_expr_name)(*common_symrefs)

            true_branch = self.visit(node.true_branch, inner_expr=inner_expr, **kwargs)
            false_branch = self.visit(node.false_branch, inner_expr=inner_expr, **kwargs)

            return im.let(inner_expr_name, inner_expr_evaluator)(
                im.if_(cond, true_branch, false_branch)
            )

        assert return_kind is StmtReturnKind.UNCONDIOTIONAL_RETURN

        # note that we do not duplicate `inner_expr` here since if both branches
        #  return, `inner_expr` is ignored.
        true_branch = self.visit(node.true_branch, inner_expr=inner_expr, **kwargs)
        false_branch = self.visit(node.false_branch, inner_expr=inner_expr, **kwargs)

        return im.if_(cond, true_branch, false_branch)

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> itir.Sym:
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.SymRef:
        return im.ref(node.id)

    def _lift_lambda(self, node: foast.LocatedNode):
        if any(
            node.pre_walk_values()
            .if_isinstance(foast.Name)
            .filter(is_expr_with_iterator_type_kind(ITIRTypeKind.ENCAPSULATED_ITERATOR))
        ):
            raise NotImplementedError(
                "Using composite types (e.g. tuples) containing local fields not supported."
            )
        param_names = (
            node.pre_walk_values()
            .if_isinstance(foast.Name)
            .filter(is_expr_with_iterator_type_kind(ITIRTypeKind.ITERATOR))
            .getattr("id")
            .unique()
            .to_list()
        )
        return self.lifted_lambda(*param_names)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        value = self.visit(node.value, **kwargs)
        if iterator_type_kind(node.value.type) is ITIRTypeKind.ITERATOR:
            return self._lift_lambda(node)(im.tuple_get_(node.index, im.deref_(value)))
        elif iterator_type_kind(node.value.type) in (
            ITIRTypeKind.VALUE,
            ITIRTypeKind.ENCAPSULATED_ITERATOR,
        ):
            return im.tuple_get_(node.index, value)
        raise AssertionError("Unexpected `IteratorTypeKind`.")

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        # it is important to use `node` here instead of `el` to decide if we
        #  want to have a value. As soon as we have one local field in the
        #  expression we choose a tuple of iterators layout (which other
        #  parts of the lowering rely on).
        if iterator_type_kind(node.type) is ITIRTypeKind.ITERATOR:
            elts = tuple(to_value(el)(self.visit(el, **kwargs)) for el in node.elts)
            return self._lift_lambda(node)(im.make_tuple_(*elts))
        elif iterator_type_kind(node.type) in (
            ITIRTypeKind.VALUE,
            ITIRTypeKind.ENCAPSULATED_ITERATOR,
        ):
            elts = tuple(self.visit(el, **kwargs) for el in node.elts)
            return im.make_tuple_(*elts)
        raise AssertionError("Unexpected `IteratorTypeKind`.")

    def _lift_if_field(self, node: foast.Expr) -> Callable[[itir.FunCall], itir.FunCall]:
        if iterator_type_kind(node.type) is ITIRTypeKind.VALUE:
            return lambda x: x
        elif iterator_type_kind(node.type) is ITIRTypeKind.ITERATOR:
            return self._lift_lambda(node)
        raise AssertionError("Unexpected `IteratorTypeKind`.")

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        if node.op in [ct.UnaryOperator.NOT, ct.UnaryOperator.INVERT]:
            return self._lift_if_field(node)(
                im.call_("not_")(to_value(node.operand)(self.visit(node.operand, **kwargs)))
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

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> itir.FunCall:
        lowered_node_cond = self.visit(node.condition, **kwargs)
        lowered_true_expr = self.visit(node.true_expr, **kwargs)
        lowered_false_expr = self.visit(node.false_expr, **kwargs)

        return self._lift_if_field(node)(
            im.if_(
                lowered_node_cond,
                to_value(node.true_expr)(lowered_true_expr),
                to_value(node.false_expr)(lowered_false_expr),
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
        lowering = InsideReductionLowering(uid_generator=self.uid_generator)
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

    def _visit_min_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        init_expr = itir.Literal(value=str(np.finfo(np.float64).max), type="float64")
        return self._make_reduction_expr(
            node,
            lambda expr: im.call_("if_")(im.less_("acc", expr), "acc", expr),
            init_expr,
            **kwargs,
        )

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
            lowered_args = []
            for arg in node.args:
                lowered_arg = self.visit(arg, **kwargs)
                if iterator_type_kind(arg.type) == ITIRTypeKind.VALUE:
                    lowered_arg = im.lift_(im.lambda__()(lowered_arg))()
                lowered_args.append(lowered_arg)

            return self._lift_if_field(node)(im.call_(lowered_func)(*lowered_args))

        raise AssertionError(
            f"Call to object of type {type(node.func.type).__name__} not understood."
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

    def _visit_math_built_in(self, node: foast.Call, **kwargs) -> itir.FunCall:
        args = tuple(to_value(arg)(self.visit(arg, **kwargs)) for arg in node.args)
        return self._lift_if_field(node)(im.call_(self.visit(node.func, **kwargs))(*args))

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._visit_reduce(node, **kwargs)

    def _visit_type_constr(self, node: foast.Call, **kwargs) -> itir.FunCall:
        if isinstance(node.args[0], foast.Constant):
            node_kind = self.visit(node.type).kind.name.lower()
            target_type = fbuiltins.BUILTINS[node_kind]
            source_type = {**fbuiltins.BUILTINS, "string": str}[node.args[0].type.__str__().lower()]
            if target_type is bool and source_type is not bool:
                return im.literal_(str(bool(source_type(node.args[0].value))), node.func.id)
            return im.literal_(str(node.args[0].value), node_kind)
        raise FieldOperatorLoweringError(f"Encountered a type cast, which is not supported: {node}")

    def _make_literal(self, val: Any, type_: ct.SymbolType) -> itir.Literal:
        # TODO(tehrengruber): check constant of this type is supported in iterator ir
        if isinstance(type_, ct.TupleType):
            return im.make_tuple_(
                *(self._make_literal(val, type_) for val, type_ in zip(val, type_.types))
            )
        elif isinstance(type_, ct.ScalarType):
            typename = type_.kind.name.lower()
            return im.literal_(str(val), typename)
        raise ValueError(f"Unsupported literal type {type_}.")

    def visit_Constant(self, node: foast.Constant, **kwargs) -> itir.Literal:
        return self._make_literal(node.value, node.type)


@dataclass
class InsideReductionLowering(FieldOperatorLowering):
    """Variant of the lowering with special rules for inside reductions."""

    lambda_params: dict[str, itir.Expr] = field(default_factory=lambda: {})
    __counter: itertools.count = field(default_factory=lambda: itertools.count())

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.SymRef:
        uid = f"{node.id}__{self._sequential_id()}"
        if iterator_type_kind(node.type) is ITIRTypeKind.ENCAPSULATED_ITERATOR:
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

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> itir.FunCall:
        return im.call_("if_")(
            self.visit(node.condition, **kwargs),
            self.visit(node.true_expr, **kwargs),
            self.visit(node.false_expr, **kwargs),
        )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        if node.op is ct.UnaryOperator.NOT:
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
