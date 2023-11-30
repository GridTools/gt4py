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

import dataclasses
from typing import Any, Callable, Optional

from gt4py.eve import NodeTranslator
from gt4py.eve.utils import UIDGenerator
from gt4py.next.ffront import (
    dialect_ast_enums,
    fbuiltins,
    field_operator_ast as foast,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.fbuiltins import FUN_BUILTIN_NAMES, MATH_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from gt4py.next.ffront.foast_introspection import StmtReturnKind, deduce_stmt_return_kind
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_info, type_specifications as ts


def promote_to_list(
    node: foast.Symbol | foast.Expr,
) -> Callable[[itir.Expr], itir.Expr]:
    if not type_info.contains_local_field(node.type):
        return lambda x: im.promote_to_lifted_stencil("make_const_list")(x)
    return lambda x: x


@dataclasses.dataclass
class FieldOperatorLowering(NodeTranslator):
    """
    Lower FieldOperator AST (FOAST) to Iterator IR (ITIR).

    The strategy is to lower every expression to lifted stencils,
    i.e. taking iterators and returning iterator.

    Examples
    --------
    >>> from gt4py.next.ffront.func_to_foast import FieldOperatorParser
    >>> from gt4py.next import Field, Dimension, float64
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
    [Sym(id=SymbolName('inp'), kind='Iterator', dtype=('float64', False))]
    """

    uid_generator: UIDGenerator = dataclasses.field(default_factory=UIDGenerator)

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> itir.Expr:
        return cls().visit(node)

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs
    ) -> itir.FunctionDefinition:
        params = self.visit(node.params)
        return itir.FunctionDefinition(
            id=node.id,
            params=params,
            expr=self.visit_BlockStmt(node.body, inner_expr=None),
        )  # `expr` is a lifted stencil

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> itir.FunctionDefinition:
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)
        new_body = im.deref(func_definition.expr)

        return itir.FunctionDefinition(
            id=func_definition.id,
            params=func_definition.params,
            expr=new_body,
        )

    def visit_ScanOperator(self, node: foast.ScanOperator, **kwargs) -> itir.FunctionDefinition:
        # note: we don't need the axis here as this is handled by the program
        #  decorator

        # We are lowering node.forward and node.init to iterators, but here we expect values -> `deref`.
        # In iterator IR we didn't properly specify if this is legal,
        # however after lift-inlining the expressions are transformed back to literals.
        forward = im.deref(self.visit(node.forward, **kwargs))
        init = im.deref(self.visit(node.init, **kwargs))

        # lower definition function
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)
        new_body = func_definition.expr

        # promote carry to iterator
        # (this is the only place in the lowering were a variable is captured in a lifted lambda)
        new_body = im.let(
            func_definition.params[0].id,
            im.promote_to_const_iterator(func_definition.params[0].id),
        )(im.deref(new_body))
        definition = itir.Lambda(params=func_definition.params, expr=new_body)
        body = im.call(im.call("scan")(definition, forward, init))(
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
        self, node: foast.Return, *, inner_expr: Optional[itir.Expr], **kwargs
    ) -> itir.Expr:
        return self.visit(node.value, **kwargs)

    def visit_BlockStmt(
        self, node: foast.BlockStmt, *, inner_expr: Optional[itir.Expr], **kwargs
    ) -> itir.Expr:
        for stmt in reversed(node.stmts):
            inner_expr = self.visit(stmt, inner_expr=inner_expr, **kwargs)
        assert inner_expr
        return inner_expr

    def visit_IfStmt(
        self, node: foast.IfStmt, *, inner_expr: Optional[itir.Expr], **kwargs
    ) -> itir.Expr:
        # the lowered if call doesn't need to be lifted as the condition can only originate
        #  from a scalar value (and not a field)
        assert (
            isinstance(node.condition.type, ts.ScalarType)
            and node.condition.type.kind == ts.ScalarKind.BOOL
        )

        cond = self.visit(node.condition, **kwargs)

        return_kind: StmtReturnKind = deduce_stmt_return_kind(node)

        common_symbols: dict[str, foast.Symbol] = node.annex.propagated_symbols

        if return_kind is StmtReturnKind.NO_RETURN:
            # pack the common symbols into a tuple
            common_symrefs = im.make_tuple(*(im.ref(sym) for sym in common_symbols.keys()))

            # apply both branches and extract the common symbols through the prepared tuple
            true_branch = self.visit(node.true_branch, inner_expr=common_symrefs, **kwargs)
            false_branch = self.visit(node.false_branch, inner_expr=common_symrefs, **kwargs)

            # unpack the common symbols' tuple for `inner_expr`
            for i, sym in enumerate(common_symbols.keys()):
                inner_expr = im.let(sym, im.tuple_get(i, im.ref("__if_stmt_result")))(inner_expr)

            # here we assume neither branch returns
            return im.let("__if_stmt_result", im.if_(im.deref(cond), true_branch, false_branch))(
                inner_expr
            )
        elif return_kind is StmtReturnKind.CONDITIONAL_RETURN:
            common_syms = tuple(im.sym(sym) for sym in common_symbols.keys())
            common_symrefs = tuple(im.ref(sym) for sym in common_symbols.keys())

            # wrap the inner expression in a lambda function. note that this increases the
            # operation count if both branches are evaluated.
            inner_expr_name = self.uid_generator.sequential_id(prefix="__inner_expr")
            inner_expr_evaluator = im.lambda_(*common_syms)(inner_expr)
            inner_expr = im.call(inner_expr_name)(*common_symrefs)

            true_branch = self.visit(node.true_branch, inner_expr=inner_expr, **kwargs)
            false_branch = self.visit(node.false_branch, inner_expr=inner_expr, **kwargs)

            return im.let(inner_expr_name, inner_expr_evaluator)(
                im.if_(im.deref(cond), true_branch, false_branch)
            )

        assert return_kind is StmtReturnKind.UNCONDITIONAL_RETURN

        # note that we do not duplicate `inner_expr` here since if both branches
        #  return, `inner_expr` is ignored.
        true_branch = self.visit(node.true_branch, inner_expr=inner_expr, **kwargs)
        false_branch = self.visit(node.false_branch, inner_expr=inner_expr, **kwargs)

        return im.if_(im.deref(cond), true_branch, false_branch)

    def visit_Assign(
        self, node: foast.Assign, *, inner_expr: Optional[itir.Expr], **kwargs
    ) -> itir.Expr:
        return im.let(self.visit(node.target, **kwargs), self.visit(node.value, **kwargs))(
            inner_expr
        )

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> itir.Sym:
        # TODO(tehrengruber): extend to more types
        if isinstance(node.type, ts.FieldType):
            kind = "Iterator"
            dtype = node.type.dtype.kind.name.lower()
            is_list = type_info.is_local_field(node.type)
            return itir.Sym(id=node.id, kind=kind, dtype=(dtype, is_list))
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.SymRef:
        return im.ref(node.id)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.Expr:
        return im.promote_to_lifted_stencil(lambda tuple_: im.tuple_get(node.index, tuple_))(
            self.visit(node.value, **kwargs)
        )

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.Expr:
        return im.promote_to_lifted_stencil(lambda *elts: im.make_tuple(*elts))(
            *[self.visit(el, **kwargs) for el in node.elts],
        )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.Expr:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        dtype = type_info.extract_dtype(node.type)
        if node.op in [dialect_ast_enums.UnaryOperator.NOT, dialect_ast_enums.UnaryOperator.INVERT]:
            if dtype.kind != ts.ScalarKind.BOOL:
                raise NotImplementedError(f"{node.op} is only supported on `bool`s.")
            return self._map("not_", node.operand)

        return self._map(
            node.op.value,
            foast.Constant(value="0", type=dtype, location=node.location),
            node.operand,
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> itir.FunCall:
        return self._map("if_", node.condition, node.true_expr, node.false_expr)

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.Expr:
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                shift_offset = im.shift(offset_name, offset_index)
            case foast.Name(id=offset_name):
                return im.lifted_neighbors(str(offset_name), self.visit(node.func, **kwargs))
            case foast.Call(func=foast.Name(id="as_offset")):
                func_args = node.args[0]
                offset_dim = func_args.args[0]
                assert isinstance(offset_dim, foast.Name)
                shift_offset = im.shift(
                    offset_dim.id, im.deref(self.visit(func_args.args[1], **kwargs))
                )
            case _:
                raise FieldOperatorLoweringError("Unexpected shift arguments!")
        return im.lift(im.lambda_("it")(im.deref(shift_offset("it"))))(
            self.visit(node.func, **kwargs)
        )

    def visit_Call(self, node: foast.Call, **kwargs) -> itir.Expr:
        if type_info.type_class(node.func.type) is ts.FieldType:
            return self._visit_shift(node, **kwargs)
        elif isinstance(node.func, foast.Name) and node.func.id in MATH_BUILTIN_NAMES:
            return self._visit_math_built_in(node, **kwargs)
        elif isinstance(node.func, foast.Name) and node.func.id in FUN_BUILTIN_NAMES:
            visitor = getattr(self, f"_visit_{node.func.id}")
            return visitor(node, **kwargs)
        elif isinstance(node.func, foast.Name) and node.func.id in TYPE_BUILTIN_NAMES:
            return self._visit_type_constr(node, **kwargs)
        elif isinstance(
            node.func.type,
            (
                ts_ffront.FieldOperatorType,
                ts_ffront.ScanOperatorType,
            ),
        ):
            # Operators are lowered into lifted stencils.
            lowered_func = self.visit(node.func, **kwargs)
            # ITIR has no support for keyword arguments. Instead, we concatenate both positional
            # and keyword arguments and use the unique order as given in the function signature.
            lowered_args, lowered_kwargs = type_info.canonicalize_arguments(
                node.func.type,
                [self.visit(arg, **kwargs) for arg in node.args],
                {name: self.visit(arg, **kwargs) for name, arg in node.kwargs.items()},
                use_signature_ordering=True,
            )
            call_args = [f"__arg{i}" for i in range(len(lowered_args))]
            call_kwargs = [f"__kwarg_{name}" for name in lowered_kwargs.keys()]
            return im.lift(
                im.lambda_(*call_args, *call_kwargs)(
                    im.call(lowered_func)(*call_args, *call_kwargs)
                )
            )(*lowered_args, *lowered_kwargs.values())
        elif isinstance(node.func.type, ts.FunctionType):
            # ITIR has no support for keyword arguments. Instead, we concatenate both positional
            # and keyword arguments and use the unique order as given in the function signature.
            lowered_args, lowered_kwargs = type_info.canonicalize_arguments(
                node.func.type,
                self.visit(node.args, **kwargs),
                self.visit(node.kwargs, **kwargs),
                use_signature_ordering=True,
            )
            return im.call(self.visit(node.func, **kwargs))(*lowered_args, *lowered_kwargs.values())

        raise AssertionError(
            f"Call to object of type {type(node.func.type).__name__} not understood."
        )

    def _visit_astype(self, node: foast.Call, **kwargs) -> itir.FunCall:
        assert len(node.args) == 2 and isinstance(node.args[1], foast.Name)
        obj, new_type = node.args[0], node.args[1].id
        return self._process_elements(
            lambda x: im.call("cast_")(x, str(new_type)), obj, obj.type, **kwargs
        )

    def _visit_where(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._map("if_", *node.args)

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self.visit(node.args[0], **kwargs)

    def _visit_math_built_in(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return self._map(self.visit(node.func, **kwargs), *node.args)

    def _make_reduction_expr(
        self,
        node: foast.Call,
        op: str | itir.SymRef,
        init_expr: itir.Expr,
        **kwargs,
    ):
        # TODO(havogt): deal with nested reductions of the form neighbor_sum(neighbor_sum(field(off1)(off2)))
        it = self.visit(node.args[0], **kwargs)
        assert isinstance(node.kwargs["axis"].type, ts.DimensionType)
        val = im.call(im.call("reduce")(op, im.deref(init_expr)))
        return im.promote_to_lifted_stencil(val)(it)

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> itir.FunCall:
        dtype = type_info.extract_dtype(node.type)
        return self._make_reduction_expr(node, "plus", self._make_literal("0", dtype), **kwargs)

    def _visit_max_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        dtype = type_info.extract_dtype(node.type)
        min_value, _ = type_info.arithmetic_bounds(dtype)
        init_expr = self._make_literal(str(min_value), dtype)
        return self._make_reduction_expr(node, "maximum", init_expr, **kwargs)

    def _visit_min_over(self, node: foast.Call, **kwargs) -> itir.FunCall:
        dtype = type_info.extract_dtype(node.type)
        _, max_value = type_info.arithmetic_bounds(dtype)
        init_expr = self._make_literal(str(max_value), dtype)
        return self._make_reduction_expr(node, "minimum", init_expr, **kwargs)

    def _visit_type_constr(self, node: foast.Call, **kwargs) -> itir.Expr:
        if isinstance(node.args[0], foast.Constant):
            node_kind = self.visit(node.type).kind.name.lower()
            target_type = fbuiltins.BUILTINS[node_kind]
            source_type = {**fbuiltins.BUILTINS, "string": str}[node.args[0].type.__str__().lower()]
            if target_type is bool and source_type is not bool:
                return im.promote_to_const_iterator(
                    im.literal(str(bool(source_type(node.args[0].value))), "bool")
                )
            return im.promote_to_const_iterator(im.literal(str(node.args[0].value), node_kind))
        raise FieldOperatorLoweringError(f"Encountered a type cast, which is not supported: {node}")

    def _make_literal(self, val: Any, type_: ts.TypeSpec) -> itir.Expr:
        # TODO(havogt): lifted nullary lambdas are not supported in iterator.embedded due to an implementation detail;
        # the following constructs work if they are removed by inlining.
        if isinstance(type_, ts.TupleType):
            return im.promote_to_const_iterator(
                im.make_tuple(
                    *(
                        im.deref(self._make_literal(val, type_))
                        for val, type_ in zip(val, type_.types)
                    )
                )
            )
        elif isinstance(type_, ts.ScalarType):
            typename = type_.kind.name.lower()
            return im.promote_to_const_iterator(im.literal(str(val), typename))
        raise ValueError(f"Unsupported literal type {type_}.")

    def visit_Constant(self, node: foast.Constant, **kwargs) -> itir.Expr:
        return self._make_literal(node.value, node.type)

    def _map(self, op, *args, **kwargs):
        lowered_args = [self.visit(arg, **kwargs) for arg in args]
        if any(type_info.contains_local_field(arg.type) for arg in args):
            lowered_args = [promote_to_list(arg)(larg) for arg, larg in zip(args, lowered_args)]
            op = im.call("map_")(op)

        return im.promote_to_lifted_stencil(im.call(op))(*lowered_args)

    def _process_elements(
        self,
        process_func: Callable[[itir.Expr], itir.Expr],
        obj: foast.Expr,
        current_el_type: ts.TypeSpec,
        current_el_expr: itir.Expr = im.ref("expr"),
    ):
        """Recursively applies a processing function to all primitive constituents of a tuple."""
        if isinstance(current_el_type, ts.TupleType):
            # TODO(ninaburg): Refactor to avoid duplicating lowered obj expression for each tuple element.
            return im.promote_to_lifted_stencil(lambda *elts: im.make_tuple(*elts))(
                *[
                    self._process_elements(
                        process_func,
                        obj,
                        current_el_type.types[i],
                        im.tuple_get(i, current_el_expr),
                    )
                    for i in range(len(current_el_type.types))
                ]
            )
        elif type_info.contains_local_field(current_el_type):
            raise NotImplementedError("Processing fields with local dimension is not implemented.")
        else:
            return self._map(im.lambda_("expr")(process_func(current_el_expr)), obj)


class FieldOperatorLoweringError(Exception):
    ...
