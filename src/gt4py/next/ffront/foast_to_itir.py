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
import functools
from typing import Any, Callable, Optional

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.eve.extended_typing import Never
from gt4py.eve.utils import UIDGenerator
from gt4py.next import DimensionKind, common
from gt4py.next.ffront import (
    dialect_ast_enums,
    fbuiltins,
    field_operator_ast as foast,
    lowering_utils,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.experimental import EXPERIMENTAL_FUN_BUILTIN_NAMES
from gt4py.next.ffront.fbuiltins import FUN_BUILTIN_NAMES, MATH_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from gt4py.next.ffront.foast_introspection import StmtReturnKind, deduce_stmt_return_kind
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_info, type_specifications as ts


def promote_to_list(
    node: foast.Symbol | foast.Expr,
    lowered_arg: itir.Expr,
    domain: itir.Expr  # TODO: just compute the domain here
) -> itir.Expr:
    if not type_info.contains_local_field(node.type):
        return im.apply_stencil(
            im.lambda_("it")(im.call("make_const_list")(im.deref("it"))),
            im.make_tuple(lowered_arg),
            domain
        )
    return lowered_arg


@dataclasses.dataclass
class FieldOperatorLowering(PreserveLocationVisitor, NodeTranslator):
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
    ...     return inp
    >>>
    >>> parsed = FieldOperatorParser.apply_to_function(fieldop)
    >>> lowered = FieldOperatorLowering.apply(parsed)
    >>> type(lowered)
    <class 'gt4py.next.iterator.ir.FunctionDefinition'>
    >>> lowered.id
    SymbolName('fieldop')
    >>> lowered.params  # doctest: +ELLIPSIS
    [Sym(id=SymbolName('inp'), kind='Iterator', dtype=('float64', False))]
    """

    uid_generator: UIDGenerator = dataclasses.field(default_factory=UIDGenerator)

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> itir.Expr:
        return cls().visit(node)

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> itir.FunctionDefinition:
        params = self.visit(node.params)
        return itir.FunctionDefinition(
            id=node.id,
            params=params,
            expr=self.visit_BlockStmt(node.body, inner_expr=None),
        )  # `expr` is a lifted stencil

    def visit_FieldOperator(
        self, node: foast.FieldOperator, **kwargs: Any
    ) -> itir.FunctionDefinition:
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)

        new_body = func_definition.expr

        return itir.FunctionDefinition(
            id=func_definition.id,
            params=func_definition.params,
            expr=new_body,
        )

    def visit_ScanOperator(
        self, node: foast.ScanOperator, **kwargs: Any
    ) -> itir.FunctionDefinition:
        # note: we don't need the axis here as this is handled by the program
        #  decorator
        assert isinstance(node.type, ts_ffront.ScanOperatorType)

        # We are lowering node.forward and node.init to iterators, but here we expect values -> `deref`.
        # In iterator IR we didn't properly specify if this is legal,
        # however after lift-inlining the expressions are transformed back to literals.
        forward = self.visit(node.forward, **kwargs)
        init = self.visit(node.init, **kwargs)

        # lower definition function
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)
        new_body = func_definition.expr

        stencil_args: list[itir.Expr] = []
        assert not node.type.definition.pos_only_args and not node.type.definition.kw_only_args
        for param, arg_type in zip(
            func_definition.params[1:],
            [*node.type.definition.pos_or_kw_args.values()][1:],
            strict=True,
        ):
            # convert iterator to value
            new_body = im.let(
                param.id,
                im.deref(param.id),
            )(new_body)
            stencil_args.append(im.ref(param.id))

        stencil_args = [param.id for param in func_definition.params[1:]]

        definition = itir.Lambda(params=func_definition.params, expr=new_body)

        body = im.apply_stencil(
            im.call("scan")(definition, forward, init),
            im.make_tuple(*stencil_args),
            im.call("broadcast_to_common_domain")(*stencil_args)
            #im.call("intersect")(*[im.call("get_domain")(arg) for arg in stencil_args]) # doesn't work as for program we are to late with broadcasting in the call see test_tuple_scalar_scan
        )

        return itir.FunctionDefinition(
            id=node.id,
            params=definition.params[1:],
            expr=body,
        )

    def visit_Stmt(self, node: foast.Stmt, **kwargs: Any) -> Never:
        raise AssertionError("Statements must always be visited in the context of a function.")

    def visit_Return(
        self, node: foast.Return, *, inner_expr: Optional[itir.Expr], **kwargs: Any
    ) -> itir.Expr:
        return self.visit(node.value, **kwargs)

    def visit_BlockStmt(
        self, node: foast.BlockStmt, *, inner_expr: Optional[itir.Expr], **kwargs: Any
    ) -> itir.Expr:
        for stmt in reversed(node.stmts):
            inner_expr = self.visit(stmt, inner_expr=inner_expr, **kwargs)
        assert inner_expr
        return inner_expr

    def visit_IfStmt(
        self, node: foast.IfStmt, *, inner_expr: Optional[itir.Expr], **kwargs: Any
    ) -> itir.Expr:
        # the lowered if call doesn't need to be lifted as the condition can only originate
        # from a scalar value (and not a field)
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
        self, node: foast.Assign, *, inner_expr: Optional[itir.Expr], **kwargs: Any
    ) -> itir.Expr:
        return im.let(self.visit(node.target, **kwargs), self.visit(node.value, **kwargs))(
            inner_expr
        )

    def visit_Symbol(self, node: foast.Symbol, **kwargs: Any) -> itir.Sym:
        # TODO(tehrengruber): extend to more types
        if isinstance(node.type, ts.FieldType):
            kind = "Iterator"
            dtype = node.type.dtype.kind.name.lower()
            is_list = type_info.is_local_field(node.type)
            return itir.Sym(id=node.id, kind=kind, dtype=(dtype, is_list))
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs: Any) -> itir.SymRef:
        return im.ref(node.id)

    def visit_Subscript(self, node: foast.Subscript, **kwargs: Any) -> itir.Expr:
        return im.tuple_get(node.index, self.visit(node.value, **kwargs))

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs: Any) -> itir.Expr:
        return im.make_tuple(*[self.visit(el, **kwargs) for el in node.elts])

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs: Any) -> itir.Expr:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        dtype = type_info.extract_dtype(node.type)
        if node.op in [
            dialect_ast_enums.UnaryOperator.NOT,
            dialect_ast_enums.UnaryOperator.INVERT,
        ]:
            if dtype.kind != ts.ScalarKind.BOOL:
                raise NotImplementedError(f"'{node.op}' is only supported on 'bool' arguments.")
            return self._map("not_", node.operand)

        return self._map(
            node.op.value,
            foast.Constant(value="0", type=dtype, location=node.location),
            node.operand,
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs: Any) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs: Any) -> itir.FunCall:
        # TODO: if we would support if statements outside of apply_stencil calls in ITIR this
        #  would go away and we could do
        return im.if_(*self.visit([node.condition, node.true_expr, node.false_expr]))
        # op = "if_"
        # args = node.args
        # lowered_args: list[itir.Expr] = [
        #     lowering_utils.to_iterator_of_tuples(self.visit(arg, **kwargs), arg.type)
        #     for arg in args
        # ]
        # if any(type_info.contains_local_field(arg.type) for arg in args):
        #     lowered_args = [promote_to_list(arg)(larg) for arg, larg in zip(args, lowered_args)]
        #     op = im.call("map_")(op)
        #
        # return lowering_utils.to_tuples_of_iterator(
        #     im.promote_to_lifted_stencil(im.call(op))(*lowered_args), node.type
        # )

    def visit_Compare(self, node: foast.Compare, **kwargs: Any) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def _visit_shift(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                shift_args = (itir.OffsetLiteral(value=offset_name), offset_index)
            case foast.Name(id=offset_name):
                lowered_field_expr = self.visit(node.func, **kwargs)
                return im.apply_stencil(
                    im.lambda_("it")(im.neighbors(offset_name, "it")),
                    im.make_tuple(lowered_field_expr),
                    im.call("broadcast_domain")(
                        im.call("domain")(),  # inverse translate domain would make more sense, but we need to inline this anyway
                        im.make_tuple(*(itir.AxisLiteral(value=dim.value) for dim in node.type.dims))
                    )
                )
            case foast.Call(func=foast.Name(id="as_offset")):
                func_args = node.args[0]
                offset_dim = func_args.args[0]
                assert isinstance(offset_dim, foast.Name)
                shift_args = (
                    itir.OffsetLiteral(value=offset_dim.id), im.deref(self.visit(func_args.args[1], **kwargs))
                )
            case _:
                raise FieldOperatorLoweringError("Unexpected shift arguments!")

        lowered_field_expr = self.visit(node.func, **kwargs)

        domain = im.call("inverse_translate_domain")(im.call("get_domain")(lowered_field_expr), *shift_args)

        return im.apply_stencil(
            im.lambda_("it")(im.deref(im.shift(*shift_args)("it"))),
            im.make_tuple(lowered_field_expr),
            domain
            # TODO: condition
        )

    def visit_Call(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        if type_info.type_class(node.func.type) is ts.FieldType:
            return self._visit_shift(node, **kwargs)
        elif isinstance(node.func, foast.Name) and node.func.id in MATH_BUILTIN_NAMES:
            return self._visit_math_built_in(node, **kwargs)
        elif isinstance(node.func, foast.Name) and node.func.id in (
            FUN_BUILTIN_NAMES + EXPERIMENTAL_FUN_BUILTIN_NAMES
        ):
            visitor = getattr(self, f"_visit_{node.func.id}")
            return visitor(node, **kwargs)
        elif isinstance(node.func, foast.Name) and node.func.id in TYPE_BUILTIN_NAMES:
            return self._visit_type_constr(node, **kwargs)
        elif isinstance(
            node.func.type,
            (
                ts.FunctionType,
                ts_ffront.FieldOperatorType,
                ts_ffront.ScanOperatorType,
            ),
        ):
            # ITIR has no support for keyword arguments. Instead, we concatenate both positional
            # and keyword arguments and use the unique order as given in the function signature.
            lowered_args, lowered_kwargs = type_info.canonicalize_arguments(
                node.func.type,
                self.visit(node.args, **kwargs),
                self.visit(node.kwargs, **kwargs),
                use_signature_ordering=True,
            )

            result = im.call(self.visit(node.func, **kwargs))(
                *lowered_args, *lowered_kwargs.values()
            )

            # scan operators return an iterator of tuples, transform into tuples of iterator again
            #if isinstance(node.func.type, ts_ffront.ScanOperatorType):
            #    result = lowering_utils.to_tuples_of_iterator(
            #        result, node.func.type.definition.returns
            #    )

            return result

        raise AssertionError(
            f"Call to object of type '{type(node.func.type).__name__}' not understood."
        )

    def _visit_astype(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        assert len(node.args) == 2 and isinstance(node.args[1], foast.Name)
        obj, new_type = node.args[0], node.args[1].id
        lowered_obj = self.visit(obj, **kwargs)
        return lowering_utils.process_elements(
            lambda x: im.apply_stencil(im.lambda_("it")(im.call("cast_")(im.deref("it"), str(new_type))), im.make_tuple(x), im.call("get_domain")(x)),
            lowered_obj,
            obj.type,
        )

    def _visit_where(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        condition, true_value, false_value = node.args
        lowered_condition = self.visit(condition, **kwargs)

        #dims = type_info.promote(condition.type, true_value.type, false_value.type).dims

        def handle_element(tv, fv, *, path):
            tv_type = functools.reduce(lambda type_, idx: type_.types[idx], path, true_value.type)
            fv_type = functools.reduce(lambda type_, idx: type_.types[idx], path, false_value.type)
            dims = common.promote_dims(*(type_info.extract_dims(type_) for type_ in [condition.type, tv_type, fv_type]))
            broadcast_axes = [itir.AxisLiteral(value=dim.value) for dim in dims]
            broadcasted_domains = [im.call("broadcast_domain")(im.call("get_domain")(arg), im.make_tuple(*broadcast_axes)) for arg in (lowered_condition, tv, fv)]
            domain = im.call("intersect")(*broadcasted_domains)

            return im.apply_stencil(
                im.lambda_("cit", "tit", "fit")(im.if_(
                    im.deref("cit"),
                    im.deref("tit"),
                    im.deref("fit")
                )),
                im.make_tuple(lowered_condition, tv, fv),  # todo: use let for condition
                domain
            )

        return lowering_utils.process_elements(
            handle_element,
            [self.visit(true_value, **kwargs), self.visit(false_value, **kwargs)],
            [true_value.type, false_value.type],
            with_path_arg=True
        )

    _visit_concat_where = _visit_where

    def _visit_broadcast(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        lowered_arg = self.visit(node.args[0], **kwargs)
        domain = im.call("broadcast_domain")(
            im.call("get_domain")(lowered_arg),  # todo: use let
            im.make_tuple(*(itir.AxisLiteral(value=dim.value) for dim in node.type.dims))
        )
        return im.apply_stencil(
            im.ref("deref"),
            im.make_tuple(lowered_arg),
            domain
        )

    def _visit_math_built_in(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        return self._map(self.visit(node.func, **kwargs), *node.args)

    def _make_reduction_expr(
        self,
        node: foast.Call,
        op: str | itir.SymRef,
        init_expr: itir.Expr,
        **kwargs: Any,
    ) -> itir.Expr:
        # TODO(havogt): deal with nested reductions of the form neighbor_sum(neighbor_sum(field(off1)(off2)))
        lowered_field_expr = self.visit(node.args[0], **kwargs)

        local_dim = None
        assert isinstance(node.type, ts.FieldType)
        for dim in node.args[0].type.dims:
            if dim.kind == DimensionKind.LOCAL:
                local_dim = dim
        assert local_dim is not None

        domain = im.call("strip_domain_axis")(
            im.call("get_domain")(lowered_field_expr),
            itir.AxisLiteral(value=local_dim.value)
        )

        assert isinstance(node.kwargs["axis"].type, ts.DimensionType)
        stencil_body = im.call(im.call("reduce")(op, im.deref("init")))(im.deref("it"))

        return im.apply_stencil(
            im.lambda_("it", "init")(stencil_body),
            im.make_tuple(lowered_field_expr, init_expr),
            domain
        )

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        dtype = type_info.extract_dtype(node.type)
        return self._make_reduction_expr(node, "plus", self._make_literal("0", dtype), **kwargs)

    def _visit_max_over(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        dtype = type_info.extract_dtype(node.type)
        min_value, _ = type_info.arithmetic_bounds(dtype)
        init_expr = self._make_literal(str(min_value), dtype)
        return self._make_reduction_expr(node, "maximum", init_expr, **kwargs)

    def _visit_min_over(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        dtype = type_info.extract_dtype(node.type)
        _, max_value = type_info.arithmetic_bounds(dtype)
        init_expr = self._make_literal(str(max_value), dtype)
        return self._make_reduction_expr(node, "minimum", init_expr, **kwargs)

    def _visit_type_constr(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        if isinstance(node.args[0], foast.Constant):
            node_kind = self.visit(node.type).kind.name.lower()
            target_type = fbuiltins.BUILTINS[node_kind]
            source_type = {**fbuiltins.BUILTINS, "string": str}[node.args[0].type.__str__().lower()]
            if target_type is bool and source_type is not bool:
                return im.apply_stencil(
                    im.lambda_()(im.literal(str(bool(source_type(node.args[0].value))), "bool")),
                    im.make_tuple(),
                    im.call("domain")()
                )
            return im.apply_stencil(
                im.lambda_()(im.literal(str(node.args[0].value), node_kind)),
                im.make_tuple(),
                im.call("domain")()
            )
        raise FieldOperatorLoweringError(
            f"Encountered a type cast, which is not supported: {node}."
        )

    def _make_literal(self, val: Any, type_: ts.TypeSpec) -> itir.Expr:
        # TODO(havogt): lifted nullary lambdas are not supported in iterator.embedded due to an implementation detail;
        # the following constructs work if they are removed by inlining.
        if isinstance(type_, ts.TupleType):
            return im.make_tuple(
                *(self._make_literal(val, type_) for val, type_ in zip(val, type_.types))
            )
        elif isinstance(type_, ts.ScalarType):
            typename = type_.kind.name.lower()
            return im.literal(str(val), typename)
        raise ValueError(f"Unsupported literal type '{type_}'.")

    def visit_Constant(self, node: foast.Constant, **kwargs: Any) -> itir.Expr:
        return self._make_literal(node.value, node.type)

    def _map(self, op: itir.Expr | str, *args: Any, **kwargs: Any) -> itir.FunCall:
        lowered_args = [self.visit(arg, **kwargs) for arg in args]

        if all(isinstance(arg.type, ts.ScalarType) for arg in args):
            return im.call(op)(*lowered_args)

        # TODO: only broadcast scalars
        domain_args = []
        dims = type_info.promote(*(arg.type for arg in args), always_field=True).dims
        for arg, lowered_arg in zip(args, lowered_args, strict=True):
            if isinstance(arg.type, ts.ScalarType):
                continue
            domain_args.append(im.call("broadcast_domain")(
                im.call("get_domain")(lowered_arg),  # todo: use let
                im.make_tuple(*(itir.AxisLiteral(value=dim.value) for dim in dims))
            ))
        domain = im.call("intersect")(*domain_args) if len(domain_args) > 1 else domain_args[0]

        if any(type_info.contains_local_field(arg.type) for arg in args):
            lowered_args = [promote_to_list(arg, larg, domain) for arg, larg in zip(args, lowered_args)]
            op = im.call("map_")(op)

        arg_names = [f"__arg{i}" for i in range(len(lowered_args))]
        return im.apply_stencil(
            im.lambda_(*arg_names)(im.call(op)(*(im.deref(arg) for arg in arg_names))),
            im.make_tuple(*lowered_args),
            domain
        )


class FieldOperatorLoweringError(Exception): ...
