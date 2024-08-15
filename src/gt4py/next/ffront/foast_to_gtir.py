# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import dataclasses
from typing import Any, Callable, Optional

from gt4py.eve import NodeTranslator, PreserveLocationVisitor, utils as eve_utils
from gt4py.eve.extended_typing import Never
from gt4py.next.ffront import (
    dialect_ast_enums,
    fbuiltins,
    field_operator_ast as foast,
    foast_introspection,
    lowering_utils,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.experimental import EXPERIMENTAL_FUN_BUILTIN_NAMES
from gt4py.next.ffront.fbuiltins import FUN_BUILTIN_NAMES, MATH_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_info, type_specifications as ts


def foast_to_gtir(inp: ffront_stages.FoastOperatorDefinition) -> itir.Expr:
    return FieldOperatorLowering.apply(inp.foast_node)


def promote_to_list(node: foast.Symbol | foast.Expr) -> Callable[[itir.Expr], itir.Expr]:
    if not type_info.contains_local_field(node.type):
        return lambda x: im.op_as_fieldop("make_const_list")(x)
    return lambda x: x


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
    [Sym(id=SymbolName('inp'))]
    """

    uid_generator: eve_utils.UIDGenerator = dataclasses.field(
        default_factory=eve_utils.UIDGenerator
    )

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> itir.Expr:
        return cls().visit(node)

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> itir.FunctionDefinition:
        params = self.visit(node.params)
        return itir.FunctionDefinition(
            id=node.id, params=params, expr=self.visit_BlockStmt(node.body, inner_expr=None)
        )  # `expr` is a lifted stencil

    def visit_FieldOperator(
        self, node: foast.FieldOperator, **kwargs: Any
    ) -> itir.FunctionDefinition:
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)

        new_body = func_definition.expr

        return itir.FunctionDefinition(
            id=func_definition.id, params=func_definition.params, expr=new_body
        )

    def visit_ScanOperator(
        self, node: foast.ScanOperator, **kwargs: Any
    ) -> itir.FunctionDefinition:
        raise NotImplementedError("TODO")

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
        assert (
            isinstance(node.condition.type, ts.ScalarType)
            and node.condition.type.kind == ts.ScalarKind.BOOL
        )

        cond = self.visit(node.condition, **kwargs)

        return_kind = foast_introspection.deduce_stmt_return_kind(node)

        common_symbols: dict[str, foast.Symbol] = node.annex.propagated_symbols

        if return_kind is foast_introspection.StmtReturnKind.NO_RETURN:
            # TODO document why this case should be handled in this way, not by the more general CONDITIONAL_RETURN

            # pack the common symbols into a tuple
            common_symrefs = im.make_tuple(*(im.ref(sym) for sym in common_symbols.keys()))

            # apply both branches and extract the common symbols through the prepared tuple
            true_branch = self.visit(node.true_branch, inner_expr=common_symrefs, **kwargs)
            false_branch = self.visit(node.false_branch, inner_expr=common_symrefs, **kwargs)

            # unpack the common symbols' tuple for `inner_expr`
            for i, sym in enumerate(common_symbols.keys()):
                inner_expr = im.let(sym, im.tuple_get(i, im.ref("__if_stmt_result")))(inner_expr)

            # here we assume neither branch returns
            return im.let("__if_stmt_result", im.cond(cond, true_branch, false_branch))(inner_expr)
        elif return_kind is foast_introspection.StmtReturnKind.CONDITIONAL_RETURN:
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
                im.cond(cond, true_branch, false_branch)
            )

        assert return_kind is foast_introspection.StmtReturnKind.UNCONDITIONAL_RETURN

        # note that we do not duplicate `inner_expr` here since if both branches
        #  return, `inner_expr` is ignored.
        true_branch = self.visit(node.true_branch, inner_expr=inner_expr, **kwargs)
        false_branch = self.visit(node.false_branch, inner_expr=inner_expr, **kwargs)

        return im.cond(cond, true_branch, false_branch)

    def visit_Assign(
        self, node: foast.Assign, *, inner_expr: Optional[itir.Expr], **kwargs: Any
    ) -> itir.Expr:
        return im.let(self.visit(node.target, **kwargs), self.visit(node.value, **kwargs))(
            inner_expr
        )

    def visit_Symbol(self, node: foast.Symbol, **kwargs: Any) -> itir.Sym:
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
        if node.op in [dialect_ast_enums.UnaryOperator.NOT, dialect_ast_enums.UnaryOperator.INVERT]:
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
        assert (
            isinstance(node.condition.type, ts.ScalarType)
            and node.condition.type.kind == ts.ScalarKind.BOOL
        )
        return im.cond(
            self.visit(node.condition), self.visit(node.true_expr), self.visit(node.false_expr)
        )

    def visit_Compare(self, node: foast.Compare, **kwargs: Any) -> itir.FunCall:
        return self._map(node.op.value, node.left, node.right)

    def _visit_shift(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        match node.args[0]:
            case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                shift_offset = im.shift(offset_name, offset_index)
                return im.as_fieldop(im.lambda_("__it")(im.deref(shift_offset("__it"))))(
                    self.visit(node.func, **kwargs)
                )
            case foast.Name(id=offset_name):
                return im.as_fieldop_neighbors(str(offset_name), self.visit(node.func, **kwargs))
            case foast.Call(func=foast.Name(id="as_offset")):
                # TODO(havogt): discuss this representation
                func_args = node.args[0]
                offset_dim = func_args.args[0]
                assert isinstance(offset_dim, foast.Name)
                shift_offset = im.shift(offset_dim.id, im.deref("__offset"))
                return im.as_fieldop(
                    im.lambda_("__it", "__offset")(im.deref(shift_offset("__it")))
                )(self.visit(node.func, **kwargs), self.visit(func_args.args[1], **kwargs))
            case _:
                raise FieldOperatorLoweringError("Unexpected shift arguments!")

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
            (ts.FunctionType, ts_ffront.FieldOperatorType, ts_ffront.ScanOperatorType),
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
            if isinstance(node.func.type, ts_ffront.ScanOperatorType):
                raise NotImplementedError("TODO")

            return result

        raise AssertionError(
            f"Call to object of type '{type(node.func.type).__name__}' not understood."
        )

    def _visit_astype(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        assert len(node.args) == 2 and isinstance(node.args[1], foast.Name)
        obj, new_type = self.visit(node.args[0], **kwargs), node.args[1].id

        def create_cast(expr: itir.Expr) -> itir.FunCall:
            return im.as_fieldop(
                im.lambda_("__val")(im.call("cast_")(im.deref("__val"), str(new_type)))
            )(expr)

        if not isinstance(node.type, ts.TupleType):
            return create_cast(obj)

        return lowering_utils.process_elements(create_cast, obj, node.type)

    def _visit_where(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        if not isinstance(node.type, ts.TupleType):
            return im.op_as_fieldop("if_")(*self.visit(node.args))

        cond_ = self.visit(node.args[0])
        cond_symref_name = f"__cond_{eve_utils.content_hash(cond_)}"

        def create_if(cond: itir.Expr) -> Callable[[itir.Expr, itir.Expr], itir.FunCall]:
            def create_if_impl(true_: itir.Expr, false_: itir.Expr) -> itir.FunCall:
                return im.op_as_fieldop("if_")(cond, true_, false_)

            return create_if_impl

        result = lowering_utils.process_elements(
            create_if(im.ref(cond_symref_name)),
            (self.visit(node.args[1]), self.visit(node.args[2])),
            node.type,
        )

        return im.let(cond_symref_name, cond_)(result)

    _visit_concat_where = _visit_where  # TODO(havogt): upgrade concat_where

    def _visit_broadcast(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        return self.visit(node.args[0], **kwargs)

    def _visit_math_built_in(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        return self._map(self.visit(node.func, **kwargs), *node.args)

    def _make_reduction_expr(
        self, node: foast.Call, op: str | itir.SymRef, init_expr: itir.Expr, **kwargs: Any
    ) -> itir.Expr:
        # TODO(havogt): deal with nested reductions of the form neighbor_sum(neighbor_sum(field(off1)(off2)))
        it = self.visit(node.args[0], **kwargs)
        assert isinstance(node.kwargs["axis"].type, ts.DimensionType)
        val = im.call(im.call("reduce")(op, init_expr))
        return im.op_as_fieldop(val)(it)

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
                return im.literal(str(bool(source_type(node.args[0].value))), "bool")
            return im.literal(str(node.args[0].value), node_kind)
        raise FieldOperatorLoweringError(
            f"Encountered a type cast, which is not supported: {node}."
        )

    def _make_literal(self, val: Any, type_: ts.TypeSpec) -> itir.Expr:
        if isinstance(type_, ts.ScalarType):
            typename = type_.kind.name.lower()
            return im.literal(str(val), typename)
        raise ValueError(f"Unsupported literal type '{type_}'.")

    def visit_Constant(self, node: foast.Constant, **kwargs: Any) -> itir.Expr:
        return self._make_literal(node.value, node.type)

    def _map(self, op: itir.Expr | str, *args: Any, **kwargs: Any) -> itir.FunCall:
        lowered_args = [self.visit(arg, **kwargs) for arg in args]
        if all(
            isinstance(t, ts.ScalarType)
            for arg in args
            for t in type_info.primitive_constituents(arg.type)
        ):
            return im.call(op)(*lowered_args)  # scalar operation
        if any(type_info.contains_local_field(arg.type) for arg in args):
            lowered_args = [promote_to_list(arg)(larg) for arg, larg in zip(args, lowered_args)]
            op = im.call("map_")(op)

        return im.op_as_fieldop(im.call(op))(*lowered_args)


class FieldOperatorLoweringError(Exception): ...
