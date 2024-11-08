# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# FIXME[#1582](havogt): remove after refactoring to GTIR

import dataclasses
from typing import Any, Callable, Optional

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.eve.extended_typing import Never
from gt4py.eve.utils import UIDGenerator
from gt4py.next import common
from gt4py.next.ffront import (
    dialect_ast_enums,
    fbuiltins,
    field_operator_ast as foast,
    lowering_utils,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.experimental import EXPERIMENTAL_FUN_BUILTIN_NAMES
from gt4py.next.ffront.fbuiltins import FUN_BUILTIN_NAMES, MATH_BUILTIN_NAMES, TYPE_BUILTIN_NAMES
from gt4py.next.ffront.foast_introspection import StmtReturnKind, deduce_stmt_return_kind
from gt4py.next.ffront.stages import AOT_FOP, FOP
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts
try:
    import ml_dtypes
except ModuleNotFoundError:
    ml_dtypes = None


def foast_to_itir(inp: FOP) -> itir.Expr:
    """
    Lower a FOAST field operator node to Iterator IR.

    See the docstring of `FieldOperatorLowering` for details.
    """
    return FieldOperatorLowering.apply(inp.foast_node)


def foast_to_itir_factory(cached: bool = True) -> workflow.Workflow[FOP, itir.Expr]:
    """Wrap `foast_to_itir` into a chainable and, optionally, cached workflow step."""
    wf = foast_to_itir
    if cached:
        wf = workflow.CachedStep(step=wf, hash_function=ffront_stages.fingerprint_stage)
    return wf


def adapted_foast_to_itir_factory(**kwargs: Any) -> workflow.Workflow[AOT_FOP, itir.Expr]:
    """Wrap the `foast_to_itir` workflow step into an adapter to fit into backend transform workflows."""
    return toolchain.StripArgsAdapter(foast_to_itir_factory(**kwargs))


def promote_to_list(node_type: ts.TypeSpec) -> Callable[[itir.Expr], itir.Expr]:
    if not type_info.contains_local_field(node_type):
        return lambda x: im.promote_to_lifted_stencil("make_const_list")(x)
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

    uid_generator: UIDGenerator = dataclasses.field(default_factory=UIDGenerator)

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
        # note: we don't need the axis here as this is handled by the program
        #  decorator
        assert isinstance(node.type, ts_ffront.ScanOperatorType)

        # We are lowering node.forward and node.init to iterators, but here we expect values -> `deref`.
        # In iterator IR we didn't properly specify if this is legal,
        # however after lift-inlining the expressions are transformed back to literals.
        forward = im.deref(self.visit(node.forward, **kwargs))
        init = lowering_utils.process_elements(
            im.deref, self.visit(node.init, **kwargs), node.init.type
        )

        # lower definition function
        func_definition: itir.FunctionDefinition = self.visit(node.definition, **kwargs)
        new_body = im.let(
            func_definition.params[0].id,
            # promote carry to iterator of tuples
            # (this is the only place in the lowering were a variable is captured in a lifted lambda)
            lowering_utils.to_tuples_of_iterator(
                im.promote_to_const_iterator(func_definition.params[0].id),
                [*node.type.definition.pos_or_kw_args.values()][0],  # noqa: RUF015 [unnecessary-iterable-allocation-for-first-element]
            ),
        )(
            # the function itself returns a tuple of iterators, deref element-wise
            lowering_utils.process_elements(
                im.deref, func_definition.expr, node.type.definition.returns
            )
        )

        stencil_args: list[itir.Expr] = []
        assert not node.type.definition.pos_only_args and not node.type.definition.kw_only_args
        for param, arg_type in zip(
            func_definition.params[1:],
            [*node.type.definition.pos_or_kw_args.values()][1:],
            strict=True,
        ):
            if isinstance(arg_type, ts.TupleType):
                # convert into iterator of tuples
                stencil_args.append(lowering_utils.to_iterator_of_tuples(param.id, arg_type))

                new_body = im.let(
                    param.id, lowering_utils.to_tuples_of_iterator(param.id, arg_type)
                )(new_body)
            else:
                stencil_args.append(im.ref(param.id))

        definition = itir.Lambda(params=func_definition.params, expr=new_body)

        body = im.lift(im.call("scan")(definition, forward, init))(*stencil_args)

        return itir.FunctionDefinition(id=node.id, params=definition.params[1:], expr=body)

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
            return self._lower_and_map("not_", node.operand)

        return self._lower_and_map(
            node.op.value,
            foast.Constant(value="0", type=dtype, location=node.location),
            node.operand,
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs: Any) -> itir.FunCall:
        return self._lower_and_map(node.op.value, node.left, node.right)

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs: Any) -> itir.FunCall:
        op = "if_"
        args = (node.condition, node.true_expr, node.false_expr)
        lowered_args: list[itir.Expr] = [
            lowering_utils.to_iterator_of_tuples(self.visit(arg, **kwargs), arg.type)
            for arg in args
        ]
        if any(type_info.contains_local_field(arg.type) for arg in args):
            lowered_args = [
                promote_to_list(arg.type)(larg) for arg, larg in zip(args, lowered_args)
            ]
            op = im.call("map_")(op)

        return lowering_utils.to_tuples_of_iterator(
            im.promote_to_lifted_stencil(im.call(op))(*lowered_args), node.type
        )

    def visit_Compare(self, node: foast.Compare, **kwargs: Any) -> itir.FunCall:
        return self._lower_and_map(node.op.value, node.left, node.right)

    def _visit_shift(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        current_expr = self.visit(node.func, **kwargs)

        for arg in node.args:
            match arg:
                # `field(Off[idx])`
                case foast.Subscript(value=foast.Name(id=offset_name), index=int(offset_index)):
                    current_expr = im.lift(
                        im.lambda_("it")(im.deref(im.shift(offset_name, offset_index)("it")))
                    )(current_expr)
                # `field(Dim + idx)`
                case foast.BinOp(
                    op=dialect_ast_enums.BinaryOperator.ADD
                    | dialect_ast_enums.BinaryOperator.SUB,
                    left=foast.Name(id=dimension),
                    right=foast.Constant(value=offset_index),
                ):
                    if arg.op == dialect_ast_enums.BinaryOperator.SUB:
                        offset_index *= -1
                    current_expr = im.lift(
                        # TODO(SF-N): we rely on the naming-convention that the cartesian dimensions
                        #  are passed suffixed with `off`, e.g. the `K` is passed as `Koff` in the
                        #  offset provider. This is a rather unclean solution and should be
                        #  improved.
                        im.lambda_("it")(
                            im.deref(
                                im.shift(
                                    common.dimension_to_implicit_offset(dimension), offset_index
                                )("it")
                            )
                        )
                    )(current_expr)
                # `field(Off)`
                case foast.Name(id=offset_name):
                    # only a single unstructured shift is supported so returning here is fine even though we
                    # are in a loop.
                    assert len(node.args) == 1 and len(arg.type.target) > 1  # type: ignore[attr-defined] # ensured by pattern
                    return im.lifted_neighbors(str(offset_name), self.visit(node.func, **kwargs))
                # `field(as_offset(Off, offset_field))`
                case foast.Call(func=foast.Name(id="as_offset")):
                    func_args = arg
                    # TODO(tehrengruber): Use type system to deduce the offset dimension instead of
                    #  (e.g. to allow aliasing)
                    offset_dim = func_args.args[0]
                    assert isinstance(offset_dim, foast.Name)
                    offset_it = self.visit(func_args.args[1], **kwargs)
                    current_expr = im.lift(
                        im.lambda_("it", "offset")(
                            im.deref(im.shift(offset_dim.id, im.deref("offset"))("it"))
                        )
                    )(current_expr, offset_it)
                case _:
                    raise FieldOperatorLoweringError("Unexpected shift arguments!")

        return current_expr

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
                result = lowering_utils.to_tuples_of_iterator(
                    result, node.func.type.definition.returns
                )

            return result

        raise AssertionError(
            f"Call to object of type '{type(node.func.type).__name__}' not understood."
        )

    def _visit_astype(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        assert len(node.args) == 2 and isinstance(node.args[1], foast.Name)
        obj, new_type = node.args[0], node.args[1].id
        return lowering_utils.process_elements(
            lambda x: im.promote_to_lifted_stencil(
                im.lambda_("it")(im.call("cast_")("it", str(new_type)))
            )(x),
            self.visit(obj, **kwargs),
            obj.type,
        )

    def _visit_where(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        condition, true_value, false_value = node.args

        lowered_condition = self.visit(condition, **kwargs)
        return lowering_utils.process_elements(
            lambda tv, fv, types: _map(
                "if_", (lowered_condition, tv, fv), (condition.type, *types)
            ),
            [self.visit(true_value, **kwargs), self.visit(false_value, **kwargs)],
            node.type,
            (node.args[1].type, node.args[2].type),
        )

    _visit_concat_where = _visit_where

    def _visit_broadcast(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        return self.visit(node.args[0], **kwargs)

    def _visit_math_built_in(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        return self._lower_and_map(self.visit(node.func, **kwargs), *node.args)

    def _make_reduction_expr(
        self, node: foast.Call, op: str | itir.SymRef, init_expr: itir.Expr, **kwargs: Any
    ) -> itir.Expr:
        # TODO(havogt): deal with nested reductions of the form neighbor_sum(neighbor_sum(field(off1)(off2)))
        it = self.visit(node.args[0], **kwargs)
        assert isinstance(node.kwargs["axis"].type, ts.DimensionType)
        val = im.call(im.call("reduce")(op, im.deref(init_expr)))
        return im.promote_to_lifted_stencil(val)(it)

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
        el = node.args[0]
        node_kind = self.visit(node.type).kind.name.lower()
        source_type = {**fbuiltins.BUILTINS, "string": str}[el.type.__str__().lower()]
        target_type = fbuiltins.BUILTINS[node_kind]

        if isinstance(el, foast.Constant):
            val = source_type(el.value)
        elif isinstance(el, foast.UnaryOp) and isinstance(el.operand, foast.Constant):
            operand = source_type(el.operand.value)
            val = eval(f"lambda arg: {el.op}arg")(operand)
        else:
            raise FieldOperatorLoweringError(
                f"Type cast only supports literal arguments, {node.type} not supported."
            )
        # TODO: why?
        if ml_dtypes and node_kind == 'bfloat16':
            val = float(val)
        val = target_type(val)

        return im.promote_to_const_iterator(im.literal(str(val), node_kind))

    def _make_literal(self, val: Any, type_: ts.TypeSpec) -> itir.Expr:
        # TODO(havogt): lifted nullary lambdas are not supported in iterator.embedded due to an implementation detail;
        # the following constructs work if they are removed by inlining.
        if isinstance(type_, ts.TupleType):
            return im.make_tuple(
                *(self._make_literal(val, type_) for val, type_ in zip(val, type_.types))
            )
        elif isinstance(type_, ts.ScalarType):
            typename = type_.kind.name.lower()
            return im.promote_to_const_iterator(im.literal(str(val), typename))
        raise ValueError(f"Unsupported literal type '{type_}'.")

    def visit_Constant(self, node: foast.Constant, **kwargs: Any) -> itir.Expr:
        return self._make_literal(node.value, node.type)

    def _lower_and_map(self, op: itir.Expr | str, *args: Any, **kwargs: Any) -> itir.FunCall:
        return _map(
            op, tuple(self.visit(arg, **kwargs) for arg in args), tuple(arg.type for arg in args)
        )


def _map(
    op: itir.Expr | str,
    lowered_args: tuple,
    original_arg_types: tuple[ts.TypeSpec, ...],
) -> itir.FunCall:
    """
    Mapping includes making the operation an lifted stencil (first kind of mapping), but also `itir.map_`ing lists.
    """
    if any(type_info.contains_local_field(arg_type) for arg_type in original_arg_types):
        lowered_args = tuple(
            promote_to_list(arg_type)(larg)
            for arg_type, larg in zip(original_arg_types, lowered_args)
        )
        op = im.call("map_")(op)

    return im.promote_to_lifted_stencil(im.call(op))(*lowered_args)


class FieldOperatorLoweringError(Exception): ...
