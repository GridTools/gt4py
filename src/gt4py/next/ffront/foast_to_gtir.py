# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import dataclasses
from typing import Any, Callable, Optional

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.eve.extended_typing import Never
from gt4py.next import common
from gt4py.next.ffront import (
    dialect_ast_enums,
    experimental as experimental_builtins,
    fbuiltins,
    field_operator_ast as foast,
    foast_introspection,
    lowering_utils,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.foast_passes import utils as foast_utils
from gt4py.next.ffront.stages import AOT_FOP, FOP
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import constant_folding
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation as tt


def foast_to_gtir(inp: ffront_stages.FoastOperatorDefinition) -> itir.FunctionDefinition:
    """
    Lower a FOAST field operator node to GTIR.

    See the docstring of `FieldOperatorLowering` for details.
    """
    return FieldOperatorLowering.apply(inp.foast_node)


def foast_to_gtir_factory(cached: bool = True) -> workflow.Workflow[FOP, itir.FunctionDefinition]:
    """Wrap `foast_to_gtir` into a chainable and, optionally, cached workflow step."""
    wf = foast_to_gtir
    if cached:
        wf = workflow.CachedStep(step=wf, hash_function=ffront_stages.fingerprint_stage)
    return wf


def adapted_foast_to_gtir_factory(
    **kwargs: Any,
) -> workflow.Workflow[AOT_FOP, itir.FunctionDefinition]:
    """Wrap the `foast_to_gtir` workflow step into an adapter to fit into backend transform workflows."""
    return toolchain.StripArgsAdapter(foast_to_gtir_factory(**kwargs))


def promote_to_list(node_type: ts.TypeSpec) -> Callable[[itir.Expr], itir.Expr]:
    if not type_info.contains_local_field(node_type):
        return lambda x: im.op_as_fieldop("make_const_list")(x)
    return lambda x: x


@dataclasses.dataclass
class FieldOperatorLowering(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """
    Lower FieldOperator AST (FOAST) to GTIR.

    Most expressions are lowered to `as_fieldop`ed stencils.
    Pure scalar expressions are kept as scalar operations as they might appear outside of the
    stencil context. In arithmetic operations that involve a field and a scalar, the scalar is
    implicitly broadcasted to a field in the `as_fieldop` call.

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
    def apply(cls, node: foast.LocatedNode) -> itir.FunctionDefinition:
        result = cls().visit(node)
        assert isinstance(result, itir.FunctionDefinition)
        return result

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

        return itir.FunctionDefinition(
            id=func_definition.id, params=func_definition.params, expr=func_definition.expr
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
        for param in func_definition.params[1:]:
            new_body = im.let(param.id, im.deref(param.id))(new_body)
            stencil_args.append(im.ref(param.id))

        definition = itir.Lambda(params=func_definition.params, expr=new_body)

        body = im.as_fieldop(im.scan(definition, forward, init))(*stencil_args)

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
        assert (
            isinstance(node.condition.type, ts.ScalarType)
            and node.condition.type.kind == ts.ScalarKind.BOOL
        )

        cond = self.visit(node.condition, **kwargs)

        return_kind = foast_introspection.deduce_stmt_return_kind(node)

        common_symbols: dict[str, foast.Symbol] = node.annex.propagated_symbols

        if return_kind is foast_introspection.StmtReturnKind.NO_RETURN:
            # FIXME[#1582](havogt): document why this case should be handled in this way, not by the more general CONDITIONAL_RETURN

            # pack the common symbols into a tuple
            common_symrefs = im.make_tuple(*(im.ref(sym) for sym in common_symbols.keys()))

            # apply both branches and extract the common symbols through the prepared tuple
            true_branch = self.visit(node.true_branch, inner_expr=common_symrefs, **kwargs)
            false_branch = self.visit(node.false_branch, inner_expr=common_symrefs, **kwargs)

            # unpack the common symbols' tuple for `inner_expr`
            for i, sym in enumerate(common_symbols.keys()):
                inner_expr = im.let(sym, im.tuple_get(i, im.ref("__if_stmt_result")))(inner_expr)

            # here we assume neither branch returns
            return im.let("__if_stmt_result", im.if_(cond, true_branch, false_branch))(inner_expr)
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
                im.if_(cond, true_branch, false_branch)
            )

        assert return_kind is foast_introspection.StmtReturnKind.UNCONDITIONAL_RETURN

        # note that we do not duplicate `inner_expr` here since if both branches
        #  return, `inner_expr` is ignored.
        true_branch = self.visit(node.true_branch, inner_expr=inner_expr, **kwargs)
        false_branch = self.visit(node.false_branch, inner_expr=inner_expr, **kwargs)

        return im.if_(cond, true_branch, false_branch)

    def visit_Assign(
        self, node: foast.Assign, *, inner_expr: Optional[itir.Expr], **kwargs: Any
    ) -> itir.Expr:
        return im.let(self.visit(node.target, **kwargs), self.visit(node.value, **kwargs))(
            inner_expr
        )

    def visit_Symbol(self, node: foast.Symbol, **kwargs: Any) -> itir.Sym:
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs: Any) -> itir.SymRef | itir.AxisLiteral:
        if isinstance(node.type, ts.DimensionType):
            return itir.AxisLiteral(value=node.type.dim.value, kind=node.type.dim.kind)
        return im.ref(node.id)

    def visit_Attribute(self, node: foast.Attribute, **kwargs: Any) -> itir.AxisLiteral:
        if isinstance(node.type, ts.DimensionType):
            return itir.AxisLiteral(value=node.type.dim.value, kind=node.type.dim.kind)
        raise AssertionError(
            "Unexpected attribute access. At this point all accesses should have been removed by `ClosureVarFolding`."
        )

    def visit_Subscript(self, node: foast.Subscript, **kwargs: Any) -> itir.Expr:
        if isinstance(node.index.type, ts.IndexType):
            # `field[LocalDim(42)]`
            assert isinstance(node.index, foast.Call)
            assert isinstance(node.index.args[0], foast.Constant)
            return im.as_fieldop_deref_list_get(
                node.index.args[0].value, self.visit(node.value, **kwargs)
            )
        return im.tuple_get(foast_utils.expr_to_index(node.index), self.visit(node.value, **kwargs))

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs: Any) -> itir.Expr:
        return im.make_tuple(*[self.visit(el, **kwargs) for el in node.elts])

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs: Any) -> itir.Expr:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        dtype = type_info.extract_dtype(node.type)
        assert isinstance(dtype, ts.ScalarType)
        if node.op in [dialect_ast_enums.UnaryOperator.NOT, dialect_ast_enums.UnaryOperator.INVERT]:
            if dtype.kind != ts.ScalarKind.BOOL:
                raise NotImplementedError(f"'{node.op}' is only supported on 'bool' arguments.")
            return self._lower_and_map("not_", node.operand)
        if node.op in [dialect_ast_enums.UnaryOperator.USUB]:
            return self._lower_and_map("neg", node.operand)
        if node.op in [dialect_ast_enums.UnaryOperator.UADD]:
            return self.visit(node.operand)
        else:
            raise NotImplementedError(f"Unary operator '{node.op}' is not supported.")

    def visit_BinOp(self, node: foast.BinOp, **kwargs: Any) -> itir.FunCall:
        return self._lower_and_map(node.op.value, node.left, node.right)

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs: Any) -> itir.FunCall:
        assert (
            isinstance(node.condition.type, ts.ScalarType)
            and node.condition.type.kind == ts.ScalarKind.BOOL
        )
        return im.if_(
            self.visit(node.condition), self.visit(node.true_expr), self.visit(node.false_expr)
        )

    def visit_Compare(self, node: foast.Compare, **kwargs: Any) -> itir.FunCall:
        return self._lower_and_map(node.op.value, node.left, node.right)

    def _visit_shift(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        current_expr = self.visit(node.func, **kwargs)

        for arg in node.args:
            match arg:
                # `field(Off[idx])`
                case foast.Subscript(value=foast.Name(id=offset_name), index=index):
                    # Constant folding to a `Literal` ensures that `index` becomes an `OffsetLiteral`,
                    # which can be generated as compile-time value backend code.
                    new_index = constant_folding.ConstantFolding.apply(self.visit(index, **kwargs))
                    assert isinstance(new_index, itir.Literal)
                    current_expr = im.as_fieldop(
                        im.lambda_("__it")(im.deref(im.shift(offset_name, new_index)("__it")))
                    )(current_expr)
                # `field(Dim + idx)`
                case foast.BinOp(
                    op=dialect_ast_enums.BinaryOperator.ADD | dialect_ast_enums.BinaryOperator.SUB,
                    left=foast.Name(id=dimension),  # TODO(tehrengruber): use type of lhs
                    right=foast.Constant(value=offset_index),
                ):
                    if arg.op == dialect_ast_enums.BinaryOperator.SUB:
                        offset_index *= -1
                    # TODO(havogt): we rely on the naming-convention for implicit offsets, see `dimension_to_implicit_offset`
                    current_expr = im.as_fieldop(
                        im.lambda_("__it")(
                            im.deref(
                                im.shift(
                                    common.dimension_to_implicit_offset(dimension), offset_index
                                )("__it")
                            )
                        )
                    )(current_expr)
                # `field(Off)`
                case foast.Name(id=offset_name):
                    # only a single unstructured shift is supported so returning here is fine even though we
                    # are in a loop.
                    assert len(node.args) == 1 and len(arg.type.target) > 1  # type: ignore[attr-defined] # ensured by pattern
                    return im.as_fieldop_neighbors(
                        str(offset_name), self.visit(node.func, **kwargs)
                    )
                # `field(as_offset(Off, offset_field))`
                case foast.Call(func=foast.Name(id="as_offset")):
                    func_args = arg
                    # TODO(tehrengruber): Discuss representation. We could use the type system to
                    #  deduce the offset dimension instead of (e.g. to allow aliasing).
                    offset_dim = func_args.args[0]
                    assert isinstance(offset_dim, foast.Name)
                    offset_field = self.visit(func_args.args[1], **kwargs)
                    current_expr = im.as_fieldop(
                        im.lambda_("__it", "__offset")(
                            im.deref(im.shift(offset_dim.id, im.deref("__offset"))("__it"))
                        )
                    )(current_expr, offset_field)
                case _:
                    raise FieldOperatorLoweringError("Unexpected shift arguments!")

        return current_expr

    def visit_Call(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        if type_info.type_class(node.func.type) is ts.FieldType:
            return self._visit_shift(node, **kwargs)
        elif isinstance(node.func, foast.Name) and node.func.id in fbuiltins.MATH_BUILTIN_NAMES:
            return self._visit_math_built_in(node, **kwargs)
        elif isinstance(node.func, foast.Name) and node.func.id in (
            fbuiltins.FUN_BUILTIN_NAMES + experimental_builtins.EXPERIMENTAL_FUN_BUILTIN_NAMES
        ):
            visitor = getattr(self, f"_visit_{node.func.id}")
            return visitor(node, **kwargs)
        elif isinstance(node.func, foast.Name) and isinstance(node.func.type, ts.ConstructorType):
            return self._visit_type_constr(node, **kwargs)
        elif isinstance(
            node.func.type,
            (ts.FunctionType, ts_ffront.FieldOperatorType, ts_ffront.ScanOperatorType),
        ):
            # ITIR has no support for keyword arguments. Instead, we concatenate both positional
            # and keyword arguments and use the unique order as given in the function signature.
            lowered_args, lowered_kwargs = type_info.canonicalize_arguments(
                node.func.type, self.visit(node.args, **kwargs), self.visit(node.kwargs, **kwargs)
            )
            result = im.call(self.visit(node.func, **kwargs))(
                *lowered_args, *lowered_kwargs.values()
            )

            return result

        raise AssertionError(
            f"Call to object of type '{type(node.func.type).__name__}' not understood."
        )

    def _visit_astype(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        # Note: the type to convert to is uniquely identified by its GT4Py type (`ConstructorType`),
        # not by e.g. its name.
        assert len(node.args) == 2
        assert isinstance(node.args[1].type, ts.ConstructorType)
        obj = self.visit(node.args[0], **kwargs)
        new_type = node.args[1].type.definition.returns
        assert isinstance(new_type, ts.ScalarType)

        def create_cast(expr: itir.Expr, t: tuple[ts.TypeSpec]) -> itir.FunCall:
            return _map(im.lambda_("val")(im.cast_("val", new_type)), (expr,), t)

        if not isinstance(node.type, ts.TupleType):  # to keep the IR simpler
            return create_cast(obj, (node.args[0].type,))

        return lowering_utils.process_elements(
            create_cast, obj, node.type, arg_types=(node.args[0].type,)
        )

    def _visit_where(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        if not isinstance(node.type, ts.TupleType):  # to keep the IR simpler
            return self._lower_and_map("if_", *node.args)

        cond_ = self.visit(node.args[0])
        cond_symref_name = f"__cond_{eve_utils.content_hash(cond_)}"

        def create_if(
            true_: itir.Expr, false_: itir.Expr, arg_types: tuple[ts.TypeSpec, ts.TypeSpec]
        ) -> itir.FunCall:
            return _map(
                "if_",
                (im.ref(cond_symref_name), true_, false_),
                (node.args[0].type, *arg_types),
            )

        result = lowering_utils.process_elements(
            create_if,
            (self.visit(node.args[1]), self.visit(node.args[2])),
            node.type,
            arg_types=(node.args[1].type, node.args[2].type),
        )

        return im.let(cond_symref_name, cond_)(result)

    def _visit_concat_where(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        domain, true_branch, false_branch = self.visit(node.args, **kwargs)
        return im.concat_where(domain, true_branch, false_branch)

    def _visit_broadcast(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        return im.call("broadcast")(*self.visit(node.args, **kwargs))

    def _visit_math_built_in(self, node: foast.Call, **kwargs: Any) -> itir.FunCall:
        return self._lower_and_map(self.visit(node.func, **kwargs), *node.args)

    def _make_reduction_expr(
        self, node: foast.Call, op: str | itir.SymRef, init_expr: itir.Expr, **kwargs: Any
    ) -> itir.Expr:
        # TODO(havogt): deal with nested reductions of the form neighbor_sum(neighbor_sum(field(off1)(off2)))
        it = self.visit(node.args[0], **kwargs)
        assert isinstance(node.kwargs["axis"].type, ts.DimensionType)
        val = im.reduce(op, init_expr)
        return im.op_as_fieldop(val)(it)

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        dtype = type_info.extract_dtype(node.type)
        return self._make_reduction_expr(node, "plus", self._make_literal("0", dtype), **kwargs)

    def _visit_max_over(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        dtype = type_info.extract_dtype(node.type)
        assert isinstance(dtype, ts.ScalarType)
        min_value, _ = type_info.arithmetic_bounds(dtype)
        init_expr = self._make_literal(str(min_value), dtype)
        return self._make_reduction_expr(node, "maximum", init_expr, **kwargs)

    def _visit_min_over(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        dtype = type_info.extract_dtype(node.type)
        assert isinstance(dtype, ts.ScalarType)
        _, max_value = type_info.arithmetic_bounds(dtype)
        init_expr = self._make_literal(str(max_value), dtype)
        return self._make_reduction_expr(node, "minimum", init_expr, **kwargs)

    def _visit_type_constr(self, node: foast.Call, **kwargs: Any) -> itir.Expr:
        # Note: the type to convert to is uniquely identified by its GT4Py type (`ConstructorType`),
        # not by e.g. its name.
        type_constructor = node.func
        assert isinstance(type_constructor.type, ts.ConstructorType)

        value = node.args[0]

        source_type = value.type
        assert isinstance(source_type, ts.ScalarType)
        target_type = type_constructor.type.definition.returns
        assert isinstance(target_type, ts.ScalarType)

        if isinstance(value, foast.Constant):
            val = (
                value.value
                if source_type.kind == ts.ScalarKind.STRING
                else tt.unsafe_cast_to(value.value, source_type)
            )
        elif isinstance(value, foast.UnaryOp) and isinstance(value.operand, foast.Constant):
            operand = (
                value.operand.value
                if source_type.kind == ts.ScalarKind.STRING
                else tt.unsafe_cast_to(value.operand.value, source_type)
            )
            val = eval(f"lambda arg: {value.op}arg")(operand)
        else:
            raise FieldOperatorLoweringError(
                f"Type cast only supports literal arguments, {node.type} not supported."
            )
        val = tt.unsafe_cast_to(val, target_type)

        return im.literal(str(val), target_type)

    def _make_literal(self, val: Any, type_: ts.TypeSpec) -> itir.Expr:
        if isinstance(type_, ts.TupleType):
            return im.make_tuple(
                *(self._make_literal(val, type_) for val, type_ in zip(val, type_.types))
            )
        if isinstance(type_, ts.ScalarType):
            typename = type_.kind.name.lower()
            return im.literal(str(val), typename)
        raise ValueError(f"Unsupported literal type '{type_}'.")

    def visit_Constant(self, node: foast.Constant, **kwargs: Any) -> itir.Expr:
        return self._make_literal(node.value, node.type)

    def _lower_and_map(self, op: itir.Lambda | str, *args: Any, **kwargs: Any) -> itir.FunCall:
        return _map(
            op, tuple(self.visit(arg, **kwargs) for arg in args), tuple(arg.type for arg in args)
        )


def _map(
    op: itir.Lambda | str,
    lowered_args: tuple,
    original_arg_types: tuple[ts.TypeSpec, ...],
) -> itir.FunCall:
    """
    Mapping includes making the operation an `as_fieldop` (first kind of mapping), but also `itir.map_`ing lists.
    """
    if all(
        isinstance(t, (ts.ScalarType, ts.DimensionType, ts.DomainType))
        for arg_type in original_arg_types
        for t in type_info.primitive_constituents(arg_type)
    ):
        return im.call(op)(*lowered_args)  # scalar operation
    if any(type_info.contains_local_field(arg_type) for arg_type in original_arg_types):
        lowered_args = tuple(
            promote_to_list(arg_type)(larg)
            for arg_type, larg in zip(original_arg_types, lowered_args)
        )
        op = im.map_(op)

    return im.op_as_fieldop(op)(*lowered_args)


class FieldOperatorLoweringError(Exception): ...
