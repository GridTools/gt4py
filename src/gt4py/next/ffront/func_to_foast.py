# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import textwrap
import typing
from typing import Any, Type, cast

import gt4py.eve as eve
from gt4py.next import errors
from gt4py.next.ffront import (
    dialect_ast_enums,
    fbuiltins,
    field_operator_ast as foast,
    source_utils,
    stages as ffront_stages,
)
from gt4py.next.ffront.ast_passes import (
    SingleAssignTargetPass,
    SingleStaticAssignPass,
    StringifyAnnotationsPass,
    UnchainComparesPass,
)
from gt4py.next.ffront.dialect_parser import DialectParser
from gt4py.next.ffront.foast_introspection import StmtReturnKind, deduce_stmt_return_kind
from gt4py.next.ffront.foast_passes.closure_var_folding import ClosureVarFolding
from gt4py.next.ffront.foast_passes.closure_var_type_deduction import ClosureVarTypeDeduction
from gt4py.next.ffront.foast_passes.dead_closure_var_elimination import DeadClosureVarElimination
from gt4py.next.ffront.foast_passes.iterable_unpack import UnpackedAssignPass
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from gt4py.next.ffront.stages import AOT_DSL_FOP, AOT_FOP, DSL_FOP, FOP
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


def func_to_foast(inp: DSL_FOP) -> FOP:
    """
    Turn a DSL field operator definition into a FOAST operator definition, adding metadata.

    Examples:

        >>> from gt4py import next as gtx
        >>> IDim = gtx.Dimension("I")

        >>> const = gtx.float32(2.0)
        >>> def dsl_operator(a: gtx.Field[[IDim], gtx.float32]) -> gtx.Field[[IDim], gtx.float32]:
        ...     return a * const

        >>> dsl_operator_def = gtx.ffront.stages.FieldOperatorDefinition(definition=dsl_operator)
        >>> foast_definition = func_to_foast(dsl_operator_def)

        >>> print(foast_definition.foast_node.id)
        dsl_operator

        >>> foast_closure_vars = {k: str(v) for k, v in foast_definition.closure_vars.items()}
        >>> print(foast_closure_vars)
        {'const': '2.0'}
    """
    source_def = source_utils.SourceDefinition.from_function(inp.definition)
    closure_vars = source_utils.get_closure_vars_from_function(inp.definition)
    annotations = typing.get_type_hints(inp.definition)
    foast_definition_node = FieldOperatorParser.apply(source_def, closure_vars, annotations)
    loc = foast_definition_node.location
    operator_attribute_nodes = {
        key: foast.Constant(value=value, type=type_translation.from_value(value), location=loc)
        for key, value in inp.attributes.items()
    }
    untyped_foast_node = inp.node_class(
        id=foast_definition_node.id,
        definition=foast_definition_node,
        location=loc,
        **operator_attribute_nodes,
    )
    foast_node = FieldOperatorTypeDeduction.apply(untyped_foast_node)
    return ffront_stages.FoastOperatorDefinition(
        foast_node=foast_node,
        closure_vars=closure_vars,
        grid_type=inp.grid_type,
        attributes=inp.attributes,
        debug=inp.debug,
    )


def func_to_foast_factory(cached: bool = True) -> workflow.Workflow[DSL_FOP, FOP]:
    """Wrap `func_to_foast` in a chainable and optionally cached workflow step."""
    wf = workflow.make_step(func_to_foast)
    if cached:
        wf = workflow.CachedStep(step=wf, hash_function=ffront_stages.fingerprint_stage)
    return wf


def adapted_func_to_foast_factory(**kwargs: Any) -> workflow.Workflow[AOT_DSL_FOP, AOT_FOP]:
    """Wrap the `func_to_foast step in an adapter to fit into transform toolchains.`"""
    return toolchain.DataOnlyAdapter(func_to_foast_factory(**kwargs))


class FieldOperatorParser(DialectParser[foast.FunctionDefinition]):
    """
    Parse field operator function definition from source code into FOAST.

    Catch any Field Operator specific syntax errors and typing problems.

    Example
    -------
    Parse a function into a Field Operator AST (FOAST), which can
    be lowered into Iterator IR (ITIR)

    >>> from gt4py.next import Field, Dimension
    >>> float64 = float
    >>> IDim = Dimension("IDim")
    >>> def field_op(inp: Field[[IDim], float64]):
    ...     return inp
    >>> foast_tree = FieldOperatorParser.apply_to_function(field_op)
    >>> foast_tree  # doctest: +ELLIPSIS
    FunctionDefinition(..., id=SymbolName('field_op'), ...)
    >>> foast_tree.params  # doctest: +ELLIPSIS
    [Symbol(..., id=SymbolName('inp'), type=FieldType(...), ...)]
    >>> foast_tree.body.stmts  # doctest: +ELLIPSIS
    [Return(..., value=Name(..., id=SymbolRef('inp')))]


    If a syntax error is encountered, it will point to the location in the source code.

    >>> def wrong_syntax(inp: Field[[IDim], int]):
    ...     for i in [1, 2, 3]:  # for is not part of the field operator syntax
    ...         tmp = inp
    ...     return tmp
    >>>
    >>> try:  # doctest: +ELLIPSIS
    ...     FieldOperatorParser.apply_to_function(wrong_syntax)
    ... except errors.DSLError as err:
    ...     print(
    ...         f"Error at [{err.location.line}, {err.location.column}] in {err.location.filename})"
    ...     )
    Error at [2, 5] in ...func_to_foast.FieldOperatorParser[...]>)
    """

    @classmethod
    def _preprocess_definition_ast(cls, ast: ast.AST) -> ast.AST:
        ast = StringifyAnnotationsPass.apply(ast)
        ast = SingleStaticAssignPass.apply(ast)
        ast = SingleAssignTargetPass.apply(ast)
        return ast

    @classmethod
    def _postprocess_dialect_ast(
        cls,
        foast_node: foast.FunctionDefinition | foast.FieldOperator,
        closure_vars: dict[str, Any],
        annotations: dict[str, Any],
    ) -> foast.FunctionDefinition:
        foast_node = ClosureVarFolding.apply(foast_node, closure_vars)
        foast_node = DeadClosureVarElimination.apply(foast_node)
        foast_node = ClosureVarTypeDeduction.apply(foast_node, closure_vars)
        foast_node = FieldOperatorTypeDeduction.apply(foast_node)
        foast_node = UnpackedAssignPass.apply(foast_node)

        # check deduced matches annotated return type
        if "return" in annotations:
            annotated_return_type = type_translation.from_type_hint(annotations["return"])
            # TODO(tehrengruber): use `type_info.return_type` when the type of the
            #  arguments becomes available here
            if annotated_return_type != foast_node.type.returns:  # type: ignore[union-attr] # revisit when `type_info.return_type` is implemented
                raise errors.DSLError(
                    foast_node.location,
                    "Annotated return type does not match deduced return type: annotation is "
                    f"'{annotated_return_type}'"  # type: ignore[union-attr] # revisit when 'type_info.return_type' is implemented
                    f", got '{foast_node.type.returns}'.",
                )
        return foast_node

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs: Any) -> foast.FunctionDefinition:
        loc = self.get_location(node)
        closure_var_symbols: list[foast.Symbol] = []
        for name in self.closure_vars.keys():
            try:
                type_ = type_translation.from_value(self.closure_vars[name])
                closure_var_symbols.append(
                    foast.Symbol(
                        id=name,
                        type=type_,
                        namespace=dialect_ast_enums.Namespace.CLOSURE,
                        location=self.get_location(node),
                    )
                )
            except ValueError as e:
                raise errors.DSLTypeError(
                    loc,
                    f"Unexpected object '{name}' of type '{type(self.closure_vars[name])}' encountered.",
                ) from e

        new_body = self._visit_stmts(node.body, self.get_location(node), **kwargs)

        if deduce_stmt_return_kind(new_body) == StmtReturnKind.NO_RETURN:
            raise errors.DSLError(loc, "'Function' is expected to return a value.")

        return foast.FunctionDefinition(
            id=node.name,
            params=self.visit(node.args, **kwargs),
            body=new_body,
            closure_vars=closure_var_symbols,
            location=loc,
        )

    def visit_arguments(self, node: ast.arguments) -> list[foast.DataSymbol]:
        return [self.visit_arg(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg) -> foast.DataSymbol:
        loc = self.get_location(node)
        if (annotation := self.annotations.get(node.arg, None)) is None:
            raise errors.MissingParameterAnnotationError(loc, node.arg)
        new_type = type_translation.from_type_hint(annotation)
        if not isinstance(new_type, ts.DataType):
            raise errors.InvalidParameterAnnotationError(loc, node.arg, new_type)
        return foast.DataSymbol(id=node.arg, location=loc, type=new_type)

    def visit_Assign(
        self, node: ast.Assign, **kwargs: Any
    ) -> foast.Assign | foast.TupleTargetAssign:
        target = node.targets[0]  # there is only one element after assignment passes

        if isinstance(target, ast.Tuple):
            new_targets: list[
                foast.FieldSymbol | foast.TupleSymbol | foast.ScalarSymbol | foast.Starred
            ] = []

            for elt in target.elts:
                if isinstance(elt, ast.Starred):
                    new_targets.append(
                        foast.Starred(
                            id=foast.DataSymbol(
                                id=self.visit(elt.value).id,
                                location=self.get_location(elt),
                                type=ts.DeferredType(constraint=ts.DataType),
                            ),
                            location=self.get_location(elt),
                            type=ts.DeferredType(constraint=ts.DataType),
                        )
                    )
                else:
                    new_targets.append(
                        foast.DataSymbol(
                            id=self.visit(elt).id,
                            location=self.get_location(elt),
                            type=ts.DeferredType(constraint=ts.DataType),
                        )
                    )

            return foast.TupleTargetAssign(
                targets=new_targets, value=self.visit(node.value), location=self.get_location(node)
            )

        if not isinstance(target, ast.Name):
            raise errors.DSLError(self.get_location(node), "Can only assign to names.")
        new_value = self.visit(node.value)
        constraint_type: Type[ts.DataType] = ts.DataType
        if isinstance(new_value, foast.TupleExpr):
            constraint_type = ts.TupleType
        elif (
            type_info.is_concrete(new_value.type)
            and type_info.type_class(new_value.type) is ts.ScalarType
        ):
            constraint_type = ts.ScalarType
        return foast.Assign(
            target=foast.DataSymbol(
                id=target.id,
                location=self.get_location(target),
                type=ts.DeferredType(constraint=constraint_type),
            ),
            value=new_value,
            location=self.get_location(node),
        )

    def visit_AnnAssign(self, node: ast.AnnAssign, **kwargs: Any) -> foast.Assign:
        if not isinstance(node.target, ast.Name):
            raise errors.DSLError(self.get_location(node), "Can only assign to names.")

        if node.annotation is not None:
            assert isinstance(node.annotation, ast.Constant) and isinstance(
                node.annotation.value, str
            ), "Annotations should be ast.Constant(string). Use StringifyAnnotationsPass"

            context = {**fbuiltins.BUILTINS, **self.closure_vars}
            annotation = eval(node.annotation.value, context)
            target_type = type_translation.from_type_hint(annotation, globalns=context)
        else:
            target_type = ts.DeferredType()

        return foast.Assign(
            target=foast.Symbol[ts.FieldType](
                id=node.target.id, location=self.get_location(node.target), type=target_type
            ),
            value=self.visit(node.value) if node.value else None,
            location=self.get_location(node),
        )

    def visit_Subscript(self, node: ast.Subscript, **kwargs: Any) -> foast.Subscript:
        return foast.Subscript(
            value=self.visit(node.value),
            index=self.visit(node.slice),
            location=self.get_location(node),
        )

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        return foast.Attribute(
            value=self.visit(node.value), attr=node.attr, location=self.get_location(node)
        )

    def visit_Tuple(self, node: ast.Tuple, **kwargs: Any) -> foast.TupleExpr:
        return foast.TupleExpr(
            elts=[self.visit(item) for item in node.elts], location=self.get_location(node)
        )

    def visit_Return(self, node: ast.Return, **kwargs: Any) -> foast.Return:
        loc = self.get_location(node)
        if not node.value:
            raise errors.DSLError(loc, "Must return a value, not None")
        return foast.Return(value=self.visit(node.value), location=loc)

    def visit_Expr(self, node: ast.Expr) -> foast.Expr:
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name, **kwargs: Any) -> foast.Name:
        return foast.Name(id=node.id, location=self.get_location(node))

    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs: Any) -> foast.UnaryOp:
        return foast.UnaryOp(
            op=self.visit(node.op),
            operand=self.visit(node.operand),
            location=self.get_location(node),
        )

    def visit_UAdd(self, node: ast.UAdd, **kwargs: Any) -> dialect_ast_enums.UnaryOperator:
        return dialect_ast_enums.UnaryOperator.UADD

    def visit_USub(self, node: ast.USub, **kwargs: Any) -> dialect_ast_enums.UnaryOperator:
        return dialect_ast_enums.UnaryOperator.USUB

    def visit_Not(self, node: ast.Not, **kwargs: Any) -> dialect_ast_enums.UnaryOperator:
        return dialect_ast_enums.UnaryOperator.NOT

    def visit_Invert(self, node: ast.Invert, **kwargs: Any) -> dialect_ast_enums.UnaryOperator:
        return dialect_ast_enums.UnaryOperator.INVERT

    def visit_BinOp(self, node: ast.BinOp, **kwargs: Any) -> foast.BinOp:
        return foast.BinOp(
            op=self.visit(node.op),
            left=self.visit(node.left),
            right=self.visit(node.right),
            location=self.get_location(node),
        )

    def visit_Add(self, node: ast.Add, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.MULT

    def visit_Div(self, node: ast.Div, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.DIV

    def visit_FloorDiv(self, node: ast.FloorDiv, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.FLOOR_DIV

    def visit_Pow(self, node: ast.Pow, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.POW

    def visit_Mod(self, node: ast.Mod, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.MOD

    def visit_BitAnd(self, node: ast.BitAnd, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_AND

    def visit_BitOr(self, node: ast.BitOr, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_OR

    def visit_BitXor(self, node: ast.BitXor, **kwargs: Any) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_XOR

    def visit_BoolOp(self, node: ast.BoolOp, **kwargs: Any) -> None:
        raise errors.UnsupportedPythonFeatureError(
            self.get_location(node), "logical operators `and`, `or`"
        )

    def visit_IfExp(self, node: ast.IfExp, **kwargs: Any) -> foast.TernaryExpr:
        return foast.TernaryExpr(
            condition=self.visit(node.test),
            true_expr=self.visit(node.body),
            false_expr=self.visit(node.orelse),
            location=self.get_location(node),
            type=ts.DeferredType(constraint=ts.DataType),
        )

    def visit_If(self, node: ast.If, **kwargs: Any) -> foast.IfStmt:
        loc = self.get_location(node)
        return foast.IfStmt(
            condition=self.visit(node.test, **kwargs),
            true_branch=self._visit_stmts(node.body, loc, **kwargs),
            false_branch=self._visit_stmts(node.orelse, loc, **kwargs),
            location=loc,
        )

    def _visit_stmts(
        self, stmts: list[ast.stmt], location: eve.SourceLocation, **kwargs: Any
    ) -> foast.BlockStmt:
        return foast.BlockStmt(
            stmts=[self.visit(el, **kwargs) for el in stmts if not isinstance(el, ast.Pass)],
            location=location,
        )

    def visit_Compare(self, node: ast.Compare, **kwargs: Any) -> foast.Compare:
        loc = self.get_location(node)

        if len(node.ops) != 1 or len(node.comparators) != 1:
            refactored = UnchainComparesPass.apply(node)
            raise errors.DSLError(
                loc,
                textwrap.dedent(
                    f"""
                    Comparison chains are not allowed. Please replace
                        {ast.unparse(node)}
                    by
                        {ast.unparse(refactored)}
                    """,
                ),
            )
        return foast.Compare(
            op=self.visit(node.ops[0]),
            left=self.visit(node.left),
            right=self.visit(node.comparators[0]),
            location=loc,
        )

    def visit_Gt(self, node: ast.Gt, **kwargs: Any) -> foast.CompareOperator:
        return foast.CompareOperator.GT

    def visit_Lt(self, node: ast.Lt, **kwargs: Any) -> foast.CompareOperator:
        return foast.CompareOperator.LT

    def visit_Eq(self, node: ast.Eq, **kwargs: Any) -> foast.CompareOperator:
        return foast.CompareOperator.EQ

    def visit_LtE(self, node: ast.LtE, **kwargs: Any) -> foast.CompareOperator:
        return foast.CompareOperator.LTE

    def visit_GtE(self, node: ast.GtE, **kwargs: Any) -> foast.CompareOperator:
        return foast.CompareOperator.GTE

    def visit_NotEq(self, node: ast.NotEq, **kwargs: Any) -> foast.CompareOperator:
        return foast.CompareOperator.NOTEQ

    def visit_Dict(self, node: ast.Dict) -> foast.Dict:
        return foast.Dict(
            keys_=[self.visit(cast(ast.AST, param)) for param in node.keys],
            values_=[self.visit(param) for param in node.values],
            location=self.get_location(node),
        )

    def _verify_builtin_type_constructor(self, node: ast.Call) -> None:
        if len(node.args) > 0:
            arg = node.args[0]
            if not (
                isinstance(arg, ast.Constant)
                or (isinstance(arg, ast.UnaryOp) and isinstance(arg.operand, ast.Constant))
            ):
                raise errors.DSLError(
                    self.get_location(node),
                    f"'{self._func_name(node)}()' only takes literal arguments.",
                )

    def _func_name(self, node: ast.Call) -> str:
        return node.func.id  # type: ignore[attr-defined] # We want this to fail if the attribute does not exist unexpectedly.

    def visit_Call(self, node: ast.Call, **kwargs: Any) -> foast.Call:
        # TODO(tehrengruber): is this still needed or redundant with the checks in type deduction?
        if isinstance(node.func, ast.Name):
            func_name = self._func_name(node)
            if func_name in fbuiltins.TYPE_BUILTIN_NAMES:
                self._verify_builtin_type_constructor(node)

        return foast.Call(
            func=self.visit(node.func, **kwargs),
            args=[self.visit(arg, **kwargs) for arg in node.args],
            kwargs={keyword.arg: self.visit(keyword.value, **kwargs) for keyword in node.keywords},
            location=self.get_location(node),
        )

    def visit_Constant(self, node: ast.Constant, **kwargs: Any) -> foast.Constant:
        loc = self.get_location(node)
        try:
            type_ = type_translation.from_value(node.value)
        except ValueError:
            raise errors.DSLError(
                loc, f"Constants of type {type(node.value)} are not permitted."
            ) from None

        return foast.Constant(value=node.value, location=loc, type=type_)
