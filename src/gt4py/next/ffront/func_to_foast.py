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

from __future__ import annotations

import ast
import builtins
from typing import Any, Callable, Iterable, Mapping, Type, cast

import gt4py.eve as eve
from gt4py.next import errors
from gt4py.next.ffront import dialect_ast_enums, fbuiltins, field_operator_ast as foast
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
from gt4py.next.ffront.foast_passes.type_alias_replacement import TypeAliasReplacement
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


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
    ...     for i in [1, 2, 3]: # for is not part of the field operator syntax
    ...         tmp = inp
    ...     return tmp
    >>>
    >>> try:                # doctest: +ELLIPSIS
    ...     FieldOperatorParser.apply_to_function(wrong_syntax)
    ... except errors.DSLError as err:
    ...     print(f"Error at [{err.location.line}, {err.location.column}] in {err.location.filename})")
    Error at [2, 5] in ...func_to_foast.FieldOperatorParser[...]>)
    """

    @classmethod
    def _preprocess_definition_ast(cls, definition_ast: ast.AST) -> ast.AST:
        sta = StringifyAnnotationsPass.apply(definition_ast)
        ssa = SingleStaticAssignPass.apply(sta)
        sat = SingleAssignTargetPass.apply(ssa)
        ucc = UnchainComparesPass.apply(sat)
        return ucc

    @classmethod
    def _postprocess_dialect_ast(
        cls,
        foast_node: foast.FunctionDefinition | foast.FieldOperator,
        closure_vars: dict[str, Any],
        annotations: dict[str, Any],
    ) -> foast.FunctionDefinition:
        foast_node, closure_vars = TypeAliasReplacement.apply(foast_node, closure_vars)
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
                    "Annotated return type does not match deduced return type: expected "
                    f"'{foast_node.type.returns}'"  # type: ignore[union-attr] # revisit when 'type_info.return_type' is implemented
                    f", got '{annotated_return_type}'.",
                )
        return foast_node

    def _builtin_type_constructor_symbols(
        self, captured_vars: Mapping[str, Any], location: eve.SourceLocation
    ) -> tuple[list[foast.Symbol], Iterable[str]]:
        result: list[foast.Symbol] = []
        skipped_types = {"tuple"}
        python_type_builtins: dict[str, Callable[[Any], Any]] = {
            name: getattr(builtins, name)
            for name in set(fbuiltins.TYPE_BUILTIN_NAMES) - skipped_types
            if hasattr(builtins, name)
        }
        captured_type_builtins = {
            name: value
            for name, value in captured_vars.items()
            if name in fbuiltins.TYPE_BUILTIN_NAMES and value is getattr(fbuiltins, name)
        }
        to_be_inserted = python_type_builtins | captured_type_builtins
        for name, value in to_be_inserted.items():
            result.append(
                foast.Symbol(
                    id=name,
                    type=ts.FunctionType(
                        pos_only_args=[
                            ts.DeferredType(constraint=ts.ScalarType)
                        ],  # this is a constraint type that will not be inferred (as the function is polymorphic)
                        pos_or_kw_args={},
                        kw_only_args={},
                        returns=cast(ts.DataType, type_translation.from_type_hint(value)),
                    ),
                    namespace=dialect_ast_enums.Namespace.CLOSURE,
                    location=location,
                )
            )

        return result, to_be_inserted.keys()

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs) -> foast.FunctionDefinition:
        loc = self.get_location(node)
        closure_var_symbols, skip_names = self._builtin_type_constructor_symbols(
            self.closure_vars, self.get_location(node)
        )
        for name in self.closure_vars.keys():
            if name in skip_names:
                continue
            closure_var_symbols.append(
                foast.Symbol(
                    id=name,
                    type=ts.DeferredType(constraint=None),
                    namespace=dialect_ast_enums.Namespace.CLOSURE,
                    location=self.get_location(node),
                )
            )

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

    def visit_Assign(self, node: ast.Assign, **kwargs) -> foast.Assign | foast.TupleTargetAssign:
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

    def visit_AnnAssign(self, node: ast.AnnAssign, **kwargs) -> foast.Assign:
        if not isinstance(node.target, ast.Name):
            raise errors.DSLError(self.get_location(node), "Can only assign to names.")

        if node.annotation is not None:
            assert isinstance(
                node.annotation, ast.Constant
            ), "Annotations should be ast.Constant(string). Use StringifyAnnotationsPass"
            context = {**fbuiltins.BUILTINS, **self.closure_vars}
            annotation = eval(node.annotation.value, context)
            target_type = type_translation.from_type_hint(annotation, globalns=context)
        else:
            target_type = ts.DeferredType()

        return foast.Assign(
            target=foast.Symbol[ts.FieldType](
                id=node.target.id,
                location=self.get_location(node.target),
                type=target_type,
            ),
            value=self.visit(node.value) if node.value else None,
            location=self.get_location(node),
        )

    @staticmethod
    def _match_index(node: ast.expr) -> int:
        if isinstance(node, ast.Constant):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.unaryop)
            and isinstance(node.operand, ast.Constant)
        ):
            if isinstance(node.op, ast.USub):
                return -node.operand.value
            if isinstance(node.op, ast.UAdd):
                return node.operand.value
        raise ValueError(f"Not an index: '{node}'.")

    def visit_Subscript(self, node: ast.Subscript, **kwargs) -> foast.Subscript:
        try:
            index = self._match_index(node.slice)
        except ValueError:
            raise errors.DSLError(
                self.get_location(node.slice), "eXpected an integral index."
            ) from None

        return foast.Subscript(
            value=self.visit(node.value),
            index=index,
            location=self.get_location(node),
        )

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        return foast.Attribute(
            value=self.visit(node.value), attr=node.attr, location=self.get_location(node)
        )

    def visit_Tuple(self, node: ast.Tuple, **kwargs) -> foast.TupleExpr:
        return foast.TupleExpr(
            elts=[self.visit(item) for item in node.elts], location=self.get_location(node)
        )

    def visit_Return(self, node: ast.Return, **kwargs) -> foast.Return:
        loc = self.get_location(node)
        if not node.value:
            raise errors.DSLError(loc, "Must return a value, not None")
        return foast.Return(value=self.visit(node.value), location=loc)

    def visit_Expr(self, node: ast.Expr) -> foast.Expr:
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name, **kwargs) -> foast.Name:
        return foast.Name(id=node.id, location=self.get_location(node))

    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs) -> foast.UnaryOp:
        return foast.UnaryOp(
            op=self.visit(node.op),
            operand=self.visit(node.operand),
            location=self.get_location(node),
        )

    def visit_UAdd(self, node: ast.UAdd, **kwargs) -> dialect_ast_enums.UnaryOperator:
        return dialect_ast_enums.UnaryOperator.UADD

    def visit_USub(self, node: ast.USub, **kwargs) -> dialect_ast_enums.UnaryOperator:
        return dialect_ast_enums.UnaryOperator.USUB

    def visit_Not(self, node: ast.Not, **kwargs) -> dialect_ast_enums.UnaryOperator:
        return dialect_ast_enums.UnaryOperator.NOT

    def visit_Invert(self, node: ast.Invert, **kwargs) -> dialect_ast_enums.UnaryOperator:
        return dialect_ast_enums.UnaryOperator.INVERT

    def visit_BinOp(self, node: ast.BinOp, **kwargs) -> foast.BinOp:
        return foast.BinOp(
            op=self.visit(node.op),
            left=self.visit(node.left),
            right=self.visit(node.right),
            location=self.get_location(node),
        )

    def visit_Add(self, node: ast.Add, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.MULT

    def visit_Div(self, node: ast.Div, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.DIV

    def visit_FloorDiv(self, node: ast.FloorDiv, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.FLOOR_DIV

    def visit_Pow(self, node: ast.Pow, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.POW

    def visit_Mod(self, node: ast.Mod, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.MOD

    def visit_BitAnd(self, node: ast.BitAnd, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_AND

    def visit_BitOr(self, node: ast.BitOr, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_OR

    def visit_BitXor(self, node: ast.BitXor, **kwargs) -> dialect_ast_enums.BinaryOperator:
        return dialect_ast_enums.BinaryOperator.BIT_XOR

    def visit_BoolOp(self, node: ast.BoolOp, **kwargs) -> None:
        raise errors.UnsupportedPythonFeatureError(
            self.get_location(node), "logical operators `and`, `or`"
        )

    def visit_IfExp(self, node: ast.IfExp, **kwargs) -> foast.TernaryExpr:
        return foast.TernaryExpr(
            condition=self.visit(node.test),
            true_expr=self.visit(node.body),
            false_expr=self.visit(node.orelse),
            location=self.get_location(node),
            type=ts.DeferredType(constraint=ts.DataType),
        )

    def visit_If(self, node: ast.If, **kwargs) -> foast.IfStmt:
        loc = self.get_location(node)
        return foast.IfStmt(
            condition=self.visit(node.test, **kwargs),
            true_branch=self._visit_stmts(node.body, loc, **kwargs),
            false_branch=self._visit_stmts(node.orelse, loc, **kwargs),
            location=loc,
        )

    def _visit_stmts(
        self, stmts: list[ast.stmt], location: eve.SourceLocation, **kwargs
    ) -> foast.BlockStmt:
        return foast.BlockStmt(
            stmts=[self.visit(el, **kwargs) for el in stmts if not isinstance(el, ast.Pass)],
            location=location,
        )

    def visit_Compare(self, node: ast.Compare, **kwargs) -> foast.Compare:
        loc = self.get_location(node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            # Remove comparison chains in a preprocessing pass
            # TODO: maybe add a note to the error about preprocessing passes?
            raise errors.UnsupportedPythonFeatureError(loc, "comparison chains")
        return foast.Compare(
            op=self.visit(node.ops[0]),
            left=self.visit(node.left),
            right=self.visit(node.comparators[0]),
            location=loc,
        )

    def visit_Gt(self, node: ast.Gt, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.GT

    def visit_Lt(self, node: ast.Lt, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.LT

    def visit_Eq(self, node: ast.Eq, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.EQ

    def visit_LtE(self, node: ast.LtE, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.LTE

    def visit_GtE(self, node: ast.GtE, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.GTE

    def visit_NotEq(self, node: ast.NotEq, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.NOTEQ

    def _verify_builtin_type_constructor(self, node: ast.Call):
        if len(node.args) > 0 and not isinstance(node.args[0], ast.Constant):
            raise errors.DSLError(
                self.get_location(node),
                f"'{self._func_name(node)}()' only takes literal arguments.",
            )

    def _func_name(self, node: ast.Call) -> str:
        return node.func.id  # type: ignore[attr-defined] # We want this to fail if the attribute does not exist unexpectedly.

    def visit_Call(self, node: ast.Call, **kwargs) -> foast.Call:
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

    def visit_Constant(self, node: ast.Constant, **kwargs) -> foast.Constant:
        loc = self.get_location(node)
        try:
            type_ = type_translation.from_value(node.value)
        except ValueError:
            raise errors.DSLError(
                loc, f"Constants of type {type(node.value)} are not permitted."
            ) from None

        return foast.Constant(
            value=node.value,
            location=loc,
            type=type_,
        )
