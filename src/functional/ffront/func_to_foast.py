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

from __future__ import annotations

import ast
import builtins
import collections
from typing import Any, Callable, Iterable, Mapping, Type, cast

import eve
from functional import common
from functional.ffront import (
    common_types as ct,
    fbuiltins,
    field_operator_ast as foast,
    symbol_makers,
    type_info,
)
from functional.ffront.ast_passes import (
    SingleAssignTargetPass,
    SingleStaticAssignPass,
    StringifyAnnotationsPass,
    UnchainComparesPass,
    UnpackedAssignPass,
)
from functional.ffront.dialect_parser import DialectParser, DialectSyntaxError
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction


class FieldOperatorSyntaxError(DialectSyntaxError):
    dialect_name = "Field Operator"


class FieldOperatorParser(DialectParser[foast.FunctionDefinition]):
    """
    Parse field operator function definition from source code into FOAST.

    Catch any Field Operator specific syntax errors and typing problems.

    Example
    -------
    Parse a function into a Field Operator AST (FOAST), which can
    be lowered into Iterator IR (ITIR)

    >>> from functional.common import Field
    >>> float64 = float
    >>> def field_op(inp: Field[..., float64]):
    ...     return inp
    >>> foast_tree = FieldOperatorParser.apply_to_function(field_op)
    >>> foast_tree  # doctest: +ELLIPSIS
    FunctionDefinition(..., id=SymbolName('field_op'), ...)
    >>> foast_tree.params  # doctest: +ELLIPSIS
    [Symbol(..., id=SymbolName('inp'), type=FieldType(...), ...)]
    >>> foast_tree.body  # doctest: +ELLIPSIS
    [Return(..., value=Name(..., id=SymbolRef('inp')))]


    If a syntax error is encountered, it will point to the location in the source code.

    >>> def wrong_syntax(inp: Field[..., int]):
    ...     for i in [1, 2, 3]: # for is not part of the field operator syntax
    ...         tmp = inp
    ...     return tmp
    >>>
    >>> try:                # doctest: +ELLIPSIS
    ...     FieldOperatorParser.apply_to_function(wrong_syntax)
    ... except FieldOperatorSyntaxError as err:
    ...     print(f"Error at [{err.lineno}, {err.offset}] in {err.filename})")
    Error at [2, 4] in ...functional.ffront.func_to_foast.FieldOperatorParser[...]>)
    """

    syntax_error_cls = FieldOperatorSyntaxError

    @classmethod
    def _preprocess_definition_ast(cls, definition_ast: ast.AST) -> ast.AST:
        sta = StringifyAnnotationsPass.apply(definition_ast)
        ssa = SingleStaticAssignPass.apply(sta)
        sat = SingleAssignTargetPass.apply(ssa)
        las = UnpackedAssignPass.apply(sat)
        ucc = UnchainComparesPass.apply(las)
        return ucc

    @classmethod
    def _postprocess_dialect_ast(cls, dialect_ast: foast.FieldOperator) -> foast.FieldOperator:
        return FieldOperatorTypeDeduction.apply(dialect_ast)

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
                    type=ct.FunctionType(
                        args=[
                            ct.DeferredSymbolType(constraint=ct.ScalarType)
                        ],  # this is a constraint type that will not be inferred (as the function is polymorphic)
                        kwargs={},
                        returns=cast(
                            ct.DataType, symbol_makers.make_symbol_type_from_typing(value)
                        ),
                    ),
                    namespace=ct.Namespace.CLOSURE,
                    location=location,
                )
            )

        return result, to_be_inserted.keys()

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs) -> foast.FunctionDefinition:
        captured_vars: Mapping[str, Any] = collections.ChainMap(
            self.captured_vars.globals, self.captured_vars.nonlocals
        )
        captured_symbols, skip_names = self._builtin_type_constructor_symbols(
            captured_vars, self._make_loc(node)
        )
        for name, val in captured_vars.items():
            if name in skip_names:
                continue
            captured_symbols.append(
                foast.Symbol(
                    id=name,
                    type=symbol_makers.make_symbol_type_from_value(val),
                    namespace=ct.Namespace.CLOSURE,
                    location=self._make_loc(node),
                )
            )

        return foast.FunctionDefinition(
            id=node.name,
            params=self.visit(node.args, **kwargs),
            body=self.visit_stmt_list(node.body, **kwargs),
            captured_vars=captured_symbols,
            location=self._make_loc(node),
        )

    def visit_Import(self, node: ast.Import, **kwargs) -> None:
        raise FieldOperatorSyntaxError.from_AST(
            node, msg=f"Only 'from' imports from {fbuiltins.MODULE_BUILTIN_NAMES} are supported"
        )

    def visit_ImportFrom(self, node: ast.ImportFrom, **kwargs) -> foast.ExternalImport:
        if node.module not in fbuiltins.MODULE_BUILTIN_NAMES:
            raise FieldOperatorSyntaxError.from_AST(
                node,
                msg=f"Only 'from' imports from {fbuiltins.MODULE_BUILTIN_NAMES} are supported",
            )

        symbols: list[foast.Symbol] = []

        if node.module == fbuiltins.EXTERNALS_MODULE_NAME:
            for alias in node.names:
                if alias.name not in self.externals_defs:
                    raise FieldOperatorSyntaxError.from_AST(
                        node, msg=f"Missing symbol '{alias.name}' definition in {node.module}"
                    )
                symbols.append(
                    foast.Symbol(
                        id=alias.asname or alias.name,
                        type=symbol_makers.make_symbol_type_from_value(
                            self.externals_defs[alias.name]
                        ),
                        namespace=ct.Namespace.EXTERNAL,
                        location=self._make_loc(node),
                    )
                )

        return foast.ExternalImport(symbols=symbols, location=self._make_loc(node))

    def visit_arguments(self, node: ast.arguments) -> list[foast.DataSymbol]:
        return [self.visit_arg(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg) -> foast.DataSymbol:
        if (annotation := self.captured_vars.annotations.get(node.arg, None)) is None:
            raise FieldOperatorSyntaxError.from_AST(node, msg="Untyped parameters not allowed!")
        new_type = symbol_makers.make_symbol_type_from_typing(annotation)
        if not isinstance(new_type, ct.DataType):
            raise FieldOperatorSyntaxError.from_AST(
                node, msg="Only arguments of type DataType are allowed."
            )
        return foast.DataSymbol(id=node.arg, location=self._make_loc(node), type=new_type)

    def visit_Assign(self, node: ast.Assign, **kwargs) -> foast.Assign:
        target = node.targets[0]  # there is only one element after assignment passes
        if isinstance(target, ast.Tuple):
            raise FieldOperatorSyntaxError.from_AST(
                node, msg="Unpacking not allowed, run a preprocessing pass!"
            )
        if not isinstance(target, ast.Name):
            raise FieldOperatorSyntaxError.from_AST(node, msg="Can only assign to names!")
        new_value = self.visit(node.value)
        constraint_type: Type[ct.DataType] = ct.DataType
        if isinstance(new_value, foast.TupleExpr):
            constraint_type = ct.TupleType
        elif (
            type_info.is_concrete(new_value.type)
            and type_info.type_class(new_value.type) is ct.ScalarType
        ):
            constraint_type = ct.ScalarType
        return foast.Assign(
            target=foast.DataSymbol(
                id=target.id,
                location=self._make_loc(target),
                type=ct.DeferredSymbolType(constraint=constraint_type),
            ),
            value=new_value,
            location=self._make_loc(node),
        )

    def visit_AnnAssign(self, node: ast.AnnAssign, **kwargs) -> foast.Assign:
        if not isinstance(node.target, ast.Name):
            raise FieldOperatorSyntaxError.from_AST(node, msg="Can only assign to names!")

        if node.annotation is not None:
            assert isinstance(
                node.annotation, ast.Constant
            ), "Annotations should be ast.Constant(string). Use StringifyAnnotationsPass"
            globalns = {**fbuiltins.BUILTINS, **self.captured_vars.globals}
            localns = self.captured_vars.nonlocals
            annotation = eval(node.annotation.value, globalns, localns)
            target_type = symbol_makers.make_symbol_type_from_typing(
                annotation, globalns=globalns, localns=localns
            )
        else:
            target_type = ct.DeferredSymbolType()

        return foast.Assign(
            target=foast.Symbol[ct.FieldType](
                id=node.target.id,
                location=self._make_loc(node.target),
                type=target_type,
            ),
            value=self.visit(node.value) if node.value else None,
            location=self._make_loc(node),
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
        raise ValueError(f"Not an index: {node}")

    def visit_Subscript(self, node: ast.Subscript, **kwargs) -> foast.Subscript:
        try:
            index = self._match_index(node.slice)
        except ValueError:
            raise FieldOperatorSyntaxError.from_AST(
                node, msg="""Only index is supported in subscript!"""
            )

        return foast.Subscript(
            value=self.visit(node.value),
            index=index,
            location=self._make_loc(node),
        )

    def visit_Tuple(self, node: ast.Tuple, **kwargs) -> foast.TupleExpr:
        return foast.TupleExpr(
            elts=[self.visit(item) for item in node.elts], location=self._make_loc(node)
        )

    def visit_Return(self, node: ast.Return, **kwargs) -> foast.Return:
        if not node.value:
            raise FieldOperatorSyntaxError.from_AST(node, msg="Empty return not allowed")
        return foast.Return(value=self.visit(node.value), location=self._make_loc(node))

    def visit_stmt_list(self, nodes: list[ast.stmt], **kwargs) -> list[foast.Expr]:
        if not isinstance(last_node := nodes[-1], ast.Return):
            raise FieldOperatorSyntaxError.from_AST(
                last_node, msg="Field operator must return a field expression on the last line!"
            )
        return [self.visit(node, **kwargs) for node in nodes]

    def visit_Name(self, node: ast.Name, **kwargs) -> foast.Name:
        return foast.Name(id=node.id, location=self._make_loc(node))

    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs) -> foast.UnaryOp:
        return foast.UnaryOp(
            op=self.visit(node.op), operand=self.visit(node.operand), location=self._make_loc(node)
        )

    def visit_UAdd(self, node: ast.UAdd, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.UADD

    def visit_USub(self, node: ast.USub, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.USUB

    def visit_Not(self, node: ast.Not, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.NOT

    def visit_BinOp(self, node: ast.BinOp, **kwargs) -> foast.BinOp:
        new_op = self.visit(node.op)
        return foast.BinOp(
            op=new_op,
            left=self.visit(node.left),
            right=self.visit(node.right),
            location=self._make_loc(node),
        )

    def visit_Add(self, node: ast.Add, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.MULT

    def visit_Div(self, node: ast.Div, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.DIV

    def visit_Pow(self, node: ast.Pow, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.POW

    def visit_Mod(self, node: ast.Mod, **kwargs) -> None:
        raise FieldOperatorSyntaxError.from_AST(node, msg="`%` operator not supported!")

    def visit_BitAnd(self, node: ast.BitAnd, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.BIT_AND

    def visit_BitOr(self, node: ast.BitOr, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.BIT_OR

    def visit_BoolOp(self, node: ast.BoolOp, **kwargs) -> None:
        raise FieldOperatorSyntaxError.from_AST(node, msg="`and`/`or` operator not allowed!")

    def visit_Compare(self, node: ast.Compare, **kwargs) -> foast.Compare:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise FieldOperatorSyntaxError.from_AST(
                node, msg="Comparison chains not allowed, run a preprocessing pass!"
            )
        return foast.Compare(
            op=self.visit(node.ops[0]),
            left=self.visit(node.left),
            right=self.visit(node.comparators[0]),
            location=self._make_loc(node),
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

    def _verify_builtin_function(self, node: ast.Call):
        func_name = self._func_name(node)
        func_info = getattr(fbuiltins, func_name).__gt_type__()
        if not len(node.args) == len(func_info.args):
            raise FieldOperatorSyntaxError.from_AST(
                node,
                msg=f"{func_name}() expected {len(func_info.args)} positional arguments, {len(node.args)} given!",
            )
        elif unexpected_kwargs := set(k.arg for k in node.keywords) - set(func_info.kwargs):
            raise FieldOperatorSyntaxError.from_AST(
                node,
                msg=f"{self._func_name(node)}() got unexpected keyword arguments: {unexpected_kwargs}!",
            )

    def _verify_builtin_type_constructor(self, node: ast.Call):
        if not len(node.args) == 1:
            raise FieldOperatorSyntaxError.from_AST(
                node,
                msg=f"{self._func_name(node)}() expected 1 positional argument, {len(node.args)} given!",
            )
        elif node.keywords:
            unexpected_kwargs = set(k.arg for k in node.keywords)
            raise FieldOperatorSyntaxError.from_AST(
                node,
                msg=f"{self._func_name(node)}() got unexpected keyword arguments: {unexpected_kwargs}!",
            )
        elif not isinstance(node.args[0], ast.Constant):
            raise FieldOperatorSyntaxError.from_AST(
                node,
                msg=f"{self._func_name(node)}() only takes literal arguments!",
            )

    def _func_name(self, node: ast.Call) -> str:
        return node.func.id  # type: ignore[attr-defined]  # We want this to fail if the attribute does not exist unexpectedly.

    def visit_Call(self, node: ast.Call, **kwargs) -> foast.Call:
        if not isinstance(node.func, ast.Name):
            raise FieldOperatorSyntaxError.from_AST(
                node, msg="Functions can only be called directly!"
            )

        func_name = self._func_name(node)

        if func_name in fbuiltins.FUN_BUILTIN_NAMES:
            self._verify_builtin_function(node)
        if func_name in fbuiltins.TYPE_BUILTIN_NAMES:
            self._verify_builtin_type_constructor(node)

        return foast.Call(
            func=self.visit(node.func, **kwargs),
            args=[self.visit(arg, **kwargs) for arg in node.args],
            kwargs={keyword.arg: self.visit(keyword.value, **kwargs) for keyword in node.keywords},
            location=self._make_loc(node),
        )

    def visit_Constant(self, node: ast.Constant, **kwargs) -> foast.Constant:
        try:
            type_ = symbol_makers.make_symbol_type_from_value(node.value)
        except common.GTTypeError as e:
            raise FieldOperatorSyntaxError.from_AST(
                node, msg=f"Constants of type {type(node.value)} are not permitted."
            ) from e

        return foast.Constant(
            value=node.value,
            location=self._make_loc(node),
            type=type_,
        )
