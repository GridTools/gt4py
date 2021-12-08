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
import copy
import functools
import inspect
import symtable
import textwrap
import types
import typing
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from inspect import ClosureVars
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


from eve import visitors
from eve.type_definitions import SourceLocation
from functional import common
from functional.ffront import fbuiltins, field_operator_ast as foast
from functional.ffront.ast_passes import (
    SingleAssignTargetPass,
    SingleStaticAssignPass,
    UnpackedAssignPass,
)
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from functional.ffront.type_parser import FieldOperatorTypeParser


MISSING_FILENAME = "<string>"


def _make_symbol_from_value(name: str, value: Any, namespace: foast.Namespace) -> foast.Symbol:
    symbol_type = _make_type_from_value(value)
    match symbol_type:
        case foast.ScalarType() | foast.TupleType():
            return foast.DataSymbol(
                id=name,
                type=symbol_type,
                namespace=namespace,
                location=SourceLocation.empty_location(),
            )
        case foast.FunctionType():
            return foast.Function(
                id=name, type=symbol_type, namespace=namespace, params=[], returns=[]
            )
        case _:
            raise common.GTTypeError(f"Impossible to map '{value}' value to a Symbol")


def _make_type_from_value(value: Any) -> foast.SymbolType:
    match value:
        case bool() | int() | float() | np.generic():
            return _make_type_from_value(type(value))
        case type() as t if issubclass(t, (bool, int, float, np.generic)):
            return foast.ScalarType(kind=_make_scalar_kind_from_value(value))
        case tuple() as tuple_value:
            return foast.TupleType(types=[_make_type_from_value(t) for t in tuple_value])
        case common.Field():
            return foast.FieldType(..., foast.ScalarKind.FLOAT64)
        case types.FunctionType():
            args = []
            kwargs = []
            returns = []
            return foast.FunctionType(args, kwargs, returns)
        case other if other.__module__ == "typing":
            return _make_type_from_value(other.__origin__)
        case _:
            raise common.GTTypeError(f"Impossible to map '{value}' value to a SymbolType")


def _make_scalar_kind_from_value(value: npt.DTypeLike) -> foast.ScalarKind:
    try:
        dt = np.dtype(value)
    except TypeError as err:
        raise common.GTTypeError(f"Invalid scalar type definition ({value})") from err

    if dt.shape == () and dt.fields is None:
        match dt:
            case np.bool_:
                return foast.ScalarKind.BOOL
            case np.int32:
                return foast.ScalarKind.INT32
            case np.int64:
                return foast.ScalarKind.INT64
            case np.float32:
                return foast.ScalarKind.FLOAT32
            case np.float64:
                return foast.ScalarKind.FLOAT64
            case _:
                raise common.GTTypeError(f"Impossible to map '{value}' value to a ScalarKind")
    else:
        raise common.GTTypeError(f"Non-trivial dtypes like '{value}' are not yet supported")


@dataclass(frozen=True)
class SourceDefinition:
    source: str
    filename: str = MISSING_FILENAME
    starting_line: int = 1

    @staticmethod
    def from_function(func: Callable) -> SourceDefinition:
        try:
            filename = inspect.getabsfile(func) or MISSING_FILENAME
            source = textwrap.dedent(inspect.getsource(func))
            starting_line = (
                inspect.getsourcelines(func)[1] if not filename.endswith(MISSING_FILENAME) else 1
            )
        except OSError as err:
            if filename.endswith(MISSING_FILENAME):
                message = (
                    "Can not create field operator from a function that is not in a source file!"
                )
            else:
                message = f"Can not get source code of passed function ({func})"
            raise ValueError(message) from err

        return SourceDefinition(source, filename, starting_line)

    def __iter__(self) -> Iterator[tuple[str]]:
        yield self.source
        yield self.filename
        yield self.starting_line


@dataclass(frozen=True)
class SymbolNames:
    params: tuple[str, ...]
    locals: tuple[str, ...]
    imported: tuple[str, ...]
    globals: tuple[str, ...]

    @functools.cached_property
    def all_locals(self) -> tuple[str]:
        return self.params + self.locals + self.imported

    @staticmethod
    def from_source(source: str, filename: str = MISSING_FILENAME) -> SymbolNames:
        try:
            mod_st = symtable.symtable(source, filename, "exec")
        except SyntaxError as err:
            raise common.GTValueError(
                f"Unexpected error when parsing provided source code (\n{source}\n)"
            ) from err

        assert mod_st.get_type() == "module"
        if len(children := mod_st.get_children()) != 1:
            raise common.GTValueError(
                f"Sources with multiple function definitions are not yet supported (\n{source}\n)"
            )

        func_st = children[0]
        if func_st.get_frees() or func_st.get_nonlocals():
            raise common.GTValueError(
                f"Sources with function closures are not yet supported (\n{source}\n)"
            )

        param_names = set()
        imported_names = set()
        local_names = set()
        for name in func_st.get_locals():
            if (s := func_st.lookup(name)).is_imported():
                imported_names.add(name)
            elif s.is_parameter:
                param_names.add(name)
            else:
                local_names.add()

        global_names = set(func_st.get_globals())

        return SymbolNames(param_names, local_names, imported_names, global_names)

    def __iter__(self) -> Iterator[tuple[str]]:
        yield self.params
        yield self.locals
        yield self.imported
        yield self.globals


# TODO (ricoh): pass on source locations
# TODO (ricoh): SourceLocation.source <- filename
@dataclass(frozen=True, kw_only=True)
class FieldOperatorParser(visitors.ASTNodeVisitor):
    """
    Parse field operator function definition from source code into FOAST.

    Catch any Field Operator specific syntax errors and typing problems.

    Example
    -------
    Parse a function into a Field Operator AST (FOAST), which can
    be lowered into Iterator IR (ITIR)

    >>> from functional.common import Field
    >>> float64 = float
    >>> def fieldop(inp: Field[..., float64]):
    ...     return inp
    >>> foast_tree = FieldOperatorParser.apply_to_func(fieldop)
    >>> foast_tree  # doctest: +ELLIPSIS
    FieldOperator(..., id='fieldop', ...)
    >>> foast_tree.params  # doctest: +ELLIPSIS
    [FieldSymbol(..., id='inp', ...)]
    >>> foast_tree.body  # doctest: +ELLIPSIS
    [Return(..., value=Name(..., id='inp'))]


    If a syntax error is encountered, it will point to the location in the source code.

    >>> def wrong_syntax(inp: Field[..., int]):
    ...     for i in range(10): # for is not part of the field operator syntax
    ...         tmp = inp
    ...     return tmp
    >>>
    >>> try:
    ...     FieldOperatorParser.apply_to_func(wrong_syntax)
    ... except FieldOperatorSyntaxError as err:
    ...     print(err.filename)  # doctest: +ELLIPSIS
    ...     print(err.lineno)
    ...     print(err.offset)
    /...<doctest src.functional.ffront.func_to_foast.FieldOperatorParser[...]>
    2
    4
    """

    source: str
    filename: str
    starting_line: int
    closure_vars: ClosureVars
    externals_defs: dict[str, Any]

    def _getloc(self, node: ast.AST) -> SourceLocation:
        loc = SourceLocation.from_AST(node, source=self.filename)
        return SourceLocation(
            line=loc.line + self.starting_line - 1,
            column=loc.column,
            source=loc.source,
            end_line=loc.end_line + self.starting_line - 1,
            end_column=loc.end_column,
        )

    def _make_syntax_error(self, node: ast.AST, *, message: str = "") -> FieldOperatorSyntaxError:
        err = FieldOperatorSyntaxError.from_AST(
            node, msg=message, filename=self.filename, text=self.source
        )
        err.lineno = (err.lineno or 1) + self.starting_line - 1
        return err

    @classmethod
    def apply(
        cls,
        source_definition: SourceDefinition,
        closure_vars: Optional[ClosureVars] = None,
        externals_defs: Optional[dict[str, Any]] = None,
    ) -> foast.FieldOperator:
        source, filename, starting_line = source_definition
        if closure_vars:
            if closure_vars.unbound:
                raise common.GTValueError(
                    f"Closure contains references to undefined or forward-defined values ({closure_vars.unbound})"
                )

        result = None
        try:
            definition_ast = ast.parse(textwrap.dedent(source)).body[0]
            ssa = SingleStaticAssignPass.apply(definition_ast)
            sat = SingleAssignTargetPass.apply(ssa)
            las = UnpackedAssignPass.apply(sat)
            result = cls(
                source=source,
                filename=filename,
                starting_line=starting_line,
                closure_vars=closure_vars,
                externals_defs=externals_defs,
            ).visit(las)
        except SyntaxError as err:
            if not err.filename:
                err.filename = filename
            if not isinstance(err, FieldOperatorSyntaxError):
                err.lineno = (err.lineno or 1) + starting_line - 1
            raise err

        return FieldOperatorTypeDeduction.apply(result)

    @classmethod
    def apply_to_function(
        cls,
        func: types.FunctionType,
        externals: Optional[dict[str, Any]] = None,
    ) -> foast.FieldOperator:
        source_definition = SourceDefinition.from_function(func)
        closure_vars = inspect.getclosurevars(func)
        return cls.apply(source_definition, closure_vars, externals)

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs) -> foast.FieldOperator:
        symbol_names = SymbolNames.from_source(self.source, self.filename)
        global_names = set(symbol_names.globals) - set(fbuiltins.TYPE_BUILTIN_NAMES)
        if global_names:
            if not self.closure_vars or (
                missing_defs := (global_names - set(self.closure_vars.globals.keys()))
            ):
                raise common.GTValueError(f"Missing global symbol definitions: {missing_defs}")

        closure_symbols = [
            _make_symbol_from_value(name, self.closure_vars.globals[name], foast.Namespace.CLOSURE)
            for name in global_names
        ]
        return foast.FieldOperator(
            id=node.name,
            params=self.visit(node.args),
            body=self.visit_stmt_list(node.body),
            closure=closure_symbols,
            location=self._getloc(node),
        )

    def visit_ImportFrom(self, node: ast.ImportFrom, **kwargs) -> None:
        assert node.module == "__externals__"

        symbols = []
        for alias in node.names:
            if alias.name not in self.externals_defs:
                raise self._make_syntax_error(
                    node, message="Missing imported symbol '{alias.name}'"
                )

            symbols.append(
                _make_symbol_from_value(
                    alias.asname, self.externals_defs[alias.name], foast.Namespace.EXTERNAL
                )
            )

        return foast.ExternalImport(symbols=symbols)

    def visit_arguments(self, node: ast.arguments) -> list[foast.FieldSymbol]:
        return [self.visit_arg(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg) -> foast.FieldSymbol:
        new_type = FieldOperatorTypeParser.apply(node.annotation)
        if new_type is None:
            raise self._make_syntax_error(node, message="Untyped parameters not allowed!")
        return foast.FieldSymbol(id=node.arg, location=self._getloc(node), type=new_type)

    def visit_Assign(self, node: ast.Assign, **kwargs) -> foast.Assign:
        target = node.targets[0]  # can there be more than one element?
        if isinstance(target, ast.Tuple):
            raise self._make_syntax_error(node, message="Unpacking not allowed!")
        if not isinstance(target, ast.Name):
            raise self._make_syntax_error(node, message="Can only assign to names!")
        new_value = self.visit(node.value)
        return foast.Assign(
            target=foast.FieldSymbol(
                id=target.id,
                location=self._getloc(target),
                type=foast.DeferredSymbolType(constraint=foast.FieldType),
            ),
            value=new_value,
            location=self._getloc(node),
        )

    def visit_AnnAssign(self, node: ast.AnnAssign, **kwargs) -> foast.Assign:
        # TODO (ricoh): type checking
        #
        # if the annotation does not match the inferred type of value
        # then raise an exception
        # -> only store the type here and write an additional checking pass
        if not isinstance(node.target, ast.Name):
            raise self._make_syntax_error(node, message="Can only assign to names!")
        return foast.Assign(
            target=foast.FieldSymbol(
                id=node.target.id,
                location=self._getloc(node.target),
                type=FieldOperatorTypeParser.apply(node.annotation),
            ),
            value=self.visit(node.value) if node.value else None,
            location=self._getloc(node),
        )

    def visit_Subscript(self, node: ast.Subscript, **kwargs) -> foast.Subscript:
        if not isinstance(node.slice, ast.Constant):
            raise self._make_syntax_error(node, message="""Subscript slicing not allowed!""")
        return foast.Subscript(
            value=self.visit(node.value), index=node.slice.value, location=self._getloc(node)
        )

    def visit_Tuple(self, node: ast.Tuple, **kwargs) -> foast.TupleExpr:
        return foast.TupleExpr(
            elts=[self.visit(item) for item in node.elts], location=self._getloc(node)
        )

    def visit_Return(self, node: ast.Return, **kwargs) -> foast.Return:
        if not node.value:
            raise self._make_syntax_error(node, message="Empty return not allowed")
        return foast.Return(value=self.visit(node.value), location=self._getloc(node))

    def visit_stmt_list(self, nodes: list[ast.stmt]) -> list[foast.Expr]:
        if not isinstance(last_node := nodes[-1], ast.Return):
            raise self._make_syntax_error(
                last_node, message="Field operator must return a field expression on the last line!"
            )
        return [self.visit(node) for node in nodes]

    def visit_Name(self, node: ast.Name, **kwargs) -> foast.Name:
        return foast.Name(id=node.id, location=self._getloc(node))

    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs) -> foast.UnaryOp:
        return foast.UnaryOp(
            op=self.visit(node.op), operand=self.visit(node.operand), location=self._getloc(node)
        )

    def visit_UAdd(self, node: ast.UAdd, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.UADD

    def visit_USub(self, node: ast.USub, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.USUB

    def visit_Not(self, node: ast.Not, **kwargs) -> foast.UnaryOperator:
        return foast.UnaryOperator.NOT

    def visit_BinOp(self, node: ast.BinOp, **kwargs) -> foast.BinOp:
        new_op = None
        try:
            new_op = self.visit(node.op)
        except FieldOperatorSyntaxError as err:
            err.lineno = node.lineno
            err.offset = node.col_offset
            raise err
        return foast.BinOp(
            op=new_op,
            left=self.visit(node.left),
            right=self.visit(node.right),
            location=self._getloc(node),
        )

    def visit_Add(self, node: ast.Add, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.ADD

    def visit_Sub(self, node: ast.Sub, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.SUB

    def visit_Mult(self, node: ast.Mult, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.MULT

    def visit_Div(self, node: ast.Div, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.DIV

    def visit_Pow(self, node: ast.Pow, **kwargs) -> None:
        raise FieldOperatorSyntaxError(
            msg="`**` operator not supported!",
        )

    def visit_Mod(self, node: ast.Mod, **kwargs) -> None:
        raise FieldOperatorSyntaxError(
            msg="`%` operator not supported!",
        )

    def visit_BitAnd(self, node: ast.BitAnd, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.BIT_AND

    def visit_BitOr(self, node: ast.BitOr, **kwargs) -> foast.BinaryOperator:
        return foast.BinaryOperator.BIT_OR

    def visit_BoolOp(self, node: ast.BoolOp, **kwargs) -> None:
        try:
            self.visit(node.op)
        except FieldOperatorSyntaxError as err:
            err.lineno = node.lineno
            err.offset = node.col_offset
            raise err

    def visit_And(self, node: ast.And, **kwargs) -> None:
        raise FieldOperatorSyntaxError(msg="`and` operator not allowed!")

    def visit_Or(self, node: ast.Or, **kwargs) -> None:
        raise FieldOperatorSyntaxError(msg="`or` operator not allowed!")

    def visit_Compare(self, node: ast.Compare, **kwargs) -> foast.Compare:
        if len(node.comparators) == 1:
            return foast.Compare(
                op=self.visit(node.ops[0]),
                left=self.visit(node.left),
                right=self.visit(node.comparators[0]),
                location=self._getloc(node),
            )
        smaller_node = copy.copy(node)
        smaller_node.comparators = node.comparators[1:]
        smaller_node.ops = node.ops[1:]
        smaller_node.left = node.comparators[0]
        return foast.Compare(
            op=self.visit(node.ops[0]),
            left=self.visit(node.left),
            right=self.visit(smaller_node),
            location=self._getloc(node),
        )

    def visit_Gt(self, node: ast.Gt, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.GT

    def visit_Lt(self, node: ast.Lt, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.LT

    def visit_Eq(self, node: ast.Eq, **kwargs) -> foast.CompareOperator:
        return foast.CompareOperator.EQ

    def visit_Call(self, node: ast.Call, **kwargs) -> foast.CompareOperator:
        new_func = self.visit(node.func)
        if not isinstance(new_func, foast.Name):
            raise self._make_syntax_error(
                node.func, message="functions can only be called directly!"
            )

        return foast.Call(
            func=new_func,
            args=[self.visit(arg) for arg in node.args],
            location=self._getloc(node),
        )

    def generic_visit(self, node) -> None:
        raise self._make_syntax_error(node)


class FieldOperatorSyntaxError(common.GTSyntaxError):
    def __init__(
        self,
        msg="",
        *,
        lineno: int = 0,
        offset: int = 0,
        filename: Optional[str] = None,
        end_lineno: int = None,
        end_offset: int = None,
        text: Optional[str] = None,
    ):
        msg = "Invalid Field Operator Syntax: " + msg
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))

    @classmethod
    def from_AST(
        cls,
        node: ast.AST,
        *,
        msg: str = "",
        filename: Optional[str] = None,
        text: Optional[str] = None,
    ):
        return cls(
            msg,
            lineno=node.lineno,
            offset=node.col_offset,
            filename=filename,
            end_lineno=node.end_lineno,
            end_offset=node.end_col_offset,
        )
