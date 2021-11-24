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
import collections
import copy
import inspect
import numbers
import symtable
import textwrap
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from eve.type_definitions import SourceLocation
from functional import common
from functional.common import Backend, FieldOperator
from functional.ffront import field_operator_ast as foast
from functional.ffront.ast_passes import (
    SingleAssignTargetPass,
    SingleStaticAssignPass,
    UnpackedAssignPass,
)


def field_operator(
    definition: types.FunctionType | str | tuple[str, str] | tuple[str, str, int] | None = None,
    *,
    backend: Backend,
    externals: dict[str, Any] | Sequence[dict[str, Any]] | Literal["embedded"] | None = "embedded",
) -> FieldOperator:
    """
    Create a GT4Py FieldOperator from the passed function.

    Args:
        function: Definition function.

    Keyword Args:
        backend: GT4Py backend
        externals: Variable length argument list.
            Defaults to ``None``.

    Returns:
        A backend-specific FieldOperator implementing the definition function.
    """

    func = None
    if callable(definition):
        func = definition
        source, filename, starting_line = get_source_and_info(definition)
    elif isinstance(definition, str):
        source, filename = definition, "<string>", 1
    elif isinstance(definition, Sequence) and 2 <= (def_length := len(definition)) <= 3:
        if def_length == 2:
            source, filename, starting_line = *definition, 1
        else:
            source, filename, starting_line = definition
    else:
        raise common.GTValueError(f"Invalid field operator definition ({definition})")

    arg_names, local_names, global_names = extract_symbol_names(source, filename)

    externals_dict = {}
    if externals == "embedded":
        if not func:
            raise common.GTValueError(
                f"Embedded externals can only be used when using a function object as operator definition."
            )
        externals_dict.update(collect_embedded_externals(func))
    elif isinstance(externals, Mapping):
        externals_dict.update(externals)
    elif isinstance(externals, Sequence):
        externals_dict.update(collections.ChainMap(*externals))

    if missing := (set(global_names.globals) - set(externals_dict.keys())):
        raise common.GTValueError(f"Missing definitions for some external symbols ({missing})")

    symbols = SymbolsNamespace(
        args=arg_names, locals=local_names, globals=global_names, externals=externals_dict
    )

    return FieldOperatorParser.apply(source, filename, starting_line, symbols=symbols)


def get_source_and_info(func: Callable) -> tuple[str, str, int]:
    try:
        filename = inspect.getabsfile(func) or "<string>"
        source = textwrap.dedent(inspect.getsource(func))
        starting_line = inspect.getabsfile(func)[1] if not filename.endswith("<string>") else 1
    except OSError as err:
        if filename.endswith("<string>"):
            message = "Can not create field operator from a function that is not in a source file!"
        else:
            message = f"Can not get source code of passed function ({func})"
        raise ValueError(message) from err

    return source, filename, starting_line


SymbolNames = collections.namedtuple("SymbolNames", ["args", "locals", "globals"])


def extract_symbol_names(source: str, filename: str) -> SymbolNames:
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

    arg_names = func_st.get_parameters()
    local_names = func_st.get_locals()
    global_names = func_st.get_globals()

    return SymbolNames(arg_names, local_names, global_names)


def collect_embedded_externals(func):
    _nonlocals, globals, _builtins, _unbound = inspect.getclosurevars(func)
    return globals


# def make_symbol(name: str, value: Any) -> foast.Symbol:
#     symbol_type = make_type(value)
#     match value:
#         case str() | numbers.Number() | tuple():
#             assert isinstance(symbol_type, foast.DataType)
#             return foast.DataSymbol(id=name, type=symbol_type, origin=copy.deepcopy(value))
#         case types.FunctionType:
#             return foast.Function(id=name, type=symbol_type, origin=value, body=[])
#         case _:
#             raise common.GTTypeError(f"Impossible to map '{value}' value to a Symbol")


# def make_type(value: Any) -> foast.SymbolType:
#     match value:
#         case bool(), int(), float(), np.generic():
#             return make_type(type(value))
#         case type() as t if issubclass(t, (bool, int, float, np.generic)):
#             return foast.ScalarType(kind=make_scalar_kind(value))
#         case tuple() as tuple_value:
#             return foast.TupleType(types=[make_type(t) for t in tuple_value])
#         case common.Field():
#             return foast.FieldType(..., foast.ScalarKind.FLOAT64)
#         case types.FunctionType():
#             args = []
#             kwargs = []
#             returns = []
#             return foast.FunctionType(args, kwargs, returns)
#         case other:
#             if other.__module__ == "typing":
#                 return make_type(other.__origin__)

#     raise common.GTTypeError(f"Impossible to map '{value}' value to a SymbolType")


# def make_scalar_kind(value: npt.DTypeLike) -> foast.ScalarKind:
#     try:
#         dt = np.dtype(value)
#     except TypeError as err:
#         raise common.GTTypeError(f"Invalid scalar type definition ({value})") from err

#     if dt.shape == () and dt.fields is None:
#         match dt:
#             case np.bool_:
#                 return foast.ScalarKind.BOOL
#             case np.int32:
#                 return foast.ScalarKind.INT32
#             case np.int64:
#                 return foast.ScalarKind.INT64
#             case np.float32:
#                 return foast.ScalarKind.FLOAT32
#             case np.float64:
#                 return foast.ScalarKind.FLOAT64
#             case _:
#                 raise common.GTTypeError(f"Impossible to map '{value}' value to a ScalarKind")
#     else:
#         raise common.GTTypeError(f"Non-trivial dtypes like '{value}' are not yet supported")


@dataclass
class SymbolsNamespace:
    args: list[str]
    locals: list[str]
    globals: list[str]
    externals: dict[str, any]


# TODO (ricoh): pass on source locations
# TODO (ricoh): SourceLocation.source <- filename
class FieldOperatorParser(ast.NodeVisitor):
    """
    Parse field operator function definition from source code into FOAST.

    Catch any Field Operator specific syntax errors and typing problems.

    Example
    -------
    Parse a function into a Field Operator AST (FOAST), which can
    be lowered into Iterator IR (ITIR)

    >>> def fieldop(inp):
    ...     return inp

    >>> foast_tree = FieldOperatorParser.apply(fieldop)
    >>> foast_tree  # doctest: +ELLIPSIS
    FieldOperator(..., id='fieldop', ...)
    >>> foast_tree.params  # doctest: +ELLIPSIS
    [Field(..., id='inp')]
    >>> foast_tree.body  # doctest: +ELLIPSIS
    [Return(..., value=Name(..., id='inp'))]


    If a syntax error is encountered, it will point to the location in the source code.

    >>> def wrong_syntax(inp):
    ...     for i in range(10): # for is not part of the field operator syntax
    ...         tmp = inp
    ...     return tmp
    >>>
    >>> try:
    ...     FieldOperatorParser.apply(wrong_syntax)
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

    def __init__(
        self, *, source: str, filename: str, starting_line: int, symbols: SymbolsNamespace
    ) -> None:
        self.source = source
        self.filename = filename
        self.starting_line = starting_line
        self.symbols = symbols
        super().__init__()

    def _getloc(self, node: ast.AST) -> SourceLocation:
        return SourceLocation.from_AST(node, source=self.filename)

    @classmethod
    def apply(
        cls, source: str, filename: str = "<string>", starting_line: int = 1
    ) -> foast.FieldOperator:
        result = None
        try:
            definition_ast = ast.parse(textwrap.dedent(source)).body[0]
            ssa = SingleStaticAssignPass.apply(definition_ast)
            sat = SingleAssignTargetPass.apply(ssa)
            las = UnpackedAssignPass.apply(sat)
            result = cls(source=source, filename=filename, starting_line=starting_line).visit(las)
        except SyntaxError as err:
            err.filename = filename
            err.lineno = (err.lineno or 1) + starting_line - 1
            raise err

        return result

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs) -> foast.FieldOperator:
        return foast.FieldOperator(
            id=node.name,
            params=self.visit(node.args),
            body=self.visit_stmt_list(node.body),
            location=self._getloc(node),
        )

    def visit_arguments(self, node: ast.arguments) -> list[foast.Field]:
        return [foast.Field(id=arg.arg, location=self._getloc(arg)) for arg in node.args]

    def visit_Assign(self, node: ast.Assign, **kwargs) -> foast.Assign:
        target = node.targets[0]  # can there be more than one element?
        if isinstance(target, ast.Tuple):
            raise FieldOperatorSyntaxError(
                "Unpacking not allowed!",
                lineno=node.lineno,
                offset=node.col_offset,
            )
        if not isinstance(target, ast.Name):
            raise FieldOperatorSyntaxError(
                "Can only assign to names!",
                lineno=target.lineno,
                offset=target.col_offset,
            )
        return foast.Assign(
            target=foast.Field(id=target.id, location=self._getloc(target)),
            value=self.visit(node.value),
            location=self._getloc(node),
        )

    def visit_AnnAssign(self, node: ast.AnnAssign, **kwargs) -> foast.Assign:
        # TODO (ricoh): type checking
        #
        # if the annotation does not match the inferred type of value
        # then raise an exception
        if not isinstance(node.target, ast.Name):
            raise FieldOperatorSyntaxError(
                "Can only assign to names!",
                lineno=node.target.lineno,
                offset=node.target.col_offset,
            )
        return foast.Assign(
            target=foast.Name(id=node.target.id, location=self._getloc(node.target)),
            value=self.visit(node.value) if node.value else None,
            location=self._getloc(node),
        )

    def visit_Subscript(self, node: ast.Subscript, **kwargs) -> foast.Subscript:
        if not isinstance(node.slice, ast.Constant):
            raise FieldOperatorSyntaxError(
                """Subscript slicing not allowed!""",
                lineno=node.slice.lineno,
                offset=node.slice.col_offset,
            )
        return foast.Subscript(
            value=self.visit(node.value), index=node.slice.value, location=self._getloc(node)
        )

    def visit_Tuple(self, node: ast.Tuple, **kwargs) -> foast.TupleExpr:
        return foast.TupleExpr(
            elts=[self.visit(item) for item in node.elts], location=self._getloc(node)
        )

    def visit_Return(self, node: ast.Return, **kwargs) -> foast.Return:
        if not node.value:
            raise FieldOperatorSyntaxError(
                "Empty return not allowed", lineno=node.lineno, offset=node.col_offset
            )
        return foast.Return(value=self.visit(node.value), location=self._getloc(node))

    def visit_stmt_list(self, nodes: list[ast.stmt]) -> list[foast.Expr]:
        if not isinstance(last_node := nodes[-1], ast.Return):
            raise FieldOperatorSyntaxError(
                msg="Field operator must return a field expression on the last line!",
                lineno=last_node.lineno,
                offset=last_node.col_offset,
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
            raise FieldOperatorSyntaxError(
                msg="functions can only be called directly!",
                lineno=node.func.lineno,
                offset=node.func.col_offset,
            )

        return foast.Call(
            func=new_func,
            args=[self.visit(arg) for arg in node.args],
            location=self._getloc(node),
        )

    def generic_visit(self, node) -> None:
        raise FieldOperatorSyntaxError(
            lineno=node.lineno,
            offset=node.col_offset,
        )


class FieldOperatorSyntaxError(common.GTSyntaxError):
    def __init__(
        self,
        msg="",
        *,
        lineno=0,
        offset=0,
        filename=None,
        end_lineno=None,
        end_offset=None,
        text=None,
    ):
        msg = "Invalid Field Operator Syntax: " + msg
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))
