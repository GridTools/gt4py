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

import ast
import inspect
import textwrap
from types import FunctionType
from typing import Callable, List

from functional.ffront import field_operator_ir as foir
from functional.ffront.ast_passes import (
    SingleAssignTargetPass,
    SingleStaticAssignPass,
    UnpackedAssignPass,
)


class FieldOperatorParser(ast.NodeVisitor):
    """
    Parse field operator function definition from source code into FOIR.

    Catch any Field Operator specific syntax errors and typing problems.

    Example
    -------
    Parse a function into a Field Operator Internal Representation (FOIR), which can
    be lowered into Iterator IR (ITIR)

    >>> def fieldop(inp):
    ...     return inp

    >>> foir_tree = FieldOperatorParser.apply(fieldop)
    >>> foir_tree
    FieldOperator(id='fieldop', params=[Sym(id='inp')], body=[Return(value=Name(id='inp'))])


    If a syntax error is encountered, it will point to the location in the source code.

    >>> def wrong_syntax(inp):
    ...     for i in range(10): # for is not part of the field operator syntax
    ...         tmp = inp
    ...     return tmp
    >>>
    >>> try:
    ...     FieldOperatorParser.apply(wrong_syntax)
    ... except FieldOperatorSyntaxError as err:
    ...     print(err.filename[-67:])
    ...     print(err.lineno)
    ...     print(err.offset)
    <doctest src.functional.ffront.func_to_foir.FieldOperatorParser[3]>
    2
    4
    """

    @classmethod
    def apply(cls, func: FunctionType) -> foir.FieldOperator:
        result = None
        try:
            ast = get_ast_from_func(func)
            ssa = SingleStaticAssignPass.apply(ast)
            sat = SingleAssignTargetPass.apply(ssa)
            las = UnpackedAssignPass.apply(sat)
            result = cls().visit(las)
        except SyntaxError as err:
            err.filename = inspect.getabsfile(func)
            err.lineno = (err.lineno or 1) + inspect.getsourcelines(func)[1] - 1
            raise err

        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> foir.FieldOperator:
        return foir.FieldOperator(
            id=node.name,
            params=self.visit(node.args),
            body=self.visit_stmt_list(node.body),
        )

    def visit_arguments(self, node: ast.arguments) -> list[foir.Sym]:
        return [foir.Sym(id=arg.arg) for arg in node.args]

    def visit_Assign(self, node: ast.Assign) -> foir.SymExpr:
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
        return foir.SymExpr(
            id=target.id,
            expr=self.visit(node.value),
        )

    def visit_Subscript(self, node: ast.Subscript) -> foir.Subscript:
        if not isinstance(node.slice, ast.Constant):
            raise FieldOperatorSyntaxError(
                """Subscript slicing not allowed!""",
                lineno=node.slice.lineno,
                offset=node.slice.col_offset,
            )
        return foir.Subscript(expr=self.visit(node.value), index=node.slice.value)

    def visit_Tuple(self, node: ast.Tuple) -> foir.Tuple:
        return foir.Tuple(elts=[self.visit(item) for item in node.elts])

    def visit_Return(self, node: ast.Return) -> foir.Return:
        if not node.value:
            raise FieldOperatorSyntaxError(
                "Empty return not allowed", lineno=node.lineno, offset=node.col_offset
            )
        return foir.Return(value=self.visit(node.value))

    def visit_stmt_list(self, nodes: List[ast.stmt]) -> List[foir.Expr]:
        if not isinstance(last_node := nodes[-1], ast.Return):
            raise FieldOperatorSyntaxError(
                msg="Field operator must return a field expression on the last line!",
                lineno=last_node.lineno,
                offset=last_node.col_offset,
            )
        return [self.visit(node) for node in nodes]

    def visit_Name(self, node: ast.Name) -> foir.Name:
        return foir.Name(id=node.id)

    def generic_visit(self, node) -> None:
        raise FieldOperatorSyntaxError(
            lineno=node.lineno,
            offset=node.col_offset,
        )


class FieldOperatorSyntaxError(SyntaxError):
    def __init__(self, msg="", *, lineno=0, offset=0, filename=None):
        self.msg = "Invalid Field Operator Syntax: " + msg
        self.lineno = lineno
        self.offset = offset
        self.filename = filename


def get_ast_from_func(func: Callable) -> ast.stmt:
    if inspect.getabsfile(func) == "<string>":
        raise ValueError(
            "Can not create field operator from a function that is not in a source file!"
        )
    source = textwrap.dedent(inspect.getsource(func))
    return ast.parse(source).body[0]
