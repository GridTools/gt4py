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
import inspect
import textwrap
import types

from eve.type_definitions import SourceLocation
from functional import common
from functional.ffront import field_operator_ast as foast
from functional.ffront.ast_passes import (
    SingleAssignTargetPass,
    SingleStaticAssignPass,
    UnpackedAssignPass,
)
from functional.ffront.type_parser import FieldOperatorTypeParser


class FieldOperatorParser(ast.NodeVisitor):
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

    def __init__(
        self,
        *,
        source: str,
        filename: str,
        starting_line: int,
    ) -> None:
        self.source = source
        self.filename = filename
        self.starting_line = starting_line
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
            result = cls(
                source=source,
                filename=filename,
                starting_line=starting_line,
            ).visit(las)
        except SyntaxError as err:
            err.filename = filename
            err.lineno = (err.lineno or 1) + starting_line - 1
            raise err

        return result

    @classmethod
    def apply_to_func(cls, func: types.FunctionType) -> foast.FieldOperator:
        def get_src(func: types.FunctionType) -> str:
            if inspect.getabsfile(func) == "<string>":
                raise ValueError(
                    "Can not create field operator from a function that is not in a source file!"
                )
            return textwrap.dedent(inspect.getsource(func))

        source = get_src(func)
        return cls.apply(
            source, filename=inspect.getabsfile(func), starting_line=inspect.getsourcelines(func)[1]
        )

    def visit_FunctionDef(self, node: ast.FunctionDef, **kwargs) -> foast.FieldOperator:
        return foast.FieldOperator(
            id=node.name,
            params=self.visit(node.args),
            body=self.visit_stmt_list(node.body),
            location=self._getloc(node),
        )

    def visit_arguments(self, node: ast.arguments) -> list[foast.FieldSymbol]:
        return [self.visit_arg(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg) -> foast.FieldSymbol:
        new_type = FieldOperatorTypeParser.apply(node.annotation)
        if new_type is None:
            raise FieldOperatorSyntaxError(
                "Untyped parameters not allowed!",
                lineno=node.lineno,
                offset=node.col_offset,
            )
        return foast.FieldSymbol(id=node.arg, location=self._getloc(node), type=new_type)

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
            raise FieldOperatorSyntaxError(
                "Can only assign to names!",
                lineno=node.target.lineno,
                offset=node.target.col_offset,
            )
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
