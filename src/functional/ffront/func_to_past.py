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
from dataclasses import dataclass

from functional.ffront import common_types
from functional.ffront import program_ast as past
from functional.ffront import symbol_makers
from functional.ffront.dialect_parser import DialectParser, DialectSyntaxError
from functional.ffront.past_passes.type_deduction import ProgramTypeDeduction


class ProgramSyntaxError(DialectSyntaxError):
    dialect_name = "Program"


@dataclass(frozen=True, kw_only=True)
class ProgramParser(DialectParser[past.Program]):
    """Parse program definition from Python source code into PAST."""

    syntax_error_cls = ProgramSyntaxError

    @classmethod
    def _postprocess_dialect_ast(cls, output_node: past.Program) -> past.Program:
        return ProgramTypeDeduction.apply(output_node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> past.Program:
        vars_ = collections.ChainMap(self.closure_refs.globals, self.closure_refs.nonlocals)
        closure = [
            past.Symbol(
                id=name,
                type=symbol_makers.make_symbol_type_from_value(val),
                namespace=common_types.Namespace.CLOSURE,
                location=self._make_loc(node),
            )
            for name, val in vars_.items()
        ]

        return past.Program(
            id=node.name,
            params=self.visit(node.args),
            body=[self.visit(node) for node in node.body],
            closure=closure,
            location=self._make_loc(node),
        )

    def visit_arguments(self, node: ast.arguments) -> list[past.DataSymbol]:
        return [self.visit_arg(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg) -> past.DataSymbol:
        if (annotation := self.closure_refs.annotations.get(node.arg, None)) is None:
            raise ProgramSyntaxError.from_AST(node, msg="Untyped parameters not allowed!")
        new_type = symbol_makers.make_symbol_type_from_typing(annotation)
        if not isinstance(new_type, common_types.DataType):
            raise ProgramSyntaxError.from_AST(
                node, msg="Only arguments of type DataType are allowed."
            )
        return past.DataSymbol(id=node.arg, location=self._make_loc(node), type=new_type)

    def visit_Expr(self, node: ast.Expr) -> past.LocatedNode:
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> past.Name:
        return past.Name(id=node.id, location=self._make_loc(node))

    def visit_Call(self, node: ast.Call) -> past.Call:
        new_func = self.visit(node.func)
        if not isinstance(new_func, past.Name):
            raise ProgramSyntaxError.from_AST(node, msg="Functions can only be called directly!")

        return past.Call(
            func=new_func,
            args=[self.visit(arg) for arg in node.args],
            kwargs={arg.arg: self.visit(arg.value) for arg in node.keywords},
            location=self._make_loc(node),
        )

    def visit_Subscript(self, node: ast.Subscript) -> past.Subscript:
        return past.Subscript(
            value=self.visit(node.value),
            slice_=self.visit(node.slice),
            location=self._make_loc(node),
        )

    def visit_Tuple(self, node: ast.Tuple) -> past.TupleExpr:
        return past.TupleExpr(
            elts=[self.visit(item) for item in node.elts],
            location=self._make_loc(node),
            type=common_types.DeferredSymbolType(constraint=common_types.TupleType),
        )

    def visit_Slice(self, node: ast.Slice) -> past.Slice:
        return past.Slice(
            lower=self.visit(node.lower) if node.lower is not None else None,
            upper=self.visit(node.upper) if node.upper is not None else None,
            step=self.visit(node.step) if node.step is not None else None,
            location=self._make_loc(node),
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> past.Constant:
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            symbol_type = symbol_makers.make_symbol_type_from_value(node.operand.value)
            return past.Constant(
                value=-node.operand.value, type=symbol_type, location=self._make_loc(node)
            )
        raise ProgramSyntaxError.from_AST(node, msg="Unary operators can only be used on literals.")

    def visit_Constant(self, node: ast.Constant) -> past.Constant:
        symbol_type = symbol_makers.make_symbol_type_from_value(node.value)
        return past.Constant(value=node.value, type=symbol_type, location=self._make_loc(node))
