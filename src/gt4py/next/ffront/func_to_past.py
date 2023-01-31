# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
from dataclasses import dataclass
from typing import Any, cast

from gt4py.next.ffront import (
    dialect_ast_enums,
    program_ast as past,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.dialect_parser import DialectParser, DialectSyntaxError
from gt4py.next.ffront.past_passes.closure_var_type_deduction import ClosureVarTypeDeduction
from gt4py.next.ffront.past_passes.type_deduction import ProgramTypeDeduction
from gt4py.next.type_system import type_specifications as ts, type_translation


class ProgramSyntaxError(DialectSyntaxError):
    dialect_name = "Program"


@dataclass(frozen=True, kw_only=True)
class ProgramParser(DialectParser[past.Program]):
    """Parse program definition from Python source code into PAST."""

    syntax_error_cls = ProgramSyntaxError

    @classmethod
    def _postprocess_dialect_ast(
        cls, output_node: past.Program, closure_vars: dict[str, Any], annotations: dict[str, Any]
    ) -> past.Program:
        if "return" in annotations and not isinstance(None, annotations["return"]):
            raise ProgramSyntaxError("Program should not have a return value!")
        output_node = ClosureVarTypeDeduction.apply(output_node, closure_vars)
        return ProgramTypeDeduction.apply(output_node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> past.Program:
        closure_symbols: list[past.Symbol] = [
            past.Symbol(
                id=name,
                type=type_translation.from_value(val),
                namespace=dialect_ast_enums.Namespace.CLOSURE,
                location=self._make_loc(node),
            )
            for name, val in self.closure_vars.items()
        ]

        return past.Program(
            id=node.name,
            type=ts.DeferredType(constraint=ts_ffront.ProgramType),
            params=self.visit(node.args),
            body=[self.visit(node) for node in node.body],
            closure_vars=closure_symbols,
            location=self._make_loc(node),
        )

    def visit_arguments(self, node: ast.arguments) -> list[past.DataSymbol]:
        return [self.visit_arg(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg) -> past.DataSymbol:
        if (annotation := self.annotations.get(node.arg, None)) is None:
            raise ProgramSyntaxError.from_AST(node, msg="Untyped parameters not allowed!")
        new_type = type_translation.from_type_hint(annotation)
        if not isinstance(new_type, ts.DataType):
            raise ProgramSyntaxError.from_AST(
                node, msg="Only arguments of type DataType are allowed."
            )
        return past.DataSymbol(id=node.arg, location=self._make_loc(node), type=new_type)

    def visit_Expr(self, node: ast.Expr) -> past.LocatedNode:
        return self.visit(node.value)

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

    def visit_BinOp(self, node: ast.BinOp, **kwargs) -> past.BinOp:
        return past.BinOp(
            op=self.visit(node.op),
            left=self.visit(node.left),
            right=self.visit(node.right),
            location=self._make_loc(node),
        )

    def visit_Name(self, node: ast.Name) -> past.Name:
        return past.Name(id=node.id, location=self._make_loc(node))

    def visit_Dict(self, node: ast.Dict) -> past.Dict:
        return past.Dict(
            keys_=[self.visit(cast(ast.AST, param)) for param in node.keys],
            values_=[self.visit(param) for param in node.values],
            location=self._make_loc(node),
        )

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
            type=ts.DeferredType(constraint=ts.TupleType),
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
            symbol_type = type_translation.from_value(node.operand.value)
            return past.Constant(
                value=-node.operand.value, type=symbol_type, location=self._make_loc(node)
            )
        raise ProgramSyntaxError.from_AST(node, msg="Unary operators can only be used on literals.")

    def visit_Constant(self, node: ast.Constant) -> past.Constant:
        symbol_type = type_translation.from_value(node.value)
        return past.Constant(value=node.value, type=symbol_type, location=self._make_loc(node))
