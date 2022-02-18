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
import textwrap
import types
from dataclasses import dataclass
from typing import Any, Optional

from eve.type_definitions import SourceLocation
from functional import common
from functional.ffront import common_types
from functional.ffront import program_ast as past
from functional.ffront import symbol_makers
from functional.ffront.past_passes.type_deduction import ProgramTypeDeduction
from functional.ffront.source_utils import ClosureRefs, SourceDefinition, SymbolNames


# TODO(tehrengruber): disallow every ast node we don't understand yet


@dataclass(frozen=True, kw_only=True)
class ProgramParser(ast.NodeVisitor):
    """Parse program definition from Python source code into PAST."""

    source: str
    filename: str
    starting_line: int
    closure_refs: ClosureRefs
    externals_defs: dict[str, Any]

    def _make_loc(self, node: ast.AST) -> SourceLocation:
        loc = SourceLocation.from_AST(node, source=self.filename)
        return SourceLocation(
            line=loc.line + self.starting_line - 1,
            column=loc.column,
            source=loc.source,
            end_line=loc.end_line + self.starting_line - 1,
            end_column=loc.end_column,
        )

    def _make_syntax_error(self, node: ast.AST, *, message: str = "") -> ProgramSyntaxError:
        err = ProgramSyntaxError.from_AST(
            node, msg=message, filename=self.filename, text=self.source
        )
        err.lineno = (err.lineno or 1) + self.starting_line - 1
        return err

    @classmethod
    def apply(
        cls,
        source_definition: SourceDefinition,
        closure_refs: ClosureRefs,
        externals_defs: Optional[dict[str, Any]] = None,
    ) -> past.Program:
        source, filename, starting_line = source_definition
        try:
            definition_ast = ast.parse(textwrap.dedent(source)).body[0]
            untyped_past_node = cls(
                source=source,
                filename=filename,
                starting_line=starting_line,
                closure_refs=closure_refs,
                externals_defs=externals_defs,
            ).visit(definition_ast)
            result = ProgramTypeDeduction.apply(untyped_past_node)
        except SyntaxError as err:
            if not err.filename:
                err.filename = filename
            if not isinstance(err, ProgramSyntaxError):
                err.lineno = (err.lineno or 1) + starting_line - 1
            raise err

        return result

    @classmethod
    def apply_to_function(
        cls,
        func: types.FunctionType,
        externals: Optional[dict[str, Any]] = None,
    ) -> past.Program:
        source_definition = SourceDefinition.from_function(func)
        closure_refs = ClosureRefs.from_function(func)
        return cls.apply(source_definition, closure_refs, externals)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> past.Program:
        _, _, imported_names, nonlocal_names, global_names = SymbolNames.from_source(
            self.source, self.filename
        )
        # TODO(egparedes): raise the exception at the first use of the undefined symbol
        if missing_defs := (self.closure_refs.unbound - imported_names):
            raise self._make_syntax_error(
                node, message=f"Missing symbol definitions: {missing_defs}"
            )

        # 'SymbolNames.from_source()' uses the symtable module to analyze the isolated source
        # code of the function, and thus all non-local symbols are classified as 'global'.
        # However, 'closure_refs' comes from inspecting the live function object, which might
        # have not been defined at a global scope, and therefore actual symbol values could appear
        # in both 'closure_refs.globals' and 'self.closure_refs.nonlocals'.
        defs = self.closure_refs.globals | self.closure_refs.nonlocals
        closure = [
            past.Symbol(
                id=name,
                type=symbol_makers.make_symbol_type_from_value(defs[name]),
                namespace=common_types.Namespace.CLOSURE,
                location=self._make_loc(node),
            )
            for name in global_names | nonlocal_names
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
            raise self._make_syntax_error(node, message="Untyped parameters not allowed!")
        new_type = symbol_makers.make_symbol_type_from_typing(annotation)
        if not isinstance(new_type, common_types.DataType):
            raise self._make_syntax_error(
                node, message="Only arguments of type DataType are allowed."
            )
        return past.DataSymbol(id=node.arg, location=self._make_loc(node), type=new_type)

    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> past.Name:
        return past.Name(id=node.id, location=self._make_loc(node))

    def visit_Call(self, node: ast.Call) -> past.Call:
        new_func = self.visit(node.func)
        if not isinstance(new_func, past.Name):
            raise self._make_syntax_error(
                node.func, message="Functions can only be called directly!"
            )

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

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            symbol_type = symbol_makers.make_symbol_type_from_value(node.operand.value)
            return past.Constant(
                value=-node.operand.value, type=symbol_type, location=self._make_loc(node)
            )
        raise self._make_syntax_error(node, "Unary operators can only be used on literals.")

    def visit_Constant(self, node: ast.Constant) -> past.Constant:
        symbol_type = symbol_makers.make_symbol_type_from_value(node.value)
        return past.Constant(value=node.value, type=symbol_type, location=self._make_loc(node))


class ProgramSyntaxError(common.GTSyntaxError):
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
        msg = f"Invalid Program Syntax: {msg}"
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
