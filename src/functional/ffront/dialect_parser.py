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
from typing import Any, ClassVar, Generic, Optional, Type, TypeVar, cast

from eve.type_definitions import SourceLocation
from functional import common
from functional.ffront.source_utils import ClosureRefs, SourceDefinition, SymbolNames


def _assert_source_invariants(source_definition: SourceDefinition, closure_refs: ClosureRefs):
    """
    Validate information contained in the source agrees with our expectations.

    No error should ever originate from this function. This is merely double
    checking what is already done using ast nodes in the visitor.
    """
    source, filename, starting_line = source_definition
    _, _, imported_names, nonlocal_names, global_names = SymbolNames.from_source(source, filename)
    if missing_defs := (closure_refs.unbound - imported_names):
        raise AssertionError(f"Missing symbol definitions: {missing_defs}")

    # 'SymbolNames.from_source()' uses the symtable module to analyze the isolated source
    # code of the function, and thus all non-local symbols are classified as 'global'.
    # However, 'closure_refs' comes from inspecting the live function object, which might
    # have not been defined at a global scope, and therefore actual symbol values could appear
    # in both 'closure_refs.globals' and 'self.closure_refs.nonlocals'.
    if not set(closure_refs.globals) | set(closure_refs.nonlocals.keys()) == (
        global_names | nonlocal_names
    ):
        raise AssertionError("ClosureRefs do not agree with information from symtable module.")


DialectRootT = TypeVar("DialectRootT")


@dataclass(frozen=True, kw_only=True)
class DialectParser(ast.NodeVisitor, Generic[DialectRootT]):
    source: str
    filename: str
    starting_line: int
    closure_refs: ClosureRefs
    externals_defs: dict[str, Any]

    syntax_error_cls: ClassVar[Type[DialectSyntaxError]]

    @classmethod
    def apply(
        cls,
        source_definition: SourceDefinition,
        closure_refs: ClosureRefs,
        externals: Optional[dict[str, Any]] = None,
    ) -> DialectRootT:
        source, filename, starting_line = source_definition
        try:
            definition_ast = cls._preprocess_definition_ast(
                ast.parse(textwrap.dedent(source)).body[0]
            )
            output_ast = cls._postprocess_dialect_ast(
                cls(
                    source=source,
                    filename=filename,
                    starting_line=starting_line,
                    closure_refs=closure_refs,
                    externals_defs=externals or {},
                ).visit(definition_ast)
            )
            if __debug__:
                _assert_source_invariants(source_definition, closure_refs)
        except SyntaxError as err:
            # TODO(tehrengruber): Instead of enhancing the exception here and
            #  having source information in the instance we could preprocess
            #  the ast and offset the line numbers there.
            if not err.filename:
                err.filename = filename
            if not isinstance(err, DialectSyntaxError):
                err.lineno = (err.lineno or 1) + starting_line - 1
            raise err

        return output_ast

    @classmethod
    def _preprocess_definition_ast(cls, definition_ast: ast.AST) -> ast.AST:
        return definition_ast

    @classmethod
    def _postprocess_dialect_ast(cls, output_ast: DialectRootT) -> DialectRootT:
        return output_ast

    @classmethod
    def apply_to_function(
        cls,
        func: types.FunctionType,
        externals: Optional[dict[str, Any]] = None,
    ) -> DialectRootT:
        source_definition = SourceDefinition.from_function(func)
        closure_refs = ClosureRefs.from_function(func)
        return cls.apply(source_definition, closure_refs, externals)

    def generic_visit(self, node: ast.AST) -> None:
        raise self._make_syntax_error(
            node,
            message=f"Nodes of type {type(node).__module__}.{type(node).__name__} not supported in dialect.",
        )

    def _make_loc(self, node: ast.AST) -> SourceLocation:
        loc = SourceLocation.from_AST(node, source=self.filename)
        return SourceLocation(
            line=loc.line + self.starting_line - 1,
            column=loc.column,
            source=loc.source,
            end_line=cast(int, loc.end_line) + self.starting_line - 1,
            end_column=loc.end_column,
        )

    def _make_syntax_error(self, node: ast.AST, *, message: str = "") -> DialectSyntaxError:
        err = self.syntax_error_cls.from_AST(
            node, msg=message, filename=self.filename, text=self.source
        )
        err.lineno = (err.lineno or 1) + self.starting_line - 1
        return err


class DialectSyntaxError(common.GTSyntaxError):
    dialect_name: ClassVar[str] = ""

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
        msg = f"Invalid {self.dialect_name} Syntax: {msg}\n"
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
