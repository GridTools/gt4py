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

import ast
import textwrap
import typing
from dataclasses import dataclass
from typing import Callable

from gt4py.eve.concepts import SourceLocation
from gt4py.eve.extended_typing import Any, ClassVar, Generic, Optional, Type, TypeVar
from gt4py.next import common
from gt4py.next.ffront.ast_passes.fix_missing_locations import FixMissingLocations
from gt4py.next.ffront.ast_passes.remove_docstrings import RemoveDocstrings
from gt4py.next.ffront.source_utils import SourceDefinition, get_closure_vars_from_function


DialectRootT = TypeVar("DialectRootT")


class DialectSyntaxError(common.GTSyntaxError):
    dialect_name: ClassVar[str] = ""

    def __init__(
        self,
        msg="",
        *,
        lineno: int = 0,
        offset: int = 0,
        filename: Optional[str] = None,
        end_lineno: Optional[int] = None,
        end_offset: Optional[int] = None,
        text: Optional[str] = None,
    ):
        msg = f"Invalid {self.dialect_name} Syntax: {msg}"
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
            offset=node.col_offset + 1,  # offset is 1-based for syntax errors
            filename=filename,
            end_lineno=getattr(node, "end_lineno", None),
            end_offset=(node.end_col_offset + 1)
            if hasattr(node, "end_col_offset") and node.end_col_offset is not None
            else None,
            text=text,
        )

    @classmethod
    def from_location(cls, msg="", *, location: SourceLocation):
        return cls(
            msg,
            lineno=location.line,
            offset=location.column,
            filename=location.source,
            end_lineno=location.end_line,
            end_offset=location.end_column,
            text=None,
        )


def _ensure_syntax_error_invariants(err: SyntaxError):
    """Ensure syntax error invariants required to print meaningful error messages."""
    # If offsets are provided so must line numbers. For example `err.offset` determines
    # if carets (`^^^^`) are printed below `err.text`. They would be misleading if we
    # don't know on which line the error occurs.
    assert err.lineno or not err.offset
    assert err.end_lineno or not err.end_offset
    # If the ends are provided so must starts.
    assert err.lineno or not err.end_lineno
    assert err.offset or not err.end_offset


@dataclass(frozen=True, kw_only=True)
class DialectParser(ast.NodeVisitor, Generic[DialectRootT]):
    source_definition: SourceDefinition
    closure_vars: dict[str, Any]
    annotations: dict[str, Any]
    syntax_error_cls: ClassVar[Type[DialectSyntaxError]] = DialectSyntaxError

    @classmethod
    def apply(
        cls,
        source_definition: SourceDefinition,
        closure_vars: dict[str, Any],
        annotations: dict[str, Any],
    ) -> DialectRootT:

        source, filename, starting_line = source_definition
        try:
            line_offset = starting_line - 1
            definition_ast: ast.AST
            definition_ast = ast.parse(textwrap.dedent(source)).body[0]
            definition_ast = ast.increment_lineno(definition_ast, line_offset)
            line_offset = 0  # line numbers are correct from now on

            definition_ast = RemoveDocstrings.apply(definition_ast)
            definition_ast = FixMissingLocations.apply(definition_ast)
            output_ast = cls._postprocess_dialect_ast(
                cls(
                    source_definition=source_definition,
                    closure_vars=closure_vars,
                    annotations=annotations,
                ).visit(cls._preprocess_definition_ast(definition_ast)),
                closure_vars,
                annotations,
            )
        except SyntaxError as err:
            _ensure_syntax_error_invariants(err)

            # The ast nodes do not contain information about the path of the
            #  source file or its contents. We add this information here so
            #  that raising an error using :func:`DialectSyntaxError.from_AST`
            #  does not require passing the information on every invocation.
            if not err.filename:
                err.filename = filename

            # ensure line numbers are relative to the file (respects `starting_line`)
            if err.lineno:
                err.lineno = err.lineno + line_offset
            if err.end_lineno:
                err.end_lineno = err.end_lineno + line_offset

            if not err.text:
                if err.lineno:
                    source_lineno = err.lineno - starting_line
                    source_end_lineno = (
                        (err.end_lineno - starting_line) if err.end_lineno else source_lineno
                    )
                    err.text = "\n".join(source.splitlines()[source_lineno : source_end_lineno + 1])
                else:
                    err.text = source

            raise err

        return output_ast

    @classmethod
    def apply_to_function(cls, function: Callable):
        src = SourceDefinition.from_function(function)
        closure_vars = get_closure_vars_from_function(function)
        annotations = typing.get_type_hints(function)
        return cls.apply(src, closure_vars, annotations)

    @classmethod
    def _preprocess_definition_ast(cls, definition_ast: ast.AST) -> ast.AST:
        return definition_ast

    @classmethod
    def _postprocess_dialect_ast(
        cls, output_ast: DialectRootT, closure_vars: dict[str, Any], annotations: dict[str, Any]
    ) -> DialectRootT:
        return output_ast

    def generic_visit(self, node: ast.AST) -> None:
        raise self.syntax_error_cls.from_AST(
            node,
            msg=f"Nodes of type {type(node).__module__}.{type(node).__qualname__} not supported in dialect.",
        )

    def _make_loc(self, node: ast.AST) -> SourceLocation:
        return SourceLocation.from_AST(node, source=self.source_definition.filename)
