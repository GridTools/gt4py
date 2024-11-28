# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import textwrap
import typing
from dataclasses import dataclass
from typing import Callable

from gt4py.eve.concepts import SourceLocation
from gt4py.eve.extended_typing import Any, Generic, TypeVar
from gt4py.next import errors
from gt4py.next.ffront.ast_passes.fix_missing_locations import FixMissingLocations
from gt4py.next.ffront.ast_passes.remove_docstrings import RemoveDocstrings
from gt4py.next.ffront.source_utils import SourceDefinition, get_closure_vars_from_function


DialectRootT = TypeVar("DialectRootT")


def parse_source_definition(source_definition: SourceDefinition) -> ast.AST:
    try:
        return ast.parse(textwrap.dedent(source_definition.source)).body[0]
    except SyntaxError as err:
        assert err.lineno
        assert err.offset
        loc = SourceLocation(
            line=err.lineno + source_definition.line_offset,
            column=err.offset + source_definition.column_offset,
            filename=source_definition.filename,
            end_line=(
                err.end_lineno + source_definition.line_offset
                if err.end_lineno is not None
                else None
            ),
            end_column=(
                err.end_offset + source_definition.column_offset
                if err.end_offset is not None
                else None
            ),
        )
        raise errors.DSLError(loc, err.msg).with_traceback(err.__traceback__) from err


@dataclass(frozen=True, kw_only=True)
class DialectParser(ast.NodeVisitor, Generic[DialectRootT]):
    source_definition: SourceDefinition
    closure_vars: dict[str, Any]
    annotations: dict[str, Any]

    @classmethod
    def apply(
        cls,
        source_definition: SourceDefinition,
        closure_vars: dict[str, Any],
        annotations: dict[str, Any],
    ) -> DialectRootT:
        definition_ast: ast.AST
        definition_ast = parse_source_definition(source_definition)

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

        return output_ast

    @classmethod
    def apply_to_function(cls, function: Callable) -> DialectRootT:
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
        loc = self.get_location(node)
        feature = f"{type(node).__module__}.{type(node).__qualname__}"
        raise errors.UnsupportedPythonFeatureError(loc, feature)

    def get_location(self, node: ast.AST) -> SourceLocation:
        file = self.source_definition.filename
        line_offset = self.source_definition.line_offset
        col_offset = self.source_definition.column_offset

        # `FixMissingLocations` ensures that all nodes have the location attributes
        assert hasattr(node, "lineno")
        line = node.lineno + line_offset
        assert hasattr(node, "end_lineno")
        end_line = node.end_lineno + line_offset if node.end_lineno is not None else None
        assert hasattr(node, "col_offset")
        column = 1 + node.col_offset + col_offset
        assert hasattr(node, "end_col_offset")
        end_column = (
            1 + node.end_col_offset + col_offset if node.end_col_offset is not None else None
        )

        loc = SourceLocation(file, line, column, end_line=end_line, end_column=end_column)
        return loc
