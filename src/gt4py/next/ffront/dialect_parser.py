# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import textwrap
from dataclasses import dataclass
from typing import Callable, ClassVar, Collection

from gt4py.eve.concepts import SourceLocation
from gt4py.eve.extended_typing import Any, Generic, TypeVar
from gt4py.next import errors
from gt4py.next.ffront import source_utils
from gt4py.next.ffront.ast_passes.fix_missing_locations import FixMissingLocations
from gt4py.next.ffront.ast_passes.remove_docstrings import RemoveDocstrings
from gt4py.next.ffront.source_utils import SourceDefinition, get_closure_vars_from_function


DialectRootT = TypeVar("DialectRootT")


#: Friendly diagnostics for Python constructs that are not part of the DSL subset:
#: maps the offending `ast` node type to a user-facing construct name and hints on
#: what to use instead. Constructs not listed here get a generic message naming
#: the `ast` class. Keep the hints actionable: name the closest supported
#: alternative, not just the restriction.
# TODO(havogt): add 'ast.TryStar' ('try*' statement, Python >=3.11) once the
#  Python floor is >=3.12; referencing it unconditionally breaks import on 3.10.
_UNSUPPORTED_FEATURE_HINTS: dict[type[ast.AST], tuple[str, tuple[str, ...]]] = {
    ast.For: (
        "'for' loop",
        (
            "GT4Py functions describe operations on whole fields without explicit loops. "
            "Use field expressions or built-ins like 'neighbor_sum' instead; for sequential "
            "dependencies along a dimension, use a 'scan_operator'.",
        ),
    ),
    ast.While: (
        "'while' loop",
        (
            "GT4Py functions describe operations on whole fields without explicit loops. "
            "For sequential dependencies along a dimension, use a 'scan_operator'.",
        ),
    ),
    ast.ListComp: (
        "list comprehension",
        ("Operations apply to whole fields at once; use field expressions instead.",),
    ),
    ast.SetComp: ("set comprehension", ()),
    ast.DictComp: ("dictionary comprehension", ()),
    ast.GeneratorExp: ("generator expression", ()),
    ast.Lambda: (
        "'lambda' expression",
        ("Define a separate function decorated with '@field_operator' instead.",),
    ),
    ast.Try: ("'try' statement", ("Exception handling is not available inside GT4Py functions.",)),
    ast.Raise: (
        "'raise' statement",
        ("Exception handling is not available inside GT4Py functions.",),
    ),
    ast.With: ("'with' statement", ()),
    ast.Assert: ("'assert' statement", ()),
    ast.ClassDef: ("class definition", ()),
    ast.JoinedStr: ("f-string", ("Strings cannot be computed inside GT4Py functions.",)),
    ast.Match: ("'match' statement", ("Use 'if'/'elif' chains or 'where' instead.",)),
    ast.Global: (
        "'global' statement",
        (
            "Variables from the surrounding scope are read-only inside GT4Py functions; "
            "pass values as parameters and return results instead.",
        ),
    ),
    ast.Nonlocal: (
        "'nonlocal' statement",
        (
            "Variables from the surrounding scope are read-only inside GT4Py functions; "
            "pass values as parameters and return results instead.",
        ),
    ),
}


def _describe_unsupported_feature(node: ast.AST) -> tuple[str, tuple[str, ...]]:
    if (entry := _UNSUPPORTED_FEATURE_HINTS.get(type(node))) is not None:
        return entry
    return f"'{type(node).__module__}.{type(node).__qualname__}'", ()


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

    reserved_names: ClassVar[Collection[str]] = ()  # e.g. for dialect builtins

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
        annotations = source_utils.get_type_hints_from_function(function, src)
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
        feature, hints = _describe_unsupported_feature(node)
        raise errors.UnsupportedPythonFeatureError(
            loc,
            feature,
            notes=("Only a subset of Python is valid inside GT4Py functions.",),
            hints=hints,
        )

    def _check_not_a_reserved_name(self, name: str, location: SourceLocation) -> None:
        if name in self.reserved_names:
            raise errors.DSLError(
                location,
                f"Name '{name}' is a reserved GT4Py builtin and cannot be used as the "
                f"name of a function. Please choose a different name.",
            )

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
