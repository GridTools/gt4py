# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

import pydantic
import pytest

import eve


def test_sentinel():
    from eve.type_definitions import NOTHING

    values = [0, 1, 2, NOTHING, 4, 6]

    assert values.index(NOTHING) == 3
    assert values[values.index(NOTHING)] is NOTHING


def test_symbol_types():
    from eve.type_definitions import SymbolName

    assert SymbolName("valid_name_01A") == "valid_name_01A"
    assert SymbolName.from_string("valid_name_01A") == "valid_name_01A"
    with pytest.raises(ValueError, match="string does not match regex"):
        SymbolName.from_string("$name_01A")
    with pytest.raises(ValueError, match="string does not match regex"):
        SymbolName.from_string("0name_01A")
    with pytest.raises(ValueError, match="string does not match regex"):
        SymbolName.from_string("name_01A ")

    LettersOnlySymbol = SymbolName.constrained(r"[a-zA-Z]+$")
    assert LettersOnlySymbol.from_string("validNAME") == "validNAME"
    with pytest.raises(ValueError, match="string does not match regex"):
        LettersOnlySymbol.from_string("name_a")
    with pytest.raises(ValueError, match="string does not match regex"):
        LettersOnlySymbol.from_string("name01")


class TestSourceLocation:
    def test_valid_position(self):
        eve.type_definitions.SourceLocation(line=1, column=1, source="source.py")

    def test_invalid_position(self):
        with pytest.raises(pydantic.ValidationError):
            eve.type_definitions.SourceLocation(line=1, column=-1, source="source.py")

    def test_str(self):
        loc = eve.type_definitions.SourceLocation(line=1, column=1, source="dir/source.py")
        assert str(loc) == "<'dir/source.py': Line 1, Col 1>"

        loc = eve.type_definitions.SourceLocation(
            line=1, column=1, source="dir/source.py", end_line=2
        )
        assert str(loc) == "<'dir/source.py': Line 1, Col 1 to Line 2>"

        loc = eve.type_definitions.SourceLocation(
            line=1, column=1, source="dir/source.py", end_line=2, end_column=2
        )
        assert str(loc) == "<'dir/source.py': Line 1, Col 1 to Line 2, Col 2>"

    def test_construction_from_ast(self):
        import ast

        ast_node = ast.parse("a = b + 1").body[0]
        loc = eve.type_definitions.SourceLocation.from_AST(ast_node, "source.py")

        assert loc.line == ast_node.lineno
        assert loc.column == ast_node.col_offset + 1
        assert loc.source == "source.py"
        assert loc.end_line == ast_node.end_lineno
        assert loc.end_column == ast_node.end_col_offset + 1

        loc = eve.type_definitions.SourceLocation.from_AST(ast_node)

        assert loc.line == ast_node.lineno
        assert loc.column == ast_node.col_offset + 1
        assert loc.source == f"<ast.Assign at 0x{id(ast_node):x}>"
        assert loc.end_line == ast_node.end_lineno
        assert loc.end_column == ast_node.end_col_offset + 1


class TestSourceLocationGroup:
    def test_valid_locations(self):
        loc1 = eve.type_definitions.SourceLocation(line=1, column=1, source="source1.py")
        loc2 = eve.type_definitions.SourceLocation(line=2, column=2, source="source2.py")
        eve.type_definitions.SourceLocationGroup(loc1)
        eve.type_definitions.SourceLocationGroup(loc1, loc2)
        eve.type_definitions.SourceLocationGroup(loc1, loc1, loc2, loc2, context="test context")

    def test_invalid_locations(self):
        with pytest.raises(pydantic.ValidationError):
            eve.type_definitions.SourceLocationGroup()
        loc1 = eve.type_definitions.SourceLocation(line=1, column=1, source="source.py")
        with pytest.raises(pydantic.ValidationError):
            eve.type_definitions.SourceLocationGroup(loc1, "loc2")

    def test_str(self):
        loc1 = eve.type_definitions.SourceLocation(line=1, column=1, source="source1.py")
        loc2 = eve.type_definitions.SourceLocation(line=2, column=2, source="source2.py")
        loc = eve.type_definitions.SourceLocationGroup(loc1, loc2, context="some context")
        assert (
            str(loc)
            == "<#some context#[<'source1.py': Line 1, Col 1>, <'source2.py': Line 2, Col 2>]>"
        )
