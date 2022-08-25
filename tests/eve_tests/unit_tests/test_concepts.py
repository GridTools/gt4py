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

import copy
import re

import pytest

import eve

from .. import definitions


def test_symbol_types():
    from eve.concepts import SymbolName

    assert SymbolName("valid_name_01A") == "valid_name_01A"
    assert SymbolName("valid_name_01A") == "valid_name_01A"
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        SymbolName("$name_01A")
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        SymbolName("0name_01A")
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        SymbolName("name_01A ")

    class LettersOnlySymbol(SymbolName, regex=re.compile(r"[a-zA-Z]+$")):
        __slots__ = ()

    assert LettersOnlySymbol("validNAME") == "validNAME"
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        LettersOnlySymbol("name_a")
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        LettersOnlySymbol("name01")


class TestSourceLocation:
    def test_valid_position(self):
        eve.concepts.SourceLocation(line=1, column=1, source="source.py")

    def test_invalid_position(self):
        with pytest.raises(ValueError, match="column"):
            eve.concepts.SourceLocation(line=1, column=-1, source="source.py")

    def test_str(self):
        loc = eve.concepts.SourceLocation(line=1, column=1, source="dir/source.py")
        assert str(loc) == "<dir/source.py:1:1>"

        loc = eve.concepts.SourceLocation(line=1, column=1, source="dir/source.py", end_line=2)
        assert str(loc) == "<dir/source.py:1:1 to 2>"

        loc = eve.concepts.SourceLocation(
            line=1, column=1, source="dir/source.py", end_line=2, end_column=2
        )
        assert str(loc) == "<dir/source.py:1:1 to 2:2>"

    def test_construction_from_ast(self):
        import ast

        ast_node = ast.parse("a = b + 1").body[0]
        loc = eve.concepts.SourceLocation.from_AST(ast_node, "source.py")

        assert loc.line == ast_node.lineno
        assert loc.column == ast_node.col_offset + 1
        assert loc.source == "source.py"
        assert loc.end_line == ast_node.end_lineno
        assert loc.end_column == ast_node.end_col_offset + 1

        loc = eve.concepts.SourceLocation.from_AST(ast_node)

        assert loc.line == ast_node.lineno
        assert loc.column == ast_node.col_offset + 1
        assert loc.source == f"<ast.Assign at 0x{id(ast_node):x}>"
        assert loc.end_line == ast_node.end_lineno
        assert loc.end_column == ast_node.end_col_offset + 1


class TestSourceLocationGroup:
    def test_valid_locations(self):
        loc1 = eve.concepts.SourceLocation(line=1, column=1, source="source1.py")
        loc2 = eve.concepts.SourceLocation(line=2, column=2, source="source2.py")
        eve.concepts.SourceLocationGroup(loc1)
        eve.concepts.SourceLocationGroup(loc1, loc2)
        eve.concepts.SourceLocationGroup(loc1, loc1, loc2, loc2, context="test context")

    def test_invalid_locations(self):
        with pytest.raises(ValueError):
            eve.concepts.SourceLocationGroup()
        loc1 = eve.concepts.SourceLocation(line=1, column=1, source="source.py")
        with pytest.raises(TypeError):
            eve.concepts.SourceLocationGroup(loc1, "loc2")

    def test_str(self):
        loc1 = eve.concepts.SourceLocation(line=1, column=1, source="source1.py")
        loc2 = eve.concepts.SourceLocation(line=2, column=2, source="source2.py")
        loc = eve.concepts.SourceLocationGroup(loc1, loc2, context="some context")
        assert str(loc) == "<#some context#[<source1.py:1:1>, <source2.py:2:2>]>"


class TestNode:
    def test_validation(self, invalid_sample_node_maker):
        with pytest.raises((TypeError, ValueError)):
            invalid_sample_node_maker()

    def test_unique_id(self, sample_node_maker):
        node_a = sample_node_maker()
        node_b = sample_node_maker()
        node_c = sample_node_maker()

        assert id(node_a) != id(node_b) != id(node_c)

    def test_annex(self, sample_node):
        assert isinstance(sample_node.annex, eve.utils.Namespace)

        sample_node.annex.an_int = 32
        assert sample_node.annex.an_int == 32

        sample_node.annex.an_int = -32
        assert sample_node.annex.an_int == -32

        sample_node.annex.a_str = "foo"
        assert sample_node.annex.a_str == "foo"

        assert set(sample_node.annex.keys()) >= {"an_int", "a_str"}

    def test_children(self, sample_node):
        children_names = set(name for name, _ in sample_node.iter_children_items())

        assert not any(name.endswith("__") for name in children_names)
        assert not any(name.endswith("_") for name in children_names)

        assert all(
            node1 is node2
            for (name, node1), node2 in zip(
                sample_node.iter_children_items(), sample_node.iter_children_values()
            )
        )


class TestEqNonlocated:
    def test_source_location(self):
        node = definitions.make_simple_node_with_loc()

        node_different_loc = copy.copy(node)
        node_different_loc.loc = definitions.make_source_location()
        assert node != node_different_loc
        assert eve.concepts.eq_nonlocated(node, node_different_loc)

        node_different_value_and_loc = copy.copy(node_different_loc)
        node_different_value_and_loc.str_value = definitions.make_str_value()
        assert not eve.concepts.eq_nonlocated(node, node_different_value_and_loc)

    def test_source_location_group(self):
        node = definitions.make_simple_node_with_loc()

        node_different_loc_group = copy.copy(node)
        node_different_loc_group.loc = definitions.make_source_location_group()
        assert node != node_different_loc_group
        assert eve.concepts.eq_nonlocated(node, node_different_loc_group)

        node_different_value_and_loc_group = copy.copy(node_different_loc_group)
        node_different_value_and_loc_group.str_value = definitions.make_str_value()
        assert not eve.concepts.eq_nonlocated(node, node_different_value_and_loc_group)
