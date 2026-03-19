# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import re

import pytest

from gt4py import eve

from .. import definitions


def test_symbol_types():
    from gt4py.eve.concepts import SymbolName

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
        eve.concepts.SourceLocation(line=1, column=1, filename="source.py")

    def test_invalid_position(self):
        with pytest.raises(ValueError, match="column"):
            eve.concepts.SourceLocation(line=1, column=-1, filename="source.py")

    def test_str(self):
        loc = eve.concepts.SourceLocation(line=1, column=1, filename="dir/source.py")
        assert str(loc) == "<dir/source.py:1:1>"

        loc = eve.concepts.SourceLocation(line=1, column=1, filename="dir/source.py", end_line=2)
        assert str(loc) == "<dir/source.py:1:1 to 2>"

        loc = eve.concepts.SourceLocation(
            line=1, column=1, filename="dir/source.py", end_line=2, end_column=2
        )
        assert str(loc) == "<dir/source.py:1:1 to 2:2>"


class TestSourceLocationGroup:
    def test_valid_locations(self):
        loc1 = eve.concepts.SourceLocation(line=1, column=1, filename="source1.py")
        loc2 = eve.concepts.SourceLocation(line=2, column=2, filename="source2.py")
        eve.concepts.SourceLocationGroup(loc1)
        eve.concepts.SourceLocationGroup(loc1, loc2)
        eve.concepts.SourceLocationGroup(loc1, loc1, loc2, loc2, context="test context")

    def test_invalid_locations(self):
        with pytest.raises(ValueError):
            eve.concepts.SourceLocationGroup()
        loc1 = eve.concepts.SourceLocation(line=1, column=1, filename="source.py")
        with pytest.raises(TypeError):
            eve.concepts.SourceLocationGroup(loc1, "loc2")

    def test_str(self):
        loc1 = eve.concepts.SourceLocation(line=1, column=1, filename="source1.py")
        loc2 = eve.concepts.SourceLocation(line=2, column=2, filename="source2.py")
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


def test_skipping_fields_node_pickler_skips_nested_fields_and_is_cached():
    skipped_field_pickler = eve.concepts.skipping_fields_node_pickler("int_value")
    assert skipped_field_pickler is eve.concepts.skipping_fields_node_pickler("int_value")

    node_a = definitions.CompoundNode(
        int_value=1,
        location=definitions.make_location_node(fixed=True),
        simple=definitions.make_simple_node(fixed=True),
        simple_loc=definitions.make_simple_node_with_loc(fixed=True),
        simple_opt=definitions.make_simple_node_with_optionals(fixed=True),
        other_simple_opt=None,
    )
    node_b = copy.deepcopy(node_a)

    node_b.int_value += 100
    node_b.simple.int_value += 100
    node_b.simple_loc.int_value += 100
    node_b.simple_opt.int_value += 100

    assert eve.utils.content_hash(node_a, pickler=skipped_field_pickler) == eve.utils.content_hash(
        node_b, pickler=skipped_field_pickler
    )

    node_b.simple.str_value = "changed"
    assert eve.utils.content_hash(node_a, pickler=skipped_field_pickler) != eve.utils.content_hash(
        node_b, pickler=skipped_field_pickler
    )
