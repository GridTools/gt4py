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

from typing import Any

import pytest

from eve import Node
from eve.pattern_matching import ObjectPattern, get_differences


class SampleNode(Node):
    foo: str
    bar: str


class NestedSampleNode(Node):
    foo: str
    bar: SampleNode


test_data: list[tuple[str, Any, Any, list[tuple[str, str]]]] = [
    ("int_equal", 1, 1, []),
    ("int_unequal", 1, 2, [("a", "Values are not equal. `1` != `2`")]),
    ("list_equal", [1], [1], []),
    ("list_unequal", [1], [1, 2], [("a", "Expected list of length 1, but got length 2")]),
    ("dict_equal", {"foo": 1}, {"foo": 1}, []),
    (
        "dict_unequal_val",
        {"foo": 1},
        {"foo": 2},
        [('a["foo"]', "Values are not equal. `1` != `2`")],
    ),
    (
        "dict_unequal_key",
        {"foo": 1},
        {"bar": 1},
        [
            ("a", "Expected dictionary with keys `foo`, but the following keys are missing: `foo`"),
            ("a", "Expected dictionary with keys `foo`, but the following keys are extra: `bar`"),
        ],
    ),
    (
        "dict_unequal_empty",
        {"foo": 1},
        {},
        [("a", "Expected dictionary with keys `foo`, but the following keys are missing: `foo`")],
    ),
    ("dict_nested_equal", {"bar": {"foo": 1}}, {"bar": {"foo": 1}}, []),
    (
        "dict_nested_unequal_val",
        {"bar": {"foo": 1}},
        {"bar": {"foo": 2}},
        [('a["bar"]["foo"]', "Values are not equal. `1` != `2`")],
    ),
    (
        "dict_unequal_key",
        {"bar": {"foo": 1}},
        {"bar": {"bar": 2}},
        [
            (
                'a["bar"]',
                "Expected dictionary with keys `foo`, but the following keys are missing: `foo`",
            ),
            (
                'a["bar"]',
                "Expected dictionary with keys `foo`, but the following keys are extra: `bar`",
            ),
        ],
    ),
    (
        "node_pattern_match",
        ObjectPattern(SampleNode, bar="baz"),
        SampleNode(bar="baz", foo="bar"),
        [],
    ),
    (
        "node_pattern_no_match",
        ObjectPattern(SampleNode, bar="bar"),
        SampleNode(bar="baz", foo="bar"),
        [("a.bar", "Values are not equal. `bar` != `baz`")],
    ),
    (
        "nested_node_pattern_match",
        ObjectPattern(NestedSampleNode, bar=ObjectPattern(SampleNode, foo="baz")),
        NestedSampleNode(foo="bar", bar=SampleNode(bar="baz", foo="baz")),
        [],
    ),
    (
        "nested_node_pattern_no_match",
        ObjectPattern(NestedSampleNode, bar=ObjectPattern(SampleNode, foo="bar")),
        NestedSampleNode(foo="bar", bar=SampleNode(bar="baz", foo="baz")),
        [("a.bar.foo", "Values are not equal. `bar` != `baz`")],
    ),
]


@pytest.mark.parametrize("name,a,b,expected_diff", test_data)
def test_all(name, a, b, expected_diff):
    diff = list(get_differences(a, b, path="a"))
    assert diff == expected_diff
