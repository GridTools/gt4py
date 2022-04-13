from typing import Any

import pytest

from eve import Node
from eve.pattern_matching import ObjectPattern, get_differences


class TestNode(Node):
    foo: str
    bar: str


class NestedTestNode(Node):
    foo: str
    bar: TestNode


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
        ObjectPattern(TestNode, bar="baz"),
        TestNode(bar="baz", foo="bar"),
        [],
    ),
    (
        "node_pattern_no_match",
        ObjectPattern(TestNode, bar="bar"),
        TestNode(bar="baz", foo="bar"),
        [("a.bar", "Values are not equal. `bar` != `baz`")],
    ),
    (
        "nested_node_pattern_match",
        ObjectPattern(NestedTestNode, bar=ObjectPattern(TestNode, foo="baz")),
        NestedTestNode(foo="bar", bar=TestNode(bar="baz", foo="baz")),
        [],
    ),
    (
        "nested_node_pattern_no_match",
        ObjectPattern(NestedTestNode, bar=ObjectPattern(TestNode, foo="bar")),
        NestedTestNode(foo="bar", bar=TestNode(bar="baz", foo="baz")),
        [("a.bar.foo", "Values are not equal. `bar` != `baz`")],
    ),
]


@pytest.mark.parametrize("name,a,b,expected_diff", test_data)
def test_all(name, a, b, expected_diff):
    diff = list(get_differences(a, b, path="a"))
    assert diff == expected_diff
