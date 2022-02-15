import types
from typing import Any, TypedDict

import pytest

from eve import Node
from eve.pattern_matching import ModuleWrapper, ObjectPattern, get_differences


class TestData(TypedDict):
    name: str
    a: Any
    b: Any
    expected_diff: list[tuple[str, str]]


test_data: list[tuple[str, TestData]] = []


def register_test_case(name, a, b, expected_diff):
    test_data.append((name, {"a": a, "b": b, "expected_diff": expected_diff}))


class TestNode(Node):
    foo: str
    bar: str


register_test_case("int_equal", 1, 1, [])
register_test_case("int_unequal", 1, 2, [("a", "Values are not equal. `1` != `2`")])
register_test_case("list_equal", [1], [1], [])
register_test_case(
    "list_unequal", [1], [1, 2], [("a", "Expected list of length 1, but got length 2")]
)
register_test_case("dict_equal", {"foo": 1}, {"foo": 1}, [])
register_test_case(
    "dict_unequal_val", {"foo": 1}, {"foo": 2}, [('a["foo"]', "Values are not equal. `1` != `2`")]
)
register_test_case(
    "dict_unequal_key",
    {"foo": 1},
    {"bar": 1},
    [
        ("a", "Expected dictionary with keys `foo`, but the following keys are missing: `foo`"),
        ("a", "Expected dictionary with keys `foo`, but the following keys are extra: `bar`"),
    ],
)
register_test_case(
    "dict_unequal_empty",
    {"foo": 1},
    {},
    [("a", "Expected dictionary with keys `foo`, but the following keys are missing: `foo`")],
)
register_test_case(
    "node_pattern_match",
    ObjectPattern(TestNode, {"bar": "baz"}),
    TestNode(bar="baz", foo="bar"),
    [],
)
register_test_case(
    "node_pattern_match",
    ObjectPattern(TestNode, {"bar": "bar"}),
    TestNode(bar="baz", foo="bar"),
    [("a.bar", "Values are not equal. `bar` != `baz`")],
)


@pytest.mark.parametrize("name,test_data", test_data)
def test_all(name, test_data):
    a, b, expected_diff = test_data["a"], test_data["b"], test_data["expected_diff"]
    diff = list(get_differences(a, b, path="a"))
    assert diff == expected_diff


def test_module_wrapper():
    test_mod = types.ModuleType("test_mod")
    test_mod.TestNode = TestNode
    test_mod_ = ModuleWrapper(test_mod)

    assert test_mod_.TestNode(bar="baz").matches(test_mod.TestNode(bar="baz", foo="bar"))
    assert not test_mod_.TestNode(bar="bar").matches(test_mod.TestNode(bar="baz", foo="bar"))
