# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import pytest

from gt4py.next import utils
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.type_system import type_info
from gt4py.next.common import Field
from numpy import int64


def test_tree_map_scalar():
    @utils.tree_map(collection_type=ts.ScalarType, result_collection_constructor=tuple)
    def testee(x):
        return x + 1

    assert testee(1) == (2)


def test_apply_to_primitive_constituents():
    int_type = ts.ScalarType(kind=ts.ScalarKind.INT64)
    tuple_type = ts.TupleType(types=[ts.TupleType(types=[int_type, int_type]), int_type])

    tree = type_info.type_tree_map(
        lambda primitive_type: ts.FieldType(dims=[], dtype=primitive_type)
    )(tuple_type)

    prim = type_info.apply_to_primitive_constituents(
        lambda primitive_type: ts.FieldType(dims=[], dtype=primitive_type), tuple_type
    )

    assert tree == prim


def test_tree_map_default():
    expected_result = ((2, 3), 4)

    @utils.tree_map
    def testee1(x):
        return x + 1

    assert testee1(((1, 2), 3)) == expected_result
    assert utils.tree_map(lambda x: x + 1)(((1, 2), 3)) == expected_result
    assert utils.tree_map(lambda x: x + 1, ((1, 2), 3)) == expected_result


def test_tree_map_multi_arg():
    @utils.tree_map
    def testee(x, y):
        return x + y

    assert testee(((1, 2), 3), ((4, 5), 6)) == ((5, 7), 9)


def test_tree_map_custom_input_type():
    @utils.tree_map(collection_type=list)
    def testee(x):
        return x + 1

    assert testee([[1, 2], 3]) == [[2, 3], 4]

    with pytest.raises(TypeError):
        testee(((1, 2), 3))  # tries to `((1, 2), 3)` because `tuple_type` is `list`


def test_tree_map_custom_output_type():
    @utils.tree_map(result_collection_constructor=list)
    def testee(x):
        return x + 1

    assert testee(((1, 2), 3)) == [[2, 3], 4]


def test_tree_map_multiple_input_types():
    @utils.tree_map(collection_type=(list, tuple), result_collection_constructor=tuple)
    def testee(x):
        return x + 1

    assert testee([(1, [2]), 3]) == ((2, (3,)), 4)
