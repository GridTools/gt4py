# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import pytest

from gt4py.next import utils


def test_tree_map_default():
    @utils.tree_map
    def testee(x):
        return x + 1

    assert testee(((1, 2), 3)) == ((2, 3), 4)


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


@pytest.mark.parametrize(
    "options",
    [
        dict(allow_kwargs_mutation=allow_kwargs_mutation, sort_kwargs=sort_kwargs)
        for allow_kwargs_mutation in (True, False)
        for sort_kwargs in (True, False)
    ],
    ids=lambda option: str.join(", ", (f"{k}={v}" for k, v in option.items())),
)
def test_default_make_args_canonicalizer(options: dict[str, bool]):
    def func(a, /, b, *, c, d):
        return a + b + c + d

    signature = inspect.signature(func)
    canonicalizer = utils.make_args_canonicalizer(signature, **options)
    canonical_form = ((1, 2), {"c": 3, "d": 4})

    in_place_kwargs = options["allow_kwargs_mutation"] is False or options["sort_kwargs"] is True
    output = canonicalizer((1, 2), kwargs := {"c": 3, "d": 4})
    assert output == canonical_form
    assert output[1] is kwargs or in_place_kwargs
    assert canonicalizer.cache_info().currsize == 1

    assert canonicalizer((1, 2), {"d": 4, "c": 3}) == canonical_form
    assert output[1] is kwargs or in_place_kwargs
    assert tuple(output[1].keys()) == ("c", "d") or not options["sort_kwargs"]
    assert canonicalizer.cache_info().currsize == 1

    assert canonicalizer((1,), {"b": 2, "c": 3, "d": 4}) == canonical_form
    assert canonicalizer.cache_info().currsize == 2

    assert canonicalizer((), {"d": 4, "a": 1, "b": 2, "c": 3}) == canonical_form
    assert canonicalizer.cache_info().currsize == 3

    with pytest.raises(ValueError, match="Missing keyword arguments: {'c'}"):
        canonicalizer((1, 2), {"d": 4})

    with pytest.raises(ValueError, match="Got unexpected keyword arguments: {'foo'}"):
        canonicalizer((1, 2), {"d": 4, "foo": 5})

    with pytest.raises(ValueError, match="Too many positional arguments"):
        canonicalizer((1, 2, 3, 4, 5), {})  # too many positional arguments
