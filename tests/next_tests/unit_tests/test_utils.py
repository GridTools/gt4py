# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import pytest
from typing import Final

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
    @utils.tree_map(result_collection_constructor=lambda _, elts: list(elts))
    def testee(x):
        return x + 1

    assert testee(((1, 2), 3)) == [[2, 3], 4]


def test_tree_map_multiple_input_types():
    @utils.tree_map(
        collection_type=(list, tuple),
        result_collection_constructor=lambda value, elts: tuple(elts)
        if isinstance(value, list)
        else list(elts),
    )
    def testee(x):
        return x + 1

    assert testee([(1, [2]), 3]) == ([2, (3,)], 4)


class TestArgsCanonicalizer:
    @pytest.fixture(autouse=True)
    def setup(self, request):
        def func(a, /, b, *, c, d):
            return a + b + c + d

        self.func = func
        self.func_signature = inspect.signature(func)
        self.canonicalizer = utils.make_args_canonicalizer(
            self.func_signature, name=self.func.__name__
        )

    def test_canonical_form(self):
        args, kwargs = (1, 2), {"c": 3, "d": 4}
        bound_args = self.func_signature.bind(*args, **kwargs)
        canonical_form = bound_args.args, bound_args.kwargs
        assert self.canonicalizer(*args, **kwargs) == canonical_form

        args, kwargs = (1,), {"c": 3, "d": 4, "b": 2}
        bound_args = self.func_signature.bind(*args, **kwargs)
        canonical_form = bound_args.args, bound_args.kwargs
        assert self.canonicalizer(*args, **kwargs) == canonical_form

    def test_wrong_input(self):
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'c'"):
            self.canonicalizer(*(1, 2), **{"d": 4})

        with pytest.raises(TypeError, match="got an unexpected keyword argument 'foo'"):
            self.canonicalizer(*(1, 2), **{"d": 4, "foo": 5})

        with pytest.raises(
            TypeError, match="got some positional-only arguments passed as keyword arguments: 'a'"
        ):
            self.canonicalizer(*(2,), **{"a": 1, "c": 3, "d": 4})

        with pytest.raises(TypeError, match="takes 2 positional arguments but 5 were given"):
            self.canonicalizer(*(1, 2, 3, 4, 5), **{})
