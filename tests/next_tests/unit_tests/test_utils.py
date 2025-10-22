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
    @pytest.fixture(
        autouse=True,
        params=(
            options := [
                dict(allow_kwargs_mutation=allow_kwargs_mutation, sort_kwargs=sort_kwargs)
                for allow_kwargs_mutation in (True, False)
                for sort_kwargs in (True, False)
            ]
        ),
        ids=[str.join(", ", (f"{k}={v}" for k, v in option.items())) for option in options],
    )
    def setup(self, request):
        def func(a, /, b, *, c, d):
            return a + b + c + d

        self.func = func
        self.func_signature = inspect.signature(func)

        self.options = request.param
        self.canonicalizer = utils.make_args_canonicalizer(
            self.func_signature, name=self.func.__name__, **request.param
        )
        self.in_place_kwargs = (
            request.param["allow_kwargs_mutation"] is True and request.param["sort_kwargs"] is False
        )

    def test_canonical_form(self):
        args, kwargs = (1, 2), {"c": 3, "d": 4}
        bound_args = self.func_signature.bind(*args, **kwargs)
        canonical_form = bound_args.args, bound_args.kwargs
        assert self.canonicalizer(args, kwargs) == canonical_form

        args, kwargs = (1,), {"c": 3, "d": 4, "b": 2}
        bound_args = self.func_signature.bind(*args, **kwargs)
        canonical_form = bound_args.args, bound_args.kwargs
        assert self.canonicalizer(args, kwargs) == canonical_form

    def test_wrong_input(self):
        with pytest.raises(ValueError, match="Missing keyword arguments: {'c'}"):
            self.canonicalizer((1, 2), {"d": 4})

        with pytest.raises(ValueError, match="Got unexpected keyword arguments: {'foo'}"):
            self.canonicalizer((1, 2), {"d": 4, "foo": 5})

        with pytest.raises(ValueError, match="Got unexpected keyword arguments: {'a'}"):
            self.canonicalizer((2,), {"a": 1, "c": 3, "d": 4})

        with pytest.raises(ValueError, match="Too many positional arguments"):
            self.canonicalizer((1, 2, 3, 4, 5), {})  # too many positional arguments

    def test_allow_kwargs_mutation(self):
        args, kwargs = (1,), {"c": 3, "b": 2, "d": 4}
        canonical_args, canonical_kwargs = self.canonicalizer(args, kwargs)

        assert canonical_kwargs.keys() == {"c", "d"}
        assert (canonical_kwargs is not kwargs) or self.canonicalizer.options.allow_kwargs_mutation
        assert (
            self.canonicalizer.options.allow_kwargs_mutation
            == self.options["allow_kwargs_mutation"]
        )
        assert (
            self.in_place_kwargs is self.options["allow_kwargs_mutation"]
            or self.options["sort_kwargs"]
        )

    def test_cache(self):
        args, kwargs = (1, 2), {"c": 3, "d": 4}

        assert self.canonicalizer.cache_info().currsize == 0
        _ = self.canonicalizer(args, kwargs)
        assert self.canonicalizer.cache_info().currsize == 1
        _ = self.canonicalizer(args, kwargs)
        assert self.canonicalizer.cache_info().currsize == 1

        self.canonicalizer.cache_clear()
        assert self.canonicalizer.cache_info().currsize == 0
        _ = self.canonicalizer(args, kwargs)
        assert self.canonicalizer.cache_info().currsize == 1

    def test_sort_kwargs(self):
        args, kwargs = (1, 2), {"d": 3, "c": 4}
        canonical_form = self.func_signature.bind(*args, **kwargs)
        canonical_args, canonical_kwargs = self.canonicalizer(args, kwargs)

        assert canonical_kwargs.keys() == {"c", "d"}
        assert tuple(canonical_kwargs.keys()) == ("c", "d") or not self.options["sort_kwargs"]
        assert (canonical_kwargs is not kwargs) or not self.canonicalizer.options.sort_kwargs
        assert self.canonicalizer.options.sort_kwargs == self.options["sort_kwargs"]
