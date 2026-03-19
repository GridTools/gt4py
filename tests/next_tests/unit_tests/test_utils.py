# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import inspect
import pickle
import pytest

from gt4py.eve import datamodels
from gt4py.next import utils


# Module-level classes so pickle can resolve them by qualified name.
@dataclasses.dataclass
class _DataclassModel(utils.MetadataBasedPickling):
    value: int
    transient: str = dataclasses.field(default="skip", metadata=utils.gt4py_metadata(pickle=False))


@dataclasses.dataclass(slots=True)
class _SlottedDataclassModel(utils.MetadataBasedPickling):
    value: int
    transient: str = dataclasses.field(default="skip", metadata=utils.gt4py_metadata(pickle=False))


@datamodels.datamodel(slots=False)
class _DatamodelModel(utils.MetadataBasedPickling):
    value: int
    transient: str = datamodels.field(default="skip", metadata=utils.gt4py_metadata(pickle=False))


@dataclasses.dataclass
class _EmptyDataclassModel(utils.MetadataBasedPickling):
    pass


@dataclasses.dataclass(slots=True)
class _EmptySlottedDataclassModel(utils.MetadataBasedPickling):
    pass


@datamodels.datamodel(slots=False)
class _EmptyDatamodelModel(utils.MetadataBasedPickling):
    pass


class TestMetadataBasedPickling:
    def test_get_metadata_based_getstate_rejects_non_dataclass_like_type(self):
        with pytest.raises(TypeError, match="Expected a dataclass or datamodel type"):
            utils._get_metadata_based_state_getstate(object)

    @pytest.mark.parametrize(
        "instance,expected_state",
        [
            (_DataclassModel(1, "foo"), {"value": 1}),
            (_SlottedDataclassModel(1, "foo"), (None, {"value": 1})),
            (_DatamodelModel(1, "foo"), {"value": 1}),
            (_EmptyDataclassModel(), {}),
            (_EmptySlottedDataclassModel(), None),
            (_EmptyDatamodelModel(), {}),
        ],
    )
    def test_get_metadata_based_getstate(self, instance, expected_state):
        cls = type(instance)
        getstate = utils._get_metadata_based_state_getstate(cls)
        assert getstate is utils._get_metadata_based_state_getstate(cls)  # cached

        assert getstate(instance) == expected_state

    @pytest.mark.parametrize(
        "instance,expected_fields",
        [
            (_DataclassModel(42, "skip"), {"value": 42}),
            (_SlottedDataclassModel(42, "skip"), {"value": 42}),
            (_DatamodelModel(42, "skip"), {"value": 42}),
            (_EmptyDataclassModel(), {}),
            (_EmptySlottedDataclassModel(), {}),
            (_EmptyDatamodelModel(), {}),
        ],
    )
    def test_pickle_roundtrip(self, instance, expected_fields):
        obj = instance
        restored = pickle.loads(pickle.dumps(obj))

        for field_name, expected_value in expected_fields.items():
            assert getattr(restored, field_name) == expected_value


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

    def test_empty_signature(self):
        def func():
            return 42

        canonicalizer = utils.make_args_canonicalizer(inspect.signature(func), name=func.__name__)
        assert canonicalizer() == ((), {})

    def pass_through_signature(self):
        def func(*all_pos_args, **all_kwargs):
            return 42

        canonicalizer = utils.make_args_canonicalizer(inspect.signature(func), name=func.__name__)
        assert canonicalizer(1, 2, 3, foo="foo value", bar="bar value") == (
            (1, 2, 3),
            {"foo": "foo value", "bar": "bar_value"},
        )


def test_make_args_canonicalizer_for_function():
    def func(a, /, b, *, c, d):
        return a + b + c + d

    canonicalizer = utils.make_args_canonicalizer_for_function(func)

    args, kwargs = (1,), {"c": 3, "d": 4, "b": 2}
    bound_args = inspect.signature(func).bind(*args, **kwargs)
    canonical_form = bound_args.args, bound_args.kwargs
    assert canonicalizer(*args, **kwargs) == canonical_form

    # Test caching
    assert canonicalizer is utils.make_args_canonicalizer_for_function(func)


def test_canonicalize_call_args():
    def func(a, /, b, *, c, d):
        return a + b + c + d

    args, kwargs = (1,), {"c": 3, "d": 4, "b": 2}
    bound_args = inspect.signature(func).bind(*args, **kwargs)
    canonical_form = bound_args.args, bound_args.kwargs
    assert utils.canonicalize_call_args(func, args, kwargs) == canonical_form
