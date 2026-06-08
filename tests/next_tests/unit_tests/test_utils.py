# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
import inspect
import pickle
import pytest

from gt4py.eve import datamodels
from gt4py.next import utils

from eve_tests import definitions


# Module-level classes so pickle can resolve them by qualified name.
@dataclasses.dataclass
class _DataclassModel(utils.MetadataBasedPicklingMixin):
    value: int
    transient: str = dataclasses.field(default="skip", metadata=utils.gt4py_metadata(pickle=False))


@dataclasses.dataclass(slots=True)
class _SlottedDataclassModel(utils.MetadataBasedPicklingMixin):
    value: int
    transient: str = dataclasses.field(default="skip", metadata=utils.gt4py_metadata(pickle=False))


@datamodels.datamodel(slots=False)
class _DatamodelModel(utils.MetadataBasedPicklingMixin):
    value: int
    transient: str = datamodels.field(default="skip", metadata=utils.gt4py_metadata(pickle=False))


@dataclasses.dataclass
class _EmptyDataclassModel(utils.MetadataBasedPicklingMixin):
    pass


@dataclasses.dataclass(slots=True)
class _EmptySlottedDataclassModel(utils.MetadataBasedPicklingMixin):
    pass


@datamodels.datamodel(slots=False)
class _EmptyDatamodelModel(utils.MetadataBasedPicklingMixin):
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


def test_skipping_fields_node_fingerprinter_skips_nested_fields_and_is_cached():
    fingerprinter = utils.skipping_fields_node_fingerprinter("int_value")
    assert fingerprinter is utils.skipping_fields_node_fingerprinter("int_value")

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

    assert fingerprinter(node_a) == fingerprinter(node_b)

    node_b.simple.str_value = "changed"
    assert fingerprinter(node_a) != fingerprinter(node_b)


def test_skipping_fields_node_fingerprinter_returns_pickler_when_requested():
    fingerprinter, pickler = utils.skipping_fields_node_fingerprinter(
        "int_value", return_pickler=True
    )
    assert callable(fingerprinter)
    assert isinstance(pickler, type) and issubclass(pickler, pickle.Pickler)
    assert pickler.skipped_fields == {"int_value"}

    # Without `return_pickler` only the fingerprinter callable is returned.
    assert callable(utils.skipping_fields_node_fingerprinter("int_value"))


class TestStableFingerprinter:
    def test_returns_stable_string(self):
        fingerprint = utils.stable_fingerprinter({"a": 1})
        assert isinstance(fingerprint, str)
        assert fingerprint == utils.stable_fingerprinter({"a": 1})

    def test_dicts_are_order_independent(self):
        assert utils.stable_fingerprinter({"a": 1, "b": 2}) == utils.stable_fingerprinter(
            {"b": 2, "a": 1}
        )

    def test_sets_are_order_independent(self):
        assert utils.stable_fingerprinter({1, 2, 3}) == utils.stable_fingerprinter({3, 2, 1})

    def test_distinguishes_different_content(self):
        assert utils.stable_fingerprinter({"a": 1}) != utils.stable_fingerprinter({"a": 2})

    def test_dicts_keyed_by_types_are_order_independent(self):
        # Dicts keyed by types occur in compile-time metadata (e.g. argument descriptors).
        assert utils.stable_fingerprinter({int: 1, str: 2}) == utils.stable_fingerprinter(
            {str: 2, int: 1}
        )

    def test_types_are_fingerprinted_by_reference(self):
        # Regression for the `type` reducer that used to recurse infinitely: types nested in
        # containers must be fingerprinted by reference, deterministically and distinctly.
        assert utils.stable_fingerprinter({"t": int}) == utils.stable_fingerprinter({"t": int})
        assert utils.stable_fingerprinter({"t": int}) != utils.stable_fingerprinter({"t": str})

    def test_functions_are_fingerprinted_by_reference(self):
        def foo() -> None: ...

        def bar() -> None: ...

        assert utils.stable_fingerprinter({"f": foo}) == utils.stable_fingerprinter({"f": foo})
        assert utils.stable_fingerprinter({"f": foo}) != utils.stable_fingerprinter({"f": bar})

    def test_modules_are_fingerprinted_by_reference(self):
        import os
        import sys

        # Modules are unpicklable by the pure-Python pickler unless reduced by reference.
        assert utils.stable_fingerprinter({"m": os}) == utils.stable_fingerprinter({"m": os})
        assert utils.stable_fingerprinter({"m": os}) != utils.stable_fingerprinter({"m": sys})

    def test_does_not_recurse_on_nested_types_modules_and_functions(self):
        import os

        # Regression: fingerprinting a built-in container holding types/modules/functions
        # used to raise RecursionError (and later TypeError); it must just produce a hash.
        payload = {"types": [int, str], "module": os, "nested": {float: ("x",)}}
        assert isinstance(utils.stable_fingerprinter(payload), str)


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
        result_collection_constructor=lambda value, elts: (
            tuple(elts) if isinstance(value, list) else list(elts)
        ),
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
