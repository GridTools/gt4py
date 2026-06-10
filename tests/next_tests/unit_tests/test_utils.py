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
import pytest

from gt4py.eve import datamodels
from gt4py.next import utils

from eve_tests import definitions


@dataclasses.dataclass
class _DataclassModel:
    value: int
    transient: str = dataclasses.field(
        default="skip", metadata=utils.gt4py_metadata(fingerprint=False)
    )


@dataclasses.dataclass(slots=True)
class _SlottedDataclassModel:
    value: int
    transient: str = dataclasses.field(
        default="skip", metadata=utils.gt4py_metadata(fingerprint=False)
    )


@datamodels.datamodel(slots=False)
class _DatamodelModel:
    value: int
    transient: str = datamodels.field(
        default="skip", metadata=utils.gt4py_metadata(fingerprint=False)
    )


class TestFingerprintFieldMetadata:
    @pytest.mark.parametrize(
        "model_class", [_DataclassModel, _SlottedDataclassModel, _DatamodelModel]
    )
    def test_fields_marked_fingerprint_false_do_not_affect_fingerprint(self, model_class):
        assert utils.stable_fingerprinter(model_class(1, "foo")) == utils.stable_fingerprinter(
            model_class(1, "bar")
        )

    @pytest.mark.parametrize(
        "model_class", [_DataclassModel, _SlottedDataclassModel, _DatamodelModel]
    )
    def test_other_fields_affect_fingerprint(self, model_class):
        assert utils.stable_fingerprinter(model_class(1, "foo")) != utils.stable_fingerprinter(
            model_class(2, "foo")
        )

    def test_different_classes_with_equal_fields_differ(self):
        assert utils.stable_fingerprinter(_DataclassModel(1)) != utils.stable_fingerprinter(
            _SlottedDataclassModel(1)
        )


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


def test_skipping_fields_node_fingerprinter_returns_handlers_when_requested():
    from gt4py.eve import concepts

    fingerprinter, handlers = utils.skipping_fields_node_fingerprinter(
        "int_value", return_handlers=True
    )
    assert callable(fingerprinter)
    # The handlers can be composed into another fingerprinter.
    assert set(handlers) == {concepts.Node}
    assert callable(utils.make_fingerprinter(handlers))

    # Without `return_handlers` only the fingerprinter callable is returned.
    assert callable(utils.skipping_fields_node_fingerprinter("int_value"))


class TestTreeCata:
    @staticmethod
    def _decompose(obj):
        if isinstance(obj, (list, tuple)):
            return utils.TreeNode(metadata=type(obj), children=tuple(obj))
        return utils.TreeLeaf(obj)

    def test_carrier_type_is_generic(self):
        # The reduction result can be of any type, e.g. the structure depth.
        depth = utils.tree_cata(
            [[1, [2]], 3],
            decompose=self._decompose,
            leaf_alg=lambda leaf: 0,
            node_alg=lambda node, child_depths: 1 + max(child_depths, default=0),
        )
        assert depth == 3

    def test_children_are_reduced_before_parents(self):
        # Post-order traversal: the node algebra must always see the already
        # reduced results of its children, in child order.
        visited = []

        def leaf_alg(leaf):
            visited.append(leaf.value)
            return leaf.value

        def node_alg(node, child_results):
            result = list(child_results)
            visited.append(result)
            return result

        utils.tree_cata(
            [1, [2, 3]], decompose=self._decompose, leaf_alg=leaf_alg, node_alg=node_alg
        )
        assert visited == [1, 2, 3, [2, 3], [1, [2, 3]]]

    def test_empty_containers_are_nodes(self):
        # Zero-children nodes must not corrupt the result bookkeeping.
        def leaf_alg(leaf: utils.TreeLeaf) -> tuple:
            return (leaf.value,)

        def node_alg(node: utils.TreeNode, child_results: list[tuple]) -> tuple:
            return (node.metadata, *child_results)

        result = utils.tree_cata(
            [[], ()], decompose=self._decompose, leaf_alg=leaf_alg, node_alg=node_alg
        )
        assert result == (list, (list,), (tuple,))

    def test_deep_structures_do_not_hit_the_recursion_limit(self):
        deeply_nested: tuple = ()
        for _ in range(100_000):
            deeply_nested = (deeply_nested,)
        depth = utils.tree_cata(
            deeply_nested,
            decompose=self._decompose,
            leaf_alg=lambda leaf: 0,
            node_alg=lambda node, child_depths: 1 + max(child_depths, default=0),
        )
        assert depth == 100_001  # 100_000 wrappers + the innermost empty tuple node

    def test_memoization_reduces_shared_subobjects_once(self):
        decompose_calls = []

        def decompose(obj):
            decompose_calls.append(obj)
            return self._decompose(obj)

        shared = [1, 2]
        utils.tree_cata(
            [shared, shared],
            decompose=decompose,
            leaf_alg=lambda leaf: 0,
            node_alg=lambda node, child_results: 0,
        )
        assert sum(1 for obj in decompose_calls if obj is shared) == 1

        decompose_calls.clear()
        utils.tree_cata(
            [shared, shared],
            decompose=decompose,
            leaf_alg=lambda leaf: 0,
            node_alg=lambda node, child_results: 0,
            memoize=False,
        )
        assert sum(1 for obj in decompose_calls if obj is shared) == 2

    def test_cycles_raise_without_cycle_alg(self):
        cyclic: list = [1]
        cyclic.append(cyclic)
        with pytest.raises(ValueError, match="Cycle detected"):
            utils.tree_cata(
                cyclic,
                decompose=self._decompose,
                leaf_alg=lambda leaf: 0,
                node_alg=lambda node, child_results: 0,
            )

    def test_cycles_are_reduced_via_cycle_alg(self):
        cyclic: list = [1]
        cyclic.append(cyclic)
        rendered = utils.tree_cata(
            cyclic,
            decompose=self._decompose,
            leaf_alg=lambda leaf: str(leaf.value),
            node_alg=lambda node, child_results: f"[{','.join(child_results)}]",
            cycle_alg=lambda relative_depth: f"<up:{relative_depth}>",
        )
        assert rendered == "[1,<up:1>]"


def _module_level_func_a() -> None: ...


def _module_level_func_b() -> None: ...


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
        # Importable, module-level functions are identified by their qualified name.
        foo = _module_level_func_a
        bar = _module_level_func_b
        assert utils.stable_fingerprinter({"f": foo}) == utils.stable_fingerprinter({"f": foo})
        assert utils.stable_fingerprinter({"f": foo}) != utils.stable_fingerprinter({"f": bar})

    def test_non_importable_callables_raise(self):
        # Locally defined and anonymous callables can share a qualified name with
        # distinct objects (e.g. each call of a factory produces a new closure named
        # `...<locals>.inner`, and every lambda is named `<lambda>`), so reducing
        # them by reference would silently collide. They must raise instead.
        def local_func() -> None: ...  # qualified name contains `<locals>`

        def make_closure(n: int):
            def inner() -> int:  # distinct objects, identical qualified name
                return n

            return inner

        # Two distinct closures that would otherwise collide by reference.
        assert make_closure(1) is not make_closure(2)

        for non_importable in (local_func, lambda: None, make_closure(1)):
            with pytest.raises(TypeError, match="not importable"):
                utils.stable_fingerprinter({"f": non_importable})

    def test_shadowed_globals_raise(self, monkeypatch):
        import sys

        # An object whose qualified name no longer resolves to itself (e.g. after
        # being redefined interactively) cannot be safely identified by that name.
        original = _module_level_func_a
        monkeypatch.setattr(sys.modules[__name__], "_module_level_func_a", _module_level_func_b)
        with pytest.raises(TypeError, match="not importable"):
            utils.stable_fingerprinter({"f": original})

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

    def test_deep_structures_do_not_hit_the_recursion_limit(self):
        # Regression: lowered IR trees (and other inputs) can nest far deeper than
        # the Python recursion limit allows for recursive traversals.
        deeply_nested: tuple = ()
        for _ in range(100_000):
            deeply_nested = (deeply_nested,)
        assert isinstance(utils.stable_fingerprinter(deeply_nested), str)

        nested_dicts: dict = {}
        for _ in range(10_000):
            nested_dicts = {"k": nested_dicts}
        assert isinstance(utils.stable_fingerprinter(nested_dicts), str)

    def test_dicts_with_unorderable_keys_are_order_independent(self):
        from gt4py.next import common

        # `Dimension`s (and other dataclasses without `__lt__`) occur as dict keys
        # e.g. in user closure variables.
        i, j = common.Dimension("I"), common.Dimension("J")
        assert utils.stable_fingerprinter({i: 1, j: 2}) == utils.stable_fingerprinter({j: 2, i: 1})
        assert utils.stable_fingerprinter({i: 1, j: 2}) != utils.stable_fingerprinter({i: 2, j: 1})

    def test_fingerprint_is_a_function_of_value_not_identity(self):
        # Shared sub-objects and equal copies must produce the same fingerprint.
        d = {"a": 1}
        assert utils.stable_fingerprinter((d, d)) == utils.stable_fingerprinter((d, dict(d)))

    def test_ordered_dicts_are_order_sensitive(self):
        import collections

        # Unlike `dict`, `collections.OrderedDict` equality is order-sensitive,
        # so differently ordered instances must not collide.
        od1 = collections.OrderedDict([("a", 1), ("b", 2)])
        od2 = collections.OrderedDict([("b", 2), ("a", 1)])
        assert od1 != od2
        assert utils.stable_fingerprinter(od1) != utils.stable_fingerprinter(od2)

    def test_defaultdict_factories_are_fingerprinted(self):
        import collections

        # The `default_factory` is part of a `defaultdict`'s observable behavior.
        dd_int = collections.defaultdict(int, {"a": 1})
        assert utils.stable_fingerprinter(dd_int) == utils.stable_fingerprinter(
            collections.defaultdict(int, {"a": 1})
        )
        assert utils.stable_fingerprinter(dd_int) != utils.stable_fingerprinter(
            collections.defaultdict(list, {"a": 1})
        )
        # A `defaultdict` is also distinguished from a plain `dict` with equal items.
        assert utils.stable_fingerprinter(dd_int) != utils.stable_fingerprinter({"a": 1})
        # The items contribution remains order-insensitive.
        assert utils.stable_fingerprinter(
            collections.defaultdict(int, {"a": 1, "b": 2})
        ) == utils.stable_fingerprinter(collections.defaultdict(int, {"b": 2, "a": 1}))

    def test_distinguishes_equal_values_of_different_types(self):
        assert utils.stable_fingerprinter(True) != utils.stable_fingerprinter(1)
        assert utils.stable_fingerprinter(1) != utils.stable_fingerprinter(1.0)
        assert utils.stable_fingerprinter((1, 2)) != utils.stable_fingerprinter([1, 2])
        assert utils.stable_fingerprinter("1") != utils.stable_fingerprinter(b"1")

    def test_self_referential_structures_are_supported(self):
        # E.g. module-level recursive functions appear in their own closure variables.
        cyclic_a: dict = {"value": 1}
        cyclic_a["self"] = cyclic_a
        cyclic_b: dict = {"value": 1}
        cyclic_b["self"] = cyclic_b
        assert utils.stable_fingerprinter(cyclic_a) == utils.stable_fingerprinter(cyclic_b)

        cyclic_c: dict = {"value": 2}
        cyclic_c["self"] = cyclic_c
        assert utils.stable_fingerprinter(cyclic_a) != utils.stable_fingerprinter(cyclic_c)

    def test_enums_are_fingerprinted_by_content(self):
        import enum

        # Enum classes (commonly defined locally, e.g. as DSL constants) are
        # fingerprinted by their member content, not by reference.
        def make_enum(pi):
            class Constants(enum.Enum):
                PI = pi
                E = 2.718

            return Constants

        c1, c2, c3 = make_enum(3.142), make_enum(3.142), make_enum(3.0)
        assert utils.stable_fingerprinter(c1) == utils.stable_fingerprinter(c2)
        assert utils.stable_fingerprinter(c1) != utils.stable_fingerprinter(c3)
        assert utils.stable_fingerprinter(c1.PI) == utils.stable_fingerprinter(c2.PI)
        assert utils.stable_fingerprinter(c1.PI) != utils.stable_fingerprinter(c3.PI)

        # Mixin-based enums dispatch to the enum handler, not the value type.
        class Device(enum.IntEnum):
            CPU = 0
            GPU = 1

        assert utils.stable_fingerprinter(Device.CPU) != utils.stable_fingerprinter(0)
        assert utils.stable_fingerprinter(Device.CPU) != utils.stable_fingerprinter(Device.GPU)

    def test_ndarray_content_is_fingerprinted(self):
        import numpy as np

        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b = np.array([[1, 2], [3, 4]], dtype=np.int32)
        assert utils.stable_fingerprinter(a) == utils.stable_fingerprinter(b)
        assert utils.stable_fingerprinter(a) != utils.stable_fingerprinter(a.T)
        assert utils.stable_fingerprinter(a) != utils.stable_fingerprinter(a.astype(np.int64))


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
