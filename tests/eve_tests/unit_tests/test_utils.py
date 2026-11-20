# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import os
import pathlib
import random
import string
import subprocess
import sys
import textwrap
import threading
import time
import types
from typing import Any

import pytest

from gt4py import eve
from gt4py.eve.utils import XIterable


def test_first():
    from gt4py.eve.utils import first

    # Test case 1: Non-empty iterable
    iterable = [1, 2, 3, 4, 5]
    result = first(iterable)
    assert result == 1

    # Test case 2: Empty iterable with default value
    iterable = []
    default = "default"
    result = first(iterable, default=default)
    assert result == default

    # Test case 3: Empty iterable without default value
    iterable = []
    with pytest.raises(StopIteration):
        first(iterable)

    # Test case 4: Iterable with single element
    iterable = [42]
    result = first(iterable)
    assert result == 42


def test_getitem_():
    from gt4py.eve.utils import getitem_

    mapping = {"true": True, 1: True, "false": False, 0: False}

    sequence = [False, True, True]

    # Items in collections
    assert getitem_(mapping, "true")
    assert not getitem_(mapping, "false")
    assert getitem_(sequence, 1)
    assert not getitem_(sequence, 0)

    # Items in mapping and providing default value
    assert getitem_(mapping, "true", False)
    assert not getitem_(mapping, "false", True)
    assert getitem_(sequence, 1, False)
    assert not getitem_(sequence, 0, True)

    # Missing items in mapping and providing default value
    assert getitem_(mapping, "", True)
    assert not getitem_(mapping, "", False)
    assert getitem_(sequence, 1000, True)
    assert not getitem_(sequence, 1000, False)

    # Missing items in mapping without providing default value
    with pytest.raises(KeyError):
        assert getitem_(mapping, "")
    with pytest.raises(IndexError):
        assert getitem_(sequence, 1000)


def test_register_subclasses():
    import abc

    class MyVirtualSubclassA:
        pass

    class MyVirtualSubclassB:
        pass

    @eve.utils.register_subclasses(MyVirtualSubclassA, MyVirtualSubclassB)
    class MyBaseClass(abc.ABC):
        pass

    assert issubclass(MyVirtualSubclassA, MyBaseClass) and issubclass(
        MyVirtualSubclassB, MyBaseClass
    )


def test_singledispatcher():
    from gt4py.eve.utils import singledispatcher

    class Base:
        pass

    class Derived(Base):
        pass

    dispatcher = singledispatcher(
        lambda _: "default",
        implementations={
            Base: lambda _: "base",
            Derived: lambda _: "derived",
        },
    )

    assert dispatcher(1) == "default"
    assert dispatcher(Base()) == "base"
    assert dispatcher(Derived()) == "derived"
    assert dispatcher.registry.keys() == {object, Base, Derived}


def test_singledispatcher_default_from_object_implementation():
    from gt4py.eve.utils import singledispatcher

    class Base:
        pass

    # When no explicit `default` is given, the 'object' implementation is used as default.
    dispatcher = singledispatcher(
        implementations={
            object: lambda _: "default",
            Base: lambda _: "base",
        }
    )

    assert dispatcher(1) == "default"
    assert dispatcher(Base()) == "base"
    assert dispatcher.registry.keys() == {object, Base}


def test_singledispatcher_dispatcher_as_default_is_not_mutated():
    from gt4py.eve.utils import singledispatcher

    # Chaining dispatchers: a dispatcher is used as the `default` (fallback) of
    # another. Registering implementations on the new dispatcher must NOT leak
    # into the one used as default (a regression: `functools.singledispatch`
    # aliases `register`/`registry` of a single-dispatch default via
    # `update_wrapper`).
    class Base:
        pass

    base = singledispatcher(lambda _: "default", implementations={Base: lambda _: "base"})
    chained = singledispatcher(base, implementations={int: lambda _: "int"})

    assert chained(1) == "int"
    assert chained(Base()) == "base"  # delegated to `base`
    assert chained("x") == "default"

    # `base` is unchanged: its registry never learned about `int`.
    assert base.registry.keys() == {object, Base}
    assert base(1) == "default"


def test_singledispatcher_validation_errors():
    from gt4py.eve.utils import singledispatcher

    # Neither an explicit `default` nor an 'object' implementation is provided.
    with pytest.raises(ValueError, match="default implementation for 'object' must be provided"):
        singledispatcher(implementations={})

    # A non-callable default is rejected.
    with pytest.raises(ValueError, match="must be callable"):
        singledispatcher("not-callable", implementations={})  # type: ignore[arg-type]  # intentionally wrong

    # Providing both an explicit `default` and an 'object' implementation is ambiguous.
    with pytest.raises(ValueError, match="already provided in 'implementations'"):
        singledispatcher(lambda _: "default", implementations={object: lambda _: "obj"})


def test_get_fully_qualified_name():
    import json

    from gt4py.eve.utils import get_fully_qualified_name

    # Built-in type and a regular function: "<module>.<qualname>".
    assert get_fully_qualified_name(dict) == "builtins.dict"
    assert get_fully_qualified_name(json.dumps) == "json.dumps"

    # Nested objects use the dotted __qualname__.
    class Outer:
        class Inner: ...

    assert get_fully_qualified_name(Outer.Inner).endswith(".Outer.Inner")

    # Modules are identified by their import name only (no qualname).
    assert get_fully_qualified_name(json) == "json"


def test_merge_dispatchers_merges_registries_and_uses_last_default():
    from gt4py.eve.utils import merge_dispatchers, singledispatcher

    class A:
        pass

    class B:
        pass

    d1 = singledispatcher(lambda _: "first-default", implementations={A: lambda _: "a"})
    d2 = singledispatcher(lambda _: "second-default", implementations={B: lambda _: "b"})

    merged = merge_dispatchers(d1, d2)

    # Without an explicit default, the default of the *last* dispatcher is used.
    assert merged("x") == "second-default"
    assert merged(A()) == "a"
    assert merged(B()) == "b"
    assert merged.registry.keys() == {object, A, B}


def test_merge_dispatchers_uses_explicit_default_and_last_registration_wins():
    from gt4py.eve.utils import merge_dispatchers, singledispatcher

    class A:
        pass

    d1 = singledispatcher(lambda _: "default-1", implementations={A: lambda _: "a-1"})
    d2 = singledispatcher(lambda _: "default-2", implementations={A: lambda _: "a-2"})

    merged = merge_dispatchers(d1, d2, default=lambda _: "custom-default")

    assert merged("x") == "custom-default"
    assert merged(A()) == "a-2"


def test_merge_dispatchers_validation_errors():
    from gt4py.eve.utils import merge_dispatchers, singledispatcher

    with pytest.raises(ValueError, match="At least one dispatcher"):
        merge_dispatchers()

    dispatcher = singledispatcher(lambda _: "default", implementations={})
    with pytest.raises(TypeError, match="single-dispatch callables"):
        merge_dispatchers(dispatcher, lambda _: "not-a-dispatcher")


class ModelClass(eve.datamodels.DataModel):
    data: Any


@dataclasses.dataclass
class DataClass:
    data: Any


@pytest.fixture(
    params=[
        [
            1,
            "1",
            True,
            "True",
            "true",
            False,
            (True,),
            [True],
            {True},
            {True: True},
            {True: ()},
            [(True,)],
            ["true"],
        ],
        [0, False, (False,), [False], {False}],
        [(), [], frozenset()],
        [[(1,)], [[1]], [[[1]]], [1], {1}],
        [
            {"a": 0},
            {"A": 0},
            {"b": 0},
            {"a": False},
            {"b": False},
            {"a": "0"},
            {"a": "False"},
            {"a": ("False",)},
            {"a": ["False"]},
            {"a": (False,)},
            {"a": "false"},
            {"a": [0]},
            {"a": [[0]]},
            {"a": [(0,)]},
        ],
    ]
)
def unique_data_items(request):
    input_data = request.param

    yield input_data + [
        DataClass(data=input_data),
        DataClass(data=input_data[0]),
        ModelClass(data=input_data),
        ModelClass(data=input_data[0]),
    ]


def test_custom_default_dict_base():
    from gt4py.eve.utils import CustomDefaultDictBase

    class TestDefaultDict(CustomDefaultDictBase):
        def value_factory(self, key):
            return key * 2

    d = TestDefaultDict()

    # Test default value creation
    assert d[5] == 10
    assert d[3] == 6

    # Test override
    d[5] = 100
    assert d[5] == 100

    # Test length
    assert len(d) == 2

    # Test iteration
    keys = list(d.keys())
    assert 5 in keys and 3 in keys


def test_custom_mapping():
    from gt4py.eve.utils import CustomMapping

    keys = [[1, 2], {"foo", "bar"}]
    values = ["value1", "value2"]

    mapping = CustomMapping(lambda x: hash(repr(x)))
    for key, value in zip(keys, values):
        mapping[key] = value

    assert len(mapping) == 2
    assert all(mapping[key] == value for key, value in zip(keys, values))

    # Test iteration
    assert keys == list(mapping)

    # Test deletion
    del mapping[keys[0]]
    assert len(mapping) == 1
    assert keys[-1] in mapping

    with pytest.raises(KeyError):
        _ = mapping[[3, 4]]

    # Test with different key function
    id_mapping = CustomMapping(id)
    obj1 = [1, 2, 3]
    obj2 = [1, 2, 3]

    id_mapping[obj1] = "obj1"
    id_mapping[obj2] = "obj2"

    assert id_mapping[obj1] == "obj1"
    assert id_mapping[obj2] == "obj2"
    assert len(id_mapping) == 2


def test_HashableBy():
    from gt4py.eve.utils import HashableBy

    assert hash(HashableBy(id, 345)) == id(345)
    assert "value=345" in str(HashableBy(lambda x: "FOO", 345))
    assert "hashed_value='FOO'" in str(HashableBy(lambda x: "FOO", 345))


def test_hashable_by():
    from gt4py.eve.utils import hashable_by

    @hashable_by
    def make_hashable(obj):
        return len(obj)

    assert hash(make_hashable({1: 2})) == 1


def test_hashable_by_id():
    from gt4py.eve.utils import hashable_by_id

    testee = {1: 2}

    assert hash(hashable_by_id(testee)) == id(testee)


def test_cached_hash():
    from gt4py.eve.utils import cached_hash

    testee = (1, 2)

    assert hash(cached_hash(testee)) == hash(testee)


def test_lru_cache_key_id_called_once():
    from gt4py.eve.utils import lru_cache

    call_count = 0

    def func(x):
        nonlocal call_count
        call_count += 1
        return x

    cached = lru_cache(func, key=id)

    assert cached.__wrapped__ == func

    obj = object()
    assert cached(obj) is obj
    assert cached(obj) is obj
    assert call_count == 1

    assert cached.cache_info().hits == 1
    assert cached.cache_info().misses == 1


def test_lru_cache_no_eq_call():
    class A:
        def __hash__(self) -> int:
            return 1

        def __eq__(self, other):
            raise ValueError()  # this function should never be called

    @eve.utils.lru_cache(key=lambda x: hash(x))
    def func(x):
        pass

    func(A())
    func(A())


def test_fluid_partial():
    from gt4py.eve.utils import fluid_partial

    def func(a, b, c):
        return a + b + c

    fp1 = fluid_partial(func, 1)
    fp2 = fp1.partial(2)
    fp3 = fp2.partial(3)

    assert fp1(2, 3) == 6
    assert fp2(3) == 6
    assert fp3() == 6


def test_noninstantiable_class():
    @eve.utils.noninstantiable
    class NonInstantiableClass(eve.datamodels.DataModel):
        param: int

    with pytest.raises(
        TypeError, match="Trying to instantiate 'NonInstantiableClass' non-instantiable class"
    ):
        NonInstantiableClass(param=0)

    assert eve.utils.is_noninstantiable(NonInstantiableClass)

    class InstantiableSubclass(NonInstantiableClass):
        pass

    instance = InstantiableSubclass(param=0)
    assert isinstance(instance, InstantiableSubclass)
    assert isinstance(instance, NonInstantiableClass)

    assert not eve.utils.is_noninstantiable(InstantiableSubclass)


@pytest.fixture(params=[None, hashlib.md5(), hashlib.sha1(), hashlib.sha256()])
def hash_algorithm(request):
    yield request.param


def test_shash(unique_data_items, hash_algorithm):
    from gt4py.eve.utils import content_hash

    # Test hash consistency
    for item in unique_data_items:
        if hasattr(hash_algorithm, "copy"):
            h1 = hash_algorithm.copy()
            h2 = hash_algorithm.copy()
        else:
            h1 = hash_algorithm
            h2 = hash_algorithm
        assert content_hash(item, hash_algorithm=h1) == content_hash(
            copy.deepcopy(item), hash_algorithm=h2
        )

    # Test hash specificity
    hashes = set(content_hash(item, hash_algorithm=hash_algorithm) for item in unique_data_items)
    assert len(hashes) == len(unique_data_items)


# -- CaseStyleConverter --
@pytest.fixture
def name_with_cases():
    from gt4py.eve.utils import CaseStyleConverter

    cases = {
        "words": ["first", "second", "UPPER", "Title"],
        CaseStyleConverter.CASE_STYLE.CONCATENATED: "firstseconduppertitle",
        CaseStyleConverter.CASE_STYLE.CANONICAL: "first second upper title",
        CaseStyleConverter.CASE_STYLE.CAMEL: "firstSecondUpperTitle",
        CaseStyleConverter.CASE_STYLE.PASCAL: "FirstSecondUpperTitle",
        CaseStyleConverter.CASE_STYLE.SNAKE: "first_second_upper_title",
        CaseStyleConverter.CASE_STYLE.KEBAB: "first-second-upper-title",
    }

    yield cases


def test_case_style_converter(name_with_cases):
    from gt4py.eve.utils import CaseStyleConverter

    words = name_with_cases.pop("words")
    for case, cased_string in name_with_cases.items():
        # Try also passing case as a string
        if len(case.value) % 2:
            case = case.value

        assert CaseStyleConverter.join(words, case) == cased_string
        if case == CaseStyleConverter.CASE_STYLE.CONCATENATED:
            with pytest.raises(ValueError, match="Impossible to split"):
                CaseStyleConverter.split(cased_string, case)
        else:
            assert [w.lower() for w in CaseStyleConverter.split(cased_string, case)] == [
                w.lower() for w in words
            ]


# -- SequentialIDGenerator --
class TestSequentialIDGenerator:
    def test_basic(self):
        from gt4py.eve.utils import SequentialIDGenerator

        uids = SequentialIDGenerator()
        first = next(uids)
        second = uids.next()
        assert next(uids) != first != second

    def test_prefix(self):
        from gt4py.eve.utils import SequentialIDGenerator

        uids = SequentialIDGenerator(prefix="test_")
        uid = next(uids)
        assert uid.startswith("test_")

    def test_format(self):
        from gt4py.eve.utils import SequentialIDGenerator

        prefix = "UID"
        uids = SequentialIDGenerator(format="{prefix}-{id:04d}", prefix=prefix)
        uid = next(uids)
        assert len(uid) == len(prefix) + 1 + 4  # prefix + '-' + zero-padded id


# -- Iterators --
def test_xiter():
    from gt4py.eve.utils import xiter

    it = xiter(range(6))
    assert isinstance(it, XIterable)
    assert list(it) == [0, 1, 2, 3, 4, 5]

    it = xiter([0, 1, 2, 3, 4, 5])
    assert isinstance(it, XIterable)
    assert list(it) == [0, 1, 2, 3, 4, 5]


def test_xenumerate():
    from gt4py.eve.utils import xenumerate

    assert list(xenumerate(string.ascii_letters[:3])) == [(0, "a"), (1, "b"), (2, "c")]


# -- Lazy module imports --
class TestLazyImportModule:
    """Tests for `lazy_import_module()` using toy packages written to `tmp_path`.

    Every toy module records its execution in `lztrace.ORDER`, so tests can
    check what was executed and in which order.
    """

    @pytest.fixture
    def make_modules(self, tmp_path, monkeypatch):
        """Return a function writing toy modules (`{"pkg/__init__.py": "source"}`) to an import path."""
        saved_modules = dict(sys.modules)
        monkeypatch.syspath_prepend(str(tmp_path))
        (tmp_path / "lztrace.py").write_text("ORDER = []\n")

        def make(files: dict[str, str]) -> pathlib.Path:
            for rel_path, source in files.items():
                path = tmp_path / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(textwrap.dedent(source))
            importlib.invalidate_caches()
            return tmp_path

        yield make

        for name in set(sys.modules) - set(saved_modules):
            del sys.modules[name]
        sys.modules.update(saved_modules)

    TRACE = "import lztrace; lztrace.ORDER.append(__name__)\n"

    @staticmethod
    def trace() -> list[str]:
        import lztrace

        return lztrace.ORDER

    def test_no_code_executed(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": self.TRACE,
                "lzpkg/sub/__init__.py": self.TRACE,
                "lzpkg/sub/leaf.py": self.TRACE,
            }
        )
        modules_before = set(sys.modules)

        proxy = eve.utils.lazy_import_module("lzpkg.sub.leaf")

        assert set(sys.modules) == modules_before
        assert eve.utils.is_lazy_module(proxy)
        assert isinstance(proxy, types.ModuleType)
        assert proxy.__name__ == "lzpkg.sub.leaf"
        assert repr(proxy) == "<lazy module 'lzpkg.sub.leaf' (unloaded)>"
        assert set(sys.modules) == modules_before

        assert proxy.__file__.endswith("leaf.py")
        assert self.trace() == ["lzpkg", "lzpkg.sub", "lzpkg.sub.leaf"]  # parent first
        assert repr(proxy) == repr(sys.modules["lzpkg.sub.leaf"])

    def test_already_imported(self):
        assert eve.utils.lazy_import_module("sys") is sys
        assert eve.utils.lazy_import_module("importlib.util") is importlib.util
        assert not eve.utils.is_lazy_module(sys)
        assert eve.utils.resolve_lazy_module(sys) is sys

    def test_identity(self, make_modules):
        make_modules({"lzpkg/__init__.py": "X = 1\n"})
        proxy = eve.utils.lazy_import_module("lzpkg")
        proxy_hash = hash(proxy)

        real = eve.utils.resolve_lazy_module(proxy)

        assert real is sys.modules["lzpkg"]
        assert proxy is not real
        assert proxy != real
        assert hash(proxy) == proxy_hash
        assert eve.utils.is_lazy_module(proxy)
        assert not eve.utils.is_lazy_module(real)

    def test_attribute_forwarding(self, make_modules):
        make_modules({"lzpkg/__init__.py": "X = 1\n"})
        proxy = eve.utils.lazy_import_module("lzpkg")

        proxy.Y = 2
        real = sys.modules["lzpkg"]
        assert real.Y == 2
        del proxy.X
        assert not hasattr(real, "X")
        assert vars(proxy) is vars(real)
        assert proxy.__dict__ is real.__dict__
        assert "Y" in dir(proxy)

    def test_module_getattr(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": """
                def __getattr__(name):
                    if name == "dynamic":
                        return "from __getattr__"
                    raise AttributeError(name)
                """
            }
        )
        assert eve.utils.lazy_import_module("lzpkg").dynamic == "from __getattr__"

    def test_circular_reexport(self, make_modules):
        # The parent re-exports from the child, while the child imports the parent.
        make_modules(
            {
                "lzpkg/__init__.py": self.TRACE + "from .sub import Y\nZ = Y * 10\n",
                "lzpkg/sub/__init__.py": self.TRACE + "import lzpkg\nY = 2\n",
            }
        )
        child = eve.utils.lazy_import_module("lzpkg.sub")
        parent = eve.utils.lazy_import_module("lzpkg")

        assert child.Y == 2  # touch the child first
        assert parent.Z == 20
        assert self.trace() == ["lzpkg", "lzpkg.sub"]
        assert sys.modules["lzpkg"].sub is eve.utils.resolve_lazy_module(child)

    def test_child_import_side_effects(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": "REGISTRY = []\nfrom . import reg\nN = len(REGISTRY)\n",
                "lzpkg/reg.py": "from . import REGISTRY\nREGISTRY.append(__name__)\n",
            }
        )
        eve.utils.lazy_import_module("lzpkg.reg").__file__

        assert sys.modules["lzpkg"].N == 1

    def test_parent_binding_shadows_child(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": "from .sub import sub\n",
                "lzpkg/sub.py": "def sub(): pass\n",
            }
        )
        eve.utils.lazy_import_module("lzpkg.sub").__file__

        assert isinstance(sys.modules["lzpkg"].sub, types.FunctionType)

    def test_self_replacing_module(self, make_modules):
        make_modules(
            {
                "lzpkg/__init__.py": """
                import sys, types
                sys.modules[__name__] = types.SimpleNamespace(X="wrapped")
                """
            }
        )
        proxy = eve.utils.lazy_import_module("lzpkg")

        assert proxy.X == "wrapped"
        assert proxy.X == "wrapped"

    def test_runtime_package_path(self, make_modules, tmp_path):
        make_modules(
            {
                "lzpkg/__init__.py": f"""
                import os, sys
                __path__.append({str(tmp_path / "extra")!r})
                sys.modules[__name__ + ".alias"] = os
                """,
                "extra/plug.py": "WHERE = 'extra'\n",
            }
        )

        # Not found statically: the parent is imported to look into its runtime `__path__`
        plug = eve.utils.lazy_import_module("lzpkg.plug")
        assert "lzpkg" in sys.modules
        assert plug.WHERE == "extra"

        assert eve.utils.lazy_import_module("lzpkg.alias") is os
        assert eve.utils.lazy_import_module("lzpkg.missing", on_missing="none") is None
        with pytest.raises(ModuleNotFoundError):
            eve.utils.lazy_import_module("lzpkg.missing")

    def test_alias_registered_by_parent(self, make_modules):
        # The alias is looked up before the parent is imported, and it has no `__spec__`.
        make_modules(
            {
                "lzpkg/__init__.py": """
                import sys, types
                sys.modules[__name__ + ".alias"] = ALIAS = types.ModuleType(__name__ + ".alias")
                """
            }
        )
        alias = eve.utils.lazy_import_module("lzpkg.alias")

        assert alias is sys.modules["lzpkg"].ALIAS

    @pytest.mark.parametrize(
        "fullname",
        [
            "lz_not_installed",
            "lz_not_installed.child",
            "lzpkg.missing",
            "lzpkg.missing.child",
            "lzpkg.module.child",  # parent is not a package
            "typing.numpy",  # parent is an imported module, not a package
            "lzblocked",
            "lzblocked.child",
        ],
    )
    def test_missing(self, make_modules, monkeypatch, fullname):
        make_modules({"lzpkg/__init__.py": "", "lzpkg/module.py": ""})
        monkeypatch.setitem(sys.modules, "lzblocked", None)  # standard way to block an import

        assert eve.utils.lazy_import_module(fullname, on_missing="none") is None
        with pytest.raises(ModuleNotFoundError):
            eve.utils.lazy_import_module(fullname)
        with pytest.raises(ModuleNotFoundError):
            eve.utils.lazy_import_module(fullname, on_missing="raise")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(fullname)  # check plain import agrees

    def test_broken_parent_in_fallback(self, make_modules):
        # The child is not found statically, so the broken parent is imported at call time.
        make_modules({"lzpkg/__init__.py": "raise ImportError('broken parent')\n"})

        with pytest.raises(ImportError, match="broken parent"):
            eve.utils.lazy_import_module("lzpkg.missing", on_missing="none")

    def test_invalid_on_missing(self):
        with pytest.raises(ValueError, match="on_missing"):
            eve.utils.lazy_import_module("sys", on_missing="ignore")

    def test_broken_library(self, make_modules):
        make_modules({"lzbroken/__init__.py": "A = 1\nraise ImportError('broken library')\n"})
        proxy = eve.utils.lazy_import_module("lzbroken")

        for _ in range(2):  # errors are not cached: every access retries the import
            with pytest.raises(ImportError, match="broken library"):
                proxy.A
            assert "lzbroken" not in sys.modules

        assert not hasattr(proxy, "__wrapped__")
        with pytest.raises(TypeError):
            vars(proxy)
        inspect.stack()
        assert repr(proxy) == "<lazy module 'lzbroken' (unloaded)>"

    def test_half_initialized_module_not_cached(self, make_modules):
        # The module touches its own proxy while it is being imported, then fails.
        make_modules(
            {
                "lzholder.py": "PROXY = None\n",
                "lzhalf/__init__.py": """
                import lzholder
                lzholder.SEEN = getattr(lzholder.PROXY, "GOOD", "not yet")
                GOOD = 1
                raise ImportError("fails after being touched")
                """,
            }
        )
        import lzholder

        lzholder.PROXY = proxy = eve.utils.lazy_import_module("lzhalf")

        for _ in range(2):
            with pytest.raises(ImportError, match="fails after being touched"):
                proxy.GOOD
            assert lzholder.SEEN == "not yet"

    @pytest.mark.parametrize("circular", [False, True])
    def test_threads(self, make_modules, circular):
        make_modules(
            {
                "lzpkg/__init__.py": (
                    "import time; time.sleep(0.002)\n"
                    + ("from .sub import Y\n" if circular else "")
                    + "X = 1\n"
                ),
                "lzpkg/sub/__init__.py": (
                    "import time; time.sleep(0.002)\n"
                    + ("import lzpkg\n" if circular else "")
                    + "Y = 2\n"
                ),
            }
        )
        rng = random.Random(42)
        for _ in range(20):
            for name in ("lzpkg.sub", "lzpkg"):
                sys.modules.pop(name, None)
            proxies = [eve.utils.lazy_import_module(name) for name in ("lzpkg", "lzpkg.sub")]
            errors: list[BaseException] = []

            def touch(proxies=proxies, errors=errors, delay=rng.random() / 500):
                time.sleep(delay)
                for proxy in random.sample(proxies, 2):
                    try:
                        getattr(proxy, "X" if proxy.__name__ == "lzpkg" else "Y")
                    except Exception as error:
                        errors.append(error)

            threads = [threading.Thread(target=touch, daemon=True) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            assert not any(thread.is_alive() for thread in threads), "deadlock"

            if not circular:
                # Concurrent imports of circular packages may fail as with plain imports
                assert not errors
            for proxy in proxies:
                cached = object.__getattribute__(proxy, "__dict__")[eve.utils._LAZY_MODULE_SLOT]
                assert cached is None or cached is sys.modules[proxy.__name__]

    def test_introspection_does_not_import(self, make_modules, tmp_path):
        # Run in a subprocess: `inspect.getmodule()` (used by `inspect.stack()`) scans `sys.modules`.
        make_modules({"lzpkg/__init__.py": "raise ImportError('must not be imported')\n"})
        script = textwrap.dedent(
            """
            import inspect, sys
            from gt4py import eve
            proxy = eve.utils.lazy_import_module("lzpkg")
            inspect.stack()
            print("lzpkg" in sys.modules, eve.utils.is_lazy_module(proxy))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])},
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.split() == ["False", "True"]
