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
import string
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


@pytest.fixture(
    params=[None, hashlib.md5(), "md5", hashlib.sha1(), "sha1", hashlib.sha256(), "sha256"]
)
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


# -- Custom Pickler utilities --
@dataclasses.dataclass
class _Offseter:
    offset: int
    calls_log: list[int] = dataclasses.field(default_factory=list)

    def __call__(self, a: int) -> int:
        self.calls_log.append(a)
        return a + self.offset


def test_custom_pickler_with_singledispatch_reducer():
    import functools
    import io
    import pickle

    from gt4py.eve.utils import custom_pickler

    offseter = _Offseter(4)
    offseter(2)
    offseter(3)
    assert offseter.calls_log == [2, 3]

    @functools.singledispatch
    def my_reducer(obj):
        return NotImplemented

    my_reducer.register(
        _Offseter,
        lambda obj: (
            obj.__class__,
            (),
            (
                ("offset", obj.offset),
                ("calls_log", []),
            ),
        ),
    )

    pickler_cls = custom_pickler(my_reducer, name="MyCustomPickler")

    assert issubclass(pickler_cls, pickle.Pickler)
    assert pickler_cls.__name__ == "MyCustomPickler"

    # Verify we can instantiate the pickler
    buf = io.BytesIO()
    pickler = pickler_cls(buf)
    assert hasattr(pickler, "reducer_override")

    # Verify the custom pickler is used for objects of type _Offseter
    # and that calls_log is ignored
    pickler.dump([1, offseter])
    custom_dump_1 = buf.getvalue()

    std_buf = io.BytesIO()
    std_pickler = pickle.Pickler(std_buf)
    std_pickler.dump([1, offseter])
    std_dump_1 = std_buf.getvalue()

    assert custom_dump_1 != std_dump_1

    buf2 = io.BytesIO()
    pickler = pickler_cls(buf2)
    pickler.dump([1, _Offseter(4)])  # calls_log should be an empty list
    custom_dump_2 = buf2.getvalue()

    assert custom_dump_1 == custom_dump_2

    # Verify the custom pickler is not used for objects of other types
    buf = io.BytesIO()
    pickler = pickler_cls(buf)
    pickler.dump([1])
    custom_dump = buf.getvalue()

    std_buf = io.BytesIO()
    std_pickler = pickle.Pickler(std_buf)
    std_pickler.dump([1])
    std_dump = std_buf.getvalue()

    # This time they should be equal
    assert custom_dump == std_dump


def test_custom_pickler_from_reducers_creates_subclass():
    import pickle
    from gt4py.eve.utils import custom_pickler_from_reducers

    reducers = {int: (lambda obj: (int, (obj,)))}
    pickler_cls = custom_pickler_from_reducers(reducers, name="TestPickler")

    assert issubclass(pickler_cls, pickle.Pickler)
    assert pickler_cls.__name__ == "TestPickler"
    assert hasattr(pickler_cls, "reducer_override")
