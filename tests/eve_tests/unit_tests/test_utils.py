# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
import dataclasses
import hashlib
import string
from typing import Any

import pytest

import eve
from eve.utils import XIterable


def test_getitem_():
    from eve.utils import getitem_

    mapping = {
        "true": True,
        1: True,
        "false": False,
        0: False,
    }

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
    class MyBaseClass(abc.ABC):  # noqa: B024
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
    from eve.utils import content_hash

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
    from eve.utils import CaseStyleConverter

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
    from eve.utils import CaseStyleConverter

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


# -- UIDGenerator --
class TestUIDGenerator:
    def test_random_id(self):
        from eve.utils import UIDGenerator, UIDs

        a = UIDs.random_id()
        b = UIDs.random_id()
        c = UIDs.random_id()
        assert a != b and a != c and b != c
        assert UIDs.random_id(prefix="abcde").startswith("abcde")
        assert len(UIDs.random_id(width=10)) == 10
        with pytest.raises(ValueError, match="Width"):
            UIDs.random_id(width=-1)
        with pytest.raises(ValueError, match="Width"):
            UIDs.random_id(width=4)

        uids = UIDGenerator(prefix="abcde")
        assert uids.sequential_id().startswith("abcde")
        assert uids.sequential_id(prefix="xyz").startswith("xyz")

        uids = UIDGenerator(width=12)
        assert len(uids.sequential_id()) == 12
        assert len(UIDs.sequential_id(width=10)) == 10

    def test_sequential_id(self):
        from eve.utils import UIDGenerator, UIDs

        i = UIDs.sequential_id()
        assert UIDs.sequential_id() != i
        assert UIDs.sequential_id(prefix="abcde").startswith("abcde")
        assert len(UIDs.sequential_id(width=10)) == 10
        assert not UIDs.sequential_id().startswith("0")
        with pytest.raises(ValueError, match="Width"):
            UIDs.sequential_id(width=-1)

        uids = UIDGenerator(prefix="abcde")
        assert uids.prefix == "abcde"
        assert uids.sequential_id().startswith("abcde")
        assert uids.sequential_id(prefix="xyz").startswith("xyz")

        uids = UIDGenerator(width=12)
        assert uids.width == 12
        assert len(uids.sequential_id()) == 12
        assert len(UIDs.sequential_id(width=10)) == 10

    def test_reset_sequence(self):
        import warnings

        from eve.utils import UIDGenerator, UIDs

        i = UIDs.sequential_id()
        counter = int(i)
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            UIDs.reset_sequence(counter + 10)
            assert int(UIDs.sequential_id()) == counter + 10

            UIDs.reset_sequence(counter + 1, warn_unsafe=False)

        with pytest.warns(UserWarning, match="Unsafe reset"):
            UIDs.reset_sequence(counter, warn_unsafe=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")

            uids = UIDGenerator(warn_unsafe=True).reset_sequence(10)
            counter = int(uids.sequential_id())
            assert uids.warn_unsafe is True
            assert counter == 10

            uids.reset_sequence(counter + 10)
            uids.reset_sequence(1, warn_unsafe=False)

        uids.reset_sequence(10, warn_unsafe=False)
        with pytest.warns(UserWarning, match="Unsafe reset"):
            uids.reset_sequence(1)

        with pytest.raises(ValueError, match="must be a positive number"):
            uids.reset_sequence(-1)


# -- Iterators --
def test_xiter():
    from eve.utils import xiter

    it = xiter(range(6))
    assert isinstance(it, XIterable)
    assert list(it) == [0, 1, 2, 3, 4, 5]

    it = xiter([0, 1, 2, 3, 4, 5])
    assert isinstance(it, XIterable)
    assert list(it) == [0, 1, 2, 3, 4, 5]


def test_xenumerate():
    from eve.utils import xenumerate

    assert list(xenumerate(string.ascii_letters[:3])) == [(0, "a"), (1, "b"), (2, "c")]
