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


from __future__ import annotations

import collections.abc
import sys
import timeit
import types
import typing

import pytest

from eve import extended_typing as xtyping
from eve.extended_typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    ForwardRef,
    FrozenSet,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
)


@pytest.fixture
def sample_class_defs():
    TEST_SRC = """
from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

class NoDataProto(Protocol):
    def method_1(self) -> int:
        ...

    def method_2(self) -> int:
        ...

    def method_3(self) -> int:
        ...

    def method_4(self) -> int:
        ...

    def method_5(self) -> int:
        ...

    def method_6(self) -> int:
        ...

    def method_7(self) -> int:
        ...

class DataProto(NoDataProto, Protocol):
    foo: int
    BAR: ClassVar[int]

class ConcreteClass:
    foo: int = 32
    BAR = 32

    def method_1(self) -> int:
        return 42

    def method_2(self) -> int:
        return 42

    def method_3(self) -> int:
        return 42

    def method_4(self) -> int:
        return 42

    def method_5(self) -> int:
        return 42

    def method_6(self) -> int:
        return 42

    def method_7(self) -> int:
        return 42

    def method_foo(self) -> int:
        return 42

class IncompleteClass:
    def method_1(self) -> int:
        return 42

    def method_2(self) -> int:
        return 42
    """
    DEFINITIONS = {}
    exec(TEST_SRC, None, DEFINITIONS)
    yield types.SimpleNamespace(**DEFINITIONS)


class TestExtendedProtocol:
    def test_instance_check_shortcut(self, sample_class_defs):
        ConcreteClass = sample_class_defs.ConcreteClass
        IncompleteClass = sample_class_defs.IncompleteClass
        NoDataProto = sample_class_defs.NoDataProto
        DataProto = sample_class_defs.DataProto

        # Undecorated runtime protocol checks should fail
        with pytest.raises(
            TypeError, match="checks can only be used with @runtime_checkable protocols"
        ):
            assert isinstance(ConcreteClass(), NoDataProto)

        # Standard runtime protocol checks
        xtyping.runtime_checkable(NoDataProto)
        assert isinstance(ConcreteClass(), NoDataProto)
        assert isinstance(ConcreteClass(), DataProto)
        assert not isinstance(IncompleteClass(), NoDataProto)

        # Standard runtime protocol checks from extended decorator (behavior should be the same)
        xtyping.extended_runtime_checkable(
            NoDataProto, instance_check_shortcut=False, subclass_check_with_data_members=False
        )
        assert isinstance(ConcreteClass(), NoDataProto)
        assert isinstance(ConcreteClass(), DataProto)
        assert not isinstance(IncompleteClass(), NoDataProto)

        # Runtime protocol checks from extended decorator with shortcuts
        xtyping.extended_runtime_checkable(
            instance_check_shortcut=True, subclass_check_with_data_members=False
        )(NoDataProto)
        assert isinstance(ConcreteClass(), NoDataProto)
        assert isinstance(ConcreteClass(), DataProto)
        assert not isinstance(IncompleteClass(), NoDataProto)

    def test_instance_check_shortcut_performance(self, sample_class_defs):
        PASS_STMT = "isinstance(ConcreteClass(), NoDataProto)"
        FAIL_STMT = "isinstance(IncompleteClass(), NoDataProto)"
        DEFINITIONS = sample_class_defs.__dict__
        NUM_REPETITIONS = 10000

        # Timings for standard runtime_checkable()
        xtyping.runtime_checkable(sample_class_defs.NoDataProto)
        std_pass_time = timeit.timeit(stmt=PASS_STMT, number=NUM_REPETITIONS, globals=DEFINITIONS)
        std_fail_time = timeit.timeit(stmt=FAIL_STMT, number=NUM_REPETITIONS, globals=DEFINITIONS)

        # Standard runtime protocol checks from extended decorator.
        # Expected performance should be roughly the same.
        xtyping.runtime_checkable(sample_class_defs.NoDataProto)
        std_from_ext_pass_time = timeit.timeit(
            stmt=PASS_STMT, number=NUM_REPETITIONS, globals=DEFINITIONS
        )
        std_from_ext_fail_time = timeit.timeit(
            stmt=FAIL_STMT, number=NUM_REPETITIONS, globals=DEFINITIONS
        )
        bound_factor = 3.0

        assert (1 / bound_factor) < (std_pass_time / std_from_ext_pass_time) < bound_factor
        assert (1 / bound_factor) < (std_fail_time / std_from_ext_fail_time) < bound_factor

        # Runtime protocol checks from extended decorator with shortcuts
        # Expected performance should be much better.
        xtyping.extended_runtime_checkable(
            instance_check_shortcut=True, subclass_check_with_data_members=False
        )(sample_class_defs.NoDataProto)
        ext_pass_time = timeit.timeit(stmt=PASS_STMT, number=NUM_REPETITIONS, globals=DEFINITIONS)
        ext_fail_time = timeit.timeit(stmt=FAIL_STMT, number=NUM_REPETITIONS, globals=DEFINITIONS)
        bound_factor = 10.0

        assert std_pass_time / ext_pass_time > bound_factor
        assert std_pass_time / ext_fail_time > bound_factor

    def test_subclass_check_with_data_members(self, sample_class_defs):
        ConcreteClass = sample_class_defs.ConcreteClass
        NoDataProto = sample_class_defs.NoDataProto
        DataProto = sample_class_defs.DataProto

        # Undecorated runtime protocol checks should fail
        with pytest.raises(
            TypeError, match="checks can only be used with @runtime_checkable protocols"
        ):
            assert issubclass(ConcreteClass, DataProto)

        # Standard runtime protocol checks
        xtyping.runtime_checkable(NoDataProto)
        assert isinstance(ConcreteClass(), NoDataProto)
        assert issubclass(ConcreteClass, NoDataProto)

        with pytest.raises(
            TypeError, match="Protocols with non-method members don't support issubclass()"
        ):
            assert issubclass(ConcreteClass, DataProto)

        # Standard runtime protocol checks from extended decorator.
        # Expected behavior and performance should be roughly the same.
        xtyping.extended_runtime_checkable(
            instance_check_shortcut=False, subclass_check_with_data_members=False
        )(DataProto)
        assert isinstance(ConcreteClass(), NoDataProto)
        assert issubclass(ConcreteClass, NoDataProto)

        with pytest.raises(
            TypeError, match="Protocols with non-method members don't support issubclass()"
        ):
            assert issubclass(ConcreteClass, DataProto)

        # Runtime protocol checks from extended decorator with shortcuts
        xtyping.extended_runtime_checkable(
            instance_check_shortcut=False, subclass_check_with_data_members=True
        )(DataProto)
        assert isinstance(ConcreteClass(), DataProto)
        assert issubclass(ConcreteClass, DataProto)
        assert isinstance(ConcreteClass(), NoDataProto)
        assert issubclass(ConcreteClass, NoDataProto)


@pytest.mark.parametrize("t", (int, float, dict, tuple, frozenset, collections.abc.Mapping))
def test_is_actual_valid_type(t):
    assert xtyping.is_actual_type(t)


@pytest.mark.parametrize(
    "t",
    (
        Tuple[int],
        Tuple[int, ...],
        Tuple[int, int],
        Dict[str, Any],
        Dict[str, float],
        Mapping[int, float],
    ),
)
def test_is_actual_wrong_type(t):
    assert not xtyping.is_actual_type(t)


ACTUAL_TYPE_SAMPLES = [
    (3, int),
    (4.5, float),
    ({}, dict),
    (int, type),
    (tuple, type),
    (list, type),
    (Tuple[int, float], type(Tuple[int, float])),
    (List[int], type(List[int])),
]
if sys.version_info >= (3, 9):
    ACTUAL_TYPE_SAMPLES.extend(
        [
            (tuple[int, float], types.GenericAlias),  # type: ignore[misc]   # ignore false positive bug: https://github.com/python/mypy/issues/11098
            (list[int], types.GenericAlias),
        ]
    )


@pytest.mark.parametrize(["instance", "expected"], ACTUAL_TYPE_SAMPLES)
def test_get_actual_type(instance, expected):
    assert xtyping.get_actual_type(instance) == expected


class TestHashableTypings:
    @pytest.mark.parametrize(
        "x",
        [
            int,
            float,
            complex,
            str,
            tuple,
            frozenset,
            1,
            -2.0,
            "foo",
            (),
            (1, 3.0),
            frozenset([1, 2, 3]),
        ],
    )
    def test_is_value_hashable(self, x):
        assert xtyping.is_value_hashable(x)

    @pytest.mark.parametrize("x", [list(), {1, 2, 3}, dict()])
    def test_is_not_value_hashable(self, x):
        assert not xtyping.is_value_hashable(x)

    @pytest.mark.parametrize(
        "t",
        [
            int,
            str,
            float,
            tuple,
            Tuple,
            Tuple[int],
            Tuple[int, ...],
            Tuple[Tuple[int, ...], ...],
            FrozenSet,
            Type,
            type(None),
            None,
        ],
    )
    def test_is_value_hashable_typing(self, t):
        assert xtyping.is_value_hashable_typing(t)

    @pytest.mark.parametrize(
        "t", [dict, Dict, Dict[str, int], Sequence[int], List[str], Any, TypeVar("T")]
    )
    def test_is_not_value_hashable_type(self, t):
        assert not xtyping.is_value_hashable_typing(t)

    def test_has_custom_hash_abc(self):
        assert isinstance(4, xtyping.HasCustomHash)
        assert isinstance((), xtyping.HasCustomHash)

        class A:
            def __hash__(self):
                return 3

        assert isinstance(A(), xtyping.HasCustomHash)

        class B:
            __hash__ = None

        assert not isinstance(B(), xtyping.HasCustomHash)

        assert not isinstance(None, xtyping.HasCustomHash)
        assert not isinstance(object(), xtyping.HasCustomHash)
        assert not isinstance(tuple, xtyping.HasCustomHash)
        assert not isinstance(type, xtyping.HasCustomHash)


def test_is_protocol():
    class AProtocol(typing.Protocol):
        def do_something(self, value: int) -> int:
            ...

    class NotProtocol(AProtocol):
        def do_something_else(self, value: float) -> float:
            ...

    class AXProtocol(xtyping.Protocol):
        A = 1

    class NotXProtocol(AXProtocol):
        A = 1

    class AgainProtocol(AProtocol, xtyping.Protocol):
        def do_something_else(self, value: float) -> float:
            ...

    assert xtyping.is_protocol(AProtocol)
    assert xtyping.is_protocol(AXProtocol)

    assert not xtyping.is_protocol(NotProtocol)
    assert not xtyping.is_protocol(NotXProtocol)

    assert xtyping.is_protocol(AgainProtocol)


def test_get_partial_type_hints():
    def f1(a: int) -> float:
        ...

    assert xtyping.get_partial_type_hints(f1) == {"a": int, "return": float}

    class MissingRef:
        ...

    def f_partial(a: int) -> MissingRef:
        ...

    # This is expected behavior because this test file uses
    # 'from __future__ import annotations' and therefore local
    # references cannot be automatically resolved
    assert xtyping.get_partial_type_hints(f_partial) == {
        "a": int,
        "return": ForwardRef("MissingRef"),
    }
    assert xtyping.get_partial_type_hints(f_partial, localns={"MissingRef": MissingRef}) == {
        "a": int,
        "return": MissingRef,
    }
    assert xtyping.get_partial_type_hints(f_partial, globalns={"MissingRef": int}) == {
        "a": int,
        "return": int,
    }

    def f_nested_partial(a: int) -> Dict[str, MissingRef]:
        ...

    assert xtyping.get_partial_type_hints(f_nested_partial) == {
        "a": int,
        "return": ForwardRef("Dict[str, MissingRef]"),
    }
    assert xtyping.get_partial_type_hints(f_nested_partial, localns={"MissingRef": MissingRef}) == {
        "a": int,
        "return": Dict[str, MissingRef],
    }

    def f_annotated(a: Annotated[int, "Foo"]) -> float:  # type: ignore[name-defined]  # used to work, now mypy is going berserk for unknown reasons
        ...

    assert xtyping.get_partial_type_hints(f_annotated) == {"a": int, "return": float}
    assert xtyping.get_partial_type_hints(f_annotated, include_extras=True) == {
        "a": Annotated[int, "Foo"],
        "return": float,
    }
    assert xtyping.get_partial_type_hints(f_annotated, include_extras=True) != {
        "a": Annotated[int, "Bar"],
        "return": float,
    }


def test_eval_forward_ref():
    assert xtyping.eval_forward_ref("Dict[str, Tuple[int, float]]") == Dict[str, Tuple[int, float]]
    assert (
        xtyping.eval_forward_ref(ForwardRef("Dict[str, Tuple[int, float]]"))
        == Dict[str, Tuple[int, float]]
    )

    class MissingRef:
        ...

    assert (
        xtyping.eval_forward_ref("Callable[[int], MissingRef]", localns={"MissingRef": MissingRef})
        == Callable[[int], MissingRef]
    )

    assert (
        xtyping.eval_forward_ref(
            "Callable[[int], MissingRef]",
            globalns={"Callable": Callable},
            localns={"MissingRef": MissingRef},
        )
        == Callable[[int], MissingRef]
    )

    assert (
        xtyping.eval_forward_ref(
            "Callable[[Annotated[int, 'Foo']], MissingRef]",
            globalns={"Annotated": Annotated, "Callable": Callable},
            localns={"MissingRef": MissingRef},
        )
        == Callable[[int], MissingRef]
    )
    assert (
        xtyping.eval_forward_ref(
            "Callable[[Annotated[int, 'Foo']], MissingRef]",
            globalns={"Annotated": Annotated, "Callable": Callable},
            localns={"MissingRef": MissingRef},
            include_extras=True,
        )
        == Callable[[Annotated[int, "Foo"]], MissingRef]
    )


def test_infer_type():
    assert xtyping.infer_type(3) == int

    assert xtyping.infer_type(None) is type(None)  # noqa: E721  # do not compare types
    assert xtyping.infer_type(type(None)) is type(None)  # noqa: E721  # do not compare types
    assert xtyping.infer_type(None, none_as_type=False) is None
    assert xtyping.infer_type(type(None), none_as_type=False) is None

    assert xtyping.infer_type(Dict[str, int]) == Dict[str, int]

    assert xtyping.infer_type({1, 2, 3}) == Set[int]
    assert xtyping.infer_type(frozenset({"1", "2", "3"})) == FrozenSet[str]

    assert xtyping.infer_type({"a": [0], "b": [1]}) == Dict[str, List[int]]

    assert xtyping.infer_type(str) == Type[str]

    class A:
        ...

    assert xtyping.infer_type(A()) == A
    assert xtyping.infer_type(A) == Type[A]

    def f1():
        ...

    assert xtyping.infer_type(f1) == Callable[[], Any]

    def f2(a: int, b: float) -> None:
        ...

    assert xtyping.infer_type(f2) == Callable[[int, float], type(None)]

    def f3(
        a: Dict[Tuple[str, ...], List[int]],
        b: List[Callable[[List[int]], Set[Set[int]]]],
        c: Type[List[int]],
    ) -> Any:
        ...

    assert (
        xtyping.infer_type(f3)
        == Callable[
            [
                Dict[Tuple[str, ...], List[int]],
                List[Callable[[List[int]], Set[Set[int]]]],
                Type[List[int]],
            ],
            Any,
        ]
    )

    def f4(a: int, b: float, *, foo: Tuple[str, ...] = ()) -> None:
        ...

    assert xtyping.infer_type(f4) == Callable[[int, float], type(None)]
    assert (
        xtyping.infer_type(f4, annotate_callable_kwargs=True)
        == Annotated[
            Callable[[int, float], type(None)], xtyping.CallableKwargsInfo({"foo": Tuple[str, ...]})
        ]
    )
