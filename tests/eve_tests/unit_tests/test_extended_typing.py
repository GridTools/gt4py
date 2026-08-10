# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import collections
import collections.abc
import contextlib
import re
import sys
import types
import typing

import pytest

from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import (
    Annotated,
    Any,
    Callable,
    ForwardRef,
    Mapping,
    Sequence,
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


def test_supports_array_interface():
    from gt4py.eve.extended_typing import supports_array_interface

    class ArrayInterface:
        __array_interface__ = "interface"

    class NoArrayInterface:
        pass

    assert supports_array_interface(ArrayInterface())
    assert not supports_array_interface(NoArrayInterface())
    assert not supports_array_interface("array")
    assert not supports_array_interface(None)


def test_supports_cuda_array_interface():
    from gt4py.eve.extended_typing import supports_cuda_array_interface

    class CudaArray:
        def __cuda_array_interface__(self):
            return {}

    class NoCudaArray:
        pass

    assert supports_cuda_array_interface(CudaArray())
    assert not supports_cuda_array_interface(NoCudaArray())
    assert not supports_cuda_array_interface("cuda")
    assert not supports_cuda_array_interface(None)


def test_supports_dlpack():
    from gt4py.eve.extended_typing import supports_dlpack

    class DummyDLPackBuffer:
        def __dlpack__(self):
            pass

        def __dlpack_device__(self):
            pass

    class DLPackBufferWithWrongBufferMethod:
        __dlpack__ = "buffer"

        def __dlpack_device__(self):
            pass

    class DLPackBufferWithoutDevice:
        def __dlpack__(self):
            pass

    class DLPackBufferWithWrongDevice:
        def __dlpack__(self):
            pass

        __dlpack_device__ = "device"

    assert supports_dlpack(DummyDLPackBuffer())
    assert not supports_dlpack(DLPackBufferWithWrongBufferMethod())
    assert not supports_dlpack(DLPackBufferWithoutDevice())
    assert not supports_dlpack(DLPackBufferWithWrongDevice())


DEPRECATED_TYPING_ALIASES = [
    ("Dict", "dict"),
    ("FrozenSet", "frozenset"),
    ("List", "list"),
    ("Set", "set"),
    ("Tuple", "tuple"),
    ("Type", "type"),
]


@pytest.mark.parametrize(["name", "replacement"], DEPRECATED_TYPING_ALIASES)
def test_deprecated_typing_alias_is_not_exported(name, replacement):
    # These names are still bound by the 'typing' / 'typing_extensions' star imports in
    # 'extended_typing', and its module '__getattr__' would otherwise forward them. Pin
    # the rejection: dropping the guard would not make the names disappear, it would
    # silently resolve them to the deprecated 'typing' objects instead of the builtins.
    with pytest.raises(AttributeError, match=f"'{name}' is a deprecated 'typing' alias"):
        getattr(xtyping, name)

    assert replacement in str(pytest.raises(AttributeError, lambda: getattr(xtyping, name)).value)

    with pytest.raises(ImportError):
        exec(f"from gt4py.eve.extended_typing import {name}")

    assert name not in dir(xtyping)


@pytest.mark.parametrize(
    ["name", "expected"],
    [
        ("Sequence", collections.abc.Sequence),
        ("Callable", collections.abc.Callable),
        ("AbstractSet", collections.abc.Set),
        ("Match", re.Match),
        ("ContextManager", contextlib.AbstractContextManager),
        ("deque", collections.deque),
    ],
)
def test_non_deprecated_aliases_are_still_re_exported(name, expected):
    # Unlike the builtin generics above, these names are not deprecated -- only their
    # 'typing' home is -- so 'extended_typing' keeps pointing them at the modern object.
    assert getattr(xtyping, name) is expected


@pytest.mark.parametrize(
    ["name", "replacement", "args"],
    [
        (name, replacement, "int, str" if name == "Dict" else "int")
        for name, replacement in DEPRECATED_TYPING_ALIASES
    ],
)
def test_deprecated_typing_alias_still_resolves_in_forward_refs(name, replacement, args):
    # These names stay valid in user-written annotations, so resolving a forward
    # reference must not hit the rejection above. Both the bare and the 'typing.'-
    # qualified spelling normalize to the *builtin* generic, which is what resolving
    # through this module has always produced.
    expected_args = tuple(eval(a) for a in args.split(", "))

    for ref in (f"{name}[{args}]", f"typing.{name}[{args}]"):
        resolved = xtyping.eval_forward_ref(ref)

        assert xtyping.get_origin(resolved) is getattr(builtins, replacement)
        assert xtyping.get_args(resolved) == expected_args
        # A builtin generic alias, not the deprecated 'typing._GenericAlias' object.
        assert type(resolved) is types.GenericAlias

    # An explicit 'globalns' is left alone, so the bare name is not injected there.
    with pytest.raises(NameError):
        xtyping.eval_forward_ref(f"{name}[{args}]", globalns={})


@pytest.mark.parametrize("t", (int, float, dict, tuple, frozenset, collections.abc.Mapping))
def test_is_actual_valid_type(t):
    assert xtyping.is_actual_type(t)


@pytest.mark.parametrize(
    "t",
    (
        tuple[int],
        tuple[int, ...],
        tuple[int, int],
        dict[str, Any],
        dict[str, float],
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
    # The deprecated 'typing' aliases and the builtin generics are distinct objects with
    # distinct alias types, so both spellings are covered.
    (typing.Tuple[int, float], type(typing.Tuple[int, float])),
    (typing.List[int], type(typing.List[int])),
    (tuple[int, float], types.GenericAlias),
    (list[int], types.GenericAlias),
]


@pytest.mark.parametrize(["instance", "expected"], ACTUAL_TYPE_SAMPLES)
def test_get_actual_type(instance, expected):
    assert xtyping.get_actual_type(instance) == expected


def test_has_custom_hash_abc():
    assert isinstance(4, xtyping.HasCustomHash)
    assert isinstance(True, xtyping.HasCustomHash)
    assert isinstance((), xtyping.HasCustomHash)

    class A:
        def __hash__(self):
            return 3

    assert isinstance(A(), xtyping.HasCustomHash)

    # PEP-683 Immortal objects have custom hash
    assert isinstance(None, xtyping.HasCustomHash) == (sys.version_info >= (3, 12))

    class B:
        __hash__ = None

    assert not isinstance(B(), xtyping.HasCustomHash)

    assert not isinstance(object(), xtyping.HasCustomHash)
    assert not isinstance(type, xtyping.HasCustomHash)


def test_is_protocol():
    class AProtocol(typing.Protocol):
        def do_something(self, value: int) -> int: ...

    class NotProtocol(AProtocol):
        def do_something_else(self, value: float) -> float: ...

    class AXProtocol(xtyping.Protocol):
        A = 1

    class NotXProtocol(AXProtocol):
        A = 1

    class AgainProtocol(AProtocol, xtyping.Protocol):
        def do_something_else(self, value: float) -> float: ...

    assert xtyping.is_protocol(AProtocol)
    assert xtyping.is_protocol(AXProtocol)

    assert not xtyping.is_protocol(NotProtocol)
    assert not xtyping.is_protocol(NotXProtocol)

    assert xtyping.is_protocol(AgainProtocol)


def test_get_partial_type_hints():
    def f1(a: int) -> float: ...

    assert xtyping.get_partial_type_hints(f1) == {"a": int, "return": float}

    class MissingRef: ...

    def f_partial(a: int) -> MissingRef: ...

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

    def f_nested_partial(a: int) -> dict[str, MissingRef]: ...

    assert xtyping.get_partial_type_hints(f_nested_partial) == {
        "a": int,
        "return": ForwardRef("dict[str, MissingRef]"),
    }
    assert xtyping.get_partial_type_hints(f_nested_partial, localns={"MissingRef": MissingRef}) == {
        "a": int,
        "return": dict[str, MissingRef],
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
    assert xtyping.eval_forward_ref("dict[str, tuple[int, float]]") == dict[str, tuple[int, float]]
    assert (
        xtyping.eval_forward_ref(ForwardRef("dict[str, tuple[int, float]]"))
        == dict[str, tuple[int, float]]
    )

    class MissingRef: ...

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
        ref := xtyping.eval_forward_ref(
            "Callable[[Annotated[int, 'Foo']], MissingRef]",
            globalns={"Annotated": Annotated, "Callable": Callable},
            localns={"MissingRef": MissingRef},
        )
    ) == Callable[[int], MissingRef]

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

    assert xtyping.infer_type(dict[str, int]) == dict[str, int]

    assert xtyping.infer_type({1, 2, 3}) == set[int]
    assert xtyping.infer_type(frozenset({"1", "2", "3"})) == frozenset[str]

    assert xtyping.infer_type({"a": [0], "b": [1]}) == dict[str, list[int]]

    assert xtyping.infer_type(str) == type[str]

    class A: ...

    assert xtyping.infer_type(A()) == A
    assert xtyping.infer_type(A) == type[A]

    def f1(): ...

    assert xtyping.infer_type(f1) == Callable[[], Any]

    def f2(a: int, b: float) -> None: ...

    assert xtyping.infer_type(f2) == Callable[[int, float], type(None)]

    def f3(
        a: dict[tuple[str, ...], list[int]],
        b: list[Callable[[list[int]], set[set[int]]]],
        c: type[list[int]],
    ) -> Any: ...

    assert (
        xtyping.infer_type(f3)
        == Callable[
            [
                dict[tuple[str, ...], list[int]],
                list[Callable[[list[int]], set[set[int]]]],
                type[list[int]],
            ],
            Any,
        ]
    )

    def f4(a: int, b: float, *, foo: tuple[str, ...] = ()) -> None: ...

    assert xtyping.infer_type(f4) == Callable[[int, float], type(None)]
    assert (
        xtyping.infer_type(f4, annotate_callable_kwargs=True)
        == Annotated[
            Callable[[int, float], type(None)], xtyping.CallableKwargsInfo({"foo": tuple[str, ...]})
        ]
    )


def test_is_single_dispatch_callable():
    import functools

    # `functools.singledispatch` results (and wrappers exposing `dispatch`/`register`).
    assert xtyping.is_single_dispatch_callable(functools.singledispatch(lambda _: None))

    # Plain callables and non-callables are rejected.
    assert not xtyping.is_single_dispatch_callable(lambda _: None)
    assert not xtyping.is_single_dispatch_callable(42)
