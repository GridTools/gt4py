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
import typing_extensions

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, ForwardRef, Optional, TypeVar, Union
import typing
import typing_extensions
from gt4py.eve.extra_typing import (
    supports_array_interface,
    supports_cuda_array_interface,
    supports_dlpack,
)
from gt4py.eve import extra_typing


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

    class ArrayInterface:
        __array_interface__ = "interface"

    class NoArrayInterface:
        pass

    assert supports_array_interface(ArrayInterface())
    assert not supports_array_interface(NoArrayInterface())
    assert not supports_array_interface("array")
    assert not supports_array_interface(None)


def test_supports_cuda_array_interface():

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


def test_module_does_not_re_export_typing():
    """`extra_typing` must export only definitions the standard modules do not provide.

    This is the invariant the module exists to hold: it used to star-import `typing`
    and `typing_extensions` and forward anything else through a module `__getattr__`,
    which meant mypy could not check a single `extra_typing.X` reference (an unknown
    name simply resolved to `Any`) and ruff could not see through it either. If a
    re-export creeps back in, both of those go quiet again.
    """
    borrowed = (typing, typing_extensions, collections.abc, collections, contextlib, re)
    re_exported = sorted(
        name
        for name in extra_typing.__all__
        if any(
            getattr(module, name, object()) is getattr(extra_typing, name) for module in borrowed
        )
    )
    assert re_exported == [], (
        f"'extra_typing' exports {re_exported}, which it does not define; import "
        f"those from 'typing', 'typing_extensions' or 'collections.abc' at the use "
        f"site instead."
    )


def test_module_has_no_getattr_fallback():
    """An unknown name must raise, not be forwarded to `typing`.

    The forwarding `__getattr__` is what made every `extra_typing.X` reference
    unverifiable: a typo resolved to `Any` and type-checked clean.
    """
    with pytest.raises(AttributeError):
        extra_typing.ThisNameDoesNotExist


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
        resolved = extra_typing.eval_forward_ref(ref)

        assert typing.get_origin(resolved) is getattr(builtins, replacement)
        assert typing.get_args(resolved) == expected_args
        # A builtin generic alias, not the deprecated 'typing._GenericAlias' object.
        assert type(resolved) is types.GenericAlias

    # An explicit 'globalns' is left alone, so the bare name is not injected there.
    with pytest.raises(NameError):
        extra_typing.eval_forward_ref(f"{name}[{args}]", globalns={})


@pytest.mark.parametrize("t", (int, float, dict, tuple, frozenset, collections.abc.Mapping))
def test_is_actual_valid_type(t):
    assert extra_typing.is_actual_type(t)


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
    assert not extra_typing.is_actual_type(t)


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
    assert extra_typing.get_actual_type(instance) == expected


def test_has_custom_hash_abc():
    assert isinstance(4, extra_typing.HasCustomHash)
    assert isinstance(True, extra_typing.HasCustomHash)
    assert isinstance((), extra_typing.HasCustomHash)

    class A:
        def __hash__(self):
            return 3

    assert isinstance(A(), extra_typing.HasCustomHash)

    # PEP-683 Immortal objects have custom hash
    assert isinstance(None, extra_typing.HasCustomHash) == (sys.version_info >= (3, 12))

    class B:
        __hash__ = None

    assert not isinstance(B(), extra_typing.HasCustomHash)

    assert not isinstance(object(), extra_typing.HasCustomHash)
    assert not isinstance(type, extra_typing.HasCustomHash)


def test_is_protocol():
    class AProtocol(typing.Protocol):
        def do_something(self, value: int) -> int: ...

    class NotProtocol(AProtocol):
        def do_something_else(self, value: float) -> float: ...

    class AXProtocol(typing_extensions.Protocol):
        A = 1

    class NotXProtocol(AXProtocol):
        A = 1

    class AgainProtocol(AProtocol, typing_extensions.Protocol):
        def do_something_else(self, value: float) -> float: ...

    assert typing_extensions.is_protocol(AProtocol)
    assert typing_extensions.is_protocol(AXProtocol)

    assert not typing_extensions.is_protocol(NotProtocol)
    assert not typing_extensions.is_protocol(NotXProtocol)

    assert typing_extensions.is_protocol(AgainProtocol)


def test_get_partial_type_hints():
    def f1(a: int) -> float: ...

    assert extra_typing.get_partial_type_hints(f1) == {"a": int, "return": float}

    class MissingRef: ...

    def f_partial(a: int) -> MissingRef: ...

    # This is expected behavior because this test file uses
    # 'from __future__ import annotations' and therefore local
    # references cannot be automatically resolved
    assert extra_typing.get_partial_type_hints(f_partial) == {
        "a": int,
        "return": ForwardRef("MissingRef"),
    }
    assert extra_typing.get_partial_type_hints(f_partial, localns={"MissingRef": MissingRef}) == {
        "a": int,
        "return": MissingRef,
    }
    assert extra_typing.get_partial_type_hints(f_partial, globalns={"MissingRef": int}) == {
        "a": int,
        "return": int,
    }

    def f_nested_partial(a: int) -> dict[str, MissingRef]: ...

    assert extra_typing.get_partial_type_hints(f_nested_partial) == {
        "a": int,
        "return": ForwardRef("dict[str, MissingRef]"),
    }
    assert extra_typing.get_partial_type_hints(
        f_nested_partial, localns={"MissingRef": MissingRef}
    ) == {
        "a": int,
        "return": dict[str, MissingRef],
    }

    def f_annotated(a: Annotated[int, "Foo"]) -> float:  # type: ignore[name-defined]  # used to work, now mypy is going berserk for unknown reasons
        ...

    assert extra_typing.get_partial_type_hints(f_annotated) == {"a": int, "return": float}
    assert extra_typing.get_partial_type_hints(f_annotated, include_extras=True) == {
        "a": Annotated[int, "Foo"],
        "return": float,
    }
    assert extra_typing.get_partial_type_hints(f_annotated, include_extras=True) != {
        "a": Annotated[int, "Bar"],
        "return": float,
    }


def test_eval_forward_ref():
    assert (
        extra_typing.eval_forward_ref("dict[str, tuple[int, float]]")
        == dict[str, tuple[int, float]]
    )
    assert (
        extra_typing.eval_forward_ref(ForwardRef("dict[str, tuple[int, float]]"))
        == dict[str, tuple[int, float]]
    )

    class MissingRef: ...

    assert (
        extra_typing.eval_forward_ref(
            "Callable[[int], MissingRef]", localns={"MissingRef": MissingRef}
        )
        == Callable[[int], MissingRef]
    )

    assert (
        extra_typing.eval_forward_ref(
            "Callable[[int], MissingRef]",
            globalns={"Callable": Callable},
            localns={"MissingRef": MissingRef},
        )
        == Callable[[int], MissingRef]
    )

    assert (
        ref := extra_typing.eval_forward_ref(
            "Callable[[Annotated[int, 'Foo']], MissingRef]",
            globalns={"Annotated": Annotated, "Callable": Callable},
            localns={"MissingRef": MissingRef},
        )
    ) == Callable[[int], MissingRef]

    assert (
        extra_typing.eval_forward_ref(
            "Callable[[Annotated[int, 'Foo']], MissingRef]",
            globalns={"Annotated": Annotated, "Callable": Callable},
            localns={"MissingRef": MissingRef},
            include_extras=True,
        )
        == Callable[[Annotated[int, "Foo"]], MissingRef]
    )


def test_infer_type():
    assert extra_typing.infer_type(3) == int

    assert extra_typing.infer_type(None) is type(None)  # noqa: E721  # do not compare types
    assert extra_typing.infer_type(type(None)) is type(None)  # noqa: E721  # do not compare types
    assert extra_typing.infer_type(None, none_as_type=False) is None
    assert extra_typing.infer_type(type(None), none_as_type=False) is None

    assert extra_typing.infer_type(dict[str, int]) == dict[str, int]

    assert extra_typing.infer_type({1, 2, 3}) == set[int]
    assert extra_typing.infer_type(frozenset({"1", "2", "3"})) == frozenset[str]

    assert extra_typing.infer_type({"a": [0], "b": [1]}) == dict[str, list[int]]

    assert extra_typing.infer_type(str) == type[str]

    class A: ...

    assert extra_typing.infer_type(A()) == A
    assert extra_typing.infer_type(A) == type[A]

    def f1(): ...

    assert extra_typing.infer_type(f1) == Callable[[], Any]

    def f2(a: int, b: float) -> None: ...

    assert extra_typing.infer_type(f2) == Callable[[int, float], type(None)]

    def f3(
        a: dict[tuple[str, ...], list[int]],
        b: list[Callable[[list[int]], set[set[int]]]],
        c: type[list[int]],
    ) -> Any: ...

    assert (
        extra_typing.infer_type(f3)
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

    assert extra_typing.infer_type(f4) == Callable[[int, float], type(None)]
    assert (
        extra_typing.infer_type(f4, annotate_callable_kwargs=True)
        == Annotated[
            Callable[[int, float], type(None)],
            extra_typing.CallableKwargsInfo({"foo": tuple[str, ...]}),
        ]
    )


def test_is_single_dispatch_callable():
    import functools

    # `functools.singledispatch` results (and wrappers exposing `dispatch`/`register`).
    assert extra_typing.is_single_dispatch_callable(functools.singledispatch(lambda _: None))

    # Plain callables and non-callables are rejected.
    assert not extra_typing.is_single_dispatch_callable(lambda _: None)
    assert not extra_typing.is_single_dispatch_callable(42)


# -- PEP 695 type aliases --
type SampleIntAlias = int
type SampleChainedAlias = SampleIntAlias
type SampleGenericAlias[T] = tuple[T, T]
type SampleRecursiveAlias = SampleRecursiveAlias
type SampleMutualAliasA = SampleMutualAliasB
type SampleMutualAliasB = SampleMutualAliasA
type SampleDivergingGenericAlias[T] = SampleDivergingGenericAlias[tuple[T]]
type SampleNestedTupleAlias[T] = tuple[T | SampleNestedTupleAlias[T], ...]


def test_is_type_alias():
    # Both the native 'typing' class and the 'typing_extensions' backport have to be
    # recognized: they are distinct classes and a native alias is not an instance
    # of the backport.
    assert extra_typing.is_type_alias(SampleIntAlias)
    assert extra_typing.is_type_alias(typing_extensions.TypeAliasType("Backported", str))

    assert not extra_typing.is_type_alias(int)
    assert not extra_typing.is_type_alias(list[int])
    assert not extra_typing.is_type_alias(SampleGenericAlias[int])


def test_eval_type_alias():
    assert extra_typing.eval_type_alias(SampleIntAlias) is int
    assert extra_typing.eval_type_alias(SampleChainedAlias) is int
    assert extra_typing.eval_type_alias(SampleGenericAlias[int]) == tuple[int, int]


def test_eval_type_alias_passes_through_non_aliases():
    for annotation in (int, list[int], typing.Any, None):
        assert extra_typing.eval_type_alias(annotation) is annotation


def test_eval_type_alias_with_undefined_value():
    # Alias values are evaluated lazily, so the name is only looked up here.
    type LazyAlias = _defined_later  # noqa: F821 [undefined-name]  # defined below

    with pytest.raises(NameError):
        extra_typing.eval_type_alias(LazyAlias)

    _defined_later = int

    assert extra_typing.eval_type_alias(LazyAlias) is int


def test_eval_type_alias_with_recursive_alias():
    with pytest.raises(TypeError, match="recursive definition"):
        extra_typing.eval_type_alias(SampleRecursiveAlias)


def test_eval_type_alias_with_alias_recursing_through_a_container():
    # Unlike 'type A = A', an alias recursing through a container is well founded: a
    # single resolution step already yields an annotation which is not an alias itself,
    # so it resolves instead of being reported as a cycle. The alias is still present
    # in the result, so consumers walking it have to break the recursion themselves.
    assert (
        extra_typing.eval_type_alias(SampleNestedTupleAlias[int])
        == tuple[int | SampleNestedTupleAlias[int], ...]
    )
    assert (
        extra_typing.eval_type_alias(extra_typing.NestedTuple[int])
        == tuple[int | extra_typing.NestedTuple[int], ...]
    )


def test_eval_type_alias_with_mutually_recursive_aliases():
    # Reported as recursive rather than as too deeply nested, which is what a bare
    # depth bound would have to say: a cycle repeats an alias, so it is detected as
    # soon as one is seen twice, however long the cycle is.
    with pytest.raises(TypeError, match="'SampleMutualAliasA' cannot be resolved.*recursive"):
        extra_typing.eval_type_alias(SampleMutualAliasA)


def test_eval_type_alias_with_diverging_generic_alias():
    # A parametrized alias which grows on every step never repeats an annotation, so
    # the visited-alias check cannot see it and the depth bound is what stops it.
    with pytest.raises(TypeError, match="nested too deeply"):
        extra_typing.eval_type_alias(SampleDivergingGenericAlias[int])


def test_eval_type_alias_with_failing_value():
    # Anything the lazily evaluated alias value raises is a problem with the alias
    # itself, so it is reported as such instead of escaping as a raw error. Only
    # 'NameError' stays unwrapped, since callers defer on it (see the test above).
    type BrokenAlias = _empty_module.missing_attribute  # noqa: F821 [undefined-name]  # defined below

    _empty_module = types.ModuleType("_empty_module")

    with pytest.raises(TypeError, match="'BrokenAlias' cannot be resolved") as exc_info:
        extra_typing.eval_type_alias(BrokenAlias)

    # The actual cause is kept, both in the message and as the chained exception.
    assert "missing_attribute" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, AttributeError)


# -- get_represented_types --
class SampleReprA: ...


class SampleReprB: ...


type SampleUnionAlias = SampleReprA | SampleReprB
type SampleNestedUnionAlias = SampleUnionAlias | int


def test_get_represented_types():
    assert extra_typing.get_represented_types(int) == (int,)
    assert extra_typing.get_represented_types(Union[SampleReprA, SampleReprB]) == (
        SampleReprA,
        SampleReprB,
    )
    assert extra_typing.get_represented_types(SampleReprA | SampleReprB) == (
        SampleReprA,
        SampleReprB,
    )
    assert extra_typing.get_represented_types(list[int]) == (list,)


def test_get_represented_types_resolves_type_aliases():
    # An unresolved alias would silently yield an empty tuple, which turns every
    # downstream 'isinstance()' check against the result into a constant 'False'.
    assert extra_typing.get_represented_types(SampleIntAlias) == (int,)
    assert extra_typing.get_represented_types(SampleUnionAlias) == (SampleReprA, SampleReprB)
    assert extra_typing.get_represented_types(SampleNestedUnionAlias) == (
        SampleReprA,
        SampleReprB,
        int,
    )


def test_get_represented_types_with_alias_recursing_through_a_container():
    # The recursion stops at the container, since generic annotations are represented
    # by their origin and are not walked any further.
    assert extra_typing.get_represented_types(SampleNestedTupleAlias[int]) == (tuple,)
    assert extra_typing.get_represented_types(extra_typing.NestedTuple) == (tuple,)
    assert extra_typing.get_represented_types(extra_typing.NestedTuple[int]) == (tuple,)
    assert extra_typing.get_represented_types(extra_typing.MaybeNestedInTuple[int]) == (int, tuple)


def test_get_represented_types_with_alias_nested_in_annotation():
    assert extra_typing.get_represented_types(Optional[SampleIntAlias]) == (int, type(None))
    assert extra_typing.get_represented_types(Union[SampleUnionAlias, int]) == (
        SampleReprA,
        SampleReprB,
        int,
    )
