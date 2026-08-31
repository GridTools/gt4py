# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Typing definitions that `typing` and `typing_extensions` do not provide.

This module is **not** a replacement for `typing`: it does not re-export it.
Import the standard names from `typing`, `typing_extensions` or
`collections.abc` directly, and take only GT4Py's own definitions from here --
the array and DLPack protocols, the nested-collection aliases, the descriptor
and dataclass protocols, and the annotation utilities.

Where a name lives, and why:

- `collections.abc` -- the container protocols (`Sequence`, `Mapping`,
  `Callable`, `Iterable`, `Iterator`, `Collection`, `Generator`, ...). Their
  `typing` spellings are deprecated by PEP 585.
- `typing` -- everything the standard library provides on the supported floor,
  including `TypeVar`, `TypeVarTuple`, `ParamSpec` and `NamedTuple`. The
  `typing_extensions` versions of those four differ only in PEP 696
  `default=` support and generic/defaulted named tuples, neither of which this
  codebase uses.
- `typing_extensions` -- `TypeIs` and `is_protocol`, which Python 3.12 lacks
  (both land in `typing` at 3.13, so they move when the floor rises); plus
  `Protocol`, `runtime_checkable` and `get_type_hints`, whose implementations
  genuinely differ from the `typing` ones. `eve` runs every annotation through
  `get_type_hints`, and `gt4py.next` declares `@runtime_checkable` protocols
  that are checked on hot paths, so those keep the implementation they have
  always had.

Run `./scripts/run check typing-extensions-usage` after raising the Python
floor to find the `typing_extensions` imports that can move to `typing`.
"""

from __future__ import annotations

import abc as _abc
import array as _array
import collections as _collections
import collections.abc as _collections_abc
import contextlib as _contextlib
import dataclasses as _dataclasses
import functools as _functools
import inspect as _inspect
import mmap as _mmap
import pickle as _pickle
import re as _re
import types as _types
import typing as _typing
from collections.abc import (
    Buffer,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    ForwardRef,
    Generic,
    Literal,
    Never,
    NotRequired,
    Optional,
    ParamSpec,
    Self,
    SupportsBytes,
    TypeAlias,
    TypeAliasType,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    final,
    get_args,
    get_origin,
    overload,
)

import numpy.typing as npt
import typing_extensions as _typing_extensions
from typing_extensions import Protocol, TypeIs, get_type_hints, runtime_checkable


#: Only this module's own definitions are exported. The names it imports for its own
#: use stay reachable as attributes -- that is true of any module -- but they are
#: deliberately not offered here: import them from 'typing', 'typing_extensions'
#: or 'collections.abc' at the use site.
__all__ = [
    "ArgsOnlyCallable",
    "ArrayInterface",
    "ArrayInterfaceTypedDict",
    "CUDAArrayInterface",
    "CUDAArrayInterfaceTypedDict",
    "CallableKwargsInfo",
    "DLPackBuffer",
    "DLPackDevice",
    "DataDescriptor",
    "DataclassABC",
    "DevToolsPrettyPrintable",
    "FrozenDataclass",
    "HasCustomHash",
    "HashlibAlgorithm",
    "MaybeNested",
    "MaybeNestedInList",
    "MaybeNestedInSequence",
    "MaybeNestedInTuple",
    "MultiStreamDLPackBuffer",
    "NestedList",
    "NestedSequence",
    "NestedTuple",
    "NoArgsCallable",
    "NonDataDescriptor",
    "OpaqueMutableMapping",
    "ReadOnlyBuffer",
    "ReadableBuffer",
    "SingleDispatchCallable",
    "SingleStreamDLPackBuffer",
    "SingleTypeAnnotation",
    "SolvedTypeAnnotation",
    "SourceTypeAnnotation",
    "StdGenericAliasType",
    "StrictArrayInterface",
    "StrictCUDAArrayInterface",
    "SupportsArray",
    "TypeAnnotation",
    "TypedNamedTupleABC",
    "WriteableBuffer",
    "annotations",
    "eval_forward_ref",
    "eval_type_alias",
    "get_actual_type",
    "get_partial_type_hints",
    "get_represented_types",
    "has_type_parameters",
    "infer_type",
    "is_Any",
    "is_actual_type",
    "is_maybe_nested_in_tuple_of",
    "is_nested_tuple_of",
    "is_single_dispatch_callable",
    "is_type_alias",
    "is_type_with_custom_hash",
    "normalize_union",
    "resolve_annotation",
    "strip_annotated",
    "supports_array",
    "supports_array_interface",
    "supports_cuda_array_interface",
    "supports_dlpack",
]


# -- Forward-reference resolution --
#
# A forward reference is a string, so resolving it needs a namespace to evaluate it
# in. This module used to be that namespace: it star-imported 'typing' and
# 'typing_extensions', so 'globals()' happened to contain every typing name and
# 'eval_forward_ref' could fall back to it. Nothing re-exports anything now, so the
# namespace is built explicitly below -- which is also what it always meant.

#: The 'typing' aliases of the builtin generics are deprecated by PEP 585, but they
#: stay valid in user-written annotations, and resolving one through this module has
#: always normalized it to the builtin generic ('typing.List[int]' -> 'list[int]').
_DEPRECATED_ALIAS_REPLACEMENTS: Final[Mapping[str, Any]] = {
    "Dict": dict,
    "FrozenSet": frozenset,
    "List": list,
    "Set": set,
    "Tuple": tuple,
    "Type": type,
}

#: Names that a forward reference may use and that must resolve to the
#: 'collections.abc' / 'collections' / 'contextlib' / 're' object rather than to the
#: deprecated 'typing' alias of the same name.
_NON_TYPING_ALIASES: Final[Mapping[str, Any]] = {
    **{
        name: getattr(_collections_abc, name)
        for name in (
            "AsyncGenerator",
            "AsyncIterable",
            "AsyncIterator",
            "Awaitable",
            "Callable",
            "Collection",
            "Container",
            "Coroutine",
            "Generator",
            "ItemsView",
            "Iterable",
            "Iterator",
            "KeysView",
            "Mapping",
            "MappingView",
            "MutableMapping",
            "MutableSequence",
            "MutableSet",
            "Reversible",
            "Sequence",
            "ValuesView",
        )
    },
    "AbstractSet": _collections_abc.Set,
    "ChainMap": _collections.ChainMap,
    "Counter": _collections.Counter,
    "OrderedDict": _collections.OrderedDict,
    "defaultdict": _collections.defaultdict,
    "deque": _collections.deque,
    "AsyncContextManager": _contextlib.AbstractAsyncContextManager,
    "ContextManager": _contextlib.AbstractContextManager,
    "Match": _re.Match,
    "Pattern": _re.Pattern,
}


@_functools.cache
def _forward_ref_namespace() -> dict[str, Any]:
    """Namespace used to resolve a forward reference when the caller provides none.

    Later entries win, so the precedence is: 'typing', then 'typing_extensions',
    then the non-'typing' spellings of the container protocols, then this module's
    own definitions, then the builtin generics standing in for the deprecated
    'typing' aliases.
    """
    namespace: dict[str, Any] = {}
    namespace.update(vars(_typing))
    namespace.update(vars(_typing_extensions))
    namespace.update(_NON_TYPING_ALIASES)
    namespace.update(globals())
    namespace.update(_DEPRECATED_ALIAS_REPLACEMENTS)
    return namespace


class _ForwardRefTypingNamespace:
    """Namespace bound to the name 'typing' while evaluating forward references.

    A reference spelled 'typing.Sequence[int]' resolves through this object, so that
    'typing_extensions' definitions take priority, the container protocols come from
    'collections.abc', and the deprecated builtin aliases give back the builtin
    generic instead of the deprecated 'typing' object.
    """

    def __getattr__(self, name: str) -> Any:
        namespace = _forward_ref_namespace()
        if name in namespace:
            return namespace[name]
        raise AttributeError(f"Module 'typing' has no attribute '{name}'.")


_FORWARD_REF_TYPING_NS: Final = _ForwardRefTypingNamespace()


# -- Common type aliases --
NoArgsCallable = Callable[[], Any]

_A = TypeVar("_A", contravariant=True)
_R = TypeVar("_R", covariant=True)


class ArgsOnlyCallable(Protocol[_A, _R]):
    def __call__(self, *args: _A) -> _R: ...


_T_co = TypeVar("_T_co", covariant=True)

type NestedSequence[_T_co] = Sequence[_T_co | NestedSequence[_T_co]]
type NestedList[_T_co] = list[_T_co | NestedList[_T_co]]
type NestedTuple[_T_co] = tuple[_T_co | NestedTuple[_T_co], ...]

type MaybeNested[_T_co] = _T_co | NestedSequence[_T_co]
type MaybeNestedInSequence[_T_co] = _T_co | NestedSequence[_T_co]
type MaybeNestedInList[_T_co] = _T_co | NestedList[_T_co]
type MaybeNestedInTuple[_T_co] = _T_co | NestedTuple[_T_co]


def is_nested_tuple_of(value: object, type_: type[_T_co]) -> TypeIs[NestedTuple[_T_co]]:
    """Check if `value` is a nested tuple of elements of type `type_`."""
    return isinstance(value, tuple) and all(
        isinstance(v, type_) or (isinstance(v, tuple) and is_nested_tuple_of(v, type_))
        for v in value
    )


def is_maybe_nested_in_tuple_of(
    value: object, type_: type[_T_co]
) -> TypeIs[MaybeNestedInTuple[_T_co]]:
    """Check if `value` is either of type `type_` or a nested tuple of elements of type `type_`."""
    return isinstance(value, type_) or is_nested_tuple_of(value, type_)


# -- Typing annotations --
SingleTypeAnnotation = Union[
    type[Any],
    _types.GenericAlias,
    _typing._BaseGenericAlias,  # type: ignore[name-defined]  # _BaseGenericAlias is not exported in stub
    # Both PEP 695 type alias implementations, for the same reason as in `_TypeAliasTypes`
    _typing.TypeAliasType,
    _typing_extensions.TypeAliasType,
]

SolvedTypeAnnotation = Union[SingleTypeAnnotation, _typing._SpecialForm]

TypeAnnotation = Union[ForwardRef, SolvedTypeAnnotation]
SourceTypeAnnotation = Union[str, TypeAnnotation]

StdGenericAliasType: Final[type[Any]] = type(list[int])

if TYPE_CHECKING:
    StdGenericAlias: TypeAlias = _types.GenericAlias

_TypingSpecialFormType: Final[type[Any]] = _typing._SpecialForm
_TypingGenericAliasType: Final[type[Any]] = _typing._BaseGenericAlias  # type: ignore[attr-defined]  # _BaseGenericAlias / _GenericAlias are not exported in stub


# -- Standard Python protocols --
_C = TypeVar("_C")
_V = TypeVar("_V")


class NonDataDescriptor(Protocol[_C, _V]):
    """Typing protocol for non-data descriptor classes.

    See https://docs.python.org/3/howto/descriptor.html for further information.
    """

    @overload
    def __get__(
        self, _instance: Literal[None], _owner_type: Optional[type[_C]] = None
    ) -> NonDataDescriptor[_C, _V]: ...

    @overload
    def __get__(self, _instance: _C, _owner_type: Optional[type[_C]] = None) -> _V: ...

    def __get__(
        self, _instance: Optional[_C], _owner_type: Optional[type[_C]] = None
    ) -> _V | NonDataDescriptor[_C, _V]: ...


class DataDescriptor(NonDataDescriptor[_C, _V], Protocol):
    """Typing protocol for data descriptor classes.

    See https://docs.python.org/3/howto/descriptor.html for further information.
    """

    def __set__(self, _instance: _C, _value: _V) -> None: ...

    def __delete__(self, _instance: _C) -> None: ...


# -- Based on typeshed definitions --
ReadOnlyBuffer: TypeAlias = Union[bytes, SupportsBytes]
WriteableBuffer: TypeAlias = Union[
    bytearray, memoryview, _array.array, _mmap.mmap, _pickle.PickleBuffer
]
ReadableBuffer: TypeAlias = Union[ReadOnlyBuffer, WriteableBuffer]


_P = ParamSpec("_P")
_T = TypeVar("_T")


@runtime_checkable
class SingleDispatchCallable(Protocol[_P, _T]):
    # `functools.singledispatch` copies the wrapped function's identity onto
    # the dispatcher; declaring these attributes allows callers to overwrite
    # it (e.g. to give a dispatcher its own pickle identity).
    __name__: str
    __qualname__: str
    registry: Mapping[Any, Callable[_P, _T]]

    def dispatch(self, cls: Any) -> Callable[_P, _T]: ...

    @overload
    def register(
        self, cls: Any, func: Literal[None] = None
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...

    @overload
    def register(self, cls: Any, func: Callable[_P, _T]) -> Callable[_P, _T]: ...

    def register(
        self, cls: Any, func: Callable[_P, _T] | None = None
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]] | Callable[_P, _T]: ...

    def _clear_cache(self) -> None: ...

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T: ...


def is_single_dispatch_callable(
    func: Callable[_P, _T],
) -> TypeGuard[SingleDispatchCallable[_P, _T]]:
    return (
        callable(func)
        and getattr(func, "registry", None) is not None
        and callable(getattr(func, "dispatch", None))
        and callable(getattr(func, "register", None))
        and callable(getattr(func, "_clear_cache", None))
    )


class HashlibAlgorithm(Protocol):
    """Used in the hashlib module of the standard library."""

    @property
    def block_size(self) -> int: ...

    @property
    def digest_size(self) -> int: ...

    @property
    def name(self) -> str: ...

    def __init__(self, data: ReadableBuffer = ...) -> None: ...

    def copy(self) -> Self: ...

    def update(self, data: Buffer, /) -> None: ...

    def digest(self) -> bytes: ...

    def hexdigest(self) -> str: ...


# -- Third party protocols --
class SupportsArray(Protocol):
    def __array__(self, dtype: Optional[npt.DTypeLike] = None, /) -> npt.NDArray[Any]: ...


def supports_array(value: Any) -> TypeGuard[SupportsArray]:
    return hasattr(value, "__array__")


class ArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> dict[str, Any]: ...


class ArrayInterfaceTypedDict(TypedDict):
    shape: tuple[int, ...]
    typestr: str
    descr: NotRequired[list[tuple]]
    data: NotRequired[tuple[int, bool]]
    strides: NotRequired[Optional[tuple[int, ...]]]
    mask: NotRequired[Optional["StrictArrayInterface"]]
    offset: NotRequired[int]
    version: int


class StrictArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> ArrayInterfaceTypedDict: ...


def supports_array_interface(value: Any) -> TypeGuard[ArrayInterface]:
    return hasattr(value, "__array_interface__")


class CUDAArrayInterface(Protocol):
    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...


class CUDAArrayInterfaceTypedDict(TypedDict):
    shape: tuple[int, ...]
    typestr: str
    data: tuple[int, bool]
    version: int
    strides: NotRequired[Optional[tuple[int, ...]]]
    descr: NotRequired[list[tuple]]
    mask: NotRequired[Optional["StrictCUDAArrayInterface"]]
    stream: NotRequired[Optional[int]]


class StrictCUDAArrayInterface(Protocol):
    @property
    def __cuda_array_interface__(self) -> CUDAArrayInterfaceTypedDict: ...


def supports_cuda_array_interface(value: Any) -> TypeGuard[CUDAArrayInterface]:
    """Check if the given value supports the CUDA Array Interface."""
    return hasattr(value, "__cuda_array_interface__")


DLPackDevice = tuple[int, int]


class MultiStreamDLPackBuffer(Protocol):
    def __dlpack__(self, *, stream: Optional[int] = None) -> Any: ...

    def __dlpack_device__(self) -> DLPackDevice: ...


class SingleStreamDLPackBuffer(Protocol):
    def __dlpack__(self, *, stream: None = None) -> Any: ...

    def __dlpack_device__(self) -> DLPackDevice: ...


DLPackBuffer: TypeAlias = Union[MultiStreamDLPackBuffer, SingleStreamDLPackBuffer]


def supports_dlpack(value: Any) -> TypeGuard[DLPackBuffer]:
    """Check if a given object supports the DLPack protocol."""
    return callable(getattr(value, "__dlpack__", None)) and callable(
        getattr(value, "__dlpack_device__", None)
    )


class DevToolsPrettyPrintable(Protocol):
    """Used by python-devtools (https://python-devtools.helpmanual.io/)."""

    def __pretty__(
        self, fmt: Callable[[Any], Any], **kwargs: Any
    ) -> Generator[Any, None, None]: ...


# -- Added functionality --
_ArtefactTypes: Final[tuple[type, ...]] = (_types.GenericAlias, _typing.Any)


def is_actual_type(obj: Any) -> TypeGuard[type[Any]]:
    """Check if an object has an actual type and instead of a typing artefact like ``GenericAlias`` or ``Any``.

    This is needed because since Python 3.9:
        ``isinstance(types.GenericAlias(), type) is True``
    and since Python 3.11:
        ``isinstance(typing.Any, type) is True``
    """
    return (
        isinstance(obj, type) and (obj not in _ArtefactTypes) and (type(obj) not in _ArtefactTypes)
    )


# -- PEP 604 unions (``X | Y``) --


def normalize_union(annotation: Any) -> Any:
    """Rewrite a PEP 604 union (``X | Y``) as the equivalent ``typing.Union[X, Y]``.

    Before 3.14 the two are different runtime objects: ``get_origin(int | None)`` is
    ``types.UnionType``, ``get_origin(Optional[int])`` is ``typing.Union``. A site
    comparing against only one does not raise on the other, it falls through to a
    no-match branch -- so call this at the head of any funnel dispatching on
    union-ness. Anything already normalized, and anything that is not a union at all,
    is returned as the identical object.

    See https://github.com/python/cpython/issues/105499.

    Examples:
        >>> normalize_union(int | None) == Optional[int]
        True

        >>> normalize_union(int) is int
        True
    """
    if isinstance(annotation, _types.UnionType) and get_origin(annotation) is not _typing.Union:
        # The second test matters only on 3.14, where 'typing.Union' *is*
        # 'types.UnionType': every union passes the 'isinstance' there, so without it
        # an already-normalized annotation would be rebuilt into an equal but not
        # identical object, breaking the ``is`` check this function promises.
        return _typing.Union[annotation.__args__]
    return annotation


def strip_annotated(annotation: Any) -> Any:
    """Return the type wrapped by ``Annotated[X, ...]``, or ``annotation`` unchanged.

    ``Annotated`` metadata is not part of the type, so every funnel dispatching on an
    annotation has to see through it -- and none of them fails loudly otherwise. On
    3.12 ``typing.Annotated`` is a class, so a funnel testing "is this a class" claims
    it and builds nonsense: a ``typing.Annotated(value)`` call, an ``isinstance``
    check nothing satisfies, a mutability verdict read off the metadata. From 3.13 it
    is a special form again and falls through to a no-match branch instead.

    ``typing`` flattens nested ``Annotated``, so one step is enough. Anything else is
    returned as the identical object, so callers can detect the no-op with ``is``.

    Examples:
        >>> strip_annotated(Annotated[int, "meta"]) is int
        True

        >>> strip_annotated(Annotated[Optional[int], "meta"]) == Optional[int]
        True

        >>> strip_annotated(int) is int
        True
    """
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


# -- PEP 695 type aliases (``type X = ...``) --
#
# Resolved at the annotation-dispatch funnels (`type_validation`,
# `datamodels.core._make_type_converter`, `get_represented_types` below), not by
# rewriting stored annotations: that would force `__value__` evaluation at class
# creation and break aliases defined before their target. Wire any new shape-dispatching
# consumer in as well; a funnel left out does not raise, it falls through to its
# no-match branch (``()`` for `get_represented_types`), silently making every
# `isinstance()` against the result `False`.
#
# Not supported: `ClassVar` behind an alias, aliases used as runtime values (base
# class, constructor, `isinstance()` argument) -- the reason ruff's `UP040` stays off --
# and the `Coerced`/`Unchecked` tag lookup in `datamodels.core`, which reads the
# `Annotated` metadata off the raw annotation: `type MyCoerced = Coerced[int]` loses
# the tag rather than raising.

#: Both PEP 695 alias implementations: they are distinct classes and neither is an
#: instance of the other, so both have to be checked. ``hasattr(obj, "__value__")`` is
#: not equivalent -- ``MyGenericAlias[int]`` proxies attribute lookups to its origin and
#: passes it without being an alias.
_TypeAliasTypes: Final[tuple[type, ...]] = (
    _typing.TypeAliasType,
    _typing_extensions.TypeAliasType,
)

#: Upper bound for the number of resolution steps in `eval_type_alias`. True cycles are
#: detected exactly by the set of visited aliases; this bound is the backstop for the
#: chains that diverge without ever repeating, like ``type G[T] = G[Tuple[T]]``.
_MAX_TYPE_ALIAS_DEPTH: Final = 64


def is_type_alias(obj: Any) -> TypeGuard[TypeAliasType]:
    """Check if an object is a PEP 695 type alias (``type X = ...``)."""
    return isinstance(obj, _TypeAliasTypes)


def eval_type_alias(annotation: Any) -> Any:
    """Replace a PEP 695 type alias by the annotation it stands for.

    Chained aliases are followed until a non-alias annotation is reached, and
    parametrized generic aliases (``MyAlias[int]``) get their type parameters
    substituted. Any other annotation is returned unchanged (as the identical
    object), so callers can use an identity check to find out whether anything
    was actually resolved.

    Note that alias values are evaluated lazily by the interpreter, so this is
    the point where the names used in the alias definition are looked up for
    the first time.

    Args:
        annotation: Any type annotation.

    Returns:
        The annotation the alias stands for, or `annotation` itself.

    Raises:
        NameError: If the alias value references a name which is not defined yet.
        TypeError: If the alias is recursive (reported as such), nested too
            deeply, cannot be parametrized with the given type arguments, or its
            value fails to evaluate for any other reason.

    Examples:
        >>> type MyInt = int
        >>> eval_type_alias(MyInt)
        <class 'int'>

        >>> eval_type_alias(float)
        <class 'float'>
    """
    original_annotation = annotation
    # A set of the aliases already walked through detects a true cycle ('type A = A',
    # or a mutual 'A -> B -> A') exactly and reports it as such. It cannot replace the
    # depth bound, though: a parametrized alias like 'type G[T] = G[Tuple[T]]' builds a
    # bigger annotation on every step and so never repeats one. The two are complements.
    seen: set[TypeAliasType] = set()
    recursive = False
    try:
        for _ in range(_MAX_TYPE_ALIAS_DEPTH):
            if is_type_alias(annotation):
                if annotation in seen:
                    recursive = True
                    break
                seen.add(annotation)
                annotation = annotation.__value__
            elif is_type_alias(alias := get_origin(annotation)):
                annotation = alias.__value__[get_args(annotation)]
            else:
                return annotation
    except NameError:
        # A signal, not an error: an alias may be defined before its target, so a name
        # missing here may exist by the time the field is used. 'datamodels' catches it
        # and defers to the first instantiation, as it already does for string forward
        # references, at the cost of a later error naming the bare field.
        raise
    except Exception as error:
        # An alias value is arbitrary user code, so it can fail in any way (a typo'd
        # dtype raises 'AttributeError'). Unlike 'NameError', none of those will ever
        # resolve, so they fail here rather than being deferred. Wrapping them into one
        # type lets consumers keep a single narrow 'except' and propagate this message
        # verbatim, as it is the only text naming the actual cause.
        raise TypeError(
            f"Type alias '{original_annotation}' cannot be resolved ({error})."
        ) from error

    raise TypeError(
        f"Type alias '{original_annotation}' cannot be resolved "
        f"({'recursive definition' if recursive else 'nested too deeply'})."
    )


def resolve_annotation(annotation: Any) -> Any:
    """Resolve PEP 695 aliases and strip ``Annotated`` metadata until neither applies.

    The two wrappers nest into each other any number of times
    (``type A = Annotated[B, ...]``, ``type B = Annotated[int, ...]``), so neither
    `eval_type_alias` nor `strip_annotated` alone reaches the underlying type -- and a
    funnel applying them in separate branches recurses forever when the two wrap each
    other in a cycle. Anything else is returned as the identical object, so callers
    can detect the no-op with ``is``.

    Args:
        annotation: Any type annotation.

    Returns:
        The annotation left once every alias and ``Annotated`` layer is gone, or
        `annotation` itself.

    Raises:
        NameError: As `eval_type_alias`, so that consumers can still defer.
        TypeError: If the layers cycle (``type R = Annotated[R, ...]``) or nest too
            deeply, plus everything `eval_type_alias` reports.

    Examples:
        >>> type MyInt = Annotated[int, "meta"]
        >>> resolve_annotation(MyInt)
        <class 'int'>

        >>> resolve_annotation(float) is float
        True
    """
    original_annotation = annotation
    # Identity, not equality: '__eq__' on arbitrary annotation objects is not
    # guaranteed to be usable here.
    seen: list[Any] = []
    for _ in range(_MAX_TYPE_ALIAS_DEPTH):
        unaliased = eval_type_alias(annotation)
        stripped = strip_annotated(unaliased)
        if unaliased is annotation and stripped is annotation:
            return annotation
        if any(stripped is s for s in seen):
            raise TypeError(
                f"Type annotation '{original_annotation}' cannot be resolved "
                "(recursive definition)."
            )
        seen.append(annotation)
        annotation = stripped

    raise TypeError(
        f"Type annotation '{original_annotation}' cannot be resolved (nested too deeply)."
    )


def is_Any(obj: Any) -> bool:
    """Check if an object is the ``Any`` special form."""
    # 'typing_extensions' re-exports 'typing.Any' on every supported version, so the
    # two implementations that used to exist below the 3.11 floor are now one object.
    return obj is _typing.Any


def has_type_parameters(cls: type[Any]) -> bool:
    """Return ``True`` if obj is a generic class with type parameters."""
    return issubclass(cls, Generic) and len(getattr(cls, "__parameters__", [])) > 0


def get_actual_type(obj: _T) -> type[_T]:
    """Return type of an object (also working for GenericAlias instances which pretend to be an actual type)."""
    return StdGenericAliasType if isinstance(obj, StdGenericAliasType) else type(obj)


def get_represented_types(
    type_annotation: TypeAnnotation,
    *,
    globalns: Optional[dict[str, Any]] = None,
    localns: Optional[dict[str, Any]] = None,
) -> tuple[type, ...]:
    """Return a tuple with all the actual types contained in a type annotation."""
    recurse = _functools.partial(get_represented_types, globalns=globalns, localns=localns)

    def recurse_all(annotations: Iterable[TypeAnnotation]) -> tuple[type, ...]:
        return _functools.reduce(lambda acc, c: acc + recurse(c), annotations, ())

    # PEP 695 aliases are opaque objects which no other branch below matches, so an
    # unresolved one would silently yield an empty tuple, and the 'Annotated' special
    # form is not a represented type either. Nested occurrences of both are covered
    # for free, since the generic branches recurse through this same function.
    type_annotation = resolve_annotation(type_annotation)

    if type_annotation is Ellipsis:
        return ()

    if is_actual_type(type_annotation):
        return (type_annotation,)

    if isinstance(type_annotation, TypeVar):
        if type_annotation.__bound__:
            return recurse(type_annotation.__bound__)
        if type_annotation.__constraints__:
            return recurse_all(type_annotation.__constraints__)
        if typevar_default := getattr(type_annotation, "__default__", None):
            return recurse(typevar_default)

    if isinstance(type_annotation, ForwardRef):
        return recurse(eval_forward_ref(type_annotation, globalns=globalns, localns=localns))

    # Generic types
    origin_type = get_origin(type_annotation)
    type_args = get_args(type_annotation)

    if origin_type is Literal:
        # 'Literal' arguments are values, not types, so they cannot be recursed
        # into: the type each one represents is its own type.
        return tuple(dict.fromkeys(type(arg) for arg in type_args))

    if origin_type in [Union, _types.UnionType]:
        return recurse_all(t for t in type_args)

    if origin_type is not None:
        return (origin_type,)

    return ()


def is_type_with_custom_hash(type_: type[Any]) -> bool:
    return type_.__hash__ not in (None, object.__hash__)


class HasCustomHash(Hashable):
    """ABC for types defining a custom hash function."""

    @classmethod
    def __subclasshook__(cls, candidate_cls: type) -> bool:
        return is_type_with_custom_hash(candidate_cls)


class TypedNamedTupleABC(_abc.ABC, Generic[_T_co]):
    """ABC for `tuple` subclasses created with `collections.abc.namedtuple()`."""

    # Replicate the standard tuple API
    @overload
    @_abc.abstractmethod
    def __getitem__(self, index: int) -> _T_co: ...

    @overload
    @_abc.abstractmethod
    def __getitem__(self, index: slice) -> Self: ...

    @_abc.abstractmethod
    def __getitem__(self, index: Union[int, slice]) -> Union[_T_co, Self]: ...

    @_abc.abstractmethod
    def __len__(self) -> int: ...

    @_abc.abstractmethod
    def __contains__(self, value: object) -> bool: ...

    @_abc.abstractmethod
    def __iter__(self) -> Iterator[_T_co]: ...

    @_abc.abstractmethod
    def __add__(self, other: Self) -> Self: ...

    @_abc.abstractmethod
    def __mul__(self, other: int) -> Self: ...

    @_abc.abstractmethod
    def __rmul__(self, other: int) -> Self: ...

    @_abc.abstractmethod
    def index(self, value: Any, start: int = 0, stop: Optional[int] = None) -> int: ...

    @_abc.abstractmethod
    def count(self, value: Any) -> int: ...

    # Add specific namedtuple methods
    _fields: ClassVar[tuple[str, ...]]

    @_abc.abstractmethod
    def _make(self, iterable: Iterable) -> Self: ...

    @_abc.abstractmethod
    def _asdict(self) -> dict[str, Any]: ...

    @_abc.abstractmethod
    def _replace(self, **kwargs: Any) -> Self: ...

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return (
            issubclass(subclass, tuple)
            and (_typing.NamedTuple in getattr(subclass, "__orig_bases__", ()))
        ) or (
            (field_names := getattr(subclass, "_fields", None)) is not None
            and {*field_names} <= _typing.get_type_hints(subclass).keys()
        )


class DataclassABC(_abc.ABC):
    """ABC for data classes."""

    __dataclass_fields__: ClassVar[dict[str, _dataclasses.Field]]
    __dataclass_params__: ClassVar[_DataclassParamsABC]

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return _dataclasses.is_dataclass(subclass)


class _DataclassParamsABC(_abc.ABC):
    init: bool
    repr: bool
    eq: bool
    order: bool
    unsafe_hash: bool
    frozen: bool
    match_args: bool
    kw_only: bool
    slots: bool
    weakref_slot: bool


class FrozenDataclass(DataclassABC):
    """ABC for frozen data classes."""

    __dataclass_params__: ClassVar[_FrozenDataclassParamsABC]

    @_abc.abstractmethod
    def __setattr__(self, name: str, value: Any) -> Never: ...

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        try:
            return _dataclasses.is_dataclass(subclass) and (
                subclass.__dataclass_params__.frozen is True  # type: ignore[attr-defined]  # subclass.__dataclass_params__ is ok after check
            )
        except AttributeError:
            return False


class _FrozenDataclassParamsABC(_DataclassParamsABC):
    frozen: Literal[True]


_KT = TypeVar("_KT", contravariant=True)
_VT = TypeVar("_VT")


class OpaqueMutableMapping(Protocol[_KT, _VT]):
    """
    Mutable mapping without access to the keys, just setting, getting, deleting with a given key.
    """

    def __getitem__(self, key: _KT) -> _VT: ...

    def __setitem__(self, key: _KT, value: _VT) -> None: ...

    def __delitem__(self, key: _KT) -> None: ...


def get_partial_type_hints(
    obj: Union[
        object,
        Callable,
        _types.FunctionType,
        _types.BuiltinFunctionType,
        _types.MethodType,
        _types.ModuleType,
        _types.WrapperDescriptorType,
        _types.MethodWrapperType,
        _types.MethodDescriptorType,
    ],
    globalns: Optional[dict[str, Any]] = None,
    localns: Optional[dict[str, Any]] = None,
    include_extras: bool = False,
) -> dict[str, Union[type[Any], ForwardRef]]:
    """Return a dictionary with type hints (using forward refs for undefined names) for a function, method, module or class object.

    For each member type hint in the object a :class:`typing.ForwardRef` instance will be
    returned if some names in the string annotation have not been found. For additional
    information see :func:`typing.get_type_hints`.
    """
    if getattr(obj, "__no_type_check__", None):
        return {}
    if not hasattr(obj, "__annotations__"):
        return get_type_hints(
            obj, globalns=globalns, localns=localns, include_extras=include_extras
        )

    hints: dict[str, Union[type[Any], ForwardRef]] = {}
    annotations = getattr(obj, "__annotations__", {})
    for name, hint in annotations.items():
        obj.__annotations__ = {name: hint}
        try:
            resolved_hints = get_type_hints(
                obj, globalns=globalns, localns=localns, include_extras=include_extras
            )
            hints[name] = resolved_hints[name]
        except NameError as error:
            if isinstance(hint, str):
                # This conversion could be probably skipped in Python versions containing
                # the fix applied in bpo-41370. Check:
                # https://github.com/python/cpython/commit/b465b606049f6f7dd0711cb031fdaa251818741a#diff-ddb987fca5f5df0c9a2f5521ed687919d70bb3d64eaeb8021f98833a2a716887R344
                hints[name] = ForwardRef(hint)
            elif isinstance(hint, (ForwardRef, _typing.ForwardRef)):
                hints[name] = hint
            else:
                raise error

    obj.__annotations__ = annotations

    return hints


def eval_forward_ref(
    ref: Union[str, ForwardRef],
    globalns: Optional[dict[str, Any]] = None,
    localns: Optional[dict[str, Any]] = None,
    *,
    include_extras: bool = False,
) -> SolvedTypeAnnotation:
    """Resolve forward references in type annotations.

    Arguments:
        globalns: globals ``dict`` used in the evaluation of the annotations.
        localns: locals ``dict`` used in the evaluation of the annotations.

    Keyword Arguments:
        include_extras: if ``True``, ``Annotated`` hints will preserve the annotation.

    Examples:
        >>> print("Result:", eval_forward_ref("dict[str, tuple[int, float]]"))
        Result: dict[str, tuple[int, float]]

    """

    def f() -> None: ...

    f.__annotations__ = {"return": ForwardRef(ref) if isinstance(ref, str) else ref}

    safe_localns = {**localns} if localns else {}
    safe_localns.setdefault("typing", _FORWARD_REF_TYPING_NS)
    safe_localns.setdefault("NoneType", type(None))

    if globalns is None:
        # No caller-provided namespace: resolve against the explicit one, which spells
        # the container protocols as their 'collections.abc' objects and the deprecated
        # 'typing' aliases as the builtin generics. A caller-provided 'globalns' is left
        # untouched.
        globalns = _forward_ref_namespace()

    actual_type = get_type_hints(f, globalns, safe_localns, include_extras=include_extras)["return"]
    assert not isinstance(actual_type, ForwardRef)

    return actual_type


def _collapse_type_args(*args: Any) -> tuple[bool, tuple]:
    if args and all(args[0] == a for a in args[1:]):
        return (True, args)
    else:
        return (False, args)


@final
@_dataclasses.dataclass
class CallableKwargsInfo:
    data: dict[str, Any]


def infer_type(
    value: Any, *, annotate_callable_kwargs: bool = False, none_as_type: bool = True
) -> TypeAnnotation:
    """Generate a typing definition from a value.

    Keyword Arguments:
        annotate_callable_kwargs: if ``True``, ``Callable``s will be returned as
            a ``Annotated[Callable, CallableKwargsInfo]`` hint, where :class:`CallableKwargsInfo`
            contains the inferred typings for the keyword arguments, if any.
        none_as_type:  if ``True``, ``None`` hints will be transformed to ``type(None)``.

    Examples:
        >>> infer_type(3)
        <class 'int'>

        >>> infer_type((3, "four"))
        tuple[int, str]

        >>> infer_type((3, 4))
        tuple[int, int]

        >>> infer_type(frozenset([1, 2, 3]))
        frozenset[int]

        >>> infer_type({"a": 0, "b": 1})
        dict[str, int]

        >>> infer_type({"a": 0, "b": "B"})
        dict[str, ...Any]

        >>> print("Result:", infer_type(lambda a, b: a + b))
        Result: ...Callable[[...Any, ...Any], ...Any]

        # Note that some patch versions of cpython3.9 show weird behaviors
        >>> def f(a: int, b) -> int: ...
        >>> print("Result:", infer_type(f))
        Result: ...Callable[[...int..., ...Any], int]

        >>> def f(a: int, b) -> int: ...
        >>> print("Result:", infer_type(f))
        Result: ...Callable[..., int]

        >>> print("Result:", infer_type(dict[int, Union[int, float]]))
        Result: ...ict[int, ...int...float...]

    For advanced cases, using :func:`functools.singledispatch` with custom hooks
    is a simple way to extend and customize this base implementation.

    Example:
        >>> import functools, numbers
        >>> extended_infer_type = functools.singledispatch(infer_type)
        >>> @extended_infer_type.register(int)
        ... @extended_infer_type.register(float)
        ... @extended_infer_type.register(complex)
        ... def _infer_type_number(value, *, annotate_callable_kwargs: bool = False):
        ...     return numbers.Number
        >>> extended_infer_type(3.4)
        <class 'numbers.Number'>
        >>> infer_type(3.4)
        <class 'float'>

    """
    _infer = _functools.partial(infer_type, annotate_callable_kwargs=annotate_callable_kwargs)

    # Annotations are returned unchanged rather than described: PEP 585 generics,
    # bare special forms, parametrized 'typing' aliases ('Optional[int]'), PEP 604
    # unions and PEP 695 aliases. Otherwise the 'type(value)' fallback below reports
    # the CPython-internal class implementing them ('typing._GenericAlias', ...).
    if (
        isinstance(value, (StdGenericAliasType, _TypingSpecialFormType))
        or is_type_alias(value)
        or get_origin(value) is not None
    ):
        return value

    # note: identity check instead of `in`, which would use `__eq__` and fail
    # for values with non-boolean equality (e.g. NumPy arrays)
    if value is None or value is type(None):
        return type(None) if none_as_type else None

    if isinstance(value, type):
        return type[value]

    if isinstance(value, tuple) and not isinstance(value, TypedNamedTupleABC):
        # Special case for tuples, which can have multiple types.
        # We should not confuse them with namedtuples, which are
        # treated as normal classes.
        _, args = _collapse_type_args(*(_infer(item) for item in value))
        if args:
            return StdGenericAliasType(tuple, args)
        else:
            return StdGenericAliasType(tuple, (Any, ...))

    if isinstance(value, (list, set, frozenset)):
        t: Union[type[list], type[set], type[frozenset]] = type(value)
        unique_type, args = _collapse_type_args(*(_infer(item) for item in value))
        return StdGenericAliasType(t, args[0] if unique_type else Any)

    if isinstance(value, dict):
        unique_key_type, keys = _collapse_type_args(*(_infer(key) for key in value.keys()))
        unique_value_type, values = _collapse_type_args(*(_infer(v) for v in value.values()))
        kt = keys[0] if unique_key_type else Any
        vt = values[0] if unique_value_type else Any
        return StdGenericAliasType(dict, (kt, vt))

    if isinstance(value, _types.FunctionType):
        try:
            annotations = get_type_hints(value)
            return_type = annotations.get("return", Any)

            sig = _inspect.signature(value)
            arg_types: list = []
            kwonly_arg_types: dict[str, Any] = {}
            for p in sig.parameters.values():
                if p.kind in (
                    _inspect.Parameter.POSITIONAL_ONLY,
                    _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    arg_types.append(annotations.get(p.name, None) or Any)
                elif p.kind == _inspect.Parameter.KEYWORD_ONLY:
                    kwonly_arg_types[p.name] = annotations.get(p.name, None) or Any
                elif p.kind in (
                    _inspect.Parameter.VAR_POSITIONAL,
                    _inspect.Parameter.VAR_KEYWORD,
                ):
                    raise TypeError("Variadic callables are not supported")

            result: Any = Callable[arg_types, return_type]
            if annotate_callable_kwargs:
                result = Annotated[result, CallableKwargsInfo(kwonly_arg_types)]
            return result
        except Exception:
            return Callable

    return type(value)


# TODO(egparedes): traversing a typing definition is an operation needed in several places
#   but it currently requires custom and cumbersome code due to the messy implementation details
#   in the standard library. Ideally, this code could be replaced by translating it once to a
#   custom "typing tree" data structure which could be then traversed in a generic way.
#
