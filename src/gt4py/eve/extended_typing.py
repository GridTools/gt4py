# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Typing definitions working across different Python versions (via `typing_extensions`).

Definitions in 'typing_extensions' take priority over those in 'typing'.
"""

from __future__ import annotations

# ruff: noqa: F401, F405
import abc as _abc
import array as _array
import builtins as _builtins
import collections.abc as _collections_abc
import dataclasses as _dataclasses
import functools as _functools
import inspect as _inspect
import mmap as _mmap
import pickle as _pickle
import sys as _sys
import types as _types
import typing as _typing
from typing import *  # noqa: F403 [undefined-local-with-import-star]
from typing import overload

import numpy.typing as npt
import typing_extensions as _typing_extensions
from typing_extensions import *  # type: ignore[assignment,no-redef]  # noqa: F403 [undefined-local-with-import-star]


# Re-export the standard collection types under the names the star imports above bind to
# their deprecated 'typing' counterparts, so that e.g. 'Sequence' is
# 'collections.abc.Sequence' rather than 'typing.Sequence'. This block must stay *below*
# the star imports, since it deliberately rebinds names those imports also define; the
# 'isort: split' marker keeps the import sorter from hoisting it. The builtin generics
# are deliberately not re-exported under their old 'typing' spellings; see
# '_DEPRECATED_TYPING_ALIASES' below.
# isort: split
from collections import (
    ChainMap as ChainMap,
    Counter as Counter,
    OrderedDict as OrderedDict,
    defaultdict as defaultdict,
    deque as deque,
)
from collections.abc import (
    AsyncGenerator as AsyncGenerator,
    AsyncIterable as AsyncIterable,
    AsyncIterator as AsyncIterator,
    Awaitable as Awaitable,
    ByteString as ByteString,
    Callable as Callable,
    Collection as Collection,
    Container as Container,
    Coroutine as Coroutine,
    Generator as Generator,
    ItemsView as ItemsView,
    Iterable as Iterable,
    Iterator as Iterator,
    KeysView as KeysView,
    Mapping as Mapping,
    MappingView as MappingView,
    MutableMapping as MutableMapping,
    MutableSequence as MutableSequence,
    MutableSet as MutableSet,
    Reversible as Reversible,
    Sequence as Sequence,
    Set as AbstractSet,
    ValuesView as ValuesView,
)
from contextlib import (
    AbstractAsyncContextManager as AsyncContextManager,
    AbstractContextManager as ContextManager,
)
from re import Match as Match, Pattern as Pattern


# The 'typing' aliases of the builtin generics are deprecated since PEP 585 and are no
# longer re-exported: use the builtin spelling instead. They are not simply absent from
# this module -- the star imports above bind them, and '__getattr__' below would happily
# forward them to 'typing' -- so they are dropped from the namespace here and rejected
# explicitly. Without this, removing them would silently downgrade every use site from
# the builtin to the deprecated 'typing' object. Note that this is a runtime guarantee
# only: a type checker still resolves the names through the star imports.
_DEPRECATED_TYPING_ALIASES: Final[Mapping[str, str]] = {
    "Dict": "dict",
    "FrozenSet": "frozenset",
    "List": "list",
    "Set": "set",
    "Tuple": "tuple",
    "Type": "type",
}

for _alias in _DEPRECATED_TYPING_ALIASES:
    globals().pop(_alias, None)
del _alias

# The names are still valid in user-written annotations, and resolving a forward
# reference through this module has always normalized them to the builtin generic
# ('typing.List[int]' -> 'list[int]'). Keep that mapping available to the forward-ref
# machinery below, which would otherwise either raise or hand back the deprecated
# 'typing' object.
_DEPRECATED_ALIAS_REPLACEMENTS: Final[Mapping[str, Any]] = {
    name: getattr(_builtins, replacement)
    for name, replacement in _DEPRECATED_TYPING_ALIASES.items()
}


class _ForwardRefTypingNamespace:
    """Namespace bound to the name 'typing' while evaluating forward references.

    Annotations resolve through this module, so that 'typing_extensions' definitions
    take priority and the standard collection types are used. The deprecated builtin
    aliases are not re-exported here, but they stay valid in user-written annotations,
    so 'typing.List[int]' resolves to 'list[int]' rather than raising.
    """

    def __getattr__(self, name: str) -> Any:
        if (replacement := _DEPRECATED_ALIAS_REPLACEMENTS.get(name)) is not None:
            return replacement
        return getattr(_sys.modules[__name__], name)


_FORWARD_REF_TYPING_NS: Final = _ForwardRefTypingNamespace()


# These fallbacks are useful for public symbols not exported by default.
# Again, definitions in 'typing_extensions' take priority over those in 'typing'
def __getattr__(name: str) -> Any:
    import sys

    import typing_extensions

    if (replacement := _DEPRECATED_TYPING_ALIASES.get(name)) is not None:
        raise AttributeError(
            f"'{name}' is a deprecated 'typing' alias (PEP 585) and is not exported by"
            f" '{__name__}'. Use '{replacement}' instead."
        )

    result = SENTINEL = object()
    if not (name.startswith("__") and name.endswith("__")):
        result = getattr(typing_extensions, name, SENTINEL)
        if result is SENTINEL:
            import typing

            result = getattr(typing, name, SENTINEL)

    if result is SENTINEL:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    setattr(sys.modules[__name__], name, result)  # cache result

    return result


def __dir__() -> list[str]:
    if not hasattr(self_func := (globals()["__dir__"]), "__cached_dir"):
        import typing

        import typing_extensions

        # Everything reachable through '__getattr__' plus this module's own definitions,
        # minus the aliases '__getattr__' explicitly rejects.
        names = {*typing.__dir__(), *typing_extensions.__dir__(), *globals()}
        self_func.__cached_dir = sorted(names - _DEPRECATED_TYPING_ALIASES.keys())

    return self_func.__cached_dir


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
# Not supported: `ClassVar` behind an alias, and aliases used as runtime values (base
# class, constructor, `isinstance()` argument) -- the reason ruff's `UP040` stays off.

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


def is_Any(obj: Any) -> bool:
    """Check if an object is the ``Any`` special form."""
    # 'typing_extensions' re-exports 'typing.Any' on every supported version, so the
    # two implementations that used to exist below the 3.11 floor are now one object.
    return obj is _typing.Any


def has_type_parameters(cls: type[Any]) -> bool:
    """Return ``True`` if obj is a generic class with type parameters."""
    return issubclass(cls, Generic) and len(getattr(cls, "__parameters__", [])) > 0  # type: ignore[arg-type]  # Generic not considered as a class


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
    # unresolved one would silently yield an empty tuple. Nested aliases are covered
    # for free, since the generic branches recurse through this same function.
    type_annotation = eval_type_alias(type_annotation)

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

    if origin_type in [Literal, Union, _types.UnionType]:
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


is_protocol = _typing_extensions.is_protocol


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
        # Without an explicit 'globalns' the reference is resolved in this module's
        # namespace, which used to spell the deprecated aliases as the builtin generics.
        # They are no longer defined here, so re-add them for this evaluation only; a
        # caller-provided 'globalns' is left untouched, exactly as before.
        globalns = {**globals(), **_DEPRECATED_ALIAS_REPLACEMENTS}

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

    if isinstance(value, (StdGenericAliasType, _TypingSpecialFormType)):
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
