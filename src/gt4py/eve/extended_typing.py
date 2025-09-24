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


if _sys.version_info >= (3, 9):
    # Standard library already supports PEP 585 (Type Hinting Generics In Standard Collections)
    from builtins import (  # type: ignore[assignment]
        dict as Dict,
        frozenset as FrozenSet,
        list as List,
        set as Set,
        tuple as Tuple,
        type as Type,
    )
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


# These fallbacks are useful for public symbols not exported by default.
# Again, definitions in 'typing_extensions' take priority over those in 'typing'
def __getattr__(name: str) -> Any:
    import sys

    import typing_extensions

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


def __dir__() -> List[str]:
    if not hasattr(self_func := (globals()["__dir__"]), "__cached_dir"):
        import typing

        import typing_extensions

        orig_dir = typing.__dir__()
        self_func.__cached_dir = [*orig_dir] + [
            name for name in typing_extensions.__dir__() if name not in orig_dir
        ]

    return self_func.__cached_dir


# -- Common type aliases --
NoArgsCallable = Callable[[], Any]

_A = TypeVar("_A", contravariant=True)
_R = TypeVar("_R", covariant=True)


class ArgsOnlyCallable(Protocol[_A, _R]):
    def __call__(self, *args: _A) -> _R: ...


_T_co = TypeVar("_T_co", covariant=True)
NestedSequence = Sequence[Union[_T_co, "NestedSequence[_T_co]"]]
NestedList = List[Union[_T_co, "NestedList[_T_co]"]]
NestedTuple = Tuple[Union[_T_co, "NestedTuple[_T_co]"], ...]

MaybeNested = Union[_T_co, NestedSequence[_T_co]]
MaybeNestedInSequence = Union[_T_co, NestedSequence[_T_co]]
MaybeNestedInList = Union[_T_co, NestedList[_T_co]]
MaybeNestedInTuple = Union[_T_co, NestedTuple[_T_co]]

# -- Typing annotations --
SolvedTypeAnnotation = Union[
    Type,
    _typing._SpecialForm,
    _types.GenericAlias,
    _typing._BaseGenericAlias,  # type: ignore[name-defined]  # _BaseGenericAlias is not exported in stub
]

TypeAnnotation = Union[ForwardRef, SolvedTypeAnnotation]
SourceTypeAnnotation = Union[str, TypeAnnotation]

StdGenericAliasType: Final[Type] = type(List[int])

if TYPE_CHECKING:
    StdGenericAlias: TypeAlias = _types.GenericAlias

_TypingSpecialFormType: Final[Type] = _typing._SpecialForm
_TypingGenericAliasType: Final[Type] = _typing._BaseGenericAlias  # type: ignore[attr-defined]  # _BaseGenericAlias / _GenericAlias are not exported in stub


# -- Standard Python protocols --
_C = TypeVar("_C")
_V = TypeVar("_V")


class NonDataDescriptor(Protocol[_C, _V]):
    """Typing protocol for non-data descriptor classes.

    See https://docs.python.org/3/howto/descriptor.html for further information.
    """

    @overload
    def __get__(
        self, _instance: Literal[None], _owner_type: Optional[Type[_C]] = None
    ) -> NonDataDescriptor[_C, _V]: ...

    @overload
    def __get__(self, _instance: _C, _owner_type: Optional[Type[_C]] = None) -> _V: ...

    def __get__(
        self, _instance: Optional[_C], _owner_type: Optional[Type[_C]] = None
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
    def __array_interface__(self) -> Dict[str, Any]: ...


class ArrayInterfaceTypedDict(TypedDict):
    shape: Tuple[int, ...]
    typestr: str
    descr: NotRequired[List[Tuple]]
    data: NotRequired[Tuple[int, bool]]
    strides: NotRequired[Optional[Tuple[int, ...]]]
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
    def __cuda_array_interface__(self) -> Dict[str, Any]: ...


class CUDAArrayInterfaceTypedDict(TypedDict):
    shape: Tuple[int, ...]
    typestr: str
    data: Tuple[int, bool]
    version: int
    strides: NotRequired[Optional[Tuple[int, ...]]]
    descr: NotRequired[List[Tuple]]
    mask: NotRequired[Optional["StrictCUDAArrayInterface"]]
    stream: NotRequired[Optional[int]]


class StrictCUDAArrayInterface(Protocol):
    @property
    def __cuda_array_interface__(self) -> CUDAArrayInterfaceTypedDict: ...


def supports_cuda_array_interface(value: Any) -> TypeGuard[CUDAArrayInterface]:
    """Check if the given value supports the CUDA Array Interface."""
    return hasattr(value, "__cuda_array_interface__")


DLPackDevice = Tuple[int, int]


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
_ArtefactTypes: tuple[type, ...] = (_types.GenericAlias,)

# `Any` is a class since Python 3.11
if isinstance(_typing.Any, type):  # Python >= 3.11
    _ArtefactTypes = (*_ArtefactTypes, _typing.Any)

# `Any` is a class since typing_extensions >= 4.4 and Python 3.11
if (typing_exts_any := getattr(_typing_extensions, "Any", None)) is not _typing.Any and isinstance(
    typing_exts_any, type
):
    _ArtefactTypes = (*_ArtefactTypes, typing_exts_any)


def is_actual_type(obj: Any) -> TypeGuard[Type]:
    """Check if an object has an actual type and instead of a typing artefact like ``GenericAlias`` or ``Any``.

    This is needed because since Python 3.9:
        ``isinstance(types.GenericAlias(), type) is True``
    and since Python 3.11:
        ``isinstance(typing.Any, type) is True``
    """
    return (
        isinstance(obj, type) and (obj not in _ArtefactTypes) and (type(obj) not in _ArtefactTypes)
    )


if hasattr(_typing_extensions, "Any") and _typing.Any is not _typing_extensions.Any:  # type: ignore[attr-defined] # _typing_extensions.Any only from >= 4.4
    # When using Python < 3.11 and typing_extensions >= 4.4 there are
    # two different implementations of `Any`

    def is_Any(obj: Any) -> bool:
        return obj is _typing.Any or obj is _typing_extensions.Any  # type: ignore[attr-defined] # _typing_extensions.Any only from >= 4.4

else:

    def is_Any(obj: Any) -> bool:
        return obj is _typing.Any


def has_type_parameters(cls: Type) -> bool:
    """Return ``True`` if obj is a generic class with type parameters."""
    return issubclass(cls, Generic) and len(getattr(cls, "__parameters__", [])) > 0  # type: ignore[arg-type]  # Generic not considered as a class


_T = TypeVar("_T")


def get_actual_type(obj: _T) -> Type[_T]:
    """Return type of an object (also working for GenericAlias instances which pretend to be an actual type)."""
    return StdGenericAliasType if isinstance(obj, StdGenericAliasType) else type(obj)


def is_type_with_custom_hash(type_: Type) -> bool:
    return type_.__hash__ not in (None, object.__hash__)


class HasCustomHash(Hashable):
    """ABC for types defining a custom hash function."""

    @classmethod
    def __subclasshook__(cls, candidate_cls: type) -> bool:
        return is_type_with_custom_hash(candidate_cls)


def is_value_hashable(obj: Any) -> TypeGuard[HasCustomHash]:
    return isinstance(obj, type) or obj is None or is_type_with_custom_hash(type(obj))


def is_value_hashable_typing(
    type_annotation: TypeAnnotation,
    *,
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check if a type annotation describes a type hashable by value."""
    if is_actual_type(type_annotation):
        assert not get_args(type_annotation)
        return (
            True
            if type_annotation in (type, type(None))
            else is_type_with_custom_hash(type_annotation)
        )

    if isinstance(type_annotation, TypeVar):
        if type_annotation.__bound__:
            return is_value_hashable_typing(type_annotation.__bound__)
        if type_annotation.__constraints__:
            return all(is_value_hashable_typing(c) for c in type_annotation.__constraints__)
        return False

    if isinstance(type_annotation, ForwardRef):
        return is_value_hashable_typing(
            eval_forward_ref(type_annotation, globalns=globalns, localns=localns)
        )

    if type_annotation is Any:
        return False

    # Generic types
    origin_type = get_origin(type_annotation)
    type_args = get_args(type_annotation)

    if origin_type is Literal:
        return True

    if origin_type is Union:
        return all(is_value_hashable_typing(t) for t in type_args)

    if isinstance(origin_type, type) and is_value_hashable_typing(origin_type):
        return all(is_value_hashable_typing(t) for t in type_args if t != Ellipsis)

    return type_annotation is None


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
                subclass.__dataclass_params__.frozen is not None  # type: ignore[attr-defined]  # subclass.__dataclass_params__ is ok after check
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
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
    include_extras: bool = False,
) -> Dict[str, Union[Type, ForwardRef]]:
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

    hints: Dict[str, Union[Type, ForwardRef]] = {}
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
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
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
        >>> from typing import Dict, Tuple
        >>> print("Result:", eval_forward_ref("Dict[str, Tuple[int, float]]"))
        Result: ...ict[str, ...uple[int, float]]

    """

    def f() -> None: ...

    f.__annotations__ = {"return": ForwardRef(ref) if isinstance(ref, str) else ref}

    safe_localns = {**localns} if localns else {}
    safe_localns.setdefault("typing", _sys.modules[__name__])
    safe_localns.setdefault("NoneType", type(None))

    actual_type = get_type_hints(f, globalns, safe_localns, include_extras=include_extras)["return"]
    assert not isinstance(actual_type, ForwardRef)

    return actual_type


def _collapse_type_args(*args: Any) -> Tuple[bool, Tuple]:
    if args and all(args[0] == a for a in args[1:]):
        return (True, args)
    else:
        return (False, args)


@final
@_dataclasses.dataclass
class CallableKwargsInfo:
    data: Dict[str, Any]


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

        >>> print("Result:", infer_type(Dict[int, Union[int, float]]))
        Result: ...ict[int, typing.Union[int, float]]

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

    if value in (None, type(None)):
        return type(None) if none_as_type else None

    if isinstance(value, type):
        return Type[value]

    if isinstance(value, tuple):
        _, args = _collapse_type_args(*(_infer(item) for item in value))
        if args:
            return StdGenericAliasType(tuple, args)
        else:
            return StdGenericAliasType(tuple, (Any, ...))

    if isinstance(value, (list, set, frozenset)):
        t: Union[Type[List], Type[Set], Type[FrozenSet]] = type(value)
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
            arg_types: List = []
            kwonly_arg_types: Dict[str, Any] = {}
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
