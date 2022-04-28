# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

"""Typing definitions working across different Python versions (via `typing_extensions`)."""

from __future__ import annotations

import dataclasses as _dataclasses
import functools as _functools
import inspect as _inspect
import sys as _sys
import types as _types
import typing as _typing

# Definitions in 'typing_extensions' take priority over those in 'typing'
from typing import *  # noqa: F403

from typing_extensions import *  # type: ignore[misc]  # noqa: F403


if _sys.version_info >= (3, 9):
    # Standard library already supports PEP 585 (Type Hinting Generics In Standard Collections)
    from builtins import (  # type: ignore[misc]  # isort:skip
        tuple as Tuple,
        list as List,
        dict as Dict,
        set as Set,
        frozenset as FrozenSet,
        type as Type,
    )
    from collections import (  # isort:skip
        ChainMap as ChainMap,
        Counter as Counter,
        OrderedDict as OrderedDict,
        defaultdict as defaultdict,
        deque as deque,
    )
    from collections.abc import (  # isort:skip
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
    )
    from collections.abc import Set as AbstractSet  # isort:skip
    from collections.abc import ValuesView as ValuesView  # isort:skip
    from contextlib import (  # isort:skip
        AbstractAsyncContextManager as AsyncContextManager,
    )
    from contextlib import AbstractContextManager as ContextManager  # isort:skip
    from re import Match as Match, Pattern as Pattern  # isort:skip


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


# Common type aliases
_T_co = TypeVar("_T_co", covariant=True)

FrozenList: TypeAlias = Tuple[_T_co, ...]
NoArgsCallable = Callable[[], Any]


# Typing annotations
if _sys.version_info >= (3, 9):
    SolvedTypingAnnotation = Union[
        Type,
        _types.GenericAlias,
        _typing._BaseGenericAlias,  # type: ignore[name-defined]  # _BaseGenericAlias is private
        _typing._SpecialForm,
    ]
else:
    SolvedTypingAnnotation = Union[  # type: ignore[misc]  # mypy consider this assignment a redefinition
        Type,
        _typing._GenericAlias,  # type: ignore[attr-defined]  # _GenericAlias is private
        _typing._SpecialForm,
    ]

TypingAnnotation = Union[ForwardRef, SolvedTypingAnnotation]
SourceTypingAnnotation = Union[str, TypingAnnotation]

_TypingSpecialFormType = _typing._SpecialForm
_GenericAliasType: Final[Type] = (
    _types.GenericAlias if _sys.version_info >= (3, 9) else _typing._GenericAlias  # type: ignore[attr-defined]  # _GenericAlias is private
)


# TODO(egparedes): remove these types only needed from pydantic models
RootValidatorValuesType = Dict[str, Any]
RootValidatorType = Callable[[Type, RootValidatorValuesType], RootValidatorValuesType]


# Third party protocols
class DevToolsPrettyPrintable(Protocol):
    """Used by python-devtools (https://python-devtools.helpmanual.io/)."""

    def __pretty__(self, fmt: Callable[[Any], Any], **kwargs: Any) -> Generator[Any, None, None]:
        ...


# Extra functionality
def is_protocol(tp: type) -> bool:
    """Check if a type is a Protocol definition."""
    this_module = _sys.modules[is_protocol.__module__]
    return isinstance(tp, this_module._ProtocolMeta) and tp.__bases__[-1] is this_module.Protocol


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

    For each member type hint in the object a :class:`typing.ForwarRef` instance will be
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
            hints.update(resolved_hints)
        except NameError as error:
            if isinstance(hint, str):
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
) -> SolvedTypingAnnotation:
    """Resolve forward references in type annotations.

    Arguments:
        globalns: globals dict used in the evaluation of the annotations.
        localns: locals dict used in the evaluation of the annotations.

    Keyword Arguments:
        include_extras: if ``True``, ``Annotated`` hints will preserve the annotation.

    Examples:
        >>> from typing import Dict, Tuple
        >>> print("Result:", eval_forward_ref('Dict[str, Tuple[int, float]]'))
        Result: ...ict[str, ...uple[int, float]]

    """
    actual_type = ForwardRef(ref) if isinstance(ref, str) else ref

    def _f() -> None:
        pass

    _f.__annotations__ = {"ref": actual_type}

    if localns:
        safe_localns = {**localns}
        safe_localns.setdefault("typing", _sys.modules[__name__])
        safe_localns.setdefault("NoneType", type(None))
    else:
        safe_localns = {"typing": _sys.modules[__name__], "NoneType": type(None)}

    actual_type = get_type_hints(_f, globalns, safe_localns, include_extras=include_extras)["ref"]
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


def infer_type(  # noqa: C901  # function is complex but well organized in independent cases
    value: Any,
    *,
    annotate_callable_kwargs: bool = False,
    none_as_type: bool = True,
) -> TypingAnnotation:
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
        tuple[int, ...]

        >>> infer_type(frozenset([1, 2, 3]))
        frozenset[int]

        >>> infer_type({'a': 0, 'b': 1})
        dict[str, int]

        >>> infer_type({'a': 0, 'b': 'B'})
        dict[str, typing.Any]

        >>> print("Result:", infer_type(lambda a, b: a + b))
        Result: ...Callable[[typing.Any, typing.Any], typing.Any]

        >>> def f(a: int, b) -> int: ...
        >>> print("Result:", infer_type(f))
        Result: ...Callable[[int, typing.Any], int]

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
        ...    return numbers.Number
        >>> extended_infer_type(3.4)
        <class 'numbers.Number'>
        >>> infer_type(3.4)
        <class 'float'>

    """
    _reveal = _functools.partial(infer_type, annotate_callable_kwargs=annotate_callable_kwargs)

    if isinstance(value, (_GenericAliasType, _TypingSpecialFormType)):
        return value

    if value in (None, type(None)):
        return type(None) if none_as_type else None

    if isinstance(value, type):
        return Type[value]

    if isinstance(value, tuple):
        unique_type, args = _collapse_type_args(*(_reveal(item) for item in value))
        if unique_type and len(args) > 1:
            return _GenericAliasType(tuple, (args[0], ...))
        elif args:
            return _GenericAliasType(tuple, args)
        else:
            return _GenericAliasType(tuple, (Any, ...))

    if isinstance(value, (list, set, frozenset)):
        t: Union[Type[List], Type[Set], Type[FrozenSet]] = type(value)
        unique_type, args = _collapse_type_args(*(_reveal(item) for item in value))
        return _GenericAliasType(t, args[0] if unique_type else Any)

    if isinstance(value, dict):
        unique_key_type, keys = _collapse_type_args(*(_reveal(key) for key in value.keys()))
        unique_value_type, values = _collapse_type_args(*(_reveal(v) for v in value.values()))
        kt = keys[0] if unique_key_type else Any
        vt = values[0] if unique_value_type else Any
        return _GenericAliasType(dict, (kt, vt))

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
                elif p.kind in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD):
                    raise TypeError("Variadic callables are not supported")

            result: Any = Callable[arg_types, return_type]  # type: ignore[misc]  # explicitly build annotation at runtime
            if annotate_callable_kwargs:
                result = Annotated[result, CallableKwargsInfo(kwonly_arg_types)]
            return result
        except Exception:
            return Callable

    return type(value)
