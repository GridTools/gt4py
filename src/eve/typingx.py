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

"""Python version independent typings."""


# flake8: noqa

from __future__ import annotations

import dataclasses
import functools
import inspect
import sys
import types
import typing
from typing import *
from typing import IO, BinaryIO, TextIO


AnyCallable = Callable[..., Any]
AnyNoneCallable = Callable[..., None]
AnyNoArgCallable = Callable[[], Any]


T = TypeVar("T")
V = TypeVar("V")
T_contra = TypeVar("T_contra", contravariant=True)
V_co = TypeVar("V_co", covariant=True)

ZeroOrMore = Union[None, T, Tuple[T, ...]]
OneOrMore = Union[T, Tuple[T, ...]]


class NonDataDescriptor(Protocol[T_contra, V_co]):
    @overload
    def __get__(self, _instance: None, _owner_type: Type[T_contra]) -> NonDataDescriptor:
        ...

    @overload
    def __get__(self, _instance: T_contra, _owner_type: Optional[Type[T_contra]] = None) -> V_co:
        ...


class DataDescriptor(NonDataDescriptor[T_contra, V], Protocol):
    def __set__(self, _instance: T, _value: V) -> None:
        ...


RootValidatorValuesType = Dict[str, Any]
RootValidatorType = Callable[[Type, RootValidatorValuesType], RootValidatorValuesType]


def canonicalize_forward_ref(type_hint: Union[str, Type, ForwardRef]) -> Union[Type, ForwardRef]:
    """Return the original type hint or a ``ForwardRef``s without nested ``ForwardRef``s.

    Examples:
        >>> import typing
        >>> canonicalize_forward_ref(typing.List[typing.ForwardRef('my_type')])
        ForwardRef('List[my_type]')
        >>> canonicalize_forward_ref("List[typing.ForwardRef('my_type')]")
        ForwardRef('List[my_type]')
        >>> canonicalize_forward_ref(typing.ForwardRef("typing.List[typing.ForwardRef('my_type')]"))
        ForwardRef('typing.List[my_type]')

    """
    if isinstance(type_hint, ForwardRef):
        return canonicalize_forward_ref(type_hint.__forward_arg__)
    if isinstance(type_hint, str):
        new_hint = type_hint
        for pattern in ("typing.ForwardRef(", "ForwardRef("):
            offset = len(pattern)
            while pattern in new_hint:
                start = new_hint.find(pattern)
                nested = 0
                for i, c in enumerate(new_hint[start + offset :]):
                    if c == ")":
                        if nested == 0:
                            end = start + offset + i
                            break
                        else:
                            nested -= 1
                    elif c == "(":
                        nested += 1
                content = (new_hint[start + offset : end]).strip(" \n\r\t\"'")
                new_hint = new_hint[:start] + content + new_hint[end + 1 :]

        return ForwardRef(new_hint)

    if not (type_args := typing.get_args(type_hint)):
        return type_hint

    assert isinstance(type_hint, typing._GenericAlias)  # type: ignore[attr-defined]  # typing._GenericAlias is not public
    new_type_args = tuple(canonicalize_forward_ref(t) for t in type_args)
    if not any(isinstance(t, ForwardRef) for t in new_type_args):
        return type_hint

    str_args = []
    for t in new_type_args:
        if isinstance(t, str):
            str_args.append(t)
        elif isinstance(t, ForwardRef):
            str_args.append(t.__forward_arg__)
        else:
            str_args.append(repr(t))

    return ForwardRef(f"{type_hint._name}[{','.join(str_args)}]")


def get_canonical_type_hints(cls: Type) -> Dict[str, Union[Type, ForwardRef]]:
    """Extract class type annotations returning forward references for partially undefined types.

    The canonicalization consists in returning either a fully-specified type
    annotation or a :class:`typing.ForwarRef` instance with the type annotation
    for not fully-specified types.

    Based on :func:`typing.get_type_hints` implementation.
    """
    hints: Dict[str, Union[Type, ForwardRef]] = {}

    for base in reversed(cls.__mro__):
        base_globals = sys.modules[base.__module__].__dict__
        ann = base.__dict__.get("__annotations__", {})
        for name, value in ann.items():
            if value is None:
                value = type(None)
            elif isinstance(value, str):
                value = ForwardRef(value, is_argument=False)

            try:
                value = typing._eval_type(value, base_globals, None)  # type: ignore[attr-defined]  # typing._eval_type is not public
            except NameError as e:
                if "ForwardRef(" not in repr(value):
                    raise e
            if not isinstance(value, ForwardRef) and "ForwardRef(" in repr(value):
                value = canonicalize_forward_ref(value)

            hints[name] = value

    return hints


@typing.overload
def resolve_type(
    type_hint: Any,
    global_ns: Optional[Dict[str, Any]] = None,
    local_ns: Optional[Dict[str, Any]] = None,
    *,
    allow_partial: Literal[False],
) -> Type:
    ...


@typing.overload
def resolve_type(
    type_hint: Any,
    global_ns: Optional[Dict[str, Any]] = None,
    local_ns: Optional[Dict[str, Any]] = None,
    *,
    allow_partial: Literal[True],
) -> Union[Type, ForwardRef]:
    ...


def resolve_type(
    type_hint: Any,
    global_ns: Optional[Dict[str, Any]] = None,
    local_ns: Optional[Dict[str, Any]] = None,
    *,
    allow_partial: bool = False,
) -> Union[Type, ForwardRef]:
    """Resolve forward references in type annotations.

    Arguments:
        global_ns: globals dict used in the evaluation of the annotations.
        local_ns: locals dict used in the evaluation of the annotations.

    Keyword Arguments:
        allow_partial: if ``True``, the resolution is allowed to fail and
            a :class:`typing.ForwardRef` will be returned.

    Examples:
        >>> import typing
        >>> resolve_type(
        ...     typing.Dict[typing.ForwardRef('str'), 'typing.Tuple["int", typing.ForwardRef("float")]']
        ... )
        typing.Dict[str, typing.Tuple[int, float]]

    """
    actual_type = ForwardRef(type_hint) if isinstance(type_hint, str) else type_hint
    while "ForwardRef(" in repr(actual_type):
        try:
            if local_ns:
                safe_local_ns = {**local_ns}
                safe_local_ns.setdefault("typing", sys.modules["typing"])
                safe_local_ns.setdefault("NoneType", type(None))
            else:
                safe_local_ns = {"typing": sys.modules["typing"], "NoneType": type(None)}
            actual_type = typing._eval_type(  # type: ignore[attr-defined]  # typing._eval_type is not visible for mypy
                actual_type,
                global_ns,
                safe_local_ns,
            )
        except Exception as e:
            if allow_partial:
                actual_type = canonicalize_forward_ref(actual_type)
                break
            else:
                raise e

    return actual_type


def _collapsable_type_args(*args: Any) -> tuple[bool, tuple]:
    if args and all(args[0] == a for a in args):
        return (True, args)
    else:
        return (False, args)


@dataclasses.dataclass
class CallableKwargsInfo:
    data: dict[str, Any]


@functools.singledispatch
def get_typing(value: Any, *, annotate_callable_kwargs: bool = False) -> Any:
    """Generate a typing definition from a value.

    The implementation uses :func:`functools.singledispatch`. Customized or
    more detailed annotations can be generated by registering alternative
    implementations.

    Examples:
        >>> get_typing(3)
        int

        >>> get_typing((3, "four"))
        tuple[int, str]

        >>> get_typing((3, 4))
        tuple[int, ...]

        >>> get_typing(frozenset([1, 2, 3]))
        frozenset[int]

        >>> get_typing({'a': 0, 'b': 1})
        dict[str, int]

        >>> get_typing({'a': 0, 'b': 'B'})
        dict[str, typing.Any]

        >>> get_typing(lambda a, b: a + b)
        collections.abc.Callable[[typing.Any, typing.Any], typing.Any]

        >>> def f(a: int, b) -> int: ...
        >>> get_typing(f)
        collections.abc.Callable[['int', typing.Any], 'int']

        >>> def f(a: int, b) -> int: ...
        >>> get_typing(f)
        collections.abc.Callable[..., 'int']

        >>> get_typing(Dict[int, Union[int, float]])
        dict[int, typing.Union[int, float]]

        >>> get_typing(Dict[int, Union[TypeVar("T", bound=int), float]])
        dict[int, typing.Union[int, float]]

        >>> get_typing(Dict[int, Union[int, float]], get_origins=False)
        typing.Dict[int, typing.Union[int, float]]

        >>> import numbers
        >>> @get_typing.register(int)
        >>> @get_typing.register(float)
        >>> @get_typing.register(complex)
        >>> def _get_typing_number(value, *, get_origins: bool = True):
        ...    return numbers.Number
        >>> get_typing(3.4)
        numbers.Number

    """
    recursive_get = functools.partial(get_typing, annotate_callable_kwargs=annotate_callable_kwargs)
    if isinstance(value, type):
        return value

    elif value in (None, types.NoneType):
        return None

    elif isinstance(value, tuple):
        unique_type, args = _collapsable_type_args(*(recursive_get(item) for item in value))
        if unique_type and len(args) > 1:
            return tuple[args[0], ...]
        elif args:
            return tuple[args]
        else:
            return tuple[Any, ...]

    elif isinstance(value, (list, set, frozenset)):
        t = type(value)
        unique_type, args = _collapsable_type_args(*(recursive_get(item) for item in value))
        return t[args[0]] if unique_type else t[Any]

    elif isinstance(value, dict):
        unique_key_type, keys = _collapsable_type_args(
            *(recursive_get(key) for key in value.keys())
        )
        unique_value_type, values = _collapsable_type_args(
            *(recursive_get(v) for v in value.values())
        )
        kt = keys[0] if unique_key_type else Any
        vt = values[0] if unique_value_type else Any
        return dict[kt, vt]

    elif isinstance(value, types.FunctionType):
        try:
            annotations = get_type_hints(value)
            return_type = annotations.get("return", Any)

            sig = inspect.signature(value)
            arg_types: Union[list, Ellipsis] = []
            kwonly_arg_types: dict[str, Any] = {}
            for p in sig.parameters.values():
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    arg_types.append(annotations.get(p.name, None) or Any)
                elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                    kwonly_arg_types.append(annotations.get(p.name, None) or Any)
                elif p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise TypeError(f"Variadic callables are not supported")

            result = Callable[arg_types, return_type]
            if annotate_callable_kwargs:
                result = Annotated[result, CallableKwargsInfo(kwonly_arg_types)]
            return result
        except:
            return Callable

    else:
        return type(value)
