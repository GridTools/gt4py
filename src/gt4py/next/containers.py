# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

from gt4py.next import common


if TYPE_CHECKING:
    from gt4py.next.type_system import type_specifications as ts

# functools.singledispatch would be nice, but would have to be on the metaclass level
_registry: list[tuple[Callable[[type], bool], Callable[[type], ts.TupleType] | ts.TupleType]] = []


# for users
def register(
    predicate: Callable[[type], bool], func_or_type: Callable[[type], ts.TupleType] | ts.TupleType
) -> None:
    _registry.append((predicate, func_or_type))


# only internally
def get_constructor_type(cls: type) -> ts.TupleType:
    for predicate, func_or_type in _registry:
        if predicate(cls):
            if callable(func_or_type):
                return func_or_type(cls)
            return func_or_type
    raise KeyError(f"get_constructor_type not implemented for {cls}.")


# TODO customize
def flatten(arg) -> tuple | common.Field:
    return (
        arg
        if isinstance(arg, common.Field) or not dataclasses.is_dataclass(arg)
        else tuple(getattr(arg, f.name) for f in dataclasses.fields(arg))
    )


# -- Field containers --

import functools
import inspect
import typing
from collections.abc import Callable
from typing import Any, Final, TypeAlias, TypeVar

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import NestedTuple


if TYPE_CHECKING:
    from gt4py.next.type_system import type_specifications as ts


PythonContainer: TypeAlias = xtyping.TypedNamedTupleABC | xtyping.DataclassABC
PYTHON_CONTAINER_TYPES: Final[tuple[type, ...]] = typing.cast(
    tuple[type, ...],
    PythonContainer.__args__,  # type: ignore[attr-defined]
)

NestedTupleConstructor: TypeAlias = Callable[[Any], NestedTuple[core_defs.Scalar | common.Field]]


@functools.cache
def container_keys(container_type: type[PythonContainer]) -> tuple[str, ...]:
    assert issubclass(container_type, PYTHON_CONTAINER_TYPES)

    if issubclass(container_type, xtyping.TypedNamedTupleABC):
        return container_type._fields
    if issubclass(container_type, xtyping.DataclassABC):
        return tuple(container_type.__dataclass_fields__.keys())

    return ()


@functools.cache
def make_container_extractor(
    container_type: type[PythonContainer],
    type_spec: ts.TypeSpec,
) -> NestedTupleConstructor:
    assert issubclass(container_type, PythonContainer)
    assert isinstance(type_spec, ts.NamedTupleType)

    def make_extractor_expr(
        in_arg: str,
        container_type: type[PythonContainer],
        type_spec: ts.NamedTupleType,
    ):
        children_hints = xtyping.get_type_hints(container_type)
        args = {
            key: make_extractor_expr(f"{in_arg}.{key}", children_hints[key], spec)
            if isinstance(spec, ts.NamedTupleType)
            else f"{in_arg}.{key}"
            for key, spec in zip(type_spec.keys, type_spec.types)
        }
        if isinstance(container_type, tuple) and all(
            value.endswith(f".{key}") for key, value in args.items()
        ):
            return "{in_arg}"  # Fast path: input argument is already a pure nested tuple
        else:
            return f"({', '.join(args)})"

    return eval(f"lambda x: {make_extractor_expr('x', container_type, type_spec)}")


PythonContainerT = TypeVar("PythonContainerT", bound=PythonContainer)


@functools.cache
def make_container_constructor(
    container_type: type[PythonContainerT],
) -> Callable[[ts.NestedTuple], PythonContainerT]:
    assert isinstance(container_type, type)
    global_ns = {}

    def make_constructor_expr(
        in_arg: str, container_type: type[PythonContainer], container_type_key: str
    ):
        if isinstance(container_type, core_defs.SCALAR_TYPES):
            return in_arg
        
        assert isinstance(container_type, PythonContainer)
        global_ns[container_type_key] = container_type
        params = inspect.signature(container_type.__init__).parameters
        args_count = sum(
            p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            for p in params.values()
        )
        kwargs_keys = [key for key, param in params.items() if param.kind == inspect.Parameter.KEYWORD_ONLY]
        assert args_count + len(kwargs_keys) == len(params), "Variadic parameters are not supported"

        call_args = (
            f"*{in_arg}[:args_count], {', '.join(f'{k}={in_arg}[{args_count + i}]' for i, k in enumerate(kwargs_keys))}"
            if kwargs_keys
            else f"*{in_arg}"
        )


        children_hints = xtyping.get_type_hints(container_type)
        call_args = {
            key: make_constructor_expr(
                f"{in_arg}.{key}", children_hints[key], f"{container_type_key}_{key}"
            )
            for key in container_keys(container_type)
        }



        return f"{container_type_key}({call_args})"

    return eval(
        f"lambda x: {make_constructor_expr('x', container_type, 'container_type')}", global_ns
    )
