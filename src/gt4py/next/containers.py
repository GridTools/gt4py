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
PythonContainerValue: TypeAlias = core_defs.Scalar | common.Field
NestedTupleConstructor: TypeAlias = Callable[[Any], NestedTuple[PythonContainerValue]]

PYTHON_CONTAINER_TYPES: Final[tuple[type, ...]] = typing.cast(
    tuple[type, ...],
    PythonContainer.__args__,  # type: ignore[attr-defined]
)

VALUE_TYPES: Final[tuple[type, ...]] = tuple([*core_defs.SCALAR_TYPES, common.Field])


@functools.cache
def container_keys(container_type: type[PythonContainer]) -> tuple[str, ...]:
    """Get the keys of the container type."""
    assert issubclass(container_type, PythonContainer)

    if issubclass(container_type, xtyping.TypedNamedTupleABC):
        return container_type._fields
    if issubclass(container_type, xtyping.DataclassABC):
        return tuple(container_type.__dataclass_fields__.keys())

    return ()


@functools.cache
def make_container_extractor(container_type: type[PythonContainer]) -> NestedTupleConstructor:
    """
    Create an extractor function for the given container type.

    The returned function takes an instance of the specified container type
    and extracts its values into a nested tuple according to the provided
    type specification.
    """
    assert issubclass(container_type, PythonContainer)

    def make_extractor_expr(in_arg: str, container_type: type[PythonContainer]):
        children_hints = {
            key: xtyping.get_origin(value) or value
            for key, value in xtyping.get_type_hints(container_type).items()
        }
        args = {
            key: make_extractor_expr(f"{in_arg}.{key}", children_hints[key])
            if issubclass(children_hints[key], PYTHON_CONTAINER_TYPES)
            else f"{in_arg}.{key}"
            for key in container_keys(container_type)
        }
        if issubclass(container_type, tuple) and all(
            value.endswith(f".{key}") for key, value in args.items()
        ):
            # Fast path: input argument is already a pure nested tuple
            return f"{in_arg}"
        else:
            return f"({', '.join(args.values())})"

    extractor_func_src = f"lambda x: {make_extractor_expr('x', container_type)}"
    return eval(extractor_func_src)


PythonContainerT = TypeVar("PythonContainerT", bound=PythonContainer)


def _get_args_info(container_type: type[PythonContainer]) -> tuple[int, list[str]]:
    assert issubclass(container_type, PythonContainer)

    # Use a constructor signature without variadic parameters
    for method_name in ("__new__", "__init__"):
        params = inspect.signature(getattr(container_type, method_name)).parameters
        if not (
            {p.kind for p in params.values()}
            & {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        ):
            break
    else:
        raise TypeError(
            f"Type {container_type} with variadic parameters in constructor are not supported"
        )
    if len(params) == 1:
        raise TypeError(f"Type {container_type} has no parameters in constructor")

    params = {k: v for k, v in [*params.items()][1:]}  # drop 'self' / 'cls' parameter
    args_count = sum(
        p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        for p in params.values()
    )
    kwargs_keys = [
        key for key, param in params.items() if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    assert args_count + len(kwargs_keys) == len(params), "Variadic parameters are not supported"

    return args_count, kwargs_keys


@functools.cache
def make_container_constructor(
    container_type: type[PythonContainerT],
) -> Callable[[ts.NestedTuple], PythonContainerT]:
    """
    Create a constructor function for the given container type.

    The returned function takes a nested tuple of values and constructs
    an instance of the specified container type.
    """
    assert issubclass(container_type, PythonContainer)

    global_ns = {}

    def make_constructor_expr(
        in_arg: str, container_type: type[PythonContainer], container_type_alias: str
    ):
        # We reached the leaf of the nested container construction so we just return the argument as is
        if container_type in VALUE_TYPES or xtyping.get_origin(container_type) in VALUE_TYPES:
            return in_arg

        assert issubclass(container_type, PythonContainer)

        # Store the container type alias in the global namespace for eval
        global_ns[container_type_alias] = container_type

        # Recursively generate the constructor call
        children_hints = {
            key: xtyping.get_origin(value) or value
            for key, value in xtyping.get_type_hints(container_type).items()
        }
        args_count, kwargs_keys = _get_args_info(container_type)

        if {children_hints[key] for key in container_keys(container_type)} <= {*VALUE_TYPES}:
            # Fast path: all children are values, so we can just use argument unpacking
            call_args = (
                f"*{in_arg}[:{args_count}], {', '.join(f'{k}={in_arg}[{args_count + i}]' for i, k in enumerate(kwargs_keys))}"
                if kwargs_keys
                else f"*{in_arg}"
            )
        else:
            # General path: we need to recursively construct children containers
            call_args_items = []
            for i, key in enumerate(container_keys(container_type)):
                if children_hints[key] in VALUE_TYPES:
                    call_args_items.append(f"{in_arg}[{i}]")
                else:
                    *path_keys, _ = container_type_alias[2:].split("__")
                    children_container = children_hints[key].__name__
                    children_container_alias = (
                        f"{'__'.join(path_keys)}__{key}__{children_container}"
                    )
                    value = make_constructor_expr(
                        f"{in_arg}[{i}]", children_hints[key], children_container_alias
                    )
                    call_args_items.append(f"{key}={value}")
            call_args = ", ".join(call_args_items)

        return f"{container_type_alias}({call_args})"

    constructor_func_src = (
        f"lambda x: {make_constructor_expr('x', container_type, f'{container_type.__name__}')}"
    )

    return eval(constructor_func_src, global_ns)
