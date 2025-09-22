# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import inspect
import pkgutil
import typing
from collections.abc import Callable
from typing import Any, Final, TypeAlias, TypeVar

from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts


NUMERIC_VALUE_TYPES: Final[tuple[type, ...]] = typing.cast(
    tuple[type, ...],
    common.NumericValue.__args__,
)


class PyContainerDataclassABC(xtyping.DataclassABC):
    """ABC Check if a dataclass respects current constraints for dataclasses."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if dataclasses.is_dataclass(subclass) and not subclass.__module__.startswith("gt4py."):
            fields = dataclasses.fields(subclass)
            return len(fields) > 0 and all(
                f.init is True
                and f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
                and f._field_type is dataclasses._FIELD  # type: ignore[attr-defined]  # private member
                for f in fields
            )

        return False


PyContainer: TypeAlias = xtyping.TypedNamedTupleABC | PyContainerDataclassABC
PyContainerT = TypeVar("PyContainerT", bound=PyContainer)

PyContainerExtractor: TypeAlias = Callable[[Any], NestedTuple[common.NumericValue]]

PyContainerConstructor: TypeAlias = Callable[[NestedTuple[common.NumericValue]], PyContainerT]


PY_CONTAINER_TYPES: Final[tuple[type, ...]] = typing.cast(
    tuple[type, ...],
    PyContainer.__args__,
)


@functools.cache
def keys(container_type: type[PyContainer]) -> tuple[str, ...]:
    """Get the keys of the container type."""
    assert issubclass(container_type, PyContainer)

    if issubclass(container_type, xtyping.TypedNamedTupleABC):
        return container_type._fields
    if issubclass(container_type, xtyping.DataclassABC):
        return tuple(container_type.__dataclass_fields__.keys())

    return ()


def make_container_extractor_from_type_spec(
    container_type_spec: ts.NamedTupleType,
) -> PyContainerExtractor:
    assert isinstance(container_type_spec, ts.NamedTupleType)
    return make_container_extractor(pkgutil.resolve_name(container_type_spec.original_python_type))


def make_container_constructor_from_type_spec(
    container_type_spec: ts.NamedTupleType,
) -> PyContainerConstructor:
    assert isinstance(container_type_spec, ts.NamedTupleType)
    return make_container_constructor(
        pkgutil.resolve_name(container_type_spec.original_python_type)
    )


def make_container_extractor(container_type: type[PyContainer]) -> PyContainerExtractor:
    """
    Create an extractor function for the given container type.

    The returned function takes an instance of the specified container type
    and extracts its values into a nested tuple according to the provided
    type specification.
    """
    assert issubclass(container_type, PyContainer)

    extractor_func_src = f"lambda x: {make_extractor_expr('x', container_type)}"
    return eval(extractor_func_src)


def make_extractor_expr(value_name: str, container_type: type[PyContainer]) -> str:
    children_hints = {
        key: xtyping.get_origin(value) or value
        for key, value in xtyping.get_type_hints(container_type).items()
    }
    args = {
        key: make_extractor_expr(f"{value_name}.{key}", children_hints[key])
        if issubclass(children_hints[key], PY_CONTAINER_TYPES)
        else f"{value_name}.{key}"
        for key in keys(container_type)
    }
    if issubclass(container_type, tuple) and all(
        value.endswith(f".{key}") for key, value in args.items()
    ):
        # Fast path: input argument is already a pure nested tuple
        return f"{value_name}"
    else:
        return f"({', '.join(args.values())})"


def _get_args_info(container_type: type[PyContainer]) -> tuple[int, list[str]]:
    assert issubclass(container_type, PyContainer)

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
    container_type: type[PyContainerT],
) -> PyContainerConstructor[PyContainerT]:
    """
    Create a constructor function for the given container type.

    The returned function takes a nested tuple of values and constructs
    an instance of the specified container type.
    """
    assert issubclass(container_type, PyContainer)

    global_ns = {}

    def make_constructor_expr(
        in_arg: str, container_type: type[PyContainer], container_type_alias: str
    ) -> str:
        # We reached the leaf of the nested container construction so we just return the argument as is
        if (
            container_type in NUMERIC_VALUE_TYPES
            or xtyping.get_origin(container_type) in NUMERIC_VALUE_TYPES
        ):
            return in_arg

        assert issubclass(container_type, PyContainer)

        # Store the container type alias in the global namespace for eval()
        global_ns[container_type_alias] = container_type

        # Recursively generate the constructor call
        children_hints = {
            key: xtyping.get_origin(value) or value
            for key, value in xtyping.get_type_hints(container_type).items()
        }
        args_count, kwargs_keys = _get_args_info(container_type)

        if {children_hints[key] for key in keys(container_type)} <= {*NUMERIC_VALUE_TYPES}:
            # Fast path: all children are values, so we can just use argument unpacking
            call_args = (
                f"*{in_arg}[:{args_count}], {', '.join(f'{k}={in_arg}[{args_count + i}]' for i, k in enumerate(kwargs_keys))}"
                if kwargs_keys
                else f"*{in_arg}"
            )
        else:
            # General path: we need to recursively construct children containers
            call_args_items = []
            for i, key in enumerate(keys(container_type)):
                if children_hints[key] in NUMERIC_VALUE_TYPES:
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


# TODO added by havogt
def extract(value: common.NumericValue | PyContainer) -> common.NumericValue:
    # TODO the input is NestedAnyContainer[PythonContainerValue], to be defined
    """Extract the values from a container into a nested tuple."""
    if isinstance(value, PyContainer):
        return make_container_extractor(type(value))(value)
    return value
