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

from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import Any, Final, NestedTuple, TypeAlias, TypeVar
from gt4py.next import common
from gt4py.next.type_system import type_info, type_specifications as ts


class PyContainerDataclassABC(xtyping.DataclassABC):
    """ABC Check if a dataclass respects current constraints for dataclasses."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if dataclasses.is_dataclass(subclass) and not subclass.__module__.startswith("gt4py."):
            fields = subclass.__dataclass_fields__.values()
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

AnyContainer = tuple | PyContainer
AnyInput = AnyContainer | common.NumericValue

PY_CONTAINER_TYPES: Final[tuple[type, ...]] = typing.cast(
    tuple[type, ...],
    PyContainer.__args__,
)

ANY_CONTAINER_TYPES: Final[tuple[type, ...]] = typing.cast(tuple[type, ...], AnyContainer.__args__)


@functools.cache
def keys(container_type: type[PyContainer]) -> tuple[str, ...]:
    """Get the keys of the container type."""
    assert issubclass(container_type, PY_CONTAINER_TYPES)

    if issubclass(container_type, xtyping.TypedNamedTupleABC):
        return container_type._fields
    if issubclass(container_type, xtyping.DataclassABC):
        return tuple(container_type.__dataclass_fields__.keys())

    return ()


def make_container_extractor_from_type_spec(
    container_type_spec: ts.NamedTupleType,
) -> PyContainerExtractor:
    """Create an extractor function for the given container type specification."""
    assert isinstance(container_type_spec, ts.NamedTupleType)
    extractor_func_src = f"lambda x: {make_extractor_expr_from_type_spec(container_type_spec, 'x')}"
    return eval(extractor_func_src)


def make_container_constructor_from_type_spec(
    container_type_spec: ts.NamedTupleType,
) -> PyContainerConstructor:
    """Create a constructor function for the given container type specification."""
    assert isinstance(container_type_spec, ts.NamedTupleType)
    return make_container_constructor(
        pkgutil.resolve_name(container_type_spec.original_python_type)
    )


@functools.cache
def make_container_extractor(container_type: xtyping.TypeAnnotation) -> PyContainerExtractor:
    """
    Create an extractor function for the given container type.

    The returned function takes an instance of the specified container type
    and extracts its values into a nested tuple according to the provided
    type specification.
    """

    extractor_func_src = f"lambda x: {make_extractor_expr(container_type, 'x')}"
    return eval(extractor_func_src)


@functools.singledispatch
def make_extractor_expr(type_: xtyping.TypeAnnotation, value_expr: str) -> str:
    """Make an expression from a type description to extract numeric values out of its instances."""

    expr_parts: list[str] = []
    actual_type = xtyping.get_origin(type_) or type_

    if isinstance(actual_type, type) and issubclass(actual_type, PY_CONTAINER_TYPES):
        if children_type_hints := xtyping.get_type_hints(type_):
            expr_parts = [
                make_extractor_expr(value, f"{value_expr}.{key}")
                for key, value in children_type_hints.items()
            ]
            # Check if we can just skip extraction if the input type
            # is already a pure nested tuple and we just return it as is.
            actual_type = xtyping.get_origin(type_) or type_
            if (
                isinstance(actual_type, type)
                and issubclass(actual_type, tuple)
                and all(
                    value == f"{value_expr}.{key}"
                    for key, value in zip(children_type_hints, expr_parts, strict=True)
                )
            ):
                return f"{value_expr}"

        elif xtyping.get_origin(type_) is tuple:
            # This is a `tuple` with type arguments
            expr_parts = [
                make_extractor_expr(f"{value_expr}[{i}]", value)
                for i, value in enumerate(xtyping.get_args(type_))
            ]

    # If we don't have information about the inner structure of the type,
    # assume it doesn't need extraction and just return it as is.
    return f"({', '.join(expr_parts)})" if expr_parts else value_expr


def make_extractor_expr_from_type_spec(type_: ts.TypeSpec, value_expr: str) -> str:
    """Create an expression string that extracts values from a container."""
    match type_:
        case ts.NamedTupleType():
            return make_extractor_expr(pkgutil.resolve_name(type_.original_python_type), value_expr)
        case ts.TupleType() if type_info.needs_value_extraction(type_):
            args = [
                make_extractor_expr_from_type_spec(t, f"{value_expr}[{i}]")
                for i, t in enumerate(type_.types)
            ]
            return f"({', '.join(args)})"

    return value_expr


def _get_pycontainer_constructor_args_info(
    container_type: type[PyContainer],
) -> tuple[int, list[str]]:
    assert issubclass(container_type, PY_CONTAINER_TYPES)

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
    assert issubclass(container_type, PY_CONTAINER_TYPES)

    global_ns = {}

    constructor_func_body = make_constructor_expr(
        container_type, "x", item_path="", global_ns=global_ns
    )
    constructor_func_src = f"lambda x: {constructor_func_body}"

    return eval(constructor_func_src, global_ns)


def make_constructor_expr(
    container_type: type[PyContainer],
    value_expr: str,
    *,
    item_path: str,
    global_ns: dict[str, Any],
) -> str:
    """
    Create an expression string that constructs a container from a nested tuple of values.

    The expression assumes that `value_expr` is a nested tuple of values that
    correspond to the structure of `container_type`. The expression constructs
    an instance of `container_type` using these values.

    Arguments:
        container_type: The type of the container to construct.
        value_expr: The expression that evaluates to the nested tuple of values.

    Keyword Arguments:
        item_path: A unique path string used to create unique names for container
            types in the constructor building expression.
        global_ns: The global namespace where the unique container type aliases are
            stored for final evaluation of the constructor expression.
    """
    if not (isinstance(container_type, type) and issubclass(container_type, PY_CONTAINER_TYPES)):
        # We don't need to construct anything, just return the argument as is
        return value_expr

    # Store the container type alias in the global namespace for eval()
    container_type_alias = f"__{item_path}__{container_type.__name__}"
    global_ns[container_type_alias] = container_type

    # Get the type hints of the container's members
    container_keys = keys(container_type)
    type_hints = xtyping.get_type_hints(container_type)
    container_keys_hints = {
        key: xtyping.get_origin(value := type_hints.get(key)) or value for key in container_keys
    }

    call_args = [
        make_constructor_expr(
            container_keys_hints[key],
            f"{value_expr}[{i}]",
            item_path=f"{item_path}__{key}",
            global_ns=global_ns,
        )
        for i, key in enumerate(container_keys)
    ]

    # Optimize the call expression if none of the children needs further construction
    # use argument unpacking to pass them to the constructor
    if all(arg == f"{value_expr}[{i}]" for i, arg in enumerate(call_args)):
        # Fast path: none of the children needs construction, so we can just use argument
        # unpacking to pass them to the constructor
        args_count, kwargs_keys = _get_pycontainer_constructor_args_info(container_type)
        if not kwargs_keys:
            call_args = [f"*{value_expr}"]
        else:
            call_args = [
                f"*{value_expr}[:{args_count}]",
                *(f"{k}={value_expr}[{args_count + i}]" for i, k in enumerate(kwargs_keys)),
            ]

    return f"{container_type_alias}({', '.join(call_args)})"
