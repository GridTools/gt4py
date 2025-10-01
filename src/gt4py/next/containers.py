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
from gt4py.eve.extended_typing import (
    Any,
    Final,
    Mapping,
    NestedTuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
)
from gt4py.next import common
from gt4py.next.type_system import type_info, type_specifications as ts


class CustomDataclassContainerABC(xtyping.DataclassABC):
    """ABC for dataclasses satisfying the current constraints to be custom containers."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if dataclasses.is_dataclass(subclass) and (
            not subclass.__module__.startswith("gt4py.")  # Exclude GT4Py internal dataclasses
        ):
            fields = subclass.__dataclass_fields__.values()
            return len(fields) > 0 and all(
                f.init is True
                and f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
                and f._field_type is dataclasses._FIELD  # type: ignore[attr-defined]  # private member
                for f in fields
            )

        return False


CustomContainer: TypeAlias = xtyping.TypedNamedTupleABC | CustomDataclassContainerABC

CUSTOM_CONTAINER_TYPES: Final[tuple[type, ...]] = xtyping.get_represented_types(CustomContainer)

Container: TypeAlias = NestedTuple | CustomContainer

CONTAINER_TYPES: Final[tuple[type, ...]] = xtyping.get_represented_types(Container)


ContainerT = TypeVar("ContainerT", bound=Container)

ContainerExtractor: TypeAlias = Callable[[ContainerT], NestedTuple[common.PrimitiveValue]]
ContainerConstructor: TypeAlias = Callable[[NestedTuple[common.PrimitiveValue]], ContainerT]

ContainerKey = str | int


def container_type(type_hint: xtyping.TypeAnnotation) -> type[Container] | None:
    """Get the type if the given type hint represents a supported Python container type."""
    class_ = xtyping.get_origin(type_hint) or type_hint
    if class_ is tuple or (isinstance(class_, type) and issubclass(class_, CUSTOM_CONTAINER_TYPES)):
        return typing.cast(type[Container], class_)
    return None


def is_container_type(type_hint: xtyping.TypeAnnotation) -> TypeGuard[type[Container]]:
    """Check if a type annotation represents a supported Python container type."""
    return container_type(type_hint) is not None


@functools.cache
def elements_keys(container_type_hint: xtyping.SingleTypeAnnotation) -> tuple[ContainerKey, ...]:
    """Get the keys of the elements in a container type."""
    class_ = xtyping.get_origin(container_type_hint) or container_type_hint
    if class_ is tuple:
        return tuple(range(len(xtyping.get_args(container_type_hint))))

    return getattr(class_, "__match_args__", ()) if isinstance(class_, type) else ()


def elements_types(
    container_type_hint: xtyping.SingleTypeAnnotation,
    *,
    globalns: dict[str, Any] | None = None,
    localns: dict[str, Any] | None = None,
) -> Mapping[ContainerKey, type]:
    """Get the types of the elements of a container type."""

    if xtyping.get_origin(container_type_hint) is tuple:
        return {i: value for i, value in enumerate(xtyping.get_args(container_type_hint))}

    type_ = container_type(container_type_hint)
    if type_ is not None:
        keys = elements_keys(container_type_hint)
        all_hints = xtyping.get_type_hints(container_type_hint, globalns=globalns, localns=localns)
        if not {*keys} <= all_hints.keys():
            raise TypeError(f"Missing type hints for container elements: {keys - all_hints.keys()}")
        assert all(isinstance(k, str) for k in keys)
        return {key: all_hints[key] for key in typing.cast(dict[str, type], keys)}

    return {}


@functools.cache
def make_container_extractor(container_type_hint: xtyping.TypeAnnotation) -> ContainerExtractor:
    """
    Create an extractor function for the given container type.

    The returned function takes an instance of the specified container type
    and extracts its values into a nested tuple according to the provided
    type specification.
    """

    extractor_func_src = f"lambda x: {make_extractor_expr(container_type_hint, 'x')}"
    return eval(extractor_func_src)


def make_container_extractor_from_type_spec(
    container_type_spec: ts.NamedTupleType,
) -> ContainerExtractor:
    """Create an extractor function for the given container type specification."""
    assert isinstance(container_type_spec, ts.NamedTupleType)
    extractor_func_src = f"lambda x: {make_extractor_expr_from_type_spec(container_type_spec, 'x')}"
    return eval(extractor_func_src)


def make_extractor_expr(container_type_hint: xtyping.TypeAnnotation, value_expr: str) -> str:
    """Make an expression from a type description to extract numeric values out of its instances."""

    children_type_hints: dict[str, xtyping.TypeAnnotation] = {}
    expr_parts: list[str] = []
    actual_type = xtyping.get_origin(container_type_hint) or container_type_hint

    if isinstance(actual_type, type):
        if issubclass(actual_type, CUSTOM_CONTAINER_TYPES):
            children_type_hints = xtyping.get_type_hints(container_type_hint)
            assert len(children_type_hints)
            expr_parts = [
                make_extractor_expr(value, f"{value_expr}.{key}")
                for key, value in children_type_hints.items()
            ]
        elif issubclass(actual_type, tuple) and (
            tuple_arg_hints := xtyping.get_args(container_type_hint)
        ):
            # This is a `tuple` with type arguments
            expr_parts = [
                make_extractor_expr(value, f"{value_expr}[{i}]")
                for i, value in enumerate(tuple_arg_hints)
            ]

    # Check if we can just skip extraction if the input type
    # is already a pure nested tuple and we just return it as is.
    if isinstance(actual_type, type) and issubclass(actual_type, tuple):
        if (
            children_type_hints
            and all(
                value == f"{value_expr}.{key}"
                for key, value in zip(children_type_hints, expr_parts, strict=True)
            )
        ) or (
            not children_type_hints
            and all(value == f"{value_expr}[{i}]" for i, value in enumerate(expr_parts))
        ):
            return f"{value_expr}"

    # If we don't have information about the inner structure of the type,
    # assume it doesn't need extraction and just return it as is.
    return f"({', '.join(expr_parts)},)" if expr_parts else value_expr


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


def _get_container_constructor_args_info(
    container_type_hint: xtyping.SingleTypeAnnotation,
) -> tuple[int, list[str]]:
    if xtyping.get_origin(container_type_hint) is tuple:
        # For plain tuples, we assume all arguments are positional
        args_count = len(xtyping.get_args(container_type_hint))
        return args_count, []

    assert isinstance(container_type_hint, type)
    assert issubclass(container_type_hint, CUSTOM_CONTAINER_TYPES)

    # Use a constructor signature without variadic parameters
    for method_name in ("__new__", "__init__"):
        params = inspect.signature(getattr(container_type_hint, method_name)).parameters
        if not (
            {p.kind for p in params.values()}
            & {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        ):
            break
    else:
        raise TypeError(
            f"Type {container_type_hint} with variadic parameters in constructor are not supported"
        )
    if len(params) == 1:
        raise TypeError(f"Type {container_type_hint} has no parameters in constructor")

    actual_params: dict[str, inspect.Parameter] = {
        k: v for k, v in [*params.items()][1:]
    }  # drop 'self' / 'cls' parameter
    args_count = sum(
        p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        for p in actual_params.values()
    )
    kwargs_keys = [
        key for key, param in actual_params.items() if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    assert args_count + len(kwargs_keys) == len(actual_params), (
        "Variadic parameters are not supported"
    )

    return args_count, kwargs_keys


@functools.cache
def make_container_constructor(
    container_type_hint: type[ContainerT] | xtyping.TypeAnnotation,
    *,
    nested: bool = True,
) -> ContainerConstructor[ContainerT]:
    """
    Create a constructor function for the given container type.

    The returned function takes a nested tuple of values and constructs
    an instance of the specified container type.
    """
    assert (type_ := container_type(container_type_hint)) is not None
    assert issubclass(type_, CONTAINER_TYPES)

    global_ns: dict[str, Any] = {}
    constructor_func_body = make_constructor_expr(
        container_type_hint, "x", item_path="", global_ns=global_ns, nested=nested
    )
    constructor_func_src = f"lambda x: {constructor_func_body}"

    return eval(constructor_func_src, global_ns)


def make_container_constructor_from_type_spec(
    container_type_spec: ts.NamedTupleType, nested: bool = True,
) -> ContainerConstructor:
    """Create a constructor function for the given container type specification."""
    assert isinstance(container_type_spec, ts.NamedTupleType)
    return make_container_constructor(
        pkgutil.resolve_name(container_type_spec.original_python_type), nested=True
    )


def make_constructor_expr(
    container_type_hint: xtyping.TypeAnnotation,
    value_expr: str,
    *,
    item_path: str,
    global_ns: dict[str, Any],
    nested: bool = True,  # TODO: consider splitting into separate function
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
    actual_type = xtyping.get_origin(container_type_hint) or container_type_hint
    nested_types: dict[int | str, xtyping.TypeAnnotation] = {}
    if isinstance(actual_type, type):
        if issubclass(actual_type, CUSTOM_CONTAINER_TYPES):
            # Store the container type alias in the global namespace for eval()
            container_type_alias = f"__{item_path}__{actual_type.__name__}"
            global_ns[container_type_alias] = actual_type

            # Get the type hints of the container's members
            container_type = typing.cast(str | type, container_type_hint)
            container_keys = elements_keys(container_type)
            type_hints = xtyping.get_type_hints(container_type)
            assert {*container_keys} <= type_hints.keys(), "Mismatch between keys and type hints"
            nested_types = {
                key: type_hints[key] for key in typing.cast(dict[str, type], container_keys)
            }

        elif issubclass(actual_type, tuple) and (
            tuple_args_hint := xtyping.get_args(container_type_hint)
        ):
            container_type_alias = ""
            nested_types = {i: type_hint for i, type_hint in enumerate(tuple_args_hint)}

    if nested_types:
        if nested:
            call_args = [
                make_constructor_expr(
                    type_hint,
                    f"{value_expr}[{i}]",
                    item_path=f"{item_path}__{key}",
                    global_ns=global_ns,
                )
                for i, (key, type_hint) in enumerate(nested_types.items())
            ]
        else:
            call_args = [f"{value_expr}[{i}]" for i in range(len(nested_types))]

        # Optimize the call expression if none of the children needs further construction
        # use argument unpacking to pass them to the constructor
        if all(arg == f"{value_expr}[{i}]" for i, arg in enumerate(call_args)):
            if not container_type_alias:
                # Fast path for plain tuples: none of the children needs construction, so we can just
                # return the value expression as is
                return value_expr

            # Fast path: none of the children needs construction, so we can just use argument
            # unpacking to pass them to the constructor
            args_count, kwargs_keys = _get_container_constructor_args_info(container_type_hint)
            if not kwargs_keys:
                call_args = [f"*{value_expr}"]
            else:
                call_args = [
                    f"*{value_expr}[:{args_count}]",
                    *(f"{k}={value_expr}[{args_count + i}]" for i, k in enumerate(kwargs_keys)),
                ]

        return f"{container_type_alias}({', '.join(call_args)})"

    # We don't need to construct anything, just return the argument as is
    return value_expr
