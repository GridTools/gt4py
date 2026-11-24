# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The embedded execution of an entry point, as the embedded backends run it."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from gt4py._core import ndarray_utils as core_ndarray_utils
from gt4py.next import common, errors, named_collections, utils
from gt4py.next.embedded import array_ops, context as embedded_context, nd_array_field
from gt4py.next.ffront import type_info as ffront_type_info, type_specifications as ts_ffront
from gt4py.next.otf import arguments
from gt4py.next.type_system import type_info, type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.ffront import decorator


def runner(
    entry_point: decorator.GTEntryPoint,
    wrap: Callable[[Callable[..., None], inspect.Signature, Sequence[str]], Callable[..., None]],
    allocator: Any,
) -> Callable[..., None]:
    """
    `entry_point` run embedded, wrapped by `wrap`, called as the entry point is.

    It raises for fields and connectivities of another array library than `allocator`, if that
    is an array namespace.
    """
    callable_type = entry_point.__gt_type__()
    program_type = ffront_type_info.type_in_program_context(callable_type)
    assert isinstance(program_type, ts_ffront.ProgramType)
    is_program = isinstance(callable_type, ts_ffront.ProgramType)

    signature = ffront_type_info.signature_in_program_context(program_type)
    keyword_only = [
        *([] if is_program else [_keyword_only("domain", default=None)]),
        _keyword_only("offset_provider"),
    ]
    signature = signature.replace(parameters=[*signature.parameters.values(), *keyword_only])
    static_params = (
        *(entry_point.compilation_options.static_params or ()),
        *([] if is_program else ["domain"]),
    )
    if unknown := set(static_params) - signature.parameters.keys():
        raise ValueError(
            f"Static parameters {sorted(unknown)} are not parameters of the entry point."
        )
    expected = (
        array_ops.of(allocator.empty((0,)))
        if core_ndarray_utils.is_array_namespace(allocator)
        else None
    )

    # a field operator called in an embedded context runs embedded
    run = entry_point.definition_stage.definition if is_program else entry_point

    def run_embedded(*args: Any, offset_provider: common.OffsetProvider, **kwargs: Any) -> None:
        with embedded_context.update(offset_provider=offset_provider):
            run(*args, **kwargs)

    # named collections reach `wrap` as tuples, which are pytrees in every array library
    params = program_type.definition.pos_or_kw_args | program_type.definition.kw_only_args
    constructors = {
        name: _named_collection_constructor(type_)
        for name, type_ in params.items()
        if type_info.needs_value_extraction(type_)
    }

    def with_named_collections(**kwargs: Any) -> None:
        run_embedded(
            **{name: constructors.get(name, _identity)(value) for name, value in kwargs.items()}
        )

    wrapped = wrap(
        with_named_collections if constructors else run_embedded, signature, static_params
    )

    def call(*args: Any, **kwargs: Any) -> None:
        if not is_program:
            if "out" not in kwargs:
                raise errors.MissingArgumentError(None, "out", True)
            if "domain" in kwargs:
                # a `Domain` can be a static argument of a tracing compiler, a `dict` cannot
                kwargs["domain"] = utils.tree_map(common.domain)(kwargs["domain"])
        if constructors:
            args, kwargs = (
                (),
                {
                    name: arguments.extract(value) if name in constructors else value
                    for name, value in signature.bind(*args, **kwargs).arguments.items()
                },
            )
        if expected is not None:
            _check_array_library(expected, (args, kwargs))
        wrapped(*args, **kwargs)

    call.__wrapped__ = wrapped  # type: ignore[attr-defined]
    return call


def _check_array_library(expected: array_ops.ArrayOps, value: Any) -> None:
    if isinstance(value, nd_array_field.NdArrayField):
        if array_ops.of(value._ndarray) is not expected:
            raise TypeError(
                f"A field of '{type(value._ndarray).__module__}.{type(value._ndarray).__name__}'"
                f" arrays, but the backend runs on '{expected.xp.__name__}' arrays."
            )
    elif isinstance(value, tuple | list):
        for element in value:
            _check_array_library(expected, element)
    elif isinstance(value, dict):
        for element in value.values():
            _check_array_library(expected, element)


def _keyword_only(name: str, **kwargs: Any) -> inspect.Parameter:
    return inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, **kwargs)


def _identity(value: Any) -> Any:
    return value


def _named_collection_constructor(type_: ts.TypeSpec) -> Callable[[Any], Any]:
    """Rebuild the named collections of a value of `type_` from the tuples of `arguments.extract`."""
    if isinstance(type_, ts.NamedCollectionType):
        return named_collections.make_named_collection_constructor_from_type_spec(type_)
    if isinstance(type_, ts.TupleType) and type_info.needs_value_extraction(type_):
        constructors = [_named_collection_constructor(element) for element in type_.types]
        return lambda values: tuple(
            constructor(value) for constructor, value in zip(constructors, values, strict=True)
        )
    return _identity
