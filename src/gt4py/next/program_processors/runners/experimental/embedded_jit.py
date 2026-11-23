# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
What the embedded jit backends share: the embedded execution of an entry point, as they wrap it.

A stopgap until embedded execution is represented by backends (ADR 0030).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from gt4py.next import common, errors, named_collections, utils
from gt4py.next.embedded import context as embedded_context
from gt4py.next.ffront import type_info as ffront_type_info, type_specifications as ts_ffront
from gt4py.next.otf import arguments
from gt4py.next.type_system import type_info, type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.ffront import decorator


#: Wraps the embedded execution, called as the signature describes, in a tracing compiler, with
#: the named parameters static. The arguments are fields, connectivities, scalars and (nested)
#: tuples of them.
Wrap: TypeAlias = Callable[
    [Callable[..., None], inspect.Signature, Sequence[str]], Callable[..., None]
]


def jit(entry_point: decorator.GTEntryPoint, wrap: Wrap) -> Callable[..., None]:
    """`entry_point` run embedded in the tracing compiler of `wrap`, called as the entry point is."""
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

    # a field operator called in an embedded context runs embedded
    run = entry_point.definition_stage.definition if is_program else entry_point

    def run_embedded(*args: Any, offset_provider: common.OffsetProvider, **kwargs: Any) -> None:
        with embedded_context.update(offset_provider=offset_provider):
            run(*args, **kwargs)

    # named collections reach the tracing compiler as tuples, which are pytrees in every library
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

    jitted = wrap(
        with_named_collections if constructors else run_embedded, signature, static_params
    )
    param_names = tuple(program_type.definition.pos_or_kw_args)

    def call(*args: Any, **kwargs: Any) -> None:
        if not is_program:
            if "out" not in kwargs:
                raise errors.MissingArgumentError(None, "out", True)
            if "domain" in kwargs:
                # a `Domain` is static in the tracing compiler, a `dict` is not
                kwargs["domain"] = utils.tree_map(common.domain)(kwargs["domain"])
        if constructors:
            kwargs = {**dict(zip(param_names, args)), **kwargs}
            args = ()
            kwargs = {
                name: arguments.extract(value) if name in constructors else value
                for name, value in kwargs.items()
            }
        jitted(*args, **kwargs)

    call.__wrapped__ = jitted  # type: ignore[attr-defined]
    return call


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
