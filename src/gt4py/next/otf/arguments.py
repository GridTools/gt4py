# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import (
    Any,
    Generic,
    Literal,
    MaybeNestedInTuple,
    Optional,
    Self,
    TypeAlias,
    TypeIs,
    TypeVar,
    TypeVarTuple,
    Unpack,
)
from gt4py.next import common, containers
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


DATA_T = typing.TypeVar("DATA_T")


@dataclasses.dataclass(frozen=True)
class StaticArg(Generic[core_defs.ScalarT]):
    value: core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...]
    type_: ts.TypeSpec


@dataclasses.dataclass(frozen=True)
class JITArgs:
    """Concrete (runtime) arguments to a GTX program in a format that can be passed into the toolchain."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @classmethod
    def from_signature(cls, *args: Any, **kwargs: Any) -> Self:
        return cls(args=args, kwargs=kwargs)


@dataclasses.dataclass(frozen=True)
class CompileTimeArgs:
    """Compile-time standins for arguments to a GTX program to be used in ahead-of-time compilation."""

    args: tuple[ts.TypeSpec | StaticArg, ...]
    kwargs: dict[str, ts.TypeSpec | StaticArg]
    offset_provider: common.OffsetProvider  # TODO(havogt): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    column_axis: Optional[common.Dimension]

    @property
    def offset_provider_type(self) -> common.OffsetProviderType:
        return common.offset_provider_to_type(self.offset_provider)

    @classmethod
    def from_concrete(cls, *args: Any, **kwargs: Any) -> Self:
        """Convert concrete GTX program arguments into their compile-time counterparts."""
        compile_args = tuple(type_translation.from_value(arg) for arg in args)
        kwargs_copy = kwargs.copy()
        offset_provider = kwargs_copy.pop("offset_provider", {})
        return cls(
            args=compile_args,
            offset_provider=offset_provider,
            column_axis=kwargs_copy.pop("column_axis", None),
            kwargs={
                k: type_translation.from_value(v) for k, v in kwargs_copy.items() if v is not None
            },
        )

    @classmethod
    def empty(cls) -> Self:
        return cls(tuple(), {}, {}, None)


def jit_to_aot_args(
    inp: JITArgs,
) -> CompileTimeArgs:
    return CompileTimeArgs.from_concrete(*inp.args, **inp.kwargs)


def adapted_jit_to_aot_args_factory() -> workflow.Workflow[
    toolchain.CompilableProgram[DATA_T, JITArgs],
    toolchain.CompilableProgram[DATA_T, CompileTimeArgs],
]:
    """Wrap `jit_to_aot` into a workflow adapter to fit into backend transform workflows."""
    return toolchain.ArgsOnlyAdapter(jit_to_aot_args)


def find_first_field(tuple_arg: tuple[Any, ...]) -> Optional[common.Field]:
    for element in tuple_arg:
        match element:
            case tuple():
                found = find_first_field(element)
                if found:
                    return found
            case common.Field():
                return element
            case _:
                pass
    return None


Ts = TypeVarTuple("Ts")
NeedsValueExtraction: TypeAlias = (
    containers.PyContainer
    | tuple[Unpack[Ts], "NeedsValueExtraction"]
    | tuple["NeedsValueExtraction", Unpack[Ts]]
)


def needs_value_extraction(value: object) -> TypeIs[NeedsValueExtraction]:
    return isinstance(value, containers.PY_CONTAINER_TYPES) or (
        isinstance(value, tuple) and any(needs_value_extraction(v) for v in value)
    )


T = TypeVar("T")


@typing.overload
def extract(
    value: T,
    *,
    pass_through_values: Literal[True],
) -> T | MaybeNestedInTuple[common.NumericValue]: ...


@typing.overload
def extract(
    value: common.NumericValue | NeedsValueExtraction,
    *,
    pass_through_values: Literal[False],
) -> MaybeNestedInTuple[common.NumericValue]: ...


def extract(
    value: Any, pass_through_values: bool = True
) -> MaybeNestedInTuple[common.NumericValue]:
    """
    Extract the values from a run-time value into a digestable numeric value form.

    This functions is useful to do the extraction from run-time values
    when the full type information is not available. For non-container values,
    return them as-is if `pass_through_values` is `True`, otherwise raise a `TypeError`.
    """
    if isinstance(value, common.NUMERIC_VALUE_TYPES):
        return typing.cast(common.NumericValue, value)
    if isinstance(value, containers.PY_CONTAINER_TYPES):
        return containers.make_container_extractor(type(value))(value)
    if isinstance(value, tuple):
        return tuple(extract(v, pass_through_values=pass_through_values) for v in value)
    if pass_through_values:
        return value

    raise TypeError(f"Cannot extract numeric value from {type(value)}.")


# TODO(egparedes): memoize this function (and/or the one above) if TypeSpecs become hashable
def make_numeric_value_args_extractor(
    function: ts.FunctionType,
) -> Callable[..., tuple[tuple, dict[str, Any]]] | None:
    """
    Make a function to extract numeric values from arguments that need it.

    If no arguments need extraction, return `None`.

    The returned function has the signature `(*args, **kwargs) -> (args, kwargs)`,
    where `args` is a tuple of positional arguments and `kwargs` is a dictionary of
    keyword arguments containing the extracted numeric values where needed.
    """
    args_param = "args"
    kwargs_param = "kwargs"
    num_args_to_extract = 0
    num_kwargs_to_extract = 0
    extractor_exprs: dict[int | str, str] = {}

    for i, type_spec in enumerate(
        pos_args := [
            *function.pos_only_args,
            *function.pos_or_kw_args.values(),
        ]
    ):
        if type_info.needs_value_extraction(type_spec):
            num_args_to_extract += 1
            extractor_exprs[i] = containers.make_extractor_expr(type_spec, f"{args_param}[{i}]")
        else:
            extractor_exprs[i] = f"{args_param}[{i}]"

    for name, type_spec in function.kw_only_args.items():
        if type_info.needs_value_extraction(type_spec):
            num_kwargs_to_extract += 1
            extractor_exprs[name] = containers.make_extractor_expr(
                type_spec, f"{kwargs_param}[{name}]"
            )
        else:
            extractor_exprs[name] = f"{kwargs_param}[{name}]"

    if num_args_to_extract + num_kwargs_to_extract:
        args_expr = (
            f"({str.join(', ', (extractor_exprs[i] for i, _ in enumerate(pos_args)))})"
            if num_args_to_extract
            else args_param
        )
        kwargs_expr = (
            f"{{ {str.join(', ', (f'{k}: {extractor_exprs[k]}' for k in function.kw_only_args))} }}"
            if num_kwargs_to_extract
            else kwargs_param
        )

        extractor_func_src = f"lambda *{args_param}, **{kwargs_param}: ({args_expr}, {kwargs_expr})"

        return eval(extractor_func_src)

    return None
