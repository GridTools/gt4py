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
from typing import Any, Generic, Iterable, Iterator, Optional

from typing_extensions import Self

from gt4py._core import definitions as core_defs
from gt4py.next import common
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