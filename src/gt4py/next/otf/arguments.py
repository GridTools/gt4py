# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Iterable, Iterator, Optional

from typing_extensions import Self

from gt4py.next import common
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


DATA_T = typing.TypeVar("DATA_T")


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

    args: tuple[ts.TypeSpec, ...]
    kwargs: dict[str, ts.TypeSpec]
    offset_provider: common.OffsetProvider  # TODO(havogt): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    column_axis: Optional[common.Dimension]

    @property
    def offset_provider_type(self) -> common.OffsetProviderType:
        return common.offset_provider_to_type(self.offset_provider)

    @classmethod
    def from_concrete_no_size(cls, *args: Any, **kwargs: Any) -> Self:
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
    def from_concrete(cls, *args: Any, **kwargs: Any) -> Self:
        """Convert concrete GTX program arguments to compile-time, adding (compile-time) dimension size arguments."""
        no_size = cls.from_concrete_no_size(*args, **kwargs)
        return cls(
            args=(*no_size.args, *iter_size_compile_args(no_size.args)),
            offset_provider=no_size.offset_provider,
            column_axis=no_size.column_axis,
            kwargs=no_size.kwargs,
        )

    @classmethod
    def empty(cls) -> Self:
        return cls(tuple(), {}, {}, None)


def jit_to_aot_args(
    inp: JITArgs,
) -> CompileTimeArgs:
    return CompileTimeArgs.from_concrete_no_size(*inp.args, **inp.kwargs)


def adapted_jit_to_aot_args_factory() -> (
    workflow.Workflow[
        toolchain.CompilableProgram[DATA_T, JITArgs],
        toolchain.CompilableProgram[DATA_T, CompileTimeArgs],
    ]
):
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


def iter_size_args(args: tuple[Any, ...]) -> Iterator[int]:
    """
    Yield the size of each field argument in each dimension.

    This can be used to generate domain size arguments for FieldView Programs that use an implicit domain.
    """
    for arg in args:
        match arg:
            case tuple():
                # we only need the first field, because all fields in a tuple must have the same dims and sizes
                first_field = find_first_field(arg)
                if first_field:
                    yield from iter_size_args((first_field,))
            case common.Field():
                yield from arg.ndarray.shape
            case _:
                pass


def iter_size_compile_args(
    args: Iterable[ts.TypeSpec],
) -> Iterator[ts.TypeSpec]:
    """
    Yield a compile-time size argument for every compile-time field argument in each dimension.

    This can be used inside transformation workflows to generate compile-time domain size arguments for FieldView Programs that use an implicit domain.
    """
    for arg in args:
        field_constituents: list[ts.FieldType] = typing.cast(
            list[ts.FieldType],
            type_info.primitive_constituents(arg).if_isinstance(ts.FieldType).to_list(),
        )
        if field_constituents:
            # we only need the first field, because all fields in a tuple must have the same dims and sizes
            yield from [
                ts.ScalarType(kind=ts.ScalarKind.INT32) for dim in field_constituents[0].dims
            ]
