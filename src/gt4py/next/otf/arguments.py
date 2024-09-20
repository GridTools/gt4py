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

import numpy as np
from typing_extensions import Self

from gt4py.next import common
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_specifications as ts, type_translation


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
class CompileTimeArg:
    """Standin (at compile-time) for a GTX program argument, retaining only the type information."""

    gt_type: ts.TypeSpec

    def __gt_type__(self) -> ts.TypeSpec:
        return self.gt_type

    @classmethod
    def from_concrete(cls, value: Any) -> Self | tuple[Self | tuple, ...]:
        gt_type = type_translation.from_value(value)
        match gt_type:
            case ts.TupleType():
                return tuple(cls.from_concrete(element) for element in value)
            case _:
                return cls(gt_type)


@dataclasses.dataclass(frozen=True)
class CompileTimeConnectivity(common.Connectivity):
    """Compile-time standin for a GTX connectivity, retaining everything except the connectivity tables."""

    max_neighbors: int
    has_skip_values: bool
    origin_axis: common.Dimension
    neighbor_axis: common.Dimension
    index_type: type[int] | type[np.int32] | type[np.int64]

    def mapped_index(
        self, cur_index: int | np.integer, neigh_index: int | np.integer
    ) -> Optional[int | np.integer]:
        raise NotImplementedError(
            "A CompileTimeConnectivity instance should not call `mapped_index`."
        )

    @classmethod
    def from_connectivity(cls, connectivity: common.Connectivity) -> Self:
        return cls(
            max_neighbors=connectivity.max_neighbors,
            has_skip_values=connectivity.has_skip_values,
            origin_axis=connectivity.origin_axis,
            neighbor_axis=connectivity.neighbor_axis,
            index_type=connectivity.index_type,
        )

    @property
    def table(self) -> None:
        return None


@dataclasses.dataclass(frozen=True)
class CompileTimeArgs:
    """Compile-time standins for arguments to a GTX program to be used in ahead-of-time compilation."""

    args: tuple[CompileTimeArg | tuple, ...]
    kwargs: dict[str, CompileTimeArg | tuple]
    offset_provider: dict[str, common.Connectivity | common.Dimension]
    column_axis: Optional[common.Dimension]

    @classmethod
    def from_concrete_no_size(cls, *args: Any, **kwargs: Any) -> Self:
        """Convert concrete GTX program arguments into their compile-time counterparts."""
        compile_args = tuple(CompileTimeArg.from_concrete(arg) for arg in args)
        kwargs_copy = kwargs.copy()
        offset_provider = kwargs_copy.pop("offset_provider", {})
        return cls(
            args=compile_args,
            offset_provider=offset_provider,  # TODO(ricoh): replace with the line below once the temporaries pass is AOT-ready. If unsure, just try it and run the tests.
            # offset_provider={k: connectivity_or_dimension(v) for k, v in offset_provider.items()},  # noqa: ERA001 [commented-out-code]
            column_axis=kwargs_copy.pop("column_axis", None),
            kwargs={
                k: CompileTimeArg.from_concrete(v) for k, v in kwargs_copy.items() if v is not None
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


def connectivity_or_dimension(
    some_offset_provider: common.Connectivity | common.Dimension,
) -> CompileTimeConnectivity | common.Dimension:
    match some_offset_provider:
        case common.Dimension():
            return some_offset_provider
        case common.Connectivity():
            return CompileTimeConnectivity.from_connectivity(some_offset_provider)
        case _:
            raise ValueError


def iter_size_args(args: tuple[Any, ...], inside_tuple: bool = False) -> Iterator[int]:
    """
    Yield the size of each field argument in each dimension.

    This can be used to generate domain size arguments for FieldView Programs that use an implicit domain.
    """
    yielded = False
    for arg in args:
        match arg:
            case tuple():
                # TODO(ricoh) getting size args for the first element is not correct
                #  as the first argument might not be a field
                yield from iter_size_args(arg, inside_tuple=True)
            case common.Field():
                yield from arg.ndarray.shape
                yielded = True
            case _:
                pass
        if inside_tuple and yielded:
            break


def iter_size_compile_args(
    args: Iterable[CompileTimeArg | tuple],
) -> Iterator[CompileTimeArg | tuple]:
    """
    Yield a compile-time size argument for every compile-time field argument in each dimension.

    This can be used inside transformation workflows to generate compile-time domain size arguments for FieldView Programs that use an implicit domain.
    """
    for arg in args:
        match argt := type_translation.from_value(arg):
            case ts.TupleType():
                yield from iter_size_compile_args((CompileTimeArg(t) for t in argt))
            case ts.FieldType():
                yield from [
                    CompileTimeArg(ts.ScalarType(kind=ts.ScalarKind.INT32)) for dim in argt.dims
                ]
            case _:
                pass
