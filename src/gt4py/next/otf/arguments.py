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
from typing import Any, Iterable, Iterator, Optional

import numpy as np
from typing_extensions import Self

from gt4py.next import common
from gt4py.next.otf import workflow
from gt4py.next.type_system import type_specifications as ts, type_translation


@dataclasses.dataclass(frozen=True)
class JITArgs:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @classmethod
    def from_signature(cls, *args: Any, **kwargs: Any) -> Self:
        return cls(args=args, kwargs=kwargs)


@dataclasses.dataclass(frozen=True)
class CompileArg:
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
class CompileConnectivity(common.Connectivity):
    max_neighbors: int
    has_skip_values: bool
    origin_axis: common.Dimension
    neighbor_axis: common.Dimension
    index_type: type[int] | type[np.int32] | type[np.int64]

    @classmethod
    def from_connectivity(cls, connectivity: common.Connectivity) -> Self:
        return cls(
            max_neighbors=connectivity.max_neighbors,
            has_skip_values=connectivity.has_skip_values,
            origin_axis=connectivity.origin_axis,
            neighbor_axis=connectivity.neighbor_axis,
            index_type=connectivity.index_type,
        )


@dataclasses.dataclass(frozen=True)
class CompileArgSpec:
    args: tuple[CompileArg | tuple, ...]
    kwargs: dict[str, CompileArg | tuple]
    offset_provider: dict[str, common.Connectivity | common.Dimension]
    column_axis: Optional[common.Dimension]

    @classmethod
    def from_concrete_no_size(cls, *args: Any, **kwargs: Any) -> Self:
        compile_args = tuple(CompileArg.from_concrete(arg) for arg in args)
        kwargs_copy = kwargs.copy()
        offset_provider = kwargs_copy.pop("offset_provider", {})
        return cls(
            args=compile_args,
            offset_provider=offset_provider,  # TODO(ricoh): replace with the line below once the temporaries pass is AOT-ready. If unsure, just try it and run the tests.
            # offset_provider={k: connectivity_or_dimension(v) for k, v in offset_provider.items()},  # noqa: ERA001 [commented-out-code]
            column_axis=kwargs_copy.pop("column_axis", None),
            kwargs={
                k: CompileArg.from_concrete(v) for k, v in kwargs_copy.items() if v is not None
            },
        )

    @classmethod
    def from_concrete(cls, *args: Any, **kwargs: Any) -> Self:
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
) -> CompileArgSpec:
    return CompileArgSpec.from_concrete_no_size(*inp.args, **inp.kwargs)


def jit_to_aot_args_factory(
    adapter: bool = True,
) -> workflow.Workflow[JITArgs, CompileArgSpec]:
    wf = jit_to_aot_args
    if adapter:
        wf = workflow.ArgsOnlyAdapter(wf)
    return wf


def connectivity_or_dimension(
    some_offset_provider: common.Connectivity | common.Dimension,
) -> CompileConnectivity | common.Dimension:
    match some_offset_provider:
        case common.Dimension():
            return some_offset_provider
        case common.Connectivity():
            return CompileConnectivity.from_connectivity(some_offset_provider)
        case _:
            raise ValueError


def iter_size_compile_args(args: Iterable[CompileArg | tuple]) -> Iterator[CompileArg | tuple]:
    for arg in args:
        match argt := type_translation.from_value(arg):
            case ts.TupleType():
                yield from iter_size_compile_args((CompileArg(t) for t in argt))
            case ts.FieldType():
                yield from [
                    CompileArg(ts.ScalarType(kind=ts.ScalarKind.INT32)) for dim in argt.dims
                ]
            case _:
                pass
