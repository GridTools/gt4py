# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, Final, TypeAlias

from gt4py._core.definitions import Scalar
from gt4py.next import backend, common, constructors
from gt4py.next.ffront import decorator, stages as ffront_stages
from gt4py.next.otf import compiled_program


_ONLY_FOR_TYPING: Final[str] = "only for typing"

# TODO(havogt): alternatively we could introduce Protocols
DSLDefinition: TypeAlias = Annotated[ffront_stages.DSLDefinition, _ONLY_FOR_TYPING]

Program: TypeAlias = Annotated[decorator.Program, _ONLY_FOR_TYPING]
FieldOperator: TypeAlias = Annotated[decorator.FieldOperator, _ONLY_FOR_TYPING]
GTEntryPoint: TypeAlias = Annotated[decorator.GTEntryPoint, _ONLY_FOR_TYPING]

CompiledProgramsKey: TypeAlias = Annotated[compiled_program.CompiledProgramsKey, _ONLY_FOR_TYPING]

Backend: TypeAlias = Annotated[backend.Backend, _ONLY_FOR_TYPING]

Allocator: TypeAlias = Annotated[constructors.Allocator, _ONLY_FOR_TYPING]

OffsetProvider: TypeAlias = Annotated[common.OffsetProvider, _ONLY_FOR_TYPING]


__all__ = [
    "Allocator",
    "Backend",
    "FieldOperator",
    "OffsetProvider",
    "Program",
    # from _core.definitions for convenience
    "Scalar",
]
