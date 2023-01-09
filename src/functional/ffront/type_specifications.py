# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either vervisit_Constantsion 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass

from functional import common as func_common
from functional.type_system.type_specifications import (  # noqa: F401
    CallableType as CallableType,
    DataType as DataType,
    DeferredType as DeferredType,
    DimensionType as DimensionType,
    FieldType as FieldType,
    FunctionType as FunctionType,
    OffsetType as OffsetType,
    ScalarKind as ScalarKind,
    ScalarType as ScalarType,
    TupleType as TupleType,
    TypeSpec as TypeSpec,
    VoidType as VoidType,
)


@dataclass(frozen=True)
class ProgramType(TypeSpec, CallableType):
    definition: FunctionType


@dataclass(frozen=True)
class FieldOperatorType(TypeSpec, CallableType):
    definition: FunctionType


@dataclass(frozen=True)
class ScanOperatorType(TypeSpec, CallableType):
    axis: func_common.Dimension
    definition: FunctionType
