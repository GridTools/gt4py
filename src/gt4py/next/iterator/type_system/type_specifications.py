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

import dataclasses
import typing

from gt4py._core.definitions import IntegralScalar
from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class NamedRangeType(ts.TypeSpec):
    dim: common.Dimension


@dataclasses.dataclass(frozen=True)
class DomainType(ts.DataType):
    dims: list[common.Dimension]


# TODO: how about ts.OffsetType?
@dataclasses.dataclass(frozen=True)
class OffsetLiteralType(ts.TypeSpec):
    value: IntegralScalar | common.Dimension


@dataclasses.dataclass(frozen=True)
class ListType(ts.DataType):
    element_type: ts.DataType


@dataclasses.dataclass(frozen=True)
class IteratorType(ts.DataType, ts.CallableType):  # todo: rename to iterator
    position_dims: list[common.Dimension] | typing.Literal["unknown"]
    defined_dims: list[common.Dimension]
    element_type: ts.DataType


@dataclasses.dataclass(frozen=True)
class StencilClosureType(ts.TypeSpec):
    domain: DomainType
    stencil: ts.FunctionType
    output: ts.FieldType | ts.TupleType  # todo: validate tuple of fields
    inputs: list[ts.FieldType]


@dataclasses.dataclass(frozen=True)
class FencilType(ts.TypeSpec):
    params: list[ts.DataType]
    closures: list[StencilClosureType]
