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
from typing import Literal

from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class NamedRangeType(ts.TypeSpec):
    dim: common.Dimension


@dataclasses.dataclass(frozen=True)
class DomainType(ts.DataType):
    dims: list[common.Dimension]


@dataclasses.dataclass(frozen=True)
class OffsetLiteralType(ts.TypeSpec):
    value: ts.ScalarType | common.Dimension


@dataclasses.dataclass(frozen=True)
class ListType(ts.DataType):
    element_type: ts.DataType


@dataclasses.dataclass(frozen=True)
class IteratorType(ts.DataType, ts.CallableType):
    position_dims: list[common.Dimension] | Literal["unknown"]
    defined_dims: list[common.Dimension]
    element_type: ts.DataType


@dataclasses.dataclass(frozen=True)
class StencilClosureType(ts.TypeSpec):
    domain: DomainType
    stencil: ts.FunctionType
    output: ts.FieldType | ts.TupleType
    inputs: list[ts.FieldType]

    def __post_init__(self):
        # local import to avoid importing type_info from a type_specification module
        from gt4py.next.type_system import type_info

        for i, el_type in enumerate(type_info.primitive_constituents(self.output)):
            assert isinstance(
                el_type, ts.FieldType
            ), f"All constituent types must be field types, but the {i}-th element is of type '{el_type}'."


# TODO(tehrengruber): Remove after new ITIR format with apply_stencil is used everywhere
@dataclasses.dataclass(frozen=True)
class FencilType(ts.TypeSpec):
    params: dict[str, ts.DataType]
    closures: list[StencilClosureType]


@dataclasses.dataclass(frozen=True)
class ProgramType(ts.TypeSpec):
    params: dict[str, ts.DataType]
