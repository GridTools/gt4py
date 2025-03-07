# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts


class NamedRangeType(ts.TypeSpec):
    dim: common.Dimension


class DomainType(ts.DataType):
    dims: list[common.Dimension] | Literal["unknown"]


class OffsetLiteralType(ts.TypeSpec):
    value: ts.ScalarType | str


class IteratorType(ts.DataType, ts.CallableType):
    position_dims: list[common.Dimension] | Literal["unknown"]
    defined_dims: list[common.Dimension]
    element_type: ts.DataType


class ProgramType(ts.TypeSpec):
    params: dict[str, ts.DataType]
