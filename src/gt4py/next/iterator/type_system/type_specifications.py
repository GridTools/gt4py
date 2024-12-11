# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Literal, Optional

from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class NamedRangeType(ts.TypeSpec):
    dim: common.Dimension


@dataclasses.dataclass(frozen=True)
class DomainType(ts.DataType):
    dims: list[common.Dimension] | Literal["unknown"]


@dataclasses.dataclass(frozen=True)
class OffsetLiteralType(ts.TypeSpec):
    value: ts.ScalarType | common.Dimension


@dataclasses.dataclass(frozen=True)
class ListType(ts.DataType):
    element_type: ts.DataType
    # TODO(havogt): the `offset_type` is not yet used in type_inference,
    # it is meant to describe the neighborhood (via the local dimension)
    offset_type: Optional[common.Dimension] = None


@dataclasses.dataclass(frozen=True)
class IteratorType(ts.DataType, ts.CallableType):
    position_dims: list[common.Dimension] | Literal["unknown"]
    defined_dims: list[common.Dimension]
    element_type: ts.DataType


@dataclasses.dataclass(frozen=True)
class ProgramType(ts.TypeSpec):
    params: dict[str, ts.DataType]
