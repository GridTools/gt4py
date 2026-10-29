# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

from gt4py.eve import datamodels as eve_datamodels
from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts


class NamedRangeType(ts.TypeSpec):
    dim: common.DimensionRef = ts.dimension_field()


class OffsetLiteralType(ts.TypeSpec):
    value: ts.ScalarType | str


class CartesianOffsetType(ts.TypeSpec):
    domain: common.DimensionRef = ts.dimension_field()
    codomain: common.DimensionRef = ts.dimension_field()


class IteratorType(ts.DataType, ts.CallableType):
    position_dims: list[common.DimensionRef] | Literal["unknown"] = eve_datamodels.field(
        converter=lambda dims: dims if dims == "unknown" else common.dimension_refs(dims)
    )
    defined_dims: list[common.DimensionRef] = ts.dimension_list_field()
    element_type: ts.DataType


class ProgramType(ts.TypeSpec):
    params: dict[str, ts.DataType]
