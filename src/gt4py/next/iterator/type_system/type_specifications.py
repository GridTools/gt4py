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
    value: ts.ScalarType | common.Dimension


class IteratorType(ts.DataType, ts.CallableType):
    position_dims: list[common.Dimension] | Literal["unknown"]
    defined_dims: list[common.Dimension]
    element_type: ts.DataType


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
class FencilType(ts.TypeSpec):
    params: dict[str, ts.DataType]
    closures: list[StencilClosureType]


class ProgramType(ts.TypeSpec):
    params: dict[str, ts.DataType]
