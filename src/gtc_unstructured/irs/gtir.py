# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

import enum
from typing import List, Optional, Union

from devtools import debug  # noqa: F401
from pydantic import root_validator, validator

from eve import Node, Str, StrEnum
from gtc_unstructured.irs import common


class Expr(Node):
    location_type: common.LocationType
    pass


class Stmt(Node):
    location_type: common.LocationType
    pass


class Literal(common.Literal, Expr):
    pass


class NeighborChain(Node):
    elements: List[common.LocationType]

    @validator("elements")
    def not_empty(cls, elements):
        if len(elements) < 1:
            raise ValueError("NeighborChain must contain at least one locations")
        return elements


@enum.unique
class ReduceOperator(StrEnum):
    """Reduction operator identifier."""

    ADD = "ADD"
    MUL = "MUL"
    MAX = "MAX"
    MIN = "MIN"


class Domain(Node):
    pass


class LocationRef(Node):
    name: str


class LocationComprehension(Node):
    name: str
    chain: NeighborChain
    of: Union[LocationRef, Domain]


class NeighborReduce(Expr):
    operand: Expr
    op: ReduceOperator
    neighbors: LocationComprehension

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["neighbors"].chain.elements[-1] != values["operand"].location_type:
            raise ValueError("Location type mismatch")
        return values


class FieldAccess(Expr):
    name: Str  # via symbol table
    subscript: List[LocationRef]  # maybe remove the separate LocationRef


class AssignStmt(common.AssignStmt[FieldAccess, Expr], Stmt):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    pass


class VerticalDimension(Node):
    pass


class HorizontalDimension(Node):
    primary: common.LocationType
    secondary: Optional[NeighborChain]


class Dimensions(Node):
    horizontal: Optional[HorizontalDimension]
    vertical: Optional[VerticalDimension]
    # other: TODO


class UField(Node):
    name: Str
    vtype: common.DataType
    dimensions: Dimensions


class TemporaryField(UField):
    pass


class HorizontalLoop(Node):
    stmt: Stmt
    location: LocationComprehension

    @root_validator(pre=True)
    def check_location_type(cls, values):
        # Don't infer here! The location type of the loop should always come from the frontend!
        if len(values["location"].chain.elements) != 1:
            raise ValueError(
                "LocationComprehension on HorizontalLoop must have NeighborChain of length 1"
            )
        if values["stmt"].location_type != values["location"].chain.elements[0]:
            raise ValueError("Location type mismatch")
        return values


class VerticalLoop(Node):
    # each statement inside a `with location_type` is interpreted as a full horizontal loop (see parallel model of SIR)
    horizontal_loops: List[HorizontalLoop]
    loop_order: common.LoopOrder


class Stencil(Node):
    vertical_loops: List[VerticalLoop]


class Computation(Node):
    name: Str
    params: List[UField]
    declarations: Optional[List[TemporaryField]]
    stencils: List[Stencil]
