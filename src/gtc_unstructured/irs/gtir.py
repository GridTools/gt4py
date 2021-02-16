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
from typing import List, Optional, Union, ClassVar, Dict

from devtools import debug  # noqa: F401
from pydantic import root_validator

from eve import Node, Str, StrEnum, SymbolTableTrait
from eve.type_definitions import SymbolName, SymbolRef
from gtc_unstructured.irs import common
from gtc import common as stable_gtc_common


class NativeFuncCall(GenericNode, Generic[ExprT]):
    func: NativeFunction
    args: List[ExprT]

    @root_validator(skip_on_failure=True)
    def arity_check(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
        if values["func"].arity != len(values["args"]):
            raise ValueError(
                "{} accepts {} arguments, {} where passed.".format(
                    values["func"], values["func"].arity, len(values["args"])
                )
            )
        return values

@enum.unique
class NativeFunction(StrEnum):
    ABS = "abs"
    MIN = "min"
    MAX = "max"
    MOD = "mod"

    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ARCSIN = "arcsin"
    ARCCOS = "arccos"
    ARCTAN = "arctan"

    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"

    ISFINITE = "isfinite"
    ISINF = "isinf"
    ISNAN = "isnan"
    FLOOR = "floor"
    CEIL = "ceil"
    TRUNC = "trunc"

    IR_OP_TO_NUM_ARGS: ClassVar[Dict["NativeFunction", int]]

    @property
    def arity(self) -> int:
        return self.IR_OP_TO_NUM_ARGS[self]


NativeFunction.IR_OP_TO_NUM_ARGS = {
    NativeFunction.ABS: 1,
    NativeFunction.MIN: 2,
    NativeFunction.MAX: 2,
    NativeFunction.MOD: 2,
    NativeFunction.SIN: 1,
    NativeFunction.COS: 1,
    NativeFunction.TAN: 1,
    NativeFunction.ARCSIN: 1,
    NativeFunction.ARCCOS: 1,
    NativeFunction.ARCTAN: 1,
    NativeFunction.SQRT: 1,
    NativeFunction.EXP: 1,
    NativeFunction.LOG: 1,
    NativeFunction.ISFINITE: 1,
    NativeFunction.ISINF: 1,
    NativeFunction.ISNAN: 1,
    NativeFunction.FLOOR: 1,
    NativeFunction.CEIL: 1,
    NativeFunction.TRUNC: 1,
}


class Expr(Node):
    location_type: common.LocationType
    pass


class Stmt(Node):
    location_type: common.LocationType
    pass


class Literal(common.Literal, Expr):
    pass


class LocationRef(Node):
    name: SymbolRef  # to PrimaryLocation or LocationComprehension


# TODO indirection not needed?
class ConnectivityRef(Node):
    name: SymbolRef


class NeighborVectorAccess(Expr):
    exprs: List[Expr]
    location_ref: LocationRef

    # TODO check that size of list equals connectivity max_neighbors


# TODO using a LocalFieldDecl seems natural, but doesn't work with the parallel model (it would live in its own scope)
# class LocalFieldDecl(Stmt):
#     name: SymbolName
#     connectivity: ConnectivityRef
#     init: List[Expr]


@enum.unique
class ReduceOperator(StrEnum):
    """Reduction operator identifier."""

    ADD = "ADD"
    MUL = "MUL"
    MAX = "MAX"
    MIN = "MIN"


class PrimaryLocation(Node):
    name: SymbolName
    location_type: common.LocationType


class LocationComprehension(Node):
    name: SymbolName
    of: ConnectivityRef


class NeighborReduce(Expr):
    operand: Expr
    op: ReduceOperator
    neighbors: LocationComprehension

    # TODO to validate we would need to lookup the connectivity,
    # i.e. can only be done when symbols are resolvable
    # @root_validator(pre=True)
    # def check_location_type(cls, values):
    #     if values["neighbors"].chain.elements[-1] != values["operand"].location_type:
    #         raise ValueError("Location type mismatch")
    #     return values


class FieldAccess(Expr):
    name: SymbolRef
    subscript: List[LocationRef]


# to SparseField (or TODO LocalField)
class NeighborAssignStmt(common.AssignStmt[FieldAccess, Expr], Stmt, SymbolTableTrait):
    neighbors: LocationComprehension


class AssignStmt(common.AssignStmt[FieldAccess, Expr], Stmt):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    pass


class VerticalDimension(Node):
    pass


class HorizontalDimension(Node):
    primary: common.LocationType


class Dimensions(Node):
    horizontal: Optional[HorizontalDimension]
    vertical: Optional[VerticalDimension]
    # other: TODO


class UField(Node):
    name: SymbolName
    vtype: common.DataType  # TODO rename vtype
    dimensions: Dimensions


class SparseField(Node):
    name: SymbolName
    connectivity: SymbolRef
    vtype: common.DataType  # TODO rename vtype
    dimensions: Dimensions


class TemporaryField(UField):
    pass


class TemporarySparseField(SparseField):
    pass


class HorizontalLoop(Node):
    stmt: Stmt
    location: PrimaryLocation

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["stmt"].location_type != values["location"].location_type:
            raise ValueError("Location type mismatch")
        return values


class VerticalLoop(Node):
    # each statement inside a `with location_type` is interpreted as a full horizontal loop (see parallel model)
    horizontal_loops: List[HorizontalLoop]
    loop_order: common.LoopOrder


class Stencil(Node):
    vertical_loops: List[VerticalLoop]


class Connectivity(Node):
    name: SymbolName
    primary: common.LocationType
    secondary: common.LocationType
    max_neighbors: int
    has_skip_values: bool


class Computation(Node, SymbolTableTrait):
    name: Str
    connectivities: List[Connectivity]
    params: List[Union[UField, SparseField]]
    declarations: Optional[List[Union[TemporaryField, TemporarySparseField]]]
    stencils: List[Stencil]

    _validate_symbol_refs = stable_gtc_common.validate_symbol_refs()
