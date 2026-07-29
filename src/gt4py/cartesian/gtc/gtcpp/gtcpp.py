# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
from typing import Any, List, Tuple, Union

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.common import LocNode
from gt4py.eve import datamodels


@eve.utils.noninstantiable
class Expr(common.Expr):
    pass


@eve.utils.noninstantiable
class Stmt(common.Stmt):
    pass


class Offset(common.CartesianOffset):
    pass


class Literal(common.Literal, Expr):
    pass


class LocalAccess(common.ScalarAccess, Expr):
    pass


class VariableKOffset(common.VariableKOffset[Expr]):
    pass


class AccessorRef(common.FieldAccess[Expr, VariableKOffset], Expr):
    pass


class BlockStmt(common.BlockStmt[Stmt], Stmt):
    pass


class AssignStmt(common.AssignStmt[Union[LocalAccess, AccessorRef], Expr], Stmt):
    @datamodels.validator("left")
    def no_horizontal_offset_in_assignment(
        self, attribute: datamodels.Attribute, value: Union[LocalAccess, AccessorRef]
    ) -> None:
        if isinstance(value, AccessorRef):
            offsets = value.offset.to_dict()
            if offsets["i"] != 0 or offsets["j"] != 0:
                raise ValueError("Lhs of assignment must not have a horizontal offset.")

    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


class IfStmt(common.IfStmt[BlockStmt, Expr], Stmt):
    pass


class While(common.While[Stmt, Expr], Stmt):
    pass


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


class Cast(common.Cast[Expr], Expr):
    pass


class Temporary(LocNode):
    name: eve.Coerced[eve.SymbolName]
    dtype: common.DataType
    dimensions: tuple[bool, bool, bool] = (True, True, True)
    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)


class GTLevel(LocNode):
    splitter: int
    offset: int

    @datamodels.validator("offset")
    def offset_must_not_be_zero(self, attribute: datamodels.Attribute, value: int) -> None:
        if value == 0:
            raise ValueError("GridTools level offset must be != 0")


class GTInterval(LocNode):
    from_level: GTLevel
    to_level: GTLevel


class LocalVarDecl(LocNode):
    name: eve.SymbolName
    dtype: common.DataType


class GTApplyMethod(LocNode):
    interval: GTInterval
    body: List[Stmt]
    local_variables: List[LocalVarDecl]


@enum.unique
class Intent(eve.StrEnum):
    IN = "in"
    INOUT = "inout"


class GTExtent(LocNode):
    i: Tuple[int, int]
    j: Tuple[int, int]
    k: Tuple[int, int]

    @classmethod
    def zero(cls) -> GTExtent:
        return cls(i=(0, 0), j=(0, 0), k=(0, 0))

    def __add__(self, offset: Union[common.CartesianOffset, VariableKOffset]) -> GTExtent:
        if isinstance(offset, common.CartesianOffset):
            return GTExtent(
                i=(min(self.i[0], offset.i), max(self.i[1], offset.i)),
                j=(min(self.j[0], offset.j), max(self.j[1], offset.j)),
                k=(min(self.k[0], offset.k), max(self.k[1], offset.k)),
            )
        elif isinstance(offset, VariableKOffset):
            MAX_OFFSET = 1000
            return GTExtent(i=self.i, j=self.j, k=(-MAX_OFFSET, MAX_OFFSET))
        else:
            raise AssertionError(f"Unrecognized offset type: {type(offset)}")


class GTAccessor(LocNode):
    name: eve.Coerced[eve.SymbolName]
    id: int  # shadowing python builtin
    intent: Intent
    extent: GTExtent
    ndim: int = 3


class GTParamList(LocNode):
    accessors: List[GTAccessor]


class GTFunctor(LocNode, eve.SymbolTableTrait):
    name: eve.Coerced[eve.SymbolName]
    applies: List[GTApplyMethod]
    param_list: GTParamList


class Arg(LocNode):
    name: eve.Coerced[eve.SymbolRef]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Arg):
            return NotImplemented
        return self.name == other.name


class ApiParamDecl(LocNode):
    name: eve.Coerced[eve.SymbolName]
    dtype: common.DataType

    def __init__(self, *args: Any, **kwargs: Any):
        if type(self) is ApiParamDecl:
            raise TypeError("Trying to instantiate `ApiParamDecl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(ApiParamDecl):
    dimensions: Tuple[bool, bool, bool]
    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)


class GlobalParamDecl(ApiParamDecl):
    pass


class ComputationDecl(LocNode):
    name: eve.Coerced[eve.SymbolName]
    dtype = common.DataType.INT32
    kind = common.ExprKind.SCALAR


class Positional(ComputationDecl):
    axis_name: str


class AxisLength(ComputationDecl):
    axis: int


class GTStage(LocNode):
    functor: eve.Coerced[eve.SymbolRef]
    # `args` are SymbolRefs to GTComputation `arguments` (interpreted as parameters)
    # or `temporaries`
    args: List[Arg]

    @datamodels.validator("args")
    def has_args(self, attribute: datamodels.Attribute, value: List[Arg]) -> None:
        if not value:
            raise ValueError("At least one argument required")


class Cache(LocNode):
    name: eve.Coerced[eve.SymbolRef]  # symbol ref to GTComputation params or temporaries


class IJCache(Cache):
    pass


class KCache(Cache):
    fill: bool
    flush: bool


class GTMultiStage(LocNode):
    loop_order: common.LoopOrder
    stages: List[GTStage]
    caches: List[Cache]


class GTComputationCall(LocNode, eve.SymbolTableTrait):
    # In the generated C++ code `arguments` represent both the arguments in the call to `run`
    # and the parameters of the function object.
    # We could represent this closer to the C++ code by splitting call and definition of the
    # function object.
    arguments: List[Arg]
    extra_decls: List[ComputationDecl]
    temporaries: List[Temporary]
    multi_stages: List[GTMultiStage]


class Program(LocNode, eve.ValidatedSymbolTableTrait):
    name: str
    parameters: List[
        ApiParamDecl
    ]  # in the current implementation these symbols can be accessed by the functor body
    functors: List[GTFunctor]
    gt_computation: GTComputationCall  # here could be the CtrlFlow region

    _validate_dtype_is_set = common.validate_dtype_is_set()
