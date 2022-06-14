# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
from typing import Any, List, Tuple, Union

from pydantic.class_validators import validator

import eve
from eve import Str, StrEnum, SymbolName, SymbolTableTrait, field, utils
from eve.type_definitions import SymbolRef
from gtc import common
from gtc.common import LocNode


@utils.noninstantiable
class Expr(common.Expr):
    pass


@utils.noninstantiable
class Stmt(common.Stmt):
    pass


class Offset(common.CartesianOffset):
    pass


class Literal(common.Literal, Expr):  # type: ignore
    pass


class LocalAccess(common.ScalarAccess, Expr):  # type: ignore
    pass


class VariableKOffset(common.VariableKOffset[Expr]):
    pass


class AccessorRef(common.FieldAccess[Expr, VariableKOffset], Expr):  # type: ignore
    pass


class BlockStmt(common.BlockStmt[Stmt], Stmt):
    pass


class AssignStmt(common.AssignStmt[Union[LocalAccess, AccessorRef], Expr], Stmt):
    @validator("left")
    def no_horizontal_offset_in_assignment(
        cls, v: Union[LocalAccess, AccessorRef]
    ) -> Union[LocalAccess, AccessorRef]:
        if isinstance(v, AccessorRef) and (v.offset.i != 0 or v.offset.j != 0):
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v

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


class Cast(common.Cast[Expr], Expr):  # type: ignore
    pass


class Temporary(LocNode):
    name: SymbolName
    dtype: common.DataType
    data_dims: Tuple[int, ...] = field(default_factory=tuple)


class GTLevel(LocNode):
    splitter: int
    offset: int

    @validator("offset")
    def offset_must_not_be_zero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("GridTools level offset must be != 0")
        return v


class GTInterval(LocNode):
    from_level: GTLevel
    to_level: GTLevel


class LocalVarDecl(LocNode):
    name: SymbolName
    dtype: common.DataType


class GTApplyMethod(LocNode):
    interval: GTInterval
    body: List[Stmt]
    local_variables: List[LocalVarDecl]


@enum.unique
class Intent(StrEnum):
    IN = "in"
    INOUT = "inout"


class GTExtent(LocNode):
    i: Tuple[int, int]
    j: Tuple[int, int]
    k: Tuple[int, int]

    @classmethod
    def zero(cls) -> "GTExtent":
        return cls(i=(0, 0), j=(0, 0), k=(0, 0))

    def __add__(self, offset: Union[common.CartesianOffset, VariableKOffset]) -> "GTExtent":
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
    name: SymbolName
    id: int  # noqa: A003  # shadowing python builtin
    intent: Intent
    extent: GTExtent
    ndim: int = 3


class GTParamList(LocNode):
    accessors: List[GTAccessor]


class GTFunctor(LocNode, SymbolTableTrait):
    name: SymbolName
    applies: List[GTApplyMethod]
    param_list: GTParamList


class Arg(LocNode):
    name: SymbolRef

    class Config(eve.concepts.FrozenModel.Config):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Arg):
            return NotImplemented
        return self.name == other.name


class ApiParamDecl(LocNode):
    name: SymbolName
    dtype: common.DataType

    def __init__(self, *args: Any, **kwargs: Any):
        if type(self) is ApiParamDecl:
            raise TypeError("Trying to instantiate `ApiParamDecl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(ApiParamDecl):
    dimensions: Tuple[bool, bool, bool]
    data_dims: Tuple[int, ...] = field(default_factory=tuple)


class GlobalParamDecl(ApiParamDecl):
    pass


class ComputationDecl(LocNode):
    name: SymbolName
    dtype = common.DataType.INT32
    kind = common.ExprKind.SCALAR


class Positional(ComputationDecl):
    axis_name: Str


class AxisLength(ComputationDecl):
    axis: int


class GTStage(LocNode):
    functor: SymbolRef
    # `args` are SymbolRefs to GTComputation `arguments` (interpreted as parameters)
    # or `temporaries`
    args: List[Arg]

    @validator("args")
    def has_args(cls, v: List[Arg]) -> List[Arg]:
        if not v:
            raise ValueError("At least one argument required")
        return v


class Cache(LocNode):
    name: SymbolRef  # symbol ref to GTComputation params or temporaries


class IJCache(Cache):
    pass


class KCache(Cache):
    fill: bool
    flush: bool


class GTMultiStage(LocNode):
    loop_order: common.LoopOrder
    stages: List[GTStage]
    caches: List[Cache]


class GTComputationCall(LocNode, SymbolTableTrait):
    # In the generated C++ code `arguments` represent both the arguments in the call to `run`
    # and the parameters of the function object.
    # We could represent this closer to the C++ code by splitting call and definition of the
    # function object.
    arguments: List[Arg]
    extra_decls: List[ComputationDecl]
    temporaries: List[Temporary]
    multi_stages: List[GTMultiStage]


class Program(LocNode, SymbolTableTrait):
    name: Str
    parameters: List[
        ApiParamDecl
    ]  # in the current implementation these symbols can be accessed by the functor body
    functors: List[GTFunctor]
    gt_computation: GTComputationCall  # here could be the CtrlFlow region

    _validate_dtype_is_set = common.validate_dtype_is_set()
    _validate_symbol_refs = common.validate_symbol_refs()
