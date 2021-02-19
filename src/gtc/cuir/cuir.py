# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import Any, List, Optional, Tuple, Union

from pydantic import validator

from eve import Str, SymbolName, SymbolRef, SymbolTableTrait
from gtc import common
from gtc.common import AxisBound, DataType, LocNode, LoopOrder


class Expr(common.Expr):
    dtype: common.DataType

    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(common.Stmt):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Stmt` abstract class.")
        super().__init__(*args, **kwargs)


class Literal(common.Literal, Expr):  # type: ignore
    pass


class ScalarAccess(common.ScalarAccess, Expr):  # type: ignore
    pass


class FieldAccess(common.FieldAccess, Expr):  # type: ignore
    pass


class IJCacheAccess(Expr):
    name: SymbolRef
    offset: Tuple[int, int]
    kind = common.ExprKind.FIELD


class KCacheAccess(Expr):
    name: SymbolRef
    offset: int
    kind = common.ExprKind.FIELD


class AssignStmt(
    common.AssignStmt[Union[ScalarAccess, FieldAccess, IJCacheAccess, KCacheAccess], Expr], Stmt
):
    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class Cast(common.Cast[Expr], Expr):  # type: ignore
    pass


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


class Decl(LocNode):
    name: SymbolName
    dtype: DataType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    pass


class ScalarDecl(Decl):
    pass


class LocalScalar(Decl):
    pass


class Temporary(Decl):
    pass


class IJCacheDecl(Decl):
    pass


class KCacheDecl(Decl):
    min_offset: int
    max_offset: int


class IJExtent(LocNode):
    i: Tuple[int, int]
    j: Tuple[int, int]

    @classmethod
    def zero(cls) -> "IJExtent":
        return cls(i=(0, 0), j=(0, 0))

    def union(*extents: "IJExtent") -> "IJExtent":
        return IJExtent(
            i=(min(e.i[0] for e in extents), max(e.i[1] for e in extents)),
            j=(min(e.j[0] for e in extents), max(e.j[1] for e in extents)),
        )

    def __add__(self, other: "IJExtent") -> "IJExtent":
        return IJExtent(
            i=(self.i[0] + other.i[0], self.i[1] + other.i[1]),
            j=(self.j[0] + other.j[0], self.j[1] + other.j[1]),
        )


class HorizontalExecution(LocNode):
    body: List[Stmt]
    mask: Optional[Expr]
    declarations: List[LocalScalar]
    extent: Optional[IJExtent]


class VerticalLoopSection(LocNode):
    start: AxisBound
    end: AxisBound
    horizontal_executions: List[HorizontalExecution]


class VerticalLoop(LocNode):
    loop_order: LoopOrder
    sections: List[VerticalLoopSection]
    ij_caches: List[IJCacheDecl]
    k_caches: List[KCacheDecl]


class Kernel(LocNode):
    vertical_loops: List[VerticalLoop]

    @validator("vertical_loops")
    def check_parallelism(cls, v: List[VerticalLoop]) -> List[VerticalLoop]:
        parallel = [loop.loop_order == LoopOrder.PARALLEL for loop in v]
        if any(parallel) and not all(parallel):
            raise ValueError("Mixed k-parallelism in kernel")
        return v


class Program(LocNode, SymbolTableTrait):
    name: Str
    params: List[Decl]
    temporaries: List[Temporary]
    kernels: List[Kernel]
