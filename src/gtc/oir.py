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

"""
Optimizable Intermediate Representation (working title).

OIR represents a computation at the level of GridTools stages and multistages,
e.g. stage merging, staged computations to compute-on-the-fly, cache annotations, etc.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import root_validator, validator

from eve import Str, SymbolName, SymbolRef, SymbolTableTrait
from gtc import common
from gtc.common import AxisBound, LocNode


class Expr(common.Expr):
    dtype: Optional[common.DataType]

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


class AssignStmt(common.AssignStmt[Union[ScalarAccess, FieldAccess], Expr], Stmt):
    @validator("left")
    def no_horizontal_offset_in_assignment(
        cls, v: Union[ScalarAccess, FieldAccess]
    ) -> Union[ScalarAccess, FieldAccess]:
        if isinstance(v, FieldAccess) and (v.offset.i != 0 or v.offset.j != 0):
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v

    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


# TODO(havogt) consider introducing BlockStmt
# class BlockStmt(common.BlockStmt[Stmt], Stmt):
#     pass

# TODO(havogt) should we have an IfStmt or is masking the final solution?
# class IfStmt(common.IfStmt[List[Stmt], Expr], Stmt):  # TODO replace List[Stmt] by BlockStmt?
#     pass


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
    dtype: common.DataType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    # TODO dimensions
    pass


class ScalarDecl(Decl):
    pass


class LocalScalar(Decl):
    pass


class Temporary(Decl):
    pass


class Interval(LocNode):
    start: AxisBound
    end: AxisBound

    @root_validator
    def check(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        start, end = values["start"], values["end"]
        if start.level == common.LevelMarker.END and end.level == common.LevelMarker.START:
            raise ValueError("Start level must be smaller or equal end level")
        if start.level == end.level and not start.offset < end.offset:
            raise ValueError(
                "Start offset must be smaller than end offset if start and end levels are equal"
            )
        return values


class CartesianIterationSpace(LocNode):
    i_interval: Interval
    j_interval: Interval

    @validator("i_interval", "j_interval")
    def minimum_domain(cls, v: Interval) -> Interval:
        if (
            v.start.level != common.LevelMarker.START
            or v.start.offset > 0
            or v.end.level != common.LevelMarker.END
            or v.end.offset < 0
        ):
            raise ValueError("iteration space must include the whole domain")
        return v

    @classmethod
    def domain(cls) -> "CartesianIterationSpace":
        return CartesianIterationSpace(
            i_interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
            j_interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
        )

    @classmethod
    def from_offset(cls, offset: common.CartesianOffset) -> "CartesianIterationSpace":

        return CartesianIterationSpace(
            i_interval=Interval(
                start=AxisBound.from_start(min(0, offset.i)),
                end=AxisBound.from_end(max(0, offset.i)),
            ),
            j_interval=Interval(
                start=AxisBound.from_start(min(0, offset.j)),
                end=AxisBound.from_end(max(0, offset.j)),
            ),
        )

    def compose(self, other: "CartesianIterationSpace") -> "CartesianIterationSpace":
        i_interval = Interval(
            start=AxisBound.from_start(
                self.i_interval.start.offset + other.i_interval.start.offset,
            ),
            end=AxisBound.from_end(
                self.i_interval.end.offset + other.i_interval.end.offset,
            ),
        )
        j_interval = Interval(
            start=AxisBound.from_start(
                self.j_interval.start.offset + other.j_interval.start.offset,
            ),
            end=AxisBound.from_end(
                self.j_interval.end.offset + other.j_interval.end.offset,
            ),
        )
        return CartesianIterationSpace(i_interval=i_interval, j_interval=j_interval)

    def __or__(self, other: "CartesianIterationSpace") -> "CartesianIterationSpace":
        i_interval = Interval(
            start=AxisBound.from_start(
                min(
                    self.i_interval.start.offset,
                    other.i_interval.start.offset,
                )
            ),
            end=AxisBound.from_end(
                max(
                    self.i_interval.end.offset,
                    other.i_interval.end.offset,
                )
            ),
        )
        j_interval = Interval(
            start=AxisBound.from_start(
                min(
                    self.j_interval.start.offset,
                    other.j_interval.start.offset,
                )
            ),
            end=AxisBound.from_end(
                max(
                    self.j_interval.end.offset,
                    other.j_interval.end.offset,
                )
            ),
        )
        return CartesianIterationSpace(i_interval=i_interval, j_interval=j_interval)


class HorizontalExecution(LocNode):
    body: List[Stmt]
    mask: Optional[Expr]
    declarations: List[LocalScalar]
    iteration_space: Optional[CartesianIterationSpace]

    @validator("mask")
    def mask_is_boolean_field_expr(cls, v: Optional[Expr]) -> Optional[Expr]:
        if v:
            if v.dtype != common.DataType.BOOL:
                raise ValueError("Mask must be a boolean expression.")
        return v


class CacheDesc(LocNode):
    name: SymbolRef


class IJCache(CacheDesc):
    pass


class KCache(CacheDesc):
    fill: bool
    flush: bool


class VerticalLoopSection(LocNode):
    interval: Interval
    horizontal_executions: List[HorizontalExecution]


class VerticalLoop(LocNode):
    loop_order: common.LoopOrder
    sections: List[VerticalLoopSection]
    caches: List[CacheDesc]

    @validator("sections")
    def nonempty_loop(cls, v: List[VerticalLoopSection]) -> List[VerticalLoopSection]:
        if not v:
            raise ValueError("Empty vertical loop is not allowed")
        return v

    @root_validator
    def valid_section_intervals(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        loop_order, sections = values["loop_order"], values["sections"]
        starts, ends = zip(*((s.interval.start, s.interval.end) for s in sections))
        if loop_order == common.LoopOrder.BACKWARD:
            starts, ends = starts[:-1], ends[1:]
        else:
            starts, ends = starts[1:], ends[:-1]

        if not all(
            start.level == end.level and start.offset == end.offset
            for start, end in zip(starts, ends)
        ):
            raise ValueError("Loop intervals not contiguous or in wrong order")
        return values


class Stencil(LocNode, SymbolTableTrait):
    name: Str
    # TODO: fix to be List[Union[ScalarDecl, FieldDecl]]
    params: List[Decl]
    vertical_loops: List[VerticalLoop]
    declarations: List[Temporary]

    _validate_dtype_is_set = common.validate_dtype_is_set()
    _validate_symbol_refs = common.validate_symbol_refs()
