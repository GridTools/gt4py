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

from typing import Any, Dict, List, Tuple, Union

from pydantic import root_validator, validator

from eve import Str, SymbolName, SymbolRef, SymbolTableTrait, field, utils
from eve.typingx import RootValidatorValuesType
from gtc import common
from gtc.common import AxisBound, LocNode


@utils.noninstantiable
class Expr(common.Expr):
    pass


@utils.noninstantiable
class Stmt(common.Stmt):
    pass


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


class MaskStmt(Stmt):
    mask: Expr
    body: List[Stmt]

    @validator("mask")
    def mask_is_boolean_field_expr(cls, v: Expr) -> Expr:
        if v.dtype != common.DataType.BOOL:
            raise ValueError("Mask must be a boolean expression.")
        return v


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
    dimensions: Tuple[bool, bool, bool]
    data_dims: Tuple[int, ...] = field(default_factory=tuple)


class ScalarDecl(Decl):
    pass


class LocalScalar(Decl):
    pass


class Temporary(FieldDecl):
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

    def covers(self, other: "Interval") -> bool:
        outer_starts_lower = self.start < other.start or self.start == other.start
        outer_ends_higher = self.end > other.end or self.end == other.end
        return outer_starts_lower and outer_ends_higher

    def intersects(self, other: "Interval") -> bool:
        return not (other.start >= self.end or self.start >= other.end)

    def shifted(self, offset: int) -> "Interval":
        start = AxisBound(level=self.start.level, offset=self.start.offset + offset)
        end = AxisBound(level=self.end.level, offset=self.end.offset + offset)
        return Interval(start=start, end=end)

    @classmethod
    def full(cls):
        return cls(start=AxisBound.start(), end=AxisBound.end())


class HorizontalExecution(LocNode):
    body: List[Stmt]
    declarations: List[LocalScalar]


def horizontal_intervals_are_disjoint(
    self_interval: common.HorizontalInterval,
    other_interval: common.HorizontalInterval,
) -> bool:
    DOMAIN_SIZE = 1000
    OFFSET_SIZE = 1000

    if isinstance(self_interval.start, AxisBound):
        s_start = (
            0 + self_interval.start.offset
            if self_interval.start.level == common.LevelMarker.START
            else DOMAIN_SIZE + self_interval.start.offset
        )
    else:
        s_start = -OFFSET_SIZE

    if isinstance(self_interval.end, AxisBound):
        s_end = (
            0 + self_interval.end.offset
            if self_interval.end.level == common.LevelMarker.START
            else DOMAIN_SIZE + self_interval.end.offset
        )
    else:
        s_end = DOMAIN_SIZE + OFFSET_SIZE

    if isinstance(other_interval.start, AxisBound):
        o_start = (
            0 + other_interval.start.offset
            if other_interval.start.level == common.LevelMarker.START
            else DOMAIN_SIZE + other_interval.start.offset
        )
    else:
        o_start = -OFFSET_SIZE

    if isinstance(other_interval.end, AxisBound):
        o_end = (
            0 + other_interval.end.offset
            if other_interval.end.level == common.LevelMarker.START
            else DOMAIN_SIZE + other_interval.end.offset
        )
    else:
        o_end = -OFFSET_SIZE

    return not (s_start <= o_start < s_end) and not (o_start <= s_start < o_end)


class HorizontalMask(common.HorizontalMask[Expr], Expr):
    pass


class HorizontalSpecialization(Expr):
    mask: HorizontalMask
    expr: Expr

    @root_validator(skip_on_failure=True)
    def dtype_propagation(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
        values["dtype"] = values["expr"].dtype
        return values

    @root_validator(pre=True)
    def kind_propagation(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
        values["kind"] = values["expr"].kind
        return values


def horizontal_specializations_are_disjoint(
    self: HorizontalSpecialization, other: HorizontalSpecialization
) -> bool:
    return any(
        horizontal_intervals_are_disjoint(interval1, interval2)
        for interval1, interval2 in zip(self.mask.intervals, other.mask.intervals)
    )


class HorizontalSwitch(Expr):
    values: List[HorizontalSpecialization]
    default: Expr

    @root_validator(skip_on_failure=True)
    def dtype_propagation(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
        values["dtype"] = common.verify_and_get_common_dtype(cls, values["values"], strict=True)
        return values

    @root_validator(pre=True)
    def kind_propagation(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
        values["kind"] = common.compute_kind(values["values"])
        return values

    @validator("values")
    def check_disjointness(
        cls, values: List[HorizontalSpecialization]
    ) -> List[HorizontalSpecialization]:
        for i, value in enumerate(values[:-1]):
            for other in values[i + 1 :]:
                if not horizontal_specializations_are_disjoint(value, other):
                    raise ValueError("Horizontal switch values must be disjoint specializations.")
        return values


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
    _validate_lvalue_dims = common.validate_lvalue_dims(VerticalLoop, FieldDecl)
