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

"""
Optimizable Intermediate Representation (working title).

OIR represents a computation at the level of GridTools stages and multistages,
e.g. stage merging, staged computations to compute-on-the-fly, cache annotations, etc.
"""

from typing import Any, List, Optional, Tuple, Type, Union

import eve
from eve import datamodels
from gtc import common
from gtc.common import AxisBound, LocNode


@eve.utils.noninstantiable
class Expr(common.Expr):
    pass


@eve.utils.noninstantiable
class Stmt(common.Stmt):
    pass


class Literal(common.Literal, Expr):  # type: ignore
    pass


class ScalarAccess(common.ScalarAccess, Expr):  # type: ignore
    pass


class VariableKOffset(common.VariableKOffset[Expr]):
    pass


class FieldAccess(common.FieldAccess[Expr, VariableKOffset], Expr):  # type: ignore
    pass


class AssignStmt(common.AssignStmt[Union[ScalarAccess, FieldAccess], Expr], Stmt):
    @datamodels.validator("left")
    def no_horizontal_offset_in_assignment(
        self, attribute: datamodels.Attribute, value: Union[ScalarAccess, FieldAccess]
    ) -> None:
        if isinstance(value, FieldAccess):
            offsets = value.offset.to_dict()
            if offsets["i"] != 0 or offsets["j"] != 0:
                raise ValueError("Lhs of assignment must not have a horizontal offset.")

    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


class MaskStmt(Stmt):
    mask: Expr
    body: List[Stmt]

    @datamodels.validator("mask")
    def mask_is_boolean_field_expr(self, attribute: datamodels.Attribute, v: Expr) -> None:
        if v.dtype != common.DataType.BOOL:
            raise ValueError("Mask must be a boolean expression.")


class HorizontalRestriction(common.HorizontalRestriction[Stmt], Stmt):
    pass


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


class While(common.While[Stmt, Expr], Stmt):
    pass


class Decl(LocNode):
    name: eve.Coerced[eve.SymbolName]
    dtype: common.DataType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    dimensions: Tuple[bool, bool, bool]
    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)


class ScalarDecl(Decl):
    pass


class LocalScalar(Decl):
    pass


class Temporary(FieldDecl):
    pass


def _check_interval(instance: Union["Interval", "UnboundedInterval"]) -> None:
    start, end = instance.start, instance.end
    if (
        start is not None
        and start.level == common.LevelMarker.END
        and end is not None
        and end.level == common.LevelMarker.START
    ):
        raise ValueError("Start level must be smaller or equal end level")

    if (
        start is not None
        and end is not None
        and start.level == end.level
        and not start.offset < end.offset
    ):
        raise ValueError(
            "Start offset must be smaller than end offset if start and end levels are equal"
        )


class Interval(LocNode):
    start: AxisBound
    end: AxisBound

    @datamodels.root_validator
    @classmethod
    def check(cls: Type["Interval"], instance: "Interval") -> None:
        _check_interval(instance)

    def covers(self, other: "Interval") -> bool:
        outer_starts_lower = self.start < other.start or self.start == other.start
        outer_ends_higher = self.end > other.end or self.end == other.end
        return outer_starts_lower and outer_ends_higher

    def intersects(self, other: "Interval") -> bool:
        return not (other.start >= self.end or self.start >= other.end)

    def shifted(self, offset: Optional[int]) -> Union["Interval", "UnboundedInterval"]:
        if offset is None:
            return UnboundedInterval()
        start = AxisBound(level=self.start.level, offset=self.start.offset + offset)
        end = AxisBound(level=self.end.level, offset=self.end.offset + offset)
        return Interval(start=start, end=end)

    @classmethod
    def full(cls):
        return cls(start=AxisBound.start(), end=AxisBound.end())


class UnboundedInterval:
    start: Optional[AxisBound] = None
    end: Optional[AxisBound] = None

    @datamodels.root_validator
    @classmethod
    def check(cls: Type["UnboundedInterval"], instance: "UnboundedInterval") -> None:
        _check_interval(instance)

    def covers(self, other: Union[Interval, "UnboundedInterval"]) -> bool:
        if self.start is None and self.end is None:
            return True
        if (
            self.end is None
            and other.start is not None
            and self.start is not None
            and other.start >= self.start
        ):
            return True
        if (
            self.start is None
            and other.end is not None
            and self.end is not None
            and other.end <= self.end
        ):
            return True

        # at this point, we know self is actually bounded, so can't cover unbounded intervals
        if other.start is None or other.end is None:
            return False

        assert isinstance(other, Interval)
        return Interval(start=self.start, end=self.end).covers(other)

    def intersects(self, other: Union[Interval, "UnboundedInterval"]) -> bool:
        no_overlap_high = (
            self.end is not None and other.start is not None and other.start >= self.end
        )
        no_overlap_low = (
            self.start is not None and other.end is not None and self.start >= other.end
        )
        return not (no_overlap_low or no_overlap_high)

    def shifted(self, offset: Optional[int]) -> "UnboundedInterval":
        if offset is None:
            return UnboundedInterval()

        start = (
            None
            if self.start is None
            else AxisBound(level=self.start.level, offset=self.start.offset + offset)
        )
        end = (
            None
            if self.end is None
            else AxisBound(level=self.end.level, offset=self.end.offset + offset)
        )
        return UnboundedInterval(start=start, end=end)  # type: ignore

    @classmethod
    def full(cls):
        return cls()


class HorizontalExecution(LocNode, eve.SymbolTableTrait):
    body: List[Stmt]
    declarations: List[LocalScalar]


class CacheDesc(LocNode):
    name: eve.Coerced[eve.SymbolRef]


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
    caches: List[CacheDesc] = eve.field(default_factory=list)

    @datamodels.validator("sections")
    def nonempty_loop(self, attribute: datamodels.Attribute, v: List[VerticalLoopSection]) -> None:
        if not v:
            raise ValueError("Empty vertical loop is not allowed")

    @datamodels.root_validator
    @classmethod
    def valid_section_intervals(cls: Type["VerticalLoop"], instance: "VerticalLoop") -> None:
        starts, ends = zip(*((s.interval.start, s.interval.end) for s in instance.sections))
        if instance.loop_order == common.LoopOrder.BACKWARD:
            starts, ends = starts[:-1], ends[1:]
        else:
            starts, ends = starts[1:], ends[:-1]

        if not all(
            start.level == end.level and start.offset == end.offset
            for start, end in zip(starts, ends)
        ):
            raise ValueError("Loop intervals not contiguous or in wrong order")


class Stencil(LocNode, eve.ValidatedSymbolTableTrait):
    name: str
    # TODO: fix to be List[Union[ScalarDecl, FieldDecl]]
    params: List[Decl]
    vertical_loops: List[VerticalLoop]
    declarations: List[Temporary]

    _validate_dtype_is_set = common.validate_dtype_is_set()
    _validate_lvalue_dims = common.validate_lvalue_dims(VerticalLoop, FieldDecl)
