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

from typing import Any, List, Optional, Tuple, Union

from pydantic import root_validator, validator

from eve import Str, SymbolName, SymbolTableTrait
from eve.typingx import RootValidatorValuesType
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


class CartesianIterationOffset(LocNode):
    i_offsets: Tuple[int, int]
    j_offsets: Tuple[int, int]

    @validator("i_offsets", "j_offsets")
    def offsets_ordered(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if not v[0] <= v[1]:
            raise ValueError(
                "Lower bound of iteration offset must be less or equal to upper bound."
            )
        return v


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


class HorizontalExecution(LocNode):
    body: List[Stmt]
    mask: Optional[Expr]
    declarations: List[LocalScalar]
    iteration_space: Optional[CartesianIterationOffset]

    @validator("mask")
    def mask_is_boolean_field_expr(cls, v: Optional[Expr]) -> Optional[Expr]:
        if v:
            if v.dtype != common.DataType.BOOL:
                raise ValueError("Mask must be a boolean expression.")
        return v


class Interval(LocNode):
    start: AxisBound
    end: AxisBound

    def covers(self, other: "Interval") -> bool:
        outer_starts_lower = self.start < other.start or self.start == other.start
        outer_ends_higher = self.end > other.end or self.end == other.end
        return outer_starts_lower and outer_ends_higher

    def intersects(self, other: "Interval") -> bool:
        return not (other.start >= self.end or self.start >= other.end)

    def shift(self, offset: int) -> "Interval":
        start = AxisBound(level=self.start.level, offset=self.start.offset + offset)
        end = AxisBound(level=self.end.level, offset=self.end.offset + offset)
        return Interval(start=start, end=end)

    @root_validator(skip_on_failure=True)
    def ordered_check(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
        if not values["start"] < values["end"]:
            raise ValueError("Interval start needs to be lower than end.")
        return values


class VerticalLoop(LocNode):
    interval: Interval
    horizontal_executions: List[HorizontalExecution]
    loop_order: common.LoopOrder
    declarations: List[Temporary]


class Stencil(LocNode, SymbolTableTrait):
    name: Str
    # TODO: fix to be List[Union[ScalarDecl, FieldDecl]]
    params: List[Decl]
    vertical_loops: List[VerticalLoop]

    _validate_dtype_is_set = common.validate_dtype_is_set()
    _validate_symbol_refs = common.validate_symbol_refs()


class IntervalMapping:
    def __init__(self) -> None:
        self.interval_starts: List[AxisBound] = list()
        self.interval_ends: List[AxisBound] = list()
        self.values: List[Any] = list()

    def _setitem_subset_of_existing(self, i: int, key: Interval, value: Any) -> None:
        start = self.interval_starts[i]
        end = self.interval_ends[i]
        if self.values[i] is not value:
            idx = i
            if key.start != start:
                self.interval_ends[i] = key.start
                self.interval_starts.insert(i + 1, key.start)
                self.interval_ends.insert(i + 1, key.end)
                self.values.insert(i + 1, value)
                idx = i + 1
            if key.end != end:
                self.interval_starts.insert(idx + 1, key.end)
                self.interval_ends.insert(idx + 1, end)
                self.values.insert(idx + 1, self.values[i])
                self.interval_ends[idx] = key.end
                self.values[idx] = value

    def _setitem_partial_overlap(self, i: int, key: Interval, value: Any) -> None:
        start = self.interval_starts[i]
        if key.start < start:
            if self.values[i] is value:
                self.interval_starts[i] = key.start
            else:
                self.interval_starts[i] = key.end
                self.interval_starts.insert(i, key.start)
                self.interval_ends.insert(i, key.end)
                self.values.insert(i, value)
        else:  # key.end > end
            if self.values[i] is value:
                self.interval_ends[i] = key.end
                nextidx = i + 1
            else:
                self.interval_ends[i] = key.start
                self.interval_starts.insert(i + 1, key.start)
                self.interval_ends.insert(i + 1, key.end)
                self.values.insert(i + 1, value)
                nextidx = i + 2
            if nextidx < len(self.interval_starts) and (
                key.intersects(
                    Interval(start=self.interval_starts[nextidx], end=self.interval_ends[nextidx])
                )
                or self.interval_starts[nextidx] == key.end
            ):
                if self.values[nextidx] is value:
                    self.interval_ends[nextidx - 1] = self.interval_ends[nextidx]
                    del self.interval_starts[nextidx]
                    del self.interval_ends[nextidx]
                    del self.values[nextidx]
                else:
                    self.interval_starts[nextidx] = key.end

    def __setitem__(self, key: Interval, value: Any) -> None:
        if not isinstance(key, Interval):
            raise TypeError("Only OIR intervals supported for method add of IntervalSet.")

        delete = list()
        for i, (start, end) in enumerate(zip(self.interval_starts, self.interval_ends)):
            if key.covers(Interval(start=start, end=end)):
                delete.append(i)

        for i in reversed(delete):  # so indices keep validity while deleting
            del self.interval_starts[i]
            del self.interval_ends[i]
            del self.values[i]

        if len(self.interval_starts) == 0:
            self.interval_starts.append(key.start)
            self.interval_ends.append(key.end)
            self.values.append(value)
            return

        for i, (start, end) in enumerate(zip(self.interval_starts, self.interval_ends)):
            if Interval(start=start, end=end).covers(key):
                self._setitem_subset_of_existing(i, key, value)
                return

        for i, (start, end) in enumerate(zip(self.interval_starts, self.interval_ends)):
            if (
                key.intersects(Interval(start=start, end=end))
                or start == key.end
                or end == key.start
            ):
                self._setitem_partial_overlap(i, key, value)
                return

        for i, start in enumerate(self.interval_starts):
            if start > key.start:
                self.interval_starts.insert(i, key.start)
                self.interval_ends.insert(i, key.end)
                self.values.insert(i, value)
                return
        self.interval_starts.append(key.start)
        self.interval_ends.append(key.end)
        self.values.append(value)
        return

    def __getitem__(self, key: Interval) -> List[Any]:
        if not isinstance(key, Interval):
            raise TypeError("Only OIR intervals supported for keys of IntervalMapping.")

        res = []
        for start, end, value in zip(self.interval_starts, self.interval_ends, self.values):
            if key.intersects(Interval(start=start, end=end)):
                res.append(value)
        return res
