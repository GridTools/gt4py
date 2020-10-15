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
from typing import Dict, List, Optional, Tuple, Union

from devtools import debug  # noqa: F401
from eve import IntEnum, Node, SourceLocation, Str

from . import common


class LocNode(Node):
    loc: Optional[SourceLocation]


class Expr(LocNode):
    pass


class Stmt(LocNode):
    pass


class Literal(Expr):
    # TODO when coming from python AST we know more than just the string representation, I suppose
    value: Str
    dtype: common.DataType


class Domain(LocNode):
    pass


class CartesianOffset(Node):
    i: int
    j: int
    k: int

    @classmethod
    def zero(cls):
        return cls(i=0, j=0, k=0)

    def to_dict(self):
        return {"i": self.i, "j": self.j, "k": self.k}


class FieldAccess(Expr):
    name: Str  # via symbol table
    offset: CartesianOffset

    @classmethod
    def centered(cls, *, name, loc=None):
        return cls(name=name, loc=loc, offset=CartesianOffset.zero())


class AssignStmt(Stmt):
    left: FieldAccess  # there are no local variables in gtir, only fields
    right: Expr


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


class FieldDecl(LocNode):
    name: Str
    dtype: common.DataType


class HorizontalLoop(LocNode):
    stmt: Stmt


class AxisBound(Node):
    level: common.LevelMarker
    offset: int = 0

    @classmethod
    def from_start(cls, offset: int):
        return cls(level=common.LevelMarker.START, offset=offset)

    @classmethod
    def from_end(cls, offset: int):
        return cls(level=common.LevelMarker.END, offset=offset)

    @classmethod
    def start(cls):
        return cls.from_start(0)

    @classmethod
    def end(cls):
        return cls.from_end(0)


class VerticalInterval(LocNode):
    horizontal_loops: List[HorizontalLoop]
    start: AxisBound
    end: AxisBound


class VerticalLoop(LocNode):
    vertical_intervals: List[VerticalInterval]
    loop_order: common.LoopOrder


class Stencil(LocNode):
    vertical_loops: List[VerticalLoop]


@enum.unique
class AccessKind(IntEnum):
    READ_ONLY = 0
    READ_WRITE = 1


class FieldBoundary(Node):
    i: Tuple[int, int]
    j: Tuple[int, int]
    k: Tuple[int, int]

    def to_dict(self):
        return {"i": self.i, "j": self.j, "k": self.k}


class FieldBoundaryAccumulator:
    def __init__(self):
        self.bounds = {
            "i": {"lower": 0, "upper": 0},
            "j": {"lower": 0, "upper": 0},
            "k": {"lower": 0, "upper": 0},
        }

    def update_from_offset(self, offset: CartesianOffset):
        for idx, values in self.bounds.items():
            offset_at_idx = offset.to_dict()[idx]
            sign, end = (-1, "lower") if offset_at_idx < 0 else (1, "upper")
            values[end] = max(sign * offset_at_idx, values[end])

    def to_boundary(self):
        return FieldBoundary(**{k: (v["lower"], v["upper"]) for k, v in self.bounds.items()})


class FieldMetadata(Node):
    name: str
    access: AccessKind
    boundary: FieldBoundary
    dtype: common.DataType


class FieldMetadataBuilder:
    def __init__(self) -> None:
        self._name: Optional[str] = None
        self._access: int = AccessKind.READ_WRITE
        self._dtype: Optional[common.DataType] = None
        self.boundary = FieldBoundaryAccumulator()

    def name(self, name: str) -> "FieldMetadataBuilder":
        self._name = name
        return self

    def access(self, access: AccessKind) -> "FieldMetadataBuilder":
        self._access = access
        return self

    def dtype(self, dtype: common.DataType) -> "FieldMetadataBuilder":
        self._dtype = dtype
        return self

    def build(self):
        return FieldMetadata(
            name=self._name,
            access=self._access,
            boundary=self.boundary.to_boundary(),
            dtype=self._dtype,
        )


class FieldsMetadata(Node):
    metas: Dict[str, FieldMetadata] = {}


class FieldsMetadataBuilder:
    def __init__(self) -> None:
        self.metas: Dict[str, FieldMetadataBuilder] = {}

    def get_or_create(self, node: Union[FieldAccess, FieldDecl]) -> FieldMetadataBuilder:
        return self.metas.setdefault(node.name, FieldMetadataBuilder().name(node.name))

    def build(self) -> FieldsMetadata:
        return FieldsMetadata(metas={k: v.build() for k, v in self.metas.items()})


class Computation(LocNode):
    name: Str
    params: List[FieldDecl]
    stencils: List[Stencil]
    fields_metadata: FieldsMetadata = FieldsMetadata()

    @property
    def param_names(self) -> List:
        return [p.name for p in self.params]
