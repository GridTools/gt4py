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
from pydantic import validator

from devtools import debug  # noqa: F401
from eve import IntEnum, Node, Str, SymbolName
from eve.type_definitions import SymbolRef


from gt4py.gtc import common
from gt4py.gtc.common import LocNode


class Expr(common.Expr):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(common.Stmt):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Stmt` abstract class.")
        super().__init__(*args, **kwargs)


class Literal(common.Literal, Expr):
    pass


class Domain(LocNode):
    # TODO
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


class ScalarAccess(Expr):
    name: SymbolRef
    kind = common.ExprKind.SCALAR


class FieldAccess(Expr):
    name: SymbolRef
    offset: CartesianOffset
    kind = common.ExprKind.FIELD

    @classmethod
    def centered(cls, *, name, loc=None):
        return cls(name=name, loc=loc, offset=CartesianOffset.zero())


class ParAssignStmt(common.AssignStmt[FieldAccess, Expr], Stmt):
    """Parallel assignment.

    R.h.s. is evaluated for all points and the resulting field is assigned
    (GTScript parallel model).
    Scalar variables on the l.h.s. are not allowed,
    as the only scalar variables are read-only stencil parameters.
    """

    @validator("left")
    def no_horizontal_offset_in_assignment(cls, v):
        if v.offset.i != 0 or v.offset.j != 0:
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v


class FieldIfStmt(common.IfStmt[Stmt, Expr], Stmt):
    @validator("cond")
    def verify_scalar_condition(cls, cond):
        if cond.kind != common.ExprKind.FIELD:
            raise ValueError("Condition is not a field expression")
        return cond


class ScalarIfStmt(common.IfStmt[Stmt, Expr], Stmt):
    @validator("cond")
    def verify_scalar_condition(cls, cond):
        if cond.kind != common.ExprKind.SCALAR:
            raise ValueError("Condition is not scalar")
        return cond


class BinaryOp(common.BinaryOp[Expr], Expr):
    pass


class TernaryOp(common.TernaryOp[Expr], Expr):
    pass


class Decl(LocNode):
    name: SymbolName
    dtype: common.DataType

    def __init__(self, *args, **kwargs):
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    # TODO dimensions
    pass


class ScalarDecl(Decl):
    pass


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


class Interval(LocNode):
    start: AxisBound
    end: AxisBound


class VerticalLoop(LocNode):
    interval: Interval
    loop_order: common.LoopOrder
    body: List[Stmt]


@enum.unique
class AccessKind(IntEnum):
    READ_ONLY = 0
    READ_WRITE = 1
    # TODO add WRITE_ONLY using flag enums


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
    name: Str
    access: AccessKind
    boundary: FieldBoundary
    dtype: common.DataType


class FieldMetadataBuilder:
    def __init__(self) -> None:
        self._name: Optional[Str] = None
        self._access: int = AccessKind.READ_WRITE
        self._dtype: Optional[int] = None
        self.boundary = FieldBoundaryAccumulator()

    def name(self, name: str) -> "FieldMetadataBuilder":
        self._name = name
        return self

    def access(self, access: int) -> "FieldMetadataBuilder":
        self._access = access
        return self

    def dtype(self, dtype: int) -> "FieldMetadataBuilder":
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
    metas: Dict[Str, FieldMetadata] = {}


class FieldsMetadataBuilder:
    def __init__(self) -> None:
        self.metas: Dict[str, FieldMetadataBuilder] = {}

    def get_or_create(self, node: Union[FieldAccess, FieldDecl]) -> FieldMetadataBuilder:
        return self.metas.setdefault(node.name, FieldMetadataBuilder().name(node.name))

    def build(self) -> FieldsMetadata:
        return FieldsMetadata(metas={k: v.build() for k, v in self.metas.items()})


class Stencil(LocNode):
    # TODO remove `__main__.`` from name and use default SymbolName constraint
    name: SymbolName.constrained(r"[a-zA-Z_][\w\.]*")
    # TODO deal with gtscript externals
    params: List[Decl]
    vertical_loops: List[VerticalLoop]
    fields_metadata: Optional[FieldsMetadata]

    @property
    def param_names(self) -> List:
        return [p.name for p in self.params]
