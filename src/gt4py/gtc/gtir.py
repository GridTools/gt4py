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
from gt4py.gtc.common import DataType
from pydantic import validator

from devtools import debug  # noqa: F401
from eve import IntEnum, Node, SourceLocation, Str, SymbolName
from eve.type_definitions import SymbolRef


from gt4py.gtc import common
from pydantic.class_validators import root_validator


class LocNode(Node):
    loc: Optional[SourceLocation]


class Expr(LocNode):
    dtype: Optional[common.DataType]

    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(LocNode):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Literal(Expr):
    # TODO when coming from python AST we know more than just the string representation, I suppose
    value: Str
    dtype: common.DataType


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


class FieldAccess(Expr):
    name: SymbolRef
    offset: CartesianOffset

    @classmethod
    def centered(cls, *, name, loc=None):
        return cls(name=name, loc=loc, offset=CartesianOffset.zero())


class ParAssignStmt(Stmt):
    """Parallel assignment.

    R.h.s. is evaluated for all points and the resulting field is assigned
    (GTScript parallel model).
    """

    left: FieldAccess  # there are no local variables in gtir, only fields
    right: Expr

    @validator("left")
    def no_horizontal_offset_in_assignment(cls, v):
        if v.offset.i != 0 or v.offset.j != 0:
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v


def condition_is_boolean(parent_node_cls, cond: Expr) -> Expr:
    if cond.dtype and cond.dtype is not common.DataType.BOOL:
        raise ValueError("Condition in `{}` must be boolean.".format(parent_node_cls.__name__))
    return cond


class IfStmt(Stmt):
    cond: Expr
    true_branch: List[Stmt]
    false_branch: List[Stmt]

    @validator("cond")
    def condition_is_boolean(cls, cond):
        return condition_is_boolean(cls, cond)

    # TODO or like this (but how to pass the name)
    # _cond_is_bool = validator("cond", allow_reuse=True)(condition_is_boolean)


def verify_and_get_common_dtype(node_cls, values: List[Expr]) -> common.DataType:
    assert len(values) > 0
    if all([v.dtype for v in values]):
        dtype = values[0].dtype
        if all([v.dtype == dtype for v in values]):
            return dtype
        else:
            raise ValueError(
                "Type mismatch in `{}`. Types are ".format(node_cls.__name__)
                + ", ".join(v.dtype.name for v in values)
            )
    else:
        return None


class TernaryOp(Expr):
    cond: Expr
    true_expr: Expr
    false_expr: Expr

    @validator("cond")
    def condition_is_boolean(cls, cond):
        return condition_is_boolean(cls, cond)

    @root_validator(pre=True)
    def type_propagation_and_check(cls, values):
        common_dtype = verify_and_get_common_dtype(
            cls, [values["true_expr"], values["false_expr"]]
        )
        if common_dtype:
            values["dtype"] = common_dtype
        return values


class BinaryOp(Expr):
    op: Union[common.BinaryOperator, common.ComparisonOperator, common.LogicalOperator]
    left: Expr
    right: Expr

    @root_validator(pre=True)
    def type_propagation_and_check(cls, values):
        common_dtype = verify_and_get_common_dtype(cls, [values["left"], values["right"]])

        if common_dtype:
            if isinstance(values["op"], common.BinaryOperator):
                if common_dtype is not common.DataType.BOOL:
                    values["dtype"] = common_dtype
                else:
                    raise ValueError(
                        "Boolean expression is not allowed with arithmetic operation."
                    )
            elif isinstance(values["op"], common.LogicalOperator):
                if common_dtype is common.DataType.BOOL:
                    values["dtype"] = common.DataType.BOOL
                else:
                    raise ValueError("Arithmetic expression is not allowed in logical operation.")
            elif isinstance(values["op"], common.ComparisonOperator):
                values["dtype"] = common.DataType.BOOL

        return values


class FieldDecl(LocNode):
    # name: SymbolName
    name: Str
    dtype: common.DataType


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
    body: List[Stmt]
    start: AxisBound
    end: AxisBound


class VerticalLoop(LocNode):
    vertical_intervals: List[VerticalInterval]
    loop_order: common.LoopOrder

    # TODO validate that intervals are contiguous


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
    metas: Dict[str, FieldMetadata] = {}


class FieldsMetadataBuilder:
    def __init__(self) -> None:
        self.metas: Dict[str, FieldMetadataBuilder] = {}

    def get_or_create(self, node: Union[FieldAccess, FieldDecl]) -> FieldMetadataBuilder:
        return self.metas.setdefault(node.name, FieldMetadataBuilder().name(node.name))

    def build(self) -> FieldsMetadata:
        return FieldsMetadata(metas={k: v.build() for k, v in self.metas.items()})


class Stencil(LocNode):
    name: SymbolName
    params: List[FieldDecl]
    vertical_loops: List[VerticalLoop]
    fields_metadata: Optional[FieldsMetadata]

    @property
    def param_names(self) -> List:
        return [p.name for p in self.params]
