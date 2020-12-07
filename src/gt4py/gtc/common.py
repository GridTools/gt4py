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
from typing import Generic, List, Optional, TypeVar, Union

from eve import GenericNode, IntEnum, Node, SourceLocation, Str, StrEnum, SymbolTableTrait
from eve.type_definitions import SymbolRef
from pydantic import validator
from pydantic.class_validators import root_validator


class AssignmentKind(StrEnum):
    """Kind of assignment: plain or combined with operations."""

    PLAIN = "="
    ADD = "+="
    SUB = "-="
    MUL = "*="
    DIV = "/="


@enum.unique
class UnaryOperator(StrEnum):
    """Unary operator indentifier."""

    POS = "+"
    NEG = "-"
    NOT = "not"


@enum.unique
class ArithmeticOperator(StrEnum):
    """Arithmetic operators."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


@enum.unique
class ComparisonOperator(StrEnum):
    """Comparison operators."""

    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    EQ = "=="
    NE = "!="


@enum.unique
class LogicalOperator(StrEnum):
    """Logical operators."""

    AND = "and"
    OR = "or"


@enum.unique
class DataType(IntEnum):
    """Data type identifier."""

    # IDs from gt4py
    INVALID = -1
    AUTO = 0
    DEFAULT = 1
    BOOL = 10
    INT8 = 11
    INT16 = 12
    INT32 = 14
    INT64 = 18
    FLOAT32 = 104
    FLOAT64 = 108


@enum.unique
class LoopOrder(IntEnum):
    """Loop order identifier."""

    PARALLEL = 0
    FORWARD = 1
    BACKWARD = 2


@enum.unique
class BuiltInLiteral(IntEnum):
    MAX_VALUE = 0
    MIN_VALUE = 1
    ZERO = 2
    ONE = 3


@enum.unique
class NativeFunction(StrEnum):
    ABS = "abs"
    MIN = "min"
    MAX = "max"
    MOD = "mod"

    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ARCSIN = "arcsin"
    ARCCOS = "arccos"
    ARCTAN = "arctan"

    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"

    ISFINITE = "isfinite"
    ISINF = "isinf"
    ISNAN = "isnan"
    FLOOR = "floor"
    CEIL = "ceil"
    TRUNC = "trunc"

    @property
    def arity(self):
        return type(self).IR_OP_TO_NUM_ARGS[self]


NativeFunction.IR_OP_TO_NUM_ARGS = {
    NativeFunction.ABS: 1,
    NativeFunction.MIN: 2,
    NativeFunction.MAX: 2,
    NativeFunction.MOD: 2,
    NativeFunction.SIN: 1,
    NativeFunction.COS: 1,
    NativeFunction.TAN: 1,
    NativeFunction.ARCSIN: 1,
    NativeFunction.ARCCOS: 1,
    NativeFunction.ARCTAN: 1,
    NativeFunction.SQRT: 1,
    NativeFunction.EXP: 1,
    NativeFunction.LOG: 1,
    NativeFunction.ISFINITE: 1,
    NativeFunction.ISINF: 1,
    NativeFunction.ISNAN: 1,
    NativeFunction.FLOOR: 1,
    NativeFunction.CEIL: 1,
    NativeFunction.TRUNC: 1,
}


@enum.unique
class LevelMarker(StrEnum):
    START = "start"
    END = "end"


@enum.unique
class ExprKind(IntEnum):
    SCALAR = 0
    FIELD = 1


class LocNode(Node):
    loc: Optional[SourceLocation]


class Expr(LocNode):
    """
    Expression base class.

    All expressions have
    - an optional `dtype`
    - an expression `kind` (scalar or field)
    """

    dtype: Optional[DataType]
    kind: ExprKind

    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(LocNode):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Stmt` abstract class.")
        super().__init__(*args, **kwargs)


def verify_condition_is_boolean(parent_node_cls, cond: Expr) -> Expr:
    if cond.dtype and cond.dtype is not DataType.BOOL:
        raise ValueError("Condition in `{}` must be boolean.".format(parent_node_cls.__name__))
    return cond


def verify_and_get_common_dtype(node_cls, values: List[Expr]) -> DataType:
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


def compute_kind(values: List[Expr]) -> ExprKind:
    if any([v.kind == ExprKind.FIELD for v in values]):
        return ExprKind.FIELD
    else:
        return ExprKind.SCALAR


class Literal(Node):
    # TODO when coming from python AST we know more than just the string representation, I suppose
    value: Union[BuiltInLiteral, Str]
    dtype: DataType
    kind = ExprKind.SCALAR


StmtT = TypeVar("StmtT")
ExprT = TypeVar("ExprT")
TargetT = TypeVar("TargetT")


class CartesianOffset(Node):
    i: int
    j: int
    k: int

    @classmethod
    def zero(cls):
        return cls(i=0, j=0, k=0)

    def to_dict(self):
        return {"i": self.i, "j": self.j, "k": self.k}


class ScalarAccess(Node):
    name: SymbolRef
    kind = ExprKind.SCALAR


class FieldAccess(Node):
    name: SymbolRef
    offset: CartesianOffset
    kind = ExprKind.FIELD

    @classmethod
    def centered(cls, *, name, loc=None):
        return cls(name=name, loc=loc, offset=CartesianOffset.zero())


class BlockStmt(GenericNode, SymbolTableTrait, Generic[StmtT]):
    body: List[StmtT]


class IfStmt(GenericNode, Generic[StmtT, ExprT]):
    """
    Generic if statement.

    Verifies that `cond` is a boolean expr (if `dtype` is set).
    """

    cond: ExprT
    true_branch: StmtT
    false_branch: Optional[StmtT]

    @validator("cond")
    def condition_is_boolean(cls, cond):
        return verify_condition_is_boolean(cls, cond)


class AssignStmt(GenericNode, Generic[TargetT, ExprT]):
    left: TargetT
    right: ExprT


class UnaryOp(GenericNode, Generic[ExprT]):
    """
    Generic unary operation with type propagation.

    The generic `UnaryOp` already contains logic for type propagation.
    """

    op: UnaryOperator
    expr: ExprT

    @root_validator(pre=True)
    def dtype_propagation(cls, values):
        values["dtype"] = values["expr"].dtype
        return values

    @root_validator(pre=True)
    def kind_propagation(cls, values):
        values["kind"] = values["expr"].kind
        return values

    @root_validator(pre=True)
    def op_to_dtype_check(cls, values):
        if values["expr"].dtype:
            if values["op"] == UnaryOperator.NOT:
                if not values["expr"].dtype == DataType.BOOL:
                    raise ValueError("Unary operator `NOT` only allowed with boolean expression.")
            else:
                if values["expr"].dtype == DataType.BOOL:
                    raise ValueError(
                        "Unary operator `{}` not allowed with boolean expression.".format(
                            values["op"].name
                        )
                    )
        return values


class BinaryOp(GenericNode, Generic[ExprT]):
    """Generic binary operation with type propagation.

    The generic BinaryOp already contains logic for
    - strict type checking if the `dtype` for `left` and `right` is set.
    - type propagation (taking `operator` type into account).
    """

    # TODO parametrize on op?
    op: Union[ArithmeticOperator, ComparisonOperator, LogicalOperator]
    left: ExprT
    right: ExprT

    @root_validator(pre=True)
    def dtype_propagation_and_check(cls, values):
        common_dtype = verify_and_get_common_dtype(cls, [values["left"], values["right"]])

        if common_dtype:
            if isinstance(values["op"], ArithmeticOperator):
                if common_dtype is not DataType.BOOL:
                    values["dtype"] = common_dtype
                else:
                    raise ValueError(
                        "Boolean expression is not allowed with arithmetic operation."
                    )
            elif isinstance(values["op"], LogicalOperator):
                if common_dtype is DataType.BOOL:
                    values["dtype"] = DataType.BOOL
                else:
                    raise ValueError("Arithmetic expression is not allowed in boolean operation.")
            elif isinstance(values["op"], ComparisonOperator):
                values["dtype"] = DataType.BOOL

        return values

    @root_validator(pre=True)
    def kind_propagation(cls, values):
        values["kind"] = compute_kind([values["left"], values["right"]])
        return values


class TernaryOp(GenericNode, Generic[ExprT]):
    """
    Generic ternary operation with type propagation.

    The generic TernaryOp already contains logic for
    - strict type checking if the `dtype` for `true_expr` and `false_expr` is set.
    - type checking for `cond`
    - type propagation.
    """

    # TODO parametrize cond expr separately?
    cond: ExprT
    true_expr: ExprT
    false_expr: ExprT

    @validator("cond")
    def condition_is_boolean(cls, cond):
        return verify_condition_is_boolean(cls, cond)

    @root_validator(pre=True)
    def dtype_propagation_and_check(cls, values):
        common_dtype = verify_and_get_common_dtype(
            cls, [values["true_expr"], values["false_expr"]]
        )
        if common_dtype:
            values["dtype"] = common_dtype
        return values

    @root_validator(pre=True)
    def kind_propagation(cls, values):
        values["kind"] = compute_kind([values["true_expr"], values["false_expr"]])
        return values


class Cast(GenericNode, Generic[ExprT]):
    dtype: DataType
    expr: ExprT


class NativeFuncCall(GenericNode, Generic[ExprT]):
    """"""

    func: NativeFunction
    args: List[ExprT]

    @root_validator(pre=True)
    def arity_check(cls, values):
        if values["func"].arity != len(values["args"]):
            raise ValueError(
                "{} accepts {} arguments, {} where passed.".format(
                    values["func"], values["func"].arity, len(values["args"])
                )
            )
        return values
