# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Implementation of the intermediate representations used in GT4Py.

-----------
Definitions
-----------

Empty
    Empty node value (`None` is a valid Python value)

InvalidBranch
    Sentinel value for wrongly build conditional expressions

Builtin enumeration (:class:`Builtin`)
    Named Python constants
    [`NONE`, `FALSE`, `TRUE`]

DataType enumeration (:class:`DataType`)
    Native numeric data types
    [`INVALID`, `AUTO`, `DEFAULT`, `BOOL`,
    `INT8`, `INT16`, `INT32`, `INT64`, `FLOAT32`, `FLOAT64`]

UnaryOperator enumeration (:class:`UnaryOperator`)
    Unary operators
    [`POS`, `NEG`, `NOT`]

BinaryOperator enumeration (:class:`BinaryOperator`)
    Binary operators
    [`ADD`, `SUB`, `MUL`, `DIV`, `POW`, `AND`, `OR`,
    `EQ`, `NE`, `LT`, `LE`, `GT`, `GE`]

NativeFunction enumeration (:class:`NativeFunction`)
    Native function identifier
    [`ABS`, `MAX`, `MIN, `MOD`, `SIN`, `COS`, `TAN`, `ARCSIN`, `ARCCOS`, `ARCTAN`,
    `SQRT`, `EXP`, `LOG`, `LOG10`, `ISFINITE`, `ISINF`, `ISNAN`, `FLOOR`, `CEIL`,
    `TRUNC`, `ERF`, `ERFC`, `INT32`, `INT64`, `FLOAT32`, `FLOAT64`, `ROUND`,
    `ROUND_AWAY_FROM_ZERO`]

LevelMarker enumeration (:class:`LevelMarker`)
    Special axis levels
    [`START`, `END`]

IterationOrder enumeration (:class:`IterationOrder`)
    Execution order
    [`BACKWARD`, `PARALLEL`, `FORWARD`]

Index (:class:`gt4py.definitions.Index`)
    Multidimensional integer offset
    [int+]

Extent (:class:`gt4py.definitions.Extent`)
    Multidimensional integer extent
    [(lower: `int`, upper: `int`)+]



-------------
Definition IR
-------------

All nodes have an optional attribute `loc` [`Location(line: int, column: int, scope: str)`]
storing a reference to the piece of source code which originated the node.

 ::

    Axis(name: str)

    Domain(parallel_axes: List[Axis], [sequential_axis: Axis])
        # LatLonGrids -> parallel_axes: ["I", "J"], sequential_axis: "K"

    Literal     = ScalarLiteral(value: Any (should match DataType), data_type: DataType)
                | BuiltinLiteral(value: Builtin)

    Ref         = VarRef(name: str, [index: int])
                | FieldRef(name: str, offset: Dict[str, int | Expr])
                # Horizontal indices must be ints

    NativeFuncCall(func: NativeFunction, args: List[Expr], data_type: DataType)

    Cast(expr: Expr, data_type: DataType)

    AxisPosition(axis: str, data_type: DataType)

    AxisIndex(axis: str, endpt: LevelMarker, offset: int, data_type: DataType)

    Expr        = Literal | Ref | NativeFuncCall | Cast | CompositeExpr | InvalidBranch | AxisPosition | AxisIndex

    CompositeExpr   = UnaryOpExpr(op: UnaryOperator, arg: Expr)
                    | BinOpExpr(op: BinaryOperator, lhs: Expr, rhs: Expr)
                    | TernaryOpExpr(condition: Expr, then_expr: Expr, else_expr: Expr)

    Decl        = FieldDecl(name: str, data_type: DataType, axes: List[str],
                            is_api: bool, layout_id: str)
                | VarDecl(name: str, data_type: DataType, length: int,
                          is_api: bool, [init: Literal])

    BlockStmt(stmts: List[Statement])

    Statement   = Decl
                | Assign(target: Ref, value: Expr)
                | If(condition: expr, main_body: BlockStmt, else_body: BlockStmt)
                | HorizontalIf(intervals: Dict[str, Interval], body: BlockStmt)
                | While(condition: expr, body: BlockStmt)
                | BlockStmt

    AxisBound(level: LevelMarker | VarRef, offset: int)
        # bound = level + offset
        # level: LevelMarker = special START or END level
        # level: VarRef = access to `int` or `[int]` variable holding the run-time value of the level
        # offset: int

    AxisInterval(start: AxisBound, end: AxisBound)
        # start is included
        # end is excluded

    ComputationBlock(interval: AxisInterval, iteration_order: IterationOrder, body: BlockStmt)

    ArgumentInfo(name: str, is_keyword: bool, [default: Any])

    StencilDefinition(name: str,
                      domain: Domain,
                      api_signature: List[ArgumentInfo],
                      api_fields: List[FieldDecl],
                      parameters: List[VarDecl],
                      computations: List[ComputationBlock],
                      [externals: Dict[str, Any], sources: Dict[str, str]])
"""

from __future__ import annotations

import enum
import operator
import sys
from typing import List, Optional, Sequence

import numpy as np

from gt4py.cartesian.definitions import CartesianSpace
from gt4py.cartesian.utils.attrib import (
    Any as Any,
    Dict as DictOf,
    List as ListOf,
    Union as UnionOf,
    attribkwclass as attribclass,
    attribute,
)


# ---- Foundations ----
class Empty:
    pass


class Node:
    pass


@attribclass
class Location(Node):
    line = attribute(of=int)
    column = attribute(of=int)
    scope = attribute(of=str, default="<source>")

    @classmethod
    def from_ast_node(cls, ast_node, scope: str | None = None):
        lineno = getattr(ast_node, "lineno", 0)
        col_offset = getattr(ast_node, "col_offset", 0)
        scope = (
            "<source>" if scope is None else scope
        )  # scope is sometimes explicitly passed down as `None`
        return cls(line=lineno, column=col_offset + 1, scope=scope)


# ---- IR: domain ----
@attribclass
class Axis(Node):
    name = attribute(of=str)


@enum.unique
class LevelMarker(enum.Enum):
    START = 0
    END = -1

    def __str__(self) -> str:
        return self.name


@attribclass
class Domain(Node):
    parallel_axes = attribute(of=ListOf[Axis])
    sequential_axis = attribute(of=Axis, optional=True)

    @classmethod
    def LatLonGrid(cls):
        return cls(
            parallel_axes=[
                Axis(name=CartesianSpace.Axis.I.name),
                Axis(name=CartesianSpace.Axis.J.name),
            ],
            sequential_axis=Axis(name=CartesianSpace.Axis.K.name),
        )

    @property
    def axes(self):
        result = list(self.parallel_axes)
        if self.sequential_axis:
            result.append(self.sequential_axis)
        return result

    @property
    def axes_names(self):
        return [ax.name for ax in self.axes]

    @property
    def domain_ndims(self):
        return len(self.parallel_axes) + (1 if self.sequential_axis else 0)

    ndims = domain_ndims

    def index(self, axis):
        if isinstance(axis, Axis):
            axis = axis.name
        assert isinstance(axis, str)
        return self.axes_names.index(axis)


# ---- IR: types ----
@enum.unique
class Builtin(enum.Enum):
    NONE = -1
    FALSE = 0
    TRUE = 1

    @classmethod
    def from_value(cls, value: bool | None) -> Builtin:
        if value is None:
            return cls.NONE
        if value is True:
            return cls.TRUE
        if value is False:
            return cls.FALSE

    def __str__(self) -> str:
        return self.name


@enum.unique
class DataType(enum.Enum):
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

    def __str__(self) -> str:
        return self.name

    @property
    def dtype(self):
        return np.dtype(self.NATIVE_TYPE_TO_NUMPY[self])

    @classmethod
    def from_dtype(cls, py_dtype):
        if isinstance(py_dtype, type):
            py_dtype = np.dtype(py_dtype)
        assert isinstance(py_dtype, np.dtype)
        return cls.NUMPY_TO_NATIVE_TYPE.get(py_dtype.name, cls.INVALID)

    @classmethod
    def merge(cls, *args):
        result = cls(max(arg.value for arg in args))
        return result


def frontend_type_to_native_type(
    literal_int_precision: int, literal_float_precision: int
) -> dict[str, DataType]:
    """Return the mapping of frontend types to native types.

    Args:
        literal_int_precision (int): Literal precision used for mapping `int` to either 32 or 64 bit precision.
        literal_float_precision (int): Literal precision used for mapping `float` to either 32 or 64 bit precision.

    Returns:
        dict[str, DataType]: Mapping of the frontend types to our DataTypes.
    """
    return {
        "int32": DataType.INT32,
        "int64": DataType.INT64,
        "int": DataType.INT32 if literal_int_precision == 32 else DataType.INT64,
        "float32": DataType.FLOAT32,
        "float64": DataType.FLOAT64,
        "float": DataType.FLOAT32 if literal_float_precision == 32 else DataType.FLOAT64,
    }


DataType.NATIVE_TYPE_TO_NUMPY = {
    DataType.DEFAULT: "float_",
    DataType.BOOL: "bool",
    DataType.INT8: "int8",
    DataType.INT16: "int16",
    DataType.INT32: "int32",
    DataType.INT64: "int64",
    DataType.FLOAT32: "float32",
    DataType.FLOAT64: "float64",
}

DataType.NUMPY_TO_NATIVE_TYPE = {value: key for key, value in DataType.NATIVE_TYPE_TO_NUMPY.items()}


# ---- IR: expressions ----
class Expr(Node):
    pass


class Literal(Expr):
    pass


class InvalidBranch(Expr):
    pass


@attribclass
class ScalarLiteral(Literal):
    value = attribute(of=Any)  # Potentially an array of numeric structs
    data_type = attribute(of=DataType)
    loc = attribute(of=Location, optional=True)


@attribclass
class BuiltinLiteral(Literal):
    value = attribute(of=Builtin)
    loc = attribute(of=Location, optional=True)


class Ref(Expr):
    pass


@attribclass
class VarRef(Ref):
    name = attribute(of=str)
    index = attribute(of=int, optional=True)
    loc = attribute(of=Location, optional=True)


@attribclass
class AbsoluteKIndex(Expr):
    """See gtc.common.AbsoluteKIndex"""

    k = attribute(of=Any)


@attribclass
class FieldRef(Ref):
    name = attribute(of=str)
    offset = attribute(of=DictOf[str, UnionOf[int, Expr, AbsoluteKIndex]])
    data_index = attribute(of=ListOf[Expr], factory=list)
    loc = attribute(of=Location, optional=True)

    @classmethod
    def at_center(
        cls, name: str, axes: Sequence[str], data_index: Optional[List[int]] = None, loc=None
    ):
        return cls(
            name=name, offset={axis: 0 for axis in axes}, data_index=data_index or [], loc=loc
        )

    @classmethod
    def datadims_index(cls, name: str, loc=None):
        return cls(name=name, offset={}, data_index=[], loc=loc)


@attribclass
class Cast(Expr):
    data_type = attribute(of=DataType)
    expr = attribute(of=Expr)
    loc = attribute(of=Location, optional=True)


@attribclass
class AxisPosition(Expr):
    axis = attribute(of=str)
    data_type = attribute(of=DataType, default=DataType.INT32)


@attribclass
class AxisIndex(Expr):
    axis = attribute(of=str)
    endpt = attribute(of=LevelMarker)
    offset = attribute(of=int)
    data_type = attribute(of=DataType, default=DataType.INT32)


@enum.unique
class NativeFunction(enum.Enum):
    ABS = enum.auto()
    MIN = enum.auto()
    MAX = enum.auto()
    MOD = enum.auto()

    SIN = enum.auto()
    COS = enum.auto()
    TAN = enum.auto()
    ARCSIN = enum.auto()
    ARCCOS = enum.auto()
    ARCTAN = enum.auto()

    SINH = enum.auto()
    COSH = enum.auto()
    TANH = enum.auto()
    ARCSINH = enum.auto()
    ARCCOSH = enum.auto()
    ARCTANH = enum.auto()

    SQRT = enum.auto()
    EXP = enum.auto()
    LOG = enum.auto()
    LOG10 = enum.auto()
    GAMMA = enum.auto()
    CBRT = enum.auto()

    ISFINITE = enum.auto()
    ISINF = enum.auto()
    ISNAN = enum.auto()
    FLOOR = enum.auto()
    CEIL = enum.auto()
    TRUNC = enum.auto()
    ERF = enum.auto()
    ERFC = enum.auto()
    ROUND = enum.auto()
    ROUND_AWAY_FROM_ZERO = enum.auto()

    # Cast operations - share a keyword with type hints
    INT32 = enum.auto()
    INT64 = enum.auto()
    FLOAT32 = enum.auto()
    FLOAT64 = enum.auto()

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
    NativeFunction.SINH: 1,
    NativeFunction.COSH: 1,
    NativeFunction.TANH: 1,
    NativeFunction.ARCSINH: 1,
    NativeFunction.ARCCOSH: 1,
    NativeFunction.ARCTANH: 1,
    NativeFunction.SQRT: 1,
    NativeFunction.EXP: 1,
    NativeFunction.LOG: 1,
    NativeFunction.LOG10: 1,
    NativeFunction.GAMMA: 1,
    NativeFunction.CBRT: 1,
    NativeFunction.ISFINITE: 1,
    NativeFunction.ISINF: 1,
    NativeFunction.ISNAN: 1,
    NativeFunction.FLOOR: 1,
    NativeFunction.CEIL: 1,
    NativeFunction.TRUNC: 1,
    NativeFunction.INT32: 1,
    NativeFunction.INT64: 1,
    NativeFunction.FLOAT32: 1,
    NativeFunction.FLOAT64: 1,
    NativeFunction.ERF: 1,
    NativeFunction.ERFC: 1,
    NativeFunction.ROUND: 1,
    NativeFunction.ROUND_AWAY_FROM_ZERO: 1,
}


@attribclass
class NativeFuncCall(Expr):
    func = attribute(of=NativeFunction)
    args = attribute(of=ListOf[Expr])
    data_type = attribute(of=DataType)
    loc = attribute(of=Location, optional=True)


class CompositeExpr(Expr):
    pass


@enum.unique
class UnaryOperator(enum.Enum):
    POS = 1
    NEG = 2

    TRANSPOSED = 5

    NOT = 11

    @property
    def python_op(self):
        return type(self).IR_OP_TO_PYTHON_OP[self]

    @property
    def python_symbol(self):
        return type(self).IR_OP_TO_PYTHON_SYMBOL[self]


UnaryOperator.IR_OP_TO_PYTHON_OP = {
    UnaryOperator.POS: operator.pos,
    UnaryOperator.NEG: operator.neg,
    UnaryOperator.NOT: operator.not_,
}

UnaryOperator.IR_OP_TO_PYTHON_SYMBOL = {
    UnaryOperator.POS: "+",
    UnaryOperator.NEG: "-",
    UnaryOperator.NOT: "not",
}


@attribclass
class UnaryOpExpr(CompositeExpr):
    op = attribute(of=UnaryOperator)
    arg = attribute(of=Expr)
    loc = attribute(of=Location, optional=True)


@enum.unique
class BinaryOperator(enum.Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    POW = 5
    MOD = 6

    MATMULT = 8

    AND = 11
    OR = 12

    EQ = 21
    NE = 22
    LT = 23
    LE = 24
    GT = 25
    GE = 26

    @property
    def python_op(self):
        return type(self).IR_OP_TO_PYTHON_OP[self]

    @property
    def python_symbol(self):
        return type(self).IR_OP_TO_PYTHON_SYMBOL[self]


BinaryOperator.IR_OP_TO_PYTHON_OP = {
    BinaryOperator.ADD: operator.add,
    BinaryOperator.SUB: operator.sub,
    BinaryOperator.MUL: operator.mul,
    BinaryOperator.DIV: operator.truediv,
    BinaryOperator.POW: operator.pow,
    BinaryOperator.MOD: operator.mod,
    # BinaryOperator.AND: lambda a, b: a and b,  # non short-circuit emulation
    # BinaryOperator.OR: lambda a, b: a or b,  # non short-circuit emulation
    BinaryOperator.LT: operator.lt,
    BinaryOperator.LE: operator.le,
    BinaryOperator.EQ: operator.eq,
    BinaryOperator.GE: operator.ge,
    BinaryOperator.GT: operator.gt,
    BinaryOperator.NE: operator.ne,
}

BinaryOperator.IR_OP_TO_PYTHON_SYMBOL = {
    BinaryOperator.ADD: "+",
    BinaryOperator.SUB: "-",
    BinaryOperator.MUL: "*",
    BinaryOperator.DIV: "/",
    BinaryOperator.POW: "**",
    BinaryOperator.MOD: "%",
    BinaryOperator.AND: "and",
    BinaryOperator.OR: "or",
    BinaryOperator.LT: "<",
    BinaryOperator.LE: "<=",
    BinaryOperator.EQ: "==",
    BinaryOperator.GE: ">=",
    BinaryOperator.GT: ">",
    BinaryOperator.NE: "!=",
}


@attribclass
class BinOpExpr(CompositeExpr):
    op = attribute(of=BinaryOperator)
    lhs = attribute(of=Expr)
    rhs = attribute(of=Expr)
    loc = attribute(of=Location, optional=True)


@attribclass
class TernaryOpExpr(CompositeExpr):
    condition = attribute(of=Expr)
    then_expr = attribute(of=Expr)
    else_expr = attribute(of=Expr)
    loc = attribute(of=Location, optional=True)


# ---- IR: statements ----
class Statement(Node):
    pass


class Decl(Statement):
    pass


@attribclass
class FieldDecl(Decl):
    name = attribute(of=str)
    data_type = attribute(of=DataType)
    axes = attribute(of=ListOf[str])
    is_api = attribute(of=bool)
    data_dims = attribute(of=ListOf[int], factory=list)
    layout_id = attribute(of=str, default="_default_")
    loc = attribute(of=Location, optional=True)


@attribclass
class VarDecl(Decl):
    name = attribute(of=str)
    data_type = attribute(of=DataType)
    length = attribute(of=int)
    is_api = attribute(of=bool)
    init = attribute(of=Literal, optional=True)
    loc = attribute(of=Location, optional=True)

    @property
    def is_parameter(self):
        return self.is_api

    @property
    def is_scalar(self):
        return self.length == 0


@attribclass
class BlockStmt(Statement):
    stmts = attribute(of=ListOf[Statement])
    loc = attribute(of=Location, optional=True)


@attribclass
class Assign(Statement):
    target = attribute(of=Ref)
    value = attribute(of=Expr)
    loc = attribute(of=Location, optional=True)


@attribclass
class If(Statement):
    condition = attribute(of=Expr)
    main_body = attribute(of=BlockStmt)
    else_body = attribute(of=BlockStmt, optional=True)
    loc = attribute(of=Location, optional=True)


@attribclass
class While(Statement):
    condition = attribute(of=Expr)
    body = attribute(of=BlockStmt)
    loc = attribute(of=Location, optional=True)


# ---- IR: computations ----
@enum.unique
class IterationOrder(enum.Enum):
    BACKWARD = -1
    PARALLEL = 0
    FORWARD = 1

    @property
    def symbol(self):
        if self == self.BACKWARD:
            return "<-"
        if self == self.PARALLEL:
            return "||"
        if self == self.FORWARD:
            return "->"

    def __str__(self) -> str:
        return self.name

    def __lshift__(self, steps: int):
        return self.cycle(steps=-steps)

    def __rshift__(self, steps: int):
        return self.cycle(steps=steps)


@attribclass
class AxisBound(Node):
    level = attribute(of=LevelMarker)
    offset = attribute(of=int, default=0)
    loc = attribute(of=Location, optional=True)


@attribclass
class AxisInterval(Node):
    start = attribute(of=AxisBound)
    end = attribute(of=AxisBound)
    loc = attribute(of=Location, optional=True)

    @classmethod
    def full_interval(cls, order=IterationOrder.PARALLEL) -> AxisInterval:
        if order != IterationOrder.BACKWARD:
            return cls(
                start=AxisBound(level=LevelMarker.START, offset=0),
                end=AxisBound(level=LevelMarker.END, offset=0),
            )

        return cls(
            start=AxisBound(level=LevelMarker.END, offset=-1),
            end=AxisBound(level=LevelMarker.START, offset=-1),
        )

    @property
    def is_single_index(self) -> bool:
        if not isinstance(self.start, AxisBound) or not isinstance(self.end, AxisBound):
            return False

        return self.start.level == self.end.level and self.start.offset == self.end.offset - 1

    def disjoint_from(self, other: AxisInterval) -> bool:
        def get_offset(bound: AxisBound) -> int:
            return (
                0 + bound.offset if bound.level == LevelMarker.START else sys.maxsize + bound.offset
            )

        self_start = get_offset(self.start)
        self_end = get_offset(self.end)

        other_start = get_offset(other.start)
        other_end = get_offset(other.end)

        return not (self_start <= other_start < self_end) and not (
            other_start <= self_start < other_end
        )


# TODO Find a better place for this in the file.
@attribclass
class HorizontalIf(Statement):
    intervals = attribute(of=DictOf[str, AxisInterval])
    body = attribute(of=BlockStmt)


@attribclass
class ComputationBlock(Node):
    interval = attribute(of=AxisInterval)
    iteration_order = attribute(of=IterationOrder)
    body = attribute(of=BlockStmt)
    loc = attribute(of=Location, optional=True)


@attribclass
class ArgumentInfo(Node):
    name = attribute(of=str)
    is_keyword = attribute(of=bool, default=False)
    default = attribute(of=Any, default=Empty)


@attribclass
class StencilDefinition(Node):
    name = attribute(of=str)
    domain = attribute(of=Domain)
    api_signature = attribute(of=ListOf[ArgumentInfo])
    api_fields = attribute(of=ListOf[FieldDecl])
    parameters = attribute(of=ListOf[VarDecl])
    computations = attribute(of=ListOf[ComputationBlock])
    externals = attribute(of=DictOf[str, Any], optional=True)
    sources = attribute(of=DictOf[str, str], optional=True)
    docstring = attribute(of=str, default="")
    loc = attribute(of=Location, optional=True)
