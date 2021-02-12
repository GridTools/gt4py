# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
    `SQRT`, `EXP`, `LOG`, `ISFINITE`, `ISINF`, `ISNAN`, `FLOOR`, `CEIL`, `TRUNC`]

AccessIntent enumeration (:class:`AccessIntent`)
    Access permissions
    [`READ_ONLY`, `READ_WRITE`]

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

    Domain(parallel_axes: List[Axis], [sequential_axis: Axis, data_axes: List[Axis]])
        # LatLonGrids -> parallel_axes: ["I", "J], sequential_axis: "K"

    Literal     = ScalarLiteral(value: Any (should match DataType), data_type: DataType)
                | BuiltinLiteral(value: Builtin)

    Ref         = VarRef(name: str, [index: int])
                | FieldRef(name: str, offset: Dict[str, int])

    NativeFuncCall(func: NativeFunction, args: List[Expr], data_type: DataType)

    Cast(dtype: DataType, expr: Expr)

    Expr        = Literal | Ref | NativeFuncCall | Cast | CompositeExpr | InvalidBranch

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


-----------------
Implementation IR
-----------------
 ::

    Accessor    = ParameterAccessor(symbol: str)
                | FieldAccessor(symbol: str, intent: AccessIntent, extent: Extent)

    ApplyBlock(interval: AxisInterval,
               local_symbols: Dict[str, VarDecl],
               body: BlockStmt)

    Stage(name: str,
          accessors: List[Accessor],
          apply_blocks: List[ApplyBlock],
          compute_extent: Extent)

    StageGroup(stages: List[Stage])

    MultiStage(name: str, iteration_order: IterationOrder, groups: List[StageGroups])

    StencilImplementation(name: str,
                          api_signature: List[ArgumentInfo],
                          domain: Domain,
                          fields: Dict[str, FieldDecl],
                          parameters: Dict[str, VarDecl],
                          multi_stages: List[MultiStage],
                          fields_extents: Dict[str, Extent],
                          unreferenced: List[str],
                          [axis_splitters_var: str, externals: Dict[str, Any], sources: Dict[str, str]])

"""

import collections
import copy
import enum
import operator
from typing import List, Sequence

import numpy as np

from gt4py import utils as gt_utils
from gt4py.definitions import CartesianSpace, Extent, Index
from gt4py.utils.attrib import Any as Any
from gt4py.utils.attrib import Dict as DictOf
from gt4py.utils.attrib import List as ListOf
from gt4py.utils.attrib import Optional as OptionalOf
from gt4py.utils.attrib import Tuple as TupleOf
from gt4py.utils.attrib import Union as UnionOf
from gt4py.utils.attrib import attribkwclass as attribclass
from gt4py.utils.attrib import attribute, attributes_of


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
    def from_ast_node(cls, ast_node, scope="<source>"):
        return cls(line=ast_node.lineno, column=ast_node.col_offset + 1, scope=scope)


# ---- IR: domain ----
@attribclass
class Axis(Node):
    name = attribute(of=str)


@attribclass
class Domain(Node):
    parallel_axes = attribute(of=ListOf[Axis])
    sequential_axis = attribute(of=Axis, optional=True)
    data_axes = attribute(of=ListOf[Axis], optional=True)

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
        if self.data_axes:
            result.extend(self.data_axes)
        return result

    @property
    def axes_names(self):
        return [ax.name for ax in self.axes]

    @property
    def ndims(self):
        return self.domain_ndims + self.data_ndims

    @property
    def domain_ndims(self):
        return len(self.parallel_axes) + (1 if self.sequential_axis else 0)

    @property
    def data_ndims(self):
        return len(self.data_axes) if self.data_axes else 0

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
    def from_value(cls, value):
        if value is None:
            result = cls.NONE
        elif value is True:
            result = cls.TRUE
        elif value is False:
            result = cls.FALSE

        return result

    def __str__(self):
        return self.name


@enum.unique
class AccessIntent(enum.Enum):
    READ_ONLY = 0
    READ_WRITE = 1

    def __str__(self):
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

    def __str__(self):
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

DataType.NUMPY_TO_NATIVE_TYPE = {
    value: key for key, value in DataType.NATIVE_TYPE_TO_NUMPY.items() if key != DataType.DEFAULT
}


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


# @attribclass
# class TupleLiteral(Node):
#     items = attribute(of=TupleOf[Expr])
#
#     @property
#     def length(self):
#         return len(self.items)


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
class FieldRef(Ref):
    name = attribute(of=str)
    offset = attribute(of=DictOf[str, int])
    loc = attribute(of=Location, optional=True)

    @classmethod
    def at_center(cls, name: str, axes: Sequence[str], loc=None):
        return cls(name=name, offset={axis: 0 for axis in axes}, loc=loc)


@attribclass
class Cast(Expr):
    data_type = attribute(of=DataType)
    expr = attribute(of=Expr)
    loc = attribute(of=Location, optional=True)


@enum.unique
class NativeFunction(enum.Enum):
    ABS = 1
    MIN = 2
    MAX = 3
    MOD = 4

    SIN = 5
    COS = 6
    TAN = 7
    ARCSIN = 8
    ARCCOS = 9
    ARCTAN = 10

    SQRT = 11
    EXP = 12
    LOG = 13

    ISFINITE = 14
    ISINF = 15
    ISNAN = 16
    FLOOR = 17
    CEIL = 18
    TRUNC = 19

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


# @attribclass
# class ExprStmt(Statement):
#     expr = attribute(of=Expr)
#     loc = attribute(of=Location, optional=True)


class Decl(Statement):
    pass


@attribclass
class FieldDecl(Decl):
    name = attribute(of=str)
    data_type = attribute(of=DataType)
    axes = attribute(of=ListOf[str])
    is_api = attribute(of=bool)
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


# ---- IR: computations ----
@enum.unique
class LevelMarker(enum.Enum):
    START = 0
    END = -1

    def __str__(self):
        return self.name


@enum.unique
class IterationOrder(enum.Enum):
    BACKWARD = -1
    PARALLEL = 0
    FORWARD = 1

    @property
    def symbol(self):
        if self == self.BACKWARD:
            return "<-"
        elif self == self.PARALLEL:
            return "||"
        elif self == self.FORWARD:
            return "->"

    def __str__(self):
        return self.name

    def __lshift__(self, steps: int):
        return self.cycle(steps=-steps)

    def __rshift__(self, steps: int):
        return self.cycle(steps=steps)


@attribclass
class AxisBound(Node):
    level = attribute(of=UnionOf[LevelMarker, VarRef])
    offset = attribute(of=int, default=0)
    loc = attribute(of=Location, optional=True)


@attribclass
class AxisInterval(Node):
    start = attribute(of=AxisBound)
    end = attribute(of=AxisBound)
    loc = attribute(of=Location, optional=True)

    @classmethod
    def full_interval(cls, order=IterationOrder.PARALLEL):
        if order != IterationOrder.BACKWARD:
            interval = cls(
                start=AxisBound(level=LevelMarker.START, offset=0),
                end=AxisBound(level=LevelMarker.END, offset=0),
            )
        else:
            interval = cls(
                start=AxisBound(level=LevelMarker.END, offset=-1),
                end=AxisBound(level=LevelMarker.START, offset=-1),
            )

        return interval


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


# ---- Implementation IR (IIR) ----
class IIRNode(Node):
    pass


class Accessor(IIRNode):
    pass


@attribclass
class ParameterAccessor(Accessor):
    symbol = attribute(of=str)


@attribclass
class FieldAccessor(Accessor):
    symbol = attribute(of=str)
    intent = attribute(of=AccessIntent)
    extent = attribute(of=Extent, default=Extent.zeros())


@attribclass
class ApplyBlock(Node):
    interval = attribute(of=AxisInterval)
    local_symbols = attribute(of=DictOf[str, VarDecl])
    body = attribute(of=BlockStmt)


@attribclass
class Stage(IIRNode):
    name = attribute(of=str)
    accessors = attribute(of=ListOf[Accessor])
    apply_blocks = attribute(of=ListOf[ApplyBlock])
    compute_extent = attribute(of=Extent)


# @enum.unique
# class CacheKind(enum.Enum):
#     PARALLEL = 0
#     LAYERED = 1
#
#
# @enum.unique
# class CachePolicy(enum.Enum):
#     NONE = 0b0
#     FILL = 0b01
#     FLUSH = 0b10
#     FILL_AND_FLUSH = 0b11
#
#     @property
#     def has_fill(self):
#         return self.value & self.FILL
#
#     @property
#     def has_flush(self):
#         return self.value & self.FLUSH
#
#
# @attribclass
# class Cache(IIRNode):
#     field = attribute(of=str)
#     kind = attribute(of=CacheKind)
#     policy = attribute(of=CachePolicy)


@attribclass
class StageGroup(IIRNode):
    stages = attribute(of=ListOf[Stage])


@attribclass
class MultiStage(IIRNode):
    name = attribute(of=str)
    iteration_order = attribute(of=IterationOrder)
    # caches = attribute(of=ListOf[Cache])
    groups = attribute(of=ListOf[StageGroup])


@attribclass
class StencilImplementation(IIRNode):
    name = attribute(of=str)
    api_signature = attribute(of=ListOf[ArgumentInfo])
    domain = attribute(of=Domain)
    fields = attribute(of=DictOf[str, FieldDecl])  # All fields, including temporaries
    parameters = attribute(of=DictOf[str, VarDecl])
    multi_stages = attribute(of=ListOf[MultiStage])
    fields_extents = attribute(of=DictOf[str, Extent])
    unreferenced = attribute(of=ListOf[str], factory=list)
    axis_splitters_var = attribute(of=str, optional=True)
    externals = attribute(of=DictOf[str, Any], optional=True)
    sources = attribute(of=DictOf[str, str], optional=True)
    docstring = attribute(of=str)

    @property
    def has_effect(self):
        """
        Determine whether the stencil modifies any of its arguments.

        Note that the only guarantee of this function is that the stencil has no effect if it returns ``false``. It
        might however return true in cases where the optimization passes were not able to deduce this.
        """
        return self.multi_stages and not all(
            arg_field in self.unreferenced for arg_field in self.arg_fields
        )

    @property
    def arg_fields(self):
        result = [f.name for f in self.fields.values() if f.is_api]
        return result

    @property
    def temporary_fields(self):
        result = [f.name for f in self.fields.values() if not f.is_api]
        return result


# ---- Helpers ----
def iter_attributes(node: Node):
    """
    Yield a tuple of ``(attrib_name, value)`` for each attribute in ``node.attributes``
    that is present on *node*.
    """
    for attrib_name in node.attributes:
        try:
            yield attrib_name, getattr(node, attrib_name)
        except AttributeError:
            pass


class IRNodeVisitor:
    def visit(self, node: Node, **kwargs):
        return self._visit(node, **kwargs)

    def _visit(self, node: Node, **kwargs):
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(node, **kwargs)

    def generic_visit(self, node: Node, **kwargs):
        items = []
        if isinstance(node, (str, bytes, bytearray)):
            pass
        elif isinstance(node, collections.abc.Mapping):
            items = node.items()
        elif isinstance(node, collections.abc.Iterable):
            items = enumerate(node)
        elif isinstance(node, Node):
            items = iter_attributes(node)
        else:
            pass

        for key, value in items:
            self._visit(value, **kwargs)


class IRNodeInspector:
    def visit(self, node: Node):
        return self._visit((), None, node)

    def _visit(self, path: tuple, node_name: str, node):
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(path, node_name, node)

    def generic_visit(self, path: tuple, node_name: str, node: Node):
        items = []
        if isinstance(node, (str, bytes, bytearray)):
            pass
        elif isinstance(node, collections.abc.Mapping):
            items = node.items()
        elif isinstance(node, collections.abc.Iterable):
            items = enumerate(node)
        elif isinstance(node, Node):
            items = iter_attributes(node)
        else:
            pass

        for key, value in items:
            self._visit((*path, node_name), key, value)


class IRNodeMapper:
    def visit(self, node: Node):
        keep_node, new_node = self._visit((), None, node)
        return new_node if keep_node else None

    def _visit(self, path: tuple, node_name: str, node: Node):
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(path, node_name, node)

    def generic_visit(self, path: tuple, node_name: str, node: Node):
        if isinstance(node, (str, bytes, bytearray)):
            return True, node
        elif isinstance(node, collections.abc.Iterable):
            if isinstance(node, collections.abc.Mapping):
                items = node.items()
            else:
                items = enumerate(node)
            setattr_op = operator.setitem
            delattr_op = operator.delitem
        elif isinstance(node, Node):
            items = iter_attributes(node)
            setattr_op = setattr
            delattr_op = delattr
        else:
            return True, node

        del_items = []
        for key, old_value in items:
            keep_item, new_value = self._visit((*path, node_name), key, old_value)
            if not keep_item:
                del_items.append(key)
            elif new_value != old_value:
                setattr_op(node, key, new_value)
        for key in reversed(del_items):  # reversed, so that keys remain valid in sequences
            delattr_op(node, key)

        return True, node


class IRNodeDumper(IRNodeMapper):
    @classmethod
    def apply(cls, root_node, *, as_json=False):
        return cls(as_json=as_json)(root_node)

    def __init__(self, as_json: bool):
        self.as_json = as_json

    def __call__(self, node):
        result = self.visit(copy.deepcopy(node))
        if self.as_json:
            result = gt_utils.jsonify(result)
        return result

    def visit_Node(self, path: tuple, node_name: str, node: Node):
        object_name = node.__class__.__name__.split(".")[-1]
        keep_node, new_node = self.generic_visit(path, node_name, node)
        return keep_node, {object_name: new_node.as_dict()}


dump_ir = IRNodeDumper.apply
