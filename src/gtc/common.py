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

import enum
import functools
import typing
from typing import Any, ClassVar, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import scipy.special

import eve
from eve import datamodels
from gtc.utils import dimension_flags_to_names, flatten_list


class GTCPreconditionError(eve.exceptions.EveError, RuntimeError):
    message_template = "GTC pass precondition error: [{info}]"

    def __init__(self, *, expected: str, **kwargs: Any) -> None:
        super().__init__(expected=expected, **kwargs)  # type: ignore


class GTCPostconditionError(eve.exceptions.EveError, RuntimeError):
    message_template = "GTC pass postcondition error: [{info}]"

    def __init__(self, *, expected: str, **kwargs: Any) -> None:
        super().__init__(expected=expected, **kwargs)  # type: ignore


class AssignmentKind(eve.StrEnum):
    """Kind of assignment: plain or combined with operations."""

    PLAIN = "="
    ADD = "+="
    SUB = "-="
    MUL = "*="
    DIV = "/="


@enum.unique
class UnaryOperator(eve.StrEnum):
    """Unary operator indentifier."""

    POS = "+"
    NEG = "-"
    NOT = "not"


@enum.unique
class ArithmeticOperator(eve.StrEnum):
    """Arithmetic operators."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MATMULT = "@"


@enum.unique
class ComparisonOperator(eve.StrEnum):
    """Comparison operators."""

    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    EQ = "=="
    NE = "!="


@enum.unique
class LogicalOperator(eve.StrEnum):
    """Logical operators."""

    AND = "and"
    OR = "or"


@enum.unique
class DataType(eve.IntEnum):
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

    def isbool(self):
        return self == self.BOOL

    def isinteger(self):
        return self in (self.INT8, self.INT32, self.INT64)

    def isfloat(self):
        return self in (self.FLOAT32, self.FLOAT64)


@enum.unique
class LoopOrder(eve.StrEnum):
    """Loop order identifier."""

    PARALLEL = "parallel"
    FORWARD = "forward"
    BACKWARD = "backward"


@enum.unique
class BuiltInLiteral(eve.StrEnum):
    MAX_VALUE = "max"
    MIN_VALUE = "min"
    ZERO = "zero"
    ONE = "one"
    TRUE = "true"
    FALSE = "false"


@enum.unique
class NativeFunction(eve.StrEnum):
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
    SINH = "sinh"
    COSH = "cosh"
    TANH = "tanh"
    ARCSINH = "arcsinh"
    ARCCOSH = "arccosh"
    ARCTANH = "arctanh"

    SQRT = "sqrt"
    POW = "pow"
    EXP = "exp"
    LOG = "log"
    GAMMA = "gamma"
    CBRT = "cbrt"

    ISFINITE = "isfinite"
    ISINF = "isinf"
    ISNAN = "isnan"
    FLOOR = "floor"
    CEIL = "ceil"
    TRUNC = "trunc"

    IR_OP_TO_NUM_ARGS: ClassVar[Dict["NativeFunction", int]]

    @property
    def arity(self) -> int:
        return self.IR_OP_TO_NUM_ARGS[self]


NativeFunction.IR_OP_TO_NUM_ARGS = {
    NativeFunction(f): v  # instead of noqa on every line
    for f, v in {
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
        NativeFunction.POW: 2,
        NativeFunction.EXP: 1,
        NativeFunction.LOG: 1,
        NativeFunction.GAMMA: 1,
        NativeFunction.CBRT: 1,
        NativeFunction.ISFINITE: 1,
        NativeFunction.ISINF: 1,
        NativeFunction.ISNAN: 1,
        NativeFunction.FLOOR: 1,
        NativeFunction.CEIL: 1,
        NativeFunction.TRUNC: 1,
    }.items()
}


@enum.unique
class LevelMarker(eve.StrEnum):
    START = "start"
    END = "end"


@enum.unique
class ExprKind(eve.IntEnum):
    SCALAR: "ExprKind" = typing.cast("ExprKind", enum.auto())
    FIELD: "ExprKind" = typing.cast("ExprKind", enum.auto())


class LocNode(eve.Node):
    loc: Optional[eve.SourceLocation] = None


@eve.utils.noninstantiable
class Expr(LocNode):
    """
    Expression base class.

    All expressions have
    - an optional `dtype`
    - an expression `kind` (scalar or field)
    """

    # Both kind and dtype are set to default here and root validators propagate the correct value after the __init__ is called.
    kind: ExprKind = ExprKind.FIELD
    dtype: DataType = DataType.AUTO


@eve.utils.noninstantiable
class Stmt(LocNode):
    pass


def verify_condition_is_boolean(parent_node_cls: datamodels.DataModel, cond: Expr) -> None:
    if cond.dtype and cond.dtype is not DataType.BOOL:
        raise ValueError("Condition in `{}` must be boolean.".format(type(parent_node_cls)))


def verify_and_get_common_dtype(
    node_cls: Type[datamodels.DataModel], exprs: List[Expr], *, strict: bool = True
) -> Optional[DataType]:
    assert len(exprs) > 0
    if all(e.dtype is not DataType.AUTO for e in exprs):
        dtypes: List[DataType] = [e.dtype for e in exprs]  # type: ignore # guaranteed to be not None
        dtype = dtypes[0]
        if strict:
            if all(dt == dtype for dt in dtypes):
                return dtype
            else:
                raise ValueError(
                    f"Type mismatch in `{node_cls.__name__}`. Types are "
                    + ", ".join(dt.name for dt in dtypes)
                )
        else:
            # upcasting
            return max(dt for dt in dtypes)
    else:
        return None


def compute_kind(*values) -> ExprKind:
    if any(v.kind == ExprKind.FIELD for v in values):
        return ExprKind.FIELD
    else:
        return ExprKind.SCALAR


class Literal(eve.Node):
    # TODO(havogt) reconsider if `str` is a good representation for value,
    # maybe it should be Union[float,int,str] etc?
    value: Union[BuiltInLiteral, str]
    dtype: DataType
    kind: ExprKind = ExprKind.SCALAR


StmtT = TypeVar("StmtT", bound=Stmt)
ExprT = TypeVar("ExprT", bound=Expr)
TargetT = TypeVar("TargetT", bound=Expr)
VariableKOffsetT = TypeVar("VariableKOffsetT")


class CartesianOffset(eve.Node):
    i: int
    j: int
    k: int

    @classmethod
    def zero(cls) -> "CartesianOffset":
        return cls(i=0, j=0, k=0)

    def to_dict(self) -> Dict[str, int]:
        return {"i": self.i, "j": self.j, "k": self.k}


class VariableKOffset(eve.GenericNode, Generic[ExprT]):
    k: ExprT

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {"i": 0, "j": 0, "k": None}

    @datamodels.validator("k")
    def offset_expr_is_int(self, attribute: datamodels.Attribute, value: Any) -> None:
        value = typing.cast(Expr, value)
        if value.dtype is not DataType.AUTO and not value.dtype.isinteger():
            raise ValueError("Variable vertical index must be an integer expression")


class ScalarAccess(LocNode):
    name: eve.Coerced[eve.SymbolRef]
    kind: ExprKind = ExprKind.SCALAR


class FieldAccess(eve.GenericNode, Generic[ExprT, VariableKOffsetT]):
    name: eve.Coerced[eve.SymbolRef]
    offset: Union[CartesianOffset, VariableKOffsetT]
    data_index: List[ExprT] = eve.field(default_factory=list)
    kind: ExprKind = ExprKind.FIELD

    @classmethod
    def centered(cls, *, name: str, loc: eve.SourceLocation = None) -> "FieldAccess":
        return cls(name=name, loc=loc, offset=CartesianOffset.zero())

    @datamodels.validator("data_index")
    def data_index_exprs_are_int(self, attribute: datamodels.Attribute, value: Any) -> None:
        value = typing.cast(List[Expr], value)
        if value and any(
            index.dtype is not DataType.AUTO and not index.dtype.isinteger() for index in value
        ):
            raise ValueError("Data indices must be integer expressions")


class BlockStmt(eve.GenericNode, eve.SymbolTableTrait, Generic[StmtT]):
    body: List[StmtT]


class IfStmt(eve.GenericNode, Generic[StmtT, ExprT]):
    """
    Generic if statement.

    Verifies that `cond` is a boolean expr (if `dtype` is set).
    """

    cond: ExprT
    true_branch: StmtT
    false_branch: Optional[StmtT] = None

    @datamodels.validator("cond")
    def condition_is_boolean(self, attribute: datamodels.Attribute, value: Expr) -> None:
        verify_condition_is_boolean(self, value)


class While(eve.GenericNode, Generic[StmtT, ExprT]):
    """
    Generic while loop.

    Verifies that `cond` is a boolean expr (if `dtype` is set).
    """

    cond: ExprT
    body: List[StmtT]

    @datamodels.validator("cond")
    def condition_is_boolean(self, attribute: datamodels.Attribute, value: Expr) -> None:
        verify_condition_is_boolean(self, value)


class AssignStmt(eve.GenericNode, Generic[TargetT, ExprT]):
    left: TargetT
    right: ExprT


def _make_root_validator(impl: datamodels.RootValidator) -> datamodels.RootValidator:
    return datamodels.root_validator(typing.cast(datamodels.RootValidator, classmethod(impl)))


def assign_stmt_dtype_validation(*, strict: bool) -> datamodels.RootValidator:
    def _impl(
        cls: Type[datamodels.DataModel],
        instance: datamodels.DataModel,
    ) -> None:
        assert isinstance(instance, AssignStmt)
        verify_and_get_common_dtype(cls, [instance.left, instance.right], strict=strict)

    return _make_root_validator(_impl)


class UnaryOp(eve.GenericNode, Generic[ExprT]):
    """
    Generic unary operation with type propagation.

    The generic `UnaryOp` already contains logic for type propagation.
    """

    op: UnaryOperator
    expr: ExprT

    @datamodels.root_validator
    @classmethod
    def dtype_propagation(cls: Type["UnaryOp"], instance: "UnaryOp") -> None:
        instance.dtype = instance.expr.dtype  # type: ignore[attr-defined]

    @datamodels.root_validator
    @classmethod
    def kind_propagation(cls: Type["UnaryOp"], instance: "UnaryOp") -> None:
        instance.kind = instance.expr.kind  # type: ignore[attr-defined]

    @datamodels.root_validator
    @classmethod
    def op_to_dtype_check(cls: Type["UnaryOp"], instance: "UnaryOp") -> None:
        if instance.expr.dtype:
            if instance.op == UnaryOperator.NOT:
                if not instance.expr.dtype == DataType.BOOL:
                    raise ValueError("Unary operator `NOT` only allowed with boolean expression.")
            else:
                if instance.expr.dtype == DataType.BOOL:
                    raise ValueError(
                        f"Unary operator `{instance.op.name}` not allowed with boolean expression."
                    )


class BinaryOp(eve.GenericNode, Generic[ExprT]):
    """Generic binary operation with type propagation.

    The generic BinaryOp already contains logic for
    - strict type checking if the `dtype` for `left` and `right` is set.
    - type propagation (taking `operator` type into account).
    """

    # consider parametrizing on op
    op: Union[ArithmeticOperator, ComparisonOperator, LogicalOperator]
    left: ExprT
    right: ExprT

    @datamodels.root_validator
    @classmethod
    def kind_propagation(cls: Type["BinaryOp"], instance: "BinaryOp") -> None:
        instance.kind = compute_kind(instance.left, instance.right)  # type: ignore[attr-defined]


def binary_op_dtype_propagation(*, strict: bool) -> datamodels.RootValidator:
    def _impl(cls: Type[BinaryOp], instance: BinaryOp) -> None:
        common_dtype = verify_and_get_common_dtype(
            cls, [instance.left, instance.right], strict=strict
        )

        if common_dtype:
            if isinstance(instance.op, ArithmeticOperator):
                if common_dtype is not DataType.BOOL:
                    instance.dtype = common_dtype  # type: ignore[attr-defined]
                else:
                    raise ValueError("Boolean expression is not allowed with arithmetic operation.")
            elif isinstance(instance.op, LogicalOperator):
                if common_dtype is DataType.BOOL:
                    instance.dtype = DataType.BOOL  # type: ignore[attr-defined]
                else:
                    raise ValueError("Arithmetic expression is not allowed in boolean operation.")
            elif isinstance(instance.op, ComparisonOperator):
                instance.dtype = DataType.BOOL  # type: ignore[attr-defined]

    return _make_root_validator(_impl)


class TernaryOp(eve.GenericNode, Generic[ExprT]):
    """
    Generic ternary operation with type propagation.

    The generic TernaryOp already contains logic for
    - strict type checking if the `dtype` for `true_expr` and `false_expr` is set.
    - type checking for `cond`
    - type propagation.
    """

    # consider parametrizing cond type and expr separately
    cond: ExprT
    true_expr: ExprT
    false_expr: ExprT

    @datamodels.validator("cond")
    def condition_is_boolean(self, attribute: datamodels.Attribute, value: Expr) -> None:
        return verify_condition_is_boolean(self, value)

    @datamodels.root_validator
    @classmethod
    def kind_propagation(cls: Type["TernaryOp"], instance: "TernaryOp") -> None:
        instance.kind = compute_kind(instance.true_expr, instance.false_expr)  # type: ignore[attr-defined]


def ternary_op_dtype_propagation(*, strict: bool) -> datamodels.RootValidator:
    def _impl(cls: Type[TernaryOp], instance: TernaryOp) -> None:
        common_dtype = verify_and_get_common_dtype(
            cls, [instance.true_expr, instance.false_expr], strict=strict
        )
        if common_dtype:
            instance.dtype = common_dtype  # type: ignore[attr-defined]

    return _make_root_validator(_impl)


class Cast(eve.GenericNode, Generic[ExprT]):
    dtype: DataType
    expr: ExprT

    @datamodels.root_validator
    @classmethod
    def kind_propagation(cls: Type["Cast"], instance: "Cast") -> None:
        instance.kind = compute_kind(instance.expr)  # type: ignore[attr-defined]


class NativeFuncCall(eve.GenericNode, Generic[ExprT]):
    func: NativeFunction
    args: List[ExprT]

    @datamodels.root_validator
    @classmethod
    def arity_check(cls: Type["NativeFuncCall"], instance: "NativeFuncCall") -> None:
        if instance.func.arity != len(instance.args):
            raise ValueError(
                f"{instance.func} accepts {instance.func.arity} arguments, {len(instance.args)} where passed."
            )

    @datamodels.root_validator
    @classmethod
    def kind_propagation(cls: Type["NativeFuncCall"], instance: "NativeFuncCall") -> None:
        instance.kind = compute_kind(*instance.args)  # type: ignore[attr-defined]


def native_func_call_dtype_propagation(*, strict: bool = True) -> datamodels.RootValidator:
    def _impl(cls: Type[NativeFuncCall], instance: NativeFuncCall) -> None:
        if instance.func in (NativeFunction.ISFINITE, NativeFunction.ISINF, NativeFunction.ISNAN):
            instance.dtype = DataType.BOOL  # type: ignore[attr-defined]
        else:
            # assumes all NativeFunction args have a common dtype
            common_dtype = verify_and_get_common_dtype(cls, instance.args, strict=strict)
            if common_dtype:
                instance.dtype = common_dtype  # type: ignore[attr-defined]

    return _make_root_validator(_impl)


def validate_dtype_is_set() -> datamodels.RootValidator:
    def _impl(cls: Type[ExprT], instance: ExprT) -> None:
        dtype_nodes: List[ExprT] = []
        for v in flatten_list(datamodels.astuple(instance)):
            if isinstance(v, eve.Node):
                dtype_nodes.extend(v.walk_values().if_hasattr("dtype"))

        nodes_without_dtype = []
        for node in dtype_nodes:
            if not node.dtype:
                nodes_without_dtype.append(node)

        if len(nodes_without_dtype) > 0:
            raise ValueError("Nodes without dtype detected {}".format(nodes_without_dtype))

    return _make_root_validator(_impl)


class _LvalueDimsValidator(eve.VisitorWithSymbolTableTrait):
    def __init__(self, vertical_loop_type: Type[eve.Node], decl_type: Type[eve.Node]) -> None:
        if not vertical_loop_type.__annotations__.get("loop_order") is LoopOrder:
            raise ValueError(
                f"Vertical loop type {vertical_loop_type} has no `loop_order` attribute"
            )
        if not decl_type.__annotations__.get("dimensions") == Tuple[bool, bool, bool]:
            raise ValueError(
                f"Field decl type {decl_type} must have a `dimensions` "
                "attribute of type `Tuple[bool, bool, bool]`."
            )
        self.vertical_loop_type = vertical_loop_type
        self.decl_type = decl_type

    def visit_Node(
        self, node: eve.Node, *, loop_order: Optional[LoopOrder] = None, **kwargs: Any
    ) -> None:
        if isinstance(node, self.vertical_loop_type):
            loop_order = getattr(node, "loop_order")  # noqa: B009
        self.generic_visit(node, loop_order=loop_order, **kwargs)

    def visit_AssignStmt(
        self, node: AssignStmt, *, loop_order: LoopOrder, symtable: Dict[str, Any], **kwargs: Any
    ) -> None:
        decl = symtable.get(node.left.name, None)
        if decl is None:
            raise ValueError("Symbol {} not found.".format(node.left.name))
        if not isinstance(decl, self.decl_type):
            return None

        allowed_flags = self._allowed_flags(loop_order)
        flags = getattr(decl, "dimensions")  # noqa: B009
        if flags not in allowed_flags:
            dims = dimension_flags_to_names(flags)
            raise ValueError(
                f"Not allowed to assign to {dims}-field `{node.left.name}` in {loop_order.name}."
            )
        return None

    def _allowed_flags(self, loop_order: LoopOrder) -> List[Tuple[bool, bool, bool]]:
        allowed_flags = [(True, True, True)]  # ijk always allowed
        if loop_order is not LoopOrder.PARALLEL:
            allowed_flags.append((True, True, False))  # ij only allowed in FORWARD and BACKWARD
        return allowed_flags


# TODO(ricoh) consider making gtir.Decl & oir.Decl common and / or adding a VerticalLoop baseclass
# TODO(ricoh) in common instead of passing type arguments
def validate_lvalue_dims(
    vertical_loop_type: Type[eve.Node], decl_type: Type[eve.Node]
) -> datamodels.RootValidator:
    """
    Validate lvalue dimensions using the root node symbol table.

    The following tree structure is expected::

        Root(`SymTableTrait`)
        |- *
           |- `vertical_loop_type`
               |- loop_order: `LoopOrder`
               |- *
                  |- AssignStmt(`AssignStmt`)
                  |- left: `Node`, validated only if reference to `decl_type` in symtable
        |- symtable_: Symtable[name, Union[`decl_type`, *]]

        DeclType
        |- dimensions: `Tuple[bool, bool, bool]`

    Parameters
    ----------
    vertical_loop_type:
        A node type with an `LoopOrder` attribute named `loop_order`
    decl_type:
        A declaration type with field dimension information in the format
        `Tuple[bool, bool, bool]` in an attribute named `dimensions`.
    """

    def _impl(cls: Type[datamodels.DataModel], instance: datamodels.DataModel) -> None:
        _LvalueDimsValidator(vertical_loop_type, decl_type).visit(instance)

    return _make_root_validator(_impl)


class AxisBound(eve.Node):
    level: LevelMarker
    offset: int = 0

    @classmethod
    def from_start(cls, offset: int) -> "AxisBound":
        return cls(level=LevelMarker.START, offset=offset)

    @classmethod
    def from_end(cls, offset: int) -> "AxisBound":
        return cls(level=LevelMarker.END, offset=offset)

    @classmethod
    def start(cls, offset: int = 0) -> "AxisBound":
        return cls.from_start(offset)

    @classmethod
    def end(cls, offset: int = 0) -> "AxisBound":
        return cls.from_end(offset)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AxisBound):
            return False
        return self.level == other.level and self.offset == other.offset

    def __lt__(self, other: "AxisBound") -> bool:
        if not isinstance(other, AxisBound):
            return NotImplemented
        return (self.level == LevelMarker.START and other.level == LevelMarker.END) or (
            self.level == other.level and self.offset < other.offset
        )

    def __le__(self, other: "AxisBound") -> bool:
        if not isinstance(other, AxisBound):
            return NotImplemented
        return self < other or self == other

    def __gt__(self, other: "AxisBound") -> bool:
        if not isinstance(other, AxisBound):
            return NotImplemented
        return not self < other and not self == other

    def __ge__(self, other: "AxisBound") -> bool:
        if not isinstance(other, AxisBound):
            return NotImplemented
        return not self < other


class HorizontalInterval(eve.Node):
    """Represents an interval of the index space in the horizontal.

    This is separate from `gtir.Interval` because the endpoints may
    be outside the compute domain.

    """

    start: Optional[AxisBound]
    end: Optional[AxisBound]

    @classmethod
    def compute_domain(cls, start_offset: int = 0, end_offset: int = 0) -> "HorizontalInterval":
        return cls(start=AxisBound.start(start_offset), end=AxisBound.end(end_offset))

    @classmethod
    def full(cls) -> "HorizontalInterval":
        return cls(start=None, end=None)

    @classmethod
    def at_endpt(
        cls, level: LevelMarker, start_offset: int, end_offset: Optional[int] = None
    ) -> "HorizontalInterval":
        if end_offset is None:
            end_offset = start_offset + 1
        return cls(
            start=AxisBound(level=level, offset=start_offset),
            end=AxisBound(level=level, offset=end_offset),
        )

    @datamodels.root_validator
    @classmethod
    def check_start_before_end(
        cls: Type["HorizontalInterval"], instance: "HorizontalInterval"
    ) -> None:
        if instance.start and instance.end and not (instance.start <= instance.end):
            raise ValueError(
                f"End ({instance.end}) is not after or equal to start ({instance.start})"
            )

    def is_single_index(self) -> bool:
        if self.start is None or self.end is None or self.start.level != self.end.level:
            return False

        return abs(self.end.offset - self.start.offset) == 1

    def overlaps(self, other: "HorizontalInterval") -> bool:
        if self.start is None and other.start is None:
            return True

        if self.start is None and other.start is not None:
            left_interval = self
            right_interval = other
        elif other.start is None or (self.start is not None and other.start < self.start):
            left_interval = other
            right_interval = self
        elif self.start is not None and other.start is not None and self.start < other.start:
            left_interval = self
            right_interval = other
        else:
            assert self.start == other.start
            return True

        if left_interval.end is None or (
            right_interval.start is not None and right_interval.start < left_interval.end
        ):
            return True

        return False


class HorizontalMask(LocNode):
    """Expr to represent a convex portion of the horizontal iteration space."""

    i: HorizontalInterval
    j: HorizontalInterval

    @property
    def intervals(self) -> Tuple[HorizontalInterval, HorizontalInterval]:
        return (self.i, self.j)


class HorizontalRestriction(eve.GenericNode, Generic[StmtT]):
    """A specialization of the horizontal space."""

    mask: HorizontalMask
    body: List[StmtT]


def data_type_to_typestr(dtype: DataType) -> str:

    table = {
        DataType.BOOL: "bool",
        DataType.INT8: "int8",
        DataType.INT16: "int16",
        DataType.INT32: "int32",
        DataType.INT64: "int64",
        DataType.FLOAT32: "float32",
        DataType.FLOAT64: "float64",
    }
    if not isinstance(dtype, DataType):
        raise TypeError("Can only convert instances of DataType to typestr.")

    if dtype not in table:
        raise ValueError("Can not convert INVALID, AUTO or DEFAULT to typestr.")
    return np.dtype(table[dtype]).str


@functools.lru_cache(maxsize=None, typed=True)  # typed since uniqueness is only guaranteed per enum
def op_to_ufunc(
    op: Union[
        UnaryOperator, ArithmeticOperator, ComparisonOperator, LogicalOperator, NativeFunction
    ]
) -> np.ufunc:
    table: Dict[
        Union[
            UnaryOperator, ArithmeticOperator, ComparisonOperator, LogicalOperator, NativeFunction
        ],
        np.ufunc,
    ]
    # Can't put all in single table since UnaryOperator.POS == BinaryOperator.ADD
    if isinstance(op, UnaryOperator):
        table = {
            UnaryOperator.POS: np.positive,
            UnaryOperator.NEG: np.negative,
            UnaryOperator.NOT: np.logical_not,
        }
    elif isinstance(op, ArithmeticOperator):
        table = {
            ArithmeticOperator.ADD: np.add,
            ArithmeticOperator.SUB: np.subtract,
            ArithmeticOperator.MUL: np.multiply,
            ArithmeticOperator.DIV: np.true_divide,
        }
    elif isinstance(op, ComparisonOperator):
        table = {
            ComparisonOperator.GT: np.greater,
            ComparisonOperator.LT: np.less,
            ComparisonOperator.GE: np.greater_equal,
            ComparisonOperator.LE: np.less_equal,
            ComparisonOperator.EQ: np.equal,
            ComparisonOperator.NE: np.not_equal,
        }
    elif isinstance(op, LogicalOperator):
        table = {
            LogicalOperator.AND: np.logical_and,
            LogicalOperator.OR: np.logical_or,
        }
    elif isinstance(op, NativeFunction):
        table = {
            NativeFunction.ABS: np.abs,
            NativeFunction.MIN: np.minimum,
            NativeFunction.MAX: np.maximum,
            NativeFunction.MOD: np.remainder,
            NativeFunction.SIN: np.sin,
            NativeFunction.COS: np.cos,
            NativeFunction.TAN: np.tan,
            NativeFunction.ARCSIN: np.arcsin,
            NativeFunction.ARCCOS: np.arccos,
            NativeFunction.ARCTAN: np.arctan,
            NativeFunction.SINH: np.sinh,
            NativeFunction.COSH: np.cosh,
            NativeFunction.TANH: np.tanh,
            NativeFunction.ARCSINH: np.arcsinh,
            NativeFunction.ARCCOSH: np.arccosh,
            NativeFunction.ARCTANH: np.arctanh,
            NativeFunction.SQRT: np.sqrt,
            NativeFunction.POW: np.power,
            NativeFunction.EXP: np.exp,
            NativeFunction.LOG: np.log,
            NativeFunction.GAMMA: scipy.special.gamma,
            NativeFunction.CBRT: np.cbrt,
            NativeFunction.ISFINITE: np.isfinite,
            NativeFunction.ISINF: np.isinf,
            NativeFunction.ISNAN: np.isnan,
            NativeFunction.FLOOR: np.floor,
            NativeFunction.CEIL: np.ceil,
            NativeFunction.TRUNC: np.trunc,
        }
    else:
        raise TypeError(
            "Can only convert instances of GTC operators and supported native functions to typestr."
        )
    return table[op]


@functools.lru_cache(maxsize=None)
def typestr_to_data_type(typestr: str) -> DataType:
    if not isinstance(typestr, str) or len(typestr) < 3 or not typestr[2:].isnumeric():
        return DataType.INVALID  # type: ignore
    table = {
        ("b", 1): DataType.BOOL,
        ("i", 1): DataType.INT8,
        ("i", 2): DataType.INT16,
        ("i", 4): DataType.INT32,
        ("i", 8): DataType.INT64,
        ("f", 4): DataType.FLOAT32,
        ("f", 8): DataType.FLOAT64,
    }
    key = (typestr[1], int(typestr[2:]))
    return table.get(key, DataType.INVALID)  # type: ignore
