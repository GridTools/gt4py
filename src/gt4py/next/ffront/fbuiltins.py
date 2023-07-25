# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from builtins import bool, float, int, tuple
from dataclasses import dataclass
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    ParamSpec,
    TypeAlias,
    TypeVar,
    Union,
    Sequence,
    Tuple,
)
import inspect
import enum
from numpy import float32, float64, int32, int64
import functools

from gt4py.next.common import Dimension, DimensionKind, Field, ScalarT
from gt4py.next.ffront.experimental import as_offset  # noqa F401
from gt4py.next.iterator import runtime
from gt4py.next.type_system import type_specifications as ts


PYTHON_TYPE_BUILTINS = [bool, int, float, tuple]
PYTHON_TYPE_BUILTIN_NAMES = [t.__name__ for t in PYTHON_TYPE_BUILTINS]

TYPE_BUILTINS = [Field, Dimension, int32, int64, float32, float64] + PYTHON_TYPE_BUILTINS
TYPE_BUILTIN_NAMES = [t.__name__ for t in TYPE_BUILTINS]

# Be aware: Type aliases are not fully supported in the frontend yet, e.g. `IndexType(1)` will not
# work.
IndexType: TypeAlias = int32

TYPE_ALIAS_NAMES = ["IndexType"]


Value: TypeAlias = Any  # definitions.ScalarT, Field
P = ParamSpec("P")
R = TypeVar("R", Value, tuple[Value, ...])


def _type_conversion_helper(t: type):
    if t is Field:
        return ts.FieldType
    elif t is Dimension:
        return ts.DimensionType
    elif t is ScalarT:
        return ts.ScalarType
    elif t is Tuple:
        return ts.TupleType
    elif hasattr(t, "__origin__") and t.__origin__ is Union:
        return tuple(_type_conversion_helper(e) for e in t.__args__)
    else:
        assert False, t


def _type_conversion(t):
    return ts.DeferredType(constraint=_type_conversion_helper(t))


@dataclass  # (frozen=True)
class BuiltInFunction(Generic[R, P]):
    # name: str
    __gt_type: ts.FunctionType
    # `function` can be used to provide a default implementation for all `Field` implementations,
    # e.g. a fused multiply add could have a default implementation as a*b+c, but an optimized implementation for a specific `Field`
    function: Callable[P, R] = None  # TODO remove None

    def __post_init__(self):
        functools.update_wrapper(
            self, self.function
        )  # TODO figure out keeping function annotations in autocomplete

    def __call__(self, *args: Value, **options: Any) -> Value | tuple[Value, ...]:
        impl = self.dispatch(*args)
        return impl(*args, **options)

    def dispatch(self, *args: Value) -> Callable[P, R]:
        arg_types = tuple(type(arg) for arg in args)
        for atype in arg_types:
            if (dispatcher := getattr(atype, "__gt_op_func__", None)) is not None and (
                op_func := dispatcher(self)
            ) is not NotImplemented:
                return op_func
        else:
            return self.function

    def __gt_type__(self) -> ts.FunctionType:
        if self.__gt_type is not None:
            return self.__gt_type  # TODO remove
        signature = inspect.signature(self.function)
        params = signature.parameters

        return ts.FunctionType(
            pos_only_args=[
                _type_conversion(param.annotation)
                for param in params.values()
                if param.kind == inspect.Parameter.POSITIONAL_ONLY
            ],
            pos_or_kw_args={
                k: _type_conversion(v.annotation)
                for k, v in params.items()
                if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            },
            kw_only_args={
                k: _type_conversion(v.annotation)
                for k, v in params.items()
                if v.kind == inspect.Parameter.KEYWORD_ONLY
            },
            returns=_type_conversion(signature.return_annotation),
        )

    def __hash__(self) -> int:
        return hash(id(self))  # TODO fix


_reduction_like = lambda: BuiltInFunction(
    ts.FunctionType(
        pos_only_args=[ts.DeferredType(constraint=ts.FieldType)],
        pos_or_kw_args={"axis": ts.DeferredType(constraint=ts.DimensionType)},
        kw_only_args={},
        returns=ts.DeferredType(constraint=ts.FieldType),
    )
)


# def _filter_params(
#     params: Sequence[inspect.Parameter], kind: enum.IntEnum
# ) -> list[inspect.Parameter]:
#     return [param.annotation for param in params.values() if param.kind == kind]


def builtin_function(fun: Callable[P, R]) -> BuiltInFunction[R, P]:
    return BuiltInFunction(None, fun)  # TODO remove None


@builtin_function
def neighbor_sum(
    field: Field,
    /,
    axis: Dimension,
) -> Field:
    ...


@builtin_function
def max_over(
    field: Field,
    /,
    axis: Dimension,
) -> Field:
    ...


@builtin_function
def min_over(
    field: Field,
    /,
    axis: Dimension,
) -> Field:
    ...


@builtin_function
def broadcast(_: Field | ScalarT, __: Tuple, /) -> Field:
    ...


@builtin_function
def where(
    _: Field,
    __: Field | ScalarT | Tuple,
    ___: Field | ScalarT | Tuple,
    /,
) -> Field | Tuple:
    ...


astype = BuiltInFunction(
    ts.FunctionType(
        pos_only_args=[
            ts.DeferredType(constraint=ts.FieldType),
            ts.DeferredType(constraint=ts.FunctionType),
        ],
        pos_or_kw_args={},
        kw_only_args={},
        returns=ts.DeferredType(constraint=ts.FieldType),
    )
)

_unary_math_builtin = lambda: BuiltInFunction(
    ts.FunctionType(
        pos_only_args=[ts.DeferredType(constraint=(ts.ScalarType, ts.FieldType))],
        pos_or_kw_args={},
        kw_only_args={},
        returns=ts.DeferredType(constraint=(ts.ScalarType, ts.FieldType)),
    )
)

# unary math builtins (number) -> number
abs = _unary_math_builtin()  # noqa: A001

UNARY_MATH_NUMBER_BUILTIN_NAMES = ["abs"]

# unary math builtins (float) -> float
sin = _unary_math_builtin()
cos = _unary_math_builtin()
tan = _unary_math_builtin()
arcsin = _unary_math_builtin()
arccos = _unary_math_builtin()
arctan = _unary_math_builtin()
sinh = _unary_math_builtin()
cosh = _unary_math_builtin()
tanh = _unary_math_builtin()
arcsinh = _unary_math_builtin()
arccosh = _unary_math_builtin()
arctanh = _unary_math_builtin()
sqrt = _unary_math_builtin()
exp = _unary_math_builtin()
log = _unary_math_builtin()
gamma = _unary_math_builtin()
cbrt = _unary_math_builtin()
floor = _unary_math_builtin()
ceil = _unary_math_builtin()
trunc = _unary_math_builtin()

UNARY_MATH_FP_BUILTIN_NAMES = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sqrt",
    "exp",
    "log",
    "gamma",
    "cbrt",
    "floor",
    "ceil",
    "trunc",
]

# unary math predicates (float) -> bool
_unary_math_predicate_builtin = lambda: BuiltInFunction(
    ts.FunctionType(
        pos_only_args=[ts.DeferredType(constraint=(ts.ScalarType, ts.FieldType))],
        pos_or_kw_args={},
        kw_only_args={},
        returns=ts.DeferredType(constraint=(ts.ScalarType, ts.FieldType)),
    )
)

isfinite = _unary_math_predicate_builtin()
isinf = _unary_math_predicate_builtin()
isnan = _unary_math_predicate_builtin()

UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES = ["isfinite", "isinf", "isnan"]

# binary math builtins (number, number) -> number
_binary_math_builtin = lambda: BuiltInFunction(
    ts.FunctionType(
        pos_only_args=[
            ts.DeferredType(constraint=(ts.ScalarType, ts.FieldType)),
            ts.DeferredType(constraint=(ts.ScalarType, ts.FieldType)),
        ],
        pos_or_kw_args={},
        kw_only_args={},
        returns=ts.DeferredType(constraint=(ts.ScalarType, ts.FieldType)),
    )
)

minimum = _binary_math_builtin()
maximum = _binary_math_builtin()
fmod = _binary_math_builtin()
power = _binary_math_builtin()

BINARY_MATH_NUMBER_BUILTIN_NAMES = ["minimum", "maximum", "fmod", "power"]

MATH_BUILTIN_NAMES = (
    UNARY_MATH_NUMBER_BUILTIN_NAMES
    + UNARY_MATH_FP_BUILTIN_NAMES
    + UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + BINARY_MATH_NUMBER_BUILTIN_NAMES
)

FUN_BUILTIN_NAMES = [
    "neighbor_sum",
    "max_over",
    "min_over",
    "broadcast",
    "where",
    "astype",
    "as_offset",
] + MATH_BUILTIN_NAMES

BUILTIN_NAMES = TYPE_BUILTIN_NAMES + FUN_BUILTIN_NAMES

BUILTINS = {name: globals()[name] for name in BUILTIN_NAMES}

__all__ = [*((set(BUILTIN_NAMES) | set(TYPE_ALIAS_NAMES)) - {"Dimension", "Field"})]


# TODO(tehrengruber): FieldOffset and runtime.Offset are not an exact conceptual
#  match. Revisit if we want to continue subclassing here. If we split
#  them also check whether Dimension should continue to be the shared or define
#  guidelines for decision.
@dataclass(frozen=True)
class FieldOffset(runtime.Offset):
    source: Dimension
    target: tuple[Dimension] | tuple[Dimension, Dimension]
    connectivity: Optional[Any] = None  # TODO

    def __post_init__(self):
        if len(self.target) == 2 and self.target[1].kind != DimensionKind.LOCAL:
            raise ValueError("Second dimension in offset must be a local dimension.")

    def __gt_type__(self):
        return ts.OffsetType(source=self.source, target=self.target)

    def __getitem__(self, index):
        return lambda i: i  # TODO
        # return FieldOffset(self.source, self.target, index)
