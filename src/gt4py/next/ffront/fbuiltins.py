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

import dataclasses
import inspect
from builtins import bool, float, int, tuple
from typing import Any, Callable, Generic, Optional, ParamSpec, Tuple, TypeAlias, TypeVar, Union

from numpy import float32, float64, int32, int64

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
    elif t is type:
        return (
            ts.FunctionType
        )  # our type of type is currently represented by the type constructor function
    elif t is Tuple:
        return ts.TupleType
    elif hasattr(t, "__origin__") and t.__origin__ is Union:
        return tuple(_type_conversion_helper(e) for e in t.__args__)
    else:
        raise AssertionError("Illegal type encountered.")


def _type_conversion(t):
    return ts.DeferredType(constraint=_type_conversion_helper(t))


@dataclasses.dataclass(frozen=True)
class BuiltInFunction(Generic[R, P]):
    name: str = dataclasses.field(init=False)
    # `function` can be used to provide a default implementation for all `Field` implementations,
    # e.g. a fused multiply add could have a default implementation as a*b+c, but an optimized implementation for a specific `Field`
    function: Callable[P, R]

    def __post_init__(self):
        # functools.update_wrapper(
        #     self, self.function
        # )  # TODO figure out keeping function annotations in autocomplete
        object.__setattr__(self, "name", f"{self.function.__module__}.{self.function.__name__}")

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


def builtin_function(fun: Callable[P, R]) -> BuiltInFunction[R, P]:
    return BuiltInFunction(fun)


@builtin_function
def neighbor_sum(
    field: Field,
    /,
    axis: Dimension,
) -> Field:
    raise NotImplementedError()


@builtin_function
def max_over(
    field: Field,
    /,
    axis: Dimension,
) -> Field:
    raise NotImplementedError()


@builtin_function
def min_over(
    field: Field,
    /,
    axis: Dimension,
) -> Field:
    raise NotImplementedError()


@builtin_function
def broadcast(field: Field | ScalarT, dims: Tuple, /) -> Field:
    raise NotImplementedError()


@builtin_function
def where(
    mask: Field,
    true_field: Field | ScalarT | Tuple,
    false_field: Field | ScalarT | Tuple,
    /,
) -> Field | Tuple:
    raise NotImplementedError()


@builtin_function
def astype(field: Field, type_: type, /) -> Field:
    raise NotImplementedError()


UNARY_MATH_NUMBER_BUILTIN_NAMES = ["abs"]

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


UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES = ["isfinite", "isinf", "isnan"]


def _make_unary_math_builtin(name):
    def impl(value: Field | ScalarT, /) -> Field | ScalarT:
        raise NotImplementedError()

    impl.__name__ = name
    globals()[name] = builtin_function(impl)


for f in (
    UNARY_MATH_NUMBER_BUILTIN_NAMES
    + UNARY_MATH_FP_BUILTIN_NAMES
    + UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
):
    _make_unary_math_builtin(f)


BINARY_MATH_NUMBER_BUILTIN_NAMES = ["minimum", "maximum", "fmod", "power"]


def _make_binary_math_builtin(name):
    def impl(lhs: Field | ScalarT, rhs: Field | ScalarT, /) -> Field | ScalarT:
        raise NotImplementedError()

    impl.__name__ = name
    globals()[name] = builtin_function(impl)


for f in BINARY_MATH_NUMBER_BUILTIN_NAMES:
    _make_binary_math_builtin(f)

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
@dataclasses.dataclass(frozen=True)
class FieldOffset(runtime.Offset):
    source: Dimension
    target: tuple[Dimension] | tuple[Dimension, Dimension]
    connectivity: Optional[Any] = None  # TODO

    def __post_init__(self):
        if len(self.target) == 2 and self.target[1].kind != DimensionKind.LOCAL:
            raise ValueError("Second dimension in offset must be a local dimension.")

    def __gt_type__(self):
        return ts.OffsetType(source=self.source, target=self.target)
