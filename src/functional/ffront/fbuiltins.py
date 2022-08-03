# GT4Py Project - GridTools Framework
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

from builtins import bool, float, int, tuple
from dataclasses import dataclass

from numpy import float32, float64, int32, int64

from functional.common import Dimension, DimensionKind, Field
from functional.ffront import common_types as ct
from functional.iterator import runtime


PYTHON_TYPE_BUILTINS = [bool, int, float, tuple]
PYTHON_TYPE_BUILTIN_NAMES = [t.__name__ for t in PYTHON_TYPE_BUILTINS]

TYPE_BUILTINS = [Field, Dimension, int32, int64, float32, float64] + PYTHON_TYPE_BUILTINS
TYPE_BUILTIN_NAMES = [t.__name__ for t in TYPE_BUILTINS]


@dataclass
class BuiltInFunction:
    __gt_type: ct.FunctionType

    def __call__(self, *args, **kwargs):
        """Act as an empty place holder for the built in function."""

    def __gt_type__(self):
        return self.__gt_type


_reduction_like = BuiltInFunction(
    ct.FunctionType(
        args=[ct.DeferredSymbolType(constraint=ct.FieldType)],
        kwargs={"axis": ct.DeferredSymbolType(constraint=ct.DimensionType)},
        returns=ct.DeferredSymbolType(constraint=ct.FieldType),
    )
)

neighbor_sum = _reduction_like
max_over = _reduction_like

broadcast = BuiltInFunction(
    ct.FunctionType(
        args=[
            ct.DeferredSymbolType(constraint=(ct.FieldType, ct.ScalarType)),
            ct.DeferredSymbolType(constraint=ct.TupleType),
        ],
        kwargs={},
        returns=ct.DeferredSymbolType(constraint=ct.FieldType),
    )
)

where = BuiltInFunction(
    ct.FunctionType(
        args=[
            ct.DeferredSymbolType(constraint=ct.FieldType),
            ct.DeferredSymbolType(constraint=(ct.FieldType, ct.ScalarType)),
            ct.DeferredSymbolType(constraint=(ct.FieldType, ct.ScalarType)),
        ],
        kwargs={},
        returns=ct.DeferredSymbolType(constraint=ct.FieldType),
    )
)

_unary_math_builtin = BuiltInFunction(
    ct.FunctionType(
        args=[ct.DeferredSymbolType(constraint=(ct.ScalarType, ct.FieldType))],
        kwargs={},
        returns=ct.DeferredSymbolType(constraint=(ct.ScalarType, ct.FieldType)),
    )
)

# unary math builtins (number) -> number
abs = _unary_math_builtin  # noqa: A001

UNARY_MATH_NUMBER_BUILTIN_NAMES = ["abs"]

# unary math builtins (float) -> float
sin = _unary_math_builtin
cos = _unary_math_builtin
tan = _unary_math_builtin
arcsin = _unary_math_builtin
arccos = _unary_math_builtin
arctan = _unary_math_builtin
sinh = _unary_math_builtin
cosh = _unary_math_builtin
tanh = _unary_math_builtin
arcsinh = _unary_math_builtin
arccosh = _unary_math_builtin
arctanh = _unary_math_builtin
sqrt = _unary_math_builtin
exp = _unary_math_builtin
log = _unary_math_builtin
gamma = _unary_math_builtin
cbrt = _unary_math_builtin
floor = _unary_math_builtin
ceil = _unary_math_builtin
trunc = _unary_math_builtin

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
_unary_math_predicate_builtin = BuiltInFunction(
    ct.FunctionType(
        args=[ct.DeferredSymbolType(constraint=(ct.ScalarType, ct.FieldType))],
        kwargs={},
        returns=ct.DeferredSymbolType(constraint=(ct.ScalarType, ct.FieldType)),
    )
)

isfinite = _unary_math_predicate_builtin
isinf = _unary_math_predicate_builtin
isnan = _unary_math_predicate_builtin

UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES = ["isfinite", "isinf", "isnan"]

# binary math builtins (number, number) -> number
_binary_math_builtin = BuiltInFunction(
    ct.FunctionType(
        args=[
            ct.DeferredSymbolType(constraint=(ct.ScalarType, ct.FieldType)),
            ct.DeferredSymbolType(constraint=(ct.ScalarType, ct.FieldType)),
        ],
        kwargs={},
        returns=ct.DeferredSymbolType(constraint=(ct.ScalarType, ct.FieldType)),
    )
)

minimum = _binary_math_builtin
maximum = _binary_math_builtin
fmod = _binary_math_builtin
power = _binary_math_builtin

BINARY_MATH_NUMBER_BUILTIN_NAMES = ["minimum", "maximum", "fmod", "power"]

MATH_BUILTIN_NAMES = (
    UNARY_MATH_NUMBER_BUILTIN_NAMES
    + UNARY_MATH_FP_BUILTIN_NAMES
    + UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + BINARY_MATH_NUMBER_BUILTIN_NAMES
)

FUN_BUILTIN_NAMES = ["neighbor_sum", "max_over", "broadcast", "where"] + MATH_BUILTIN_NAMES


EXTERNALS_MODULE_NAME = "__externals__"
MODULE_BUILTIN_NAMES = [EXTERNALS_MODULE_NAME]

BUILTIN_NAMES = TYPE_BUILTIN_NAMES + FUN_BUILTIN_NAMES

BUILTINS = {name: globals()[name] for name in BUILTIN_NAMES}

__all__ = BUILTIN_NAMES


# TODO(tehrengruber): FieldOffset and runtime.Offset are not an exact conceptual
#  match. Revisit if we want to continue subclassing here. If we split
#  them also check whether Dimension should continue to be the shared or define
#  guidelines for decision.
@dataclass(frozen=True)
class FieldOffset(runtime.Offset):
    source: Dimension
    target: tuple[Dimension] | tuple[Dimension, Dimension]

    def __post_init__(self):
        if len(self.target) == 2 and self.target[1].kind != DimensionKind.LOCAL:
            raise ValueError("Second dimension in offset must be a local dimension.")

    def __gt_type__(self):
        return ct.OffsetType(source=self.source, target=self.target)
