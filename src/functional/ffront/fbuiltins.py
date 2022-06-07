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

from builtins import bool, float, int
from dataclasses import dataclass
from typing import Optional

from numpy import float32, float64, int32, int64

from functional.common import Dimension, Field
from functional.ffront import common_types as ct
from functional.iterator import runtime


# FIXME(ben): need to clean up changes of math built-ins (quite messy currently)
__all__ = ["Field", "Dimension", "float32", "float64", "int32", "int64", "neighbor_sum", "abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "sqrt", "exp", "log", "gamma", "cbrt", "isfinite", "isinf", "isnan", "floor", "ceil", "trunc", "min", "max", "mod"]

TYPE_BUILTINS = [Field, bool, int, int32, int64, float, float32, float64, tuple]
TYPE_BUILTIN_NAMES = [t.__name__ for t in TYPE_BUILTINS]


@dataclass
class BuiltInFunction:
    __gt_type: ct.FunctionType

    def __call__(self, *args, **kwargs):
        """Act as an empty place holder for the built in function."""

    def __gt_type__(self):
        return self.__gt_type


neighbor_sum = BuiltInFunction(
    ct.FunctionType(
        args=[ct.DeferredSymbolType(constraint=ct.FieldType)],
        kwargs={"axis": ct.DeferredSymbolType(constraint=ct.DimensionType)},
        returns=ct.DeferredSymbolType(constraint=ct.FieldType),
    )
)


_single_arg_math_built_in = BuiltInFunction(
    ct.FunctionType(
        args=[ct.DeferredSymbolType(constraint=ct.FieldType)],
        kwargs={},
        returns=ct.DeferredSymbolType(constraint=ct.FieldType),
    )
)

abs = _single_arg_math_built_in
sin = _single_arg_math_built_in
cos = _single_arg_math_built_in
tan = _single_arg_math_built_in
arcsin = _single_arg_math_built_in
arccos = _single_arg_math_built_in
arctan = _single_arg_math_built_in
sinh = _single_arg_math_built_in
cosh = _single_arg_math_built_in
tanh = _single_arg_math_built_in
arcsinh = _single_arg_math_built_in
arccosh = _single_arg_math_built_in
arctanh = _single_arg_math_built_in
sqrt = _single_arg_math_built_in
exp = _single_arg_math_built_in
log = _single_arg_math_built_in
gamma = _single_arg_math_built_in
cbrt = _single_arg_math_built_in
isfinite = _single_arg_math_built_in
isinf = _single_arg_math_built_in
isnan = _single_arg_math_built_in
floor = _single_arg_math_built_in
ceil = _single_arg_math_built_in
trunc = _single_arg_math_built_in

del _single_arg_math_built_in
SINGLE_ARG_MATH_BUILT_INS = ["abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "sqrt", "exp", "log", "gamma", "cbrt", "isfinite", "isinf", "isnan", "floor", "ceil", "trunc"]

_double_arg_math_built_in = BuiltInFunction(
    ct.FunctionType(
        args=[ct.DeferredSymbolType(constraint=ct.FieldType), ct.DeferredSymbolType(constraint=ct.FieldType)],
        kwargs={},
        returns=ct.DeferredSymbolType(constraint=ct.FieldType),
    )
)

min = _double_arg_math_built_in
max = _double_arg_math_built_in
mod = _double_arg_math_built_in

del _double_arg_math_built_in
DOUBLE_ARG_MATH_BUILT_IN = ["min", "max", "mod"]

FUN_BUILTIN_NAMES = ["neighbor_sum"] + SINGLE_ARG_MATH_BUILT_INS + DOUBLE_ARG_MATH_BUILT_IN

EXTERNALS_MODULE_NAME = "__externals__"
MODULE_BUILTIN_NAMES = [EXTERNALS_MODULE_NAME]

ALL_BUILTIN_NAMES = TYPE_BUILTIN_NAMES + MODULE_BUILTIN_NAMES

BUILTINS = {name: globals()[name] for name in __all__ + ["bool", "int", "float"]}


# TODO(tehrengruber): FieldOffset and runtime.Offset are not an exact conceptual
#  match. Revisit if we want to continue subclassing here. If we split
#  them also check whether Dimension should continue to be the shared or define
#  guidelines for decision.
@dataclass(frozen=True)
class FieldOffset(runtime.Offset):
    source: Optional[Dimension] = None
    target: Optional[tuple[Dimension, ...]] = None

    def __gt_type__(self):
        return ct.OffsetType(source=self.source, target=self.target)
