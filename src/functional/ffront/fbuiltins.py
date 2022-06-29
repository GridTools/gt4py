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

from numpy import float32, float64, int32, int64

from functional.common import Dimension, Field
from functional.ffront import common_types as ct
from functional.iterator import runtime


__all__ = [
    "Field",
    "Dimension",
    "float32",
    "float64",
    "int32",
    "int64",
    "neighbor_sum",
    "broadcast",
    "where",
]


TYPE_BUILTINS = [Field, bool, int, int32, int64, float, float32, float64, tuple]
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

FUN_BUILTIN_NAMES = ["neighbor_sum", "max_over", "broadcast", "where"]


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
    source: Dimension
    target: tuple[Dimension, ...]

    def __gt_type__(self):
        return ct.OffsetType(source=self.source, target=self.target)
