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

from dataclasses import dataclass
from typing import Any, Callable, Optional

from numpy import float32, float64, int32, int64

from functional.common import Dimension, Field
from functional.ffront import common_types as ct
from functional.iterator import runtime


__all__ = ["Field", "float32", "float64", "int32", "int64", "neighbor_sum"]


@dataclass
class BuiltInFunction:
    __gt_type: ct.FunctionType

    def __call__(self, *args, **kwargs):
        """Act as an empty place holder for the built in function."""

    def __gt_type__(self):
        return self.__gt_type


@dataclass
class ConstantConstructor(BuiltInFunction):
    constructor: Callable[[str], Any]

    def __init__(self, constructor):
        super().__init__(
            ct.FunctionType(
                args=[ct.DeferredSymbolType(constraint=None)],
                kwargs={},
                returns=ct.DeferredSymbolType(constraint=ct.FieldType),
            )
        )
        self.constructor = constructor

    def __call__(self, value: str, *args, **kwargs):
        return self.constructor(value)

    @property
    def __name__(self):
        return self.constructor.__name__


float32_ = ConstantConstructor(constructor=float32)
float64_ = ConstantConstructor(constructor=float64)
int32_ = ConstantConstructor(constructor=int32)
int64_ = ConstantConstructor(constructor=int64)


TYPE_BUILTINS = [Field, float, float32, float64, int, int32, int64, bool, tuple]
TYPE_BUILTIN_NAMES = [t.__name__ for t in TYPE_BUILTINS]


neighbor_sum = BuiltInFunction(
    ct.FunctionType(
        args=[ct.DeferredSymbolType(constraint=ct.FieldType)],
        kwargs={"axis": ct.ScalarType(kind=ct.ScalarKind.DIMENSION)},
        returns=ct.DeferredSymbolType(constraint=ct.FieldType),
    )
)


FUN_BUILTIN_NAMES = ["neighbor_sum", "float32_", "float64_", "int32_", "int64_"]


EXTERNALS_MODULE_NAME = "__externals__"
MODULE_BUILTIN_NAMES = [EXTERNALS_MODULE_NAME]

ALL_BUILTIN_NAMES = TYPE_BUILTIN_NAMES + MODULE_BUILTIN_NAMES

BUILTINS = {name: globals()[name] for name in __all__}


# TODO(ricoh): This should probably be reunified with ``iterator.runtime.Offset``
# potentially in ``functional.common``, which requires lifting of
# ``ffront.common_types`` into ``functional``.
@dataclass(frozen=True)
class FieldOffset(runtime.Offset):
    source: Optional[Dimension] = None
    target: Optional[tuple[Dimension, ...]] = None

    def __gt_type__(self):
        return ct.OffsetType(source=self.source, target=self.target)
