# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing

from mypy import plugin as mplugin, types


FLOAT_TYPES = ["builtins.float", "numpy.float32", "numpy.float64"]
INT_TYPES = [
    "builtins.int",
    "numpy.int32",
    "numpy.int64",
    "numpy.integer",
    "numpy.signedinteger",
    # "numpy.unsignedinteger",
    # "optype.numpy._scalar.integer"
]


def ignore_type(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
    return types.AnyType(types.TypeOfAny.explicit)


def blur_float_precision(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
    return ctx.api.named_type("builtins.float", [])


def blur_int_precision(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
    return ctx.api.named_type("builtins.int", [])


class TreatDimensionsAsTypes(mplugin.Plugin):
    def get_type_analyze_hook(
        self, fullname: str
    ) -> typing.Callable[[mplugin.AnalyzeTypeContext], types.Type] | None:
        if fullname.endswith("Dim"):
            return ignore_type
        elif fullname in FLOAT_TYPES:
            return blur_float_precision
        elif fullname in INT_TYPES:
            return blur_int_precision
        return None


def plugin(version: str) -> type[mplugin.Plugin]:
    return TreatDimensionsAsTypes
