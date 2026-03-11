# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing


DIM_MAP = {}


def iter_dim_names() -> typing.Iterator[str]:
    yield "DimA"
    yield "DimB"
    yield "DimC"
    yield "DimD"
    while True:
        yield "AnyDim"


try:
    from mypy import plugin as mplugin, types

    FLOAT_TYPES = ["builtins.float", "numpy.float32", "numpy.float64"]
    INT_TYPES = [
        "builtins.int",
        "numpy.int32",
        "numpy.int64",
        "numpy.integer",
        "numpy.signedinteger",
    ]

    def fixup_dims_type(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
        module_name = "gt4py.next.common"
        dims = iter_dim_names()
        args = []
        if ctx.type.args:
            for arg in ctx.type.args:
                argname = getattr(arg, "name", "unknown")
                if argname not in DIM_MAP:
                    DIM_MAP[argname] = ctx.api.analyze_type(
                        ctx.api.named_type(f"{module_name}.{next(dims)}", [])
                    )
                args.append(DIM_MAP[argname])
        result = ctx.api.analyze_type(ctx.api.named_type("gt4py.next.common.Dims", args))
        return result

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
            if fullname == "gt4py.next.common.Dims":
                return fixup_dims_type
            elif fullname in FLOAT_TYPES:
                return blur_float_precision
            elif fullname in INT_TYPES:
                return blur_int_precision
            return None

    def plugin(version: str) -> type[mplugin.Plugin]:
        return TreatDimensionsAsTypes

except ImportError:
    pass
