# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
This module contains a mypy plugin that can be used in downstream code to simplify type checking.

Usage:

    ```toml
    # pyproject.toml
    [tool.mypy]
    plugins = ['gt4py.next.type_system.mypy_plugin']
    ```

The goal of this plugin is to reduce the amount of false positives from mypy that arise from correct
usage of GT4Py. The following are examples for such false positives:

Dimensions are not fields:

    IDim = gtx.Dimension("IDim")
    gtx.Field[gtx.Dims[IDim], float]  # IDim is not a valid type

mixed precision math / different ways of describing the same dtype:

    a: gtx.Field[gtx.Dims[IDim], gtx.float64]
    a * 0.1  # no operator overload for types 'float' and 'float64'
    # same happens for int

Every false positive fixed in here should have a test in 'typing_tests/test_next.yaml'.
The documentation for how to write tests in that format is at https://github.com/typeddjango/pytest-mypy-plugins.

The documentation on mypy plugins is at https://mypy.readthedocs.io/en/latest/extending_mypy.html
"""

from __future__ import annotations

import typing


def iter_dim_names() -> typing.Iterator[str]:
    """Go through the four distinct place holders, then yield _AnyDim for everything after."""
    yield "_DimA"
    yield "_DimB"
    yield "_DimC"
    yield "_DimD"
    while True:
        yield "_AnyDim"


# if we can not import mypy we are not type checking with mypy, so we can skip all this
try:
    from mypy import plugin as mplugin, types

    DIM_MAP: dict[str, types.Type] = {}

    FLOAT_TYPES = ["builtins.float", "numpy.float32", "numpy.float64"]
    INT_TYPES = [
        "builtins.int",
        "numpy.int32",
        "numpy.int64",
        "numpy.integer",
        "numpy.signedinteger",
    ]

    def fixup_dims_type(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
        """
        Overwrite Dims[...] to contain actual types.

        Example:

            CellDim = Dimension("CellDim")  # this is not a type
            a: Field[Dims[CellDim], gtx.float]  # we transform this to Field[Dims[_DimA]] where _DimA *is* a type

        The actual types are defined in 'gt4py.next.common', if typing.TYPE_CHECKING is true.
        """
        module_name = "gt4py.next.common"
        dims = iter_dim_names()
        args = []
        if ctx.type.args:
            # replacement 'Dims[OneDim, OtherDim]' -> 'Dims[_DimA, _DimB]' happens here
            for arg in ctx.type.args:
                argname = getattr(arg, "name", "unknown")
                if argname not in DIM_MAP:
                    DIM_MAP[argname] = ctx.api.analyze_type(
                        ctx.api.named_type(f"{module_name}.{next(dims)}", [])
                    )
                args.append(DIM_MAP[argname])
        else:
            # do not accidentally replace 'Dims' -> 'Dims[Any]' (the former matches any number of dims, the latter only one)
            args = [types.UnpackType(typ=types.AnyType(types.TypeOfAny.explicit))]
        result = ctx.api.analyze_type(ctx.api.named_type("gt4py.next.common.Dims", args))
        return result

    def fixup_dims_from_typealiases(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
        """
        Catch 'Dimension' instances that made it through other replacements, this seems to happen in type aliases.

        Example:

            T = TypeVar("T", bound=(float, float64))
            CellDim = Dimension("CellDim")
            CellField: TypeAlias = Field[Dims[CellDim], T]

        If we have seen this dimension instance before, reuse the same dim type in the replacement, else use `_AnyDim`
        """
        result: types.Type | types.AnyType = types.AnyType(
            types.TypeOfAny.explicit
        )  # Fallback to Any if _AnyDim is not found
        try:
            if ctx.type.name not in DIM_MAP:
                DIM_MAP[ctx.type.name] = ctx.api.analyze_type(
                    ctx.api.named_type("gt4py.next.common._AnyDim", [])
                )
            result = DIM_MAP[ctx.type.name]
        except AssertionError:  # this probably happens when a dim type is analyzed in a context from where _AnyDim is unreachable
            pass
        return result

    def blur_float_precision(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
        """Turn everything into 'builtins.float'."""
        # Note(ricoh): have tried to return numpy dtypes from here but ran into some error from mypy
        return ctx.api.named_type("builtins.float", [])

    def blur_int_precision(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
        """Turn everything into 'builtins.int'"""
        return ctx.api.named_type("builtins.int", [])

    class TreatDimensionsAsTypes(mplugin.Plugin):
        def get_type_analyze_hook(
            self, fullname: str
        ) -> typing.Callable[[mplugin.AnalyzeTypeContext], types.Type] | None:
            """
            Decide whether to return a callback which can modify type hints before mypy analyzes them.

            If a callback is returned, it has to return a mypy-representation of a valid type.
            """
            # replace Dims args with actual types
            if fullname == "gt4py.next.common.Dims":
                return fixup_dims_type
            # replace stray 'Dimension' instances
            elif fullname.endswith("Dim") and not fullname == "gt4py.next.common._AnyDim":
                return fixup_dims_from_typealiases
            # treat all float precision types the same (GT4Py dsl will catch actual problems)
            elif fullname in FLOAT_TYPES:
                return blur_float_precision
            # treat all int precision types the same (GT4Py dsl will catch actual problems)
            elif fullname in INT_TYPES:
                return blur_int_precision
            return None

    def plugin(version: str) -> type[mplugin.Plugin]:
        """
        This is the entry point mypy looks for if this module was pointed to in config as a plugin.
        """
        return TreatDimensionsAsTypes

except ImportError:
    pass
