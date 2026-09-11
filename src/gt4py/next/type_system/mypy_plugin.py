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

mixed precision math / different ways of describing the same dtype:

    a: gtx.Field[gtx.Dims[IDim], gtx.float64]
    a * 0.1  # no operator overload for types 'float' and 'float64'
    # same happens for int

Every false positive fixed in here should have a test in 'typing_tests/test_next.yaml'.
The documentation for how to write tests in that format is at https://github.com/typeddjango/pytest-mypy-plugins.

The documentation on mypy plugins is at https://mypy.readthedocs.io/en/latest/extending_mypy.html

Dimensions no longer need plugin support: a concrete dimension is a class
('class IDim(gtx.DimensionIndex): ...'), which is a valid annotation for any type checker. See ADR
0028. Only the mixed-precision hooks below remain; this plugin is scheduled for removal once
dtype-generic fields land.
"""

from __future__ import annotations

import typing


# if we can not import mypy we are not type checking with mypy, so we can skip all this
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

    def blur_float_precision(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
        """Turn everything into 'builtins.float'."""
        # Note(ricoh): have tried to return numpy dtypes from here but ran into some error from mypy
        return ctx.api.named_type("builtins.float", [])

    def blur_int_precision(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
        """Turn everything into 'builtins.int'"""
        return ctx.api.named_type("builtins.int", [])

    class BlurScalarPrecision(mplugin.Plugin):
        def get_type_analyze_hook(
            self, fullname: str
        ) -> typing.Callable[[mplugin.AnalyzeTypeContext], types.Type] | None:
            """
            Decide whether to return a callback which can modify type hints before mypy analyzes them.

            If a callback is returned, it has to return a mypy-representation of a valid type.
            """
            # treat all float precision types the same (GT4Py dsl will catch actual problems)
            if fullname in FLOAT_TYPES:
                return blur_float_precision
            # treat all int precision types the same (GT4Py dsl will catch actual problems)
            elif fullname in INT_TYPES:
                return blur_int_precision
            return None

    #: Deprecated alias: downstream configs may still name the old symbol.
    TreatDimensionsAsTypes = BlurScalarPrecision

    def plugin(version: str) -> type[mplugin.Plugin]:
        """
        This is the entry point mypy looks for if this module was pointed to in config as a plugin.
        """
        return BlurScalarPrecision

except ImportError:
    pass
