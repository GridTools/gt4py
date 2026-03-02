# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import typing

from mypy import plugin as mplugin, types


class TreatDimensionsAsTypes(mplugin.Plugin):
    def get_type_analyze_hook(
        self, fullname: str
    ) -> typing.Callable[[mplugin.AnalyzeTypeContext], types.Type] | None:
        def foo(ctx: mplugin.AnalyzeTypeContext) -> types.Type:
            return types.AnyType(types.TypeOfAny.explicit)

        if re.match(r".*Dim$", fullname):
            return foo
        return None


def plugin(version: str) -> type[mplugin.Plugin]:
    return TreatDimensionsAsTypes
