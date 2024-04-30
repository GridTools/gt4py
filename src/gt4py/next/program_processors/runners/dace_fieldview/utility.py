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

from typing import Any, Mapping

import dace

from gt4py.next.common import Connectivity
from gt4py.next.type_system import type_specifications as ts


def as_dace_type(type_: ts.ScalarType) -> dace.dtypes.typeclass:
    if type_.kind == ts.ScalarKind.BOOL:
        return dace.bool_
    elif type_.kind == ts.ScalarKind.INT32:
        return dace.int32
    elif type_.kind == ts.ScalarKind.INT64:
        return dace.int64
    elif type_.kind == ts.ScalarKind.FLOAT32:
        return dace.float32
    elif type_.kind == ts.ScalarKind.FLOAT64:
        return dace.float64
    raise ValueError(f"Scalar type '{type_}' not supported.")


def filter_connectivities(offset_provider: Mapping[str, Any]) -> dict[str, Connectivity]:
    return {
        offset: table
        for offset, table in offset_provider.items()
        if isinstance(table, Connectivity)
    }


def unique_name(prefix: str) -> str:
    unique_id = getattr(unique_name, "_unique_id", 0)  # static variable
    setattr(unique_name, "_unique_id", unique_id + 1)  # noqa: B010 [set-attr-with-constant]

    return f"{prefix}_{unique_id}"
