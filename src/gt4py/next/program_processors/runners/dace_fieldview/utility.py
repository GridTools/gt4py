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

import dace

from gt4py.next.type_system import type_specifications as ts


def as_dace_type(type_: ts.ScalarKind) -> dace.dtypes.typeclass:
    if type_ == ts.ScalarKind.BOOL:
        return dace.bool_
    elif type_ == ts.ScalarKind.INT32:
        return dace.int32
    elif type_ == ts.ScalarKind.INT64:
        return dace.int64
    elif type_ == ts.ScalarKind.FLOAT32:
        return dace.float32
    elif type_ == ts.ScalarKind.FLOAT64:
        return dace.float64
    raise ValueError(f"Data type '{type_}' not supported.")
