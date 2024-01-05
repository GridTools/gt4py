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

import numpy as np

from gt4py.next import common
from gt4py.next.ffront.fbuiltins import BuiltInFunction, FieldOffset


@BuiltInFunction
def as_offset(
    offset_: FieldOffset,
    field: common.Field,
    /,
) -> common.ConnectivityField:
    offset_dim = np.squeeze(
        np.where(list(map(lambda x: x == offset_.source, field.domain.dims)))
    ).item()
    new_connectivity = np.indices(field.ndarray.shape)[offset_dim] + field.ndarray
    return common.connectivity(new_connectivity, codomain=offset_.source, domain=field.domain)


EXPERIMENTAL_FUN_BUILTIN_NAMES = ["as_offset"]
