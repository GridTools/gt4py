# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from functools import lru_cache

import numpy as np
from dace import dtypes, symbolic

from gt4py import eve
from gt4py.cartesian.gtc import common


def data_type_to_dace_typeclass(data_type: common.DataType) -> dtypes.typeclass:
    dtype = np.dtype(common.data_type_to_typestr(data_type))
    return dtypes.typeclass(dtype.type)


@lru_cache(maxsize=None)
def get_dace_symbol(
    name: eve.SymbolRef, dtype: common.DataType = common.DataType.INT32
) -> symbolic.symbol:
    return symbolic.symbol(name, dtype=data_type_to_dace_typeclass(dtype))
