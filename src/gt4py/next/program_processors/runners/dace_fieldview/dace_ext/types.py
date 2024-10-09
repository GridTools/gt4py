# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
from dace import properties as dace_properties


@dace_properties.make_properties
class MaskedArray(dace.data.Array):
    offset: str
