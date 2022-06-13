# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
import dace.data
import dace.library
import dace.subsets

from gtc import common


def get_dace_debuginfo(node: common.LocNode):

    if node.loc is not None:
        return dace.dtypes.DebugInfo(
            node.loc.line,
            node.loc.column,
            node.loc.line,
            node.loc.column,
            node.loc.source,
        )
    else:
        return dace.dtypes.DebugInfo(0)
