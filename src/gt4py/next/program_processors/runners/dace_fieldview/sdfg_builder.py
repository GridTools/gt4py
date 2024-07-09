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
"""
Visitor interface to build an SDFG dataflow.

"""

import dataclasses

from gt4py import eve
from gt4py.next import common as gtx_common
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class SDFGBuilder(eve.NodeVisitor):
    offset_provider: dict[str, gtx_common.Connectivity | gtx_common.Dimension]
    symbol_types: dict[str, ts.FieldType | ts.ScalarType]
