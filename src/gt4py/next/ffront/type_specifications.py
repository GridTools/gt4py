# GT4Py - GridTools Framework
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

from dataclasses import dataclass

import gt4py.next.type_system.type_specifications as ts
from gt4py.next import common as func_common


@dataclass(frozen=True)
class ProgramType(ts.TypeSpec, ts.CallableType):
    definition: ts.FunctionType


@dataclass(frozen=True)
class FieldOperatorType(ts.TypeSpec, ts.CallableType):
    definition: ts.FunctionType


@dataclass(frozen=True)
class ScanOperatorType(ts.TypeSpec, ts.CallableType):
    axis: func_common.Dimension
    definition: ts.FunctionType
