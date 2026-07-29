# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next.type_system.type_specifications as ts
from gt4py.next import common


class ProgramType(ts.CallableType):
    definition: ts.FunctionType


class FieldOperatorType(ts.CallableType):
    definition: ts.FunctionType


class ScanOperatorType(ts.CallableType):
    axis: common.Dimension
    definition: ts.FunctionType
