# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Contains type definitions and data utilities used during lowering to SDFG.

TODO(edopao): Move existing types to this module in a refactoring PR.
"""

from __future__ import annotations

import dataclasses

import dace

from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class SymbolicData:
    gt_type: ts.ScalarType
    value: dace.symbolic.SymbolicType
