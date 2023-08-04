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

from dataclasses import dataclass

from gt4py.next.type_system import type_specifications as ts


@dataclass
class BuiltInFunction:
    __gt_type: ts.FunctionType

    def __call__(self, *args, **kwargs):
        """Act as an empty place holder for the built in function."""

    def __gt_type__(self):
        return self.__gt_type


as_offset = BuiltInFunction(
    ts.FunctionType(
        pos_only_args=[
            ts.DeferredType(constraint=ts.OffsetType),
            ts.DeferredType(constraint=ts.FieldType),
        ],
        pos_or_kw_args={},
        kw_only_args={},
        returns=ts.DeferredType(constraint=ts.OffsetType),
    )
)
