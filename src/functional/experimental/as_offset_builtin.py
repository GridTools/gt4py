# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
import ast
from builtins import bool, float, int, tuple
from dataclasses import dataclass

from numpy import float32, float64, int32, int64

from functional.common import Dimension, DimensionKind, Field
from functional.ffront import fbuiltins, field_operator_ast as foast, type_specifications as ts
from functional.ffront.fbuiltins import BuiltInFunction
from functional.ffront.func_to_foast import FieldOperatorSyntaxError
from functional.iterator import runtime


as_offset = BuiltInFunction(
    ts.FunctionType(
        args=[
            ts.DeferredType(constraint=ts.DimensionType),
            ts.DeferredType(constraint=ts.FieldType),
        ],
        kwargs={},
        returns=ts.DeferredType(constraint=ts.FieldType),
    )
)
