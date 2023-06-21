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

from __future__ import annotations

import numpy as np

from gt4py.eve.extended_typing import Final, Union, TypeAlias

# Scalar types supported by GT4Py
bool_ = np.bool_

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

float32 = np.float32
float64 = np.float64

BOOL_TYPES: Final[tuple[type, ...]] = (bool, bool_)
ScalarBoolType: TypeAlias = Union[bool, bool_]

SINT_TYPES: Final[tuple[type, ...]] = (int8, int16, int32, int64, int)
ScalarSignedIntType: TypeAlias = Union[int8, int16, int32, int64, int]

UINT_TYPES: Final[tuple[type, ...]] = (uint8, uint16, uint32, uint64)
ScalarUnsignedIntType: TypeAlias = Union[uint8, uint16, uint32, uint64]

INT_TYPES: Final[tuple[type, ...]] = (*SINT_TYPES, *UINT_TYPES)
ScalarIntType: TypeAlias = Union[ScalarSignedIntType, ScalarUnsignedIntType]

FLOAT_TYPES: Final[tuple[type, ...]] = (float32, float64, float)
ScalarFloatType: TypeAlias = Union[float32, float64, float]

SCALAR_TYPES: Final[tuple[type, ...]] = (*BOOL_TYPES, *INT_TYPES, *FLOAT_TYPES)

#: Type alias for all scalar types supported by GT4Py
ScalarType: TypeAlias = Union[ScalarBoolType, ScalarIntType, ScalarFloatType]
