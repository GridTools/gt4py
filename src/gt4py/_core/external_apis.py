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

import enum

from typing import Union, overload, Literal

import gt4py._core.definitions as definitions


# class DeviceType(enum.Enum):
#     """The type of the device where a memory buffer is allocated.

#     Enum values taken from DLPack reference implementation at:
#     https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
#     """

#     CPU = 1
#     CUDA = 2
#     CPU_PINNED = 3
#     OPENCL = 4
#     VULKAN = 7
#     METAL = 8
#     VPI = 9
#     ROCM = 10


# @overload
# def dtype_kind(sc_type: type[BoolScalar]) -> Literal[DTypeKind.BOOL]:  # type: ignore[misc]
#     ...


# @overload
# def dtype_kind(sc_type: type[Union[int8, int16, int32, int64, int]]) -> Literal[DTypeKind.INT]:
#     ...


# @overload
# def dtype_kind(sc_type: type[UnsignedIntScalar]) -> Literal[DTypeKind.UINT]:
#     ...


# @overload
# def dtype_kind(sc_type: type[FloatingScalar]) -> Literal[DTypeKind.FLOAT]:
#     ...


# @overload
# def dtype_kind(sc_type: type[Scalar]) -> DTypeKind:
#     ...


# def dtype_kind(sc_type: type[Scalar]) -> DTypeKind:
#     """Return the data type kind of the given scalar type."""
#     if issubclass(sc_type, numbers.Integral):
#         if is_boolean_integral_type(sc_type):
#             return DTypeKind.BOOL
#         elif is_unsigned_integral_type(sc_type):
#             return DTypeKind.UINT
#         else:
#             return DTypeKind.INT
#     if issubclass(sc_type, numbers.Real):
#         return DTypeKind.FLOAT
#     if issubclass(sc_type, numbers.Complex):
#         return DTypeKind.COMPLEX

#     raise TypeError("Unknown scalar type kind")
