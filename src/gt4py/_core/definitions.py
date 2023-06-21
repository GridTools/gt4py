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

import dataclasses
import enum
import functools
import math
import types
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from gt4py.eve.extended_typing import (
    Final,
    Literal,
    Mapping,
)
from gt4py._core import scalars


class DeviceType(enum.Enum):
    """The type of the device where a memory buffer is allocated.

    Enum values taken from DLPack reference implementation at:
    https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
    """

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


@dataclasses.dataclass(frozen=True, slots=True)
class Device:
    """
    Representation of a computing device.

    This definition is based on the DLPack device definition. A device is
    described by a pair of `DeviceType` and `device_id`. The `device_id`
    is an integer that is interpreted differently depending on the
    `DeviceType`. For example, for `DeviceType.CPU` it could be the CPU
    core number, for `DeviceType.CUDA` it could be the CUDA device number, etc.
    """

    device_type: DeviceType
    device_id: int

    def __iter__(self) -> Iterable[int]:
        return iter((self.device_type, self.device_id))


class DTypeCode(enum.Enum):
    """
    Kind of a specific data type.

    Character codes match the value for the corresponding kind in NumPy `dtype.kind`.
    """

    INT = "i"
    UINT = "u"
    FLOAT = "f"
    OPAQUE_POINTER = "V"
    COMPLEX = "c"

    def dlpack_type_code(self) -> int:
        """
        DLPAck type code.

        Actual values taken from DLPack reference implementation at:
            https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
        """
        return _DTYPECODE_TO_DLPACK_CODE[self.value]


_DTYPECODE_TO_DLPACK_CODE: Final[
    Mapping[DTypeCode, Literal[0, 1, 2, 3, 5]]
] = types.MappingProxyType(
    {
        DTypeCode.INT: 0,
        DTypeCode.UINT: 1,
        DTypeCode.FLOAT: 2,
        DTypeCode.OPAQUE_POINTER: 3,
        DTypeCode.COMPLEX: 5,  # In DLPack bfloat16 is defined as BFLOAT = 4
    }
)


class DType(enum.Enum):
    """
    Descriptor of data type for field elements.

    This definition is based on DLPack and Array API standards. The data
    type is described by a name and a triple, `DTypeCode`, `bits`, and
    `lanes`, which should be interpreted as packed `lanes` repetitions
    of elements from `type_code` data-category of width `bits`.

    The Array API standard only requires DTypes to be comparable with `__eq__`.

    Additionally, instances of this class can also be used as valid NumPy
    `dtype`s definitions due to the `.dtype` attribute.

    Note:
        This DType definition is implemented in a non-extensible way on purpose
        to avoid the complexity of dealing with user-defined data types.
    """

    bool_ = scalars.bool_
    int8 = scalars.int8
    int16 = scalars.int16
    int32 = scalars.int32
    int64 = scalars.int64
    uint8 = scalars.uint8
    uint16 = scalars.uint16
    uint32 = scalars.uint32
    uint64 = scalars.uint64
    float32 = scalars.float32
    float64 = scalars.float64

    # Definition properties
    @functools.cached_property
    def type_code(self) -> DTypeCode:
        return DTypeCode(self.dtype.kind)

    @functools.cached_property
    def bits(self) -> int:
        assert self.dtype.itemsize % self.lanes == 0
        return 8 * (self.dtype.itemsize // self.lanes)

    @functools.cached_property
    def lanes(self) -> int:
        shape = self.dtype.shape or (1,)
        return math.prod(shape)

    # Convenience functions
    @functools.cached_property
    def scalar_type(self) -> type:
        return self.value

    @functools.cached_property
    def byte_size(self) -> int:
        return self.dtype.itemsize

    # NumPy compatibility functions
    @functools.cached_property
    def dtype(self) -> np.dtype:
        return np.dtype(self.value)

    @classmethod
    def from_np_dtype(cls, np_dtype: npt.DTypeLike) -> DType:
        return cls(np.dtype(np_dtype).type)


# DTypeLike = Union[DType, npt.DTypeLike]
