# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Final, Literal

from gt4py._core.definitions import DeviceType


try:
    import cupy as cp
except ImportError:
    cp = None


CUPY_DEVICE: Final[Literal[DeviceType.CUDA, DeviceType.ROCM] | None] = (
    None if not cp else (DeviceType.ROCM if cp.cuda.get_hipcc_path() else DeviceType.CUDA)
)
