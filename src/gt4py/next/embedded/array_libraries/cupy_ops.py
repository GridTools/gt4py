# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` for CuPy arrays."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import array_api_compat.cupy as compat_cupy
import cupy as cp
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops


class CuPyOps(array_ops.ArrayOps):
    @property
    def array_ns(self) -> ModuleType:
        return cp

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        return value

    def _elementwise_function(self, name: str) -> Callable[..., Any]:
        if name == "gamma":
            import cupyx.scipy.special

            return cupyx.scipy.special.gamma
        return super()._elementwise_function(name)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        array[key] = value  # type: ignore[index] # `NDArrayObject` typing is not complete
        return array

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return cp.asnumpy(array)


array_ops.register(CuPyOps(compat_cupy))
