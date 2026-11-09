# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` for NumPy arrays."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import array_api_compat
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops


def _np_gamma(a: core_defs.NDArrayObject) -> core_defs.NDArrayObject:
    try:
        from scipy.special import gamma
    except ImportError:
        return np.vectorize(math.gamma, otypes=[a.dtype])(a)
    return gamma(a)


class NumPyOps(array_ops.ArrayOps):
    @property
    def array_ns(self) -> ModuleType:
        return np

    def asarray(
        self,
        data: Any,
        *,
        dtype: Any = None,
        like: core_defs.NDArrayObject | None = None,
    ) -> core_defs.NDArrayObject:
        return np.asarray(data, dtype=dtype)  # type: ignore[return-value] # `NDArrayObject` typing is not complete

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        return value

    def _elementwise(self, name: str) -> Callable[..., Any]:
        if name == "gamma":
            return _np_gamma
        return super()._elementwise(name)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        array[key] = value  # type: ignore[index] # `NDArrayObject` typing is not complete
        return array

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return np.asarray(array)


array_ops.register(array_api_compat.is_numpy_namespace, NumPyOps)
