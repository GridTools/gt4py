# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
`ArrayOps` for MLX arrays.

Field code computes on MLX arrays through `_mlx_namespace`, a namespace of the Array API
standard around `mlx.core`. `array_api_compat` gives `mlx.core` itself as the namespace of
MLX arrays, so the `ArrayOps` are registered for it.

MLX arrays have no device: MLX runs every operation on its default device, which GT4Py
never changes, and a field reports it as its device. MLX does not support float64 on the
GPU, see `_mlx_namespace`.

MLX evaluates lazily: an operation adds a node to a graph, which is computed when its
values are needed. A write into a field (e.g. to the `out` of a field operator) and the
fields returned by a field operator call outside of any field operator are evaluated, so
that a time loop does not build up an ever larger graph.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import mlx.core as mx
import numpy as np

from gt4py._core import definitions as core_defs, ndarray_utils as core_ndarray_utils
from gt4py.next import common
from gt4py.next.embedded import array_ops
from gt4py.next.embedded.array_libraries import _mlx_namespace


class MlxOps(array_ops.ArrayOps):
    @property
    def array_ns(self) -> ModuleType:
        return mx

    @property
    def own_allocator(self) -> Any:
        # an MLX array stays one: the default allocator of its device would copy it to NumPy
        return mx

    def device(self, array: core_defs.NDArrayObject) -> Any:
        return mx.default_device()

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        # MLX replaces the array behind `array` by the result of the write
        array = super().setitem(array, key, value)
        self.evaluate(array)
        return array

    def evaluate(self, array: core_defs.NDArrayObject) -> None:
        mx.eval(array)

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        # a copy: a NumPy view of the memory of an MLX array changes with writes into it
        return np.array(array, copy=True)

    def buffer_info(self, array: core_defs.NDArrayObject) -> core_ndarray_utils.BufferInfo:
        raise NotImplementedError(
            "Fields of MLX arrays have no buffer info: they run only in embedded execution."
        )


array_ops.register(MlxOps(_mlx_namespace), namespace=mx)
