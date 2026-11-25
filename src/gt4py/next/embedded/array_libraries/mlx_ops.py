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
MLX arrays, so the `ArrayOps` are registered for it. What MLX lacks is added in that one
namespace, and not at the places using it, to keep `NdArrayField` and the builtins free of
library-specific code.

Devices: MLX arrays have no device. MLX runs every operation on its default device, so a
field reports that one as its device, and a `device` argument is only accepted if it names
it: any other would claim a device the arrays do not compute on. The default device is
global state of the process, owned by the user: GT4Py never changes it.

Dtypes: MLX does not support float64 on the GPU, and by itself converts float64 data to
float32. Fields never change dtype silently, as results would then differ from the ones of
the other array libraries without notice: float64 on the GPU raises, and users pass
`dtype=float32` there, also for the `init` of a scan. See `_mlx_namespace`.

Evaluation: MLX evaluates lazily, an operation adds a node to a graph, which is computed
when its values are needed. A write into a field (e.g. to the `out` of a field operator)
and the fields returned by a field operator call outside of any field operator are
evaluated, so that a time loop does not build up an ever larger graph. Operations inside a
field operator are not evaluated one by one, to keep the fusion MLX does on the graph, and
field arithmetic outside of field operators stays lazy until its values are read.

Limits: MLX fields run only in embedded execution, as MLX gives no stable pointer and
strides for the buffer info compiled backends need. Slices of MLX arrays are copies, so
writes into a sliced `out` field are lost, as with JAX. A scan evaluates at every point.
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
        if isinstance(value, np.generic):
            # in the dtype of `array`: a float64 0-d array is not allowed on the GPU
            value = self.xp.asarray(value, dtype=array.dtype)
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
