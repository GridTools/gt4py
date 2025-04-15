# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import types

import numpy as np

from gt4py._core import definitions as core_defs


try:
    import cupy
except ImportError:
    cupy = None


def array_namespace(array: core_defs.NDArrayObject) -> types.ModuleType:
    """
    Get the namespace of the array.

    This is defined in https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html,
    however not implemented in CuPy < 14.
    """
    if hasattr(array, "__array_namespace__"):
        return array.__array_namespace__()
    else:
        if isinstance(array, np.ndarray):
            return np
        if cupy is not None and isinstance(array, cupy.ndarray):
            return cupy
        raise TypeError(f"Could not determine array namespace of {array} of type {type(array)}")
