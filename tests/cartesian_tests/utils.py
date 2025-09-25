# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import Any, Union

try:
    import dace
except ImportError:
    dace = None

try:
    import cupy as cp
except ImportError:
    cp = None


class ArrayWrapper:
    def __init__(self, array: Union[np.ndarray, "cp.ndarray"], **_kwargs: Any) -> None:
        self.array = array

    @property
    def __array_interface__(self):
        return self.array.__array_interface__

    @property
    def __cuda_array_interface__(self):
        return self.array.__cuda_array_interface__

    def __descriptor__(self):
        return dace.data.create_datadescriptor(self.array)


class DimensionsWrapper(ArrayWrapper):
    def __init__(self, dimensions: tuple[str, ...], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if len(self.array.shape) != len(dimensions):
            raise ValueError(
                f"Non matching dimensions of array.shape {self.array.shape} and dimensions {dimensions}."
            )
        self.__gt_dims__ = dimensions


class OriginWrapper(ArrayWrapper):
    def __init__(self, *, origin: tuple[int, ...], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if len(self.array.shape) != len(origin):
            raise ValueError(
                f"Non matching dimensions of array.shape {self.array.shape} and origin {origin}."
            )
        self.__gt_origin__ = origin

    def __descriptor__(self):
        res = super().__descriptor__()
        res.__gt_origin__ = self.__gt_origin__
        return res
