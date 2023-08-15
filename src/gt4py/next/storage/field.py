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

from typing import Optional, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.storage import allocators


# Public interface
def empty(
    domain: common.Domain,
    dtype: core_defs.DTypeLike = np.float64,
    *,
    layout_info,  #: LayoutInfoInterface,
    aligned_index: Optional[Sequence[int]] = None,
) -> common.Field:
    """Allocate an array of uninitialized (undefined) values with performance-optimal strides and alignment.

    !!!TODO!!!

    Parameters
    ----------
        shape : `Sequence` of `int`
            The shape of the resulting `ndarray`
        dtype :  DTypeLike, optional
            The dtype of the resulting `ndarray`

    Keyword Arguments
    -----------------
        backend : `str`
            The target backend for which the allocation is optimized.
        aligned_index: `Sequence` of `int`, optional
            Indicate the index of the resulting array that most commonly corresponds to the origin of the compute
            domain. If not passed, it is aligned at the data origin.
        dimensions: `Sequence` of `str`, optional
            Indicate the semantic meaning of the dimensions in the provided array. Only used for determining optimal
            strides, the information is not stored.

    Returns
    -------
        NumPy or CuPy ndarray
            With uninitialized values, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    # assert storage_info is not None
    # if storage_info["device"] == "gpu":
    #     allocate_f = storage_utils.allocate_gpu
    # else:
    #     allocate_f = storage_utils.allocate_cpu

    # aligned_index, shape, dtype, dimensions = storage_utils.normalize_storage_spec(
    #     aligned_index, shape, dtype, dimensions
    # )

    # _error_on_invalid_preset(backend)

    # alignment = storage_info["alignment"]
    # layout_map = storage_info["layout_map"](dimensions)
    # assert allocators.is_valid_layout_map(layout_map)

    # _, res = allocate_f(shape, layout_map, dtype, alignment * dtype.itemsize, aligned_index)

    dtype = core_defs.dtype(dtype)
    shape = domain.shape
    layout_map = list(range(len(shape)))
    device = layout_info.device  # TODO define LayoutInfo

    buffer = allocators.allocate(shape, dtype, layout_map, byte_alignment=1, device=device)

    return common.field(buffer.ndarray, domain=domain)  # TODO alignment etc


def ones(
    domain: common.Domain,
    dtype: core_defs.DTypeLike = np.float64,
    *,
    layout_info,  #: LayoutInfoInterface,
    aligned_index: Optional[Sequence[int]] = None,
) -> common.Field:
    """Allocate an array with values initialized to 1.0 with performance-optimal strides and alignment.

    Parameters
    ----------
        shape : `Sequence` of `int`
            The shape of the resulting `ndarray`
        dtype :  DTypeLike, optional
            The dtype of the resulting `ndarray`

    Keyword Arguments
    -----------------
        backend : `str`
            The target backend for which the allocation is optimized.
        aligned_index: `Sequence` of `int`, optional
            Indicate the index of the resulting array that most commonly corresponds to the origin of the compute
            domain. If not passed, it is aligned at the data origin.
        dimensions: `Sequence` of `str`, optional
            Indicate the semantic meaning of the dimensions in the provided array. Only used for determining optimal
            strides, the information is not stored.

    Returns
    -------
        NumPy or CuPy ndarray
            Initialized to 1.0, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    field = empty(
        domain=domain,
        dtype=dtype,
        layout_info=layout_info,
        aligned_index=aligned_index,
    )
    field.ndarray[...] = field.value_type(1)
    return field
