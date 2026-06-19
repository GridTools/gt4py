# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING, Any, Callable, Literal, Tuple, TypedDict, Union

import numpy as np


if TYPE_CHECKING:
    try:
        import cupy as cp
    except ImportError:
        cp = None


class LayoutInfo(TypedDict):
    alignment: int  # measured in bytes
    device: Literal["cpu", "gpu"]
    layout_map: Callable[[Tuple[str, ...]], Tuple[int, ...]]
    is_optimal_layout: Callable[[Any, Tuple[str, ...]], bool]


def layout_maker_factory(
    base_layout: Tuple[int, ...],
) -> Callable[[Tuple[str, ...]], Tuple[int, ...]]:
    def layout_maker(dimensions: Tuple[str, ...]) -> Tuple[int, ...]:
        """Create layout from a given list of dimensions. Cartesian dimensions
        are given precedence so they follow the requested layout, other dimensions
        will be appended to it."""

        # Sort cartesian layout
        mask = [dim in dimensions for dim in "IJK"]
        ranks = []
        for m, bl in zip(mask, base_layout):
            if m:
                ranks.append(bl)

        # Sort data dimensions
        # - shift all cartesian layout
        data_dimensions_size = len(dimensions) - sum(mask)
        ranks = [data_dimensions_size + r for r in ranks]
        # - extend ranks with the required amount of data dimensions
        for ddim in range(data_dimensions_size):
            ranks.append(ddim)

        # Turn ranks into memory layout
        res_layout = [0] * len(ranks)
        for i, idx in enumerate(np.argsort(ranks)):
            res_layout[idx] = i
        return tuple(res_layout)

    return layout_maker


def _check_layout(layout_map, strides):
    if not (len(strides) == len(layout_map)):
        return False
    stride = 0
    for dim in reversed(np.argsort(layout_map)):
        if strides[dim] < stride:
            return False
        stride = strides[dim]
    return True


def layout_checker_factory(layout_maker):
    def layout_checker(field: Union[np.ndarray, "cp.ndarray"], dimensions: Tuple[str, ...]) -> bool:
        layout_map = layout_maker(dimensions)
        return _check_layout(layout_map, field.strides)

    return layout_checker
