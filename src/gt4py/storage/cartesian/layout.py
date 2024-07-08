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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np


if TYPE_CHECKING:
    try:
        import cupy as cp
    except ImportError:
        cp = None


class LayoutInfo(TypedDict):
    alignment: int  # measured in bytes
    device: Literal["cpu", "gpu"]
    layout_map: Callable[[Tuple[str, ...]], Tuple[Optional[int], ...]]
    is_optimal_layout: Callable[[Any, Tuple[str, ...]], bool]


REGISTRY: Dict[str, LayoutInfo] = {}


def from_name(name: str) -> Optional[LayoutInfo]:
    return REGISTRY.get(name, None)


def register(name: str, info: Optional[LayoutInfo]) -> None:
    if info is None:
        if name in REGISTRY:
            del REGISTRY[name]
    else:
        assert isinstance(name, str)
        assert isinstance(info, dict)

        REGISTRY[name] = info


def check_layout(layout_map, strides):
    if not (len(strides) == len(layout_map)):
        return False
    stride = 0
    for dim in reversed(np.argsort(layout_map)):
        if strides[dim] < stride:
            return False
        stride = strides[dim]
    return True


def layout_maker_factory(
    base_layout: Tuple[int, ...],
) -> Callable[[Tuple[str, ...]], Tuple[int, ...]]:
    def layout_maker(dimensions: Tuple[str, ...]) -> Tuple[int, ...]:
        mask = [dim in dimensions for dim in "IJK"]
        mask += [True] * (len(dimensions) - sum(mask))
        ranks = []
        for m, bl in zip(mask, base_layout):
            if m:
                ranks.append(bl)
        if len(mask) > 3:
            if base_layout[2] == 2:
                ranks.extend(3 + c for c in range(len(mask) - 3))
            else:
                ranks.extend(-c for c in range(len(mask) - 3))

        res_layout = [0] * len(ranks)
        for i, idx in enumerate(np.argsort(ranks)):
            res_layout[idx] = i
        return tuple(res_layout)

    return layout_maker


def layout_checker_factory(layout_maker):
    def layout_checker(field: Union[np.ndarray, "cp.ndarray"], dimensions: Tuple[str, ...]) -> bool:
        layout_map = layout_maker(dimensions)
        return check_layout(layout_map, field.strides)

    return layout_checker


def _permute_layout_to_dimensions(
    layout: Sequence[int], dimensions: Tuple[str, ...]
) -> Tuple[int, ...]:
    data_dims = [int(d) for d in dimensions if d.isdigit()]
    canonical_dimensions = [d for d in "IJK" if d in dimensions] + [
        str(d) for d in sorted(data_dims)
    ]
    res_layout = []
    for d in dimensions:
        res_layout.append(layout[canonical_dimensions.index(d)])
    return tuple(res_layout)


def make_gtcpu_kfirst_layout_map(dimensions: Tuple[str, ...]) -> Tuple[int, ...]:
    layout = [i for i in range(len(dimensions))]
    naxes = sum(dim in dimensions for dim in "IJK")
    layout = [*layout[-naxes:], *layout[:-naxes]]
    return _permute_layout_to_dimensions([lt for lt in layout if lt is not None], dimensions)


def make_gtcpu_ifirst_layout_map(dimensions: Tuple[str, ...]) -> Tuple[int, ...]:
    ctr = reversed(range(len(dimensions)))
    layout = [next(ctr) for dim in "IJK" if dim in dimensions] + list(ctr)
    if "K" in dimensions and "J" in dimensions:
        if "I" in dimensions:
            layout = [layout[0], layout[2], layout[1], *layout[3:]]
        else:
            layout = [layout[1], layout[0], *layout[2:]]
    return _permute_layout_to_dimensions(layout, dimensions)


def make_cuda_layout_map(dimensions: Tuple[str, ...]) -> Tuple[Optional[int], ...]:
    layout = tuple(reversed(range(len(dimensions))))
    return _permute_layout_to_dimensions(layout, dimensions)


NaiveCPULayout: Final[LayoutInfo] = {
    "alignment": 1,
    "device": "cpu",
    "layout_map": lambda axes: tuple(i for i in range(len(axes))),
    "is_optimal_layout": lambda *_: True,
}
register("naive_cpu", NaiveCPULayout)

CPUIFirstLayout: Final[LayoutInfo] = {
    "alignment": 8,
    "device": "cpu",
    "layout_map": make_gtcpu_ifirst_layout_map,
    "is_optimal_layout": layout_checker_factory(make_gtcpu_ifirst_layout_map),
}
register("cpu_ifirst", CPUIFirstLayout)


CPUKFirstLayout: Final[LayoutInfo] = {
    "alignment": 1,
    "device": "cpu",
    "layout_map": make_gtcpu_kfirst_layout_map,
    "is_optimal_layout": layout_checker_factory(make_gtcpu_kfirst_layout_map),
}
register("cpu_kfirst", CPUKFirstLayout)


CUDALayout: Final[LayoutInfo] = {
    "alignment": 32,
    "device": "gpu",
    "layout_map": make_cuda_layout_map,
    "is_optimal_layout": layout_checker_factory(make_cuda_layout_map),
}
register("cuda", CUDALayout)

GPULayout: Final[LayoutInfo] = CUDALayout
register("gpu", GPULayout)
