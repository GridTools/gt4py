# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

import numpy as np

import gt4py._core.definitions as core_defs
from gt4py.storage.cartesian.cupy_device import CUPY_DEVICE


if TYPE_CHECKING:
    try:
        import cupy as cp
    except ImportError:
        cp = None

LayoutMap: TypeAlias = Callable[[Tuple[str, ...]], Tuple[Optional[int], ...]]


class LayoutInfo(TypedDict):
    alignment: int  # measured in bytes
    device: core_defs.DeviceType | None
    layout_map: LayoutMap
    is_optimal_layout: Callable[[Any, Tuple[str, ...]], bool]


# Registry of LayoutInfos per backend
REGISTRY: Dict[str, LayoutInfo] = {}


def from_name(backend_name: str) -> Optional[LayoutInfo]:
    """Fetch LayoutInfo from the registry for a given backend name."""
    return REGISTRY.get(backend_name, None)


def register(backend_name: str, info: Optional[LayoutInfo]) -> None:
    """ "Register LayoutInfo under the given backend name. Clears an existing registry entry if None is given as info."""
    if info is None:
        if backend_name in REGISTRY:
            del REGISTRY[backend_name]
        return

    assert isinstance(backend_name, str)
    assert isinstance(info, dict)

    REGISTRY[backend_name] = info


def is_cpu_device(layout_info: LayoutInfo) -> bool:
    device = layout_info["device"]
    if device is None:
        raise ValueError("Can't determine if device is CPU because layout_info['device'] is None.")
    return device == core_defs.DeviceType.CPU


def is_gpu_device(layout_info: LayoutInfo) -> bool:
    device = layout_info["device"]
    if device is None:
        raise ValueError("Can't determine if device is GPU because layout_info['device'] is None.")
    return device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM]


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
    "device": core_defs.DeviceType.CPU,
    "layout_map": lambda axes: tuple(i for i in range(len(axes))),
    "is_optimal_layout": lambda *_: True,
}
register("naive_cpu", NaiveCPULayout)

CPUIFirstLayout: Final[LayoutInfo] = {
    "alignment": 8,
    "device": core_defs.DeviceType.CPU,
    "layout_map": make_gtcpu_ifirst_layout_map,
    "is_optimal_layout": layout_checker_factory(make_gtcpu_ifirst_layout_map),
}
register("cpu_ifirst", CPUIFirstLayout)


CPUKFirstLayout: Final[LayoutInfo] = {
    "alignment": 1,
    "device": core_defs.DeviceType.CPU,
    "layout_map": make_gtcpu_kfirst_layout_map,
    "is_optimal_layout": layout_checker_factory(make_gtcpu_kfirst_layout_map),
}
register("cpu_kfirst", CPUKFirstLayout)


CUDALayout: Final[LayoutInfo] = {
    "alignment": 32,
    "device": CUPY_DEVICE,
    "layout_map": make_cuda_layout_map,
    "is_optimal_layout": layout_checker_factory(make_cuda_layout_map),
}
register("cuda", CUDALayout)

GPULayout: Final[LayoutInfo] = CUDALayout
register("gpu", GPULayout)

DaceCPULayout: Final[LayoutInfo] = {
    "alignment": 1,
    "device": core_defs.DeviceType.CPU,
    "layout_map": layout_maker_factory((0, 1, 2)),
    "is_optimal_layout": layout_checker_factory(layout_maker_factory((0, 1, 2))),
}
register("dace:cpu", DaceCPULayout)

DaceGPULayout: Final[LayoutInfo] = {
    "alignment": 32,
    "device": CUPY_DEVICE,
    "layout_map": layout_maker_factory((2, 1, 0)),
    "is_optimal_layout": layout_checker_factory(layout_maker_factory((2, 1, 0))),
}
register("dace:gpu", DaceGPULayout)
