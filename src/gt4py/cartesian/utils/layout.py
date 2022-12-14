from typing import TYPE_CHECKING, Tuple, Union

import numpy as np


if TYPE_CHECKING:
    try:
        import cupy as cp
    except ImportError:
        cp = None


def dimensions_to_mask(dimensions: Tuple[str, ...]) -> Tuple[bool, ...]:
    ndata_dims = sum(d.isdigit() for d in dimensions)
    mask = [(d in dimensions) for d in "IJK"] + [True for _ in range(ndata_dims)]
    return tuple(mask)


def check_layout(layout_map, strides):
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
        return check_layout(layout_map, field.strides)

    return layout_checker
