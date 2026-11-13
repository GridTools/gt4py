# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` for torch tensors; importing it does not import torch."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops


if TYPE_CHECKING:
    import torch


def _cbrt(x: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


def _gamma(x: torch.Tensor) -> torch.Tensor:
    import torch

    sign = torch.where((x < 0) & (torch.floor(x) % 2 == 1), -1.0, 1.0).to(x.dtype)
    result = sign * torch.exp(torch.lgamma(x))
    # the poles as in SciPy: +-inf at +-0, NaN at the negative integers
    result = torch.where(x == 0, 1 / x, result)
    return torch.where((x < 0) & (x == torch.floor(x)), torch.nan, result)


class TorchOps(array_ops.ArrayOps):
    @property
    def array_ns(self) -> ModuleType:
        import torch

        return torch

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        """
        `value` as an operand of an elementwise function next to the tensor `like`.

        Scalars become 0-d tensors, as several torch functions reject Python and NumPy
        scalars. They are created on the CPU: torch takes 0-d CPU tensors as operands of
        CUDA tensors, without a copy to the device. A Python scalar of the kind of `like`
        takes its dtype, to keep the weak typing of the standard.
        """
        if isinstance(value, bool | int | float):
            like_dtype = like.dtype
            like_kind = (
                bool
                if like_dtype == self.xp.bool
                else float
                if like_dtype.is_floating_point
                else int
            )
            dtype = like_dtype if type(value) is like_kind else self.dtype(type(value))
            return self.xp.asarray(value, dtype=dtype, device="cpu")
        if isinstance(value, np.generic):
            return self.xp.asarray(value, device="cpu")
        return value

    def _elementwise_function(self, name: str) -> Callable[..., Any]:
        if name == "cbrt":
            return _cbrt
        if name == "gamma":
            return _gamma
        return super()._elementwise_function(name)

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return array.detach().cpu().numpy()  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

    def buffer_info(self, array: core_defs.NDArrayObject) -> common.BufferInfo:
        raise NotImplementedError("'__gt_buffer_info__' for torch tensors not yet implemented.")


array_ops.register("array_api_compat.torch", TorchOps)
