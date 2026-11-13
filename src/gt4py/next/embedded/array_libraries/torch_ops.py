# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` for torch tensors."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import array_api_compat.torch as compat_torch
import numpy as np
import torch

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops


def _cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


def _gamma(x: torch.Tensor) -> torch.Tensor:
    sign = torch.where((x < 0) & (torch.floor(x) % 2 == 1), -1.0, 1.0).to(x.dtype)
    return sign * torch.exp(torch.lgamma(x))


class TorchOps(array_ops.ArrayOps):
    @property
    def array_ns(self) -> ModuleType:
        return torch

    def asarray(
        self,
        data: Any,
        *,
        dtype: Any = None,
        like: core_defs.NDArrayObject | None = None,
    ) -> core_defs.NDArrayObject:
        # before torch 2.15, `torch.asarray` reads an object with the buffer protocol but
        # without `__array__` (e.g. an array-api-strict array) as raw bytes of the default dtype
        if not isinstance(data, torch.Tensor | np.ndarray) and hasattr(data, "__dlpack__"):
            data = torch.from_dlpack(data)
        return super().asarray(data, dtype=dtype, like=like)

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
                bool if like_dtype == torch.bool else float if like_dtype.is_floating_point else int
            )
            dtype = like_dtype if type(value) is like_kind else self.dtype(type(value))
            return torch.asarray(value, dtype=dtype, device="cpu")
        if isinstance(value, np.generic):
            return torch.asarray(value, device="cpu")
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


array_ops.register(TorchOps(compat_torch))
