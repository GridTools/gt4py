# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` of the array libraries whose arrays need more than the Array API standard."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import array_api_compat
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops, nd_array_field


def _np_gamma(a: core_defs.NDArrayObject) -> core_defs.NDArrayObject:
    try:
        from scipy.special import gamma
    except ImportError:
        return np.vectorize(math.gamma, otypes=[a.dtype])(a)
    return gamma(a)


class NumPyOps(array_ops.ArrayOps):
    @property
    def array_ns(self) -> ModuleType:
        return np

    def asarray(
        self,
        data: Any,
        *,
        dtype: Any = None,
        like: core_defs.NDArrayObject | None = None,
    ) -> core_defs.NDArrayObject:
        return np.asarray(data, dtype=dtype)  # type: ignore[return-value] # `NDArrayObject` typing is not complete

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        return value

    def _elementwise(self, name: str) -> Callable[..., Any]:
        if name == "gamma":
            return _np_gamma
        return super()._elementwise(name)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        array[key] = value  # type: ignore[index] # `NDArrayObject` typing is not complete
        return array

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return np.asarray(array)


class CuPyOps(array_ops.ArrayOps):
    @property
    def array_ns(self) -> ModuleType:
        import cupy as cp

        return cp

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        return value

    def _elementwise(self, name: str) -> Callable[..., Any]:
        if name == "gamma":
            import cupyx.scipy.special

            return cupyx.scipy.special.gamma
        return super()._elementwise(name)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        array[key] = value  # type: ignore[index] # `NDArrayObject` typing is not complete
        return array

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        import cupy as cp

        return cp.asnumpy(array)


class JaxOps(array_ops.ArrayOps):
    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        return value

    def _elementwise(self, name: str) -> Callable[..., Any]:
        if name == "gamma":
            import jax.scipy.special

            return jax.scipy.special.gamma
        return super()._elementwise(name)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        return array.at[key].set(value)  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return np.asarray(array)

    def buffer_info(self, array: core_defs.NDArrayObject) -> common.BufferInfo:
        raise NotImplementedError("'__gt_buffer_info__' for JAX arrays not yet implemented.")


class TorchOps(array_ops.ArrayOps):
    def __init__(self, xp: ModuleType) -> None:
        super().__init__(xp)
        import torch

        self._torch = torch

    @property
    def array_ns(self) -> ModuleType:
        return self._torch

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        # torch functions reject scalar operands; 0-d tensors keep NumPy's promotion
        # because a Python float is made float64 here, not torch's default float32
        if isinstance(value, self._torch.Tensor):
            return value
        scalar_type = (
            type(value)
            if isinstance(value, np.generic)
            else core_defs.dtype(type(value)).scalar_type
        )
        return self.asarray(value, dtype=scalar_type, like=like)

    def _elementwise(self, name: str) -> Callable[..., Any]:
        torch = self._torch
        if name == "cbrt":
            return lambda x: torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)
        if name == "gamma":

            def gamma(x: Any) -> Any:
                sign = torch.where((x < 0) & (torch.floor(x) % 2 == 1), -1.0, 1.0).to(x.dtype)
                return sign * torch.exp(torch.lgamma(x))

            return gamma
        return super()._elementwise(name)

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return array.detach().cpu().numpy()  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete


array_ops.register(array_api_compat.is_numpy_namespace, NumPyOps)
array_ops.register(array_api_compat.is_cupy_namespace, CuPyOps)
array_ops.register(array_api_compat.is_jax_namespace, JaxOps)
array_ops.register(array_api_compat.is_torch_namespace, TorchOps)


if importlib.util.find_spec("jax") is not None:
    import jax

    # TODO(havogt): we currently enable 64-bit support by default, but we might want to make this configurable via the GT4Py config
    jax.config.update("jax_enable_x64", True)

    def _flatten_field(
        field: nd_array_field.NdArrayField,
    ) -> tuple[tuple[core_defs.NDArrayObject], common.Domain]:
        return (field.ndarray,), field.domain

    def _unflatten_field(
        domain: common.Domain, children: tuple[core_defs.NDArrayObject]
    ) -> nd_array_field.NdArrayField:
        return nd_array_field.NdArrayField(domain, children[0])  # type: ignore[abstract] # mypy does not see '__gt_builtin_func__' as implemented by 'FieldBuiltinFuncRegistry'

    jax.tree_util.register_pytree_node(
        nd_array_field.NdArrayField,  # type: ignore[type-abstract, unused-ignore] # only reported when 'jax' is installed, see '_unflatten_field'
        _flatten_field,
        _unflatten_field,
    )
