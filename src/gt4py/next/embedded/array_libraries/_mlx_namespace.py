# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The Array API standard namespace of MLX arrays, around `mlx.core`.

`mlx.core` is close to the standard, but it is no conforming namespace: it has no `device`
arguments, since MLX runs every operation on its default device, or on a `stream` given to
the operation; it has `bool_` instead of `bool`, no `nonzero`, `cbrt` or `fmod`, and its
`sum` has no `dtype`. This module adds what `gt4py.next` uses, as `ArrayOps.xp` of MLX
arrays; any other name is the one of `mlx.core`. It is meant to be deleted when
`array-api-compat` supports MLX.

`mlx.core` converts float64 NumPy data and Python floats to float32. The functions here
never downcast silently: data without a `dtype` keeps the dtype NumPy gives it, creation
functions without `dtype` use NumPy's default dtypes, integers are divided in float64 as in
NumPy, and the float64 functions MLX computes with float32 accuracy are computed by NumPy.
MLX does not support float64 on the GPU, so requesting a float64 array there raises a
`TypeError`. Arrays of different dtypes are otherwise promoted as MLX does, e.g. int64 and
float32 to float32, unlike NumPy.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Any, Final

import mlx.core as mx
import numpy as np


FLOAT64_ON_GPU_MESSAGE: Final = (
    "MLX does not support 'float64' on the GPU: use 'dtype=float32', or compute on the CPU"
    " with 'mlx.core.set_default_device(mlx.core.cpu)'."
)

bool = mx.bool_  # noqa: A001 # the standard's name of the boolean dtype


def __getattr__(name: str) -> Any:
    return getattr(mx, name)


def _from_numpy_dtype(dtype: np.dtype) -> Any:
    return mx.bool_ if dtype == np.bool_ else getattr(mx, dtype.name)


def _check_dtype(dtype: Any, device: Any) -> None:
    """Raise for float64 on the GPU, which MLX does not support."""
    if dtype == mx.float64 and (mx.default_device() if device is None else device) == mx.gpu:
        raise TypeError(FLOAT64_ON_GPU_MESSAGE)


def _stream(device: Any) -> Any:
    """The `stream` argument of MLX for `device`: none for the default device."""
    return None if device is None or device == mx.default_device() else device


def asarray(
    obj: Any, /, *, dtype: Any = None, device: Any = None, copy: builtins.bool | None = None
) -> mx.array:
    if not isinstance(obj, mx.array):
        # through NumPy, so that the dtype of NumPy data, and NumPy's dtypes of Python scalars, are kept
        obj = np.asarray(obj)
        if dtype is None:
            dtype = _from_numpy_dtype(obj.dtype)
    elif (dtype is None or dtype == obj.dtype) and not copy:
        _check_dtype(obj.dtype, device)
        return obj
    _check_dtype(obj.dtype if dtype is None else dtype, device)
    return mx.asarray(obj, dtype=dtype, copy=copy)


def _creation_dtype(dtype: Any, device: Any) -> Any:
    dtype = mx.float64 if dtype is None else dtype
    _check_dtype(dtype, device)
    return dtype


def empty(shape: int | Sequence[int], *, dtype: Any = None, device: Any = None) -> mx.array:
    return mx.zeros(shape, _creation_dtype(dtype, device), stream=_stream(device))


def zeros(shape: int | Sequence[int], *, dtype: Any = None, device: Any = None) -> mx.array:
    return mx.zeros(shape, _creation_dtype(dtype, device), stream=_stream(device))


def ones(shape: int | Sequence[int], *, dtype: Any = None, device: Any = None) -> mx.array:
    return mx.ones(shape, _creation_dtype(dtype, device), stream=_stream(device))


def full(
    shape: int | Sequence[int], fill_value: Any, *, dtype: Any = None, device: Any = None
) -> mx.array:
    if dtype is None:
        dtype = (
            fill_value.dtype
            if isinstance(fill_value, mx.array)
            else _from_numpy_dtype(np.asarray(fill_value).dtype)
        )
    _check_dtype(dtype, device)
    return mx.full(shape, fill_value, dtype, stream=_stream(device))


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: Any = None,
    device: Any = None,
) -> mx.array:
    if stop is None:
        start, stop = 0, start
    if dtype is None:
        dtype = _from_numpy_dtype(np.result_type(start, stop, step))
    _check_dtype(dtype, device)
    return mx.arange(start, stop, step, dtype, stream=_stream(device))


def astype(
    x: mx.array, dtype: Any, /, *, copy: builtins.bool = True, device: Any = None
) -> mx.array:
    _check_dtype(dtype, device)
    if not copy and x.dtype == dtype:
        return x
    return mx.astype(x, dtype, stream=_stream(device))


def sum(  # noqa: A001 # the standard's name
    x: mx.array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: Any = None,
    keepdims: builtins.bool = False,
) -> mx.array:
    result = mx.sum(x, axis=axis, keepdims=keepdims)
    return result if dtype is None or result.dtype == dtype else mx.astype(result, dtype)


def nonzero(x: mx.array, /) -> tuple[mx.array, ...]:
    # MLX has no `nonzero`, as its output shape depends on the values
    return tuple(mx.array(indices) for indices in np.nonzero(np.array(x)))


def cbrt(x: mx.array, /) -> mx.array:
    return mx.sign(x) * mx.abs(x) ** (1.0 / 3.0)


def fmod(x1: mx.array, x2: mx.array, /) -> mx.array:
    # `remainder` has the sign of `x2`, `fmod` the sign of `x1`
    remainder = mx.remainder(x1, x2)
    return mx.where((remainder != 0) & ((remainder < 0) != (x1 < 0)), remainder - x2, remainder)


def divide(x1: Any, x2: Any, /) -> mx.array:
    # MLX divides integers in float32
    if all(
        (x.dtype == mx.bool_ or mx.issubdtype(x.dtype, mx.integer))
        if isinstance(x, mx.array)
        else isinstance(x, builtins.bool | int | np.integer | np.bool_)
        for x in (x1, x2)
    ):
        x1, x2 = (astype(x, mx.float64) if isinstance(x, mx.array) else float(x) for x in (x1, x2))
    return mx.divide(x1, x2)


def _float64_by_numpy(name: str) -> Any:
    mlx_function, numpy_function = getattr(mx, name), getattr(np, name)

    def function(x: mx.array, /) -> mx.array:
        if x.dtype == mx.float64:  # only on the CPU, see `_check_dtype`
            return mx.array(numpy_function(np.array(x)), dtype=mx.float64)
        return mlx_function(x)

    function.__name__ = name
    return function


# MLX 0.32.4 computes them in float64 with float32 accuracy, all others with float64 accuracy
sin = _float64_by_numpy("sin")
cos = _float64_by_numpy("cos")
exp = _float64_by_numpy("exp")
