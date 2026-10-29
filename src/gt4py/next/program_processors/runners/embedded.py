# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Embedded backends: run programs and operators directly in Python on fields.

- `embedded_numpy` allocates with NumPy. It is the default backend and is also
  exported as `gt4py.next.embedded_numpy`.
- `embedded_cupy` allocates with CuPy and `embedded_jax` with `jax.numpy`. They
  are created on first access and require the optional package; accessing them
  without it raises `AttributeError`.
"""

from __future__ import annotations

import typing
from typing import Final

import numpy as np

from gt4py._core import ndarray_utils as core_ndarray_utils
from gt4py.next import backend as next_backend


embedded_numpy: Final = next_backend.EmbeddedBackend(
    name="embedded_numpy", allocator=typing.cast(core_ndarray_utils.ArrayNamespace, np)
)


def _import_cupy() -> core_ndarray_utils.ArrayNamespace:
    import cupy

    return typing.cast(core_ndarray_utils.ArrayNamespace, cupy)


def _import_jax_numpy() -> core_ndarray_utils.ArrayNamespace:
    import jax.numpy

    return typing.cast(core_ndarray_utils.ArrayNamespace, jax.numpy)


# Backends needing an optional package: name -> (package, namespace importer)
_OPTIONAL_BACKENDS: Final[
    dict[str, tuple[str, typing.Callable[[], core_ndarray_utils.ArrayNamespace]]]
] = {
    "embedded_cupy": ("cupy", _import_cupy),
    "embedded_jax": ("jax", _import_jax_numpy),
}


def __getattr__(name: str) -> next_backend.EmbeddedBackend:
    if name not in _OPTIONAL_BACKENDS:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")
    package, import_namespace = _OPTIONAL_BACKENDS[name]
    try:
        array_ns = import_namespace()
    except ImportError as error:
        raise AttributeError(f"'{name}' requires the optional '{package}' package.") from error
    backend = next_backend.EmbeddedBackend(name=name, allocator=array_ns)
    # Cache it, so that every access returns the same instance.
    globals()[name] = backend
    return backend
