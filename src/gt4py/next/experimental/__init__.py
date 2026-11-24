# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
from typing import Any, Final

from gt4py.next.ffront.experimental import as_offset, concat_where


__all__ = [
    "as_offset",
    "concat_where",
]

# the embedded backends, created on first access: they need an optional package
_EMBEDDED_JIT_BACKENDS: Final = {"jax_jit": "jax", "torch_compile": "torch"}
#: The array namespace each eager embedded backend allocates with
_EMBEDDED_BACKENDS: Final = {
    "embedded_numpy": "numpy",
    "embedded_cupy": "cupy",
    "embedded_jax": "jax.numpy",
    "embedded_torch": "torch",
}


def __getattr__(name: str) -> Any:
    if name in _EMBEDDED_BACKENDS:
        from gt4py.next import backend as next_backend

        namespace = _EMBEDDED_BACKENDS[name]
        try:
            allocator = importlib.import_module(namespace)
        except ImportError as error:
            raise AttributeError(
                f"'{name}' requires the optional '{namespace}' package."
            ) from error
        backend = next_backend.EmbeddedBackend(name=name, allocator=allocator)
    elif name in _EMBEDDED_JIT_BACKENDS:
        try:
            module = importlib.import_module(
                f"gt4py.next.program_processors.runners.experimental.{name}"
            )
        except ImportError as error:
            raise AttributeError(
                f"'{name}' requires the optional '{_EMBEDDED_JIT_BACKENDS[name]}' package."
            ) from error
        backend = getattr(module, name)
    else:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")
    globals()[name] = backend
    return backend
