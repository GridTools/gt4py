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

#: The eager embedded backends, and the array namespace each allocates with
_EAGER_BACKENDS: Final = {
    "embedded_numpy": "numpy",
    "embedded_cupy": "cupy",
    "embedded_jax": "jax.numpy",
    "embedded_torch": "torch",
}
#: The embedded jit backends, and the optional package each needs
_JIT_BACKENDS: Final = {"jax_jit": "jax", "torch_compile": "torch"}


# the backends are created on first access, since most of them need an optional package
def __getattr__(name: str) -> Any:
    if name in _EAGER_BACKENDS:
        from gt4py.next import backend as next_backend

        namespace = _EAGER_BACKENDS[name]
        allocator = _import(name, namespace, namespace.partition(".")[0])
        backend = next_backend.EmbeddedBackend(name=name, allocator=allocator)
    elif name in _JIT_BACKENDS:
        module = _import(
            name, f"gt4py.next.program_processors.runners.experimental.{name}", _JIT_BACKENDS[name]
        )
        backend = getattr(module, name)
    else:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")
    globals()[name] = backend
    return backend


def __dir__() -> list[str]:
    return [*globals(), *_EAGER_BACKENDS, *_JIT_BACKENDS]


def _import(backend: str, module: str, package: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] != package:
            raise
        raise AttributeError(f"'{backend}' requires the optional '{package}' package.") from error
