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

# the embedded jit backends, created on first access: they need an optional package
_EMBEDDED_JIT_BACKENDS: Final = {"jax_jit": "jax", "torch_compile": "torch"}


def __getattr__(name: str) -> Any:
    if name not in _EMBEDDED_JIT_BACKENDS:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")
    try:
        module = importlib.import_module(
            f"gt4py.next.program_processors.runners.experimental.{name}"
        )
    except ImportError as error:
        raise AttributeError(
            f"'{name}' requires the optional '{_EMBEDDED_JIT_BACKENDS[name]}' package."
        ) from error
    backend = globals()[name] = getattr(module, name)
    return backend
