# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Profiler time ranges for the compilation machinery.

Enabled by `config.ADD_COMPILATION_TRACE_MARKERS`; the ranges are emitted through
CuPy (NVTX or ROC-TX) and are no-ops when disabled or CuPy is not available. The
flag is read at each call, so compilation worker processes follow the config
shipped to them by the runner.
"""

from __future__ import annotations

import contextlib
from typing import ContextManager, Final

from gt4py.next import config


#: Colors of ranges of work skipped thanks to a cache, and of work actually done.
CACHE_HIT_COLOR: Final = 0xFF2CA02C
BUILD_COLOR: Final = 0xFFD62728


def time_range(
    message: str, argb_color: int | None = None, enabled: bool = True
) -> ContextManager[None]:
    """Return a time range named `message`, or a no-op context manager if disabled."""
    if not (enabled and config.ADD_COMPILATION_TRACE_MARKERS):
        return contextlib.nullcontext()
    try:
        import cupyx.profiler as cupyx_profiler  # noqa: PLC0415 [import-outside-top-level]
    except ImportError:
        return contextlib.nullcontext()
    return cupyx_profiler.time_range(message, argb_color=argb_color)
