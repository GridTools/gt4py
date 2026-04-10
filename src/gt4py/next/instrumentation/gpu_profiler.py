# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
GPU profiling utilities for GT4Py programs.

Provides context managers and hooks to emit NVTX/ROCTX time ranges around program
calls and to start/stop CUDA/ROCm profiling sessions, so that GT4Py program
executions show up as annotated regions in GPU profilers (e.g. Nsight Systems).

Requires a CuPy-compatible GPU; otherwise the public API just emits warnings.
"""

from __future__ import annotations

import contextlib
import threading
import types
import warnings
from collections.abc import Generator
from typing import Any, ContextManager, Final

from gt4py._core import definitions as core_definitions, types as core_types
from gt4py.next import common, typing as gtx_typing
from gt4py.next.instrumentation import hooks
from gt4py.next.otf import compiled_program


try:
    import cupyx.profiler as cupyx_profiler

    assert core_definitions.CUPY_DEVICE_TYPE is not None

    time_range: Final = cupyx_profiler.time_range
    profile: Final = cupyx_profiler.profile

except ImportError:
    cupy = None

    class ProfilingContextManager(contextlib.AbstractContextManager):
        """Fallback profiling context manager that does nothing but emit a warning."""

        def __init__(self) -> None:
            warnings.warn(
                "GT4Py profiling is only supported when using a GPU and CuPy is installed.",
                UserWarning,
                stacklevel=2,
            )

        def __enter__(self) -> None:
            pass

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: types.TracebackType | None,
        ) -> None:
            pass

    class TimeRangeContextManager(contextlib.AbstractContextManager):
        """Fallback time range context manager that does nothing but emit a warning."""

        def __init__(
            self,
            message: str | None = None,
            color_id: int | None = None,
            argb_color: core_types.int32 | None = None,
            sync: bool = False,
        ) -> None:
            warnings.warn(
                "GT4Py profiling is only supported when using a GPU and CuPy is installed.",
                UserWarning,
                stacklevel=2,
            )

        def __enter__(self) -> None:
            pass

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: types.TracebackType | None,
        ) -> None:
            pass

    time_range: Final = TimeRangeContextManager  # type: ignore[misc]
    profile: Final = ProfilingContextManager  # type: ignore[misc]


_profile_ctx_manager: ContextManager | None = None
_profile_ctx_manager_lock: threading.Lock = threading.Lock()


@contextlib.contextmanager
def profile_calls() -> Generator[None, None, None]:
    """Context manager that enables GPU profiling of GT4Py program calls within its scope."""
    start_profiling_calls()
    yield
    stop_profiling_calls()


def start_profiling_calls() -> None:
    """
    Start a GPU profiling session and register hooks that annotate each program call.

    Repeated calls will not start a second session. Pairs with :func:`stop_profiling_calls`.
    """
    global _profile_ctx_manager
    hooks.program_call_context.register(ProgramCallProfiler, index=0)
    hooks.compiled_program_call_context.register(CompiledProgramCallProfiler, index=0)
    with _profile_ctx_manager_lock:
        if _profile_ctx_manager is None:
            _profile_ctx_manager = profile()
            _profile_ctx_manager.__enter__()


def stop_profiling_calls() -> None:
    """Stop the active GPU profiling session and unregister the program call hooks."""
    global _profile_ctx_manager
    with _profile_ctx_manager_lock:
        if _profile_ctx_manager is not None:
            _profile_ctx_manager.__exit__(None, None, None)
            _profile_ctx_manager = None
    hooks.program_call_context.remove(ProgramCallProfiler)
    hooks.compiled_program_call_context.remove(CompiledProgramCallProfiler)


class ProgramProfiler(contextlib.AbstractContextManager):
    """Base context manager that wraps a program execution in a time range."""

    name: str
    time_range: cupyx_profiler.time_range
    color_id: int

    __slots__ = ("color_id", "name", "time_range")

    def __enter__(self) -> None:
        self.time_range = time_range(self.name, color_id=self.color_id).__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.time_range.__exit__(exc_type, exc_value, traceback)


class ProgramCallProfiler(ProgramProfiler):
    """Hook-compatible profiler that emits a time range around each program call."""

    color_id: int = (
        1  # default color ID for program calls, can be overridden by program definitions
    )

    def __init__(
        self,
        program: gtx_typing.Program,
        args: tuple[Any, ...],
        offset_provider: common.OffsetProvider,
        enable_jit: bool,
        kwargs: dict[str, Any],
    ) -> None:
        self.name = program.__name__
        if (color_id := getattr(program.definition, "color_id", None)) is not None:
            self.color_id = color_id


class CompiledProgramCallProfiler(ProgramProfiler):
    """Hook-compatible profiler that emits a time range around each compiled program dispatch."""

    color_id: int = (
        2  # default color ID for program calls, can be overridden by program definitions
    )

    def __init__(
        self,
        program_pool: compiled_program.CompiledProgramsPool,
        key: gtx_typing.CompiledProgramsKey,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        offset_provider: common.OffsetProvider,
    ) -> None:
        self.name = compiled_program.metrics_source_key(program_pool, key)
        if (color_id := getattr(program_pool.definition, "color_id", None)) is not None:
            self.color_id = color_id
