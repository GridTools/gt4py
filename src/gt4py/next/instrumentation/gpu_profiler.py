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
    """Context manager for marking a time range in GPU profiling sessions, with optional message and color."""

    profile: Final = cupyx_profiler.profile
    """Context manager for signaling the GPU profiler tool the beginning and end of a profiling session."""

except ImportError:
    cupyx_profiler = None

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
            **kwargs: Any,
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
_profile_ctx_manager_count: int = 0
_profile_ctx_manager_lock: threading.Lock = threading.Lock()


@contextlib.contextmanager
def profile_calls() -> Generator[None, None, None]:
    """Context manager that enables GPU profiling of GT4Py program calls within its scope."""
    if not start_profiling_calls():
        warnings.warn(
            "GPU profiling of GT4Py program calls is already active."
            "Nested 'profile_calls' should not be used.",
            UserWarning,
            stacklevel=2,
        )
    try:
        yield
    finally:
        stop_profiling_calls()


def start_profiling_calls() -> bool:
    """
    Start a GPU profiling session and register hooks that annotate each program call.

    Repeated calls will not start a second session. Pairs with :func:`stop_profiling_calls`.

    Returns:
        True if a new profiling session was started, False if a session was already active.
    """
    global _profile_ctx_manager, _profile_ctx_manager_count
    with _profile_ctx_manager_lock:
        _profile_ctx_manager_count += 1
        if _profile_ctx_manager is None:
            assert _profile_ctx_manager_count == 1
            _profile_ctx_manager = profile()
            hooks.program_call_context.register(ProgramCallProfiler, index=0)
            hooks.compiled_program_call_context.register(CompiledProgramCallProfiler, index=0)
            try:
                _profile_ctx_manager.__enter__()
            except Exception:
                _profile_ctx_manager_count = 0
                hooks.compiled_program_call_context.remove(CompiledProgramCallProfiler)
                hooks.program_call_context.remove(ProgramCallProfiler)
                _profile_ctx_manager = None
                raise

            return True

    return False


def stop_profiling_calls() -> bool:
    """
    Stop the active GPU profiling session and unregister the program call hooks.

    Returns:
        True if a profiling session was stopped, False if no session was active.
    """
    global _profile_ctx_manager, _profile_ctx_manager_count
    with _profile_ctx_manager_lock:
        # Only stop the profiling session when the outermost context manager is exited,
        # so nesting start/stop calls still works (although is heavily discouraged).
        if _profile_ctx_manager_count > 0:
            _profile_ctx_manager_count -= 1
            if _profile_ctx_manager_count == 0:
                assert _profile_ctx_manager is not None
                try:
                    _profile_ctx_manager.__exit__(None, None, None)
                finally:
                    hooks.compiled_program_call_context.remove(CompiledProgramCallProfiler)
                    hooks.program_call_context.remove(ProgramCallProfiler)
                    _profile_ctx_manager = None

                return True

    return False


class ProgramProfiler(contextlib.AbstractContextManager):
    """Base context manager that wraps a program execution in a time range."""

    time_range_ctx: ContextManager[None]

    __slots__ = ("time_range_ctx",)

    def __enter__(self) -> None:
        self.time_range_ctx.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.time_range_ctx.__exit__(exc_type, exc_value, traceback)


class ProgramCallProfiler(ProgramProfiler):
    """
    Hook-compatible profiler that emits a time range around each program call.

    By default it uses a hardcoded color ID for the time ranges. GT4Py
    programs can override this by explicitly setting a `program_color_id`
    attribute on their python definition function.
    """

    color_id: int = 1  # default for program calls, program definitions can override it

    def __init__(
        self,
        program: gtx_typing.Program,
        args: tuple[Any, ...],
        offset_provider: common.OffsetProvider,
        enable_jit: bool,
        kwargs: dict[str, Any],
    ) -> None:
        self.time_range_ctx = time_range(
            program.__name__,
            color_id=getattr(program.definition, "program_color_id", self.color_id),
        )


class CompiledProgramCallProfiler(ProgramProfiler):
    """
    Hook-compatible profiler that emits a time range around each compiled program dispatch.

    By default it uses a hardcoded color ID for the time ranges. GT4Py
    programs can override this by explicitly setting a `compiled_program_color_id`
    attribute on their python definition function.
    """

    color_id: int = 2  # default for compiled program calls, program definitions can override it

    def __init__(
        self,
        program_pool: compiled_program.CompiledProgramsPool,
        key: gtx_typing.CompiledProgramsKey,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        offset_provider: common.OffsetProvider,
    ) -> None:
        self.time_range_ctx = time_range(
            compiled_program.metrics_source_key(program_pool, key),
            color_id=getattr(program_pool.definition, "compiled_program_color_id", self.color_id),
        )
