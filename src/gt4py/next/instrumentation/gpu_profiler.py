# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import contextlib
import threading
import types
import warnings
from collections.abc import Generator
from typing import Any, ClassVar, ContextManager

from gt4py._core import definitions as core_definitions, types as core_types
from gt4py.next import common, typing as gtx_typing
from gt4py.next.instrumentation import hooks
from gt4py.next.otf import compiled_program


if core_definitions.CUPY_DEVICE_TYPE is not None:
    import cupyx.profiler as cupyx_profiler

    time_range = cupyx_profiler.time_range

else:

    class time_range(contextlib.AbstractContextManager):  # type: ignore[no-redef]
        def __init__(
            self,
            message: str | None = None,
            color_id: int | None = None,
            argb_color: core_types.int32 | None = None,
            sync: bool = False,
        ) -> None:
            warnings.warn(
                "GT4Py profiling is only supported when using a GPU.",
                UserWarning,
                stacklevel=2,
            )


_profile_ctx_manager: ContextManager | None = None
_profile_ctx_manager_lock: threading.Lock = threading.Lock()


@contextlib.contextmanager
def profile_calls() -> Generator[None, None, None]:
    start_profiling_calls()
    yield
    stop_profiling_calls()


def start_profiling_calls() -> None:
    global _profile_ctx_manager
    hooks.program_call_context.register(ProgramCallProfiler, index=0)
    hooks.compiled_program_call_context.register(CompiledProgramCallProfiler, index=0)
    with _profile_ctx_manager_lock:
        if _profile_ctx_manager is None:
            _profile_ctx_manager = cupyx_profiler.profile()
            _profile_ctx_manager.__enter__()


def stop_profiling_calls() -> None:
    global _profile_ctx_manager
    with _profile_ctx_manager_lock:
        if _profile_ctx_manager is not None:
            _profile_ctx_manager.__exit__(None, None, None)
            _profile_ctx_manager = None
    hooks.program_call_context.remove(ProgramCallProfiler)
    hooks.compiled_program_call_context.remove(CompiledProgramCallProfiler)


class ProgramProfiler(contextlib.AbstractContextManager):
    name: str
    time_range: cupyx_profiler.time_range

    COLOR_ID: ClassVar[int]

    __slots__ = ("name", "time_range")

    def __enter__(self) -> None:
        self.time_range = time_range(self.name, color_id=self.COLOR_ID).__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.time_range.__exit__(exc_type, exc_value, traceback)


class ProgramCallProfiler(ProgramProfiler):
    COLOR_ID: ClassVar[int] = 1

    def __init__(
        self,
        program: gtx_typing.Program,
        args: tuple[Any, ...],
        offset_provider: common.OffsetProvider,
        enable_jit: bool,
        kwargs: dict[str, Any],
    ) -> None:
        self.name = program.__name__


class CompiledProgramCallProfiler(ProgramProfiler):
    COLOR_ID: ClassVar[int] = 2

    def __init__(
        self,
        program_pool: compiled_program.CompiledProgramsPool,
        key: gtx_typing.CompiledProgramsKey,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        offset_provider: common.OffsetProvider,
    ) -> None:
        self.name = compiled_program.metrics_source_key(program_pool, key)
