# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from typing import Any, ClassVar

from gt4py._core import definitions as core_definitions, types as core_types
from gt4py.next import common, typing as gtx_typing
from gt4py.next.instrumentation import hooks


if core_definitions.CUPY_DEVICE_TYPE is not None:
    import cupyx.profiler as cupy_profiler

    time_range = cupy_profiler.time_range

else:

    class time_range(contextlib.AbstractContextManager):
        def __init__(
            self,
            message: str | None = None,
            color_id: int | None = None,
            argb_color: core_types.int32 | None = None,
            sync=False,
        ) -> None:
            warnings.warn(
                "GT4Py profiling is only supported when using a GPU.",
                UserWarning,
                stacklevel=2,
            )


@contextlib.contextmanager
def profile_calls():
    start_profiling_calls()
    yield
    stop_profiling_calls()


def start_profiling_calls() -> None:
    hooks.program_call_context.register(ProgramCallProfiler, index=0)
    hooks.compiled_program_call_context.register(CompiledProgramCallProfiler, index=0)


def stop_profiling_calls() -> None:
    hooks.program_call_context.remove(ProgramCallProfiler)
    hooks.compiled_program_call_context.remove(CompiledProgramCallProfiler)


class ProgramProfiler(contextlib.AbstractContextManager):
    name: str
    time_range: cupy_profiler.time_range

    COLOR_ID: ClassVar[int]

    __slots__ = ("name", "time_range")

    def __enter__(self) -> None:
        print(f"\n\n\n\nProfiling {self.name}...")
        self.time_range = time_range(self.name, color_id=self.COLOR_ID).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.time_range.__exit__(exc_type, exc_val, exc_tb)
        print(f"Finished profiling {self.name}.\n\n\n\n")


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
        compiled_program: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        offset_provider: common.OffsetProvider,
        root: tuple[str, str],
        key: gtx_typing.CompiledProgramsKey,
    ) -> None:
        self.name = f"{root[0]}<{root[1]}>"
