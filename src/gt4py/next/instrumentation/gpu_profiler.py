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
from typing import TYPE_CHECKING, Any, ClassVar

from gt4py._core import definitions as core_definitions, types as core_types
from gt4py.next import common, typing as gtx_typing


_current_profiler = None


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
def profile():
    warnings.warn(
        "GT4Py profiling is only supported when using a GPU.",
        UserWarning,
        stacklevel=2,
    )
    yield


def start_profiling() -> None:
    global _current_profiler
    _current_profiler = profile()
    _current_profiler.__enter__()


def stop_profiling() -> None:
    if _current_profiler is not None:
        _current_profiler.__exit__(None, None, None)


class ProgramProfiler(contextlib.AbstractContextManager):
    __slots__ = ("name", "time_range")
    name: str
    time_range: cupy_profiler.time_range
    color_id: ClassVar[int]

    def __enter__(self) -> None:
        self.time_range = time_range(self.name, color_id=self.color_id).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.time_range.__exit__(exc_type, exc_val, exc_tb)


class ProgramCallProfiler(ProgramProfiler):
    color_id: ClassVar[int] = 1

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
    color_id: ClassVar[int] = 2

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


if not TYPE_CHECKING and core_definitions.CUPY_DEVICE_TYPE is not None:
    import cupyx.profiler as cupy_profiler

    time_range = cupy_profiler.time_range
    profile = cupy_profiler.profile
