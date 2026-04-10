# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import unittest.mock as mock
from typing import Any, cast

import pytest

from gt4py._core import definitions as core_definitions
from gt4py.next import typing as gtx_typing
from gt4py.next.instrumentation import gpu_profiler
from gt4py.next.otf import compiled_program


HAS_CUPY = core_definitions.CUPY_DEVICE_TYPE is not None


# --- Helpers ---
class _FakeDefinition:
    # Declared so assigning/deleting these attributes does not trip the type checker.
    # They are intentionally left unset by default so `getattr(..., default)` is exercised.
    program_color_id: int
    compiled_program_color_id: int
    color_id: int


class _FakeProgram:
    __name__ = "my_program"
    definition: Any = _FakeDefinition()


class _FakeCompiledProgramsPool:
    root = ("my_program", "my_backend")
    definition: Any = _FakeDefinition()


def _fake_program() -> gtx_typing.Program:
    return cast(gtx_typing.Program, _FakeProgram())


# --- Fallback context managers (no CuPy) ---
class TestTimeRangeFallback:
    @pytest.mark.skipif(HAS_CUPY, reason="Only tests fallback path without CuPy")
    def test_warns_on_creation(self):
        with pytest.warns(UserWarning, match="CuPy is installed"):
            gpu_profiler.time_range("test")

    @pytest.mark.skipif(HAS_CUPY, reason="Only tests fallback path without CuPy")
    def test_usable_as_context_manager(self):
        with pytest.warns(UserWarning):
            ctx = gpu_profiler.time_range("test")
        ctx.__enter__()
        ctx.__exit__(None, None, None)


class TestProfileFallback:
    @pytest.mark.skipif(HAS_CUPY, reason="Only tests fallback path without CuPy")
    def test_warns_on_creation(self):
        with pytest.warns(UserWarning, match="CuPy is installed"):
            gpu_profiler.profile()

    @pytest.mark.skipif(HAS_CUPY, reason="Only tests fallback path without CuPy")
    def test_usable_as_context_manager(self):
        with pytest.warns(UserWarning):
            ctx = gpu_profiler.profile()
        ctx.__enter__()
        ctx.__exit__(None, None, None)


# --- ProgramCallProfiler ---
class TestProgramCallProfiler:
    def test_creates_time_range_with_program_name(self):
        with mock.patch.object(gpu_profiler, "time_range") as mock_time_range:
            gpu_profiler.ProgramCallProfiler(
                program=_fake_program(),
                args=(),
                offset_provider={},
                enable_jit=False,
                kwargs={},
            )
        mock_time_range.assert_called_once_with("my_program", color_id=1)

    def test_default_color_id(self):
        with mock.patch.object(gpu_profiler, "time_range") as mock_time_range:
            gpu_profiler.ProgramCallProfiler(
                program=_fake_program(),
                args=(),
                offset_provider={},
                enable_jit=False,
                kwargs={},
            )
        assert mock_time_range.call_args.kwargs["color_id"] == 1

    def test_custom_color_id_from_definition(self):
        program = _FakeProgram()
        program.definition.program_color_id = 42
        try:
            with mock.patch.object(gpu_profiler, "time_range") as mock_time_range:
                gpu_profiler.ProgramCallProfiler(
                    program=cast(gtx_typing.Program, program),
                    args=(),
                    offset_provider={},
                    enable_jit=False,
                    kwargs={},
                )
            assert mock_time_range.call_args.kwargs["color_id"] == 42
        finally:
            del program.definition.program_color_id

    def test_stores_time_range_on_instance(self):
        sentinel_tr = mock.MagicMock(name="time_range_instance")
        with mock.patch.object(gpu_profiler, "time_range", return_value=sentinel_tr):
            profiler = gpu_profiler.ProgramCallProfiler(
                program=_fake_program(),
                args=(),
                offset_provider={},
                enable_jit=False,
                kwargs={},
            )
        assert profiler.time_range is sentinel_tr


# --- CompiledProgramCallProfiler ---
class TestCompiledProgramCallProfiler:
    def test_creates_time_range_with_metrics_source_key(self):
        pool = _FakeCompiledProgramsPool()
        key = (("desc",), 123, None)
        with (
            mock.patch(
                "gt4py.next.instrumentation.gpu_profiler.compiled_program.metrics_source_key",
                return_value="mocked_key",
            ) as mock_key,
            mock.patch.object(gpu_profiler, "time_range") as mock_time_range,
        ):
            gpu_profiler.CompiledProgramCallProfiler(
                program_pool=cast(compiled_program.CompiledProgramsPool, pool),
                key=key,
                args=(),
                kwargs={},
                offset_provider={},
            )
            mock_key.assert_called_once_with(pool, key)
        mock_time_range.assert_called_once_with("mocked_key", color_id=2)

    def test_default_color_id(self):
        pool = _FakeCompiledProgramsPool()
        key = (("desc",), 123, None)
        with (
            mock.patch(
                "gt4py.next.instrumentation.gpu_profiler.compiled_program.metrics_source_key",
                return_value="k",
            ),
            mock.patch.object(gpu_profiler, "time_range") as mock_time_range,
        ):
            gpu_profiler.CompiledProgramCallProfiler(
                program_pool=cast(compiled_program.CompiledProgramsPool, pool),
                key=key,
                args=(),
                kwargs={},
                offset_provider={},
            )
        assert mock_time_range.call_args.kwargs["color_id"] == 2

    def test_custom_color_id_from_definition(self):
        pool = _FakeCompiledProgramsPool()
        pool.definition.compiled_program_color_id = 99
        key = (("desc",), 123, None)
        try:
            with (
                mock.patch(
                    "gt4py.next.instrumentation.gpu_profiler.compiled_program.metrics_source_key",
                    return_value="k",
                ),
                mock.patch.object(gpu_profiler, "time_range") as mock_time_range,
            ):
                gpu_profiler.CompiledProgramCallProfiler(
                    program_pool=cast(compiled_program.CompiledProgramsPool, pool),
                    key=key,
                    args=(),
                    kwargs={},
                    offset_provider={},
                )
            assert mock_time_range.call_args.kwargs["color_id"] == 99
        finally:
            del pool.definition.compiled_program_color_id

    def test_stores_time_range_on_instance(self):
        pool = _FakeCompiledProgramsPool()
        key = (("desc",), 123, None)
        sentinel_tr = mock.MagicMock(name="time_range_instance")
        with (
            mock.patch(
                "gt4py.next.instrumentation.gpu_profiler.compiled_program.metrics_source_key",
                return_value="k",
            ),
            mock.patch.object(gpu_profiler, "time_range", return_value=sentinel_tr),
        ):
            profiler = gpu_profiler.CompiledProgramCallProfiler(
                program_pool=cast(compiled_program.CompiledProgramsPool, pool),
                key=key,
                args=(),
                kwargs={},
                offset_provider={},
            )
        assert profiler.time_range is sentinel_tr


# --- ProgramProfiler enter/exit ---
class TestProgramProfilerContextManager:
    def test_enter_exit_delegates_to_time_range(self):
        profiler = gpu_profiler.ProgramCallProfiler.__new__(gpu_profiler.ProgramCallProfiler)
        fake_tr = mock.MagicMock()
        profiler.time_range = fake_tr

        profiler.__enter__()
        fake_tr.__enter__.assert_called_once_with()

        profiler.__exit__(None, None, None)
        fake_tr.__exit__.assert_called_once_with(None, None, None)

    def test_exit_forwards_exception_info(self):
        profiler = gpu_profiler.CompiledProgramCallProfiler.__new__(
            gpu_profiler.CompiledProgramCallProfiler
        )
        fake_tr = mock.MagicMock()
        profiler.time_range = fake_tr

        exc_type = ValueError
        exc = ValueError("boom")
        profiler.__exit__(exc_type, exc, None)
        fake_tr.__exit__.assert_called_once_with(exc_type, exc, None)
