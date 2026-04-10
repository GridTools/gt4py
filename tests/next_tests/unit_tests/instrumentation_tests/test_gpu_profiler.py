# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import unittest.mock as mock

import pytest

from gt4py._core import definitions as core_definitions
from gt4py.next.instrumentation import gpu_profiler, hooks


HAS_CUPY = core_definitions.CUPY_DEVICE_TYPE is not None


# --- Helpers ---
class _FakeDefinition:
    pass


class _FakeProgram:
    __name__ = "my_program"
    definition = _FakeDefinition()


class _FakeCompiledProgramsPool:
    root = ("my_program", "my_backend")
    definition = _FakeDefinition()


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
    def test_stores_program_name(self):
        profiler = gpu_profiler.ProgramCallProfiler(
            program=_FakeProgram(),
            args=(),
            offset_provider={},
            enable_jit=False,
            kwargs={},
        )
        assert profiler.name == "my_program"

    def test_default_color_id(self):
        profiler = gpu_profiler.ProgramCallProfiler(
            program=_FakeProgram(),
            args=(),
            offset_provider={},
            enable_jit=False,
            kwargs={},
        )
        assert profiler.color_id == 1

    def test_custom_color_id_from_definition(self):
        program = _FakeProgram()
        program.definition.color_id = 42
        profiler = gpu_profiler.ProgramCallProfiler(
            program=program,
            args=(),
            offset_provider={},
            enable_jit=False,
            kwargs={},
        )
        assert profiler.color_id == 42


# --- CompiledProgramCallProfiler ---
class TestCompiledProgramCallProfiler:
    def test_stores_metrics_source_key(self):
        pool = _FakeCompiledProgramsPool()
        key = (("desc",), 123, None)
        with mock.patch(
            "gt4py.next.instrumentation.gpu_profiler.compiled_program.metrics_source_key",
            return_value="mocked_key",
        ) as mock_key:
            profiler = gpu_profiler.CompiledProgramCallProfiler(
                program_pool=pool,
                key=key,
                args=(),
                kwargs={},
                offset_provider={},
            )
            mock_key.assert_called_once_with(pool, key)
        assert profiler.name == "mocked_key"

    def test_default_color_id(self):
        pool = _FakeCompiledProgramsPool()
        key = (("desc",), 123, None)
        with mock.patch(
            "gt4py.next.instrumentation.gpu_profiler.compiled_program.metrics_source_key",
            return_value="k",
        ):
            profiler = gpu_profiler.CompiledProgramCallProfiler(
                program_pool=pool,
                key=key,
                args=(),
                kwargs={},
                offset_provider={},
            )
        assert profiler.color_id == 2

    def test_custom_color_id_from_definition(self):
        pool = _FakeCompiledProgramsPool()
        pool.definition.color_id = 99
        key = (("desc",), 123, None)
        with mock.patch(
            "gt4py.next.instrumentation.gpu_profiler.compiled_program.metrics_source_key",
            return_value="k",
        ):
            profiler = gpu_profiler.CompiledProgramCallProfiler(
                program_pool=pool,
                key=key,
                args=(),
                kwargs={},
                offset_provider={},
            )
        assert profiler.color_id == 99


# --- ProgramProfiler enter/exit ---
class TestProgramProfilerContextManager:
    def test_enter_exit_delegates_to_time_range(self):
        profiler = gpu_profiler.ProgramCallProfiler.__new__(gpu_profiler.ProgramCallProfiler)
        profiler.name = "test_op"
        profiler.color_id = 5

        fake_tr = mock.MagicMock()
        fake_tr.__enter__ = mock.MagicMock(return_value=fake_tr)
        fake_tr.__exit__ = mock.MagicMock(return_value=None)

        with mock.patch.object(gpu_profiler, "time_range", return_value=fake_tr) as mock_time_range:
            profiler.__enter__()
            mock_time_range.assert_called_once_with("test_op", color_id=5)
            fake_tr.__enter__.assert_called_once()

            profiler.__exit__(None, None, None)
            fake_tr.__exit__.assert_called_once_with(None, None, None)
