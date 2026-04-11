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


class TestProfilingCallsLifecycle:
    @pytest.fixture(autouse=True)
    def _clean_profiling_state(self):
        """Ensure clean state before and after each test."""
        gpu_profiler._profile_ctx_manager = None
        yield
        gpu_profiler._profile_ctx_manager = None
        for cls in (gpu_profiler.ProgramCallProfiler, gpu_profiler.CompiledProgramCallProfiler):
            try:
                hooks.program_call_context.remove(cls)
            except KeyError:
                pass
            try:
                hooks.compiled_program_call_context.remove(cls)
            except KeyError:
                pass

    def test_start_registers_hooks(self):
        with mock.patch.object(gpu_profiler, "profile", return_value=contextlib.nullcontext()):
            gpu_profiler.start_profiling_calls()

        assert gpu_profiler.ProgramCallProfiler in hooks.program_call_context.callbacks
        assert (
            gpu_profiler.CompiledProgramCallProfiler
            in hooks.compiled_program_call_context.callbacks
        )

    def test_stop_removes_hooks(self):
        with mock.patch.object(gpu_profiler, "profile", return_value=contextlib.nullcontext()):
            gpu_profiler.start_profiling_calls()
            gpu_profiler.stop_profiling_calls()

        assert gpu_profiler.ProgramCallProfiler not in hooks.program_call_context.callbacks
        assert (
            gpu_profiler.CompiledProgramCallProfiler
            not in hooks.compiled_program_call_context.callbacks
        )

    def test_start_creates_profile_context_manager(self):
        fake_profile = mock.MagicMock()
        fake_profile.__enter__ = mock.MagicMock(return_value=fake_profile)
        fake_profile.__exit__ = mock.MagicMock(return_value=None)

        with mock.patch.object(gpu_profiler, "profile", return_value=fake_profile):
            gpu_profiler.start_profiling_calls()

        assert gpu_profiler._profile_ctx_manager is fake_profile
        fake_profile.__enter__.assert_called_once()

    def test_stop_exits_profile_context_manager(self):
        fake_profile = mock.MagicMock()
        fake_profile.__enter__ = mock.MagicMock(return_value=fake_profile)
        fake_profile.__exit__ = mock.MagicMock(return_value=None)

        with mock.patch.object(gpu_profiler, "profile", return_value=fake_profile):
            gpu_profiler.start_profiling_calls()
            gpu_profiler.stop_profiling_calls()

        fake_profile.__exit__.assert_called_once_with(None, None, None)
        assert gpu_profiler._profile_ctx_manager is None

    def test_start_is_idempotent_for_profile_ctx(self):
        fake_profile = mock.MagicMock()
        fake_profile.__enter__ = mock.MagicMock(return_value=fake_profile)

        with mock.patch.object(gpu_profiler, "profile", return_value=fake_profile):
            gpu_profiler.start_profiling_calls()
            gpu_profiler.start_profiling_calls()

        # profile().__enter__ should only be called once
        fake_profile.__enter__.assert_called_once()

    def test_profile_calls_context_manager(self):
        fake_profile = mock.MagicMock()
        fake_profile.__enter__ = mock.MagicMock(return_value=fake_profile)
        fake_profile.__exit__ = mock.MagicMock(return_value=None)

        with mock.patch.object(gpu_profiler, "profile", return_value=fake_profile):
            with gpu_profiler.profile_calls():
                assert gpu_profiler._profile_ctx_manager is not None
                assert gpu_profiler.ProgramCallProfiler in hooks.program_call_context.callbacks

        assert gpu_profiler._profile_ctx_manager is None
        assert gpu_profiler.ProgramCallProfiler not in hooks.program_call_context.callbacks

    def test_profile_calls_cleans_up_on_exception(self):
        """`profile_calls` must call `stop_profiling_calls` even if the body raises."""
        fake_profile = mock.MagicMock()
        fake_profile.__enter__ = mock.MagicMock(return_value=fake_profile)
        fake_profile.__exit__ = mock.MagicMock(return_value=None)

        with mock.patch.object(gpu_profiler, "profile", return_value=fake_profile):
            with pytest.raises(RuntimeError, match="boom"):
                with gpu_profiler.profile_calls():
                    raise RuntimeError("boom")

        assert gpu_profiler._profile_ctx_manager is None
        fake_profile.__exit__.assert_called_once_with(None, None, None)
        assert gpu_profiler.ProgramCallProfiler not in hooks.program_call_context.callbacks
        assert (
            gpu_profiler.CompiledProgramCallProfiler
            not in hooks.compiled_program_call_context.callbacks
        )

    def test_stop_without_start_is_noop(self):
        """Calling stop without a prior start should not raise (hooks were never registered)."""
        # Sanity-check preconditions set by the fixture.
        assert gpu_profiler._profile_ctx_manager is None
        assert gpu_profiler.ProgramCallProfiler not in hooks.program_call_context.callbacks

        gpu_profiler.stop_profiling_calls()

        assert gpu_profiler._profile_ctx_manager is None
        assert gpu_profiler.ProgramCallProfiler not in hooks.program_call_context.callbacks
        assert (
            gpu_profiler.CompiledProgramCallProfiler
            not in hooks.compiled_program_call_context.callbacks
        )

    def test_double_start_registers_hooks_once(self):
        """Repeated `start_profiling_calls` must not re-register the hooks."""
        fake_profile = mock.MagicMock()
        fake_profile.__enter__ = mock.MagicMock(return_value=fake_profile)

        with mock.patch.object(gpu_profiler, "profile", return_value=fake_profile):
            gpu_profiler.start_profiling_calls()
            gpu_profiler.start_profiling_calls()

        program_occurrences = sum(
            1
            for cb in hooks.program_call_context.callbacks
            if cb is gpu_profiler.ProgramCallProfiler
        )
        compiled_occurrences = sum(
            1
            for cb in hooks.compiled_program_call_context.callbacks
            if cb is gpu_profiler.CompiledProgramCallProfiler
        )
        assert program_occurrences == 1
        assert compiled_occurrences == 1
