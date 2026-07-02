# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import multiprocessing
import unittest.mock as mock

import pytest

from gt4py.next import backend as next_backend, config
from gt4py.next.otf import compilation_runner


@pytest.fixture
def process_runner(tmp_path):
    runner = compilation_runner.ProcessRunner(max_workers=1, shared_session_cache_dir=str(tmp_path))
    yield runner
    runner.shutdown(wait=True)


@dataclasses.dataclass(frozen=True)
class _NoOpArtifact:
    def load(self):
        return lambda *args, **kwargs: None


def _make_backend(executor):
    return next_backend.Backend(
        name="test_backend", executor=executor, allocator=None, transforms=lambda inp: inp
    )


def test_process_runner_falls_back_on_unpicklable_executor(process_runner):
    backend = _make_backend(executor=lambda compilable: _NoOpArtifact())

    with pytest.warns(UserWarning, match="not picklable"):
        future = process_runner.submit(backend, definition_stage=None, compile_time_args=None)

    assert future.done()
    assert callable(future.result())


def test_process_runner_falls_back_on_customized_compile(process_runner):
    class _WrapperBackend:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def compile(self, program, compile_time_args):
            self.compile_called = True
            return self._wrapped.compile(program, compile_time_args=compile_time_args)

    backend = _WrapperBackend(_make_backend(executor=lambda compilable: _NoOpArtifact()))

    with pytest.warns(UserWarning, match="customizes 'compile'"):
        future = process_runner.submit(backend, definition_stage=None, compile_time_args=None)

    assert future.done()
    assert backend.compile_called
    assert callable(future.result())


def test_default_runner_is_created_lazily_and_reset_makes_a_fresh_one():
    with mock.patch.object(config, "BUILD_JOBS_MODE", config.BuildJobsMode.SERIAL):
        compilation_runner.reset_default_runner()
        first = compilation_runner.get_default_runner()
        assert compilation_runner.get_default_runner() is first
        compilation_runner.reset_default_runner()
        assert compilation_runner.get_default_runner() is not first
        compilation_runner.reset_default_runner()


def test_default_runner_is_serial_in_worker_process():
    with (
        mock.patch.object(config, "BUILD_JOBS_MODE", config.BuildJobsMode.PROCESS),
        mock.patch.object(multiprocessing, "parent_process", return_value=mock.Mock()),
    ):
        compilation_runner.reset_default_runner()
        assert isinstance(compilation_runner.get_default_runner(), compilation_runner.SerialRunner)
        compilation_runner.reset_default_runner()
