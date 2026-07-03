# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import multiprocessing
import os
import pickle
import unittest.mock as mock

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend, config
from gt4py.next.otf import arguments, compilation_runner, compile_jobs, compiled_program


@pytest.fixture
def process_runner(tmp_path):
    runner = compilation_runner.ProcessRunner(max_workers=1, shared_session_cache_dir=str(tmp_path))
    yield runner
    runner.shutdown(wait=True)


@dataclasses.dataclass(frozen=True)
class _NoOpArtifact:
    def load(self):
        return lambda *args, **kwargs: None


def _decomposed_job(executor):
    compilable = "compilable"
    return compilation_runner.CompileJob(
        name="test_job",
        run=lambda: executor(compilable).load(),
        offload=compilation_runner.OffloadableWork(compilable=compilable, executor=executor),
    )


def test_process_runner_falls_back_on_unpicklable_executor(process_runner):
    job = _decomposed_job(executor=lambda compilable: _NoOpArtifact())

    with pytest.warns(UserWarning, match="not picklable"):
        future = process_runner.submit(job)

    assert future.done()
    assert callable(future.result())


def test_process_runner_falls_back_on_non_offloadable_job(process_runner):
    job = compilation_runner.CompileJob(name="opaque", run=lambda: _NoOpArtifact().load())

    with pytest.warns(UserWarning, match="standard compilation workflow"):
        future = process_runner.submit(job)

    assert future.done()
    assert callable(future.result())


def test_make_compile_job_decomposes_standard_backend():
    backend = next_backend.Backend(
        name="test_backend",
        executor=lambda compilable: _NoOpArtifact(),
        allocator=None,
        transforms=lambda inp: inp,
    )

    job = compile_jobs.make_compile_job(
        backend, definition_stage=None, compile_time_args=arguments.CompileTimeArgs.empty()
    )

    assert job.offload is not None
    assert job.offload.executor is backend.executor
    assert callable(job.run())


def test_make_compile_job_is_opaque_for_customized_compile():
    class _WrapperBackend:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def compile(self, program, compile_time_args):
            self.compile_called = True
            return self._wrapped.compile(program, compile_time_args=compile_time_args)

    backend = _WrapperBackend(
        next_backend.Backend(
            name="test_backend",
            executor=lambda compilable: _NoOpArtifact(),
            allocator=None,
            transforms=lambda inp: inp,
        )
    )

    job = compile_jobs.make_compile_job(
        backend, definition_stage=None, compile_time_args=arguments.CompileTimeArgs.empty()
    )

    assert job.offload is None
    assert callable(job.run())
    assert backend.compile_called


def test_offloaded_job_ships_connectivities_as_file_refs():
    Vertex = gtx.Dimension("Vertex")
    Edge = gtx.Dimension("Edge")
    V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
    conn = gtx.as_connectivity([Vertex, V2EDim], Edge, np.array([[0, 1], [1, 2], [2, 0]]))
    compile_time_args = dataclasses.replace(
        arguments.CompileTimeArgs.empty(), offset_provider={"V2E": conn}
    )
    backend = next_backend.Backend(
        name="test_backend",
        executor=lambda compilable: _NoOpArtifact(),
        allocator=None,
        transforms=lambda inp: inp,
    )

    job = compile_jobs.make_compile_job(
        backend, definition_stage=None, compile_time_args=compile_time_args
    )

    ref = job.offload.compilable.args.offset_provider["V2E"]
    assert isinstance(ref, compile_jobs._ConnectivityFileRef)
    # the same connectivity is dumped only once
    job2 = compile_jobs.make_compile_job(
        backend, definition_stage=None, compile_time_args=compile_time_args
    )
    assert job2.offload.compilable.args.offset_provider["V2E"].path == ref.path
    # unpickling (as in a worker) yields the memory-mapped connectivity
    restored = pickle.loads(pickle.dumps(ref))
    assert np.array_equal(restored.asnumpy(), conn.asnumpy())
    assert restored.domain == conn.domain
    assert restored.codomain == conn.codomain
    assert restored.skip_value == conn.skip_value


def test_detect_cuda_archs_prefers_cudaarchs_env():
    with (
        mock.patch.dict(os.environ, {"CUDAARCHS": "80;90"}),
        mock.patch.object(
            compilation_runner.core_defs,
            "CUPY_DEVICE_TYPE",
            compilation_runner.core_defs.DeviceType.CUDA,
        ),
    ):
        assert compilation_runner._detect_cuda_archs() == "80;90"


def test_detect_cuda_archs_queries_device():
    with (
        mock.patch.dict(os.environ),
        mock.patch.object(
            compilation_runner.compilation_common, "get_device_arch", return_value="90"
        ),
        mock.patch.object(
            compilation_runner.core_defs,
            "CUPY_DEVICE_TYPE",
            compilation_runner.core_defs.DeviceType.CUDA,
        ),
    ):
        os.environ.pop("CUDAARCHS", None)
        assert compilation_runner._detect_cuda_archs() == "90"


def test_detect_cuda_archs_none_without_cuda_device_type():
    with (
        mock.patch.dict(os.environ),
        mock.patch.object(compilation_runner.core_defs, "CUPY_DEVICE_TYPE", None),
    ):
        os.environ.pop("CUDAARCHS", None)
        assert compilation_runner._detect_cuda_archs() is None


def test_pool_worker_initializer_hides_gpus_when_archs_known(tmp_path):
    with (
        mock.patch.dict(os.environ),
        mock.patch.object(compilation_runner._cache, "_session_cache_dir_path"),
    ):
        compilation_runner._pool_worker_initializer(str(tmp_path), "90")
        assert os.environ["CUDAARCHS"] == "90"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert compilation_runner._cache._session_cache_dir_path == tmp_path


def test_pool_worker_initializer_leaves_gpus_visible_without_archs(tmp_path):
    with (
        mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}),
        mock.patch.object(compilation_runner._cache, "_session_cache_dir_path"),
    ):
        compilation_runner._pool_worker_initializer(str(tmp_path), None)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
        assert "CUDAARCHS" not in os.environ or os.environ["CUDAARCHS"] == ""


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
