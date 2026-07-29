# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import concurrent.futures
import dataclasses
import gc
import multiprocessing
import os
import pickle
import sys
import unittest.mock as mock

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend, config
from gt4py.next.otf import arguments, compilation_tasks, compiled_program, runners


@pytest.fixture
def process_runner(tmp_path):
    runner = runners.ProcessRunner(max_workers=1, shared_session_cache_dir=str(tmp_path))
    yield runner
    runner.shutdown(wait=True)


@dataclasses.dataclass(frozen=True)
class _NoOpArtifact:
    def load(self):
        return lambda *args, **kwargs: None


def _decomposed_task(executor):
    return runners.CompilationTask(
        name="test_task",
        construct_compilable=lambda with_refs: "compilable",
        executor=executor,
    )


def test_process_runner_falls_back_on_unpicklable_executor(process_runner):
    task = _decomposed_task(executor=lambda compilable: _NoOpArtifact())

    with pytest.warns(UserWarning, match="not picklable"):
        future = process_runner.submit(task)

    assert future.done()
    assert callable(future.result().load())


def _recurse(depth: int) -> None:
    if depth > 0:
        _recurse(depth - 1)


@dataclasses.dataclass(frozen=True)
class _RecursingExecutor:
    depth: int

    def __call__(self, compilable):
        _recurse(self.depth)
        return _NoOpArtifact()


def test_process_runner_propagates_recursion_limit(process_runner):
    """A raised ``sys.setrecursionlimit`` must reach the worker: it is per-process,
    and ``spawn`` workers start back at the interpreter default, so deep-IR
    compiles that succeed in the parent would otherwise fail worker-side."""
    old_limit = sys.getrecursionlimit()
    depth = old_limit + 1000
    sys.setrecursionlimit(depth + 1000)
    try:
        future = process_runner.submit(_decomposed_task(executor=_RecursingExecutor(depth)))
        assert callable(future.result(timeout=120).load())
    finally:
        sys.setrecursionlimit(old_limit)


def test_process_runner_falls_back_on_non_offloadable_task(process_runner):
    task = runners.CompilationTask(
        name="opaque",
        construct_compilable=lambda with_refs: None,
        executor=lambda _: _NoOpArtifact(),
        no_offload_reason="it does not use the standard compilation workflow",
    )

    with pytest.warns(UserWarning, match="standard compilation workflow"):
        future = process_runner.submit(task)

    assert future.done()
    assert callable(future.result().load())


def test_make_compilation_task_decomposes_standard_backend():
    backend = next_backend.Backend(
        name="test_backend",
        executor=lambda compilable: _NoOpArtifact(),
        allocator=None,
        transforms=lambda inp: inp,
    )

    task = compilation_tasks.make_compilation_task(
        backend, definition_stage=None, compile_time_args=arguments.CompileTimeArgs.empty()
    )

    assert task.no_offload_reason is None
    assert task.executor is backend.executor
    assert callable(task.compile().load())


def test_make_compilation_task_is_opaque_for_customized_compile():
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

    task = compilation_tasks.make_compilation_task(
        backend, definition_stage=None, compile_time_args=arguments.CompileTimeArgs.empty()
    )

    assert task.no_offload_reason is not None
    assert callable(task.compile().load())
    assert backend.compile_called


def test_offloaded_task_ships_connectivities_as_file_refs():
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

    task = compilation_tasks.make_compilation_task(
        backend, definition_stage=None, compile_time_args=compile_time_args
    )

    # without refs the original compilable is used as is
    assert task.construct_compilable(False).args.offset_provider["V2E"] is conn
    shipped = task.construct_compilable(True)
    ref = shipped.args.offset_provider["V2E"]
    assert isinstance(ref, compilation_tasks._ConnectivityFileRef)
    # task preparation is pure: nothing is dumped until a runner ships the task
    assert id(conn) not in compilation_tasks._connectivity_files
    # unpickling (as in a worker) yields the memory-mapped connectivity
    restored = pickle.loads(pickle.dumps(ref))
    assert np.array_equal(restored.asnumpy(), conn.asnumpy())
    assert restored.domain == conn.domain
    assert restored.codomain == conn.codomain
    assert restored.skip_value == conn.skip_value
    # the same connectivity is dumped only once across tasks
    path = compilation_tasks._connectivity_files[id(conn)][1]
    task2 = compilation_tasks.make_compilation_task(
        backend, definition_stage=None, compile_time_args=compile_time_args
    )
    pickle.dumps(task2.construct_compilable(True).args.offset_provider["V2E"])
    assert compilation_tasks._connectivity_files[id(conn)][1] == path


def test_connectivity_file_registry_prunes_on_gc():
    Vertex = gtx.Dimension("Vertex")
    Edge = gtx.Dimension("Edge")
    V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
    conn = gtx.as_connectivity([Vertex, V2EDim], Edge, np.array([[0, 1], [1, 2], [2, 0]]))

    compilation_tasks._dump_connectivity(conn)
    key = id(conn)
    assert key in compilation_tasks._connectivity_files

    del conn
    gc.collect()
    assert key not in compilation_tasks._connectivity_files


def test_wait_for_compilation_untracks_successful_futures():
    future = concurrent.futures.Future()
    future.set_result(lambda: None)
    compiled_program._ongoing_compilations[future] = "testee (backend)"

    compiled_program.wait_for_compilation()

    assert future not in compiled_program._ongoing_compilations


def test_detect_cuda_archs_prefers_cudaarchs_env():
    with (
        mock.patch.dict(os.environ, {"CUDAARCHS": "80;90"}),
        mock.patch.object(
            runners.core_defs,
            "CUPY_DEVICE_TYPE",
            runners.core_defs.DeviceType.CUDA,
        ),
    ):
        assert runners._detect_cuda_archs() == "80;90"


def test_detect_cuda_archs_queries_device():
    with (
        mock.patch.dict(os.environ),
        mock.patch.object(runners.compilation_common, "get_device_arch", return_value="90"),
        mock.patch.object(
            runners.core_defs,
            "CUPY_DEVICE_TYPE",
            runners.core_defs.DeviceType.CUDA,
        ),
    ):
        os.environ.pop("CUDAARCHS", None)
        assert runners._detect_cuda_archs() == "90"


def test_detect_cuda_archs_none_without_cuda_device_type():
    with (
        mock.patch.dict(os.environ),
        mock.patch.object(runners.core_defs, "CUPY_DEVICE_TYPE", None),
    ):
        os.environ.pop("CUDAARCHS", None)
        assert runners._detect_cuda_archs() is None


def test_pool_worker_initializer_hides_gpus_when_archs_known(tmp_path):
    with (
        mock.patch.dict(os.environ),
        mock.patch.object(runners._cache, "_session_cache_dir_path"),
    ):
        runners._pool_worker_initializer(str(tmp_path), "90")
        assert os.environ["CUDAARCHS"] == "90"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert runners._cache._session_cache_dir_path == tmp_path


def test_pool_worker_initializer_leaves_gpus_visible_without_archs(tmp_path):
    with (
        mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}),
        mock.patch.object(runners._cache, "_session_cache_dir_path"),
    ):
        # `patch.dict` only layers on top of the inherited environment; pin the
        # precondition so an ambient CUDAARCHS (e.g. set by CI) cannot leak in.
        os.environ.pop("CUDAARCHS", None)
        runners._pool_worker_initializer(str(tmp_path), None)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
        assert "CUDAARCHS" not in os.environ


def test_default_runner_is_created_lazily_and_reset_makes_a_fresh_one():
    with mock.patch.object(config, "BUILD_JOBS_MODE", config.BuildJobsMode.SERIAL):
        runners.reset_default_runner()
        first = runners.get_default_runner()
        assert runners.get_default_runner() is first
        runners.reset_default_runner()
        assert runners.get_default_runner() is not first
        runners.reset_default_runner()


def test_default_runner_is_serial_in_worker_process():
    with (
        mock.patch.object(config, "BUILD_JOBS_MODE", config.BuildJobsMode.PROCESS),
        mock.patch.object(multiprocessing, "parent_process", return_value=mock.Mock()),
    ):
        runners.reset_default_runner()
        assert isinstance(runners.get_default_runner(), runners.SerialRunner)
        runners.reset_default_runner()
