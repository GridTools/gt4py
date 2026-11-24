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
import hashlib
import multiprocessing
import os
import pickle
import sys
import unittest.mock as mock

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import backend as next_backend, common, config, fingerprinting
from gt4py.next.otf import arguments, compilation_tasks, compiled_program, runners

from next_tests.fixtures import compilation as fixtures_compilation


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
        compilable="compilable",
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
        compilable=None,
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


_xp_params = pytest.mark.parametrize(
    "xp",
    [np, pytest.param(core_defs.cp, id="cupy", marks=pytest.mark.requires_gpu)],
    ids=lambda xp: getattr(xp, "__name__", "cupy"),
)


def _v2e_tables(xp):
    Vertex = gtx.Dimension("Vertex")
    Edge = gtx.Dimension("Edge")
    V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
    table = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
    c_order = gtx.as_connectivity([Vertex, V2EDim], Edge, xp.asarray(table))
    f_order = type(c_order).from_array(
        xp.asarray(np.asfortranarray(table)), Edge, domain=c_order.domain
    )
    assert f_order.ndarray.flags.f_contiguous
    return table, {"V2E": c_order, "V2EF": f_order, "Vertex": Vertex}


@_xp_params
def test_compile_time_offset_provider_is_canonical_and_memoized(xp):
    table, offset_provider = _v2e_tables(xp)

    compile_time = compilation_tasks.compile_time_offset_provider(offset_provider)

    assert list(compile_time) == list(offset_provider)
    assert compile_time["Vertex"] is offset_provider["Vertex"]
    for name in ("V2E", "V2EF"):
        conn, original = compile_time[name], offset_provider[name]
        assert type(conn) is compilation_tasks._CompileTimeConnectivity
        assert conn.ndarray.flags.c_contiguous
        assert not conn.ndarray.flags.writeable
        assert np.array_equal(conn.ndarray, table)
        assert conn.domain == original.domain
        assert conn.codomain == original.codomain
        assert conn.skip_value == original.skip_value
        assert compilation_tasks.compile_time_offset_provider(offset_provider)[name] is conn
        assert compilation_tasks.compile_time_offset_provider(compile_time)[name] is conn


@_xp_params
def test_compile_time_connectivity_copies_each_table_once(xp, monkeypatch):
    _, offset_provider = _v2e_tables(xp)
    conn_type = type(offset_provider["V2E"])
    asnumpy = mock.Mock(side_effect=conn_type.asnumpy)
    monkeypatch.setattr(conn_type, "asnumpy", lambda self: asnumpy(self))

    for _ in range(2):
        compile_time = compilation_tasks.compile_time_offset_provider(offset_provider)
        pickle.loads(pickle.dumps(compile_time))
        fingerprinting.strict_fingerprinter(compile_time)

    assert asnumpy.call_count == 2  # once per table


def test_compile_time_connectivity_fingerprint_depends_only_on_content():
    _, offset_provider = _v2e_tables(np)
    conn = offset_provider["V2E"]
    Vertex, Edge = offset_provider["Vertex"], conn.codomain
    variants = {
        "content": gtx.as_connectivity(
            conn.domain, Edge, np.array([[0, 1], [1, 2], [2, 2]], dtype=np.int32)
        ),
        "skip_value": gtx.as_connectivity(
            conn.domain, Edge, np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32), skip_value=-1
        ),
        "codomain": gtx.as_connectivity(
            conn.domain, Vertex, np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
        ),
        "domain": gtx.as_connectivity(
            [gtx.Dimension("Cell"), conn.domain.dims[1]],
            Edge,
            np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        ),
        "dtype": gtx.as_connectivity(
            conn.domain, Edge, np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
        ),
    }

    def fingerprint(c):
        return fingerprinting.strict_fingerprinter(
            compilation_tasks.compile_time_offset_provider({"X": c})["X"]
        )

    assert fingerprint(conn) == fingerprint(offset_provider["V2EF"])
    for name, other in variants.items():
        assert fingerprint(other) != fingerprint(conn), name


def test_compile_time_connectivity_pickles_as_file_reference():
    Vertex = gtx.Dimension("Vertex")
    V2VDim = gtx.Dimension("V2V", kind=gtx.DimensionKind.LOCAL)
    table = np.arange(4000, dtype=np.int32).reshape(1000, 4) % 1000
    table[::7, 3] = -1
    conn = compilation_tasks.compile_time_offset_provider(
        {"V2V": gtx.as_connectivity([Vertex, V2VDim], Vertex, table, skip_value=-1)}
    )["V2V"]

    data = pickle.dumps(conn)
    restored = pickle.loads(data)

    assert len(data) < table.nbytes
    assert isinstance(restored.ndarray.base, np.memmap)
    assert np.array_equal(restored.ndarray, table)
    assert restored.domain == conn.domain
    assert restored.codomain == conn.codomain
    assert restored.skip_value == conn.skip_value == -1
    assert fingerprinting.strict_fingerprinter(restored) == fingerprinting.strict_fingerprinter(
        conn
    )


def test_compile_time_connectivity_copies_are_valid():
    _, offset_provider = _v2e_tables(np)
    conn = compilation_tasks.compile_time_offset_provider(offset_provider)["V2E"]

    rebuilt = type(conn)(conn.domain, conn.ndarray, conn.codomain, conn.skip_value)
    restricted = conn.restrict((slice(0, 2),))

    assert fingerprinting.strict_fingerprinter(rebuilt) == fingerprinting.strict_fingerprinter(conn)
    assert np.array_equal(pickle.loads(pickle.dumps(restricted)).ndarray, conn.ndarray[0:2])
    assert fingerprinting.strict_fingerprinter(restricted) != fingerprinting.strict_fingerprinter(
        conn
    )


def test_compile_time_table_files_are_content_addressed_and_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(compilation_tasks._cache, "_session_cache_dir_path", tmp_path)
    Vertex = gtx.Dimension("Vertex")
    V2VDim = gtx.Dimension("V2V", kind=gtx.DimensionKind.LOCAL)
    table = np.array([[0, 1], [1, 0]], dtype=np.int32)
    sources = [gtx.as_connectivity([Vertex, V2VDim], Vertex, table) for _ in range(8)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        views = list(
            pool.map(
                lambda c: compilation_tasks.compile_time_offset_provider({"V": c})["V"], sources
            )
        )
    (path,) = (tmp_path / "connectivities").iterdir()
    inode = path.stat().st_ino
    compilation_tasks.compile_time_offset_provider(
        {"V": gtx.as_connectivity([Vertex, V2VDim], Vertex, table)}
    )

    assert len({view._digest for view in views}) == 1
    assert path.name == f"{views[0]._digest}.npy"
    assert path.stat().st_ino == inode
    assert [p.name for p in (tmp_path / "connectivities").iterdir()] == [path.name]


def test_compile_time_table_write_failure_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(compilation_tasks._cache, "_session_cache_dir_path", tmp_path)

    def failing_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(compilation_tasks.np, "save", failing_save)
    _, offset_provider = _v2e_tables(np)

    with pytest.raises(OSError, match="disk full"):
        compilation_tasks.compile_time_offset_provider(offset_provider)
    assert list((tmp_path / "connectivities").iterdir()) == []


def test_make_compilation_task_compiles_against_compile_time_view():
    _, offset_provider = _v2e_tables(np)
    compile_time_args = dataclasses.replace(
        arguments.CompileTimeArgs.empty(), offset_provider=offset_provider
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

    compiled_against = task.compilable.args.offset_provider
    assert type(compiled_against["V2E"]) is compilation_tasks._CompileTimeConnectivity
    assert type(compiled_against["V2EF"]) is compilation_tasks._CompileTimeConnectivity
    assert compiled_against["Vertex"] is offset_provider["Vertex"]


def test_compile_time_connectivity_registry_prunes_on_gc():
    _, offset_provider = _v2e_tables(np)
    conn = offset_provider.pop("V2EF")
    key = id(conn)
    compilation_tasks.compile_time_offset_provider({"V2EF": conn})
    assert key in compilation_tasks._compile_time_connectivities
    del conn
    gc.collect()
    assert key not in compilation_tasks._compile_time_connectivities


def test_compile_time_connectivity_digest_is_derived_from_table():
    _, offset_provider = _v2e_tables(np)
    conn = compilation_tasks.compile_time_offset_provider(offset_provider)["V2E"]

    replaced = dataclasses.replace(
        conn, _ndarray=np.array([[0, 1], [1, 2], [2, 2]], dtype=np.int32)
    )

    assert replaced._digest != conn._digest
    with pytest.raises(TypeError):
        type(conn)(conn.domain, conn.ndarray, conn.codomain, conn.skip_value, conn._digest)


def test_compile_time_connectivity_rejects_mismatching_table_file():
    _, offset_provider = _v2e_tables(np)
    conn = compilation_tasks.compile_time_offset_provider(offset_provider)["V2E"]
    reconstruct, args = conn.__reduce__()
    path, digest, domain, codomain, skip_value, shape, dtype = args

    with pytest.raises(ValueError, match="expected"):
        reconstruct(path, digest, domain, codomain, skip_value, (shape[0] + 1, shape[1]), dtype)
    with pytest.raises(ValueError, match="expected"):
        reconstruct(path, digest, domain, codomain, skip_value, shape, "<i8")


def _fail_to_unpickle():
    raise FileNotFoundError("connectivity table file is gone")


class _FailsToUnpickle:
    def __reduce__(self):
        return (_fail_to_unpickle, ())


def test_process_runner_fails_only_the_task_whose_compilable_cannot_be_unpickled(process_runner):
    broken = runners.CompilationTask(
        name="broken", compilable=_FailsToUnpickle(), executor=_RecursingExecutor(0)
    )

    with pytest.raises(FileNotFoundError, match="is gone"):
        process_runner.submit(broken).result(timeout=120)
    assert callable(
        process_runner.submit(_decomposed_task(executor=_RecursingExecutor(0)))
        .result(timeout=120)
        .load()
    )


@dataclasses.dataclass(frozen=True)
class _CompilableSummaryArtifact:
    fingerprint: str
    tables: tuple
    contents: tuple

    def load(self):
        return lambda *args, **kwargs: None


@dataclasses.dataclass(frozen=True)
class _SummarizingExecutor:
    def __call__(self, compilable):
        return _CompilableSummaryArtifact(
            fingerprinting.strict_fingerprinter(compilable),
            tuple(
                (name, type(conn).__name__, conn.ndarray.flags.c_contiguous)
                for name, conn in compilable.args.offset_provider.items()
                if isinstance(conn, common.GatherConnectivity)
            ),
            tuple(
                (name, _sha256(conn.ndarray))
                for name, conn in compilable.args.offset_provider.items()
                if isinstance(conn, common.GatherConnectivity)
            ),
        )


def _sha256(table) -> str:
    table = np.ascontiguousarray(np.asarray(table))
    return hashlib.sha256(
        table.dtype.str.encode() + repr(table.shape).encode() + table.tobytes()
    ).hexdigest()


@_xp_params
def test_compilable_is_identical_in_every_runner(xp, tmp_path):
    table, offset_provider = _v2e_tables(xp)
    compile_time_args = dataclasses.replace(
        arguments.CompileTimeArgs.empty(),
        offset_provider=compilation_tasks.compile_time_offset_provider(offset_provider),
    )
    backend = next_backend.Backend(
        name="test_backend",
        executor=_SummarizingExecutor(),
        allocator=None,
        transforms=lambda inp: inp,
    )
    task = compilation_tasks.make_compilation_task(
        backend, definition_stage=None, compile_time_args=compile_time_args
    )

    all_runners = [
        runners.SerialRunner(),
        runners.ThreadRunner(max_workers=1),
        runners.ProcessRunner(max_workers=1, shared_session_cache_dir=str(tmp_path)),
    ]
    try:
        summaries = [runner.submit(task).result(timeout=120) for runner in all_runners]
    finally:
        for runner in all_runners:
            runner.shutdown(wait=True)

    assert len({summary.fingerprint for summary in summaries}) == 1
    expected = (
        ("V2E", "_CompileTimeConnectivity", True),
        ("V2EF", "_CompileTimeConnectivity", True),
    )
    assert [summary.tables for summary in summaries] == [expected] * 3
    expected_contents = (("V2E", _sha256(table)), ("V2EF", _sha256(table)))
    assert [summary.contents for summary in summaries] == [expected_contents] * 3


def test_wait_for_compilation_untracks_successful_futures():
    future = concurrent.futures.Future()
    future.set_result(lambda: None)
    compiled_program._ongoing_compilations[future] = "testee (backend)"

    compiled_program.wait_for_compilation()

    assert future not in compiled_program._ongoing_compilations


def test_wait_for_compilation_groups_multiple_failures():
    errors = [ValueError("first boom"), TypeError("second boom")]
    # The futures have to stay referenced: tracking is weak, so a collected future
    # is not reported and this would degrade to the single-failure path.
    futures = [concurrent.futures.Future() for _ in errors]
    for i, (future, error) in enumerate(zip(futures, errors)):
        future.set_exception(error)
        compiled_program._ongoing_compilations[future] = f"testee_{i} (backend)"

    with pytest.raises(ExceptionGroup) as exc_info:
        compiled_program.wait_for_compilation()

    # Every failure keeps its own traceback instead of being flattened into the
    # message of a single error, and each program is named. It is a plain
    # 'ExceptionGroup', not just a 'BaseExceptionGroup', so 'except*' on 'Exception'
    # catches it.
    assert type(exc_info.value) is ExceptionGroup
    assert exc_info.value.exceptions == tuple(errors)
    assert "testee_0 (backend)" in str(exc_info.value)
    assert "testee_1 (backend)" in str(exc_info.value)

    # each failure is reported only once
    compiled_program.wait_for_compilation()


def test_wait_for_compilation_drains_foreign_entries():
    """Pin the leak the `isolate_ongoing_compilations` fixture exists to contain.

    The drain is process-global, so it reports entries this test never created.
    Across tests that is what makes one test fail with another one's error.
    """
    # The future has to stay referenced: tracking is weak, so a collected future
    # is not reported at all.
    foreign = concurrent.futures.Future()
    foreign.set_exception(ValueError("foreign boom"))
    compiled_program._ongoing_compilations[foreign] = "testee (run_gtfn_gpu)"

    # Nothing in this test compiled anything, yet it drains someone else's failure.
    with pytest.raises(ValueError, match="foreign boom"):
        compiled_program.wait_for_compilation()


def test_reset_ongoing_compilations_untracks_pending_failures():
    """A reset test must not see a failure a previous one left behind."""
    foreign = concurrent.futures.Future()
    foreign.set_exception(ValueError("foreign boom"))
    compiled_program._ongoing_compilations[foreign] = "testee (run_gtfn_gpu)"

    fixtures_compilation.reset_ongoing_compilations()

    assert foreign not in compiled_program._ongoing_compilations
    compiled_program.wait_for_compilation()  # must not raise


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
