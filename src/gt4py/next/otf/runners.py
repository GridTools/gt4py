# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Runners executing prepared compilation tasks serially, in threads, or in worker processes."""

from __future__ import annotations

import atexit
import concurrent.futures
import dataclasses
import multiprocessing
import os
import pathlib
import pickle
import sys
import threading
import warnings
from typing import Any, Callable, Protocol, runtime_checkable

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import stages
from gt4py.next.otf.compilation import cache as _cache, common as compilation_common


@dataclasses.dataclass(frozen=True)
class CompilationTask:
    """A task yielding a `CompilationArtifact`, fully prepared by the caller.

    Loading the artifact is the caller's concern; a runner never loads.
    """

    #: Label used in user-facing messages.
    name: str
    #: Constructs the program to compile; with ``with_refs`` the connectivity
    #: buffers are replaced by file references for crossing a process boundary.
    construct_compilable: Callable[[bool], Any]
    #: The artifact-producing step. A runner may execute it anywhere, including
    #: another process; its picklability is the runner's concern.
    executor: Callable[[Any], stages.CompilationArtifact]
    #: Reason this task is known not to be shippable to another process, if any.
    no_offload_reason: str | None = None

    def compile(self, with_refs: bool = False) -> stages.CompilationArtifact:
        return self.executor(self.construct_compilable(with_refs))


@runtime_checkable
class Runner(Protocol):
    def submit(
        self, task: CompilationTask
    ) -> concurrent.futures.Future[stages.CompilationArtifact]:
        """Schedule `task`.

        Returns:
            A future yielding the ``CompilationArtifact``.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Release the resources backing this runner (workers, threads)."""
        ...


def _run_in_calling_thread(
    task: CompilationTask,
) -> concurrent.futures.Future[stages.CompilationArtifact]:
    future: concurrent.futures.Future[stages.CompilationArtifact] = concurrent.futures.Future()
    try:
        future.set_result(task.compile())
    except BaseException as exception:  # re-raised via the future
        future.set_exception(exception)
    return future


class SerialRunner:
    """Runs compilation in the calling thread; the returned future is already done."""

    def submit(
        self, task: CompilationTask
    ) -> concurrent.futures.Future[stages.CompilationArtifact]:
        return _run_in_calling_thread(task)

    def shutdown(self, wait: bool = True) -> None:
        return None


class ThreadRunner:
    """Compiles in a ``ThreadPoolExecutor``."""

    def __init__(self, max_workers: int) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self, task: CompilationTask
    ) -> concurrent.futures.Future[stages.CompilationArtifact]:
        return self._pool.submit(task.compile)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


def _detect_cuda_archs() -> str | None:
    """CUDA architecture for worker builds (CMake style, e.g. ``"90"``), or None.

    CUDA-only: the worker initializer communicates the value via ``CUDAARCHS``,
    which has no HIP counterpart with settled semantics yet.
    """
    if core_defs.CUPY_DEVICE_TYPE is not core_defs.DeviceType.CUDA:
        return None
    return compilation_common.get_device_arch()


def _pool_worker_initializer(shared_session_cache_dir: str, cuda_archs: str | None) -> None:
    """Prepare a worker process for building without touching the parent's GPU.

    Points the worker's session-lifetime build cache at the main process's temp
    dir: each worker would otherwise create its own ``TemporaryDirectory`` and
    scrub it on exit — taking the compiled artifacts with it before main can
    ``dlopen`` them.

    When the CUDA architectures are known, hides all GPUs from the worker and
    exports ``CUDAARCHS`` instead: GPU builds otherwise probe the device for its
    architecture (gt4py's cmake build system via cupy, dace's CMake via a
    ``try_run`` binary), each probe creating a CUDA context on the GPU of the
    parent — which may be running kernels concurrently.

    TODO(havogt): change the contract of GPU backends to take the device
    architecture as an explicit argument of their compile steps (resolved once
    at backend construction, no detection inside the build). The architecture
    then travels to workers as part of the pickled executor and this
    ``CUDAARCHS`` export becomes obsolete.
    """
    _cache._session_cache_dir_path = pathlib.Path(shared_session_cache_dir)
    if cuda_archs is not None:
        os.environ["CUDAARCHS"] = cuda_archs
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _config_snapshot() -> dict[str, Any]:
    """Capture main's ``gt4py.next.config`` values to ship to a worker.

    ``spawn`` workers only inherit the OS environment at startup, so post-import
    changes to ``config.X`` are invisible without explicit shipment.
    """
    overrides: dict[str, Any] = {}
    for name, value in vars(config).items():
        if not name.isupper() or name.startswith("_"):
            continue
        if callable(value) or isinstance(value, type):
            continue
        overrides[name] = value
    return overrides


def _apply_config_overrides(overrides: dict[str, Any]) -> None:
    for name, value in overrides.items():
        setattr(config, name, value)


# Top-level (must be top-level for pickle).
def _run_compilation_task_in_worker(
    executor_blob: bytes,
    compilable: Any,
    config_overrides: dict[str, Any],
    recursion_limit: int,
) -> stages.CompilationArtifact:
    """Worker entry point: deserialize the executor and run it."""
    # `sys.setrecursionlimit` is per-process: a limit the parent raised for
    # deeply nested IR would silently reset to the interpreter default in a
    # `spawn` worker. Adopt the parent's limit, but never lower the worker's.
    sys.setrecursionlimit(max(recursion_limit, sys.getrecursionlimit()))
    _apply_config_overrides(config_overrides)
    executor = pickle.loads(executor_blob)
    return executor(compilable)


class ProcessRunner:
    """Compiles in a ``ProcessPoolExecutor`` (``spawn``).

    The worker runs the task's executor (post-lowering compile) and returns
    the picklable ``CompilationArtifact``.

    Tasks that cannot be offloaded — a known ``no_offload_reason`` or an executor
    that stdlib ``pickle`` cannot serialize — are compiled in the calling thread
    instead (with a warning), so they behave as under ``SerialRunner``.
    """

    def __init__(self, max_workers: int, shared_session_cache_dir: str) -> None:
        # spawn (not fork): the parent may already hold running threads (BLAS/OpenMP,
        # this pool's own manager thread), CUDA contexts and MPI state, none of which
        # survive fork safely. The gtfn GPU build even touches CUDA at *compile* time
        # (`build_systems/cmake.py` queries `cp.cuda.Device(0).compute_capability`
        # when `CUDAARCHS` is unset), so a forked worker with an inherited CUDA
        # context would fail right there. Fork also would not relax the picklability
        # requirement: ProcessPoolExecutor pickles every submitted task regardless
        # of start method.
        ctx = multiprocessing.get_context("spawn")
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_pool_worker_initializer,
            initargs=(shared_session_cache_dir, _detect_cuda_archs()),
        )

    def submit(
        self, task: CompilationTask
    ) -> concurrent.futures.Future[stages.CompilationArtifact]:
        reason = task.no_offload_reason
        executor_blob: bytes | None = None
        if reason is None:
            try:
                executor_blob = pickle.dumps(task.executor)
            except Exception as error:  # pickling arbitrary object graphs raises arbitrary errors
                reason = f"its executor is not picklable ({error!s})"
        if executor_blob is None:
            warnings.warn(
                f"Compiling '{task.name}' in the calling thread instead of a worker process "
                f"because {reason}.",
                stacklevel=2,
            )
            return _run_in_calling_thread(task)

        return self._pool.submit(
            _run_compilation_task_in_worker,
            executor_blob=executor_blob,
            compilable=task.construct_compilable(True),
            config_overrides=_config_snapshot(),
            recursion_limit=sys.getrecursionlimit(),
        )

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


def from_config() -> Runner:
    """Create a runner as configured by `config.BUILD_JOBS_MODE` and `config.BUILD_JOBS`."""
    mode = config.BUILD_JOBS_MODE
    if mode is config.BuildJobsMode.SERIAL or config.BUILD_JOBS <= 0:
        return SerialRunner()
    if mode is config.BuildJobsMode.THREAD:
        return ThreadRunner(max_workers=config.BUILD_JOBS)
    if mode is config.BuildJobsMode.PROCESS:
        return ProcessRunner(
            max_workers=config.BUILD_JOBS,
            shared_session_cache_dir=str(_cache._session_cache_dir_path),
        )
    raise ValueError(f"Unsupported BUILD_JOBS_MODE={mode!r}.")


_default_runner: Runner | None = None
_default_runner_lock = threading.Lock()


def _is_worker_process() -> bool:
    return multiprocessing.parent_process() is not None


def get_default_runner() -> Runner:
    """Return the process-wide runner, creating it from `gt4py.next.config` on first use.

    Created lazily so that merely importing gt4py never spins up multiprocessing
    machinery (which would leak into unrelated host processes, e.g. mypy loading
    the gt4py mypy plugin). In a worker process the default is always serial:
    nested worker pools must not spawn recursively.
    """
    global _default_runner
    with _default_runner_lock:
        if _default_runner is None:
            _default_runner = SerialRunner() if _is_worker_process() else from_config()
        return _default_runner


def reset_default_runner() -> None:
    """Shut down and discard the default runner, waiting for its ongoing jobs.

    The next `get_default_runner` call creates a fresh runner from the
    current configuration.
    """
    global _default_runner
    with _default_runner_lock:
        if _default_runner is not None:
            _default_runner.shutdown(wait=True)
            _default_runner = None


# Shut down worker pools while the interpreter is still intact; relying on
# garbage collection at teardown leaks the pool's semaphores (with a
# ``resource_tracker`` warning on stderr). Ordering: the temporary directories
# workers read from (session cache, connectivity dumps) are removed by
# ``weakref.finalize``, whose exit hook is registered no later than the import
# of `otf.compilation.cache` above — atexit runs LIFO, so this shutdown runs
# before those directories disappear.
atexit.register(reset_default_runner)
