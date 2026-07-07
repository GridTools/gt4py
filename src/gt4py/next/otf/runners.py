# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Runners executing prepared load tasks serially, in threads, or in worker processes."""

from __future__ import annotations

import atexit
import concurrent.futures
import dataclasses
import multiprocessing
import os
import pathlib
import pickle
import threading
import warnings
from typing import Any, Callable, Protocol, runtime_checkable

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import definitions, stages, workflow
from gt4py.next.otf.compilation import cache as _cache, common as compilation_common


@dataclasses.dataclass(frozen=True)
class CompilationTask:
    """Decomposed compilation part of a `LoadTask`, prepared main-side."""

    #: The already-lowered program; pickle-safe by construction.
    compilable: definitions.CompilableProgramDef
    #: The backend's artifact-producing step. A runner may execute it anywhere,
    #: including another process; its picklability is the runner's concern.
    executor: workflow.Workflow[definitions.CompilableProgramDef, stages.CompilationArtifact]


@dataclasses.dataclass(frozen=True)
class LoadTask:
    """A task yielding a loaded `ExecutableProgram`, fully prepared by the caller.

    Completing a `LoadTask` yields a loaded `ExecutableProgram`; where the load
    executes is the runner's concern.
    """

    #: Label used in user-facing messages.
    name: str
    #: The whole pipeline executed in the current thread; always valid.
    compile_and_load: Callable[[], stages.ExecutableProgram]
    #: The decomposed compilation part, for runners that can execute it in
    #: another process; ``None`` when the backend customizes ``compile`` and
    #: the task can only run as-is.
    compilation: CompilationTask | None = None


@runtime_checkable
class Runner(Protocol):
    def submit(self, task: LoadTask) -> concurrent.futures.Future[stages.ExecutableProgram]:
        """Schedule `task`.

        Returns:
            A future yielding the fully loaded ``ExecutableProgram``; the
            runner is responsible for any cross-process hydration.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Release the resources backing this runner (workers, threads)."""
        ...


def _run_in_calling_thread(
    task: LoadTask,
) -> concurrent.futures.Future[stages.ExecutableProgram]:
    future: concurrent.futures.Future[stages.ExecutableProgram] = concurrent.futures.Future()
    try:
        future.set_result(task.compile_and_load())
    except BaseException as exception:  # re-raised via the future
        future.set_exception(exception)
    return future


class SerialRunner:
    """Runs compilation in the calling thread; the returned future is already done."""

    def submit(self, task: LoadTask) -> concurrent.futures.Future[stages.ExecutableProgram]:
        return _run_in_calling_thread(task)

    def shutdown(self, wait: bool = True) -> None:
        return None


class ThreadRunner:
    """Compiles in a ``ThreadPoolExecutor``."""

    def __init__(self, max_workers: int) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, task: LoadTask) -> concurrent.futures.Future[stages.ExecutableProgram]:
        return self._pool.submit(task.compile_and_load)

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
) -> stages.CompilationArtifact:
    """Worker entry point: deserialize the executor and run it."""
    _apply_config_overrides(config_overrides)
    executor = pickle.loads(executor_blob)
    return executor(compilable)


class ProcessRunner:
    """Compiles in a ``ProcessPoolExecutor`` (``spawn``).

    The worker runs the task's executor (post-lowering compile) and returns a
    picklable ``CompilationArtifact``; the main process rehydrates it via
    ``artifact.load()`` (in a done-callback) so the returned future yields a live
    ``ExecutableProgram``.

    Tasks that cannot be offloaded — no decomposed compilation or an executor
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

    def submit(self, task: LoadTask) -> concurrent.futures.Future[stages.ExecutableProgram]:
        executor_blob: bytes | None = None
        if task.compilation is None:
            blocker = "it does not use the standard compilation workflow (customized 'compile')"
        else:
            try:
                executor_blob = pickle.dumps(task.compilation.executor)
            except Exception as error:  # pickling arbitrary object graphs raises arbitrary errors
                blocker = f"its executor is not picklable ({error!s})"
        if executor_blob is None:
            warnings.warn(
                f"Compiling '{task.name}' in the calling thread instead of a worker process "
                f"because {blocker}.",
                stacklevel=2,
            )
            return _run_in_calling_thread(task)

        assert task.compilation is not None
        artifact_future = self._pool.submit(
            _run_compilation_task_in_worker,
            executor_blob=executor_blob,
            compilable=task.compilation.compilable,
            config_overrides=_config_snapshot(),
        )
        loaded: concurrent.futures.Future[stages.ExecutableProgram] = concurrent.futures.Future()

        def _load(
            artifact_future: concurrent.futures.Future[stages.CompilationArtifact],
        ) -> None:
            try:
                loaded.set_result(artifact_future.result().load())
            except BaseException as exception:  # re-raised via the future
                loaded.set_exception(exception)

        artifact_future.add_done_callback(_load)
        return loaded

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
# ``resource_tracker`` warning on stderr).
atexit.register(reset_default_runner)
