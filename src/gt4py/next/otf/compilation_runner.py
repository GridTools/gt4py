# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import atexit
import concurrent.futures
import multiprocessing
import pickle
import threading
import warnings
from typing import Any, Protocol, runtime_checkable

from gt4py.next import backend as next_backend, config
from gt4py.next.otf import arguments, definitions, stages
from gt4py.next.otf.compilation import cache as _cache


@runtime_checkable
class CompilationRunner(Protocol):
    def submit(
        self,
        backend: next_backend.Backend,
        definition_stage: Any,
        compile_time_args: arguments.CompileTimeArgs,
    ) -> concurrent.futures.Future[stages.ExecutableProgram]:
        """Compile ``definition_stage`` with ``backend``.

        The returned future always yields a fully loaded ``ExecutableProgram``;
        the runner is responsible for any cross-process hydration.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Release the resources backing this runner (workers, threads)."""
        ...


def _compile_in_calling_thread(
    backend: next_backend.Backend,
    definition_stage: Any,
    compile_time_args: arguments.CompileTimeArgs,
) -> concurrent.futures.Future[stages.ExecutableProgram]:
    future: concurrent.futures.Future[stages.ExecutableProgram] = concurrent.futures.Future()
    try:
        future.set_result(backend.compile(definition_stage, compile_time_args=compile_time_args))
    except BaseException as exception:  # re-raised via the future
        future.set_exception(exception)
    return future


class SerialRunner:
    """Runs compilation in the calling thread; the returned future is already done."""

    def submit(
        self,
        backend: next_backend.Backend,
        definition_stage: Any,
        compile_time_args: arguments.CompileTimeArgs,
    ) -> concurrent.futures.Future[stages.ExecutableProgram]:
        return _compile_in_calling_thread(backend, definition_stage, compile_time_args)

    def shutdown(self, wait: bool = True) -> None:
        return None


class ThreadRunner:
    """Compiles in a ``ThreadPoolExecutor``.

    ``Backend.compile`` performs build and load in one step, so the future
    yields the loaded ``ExecutableProgram`` directly.
    """

    def __init__(self, max_workers: int) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        backend: next_backend.Backend,
        definition_stage: Any,
        compile_time_args: arguments.CompileTimeArgs,
    ) -> concurrent.futures.Future[stages.ExecutableProgram]:
        return self._pool.submit(
            backend.compile, definition_stage, compile_time_args=compile_time_args
        )

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


def _pool_worker_initializer(shared_session_cache_dir: str) -> None:
    """Point the worker's session-lifetime build cache at the main process's temp dir.

    Each worker would otherwise create its own ``TemporaryDirectory`` and scrub it
    on exit — taking the compiled artifacts with it before main can ``dlopen`` them.
    """
    import pathlib

    _cache._session_cache_dir_path = pathlib.Path(shared_session_cache_dir)


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
def _process_pool_compile_job(
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

    The worker runs ``backend.executor`` (post-translation compile) and returns a
    picklable ``CompilationArtifact``; the main process rehydrates it via
    ``artifact.load()`` (in a done-callback) so the returned future yields a live
    ``ExecutableProgram``.

    Backends that cannot be offloaded — a customized ``compile`` or an executor
    that stdlib ``pickle`` cannot serialize — are compiled in the calling thread
    instead (with a warning), so they behave as under ``SerialRunner``.
    """

    def __init__(self, max_workers: int, shared_session_cache_dir: str) -> None:
        # spawn (not fork): the parent may already hold running threads (BLAS/OpenMP,
        # this pool's own manager thread), CUDA contexts and MPI state, none of which
        # survive fork safely. Fork also would not relax the picklability requirement:
        # ProcessPoolExecutor pickles every submitted task regardless of start method.
        ctx = multiprocessing.get_context("spawn")
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_pool_worker_initializer,
            initargs=(shared_session_cache_dir,),
        )

    @staticmethod
    def _executor_blob_for_offload(backend: next_backend.Backend) -> tuple[bytes | None, str]:
        # The transforms/executor split in `submit` replicates `Backend.compile`;
        # it is only faithful if the backend does not customize `compile`.
        if getattr(type(backend), "compile", None) is not next_backend.Backend.compile:
            return None, "it customizes 'compile'"
        try:
            return pickle.dumps(backend.executor), ""
        except Exception as error:  # pickling arbitrary object graphs raises arbitrary errors
            return None, f"its executor is not picklable ({error!s})"

    def submit(
        self,
        backend: next_backend.Backend,
        definition_stage: Any,
        compile_time_args: arguments.CompileTimeArgs,
    ) -> concurrent.futures.Future[stages.ExecutableProgram]:
        executor_blob, reason = self._executor_blob_for_offload(backend)
        if executor_blob is None:
            warnings.warn(
                f"Compiling backend '{getattr(backend, 'name', type(backend).__name__)}' "
                f"in the calling thread instead of a worker process because {reason}.",
                stacklevel=2,
            )
            return _compile_in_calling_thread(backend, definition_stage, compile_time_args)

        # Frontend lowering stays main-side: decorators rebind the user's
        # function module attribute, so the raw ``types.FunctionType`` does not
        # cross the process boundary. The worker receives the lowered,
        # pickle-safe ``CompilableProgramDef`` and runs only ``backend.executor``.
        compilable = backend.transforms(
            definitions.ConcreteProgramDef(data=definition_stage, args=compile_time_args)
        )
        artifact_future = self._pool.submit(
            _process_pool_compile_job,
            executor_blob=executor_blob,
            compilable=compilable,
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


def from_config() -> CompilationRunner:
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


_default_runner: CompilationRunner | None = None
_default_runner_lock = threading.Lock()


def _is_worker_process() -> bool:
    return multiprocessing.parent_process() is not None


def get_default_runner() -> CompilationRunner:
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
    global _default_runner
    with _default_runner_lock:
        if _default_runner is not None:
            _default_runner.shutdown(wait=True)
            _default_runner = None


# Shut down worker pools while the interpreter is still intact; relying on
# garbage collection at teardown leaks the pool's semaphores (with a
# ``resource_tracker`` warning on stderr).
atexit.register(reset_default_runner)
