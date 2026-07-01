# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import concurrent.futures
import multiprocessing
import pickle
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


class SerialRunner:
    """Runs compilation in the calling thread; the returned future is already done."""

    def submit(
        self,
        backend: next_backend.Backend,
        definition_stage: Any,
        compile_time_args: arguments.CompileTimeArgs,
    ) -> concurrent.futures.Future[stages.ExecutableProgram]:
        future: concurrent.futures.Future[stages.ExecutableProgram] = (
            concurrent.futures.Future()
        )
        try:
            future.set_result(
                backend.compile(definition_stage, compile_time_args=compile_time_args)
            )
        except BaseException as exception:  # noqa: BLE001 - re-raised via the future
            future.set_exception(exception)
        return future

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
    """

    def __init__(self, max_workers: int, shared_session_cache_dir: str) -> None:
        # spawn (not fork): fork-after-threads / fork-with-CUDA is unsafe.
        ctx = multiprocessing.get_context("spawn")
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_pool_worker_initializer,
            initargs=(shared_session_cache_dir,),
        )

    def submit(
        self,
        backend: next_backend.Backend,
        definition_stage: Any,
        compile_time_args: arguments.CompileTimeArgs,
    ) -> concurrent.futures.Future[stages.ExecutableProgram]:
        # Frontend lowering stays main-side: decorators rebind the user's
        # function module attribute, so the raw ``types.FunctionType`` does not
        # cross the process boundary. The worker receives the lowered,
        # pickle-safe ``CompilableProgramDef`` and runs only ``backend.executor``.
        compilable = backend.transforms(
            definitions.ConcreteProgramDef(data=definition_stage, args=compile_time_args)
        )
        executor_blob = pickle.dumps(backend.executor)
        artifact_future = self._pool.submit(
            _process_pool_compile_job,
            executor_blob,
            compilable,
            _config_snapshot(),
        )
        loaded: concurrent.futures.Future[stages.ExecutableProgram] = (
            concurrent.futures.Future()
        )

        def _load(
            artifact_future: concurrent.futures.Future[stages.CompilationArtifact],
        ) -> None:
            try:
                loaded.set_result(artifact_future.result().load())
            except BaseException as exception:  # noqa: BLE001 - re-raised via the future
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


def _is_worker_process() -> bool:
    return multiprocessing.parent_process() is not None


def get_default_runner() -> CompilationRunner:
    global _default_runner
    if _default_runner is None:
        _default_runner = from_config()
    return _default_runner


def reset_default_runner() -> None:
    global _default_runner
    if _default_runner is not None:
        _default_runner.shutdown(wait=True)
        _default_runner = None


# Workers re-import this module on spawn; don't let them spin up their own pool.
if not _is_worker_process():
    get_default_runner()
