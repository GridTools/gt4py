# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Preparation of runner-ready compilation tasks from a backend and a program definition.

This is the layer between the compiled-programs pool and the compilation
runners: it decides what runs main-side (frontend lowering, since the raw
user function cannot cross a process boundary), whether the task can be
decomposed for offloading at all, and what may cross the process boundary
(connectivity tables travel as memory-mapped file references, a stopgap
until `CompileTimeArgs.offset_provider` becomes type-only).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import tempfile
import weakref
from typing import Any, Callable

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.next import backend as gtx_backend, common, constructors
from gt4py.next.otf import arguments, definitions as otf_definitions, runners, stages


def _connectivity_from_file(
    path: str,
    domain: common.Domain,
    codomain: common.Dimension,
    skip_value: core_defs.IntegralScalar | None,
) -> common.Connectivity:
    """Rehydrate a `_ConnectivityFileRef`: memory-mapped, so unpickling the same
    file in many worker processes shares the physical pages instead of copying."""
    data = np.load(path, mmap_mode="r")
    return constructors.as_connectivity(
        domain=domain, codomain=codomain, data=data, skip_value=skip_value
    )


@dataclasses.dataclass(frozen=True)
class _ConnectivityFileRef:
    """Lazy stand-in for a connectivity table in the compilation part of a load task.

    Construction is pure. On first pickling the table is dumped (memoized) to an
    ``.npy`` file in the session cache dir, and unpickling yields the memory-mapped
    `Connectivity`, so the table crosses the process boundary through the page
    cache instead of the pickle stream. A task that is never shipped to a worker
    never dumps anything.
    """

    connectivity: common.Connectivity

    def __reduce__(self) -> tuple[Callable, tuple]:
        return (
            _connectivity_from_file,
            (
                _dump_connectivity(self.connectivity),
                self.connectivity.domain,
                self.connectivity.codomain,
                self.connectivity.skip_value,
            ),
        )


#: Files already written for a connectivity, keyed by its id (validated against
#: a weakref, since ids can be reused). Main-process only, consulted when a
#: task's connectivity is first pickled: the same mesh object appears in the
#: tasks of many programs and variants and must be dumped only once per run.
_connectivity_files: dict[int, tuple[weakref.ref, str]] = {}

_dump_dir: tempfile.TemporaryDirectory | None = None


def _get_dump_dir() -> pathlib.Path:
    """Directory holding the connectivity shipping buffers; removed at process exit."""
    global _dump_dir
    if _dump_dir is None:
        _dump_dir = tempfile.TemporaryDirectory(prefix="gt4py_connectivities_")
    return pathlib.Path(_dump_dir.name)


def _dump_connectivity(value: common.Connectivity) -> str:
    entry = _connectivity_files.get(id(value))
    if entry is not None and entry[0]() is value:
        return entry[1]
    # One writer per path and files are write-once: `mkstemp` uniqueness is what
    # makes concurrent dumps and the workers' mmap readers safe. A racing double
    # dump costs a duplicate file (reclaimed with the session dir), nothing else.
    fd, path = tempfile.mkstemp(suffix=".npy", prefix="connectivity_", dir=_get_dump_dir())
    os.close(fd)
    np.save(path, value.asnumpy())

    def _prune(ref: weakref.ref, key: int = id(value)) -> None:
        # Only the registry entry: the file may still be referenced by
        # in-flight tasks and is reclaimed with the session cache dir. The
        # guard protects a reused id already re-registered by a new owner.
        if (current := _connectivity_files.get(key)) is not None and current[0] is ref:
            del _connectivity_files[key]

    try:
        _connectivity_files[id(value)] = (weakref.ref(value, _prune), path)
    except TypeError:  # not weakref-able: correct but re-dumped per task
        pass
    return path


def _offset_provider_with_file_refs(
    offset_provider: common.OffsetProvider,
) -> common.OffsetProvider:
    return {
        name: value
        if isinstance(value, common.Dimension)
        else xtyping.cast(common.OffsetProviderElem, _ConnectivityFileRef(value))
        for name, value in offset_provider.items()
    }


@dataclasses.dataclass(frozen=True)
class _PreloadedArtifact:
    """Wraps the already-loaded program of a backend with a customized ``compile``."""

    program: stages.ExecutableProgram

    def load(self) -> stages.ExecutableProgram:
        return self.program


def make_compilation_task(
    backend: gtx_backend.Backend,
    definition_stage: Any,
    compile_time_args: arguments.CompileTimeArgs,
) -> runners.CompilationTask:
    """Prepare the compilation of `definition_stage` with `backend` as a task for a runner."""
    name = getattr(backend, "name", type(backend).__name__)
    if getattr(type(backend), "compile", None) is not gtx_backend.Backend.compile:
        # A customized `compile` is opaque: it cannot be decomposed into the
        # standard transforms/executor workflow (and yields an already-loaded
        # program instead of an artifact), so the executor closes over
        # everything and ignores the compilable.
        return runners.CompilationTask(
            name=name,
            construct_compilable=lambda with_refs: None,
            executor=lambda _: _PreloadedArtifact(
                backend.compile(definition_stage, compile_time_args=compile_time_args)
            ),
            no_offload_reason="it does not use the standard compilation workflow"
            " (customized 'compile')",
        )
    # Frontend lowering happens here, main-side: decorators rebind the user's
    # function module attribute, so the raw `types.FunctionType` must not cross
    # a process boundary; the lowered `CompilableProgramDef` is pickle-safe.
    compilable = backend.transforms(
        otf_definitions.ConcreteProgramDef(data=definition_stage, args=compile_time_args)
    )

    def construct_compilable(with_refs: bool) -> otf_definitions.CompilableProgramDef:
        if not with_refs or not compilable.args.offset_provider:
            return compilable
        # The shipped copy must not carry the connectivity buffers: they may
        # live on a device the worker cannot see, and shipping them through the
        # pickle queue would serialize the full tables once per task.
        return dataclasses.replace(
            compilable,
            args=dataclasses.replace(
                compilable.args,
                offset_provider=_offset_provider_with_file_refs(compilable.args.offset_provider),
            ),
        )

    return runners.CompilationTask(
        name=name, construct_compilable=construct_compilable, executor=backend.executor
    )
