# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Preparation of runner-ready compile jobs from a backend and a program definition.

This is the layer between the compiled-programs pool and the compilation
runners: it decides what runs main-side (frontend lowering, since the raw
user function cannot cross a process boundary), whether the job can be
decomposed for offloading at all, and what may cross the process boundary
(connectivity tables travel as memory-mapped file references, a stopgap
until `CompileTimeArgs.offset_provider` becomes type-only).
"""

from __future__ import annotations

import dataclasses
import functools
import os
import tempfile
import weakref
from typing import Any, Callable

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.next import backend as gtx_backend, common, constructors
from gt4py.next.otf import arguments, compilation_runner, definitions as otf_definitions
from gt4py.next.otf.compilation import cache as compilation_cache


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
    """Stand-in for a connectivity table in an offloaded compile job.

    Pickles as a reference to an ``.npy`` file in the session cache dir and
    unpickles as the memory-mapped `Connectivity` itself, so the table crosses
    the process boundary through the page cache instead of the pickle stream.
    """

    path: str
    domain: common.Domain
    codomain: common.Dimension
    skip_value: core_defs.IntegralScalar | None

    def __reduce__(self) -> tuple[Callable, tuple]:
        return (
            _connectivity_from_file,
            (self.path, self.domain, self.codomain, self.skip_value),
        )


#: Files already written for a connectivity, keyed by its id (validated against
#: a weakref, since ids can be reused): the same mesh is typically shared by all
#: programs of a run and must be dumped only once.
_connectivity_files: dict[int, tuple[weakref.ref, str]] = {}


def _connectivity_file_ref(value: common.Connectivity) -> _ConnectivityFileRef:
    entry = _connectivity_files.get(id(value))
    if entry is None or entry[0]() is not value:
        dump_dir = compilation_cache._session_cache_dir_path / "connectivities"
        dump_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=".npy", prefix="connectivity_", dir=dump_dir)
        os.close(fd)
        np.save(path, value.asnumpy())

        def _prune(ref: weakref.ref, key: int = id(value)) -> None:
            # Only the registry entry: the file may still be referenced by
            # in-flight jobs and is reclaimed with the session cache dir. The
            # guard protects a reused id already re-registered by a new owner.
            if (current := _connectivity_files.get(key)) is not None and current[0] is ref:
                del _connectivity_files[key]

        try:
            _connectivity_files[id(value)] = (weakref.ref(value, _prune), path)
        except TypeError:  # not weakref-able: correct but re-dumped per job
            pass
    else:
        path = entry[1]
    return _ConnectivityFileRef(
        path=path, domain=value.domain, codomain=value.codomain, skip_value=value.skip_value
    )


def _offset_provider_with_file_refs(
    offset_provider: common.OffsetProvider,
) -> common.OffsetProvider:
    return {
        name: value
        if isinstance(value, common.Dimension)
        else xtyping.cast(common.OffsetProviderElem, _connectivity_file_ref(value))
        for name, value in offset_provider.items()
    }


def make_compile_job(
    backend: gtx_backend.Backend,
    definition_stage: Any,
    compile_time_args: arguments.CompileTimeArgs,
) -> compilation_runner.CompileJob:
    """Prepare the compilation of `definition_stage` with `backend` as a job for a runner."""
    name = getattr(backend, "name", type(backend).__name__)
    if getattr(type(backend), "compile", None) is not gtx_backend.Backend.compile:
        # A customized `compile` is opaque: it cannot be decomposed into the
        # standard transforms/executor workflow, so the job can only run as-is.
        return compilation_runner.CompileJob(
            name=name,
            run=functools.partial(
                backend.compile, definition_stage, compile_time_args=compile_time_args
            ),
        )
    # Frontend lowering happens here, main-side: decorators rebind the user's
    # function module attribute, so the raw `types.FunctionType` must not cross
    # a process boundary; the lowered `CompilableProgramDef` is pickle-safe.
    compilable = backend.transforms(
        otf_definitions.ConcreteProgramDef(data=definition_stage, args=compile_time_args)
    )
    offload_compilable = compilable
    if compilable.args.offset_provider:
        # The offloaded copy must not carry the connectivity buffers: they may
        # live on a device the worker cannot see, and shipping them through the
        # pickle queue would serialize the full tables once per job.
        offload_compilable = dataclasses.replace(
            compilable,
            args=dataclasses.replace(
                compilable.args,
                offset_provider=_offset_provider_with_file_refs(compilable.args.offset_provider),
            ),
        )
    return compilation_runner.CompileJob(
        name=name,
        run=lambda: backend.executor(compilable).load(),
        offload=compilation_runner.OffloadableWork(
            compilable=offload_compilable, executor=backend.executor
        ),
    )
