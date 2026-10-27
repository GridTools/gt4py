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
(connectivity tables travel as memory-mapped file references).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import tempfile
import typing
import weakref
from typing import Any, Callable

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import backend as gtx_backend, common, fingerprinting, utils as next_utils
from gt4py.next.embedded import nd_array_field
from gt4py.next.otf import arguments, artifacts, runners, stages
from gt4py.next.otf.compilation import cache as _cache


@dataclasses.dataclass(frozen=True, eq=False)
class CompileTimeConnectivity(nd_array_field.NumPyArrayConnectivityField):
    """
    Compile-time view of a connectivity: a read-only, C-contiguous host table,
    memory-mapped from a file named by the digest of its content.

    It is fingerprinted by its digest instead of by the table, and pickled as a
    reference to the file, so it crosses process boundaries through the page
    cache. Each runtime connectivity is converted once: tables must not be
    modified in place after they have been used for compilation.
    """

    _ndarray: core_defs.NDArrayObject = dataclasses.field(
        metadata=next_utils.gt4py_metadata(fingerprint=False)
    )
    _digest: str

    def __reduce__(self) -> tuple[Callable, tuple]:
        return (
            _connectivity_from_file,
            (
                str(_table_path(self._digest)),
                self._digest,
                self.domain,
                self.codomain,
                self.skip_value,
            ),
        )


def _table_path(digest: str) -> pathlib.Path:
    return _cache._session_cache_dir_path / "connectivities" / f"{digest}.npy"


def _connectivity_from_file(
    path: str,
    digest: str,
    domain: common.Domain,
    codomain: common.Dimension,
    skip_value: core_defs.IntegralScalar | None,
) -> CompileTimeConnectivity:
    table = typing.cast(core_defs.NDArrayObject, np.asarray(np.load(path, mmap_mode="r")))
    return CompileTimeConnectivity(domain, table, codomain, skip_value, digest)


def _write_table(table: np.ndarray, path: pathlib.Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and renamed, so a reader never maps a partially written file.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as f:
        np.save(f, table)
    os.replace(f.name, path)


#: Compile-time views of runtime connectivities, keyed by the `id` of the
#: runtime connectivity (validated against a weakref, since ids can be reused).
_compile_time_connectivities: dict[int, tuple[weakref.ref, CompileTimeConnectivity]] = {}


def _to_compile_time_connectivity(value: common.GatherConnectivity) -> CompileTimeConnectivity:
    if isinstance(value, CompileTimeConnectivity):
        return value
    entry = _compile_time_connectivities.get(id(value))
    if entry is not None and entry[0]() is value:
        return entry[1]
    table = np.ascontiguousarray(value.asnumpy())
    digest = fingerprinting.strict_fingerprinter(table)
    path = _table_path(digest)
    _write_table(table, path)
    result = _connectivity_from_file(
        str(path), digest, value.domain, value.codomain, value.skip_value
    )

    def _prune(ref: weakref.ref, key: int = id(value)) -> None:
        if (current := _compile_time_connectivities.get(key)) is not None and current[0] is ref:
            del _compile_time_connectivities[key]

    try:
        _compile_time_connectivities[id(value)] = (weakref.ref(value, _prune), result)
    except TypeError:  # not weakref-able: correct but converted per call
        pass
    return result


def compile_time_offset_provider(offset_provider: common.OffsetProvider) -> common.OffsetProvider:
    """
    Return the compile-time view of `offset_provider`.

    Every gather connectivity is replaced by its `CompileTimeConnectivity`, so
    that compilation depends neither on the device nor on the memory layout of
    the tables, and never touches device memory. Other entries and the order of
    the entries are kept.
    """
    return {
        name: typing.cast(common.OffsetProviderElem, _to_compile_time_connectivity(value))
        if isinstance(value, common.GatherConnectivity)
        else value
        for name, value in offset_provider.items()
    }


@dataclasses.dataclass(frozen=True)
class _PreloadedArtifact:
    """Wraps the already-loaded program of a backend with a customized ``compile``."""

    program: artifacts.ExecutableProgram

    def load(self) -> artifacts.ExecutableProgram:
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
        stages.ConcreteProgramDef(data=definition_stage, args=compile_time_args)
    )

    def construct_compilable(with_refs: bool) -> stages.CompilableProgramDef:
        if not with_refs or not compilable.args.offset_provider:
            return compilable
        # The shipped copy must not carry the connectivity buffers: they may
        # live on a device the worker cannot see. Compile-time connectivities
        # pickle as references to their memory-mapped files.
        return dataclasses.replace(
            compilable,
            args=dataclasses.replace(
                compilable.args,
                offset_provider=compile_time_offset_provider(compilable.args.offset_provider),
            ),
        )

    return runners.CompilationTask(
        name=name, construct_compilable=construct_compilable, executor=backend.executor
    )
