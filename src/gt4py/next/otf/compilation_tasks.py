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
decomposed for offloading at all, and what compilation sees of the offset
provider (compile-time views of the connectivity tables, which travel as
memory-mapped file references).
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
import xxhash

from gt4py._core import definitions as core_defs
from gt4py.next import backend as gtx_backend, common, utils as next_utils
from gt4py.next.embedded import nd_array_field
from gt4py.next.otf import arguments, artifacts, runners, stages
from gt4py.next.otf.compilation import cache as _cache


def _table_digest(table: np.ndarray) -> str:
    """Digest of the content of a C-contiguous `table`, hashed in place."""
    hasher = xxhash.xxh3_128(b"table\0")
    hasher.update(table.dtype.str.encode())
    hasher.update(repr(table.shape).encode())
    hasher.update(table.data)
    return hasher.hexdigest()


@dataclasses.dataclass(frozen=True, eq=False)
class _CompileTimeConnectivity(nd_array_field.NumPyArrayConnectivityField):
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
    _digest: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        table = np.ascontiguousarray(np.asarray(self._ndarray))
        object.__setattr__(self, "_digest", _table_digest(table))

    @classmethod
    def _from_file(
        cls,
        path: str,
        digest: str,
        domain: common.Domain,
        codomain: common.Dimension,
        skip_value: core_defs.IntegralScalar | None,
        shape: tuple[int, ...],
        dtype: str,
    ) -> _CompileTimeConnectivity:
        """Map the table file written for `digest`, without hashing the table again."""
        table = np.asarray(np.load(path, mmap_mode="r"))
        if table.shape != shape or table.dtype.str != dtype:
            raise ValueError(
                f"Connectivity table file '{path}' holds a '{table.dtype.str}' table of shape"
                f" {table.shape}, expected '{dtype}' and {shape}."
            )
        result = object.__new__(cls)
        for name, value in (
            ("_domain", domain),
            ("_ndarray", table),
            ("_codomain", codomain),
            ("_skip_value", skip_value),
            ("_digest", digest),
        ):
            object.__setattr__(result, name, value)
        return result

    def __reduce__(self) -> tuple[Callable, tuple]:
        table = np.asarray(self._ndarray)
        path = _table_path(self._digest)
        _write_table(table, path)
        return (
            _connectivity_from_file,
            (
                str(path),
                self._digest,
                self.domain,
                self.codomain,
                self.skip_value,
                table.shape,
                table.dtype.str,
            ),
        )


def _connectivity_from_file(*args: Any) -> _CompileTimeConnectivity:
    return _CompileTimeConnectivity._from_file(*args)


def _table_path(digest: str) -> pathlib.Path:
    return _cache._session_cache_dir_path / "connectivities" / f"{digest}.npy"


def _write_table(table: np.ndarray, path: pathlib.Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and renamed, so a reader never maps a partially written file.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as f:
        temporary = f.name
        try:
            np.save(f, np.ascontiguousarray(table))
        except BaseException:
            f.close()
            os.unlink(temporary)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


#: Compile-time views of runtime connectivities, keyed by the `id` of the
#: runtime connectivity (validated against a weakref, since ids can be reused).
_compile_time_connectivities: dict[int, tuple[weakref.ref, _CompileTimeConnectivity]] = {}


def _to_compile_time_connectivity(value: common.GatherConnectivity) -> _CompileTimeConnectivity:
    if isinstance(value, _CompileTimeConnectivity):
        return value
    entry = _compile_time_connectivities.get(id(value))
    if entry is not None and entry[0]() is value:
        return entry[1]
    table = np.ascontiguousarray(value.asnumpy())
    digest = _table_digest(table)
    path = _table_path(digest)
    _write_table(table, path)
    result = _CompileTimeConnectivity._from_file(
        str(path),
        digest,
        value.domain,
        value.codomain,
        value.skip_value,
        table.shape,
        table.dtype.str,
    )
    del table

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

    Every gather connectivity is replaced by a read-only, C-contiguous host
    copy of it, memory-mapped from a file named by the digest of its content,
    so that compilation depends neither on the device nor on the memory layout
    of the tables, and does not read device memory after the conversion. Other
    entries and the order of the entries are kept.
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
            compilable=None,
            executor=lambda _: _PreloadedArtifact(
                backend.compile(definition_stage, compile_time_args=compile_time_args)
            ),
            no_offload_reason="it does not use the standard compilation workflow"
            " (customized 'compile')",
        )
    # Compilation runs against the compile-time view of the offset provider,
    # whichever runner executes it.
    if compile_time_args.offset_provider:
        compile_time_args = dataclasses.replace(
            compile_time_args,
            offset_provider=compile_time_offset_provider(compile_time_args.offset_provider),
        )
    # Frontend lowering happens here, main-side: decorators rebind the user's
    # function module attribute, so the raw `types.FunctionType` must not cross
    # a process boundary; the lowered `CompilableProgramDef` is pickle-safe.
    compilable = backend.transforms(
        stages.ConcreteProgramDef(data=definition_stage, args=compile_time_args)
    )
    return runners.CompilationTask(name=name, compilable=compilable, executor=backend.executor)
