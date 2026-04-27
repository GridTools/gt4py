# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic build orchestration for backends that produce a Python extension module.

Wraps the lock + build-data tracking + builder-factory invocation + post-build
validation into a single :func:`run_build` call. Returns a :class:`BuildResult`
descriptor (paths + entry-point name) that backends wrap into their own
artifact type.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TypeVar

from gt4py._core import locking
from gt4py.next import config
from gt4py.next.otf import code_specs, stages
from gt4py.next.otf.compilation import build_data, build_system, cache


CodeSpecT = TypeVar("CodeSpecT", bound=code_specs.SourceCodeSpec)
TargetCodeSpecT = TypeVar("TargetCodeSpecT", bound=code_specs.SourceCodeSpec)


@dataclasses.dataclass(frozen=True)
class BuildResult:
    """On-disk descriptor of a successful build."""

    src_dir: pathlib.Path
    module: pathlib.Path
    entry_point_name: str


class CompilationError(RuntimeError): ...


def is_compiled(data: build_data.BuildData) -> bool:
    return data.status >= build_data.BuildStatus.COMPILED


def module_exists(data: build_data.BuildData, src_dir: pathlib.Path) -> bool:
    return (src_dir / data.module).exists()


def run_build(
    inp: stages.CompilableProject[CodeSpecT, TargetCodeSpecT],
    cache_lifetime: config.BuildCacheLifetime,
    builder_factory: build_system.BuildSystemProjectGenerator[CodeSpecT, TargetCodeSpecT],
    force_recompile: bool = False,
) -> BuildResult:
    """Drive ``builder_factory`` to produce a Python extension module on disk."""
    src_dir = cache.get_cache_folder(inp, cache_lifetime)

    # If we are compiling the same program at the same time (e.g. multiple MPI ranks),
    # we need to make sure that only one of them accesses the same build directory for compilation.
    with locking.lock(src_dir):
        data = build_data.read_data(src_dir)

        if not data or not is_compiled(data) or force_recompile:
            builder_factory(inp, cache_lifetime).build()

        new_data = build_data.read_data(src_dir)

        if not new_data or not is_compiled(new_data) or not module_exists(new_data, src_dir):
            raise CompilationError(
                f"On-the-fly compilation unsuccessful for '{inp.program_source.entry_point.name}'."
            )

    return BuildResult(
        src_dir=src_dir,
        module=new_data.module,
        entry_point_name=new_data.entry_point_name,
    )
