# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import pathlib
from typing import Protocol, TypeGuard, TypeVar

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config, fingerprinting
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import build_data, cache, importer


def is_compiled(data: build_data.BuildData) -> bool:
    return data.status >= build_data.BuildStatus.COMPILED


def module_exists(data: build_data.BuildData, src_dir: pathlib.Path) -> bool:
    return (src_dir / data.module).exists()


def is_usable(
    data: build_data.BuildData | None, src_dir: pathlib.Path
) -> TypeGuard[build_data.BuildData]:
    """Check that a cached build is marked compiled *and* its module artifact is present.

    An interrupted run (or external cleanup) can leave a ``COMPILED`` marker
    without the artifact; such a build folder must be rebuilt, not reused.
    """
    return data is not None and is_compiled(data) and module_exists(data, src_dir)


CodeSpecT = TypeVar("CodeSpecT", bound=code_specs.SourceCodeSpec)
TargetCodeSpecT = TypeVar("TargetCodeSpecT", bound=code_specs.SourceCodeSpec)
CPPLikeCodeSpecT = TypeVar("CPPLikeCodeSpecT", bound=code_specs.CPPLikeCodeSpec)


class BuildSystemProjectGenerator(Protocol[CodeSpecT, TargetCodeSpecT]):
    def __call__(
        self,
        source: stages.ExtensionSource[CodeSpecT, TargetCodeSpecT],
        cache_lifetime: config.BuildCacheLifetime,
    ) -> stages.BuildSystemProject[CodeSpecT, TargetCodeSpecT]: ...


@dataclasses.dataclass(frozen=True)
class CPPCompilationArtifact:
    """On-disk result of a CPP-style compilation: a Python extension module.

    The default ``load`` is an ``importlib`` import + entry-point lookup;
    backends override to apply their own calling convention.
    """

    src_dir: pathlib.Path
    module: pathlib.Path
    entry_point_name: str
    device_type: core_defs.DeviceType

    def load(self) -> stages.ExecutableProgram:
        """Import the .so and return the raw entry point.

        Must run in the process that will call the returned program: the
        module is registered in that process's ``sys.modules`` under the
        ``gt4py.__compiled_programs__.`` prefix.
        """
        m = importer.import_from_path(
            self.src_dir / self.module,
            sys_modules_prefix="gt4py.__compiled_programs__.",
        )
        return getattr(m, self.entry_point_name)


@dataclasses.dataclass(frozen=True)
class CPPCompiler(
    workflow.ChainableWorkflowMixin[
        stages.ExtensionSource[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        CPPCompilationArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.ExtensionSource[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        CPPCompilationArtifact,
    ],
    definitions.CompilationStep[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
):
    """Drive a CPP-style build system into a ``CPPCompilationArtifact``.

    Backends override ``_make_artifact`` to use their own artifact subclass.
    """

    cache_lifetime: config.BuildCacheLifetime
    builder_factory: BuildSystemProjectGenerator[CPPLikeCodeSpecT, code_specs.PythonCodeSpec]
    device_type: core_defs.DeviceType
    fingerprint_builder_factory: bool = True
    force_recompile: bool = False

    def __call__(
        self,
        inp: stages.ExtensionSource[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
    ) -> CPPCompilationArtifact:
        build_context_id = (
            fingerprinting.strict_fingerprinter(self.builder_factory)
            if self.fingerprint_builder_factory
            else ""
        )
        src_dir = cache.get_cache_folder(inp, self.cache_lifetime, build_context_id)

        # If we are compiling the same program at the same time (e.g. multiple MPI ranks),
        # we need to make sure that only one of them accesses the same build directory for compilation.
        with locking.lock(src_dir):
            if self.force_recompile or not is_usable(build_data.read_data(src_dir), src_dir):
                self.builder_factory(inp, self.cache_lifetime).build()

            new_data = build_data.read_data(src_dir)

            if not is_usable(new_data, src_dir):
                raise CompilationError(
                    f"On-the-fly compilation unsuccessful for '{inp.program_source.entry_point.name}'."
                )

        return self._make_artifact(src_dir, new_data.module, new_data.entry_point_name)

    def _make_artifact(
        self, src_dir: pathlib.Path, module: pathlib.Path, entry_point_name: str
    ) -> CPPCompilationArtifact:
        return CPPCompilationArtifact(
            src_dir=src_dir,
            module=module,
            entry_point_name=entry_point_name,
            device_type=self.device_type,
        )


class CompilationError(RuntimeError): ...
