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
from typing import Protocol, TypeVar

import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config, utils as gtx_utils
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import build_data, cache, importer


def is_compiled(data: build_data.BuildData) -> bool:
    return data.status >= build_data.BuildStatus.COMPILED


def module_exists(data: build_data.BuildData, src_dir: pathlib.Path) -> bool:
    return (src_dir / data.module).exists()


CodeSpecT = TypeVar("CodeSpecT", bound=code_specs.SourceCodeSpec)
TargetCodeSpecT = TypeVar("TargetCodeSpecT", bound=code_specs.SourceCodeSpec)
CPPLikeCodeSpecT = TypeVar("CPPLikeCodeSpecT", bound=code_specs.CPPLikeCodeSpec)


class BuildSystemProjectGenerator(Protocol[CodeSpecT, TargetCodeSpecT]):
    def __call__(
        self,
        source: stages.CompilableProject[CodeSpecT, TargetCodeSpecT],
        cache_lifetime: config.BuildCacheLifetime,
    ) -> stages.BuildSystemProject[CodeSpecT, TargetCodeSpecT]: ...


@dataclasses.dataclass(frozen=True)
class CPPCompilationArtifact(gtx_utils.MetadataBasedPickling):
    """On-disk result of a CPP-style compilation: a Python extension module.

    Bindings are baked into the .so (e.g. via nanobind), so the default
    :meth:`load` is just an ``importlib`` import + entry-point lookup,
    returning the raw imported callable. Backends that need to wrap the
    callable in a calling convention (e.g. GTFN's gt4py-shaped argument
    conversion) subclass and override :meth:`load`.
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
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        CPPCompilationArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        CPPCompilationArtifact,
    ],
    definitions.CompilationStep[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
):
    """Drive a CPP-style build system and wrap the result in a :class:`CPPCompilationArtifact`.

    Backends that need a different artifact subclass (e.g. with a wrapped
    ``load``) subclass and override :meth:`_make_artifact`.
    """

    cache_lifetime: config.BuildCacheLifetime
    builder_factory: BuildSystemProjectGenerator[CPPLikeCodeSpecT, code_specs.PythonCodeSpec]
    device_type: core_defs.DeviceType
    force_recompile: bool = False

    def __call__(
        self,
        inp: stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
    ) -> CPPCompilationArtifact:
        src_dir = cache.get_cache_folder(inp, self.cache_lifetime)

        # If we are compiling the same program at the same time (e.g. multiple MPI ranks),
        # we need to make sure that only one of them accesses the same build directory for compilation.
        with locking.lock(src_dir):
            data = build_data.read_data(src_dir)

            if not data or not is_compiled(data) or self.force_recompile:
                self.builder_factory(inp, self.cache_lifetime).build()

            new_data = build_data.read_data(src_dir)

            if not new_data or not is_compiled(new_data) or not module_exists(new_data, src_dir):
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


class CompilerFactory(factory.Factory):
    class Meta:
        model = CPPCompiler


class CompilationError(RuntimeError): ...
