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
from typing import Callable, Protocol, TypeVar

import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config, utils as gtx_utils
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import build_data, cache, importer


CodeSpecT = TypeVar("CodeSpecT", bound=code_specs.SourceCodeSpec)
TargetCodeSpecT = TypeVar("TargetCodeSpecT", bound=code_specs.SourceCodeSpec)
CPPLikeCodeSpecT = TypeVar("CPPLikeCodeSpecT", bound=code_specs.CPPLikeCodeSpec)


class BuildSystemProjectGenerator(Protocol[CodeSpecT, TargetCodeSpecT]):
    def __call__(
        self,
        source: stages.CompilableProject[CodeSpecT, TargetCodeSpecT],
        cache_lifetime: config.BuildCacheLifetime,
    ) -> stages.BuildSystemProject[CodeSpecT, TargetCodeSpecT]: ...


def is_compiled(data: build_data.BuildData) -> bool:
    return data.status >= build_data.BuildStatus.COMPILED


def module_exists(data: build_data.BuildData, src_dir: pathlib.Path) -> bool:
    return (src_dir / data.module).exists()


class CompilationError(RuntimeError): ...


# Signature of the per-backend wrapping applied to a freshly imported entry point.
ProgramDecorator = Callable[
    [stages.ExecutableProgram, core_defs.DeviceType], stages.ExecutableProgram
]


@dataclasses.dataclass(frozen=True)
class CPPBuildArtifact(gtx_utils.MetadataBasedPickling):
    """On-disk result of a CPP-style compilation: a Python extension module.

    Bindings are baked into the .so (e.g. via nanobind), so :meth:`materialize`
    is just an ``importlib`` import + entry-point lookup, plus a per-backend
    :attr:`decorator` that adapts the raw callable to the backend's calling
    convention.
    """

    src_dir: pathlib.Path
    module: pathlib.Path
    entry_point_name: str
    device_type: core_defs.DeviceType
    decorator: ProgramDecorator

    def materialize(self) -> stages.ExecutableProgram:
        """Import the module and apply the configured per-backend decorator.

        Must run in the process that will call the returned program: the
        module is registered in that process's ``sys.modules`` under the
        ``gt4py.__compiled_programs__.`` prefix.
        """
        m = importer.import_from_path(
            self.src_dir / self.module,
            sys_modules_prefix="gt4py.__compiled_programs__.",
        )
        return self.decorator(getattr(m, self.entry_point_name), self.device_type)


@dataclasses.dataclass(frozen=True)
class CPPCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        CPPBuildArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        CPPBuildArtifact,
    ],
    definitions.CompilationStep[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
):
    """Drive a CPP-style build system and wrap the result in a :class:`CPPBuildArtifact`."""

    cache_lifetime: config.BuildCacheLifetime
    builder_factory: BuildSystemProjectGenerator[CPPLikeCodeSpecT, code_specs.PythonCodeSpec]
    device_type: core_defs.DeviceType
    decorator: ProgramDecorator
    force_recompile: bool = False

    def __call__(
        self,
        inp: stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
    ) -> CPPBuildArtifact:
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

        return CPPBuildArtifact(
            src_dir=src_dir,
            module=new_data.module,
            entry_point_name=new_data.entry_point_name,
            device_type=self.device_type,
            decorator=self.decorator,
        )


class CompilerFactory(factory.Factory):
    class Meta:
        model = CPPCompiler
