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
from typing import TypeVar

import factory

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import build_orchestrator, build_system, importer
from gt4py.next.program_processors.runners import gtfn_decoration


CPPLikeCodeSpecT = TypeVar("CPPLikeCodeSpecT", bound=code_specs.CPPLikeCodeSpec)


@dataclasses.dataclass(frozen=True)
class GTFNBuildArtifact:
    """On-disk result of a GTFN compilation: a Python extension module.

    Bindings are baked into the .so via nanobind, so :meth:`materialize` is
    just an ``importlib`` import + entry-point symbol lookup, plus a wrap in
    gt4py's calling convention.
    """

    src_dir: pathlib.Path
    module: pathlib.Path
    entry_point_name: str
    device_type: core_defs.DeviceType

    def materialize(self) -> stages.ExecutableProgram:
        """Import the module and wrap its entry point in gt4py's calling convention.

        Must run in the process that will call the returned program: the
        module is registered in that process's ``sys.modules`` under the
        ``gt4py.__compiled_programs__.`` prefix.
        """
        m = importer.import_from_path(
            self.src_dir / self.module,
            sys_modules_prefix="gt4py.__compiled_programs__.",
        )
        return gtfn_decoration.convert_args(
            getattr(m, self.entry_point_name), device=self.device_type
        )


@dataclasses.dataclass(frozen=True)
class Compiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        GTFNBuildArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        GTFNBuildArtifact,
    ],
    definitions.CompilationStep[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
):
    """Drive a build system and wrap the result in a :class:`GTFNBuildArtifact`."""

    cache_lifetime: config.BuildCacheLifetime
    builder_factory: build_system.BuildSystemProjectGenerator[
        CPPLikeCodeSpecT, code_specs.PythonCodeSpec
    ]
    device_type: core_defs.DeviceType
    force_recompile: bool = False

    def __call__(
        self,
        inp: stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
    ) -> GTFNBuildArtifact:
        result = build_orchestrator.run_build(
            inp, self.cache_lifetime, self.builder_factory, self.force_recompile
        )
        return GTFNBuildArtifact(
            src_dir=result.src_dir,
            module=result.module,
            entry_point_name=result.entry_point_name,
            device_type=self.device_type,
        )


class CompilerFactory(factory.Factory):
    class Meta:
        model = Compiler
