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

import dace
import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
)
from gt4py.next.program_processors.runners.dace.workflow.compiled_program import CompiledDaceProgram


@dataclasses.dataclass(frozen=True)
class DaCeBuildArtifact:
    """On-disk result of a DaCe compilation: a build folder + the SDFG bindings."""

    build_folder: pathlib.Path
    binding_source_code: str
    bind_func_name: str
    device_type: core_defs.DeviceType

    def materialize(self) -> stages.ExecutableProgram:
        """Re-deserialize the SDFG, link the .so, and wrap in gt4py's calling convention.

        Must run in the process that will call the returned program.
        """
        for dump_name in ("program.sdfgz", "program.sdfg"):
            sdfg_dump = self.build_folder / dump_name
            if sdfg_dump.exists():
                break
        else:
            raise RuntimeError(
                f"No SDFG dump (program.sdfgz / program.sdfg) found in '{self.build_folder}'."
            )

        sdfg = dace.SDFG.from_file(str(sdfg_dump))
        sdfg.build_folder = str(self.build_folder)

        with gtx_wfdcommon.dace_context(device_type=self.device_type):
            with dace.config.set_temporary("compiler", "use_cache", value=True):
                sdfg_program = sdfg.compile(validate=False)

        program = CompiledDaceProgram(sdfg_program, self.bind_func_name, self.binding_source_code)
        return gtx_wfddecoration.convert_args(program, device=self.device_type)


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        DaCeBuildArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        DaCeBuildArtifact,
    ],
    definitions.CompilationStep[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
):
    """Run the DaCe build system and produce an on-disk :class:`DaCeBuildArtifact`."""

    bind_func_name: str
    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(
        self,
        inp: stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
    ) -> DaCeBuildArtifact:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            sdfg_build_folder = gtx_cache.get_cache_folder(inp, self.cache_lifetime)
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)

            sdfg = dace.SDFG.from_json(inp.program_source.source_code)
            sdfg.build_folder = sdfg_build_folder
            with locking.lock(sdfg_build_folder):
                sdfg.compile(validate=False, return_program_handle=False)

        assert inp.binding_source is not None
        return DaCeBuildArtifact(
            build_folder=pathlib.Path(sdfg_build_folder),
            binding_source_code=inp.binding_source.source_code,
            bind_func_name=self.bind_func_name,
            device_type=self.device_type,
        )


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
