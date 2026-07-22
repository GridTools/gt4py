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
from typing import Any, Final

import dace
import dace.codegen.compiler as dace_compiler
import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config, fingerprinting
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
)


_COMPILE_COMPLETE_MARKER: Final = ".gt4py_compile_complete"


def _add_tx_markers(sdfg: dace.SDFG) -> None:
    has_gpu_schedule = any(
        getattr(node, "schedule", dace.dtypes.ScheduleType.Default) in dace.dtypes.GPU_SCHEDULES
        for node, _ in sdfg.all_nodes_recursive()
    )

    if has_gpu_schedule:
        sdfg.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS
        for node, _ in sdfg.all_nodes_recursive():
            # Also adds markers to map scopes that are NOT scheduled on GPU
            if isinstance(node, (dace.nodes.MapEntry, dace.sdfg.SDFGState)):
                node.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS


class CompiledDaceProgram:
    # TODO(phimuell): Update type.
    sdfg_program: dace.CompiledSDFG

    def __init__(
        self,
        program: dace.CompiledSDFG,
    ):
        self.sdfg_program = program

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        result = self.sdfg_program(*args, **kwargs)
        assert result is None


@dataclasses.dataclass(frozen=True)
class DaCeCompilationArtifact:
    """Result of a DaCe compilation: library path + SDFG bindings + the SDFG itself.

    The SDFG is carried inline as JSON because dace's load path
    (``get_program_handle``) needs an SDFG instance to wrap into the
    returned ``CompiledSDFG``, and the build folder may not contain a
    ``program.sdfg(z)`` dump under the upcoming minimal-build-dir mode.
    """

    sdfg_build_folder: pathlib.Path
    device_type: core_defs.DeviceType

    def load(self) -> stages.ExecutableProgram:
        sdfg_program = dace_compiler.load_precompiled_sdfg(self.sdfg_build_folder)
        program = CompiledDaceProgram(sdfg_program)
        return gtx_wfddecoration.convert_args(program, device=self.device_type)


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.ExtensionSource[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        DaCeCompilationArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.ExtensionSource[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        DaCeCompilationArtifact,
    ],
    definitions.CompilationStep[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
):
    """Run the DaCe build system and produce an on-disk ``DaCeCompilationArtifact``."""

    bind_func_name: str
    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType
    add_gpu_trace_markers: bool = dataclasses.field(
        default_factory=lambda: config.ADD_GPU_TRACE_MARKERS
    )
    cmake_build_type: config.CMakeBuildType = dataclasses.field(
        default_factory=lambda: config.CMAKE_BUILD_TYPE
    )
    # we store the non-default values of `dace.Config` in order to include it in the stage fingerprint
    dace_config_nondefaults: dict[str, Any] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            object.__setattr__(self, "dace_config_nondefaults", dace.Config._data.nondefaults())

    def __call__(
        self,
        inp: stages.ExtensionSource[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
    ) -> DaCeCompilationArtifact:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            sdfg = dace.SDFG.from_json(inp.program_source.source_code)

            # Fingerprint the non-default ``dace.Config`` so the SDFG rebuilds when the
            # user changes the backend configuration (PR #2650).
            sdfg_build_folder = gtx_cache.get_cache_folder(
                inp,
                self.cache_lifetime,
                build_context_id=fingerprinting.strict_fingerprinter(self.dace_config_nondefaults),
            )
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)
            sdfg.build_folder = sdfg_build_folder

            # Add TX markers to the generated GPU code for trace visualization tools.
            if self.add_gpu_trace_markers and self.device_type == core_defs.CUPY_DEVICE_TYPE:
                _add_tx_markers(sdfg)

            # NOT DELETE; RESTORE IT:
            library_path = dace_compiler.get_binary_name(
                object_folder=sdfg_build_folder, sdfg_name=sdfg.name
            )

            with locking.lock(sdfg_build_folder):
                # With `compiler.use_cache=True` dace reuses a cached library on mere
                # *existence*, without validating it; an interrupted build can leave a
                # truncated, unloadable library behind. The marker is written only
                # after a completed compile: no marker -> drop the stale library so
                # dace rebuilds it instead of handing it out.
                marker = sdfg_build_folder / _COMPILE_COMPLETE_MARKER
                if not marker.exists():
                    for stale in (
                        library_path,
                        *sdfg_build_folder.glob(f"libdacestub_{sdfg.name}.*"),
                    ):
                        stale.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                sdfg.compile(validate=False, return_program_handle=False)
                marker.touch()

        assert inp.binding_source is not None
        return DaCeCompilationArtifact(
            sdfg_build_folder=sdfg_build_folder, device_type=self.device_type
        )


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
