# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import warnings
from collections.abc import Callable, MutableSequence, Sequence
from typing import Any

import dace
import dace.codegen.compiler as dace_compiler
import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.next import common, config
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
)


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
    sdfg_program: dace.CompiledSDFG

    # Sorted list of SDFG arguments as they appear in program ABI and corresponding data type;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_argtypes: list[dace.dtypes.Data]

    # The compiled program contains a callable object to update the SDFG arguments list.
    update_sdfg_ctype_arglist: Callable[
        [
            core_defs.DeviceType,
            Sequence[dace.dtypes.Data],
            Sequence[Any],
            MutableSequence[Any],
            common.OffsetProvider,
        ],
        None,
    ]

    # Processed argument vectors that are passed to `CompiledSDFG.fast_call()`. `None`
    #  means that it has not been initialized, i.e. no call was ever performed.
    #  - csdfg_argv: Arguments used for calling the actual compiled SDFG, will be updated.
    #  - csdfg_init_argv: Arguments used for initialization; used only the first time and
    #       never updated.
    csdfg_argv: MutableSequence[Any] | None
    csdfg_init_argv: Sequence[Any] | None

    def __init__(
        self,
        program: dace.CompiledSDFG,
        bind_func_name: str,
        binding_source_code: str,
    ):
        self.sdfg_program = program

        # `dace.CompiledSDFG.arglist()` returns an ordered dictionary that maps the argument
        # name to its data type, in the same order as arguments appear in the program ABI.
        # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
        self.sdfg_argtypes = list(program.sdfg.arglist().values())

        # The binding source code is Python tailored to this specific SDFG.
        # We dynamically compile that function and add it to the compiled program.
        global_namespace: dict[str, Any] = {}
        exec(binding_source_code, global_namespace)
        self.update_sdfg_ctype_arglist = global_namespace[bind_func_name]
        # For debug purpose, we set a unique module name on the compiled function.
        self.update_sdfg_ctype_arglist.__module__ = os.path.basename(program.sdfg.build_folder)

        # Since the SDFG hasn't been called yet.
        self.csdfg_argv = None
        self.csdfg_init_argv = None

    def construct_arguments(self, **kwargs: Any) -> None:
        """
        This function will process the arguments and store the processed argument
        vectors in `self.csdfg_args`, to call them use `self.fast_call()`.
        """
        with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
            csdfg_argv, csdfg_init_argv = self.sdfg_program.construct_arguments(**kwargs)
        # Note we only care about `csdfg_argv` (normal call), since we have to update it,
        #  we ensure that it is a `list`.
        self.csdfg_argv = [*csdfg_argv]
        self.csdfg_init_argv = csdfg_init_argv

    def fast_call(self) -> None:
        """
        Perform a call to the compiled SDFG using the previously generated argument
        vectors, see `self.construct_arguments()`.
        """
        assert self.csdfg_argv is not None and self.csdfg_init_argv is not None, (
            "Argument vector was not set properly."
        )
        self.sdfg_program.fast_call(
            self.csdfg_argv, self.csdfg_init_argv, do_gpu_check=config.DEBUG
        )

    def __call__(self, **kwargs: Any) -> None:
        """Call the compiled SDFG with the given arguments.

        Note that this function will not update the argument vectors stored inside
        `self`. Furthermore, it is not recommended to use this function as it is
        very slow.
        """
        warnings.warn(
            "Called an SDFG through the standard DaCe interface is not recommended, use `fast_call()` instead.",
            stacklevel=1,
        )
        result = self.sdfg_program(**kwargs)
        assert result is None


# Share the live ``CompiledDaceProgram`` across all ``load()`` calls for the same
# build folder. ``CompiledDaceProgram.csdfg_argv`` is mutable state populated on
# first call and reused via ``fast_call()``; if a second ``Backend.compile()`` for
# the same program returns a fresh instance, that state is lost and ``fast_call``
# regresses to ``construct_arguments`` every call.
_live_program_cache: dict[pathlib.Path, CompiledDaceProgram] = {}


@dataclasses.dataclass(frozen=True)
class DaCeCompilationArtifact:
    """Result of a DaCe compilation: build folder + library path + SDFG bindings + the SDFG itself.

    The SDFG is carried inline as JSON because dace's load path
    (``get_program_handle``) needs an SDFG instance to wrap into the
    returned ``CompiledSDFG``, and the build folder may not contain a
    ``program.sdfg(z)`` dump under the upcoming minimal-build-dir mode.
    """

    build_folder: pathlib.Path
    library_path: pathlib.Path
    sdfg_json: str
    binding_source_code: str
    bind_func_name: str
    device_type: core_defs.DeviceType

    def load(self) -> stages.ExecutableProgram:
        program = _live_program_cache.get(self.build_folder)
        if program is None:
            program = self._load_compiled_program()
            _live_program_cache[self.build_folder] = program
        return gtx_wfddecoration.convert_args(program, device=self.device_type)

    def _load_compiled_program(self) -> CompiledDaceProgram:
        # TODO(phimuell): Drop ``sdfg_json`` from the artifact once dace
        #   exposes a load path that doesn't require an SDFG instance to wrap
        #   into the returned ``CompiledSDFG``.
        sdfg = dace.SDFG.from_json(json.loads(self.sdfg_json))
        sdfg_program = dace_compiler.get_program_handle(self.library_path, sdfg)
        return CompiledDaceProgram(sdfg_program, self.bind_func_name, self.binding_source_code)


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        DaCeCompilationArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
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

    def __call__(
        self,
        inp: stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
    ) -> DaCeCompilationArtifact:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            sdfg_build_folder = gtx_cache.get_cache_folder(inp, self.cache_lifetime)
            pathlib.Path(sdfg_build_folder).mkdir(parents=True, exist_ok=True)

            sdfg = dace.SDFG.from_json(inp.program_source.source_code)

            # Add TX markers to the generated GPU code for trace visualization tools.
            if self.add_gpu_trace_markers and self.device_type == core_defs.CUPY_DEVICE_TYPE:
                _add_tx_markers(sdfg)

            sdfg.build_folder = sdfg_build_folder
            with locking.lock(sdfg_build_folder):
                sdfg.compile(validate=False, return_program_handle=False)
            # ``build_folder_mode`` is set by ``dace_context``; resolve the library
            # path here so ``get_binary_name`` sees the same mode dace built under.
            library_path = dace_compiler.get_binary_name(
                object_folder=sdfg_build_folder, sdfg_name=sdfg.name
            )

        assert inp.binding_source is not None
        return DaCeCompilationArtifact(
            build_folder=pathlib.Path(sdfg_build_folder),
            library_path=library_path,
            sdfg_json=json.dumps(inp.program_source.source_code),
            binding_source_code=inp.binding_source.source_code,
            bind_func_name=self.bind_func_name,
            device_type=self.device_type,
        )


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
