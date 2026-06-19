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
from gt4py.next import common, config, utils as gtx_utils
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
)


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


@dataclasses.dataclass(frozen=True)
class DaCeCompilationArtifact(gtx_utils.MetadataBasedPickling):
    """Result of a DaCe compilation: build folder + SDFG bindings + the SDFG itself.

    The SDFG is carried inline as JSON because dace's load path
    (:func:`get_program_handle`) needs an SDFG instance to wrap into the
    returned :class:`CompiledSDFG`, and the build folder may not contain a
    ``program.sdfg(z)`` dump under the upcoming minimal-build-dir mode.
    """

    build_folder: pathlib.Path
    sdfg_json: str
    binding_source_code: str
    bind_func_name: str
    device_type: core_defs.DeviceType

    # Process-local cache of the live :class:`CompiledDaceProgram`. Populated by
    # ``DaCeCompiler`` to skip the disk round-trip when the artifact stays in
    # the same process. Excluded from pickle (``pickle=False`` metadata) so
    # receivers in other processes see ``None`` and fall through to the
    # disk-based load.
    _live_program: CompiledDaceProgram | None = dataclasses.field(
        init=False,
        default=None,
        compare=False,
        repr=False,
        metadata=gtx_utils.gt4py_metadata(pickle=False),
    )

    def load(self) -> stages.ExecutableProgram:
        """Wrap the compiled program in gt4py's calling convention.

        On a miss, loads the precompiled .so directly via
        :func:`dace.codegen.compiler.get_program_handle` — no recompilation,
        no ``dace.config`` re-entry. Must run in the process that will call
        the returned program.
        """
        program = self._live_program
        if program is None:
            program = self._load_compiled_program()
            object.__setattr__(self, "_live_program", program)
        return gtx_wfddecoration.convert_args(program, device=self.device_type)

    def _load_compiled_program(self) -> CompiledDaceProgram:
        # TODO(phimuell): Drop ``sdfg_json`` from the artifact once dace
        #   exposes a load path that doesn't require an SDFG instance to wrap
        #   into the returned ``CompiledSDFG``.
        sdfg = dace.SDFG.from_json(json.loads(self.sdfg_json))
        folder_version = dace_compiler.get_folder_version(self.build_folder)
        library_path = dace_compiler.get_binary_name(
            self.build_folder, sdfg_name=sdfg.name, folder_version=folder_version
        )
        sdfg_program = dace_compiler.get_program_handle(library_path, sdfg)
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
    """Run the DaCe build system and produce an on-disk :class:`DaCeCompilationArtifact`."""

    bind_func_name: str
    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(
        self,
        inp: stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
    ) -> DaCeCompilationArtifact:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            sdfg_build_folder = pathlib.Path(gtx_cache.get_cache_folder(inp, self.cache_lifetime))
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)

            sdfg = dace.SDFG.from_json(inp.program_source.source_code)
            sdfg.build_folder = str(sdfg_build_folder)
            with locking.lock(sdfg_build_folder):
                # Keep the handle so the artifact's load() can skip the disk
                # round-trip in the same process.
                sdfg_program = sdfg.compile(validate=False)

        assert inp.binding_source is not None
        artifact = DaCeCompilationArtifact(
            build_folder=sdfg_build_folder,
            sdfg_json=json.dumps(inp.program_source.source_code),
            binding_source_code=inp.binding_source.source_code,
            bind_func_name=self.bind_func_name,
            device_type=self.device_type,
        )
        object.__setattr__(
            artifact,
            "_live_program",
            CompiledDaceProgram(
                sdfg_program, artifact.bind_func_name, artifact.binding_source_code
            ),
        )
        return artifact


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
