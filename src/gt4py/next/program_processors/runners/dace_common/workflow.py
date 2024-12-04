# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ctypes
import dataclasses
from typing import Any

import dace
import factory
from dace.codegen.compiled_sdfg import _array_interface_ptr as get_array_interface_ptr

from gt4py._core import definitions as core_defs
from gt4py.next import common, config, utils as gtx_utils
from gt4py.next.otf import arguments, languages, stages, step_types, workflow
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.runners.dace_common import dace_backend, utility as dace_utils


class CompiledDaceProgram(stages.CompiledProgram):
    sdfg_program: dace.CompiledSDFG

    # Sorted list of SDFG arguments as they appear in program ABI and corresponding data type;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_arglist: list[tuple[str, dace.dtypes.Data]]

    def __init__(self, program: dace.CompiledSDFG):
        self.sdfg_program = program
        # `dace.CompiledSDFG.arglist()` returns an ordered dictionary that maps the argument
        # name to its data type, in the same order as arguments appear in the program ABI.
        # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
        self.sdfg_arglist = [
            (arg_name, arg_type) for arg_name, arg_type in program.sdfg.arglist().items()
        ]

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        result = self.sdfg_program(*args, **kwargs)
        assert result is None

    def fast_call(self) -> None:
        result = self.sdfg_program.fast_call(*self.sdfg_program._lastargs)
        assert result is None


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        CompiledDaceProgram,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        CompiledDaceProgram,
    ],
    step_types.CompilationStep[languages.SDFG, languages.LanguageSettings, languages.Python],
):
    """Use the dace build system to compile a GT4Py program to a ``gt4py.next.otf.stages.CompiledProgram``."""

    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(
        self,
        inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    ) -> CompiledDaceProgram:
        sdfg = dace.SDFG.from_json(inp.program_source.source_code)

        src_dir = cache.get_cache_folder(inp, self.cache_lifetime)
        sdfg.build_folder = src_dir / ".dacecache"

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "build_type", value=self.cmake_build_type.value)
            if self.device_type == core_defs.DeviceType.CPU:
                compiler_args = dace.config.Config.get("compiler", "cpu", "args")
                # disable finite-math-only in order to support isfinite/isinf/isnan builtins
                if "-ffast-math" in compiler_args:
                    compiler_args += " -fno-finite-math-only"
                if "-ffinite-math-only" in compiler_args:
                    compiler_args.replace("-ffinite-math-only", "")

                dace.config.Config.set("compiler", "cpu", "args", value=compiler_args)
            sdfg_program = sdfg.compile(validate=False)

        return CompiledDaceProgram(sdfg_program)


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler


def convert_args(
    inp: CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
    use_field_canonical_representation: bool = False,
) -> stages.CompiledProgram:
    sdfg_program = inp.sdfg_program
    sdfg = sdfg_program.sdfg
    on_gpu = True if device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM] else False

    def decorated_program(
        *args: Any,
        offset_provider: common.OffsetProvider,
        out: Any = None,
    ) -> None:
        if out is not None:
            args = (*args, out)
        if len(sdfg.arg_names) > len(args):
            args = (*args, *arguments.iter_size_args(args))

        if sdfg_program._lastargs:
            kwargs = dict(zip(sdfg.arg_names, gtx_utils.flatten_nested_tuple(args), strict=True))
            kwargs.update(dace_backend.get_sdfg_conn_args(sdfg, offset_provider, on_gpu))

            use_fast_call = True
            last_call_args = sdfg_program._lastargs[0]
            # The scalar arguments should be overridden with the new value; for field arguments,
            # the data pointer should remain the same otherwise fast_call cannot be used and
            # the arguments list has to be reconstructed.
            for i, (arg_name, arg_type) in enumerate(inp.sdfg_arglist):
                if isinstance(arg_type, dace.data.Array):
                    assert arg_name in kwargs, f"argument '{arg_name}' not found."
                    data_ptr = get_array_interface_ptr(kwargs[arg_name], arg_type.storage)
                    assert isinstance(last_call_args[i], ctypes.c_void_p)
                    if last_call_args[i].value != data_ptr:
                        use_fast_call = False
                        break
                else:
                    assert isinstance(arg_type, dace.data.Scalar)
                    assert isinstance(last_call_args[i], ctypes._SimpleCData)
                    if arg_name in kwargs:
                        # override the scalar value used in previous program call
                        actype = arg_type.dtype.as_ctypes()
                        last_call_args[i] = actype(kwargs[arg_name])
                    else:
                        # shape and strides of arrays are supposed not to change, and can therefore be omitted
                        assert dace_utils.is_field_symbol(
                            arg_name
                        ), f"argument '{arg_name}' not found."

            if use_fast_call:
                return inp.fast_call()

        sdfg_args = dace_backend.get_sdfg_args(
            sdfg,
            offset_provider,
            *args,
            check_args=False,
            on_gpu=on_gpu,
            use_field_canonical_representation=use_field_canonical_representation,
        )

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "allow_view_arguments", value=True)
            return inp(**sdfg_args)

    return decorated_program
