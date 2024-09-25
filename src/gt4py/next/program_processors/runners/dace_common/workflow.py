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
import re
from typing import Any, Optional

import dace
import factory
from dace.codegen.compiled_sdfg import _array_interface_ptr as get_array_interface_ptr

from gt4py._core import definitions as core_defs
from gt4py.next import common, config
from gt4py.next.otf import arguments, languages, stages, step_types, workflow
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.runners.dace_common import dace_backend


class CompiledDaceProgram(stages.CompiledProgram):
    sdfg_program: dace.CompiledSDFG
    # Map SDFG positional arguments to the position in program ABI;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_arg_position: list[Optional[int]]
    # Map arguments for connectivity tables to the position in program ABI;
    # consider only the connectivity arrays, skip shape and stride symbols.
    sdfg_conn_position: dict[str, int]

    def __init__(self, program: dace.CompiledSDFG):
        # method `signature_arglist` returns an ordered dictionary
        self.sdfg_arglist = [
            (i, arg_name, arg_type)
            for i, (arg_name, arg_type) in enumerate(program.sdfg.arglist().items())
        ]
        self.sdfg_program = program

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.sdfg_program(*args, **kwargs)


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


def _get_ctype_value(arg: Any, dtype: dace.dtypes.dataclass) -> Any:
    if not isinstance(arg, (ctypes._SimpleCData, ctypes._Pointer)):
        actype = dtype.as_ctypes()
        return actype(arg)
    return arg


def convert_args(
    inp: CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
    use_field_canonical_representation: bool = False,
) -> stages.CompiledProgram:
    sdfg_program = inp.sdfg_program
    sdfg = sdfg_program.sdfg
    on_gpu = True if device == core_defs.DeviceType.CUDA else False

    def decorated_program(
        *args: Any,
        offset_provider: dict[str, common.Connectivity | common.Dimension],
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)
        if len(sdfg.arg_names) > len(args):
            args = (*args, *arguments.iter_size_args(args))

        if sdfg_program._lastargs:
            kwargs = dict(zip(sdfg.arg_names, args, strict=True))
            kwargs.update(dace_backend.get_sdfg_conn_args(sdfg, offset_provider, on_gpu))

            use_fast_call = True
            last_call_args = sdfg_program._lastargs[0]
            # The scalar arguments should be overridden with the new value; for field arguments,
            # the data pointer should remain the same otherwise fast-call cannot be used and
            # the arguments list has to be reconstructed.
            for i, arg_name, arg_type in inp.sdfg_arglist:
                if isinstance(arg_type, dace.data.Array):
                    assert arg_name in kwargs, f"Array argument '{arg_name}' was not passed."
                    data_ptr = get_array_interface_ptr(kwargs[arg_name], arg_type.storage)
                    assert isinstance(last_call_args[i], ctypes.c_void_p)
                    if last_call_args[i].value != data_ptr:
                        use_fast_call = False
                        break
                elif isinstance(arg_type, dace.data.Scalar):
                    if arg_name in kwargs:
                        last_call_args[i] = _get_ctype_value(kwargs[arg_name], arg_type.dtype)
                    else:
                        # shape and strides of arrays are supposed not to change, and can therefore be omitted
                        assert re.match(
                            "__.+_(size|stride)_\d+", arg_name
                        ), f"Scalar argument '{arg_name}' was not passed."
                else:
                    raise ValueError(f"Unsupported data type {arg_type}")

            if use_fast_call:
                return sdfg_program.fast_call(*sdfg_program._lastargs)

        sdfg_args = dace_backend.get_sdfg_args(
            sdfg,
            *args,
            check_args=False,
            offset_provider=offset_provider,
            on_gpu=on_gpu,
            use_field_canonical_representation=use_field_canonical_representation,
        )

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "allow_view_arguments", value=True)
            return inp(**sdfg_args)

    return decorated_program
