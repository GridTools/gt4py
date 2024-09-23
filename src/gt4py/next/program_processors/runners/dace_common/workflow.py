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
from typing import Any, Optional

import dace
import factory
from dace.codegen.compiled_sdfg import _array_interface_ptr as get_array_interface_ptr

from gt4py._core import definitions as core_defs
from gt4py.next import common, config
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.runners.dace_common import dace_backend, defs as dace_defs


class CompiledDaceProgram(stages.CompiledProgram):
    sdfg_program: dace.CompiledSDFG
    # Map SDFG positional arguments to the position in program ABI;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_arg_position: list[Optional[int]]
    # Map arguments for connectivity tables to the position in program ABI;
    # consider only the connectivity arrays, skip shape and stride symbols.
    sdfg_kwarg_position: dict[str, int]

    def __init__(self, program: dace.CompiledSDFG):
        # extract position of arguments in program ABI
        sdfg_arglist = program.sdfg.signature_arglist(with_types=False)
        sdfg_arg_pos_mapping = {param: pos for pos, param in enumerate(sdfg_arglist)}
        sdfg_used_symbols = program.sdfg.used_symbols(all_symbols=False)

        self.sdfg_arg_position = [
            sdfg_arg_pos_mapping[param]
            if param in program.sdfg.arrays or param in sdfg_used_symbols
            else None
            for param in program.sdfg.arg_names
        ]
        self.sdfg_kwarg_position = {
            param: pos
            for param, pos in sdfg_arg_pos_mapping.items()
            if param not in program.sdfg.arg_names and param in program.sdfg.arrays
        }
        assert all(
            param.startswith(dace_defs.CONNECTIVITY_PREFIX)
            for param in self.sdfg_kwarg_position.keys()
        )
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
    use_fast_call: bool = False,
    use_field_canonical_representation: bool = False,
) -> stages.CompiledProgram:
    sdfg_program = inp.sdfg_program
    sdfg = sdfg_program.sdfg
    on_gpu = True if device == core_defs.DeviceType.CUDA else False

    def decorated_program(
        *args: Any, offset_provider: dict[str, common.Connectivity | common.Dimension]
    ) -> Any:
        if sdfg_program._lastargs and use_fast_call:
            # The scalar arguments should be overridden with the new value;
            # for field arguments, DaCe fast_call will reuse the same memory buffer.
            for arg, param, pos in zip(args, sdfg.arg_names, inp.sdfg_arg_position, strict=True):
                if dace.dtypes.is_array(arg):
                    desc = sdfg.arrays[param]
                    assert isinstance(desc, dace.data.Array)
                    assert isinstance(sdfg_program._lastargs[0][pos], ctypes.c_void_p)
                    data_ptr = get_array_interface_ptr(arg, desc.storage)
                    if sdfg_program._lastargs[0][pos].value != data_ptr:
                        raise RuntimeError(f"Invalid memory buffer for argument {param}.")

                elif param in sdfg.arrays:
                    desc = sdfg.arrays[param]
                    assert isinstance(desc, dace.data.Scalar)
                    sdfg_program._lastargs[0][pos] = _get_ctype_value(arg, desc.dtype)

                elif pos:
                    sym_dtype = sdfg.symbols[param]
                    sdfg_program._lastargs[0][pos] = _get_ctype_value(arg, sym_dtype)

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
