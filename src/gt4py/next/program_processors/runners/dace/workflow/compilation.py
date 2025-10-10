# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Sequence

import dace
import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon


def _get_sdfg_ctype_arglist_callback(
    module_name: str, bind_func_name: str, python_code: str
) -> Callable[
    [core_defs.DeviceType, Sequence[dace.dtypes.Data], Sequence[Any], Sequence[Any]], None
]:
    """
    Helper method to load dynamically generated Python code which will be used
    to update the list of SDFG call arguments.

    It loads the Python code inside an empty namespace, without modifying the current
    global namespace. This is done to support parallel compilation, in which it can
    happen that two threads generate the same `bind_func_name` for the callback function.

    Args:
        module_name: Set on the loaded callback function for debugging.
        bind_func_name: Name to use for the translation function.
        python_code: String containing the Python code to load.

    Returns:
        A callable object to update the list of SDFG call arguments.
    """
    exec(python_code, global_namespace := {})  # type: ignore[var-annotated]
    assert bind_func_name not in globals()
    assert bind_func_name in global_namespace
    global_namespace[bind_func_name].__module__ = module_name
    return global_namespace[bind_func_name]


class CompiledDaceProgram(stages.CompiledProgram):
    sdfg_program: dace.CompiledSDFG

    # Sorted list of SDFG arguments as they appear in program ABI and corresponding data type;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_argtypes: list[dace.dtypes.Data]

    # The compiled program contains a callable object to update the SDFG arguments list.
    update_sdfg_ctype_arglist: Callable[
        [core_defs.DeviceType, Sequence[dace.dtypes.Data], Sequence[Any], Sequence[Any]],
        None,
    ]

    def __init__(
        self,
        program: dace.CompiledSDFG,
        bind_func_name: str,
        binding_source: stages.BindingSource[languages.SDFG, languages.Python],
    ):
        self.sdfg_program = program

        # `dace.CompiledSDFG.arglist()` returns an ordered dictionary that maps the argument
        # name to its data type, in the same order as arguments appear in the program ABI.
        # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
        self.sdfg_argtypes = list(program.sdfg.arglist().values())

        # Note that `binding_source` contains Python code tailored to this specific SDFG.
        #   We need to ensure that it is loaded as a Python module with a unique name,
        #   in order to avoid conflicts with other variants of the same program.
        #   Therefore, we use the name of the build folder as module name.
        binding_module_name = os.path.basename(program.sdfg.build_folder)
        self.update_sdfg_ctype_arglist = _get_sdfg_ctype_arglist_callback(
            binding_module_name, bind_func_name, binding_source.source_code
        )

    def __call__(self, **kwargs: Any) -> Any:
        return self.sdfg_program(**kwargs)

    def fast_call(self) -> Any:
        return self.sdfg_program.fast_call(*self.sdfg_program._lastargs)


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

    bind_func_name: str
    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(
        self,
        inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    ) -> CompiledDaceProgram:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            sdfg_build_folder = gtx_cache.get_cache_folder(inp, self.cache_lifetime)
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)

            sdfg = dace.SDFG.from_json(inp.program_source.source_code)
            sdfg.build_folder = sdfg_build_folder
            with locking.lock(sdfg_build_folder):
                sdfg_program = sdfg.compile(validate=False)

        assert inp.binding_source is not None
        return CompiledDaceProgram(
            sdfg_program,
            self.bind_func_name,
            inp.binding_source,
        )


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
