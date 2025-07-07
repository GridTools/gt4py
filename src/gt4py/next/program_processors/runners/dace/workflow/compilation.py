# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import importlib
import os
import pathlib
from typing import Any, Callable, Sequence

import dace
import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config
from gt4py.next.otf import languages, stages, step_types, workflow

from . import common as dace_common


def _get_sdfg_ctype_arglist_callback(
    module_name: str, bind_func_name: str, python_code: str
) -> Callable[
    [core_defs.DeviceType, Sequence[dace.dtypes.Data], Sequence[Any], Sequence[Any]], None
]:
    """
    Helper method to load dynamically generated Python code as a module and return
    a function to update the list of SDFG call arguments.

    Args:
        module_name: Name to use to load the python code as a module.
        bind_func_name: Name to use for the translation function.
        python_code: String containg the Python code to load.

    Returns:
        A callable object to update the list of SDFG call arguments.
    """
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    exec(python_code, module.__dict__)
    assert bind_func_name in module.__dict__
    return module.__dict__[bind_func_name]


class CompiledDaceProgram(stages.ExtendedCompiledProgram):
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
        implicit_domain: bool,
    ):
        self.sdfg_program = program
        self.implicit_domain = implicit_domain

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

    bind_func_name: str
    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(
        self,
        inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    ) -> CompiledDaceProgram:
        with dace.config.temporary_config():
            dace_common.set_dace_config(
                device_type=self.device_type,
                cmake_build_type=self.cmake_build_type,
            )
            sdfg = dace.SDFG.from_json(inp.program_source.source_code)
            sdfg_build_folder = pathlib.Path(sdfg.build_folder)
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)
            with locking.lock(sdfg_build_folder):
                sdfg_program = sdfg.compile(validate=False)

        assert inp.binding_source is not None
        return CompiledDaceProgram(
            sdfg_program,
            self.bind_func_name,
            inp.binding_source,
            inp.program_source.implicit_domain,
        )


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
